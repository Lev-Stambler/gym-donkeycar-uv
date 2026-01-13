#!/usr/bin/env python3
"""
Reward-weighted SFT training for VLM driving.

Trains the VLM using collected RL trajectories where samples are
weighted by their trajectory reward.

Usage:
    uv run python scripts/vlm_driving/train_rl.py
    uv run python scripts/vlm_driving/train_rl.py training.num_train_epochs=3
"""

import logging
import sys
from pathlib import Path

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import hydra
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    get_scheduler,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_model_and_processor(cfg: DictConfig):
    """Load Qwen3-VL model with optional QLoRA quantization."""
    logger.info(f"Loading model: {cfg.model_id}")

    bnb_config = None
    if cfg.quantization.load_in_4bit:
        logger.info("Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        )

    torch_dtype = getattr(torch, cfg.torch_dtype)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    processor = AutoProcessor.from_pretrained(
        cfg.processor_id,
        trust_remote_code=True,
    )

    return model, processor


def apply_lora(model, cfg: DictConfig):
    """Apply LoRA adapters to model."""
    logger.info("Applying LoRA adapters")

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=list(cfg.lora.target_modules),
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, lora_config


def create_weighted_collate_fn(processor):
    """
    Create data collator that includes sample weights.

    Args:
        processor: The Qwen-VL processor

    Returns:
        Collate function that returns (batch, weights)
    """

    def collate_fn(examples):
        texts = []
        images = []
        weights = []

        for example in examples:
            messages = example["messages"]

            # Load image from path
            for msg in messages:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "image":
                            img_path = content["image"]
                            img = Image.open(img_path).convert("RGB")
                            images.append(img)

            # Apply chat template
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            # Get weight (default to 1.0 if not present)
            weights.append(example.get("weight", 1.0))

        # Process batch
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Set labels for causal LM training
        batch["labels"] = batch["input_ids"].clone()

        # Add weights to batch
        batch["sample_weights"] = torch.tensor(weights, dtype=torch.float32)

        return batch

    return collate_fn


def compute_weighted_loss(model, batch, device):
    """
    Compute weighted cross-entropy loss.

    Args:
        model: The model
        batch: Batch with input_ids, labels, attention_mask, sample_weights
        device: Device to use

    Returns:
        Weighted loss tensor
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    pixel_values = batch.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    image_grid_thw = batch.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    weights = batch["sample_weights"].to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=labels,
        return_dict=True,
    )

    # Get per-sample loss
    # The model returns average loss, we need to recompute for weighting
    logits = outputs.logits

    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    # Reshape to (batch, seq_len)
    loss = loss.view(shift_labels.size())

    # Mask padding tokens
    mask = (shift_labels != -100).float()
    loss = loss * mask

    # Per-sample loss (average over sequence)
    per_sample_loss = loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # Apply sample weights
    weighted_loss = (per_sample_loss * weights).mean()

    return weighted_loss


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, cfg):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        optimizer.zero_grad()

        loss = compute_weighted_loss(model, batch, device)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.max_grad_norm
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg_loss=f"{total_loss / num_batches:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    return total_loss / num_batches


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main RL training function."""
    logger.info("Starting reward-weighted SFT training...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    torch.manual_seed(cfg.seed)

    # Check for training data
    train_file = Path(cfg.data_dir) / "processed" / "train_weighted.jsonl"
    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        logger.error("Run rl_collect.py and build_weighted_dataset.py first")
        return

    # Load dataset
    logger.info(f"Loading dataset from {train_file}")
    dataset = load_dataset("json", data_files={"train": str(train_file)})
    logger.info(f"Train samples: {len(dataset['train'])}")

    # Load model and processor
    model, processor = load_model_and_processor(cfg.model)

    # Apply LoRA
    model, lora_config = apply_lora(model, cfg.model)

    # Create dataloader
    collate_fn = create_weighted_collate_fn(processor)
    dataloader = DataLoader(
        dataset["train"],
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.training.dataloader_num_workers,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Setup scheduler
    num_training_steps = len(dataloader) * cfg.training.num_train_epochs
    num_warmup_steps = int(num_training_steps * cfg.training.warmup_ratio)

    scheduler = get_scheduler(
        cfg.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Determine device
    device = next(model.parameters()).device
    logger.info(f"Training on device: {device}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("Starting training...")
    best_loss = float("inf")

    for epoch in range(1, cfg.training.num_train_epochs + 1):
        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, device, epoch, cfg
        )

        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = output_dir / "best_model"
            logger.info(f"Saving best model to {checkpoint_path}")
            model.save_pretrained(str(checkpoint_path))
            processor.save_pretrained(str(checkpoint_path))

    # Save final model
    final_model_path = output_dir / "final_model"
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(str(final_model_path))
    processor.save_pretrained(str(final_model_path))

    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
