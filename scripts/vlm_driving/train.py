#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL-2B for RC car driving using LoRA/QLoRA.

This script trains a vision-language model to output discrete steering tokens
(A-G) given a camera image and the prompt "Drive."

Usage:
    uv run python scripts/vlm_driving/train.py
    uv run python scripts/vlm_driving/train.py training.num_train_epochs=5
    uv run python scripts/vlm_driving/train.py model.quantization.load_in_4bit=false
"""

import logging
import sys
from pathlib import Path

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def load_model_and_processor(cfg: DictConfig):
    """
    Load Qwen-VL model with optional QLoRA quantization.

    Args:
        cfg: Config with model settings

    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading model: {cfg.model_id}")

    # BitsAndBytes config for QLoRA
    bnb_config = None
    if cfg.quantization.load_in_4bit:
        logger.info("Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        )

    # Determine torch dtype
    torch_dtype = getattr(torch, cfg.torch_dtype)

    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Prepare for k-bit training if using quantization
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        cfg.processor_id,
        trust_remote_code=True,
    )

    return model, processor


def apply_lora(model, cfg: DictConfig):
    """
    Apply LoRA adapters to model.

    Args:
        model: The base model
        cfg: Config with LoRA settings

    Returns:
        Tuple of (peft_model, lora_config)
    """
    logger.info("Applying LoRA adapters")

    lora_config = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        target_modules=list(cfg.model.lora.target_modules),
        lora_dropout=cfg.model.lora.lora_dropout,
        bias=cfg.model.lora.bias,
        task_type=cfg.model.lora.task_type,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, lora_config


def create_collate_fn(processor):
    """
    Create data collator for vision-language batches.

    This function handles:
    1. Loading images from paths
    2. Applying the chat template
    3. Processing text + images into model inputs
    4. Creating labels for causal LM training

    Args:
        processor: The Qwen-VL processor

    Returns:
        Collate function
    """

    def collate_fn(examples):
        texts = []
        images = []

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

        # Process batch
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Set labels for causal LM training
        # Labels are same as input_ids for causal LM
        batch["labels"] = batch["input_ids"].clone()

        return batch

    return collate_fn


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting VLM fine-tuning...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    torch.manual_seed(cfg.seed)

    # Check for training data
    train_file = Path(cfg.train_file)
    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        logger.error("Run create_dataset.py first to generate train.jsonl")
        return

    # Load dataset
    logger.info(f"Loading dataset from {train_file}")
    data_files = {"train": str(train_file)}

    val_file = Path(cfg.val_file)
    if val_file.exists():
        data_files["validation"] = str(val_file)
        logger.info(f"Found validation data: {val_file}")

    dataset = load_dataset("json", data_files=data_files)
    logger.info(f"Train samples: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Val samples: {len(dataset['validation'])}")

    # Load model and processor
    model, processor = load_model_and_processor(cfg)

    # Apply LoRA
    model, lora_config = apply_lora(model, cfg)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        max_grad_norm=cfg.training.max_grad_norm,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps if "validation" in dataset else None,
        eval_strategy="steps" if "validation" in dataset else "no",
        save_total_limit=cfg.training.save_total_limit,
        remove_unused_columns=cfg.training.remove_unused_columns,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        report_to=cfg.training.report_to,
        # Important for VLMs: don't pack and don't truncate
        packing=cfg.training.packing,
        max_seq_length=cfg.training.max_seq_length,
        # Dataset config
        dataset_text_field=None,  # We use custom collator
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=create_collate_fn(processor),
        processing_class=processor,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_model_path = output_dir / "final_model"
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(str(final_model_path))

    # Save processor
    processor.save_pretrained(str(final_model_path))

    logger.info("Training complete!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"Next step: run evaluate.py to test the model in the simulator")


if __name__ == "__main__":
    main()
