#!/usr/bin/env python3
"""
Evaluate fine-tuned VLM in Donkey Car simulator.

Connects to the simulator and drives using the fine-tuned Qwen-VL model.

Usage:
    # Start simulator first, then:
    uv run python scripts/vlm_driving/evaluate.py inference.checkpoint=outputs/final_model
    uv run python scripts/vlm_driving/evaluate.py inference.throttle=0.4
"""

import logging
import sys
import time
from pathlib import Path

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from gym_donkeycar.core.sim_client import SimClient

from utils.vlm_handler import VLMDonkeyHandler

logger = logging.getLogger(__name__)


def load_finetuned_model(cfg: DictConfig):
    """
    Load fine-tuned LoRA model.

    Args:
        cfg: Config with model and inference settings

    Returns:
        Tuple of (model, processor)
    """
    checkpoint_path = Path(cfg.inference.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to train the model."
        )

    logger.info(f"Loading base model: {cfg.model_id}")

    # Determine torch dtype
    torch_dtype = getattr(torch, cfg.torch_dtype)

    # Load base model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        device_map=cfg.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(
        model,
        str(checkpoint_path),
    )

    # Optionally merge adapter for faster inference
    if cfg.inference.merge_adapter:
        logger.info("Merging LoRA adapter into base model...")
        model = model.merge_and_unload()

    model.eval()

    # Load processor (try from checkpoint first, fallback to base)
    processor_path = checkpoint_path
    if not (checkpoint_path / "processor_config.json").exists():
        processor_path = cfg.processor_id

    logger.info(f"Loading processor from: {processor_path}")
    processor = AutoProcessor.from_pretrained(
        str(processor_path),
        trust_remote_code=True,
    )

    return model, processor


def run_evaluation(
    model,
    processor,
    cfg: DictConfig,
) -> None:
    """
    Run evaluation in the simulator.

    Args:
        model: Fine-tuned model
        processor: Qwen-VL processor
        cfg: Config with simulator and inference settings
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create handler
    handler = VLMDonkeyHandler(
        model=model,
        processor=processor,
        constant_throttle=cfg.inference.throttle,
        num_buckets=cfg.preprocessing.num_steering_buckets,
        device=device,
        smoothing_window=cfg.inference.smoothing_window,
    )

    # Connect to simulator
    address = (cfg.simulator.host, cfg.simulator.port)
    logger.info(f"Connecting to simulator at {address}...")

    client = SimClient(address, handler)

    if not client.is_connected():
        logger.error(
            f"Failed to connect to simulator at {address}\n"
            "Make sure the Donkey Car simulator is running."
        )
        return

    logger.info("Connected! Running VLM-based driving...")
    logger.info("Press Ctrl+C to stop.")

    # Run until interrupted
    try:
        while client.is_connected():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping evaluation...")

    client.stop()
    logger.info("Evaluation complete.")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    logger.info("Starting VLM evaluation...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Check for checkpoint
    if cfg.inference.checkpoint is None:
        logger.error(
            "No checkpoint specified. Use:\n"
            "  uv run python evaluate.py inference.checkpoint=outputs/final_model"
        )
        return

    # Load model
    model, processor = load_finetuned_model(cfg)

    # Run evaluation
    run_evaluation(model, processor, cfg)


if __name__ == "__main__":
    main()
