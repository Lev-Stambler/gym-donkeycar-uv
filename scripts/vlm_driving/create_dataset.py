#!/usr/bin/env python3
"""
Dataset creation script for VLM driving training.

Converts preprocessed metadata to Qwen-VL chat format (train.jsonl).

Usage:
    uv run python scripts/vlm_driving/create_dataset.py
"""

import json
import logging
import sys
from pathlib import Path

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.dataset_utils import create_qwen_chat_entry, write_jsonl

logger = logging.getLogger(__name__)


def load_metadata(metadata_path: Path) -> list:
    """Load metadata from JSON file."""
    with open(metadata_path) as f:
        return json.load(f)


def create_dataset(
    metadata: list,
    prompt: str = "Drive.",
) -> list:
    """
    Convert metadata entries to Qwen-VL chat format.

    Args:
        metadata: List of processed metadata entries
        prompt: Text prompt for the model

    Returns:
        List of Qwen-VL formatted entries
    """
    entries = []

    for item in metadata:
        entry = create_qwen_chat_entry(
            image_path=item["image_path"],
            steering_token=item["steering_token"],
            prompt=prompt,
        )
        entries.append(entry)

    return entries


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main dataset creation function."""
    logger.info("Creating Qwen-VL format dataset...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    processed_dir = Path(cfg.data_dir) / "processed"

    # Load preprocessed metadata
    train_meta_path = processed_dir / "train_metadata.json"
    val_meta_path = processed_dir / "val_metadata.json"

    if not train_meta_path.exists():
        logger.error(f"Train metadata not found: {train_meta_path}")
        logger.error("Run preprocess_data.py first.")
        return

    train_metadata = load_metadata(train_meta_path)
    logger.info(f"Loaded {len(train_metadata)} training samples")

    val_metadata = []
    if val_meta_path.exists():
        val_metadata = load_metadata(val_meta_path)
        logger.info(f"Loaded {len(val_metadata)} validation samples")

    # Convert to Qwen-VL format
    train_entries = create_dataset(train_metadata)
    val_entries = create_dataset(val_metadata) if val_metadata else []

    # Save as JSONL
    train_output = Path(cfg.train_file)
    val_output = Path(cfg.val_file)

    write_jsonl(train_entries, train_output)
    logger.info(f"Saved training dataset: {train_output}")

    if val_entries:
        write_jsonl(val_entries, val_output)
        logger.info(f"Saved validation dataset: {val_output}")

    # Print sample entry
    if train_entries:
        logger.info("Sample entry:")
        logger.info(json.dumps(train_entries[0], indent=2))

    logger.info("Dataset creation complete!")
    logger.info(f"Next step: run train.py to fine-tune the model")


if __name__ == "__main__":
    main()
