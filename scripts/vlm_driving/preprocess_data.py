#!/usr/bin/env python3
"""
Data preprocessing script for VLM driving training.

Processes raw collected data:
1. Filters out bad frames (collisions, off-track)
2. Converts continuous steering to discrete tokens
3. Optionally balances the dataset

Usage:
    uv run python scripts/vlm_driving/preprocess_data.py
    uv run python scripts/vlm_driving/preprocess_data.py preprocessing.max_cte=3.0
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.steering_buckets import get_bucket_distribution, steering_to_token

logger = logging.getLogger(__name__)


def load_session_metadata(session_dir: Path) -> List[Dict]:
    """Load all metadata files from a session directory."""
    metadata_files = sorted(session_dir.glob("*.json"))

    # Filter out config.yaml-style files
    metadata_files = [f for f in metadata_files if f.name != "config.yaml"]

    entries = []
    for meta_path in metadata_files:
        if meta_path.stem.isdigit():  # Only load numbered files
            with open(meta_path) as f:
                entry = json.load(f)
                entry["image_path"] = str(session_dir / f"{meta_path.stem}.jpg")
                entry["session"] = session_dir.name
                entries.append(entry)

    return entries


def filter_frames(entries: List[Dict], cfg: DictConfig) -> List[Dict]:
    """
    Filter out bad frames based on criteria.

    Removes frames where:
    - Car has collided with something
    - Car is too far from track center (high CTE)
    - Car is moving too slowly
    """
    filtered = []
    stats = Counter()

    for entry in entries:
        # Check collision
        hit = entry.get("hit", "none")
        if hit != "none":
            stats["collision"] += 1
            continue

        # Check CTE
        cte = abs(entry.get("cte", 0.0))
        if cte > cfg.preprocessing.max_cte:
            stats["high_cte"] += 1
            continue

        # Check speed
        speed = entry.get("speed", 0.0)
        if speed < cfg.preprocessing.min_speed:
            stats["low_speed"] += 1
            continue

        filtered.append(entry)
        stats["kept"] += 1

    logger.info(f"Filtering stats: {dict(stats)}")
    logger.info(f"Kept {len(filtered)}/{len(entries)} frames ({100*len(filtered)/len(entries):.1f}%)")

    return filtered


def add_steering_tokens(entries: List[Dict], num_buckets: int = 7) -> List[Dict]:
    """Add steering token to each entry."""
    for entry in entries:
        entry["steering_token"] = steering_to_token(entry["steering"], num_buckets)
    return entries


def balance_dataset(entries: List[Dict], num_buckets: int = 7) -> List[Dict]:
    """
    Balance dataset by undersampling majority classes.

    Uses the minority class count as the target for all classes.
    """
    # Group by token
    by_token: Dict[str, List[Dict]] = {}
    for entry in entries:
        token = entry["steering_token"]
        if token not in by_token:
            by_token[token] = []
        by_token[token].append(entry)

    # Find minimum count
    min_count = min(len(v) for v in by_token.values())
    logger.info(f"Balancing to {min_count} samples per class")

    # Sample from each class
    import random

    balanced = []
    for token, token_entries in by_token.items():
        sampled = random.sample(token_entries, min_count)
        balanced.extend(sampled)

    # Shuffle
    random.shuffle(balanced)

    return balanced


def print_distribution(entries: List[Dict], num_buckets: int = 7) -> None:
    """Print the steering token distribution."""
    steering_values = [e["steering"] for e in entries]
    distribution = get_bucket_distribution(steering_values, num_buckets)

    total = sum(distribution.values())
    logger.info("Steering distribution:")
    for token, count in sorted(distribution.items()):
        pct = 100 * count / total if total > 0 else 0
        bar = "#" * int(pct / 2)
        logger.info(f"  {token}: {count:5d} ({pct:5.1f}%) {bar}")


def save_processed_data(
    entries: List[Dict], output_dir: Path, val_split: float = 0.1
) -> Tuple[Path, Path]:
    """
    Save processed metadata and split into train/val.

    Returns paths to train and val metadata files.
    """
    import random

    output_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.shuffle(entries)
    split_idx = int(len(entries) * (1 - val_split))
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    # Save
    train_path = output_dir / "train_metadata.json"
    val_path = output_dir / "val_metadata.json"

    with open(train_path, "w") as f:
        json.dump(train_entries, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_entries, f, indent=2)

    logger.info(f"Saved {len(train_entries)} train, {len(val_entries)} val samples")

    return train_path, val_path


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main preprocessing function."""
    logger.info("Starting data preprocessing...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    raw_dir = Path(cfg.collection.output_dir)
    processed_dir = Path(cfg.data_dir) / "processed"

    # Find all session directories
    session_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(session_dirs)} session(s) in {raw_dir}")

    if not session_dirs:
        logger.error(f"No session directories found in {raw_dir}")
        logger.error("Run collect_data.py first to collect training data.")
        return

    # Load all metadata
    all_entries = []
    for session_dir in session_dirs:
        entries = load_session_metadata(session_dir)
        logger.info(f"Loaded {len(entries)} frames from {session_dir.name}")
        all_entries.extend(entries)

    logger.info(f"Total frames loaded: {len(all_entries)}")

    # Filter bad frames
    filtered = filter_frames(all_entries, cfg)

    # Add steering tokens
    num_buckets = cfg.preprocessing.num_steering_buckets
    tokenized = add_steering_tokens(filtered, num_buckets)

    # Show distribution before balancing
    logger.info("Distribution before balancing:")
    print_distribution(tokenized, num_buckets)

    # Optionally balance
    if cfg.preprocessing.balance_classes:
        tokenized = balance_dataset(tokenized, num_buckets)
        logger.info("Distribution after balancing:")
        print_distribution(tokenized, num_buckets)

    # Save processed data
    train_path, val_path = save_processed_data(
        tokenized, processed_dir, cfg.preprocessing.val_split
    )

    logger.info(f"Preprocessing complete!")
    logger.info(f"Train metadata: {train_path}")
    logger.info(f"Val metadata: {val_path}")
    logger.info(f"Next step: run create_dataset.py to generate train.jsonl")


if __name__ == "__main__":
    main()
