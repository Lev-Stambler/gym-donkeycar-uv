#!/usr/bin/env python3
"""
Build reward-weighted dataset from RL trajectories.

Converts collected trajectories into a training dataset where samples
are weighted by their trajectory reward. Higher reward trajectories
get more weight during training.

Usage:
    uv run python scripts/vlm_driving/build_weighted_dataset.py \
        rl.output_dir=data/rl_trajectories/rl_session_YYYYMMDD_HHMMSS
"""

import json
import logging
import sys
from pathlib import Path

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from utils.prompts import get_training_messages

logger = logging.getLogger(__name__)


def load_trajectories(trajectories_dir: Path) -> list:
    """
    Load all trajectory data from a collection session.

    Args:
        trajectories_dir: Directory containing episode_XXXX subdirectories

    Returns:
        List of episode data dicts
    """
    episodes = []

    for episode_dir in sorted(trajectories_dir.glob("episode_*")):
        episode_path = episode_dir / "episode.json"
        if episode_path.exists():
            with open(episode_path) as f:
                episode_data = json.load(f)
                episodes.append(episode_data)

    logger.info(f"Loaded {len(episodes)} episodes from {trajectories_dir}")
    return episodes


def compute_sample_weights(
    episodes: list,
    temperature: float = 1.0,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> dict:
    """
    Compute sample weights based on trajectory rewards.

    Uses softmax over rewards to convert to weights.

    Args:
        episodes: List of episode data
        temperature: Softmax temperature (higher = more uniform)
        min_weight: Minimum weight clamp
        max_weight: Maximum weight clamp

    Returns:
        Dict mapping episode_id to weight
    """
    rewards = np.array([e["cumulative_reward"] for e in episodes])

    # Normalize rewards to prevent overflow
    rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Softmax with temperature
    exp_rewards = np.exp(rewards_normalized / temperature)
    weights = exp_rewards / exp_rewards.sum() * len(episodes)

    # Clamp weights
    weights = np.clip(weights, min_weight, max_weight)

    # Create mapping
    weight_map = {e["episode_id"]: float(w) for e, w in zip(episodes, weights)}

    logger.info(f"Weight range: {weights.min():.2f} - {weights.max():.2f}")
    logger.info(f"Weight mean: {weights.mean():.2f}, std: {weights.std():.2f}")

    return weight_map


def filter_episodes(episodes: list, cfg: DictConfig) -> list:
    """
    Filter out low-quality episodes.

    Args:
        episodes: List of episode data
        cfg: Config with filtering thresholds

    Returns:
        Filtered list of episodes
    """
    filtered = []

    for episode in episodes:
        # Check minimum reward
        if episode["cumulative_reward"] < cfg.rl.min_episode_reward:
            continue

        # Check minimum steps
        if episode["num_steps"] < cfg.rl.min_episode_steps:
            continue

        filtered.append(episode)

    logger.info(
        f"Filtered {len(episodes)} -> {len(filtered)} episodes "
        f"(removed {len(episodes) - len(filtered)})"
    )

    return filtered


def build_dataset(
    episodes: list,
    weights: dict,
    output_path: Path,
) -> int:
    """
    Build weighted training dataset from episodes.

    Args:
        episodes: List of episode data
        weights: Episode ID to weight mapping
        output_path: Path to output JSONL file

    Returns:
        Number of samples written
    """
    num_samples = 0

    with open(output_path, "w") as f:
        for episode in episodes:
            episode_weight = weights.get(episode["episode_id"], 1.0)

            for step in episode["trajectory"]:
                # Create training sample
                sample = get_training_messages(
                    image_path=step["image_path"],
                    action_token=step["action_token"],
                )

                # Add weight and metadata
                sample["weight"] = episode_weight
                sample["episode_id"] = episode["episode_id"]
                sample["step"] = step["step"]
                sample["reward"] = step["reward"]
                sample["cumulative_reward"] = step["cumulative_reward"]

                f.write(json.dumps(sample) + "\n")
                num_samples += 1

    logger.info(f"Wrote {num_samples} samples to {output_path}")
    return num_samples


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main dataset building function."""
    logger.info("Building reward-weighted dataset...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Find trajectories directory
    trajectories_dir = Path(cfg.rl.output_dir)

    # If it's a pattern, find the most recent session
    if not trajectories_dir.exists():
        parent = trajectories_dir.parent
        if parent.exists():
            sessions = sorted(parent.glob("rl_session_*"))
            if sessions:
                trajectories_dir = sessions[-1]
                logger.info(f"Using most recent session: {trajectories_dir}")
            else:
                logger.error(f"No RL sessions found in {parent}")
                return
        else:
            logger.error(f"Trajectories directory not found: {trajectories_dir}")
            return

    # Load trajectories
    episodes = load_trajectories(trajectories_dir)
    if not episodes:
        logger.error("No episodes found!")
        return

    # Filter episodes
    episodes = filter_episodes(episodes, cfg)
    if not episodes:
        logger.error("All episodes filtered out!")
        return

    # Compute weights
    weights = compute_sample_weights(
        episodes,
        temperature=cfg.rl.reward_temperature,
        min_weight=cfg.rl.min_weight,
        max_weight=cfg.rl.max_weight,
    )

    # Log weight distribution by reward
    for episode in sorted(episodes, key=lambda e: e["cumulative_reward"], reverse=True)[
        :5
    ]:
        logger.info(
            f"Episode {episode['episode_id']}: "
            f"reward={episode['cumulative_reward']:.1f}, "
            f"weight={weights[episode['episode_id']]:.2f}"
        )

    # Build dataset
    output_dir = Path(cfg.data_dir) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_weighted.jsonl"
    num_samples = build_dataset(episodes, weights, train_path)

    # Create summary
    summary = {
        "source_dir": str(trajectories_dir),
        "num_episodes": len(episodes),
        "num_samples": num_samples,
        "avg_weight": float(np.mean(list(weights.values()))),
        "weight_range": [
            float(min(weights.values())),
            float(max(weights.values())),
        ],
    }

    summary_path = output_dir / "weighted_dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Dataset building complete!")
    logger.info(f"Train file: {train_path}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
