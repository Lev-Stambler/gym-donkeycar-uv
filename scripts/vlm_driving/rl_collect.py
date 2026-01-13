#!/usr/bin/env python3
"""
RL trajectory collection for reward-weighted SFT.

Collects trajectories by running in the simulator with either:
- Random policy (for initial exploration)
- Current VLM policy (for on-policy data)

Each trajectory records:
- Observations (images)
- Actions taken (tokens A-G)
- Step rewards and cumulative reward

Usage:
    uv run python scripts/vlm_driving/rl_collect.py
    uv run python scripts/vlm_driving/rl_collect.py rl.policy=random
    uv run python scripts/vlm_driving/rl_collect.py rl.num_episodes=100
"""

import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import gym_donkeycar  # noqa: F401 - registers environments
from utils.image_analysis import analyze_open_space
from utils.steering_buckets import ACTION_TOKENS

logger = logging.getLogger(__name__)


def compute_step_reward(info: dict, action_token: str, prev_pos: tuple = None) -> float:
    """
    Compute reward for a single step.

    Reward structure:
    - +1.0 base survival reward (per step without collision)
    - +speed bonus (encourages forward movement)
    - -10.0 collision penalty
    - Small steering penalty (encourages smooth driving)

    Args:
        info: Telemetry from simulator
        action_token: Action taken (A-G)
        prev_pos: Previous position for distance calculation

    Returns:
        Step reward
    """
    reward = 0.0

    # Check for collision
    hit = info.get("hit", "none")
    if hit != "none":
        return -10.0  # Large penalty for collision

    # Base survival reward
    reward += 1.0

    # Speed bonus (normalized by max expected speed)
    speed = info.get("speed", 0.0)
    reward += speed * 0.1  # Small bonus for moving fast

    # Distance bonus if we have previous position
    if prev_pos is not None:
        pos = info.get("pos", (0, 0, 0))
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(pos, prev_pos)))
        reward += distance * 0.5  # Bonus for covering distance

    # Small penalty for turning (encourages going straight when possible)
    if action_token in ["A", "G"]:  # Full turn commands
        reward -= 0.5
    elif action_token in ["B", "F"]:  # Hard steering
        reward -= 0.1

    return reward


def random_policy(obs: np.ndarray, info: dict) -> str:
    """
    Random policy for exploration.

    Biased towards going straight (D) with occasional turns.

    Args:
        obs: Current observation (image)
        info: Current telemetry

    Returns:
        Action token (A-G)
    """
    # Weighted random: mostly straight, sometimes steer, rarely turn
    weights = {
        "A": 0.05,  # Turn left (rare)
        "B": 0.10,  # Hard left
        "C": 0.15,  # Slight left
        "D": 0.40,  # Straight (most common)
        "E": 0.15,  # Slight right
        "F": 0.10,  # Hard right
        "G": 0.05,  # Turn right (rare)
    }

    tokens = list(weights.keys())
    probs = list(weights.values())
    return random.choices(tokens, weights=probs, k=1)[0]


def heuristic_policy(obs: np.ndarray, info: dict) -> str:
    """
    Simple heuristic policy based on image analysis.

    Goes straight normally, turns when obstacles detected.

    Args:
        obs: Current observation (image)
        info: Current telemetry

    Returns:
        Action token (A-G)
    """
    # Check for collision in last step
    hit = info.get("hit", "none")
    if hit != "none":
        # Just hit something, need to turn
        return analyze_open_space(obs)

    # Most of the time, go straight
    if random.random() < 0.9:
        return "D"

    # Occasionally check if we should steer
    direction = analyze_open_space(obs)
    if direction == "A":
        return random.choice(["C", "D"])  # Slight left or straight
    else:
        return random.choice(["E", "D"])  # Slight right or straight


def collect_episode(
    env,
    policy_fn,
    cfg: DictConfig,
    episode_dir: Path,
    episode_id: int,
) -> dict:
    """
    Collect a single episode trajectory.

    Args:
        env: Gym environment
        policy_fn: Policy function (obs, info) -> token
        cfg: Configuration
        episode_dir: Directory to save episode data
        episode_id: Episode identifier

    Returns:
        Episode summary dict
    """
    episode_dir.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    done = False
    step = 0
    max_steps = cfg.rl.max_steps_per_episode

    trajectory = []
    cumulative_reward = 0.0
    prev_pos = None

    while not done and step < max_steps:
        # Get action from policy
        info = {"hit": "none", "speed": 0, "pos": (0, 0, 0)}  # Default for first step
        if step > 0:
            info = last_info

        action_token = policy_fn(obs, info)
        action_info = ACTION_TOKENS.get(action_token, ACTION_TOKENS["D"])

        # Convert token to action
        if action_info["type"] == "turn":
            # For turn commands, execute full steering
            steering = -1.0 if action_info["degrees"] < 0 else 1.0
            throttle = cfg.rl.turn_throttle
        else:
            steering = action_info["value"]
            throttle = cfg.rl.base_throttle

        action = np.array([steering, throttle], dtype=np.float32)

        # Execute action
        next_obs, _, done, next_info = env.step(action)

        # Compute reward
        reward = compute_step_reward(next_info, action_token, prev_pos)
        cumulative_reward += reward

        # Save frame
        img_path = episode_dir / f"{step:04d}.jpg"
        Image.fromarray(obs).save(img_path, quality=95)

        # Record step
        trajectory.append(
            {
                "step": step,
                "action_token": action_token,
                "steering": float(steering),
                "throttle": float(throttle),
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "speed": float(next_info.get("speed", 0)),
                "hit": str(next_info.get("hit", "none")),
                "pos": list(next_info.get("pos", (0, 0, 0))),
                "image_path": str(img_path),
            }
        )

        prev_pos = next_info.get("pos", (0, 0, 0))
        obs = next_obs
        last_info = next_info
        step += 1

    # Save trajectory metadata
    episode_summary = {
        "episode_id": episode_id,
        "num_steps": step,
        "cumulative_reward": cumulative_reward,
        "avg_reward": cumulative_reward / step if step > 0 else 0,
        "terminated_by_collision": last_info.get("hit", "none") != "none",
        "trajectory": trajectory,
    }

    meta_path = episode_dir / "episode.json"
    with open(meta_path, "w") as f:
        json.dump(episode_summary, f, indent=2)

    return episode_summary


def collect_trajectories(
    env,
    cfg: DictConfig,
    output_dir: Path,
) -> list:
    """
    Collect multiple episode trajectories.

    Args:
        env: Gym environment
        cfg: Configuration
        output_dir: Output directory

    Returns:
        List of episode summaries
    """
    # Select policy
    if cfg.rl.policy == "random":
        policy_fn = random_policy
        logger.info("Using random policy for exploration")
    elif cfg.rl.policy == "heuristic":
        policy_fn = heuristic_policy
        logger.info("Using heuristic policy")
    else:
        raise ValueError(f"Unknown policy: {cfg.rl.policy}")

    episodes = []
    num_episodes = cfg.rl.num_episodes

    for episode_id in range(num_episodes):
        episode_dir = output_dir / f"episode_{episode_id:04d}"

        logger.info(f"Collecting episode {episode_id + 1}/{num_episodes}...")
        summary = collect_episode(env, policy_fn, cfg, episode_dir, episode_id)

        episodes.append(summary)
        logger.info(
            f"Episode {episode_id}: {summary['num_steps']} steps, "
            f"reward={summary['cumulative_reward']:.1f}, "
            f"collision={summary['terminated_by_collision']}"
        )

    return episodes


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main RL collection function."""
    logger.info("Starting RL trajectory collection...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory
    session_name = f"rl_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(cfg.rl.output_dir) / session_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)

    # Configure environment
    env_conf = {
        "exe_path": cfg.simulator.exe_path,
        "host": cfg.simulator.host,
        "port": cfg.simulator.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "rl_collector",
        "font_size": 100,
        "max_cte": 20.0,  # High threshold
        "cam_resolution": tuple(cfg.collection.cam_resolution),
    }

    if env_conf["exe_path"] is None:
        del env_conf["exe_path"]

    # Create environment
    track_name = cfg.collection.track
    logger.info(f"Creating environment: {track_name}")
    env = gym.make(track_name, conf=env_conf)

    try:
        # Collect trajectories
        episodes = collect_trajectories(env, cfg, output_dir)

        # Compute statistics
        rewards = [e["cumulative_reward"] for e in episodes]
        steps = [e["num_steps"] for e in episodes]
        collisions = sum(1 for e in episodes if e["terminated_by_collision"])

        logger.info("=" * 50)
        logger.info("COLLECTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Episodes: {len(episodes)}")
        logger.info(f"Total steps: {sum(steps)}")
        logger.info(f"Avg reward: {np.mean(rewards):.1f} (std: {np.std(rewards):.1f})")
        logger.info(f"Avg episode length: {np.mean(steps):.1f}")
        logger.info(f"Collision rate: {collisions}/{len(episodes)} ({100*collisions/len(episodes):.1f}%)")
        logger.info("=" * 50)

        # Save overall summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "num_episodes": len(episodes),
                    "total_steps": sum(steps),
                    "avg_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "min_reward": float(np.min(rewards)),
                    "max_reward": float(np.max(rewards)),
                    "avg_episode_length": float(np.mean(steps)),
                    "collision_rate": collisions / len(episodes),
                    "episodes": [
                        {
                            "episode_id": e["episode_id"],
                            "num_steps": e["num_steps"],
                            "cumulative_reward": e["cumulative_reward"],
                            "terminated_by_collision": e["terminated_by_collision"],
                        }
                        for e in episodes
                    ],
                },
                f,
                indent=2,
            )

        logger.info(f"Data saved to: {output_dir}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
