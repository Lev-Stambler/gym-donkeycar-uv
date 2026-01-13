#!/usr/bin/env python3
"""
Data collection script for VLM driving training.

Collects images and steering/throttle data from the Donkey Car simulator
using either autopilot (CTE-based controller) or manual input.

Usage:
    uv run python scripts/vlm_driving/collect_data.py
    uv run python scripts/vlm_driving/collect_data.py collection.track=donkey-warehouse-v0
    uv run python scripts/vlm_driving/collect_data.py collection.mode=manual collection.num_frames=5000
"""

import json
import logging
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

logger = logging.getLogger(__name__)


def autopilot_action(info: dict, cfg: DictConfig) -> np.ndarray:
    """
    Simple proportional controller based on cross-track error (CTE).

    The CTE is the distance from the car to the center of the track.
    Negative CTE means car is to the left, positive means to the right.

    Args:
        info: Telemetry dict from env.step()
        cfg: Config with autopilot.kp and autopilot.base_throttle

    Returns:
        Action array [steering, throttle]
    """
    cte = info.get("cte", 0.0)

    # Proportional steering: steer opposite to CTE
    steering = -cfg.autopilot.kp * cte
    steering = np.clip(steering, -1.0, 1.0)

    # Constant throttle (or speed-based throttle)
    throttle = cfg.autopilot.base_throttle

    return np.array([steering, throttle], dtype=np.float32)


def save_frame(
    frame_num: int,
    image: np.ndarray,
    steering: float,
    throttle: float,
    info: dict,
    output_dir: Path,
    track_name: str,
) -> None:
    """Save a single frame's image and metadata."""
    # Save image
    img_path = output_dir / f"{frame_num:06d}.jpg"
    Image.fromarray(image).save(img_path, quality=95)

    # Save metadata
    metadata = {
        "frame": frame_num,
        "steering": float(steering),
        "throttle": float(throttle),
        "cte": float(info.get("cte", 0.0)),
        "speed": float(info.get("speed", 0.0)),
        "pos": list(info.get("pos", (0, 0, 0))),
        "hit": str(info.get("hit", "none")),
        "track": track_name,
        "timestamp": datetime.now().isoformat(),
    }

    meta_path = output_dir / f"{frame_num:06d}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def collect_autopilot(env, cfg: DictConfig, output_dir: Path, track_name: str) -> int:
    """
    Collect data using autopilot (CTE-based controller).

    Returns:
        Number of frames collected
    """
    frame_num = 0
    target_frames = cfg.collection.num_frames
    target_fps = cfg.collection.target_fps
    frame_interval = 1.0 / target_fps

    logger.info(f"Collecting {target_frames} frames with autopilot...")

    while frame_num < target_frames:
        # Reset environment
        obs = env.reset()
        done = False

        while not done and frame_num < target_frames:
            start_time = time.time()

            # Get autopilot action
            # Need to do a step first to get info
            if frame_num == 0:
                # First frame, use zero steering
                action = np.array([0.0, cfg.autopilot.base_throttle], dtype=np.float32)
            else:
                action = autopilot_action(last_info, cfg)

            # Execute action
            obs, reward, done, info = env.step(action)
            last_info = info

            # Save frame
            save_frame(
                frame_num=frame_num,
                image=obs,
                steering=float(action[0]),
                throttle=float(action[1]),
                info=info,
                output_dir=output_dir,
                track_name=track_name,
            )

            frame_num += 1

            # Log progress
            if frame_num % 100 == 0:
                logger.info(
                    f"Collected {frame_num}/{target_frames} frames | "
                    f"CTE: {info.get('cte', 0):.2f} | "
                    f"Speed: {info.get('speed', 0):.2f}"
                )

            # Rate limiting
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    return frame_num


def exit_scene(env) -> None:
    """Exit the current scene."""
    if hasattr(env, "viewer") and env.viewer is not None:
        env.viewer.exit_scene()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main data collection function."""
    logger.info("Starting data collection...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory with session timestamp
    session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(cfg.collection.output_dir) / session_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save config to output directory
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
        "car_name": "vlm_collector",
        "font_size": 100,
        "start_delay": 2,
        "max_cte": cfg.preprocessing.max_cte,
        "cam_resolution": tuple(cfg.collection.cam_resolution),
    }

    # Remove exe_path if None (user starts sim manually)
    if env_conf["exe_path"] is None:
        del env_conf["exe_path"]

    # Create environment
    track_name = cfg.collection.track
    logger.info(f"Creating environment: {track_name}")
    env = gym.make(track_name, conf=env_conf)

    try:
        # Make sure no track is loaded
        exit_scene(env)

        # Collect data based on mode
        if cfg.collection.mode == "autopilot":
            num_frames = collect_autopilot(env, cfg, output_dir, track_name)
        elif cfg.collection.mode == "manual":
            logger.error("Manual mode not yet implemented. Use autopilot mode.")
            raise NotImplementedError("Manual mode requires pygame for keyboard input")
        else:
            raise ValueError(f"Unknown collection mode: {cfg.collection.mode}")

        logger.info(f"Data collection complete! Collected {num_frames} frames.")
        logger.info(f"Data saved to: {output_dir}")

    finally:
        # Clean up
        exit_scene(env)
        env.close()


if __name__ == "__main__":
    main()
