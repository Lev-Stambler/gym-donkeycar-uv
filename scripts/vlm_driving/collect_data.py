#!/usr/bin/env python3
"""
Data collection script for VLM driving training.

Collects images and steering/throttle data from the Donkey Car simulator
using either autopilot (CTE-based controller) or obstacle-aware mode.

Obstacle-aware mode:
- Drives forward until collision detected
- Labels recent frames with turn direction (A or G)
- Executes turn to recover
- Repeats

Usage:
    uv run python scripts/vlm_driving/collect_data.py
    uv run python scripts/vlm_driving/collect_data.py collection.track=donkey-warehouse-v0
    uv run python scripts/vlm_driving/collect_data.py collection.mode=obstacle_aware
"""

import json
import logging
import sys
import time
from collections import deque
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
from utils.steering_buckets import steering_to_token

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
    steering_token: str = None,
    is_obstacle_frame: bool = False,
) -> None:
    """Save a single frame's image and metadata."""
    # Save image
    img_path = output_dir / f"{frame_num:06d}.jpg"
    Image.fromarray(image).save(img_path, quality=95)

    # Compute steering token if not provided
    if steering_token is None:
        steering_token = steering_to_token(steering, num_buckets=7)

    # Save metadata
    metadata = {
        "frame": frame_num,
        "steering": float(steering),
        "steering_token": steering_token,
        "throttle": float(throttle),
        "cte": float(info.get("cte", 0.0)),
        "speed": float(info.get("speed", 0.0)),
        "pos": list(info.get("pos", (0, 0, 0))),
        "hit": str(info.get("hit", "none")),
        "is_obstacle_frame": is_obstacle_frame,
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
        last_info = {"cte": 0.0}

        while not done and frame_num < target_frames:
            start_time = time.time()

            # Get autopilot action
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


def collect_obstacle_aware(
    env, cfg: DictConfig, output_dir: Path, track_name: str
) -> int:
    """
    Collect data with obstacle awareness.

    Drives forward until collision, then:
    1. Labels recent frames with turn direction
    2. Executes turn to recover
    3. Continues driving

    Returns:
        Number of frames collected
    """
    frame_num = 0
    target_frames = cfg.collection.num_frames
    target_fps = cfg.collection.target_fps
    frame_interval = 1.0 / target_fps

    # Lookback buffer for obstacle frames
    lookback_frames = cfg.obstacle.lookback_frames
    frame_buffer = deque(maxlen=lookback_frames)

    # Turn configuration
    turn_steps = cfg.obstacle.turn_steps
    turn_throttle = cfg.obstacle.turn_throttle

    collision_count = 0
    normal_frame_count = 0
    obstacle_frame_count = 0

    logger.info(f"Collecting {target_frames} frames with obstacle awareness...")
    logger.info(f"Lookback buffer: {lookback_frames} frames")
    logger.info(f"Turn steps: {turn_steps}")

    while frame_num < target_frames:
        # Reset environment
        obs = env.reset()
        done = False
        time.sleep(0.5)  # Wait for reset

        while not done and frame_num < target_frames:
            start_time = time.time()

            # Drive forward with moderate throttle
            action = np.array([0.0, cfg.autopilot.base_throttle], dtype=np.float32)

            # Execute action
            obs, reward, done, info = env.step(action)

            # Store frame in lookback buffer
            frame_buffer.append(
                {
                    "image": obs.copy(),
                    "steering": float(action[0]),
                    "throttle": float(action[1]),
                    "info": info.copy(),
                }
            )

            # Check for collision
            hit = info.get("hit", "none")
            if hit != "none":
                collision_count += 1
                logger.info(f"Collision #{collision_count} detected: {hit}")

                # Analyze image to determine turn direction
                turn_token = analyze_open_space(obs)
                turn_direction = -1.0 if turn_token == "A" else 1.0

                logger.info(
                    f"Turn direction: {'left' if turn_token == 'A' else 'right'} "
                    f"(token {turn_token})"
                )

                # Label buffered frames with turn command
                for buffered_frame in frame_buffer:
                    save_frame(
                        frame_num=frame_num,
                        image=buffered_frame["image"],
                        steering=buffered_frame["steering"],
                        throttle=buffered_frame["throttle"],
                        info=buffered_frame["info"],
                        output_dir=output_dir,
                        track_name=track_name,
                        steering_token=turn_token,
                        is_obstacle_frame=True,
                    )
                    frame_num += 1
                    obstacle_frame_count += 1

                    if frame_num >= target_frames:
                        break

                frame_buffer.clear()

                # Execute turn to recover
                if frame_num < target_frames:
                    logger.info(f"Executing {turn_steps}-step turn...")
                    for step in range(turn_steps):
                        turn_action = np.array(
                            [turn_direction, turn_throttle], dtype=np.float32
                        )
                        obs, reward, done, info = env.step(turn_action)
                        if done:
                            break

                    # Don't save turn frames (they're recovery, not training data)

            else:
                # Normal frame - save with computed steering token
                save_frame(
                    frame_num=frame_num,
                    image=obs,
                    steering=float(action[0]),
                    throttle=float(action[1]),
                    info=info,
                    output_dir=output_dir,
                    track_name=track_name,
                    steering_token="D",  # Going straight
                    is_obstacle_frame=False,
                )
                frame_num += 1
                normal_frame_count += 1

            # Log progress
            if frame_num % 100 == 0:
                logger.info(
                    f"Collected {frame_num}/{target_frames} frames | "
                    f"Normal: {normal_frame_count} | "
                    f"Obstacle: {obstacle_frame_count} | "
                    f"Collisions: {collision_count}"
                )

            # Rate limiting
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    logger.info(
        f"Collection complete. Normal: {normal_frame_count}, "
        f"Obstacle: {obstacle_frame_count}, Collisions: {collision_count}"
    )

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
        elif cfg.collection.mode == "obstacle_aware":
            num_frames = collect_obstacle_aware(env, cfg, output_dir, track_name)
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
