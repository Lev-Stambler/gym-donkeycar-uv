#!/usr/bin/env python3
"""
Simple image collector for testing VLM output format.

Just collects images from the simulator - no labels needed.
Useful for testing zero-shot inference.

Usage:
    uv run python scripts/vlm_driving/collect_images.py --num_images 100
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import gym_donkeycar  # noqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_images(
    num_images: int = 100,
    output_dir: str = "data/test_images",
    host: str = "127.0.0.1",
    port: int = 9091,
    track: str = "donkey-warehouse-v0",
):
    """Collect images from simulator."""

    output_path = Path(output_dir) / f"session_{datetime.now().strftime('%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to: {output_path}")

    env_conf = {
        "host": host,
        "port": port,
        "body_style": "donkey",
        "car_name": "image_collector",
        "max_cte": 20.0,
        "cam_resolution": (120, 160, 3),
    }

    env = gym.make(track, conf=env_conf)

    try:
        obs = env.reset()
        collected = 0

        while collected < num_images:
            # Random action (mostly forward)
            steering = np.random.uniform(-0.3, 0.3)
            throttle = 0.4

            obs, _, done, info = env.step([steering, throttle])

            # Save image
            img_path = output_path / f"{collected:04d}.jpg"
            Image.fromarray(obs).save(img_path, quality=95)
            collected += 1

            if collected % 10 == 0:
                hit = info.get("hit", "none")
                logger.info(f"Collected {collected}/{num_images} images (hit={hit})")

            if done:
                obs = env.reset()
                time.sleep(0.5)

        logger.info(f"Done! Saved {collected} images to {output_path}")
        return output_path

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect test images")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/test_images")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--track", type=str, default="donkey-warehouse-v0")
    args = parser.parse_args()

    collect_images(
        num_images=args.num_images,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        track=args.track,
    )


if __name__ == "__main__":
    main()
