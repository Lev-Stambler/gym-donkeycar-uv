#!/usr/bin/env python3
"""
Calibrate the number of steps required for an 80-degree turn.

This script drives the car with full steering for a measured number of steps,
then reports the actual yaw change to help determine the correct STEPS_FOR_80_DEGREE_TURN
constant.

Usage:
    uv run python scripts/vlm_driving/calibrate_turn.py
    uv run python scripts/vlm_driving/calibrate_turn.py --throttle 0.3 --target 80
"""

import argparse
import sys
import time
from pathlib import Path

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import gym
import numpy as np

import gym_donkeycar  # noqa: F401 - registers environments


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180] range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def calibrate_turn_steps(
    track: str = "donkey-warehouse-v0",
    host: str = "127.0.0.1",
    port: int = 9091,
    throttle: float = 0.2,
    target_degrees: float = 80.0,
    test_steps: int = 50,
    num_trials: int = 3,
) -> int:
    """
    Calibrate turn steps by measuring actual yaw change.

    Args:
        track: Environment/track name
        host: Simulator host
        port: Simulator port
        throttle: Throttle value during turn
        target_degrees: Target turn angle in degrees
        test_steps: Number of steps to execute per trial
        num_trials: Number of trials per direction

    Returns:
        Recommended steps for target_degrees turn
    """
    env_conf = {
        "host": host,
        "port": port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "calibrator",
        "font_size": 100,
        "max_cte": 20.0,  # High threshold to avoid early termination
        "cam_resolution": (120, 160, 3),
    }

    print(f"Connecting to simulator at {host}:{port}...")
    print(f"Track: {track}")
    print(f"Throttle: {throttle}")
    print(f"Target angle: {target_degrees} degrees")
    print(f"Test steps per trial: {test_steps}")
    print()

    env = gym.make(track, conf=env_conf)

    try:
        results = []

        for direction in [-1.0, 1.0]:  # Left and right
            direction_name = "left" if direction < 0 else "right"

            for trial in range(num_trials):
                print(f"Trial {trial + 1}/{num_trials} - Turn {direction_name}...")

                # Reset environment
                obs = env.reset()
                time.sleep(1.0)  # Wait for reset to complete

                # Get initial yaw
                _, _, _, info = env.step([0, 0])
                initial_yaw = info.get("car", (0, 0, 0))[2]  # yaw is third element

                # Execute full steering for test_steps
                for step in range(test_steps):
                    obs, _, done, info = env.step([direction, throttle])
                    if done:
                        print(f"  Episode ended early at step {step}")
                        break

                # Get final yaw
                final_yaw = info.get("car", (0, 0, 0))[2]

                # Calculate yaw change, handling wraparound
                yaw_change = normalize_angle(final_yaw - initial_yaw)
                yaw_change = abs(yaw_change)

                # Calculate metrics
                degrees_per_step = yaw_change / test_steps if test_steps > 0 else 0
                steps_for_target = (
                    target_degrees / degrees_per_step if degrees_per_step > 0 else 0
                )

                results.append(
                    {
                        "direction": direction_name,
                        "trial": trial,
                        "test_steps": test_steps,
                        "yaw_change": yaw_change,
                        "degrees_per_step": degrees_per_step,
                        "steps_for_target": steps_for_target,
                    }
                )

                print(
                    f"  Yaw change: {yaw_change:.1f} deg in {test_steps} steps "
                    f"({degrees_per_step:.2f} deg/step)"
                )
                print(f"  Steps for {target_degrees} deg: {steps_for_target:.1f}")

        # Calculate statistics
        all_steps = [r["steps_for_target"] for r in results if r["steps_for_target"] > 0]

        if all_steps:
            avg_steps = np.mean(all_steps)
            std_steps = np.std(all_steps)
            min_steps = np.min(all_steps)
            max_steps = np.max(all_steps)

            print()
            print("=" * 50)
            print("CALIBRATION RESULTS")
            print("=" * 50)
            print(f"Average steps for {target_degrees} deg: {avg_steps:.1f}")
            print(f"Std dev: {std_steps:.1f}")
            print(f"Range: {min_steps:.1f} - {max_steps:.1f}")
            print()
            print(f"RECOMMENDED: STEPS_FOR_80_DEGREE_TURN = {int(round(avg_steps))}")
            print("=" * 50)

            return int(round(avg_steps))
        else:
            print("ERROR: No valid measurements obtained")
            return 100  # Default fallback

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate turn step count for obstacle avoidance"
    )
    parser.add_argument(
        "--track",
        type=str,
        default="donkey-warehouse-v0",
        help="Track/environment name",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Simulator host"
    )
    parser.add_argument("--port", type=int, default=9091, help="Simulator port")
    parser.add_argument(
        "--throttle", type=float, default=0.2, help="Throttle during turn"
    )
    parser.add_argument(
        "--target", type=float, default=80.0, help="Target turn angle in degrees"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Test steps per trial"
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of trials per direction"
    )

    args = parser.parse_args()

    calibrate_turn_steps(
        track=args.track,
        host=args.host,
        port=args.port,
        throttle=args.throttle,
        target_degrees=args.target,
        test_steps=args.steps,
        num_trials=args.trials,
    )


if __name__ == "__main__":
    main()
