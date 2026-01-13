#!/usr/bin/env python3
"""
Test zero-shot VLM inference for RC car driving.

Uses the base Qwen3-VL model with a system prompt that describes
the action space (A-G). No fine-tuning required - just prompt engineering.

Usage:
    uv run python scripts/vlm_driving/test_zero_shot.py
    uv run python scripts/vlm_driving/test_zero_shot.py --image path/to/image.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from utils.prompts import SYSTEM_PROMPT, get_inference_messages
from utils.steering_buckets import ACTION_TOKENS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_id: str = "Qwen/Qwen3-VL-2B-Instruct"):
    """Load the base Qwen3-VL model (no fine-tuning)."""
    logger.info(f"Loading model: {model_id}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    return model, processor


@torch.no_grad()
def predict_action(model, processor, image: Image.Image) -> tuple:
    """
    Predict driving action from image using zero-shot prompting.

    Returns:
        Tuple of (token, full_response)
    """
    device = next(model.parameters()).device

    # Get messages with system prompt
    messages = get_inference_messages(image)

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
    ).to(device)

    # Generate (allow a few tokens for reasoning)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,  # Allow some tokens for response
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # Decode full response
    generated_ids = outputs[0, inputs.input_ids.shape[1]:]
    full_response = processor.decode(generated_ids, skip_special_tokens=True).strip()

    # Extract token (first character that's A-G)
    token = None
    for char in full_response.upper():
        if char in ACTION_TOKENS:
            token = char
            break

    if token is None:
        token = "D"  # Default to straight

    return token, full_response


def test_with_image(image_path: str):
    """Test with a specific image."""
    model, processor = load_model()

    image = Image.open(image_path).convert("RGB")
    logger.info(f"Image size: {image.size}")

    print("\n" + "=" * 60)
    print("SYSTEM PROMPT:")
    print("=" * 60)
    print(SYSTEM_PROMPT)
    print("=" * 60)

    token, response = predict_action(model, processor, image)

    print(f"\nModel response: {response}")
    print(f"Extracted token: {token}")
    print(f"Action: {ACTION_TOKENS[token]}")
    print("=" * 60)


def test_in_simulator():
    """Test in live simulator."""
    import gym
    import time
    import gym_donkeycar  # noqa

    model, processor = load_model()

    # Connect to simulator
    env_conf = {
        "host": "127.0.0.1",
        "port": 9091,
        "body_style": "donkey",
        "car_name": "zero_shot_test",
        "max_cte": 20.0,
        "cam_resolution": (120, 160, 3),
    }

    env = gym.make("donkey-warehouse-v0", conf=env_conf)

    print("\n" + "=" * 60)
    print("TESTING ZERO-SHOT VLM DRIVING")
    print("=" * 60)
    print("System prompt loaded with action descriptions A-G")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        obs = env.reset()
        done = False
        step = 0

        while not done:
            # Convert observation to PIL Image
            image = Image.fromarray(obs)

            # Get prediction
            start = time.time()
            token, response = predict_action(model, processor, image)
            inference_time = time.time() - start

            # Get action
            action_info = ACTION_TOKENS[token]
            if action_info["type"] == "turn":
                steering = -1.0 if action_info["degrees"] < 0 else 1.0
                throttle = 0.2
            else:
                steering = action_info["value"]
                throttle = 0.5

            # Execute
            obs, reward, done, info = env.step([steering, throttle])

            # Log
            if step % 10 == 0:
                hit = info.get("hit", "none")
                print(f"Step {step}: Token={token}, Response='{response[:20]}...', "
                      f"Hit={hit}, Time={inference_time*1000:.0f}ms")

            step += 1

            # Safety limit
            if step > 1000:
                break

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        env.close()

    print(f"\nTotal steps: {step}")


def main():
    parser = argparse.ArgumentParser(description="Test zero-shot VLM driving")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--sim", action="store_true", help="Run in simulator")
    args = parser.parse_args()

    if args.image:
        test_with_image(args.image)
    elif args.sim:
        test_in_simulator()
    else:
        print("Usage:")
        print("  --image PATH   Test with a specific image")
        print("  --sim          Test in live simulator")
        print("\nExample:")
        print("  uv run python scripts/vlm_driving/test_zero_shot.py --sim")


if __name__ == "__main__":
    main()
