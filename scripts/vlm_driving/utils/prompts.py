"""Prompt templates for VLM driving training."""

# System prompt with action descriptions
SYSTEM_PROMPT = """You are an autonomous RC car driver. Given a camera image, output a single letter (A-G) indicating the driving action:

A = Turn left 80° (obstacle avoidance)
B = Steer hard left while moving forward
C = Steer slight left while moving forward
D = Go straight forward
E = Steer slight right while moving forward
F = Steer hard right while moving forward
G = Turn right 80° (obstacle avoidance)

Goal: Drive forward as fast as possible. Use A or G only when you see an obstacle ahead and need to turn around."""

# Short prompt for inference (after model has learned)
INFERENCE_PROMPT = "Drive."

# Training prompt that includes action space
TRAINING_PROMPT = """Output a single letter (A-G) for the driving action:
A=turn left 80°, B=hard left, C=slight left, D=straight, E=slight right, F=hard right, G=turn right 80°
What action should the car take?"""

# Alternative: Very concise prompt
CONCISE_PROMPT = "Action (A-G):"


def get_training_messages(image_path: str, action_token: str) -> dict:
    """
    Create training message format with system prompt.

    Args:
        image_path: Path to the image file
        action_token: Ground truth action token (A-G)

    Returns:
        Messages dict in Qwen-VL format
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "What action should the car take?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": action_token}],
            },
        ]
    }


def get_inference_messages(image) -> list:
    """
    Create inference message format with system prompt.

    Args:
        image: PIL Image or image path

    Returns:
        Messages list for inference
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What action should the car take?"},
            ],
        },
    ]
