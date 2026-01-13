"""Dataset formatting utilities for Qwen-VL training."""

import json
from pathlib import Path
from typing import Any, Dict, List


def create_qwen_chat_entry(
    image_path: str,
    steering_token: str,
    prompt: str = "Drive.",
) -> Dict[str, Any]:
    """
    Create a single training entry in Qwen-VL chat format.

    This format is compatible with TRL SFTTrainer for VLMs.

    Args:
        image_path: Path to the image file
        steering_token: Discrete steering token (A-G)
        prompt: Text prompt for the model

    Returns:
        Dict in Qwen-VL message format
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": steering_token},
                ],
            },
        ]
    }


def create_llava_entry(
    image_filename: str,
    steering_token: str,
    entry_id: str,
) -> Dict[str, Any]:
    """
    Create entry in LLaVA-style format (alternative format).

    This format is used by some VLM fine-tuning frameworks.

    Args:
        image_filename: Filename of the image (not full path)
        steering_token: Discrete steering token
        entry_id: Unique identifier for this entry

    Returns:
        Dict in LLaVA conversation format
    """
    return {
        "id": entry_id,
        "image": image_filename,
        "conversations": [
            {"from": "human", "value": "<image>\nDrive."},
            {"from": "gpt", "value": steering_token},
        ],
    }


def write_jsonl(entries: List[Dict], output_path: Path) -> None:
    """
    Write entries to JSONL format.

    Args:
        entries: List of dicts to write
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def read_jsonl(input_path: Path) -> List[Dict]:
    """
    Read entries from JSONL format.

    Args:
        input_path: Path to input file

    Returns:
        List of dicts
    """
    entries = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries
