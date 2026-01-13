"""Steering value to discrete token mapping utilities."""

from typing import Dict, Tuple

# 7 steering buckets with token mapping (finer control)
STEERING_BUCKETS: Dict[str, Tuple[float, float]] = {
    "A": (-1.0, -0.6),   # Hard left
    "B": (-0.6, -0.3),   # Medium left
    "C": (-0.3, -0.1),   # Slight left
    "D": (-0.1, 0.1),    # Straight
    "E": (0.1, 0.3),     # Slight right
    "F": (0.3, 0.6),     # Medium right
    "G": (0.6, 1.01),    # Hard right (1.01 to include 1.0)
}

# 5 steering buckets (simpler, less granular)
STEERING_BUCKETS_5: Dict[str, Tuple[float, float]] = {
    "A": (-1.0, -0.4),   # Left
    "B": (-0.4, -0.1),   # Slight left
    "C": (-0.1, 0.1),    # Straight
    "D": (0.1, 0.4),     # Slight right
    "E": (0.4, 1.01),    # Right
}


def steering_to_token(steering: float, num_buckets: int = 7) -> str:
    """
    Convert continuous steering value to discrete token.

    Args:
        steering: Steering value in range [-1, 1]
        num_buckets: Number of buckets (5 or 7)

    Returns:
        Token string (A-G for 7 buckets, A-E for 5 buckets)
    """
    buckets = STEERING_BUCKETS if num_buckets == 7 else STEERING_BUCKETS_5

    # Clamp steering to valid range
    steering = max(-1.0, min(1.0, steering))

    for token, (low, high) in buckets.items():
        if low <= steering < high:
            return token

    # Default to straight if edge case
    return "D" if num_buckets == 7 else "C"


def token_to_steering(token: str, num_buckets: int = 7) -> float:
    """
    Convert discrete token back to steering value (bucket center).

    Args:
        token: Token string (A-G or A-E)
        num_buckets: Number of buckets (5 or 7)

    Returns:
        Steering value (center of bucket range)
    """
    buckets = STEERING_BUCKETS if num_buckets == 7 else STEERING_BUCKETS_5

    if token in buckets:
        low, high = buckets[token]
        # Clamp high to 1.0 for center calculation
        high = min(high, 1.0)
        return (low + high) / 2.0

    # Default to straight
    return 0.0


def get_bucket_distribution(steering_values: list, num_buckets: int = 7) -> Dict[str, int]:
    """
    Get distribution of steering values across buckets.

    Useful for checking dataset balance.

    Args:
        steering_values: List of steering floats
        num_buckets: Number of buckets

    Returns:
        Dict mapping token to count
    """
    buckets = STEERING_BUCKETS if num_buckets == 7 else STEERING_BUCKETS_5
    distribution = {token: 0 for token in buckets.keys()}

    for steering in steering_values:
        token = steering_to_token(steering, num_buckets)
        distribution[token] += 1

    return distribution
