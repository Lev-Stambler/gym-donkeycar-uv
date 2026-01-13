"""Image analysis utilities for obstacle avoidance data collection.

This module provides heuristics for determining which direction (left or right)
has more open space based on camera image analysis.
"""

import cv2
import numpy as np


def analyze_open_space(image: np.ndarray) -> str:
    """
    Analyze image to determine which direction has more open space.

    Uses a combination of edge density and brightness analysis to determine
    whether the left or right side of the image has more navigable space.

    Args:
        image: RGB image from simulator (H x W x 3), typically 120x160x3

    Returns:
        "A" for turn left (more space on left), "G" for turn right (more space on right)
    """
    h, w, _ = image.shape

    # Split image into left and right halves
    left_half = image[:, : w // 2, :]
    right_half = image[:, w // 2 :, :]

    # Calculate openness scores for each half
    left_score = _compute_openness_score(left_half)
    right_score = _compute_openness_score(right_half)

    # Return turn direction based on which side is more open
    return "A" if left_score > right_score else "G"


def _compute_openness_score(image_region: np.ndarray) -> float:
    """
    Compute an openness score for an image region.

    Higher score = more open space (fewer obstacles).

    Args:
        image_region: RGB image region

    Returns:
        Openness score in range [0, 1]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)

    # Method 1: Edge density (fewer edges = more open)
    # Obstacles and walls create more edges than open space
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0
    edge_score = 1.0 - edge_density  # Invert: fewer edges = higher score

    # Method 2: Brightness (open areas tend to be brighter/more uniform)
    brightness = np.mean(gray) / 255.0

    # Method 3: Variance (lower variance = more uniform = likely open space)
    variance = np.var(gray) / (255.0 * 255.0)  # Normalize
    variance_score = 1.0 - min(variance * 10, 1.0)  # Scale and invert

    # Combine signals with weights
    score = 0.4 * edge_score + 0.3 * brightness + 0.3 * variance_score

    return score


def analyze_open_space_horizon(image: np.ndarray) -> str:
    """
    Alternative method: analyze the horizon band where obstacles are most visible.

    Focuses on the middle third of the image where obstacles ahead are most
    likely to appear.

    Args:
        image: RGB image from simulator (H x W x 3)

    Returns:
        "A" for turn left, "G" for turn right
    """
    h, w, _ = image.shape

    # Focus on the horizon band (middle third of image)
    horizon_start = h // 3
    horizon_end = 2 * h // 3
    horizon_band = image[horizon_start:horizon_end, :, :]

    # Analyze left vs right in horizon band
    left = horizon_band[:, : w // 2, :]
    right = horizon_band[:, w // 2 :, :]

    left_score = _compute_openness_score(left)
    right_score = _compute_openness_score(right)

    return "A" if left_score > right_score else "G"


def estimate_obstacle_distance(image: np.ndarray) -> float:
    """
    Estimate rough distance to obstacle in front based on image analysis.

    Uses bottom portion of image to detect how close obstacles are.
    When obstacles are very close, they take up more of the bottom of the frame.

    Args:
        image: RGB image from simulator (H x W x 3)

    Returns:
        Estimated distance score in range [0, 1], where 0 = very close, 1 = far
    """
    h, w, _ = image.shape

    # Focus on bottom third (closest to car) and center (ahead)
    bottom_start = 2 * h // 3
    center_start = w // 4
    center_end = 3 * w // 4

    bottom_center = image[bottom_start:, center_start:center_end, :]

    # Convert to grayscale
    gray = cv2.cvtColor(bottom_center, cv2.COLOR_RGB2GRAY)

    # Edge density in bottom center indicates close obstacles
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0

    # High edge density = close obstacles = low distance
    distance_score = 1.0 - edge_density

    return distance_score


def should_turn(image: np.ndarray, threshold: float = 0.3) -> bool:
    """
    Determine if the car should execute a turn based on obstacle proximity.

    Args:
        image: RGB image from simulator
        threshold: Distance threshold below which to turn (0-1)

    Returns:
        True if a turn should be executed
    """
    distance = estimate_obstacle_distance(image)
    return distance < threshold
