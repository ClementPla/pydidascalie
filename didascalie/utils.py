"""Utility functions for Didascalie."""

import colorsys
from typing import List


def generate_distinct_colors(n: int, saturation: float = 0.7, value: float = 0.9) -> List[str]:
    """
    Generate n visually distinct colors.

    Args:
        n: Number of colors to generate
        saturation: Color saturation (0-1)
        value: Color value/brightness (0-1)

    Returns:
        List of hex color strings
    """
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        colors.append(hex_color)
    return colors


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB tuple to hex color."""
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def calculate_iou(mask1, mask2) -> float:
    """
    Calculate Intersection over Union between two binary masks.

    Args:
        mask1: Binary mask array
        mask2: Binary mask array

    Returns:
        IoU score (0-1)
    """
    import numpy as np

    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def calculate_dice(mask1, mask2) -> float:
    """
    Calculate Dice coefficient between two binary masks.

    Args:
        mask1: Binary mask array
        mask2: Binary mask array

    Returns:
        Dice score (0-1)
    """
    import numpy as np

    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    if total == 0:
        return 0.0

    return 2 * intersection / total
