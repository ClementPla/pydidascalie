# didascalie/types.py
from dataclasses import dataclass

@dataclass
class KeypointPair:
    """Future-proof type. Not used in the v0 contract — tuples on the wire."""
    ref: tuple[float, float]
    moving: tuple[float, float]