# didascalie/types.py
from dataclasses import dataclass
from typing import Any

@dataclass
class Result:
    """What a segmentation function returns to also show a message.

    `masks` is anything the function could return on its own, or None to leave
    the masks as they are. Didascalie shows `message` when the run ends.
    """
    masks: Any = None
    message: str = ""


@dataclass
class KeypointPair:
    """Future-proof type. Not used in the v0 contract — tuples on the wire."""
    ref: tuple[float, float]
    moving: tuple[float, float]