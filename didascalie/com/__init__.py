from .decorators import register, register_kpts, register_seg, register_sequence_seg
from .server import serve
from .types import KeypointPair  # exported for later, unused for now

__all__ = [
    "register",
    "register_kpts",
    "register_seg",
    "register_sequence_seg",
    "serve",
    "KeypointPair",
]
__version__ = "0.1.0"
