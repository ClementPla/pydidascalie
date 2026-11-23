from .decorators import register, register_kpts, register_seg, register_sequence_seg
from .params import Boolean, Enum, Float, Int
from .server import serve
from .types import KeypointPair, Result  # KeypointPair: for later, unused for now

__all__ = [
    "register",
    "register_kpts",
    "register_seg",
    "register_sequence_seg",
    "serve",
    "Boolean",
    "Enum",
    "Float",
    "Int",
    "KeypointPair",
    "Result",
]
__version__ = "0.1.0"
