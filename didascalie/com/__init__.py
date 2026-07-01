from .decorators import register
from .server import serve
from .types import KeypointPair  # exported for later, unused for now

__all__ = ["register", "serve", "KeypointPair"]
__version__ = "0.0.1"