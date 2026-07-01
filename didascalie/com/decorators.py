from typing import Callable

# Module-level registry. Looked up by name at dispatch time, so redefining
# the decorated cell in a notebook replaces the function transparently.
_REGISTRY: dict[str, Callable] = {}

def register(name: str | None = None):
    """
    Decorate a function that returns prefill keypoints for Didascalie.

    Expected signature:
        f(reference_image: np.ndarray,
          moving_image: np.ndarray,
          existing_keypoints: list[((rx, ry), (mx, my))]
        ) -> list[((rx, ry), (mx, my))]
    """
    def deco(fn: Callable) -> Callable:
        key = name or fn.__name__
        _REGISTRY[key] = fn
        return fn
    return deco

def _get(name: str) -> Callable | None:
    return _REGISTRY.get(name)

def _list() -> list[str]:
    return list(_REGISTRY.keys())