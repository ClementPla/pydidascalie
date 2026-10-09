import inspect
import warnings
from dataclasses import dataclass
from typing import Callable

KEYPOINTS = "keypoints"
SEG = "seg"
SEQUENCE_SEG = "sequence_seg"

# What Didascalie can hand to a segmentation function besides the image. A
# function opts in by naming the parameter, so nothing is transferred unless
# it is asked for.
SEG_EXTRAS = ("masks", "labels", "active_label", "frame_index")


@dataclass
class Entry:
    fn: Callable
    kind: str
    # Extras the function asked for, by parameter name.
    wants: tuple[str, ...] = ()

    @property
    def doc(self) -> str:
        doc = inspect.getdoc(self.fn) or ""
        return doc.strip().split("\n", 1)[0]


# Module-level registry. Looked up by name at dispatch time, so redefining
# the decorated cell in a notebook replaces the function transparently.
_REGISTRY: dict[str, Entry] = {}


def _seg_wants(fn: Callable) -> tuple[str, ...]:
    """Which extras `fn` declares, after its first (image) parameter."""
    params = list(inspect.signature(fn).parameters.values())
    if not params:
        raise TypeError(
            f"{fn.__name__}() must accept the image as its first parameter"
        )
    wants = []
    for p in params[1:]:
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return SEG_EXTRAS
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if p.name in SEG_EXTRAS:
            wants.append(p.name)
        elif p.default is inspect.Parameter.empty:
            raise TypeError(
                f"{fn.__name__}() has a parameter {p.name!r} that Didascalie "
                f"cannot fill. Available: {', '.join(SEG_EXTRAS)}"
            )
    return tuple(wants)


def _decorator(kind: str, name):
    """Supports both `@deco` and `@deco("name")`."""

    def deco(fn: Callable) -> Callable:
        key = name if isinstance(name, str) else fn.__name__
        wants = () if kind == KEYPOINTS else _seg_wants(fn)
        _REGISTRY[key] = Entry(fn=fn, kind=kind, wants=wants)
        return fn

    if callable(name):
        return deco(name)
    return deco


def register_kpts(name: str | None = None):
    """
    Decorate a function that returns prefill keypoints for Didascalie.

    Expected signature:
        f(reference_image: np.ndarray,
          moving_image: np.ndarray,
          existing_keypoints: list[((rx, ry), (mx, my))]
        ) -> list[((rx, ry), (mx, my))]
    """
    return _decorator(KEYPOINTS, name)


def register(name: str | None = None):
    """Deprecated name of `register_kpts`."""
    warnings.warn(
        "didascalie.com.register is deprecated, use register_kpts",
        DeprecationWarning,
        stacklevel=2,
    )
    return _decorator(KEYPOINTS, name)


def register_seg(name: str | None = None):
    """
    Decorate a function that segments the frame open in the Didascalie editor.

    The first parameter receives the image (H×W×3 uint8 RGB). Any of these may
    follow, by name, and are only sent when declared:
        masks:        {label name: H×W uint8}, what is currently drawn
        labels:       list of the project's label names, in order
        active_label: name of the label selected in the editor, or None
        frame_index:  index of the frame in its sequence

    Return one of:
        {label name: H×W mask}   replaces those labels
        C×H×W array              one mask per project label, in order
        H×W array                drawn onto the active label

    Masks may be bool, float (thresholded at 0.5) or integer (kept as is, for
    instance ids). Torch tensors are accepted.
    """
    return _decorator(SEG, name)


def register_sequence_seg(name: str | None = None):
    """
    Decorate a function that segments a whole sequence at once.

    Same contract as `register_seg` with a leading frame axis: the first
    parameter receives T×H×W×3 uint8 (a list of H×W×3 arrays when the frames
    differ in size), `masks` is {label name: T×H×W}, and `frame_index` is the
    position, within the frames received, of the frame open in the editor.

    Return one of:
        {label name: T×H×W}   replaces those labels on every frame
        T×C×H×W array         one mask per project label, in order
        T×H×W array           drawn onto the active label of every frame
    """
    return _decorator(SEQUENCE_SEG, name)


def _get(name: str, kind: str | None = None) -> Entry | None:
    entry = _REGISTRY.get(name)
    if entry is None or (kind is not None and entry.kind != kind):
        return None
    return entry


def _list(kind: str | None = None) -> list[str]:
    return [k for k, e in _REGISTRY.items() if kind is None or e.kind == kind]


def _describe() -> list[dict]:
    return [
        {"name": k, "kind": e.kind, "doc": e.doc, "wants": list(e.wants)}
        for k, e in _REGISTRY.items()
    ]
