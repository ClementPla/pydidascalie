"""Turn whatever a segmentation function returned into masks for the wire."""
from typing import Any, Callable

import numpy as np

# One returned layer: the label it targets (None = the editor's active label)
# and how to get its mask for frame `i`. Single-frame results ignore `i`.
Layer = tuple[str | None, Callable[[int], Any]]


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor, without importing torch
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _label_name(key: Any, labels: list[str]) -> str | None:
    """Resolve a dict key (name or index) to a label name, None if unknown."""
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        return labels[key] if 0 <= key < len(labels) else None
    return key if key in labels else None


def _from_dict(result: dict, labels: list[str]) -> tuple[list[tuple[str, Any]], list[str]]:
    known, unknown = [], []
    for key, value in result.items():
        name = _label_name(key, labels)
        if name is None:
            unknown.append(str(key))
        else:
            known.append((name, value))
    return known, unknown


def _channel_error(c: int, labels: list[str]) -> ValueError:
    return ValueError(
        f"Got {c} masks for {len(labels)} labels {labels}. Return one mask per "
        f"label, a single mask for the active label, or a dict keyed by label name."
    )


def frame_layers(result: Any, labels: list[str]) -> tuple[list[Layer], list[str]]:
    """Layers of a `register_seg` result, plus the label names nobody knows."""
    if result is None:
        return [], []
    if isinstance(result, dict):
        known, unknown = _from_dict(result, labels)
        return [(name, lambda _i, v=v: v) for name, v in known], unknown

    arr = to_numpy(result)
    if arr.ndim == 2:
        return [(None, lambda _i: arr)], []
    if arr.ndim == 3:
        if arr.shape[0] == len(labels):
            return [(name, lambda _i, c=c: arr[c]) for c, name in enumerate(labels)], []
        if arr.shape[0] == 1:
            return [(None, lambda _i: arr[0])], []
        raise _channel_error(arr.shape[0], labels)
    raise ValueError(f"Expected an H×W or C×H×W mask, got shape {arr.shape}")


def sequence_layers(
    result: Any, labels: list[str], n_frames: int
) -> tuple[list[Layer], list[str]]:
    """Layers of a `register_sequence_seg` result, each indexed by frame."""

    def check_length(value: Any, what: str) -> None:
        if len(value) != n_frames:
            raise ValueError(
                f"{what} has {len(value)} frames, the sequence has {n_frames}"
            )

    if result is None:
        return [], []
    if isinstance(result, dict):
        known, unknown = _from_dict(result, labels)
        for name, value in known:
            check_length(value, f"Masks for {name!r}")
        return [(name, lambda i, v=v: v[i]) for name, v in known], unknown

    check_length(result, "The result")
    if n_frames == 0:
        return [], []
    # Indexed per frame rather than converted whole, so a list of differently
    # sized frames works as well as a stacked array.
    first = to_numpy(result[0])
    if first.ndim == 2:
        return [(None, lambda i: result[i])], []
    if first.ndim == 3:
        if first.shape[0] == len(labels):
            return [(name, lambda i, c=c: result[i][c]) for c, name in enumerate(labels)], []
        if first.shape[0] == 1:
            return [(None, lambda i: result[i][0])], []
        raise _channel_error(first.shape[0], labels)
    raise ValueError(
        f"Expected T×H×W or T×C×H×W masks, got per-frame shape {first.shape}"
    )


def encode(layers: list[Layer], index: int, hw: tuple[int, int]) -> list[dict]:
    """Frame `index` of every layer, as uint8 wire masks."""
    out = []
    for label, get in layers:
        arr = to_numpy(get(index))
        if arr.shape != tuple(hw):
            raise ValueError(
                f"Mask for {label or 'the active label'} has shape {arr.shape}, "
                f"the frame is {tuple(hw)}"
            )
        # `binary` tells Didascalie the values carry no meaning beyond on/off,
        # so on an instance label it paints the instance selected in the editor.
        binary = arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.floating)
        if arr.dtype == np.bool_:
            mask = arr.astype(np.uint8)
        elif binary:
            mask = (arr > 0.5).astype(np.uint8)
        else:
            mask = np.clip(arr, 0, 255).astype(np.uint8)
        out.append(
            {
                "label": label,
                "buf": np.ascontiguousarray(mask).tobytes(),
                "shape": list(mask.shape),
                "binary": bool(binary),
            }
        )
    return out
