"""
Frames deeper than 8 bits, or with several channels, and how the application
shows them.

The application keeps such frames at their native depth: a microscopy file
becomes a multi-page greyscale TIFF, one page per channel (uint8, uint16 or
float32), and a 16-bit PNG or TIFF is stored as it is. It shows them through a
window, colour and gamma per channel, added together into 8-bit RGB. The
functions here do the same, after ``src-tauri/src/pixels`` in the application.
"""

from typing import Optional

import numpy as np
from PIL import Image, ImageSequence

# Percentiles the automatic window clips at.
AUTO_LOW = 0.005
AUTO_HIGH = 0.995

# Colours of 2, and 4 or more, channels, in order. A single channel is grey;
# three are red, green and blue.
PALETTE = ["#00ff00", "#ff00ff", "#00ffff", "#ff0000", "#0000ff", "#ffff00"]

# Bins of a float channel's histogram, spread over its value range.
FLOAT_BINS = 4096


def _is_tiff(data: bytes) -> bool:
    return data[:4] in (b"II*\0", b"MM\0*")


def _page_array(page: Image.Image) -> Optional[np.ndarray]:
    """A single-sample page as a 2D array of its native type, else None."""
    if page.mode in ("L", "F"):
        return np.asarray(page)
    if page.mode.startswith("I;16"):
        return np.asarray(page).astype(np.uint16)
    if page.mode == "I":
        a = np.asarray(page)
        if a.size == 0 or (a.min() >= 0 and a.max() <= 0xFFFF):
            return a.astype(np.uint16)
        return a.astype(np.float32)
    return None


def decode_raw(image: Image.Image) -> np.ndarray:
    """
    The native values of an image, ``(height, width, channels)``.

    A TIFF whose pages are all single-sample images of the same size is read as
    one channel per page, as the microscopy import writes it. Anything else
    keeps its own channels, alpha dropped. The dtype is uint8, uint16 or
    float32.
    """
    if image.format == "TIFF":
        pages = []
        for page in ImageSequence.Iterator(image):
            a = _page_array(page)
            if a is None or (pages and a.shape != pages[0].shape):
                pages = None
                break
            pages.append(a)
        image.seek(0)
        if pages:
            if len({p.dtype for p in pages}) > 1:
                raise ValueError("TIFF pages differ in sample type")
            return np.stack(pages, axis=-1)

    single = _page_array(image)
    if single is not None:
        return single[..., None]
    if image.mode in ("LA", "La"):
        return np.asarray(image)[..., :1]
    if image.mode in ("RGB", "RGBA", "RGBa"):
        return np.asarray(image)[..., :3]
    return np.asarray(image.convert("RGB"))


def needs_render(image: Image.Image) -> bool:
    """
    Whether an image is shown through display settings rather than as stored:
    deeper than 8 bits, or several channels in TIFF pages.
    """
    if image.format == "TIFF" and getattr(image, "n_frames", 1) > 1:
        return True
    return image.mode in ("F", "I") or image.mode.startswith("I;16")


def _percentile(values: np.ndarray, q: float) -> float:
    """The value below which ``q`` (0-1) of ``values`` fall, as the app counts."""
    if values.dtype == np.float32:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        lo, hi = float(values.min()), float(values.max())
        width = (hi - lo) / FLOAT_BINS if hi > lo else 1.0
        bins = np.minimum(((values - lo) / width).astype(np.int64), FLOAT_BINS - 1)
        counts = np.bincount(bins, minlength=FLOAT_BINS)
        start = lo
    else:
        counts = np.bincount(values.ravel())
        start, width = 0.0, 1.0
    total = int(counts.sum())
    if total == 0:
        return start
    target = max(int(np.ceil(q * total)), 1)
    i = int(np.searchsorted(np.cumsum(counts), target))
    return start + i * width


def auto_window(channel: np.ndarray) -> tuple[float, float]:
    """The automatic window of a channel: its 0.5-99.5 % percentiles."""
    lo = _percentile(channel, AUTO_LOW)
    hi = _percentile(channel, AUTO_HIGH)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def auto_display(raw: np.ndarray) -> dict:
    """
    A starting display for a ``(height, width, channels)`` frame. An 8-bit frame
    is shown as stored (full range, linear); deeper data gets the automatic
    window.
    """
    n = raw.shape[-1]
    if n == 1:
        colors = ["#ffffff"]
    elif n == 3:
        colors = ["#ff0000", "#00ff00", "#0000ff"]
    else:
        colors = [PALETTE[c % len(PALETTE)] for c in range(n)]
    channels = []
    for c, color in enumerate(colors):
        if raw.dtype == np.uint8:
            lo, hi = 0.0, 255.0
        else:
            lo, hi = auto_window(raw[..., c])
        channels.append(
            {"name": None, "visible": True, "color": color, "min": lo, "max": hi, "gamma": 1.0}
        )
    return {"channels": channels}


def _parse_color(hex_color: str) -> np.ndarray:
    """``#rrggbb`` as 0-1 floats; white when malformed."""
    h = (hex_color or "").lstrip("#")
    try:
        if len(h) == 6:
            return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], np.float32) / 255
    except ValueError:
        pass
    return np.ones(3, np.float32)


def render_rgb(raw: np.ndarray, display: Optional[dict] = None) -> np.ndarray:
    """
    ``raw`` (``(height, width, channels)``) as 8-bit RGB: per visible channel,
    ``clamp((x - min) / (max - min), 0, 1) ** (1 / gamma)`` times its colour,
    added up and clipped. ``display`` is ``{"channels": [{"visible", "color",
    "min", "max", "gamma"}]}``; settings that do not have one entry per channel
    are replaced by ``auto_display(raw)``, as in the application.
    """
    if raw.ndim == 2:
        raw = raw[..., None]
    if display is None or len(display.get("channels") or []) != raw.shape[-1]:
        display = auto_display(raw)
    acc = np.zeros(raw.shape[:2] + (3,), np.float32)
    for c, d in enumerate(display["channels"]):
        if not d.get("visible", True):
            continue
        lo, hi = float(d["min"]), float(d["max"])
        inv_span = np.float32(1.0 / max(hi - lo, np.finfo(np.float64).eps))
        inv_gamma = np.float32(1.0 / max(float(d.get("gamma", 1.0)), 0.01))
        t = np.clip((raw[..., c].astype(np.float32) - np.float32(lo)) * inv_span, 0, 1)
        if inv_gamma != 1:
            t = t**inv_gamma
        acc += t[..., None] * _parse_color(d.get("color"))
    return np.round(np.clip(acc, 0, 1) * 255).astype(np.uint8)
