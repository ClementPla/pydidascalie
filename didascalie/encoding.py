"""Mask encoding utilities (RLE) - Compatible with Didascalie Rust backend."""

import numpy as np
import io


def rle_encode(mask: np.ndarray) -> bytes:
    """
    Run-length encode a binary mask (COCO-style column-major).
    
    Compatible with Didascalie Rust backend.
    """
    height, width = mask.shape
    
    if height == 0 or width == 0:
        return b""
    
    # Column-major traversal (COCO-style)
    rle: list[int] = []
    count = 0
    current = 0
    
    for x in range(width):
        for y in range(height):
            val = 1 if mask[y, x] > 0 else 0
            if val == current:
                count += 1
            else:
                rle.append(count)
                count = 1
                current = val
    
    rle.append(count)
    
    # Pack as u32 little-endian
    result = bytearray()
    for length in rle:
        result.extend(length.to_bytes(4, "little"))
    
    return bytes(result)


def rle_encode_fast(mask: np.ndarray) -> bytes:
    """
    Optimized RLE encode using numpy (COCO-style column-major).
    """
    height, width = mask.shape
    
    if height == 0 or width == 0:
        return b""
    
    # Column-major flatten (Fortran order)
    flat = (mask.flatten(order='F') > 0).astype(np.uint8)
    
    if len(flat) == 0:
        return b""
    
    # Find transition indices
    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1  # +1 because diff shifts by 1
    
    # Build boundaries: [0, change1, change2, ..., len]
    boundaries = np.concatenate([[0], change_indices, [len(flat)]])
    
    # Run lengths are differences between boundaries
    run_lengths = np.diff(boundaries)
    
    # If starts with 1, prepend 0 (count of leading zeros = 0)
    if flat[0] == 1:
        run_lengths = np.concatenate([[0], run_lengths])
    
    return run_lengths.astype('<u4').tobytes()


# Value-aware RLE (`rle8`): the encoding the application writes.
#
# Unlike the binary codec above, it keeps the pixel value: 0 is background, 1 a
# semantic label, and 1..255 the instance ids of an instance label. Runs are
# stored row-major as `[value: u8][count: u32 little-endian]`.
_RLE8_RUN = np.dtype([("value", "u1"), ("count", "<u4")])


def rle8_encode(mask: np.ndarray) -> bytes:
    """
    Encode a uint8 value mask (H x W) as `rle8`.

    The values are stored as they are, so pass 0/1 for a semantic label and
    instance ids for an instance label.
    """
    flat = np.ascontiguousarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return b""

    starts = np.concatenate([[0], np.flatnonzero(np.diff(flat)) + 1])
    runs = np.empty(len(starts), dtype=_RLE8_RUN)
    runs["value"] = flat[starts]
    runs["count"] = np.diff(np.concatenate([starts, [flat.size]]))
    return runs.tobytes()


def rle8_decode(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode `rle8` data to a uint8 value mask (H x W)."""
    total = width * height
    runs = np.frombuffer(data, dtype=_RLE8_RUN, count=len(data) // _RLE8_RUN.itemsize)
    flat = np.repeat(runs["value"], runs["count"])[:total]
    if flat.size < total:
        flat = np.concatenate([flat, np.zeros(total - flat.size, dtype=np.uint8)])
    return flat.reshape((height, width))


def instance_mask(mask: np.ndarray) -> np.ndarray:
    """
    Check an instance-id mask and return it as uint8.

    Ids are stored in one byte per pixel, so a label holds at most 255
    instances per frame.
    """
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return mask.astype(np.uint8)
    if mask.size and (mask.min() < 0 or mask.max() > 255):
        raise ValueError(
            "Instance ids must be between 0 and 255 "
            f"(got {mask.min()}..{mask.max()}); renumber the instances of each frame."
        )
    return mask.astype(np.uint8)


def rle_decode(data: bytes, width: int, height: int) -> np.ndarray:
    """
    Decode RLE data to binary mask (COCO-style column-major).
    """
    if len(data) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Parse u32 little-endian
    rle = np.frombuffer(data, dtype='<u4')
    
    # Decode column-major
    total = width * height
    flat = np.zeros(total, dtype=np.uint8)
    
    pos = 0
    val = 0  # Start with 0s
    
    for count in rle:
        if val == 1 and count > 0:
            end = min(pos + count, total)
            flat[pos:end] = 255
        pos += count
        val = 1 - val
    
    # Reshape from column-major to row-major
    # flat is in column-major order: [col0, col1, col2, ...]
    # Reshape as (width, height) then transpose to (height, width)
    return flat.reshape((width, height)).T.copy()


def rle_decode_fast(data: bytes, width: int, height: int) -> np.ndarray:
    """
    Optimized RLE decode using numpy (column-major, COCO-style).
    
    Args:
        data: RLE encoded bytes
        width: Image width
        height: Image height
    
    Returns:
        2D numpy array (H x W) with values 0 or 255
    """
    if len(data) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Parse run lengths
    rle = np.frombuffer(data, dtype='<u4')
    
    # Build flat mask
    total = width * height
    flat = np.zeros(total, dtype=np.uint8)
    
    pos = 0
    val = 0
    for count in rle:
        if val == 1:
            end = min(pos + count, total)
            flat[pos:end] = 255
        pos += count
        val = 1 - val
    
    # Reshape column-major to row-major
    return flat.reshape((width, height)).T.copy()


# Use fast versions by default
def mask_to_png_bytes(mask: np.ndarray, color: str = "#FF0000") -> bytes:
    """
    Convert a binary mask to PNG bytes with color.

    Args:
        mask: 2D numpy array (binary mask)
        color: Hex color string

    Returns:
        PNG image bytes
    """
    from PIL import Image

    # Parse color
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    # Create RGBA image
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    foreground = mask > 0
    rgba[foreground, 0] = r
    rgba[foreground, 1] = g
    rgba[foreground, 2] = b
    rgba[foreground, 3] = 255

    # Encode as PNG
    img = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def png_bytes_to_mask(data: bytes) -> np.ndarray:
    """
    Convert PNG bytes to binary mask.

    Args:
        data: PNG image bytes

    Returns:
        2D numpy array with values 0 or 255
    """
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    arr = np.array(img)

    if arr.ndim == 3:
        # Use alpha channel if RGBA
        if arr.shape[2] == 4:
            return (arr[:, :, 3] > 0).astype(np.uint8) * 255
        # Otherwise use any non-zero pixel
        return (arr.any(axis=2)).astype(np.uint8) * 255

    # Grayscale
    return (arr > 0).astype(np.uint8) * 255