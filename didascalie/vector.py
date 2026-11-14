"""Vector annotations: Bézier paths, polygons and polylines drawn in the editor.

A shape is a list of nodes. Each node has a position and two handles (``in``
and ``out``), in image pixels; a segment goes from a node, through its ``out``
handle and the next node's ``in`` handle, to the next node. A node whose
handles sit on it makes straight segments.

The geometry and rasterization follow the application's export
(``src-tauri/src/commands/vector.rs``), so masks made here match the
application's.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence
import uuid

import numpy as np

# Handle length, as a fraction of the radius, that makes four cubic segments
# approximate a circle (as in the editor).
CIRCLE_KAPPA = 0.5522847498

# Half-width (px) of the stroke of open or unfilled shapes, as in the export.
STROKE_RADIUS = 1.5

# Points per curved segment when flattening, as in the export.
DEFAULT_SAMPLES = 24


@dataclass
class VectorNode:
    """A point of a path and its two Bézier handles, in image pixels."""

    x: float
    y: float
    in_x: float
    in_y: float
    out_x: float
    out_y: float
    # The two handles stay collinear when edited in the application.
    smooth: bool = False
    # Keys this version does not know about, written back untouched.
    extra: dict = field(default_factory=dict, repr=False)

    @classmethod
    def corner(cls, x: float, y: float) -> "VectorNode":
        """A node with its handles on it: straight segments on both sides."""
        return cls(x, y, x, y, x, y, smooth=False)

    @classmethod
    def from_dict(cls, d: dict) -> "VectorNode":
        known = {"x", "y", "inX", "inY", "outX", "outY", "smooth"}
        return cls(
            x=d["x"],
            y=d["y"],
            in_x=d.get("inX", d["x"]),
            in_y=d.get("inY", d["y"]),
            out_x=d.get("outX", d["x"]),
            out_y=d.get("outY", d["y"]),
            smooth=bool(d.get("smooth", False)),
            extra={k: v for k, v in d.items() if k not in known},
        )

    def to_dict(self) -> dict:
        return {
            **self.extra,
            "x": float(self.x),
            "y": float(self.y),
            "inX": float(self.in_x),
            "inY": float(self.in_y),
            "outX": float(self.out_x),
            "outY": float(self.out_y),
            "smooth": bool(self.smooth),
        }


@dataclass
class VectorShape:
    """
    One path of a vector annotation.

    ``closed`` joins the last node back to the first. ``filled`` (meaningful
    only when closed) tells the application to fill the shape when it is
    rasterized; other shapes are drawn as a thin stroke.
    """

    nodes: list = field(default_factory=list)
    closed: bool = True
    filled: bool = True
    label_id: Optional[int] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extra: dict = field(default_factory=dict, repr=False)

    # ---- construction

    @classmethod
    def from_points(
        cls,
        points: Iterable[Sequence[float]],
        closed: bool = True,
        filled: bool = True,
        label_id: Optional[int] = None,
    ) -> "VectorShape":
        """A polygon (or polyline when not closed) through ``points``, (x, y) pairs."""
        nodes = [VectorNode.corner(float(x), float(y)) for x, y in points]
        return cls(nodes=nodes, closed=closed, filled=filled, label_id=label_id)

    @classmethod
    def ellipse(
        cls,
        cx: float,
        cy: float,
        rx: float,
        ry: Optional[float] = None,
        filled: bool = True,
        label_id: Optional[int] = None,
    ) -> "VectorShape":
        """An ellipse (a circle when ``ry`` is omitted), built as the editor builds it."""
        ry = rx if ry is None else ry
        kx, ky = rx * CIRCLE_KAPPA, ry * CIRCLE_KAPPA

        def node(x, y, tx, ty):
            return VectorNode(x, y, x - tx, y - ty, x + tx, y + ty, smooth=True)

        nodes = [
            node(cx, cy - ry, kx, 0),
            node(cx + rx, cy, 0, ky),
            node(cx, cy + ry, -kx, 0),
            node(cx - rx, cy, 0, -ky),
        ]
        return cls(nodes=nodes, closed=True, filled=filled, label_id=label_id)

    @classmethod
    def from_dict(cls, d: dict) -> "VectorShape":
        known = {"id", "labelId", "closed", "filled", "nodes"}
        return cls(
            nodes=[VectorNode.from_dict(n) for n in d.get("nodes", [])],
            closed=bool(d.get("closed", False)),
            filled=bool(d.get("filled", False)),
            label_id=d.get("labelId"),
            id=d.get("id") or str(uuid.uuid4()),
            extra={k: v for k, v in d.items() if k not in known},
        )

    def to_dict(self) -> dict:
        return {
            **self.extra,
            "id": self.id,
            "labelId": self.label_id,
            "closed": bool(self.closed),
            "filled": bool(self.filled),
            "nodes": [n.to_dict() for n in self.nodes],
        }

    # ---- geometry

    def polygon(self, samples: int = DEFAULT_SAMPLES) -> np.ndarray:
        """
        The shape sampled as a polyline, ``(N, 2)`` array of (x, y) in image
        pixels. Curved segments get ``samples`` points, straight ones none in
        between. A closed shape's last point equals its first.
        """
        n = self.nodes
        if not n:
            return np.zeros((0, 2))
        pts = [np.array([[n[0].x, n[0].y]], float)]
        segs = len(n) if self.closed else len(n) - 1
        t = np.arange(1, samples + 1)[:, None] / samples
        for i in range(segs):
            a, b = n[i], n[(i + 1) % len(n)]
            p0 = np.array([a.x, a.y])
            p3 = np.array([b.x, b.y])
            straight = (a.out_x, a.out_y) == (a.x, a.y) and (b.in_x, b.in_y) == (b.x, b.y)
            if straight:
                pts.append(p3[None])
                continue
            p1 = np.array([a.out_x, a.out_y])
            p2 = np.array([b.in_x, b.in_y])
            mt = 1 - t
            pts.append(mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3)
        return np.concatenate(pts)

    def rasterize(
        self,
        width: int,
        height: int,
        fill: Optional[bool] = None,
        samples: int = DEFAULT_SAMPLES,
    ) -> np.ndarray:
        """
        The shape as a ``(height, width)`` uint8 mask, 0 or 255.

        Args:
            fill: None does what the application does: fill a shape that is
                closed and filled, stroke the others. True fills any closed
                shape (an open one is still stroked); False always strokes.
            samples: Points per curved segment.
        """
        mask = np.zeros((height, width), np.uint8)
        poly = self.polygon(samples)
        if len(poly) < 2:
            return mask
        do_fill = self.closed and (self.filled if fill is None else fill)
        if do_fill:
            _fill_polygon(poly, mask)
        else:
            _stroke_polyline(poly, mask, STROKE_RADIUS)
        return mask


def rasterize(
    shapes: Iterable[VectorShape],
    width: int,
    height: int,
    fill: Optional[bool] = None,
    samples: int = DEFAULT_SAMPLES,
) -> np.ndarray:
    """The union of several shapes as a ``(height, width)`` uint8 mask, 0 or 255.
    See ``VectorShape.rasterize`` for ``fill``."""
    mask = np.zeros((height, width), np.uint8)
    for s in shapes:
        mask |= s.rasterize(width, height, fill=fill, samples=samples)
    return mask


def _fill_polygon(poly: np.ndarray, mask: np.ndarray) -> None:
    """Even-odd scanline fill: a pixel is set when its centre is inside."""
    h, w = mask.shape
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    y0 = max(int(np.floor(y1.min())), 0)
    ye = min(int(np.ceil(y1.max())), h - 1)
    for y in range(y0, ye + 1):
        yc = y + 0.5
        cross = ((y1 <= yc) & (y2 > yc)) | ((y2 <= yc) & (y1 > yc))
        if not cross.any():
            continue
        t = (yc - y1[cross]) / (y2[cross] - y1[cross])
        xs = np.sort(x1[cross] + t * (x2[cross] - x1[cross]))
        for xa, xb in zip(xs[0::2], xs[1::2]):
            a = max(int(np.ceil(xa - 0.5)), 0)
            b = min(int(np.ceil(xb - 0.5)), w)
            if b > a:
                mask[y, a:b] = 255


def _stroke_polyline(poly: np.ndarray, mask: np.ndarray, r: float) -> None:
    """Set the pixels whose centre is within ``r`` of the polyline."""
    h, w = mask.shape
    r2 = r * r
    for a, b in zip(poly[:-1], poly[1:]):
        x0 = max(int(np.floor(min(a[0], b[0]) - r)), 0)
        x1 = min(int(np.ceil(max(a[0], b[0]) + r)), w - 1)
        y0 = max(int(np.floor(min(a[1], b[1]) - r)), 0)
        y1 = min(int(np.ceil(max(a[1], b[1]) + r)), h - 1)
        if x1 < x0 or y1 < y0:
            continue
        px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = b - a
        len2 = float(d @ d)
        t = 0.0 if len2 == 0 else np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / len2, 0, 1)
        ex, ey = px - (a[0] + t * d[0]), py - (a[1] + t * d[1])
        mask[y0 : y1 + 1, x0 : x1 + 1][ex * ex + ey * ey <= r2] = 255
