"""Main Didascalie project API."""

import sqlite3
import hashlib
import re
from pathlib import Path
from typing import Optional, Union, Iterator
import json
import io
import os
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Iterable

import numpy as np
from PIL import Image

from .schema import SCHEMA, SCHEMA_VERSION
from .models import (
    ProjectConfig,
    Label,
    Sequence,
    Frame,
    Annotation,
    Classification,
    TextDescription,
)
from .encoding import rle_encode, rle_decode, mask_to_png_bytes, png_bytes_to_mask, rle_encode_fast


def _encode_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    relative_path: Optional[str],
    embed: bool,
    format: str,
) -> tuple[Optional[str], Optional[str], Optional[bytes], int, int]:
    """
    Everything a ``frames`` row needs from an image:
    ``(relative_path, content_hash, embedded_data, width, height)``.

    Touches no database, so it can run on a worker thread.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
        if relative_path is None:
            relative_path = str(Path(image).name)
    elif isinstance(image, np.ndarray):
        # Handle different array formats
        if image.ndim == 2:
            img = Image.fromarray(image)
        elif image.ndim == 3:
            if image.shape[2] == 3:
                img = Image.fromarray(image, mode="RGB")
            elif image.shape[2] == 4:
                img = Image.fromarray(image, mode="RGBA")
            else:
                raise ValueError(f"Unsupported array shape: {image.shape}")
        else:
            raise ValueError(f"Unsupported array dimensions: {image.ndim}")
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    width, height = img.size

    embedded_data = None
    content_hash = None

    if embed:
        buffer = io.BytesIO()
        # Convert to RGB if needed for PNG
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        img.save(buffer, format=format)
        embedded_data = buffer.getvalue()
        content_hash = hashlib.sha256(embedded_data).hexdigest()

    return relative_path, content_hash, embedded_data, width, height


def _imap_ordered(pool, fn, items, pending: deque, lookahead: int):
    """
    ``map(fn, items)`` on ``pool``, in order, reading at most ``lookahead``
    items ahead of the one being yielded (``Executor.map`` reads them all up
    front, which for a video means holding every frame in memory).

    ``pending`` holds the futures still in flight, for the caller to cancel if
    it stops early.
    """
    for item in items:
        pending.append(pool.submit(fn, item))
        if len(pending) >= lookahead:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


class DidascalieProject:
    """
    Main interface for creating and manipulating Didascalie projects.

    Example:
        >>> project = DidascalieProject.create("dataset.dida", name="My Dataset")
        >>> project.add_label(Label(name="tumor", color="#FF0000"))
        >>> project.import_folder("/path/to/images")
        >>> project.close()
    """

    def __init__(self, db_path: Union[str, Path]):
        """Open an existing Didascalie project."""
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Project not found: {db_path}")

        self._connection = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._config: Optional[ProjectConfig] = None
        self._bulk_depth = 0
        self._check_schema_version()

    @property
    def schema_version(self) -> int:
        """Schema version stamped into the file (0 for pre-versioning files)."""
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def _check_schema_version(self) -> None:
        """
        Validate the on-disk schema version against what this package supports.

        - version 0: file predates schema versioning; the current schema is
          compatible, so stamp it and move on.
        - version > SCHEMA_VERSION: file was written by a newer release; refuse
          to open rather than silently corrupt it.
        - 0 < version <= SCHEMA_VERSION: compatible (add migrations here as the
          schema evolves).
        """
        version = self.schema_version
        if version == 0:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._commit()
            return
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"{self.db_path} was created with schema version {version}, but "
                f"this version of the library only supports up to {SCHEMA_VERSION}. "
                f"Please upgrade the package."
            )

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        name: str = "",
        config: Optional[ProjectConfig] = None,
        overwrite: bool = False,
    ) -> "DidascalieProject":
        """
        Create a new Didascalie project.

        Args:
            path: Path to the .dida file
            name: Project name
            config: Optional project configuration
            overwrite: If True, overwrite existing file

        Returns:
            DidascalieProject instance
        """
        path = Path(path)

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(f"Project already exists: {path}")

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create database
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        # Stamp the schema version into the database header (SQLite user_version).
        # SCHEMA_VERSION is a trusted integer constant, so inlining it is safe.
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

        # Insert config
        if config is None:
            config = ProjectConfig(name=name)
        elif name:
            config.name = name

        conn.execute(
            "INSERT INTO project (id, config) VALUES (1, ?)", (config.to_json(),)
        )
        conn.commit()
        conn.close()

        return cls(path)

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError(
                f"{self.db_path} is closed (close() was called, or a `with project:` "
                f"block has exited). Reopen it with DidascalieProject(path)."
            )
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "DidascalieProject":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ==========================================
    # Transactions
    # ==========================================

    def _commit(self) -> None:
        """Commit, unless a ``bulk()`` block is open (it commits once, on exit)."""
        if self._bulk_depth == 0:
            self._conn.commit()

    @contextmanager
    def bulk(self):
        """
        Group every write made inside the block into a single transaction.

        Each write otherwise commits on its own, and a commit is a synchronous
        write to disk: tens of milliseconds on a hard drive, which is most of
        the time of an import. Inside the block nothing is committed until it
        exits; if it raises, everything written in it is rolled back.

        Blocks nest: only the outermost one commits or rolls back.

        Example:
            >>> with project.bulk():
            ...     for image, masks in items:
            ...         project.import_with_masks(image, masks, sequence_name="video")
        """
        self._bulk_depth += 1
        try:
            yield self
        except BaseException:
            # A closed project has nothing to roll back, and failing here
            # would hide the error that got us here.
            if self._bulk_depth == 1 and self._connection is not None:
                self._connection.rollback()
            raise
        else:
            if self._bulk_depth == 1:
                self._conn.commit()
        finally:
            self._bulk_depth -= 1

    # ==========================================
    # Config
    # ==========================================

    @property
    def config(self) -> ProjectConfig:
        """Get project configuration."""
        if self._config is None:
            row = self._conn.execute(
                "SELECT config FROM project WHERE id = 1"
            ).fetchone()
            self._config = ProjectConfig.from_json(row["config"])
        return self._config

    def update_config(self, config: ProjectConfig) -> None:
        """Update project configuration."""
        self._conn.execute(
            "UPDATE project SET config = ? WHERE id = 1", (config.to_json(),)
        )
        self._commit()
        self._config = config

    # ==========================================
    # Labels
    # ==========================================

    def add_label(self, label: Label) -> int:
        """Add a label and return its ID."""
        # Get next sort order if not specified
        if label.sort_order == 0:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM labels"
            ).fetchone()
            label.sort_order = row[0]

        cursor = self._conn.execute(
            """INSERT INTO labels (name, color, is_instance, sort_order)
               VALUES (?, ?, ?, ?)""",
            (label.name, label.color, label.is_instance, label.sort_order),
        )
        self._commit()
        label.id = cursor.lastrowid
        return cursor.lastrowid

    def get_labels(self) -> list[Label]:
        """Get all labels."""
        rows = self._conn.execute(
            "SELECT * FROM labels ORDER BY sort_order"
        ).fetchall()
        return [
            Label(
                id=row["id"],
                name=row["name"],
                color=row["color"],
                is_instance=bool(row["is_instance"]),
                sort_order=row["sort_order"],
            )
            for row in rows
        ]

    def get_label_by_name(self, name: str) -> Optional[Label]:
        """Get a label by name."""
        row = self._conn.execute(
            "SELECT * FROM labels WHERE name = ?", (name,)
        ).fetchone()

        if row is None:
            return None

        return Label(
            id=row["id"],
            name=row["name"],
            color=row["color"],
            is_instance=bool(row["is_instance"]),
            sort_order=row["sort_order"],
        )

    def get_label_by_id(self, label_id: int) -> Optional[Label]:
        """Get a label by ID."""
        row = self._conn.execute(
            "SELECT * FROM labels WHERE id = ?", (label_id,)
        ).fetchone()

        if row is None:
            return None

        return Label(
            id=row["id"],
            name=row["name"],
            color=row["color"],
            is_instance=bool(row["is_instance"]),
            sort_order=row["sort_order"],
        )

    def get_or_create_label(
        self, name: str, color: str = "#FF0000", is_instance: bool = False
    ) -> Label:
        """Get existing label or create new one."""
        label = self.get_label_by_name(name)
        if label is None:
            label = Label(name=name, color=color, is_instance=is_instance)
            label.id = self.add_label(label)
        return label

    def update_label(self, label: Label) -> None:
        """Update an existing label."""
        self._conn.execute(
            """UPDATE labels SET name = ?, color = ?, is_instance = ?, sort_order = ?
               WHERE id = ?""",
            (label.name, label.color, label.is_instance, label.sort_order, label.id),
        )
        self._commit()

    def delete_label(self, label_id: int) -> None:
        """Delete a label and its annotations."""
        self._conn.execute("DELETE FROM labels WHERE id = ?", (label_id,))
        self._commit()

    # ==========================================
    # Sequences
    # ==========================================

    def add_sequence(self, name: str, sort_order: Optional[int] = None) -> int:
        """Add a sequence and return its ID."""
        if sort_order is None:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM sequences"
            ).fetchone()
            sort_order = row[0]

        cursor = self._conn.execute(
            "INSERT INTO sequences (name, sort_order) VALUES (?, ?)", (name, sort_order)
        )
        self._commit()
        return cursor.lastrowid

    def get_sequences(self) -> list[Sequence]:
        """Get all sequences with frame counts."""
        rows = self._conn.execute(
            """SELECT s.*, COUNT(f.id) as frame_count
               FROM sequences s
               LEFT JOIN frames f ON f.sequence_id = s.id
               GROUP BY s.id
               ORDER BY s.sort_order"""
        ).fetchall()

        return [
            Sequence(
                id=row["id"],
                name=row["name"],
                sort_order=row["sort_order"],
                frame_count=row["frame_count"],
            )
            for row in rows
        ]

    def get_sequence_by_name(self, name: str) -> Optional[Sequence]:
        """Get a sequence by name."""
        row = self._conn.execute(
            """SELECT s.*, COUNT(f.id) as frame_count
               FROM sequences s
               LEFT JOIN frames f ON f.sequence_id = s.id
               WHERE s.name = ?
               GROUP BY s.id""",
            (name,),
        ).fetchone()

        if row is None:
            return None

        return Sequence(
            id=row["id"],
            name=row["name"],
            sort_order=row["sort_order"],
            frame_count=row["frame_count"],
        )

    def get_or_create_sequence(self, name: str) -> Sequence:
        """Get existing sequence or create new one."""
        seq = self.get_sequence_by_name(name)
        if seq is None:
            seq_id = self.add_sequence(name)
            seq = Sequence(id=seq_id, name=name)
        return seq

    def delete_sequence(self, sequence_id: int) -> None:
        """Delete a sequence and all its frames."""
        self._conn.execute("DELETE FROM sequences WHERE id = ?", (sequence_id,))
        self._commit()

    # ==========================================
    # Frames
    # ==========================================

    def add_frame(
        self,
        sequence_id: int,
        image: Union[str, Path, np.ndarray, Image.Image],
        frame_index: Optional[int] = None,
        relative_path: Optional[str] = None,
        embed: bool = True,
        format: str = "PNG",
    ) -> int:
        """
        Add a frame to a sequence.

        Args:
            sequence_id: ID of the parent sequence
            image: Image path, numpy array, or PIL Image
            frame_index: Index within sequence (auto-incremented if None)
            relative_path: Stored path (for non-embedded)
            embed: Whether to embed the image data

        Returns:
            Frame ID
        """
        # Get next frame index if not specified
        if frame_index is None:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(frame_index), -1) + 1 FROM frames WHERE sequence_id = ?",
                (sequence_id,),
            ).fetchone()
            frame_index = row[0]

        relative_path, content_hash, embedded_data, width, height = _encode_image(
            image, relative_path, embed, format
        )

        # Insert
        cursor = self._conn.execute(
            """INSERT INTO frames
               (sequence_id, frame_index, relative_path, content_hash,
                embedded_data, width, height, reviewed)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                sequence_id,
                frame_index,
                relative_path,
                content_hash,
                embedded_data,
                width,
                height,
            ),
        )
        self._commit()
        return cursor.lastrowid

    def get_frames(self, sequence_id: int) -> list[Frame]:
        """Get all frames in a sequence."""
        rows = self._conn.execute(
            """SELECT id, sequence_id, frame_index, relative_path,
                      content_hash, width, height, reviewed
               FROM frames WHERE sequence_id = ?
               ORDER BY frame_index""",
            (sequence_id,),
        ).fetchall()

        return [
            Frame(
                id=row["id"],
                sequence_id=row["sequence_id"],
                frame_index=row["frame_index"],
                relative_path=row["relative_path"],
                content_hash=row["content_hash"],
                width=row["width"],
                height=row["height"],
                reviewed=bool(row["reviewed"]),
            )
            for row in rows
        ]

    def get_frame(self, frame_id: int) -> Optional[Frame]:
        """Get a frame by ID."""
        row = self._conn.execute(
            """SELECT id, sequence_id, frame_index, relative_path,
                      content_hash, width, height, reviewed
               FROM frames WHERE id = ?""",
            (frame_id,),
        ).fetchone()

        if row is None:
            return None

        return Frame(
            id=row["id"],
            sequence_id=row["sequence_id"],
            frame_index=row["frame_index"],
            relative_path=row["relative_path"],
            content_hash=row["content_hash"],
            width=row["width"],
            height=row["height"],
            reviewed=bool(row["reviewed"]),
        )

    def get_frame_count(self) -> int:
        """Get total number of frames."""
        row = self._conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        return row[0]

    def get_frame_image(self, frame_id: int, input_folder: Optional[str] = None) -> Image.Image:
        """
        Get the image for a frame.

        Args:
            frame_id: Frame ID
            input_folder: Base folder for non-embedded images

        Returns:
            PIL Image
        """
        row = self._conn.execute(
            "SELECT embedded_data, relative_path FROM frames WHERE id = ?",
            (frame_id,),
        ).fetchone()

        if row is None:
            raise ValueError(f"Frame not found: {frame_id}")

        if row["embedded_data"]:
            return Image.open(io.BytesIO(row["embedded_data"]))
        elif row["relative_path"] and input_folder:
            path = Path(input_folder) / row["relative_path"]
            return Image.open(path)
        else:
            raise ValueError("Frame has no embedded data and no input folder provided")

    def set_frame_reviewed(self, frame_id: int, reviewed: bool = True) -> None:
        """Mark a frame as reviewed."""
        self._conn.execute(
            "UPDATE frames SET reviewed = ? WHERE id = ?", (reviewed, frame_id)
        )
        self._commit()

    def delete_frame(self, frame_id: int) -> None:
        """Delete a frame and its annotations."""
        self._conn.execute("DELETE FROM frames WHERE id = ?", (frame_id,))
        self._commit()

    # ==========================================
    # Annotations
    # ==========================================

    def add_annotation(
        self,
        frame_id: int,
        label_id: int,
        mask: np.ndarray,
        encoding: str = "rle",
    ) -> int:
        """
        Add a segmentation annotation.

        Args:
            frame_id: Frame ID
            label_id: Label ID
            mask: Binary mask array (H x W), values 0 or 255 (or any non-zero)
            encoding: Encoding type ('rle' or 'png')

        Returns:
            Annotation ID
        """
        if encoding == "rle":
            mask_data = rle_encode_fast(mask)
        else:
            label = self.get_label_by_id(label_id)
            color = label.color if label else "#FF0000"
            mask_data = mask_to_png_bytes(mask, color)

        cursor = self._conn.execute(
            """INSERT OR REPLACE INTO annotations
               (frame_id, label_id, encoding, mask_data, modified_at)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            (frame_id, label_id, encoding, mask_data),
        )
        self._commit()
        return cursor.lastrowid

    def get_annotation(
        self,
        frame_id: int,
        label_id: int,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        """
        Get annotation mask for a frame/label pair.

        Args:
            frame_id: Frame ID
            label_id: Label ID
            width: Image width
            height: Image height

        Returns:
            Binary mask array (H x W) with values 0 or 255, or None if not found
        """
        row = self._conn.execute(
            "SELECT encoding, mask_data FROM annotations WHERE frame_id = ? AND label_id = ?",
            (frame_id, label_id),
        ).fetchone()

        if row is None:
            return None

        if row["encoding"] == "rle":
            return rle_decode(row["mask_data"], width, height)
        else:
            return png_bytes_to_mask(row["mask_data"])

    def get_annotations_for_frame(self, frame_id: int) -> list[tuple[Label, np.ndarray]]:
        """
        Get all annotations for a frame.

        Args:
            frame_id: Frame ID

        Returns:
            List of (Label, mask) tuples
        """
        frame = self.get_frame(frame_id)
        if frame is None:
            return []

        results = []
        rows = self._conn.execute(
            """SELECT a.label_id, a.encoding, a.mask_data, l.name, l.color, l.is_instance
               FROM annotations a
               JOIN labels l ON l.id = a.label_id
               WHERE a.frame_id = ?""",
            (frame_id,),
        ).fetchall()

        for row in rows:
            label = Label(
                id=row["label_id"],
                name=row["name"],
                color=row["color"],
                is_instance=bool(row["is_instance"]),
            )

            if row["encoding"] == "rle":
                mask = rle_decode(row["mask_data"], frame.width, frame.height)
            else:
                mask = png_bytes_to_mask(row["mask_data"])

            results.append((label, mask))

        return results

    def delete_annotation(self, frame_id: int, label_id: int) -> None:
        """Delete an annotation."""
        self._conn.execute(
            "DELETE FROM annotations WHERE frame_id = ? AND label_id = ?",
            (frame_id, label_id),
        )
        self._commit()

    # ==========================================
    # Classifications
    # ==========================================

    def add_classification(
        self,
        frame_id: int,
        task_name: str,
        selected_classes: list[str],
        is_multilabel: bool = False,
    ) -> int:
        """Add a classification annotation."""
        cursor = self._conn.execute(
            """INSERT OR REPLACE INTO classifications
               (frame_id, task_name, selected_classes, is_multilabel, modified_at)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            (frame_id, task_name, json.dumps(selected_classes), is_multilabel),
        )
        self._commit()
        return cursor.lastrowid

    def get_classification(
        self, frame_id: int, task_name: str
    ) -> Optional[Classification]:
        """Get classification for a frame/task pair."""
        row = self._conn.execute(
            "SELECT * FROM classifications WHERE frame_id = ? AND task_name = ?",
            (frame_id, task_name),
        ).fetchone()

        if row is None:
            return None

        return Classification(
            id=row["id"],
            frame_id=row["frame_id"],
            task_name=row["task_name"],
            selected_classes=json.loads(row["selected_classes"]),
            is_multilabel=bool(row["is_multilabel"]),
        )

    def get_classifications_for_frame(self, frame_id: int) -> list[Classification]:
        """Get all classifications for a frame."""
        rows = self._conn.execute(
            "SELECT * FROM classifications WHERE frame_id = ?", (frame_id,)
        ).fetchall()

        return [
            Classification(
                id=row["id"],
                frame_id=row["frame_id"],
                task_name=row["task_name"],
                selected_classes=json.loads(row["selected_classes"]),
                is_multilabel=bool(row["is_multilabel"]),
            )
            for row in rows
        ]

    def delete_classification(self, frame_id: int, task_name: str) -> None:
        """Delete a classification."""
        self._conn.execute(
            "DELETE FROM classifications WHERE frame_id = ? AND task_name = ?",
            (frame_id, task_name),
        )
        self._commit()

    # ==========================================
    # Text Descriptions
    # ==========================================

    def add_text_description(
        self, frame_id: int, label_name: str, content: str
    ) -> int:
        """Add a text description."""
        cursor = self._conn.execute(
            """INSERT OR REPLACE INTO text_descriptions
               (frame_id, label_name, content, modified_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (frame_id, label_name, content),
        )
        self._commit()
        return cursor.lastrowid

    def get_text_description(
        self, frame_id: int, label_name: str
    ) -> Optional[TextDescription]:
        """Get text description for a frame/label pair."""
        row = self._conn.execute(
            "SELECT * FROM text_descriptions WHERE frame_id = ? AND label_name = ?",
            (frame_id, label_name),
        ).fetchone()

        if row is None:
            return None

        return TextDescription(
            id=row["id"],
            frame_id=row["frame_id"],
            label_name=row["label_name"],
            content=row["content"],
        )

    def get_text_descriptions_for_frame(self, frame_id: int) -> list[TextDescription]:
        """Get all text descriptions for a frame."""
        rows = self._conn.execute(
            "SELECT * FROM text_descriptions WHERE frame_id = ?", (frame_id,)
        ).fetchall()

        return [
            TextDescription(
                id=row["id"],
                frame_id=row["frame_id"],
                label_name=row["label_name"],
                content=row["content"],
            )
            for row in rows
        ]

    # ==========================================
    # Registrations (keypoint pairs between two frames)
    # ==========================================

    def _ensure_registration_tables(self) -> None:
        """Projects written before registrations existed lack the tables; the
        schema is all ``CREATE ... IF NOT EXISTS``, so re-applying it adds them."""
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'keypoint_pairs'"
        ).fetchone()
        if exists is None:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    self._conn.execute(statement)

    def add_registration(
        self,
        reference_frame_id: int,
        moving_frame_id: int,
        pairs,
        homography: Optional[list[float]] = None,
        transform_type: str = "homography",
    ) -> int:
        """
        Store keypoint correspondences between two frames of a sequence, as the
        application's registration view does (it lists them as a case of that
        sequence and shows the pairs on the two frames).

        Replaces the pairs of an existing registration of the same two frames.

        Args:
            reference_frame_id: Frame the ``ref`` points are on
            moving_frame_id: Frame the ``moving`` points are on (another frame
                of the same sequence)
            pairs: ``(N, 4)`` array-like of ``(ref_x, ref_y, moving_x, moving_y)``
                in pixels, sub-pixel allowed
            homography: Optional 3x3 homography, 9 floats row-major
            transform_type: Transform the application fits to the pairs

        Returns:
            Registration ID
        """
        pairs = np.asarray(pairs, dtype=float).reshape(-1, 4)
        if not np.isfinite(pairs).all():
            raise ValueError("Keypoint coordinates must be finite")
        if homography is not None:
            homography = [float(v) for v in np.asarray(homography).ravel()]
            if len(homography) != 9:
                raise ValueError("homography must hold 9 values")

        rows = self._conn.execute(
            "SELECT id, sequence_id FROM frames WHERE id IN (?, ?)",
            (reference_frame_id, moving_frame_id),
        ).fetchall()
        sequences = {row["id"]: row["sequence_id"] for row in rows}
        if len(sequences) != 2:
            raise ValueError("A registration needs two distinct, existing frames")
        if len(set(sequences.values())) != 1:
            raise ValueError("Both frames must belong to the same sequence")

        with self.bulk():
            self._ensure_registration_tables()
            self._conn.execute(
                """INSERT INTO registrations
                   (sequence_id, reference_frame_id, moving_frame_id, homography,
                    transform_type, modified_at)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(reference_frame_id, moving_frame_id)
                   DO UPDATE SET
                      homography = excluded.homography,
                      transform_type = excluded.transform_type,
                      modified_at = CURRENT_TIMESTAMP""",
                (
                    sequences[reference_frame_id],
                    reference_frame_id,
                    moving_frame_id,
                    json.dumps(homography),
                    transform_type,
                ),
            )
            registration_id = self._conn.execute(
                """SELECT id FROM registrations
                   WHERE reference_frame_id = ? AND moving_frame_id = ?""",
                (reference_frame_id, moving_frame_id),
            ).fetchone()[0]
            self._conn.execute(
                "DELETE FROM keypoint_pairs WHERE registration_id = ?", (registration_id,)
            )
            self._conn.executemany(
                """INSERT INTO keypoint_pairs
                   (registration_id, client_uuid, ref_x, ref_y, moving_x, moving_y,
                    sort_order, modified_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                [
                    (registration_id, str(uuid.uuid4()), *map(float, pair), idx)
                    for idx, pair in enumerate(pairs)
                ],
            )
        return registration_id

    def get_registrations(self, sequence_id: int) -> list[dict]:
        """
        Registrations of a sequence, each as a dict with ``reference_frame_id``,
        ``moving_frame_id``, ``transform_type``, ``homography`` (9 floats or
        None) and ``pairs``, an ``(N, 4)`` array of
        ``(ref_x, ref_y, moving_x, moving_y)``.
        """
        self._ensure_registration_tables()
        out = []
        for row in self._conn.execute(
            """SELECT id, reference_frame_id, moving_frame_id, homography, transform_type
               FROM registrations WHERE sequence_id = ? ORDER BY id""",
            (sequence_id,),
        ).fetchall():
            pairs = self._conn.execute(
                """SELECT ref_x, ref_y, moving_x, moving_y FROM keypoint_pairs
                   WHERE registration_id = ? ORDER BY sort_order""",
                (row["id"],),
            ).fetchall()
            out.append(
                {
                    "reference_frame_id": row["reference_frame_id"],
                    "moving_frame_id": row["moving_frame_id"],
                    "transform_type": row["transform_type"],
                    "homography": json.loads(row["homography"] or "null"),
                    "pairs": np.array([tuple(p) for p in pairs], dtype=float).reshape(-1, 4),
                }
            )
        return out

    # ==========================================
    # Bulk Import
    # ==========================================

    def import_folder(
        self,
        folder: Union[str, Path],
        pattern: Optional[str] = None,
        recursive: bool = True,
        folders_as_sequences: bool = False,
        embed: bool = True,
    ) -> dict:
        """
        Import images from a folder.

        Args:
            folder: Path to image folder
            pattern: Regex pattern for filtering files
            recursive: Search subfolders
            folders_as_sequences: Group subfolders as sequences
            embed: Embed images in database

        Returns:
            Dict with import statistics
        """
        folder = Path(folder)
        pattern = pattern or self.config.input_regex
        regex = re.compile(pattern, re.IGNORECASE)

        stats = {"sequences": 0, "frames": 0, "errors": []}

        # Collect images
        images: dict[str, list[Path]] = {}  # sequence_name -> [paths]

        glob_pattern = "**/*" if recursive else "*"
        for path in folder.glob(glob_pattern):
            if not path.is_file():
                continue
            if not regex.search(path.name):
                continue

            # Determine sequence name
            if folders_as_sequences and path.parent != folder:
                seq_name = path.parent.name
            else:
                seq_name = path.stem

            images.setdefault(seq_name, []).append(path)

        # Sort and import, in a single transaction
        with self.bulk():
            for seq_name in sorted(images.keys()):
                paths = sorted(images[seq_name])

                try:
                    seq_id = self.add_sequence(seq_name)
                    stats["sequences"] += 1

                    for idx, img_path in enumerate(paths):
                        try:
                            self.add_frame(
                                sequence_id=seq_id,
                                image=img_path,
                                frame_index=idx,
                                relative_path=str(img_path.relative_to(folder)),
                                embed=embed,
                            )
                            stats["frames"] += 1
                        except Exception as e:
                            stats["errors"].append(f"{img_path}: {e}")

                except Exception as e:
                    stats["errors"].append(f"Sequence {seq_name}: {e}")

        return stats

    def import_with_masks(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        masks: dict[str, np.ndarray],
        sequence_name: Optional[str] = None,
        classification: Optional[dict[str, list[str]]] = None,
        text_descriptions: Optional[dict[str, str]] = None,
        embed: bool = True,
    ) -> int:
        """
        Import an image with pre-existing masks.

        Args:
            image: Image path or array
            masks: Dict mapping label_name -> mask_array
            sequence_name: Sequence name (uses image filename if None)
            classification: Optional dict mapping task_name -> selected_classes
            text_descriptions: Optional dict mapping label_name -> text
            embed: Embed image in database

        Returns:
            Frame ID
        """
        # Determine sequence name
        if sequence_name is None:
            if isinstance(image, (str, Path)):
                sequence_name = Path(image).stem
            else:
                sequence_name = f"image_{self.get_frame_count()}"

        # One transaction for the frame and everything attached to it
        with self.bulk():
            # Get or create sequence
            sequence = self.get_or_create_sequence(sequence_name)

            # Add frame
            frame_id = self.add_frame(sequence.id, image, embed=embed)

            # Add masks
            for label_name, mask in masks.items():
                label = self.get_or_create_label(label_name)
                self.add_annotation(frame_id, label.id, mask)

            # Add classifications
            if classification:
                for task_name, classes in classification.items():
                    is_multilabel = len(classes) > 1
                    self.add_classification(frame_id, task_name, classes, is_multilabel)

            # Add text descriptions
            if text_descriptions:
                for label_name, text in text_descriptions.items():
                    self.add_text_description(frame_id, label_name, text)

        return frame_id

    def import_sequence(
        self,
        name: str,
        frames: Iterable,
        embed: bool = True,
        format: str = "PNG",
        workers: Optional[int] = None,
    ) -> list[int]:
        """
        Import a whole sequence (a video, a volume) in one go.

        The fast path for many frames: images and masks are encoded on a thread
        pool and everything is written in a single transaction, so either the
        whole sequence is imported or, if anything fails, none of it.

        Args:
            name: Sequence name; frames are appended if it already exists
            frames: Iterable of images, or of ``(image, masks)`` pairs where
                ``masks`` maps label_name -> mask_array. Consumed lazily, a few
                frames ahead of the writes, so it can be a generator.
            embed: Embed images in database
            format: Format of the embedded images
            workers: Encoding threads (defaults to the CPU count, at most 8)

        Returns:
            Frame IDs, in order
        """
        if workers is None:
            workers = min(8, os.cpu_count() or 1)

        def encode(item):
            image, masks = item if isinstance(item, tuple) else (item, None)
            return (
                _encode_image(image, None, embed, format),
                {k: rle_encode_fast(np.asarray(m)) for k, m in (masks or {}).items()},
            )

        frame_ids = []
        label_ids: dict[str, int] = {}
        with self.bulk(), ThreadPoolExecutor(max_workers=workers) as pool:
            sequence_id = self.get_or_create_sequence(name).id
            frame_index = self._conn.execute(
                "SELECT COALESCE(MAX(frame_index), -1) + 1 FROM frames WHERE sequence_id = ?",
                (sequence_id,),
            ).fetchone()[0]

            pending: deque = deque()
            try:
                for encoded in _imap_ordered(pool, encode, frames, pending, 2 * workers):
                    (relative_path, content_hash, data, width, height), masks = encoded
                    frame_id = self._conn.execute(
                        """INSERT INTO frames
                           (sequence_id, frame_index, relative_path, content_hash,
                            embedded_data, width, height, reviewed)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                        (
                            sequence_id,
                            frame_index,
                            relative_path,
                            content_hash,
                            data,
                            width,
                            height,
                        ),
                    ).lastrowid
                    for label_name, mask_data in masks.items():
                        if label_name not in label_ids:
                            label_ids[label_name] = self.get_or_create_label(label_name).id
                        self._conn.execute(
                            """INSERT OR REPLACE INTO annotations
                               (frame_id, label_id, encoding, mask_data, modified_at)
                               VALUES (?, ?, 'rle', ?, datetime('now'))""",
                            (frame_id, label_ids[label_name], mask_data),
                        )
                    frame_ids.append(frame_id)
                    frame_index += 1
            finally:
                for future in pending:
                    future.cancel()

        return frame_ids

    # ==========================================
    # Iteration
    # ==========================================

    def iter_frames(self) -> Iterator[tuple[Sequence, Frame]]:
        """Iterate over all frames with their sequences."""
        for seq in self.get_sequences():
            for frame in self.get_frames(seq.id):
                yield seq, frame

    # ==========================================
    # Statistics
    # ==========================================

    def get_statistics(self) -> dict:
        """Get project statistics."""
        return {
            "sequences": len(self.get_sequences()),
            "frames": self.get_frame_count(),
            "labels": len(self.get_labels()),
            "annotations": self._conn.execute(
                "SELECT COUNT(*) FROM annotations"
            ).fetchone()[0],
            "classifications": self._conn.execute(
                "SELECT COUNT(*) FROM classifications"
            ).fetchone()[0],
            "reviewed": self._conn.execute(
                "SELECT COUNT(*) FROM frames WHERE reviewed = 1"
            ).fetchone()[0],
        }


# Backwards-compatible alias (deprecated): the class was renamed
# LabelMedProject -> DidascalieProject when the library was rebranded to Didascalie.
LabelMedProject = DidascalieProject
