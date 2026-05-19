"""Main LabelMed project API."""

import sqlite3
import hashlib
import re
import io
import json
from pathlib import Path
from typing import Optional, Union, Iterator

import numpy as np
from PIL import Image

from .schema import SCHEMA
from .models import (
    ProjectConfig,
    Label,
    Sequence,
    Frame,
    Annotation,
    Classification,
    TextDescription,
)
from .encoding import rle_encode, rle_decode


class LabelMedProject:
    """
    Main interface for creating and manipulating LabelMed projects.

    Example:
        >>> project = LabelMedProject.create("dataset.labelmed", name="My Dataset")
        >>> project.add_label(Label(name="tumor", color="#FF0000"))
        >>> project.import_folder("/path/to/images")
        >>> project.close()
    """

    def __init__(self, db_path: Union[str, Path]):
        """Open an existing LabelMed project."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Project not found: {db_path}")
        self.db_path = Path(db_path).with_suffix(".labelmed")
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._config: Optional[ProjectConfig] = None

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        name: str = "",
        config: Optional[ProjectConfig] = None,
        overwrite: bool = False,
    ) -> "LabelMedProject":
        """Create a new LabelMed project."""
        path = Path(path).with_suffix(".labelmed")
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(f"Project already exists: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)

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

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "LabelMedProject":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ========================================== Config
    @property
    def config(self) -> ProjectConfig:
        if self._config is None:
            row = self._conn.execute(
                "SELECT config FROM project WHERE id = 1"
            ).fetchone()
            self._config = ProjectConfig.from_json(row["config"])
        return self._config

    def update_config(self, config: ProjectConfig) -> None:
        self._conn.execute(
            "UPDATE project SET config = ? WHERE id = 1", (config.to_json(),)
        )
        self._conn.commit()
        self._config = config

    # ========================================== Labels
    def add_label(self, label: Label) -> int:
        cursor = self._conn.execute(
            """INSERT INTO labels (name, color, is_instance, sort_order)
               VALUES (?, ?, ?, ?)""",
            (label.name, label.color, label.is_instance, label.sort_order),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_labels(self) -> list[Label]:
        rows = self._conn.execute("SELECT * FROM labels ORDER BY sort_order").fetchall()
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

    def _get_label_by_id(self, label_id: int) -> Optional[Label]:
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
        label = self.get_label_by_name(name)
        if label is None:
            label = Label(name=name, color=color, is_instance=is_instance)
            label.id = self.add_label(label)
        return label

    def update_label(self, label: Label) -> None:
        self._conn.execute(
            """UPDATE labels SET name = ?, color = ?, is_instance = ?, sort_order = ?
               WHERE id = ?""",
            (label.name, label.color, label.is_instance, label.sort_order, label.id),
        )
        self._conn.commit()

    def delete_label(self, label_id: int) -> None:
        self._conn.execute("DELETE FROM labels WHERE id = ?", (label_id,))
        self._conn.commit()

    # ========================================== Sequences
    def add_sequence(self, name: str, sort_order: Optional[int] = None) -> int:
        if sort_order is None:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM sequences"
            ).fetchone()
            sort_order = row[0]

        cursor = self._conn.execute(
            "INSERT INTO sequences (name, sort_order) VALUES (?, ?)",
            (name, sort_order),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_sequences(self) -> list[Sequence]:
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
        seq = self.get_sequence_by_name(name)
        if seq is None:
            seq_id = self.add_sequence(name)
            seq = Sequence(id=seq_id, name=name, sort_order=0, frame_count=0)
        return seq

    # ========================================== Frames
    def add_frame(
        self,
        sequence_id: int,
        image: Union[str, Path, np.ndarray, Image.Image],
        frame_index: Optional[int] = None,
        relative_path: Optional[str] = None,
        embed: bool = True,
    ) -> int:
        """Add a frame to a sequence."""
        if frame_index is None:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(frame_index), -1) + 1 FROM frames WHERE sequence_id = ?",
                (sequence_id,),
            ).fetchone()
            frame_index = row[0]

        if isinstance(image, (str, Path)):
            img = Image.open(image)
            if relative_path is None:
                relative_path = str(Path(image).name)
        elif isinstance(image, np.ndarray):
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
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGB")
            img.save(buffer, format="PNG")
            embedded_data = buffer.getvalue()
            content_hash = hashlib.sha256(embedded_data).hexdigest()

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
        self._conn.commit()
        return cursor.lastrowid

    def get_frames(self, sequence_id: int) -> list[Frame]:
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

    def get_frame_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        return row[0]

    # ========================================== Annotations
    def add_annotation(
        self,
        frame_id: int,
        label_id: int,
        mask: np.ndarray,
        encoding: str = "rle",
    ) -> int:
        if encoding == "rle":
            mask_data = rle_encode(mask)
        else:
            from .encoding import mask_to_png_bytes

            label = self._get_label_by_id(label_id)
            mask_data = mask_to_png_bytes(mask, label.color if label else "#FF0000")

        cursor = self._conn.execute(
            """INSERT OR REPLACE INTO annotations
               (frame_id, label_id, encoding, mask_data, modified_at)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            (frame_id, label_id, encoding, mask_data),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_annotation(
        self,
        frame_id: int,
        label_id: int,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        row = self._conn.execute(
            "SELECT encoding, mask_data FROM annotations WHERE frame_id = ? AND label_id = ?",
            (frame_id, label_id),
        ).fetchone()
        if row is None:
            return None

        if row["encoding"] == "rle":
            return rle_decode(row["mask_data"], width, height)
        else:
            img = Image.open(io.BytesIO(row["mask_data"]))
            arr = np.array(img)
            if arr.ndim == 3:
                return (arr[:, :, 3] > 0).astype(np.uint8) * 255
            return arr

    # ========================================== Classifications
    def add_classification(
        self,
        frame_id: int,
        task_name: str,
        selected_classes: list[str],
        is_multilabel: bool = False,
    ) -> int:
        cursor = self._conn.execute(
            """INSERT OR REPLACE INTO classifications
               (frame_id, task_name, selected_classes, is_multilabel, modified_at)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            (frame_id, task_name, json.dumps(selected_classes), is_multilabel),
        )
        self._conn.commit()
        return cursor.lastrowid

    # ========================================== Bulk Import
    def import_folder(
        self,
        folder: Union[str, Path],
        pattern: Optional[str] = None,
        recursive: bool = True,
        folders_as_sequences: bool = False,
        embed: bool = True,
    ) -> dict:
        folder = Path(folder)
        pattern = pattern or self.config.input_regex
        regex = re.compile(pattern, re.IGNORECASE)

        stats = {"sequences": 0, "frames": 0, "errors": []}

        images: dict[str, list[Path]] = {}
        glob_pattern = "**/*" if recursive else "*"
        for path in folder.glob(glob_pattern):
            if not path.is_file():
                continue
            if not regex.search(path.name):
                continue

            if folders_as_sequences and path.parent != folder:
                seq_name = path.parent.name
            else:
                seq_name = path.stem

            images.setdefault(seq_name, []).append(path)

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
        if sequence_name is None:
            if isinstance(image, (str, Path)):
                sequence_name = Path(image).stem
            else:
                sequence_name = f"image_{self.get_frame_count()}"

        sequence = self.get_or_create_sequence(sequence_name)
        frame_id = self.add_frame(sequence.id, image, embed=embed)

        for label_name, mask in masks.items():
            label = self.get_or_create_label(label_name)
            self.add_annotation(frame_id, label.id, mask)

        if classification:
            for task_name, classes in classification.items():
                is_multilabel = len(classes) > 1
                self.add_classification(frame_id, task_name, classes, is_multilabel)

        if text_descriptions:
            for label_name, text in text_descriptions.items():
                self.add_text_description(frame_id, label_name, text)

        return frame_id

    # ========================================== Iteration
    def iter_frames(self) -> Iterator[tuple[Sequence, Frame]]:
        for seq in self.get_sequences():
            for frame in self.get_frames(seq.id):
                yield seq, frame

    # ========================================== Statistics
    def get_statistics(self) -> dict:
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
