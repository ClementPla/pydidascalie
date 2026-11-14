"""Main Didascalie project API."""

import sqlite3
import hashlib
import re
from pathlib import Path
from typing import Optional, Union, Iterator
import json
import io
import math
import os
import uuid
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Iterable

import numpy as np
from PIL import Image

from .schema import (
    CREATE_SCHEMA_VERSION,
    REGRESSIONS_TABLE,
    REGRESSIONS_TABLE_V3,
    SCHEMA,
    SCHEMA_V4,
    SCHEMA_VERSION,
    USER_ACCOUNTS_SCHEMA_VERSION,
    V3_PER_USER_TABLES,
    VECTOR_TABLE_V2,
)
from .models import (
    ProjectConfig,
    User,
    Label,
    Sequence,
    Frame,
    FrameSource,
    Classification,
    TextDescription,
    MICROSCOPY_EXTENSIONS,
    REGRESSION_TYPES,
    VIDEO_EXTENSIONS,
)
from . import pixels
from .vector import VectorShape, rasterize as _rasterize_shapes
from .encoding import (
    instance_mask,
    mask_to_png_bytes,
    png_bytes_to_mask,
    rle8_decode,
    rle8_encode,
    rle_decode,
)


def _encode_mask(mask: np.ndarray, is_instance: bool) -> bytes:
    """`rle8` bytes for a label's mask: instance ids, or 1 wherever it is set."""
    if is_instance:
        return rle8_encode(instance_mask(mask))
    return rle8_encode(np.asarray(mask) > 0)


def _decode_mask(
    encoding: str, data: bytes, width: int, height: int, raw: bool
) -> np.ndarray:
    """
    A stored mask as an array. With ``raw`` the stored values (instance ids)
    are returned; otherwise the mask is binary, 0 or 255.
    """
    if encoding == "rle8":
        values = rle8_decode(data, width, height)
        return values if raw else (values > 0).astype(np.uint8) * 255
    if encoding == "rle":
        mask = rle_decode(data, width, height)
    else:
        mask = png_bytes_to_mask(data)
    # The legacy encodings are binary: every pixel belongs to instance 1.
    return (mask > 0).astype(np.uint8) if raw else mask


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

    def __init__(self, db_path: Union[str, Path], user: Union[int, str, None] = None):
        """
        Open an existing Didascalie project.

        Args:
            db_path: Path to the .dida file
            user: Name or id of the account whose annotations are read and
                written, for a project with user accounts. Defaults to its
                first administrator, who owns what was annotated before
                accounts existed.
        """
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Project not found: {db_path}")

        self._connection = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._config: Optional[ProjectConfig] = None
        self._bulk_depth = 0
        self._check_schema_version()
        # Since schema version 4, annotations, classifications, texts and
        # reviews belong to a user. Both layouts are read and written.
        self._user_scoped = self._has_column("annotations", "user_id")
        self._user_id: Optional[int] = None
        if self._user_scoped:
            self.set_user(user)
        elif user is not None:
            raise ValueError(
                f"{self.db_path} has no user accounts: it was last saved by a "
                "version of Didascalie that predates them."
            )

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
            self._conn.execute("PRAGMA user_version = 1")
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
        conn.execute(f"PRAGMA user_version = {CREATE_SCHEMA_VERSION}")

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
    # Users
    # ==========================================

    def _has_table(self, table: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            ).fetchone()
            is not None
        )

    def _has_column(self, table: str, column: str) -> bool:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row["name"] == column for row in rows)

    def _mine(self, alias: str = "") -> tuple[str, tuple]:
        """SQL condition and parameter limiting a per-user table to the current
        user; empty for a project without accounts."""
        if not self._user_scoped:
            return "", ()
        return f" AND {alias}user_id = ?", (self._user_id,)

    def _owner(self) -> tuple[str, str, tuple]:
        """Column, placeholder and parameter that make an inserted row belong
        to the current user; empty for a project without accounts."""
        if not self._user_scoped:
            return "", "", ()
        return ", user_id", ", ?", (self._user_id,)

    def _reviewed(self) -> tuple[str, tuple]:
        """SQL expression (and its parameters) telling whether a row of
        ``frames`` is reviewed, by the current user where there are accounts."""
        if not self._user_scoped:
            return "reviewed", ()
        return (
            "EXISTS (SELECT 1 FROM frame_reviews r"
            " WHERE r.frame_id = frames.id AND r.user_id = ?)",
            (self._user_id,),
        )

    def _mark_rle8(self) -> None:
        """Masks are written as `rle8`, which the application reads from schema
        version 3 on: stamp older files so that older builds refuse them
        instead of misreading the masks."""
        if self.schema_version < 3:
            self._conn.execute("PRAGMA user_version = 3")

    def get_users(self) -> list[User]:
        """Accounts of the project; empty for a project without accounts."""
        if not self._user_scoped:
            return []
        rows = self._conn.execute("SELECT id, name, role FROM users ORDER BY id").fetchall()
        return [User(id=row["id"], name=row["name"], role=row["role"]) for row in rows]

    @property
    def user(self) -> Optional[User]:
        """Account whose annotations are read and written (None without accounts)."""
        return next((u for u in self.get_users() if u.id == self._user_id), None)

    def set_user(self, user: Union[int, str, None] = None) -> None:
        """
        Choose whose annotations are read and written from now on.

        Args:
            user: Account name or id; None for the first administrator
        """
        if not self._user_scoped:
            raise ValueError("This project has no user accounts")
        users = self.get_users()
        if user is None:
            admins = [u for u in users if u.role == "admin"]
            match = (admins or users or [None])[0]
        elif isinstance(user, str):
            match = next((u for u in users if u.name.lower() == user.lower()), None)
        else:
            match = next((u for u in users if u.id == user), None)
        if match is None:
            names = ", ".join(u.name for u in users)
            raise ValueError(f"No such user: {user!r} (accounts: {names})")
        self._user_id = match.id

    def enable_user_accounts(self, owner: str = "Admin") -> User:
        """
        Give the project user accounts, as the application does the first time
        it opens a project created before they existed.

        Everything annotated so far goes to a first account, an administrator
        named ``owner``, who becomes the current user. Further accounts are
        added with ``add_user``.

        The project then needs a version of Didascalie that has user accounts:
        older ones refuse to open it.
        """
        if self._bulk_depth:
            raise RuntimeError("enable_user_accounts() cannot run inside bulk()")
        if not self._user_scoped:
            old = [t for t in V3_PER_USER_TABLES if self._has_column(t, "id")]
            script = ["BEGIN;"]
            script += [f"ALTER TABLE {t} RENAME TO {t}_v3;" for t in old]
            # The tables just moved aside, in their current shape, and account 1.
            # `regressions` is newer than version 4, so it comes separately.
            current = SCHEMA_V4 + (REGRESSIONS_TABLE if "regressions" in old else "")
            script.append(current)
            for table in old:
                columns = V3_PER_USER_TABLES[table]
                script.append(
                    f"INSERT INTO {table} ({columns}, user_id)"
                    f" SELECT {columns}, 1 FROM {table}_v3;"
                    f"DROP TABLE {table}_v3;"
                )
            # Once more for the indexes, which went with the dropped tables.
            script.append(current)
            script.append(
                "INSERT OR IGNORE INTO frame_reviews (frame_id, user_id)"
                " SELECT id, 1 FROM frames WHERE reviewed = 1;"
                f"PRAGMA user_version = {USER_ACCOUNTS_SCHEMA_VERSION};"
                "COMMIT;"
            )
            self._conn.commit()
            try:
                self._conn.executescript("\n".join(script))
            except BaseException:
                self._conn.rollback()
                raise
            self._user_scoped = True
            self._conn.execute("UPDATE users SET name = ? WHERE id = 1", (owner,))
            self._commit()
        self.set_user(None)
        return self.user

    def add_user(self, name: str, role: str = "editor") -> User:
        """
        Add an account, e.g. to store a model's predictions next to the
        annotators' work instead of over it. The account has no password.

        Args:
            name: Account name, unique in the project
            role: 'editor' or 'admin'
        """
        if not self._user_scoped:
            raise ValueError(
                "This project has no user accounts: call enable_user_accounts() first"
            )
        cursor = self._conn.execute(
            "INSERT INTO users (name, role) VALUES (?, ?)", (name, role)
        )
        self._commit()
        return User(id=cursor.lastrowid, name=name, role=role)

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

    def add_classification_task(
        self, name: str, classes: list[str], default: Optional[str] = None
    ) -> None:
        """
        Declare a multiclass task (one class per frame), or add classes to it.
        The application only shows the tasks declared in the project.
        """
        config = self.config
        if config.multilabel_task and config.multilabel_task["name"] == name:
            raise ValueError(f"{name!r} is the multilabel task of this project")
        tasks = list(config.classification_tasks or [])
        task = next((t for t in tasks if t["name"] == name), None)
        if task is None:
            task = {"name": name, "classes": [], "default": default}
            tasks.append(task)
        elif default is not None:
            task["default"] = default
        task["classes"] = list(dict.fromkeys([*task["classes"], *classes]))
        config.classification_tasks = tasks
        config.classification_enabled = True
        self.update_config(config)

    def set_multilabel_task(
        self, name: str, classes: list[str], default: Optional[list[str]] = None
    ) -> None:
        """
        Declare the multilabel task (several classes per frame), or add classes
        to it. A project has one multilabel task.
        """
        config = self.config
        task = config.multilabel_task
        if task is not None and task["name"] != name:
            raise ValueError(
                f"This project already has a multilabel task, {task['name']!r}; "
                "the application supports one"
            )
        if any(t["name"] == name for t in config.classification_tasks or []):
            raise ValueError(f"{name!r} is a multiclass task of this project")
        if task is None:
            task = {"name": name, "classes": [], "default": default}
        elif default is not None:
            task["default"] = default
        task["classes"] = list(dict.fromkeys([*task["classes"], *classes]))
        config.multilabel_task = task
        config.classification_enabled = True
        self.update_config(config)

    def get_classification_tasks(self) -> dict[str, list[str]]:
        """Declared tasks and their classes, the multilabel one included."""
        config = self.config
        tasks = {t["name"]: list(t["classes"]) for t in config.classification_tasks or []}
        if config.multilabel_task:
            tasks[config.multilabel_task["name"]] = list(config.multilabel_task["classes"])
        return tasks

    def add_text_field(self, name: str) -> None:
        """Declare a free-text field, shown by the application on every frame."""
        config = self.config
        if name not in (config.text_fields or []):
            config.text_fields = [*(config.text_fields or []), name]
            config.text_description_enabled = True
            self.update_config(config)

    def add_regression_task(
        self,
        name: str,
        dtype: str = "float32",
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> None:
        """
        Declare a regression value: a number each frame can be given, shown by
        the application on every frame. Declaring a name again replaces its
        type and range; the values already stored are kept.

        Args:
            name: Name of the value, unique among the regression values
            dtype: 'float32' (default), 'float64' or 'int32'. A value is rounded
                to this type when it is stored.
            min: Smallest value accepted, or None for no lower bound
            max: Largest value accepted, or None for no upper bound. With both
                bounds, the application sets the value with a slider.
        """
        if dtype not in REGRESSION_TYPES:
            raise ValueError(f"dtype must be one of {REGRESSION_TYPES}, not {dtype!r}")
        for bound in (min, max):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("A range bound must be a finite number")
        if min is not None and max is not None and min >= max:
            raise ValueError(f"The minimum ({min}) must be below the maximum ({max})")

        task = {
            "name": name,
            "dtype": dtype,
            "min": None if min is None else float(min),
            "max": None if max is None else float(max),
        }
        config = self.config
        tasks = list(config.regression_tasks or [])
        for i, existing in enumerate(tasks):
            if existing["name"] == name:
                tasks[i] = task
                break
        else:
            tasks.append(task)
        config.regression_tasks = tasks
        config.regression_enabled = True
        self._ensure_regression_table()
        self.update_config(config)

    def get_regression_tasks(self) -> dict[str, dict]:
        """Declared regression values: ``{name: {"dtype", "min", "max"}}``."""
        return {
            t["name"]: {
                "dtype": t.get("dtype") or "float32",
                "min": t.get("min"),
                "max": t.get("max"),
            }
            for t in self.config.regression_tasks or []
        }

    def _sync_label_config(self) -> None:
        """
        Mirror the labels table into the config of a project that lists its
        labels there. The application treats that list as the reference when it
        opens a project: a label missing from it is removed if nothing uses it.
        """
        config = self.config
        if config.segmentation_labels is None:
            return
        shades = {l["name"]: l.get("shades") for l in config.segmentation_labels}
        config.segmentation_labels = [
            {"name": l.name, "color": l.color, "shades": shades.get(l.name)}
            for l in self.get_labels()
        ]
        self.update_config(config)

    # ==========================================
    # Labels
    # ==========================================

    def add_label(self, label: Label) -> int:
        """
        Add a label and return its ID.

        Instance segmentation is a setting of the whole project: adding an
        instance label turns it on, and in an instance project every label is
        an instance label.
        """
        config = self.config
        if config.instance_segmentation_enabled:
            label.is_instance = True
        elif label.is_instance:
            config.instance_segmentation_enabled = True
            self.update_config(config)

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
        self._sync_label_config()
        return label.id

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
        self._sync_label_config()

    def delete_label(self, label_id: int) -> None:
        """Delete a label and its annotations."""
        self._conn.execute("DELETE FROM labels WHERE id = ?", (label_id,))
        self._commit()
        self._sync_label_config()

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
        reviewed, mine = self._reviewed()
        rows = self._conn.execute(
            f"""SELECT id, sequence_id, frame_index, relative_path,
                       content_hash, width, height, {reviewed} AS reviewed
                FROM frames WHERE sequence_id = ?
                ORDER BY frame_index""",
            (*mine, sequence_id),
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
        reviewed, mine = self._reviewed()
        row = self._conn.execute(
            f"""SELECT id, sequence_id, frame_index, relative_path,
                       content_hash, width, height, {reviewed} AS reviewed
                FROM frames WHERE id = ?""",
            (*mine, frame_id),
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

    def get_frame_image(
        self, frame_id: int, input_folder: Optional[str] = None, raw: bool = False
    ) -> Union[Image.Image, np.ndarray]:
        """
        Get the image for a frame.

        An 8-bit image comes back as stored. A frame deeper than 8 bits, or
        with several channels (a microscopy frame: a multi-page TIFF, one page
        per channel), comes back as the application shows it: 8-bit RGB through
        its sequence's display settings, or the automatic ones when it has none
        (see ``get_display_settings``).

        Args:
            frame_id: Frame ID
            input_folder: Base folder for non-embedded images
            raw: Return the native values instead, a ``(height, width,
                channels)`` array of uint8, uint16 or float32

        Returns:
            PIL Image, or numpy array when ``raw``
        """
        row = self._conn.execute(
            "SELECT sequence_id, embedded_data, relative_path FROM frames WHERE id = ?",
            (frame_id,),
        ).fetchone()

        if row is None:
            raise ValueError(f"Frame not found: {frame_id}")

        if row["embedded_data"]:
            image = Image.open(io.BytesIO(row["embedded_data"]))
        elif row["relative_path"] and input_folder:
            image = Image.open(Path(input_folder) / row["relative_path"])
        else:
            raise ValueError("Frame has no embedded data and no input folder provided")

        if raw:
            return pixels.decode_raw(image)
        if not pixels.needs_render(image):
            return image
        values = pixels.decode_raw(image)
        display = self._stored_display_settings(row["sequence_id"])
        return Image.fromarray(pixels.render_rgb(values, display), "RGB")

    def _stored_display_settings(self, sequence_id: int) -> Optional[dict]:
        if not self._has_table("sequence_display"):
            return None
        row = self._conn.execute(
            "SELECT settings FROM sequence_display WHERE sequence_id = ?", (sequence_id,)
        ).fetchone()
        return json.loads(row["settings"]) if row else None

    def get_display_settings(self, frame_id: int) -> dict:
        """
        The display settings a frame is shown with: per channel, a window,
        colour and gamma, ``{"channels": [{"name", "visible", "color", "min",
        "max", "gamma"}]}``. They are its sequence's, set in the application,
        when they have one entry per channel of the frame; otherwise the
        automatic ones (8-bit as stored, deeper data windowed at its 0.5 and
        99.5 % percentiles).
        """
        sequence_id = self._conn.execute(
            "SELECT sequence_id FROM frames WHERE id = ?", (frame_id,)
        ).fetchone()
        if sequence_id is None:
            raise ValueError(f"Frame not found: {frame_id}")
        values = self.get_frame_image(frame_id, raw=True)
        display = self._stored_display_settings(sequence_id[0])
        if display is None or len(display.get("channels") or []) != values.shape[-1]:
            display = pixels.auto_display(values)
        return display

    def get_frame_source(self, frame_id: int) -> Optional[FrameSource]:
        """
        Where a frame converted from a microscopy file came from, or None for a
        frame imported as an image.
        """
        sources = self._frame_sources("WHERE frame_id = ?", (frame_id,))
        return sources[0] if sources else None

    def get_frame_sources(self, sequence_id: Optional[int] = None) -> list[FrameSource]:
        """The sources of the frames converted from microscopy files, of a sequence or all."""
        if sequence_id is None:
            return self._frame_sources("", ())
        return self._frame_sources(
            "WHERE frame_id IN (SELECT id FROM frames WHERE sequence_id = ?)", (sequence_id,)
        )

    def _frame_sources(self, where: str, args: tuple) -> list[FrameSource]:
        if not self._has_table("frame_sources"):
            return []
        rows = self._conn.execute(
            f"SELECT * FROM frame_sources {where} ORDER BY frame_id", args
        ).fetchall()
        return [
            FrameSource(
                frame_id=row["frame_id"],
                source_path=row["source_path"],
                series=row["series"],
                level=row["level"],
                t=row["t"],
                z=row["z"],
                channels=json.loads(row["channels"]),
                physical_size_x=row["physical_size_x"],
                physical_size_y=row["physical_size_y"],
                physical_size_z=row["physical_size_z"],
            )
            for row in rows
        ]

    def set_frame_reviewed(self, frame_id: int, reviewed: bool = True) -> None:
        """Mark a frame as reviewed (by the current user, where there are accounts)."""
        if not self._user_scoped:
            self._conn.execute(
                "UPDATE frames SET reviewed = ? WHERE id = ?", (reviewed, frame_id)
            )
        elif reviewed:
            self._conn.execute(
                "INSERT OR IGNORE INTO frame_reviews (frame_id, user_id) VALUES (?, ?)",
                (frame_id, self._user_id),
            )
        else:
            self._conn.execute(
                "DELETE FROM frame_reviews WHERE frame_id = ? AND user_id = ?",
                (frame_id, self._user_id),
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
        encoding: str = "rle8",
    ) -> int:
        """
        Add a segmentation annotation.

        Args:
            frame_id: Frame ID
            label_id: Label ID
            mask: Mask array (H x W). For an instance label, the instance id of
                each pixel (0 is background, ids go up to 255). Otherwise any
                non-zero pixel belongs to the label.
            encoding: 'rle8', the application's format, or 'png' (legacy)

        Returns:
            Annotation ID
        """
        label = self.get_label_by_id(label_id)
        if encoding == "rle8":
            mask_data = _encode_mask(mask, bool(label and label.is_instance))
            self._mark_rle8()
        elif encoding == "png":
            color = label.color if label else "#FF0000"
            mask_data = mask_to_png_bytes(mask, color)
        else:
            raise ValueError(f"Unsupported encoding: {encoding!r}")

        column, placeholder, owner = self._owner()
        cursor = self._conn.execute(
            f"""INSERT OR REPLACE INTO annotations
                (frame_id, label_id, encoding, mask_data, modified_at{column})
                VALUES (?, ?, ?, ?, datetime('now'){placeholder})""",
            (frame_id, label_id, encoding, mask_data, *owner),
        )
        self._commit()
        return cursor.lastrowid

    def get_annotation(
        self,
        frame_id: int,
        label_id: int,
        width: int,
        height: int,
        raw: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Get annotation mask for a frame/label pair.

        Args:
            frame_id: Frame ID
            label_id: Label ID
            width: Image width
            height: Image height
            raw: Return the stored values, i.e. the instance ids of an instance
                label (and 1 for a semantic one)

        Returns:
            Mask array (H x W), or None if not found. Binary with values 0 or
            255 unless ``raw`` is set.
        """
        mine, user = self._mine()
        row = self._conn.execute(
            "SELECT encoding, mask_data FROM annotations"
            f" WHERE frame_id = ? AND label_id = ?{mine}",
            (frame_id, label_id, *user),
        ).fetchone()

        if row is None:
            return None

        return _decode_mask(row["encoding"], row["mask_data"], width, height, raw)

    def get_annotations_for_frame(
        self, frame_id: int, raw: bool = False
    ) -> list[tuple[Label, np.ndarray]]:
        """
        Get all annotations for a frame.

        Args:
            frame_id: Frame ID
            raw: Return instance ids instead of binary masks (see
                ``get_annotation``)

        Returns:
            List of (Label, mask) tuples
        """
        frame = self.get_frame(frame_id)
        if frame is None:
            return []

        results = []
        mine, user = self._mine("a.")
        rows = self._conn.execute(
            f"""SELECT a.label_id, a.encoding, a.mask_data, l.name, l.color, l.is_instance
                FROM annotations a
                JOIN labels l ON l.id = a.label_id
                WHERE a.frame_id = ?{mine}""",
            (frame_id, *user),
        ).fetchall()

        for row in rows:
            label = Label(
                id=row["label_id"],
                name=row["name"],
                color=row["color"],
                is_instance=bool(row["is_instance"]),
            )

            mask = _decode_mask(
                row["encoding"], row["mask_data"], frame.width, frame.height, raw
            )

            results.append((label, mask))

        return results

    def delete_annotation(self, frame_id: int, label_id: int) -> None:
        """Delete an annotation."""
        mine, user = self._mine()
        self._conn.execute(
            f"DELETE FROM annotations WHERE frame_id = ? AND label_id = ?{mine}",
            (frame_id, label_id, *user),
        )
        self._commit()

    # ==========================================
    # Vector annotations
    # ==========================================

    def _vector_rows(self, frame_id: Optional[int], label_id: Optional[int]) -> list:
        if not self._has_table("vector_annotations"):
            return []
        where, params = "1 = 1", []
        if frame_id is not None:
            where += " AND frame_id = ?"
            params.append(frame_id)
        if label_id is not None:
            where += " AND label_id = ?"
            params.append(label_id)
        mine, user = self._mine()
        return self._conn.execute(
            f"SELECT frame_id, label_id, shapes FROM vector_annotations"
            f" WHERE {where}{mine} ORDER BY frame_id, label_id",
            (*params, *user),
        ).fetchall()

    @staticmethod
    def _parse_shapes(row) -> list[VectorShape]:
        shapes = [VectorShape.from_dict(d) for d in json.loads(row["shapes"])]
        for s in shapes:
            s.label_id = row["label_id"]  # the row is authoritative
        return shapes

    def get_vector_shapes(
        self, frame_id: int, label_id: Optional[int] = None
    ) -> list[VectorShape]:
        """
        The vector shapes of a frame, of one label or of all of them, for the
        current user. Each shape's ``label_id`` tells its label.
        """
        return [
            s for row in self._vector_rows(frame_id, label_id) for s in self._parse_shapes(row)
        ]

    def iter_vector_shapes(
        self, label_id: Optional[int] = None
    ) -> Iterator[tuple[int, list[VectorShape]]]:
        """
        ``(frame_id, shapes)`` for every frame that has vector shapes (of
        ``label_id`` if given), for the current user, in frame id order.
        """
        frame_id, shapes = None, []
        for row in self._vector_rows(None, label_id):
            if row["frame_id"] != frame_id and shapes:
                yield frame_id, shapes
                shapes = []
            frame_id = row["frame_id"]
            shapes.extend(self._parse_shapes(row))
        if shapes:
            yield frame_id, shapes

    def set_vector_shapes(
        self, frame_id: int, label_id: int, shapes: Iterable[VectorShape]
    ) -> None:
        """
        Replace the current user's vector shapes of one label on a frame.
        An empty list deletes them, as the application does.
        """
        shapes = list(shapes)
        if not shapes:
            self.delete_vector_shapes(frame_id, label_id)
            return
        if not self._has_table("vector_annotations"):
            # executescript would commit an open bulk() transaction
            for statement in VECTOR_TABLE_V2.split(";"):
                if statement.strip():
                    self._conn.execute(statement)
        data = []
        for s in shapes:
            s.label_id = label_id
            data.append(s.to_dict())
        column, placeholder, owner = self._owner()
        self._conn.execute(
            f"""INSERT INTO vector_annotations
                (frame_id, label_id, shapes, modified_at{column})
                VALUES (?, ?, ?, datetime('now'){placeholder})
                ON CONFLICT(frame_id, label_id{column})
                DO UPDATE SET shapes = excluded.shapes, modified_at = excluded.modified_at""",
            (frame_id, label_id, json.dumps(data), *owner),
        )
        self._commit()

    def add_vector_shape(
        self, frame_id: int, shape: VectorShape, label_id: Optional[int] = None
    ) -> None:
        """Add a shape to those of its label on a frame (``label_id``, or the shape's own)."""
        label_id = shape.label_id if label_id is None else label_id
        if label_id is None:
            raise ValueError("the shape has no label: pass label_id")
        self.set_vector_shapes(
            frame_id, label_id, self.get_vector_shapes(frame_id, label_id) + [shape]
        )

    def delete_vector_shapes(self, frame_id: int, label_id: Optional[int] = None) -> None:
        """Delete the current user's vector shapes on a frame, of one label or all."""
        if not self._has_table("vector_annotations"):
            return
        mine, user = self._mine()
        label = " AND label_id = ?" if label_id is not None else ""
        params = (label_id,) if label_id is not None else ()
        self._conn.execute(
            f"DELETE FROM vector_annotations WHERE frame_id = ?{label}{mine}",
            (frame_id, *params, *user),
        )
        self._commit()

    def get_vector_mask(
        self, frame_id: int, label_id: int, fill: Optional[bool] = None
    ) -> Optional[np.ndarray]:
        """
        The vector shapes of one label on a frame, rasterized at the frame's
        size: a ``(height, width)`` uint8 mask, 0 or 255, or None without shapes.

        Args:
            fill: None rasterizes as the application does (closed and filled
                shapes are filled, others stroked); True fills every closed
                shape; False strokes them all.
        """
        shapes = self.get_vector_shapes(frame_id, label_id)
        if not shapes:
            return None
        frame = self.get_frame(frame_id)
        return _rasterize_shapes(shapes, frame.width, frame.height, fill=fill)

    # ==========================================
    # Classifications
    # ==========================================

    def add_classification(
        self,
        frame_id: int,
        task_name: str,
        selected_classes: list[str],
        is_multilabel: Optional[bool] = None,
    ) -> int:
        """
        Add a classification annotation.

        The task and the classes are declared in the project if they are not
        already, so that the application shows them.

        Args:
            frame_id: Frame ID
            task_name: Task the classes belong to
            selected_classes: Chosen classes (one, for a multiclass task)
            is_multilabel: Kind of a task that is not declared yet. By default
                a task given several classes is the multilabel task.
        """
        selected_classes = list(selected_classes)
        config = self.config
        declared = config.multilabel_task and config.multilabel_task["name"] == task_name
        if declared or any(t["name"] == task_name for t in config.classification_tasks or []):
            is_multilabel = bool(declared)
        elif is_multilabel is None:
            is_multilabel = len(selected_classes) > 1
        if not is_multilabel and len(selected_classes) > 1:
            raise ValueError(
                f"{task_name!r} is a multiclass task: it takes one class per frame"
            )
        if not set(selected_classes) <= set(self.get_classification_tasks().get(task_name, [])):
            if is_multilabel:
                self.set_multilabel_task(task_name, selected_classes)
            else:
                self.add_classification_task(task_name, selected_classes)

        column, placeholder, owner = self._owner()
        cursor = self._conn.execute(
            f"""INSERT OR REPLACE INTO classifications
                (frame_id, task_name, selected_classes, is_multilabel, modified_at{column})
                VALUES (?, ?, ?, ?, datetime('now'){placeholder})""",
            (frame_id, task_name, json.dumps(selected_classes), is_multilabel, *owner),
        )
        self._commit()
        return cursor.lastrowid

    def get_classification(
        self, frame_id: int, task_name: str
    ) -> Optional[Classification]:
        """Get classification for a frame/task pair."""
        mine, user = self._mine()
        row = self._conn.execute(
            f"SELECT * FROM classifications WHERE frame_id = ? AND task_name = ?{mine}",
            (frame_id, task_name, *user),
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
        mine, user = self._mine()
        rows = self._conn.execute(
            f"SELECT * FROM classifications WHERE frame_id = ?{mine}", (frame_id, *user)
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
        mine, user = self._mine()
        self._conn.execute(
            f"DELETE FROM classifications WHERE frame_id = ? AND task_name = ?{mine}",
            (frame_id, task_name, *user),
        )
        self._commit()

    # ==========================================
    # Text Descriptions
    # ==========================================

    def add_text_description(
        self, frame_id: int, label_name: str, content: str
    ) -> int:
        """Add a text description. The field is declared in the project if it
        is not already, so that the application shows it."""
        self.add_text_field(label_name)
        column, placeholder, owner = self._owner()
        cursor = self._conn.execute(
            f"""INSERT OR REPLACE INTO text_descriptions
                (frame_id, label_name, content, modified_at{column})
                VALUES (?, ?, ?, datetime('now'){placeholder})""",
            (frame_id, label_name, content, *owner),
        )
        self._commit()
        return cursor.lastrowid

    def get_text_description(
        self, frame_id: int, label_name: str
    ) -> Optional[TextDescription]:
        """Get text description for a frame/label pair."""
        mine, user = self._mine()
        row = self._conn.execute(
            f"SELECT * FROM text_descriptions WHERE frame_id = ? AND label_name = ?{mine}",
            (frame_id, label_name, *user),
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
        mine, user = self._mine()
        rows = self._conn.execute(
            f"SELECT * FROM text_descriptions WHERE frame_id = ?{mine}", (frame_id, *user)
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
    # Regressions (numbers given to a frame)
    # ==========================================

    def _ensure_regression_table(self) -> None:
        """Add the ``regressions`` table to a project that lacks it, in the
        layout of the project: with or without user accounts."""
        if self._has_table("regressions"):
            return
        table = REGRESSIONS_TABLE if self._user_scoped else REGRESSIONS_TABLE_V3
        for statement in table.split(";"):
            if statement.strip():
                self._conn.execute(statement)

    @staticmethod
    def _conform_regression(name: str, task: dict, value: float) -> float:
        """The value to store: within the task's range, in the task's type.
        The range is checked on the value as given, before rounding, as the
        application does."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name!r} needs a finite number")
        low, high = task["min"], task["max"]
        if (low is not None and value < low) or (high is not None and value > high):
            if low is not None and high is not None:
                wanted = f"between {low} and {high}"
            elif low is not None:
                wanted = f"at least {low}"
            else:
                wanted = f"at most {high}"
            raise ValueError(f"{name!r} must be {wanted} (got {value})")

        dtype = task["dtype"]
        if dtype == "float32":
            with np.errstate(over="ignore"):
                stored = float(np.float32(value))
        elif dtype == "int32":
            # Half away from zero, like the application (not Python's round()).
            stored = math.copysign(math.floor(abs(value) + 0.5), value)
            if not -(2**31) <= stored <= 2**31 - 1:
                stored = math.inf
        else:
            stored = value
        if not math.isfinite(stored):
            raise ValueError(f"{value} does not fit in {dtype} ({name!r})")
        return stored

    def add_regression(self, frame_id: int, task_name: str, value: float) -> float:
        """
        Give a frame its value for a regression task, replacing the previous
        one. A task that is not declared yet is declared as a ``float32``
        without a range; use ``add_regression_task`` first to choose both.

        Returns:
            The value as stored: rounded to the type of the task.
        """
        task = self.get_regression_tasks().get(task_name)
        if task is None:
            self.add_regression_task(task_name)
            task = self.get_regression_tasks()[task_name]
        stored = self._conform_regression(task_name, task, value)

        self._ensure_regression_table()
        column, placeholder, owner = self._owner()
        self._conn.execute(
            f"""INSERT OR REPLACE INTO regressions
                (frame_id, task_name, value, modified_at{column})
                VALUES (?, ?, ?, datetime('now'){placeholder})""",
            (frame_id, task_name, stored, *owner),
        )
        self._commit()
        return stored

    def get_regression(self, frame_id: int, task_name: str) -> Optional[float]:
        """A frame's value for a regression task, or None when it has none."""
        return self.get_regressions_for_frame(frame_id).get(task_name)

    def get_regressions_for_frame(self, frame_id: int) -> dict[str, float]:
        """A frame's regression values by task name. A task without a value
        is absent."""
        if not self._has_table("regressions"):
            return {}
        mine, user = self._mine()
        rows = self._conn.execute(
            f"SELECT task_name, value FROM regressions WHERE frame_id = ?{mine}",
            (frame_id, *user),
        ).fetchall()
        return {row["task_name"]: row["value"] for row in rows}

    def delete_regression(self, frame_id: int, task_name: str) -> None:
        """Remove a frame's value for a regression task."""
        if not self._has_table("regressions"):
            return
        mine, user = self._mine()
        self._conn.execute(
            f"DELETE FROM regressions WHERE frame_id = ? AND task_name = ?{mine}",
            (frame_id, task_name, *user),
        )
        self._commit()

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
        extensions: Optional[Iterable[str]] = None,
    ) -> dict:
        """
        Import images from a folder.

        A file is taken when its extension is one of ``extensions`` and its
        name matches ``pattern``, as in the application. Videos and microscopy
        files are skipped: Python cannot decode them, so import them from the
        application.

        Args:
            folder: Path to image folder
            pattern: Regex filtering file names (default: the project's
                ``input_regex``; empty matches any name)
            recursive: Search subfolders
            folders_as_sequences: Group subfolders as sequences
            embed: Embed images in database
            extensions: Extensions to take, case-insensitive, with or without
                the dot (default: the project's ``extensions``)

        Returns:
            Dict with import statistics; ``skipped`` lists the files left out
            because Python cannot read them
        """
        folder = Path(folder)
        if pattern is None:
            pattern = self.config.input_regex
        regex = re.compile(pattern or ".*", re.IGNORECASE)
        if extensions is None:
            extensions = self.config.extensions
        allowed = {e.lower().lstrip(".") for e in extensions}
        unreadable = set(VIDEO_EXTENSIONS) | set(MICROSCOPY_EXTENSIONS)

        stats = {"sequences": 0, "frames": 0, "errors": [], "skipped": []}

        # Collect images
        images: dict[str, list[Path]] = {}  # sequence_name -> [paths]

        glob_pattern = "**/*" if recursive else "*"
        for path in folder.glob(glob_pattern):
            if not path.is_file():
                continue
            ext = path.suffix.lower().lstrip(".")
            if ext not in allowed or not regex.search(path.name):
                continue
            if ext in unreadable:
                stats["skipped"].append(str(path))
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

        if stats["skipped"]:
            warnings.warn(
                f"Skipped {len(stats['skipped'])} video or microscopy file(s), "
                "which pydidascalie cannot decode; import them from the "
                f"Didascalie application (first: {stats['skipped'][0]})",
                stacklevel=2,
            )
        return stats

    def import_with_masks(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        masks: dict[str, np.ndarray],
        sequence_name: Optional[str] = None,
        classification: Optional[dict[str, list[str]]] = None,
        text_descriptions: Optional[dict[str, str]] = None,
        embed: bool = True,
        regression: Optional[dict[str, float]] = None,
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
            regression: Optional dict mapping regression task name -> value

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
                    self.add_classification(frame_id, task_name, classes)

            # Add text descriptions
            if text_descriptions:
                for label_name, text in text_descriptions.items():
                    self.add_text_description(frame_id, label_name, text)

            # Add regression values
            if regression:
                for task_name, value in regression.items():
                    self.add_regression(frame_id, task_name, value)

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
                ``masks`` maps label_name -> mask_array (instance ids for the
                labels that already exist as instance labels). Consumed lazily, a few
                frames ahead of the writes, so it can be a generator.
            embed: Embed images in database
            format: Format of the embedded images
            workers: Encoding threads (defaults to the CPU count, at most 8)

        Returns:
            Frame IDs, in order
        """
        if workers is None:
            workers = min(8, os.cpu_count() or 1)

        # Read before the workers start: they must not touch the connection.
        instance_labels = {l.name for l in self.get_labels() if l.is_instance}
        # In an instance project the labels created along the way are instance
        # labels too.
        every_label = self.config.instance_segmentation_enabled

        def encode(item):
            image, masks = item if isinstance(item, tuple) else (item, None)
            return (
                _encode_image(image, None, embed, format),
                {
                    k: _encode_mask(np.asarray(m), every_label or k in instance_labels)
                    for k, m in (masks or {}).items()
                },
            )

        frame_ids = []
        label_ids: dict[str, int] = {}
        column, placeholder, owner = self._owner()
        with self.bulk(), ThreadPoolExecutor(max_workers=workers) as pool:
            self._mark_rle8()
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
                            f"""INSERT OR REPLACE INTO annotations
                                (frame_id, label_id, encoding, mask_data, modified_at{column})
                                VALUES (?, ?, 'rle8', ?, datetime('now'){placeholder})""",
                            (frame_id, label_ids[label_name], mask_data, *owner),
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
        """Get project statistics (for the current user, where there are accounts)."""
        mine, user = self._mine()
        reviewed, reviewer = self._reviewed()

        def count(table: str) -> int:
            return self._conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE 1{mine}", user
            ).fetchone()[0]

        return {
            "sequences": len(self.get_sequences()),
            "frames": self.get_frame_count(),
            "labels": len(self.get_labels()),
            "annotations": count("annotations"),
            "classifications": count("classifications"),
            "regressions": count("regressions") if self._has_table("regressions") else 0,
            "reviewed": self._conn.execute(
                f"SELECT COUNT(*) FROM frames WHERE {reviewed}", reviewer
            ).fetchone()[0],
        }


# Backwards-compatible alias (deprecated): the class was renamed
# LabelMedProject -> DidascalieProject when the library was rebranded to Didascalie.
LabelMedProject = DidascalieProject
