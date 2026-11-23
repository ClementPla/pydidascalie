"""Data models for Didascalie."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json

# File extensions (lowercase, no dot) that a folder scan can take, as listed by
# the application (IMAGE_EXTENSIONS, video::EXTENSIONS, MICROSCOPY_EXTENSIONS).
# Python decodes images only; videos and microscopy files need the application.
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
VIDEO_EXTENSIONS = ["mp4", "m4v", "mov", "mkv", "webm", "avi"]
MICROSCOPY_EXTENSIONS = [
    "czi", "lif", "nd2", "lsm", "oib", "oif", "ims", "vsi", "svs", "ndpi", "scn",
    "bif", "dv", "r3d", "zvi", "dm3", "dm4", "stk",
]

# Number types of a regression value. Every value is stored as a 64-bit float;
# the type decides how it is rounded first, and the dtype it is exported with.
REGRESSION_TYPES = ("float32", "float64", "int32")


@dataclass
class ProjectConfig:
    """
    Project configuration stored as JSON in the database.

    The application stores the definition of the labels, classification tasks,
    text fields and regression values here too. Prefer the project methods
    (``add_label``, ``add_classification_task``, ``set_multilabel_task``,
    ``add_text_field``, ``add_regression_task``) to editing those fields by hand.
    """

    name: str = ""
    input_folder: Optional[str] = None
    images_embedded: bool = False
    segmentation_enabled: bool = True
    classification_enabled: bool = False
    instance_segmentation_enabled: bool = False
    text_description_enabled: bool = False
    regression_enabled: bool = False
    # A folder scan takes a file when its extension is in ``extensions`` and its
    # name matches ``input_regex`` (empty matches any name).
    extensions: list = field(default_factory=lambda: list(IMAGE_EXTENSIONS))
    input_regex: str = ".*"
    recursive: bool = True
    folders_as_sequences: bool = False
    # ``[{"name", "color", "shades"}]``; None when the project never listed them.
    segmentation_labels: Optional[list] = None
    # ``[{"name", "classes", "default"}]``, one entry per multiclass task.
    classification_tasks: Optional[list] = None
    # ``{"name", "classes", "default"}``: the application has one multilabel task.
    multilabel_task: Optional[dict] = None
    text_fields: Optional[list] = None
    # ``[{"name", "dtype", "min", "max"}]``, one entry per regression value.
    # ``dtype`` is one of ``REGRESSION_TYPES``; a bound is a number or None.
    regression_tasks: Optional[list] = None
    # Keys this version does not know about, written back untouched so that
    # saving the config never drops what a newer application stored.
    extra: dict = field(default_factory=dict, repr=False)

    def to_json(self) -> str:
        """Serialize config to JSON string."""
        data = dict(self.extra)
        data.update({k: v for k, v in self.__dict__.items() if k != "extra"})
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "ProjectConfig":
        """Deserialize config from JSON string."""
        data = json.loads(json_str)
        data.pop("embed_threshold_kb", None)  # retired: not carried along
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        config = cls(**{k: v for k, v in data.items() if k in known})
        config.extra = {k: v for k, v in data.items() if k not in known}
        # Absent or null means every supported format: the images, for Python.
        if config.extensions is None:
            config.extensions = list(IMAGE_EXTENSIONS)
        return config


@dataclass
class User:
    """An account of the project. Each user has their own annotations."""

    id: int = 0
    name: str = ""
    role: str = "editor"


@dataclass
class Label:
    """Annotation label definition."""

    id: Optional[int] = None
    name: str = ""
    color: str = "#FF0000"
    is_instance: bool = False
    sort_order: int = 0


@dataclass
class Sequence:
    """A sequence of frames (can be single image or video/series)."""

    id: Optional[int] = None
    name: str = ""
    sort_order: int = 0
    frame_count: int = 0


@dataclass
class Frame:
    """A single image frame within a sequence."""

    id: Optional[int] = None
    sequence_id: int = 0
    frame_index: int = 0
    relative_path: Optional[str] = None
    content_hash: Optional[str] = None
    embedded_data: Optional[bytes] = None
    width: int = 0
    height: int = 0
    reviewed: bool = False

    @property
    def is_embedded(self) -> bool:
        """Check if frame has embedded image data."""
        return self.embedded_data is not None


@dataclass
class FrameSource:
    """
    Where a frame converted from a microscopy file came from. The frame holds
    the ``channels`` of time point ``t`` and slice ``z`` of that series.
    """

    frame_id: int = 0
    # Relative to the image folder, or absolute.
    source_path: str = ""
    series: int = 0
    # Resolution level, 0 being full resolution.
    level: int = 0
    t: int = 0
    # None for a maximum projection over Z.
    z: Optional[int] = None
    # The file's channel indices the frame's pages hold, in page order.
    channels: list[int] = field(default_factory=list)
    # Pixel sizes in micrometres, when the file gives them.
    physical_size_x: Optional[float] = None
    physical_size_y: Optional[float] = None
    physical_size_z: Optional[float] = None


@dataclass
class Annotation:
    """Segmentation mask annotation for a frame/label pair."""

    id: Optional[int] = None
    frame_id: int = 0
    label_id: int = 0
    encoding: str = "rle8"
    mask_data: bytes = b""
    modified_at: datetime = field(default_factory=datetime.now)


@dataclass
class Classification:
    """Classification annotation for a frame."""

    id: Optional[int] = None
    frame_id: int = 0
    task_name: str = ""
    selected_classes: list[str] = field(default_factory=list)
    is_multilabel: bool = False
    modified_at: datetime = field(default_factory=datetime.now)

    def selected_classes_json(self) -> str:
        """Serialize selected classes to JSON."""
        return json.dumps(self.selected_classes)

    @classmethod
    def from_json(cls, json_str: str) -> list[str]:
        """Deserialize selected classes from JSON."""
        return json.loads(json_str)


@dataclass
class Regression:
    """A number given to a frame for one regression task."""

    id: Optional[int] = None
    frame_id: int = 0
    task_name: str = ""
    value: float = 0.0
    modified_at: datetime = field(default_factory=datetime.now)


@dataclass
class TextDescription:
    """Text description annotation for a frame."""

    id: Optional[int] = None
    frame_id: int = 0
    label_name: str = ""
    content: str = ""
    modified_at: datetime = field(default_factory=datetime.now)
