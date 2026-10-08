"""Data models for Didascalie."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json


@dataclass
class ProjectConfig:
    """
    Project configuration stored as JSON in the database.

    The application stores the definition of the labels, classification tasks
    and text fields here too. Prefer the project methods (``add_label``,
    ``add_classification_task``, ``set_multilabel_task``, ``add_text_field``)
    to editing those four fields by hand.
    """

    name: str = ""
    input_folder: Optional[str] = None
    images_embedded: bool = False
    embed_threshold_kb: int = 100
    segmentation_enabled: bool = True
    classification_enabled: bool = False
    instance_segmentation_enabled: bool = False
    text_description_enabled: bool = False
    input_regex: str = r"\.(png|jpe?g|bmp|tiff?|dcm)$"
    recursive: bool = True
    folders_as_sequences: bool = False
    # ``[{"name", "color", "shades"}]``; None when the project never listed them.
    segmentation_labels: Optional[list] = None
    # ``[{"name", "classes", "default"}]``, one entry per multiclass task.
    classification_tasks: Optional[list] = None
    # ``{"name", "classes", "default"}``: the application has one multilabel task.
    multilabel_task: Optional[dict] = None
    text_fields: Optional[list] = None
    # Keys this version does not know about, written back untouched so that
    # saving the config never drops what a newer application stored.
    extra: dict = field(default_factory=dict, repr=False)

    def to_json(self) -> str:
        """Serialize config to JSON string."""
        data = dict(self.extra)
        data.update({k: v for k, v in self.__dict__.items() if k != "extra"})
        if data["embed_threshold_kb"] is None:
            data["embed_threshold_kb"] = 100
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "ProjectConfig":
        """Deserialize config from JSON string."""
        data = json.loads(json_str)
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        config = cls(**{k: v for k, v in data.items() if k in known})
        config.extra = {k: v for k, v in data.items() if k not in known}
        if config.embed_threshold_kb is None:
            config.embed_threshold_kb = 100
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
class TextDescription:
    """Text description annotation for a frame."""

    id: Optional[int] = None
    frame_id: int = 0
    label_name: str = ""
    content: str = ""
    modified_at: datetime = field(default_factory=datetime.now)
