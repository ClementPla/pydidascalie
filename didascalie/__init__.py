"""
Didascalie - Python library for creating and manipulating Didascalie annotation files.

Example usage:
    from didascalie import DidascalieProject, Label

    # Create a new project
    project = DidascalieProject.create("my_project.dida", name="My Dataset")

    # Add labels
    project.add_label(Label(name="tumor", color="#FF0000"))
    project.add_label(Label(name="background", color="#00FF00"))

    # Import images
    project.import_folder("/path/to/images")

    # Or import with existing masks
    project.import_with_masks(
        image_path="image.png",
        masks={"tumor": tumor_mask_array}
    )

    # Close project
    project.close()
"""

# DidascalieProject is the current name; LabelMedProject is a deprecated alias
# kept for backwards compatibility with code written before the rebrand.
from .project import DidascalieProject, LabelMedProject
from .models import (
    ProjectConfig,
    User,
    Label,
    Sequence,
    Frame,
    Annotation,
    Classification,
    TextDescription,
)

__version__ = "0.2.0"
__all__ = [
    "DidascalieProject",
    "LabelMedProject",  # deprecated alias
    "ProjectConfig",
    "User",
    "Label",
    "Sequence",
    "Frame",
    "Annotation",
    "Classification",
    "TextDescription",
]
