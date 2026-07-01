# didascalie

Python library for creating and manipulating Didascalie annotation files (`.dida`).

## Installation

Didascalie is not published to PyPI. Install it directly from GitHub:

```bash
pip install git+https://github.com/ClementPla/pydidascalie.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/ClementPla/pydidascalie.git
cd pydidascalie
pip install .
```

## Quick Start

### Create a New Project

```python
from didascalie import DidascalieProject, Label

# Create a new project
with DidascalieProject.create("dataset.dida", name="My Dataset") as project:
    # Add labels
    project.add_label(Label(name="tumor", color="#FF0000"))
    project.add_label(Label(name="background", color="#00FF00"))

    # Import images from folder
    stats = project.import_folder("/path/to/images")
    print(f"Imported {stats['frames']} images in {stats['sequences']} sequences")
```

### Import with Pre-existing Masks

```python
import numpy as np
from didascalie import DidascalieProject, Label

with DidascalieProject.create("annotated.dida") as project:
    project.add_label(Label(name="lesion", color="#FF0000"))
    
    # Load your existing mask (H x W, values 0 or 255)
    mask = np.load("lesion_mask.npy")
    
    # Import image with mask
    project.import_with_masks(
        image="patient_001.png",
        masks={"lesion": mask},
        classification={"diagnosis": ["positive"]},
    )
```

### Convert from COCO Format

```python
from didascalie import DidascalieProject
from didascalie.converters.coco import import_coco

with DidascalieProject.create("from_coco.dida") as project:
    stats = import_coco(
        project,
        coco_json="annotations.json",
        images_folder="images/",
    )
    print(f"Imported {stats['frames']} images, {stats['annotations']} masks")
```

### Read Existing Project

```python
from didascalie import DidascalieProject

with DidascalieProject("existing.dida") as project:
    print(f"Project: {project.config.name}")
    print(f"Labels: {[l.name for l in project.get_labels()]}")
    print(f"Frames: {project.get_frame_count()}")
    
    # Iterate through all frames
    for seq, frame in project.iter_frames():
        print(f"  {seq.name}/{frame.relative_path}")
        
        # Get annotation mask
        for label in project.get_labels():
            mask = project.get_annotation(frame.id, label.id, frame.width, frame.height)
            if mask is not None and mask.max() > 0:
                print(f"    - {label.name}: {mask.sum() // 255} pixels")
```

### Export to COCO Format

```python
from didascalie import DidascalieProject
from didascalie.converters.coco import export_coco

with DidascalieProject("my_project.dida") as project:
    stats = export_coco(
        project,
        output_json="output_coco.json",
    )
    print(f"Exported {stats['images']} images, {stats['annotations']} annotations")
```

## API Reference

### DidascalieProject

The main class for working with Didascalie projects. `LabelMedProject` is kept as a
deprecated alias for backwards compatibility with code written before the rebrand.

#### Class Methods

- `create(path, name, config, overwrite)` - Create a new project
- `__init__(path)` - Open an existing project

#### Instance Methods

**Labels:**
- `add_label(label)` - Add a label
- `get_labels()` - Get all labels
- `get_label_by_name(name)` - Get label by name
- `get_or_create_label(name, color)` - Get or create label

**Sequences:**
- `add_sequence(name, sort_order)` - Add a sequence
- `get_sequences()` - Get all sequences
- `get_or_create_sequence(name)` - Get or create sequence

**Frames:**
- `add_frame(sequence_id, image, frame_index, relative_path, embed)` - Add a frame
- `get_frames(sequence_id)` - Get frames in a sequence
- `get_frame_count()` - Get total frame count

**Annotations:**
- `add_annotation(frame_id, label_id, mask, encoding)` - Add annotation
- `get_annotation(frame_id, label_id, width, height)` - Get annotation mask

**Classifications:**
- `add_classification(frame_id, task_name, selected_classes, is_multilabel)` - Add classification

**Bulk Operations:**
- `import_folder(folder, pattern, recursive, folders_as_sequences, embed)` - Import folder
- `import_with_masks(image, masks, sequence_name, classification)` - Import with masks
- `iter_frames()` - Iterate over all frames

### Data Models

```python
@dataclass
class Label:
    id: Optional[int]
    name: str
    color: str  # Hex color, e.g., "#FF0000"
    is_instance: bool
    sort_order: int

@dataclass
class Sequence:
    id: Optional[int]
    name: str
    sort_order: int
    frame_count: int

@dataclass
class Frame:
    id: Optional[int]
    sequence_id: int
    frame_index: int
    relative_path: Optional[str]
    width: int
    height: int
    reviewed: bool
```

## File Format

Didascalie uses SQLite for storage. The `.dida` file contains:

- **project** - Project configuration (JSON)
- **labels** - Label definitions
- **sequences** - Image sequences/groups
- **frames** - Individual images (can be embedded or referenced)
- **annotations** - Segmentation masks (RLE encoded)
- **classifications** - Classification labels

Legacy `.labelmed` files can still be opened for backwards compatibility.

## License

MIT License
