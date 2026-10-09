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

### Instance Segmentation

A label created with `is_instance=True` stores an instance id per pixel: 0 is
background, and ids go from 1 to 255 within each frame.

Instance segmentation is a setting of the whole project, as in the application:
in an instance project, every label is an instance label.

```python
import numpy as np
from didascalie import DidascalieProject, Label, ProjectConfig

config = ProjectConfig(instance_segmentation_enabled=True)
with DidascalieProject.create("nuclei.dida", config=config) as project:
    project.add_label(Label(name="nucleus", color="#FF0000", is_instance=True))

    ids = np.load("instance_ids.npy")  # H x W, one id per nucleus
    frame_id = project.import_with_masks(image="patch.png", masks={"nucleus": ids})

    # Read the ids back; without raw=True the mask is binary (0 or 255)
    label = project.get_label_by_name("nucleus")
    height, width = ids.shape
    ids = project.get_annotation(frame_id, label.id, width, height, raw=True)
```

### Classification and Text

Tasks are declared in the project, which is where the application looks for
them. Adding a classification declares its task and classes if needed, so this is
enough:

```python
project.add_classification(frame_id, "grade", ["mild"])              # multiclass
project.add_classification(frame_id, "findings", ["drusen", "edema"])  # multilabel
project.add_text_description(frame_id, "note", "Blurry at the edges")
```

To declare them up front, with classes no frame uses yet:

```python
project.add_classification_task("grade", ["none", "mild", "severe"])
project.set_multilabel_task("findings", ["drusen", "edema", "haemorrhage"])
project.add_text_field("note")
```

A project has any number of multiclass tasks (one class per frame) and one
multilabel task.

### Users

Recent versions of the application give each annotator an account, with their
own masks, classifications, texts and reviews. In such a project you read and
write as one user: by default the first administrator, who owns everything
annotated before accounts existed.

```python
with DidascalieProject("study.dida", user="Alice") as project:
    print([u.name for u in project.get_users()])
    masks = project.get_annotations_for_frame(frame_id)   # Alice's

    project.set_user("Bob")
    masks = project.get_annotations_for_frame(frame_id)   # Bob's

    # Keep a model's predictions apart from the annotators' work
    model = project.add_user("model")
    project.set_user(model.id)
    project.add_annotation(frame_id, label_id, prediction)
```

A project without accounts has no users (`project.get_users()` is empty) and
works as before.

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
- `__init__(path, user)` - Open an existing project, as a given user if it has accounts

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
- `get_annotation(frame_id, label_id, width, height, raw)` - Get annotation mask
- `get_annotations_for_frame(frame_id, raw)` - Get every label's mask on a frame
- `set_frame_reviewed(frame_id, reviewed)` - Mark a frame reviewed

**Classifications and text:**
- `add_classification(frame_id, task_name, selected_classes, is_multilabel)` - Add classification
- `add_classification_task(name, classes, default)` - Declare a multiclass task
- `set_multilabel_task(name, classes, default)` - Declare the multilabel task
- `get_classification_tasks()` - Declared tasks and their classes
- `add_text_description(frame_id, label_name, content)` - Add a text note
- `add_text_field(name)` - Declare a text field

**Users:**
- `get_users()` - Accounts of the project
- `set_user(user)` - Choose whose annotations are read and written
- `add_user(name, role)` - Add an account

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
- **users** - Accounts, in projects saved by a version of the application that has them

### Compatibility with the application

This version reads and writes projects up to schema version 4 (user accounts).
A project saved by a newer application is refused with a message asking you to
upgrade this package.

Projects created here use the version 3 layout. Every release of the application
since 0.9 opens it, and the releases with user accounts upgrade it the first time
they open it.

Legacy `.labelmed` files can still be opened for backwards compatibility.

## License

MIT License

## Serving Python functions to Didascalie

`didascalie.com` lets the application call functions of yours over ZeroMQ
(needs `pyzmq` and `msgpack`): keypoint proposals for registration, and
segmentation of a frame or a whole sequence from the editor.

```python
from didascalie.com import register_kpts, register_seg, register_sequence_seg, serve

@register_seg
def vessels(image):                       # H x W x 3 uint8
    """Segments the vessels."""
    return {"vessel": model(image) > 0.5}  # {label: H x W}, C x H x W, or H x W

@register_seg
def refine(image, masks, active_label):   # extras are sent only when named
    return sam(image, masks[active_label])  # H x W: added to the active label

@register_sequence_seg
def track(frames, masks, frame_index):    # T x H x W x 3, {label: T x H x W}
    return {"cell": tracker(frames, masks["cell"][frame_index])}

@register_kpts
def match(reference, moving, existing):
    return [((rx, ry), (mx, my)), ...]

serve()  # blocks; 127.0.0.1:5556 is where Didascalie looks by default
```

Segmentation functions show up in the editor on their own while `serve()` is
running. `register` is the former name of `register_kpts` and still works.
See the [Didascalie documentation](https://didascalie.readthedocs.io/experimental/)
for the full contract.
