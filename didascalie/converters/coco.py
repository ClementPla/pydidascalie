"""COCO format converter."""

import json
from pathlib import Path
from typing import Union, Optional

import numpy as np

from ..project import DidascalieProject
from ..models import Label


def import_coco(
    project: DidascalieProject,
    coco_json: Union[str, Path],
    images_folder: Union[str, Path],
    embed: bool = True,
) -> dict:
    """
    Import COCO format annotations into a Didascalie project.

    Args:
        project: Didascalie project instance
        coco_json: Path to COCO JSON file
        images_folder: Path to images folder
        embed: Embed images in database

    Returns:
        Import statistics
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("pycocotools required: pip install pycocotools")

    coco = COCO(str(coco_json))
    images_folder = Path(images_folder)

    stats = {"frames": 0, "annotations": 0, "labels": 0, "errors": []}

    # Import categories as labels
    label_map = {}  # coco_cat_id -> label_id
    for cat in coco.cats.values():
        label = project.get_or_create_label(cat["name"])
        label_map[cat["id"]] = label.id
        stats["labels"] += 1

    # Import images and annotations
    for img_info in coco.imgs.values():
        img_path = images_folder / img_info["file_name"]

        if not img_path.exists():
            stats["errors"].append(f"Image not found: {img_path}")
            continue

        try:
            # Create sequence and frame
            sequence = project.get_or_create_sequence(img_path.stem)
            frame_id = project.add_frame(sequence.id, img_path, embed=embed)
            stats["frames"] += 1

            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_info["id"])
            anns = coco.loadAnns(ann_ids)

            # Group by category (merge instances for semantic segmentation)
            masks_by_label: dict[int, np.ndarray] = {}

            for ann in anns:
                cat_id = ann["category_id"]
                label_id = label_map.get(cat_id)

                if label_id is None:
                    continue

                # Decode mask
                if "segmentation" in ann:
                    if isinstance(ann["segmentation"], dict):
                        # RLE format
                        mask = coco_mask.decode(ann["segmentation"])
                    elif isinstance(ann["segmentation"], list):
                        # Polygon format
                        rles = coco_mask.frPyObjects(
                            ann["segmentation"],
                            img_info["height"],
                            img_info["width"],
                        )
                        mask = coco_mask.decode(coco_mask.merge(rles))
                    else:
                        continue

                    # Merge with existing mask for this label
                    if label_id in masks_by_label:
                        masks_by_label[label_id] = np.maximum(
                            masks_by_label[label_id], mask
                        )
                    else:
                        masks_by_label[label_id] = mask

            # Save annotations
            for label_id, mask in masks_by_label.items():
                project.add_annotation(frame_id, label_id, mask * 255)
                stats["annotations"] += 1

        except Exception as e:
            stats["errors"].append(f"{img_path}: {e}")

    return stats


def export_coco(
    project: DidascalieProject,
    output_json: Union[str, Path],
    output_images: Optional[Union[str, Path]] = None,
    include_empty: bool = False,
) -> dict:
    """
    Export Didascalie project to COCO format.

    Args:
        project: Didascalie project instance
        output_json: Output COCO JSON path
        output_images: Optional folder to export images
        include_empty: Include images without annotations

    Returns:
        Export statistics
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("pycocotools required: pip install pycocotools")

    coco_data = {
        "info": {
            "description": project.config.name,
            "version": "1.0",
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }

    stats = {"images": 0, "annotations": 0}

    # Create output images folder if specified
    if output_images:
        output_images = Path(output_images)
        output_images.mkdir(parents=True, exist_ok=True)

    # Export categories
    for label in project.get_labels():
        coco_data["categories"].append(
            {
                "id": label.id,
                "name": label.name,
                "supercategory": "",
            }
        )

    ann_id = 1

    # Export images and annotations
    for seq, frame in project.iter_frames():
        # Get annotations for this frame
        frame_annotations = project.get_annotations_for_frame(frame.id)

        if not frame_annotations and not include_empty:
            continue

        file_name = frame.relative_path or f"{seq.name}_{frame.frame_index}.png"

        coco_data["images"].append(
            {
                "id": frame.id,
                "file_name": file_name,
                "width": frame.width,
                "height": frame.height,
            }
        )
        stats["images"] += 1

        # Export image file if requested
        if output_images:
            try:
                img = project.get_frame_image(frame.id)
                output_path = output_images / file_name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_path)
            except Exception as e:
                print(f"Warning: Failed to export image {file_name}: {e}")

        # Export annotations
        for label, mask in frame_annotations:
            if mask.max() == 0:
                continue

            # Encode as RLE
            binary_mask = (mask > 0).astype(np.uint8)
            rle = coco_mask.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            # Calculate bounding box
            bbox = list(map(float, coco_mask.toBbox(rle)))

            coco_data["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": frame.id,
                    "category_id": label.id,
                    "segmentation": rle,
                    "area": float(binary_mask.sum()),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            stats["annotations"] += 1

    # Save JSON
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=2)

    return stats
