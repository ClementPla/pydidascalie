"""Raw mask converter utilities."""

from pathlib import Path
from typing import Union, Optional, Callable
import re

import numpy as np
from PIL import Image

from ..project import DidascalieProject
from ..models import Label


def import_image_mask_pairs(
    project: DidascalieProject,
    images_folder: Union[str, Path],
    masks_folder: Union[str, Path],
    label_name: str,
    label_color: str = "#FF0000",
    image_pattern: str = r"\.(png|jpe?g|bmp|tiff?)$",
    mask_suffix: str = "_mask",
    embed: bool = True,
) -> dict:
    """
    Import images with corresponding mask files.

    Expects masks to be named like: image_mask.png for image.png

    Args:
        project: Didascalie project instance
        images_folder: Path to images folder
        masks_folder: Path to masks folder
        label_name: Label name for the masks
        label_color: Label color
        image_pattern: Regex pattern for image files
        mask_suffix: Suffix added to image name for mask file
        embed: Embed images in database

    Returns:
        Import statistics
    """
    images_folder = Path(images_folder)
    masks_folder = Path(masks_folder)
    regex = re.compile(image_pattern, re.IGNORECASE)

    stats = {"frames": 0, "annotations": 0, "errors": []}

    # Get or create label
    label = project.get_or_create_label(label_name, label_color)

    # Find image files (one transaction for the whole import)
    with project.bulk():
        for img_path in sorted(images_folder.glob("*")):
            if not img_path.is_file():
                continue
            if not regex.search(img_path.name):
                continue

            # Find corresponding mask
            stem = img_path.stem
            mask_path = None

            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = masks_folder / f"{stem}{mask_suffix}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is None:
                stats["errors"].append(f"No mask found for {img_path.name}")
                continue

            try:
                # Create sequence and frame
                sequence = project.get_or_create_sequence(stem)
                frame_id = project.add_frame(
                    sequence.id,
                    img_path,
                    relative_path=str(img_path.relative_to(images_folder)),
                    embed=embed,
                )
                stats["frames"] += 1

                # Load and add mask
                mask_img = Image.open(mask_path)
                mask_arr = np.array(mask_img)

                # Handle different mask formats
                if mask_arr.ndim == 3:
                    # Use any non-zero channel
                    mask_arr = (mask_arr.any(axis=2) * 255).astype(np.uint8)
                elif mask_arr.ndim == 2:
                    mask_arr = (mask_arr > 0).astype(np.uint8) * 255

                project.add_annotation(frame_id, label.id, mask_arr)
                stats["annotations"] += 1

            except Exception as e:
                stats["errors"].append(f"{img_path}: {e}")

    return stats


def import_multilabel_masks(
    project: DidascalieProject,
    images_folder: Union[str, Path],
    masks_folder: Union[str, Path],
    labels: list[tuple[str, str]],
    image_pattern: str = r"\.(png|jpe?g|bmp|tiff?)$",
    mask_pattern: str = "{stem}_{label}.png",
    embed: bool = True,
) -> dict:
    """
    Import images with multiple mask files per image.

    Args:
        project: Didascalie project instance
        images_folder: Path to images folder
        masks_folder: Path to masks folder
        labels: List of (label_name, color) tuples
        image_pattern: Regex pattern for image files
        mask_pattern: Pattern for mask filenames, use {stem} and {label}
        embed: Embed images in database

    Returns:
        Import statistics
    """
    images_folder = Path(images_folder)
    masks_folder = Path(masks_folder)
    regex = re.compile(image_pattern, re.IGNORECASE)

    stats = {"frames": 0, "annotations": 0, "errors": []}

    # Create labels
    label_objects = {}
    for name, color in labels:
        label_objects[name] = project.get_or_create_label(name, color)

    # Find image files (one transaction for the whole import)
    with project.bulk():
        for img_path in sorted(images_folder.glob("*")):
            if not img_path.is_file():
                continue
            if not regex.search(img_path.name):
                continue

            stem = img_path.stem

            try:
                # Create sequence and frame
                sequence = project.get_or_create_sequence(stem)
                frame_id = project.add_frame(
                    sequence.id,
                    img_path,
                    relative_path=str(img_path.relative_to(images_folder)),
                    embed=embed,
                )
                stats["frames"] += 1

                # Load masks for each label
                for label_name, label in label_objects.items():
                    mask_filename = mask_pattern.format(stem=stem, label=label_name)
                    mask_path = masks_folder / mask_filename

                    if not mask_path.exists():
                        continue

                    mask_img = Image.open(mask_path)
                    mask_arr = np.array(mask_img)

                    if mask_arr.ndim == 3:
                        mask_arr = (mask_arr.any(axis=2) * 255).astype(np.uint8)
                    elif mask_arr.ndim == 2:
                        mask_arr = (mask_arr > 0).astype(np.uint8) * 255

                    if mask_arr.max() > 0:
                        project.add_annotation(frame_id, label.id, mask_arr)
                        stats["annotations"] += 1

            except Exception as e:
                stats["errors"].append(f"{img_path}: {e}")

    return stats


def import_indexed_masks(
    project: DidascalieProject,
    images_folder: Union[str, Path],
    masks_folder: Union[str, Path],
    class_mapping: dict[int, tuple[str, str]],
    image_pattern: str = r"\.(png|jpe?g|bmp|tiff?)$",
    mask_suffix: str = "",
    embed: bool = True,
) -> dict:
    """
    Import images with indexed/semantic segmentation masks.

    Each pixel value in the mask corresponds to a class ID.

    Args:
        project: Didascalie project instance
        images_folder: Path to images folder
        masks_folder: Path to masks folder
        class_mapping: Dict mapping pixel value -> (label_name, color)
        image_pattern: Regex pattern for image files
        mask_suffix: Suffix added to image name for mask file
        embed: Embed images in database

    Returns:
        Import statistics
    """
    images_folder = Path(images_folder)
    masks_folder = Path(masks_folder)
    regex = re.compile(image_pattern, re.IGNORECASE)

    stats = {"frames": 0, "annotations": 0, "errors": []}

    # Create labels
    label_objects = {}
    for pixel_value, (name, color) in class_mapping.items():
        label_objects[pixel_value] = project.get_or_create_label(name, color)

    # Find image files (one transaction for the whole import)
    with project.bulk():
        for img_path in sorted(images_folder.glob("*")):
            if not img_path.is_file():
                continue
            if not regex.search(img_path.name):
                continue

            stem = img_path.stem

            # Find corresponding mask
            mask_path = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = masks_folder / f"{stem}{mask_suffix}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is None:
                stats["errors"].append(f"No mask found for {img_path.name}")
                continue

            try:
                # Create sequence and frame
                sequence = project.get_or_create_sequence(stem)
                frame_id = project.add_frame(
                    sequence.id,
                    img_path,
                    relative_path=str(img_path.relative_to(images_folder)),
                    embed=embed,
                )
                stats["frames"] += 1

                # Load indexed mask
                mask_img = Image.open(mask_path)
                indexed_mask = np.array(mask_img)

                if indexed_mask.ndim == 3:
                    # Convert to single channel
                    indexed_mask = indexed_mask[:, :, 0]

                # Extract binary mask for each class
                for pixel_value, label in label_objects.items():
                    binary_mask = (indexed_mask == pixel_value).astype(np.uint8) * 255

                    if binary_mask.max() > 0:
                        project.add_annotation(frame_id, label.id, binary_mask)
                        stats["annotations"] += 1

            except Exception as e:
                stats["errors"].append(f"{img_path}: {e}")

    return stats


def export_masks(
    project: DidascalieProject,
    output_folder: Union[str, Path],
    format: str = "binary",
    include_images: bool = False,
) -> dict:
    """
    Export annotations as mask files.

    Args:
        project: Didascalie project instance
        output_folder: Output folder path
        format: 'binary' (one file per label) or 'indexed' (single file)
        include_images: Also export source images

    Returns:
        Export statistics
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "masks": 0}

    labels = project.get_labels()

    for seq, frame in project.iter_frames():
        base_name = frame.relative_path or f"{seq.name}_{frame.frame_index}"
        base_name = Path(base_name).stem

        # Export image if requested
        if include_images:
            try:
                img = project.get_frame_image(frame.id)
                img.save(output_folder / f"{base_name}.png")
                stats["images"] += 1
            except Exception as e:
                print(f"Warning: Failed to export image {base_name}: {e}")

        if format == "binary":
            # Export one mask per label
            for label in labels:
                mask = project.get_annotation(
                    frame.id, label.id, frame.width, frame.height
                )
                if mask is not None and mask.max() > 0:
                    mask_img = Image.fromarray(mask)
                    mask_img.save(output_folder / f"{base_name}_{label.name}.png")
                    stats["masks"] += 1

        elif format == "indexed":
            # Export single indexed mask
            indexed = np.zeros((frame.height, frame.width), dtype=np.uint8)

            for i, label in enumerate(labels, start=1):
                mask = project.get_annotation(
                    frame.id, label.id, frame.width, frame.height
                )
                if mask is not None:
                    indexed[mask > 0] = i

            if indexed.max() > 0:
                mask_img = Image.fromarray(indexed)
                mask_img.save(output_folder / f"{base_name}_mask.png")
                stats["masks"] += 1

    return stats
