"""Folder import: files are taken by extension, then filtered by name, as in the application."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from didascalie import DidascalieProject, ProjectConfig
from didascalie.models import IMAGE_EXTENSIONS

OLD_REGEX = r"\.(png|jpe?g|bmp|tiff?)$"


def write_image(path):
    Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(path)


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    write_image(root / "a.png")
    write_image(root / "b.PNG")
    (root / "c.JPG").write_bytes((root / "a.png").read_bytes())  # PIL sniffs the content
    write_image(root / "d.tif")
    (root / "notes.txt").write_text("not an image")
    (root / "clip.mp4").write_bytes(b"\0" * 16)
    (root / "slide.CZI").write_bytes(b"\0" * 16)
    return root


@pytest.fixture
def project(tmp_path):
    with DidascalieProject.create(tmp_path / "p.dida") as p:
        yield p


def imported(project):
    return sorted(s.name for s in project.get_sequences())


def test_config_from_before_extensions_takes_every_image_format():
    old = {"name": "old", "input_regex": OLD_REGEX, "future_key": 1}
    config = ProjectConfig.from_json(json.dumps(old))
    assert config.extensions == IMAGE_EXTENSIONS
    assert config.input_regex == OLD_REGEX
    assert json.loads(config.to_json())["future_key"] == 1
    # The application writes null for "every format".
    assert ProjectConfig.from_json(json.dumps({"extensions": None})).extensions == IMAGE_EXTENSIONS


def test_extensions_round_trip():
    config = ProjectConfig(extensions=["png"])
    assert ProjectConfig.from_json(config.to_json()).extensions == ["png"]


def test_default_takes_images_and_not_text(project, folder):
    assert project.config.input_regex == ".*"
    stats = project.import_folder(folder)
    assert imported(project) == ["a", "b", "c", "d"]
    assert stats["frames"] == 4 and not stats["errors"] and not stats["skipped"]


def test_videos_and_microscopy_enabled_in_the_app_are_skipped(project, folder):
    config = project.config
    config.extensions = IMAGE_EXTENSIONS + ["mp4", "czi"]
    project.update_config(config)
    with pytest.warns(UserWarning, match="cannot decode"):
        stats = project.import_folder(folder)
    assert imported(project) == ["a", "b", "c", "d"]
    assert sorted(Path(p).name for p in stats["skipped"]) == ["clip.mp4", "slide.CZI"]


def test_mixed_case_extensions(project, folder):
    stats = project.import_folder(folder, extensions=[".PNG", "Jpg"])
    assert imported(project) == ["a", "b", "c"]
    assert stats["skipped"] == []


def test_old_default_regex_still_works(project, folder):
    config = project.config
    config.input_regex = OLD_REGEX
    project.update_config(config)
    stats = project.import_folder(folder)
    assert imported(project) == ["a", "b", "c", "d"]
    assert stats["skipped"] == []


def test_empty_pattern_takes_any_name(project, folder):
    project.import_folder(folder, pattern="", extensions=["tif"])
    assert imported(project) == ["d"]


def test_pattern_filters_on_top_of_extensions(project, folder):
    project.import_folder(folder, pattern="^[ab]")
    assert imported(project) == ["a", "b"]


def test_text_not_taken_even_if_listed_by_pattern(project, folder):
    project.import_folder(folder, pattern=r"\.txt$")
    assert imported(project) == []
