"""Schema 7 projects: microscopy frames as multi-page TIFFs, shown through display settings."""
import io
import json
import sqlite3

import numpy as np
import pytest
import tifffile
from PIL import Image

from didascalie import DidascalieProject, FrameSource
from didascalie.pixels import auto_display, render_rgb
from didascalie.schema import SCHEMA_VERSION

# The tables version 7 of the application adds (`src-tauri/src/storage/schema.rs`).
V7_TABLES = """
CREATE TABLE IF NOT EXISTS sequence_display (
    sequence_id INTEGER PRIMARY KEY REFERENCES sequences(id) ON DELETE CASCADE,
    settings TEXT NOT NULL,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS frame_sources (
    frame_id INTEGER PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    series INTEGER NOT NULL,
    level INTEGER NOT NULL DEFAULT 0,
    t INTEGER NOT NULL,
    z INTEGER,
    channels TEXT NOT NULL,
    physical_size_x REAL,
    physical_size_y REAL,
    physical_size_z REAL
);
"""

H, W = 5, 7


def tiff_pages(planes: np.ndarray) -> bytes:
    """One greyscale page per channel, written as the microscopy import does."""
    buf = io.BytesIO()
    with tifffile.TiffWriter(buf) as tif:
        for plane in planes:
            tif.write(
                plane,
                photometric="minisblack",
                compression="zlib",
                predictor=plane.dtype.kind in "ui",
            )
    return buf.getvalue()


def add_raw_frame(path, sequence: str, data: bytes, source: dict = None) -> int:
    conn = sqlite3.connect(str(path))
    seq = conn.execute("SELECT id FROM sequences WHERE name = ?", (sequence,)).fetchone()
    if seq is None:
        seq_id = conn.execute("INSERT INTO sequences (name) VALUES (?)", (sequence,)).lastrowid
    else:
        seq_id = seq[0]
    index = conn.execute(
        "SELECT COUNT(*) FROM frames WHERE sequence_id = ?", (seq_id,)
    ).fetchone()[0]
    frame_id = conn.execute(
        "INSERT INTO frames (sequence_id, frame_index, embedded_data, width, height)"
        " VALUES (?, ?, ?, ?, ?)",
        (seq_id, index, data, W, H),
    ).lastrowid
    if source is not None:
        conn.execute(
            "INSERT INTO frame_sources (frame_id, source_path, series, level, t, z, channels,"
            " physical_size_x, physical_size_y, physical_size_z)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (frame_id, source["path"], 1, 0, 2, source.get("z"), json.dumps([0, 2]),
             0.25, 0.25, None),
        )
    conn.commit()
    conn.close()
    return frame_id


def set_display(path, sequence: str, settings: dict) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO sequence_display (sequence_id, settings)"
        " SELECT id, ? FROM sequences WHERE name = ?",
        (json.dumps(settings), sequence),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def v7_path(tmp_path):
    path = tmp_path / "v7.dida"
    DidascalieProject.create(path, name="raw").close()
    conn = sqlite3.connect(str(path))
    conn.executescript(V7_TABLES)
    conn.execute("PRAGMA user_version = 7")
    conn.commit()
    conn.close()
    return path


def two_channels() -> np.ndarray:
    a = np.linspace(100, 4000, H * W).reshape(H, W).astype(np.uint16)
    b = np.full((H, W), 1000, np.uint16)
    return np.stack([a, b])


def test_schema_7_opens_and_8_is_refused(tmp_path, v7_path):
    assert SCHEMA_VERSION == 7
    with DidascalieProject(v7_path) as project:
        assert project.schema_version == 7

    newer = tmp_path / "v8.dida"
    DidascalieProject.create(newer).close()
    conn = sqlite3.connect(str(newer))
    conn.execute("PRAGMA user_version = 8")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="schema version 8"):
        DidascalieProject(newer)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_raw_returns_pages_as_channels_last(v7_path, dtype):
    planes = (two_channels() / 16).astype(dtype)
    frame_id = add_raw_frame(v7_path, "cells", tiff_pages(planes))
    with DidascalieProject(v7_path) as project:
        raw = project.get_frame_image(frame_id, raw=True)
    assert raw.dtype == dtype
    assert raw.shape == (H, W, 2)
    np.testing.assert_array_equal(raw, np.moveaxis(planes, 0, -1))


def test_render_uses_the_sequence_display(v7_path):
    planes = two_channels()
    frame_id = add_raw_frame(v7_path, "cells", tiff_pages(planes))
    settings = {"channels": [
        {"name": "DAPI", "visible": True, "color": "#ff0000", "min": 100, "max": 4000, "gamma": 1},
        {"name": "GFP", "visible": True, "color": "#00ff80", "min": 0, "max": 2000, "gamma": 2},
    ]}
    set_display(v7_path, "cells", settings)
    with DidascalieProject(v7_path) as project:
        image = project.get_frame_image(frame_id)
        assert project.get_display_settings(frame_id) == settings
    assert image.mode == "RGB" and image.size == (W, H)
    rgb = np.asarray(image)
    a = np.clip((planes[0] - 100) / 3900, 0, 1)
    b = 0.5 ** 0.5  # (1000 - 0) / 2000, gamma 2
    np.testing.assert_allclose(rgb[..., 0], np.round(a * 255), atol=1)
    assert np.all(rgb[..., 1] == round(b * 255))
    assert np.all(np.abs(rgb[..., 2].astype(int) - round(b * 128 / 255 * 255)) <= 1)

    # A hidden channel adds nothing.
    settings["channels"][1]["visible"] = False
    only_a = render_rgb(np.moveaxis(planes, 0, -1), settings)
    assert not only_a[..., 1:].any()


def test_render_without_settings_is_automatic(v7_path):
    planes = two_channels()
    frame_id = add_raw_frame(v7_path, "cells", tiff_pages(planes))
    with DidascalieProject(v7_path) as project:
        display = project.get_display_settings(frame_id)
        rgb = np.asarray(project.get_frame_image(frame_id))
    first, second = display["channels"]
    assert (first["color"], second["color"]) == ("#00ff00", "#ff00ff")
    assert 100 <= first["min"] < first["max"] <= 4000
    # A flat channel gets a one-wide window starting at its value: black.
    assert (second["min"], second["max"]) == (1000, 1001)
    assert rgb.dtype == np.uint8 and rgb.shape == (H, W, 3)
    # Green growing with the first channel, nothing from the flat one.
    assert not rgb[..., 0].any() and not rgb[..., 2].any()
    assert rgb[0, 0, 1] == 0 and rgb[-1, -1, 1] == 255


def test_settings_that_do_not_fit_fall_back_to_auto(v7_path):
    frame_id = add_raw_frame(v7_path, "cells", tiff_pages(two_channels()))
    set_display(v7_path, "cells", {"channels": [
        {"name": None, "visible": True, "color": "#ffffff", "min": 0, "max": 1, "gamma": 1},
    ]})
    with DidascalieProject(v7_path) as project:
        assert len(project.get_display_settings(frame_id)["channels"]) == 2


def test_auto_colours():
    one = np.zeros((2, 2, 1), np.uint16)
    three = np.zeros((2, 2, 3), np.uint8)
    four = np.zeros((2, 2, 4), np.uint16)
    assert [c["color"] for c in auto_display(one)["channels"]] == ["#ffffff"]
    rgb = auto_display(three)["channels"]
    assert [c["color"] for c in rgb] == ["#ff0000", "#00ff00", "#0000ff"]
    assert all((c["min"], c["max"]) == (0, 255) for c in rgb)
    assert [c["color"] for c in auto_display(four)["channels"]] == [
        "#00ff00", "#ff00ff", "#00ffff", "#ff0000"]


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_single_page_deep_tiff(v7_path, dtype):
    plane = np.linspace(0, 1000, H * W).reshape(H, W).astype(dtype)
    frame_id = add_raw_frame(v7_path, "scan", tiff_pages(plane[None]))
    with DidascalieProject(v7_path) as project:
        raw = project.get_frame_image(frame_id, raw=True)
        image = project.get_frame_image(frame_id)
    assert raw.shape == (H, W, 1) and raw.dtype == dtype
    np.testing.assert_array_equal(raw[..., 0], plane)
    rgb = np.asarray(image)
    assert image.mode == "RGB"
    assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
    assert rgb[0, 0, 0] == 0 and rgb[-1, -1, 0] == 255


def test_16_bit_png(v7_path):
    plane = np.linspace(0, 60000, H * W).reshape(H, W).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(plane).save(buf, format="PNG")
    frame_id = add_raw_frame(v7_path, "scan", buf.getvalue())
    with DidascalieProject(v7_path) as project:
        raw = project.get_frame_image(frame_id, raw=True)
        assert project.get_frame_image(frame_id).mode == "RGB"
    assert raw.dtype == np.uint16
    np.testing.assert_array_equal(raw[..., 0], plane)


def test_8_bit_frames_are_returned_as_stored(v7_path):
    with DidascalieProject(v7_path) as project:
        seq = project.add_sequence("photos")
        grey = np.arange(H * W, dtype=np.uint8).reshape(H, W)
        frame_id = project.add_frame(seq, grey)
        image = project.get_frame_image(frame_id)
        assert image.mode == "L"
        np.testing.assert_array_equal(np.asarray(image), grey)
        raw = project.get_frame_image(frame_id, raw=True)
        assert raw.shape == (H, W, 1) and raw.dtype == np.uint8
        assert project.get_frame_source(frame_id) is None


def test_frame_sources(v7_path):
    projected = add_raw_frame(v7_path, "cells", tiff_pages(two_channels()),
                              {"path": "data/cells.czi"})
    sliced = add_raw_frame(v7_path, "cells", tiff_pages(two_channels()),
                           {"path": "data/cells.czi", "z": 4})
    other = add_raw_frame(v7_path, "other", tiff_pages(two_channels()), {"path": "/abs/b.lif"})
    with DidascalieProject(v7_path) as project:
        source = project.get_frame_source(projected)
        assert source == FrameSource(
            frame_id=projected, source_path="data/cells.czi", series=1, level=0, t=2,
            z=None, channels=[0, 2], physical_size_x=0.25, physical_size_y=0.25,
            physical_size_z=None,
        )
        assert project.get_frame_source(sliced).z == 4
        cells = project.get_sequence_by_name("cells").id
        assert [s.frame_id for s in project.get_frame_sources(cells)] == [projected, sliced]
        assert [s.frame_id for s in project.get_frame_sources()] == [projected, sliced, other]


def test_projects_before_7_have_no_sources(tmp_path):
    path = tmp_path / "v3.dida"
    with DidascalieProject.create(path) as project:
        assert project.get_frame_sources() == []
        assert project.get_frame_source(1) is None


def test_enable_user_accounts_stamps_version_4(tmp_path):
    path = tmp_path / "v3.dida"
    with DidascalieProject.create(path) as project:
        project.enable_user_accounts()
        # The application adds what came after 4 the next time it opens it.
        assert project.schema_version == 4
