"""Regression values: declared in the config, stored one per frame, task and user."""
import json
import sqlite3

import numpy as np
import pytest

from didascalie import DidascalieProject, ProjectConfig


@pytest.fixture
def project(tmp_path):
    with DidascalieProject.create(tmp_path / "p.dida") as p:
        yield p


def add_frame(project, name="a"):
    sequence = project.get_or_create_sequence(name)
    return project.add_frame(sequence.id, np.zeros((4, 6, 3), np.uint8))


def stored_config(project):
    row = project._conn.execute("SELECT config FROM project WHERE id = 1").fetchone()
    return json.loads(row["config"])


def test_declaring_a_task_writes_it_where_the_application_reads_it(project):
    project.add_regression_task("score", min=0, max=1)
    project.add_regression_task("age", dtype="int32", min=0)

    config = stored_config(project)
    assert config["regression_enabled"] is True
    assert config["regression_tasks"] == [
        {"name": "score", "dtype": "float32", "min": 0.0, "max": 1.0},
        {"name": "age", "dtype": "int32", "min": 0.0, "max": None},
    ]
    assert project.get_regression_tasks()["age"] == {"dtype": "int32", "min": 0.0, "max": None}


def test_declaring_again_replaces_type_and_range_and_keeps_values(project):
    frame = add_frame(project)
    project.add_regression_task("score", min=0, max=1)
    project.add_regression(frame, "score", 0.5)

    project.add_regression_task("score", dtype="float64", min=0, max=10)
    assert len(project.config.regression_tasks) == 1
    assert project.get_regression_tasks()["score"]["max"] == 10.0
    assert project.get_regression(frame, "score") == 0.5


def test_a_bad_declaration_is_refused(project):
    with pytest.raises(ValueError):
        project.add_regression_task("x", dtype="uint8")
    with pytest.raises(ValueError):
        project.add_regression_task("x", min=1, max=1)
    with pytest.raises(ValueError):
        project.add_regression_task("x", max=float("inf"))
    assert project.config.regression_tasks is None


def test_a_frame_holds_one_value_per_task(project):
    first, second = add_frame(project, "a"), add_frame(project, "b")
    project.add_regression_task("score", dtype="float64")
    project.add_regression_task("age", dtype="int32")

    project.add_regression(first, "score", 0.25)
    project.add_regression(first, "age", 41)
    project.add_regression(first, "age", 42)

    assert project.get_regressions_for_frame(first) == {"score": 0.25, "age": 42.0}
    assert project.get_regressions_for_frame(second) == {}
    assert project.get_regression(second, "age") is None
    assert project.get_statistics()["regressions"] == 2

    project.delete_regression(first, "age")
    assert project.get_regressions_for_frame(first) == {"score": 0.25}


def test_a_value_is_stored_in_the_type_of_its_task(project):
    frame = add_frame(project)
    project.add_regression_task("single")
    project.add_regression_task("whole", dtype="int32")

    stored = project.add_regression(frame, "single", 0.1)
    assert stored == float(np.float32(0.1)) != 0.1
    assert project.get_regression(frame, "single") == stored

    # Half away from zero, as the application rounds.
    assert project.add_regression(frame, "whole", 2.5) == 3.0
    assert project.add_regression(frame, "whole", -2.5) == -3.0
    assert project.add_regression(frame, "whole", 41.4) == 41.0


def test_a_value_out_of_range_or_not_finite_is_refused(project):
    frame = add_frame(project)
    project.add_regression_task("score", min=0, max=1)
    project.add_regression_task("age", dtype="int32", min=0)

    for task, value in [("score", 1.5), ("score", -0.1), ("age", -1), ("score", float("nan"))]:
        with pytest.raises(ValueError):
            project.add_regression(frame, task, value)
    with pytest.raises(ValueError):
        project.add_regression(frame, "age", 1e12)  # does not fit in int32
    assert project.get_regressions_for_frame(frame) == {}

    # The bounds themselves are accepted.
    assert project.add_regression(frame, "score", 1) == 1.0
    assert project.add_regression(frame, "score", 0) == 0.0


def test_an_undeclared_task_is_declared_as_an_unbounded_float32(project):
    frame = add_frame(project)
    project.add_regression(frame, "height", 1.75)
    assert project.get_regression_tasks() == {
        "height": {"dtype": "float32", "min": None, "max": None}
    }
    assert project.config.regression_enabled is True


def test_import_with_masks_takes_regression_values(project):
    project.add_regression_task("score", min=0, max=1)
    frame = project.import_with_masks(
        np.zeros((4, 6, 3), np.uint8), masks={}, regression={"score": 0.5, "count": 3}
    )
    assert project.get_regressions_for_frame(frame) == {"score": 0.5, "count": 3.0}


def test_a_project_without_regression_reads_as_empty(project):
    frame = add_frame(project)
    assert project.get_regression_tasks() == {}
    assert project.get_regressions_for_frame(frame) == {}
    assert project.get_statistics()["regressions"] == 0
    project.delete_regression(frame, "score")  # nothing to do, no error
    # Nothing was created just by reading.
    assert not project._has_table("regressions")


def test_a_config_from_before_regression_reads_as_off():
    config = ProjectConfig.from_json(json.dumps({"name": "old"}))
    assert config.regression_enabled is False
    assert config.regression_tasks is None


def test_a_created_project_keeps_the_layout_without_accounts(project):
    """The application gives the rows of this table to the first account when
    it upgrades the project, so the table must not claim a user itself."""
    project.add_regression_task("score")
    assert project.schema_version == 3
    assert not project._has_column("regressions", "user_id")


def test_enabling_accounts_gives_the_values_to_the_owner(project):
    frame = add_frame(project)
    project.add_regression_task("score", dtype="float64")
    project.add_regression(frame, "score", 0.5)

    project.enable_user_accounts("Ada")
    assert project._has_column("regressions", "user_id")
    assert project.get_regression(frame, "score") == 0.5

    other = project.add_user("model")
    project.set_user(other.id)
    assert project.get_regression(frame, "score") is None
    project.add_regression(frame, "score", 0.75)
    assert project.get_regression(frame, "score") == 0.75

    project.set_user("Ada")
    assert project.get_regression(frame, "score") == 0.5


def test_a_project_with_accounts_gets_the_per_user_table(project):
    project.enable_user_accounts()
    frame = add_frame(project)
    project.add_regression(frame, "score", 1.0)
    assert project._has_column("regressions", "user_id")
    assert project.get_regression(frame, "score") == 1.0


def test_a_project_stamped_by_the_application_with_regression_opens(tmp_path):
    path = tmp_path / "p.dida"
    DidascalieProject.create(path).close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 8")
    conn.commit()
    conn.close()
    with DidascalieProject(path) as project:
        assert project.schema_version == 8
