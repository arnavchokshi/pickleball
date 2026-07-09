from __future__ import annotations

import importlib
from pathlib import Path

CLI_PATH = Path("scripts/racketsport/import_w6_labelpack_tasks.py")


class _FakeProject:
    def __init__(self, project_id: int, name: str) -> None:
        self.id = project_id
        self.name = name


class _FakeTask:
    def __init__(self, task_id: int, name: str) -> None:
        self.id = task_id
        self.name = name
        self.imported_annotations: list[tuple[str, str]] = []

    def import_annotations(self, fmt: str, path: str) -> None:
        self.imported_annotations.append((fmt, path))


class _FakeProjects:
    def __init__(self) -> None:
        self.created_specs: list[dict] = []
        self._projects: list[_FakeProject] = []

    def create(self, *, spec: dict) -> _FakeProject:
        project = _FakeProject(len(self._projects) + 1, spec["name"])
        self.created_specs.append(spec)
        self._projects.append(project)
        return project

    def list(self) -> list[_FakeProject]:
        return list(self._projects)


class _FakeTasks:
    def __init__(self) -> None:
        self.created_specs: list[dict] = []
        self._tasks: list[_FakeTask] = []

    def create_from_data(self, *, spec: dict, resources: list[str], data_params: dict) -> _FakeTask:
        task = _FakeTask(len(self._tasks) + 100, spec["name"])
        self.created_specs.append({"spec": spec, "resources": resources, "data_params": data_params})
        self._tasks.append(task)
        return task

    def list(self) -> list[_FakeTask]:
        return list(self._tasks)


class _FakeClient:
    def __init__(self) -> None:
        self.projects = _FakeProjects()
        self.tasks = _FakeTasks()


def _job(name: str, image_zip: Path, prelabel_zip: Path) -> dict:
    return {
        "kind": "ball",
        "project_name": "racketsport_w6_ball_sst_20260708",
        "task_name": name,
        "labels": [{"name": "ball", "type": "rectangle"}],
        "image_zip": str(image_zip),
        "prelabel_zip": str(prelabel_zip),
        "frame_count": 640,
    }


def test_w6_labelpack_import_second_run_skips_existing_tasks(tmp_path: Path) -> None:
    assert CLI_PATH.is_file()

    try:
        importer = importlib.import_module("scripts.racketsport.import_w6_labelpack_tasks")
    except ModuleNotFoundError as exc:
        raise AssertionError("expected scripts.racketsport.import_w6_labelpack_tasks to exist") from exc

    client = _FakeClient()
    jobs = [
        _job("w6_ball_sst_ball_session_01_20260708", tmp_path / "images_1.zip", tmp_path / "labels_1.zip"),
        _job("w6_ball_sst_ball_session_02_20260708", tmp_path / "images_2.zip", tmp_path / "labels_2.zip"),
    ]
    ledger = tmp_path / "import_report.json"

    first = importer.import_labelpack_tasks(client=client, jobs=jobs, ledger_path=ledger)
    second = importer.import_labelpack_tasks(client=client, jobs=jobs, ledger_path=ledger)

    assert first["summary"]["created_task_count"] == 2
    assert first["summary"]["skipped_task_count"] == 0
    assert second["status"] == "skipped"
    assert second["summary"]["created_task_count"] == 0
    assert second["summary"]["skipped_task_count"] == 2
    assert [task["status"] for task in second["tasks"]] == ["skipped", "skipped"]
    assert {task["skip_reason"] for task in second["tasks"]} == {"already_imported"}
    assert len(client.projects.created_specs) == 1
    assert len(client.tasks.created_specs) == 2
