#!/usr/bin/env python3
"""Import the Wave-6 labelpack packages into the local CVAT labelfactory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
LANE_DIR = REPO_ROOT / "runs/lanes/w6_labelpack_20260708"
PACKAGE_MANIFEST = REPO_ROOT / "cvat_upload/w6_labelpack_20260708/package_manifest.json"
W5_PACKAGE_MANIFEST = REPO_ROOT / "cvat_upload/w5_labelpack_20260708/package_manifest.json"
CREDS_PATH = REPO_ROOT / "runs/lanes/w3_labelfactory_20260707/cvat_credentials.txt"
DEFAULT_PROJECT_NAME = "racketsport_w6_ball_sst_20260708"
SCRATCH_LABELING_MODE = "scratch"


def load_creds(path: Path) -> dict[str, str]:
    creds: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            creds[key] = value
    missing = sorted({"ADMIN_USERNAME", "ADMIN_PASSWORD"} - set(creds))
    if missing:
        raise ValueError(f"missing CVAT credential keys in {path}: {missing}")
    return creds


def ball_label_spec() -> list[dict[str, Any]]:
    return [
        {
            "name": "ball",
            "type": "rectangle",
            "attributes": [
                {"name": "visibility", "mutable": True, "input_type": "checkbox", "default_value": "true", "values": ["true", "false"]},
                {
                    "name": "visibility_level",
                    "mutable": True,
                    "input_type": "select",
                    "default_value": "clear",
                    "values": ["clear", "partial", "full", "out_of_frame"],
                },
                {"name": "center_convention", "mutable": True, "input_type": "text", "default_value": "", "values": []},
                {"name": "blur_angle_deg", "mutable": True, "input_type": "number", "default_value": "0", "values": ["0", "360", "1"]},
                {"name": "blur_length_px", "mutable": True, "input_type": "number", "default_value": "0", "values": ["0", "5000", "1"]},
                {"name": "blur_width_px", "mutable": True, "input_type": "number", "default_value": "0", "values": ["0", "5000", "1"]},
                {"name": "blur_label_quality", "mutable": True, "input_type": "text", "default_value": "", "values": []},
            ],
        }
    ]


def build_import_jobs(package_manifest: Path = PACKAGE_MANIFEST) -> list[dict[str, Any]]:
    manifest = json.loads(package_manifest.read_text(encoding="utf-8"))
    labeling_mode = str(manifest.get("labeling_mode") or "prelabel_review")
    if labeling_mode not in {"prelabel_review", SCRATCH_LABELING_MODE}:
        raise ValueError(f"unsupported labelpack labeling_mode: {labeling_mode}")
    jobs: list[dict[str, Any]] = []
    for item in manifest["ball_sessions"]:
        if labeling_mode == "prelabel_review":
            expected = f"w6_ball_sst_{item['session_id']}_20260708"
            if item["task_name"] != expected:
                raise ValueError(f"unexpected w6 task name scheme: {item['task_name']} != {expected}")
            prelabel_zip: str | None = str(REPO_ROOT / item["prelabel_zip"])
        else:
            if item.get("prelabels_present") is not False:
                raise ValueError("scratch labelpack session must declare prelabels_present=false")
            if item.get("prelabel_zip"):
                raise ValueError("scratch labelpack session must not provide prelabel_zip")
            prelabel_zip = None
        jobs.append(
            {
                "kind": "ball",
                "project_name": str(item.get("project_name") or manifest.get("project_name") or DEFAULT_PROJECT_NAME),
                "task_name": item["task_name"],
                "labels": ball_label_spec(),
                "image_zip": str(REPO_ROOT / item["image_zip"]),
                "prelabel_zip": prelabel_zip,
                "labeling_mode": labeling_mode,
                "frame_count": item["frame_count"],
            }
        )

    w5_names = load_w5_task_names()
    w6_names = {str(job["task_name"]) for job in jobs}
    collisions = sorted(w6_names & w5_names)
    if collisions:
        raise ValueError(f"w6 task names collide with w5 manifest: {collisions}")
    return jobs


def import_labelpack_tasks(*, client: Any, jobs: list[dict[str, Any]], ledger_path: Path) -> dict[str, Any]:
    ledger = _load_import_ledger(ledger_path)
    existing_by_name = _existing_tasks_by_name(ledger, client)
    project_name = _single_project_name(jobs)
    project = None
    report = {
        "schema_version": 2,
        "status": "imported",
        "projects": {},
        "tasks": [],
        "summary": {
            "requested_task_count": len(jobs),
            "created_task_count": 0,
            "skipped_task_count": 0,
        },
    }

    for job in jobs:
        fingerprint = task_fingerprint(job)
        existing = existing_by_name.get(str(job["task_name"]))
        if existing is not None:
            report["tasks"].append(_skipped_task_record(job, existing, fingerprint))
            report["summary"]["skipped_task_count"] += 1
            continue

        if project is None:
            project = _ensure_project(client, project_name, job["labels"], ledger)
            report["projects"]["ball"] = {"project_id": project.id, "project_name": project.name}

        task = client.tasks.create_from_data(
            spec={"name": job["task_name"], "project_id": project.id},
            resources=[job["image_zip"]],
            data_params={"image_quality": 95, "use_cache": True},
        )
        if job.get("prelabel_zip"):
            task.import_annotations("CVAT 1.1", job["prelabel_zip"])
        report["tasks"].append(
            {
                "kind": job["kind"],
                "task_id": task.id,
                "task_name": task.name,
                "task_fingerprint": fingerprint,
                "project_id": project.id,
                "image_zip": job["image_zip"],
                "prelabel_zip": job["prelabel_zip"],
                "frame_count": job["frame_count"],
                "status": "imported",
            }
        )
        report["summary"]["created_task_count"] += 1

    if report["summary"]["created_task_count"] == 0:
        report["status"] = "skipped"
        report["projects"] = ledger.get("projects", {})

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def task_fingerprint(job: Mapping[str, Any]) -> str:
    prelabel_zip = job.get("prelabel_zip")
    payload = {
        "frame_count": job.get("frame_count"),
        "image_zip": str(job.get("image_zip")),
        "kind": str(job.get("kind")),
        "labeling_mode": str(job.get("labeling_mode") or "prelabel_review"),
        "prelabel_zip": None if prelabel_zip is None else str(prelabel_zip),
        "project_name": str(job.get("project_name")),
        "task_name": str(job.get("task_name")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_w5_task_names() -> set[str]:
    if not W5_PACKAGE_MANIFEST.is_file():
        return set()
    manifest = json.loads(W5_PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    names = {str(item["task_name"]) for item in manifest.get("ball_sessions", []) if isinstance(item, dict) and item.get("task_name")}
    court = manifest.get("court_kp_relabel")
    if isinstance(court, dict) and court.get("task_name"):
        names.add(str(court["task_name"]))
    return names


def _load_import_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_tasks_by_name(ledger: Mapping[str, Any], client: Any) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for task in ledger.get("tasks", []):
        if isinstance(task, Mapping) and task.get("task_name"):
            existing[str(task["task_name"])] = {**task, "matched_by": "ledger"}
    for task in _safe_list(getattr(client, "tasks", None)):
        name = getattr(task, "name", None)
        if name and str(name) not in existing:
            existing[str(name)] = {"task_name": str(name), "task_id": getattr(task, "id", None), "matched_by": "cvat_api"}
    return existing


def _safe_list(collection: Any) -> list[Any]:
    if collection is None or not hasattr(collection, "list"):
        return []
    try:
        values = collection.list()
    except TypeError:
        return []
    return list(values or [])


def _single_project_name(jobs: Iterable[Mapping[str, Any]]) -> str:
    names = {str(job.get("project_name") or DEFAULT_PROJECT_NAME) for job in jobs}
    if len(names) != 1:
        raise ValueError(f"expected one CVAT project name, got {sorted(names)}")
    return next(iter(names))


def _ensure_project(client: Any, project_name: str, labels: list[dict[str, Any]], ledger: Mapping[str, Any]) -> Any:
    for project in _safe_list(getattr(client, "projects", None)):
        if getattr(project, "name", None) == project_name:
            return project
    project_ledger = ledger.get("projects", {}).get("ball") if isinstance(ledger.get("projects"), Mapping) else None
    if isinstance(project_ledger, Mapping) and project_ledger.get("project_name") == project_name and project_ledger.get("project_id"):
        class _LedgerProject:
            pass

        project = _LedgerProject()
        project.id = project_ledger["project_id"]
        project.name = project_name
        return project
    return client.projects.create(spec={"name": project_name, "labels": labels})


def _skipped_task_record(job: Mapping[str, Any], existing: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    return {
        "kind": job["kind"],
        "task_id": existing.get("task_id"),
        "task_name": job["task_name"],
        "task_fingerprint": existing.get("task_fingerprint") or fingerprint,
        "matched_by": existing.get("matched_by", "ledger"),
        "image_zip": job["image_zip"],
        "prelabel_zip": job["prelabel_zip"],
        "frame_count": job["frame_count"],
        "status": "skipped",
        "skip_reason": "already_imported",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--creds", type=Path, default=CREDS_PATH)
    parser.add_argument("--package-manifest", type=Path, default=PACKAGE_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", type=Path, default=LANE_DIR / "import_report.json")
    args = parser.parse_args()

    jobs = build_import_jobs(args.package_manifest)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "job_count": len(jobs),
                    "package_manifest": str(args.package_manifest),
                    "task_names": [job["task_name"] for job in jobs],
                    "w5_task_name_collision_count": 0,
                    "jobs": jobs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    from cvat_sdk import make_client

    creds = load_creds(args.creds)
    with make_client(host=args.host, port=args.port, credentials=(creds["ADMIN_USERNAME"], creds["ADMIN_PASSWORD"])) as client:
        client.check_server_version = lambda *a, **k: None
        report = import_labelpack_tasks(client=client, jobs=jobs, ledger_path=args.ledger)
        for task in report["tasks"]:
            if task["status"] == "imported":
                print(f"imported kind={task['kind']} task_id={task['task_id']} name={task['task_name']}")
            else:
                print(f"skipped kind={task['kind']} task_id={task['task_id']} name={task['task_name']} reason={task['skip_reason']}")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
