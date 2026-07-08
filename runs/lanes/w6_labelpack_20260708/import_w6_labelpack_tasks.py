#!/usr/bin/env python3
"""Import the Wave-6 labelpack packages into the local CVAT labelfactory.

Dry-run proof only from this lane:
  runs/lanes/w3_labelfactory_20260707/venv/bin/python runs/lanes/w6_labelpack_20260708/import_w6_labelpack_tasks.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LANE_DIR = REPO_ROOT / "runs/lanes/w6_labelpack_20260708"
PACKAGE_MANIFEST = REPO_ROOT / "cvat_upload/w6_labelpack_20260708/package_manifest.json"
W5_PACKAGE_MANIFEST = REPO_ROOT / "cvat_upload/w5_labelpack_20260708/package_manifest.json"
CREDS_PATH = REPO_ROOT / "runs/lanes/w3_labelfactory_20260707/cvat_credentials.txt"


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


def ball_label_spec() -> list[dict]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--creds", type=Path, default=CREDS_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    jobs: list[dict] = []
    for item in manifest["ball_sessions"]:
        expected = f"w6_ball_sst_{item['session_id']}_20260708"
        if item["task_name"] != expected:
            raise ValueError(f"unexpected w6 task name scheme: {item['task_name']} != {expected}")
        jobs.append(
            {
                "kind": "ball",
                "project_name": "racketsport_w6_ball_sst_20260708",
                "task_name": item["task_name"],
                "labels": ball_label_spec(),
                "image_zip": str(REPO_ROOT / item["image_zip"]),
                "prelabel_zip": str(REPO_ROOT / item["prelabel_zip"]),
                "frame_count": item["frame_count"],
            }
        )

    w5_names = load_w5_task_names()
    w6_names = {job["task_name"] for job in jobs}
    collisions = sorted(w6_names & w5_names)
    if collisions:
        raise ValueError(f"w6 task names collide with w5 manifest: {collisions}")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "job_count": len(jobs),
                    "task_name_scheme": "w6_ball_sst_<session>_20260708",
                    "w5_task_name_collision_count": len(collisions),
                    "jobs": jobs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    from cvat_sdk import make_client

    creds = load_creds(args.creds)
    report = {"schema_version": 1, "status": "imported", "projects": {}, "tasks": []}
    with make_client(host=args.host, port=args.port, credentials=(creds["ADMIN_USERNAME"], creds["ADMIN_PASSWORD"])) as client:
        client.check_server_version = lambda *a, **k: None
        project = client.projects.create(spec={"name": "racketsport_w6_ball_sst_20260708", "labels": ball_label_spec()})
        report["projects"]["ball"] = {"project_id": project.id, "project_name": project.name}
        for job in jobs:
            task = client.tasks.create_from_data(
                spec={"name": job["task_name"], "project_id": project.id},
                resources=[job["image_zip"]],
                data_params={"image_quality": 95, "use_cache": True},
            )
            task.import_annotations("CVAT 1.1", job["prelabel_zip"])
            report["tasks"].append(
                {
                    "kind": job["kind"],
                    "task_id": task.id,
                    "task_name": task.name,
                    "project_id": project.id,
                    "image_zip": job["image_zip"],
                    "prelabel_zip": job["prelabel_zip"],
                    "frame_count": job["frame_count"],
                    "status": "imported",
                }
            )
            print(f"imported kind={job['kind']} task_id={task.id} name={task.name}")

    out = LANE_DIR / "import_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def load_w5_task_names() -> set[str]:
    if not W5_PACKAGE_MANIFEST.is_file():
        return set()
    manifest = json.loads(W5_PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    names = {str(item["task_name"]) for item in manifest.get("ball_sessions", []) if isinstance(item, dict) and item.get("task_name")}
    court = manifest.get("court_kp_relabel")
    if isinstance(court, dict) and court.get("task_name"):
        names.add(str(court["task_name"]))
    return names


if __name__ == "__main__":
    raise SystemExit(main())
