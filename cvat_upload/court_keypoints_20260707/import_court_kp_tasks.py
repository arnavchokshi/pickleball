#!/usr/bin/env python3
"""Create the local CVAT project/task for the 2026-07-07 metric-15 court keypoint taskset.

Run from the repo root:
  .venv/bin/python cvat_upload/court_keypoints_20260707/import_court_kp_tasks.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKSET_DIR = REPO_ROOT / "cvat_upload/court_keypoints_20260707"
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--creds", type=Path, default=CREDS_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((TASKSET_DIR / "taskset_manifest.json").read_text(encoding="utf-8"))
    label_spec = json.loads((TASKSET_DIR / "label_spec.json").read_text(encoding="utf-8"))
    package_zip = REPO_ROOT / manifest["package_zip_path"]
    labels = label_spec["cvat_labels"]
    project_name = "racketsport_metric15_court_keypoints_20260707"
    task_name = manifest["task_name"]

    if args.dry_run:
        print(json.dumps({"project_name": project_name, "task_name": task_name, "package_zip": str(package_zip), "labels": labels}, indent=2))
        return 0

    from cvat_sdk import make_client

    creds = load_creds(args.creds)
    with make_client(host=args.host, port=args.port, credentials=(creds["ADMIN_USERNAME"], creds["ADMIN_PASSWORD"])) as client:
        client.check_server_version = lambda *a, **k: None
        project = client.projects.create(spec={"name": project_name, "labels": labels})
        task = client.tasks.create_from_data(
            spec={"name": task_name, "project_id": project.id},
            resources=[str(package_zip)],
            data_params={"image_quality": 95, "use_cache": True},
        )
        report = {
            "schema_version": 1,
            "status": "imported",
            "project_id": project.id,
            "project_name": project_name,
            "task_id": task.id,
            "task_name": task_name,
            "package_zip": str(package_zip),
            "selected_frame_count": manifest["selected_frame_count"],
            "source_ids": manifest["non_heldout_source_ids"],
        }
        out_path = TASKSET_DIR / "import_report.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
