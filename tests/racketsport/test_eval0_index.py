from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_variant_selection(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_eval0_index_discovers_variant_selection_artifacts_and_preserves_pending_status(tmp_path: Path) -> None:
    root = tmp_path
    manifest = root / "models" / "MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"schema_version": 1, "models": []}\n', encoding="utf-8")

    _write_variant_selection(
        root / "runs" / "eval0" / "body_backbone" / "variant_selection.json",
        {
            "schema_version": 1,
            "approval_status": "pending",
            "candidate_count": 2,
            "candidates": [
                {
                    "variant_id": "fast_sam_3d_body_dinov3",
                    "stage": "3d_body_backbone",
                    "overlay_path": "runs/eval0/body_backbone/fast_overlay.mp4",
                    "metric_path": "runs/eval0/body_backbone/fast_metrics.json",
                },
                {
                    "variant_id": "sat_hmr",
                    "stage": "3d_body_backbone",
                    "overlay_path": "runs/eval0/body_backbone/sat_overlay.mp4",
                    "metric_paths": ["runs/eval0/body_backbone/sat_metrics.json"],
                },
            ],
        },
    )
    _write_variant_selection(
        root / "runs" / "eval0" / "racket_segmentation" / "comparison" / "variant_selection.json",
        {
            "schema_version": 1,
            "stage": "racket_segmentation",
            "approval_status": "approved",
            "selected_candidate": "sam2_hiera_base_plus",
            "candidates": [
                {
                    "variant_id": "sam2_hiera_base_plus",
                    "overlay_paths": [
                        "runs/eval0/racket_segmentation/sam2_overlay_1.png",
                        "runs/eval0/racket_segmentation/sam2_overlay_2.png",
                    ],
                    "metrics_path": "runs/eval0/racket_segmentation/sam2_metrics.json",
                }
            ],
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_eval0_index.py",
            "--root",
            str(root),
            "--markdown",
        ],
        check=True,
    )

    payload = json.loads((root / "runs" / "eval0" / "eval0_index.json").read_text(encoding="utf-8"))
    markdown = (root / "runs" / "eval0" / "eval0_index.md").read_text(encoding="utf-8")

    assert payload["schema_version"] == 1
    assert payload["artifact_type"] == "racketsport_eval0_index"
    assert payload["run_count"] == 2
    assert payload["approval_status_counts"] == {"approved": 1, "pending": 1}
    assert [entry["stage"] for entry in payload["runs"]] == ["3d_body_backbone", "racket_segmentation"]

    pending_entry = payload["runs"][0]
    assert pending_entry["approval_status"] == "pending"
    assert pending_entry["selected_candidate"] is None
    assert pending_entry["candidate_count"] == 2
    assert pending_entry["overlay_count"] == 2
    assert pending_entry["metric_paths"] == [
        "runs/eval0/body_backbone/fast_metrics.json",
        "runs/eval0/body_backbone/sat_metrics.json",
    ]

    approved_entry = payload["runs"][1]
    assert approved_entry["selected_candidate"] == "sam2_hiera_base_plus"
    assert approved_entry["candidate_count"] == 1
    assert approved_entry["overlay_count"] == 2
    assert approved_entry["metric_paths"] == ["runs/eval0/racket_segmentation/sam2_metrics.json"]

    assert "approval_status: pending" in markdown
    assert "models/MANIFEST.json is not read or modified" in markdown
    assert manifest.read_text(encoding="utf-8") == '{"schema_version": 1, "models": []}\n'


def test_eval0_index_rejects_malformed_variant_selection_without_writing_outputs(tmp_path: Path) -> None:
    _write_variant_selection(
        tmp_path / "runs" / "eval0" / "bad_stage" / "variant_selection.json",
        {
            "schema_version": 1,
            "approval_status": "pending",
            "candidates": "not a list",
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_eval0_index.py",
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "candidates must be a list" in completed.stderr
    assert not (tmp_path / "runs" / "eval0" / "eval0_index.json").exists()
