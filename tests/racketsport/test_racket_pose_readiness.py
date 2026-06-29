from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.racket_pose_readiness import build_racket_pose_readiness


def _racket_candidates(source: str = "label_bbox:manual") -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_candidates",
        "fps": 60.0,
        "players": [
            {
                "id": 7,
                "paddle_dims_in": {"length": 16.0, "width": 8.0},
                "frames": [
                    {
                        "t": 1.0,
                        "corners_px": [[10.0, 10.0], [20.0, 10.0], [20.0, 30.0], [10.0, 30.0]],
                        "conf": 0.4,
                        "source": source,
                    }
                ],
            }
        ],
    }


def _racket_pose() -> dict:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "world_frame": "camera",
        "translation_unit": "cm",
        "players": [
            {
                "id": 7,
                "paddle_dims_in": {"length": 16.0, "width": 8.0},
                "frames": [
                    {
                        "t": 1.0,
                        "pose_se3": {
                            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "t": [0.0, 0.0, 100.0],
                        },
                        "conf": 0.8,
                        "world_frame": "camera",
                        "translation_unit": "cm",
                        "source": "sam3_keypoints:pnp_ippe",
                        "reprojection_error_px": 1.0,
                        "ambiguous": False,
                    }
                ],
                "contacts": [],
            }
        ],
    }


def test_racket_pose_readiness_flags_box_derived_preview_as_blocked() -> None:
    payload = build_racket_pose_readiness(
        clip="clip_001",
        racket_candidates=_racket_candidates(),
        racket_pose_preview=_racket_pose(),
    )

    assert payload["artifact_type"] == "racketsport_racket_pose_readiness"
    assert payload["status"] == "blocked_preview_only"
    assert payload["candidate_frame_count"] == 1
    assert payload["box_derived_frame_count"] == 1
    assert payload["true_corner_frame_count"] == 0
    assert payload["reference_gt_frame_count"] == 0
    assert payload["preview_pose_frame_count"] == 1
    assert payload["promoted_pose_frame_count"] == 0
    assert payload["summary"] == {
        "candidate_player_count": 1,
        "candidate_frame_count": 1,
        "box_derived_frame_count": 1,
        "true_corner_frame_count": 0,
        "reference_gt_frame_count": 0,
        "preview_pose_frame_count": 1,
        "promoted_pose_frame_count": 0,
    }
    assert payload["source_evidence_counts"] == {
        "box_derived": 1,
        "keypoint_or_mask": 0,
        "reference_gt": 0,
        "synthetic_or_cad": 0,
        "true_corners_or_pose": 0,
    }
    assert payload["source_counts"] == {"label_bbox:manual": 1}
    assert payload["blockers"] == [
        "box_derived_candidate_corners",
        "missing_true_paddle_keypoints_or_cad_pose",
        "missing_promoted_racket_pose_json",
        "missing_reference_pose_gt",
        "missing_racket_pose_evaluation",
    ]


def test_racket_pose_readiness_requires_reference_gt_for_non_box_promoted_pose() -> None:
    payload = build_racket_pose_readiness(
        clip="clip_001",
        racket_candidates=_racket_candidates(source="racketvision_keypoints:sam2_mask"),
        racket_pose=_racket_pose(),
    )

    assert payload["status"] == "pose_present_needs_reference_and_eval"
    assert payload["box_derived_frame_count"] == 0
    assert payload["true_corner_frame_count"] == 1
    assert payload["reference_gt_frame_count"] == 0
    assert payload["promoted_pose_frame_count"] == 1
    assert payload["summary"]["box_derived_frame_count"] == 0
    assert payload["summary"]["true_corner_frame_count"] == 1
    assert payload["summary"]["reference_gt_frame_count"] == 0
    assert payload["summary"]["promoted_pose_frame_count"] == 1
    assert payload["source_evidence_counts"] == {
        "box_derived": 0,
        "keypoint_or_mask": 1,
        "reference_gt": 0,
        "synthetic_or_cad": 0,
        "true_corners_or_pose": 1,
    }
    assert payload["blockers"] == ["missing_reference_pose_gt", "missing_racket_pose_evaluation"]


def test_racket_pose_readiness_reports_pose_present_needs_eval_with_reference_gt() -> None:
    payload = build_racket_pose_readiness(
        clip="clip_001",
        racket_candidates=_racket_candidates(source="aruco_gt:april_tag_reference"),
        racket_pose=_racket_pose(),
    )

    assert payload["status"] == "pose_present_needs_eval"
    assert payload["true_corner_frame_count"] == 1
    assert payload["reference_gt_frame_count"] == 1
    assert payload["summary"]["true_corner_frame_count"] == 1
    assert payload["summary"]["reference_gt_frame_count"] == 1
    assert payload["source_evidence_counts"]["reference_gt"] == 1
    assert payload["blockers"] == ["missing_racket_pose_evaluation"]


def test_racket_pose_readiness_classifies_cad_gt_as_cad_not_reference() -> None:
    payload = build_racket_pose_readiness(
        clip="clip_001",
        racket_candidates=_racket_candidates(source="cad_gt:measured_paddle_model"),
        racket_pose=_racket_pose(),
    )

    assert payload["source_evidence_counts"]["synthetic_or_cad"] == 1
    assert payload["source_evidence_counts"]["reference_gt"] == 0
    assert payload["reference_gt_frame_count"] == 0
    assert payload["blockers"] == ["missing_reference_pose_gt", "missing_racket_pose_evaluation"]


def test_racket_pose_readiness_cli_writes_json(tmp_path: Path) -> None:
    candidates = tmp_path / "racket_candidates.json"
    preview = tmp_path / "racket_pose_preview.json"
    out = tmp_path / "racket_pose_readiness.json"
    candidates.write_text(json.dumps(_racket_candidates()), encoding="utf-8")
    preview.write_text(json.dumps(_racket_pose()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_racket_pose_readiness.py",
            "--clip",
            "clip_001",
            "--racket-candidates",
            str(candidates),
            "--racket-pose-preview",
            str(preview),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_preview_only"
    assert payload["true_corner_frame_count"] == 0
    assert payload["reference_gt_frame_count"] == 0
    assert json.loads(completed.stdout)["status"] == "blocked_preview_only"
