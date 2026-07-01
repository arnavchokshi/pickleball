from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.racket_promotion_audit import build_racket_promotion_audit


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


def _racket_pose(source: str = "label_bbox:manual:pnp_ippe") -> dict:
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
                        "source": source,
                        "reprojection_error_px": 1.0,
                        "ambiguous": False,
                    }
                ],
                "contacts": [],
            }
        ],
    }


def test_racket_promotion_audit_marks_box_preview_without_root_pose_safe() -> None:
    payload = build_racket_promotion_audit(
        clip="clip_001",
        racket_candidates=_racket_candidates(),
        racket_pose_preview=_racket_pose(source="label_bbox:manual:pnp_ippe_preview"),
    )

    assert payload["artifact_type"] == "racketsport_racket_promotion_audit"
    assert payload["status"] == "safe_preview_only"
    assert payload["canonical_racket_pose_present"] is False
    assert payload["trusted_for_rkt_promotion"] is False
    assert payload["box_derived_candidate_frame_count"] == 1
    assert payload["true_corner_frame_count"] == 0
    assert payload["preview_pose_frame_count"] == 1
    assert payload["promoted_pose_frame_count"] == 0
    assert payload["unsafe_promoted_frame_count"] == 0
    assert payload["blockers"] == [
        "box_derived_candidate_corners",
        "missing_true_paddle_keypoints_or_cad_pose",
        "missing_promoted_racket_pose_json",
        "missing_reference_pose_gt",
        "missing_racket_pose_evaluation",
    ]
    assert payload["warnings"] == ["canonical_racket_pose_missing", "preview_only_not_gate_verified"]


def test_racket_promotion_audit_flags_box_derived_canonical_pose() -> None:
    payload = build_racket_promotion_audit(
        clip="clip_001",
        racket_candidates=_racket_candidates(),
        racket_pose=_racket_pose(source="label_bbox:manual:pnp_ippe"),
    )

    assert payload["status"] == "unsafe_box_derived_promoted"
    assert payload["canonical_racket_pose_present"] is True
    assert payload["promoted_pose_frame_count"] == 1
    assert payload["unsafe_promoted_frame_count"] == 1
    assert payload["blockers"][0] == "box_derived_racket_pose_promoted"
    assert "box_derived_racket_pose_promoted" in payload["warnings"]
    assert payload["unsafe_promoted_sources"] == {"label_bbox:manual:pnp_ippe": 1}


def test_racket_promotion_audit_requires_gt_eval_for_true_corner_pose() -> None:
    payload = build_racket_promotion_audit(
        clip="clip_001",
        racket_candidates=_racket_candidates(source="sam3_keypoints:mask_corners"),
        racket_pose=_racket_pose(source="sam3_keypoints:mask_corners:pnp_ippe"),
    )

    assert payload["status"] == "pose_present_needs_reference_and_eval"
    assert payload["box_derived_candidate_frame_count"] == 0
    assert payload["true_corner_frame_count"] == 1
    assert payload["promoted_pose_frame_count"] == 1
    assert payload["unsafe_promoted_frame_count"] == 0
    assert payload["blockers"] == ["missing_reference_pose_gt", "missing_racket_pose_evaluation"]
    assert payload["warnings"] == ["not_trusted_for_rkt_promotion"]


def test_racket_promotion_audit_classifies_cad_gt_as_cad_not_reference() -> None:
    payload = build_racket_promotion_audit(
        clip="clip_001",
        racket_candidates=_racket_candidates(source="cad_gt:measured_paddle_model"),
        racket_pose=_racket_pose(source="cad_gt:measured_paddle_model:pnp_ippe"),
    )

    assert payload["source_evidence_counts"]["synthetic_or_cad"] == 1
    assert payload["source_evidence_counts"]["reference_gt"] == 0
    assert payload["reference_gt_frame_count"] == 0
    assert payload["blockers"] == ["missing_reference_pose_gt", "missing_racket_pose_evaluation"]


def test_racket_promotion_audit_cli_writes_json(tmp_path: Path) -> None:
    candidates = tmp_path / "racket_candidates.json"
    preview = tmp_path / "racket_pose_preview.json"
    out = tmp_path / "racket_promotion_audit.json"
    candidates.write_text(json.dumps(_racket_candidates()), encoding="utf-8")
    preview.write_text(json.dumps(_racket_pose(source="label_bbox:manual:pnp_ippe_preview")), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_racket_promotion_audit.py",
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
    assert payload["status"] == "safe_preview_only"
    assert payload["canonical_racket_pose_present"] is False
    assert json.loads(completed.stdout)["status"] == "safe_preview_only"
