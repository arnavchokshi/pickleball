from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.rkt_shot_replay_lane_audit import (
    build_rkt_shot_replay_lane_audit,
    build_rkt_shot_replay_lane_audit_markdown,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_cvat_clip(root: Path, clip: str, *, paddle_boxes: int, ball_boxes: int, player_boxes: int) -> None:
    clip_dir = root / clip
    _write_json(
        clip_dir / "racket_candidates_from_cvat_paddle_boxes.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_candidates",
            "fps": 30.0,
            "players": [
                {
                    "id": 1,
                    "paddle_dims_in": {"length": 16.0, "width": 8.0},
                    "frames": [
                        {
                            "t": float(index) / 30.0,
                            "corners_px": [[10.0, 10.0], [20.0, 10.0], [20.0, 30.0], [10.0, 30.0]],
                            "conf": 0.5,
                            "source": "label_bbox:cvat_video:paddle",
                        }
                        for index in range(paddle_boxes)
                    ],
                }
            ],
        },
    )
    _write_json(
        clip_dir / "racket_pose_readiness_from_cvat_paddle_boxes.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_pose_readiness",
            "clip": clip,
            "status": "blocked_preview_only",
            "source_counts": {"label_bbox:cvat_video:paddle": paddle_boxes},
            "source_evidence_counts": {
                "box_derived": paddle_boxes,
                "keypoint_or_mask": 0,
                "reference_gt": 0,
                "synthetic_or_cad": 0,
                "true_corners_or_pose": 0,
            },
            "candidate_frame_count": paddle_boxes,
            "box_derived_frame_count": paddle_boxes,
            "true_corner_frame_count": 0,
            "reference_gt_frame_count": 0,
            "preview_pose_frame_count": 0,
            "promoted_pose_frame_count": 0,
            "blockers": [
                "box_derived_candidate_corners",
                "missing_true_paddle_keypoints_or_cad_pose",
                "missing_promoted_racket_pose_json",
                "missing_reference_pose_gt",
                "missing_racket_pose_evaluation",
            ],
        },
    )
    _write_json(
        clip_dir / "racket_promotion_audit_from_cvat_paddle_boxes.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_promotion_audit",
            "clip": clip,
            "status": "safe_preview_only",
            "trusted_for_rkt_promotion": False,
            "canonical_racket_pose_present": False,
            "candidate_frame_count": paddle_boxes,
            "box_derived_candidate_frame_count": paddle_boxes,
            "true_corner_frame_count": 0,
            "reference_gt_frame_count": 0,
            "promoted_pose_frame_count": 0,
            "unsafe_promoted_frame_count": 0,
            "blockers": [
                "box_derived_candidate_corners",
                "missing_true_paddle_keypoints_or_cad_pose",
                "missing_promoted_racket_pose_json",
                "missing_reference_pose_gt",
                "missing_racket_pose_evaluation",
            ],
        },
    )
    _write_json(
        clip_dir / "reviewed_boxes.json",
        {
            "schema_version": 1,
            "status": "human_reviewed",
            "visible_box_count_by_label": {"ball": ball_boxes, "paddle": paddle_boxes, "player": player_boxes},
        },
    )


def _write_fixture_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    cvat_root = tmp_path / "runs" / "cvat_imports" / "2026_06_30"
    clips = [
        ("clip_a", 3, 4, 8),
        ("clip_b", 2, 5, 10),
    ]
    for clip, paddle_boxes, ball_boxes, player_boxes in clips:
        _write_cvat_clip(cvat_root, clip, paddle_boxes=paddle_boxes, ball_boxes=ball_boxes, player_boxes=player_boxes)
    _write_json(
        cvat_root / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_cvat_import_manifest",
            "status": "three_active_clips_indoor_pending",
            "clips": [
                {
                    "clip_id": clip,
                    "import_dir": str(cvat_root / clip),
                    "visible_box_count_by_label": {
                        "ball": ball_boxes,
                        "paddle": paddle_boxes,
                        "player": player_boxes,
                    },
                }
                for clip, paddle_boxes, ball_boxes, player_boxes in clips
            ],
            "pending_clips": ["clip_c"],
        },
    )
    replay_path = _write_json(
        tmp_path / "runs" / "eval0" / "prototype_gate_h100_v2" / "replay_readiness_report.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_replay_readiness_report",
            "status": "blocked",
            "summary": {
                "clip_count": 2,
                "review_visual_ready_clips": 2,
                "production_replay_ready_clips": 0,
                "metrics_gate_ready_clips": 0,
            },
            "clips": [
                {
                    "clip": "clip_a",
                    "review_visual_ready": True,
                    "production_replay_ready": False,
                    "metrics_gate_ready": False,
                    "counts": {"ambiguous_paddle_frames": 3, "approx_ball_frames": 4, "mesh_player_frames": 1},
                    "glb_report": {"artifact_class": "review_static_glb"},
                    "blockers": ["review_static_glb_export", "missing_skeletal_animation", "paddle_pose_preview_only"],
                },
                {
                    "clip": "clip_b",
                    "review_visual_ready": True,
                    "production_replay_ready": False,
                    "metrics_gate_ready": False,
                    "counts": {"ambiguous_paddle_frames": 2, "approx_ball_frames": 5, "mesh_player_frames": 0},
                    "glb_report": {"artifact_class": "review_static_glb"},
                    "blockers": ["review_static_glb_export", "missing_body_mesh", "paddle_pose_preview_only"],
                },
            ],
        },
    )
    shot_root = tmp_path / "runs" / "shot_classification_review"
    _write_json(
        shot_root / "clip_a" / "shot_classification.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_shot_classification",
            "clip_id": "clip_a",
            "classifier": {
                "family": "transfer_or_heuristic",
                "not_gate_verified": True,
                "trained_model": None,
            },
            "shots": [
                {"id": "shot_001", "type": "fh_shot", "type_conf": 0.44, "gated": False},
                {"id": "shot_002", "type": "unknown", "type_conf": 0.20, "gated": True},
            ],
        },
    )
    external_eval = _write_json(
        tmp_path / "runs" / "shot_external_eval" / "summary.json",
        {
            "dataset": "external tennis sample",
            "accuracy": 0.81,
            "family_accuracy": 0.517,
            "top2_family_accuracy": 0.667,
            "by_label": {
                "overhead": {"count": 5, "correct": 0, "accuracy": 0.0},
                "serve": {"count": 15, "correct": 0, "accuracy": 0.0},
            },
            "by_truth": {
                "serve": {"count": 19, "correct": 0, "accuracy": 0.0},
                "swing": {"count": 81, "correct": 81, "accuracy": 1.0},
            },
        },
    )
    return cvat_root, replay_path, shot_root, external_eval


def test_lane_audit_keeps_rkt_shot_replay_fail_closed(tmp_path: Path) -> None:
    cvat_root, replay_path, shot_root, external_eval = _write_fixture_tree(tmp_path)

    audit = build_rkt_shot_replay_lane_audit(
        cvat_import_root=cvat_root,
        replay_readiness_path=replay_path,
        shot_review_root=shot_root,
        shot_external_eval_paths=[external_eval],
    )

    assert audit["artifact_type"] == "racketsport_rkt_shot_replay_lane_audit"
    assert audit["status"] == "blocked_not_production_ready"
    assert audit["cvat"]["summary"] == {
        "active_clip_count": 2,
        "pending_clip_count": 1,
        "player_box_count": 18,
        "ball_box_count": 9,
        "paddle_rectangle_count": 5,
    }
    assert audit["rkt"]["status"] == "blocked_preview_only"
    assert audit["rkt"]["summary"]["candidate_frame_count"] == 5
    assert audit["rkt"]["summary"]["box_derived_frame_count"] == 5
    assert audit["rkt"]["summary"]["true_corner_frame_count"] == 0
    assert audit["rkt"]["summary"]["promoted_pose_frame_count"] == 0
    assert audit["rkt"]["summary"]["unsafe_promoted_frame_count"] == 0
    assert audit["rkt"]["may_promote_rkt"] is False
    assert audit["rkt"]["blocker_frequency"]["box_derived_candidate_corners"] == 2
    assert audit["replay"]["status"] == "blocked"
    assert audit["replay"]["summary"]["review_visual_ready_clips"] == 2
    assert audit["replay"]["summary"]["production_replay_ready_clips"] == 0
    assert audit["replay"]["summary"]["metrics_gate_ready_clips"] == 0
    assert audit["replay"]["artifact_classes"] == {"review_static_glb": 2}
    assert audit["replay"]["blocker_frequency"]["review_static_glb_export"] == 2
    assert audit["shot"]["status"] == "scaffold_transfer_only"
    assert audit["shot"]["summary"]["review_prediction_count"] == 2
    assert audit["shot"]["summary"]["reviewed_truth_label_count"] == 0
    assert audit["shot"]["summary"]["trained_model_count"] == 0
    assert audit["shot"]["may_train_poseconv3d_or_bst"] is False
    assert audit["shot"]["external_eval_summaries"][0]["accuracy"] == 0.81
    assert audit["shot"]["external_eval_summaries"][0]["family_accuracy"] == 0.517
    assert audit["shot"]["external_eval_summaries"][0]["top2_family_accuracy"] == 0.667
    assert audit["shot"]["external_eval_summaries"][0]["by_label"]["serve"]["accuracy"] == 0.0
    assert audit["shot"]["external_eval_summaries"][0]["not_pickleball_gate"] is True
    assert audit["production_blockers"][:3] == [
        "RKT blocked: 5/5 candidate frames are box-derived rectangles; 0 true-corner/reference frames.",
        "RPL blocked: 0/2 clips are production replay ready; static/review GLBs remain non-production.",
        "SHOT blocked: 0 reviewed pickleball shot truth labels; transfer/heuristic predictions cannot train or verify SHOT-1.",
    ]
    assert audit["next_best_action"] == "RKT: import true paddle-face corner/keypoint/CAD/reference labels, then rerun the fail-closed RKT promotion audit before any replay promotion."


def test_lane_audit_markdown_is_handoff_ready(tmp_path: Path) -> None:
    cvat_root, replay_path, shot_root, external_eval = _write_fixture_tree(tmp_path)
    audit = build_rkt_shot_replay_lane_audit(
        cvat_import_root=cvat_root,
        replay_readiness_path=replay_path,
        shot_review_root=shot_root,
        shot_external_eval_paths=[external_eval],
    )

    markdown = build_rkt_shot_replay_lane_audit_markdown(audit)

    assert "# RKT/SHOT/RPL Lane Audit" in markdown
    assert "Paddle rectangles: 5" in markdown
    assert "RKT candidate frames: 5 (box-derived: 5, true-corner/reference: 0)" in markdown
    assert "Production replay-ready clips: 0/2" in markdown
    assert "Reviewed pickleball shot truth labels: 0" in markdown
    assert "Do not claim paddle 6DoF from rectangle boxes." in markdown
    assert "Do not claim production replay from static review GLBs." in markdown


def test_lane_audit_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    cvat_root, replay_path, shot_root, external_eval = _write_fixture_tree(tmp_path)
    out = tmp_path / "lane_audit.json"
    md_out = tmp_path / "lane_audit.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_rkt_shot_replay_lane_audit.py",
            "--cvat-import-root",
            str(cvat_root),
            "--replay-readiness",
            str(replay_path),
            "--shot-review-root",
            str(shot_root),
            "--shot-external-eval",
            str(external_eval),
            "--out",
            str(out),
            "--md-out",
            str(md_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_not_production_ready"
    assert payload["rkt"]["summary"]["candidate_frame_count"] == 5
    assert "RKT/SHOT/RPL Lane Audit" in md_out.read_text(encoding="utf-8")
    assert json.loads(completed.stdout)["status"] == "blocked_not_production_ready"
