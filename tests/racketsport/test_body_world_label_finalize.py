from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.body_world_label_finalize import finalize_body_world_labels
from threed.racketsport.eval.body_gate_report import build_body_gate_report


def _template(
    *,
    reviewed: bool = True,
    accepted_count: int = 2,
    label_source: str | None = "manual_3d_annotation",
) -> dict:
    samples = [
        {
            "sample_id": f"frame_{index + 1:06d}_player_{7 + index}",
            "frame_index": index + 1,
            "t": float(index + 1) / 30.0,
            "player_id": 7 + index,
            "accepted": index < accepted_count,
            "review_status": "reviewed" if index < accepted_count else "needs_review",
            "joints_world": [[float(index), 0.0, 0.1], [float(index) + 0.2, 0.0, 1.4]]
            if index < accepted_count
            else [],
            "predicted_joints_world": [[float(index), 0.0, 0.1], [float(index) + 0.2, 0.0, 1.4]],
            "joint_conf": [0.9, 0.8],
            "notes": "",
            **({"label_source": label_source} if label_source is not None else {}),
        }
        for index in range(2)
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_joints_labels",
        "status": "human_reviewed" if reviewed else "draft_review_template",
        "not_ground_truth": not reviewed,
        "trusted_for_world_mpjpe": reviewed,
        "clip": "clip_001",
        "source_packet": "body_world_label_packet.json",
        "source_video": "source.mp4",
        "joint_names": ["pelvis", "neck"],
        "selected_sample_ids": ["frame_000001_player_7", "frame_000002_player_8"],
        "samples": samples,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tracks() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_tracks",
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "t": 1.0 / 30.0,
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "world_xy": [0.0, 0.0],
                    }
                ],
            }
        ],
        "rally_spans": [],
    }


def _write_body_run(root: Path, clip: str) -> Path:
    run = root / clip
    tracks_path = run / "tracks.json"
    _write_json(tracks_path, _tracks())
    frame = {
        "t": 1.0 / 30.0,
        "frame_idx": 1,
        "joints_world": [[0.0, 0.0, 0.1], [0.2, 0.0, 1.4]],
        "joint_conf": [0.9, 0.8],
        "mesh_vertices_world": [[0.0, 0.0, 0.0]],
        "transl_world": [0.0, 0.0, 0.0],
    }
    _write_json(
        run / "smpl_motion.json",
        {
            "schema_version": 1,
            "model": "smplx",
            "fps": 30.0,
            "world_frame": "court_Z0",
            "players": [{"id": 7, "frames": [frame]}],
        },
    )
    _write_json(run / "skeleton3d.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
    _write_json(
        run / "body_compute_execution.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_compute_execution",
            "summary": {"scheduled_frame_count": 1, "scheduled_player_frame_count": 1},
        },
    )
    _write_json(
        run / "body_mesh_readiness.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_mesh_readiness",
            "summary": {"mesh_frame_count": 1},
            "representation_plan": {"scheduled_world_mesh_frame_count": 1, "scheduled_world_mesh_player_frame_count": 1},
        },
    )
    _write_json(
        run / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 1.0,
            "evaluated_frame_count": 1,
        },
    )
    _write_json(
        run / "body_grounding_quality.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_grounding_quality",
            "status": "pass",
            "clip": clip,
            "foot_slide_gate": {
                "name": "foot_slide_max_m",
                "threshold_m": 0.03,
                "value_m": 0.0,
                "passed": True,
            },
            "blockers": [],
        },
    )
    _write_json(
        run / "person_track_gt_score.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_track_gt_score",
            "clip_id": clip,
            "candidate": "body_tracks",
            "tracks_path": str(tracks_path),
            "id_switches": 0,
            "identity_switch_event_count": 0,
            "identity_switch_events": [],
        },
    )
    return run


def test_finalize_body_world_labels_writes_gate_consumable_reviewed_labels(tmp_path: Path) -> None:
    template = tmp_path / "body_world_joints.template.json"
    out = tmp_path / "labels" / "clip_001" / "body_world_joints.json"
    _write_json(template, _template(reviewed=True, accepted_count=2))

    report = finalize_body_world_labels(template_path=template, out_path=out)

    assert report["status"] == "finalized"
    assert report["accepted_sample_count"] == 2
    final = json.loads(out.read_text(encoding="utf-8"))
    assert final["status"] == "human_reviewed"
    assert final["not_ground_truth"] is False
    assert final["trusted_for_world_mpjpe"] is True
    assert final["samples"][0]["accepted"] is True
    assert "predicted_joints_world" not in final["samples"][0]


def test_finalize_body_world_labels_preserves_reviewer_notes(tmp_path: Path) -> None:
    template = tmp_path / "body_world_joints.template.json"
    out = tmp_path / "labels" / "clip_001" / "body_world_joints.json"
    payload = _template(reviewed=True, accepted_count=1)
    payload["selected_sample_ids"] = ["frame_000001_player_7"]
    payload["samples"] = [payload["samples"][0]]
    payload["samples"][0]["notes"] = "systematic overlay offset noted during human review"
    _write_json(template, payload)

    report = finalize_body_world_labels(template_path=template, out_path=out)

    assert report["status"] == "finalized"
    final = json.loads(out.read_text(encoding="utf-8"))
    assert final["samples"][0]["notes"] == "systematic overlay offset noted during human review"


def test_finalize_body_world_labels_refuses_candidate_predictions_without_independent_source(
    tmp_path: Path,
) -> None:
    template = tmp_path / "body_world_joints.template.json"
    out = tmp_path / "labels" / "clip_001" / "body_world_joints.json"
    _write_json(template, _template(reviewed=True, accepted_count=2, label_source=None))

    report = finalize_body_world_labels(template_path=template, out_path=out)

    assert report["status"] == "blocked"
    assert "accepted_candidate_labels_not_independent_ground_truth" in report["blockers"]
    assert not out.exists()


def test_finalize_body_world_labels_refuses_draft_template(tmp_path: Path) -> None:
    template = tmp_path / "body_world_joints.template.json"
    out = tmp_path / "body_world_joints.json"
    _write_json(template, _template(reviewed=False, accepted_count=2))

    report = finalize_body_world_labels(template_path=template, out_path=out)

    assert report["status"] == "blocked"
    assert "template_not_reviewed" in report["blockers"]
    assert "template_marked_not_ground_truth" in report["blockers"]
    assert not out.exists()


def test_finalize_body_world_labels_refuses_incomplete_selected_samples(tmp_path: Path) -> None:
    template = tmp_path / "body_world_joints.template.json"
    out = tmp_path / "body_world_joints.json"
    payload = _template(reviewed=True, accepted_count=1)
    payload["selected_sample_ids"] = ["frame_000001_player_7", "missing_sample"]
    _write_json(template, payload)

    report = finalize_body_world_labels(template_path=template, out_path=out)

    assert report["status"] == "blocked"
    assert "selected_samples_not_all_accepted" in report["blockers"]
    assert "missing_selected_samples" in report["blockers"]
    assert not out.exists()


def test_finalize_body_world_labels_refuses_selected_samples_with_overlay_warnings(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "body_world_label_review_bundle"
    template = bundle_dir / "body_world_joints.template.json"
    out = tmp_path / "body_world_joints.json"
    _write_json(template, _template(reviewed=True, accepted_count=2))
    _write_json(
        bundle_dir / "overlays" / "body_world_label_review_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_review_overlay",
            "status": "ready_for_review_with_overlay_warnings",
            "overlays": [
                {
                    "sample_id": "frame_000001_player_7",
                    "warnings": ["body_joint_overlay_alignment_warning"],
                },
                {
                    "sample_id": "frame_000002_player_8",
                    "warnings": [],
                },
            ],
        },
    )

    report = finalize_body_world_labels(template_path=template, out_path=out)

    assert report["status"] == "blocked"
    assert "selected_samples_have_overlay_warnings" in report["blockers"]
    assert report["overlay_warning_selected_sample_ids"] == ["frame_000001_player_7"]
    assert not out.exists()


def test_finalize_body_world_labels_allows_human_reviewed_overlay_warnings(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "body_world_label_review_bundle"
    template = bundle_dir / "body_world_joints.template.json"
    out = tmp_path / "body_world_joints.json"
    _write_json(template, _template(reviewed=True, accepted_count=2))
    _write_json(
        bundle_dir / "overlays" / "body_world_label_review_overlay_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_label_review_overlay",
            "status": "ready_for_review_with_overlay_warnings",
            "overlays": [
                {
                    "sample_id": "frame_000001_player_7",
                    "warnings": ["body_joint_overlay_alignment_warning"],
                    "warning_review_status": "accepted",
                    "warning_review_note": "Reviewer confirmed joints align with the intended player.",
                },
                {
                    "sample_id": "frame_000002_player_8",
                    "warnings": [],
                },
            ],
        },
    )

    report = finalize_body_world_labels(template_path=template, out_path=out)

    assert report["status"] == "finalized"
    assert report["overlay_warning_selected_sample_ids"] == []
    assert out.exists()


def test_finalized_body_world_labels_can_unblock_mpjpe_gate(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001")
    template = tmp_path / "body_world_joints.template.json"
    _write_json(
        template,
        {
            **_template(reviewed=True, accepted_count=1),
            "selected_sample_ids": ["frame_000001_player_7"],
            "samples": [_template(reviewed=True, accepted_count=1)["samples"][0]],
        },
    )

    finalize_report = finalize_body_world_labels(
        template_path=template,
        out_path=labels_root / "clip_001" / "body_world_joints.json",
    )
    gate = build_body_gate_report(
        root=root,
        clips=["clip_001"],
        labels_root=labels_root,
        world_mpjpe_threshold_m=0.01,
        world_mpjpe_min_label_samples=1,
    )

    assert finalize_report["status"] == "finalized"
    assert gate["status"] == "pass"
    assert gate["clips"][0]["world_mpjpe"]["status"] == "pass"


def test_finalize_body_world_labels_cli_blocks_drafts(tmp_path: Path) -> None:
    template = tmp_path / "body_world_joints.template.json"
    out = tmp_path / "body_world_joints.json"
    report = tmp_path / "finalize_report.json"
    _write_json(template, _template(reviewed=False, accepted_count=2))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/finalize_body_world_labels.py",
            "--template",
            str(template),
            "--out",
            str(out),
            "--report-out",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "template_not_reviewed" in completed.stdout
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "blocked"
    assert not out.exists()
