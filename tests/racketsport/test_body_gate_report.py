from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.eval.body_gate_report import build_body_gate_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _smpl_motion() -> dict:
    return {
        "schema_version": 1,
        "model": "smplx",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "players": [
            {
                "id": 1,
                "betas": [0.0] * 10,
                "skate_free": True,
                "physics": "none",
                "frames": [
                    {
                        "t": 0.0,
                        "global_orient": [0.0, 0.0, 0.0],
                        "body_pose": [0.0] * 63,
                        "left_hand_pose": [],
                        "right_hand_pose": [],
                        "transl_world": [0.0, 0.0, 0.0],
                        "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        "joint_conf": [0.9, 0.8],
                        "foot_contact": {"left": False, "right": False},
                        "grf": [],
                        "mesh_vertices_world": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ],
            }
        ],
    }


def _skeleton3d() -> dict:
    return {
        "schema_version": 1,
        "joint_names": ["pelvis", "neck"],
        "preview_only": True,
        "players": [{"id": 1, "frames": [{"t": 0.0, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "joint_conf": [0.9, 0.8]}]}],
    }


def _body_compute_execution(*, scheduled_frames: int = 1, scheduled_player_frames: int = 1) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "scheduled_frames": [],
        "skipped_frames": [],
        "summary": {
            "scheduled_frame_count": scheduled_frames,
            "scheduled_player_frame_count": scheduled_player_frames,
            "scheduled_by_target_representation": {"world_mesh": scheduled_frames} if scheduled_frames else {},
        },
    }


def _body_mesh_readiness(*, mesh_frames: int = 1) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh_readiness",
        "status": "mesh_available_needs_accuracy_gate" if mesh_frames else "missing_body_output",
        "trusted_for_body_promotion": False,
        "summary": {
            "player_count": 1 if mesh_frames else 0,
            "mesh_player_count": 1 if mesh_frames else 0,
            "mesh_frame_count": mesh_frames,
            "mesh_vertex_count_min": 3 if mesh_frames else 0,
            "mesh_vertex_count_max": 3 if mesh_frames else 0,
            "joints_player_count": 1 if mesh_frames else 0,
            "joints_frame_count": mesh_frames,
        },
        "representation_plan": {
            "scheduled_world_mesh_frame_count": mesh_frames,
            "scheduled_world_mesh_player_frame_count": mesh_frames,
            "available_mesh_frame_count": mesh_frames,
        },
        "blockers": ["missing_world_mpjpe_gate", "missing_full_clip_body_gate"] if mesh_frames else ["missing_smpl_motion_json"],
        "warnings": ["mesh_not_accuracy_verified"] if mesh_frames else ["missing_mesh_vertices"],
    }


def _write_body_run(root: Path, clip: str, *, mesh_frames: int = 1) -> Path:
    run_dir = root / clip
    _write_json(run_dir / "smpl_motion.json", _smpl_motion())
    _write_json(run_dir / "skeleton3d.json", _skeleton3d())
    _write_json(run_dir / "body_compute_execution.json", _body_compute_execution(scheduled_frames=mesh_frames))
    _write_json(run_dir / "body_mesh_readiness.json", _body_mesh_readiness(mesh_frames=mesh_frames))
    (run_dir / "virtual_world_paddle_preview.html").write_text("<html></html>", encoding="utf-8")
    return run_dir


def test_body_gate_report_keeps_mesh_smoke_unverified_without_labels_or_full_clip_gate(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001")

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert payload["status"] == "blocked"
    assert clip["status"] == "blocked"
    assert clip["mesh_smoke"]["status"] == "pass"
    assert clip["mesh_smoke"]["scheduled_frame_count"] == 1
    assert clip["mesh_smoke"]["mesh_player_frame_count"] == 1
    assert clip["world_mpjpe"]["status"] == "not_measured"
    assert clip["world_mpjpe"]["label_path"] == ""
    assert clip["full_clip_body_gate"]["status"] == "not_measured"
    assert set(clip["blockers"]) == {"missing_world_mpjpe_gate", "missing_full_clip_body_gate"}
    assert clip["inspectable_outputs"] == ["virtual_world_paddle_preview.html"]


def test_body_gate_report_computes_world_mpjpe_when_future_labels_exist(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "joints_world": [[0.0, 0.0, 0.0], [1.03, 0.0, 0.0]],
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root, world_mpjpe_threshold_m=0.05)

    mpjpe = payload["clips"][0]["world_mpjpe"]
    assert mpjpe["status"] == "pass"
    assert mpjpe["sample_count"] == 1
    assert mpjpe["joint_count"] == 2
    assert mpjpe["mean_error_m"] == 0.015
    assert mpjpe["threshold_m"] == 0.05
    assert payload["clips"][0]["blockers"] == ["missing_full_clip_body_gate"]


def test_body_gate_report_rejects_draft_world_joint_labels_and_lists_expected_gate_paths(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    label_path = labels_root / "clip_001" / "labels" / "body_world_joints.json"
    _write_json(
        label_path,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "status": "draft_prototype_unverified",
            "not_ground_truth": True,
            "samples": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "accepted": True,
                    "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                }
            ],
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    world = clip["world_mpjpe"]
    assert world["status"] == "not_measured"
    assert world["label_path"] == str(label_path)
    assert world["mean_error_m"] is None
    assert world["label_import"]["status"] == "rejected_not_ground_truth"
    assert world["label_import"]["path"] == str(label_path)
    assert world["label_import"]["payload_status"] == "draft_prototype_unverified"
    assert world["label_import"]["not_ground_truth"] is True
    assert world["label_import"]["accepted_sample_count"] == 1
    assert world["label_import"]["expected_paths"] == [
        str(labels_root / "clip_001" / "body_world_joints.json"),
        str(labels_root / "clip_001" / "body_world_mpjpe.json"),
        str(labels_root / "clip_001" / "labels" / "body_world_joints.json"),
        str(labels_root / "clip_001" / "labels" / "body_world_mpjpe.json"),
    ]
    assert world["blockers"] == ["missing_world_mpjpe_gate", "body_world_labels_not_ground_truth"]
    assert "body_world_labels_not_ground_truth" in clip["blockers"]
    assert clip["full_clip_body_gate"]["expected_paths"] == [
        str(run_dir / "body_full_clip_gate.json"),
        str(labels_root / "clip_001" / "body_full_clip_gate.json"),
        str(labels_root / "clip_001" / "labels" / "body_full_clip_gate.json"),
    ]


def test_body_gate_report_dedupes_full_clip_expected_paths_when_labels_default_to_runs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_dir = _write_body_run(root, "clip_001")

    payload = build_body_gate_report(root=root, clips=["clip_001"])

    assert payload["clips"][0]["full_clip_body_gate"]["expected_paths"] == [
        str(run_dir / "body_full_clip_gate.json"),
        str(run_dir / "labels" / "body_full_clip_gate.json"),
    ]


def test_body_gate_report_honors_full_clip_gate_when_present(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    run_dir = _write_body_run(root, "clip_001")
    _write_json(
        labels_root / "clip_001" / "body_world_joints.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_world_joints_labels",
            "samples": [
                {"frame_index": 0, "player_id": 1, "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
            ],
        },
    )
    _write_json(
        run_dir / "body_full_clip_gate.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_full_clip_gate",
            "passed": True,
            "coverage": 0.97,
            "evaluated_frame_count": 120,
        },
    )

    payload = build_body_gate_report(root=root, clips=["clip_001"], labels_root=labels_root)

    clip = payload["clips"][0]
    assert payload["status"] == "pass"
    assert clip["status"] == "pass"
    assert clip["full_clip_body_gate"]["status"] == "pass"
    assert clip["blockers"] == []


def test_body_gate_report_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    out_json = tmp_path / "body_gate_report.json"
    out_md = tmp_path / "body_gate_report.md"
    _write_body_run(root, "clip_001")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_gate_report.py",
            "--root",
            str(root),
            "--clip",
            "clip_001",
            "--out",
            str(out_json),
            "--markdown-out",
            str(out_md),
            "--write-clip-reports",
            "--allow-not-verified",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    markdown = out_md.read_text(encoding="utf-8")
    assert payload["status"] == "blocked"
    assert "| clip_001 | blocked | pass | not_measured | not_measured |" in markdown
    assert "virtual_world_paddle_preview.html" in markdown
    assert (root / "clip_001" / "body_gate_report.json").is_file()
    assert (root / "clip_001" / "body_gate_report.md").is_file()


def test_body_gate_report_default_discovery_skips_non_clip_run_dirs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_body_run(root, "clip_001")
    (root / "review_bundle").mkdir(parents=True)
    (root / "ball_tracker_benchmark").mkdir(parents=True)
    (root / "__pycache__").mkdir(parents=True)

    payload = build_body_gate_report(root=root)

    assert [clip["clip"] for clip in payload["clips"]] == ["clip_001"]
