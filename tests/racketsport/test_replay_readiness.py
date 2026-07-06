from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.replay_export import build_replay_review_export_from_virtual_world, write_replay_scene
from threed.racketsport.replay_readiness import build_replay_readiness_report


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _world_payload(*, mesh_frames: int = 2, ambiguous_paddle_frames: int = 1, approx_ball_frames: int = 3) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court": {
            "sport": "pickleball",
            "coordinate_frame": "court_Z0",
            "length_m": 13.41,
            "width_m": 6.1,
            "line_segments": {"baseline": [[-3.05, 0.0, 0.0], [3.05, 0.0, 0.0]]},
            "net": {
                "endpoints": [[-3.05, 6.705, 0.91], [3.05, 6.705, 0.91]],
                "center_height_m": 0.86,
                "post_height_m": 0.91,
            },
        },
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "representation": "mesh" if mesh_frames else "track_only",
                "frames": [
                    {
                        "t": 0.0,
                        "track_world_xy": [0.0, 1.0],
                        "track_conf": 0.91,
                        "bbox": [100.0, 100.0, 150.0, 260.0],
                        "transl_world": [0.0, 1.0, 0.0] if mesh_frames else None,
                        "mesh_vertices_world": [[0.0, 1.0, 0.0], [0.1, 1.0, 0.0]] if mesh_frames else [],
                        "joints_world": [[0.0, 1.0, 1.2]] if mesh_frames else [],
                        "joint_conf": [0.9] if mesh_frames else [],
                        "joint_count": 1 if mesh_frames else 0,
                        "mesh_vertex_count": mesh_frames,
                        "floor_world_xyz": [0.0, 1.0, 0.0],
                        "floor_source": "track_footpoint+smpl_world" if mesh_frames else "track_footpoint",
                        "floor_offset_m": 0.0,
                        "min_mesh_z_m": 0.0 if mesh_frames else None,
                        "floor_penetration_m": 0.0,
                        "foot_contact": {"left": True, "right": False} if mesh_frames else None,
                        "contact_locked": bool(mesh_frames),
                        "physics": "worldhmr_grounded_not_footlocked" if mesh_frames else None,
                        "grf": None,
                    }
                ],
            }
        ],
        "ball": {
            "source": "tracknet",
            "frames": [
                {
                    "t": 0.0,
                    "xy": [320.0, 240.0],
                    "conf": 0.91,
                    "world_xyz": [0.0, 6.0, 0.3],
                    "visible": True,
                    "approx": approx_ball_frames > 0,
                }
            ],
        },
        "paddles": [
            {
                "player_id": 1,
                "paddle_dims_in": {"length": 15.5, "width": 7.5},
                "frames": [
                    {
                        "t": 0.0,
                        "pose_se3": {
                            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "t": [0.0, 0.0, 1.0],
                        },
                        "mesh_vertices_world": [[0, 0, 0], [0.1, 0, 0], [0.1, 0.2, 0], [0, 0.2, 0]],
                        "mesh_faces": [[0, 1, 2], [0, 2, 3]],
                        "conf": 0.4,
                        "world_frame": "court_Z0",
                        "translation_unit": "m",
                        "source": "draft_box:pnp_ippe_preview:court_Z0",
                        "reprojection_error_px": None,
                        "ambiguous": ambiguous_paddle_frames > 0,
                    }
                ],
            }
        ],
        "summary": {
            "player_count": 1,
            "mesh_player_count": 1 if mesh_frames else 0,
            "mesh_player_frame_count": mesh_frames,
            "joint_player_frame_count": mesh_frames,
            "track_only_player_frame_count": 0 if mesh_frames else 1,
            "ball_frame_count": 12,
            "approx_ball_frame_count": approx_ball_frames,
            "paddle_player_count": 1,
            "paddle_frame_count": 1,
            "ambiguous_paddle_frame_count": ambiguous_paddle_frames,
            "warnings": ["ambiguous_paddle_pose"] if ambiguous_paddle_frames else [],
        },
    }


def _write_clip(run_root: Path, clip: str, *, body_status: str, mesh_frames: int = 2) -> None:
    run_dir = run_root / clip
    world = _world_payload(mesh_frames=mesh_frames)
    _write_json(run_dir / "virtual_world_paddle_preview.json", world)
    (run_dir / "virtual_world_paddle_preview.html").write_text("<html>review</html>", encoding="utf-8")
    _write_json(
        run_dir / "body_mesh_readiness.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_mesh_readiness",
            "clip": clip,
            "status": body_status,
            "world_mesh_available": body_status != "missing_body_output",
            "representation_decision": "world_mesh_required_available_unverified"
            if body_status != "missing_body_output"
            else "world_mesh_required_missing_output",
            "trusted_for_body_promotion": body_status == "gate_verified",
            "summary": {},
            "blockers": [] if body_status == "gate_verified" else ["missing_body_accuracy_gate"],
            "warnings": [],
        },
    )
    _write_json(
        run_dir / "racket_pose_readiness.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_pose_readiness",
            "clip": clip,
            "status": "blocked_preview_only",
            "blockers": ["box_derived_preview_only"],
        },
    )
    _write_json(run_dir / "contact_window_review.json", {"schema_version": 1, "status": "cleared"})
    scene = build_replay_review_export_from_virtual_world(world, export_root=run_dir / "replay_review", scene_root=run_dir)
    write_replay_scene(run_dir / "replay_scene.json", scene)


def test_build_replay_readiness_report_separates_visual_review_from_production_gate(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_clip(run_root, "clip_with_body", body_status="mesh_available_needs_accuracy_gate")
    _write_clip(run_root, "clip_without_body", body_status="missing_body_output", mesh_frames=0)

    report = build_replay_readiness_report(run_root=run_root, clips=["clip_with_body", "clip_without_body"], labels_root=tmp_path / "labels_absent")
    by_clip = {clip["clip"]: clip for clip in report["clips"]}

    assert report["status"] == "blocked"
    assert report["summary"]["review_visual_ready_clips"] == 2
    assert report["summary"]["production_replay_ready_clips"] == 0
    assert by_clip["clip_with_body"]["review_visual_ready"] is True
    assert by_clip["clip_with_body"]["production_replay_ready"] is False
    assert by_clip["clip_with_body"]["metrics_gate_ready"] is False
    assert "body_mesh_needs_accuracy_gate" in by_clip["clip_with_body"]["blockers"]
    assert "paddle_pose_preview_only" in by_clip["clip_with_body"]["blockers"]
    assert "ambiguous_paddle_pose" in by_clip["clip_with_body"]["blockers"]
    assert "approximate_ball_projection" in by_clip["clip_with_body"]["blockers"]
    assert "missing_labels_root" in by_clip["clip_with_body"]["blockers"]
    assert "review_static_glb_export" in by_clip["clip_with_body"]["blockers"]
    assert by_clip["clip_without_body"]["review_visual_ready"] is True
    assert "missing_body_mesh" in by_clip["clip_without_body"]["blockers"]
    assert by_clip["clip_with_body"]["visual_outputs"]["virtual_world_html"].endswith("virtual_world_paddle_preview.html")
    assert by_clip["clip_with_body"]["glb_report"]["valid_glb_count"] == 2
    assert by_clip["clip_with_body"]["glb_report"]["artifact_class"] == "review_static_glb"
    assert by_clip["clip_with_body"]["glb_report"]["production_replay_ready"] is False


def test_replay_readiness_fails_closed_for_unsafe_replay_scene_glb_refs(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_clip(run_root, "clip_with_body", body_status="mesh_available_needs_accuracy_gate")
    run_dir = run_root / "clip_with_body"
    scene_path = run_dir / "replay_scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    original_point_glb = run_dir / scene["points"][0]["glb_url"]
    outside_glb = run_root / "outside.glb"
    outside_glb.write_bytes(original_point_glb.read_bytes())
    scene["points"][0]["glb_url"] = "../outside.glb"
    scene_path.write_text(json.dumps(scene, indent=2, sort_keys=True), encoding="utf-8")

    report = build_replay_readiness_report(run_root=run_root, clips=["clip_with_body"])
    clip = report["clips"][0]

    assert report["status"] == "blocked"
    assert clip["review_visual_ready"] is False
    assert "invalid_replay_scene_glb_ref" in clip["blockers"]
    assert "../outside.glb" in clip["glb_report"]["invalid_glbs"]


def test_replay_readiness_rejects_status_only_readiness_sidecars(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_clip(run_root, "clip_with_body", body_status="gate_verified")
    run_dir = run_root / "clip_with_body"
    _write_json(run_dir / "body_mesh_readiness.json", {"schema_version": 1, "status": "gate_verified"})
    _write_json(run_dir / "racket_pose_readiness.json", {"schema_version": 1, "status": "gate_verified"})

    report = build_replay_readiness_report(run_root=run_root, clips=["clip_with_body"])
    clip = report["clips"][0]

    assert report["status"] == "blocked"
    assert "invalid_body_mesh_readiness" in clip["blockers"]
    assert "invalid_racket_pose_readiness" in clip["blockers"]
    assert clip["truth_status"]["body"].startswith("invalid:")
    assert clip["truth_status"]["paddle_pose"].startswith("invalid:")


def test_replay_readiness_cli_writes_json_and_html_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_clip(run_root, "clip_with_body", body_status="mesh_available_needs_accuracy_gate")
    out = tmp_path / "readiness.json"
    html = tmp_path / "readiness.html"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_replay_readiness_report.py",
            "--run-root",
            str(run_root),
            "--clips",
            "clip_with_body",
            "--out",
            str(out),
            "--html-out",
            str(html),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert out.is_file()
    assert html.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["clips"][0]["review_visual_ready"] is True
    html_text = html.read_text(encoding="utf-8")
    assert "Production replay ready" in html_text
    assert "GLB class" in html_text
    assert "review_static_glb" in html_text
    assert "Production GLB/USDZ requirements" in html_text
    assert "true skeletal animation channels" in html_text
    assert json.loads(completed.stdout)["status"] == "blocked"
