from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.orchestrator import BodyStageRunner
from threed.racketsport.body_video_smoke import run_body_video_smoke


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tracks_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 1.0 / 30.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [-1.0, -3.0], "conf": 0.92}
                ],
            }
        ],
        "rally_spans": [],
    }


def _two_frame_tracks_payload() -> dict:
    payload = _tracks_payload()
    payload["players"][0]["frames"] = [
        {"t": 0.0, "bbox": [90.0, 100.0, 190.0, 300.0], "world_xy": [-1.1, -3.0], "conf": 0.91},
        {"t": 1.0 / 30.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [-1.0, -3.0], "conf": 0.92},
    ]
    return payload


def _frame_plan_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 30.0,
        "expected_players": 1,
        "frame_count": 2,
        "frames": [
            {
                "frame_idx": 1,
                "t": 1.0 / 30.0,
                "score": 0.75,
                "recommended_tier": "deep_mesh",
                "target_representation": "world_mesh",
                "reasons": ["contact_window"],
                "active_players": 1,
                "active_player_ids": [7],
                "missing_players": 0,
                "min_track_conf": 0.92,
                "ball_conf": 0.6,
            }
        ],
        "deep_mesh_windows": [
            {
                "frame_start": 1,
                "frame_end": 1,
                "t0": 1.0 / 30.0,
                "t1": 2.0 / 30.0,
                "frame_count": 1,
                "target_representation": "world_mesh",
                "fallback_representation": "skeleton_preview",
                "target_player_ids": [7],
                "reason_counts": {"contact_window": 1},
                "max_score": 0.75,
            }
        ],
        "summary": {
            "by_tier": {"deep_mesh": 1},
            "by_reason": {"contact_window": 1},
            "by_player_target_representation": {"world_mesh": 1},
            "max_score": 0.75,
            "deep_mesh_window_count": 1,
            "deep_mesh_frame_count": 1,
            "human_review_frame_count": 0,
        },
    }


def _calibration_payload() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "image_size": [1920, 1080],
        "homography": [[20.0, 0.0, 960.0], [0.0, -20.0, 540.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "synthetic"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 10.0,
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": minimal_calibration_image_pts(),
        "world_pts": minimal_calibration_world_pts(),
    }


def _lane_a_skeleton_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": ["pelvis", "left_wrist", "right_wrist"],
        "preview_only": False,
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 1,
                        "t": 1.0 / 30.0,
                        "joints_world": [[0.0, 0.0, 1.0], [0.5, 0.0, 1.2], [0.0, 0.0, 1.2]],
                        "joint_conf": [0.9, 0.9, 0.9],
                    }
                ],
            }
        ],
        "provenance": {"lane": "A"},
    }


def _preview_skeleton_payload() -> dict:
    payload = _lane_a_skeleton_payload()
    payload["preview_only"] = True
    payload["source_model"] = "sam3dbody_world_joints"
    payload["provenance"] = {"lane": "B_preview"}
    return payload


def test_body_video_smoke_fails_closed_when_body_runtime_is_missing(tmp_path: Path, monkeypatch) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    _write_json(inputs / "frame_compute_plan.json", _frame_plan_payload())

    pipeline_stages: list[str] = []

    def fake_run_pipeline(**kwargs):
        stage = kwargs["stage"]
        pipeline_stages.append(stage)
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}
        _write_json(
            out / "body_compute_execution.json",
            {
                "schema_version": 1,
                "artifact_type": "racketsport_body_compute_execution",
                "scheduled_frames": [],
                "summary": {"scheduled_frame_count": 0, "scheduled_player_frame_count": 0},
            },
        )
        return {
            "status": "fail",
            "stages": [
                {"stage": "calibration", "status": "ran"},
                {"stage": "tracking", "status": "ran"},
                {"stage": "body", "status": "fail", "notes": ["body failed: missing FastSAM-3D-Body repo"]},
            ],
        }

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        return {
            "artifact_type": "racketsport_body_frame_manifest",
            "source_video": str(video_path),
            "body_compute_execution": str(execution_path),
            "out_dir": str(out),
            "frame_indexes": [1],
            "extracted_frame_count": 1,
            "frames": ["frame_000001.jpg"],
        }

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
    )

    assert pipeline_stages == ["tracking", "body"]
    assert report["status"] == "runtime_blocked"
    assert report["body_runtime_ran"] is False
    assert report["summary"]["scheduled_player_frame_count"] == 1
    assert report["summary"]["extracted_frame_count"] == 1
    assert report["summary"]["joint_frame_count"] == 0
    assert "missing_smpl_motion_json" in report["quality"]["quality_blockers"]
    assert "scheduled_body_output_incomplete" in report["quality"]["quality_blockers"]
    assert "missing FastSAM-3D-Body repo" in report["body_failure_note"]
    restored_execution = json.loads((run_dir / "body_compute_execution.json").read_text(encoding="utf-8"))
    assert restored_execution["summary"]["scheduled_player_frame_count"] == 1
    assert (run_dir / "body_video_smoke.json").is_file()
    assert (run_dir / "body_joint_quality.json").is_file()
    assert (run_dir / "body_full_clip_gate.json").is_file()
    assert (run_dir / "body_world_label_packet.json").is_file()
    assert (run_dir / "body_world_label_review_bundle" / "body_world_label_review_bundle.json").is_file()
    full_gate = json.loads((run_dir / "body_full_clip_gate.json").read_text(encoding="utf-8"))
    assert full_gate["passed"] is False
    assert "body_joint_quality_blocked" in full_gate["blockers"]
    assert "full_clip_body_coverage_below_threshold" in full_gate["blockers"]
    label_packet = json.loads((run_dir / "body_world_label_packet.json").read_text(encoding="utf-8"))
    assert label_packet["status"] == "blocked"
    assert label_packet["trusted_for_world_mpjpe"] is False
    assert "missing_body_predictions_for_label_packet" in label_packet["blockers"]
    assert report["paths"]["body_full_clip_gate"] == str(run_dir / "body_full_clip_gate.json")
    assert report["paths"]["body_world_label_packet"] == str(run_dir / "body_world_label_packet.json")
    assert report["paths"]["body_world_label_review_bundle"] == str(run_dir / "body_world_label_review_bundle")
    assert report["paths"]["body_world_label_review_overlay"] == str(
        run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json"
    )
    assert report["full_clip_gate"]["passed"] is False
    assert report["label_packet"]["status"] == "blocked"
    assert report["label_review_bundle"]["status"] == "no_selected_samples"
    assert report["label_review_overlay"]["status"] == "blocked_missing_overlay_inputs"
    assert report["label_review_overlay"]["floor_anchor_projection_failed_count"] == 0
    assert report["label_review_overlay"]["floor_anchor_projection_warning_count"] == 0
    assert report["label_review_overlay"]["alignment_failed_count"] == 0
    assert report["label_review_overlay"]["alignment_warning_count"] == 0
    assert "missing_overlay_inputs" in report["label_review_overlay"]["blockers"]
    assert (run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json").is_file()


def test_body_video_smoke_reuses_input_lane_a_skeleton_for_body_stage(tmp_path: Path, monkeypatch) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    _write_json(inputs / "frame_compute_plan.json", _frame_plan_payload())
    _write_json(inputs / "skeleton3d.json", _lane_a_skeleton_payload())
    body_runners: list[dict] = []

    def fake_run_pipeline(**kwargs):
        stage = kwargs["stage"]
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            _write_json(out / "court_calibration.json", _calibration_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}

        runners = kwargs.get("runners") or {}
        body_runners.append(runners)
        pose_runner = runners["pose"]
        pose_result = pose_runner.run(SimpleNamespace(inputs_dir=inputs, run_dir=out))
        assert pose_result.status == "ran"
        assert pose_result.produced_artifacts == ("skeleton3d.json",)
        skeleton = json.loads((out / "skeleton3d.json").read_text(encoding="utf-8"))
        assert skeleton["source_model"] == "sam3d_body_joints"
        assert skeleton["provenance"]["lane"] == "A"

        frame = {
            "t": 1.0 / 30.0,
            "joints_world": [[0.01 * idx, 0.0, 0.2 + 0.03 * idx] for idx in range(17)],
            "joint_conf": [0.91] * 17,
            "mesh_vertices_world": [[0.0, 0.0, 0.1], [0.1, 0.0, 1.7]],
            "transl_world": [0.0, 0.0, 0.0],
        }
        _write_json(out / "smpl_motion.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(out / "contact_splice.json", {"schema_version": 1, "artifact_type": "racketsport_contact_splice", "summary": {}})
        return {"status": "blocked", "stages": [{"stage": "body", "status": "ran"}]}

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        return {"extracted_frame_count": 1, "frame_indexes": [1], "frames": ["frame_000001.jpg"]}

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
    )

    assert body_runners
    assert "pose" in body_runners[0]
    assert report["body_runtime_ran"] is True
    assert (run_dir / "skeleton3d.json").is_file()


def test_body_video_smoke_reports_dependency_stage_failure_note(tmp_path: Path, monkeypatch) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    _write_json(inputs / "frame_compute_plan.json", _frame_plan_payload())

    def fake_run_pipeline(**kwargs):
        stage = kwargs["stage"]
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}
        return {
            "status": "fail",
            "stages": [
                {"stage": "calibration", "status": "ran"},
                {"stage": "tracking", "status": "ran"},
                {"stage": "pose", "status": "fail", "notes": ["pose failed: precomputed skeleton3d.json is preview_only"]},
            ],
        }

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        return {"extracted_frame_count": 1, "frame_indexes": [1], "frames": ["frame_000001.jpg"]}

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
    )

    assert report["status"] == "runtime_blocked"
    assert report["body_runtime_ran"] is False
    assert report["body_failure_note"] == "pose failed: precomputed skeleton3d.json is preview_only"


def test_body_video_smoke_does_not_treat_ran_body_stage_notes_as_failure_note(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    _write_json(inputs / "frame_compute_plan.json", _frame_plan_payload())

    def fake_run_pipeline(**kwargs):
        stage = kwargs["stage"]
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            _write_json(out / "court_calibration.json", _calibration_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}
        _write_json(out / "smpl_motion.json", {"schema_version": 1, "players": []})
        _write_json(out / "skeleton3d.json", _lane_a_skeleton_payload())
        _write_json(
            out / "contact_splice.json",
            {
                "schema_version": 1,
                "artifact_type": "racketsport_contact_splice",
                "summary": {
                    "scheduled_contact_count": 1,
                    "spliced_contact_count": 0,
                    "mesh_unavailable_count": 1,
                    "fallback_spliced_count": 1,
                    "overridden_joint_count": 2,
                },
            },
        )
        return {
            "status": "blocked",
            "stages": [
                {
                    "stage": "body",
                    "status": "ran",
                    "notes": ["Fast SAM-3D-Body runtime unavailable; no legacy contact fallback"],
                }
            ],
        }

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        return {"extracted_frame_count": 1, "frame_indexes": [1], "frames": ["frame_000001.jpg"]}

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
    )

    assert report["status"] == "quality_blocked"
    assert report["body_runtime_ran"] is False
    assert report["body_failure_note"] == ""
    assert "no_world_joint_frames" in report["quality"]["quality_blockers"]


def test_body_video_smoke_materializes_all_tracked_frames_for_lane_a_pose(tmp_path: Path, monkeypatch) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    _write_json(inputs / "frame_compute_plan.json", _frame_plan_payload())
    _write_json(inputs / "skeleton3d.json", _preview_skeleton_payload())
    materialize_calls: list[dict] = []
    body_runners: list[dict] = []

    def fake_run_pipeline(**kwargs):
        stage = kwargs["stage"]
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _two_frame_tracks_payload())
            _write_json(out / "court_calibration.json", _calibration_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}

        body_runners.append(kwargs.get("runners") or {})
        if not (out / "body_frames" / "frame_000000.jpg").is_file():
            return {
                "status": "fail",
                "stages": [{"stage": "body", "status": "fail", "notes": ["body failed: missing Lane A pose frame 0"]}],
            }
        frame = {
            "t": 1.0 / 30.0,
            "joints_world": [[0.01 * idx, 0.0, 0.2 + 0.03 * idx] for idx in range(17)],
            "joint_conf": [0.91] * 17,
            "mesh_vertices_world": [[0.0, 0.0, 0.1], [0.1, 0.0, 1.7]],
            "transl_world": [0.0, 0.0, 0.0],
        }
        _write_json(out / "smpl_motion.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(out / "skeleton3d.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(
            out / "contact_splice.json",
            {
                "schema_version": 1,
                "artifact_type": "racketsport_contact_splice",
                "summary": {
                    "scheduled_contact_count": 1,
                    "spliced_contact_count": 1,
                    "mesh_unavailable_count": 0,
                    "fallback_spliced_count": 0,
                    "overridden_joint_count": 2,
                },
            },
        )
        return {"status": "blocked", "stages": [{"stage": "body", "status": "ran"}]}

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        execution = json.loads(Path(execution_path).read_text(encoding="utf-8"))
        frame_indexes = [int(frame["frame_idx"]) for frame in execution.get("scheduled_frames", [])]
        materialize_calls.append({"execution_path": Path(execution_path), "frame_indexes": frame_indexes})
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for frame_idx in frame_indexes:
            (out / f"frame_{frame_idx:06d}.jpg").write_bytes(b"jpg")
        return {
            "extracted_frame_count": len(frame_indexes),
            "frame_indexes": frame_indexes,
            "frames": [f"frame_{frame_idx:06d}.jpg" for frame_idx in frame_indexes],
        }

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
    )

    assert any(
        call["execution_path"].name == "lane_a_pose_frame_execution.json" and call["frame_indexes"] == [0, 1]
        for call in materialize_calls
    )
    assert body_runners and "pose" not in body_runners[0]
    assert report["body_runtime_ran"] is True
    assert report["body_failure_note"] == ""


def test_body_video_smoke_ignores_generated_tracking_plan_when_deriving_lane_b_from_lane_a(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    inputs.mkdir()
    calls: list[str] = []

    baseline_plan = _frame_plan_payload()
    baseline_plan["frames"][0]["recommended_tier"] = "baseline"
    baseline_plan["frames"][0]["target_representation"] = "track_only"
    baseline_plan["deep_mesh_windows"] = []
    baseline_plan["summary"]["by_tier"] = {"baseline": 1}
    baseline_plan["summary"]["by_player_target_representation"] = {"track_only": 1}
    baseline_plan["summary"]["deep_mesh_frame_count"] = 0
    baseline_plan["summary"]["deep_mesh_window_count"] = 0

    def fake_run_pipeline(**kwargs):
        stage = kwargs["stage"]
        calls.append(stage)
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            _write_json(out / "court_calibration.json", _calibration_payload())
            _write_json(out / "frame_compute_plan.json", baseline_plan)
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}

        assert not (out / "frame_compute_plan.json").exists()
        frame = {
            "t": 1.0 / 30.0,
            "joints_world": [[0.01 * idx, 0.0, 0.2 + 0.03 * idx] for idx in range(17)],
            "joint_conf": [0.91] * 17,
            "mesh_vertices_world": [[0.0, 0.0, 0.1]],
            "transl_world": [0.0, 0.0, 0.0],
        }
        _write_json(out / "frame_compute_plan.json", _frame_plan_payload())
        _write_json(
            out / "body_compute_execution.json",
            {
                "schema_version": 1,
                "artifact_type": "racketsport_body_compute_execution",
                "mode": "adaptive_frame_compute_plan",
                "scheduled_frames": [
                    {
                        "frame_idx": 1,
                        "t": 1.0 / 30.0,
                        "target_representation": "world_mesh",
                        "target_player_ids": [7],
                        "active_player_ids": [7],
                    }
                ],
                "skipped_frames": [],
                "summary": {"scheduled_frame_count": 1, "scheduled_player_frame_count": 1},
            },
        )
        _write_json(out / "smpl_motion.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(out / "skeleton3d.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(
            out / "contact_splice.json",
            {
                "schema_version": 1,
                "artifact_type": "racketsport_contact_splice",
                "summary": {
                    "scheduled_contact_count": 1,
                    "spliced_contact_count": 1,
                    "mesh_unavailable_count": 0,
                    "fallback_spliced_count": 0,
                    "overridden_joint_count": 2,
                },
            },
        )
        return {"status": "blocked", "stages": [{"stage": "body", "status": "ran"}]}

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        execution = json.loads(Path(execution_path).read_text(encoding="utf-8"))
        frame_indexes = [int(frame["frame_idx"]) for frame in execution.get("scheduled_frames", [])]
        if Path(execution_path).name == "body_compute_execution.json" and not frame_indexes:
            raise ValueError("no scheduled BODY frames in execution manifest")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for frame_idx in frame_indexes:
            (out / f"frame_{frame_idx:06d}.jpg").write_bytes(b"jpg")
        return {
            "extracted_frame_count": len(frame_indexes),
            "frame_indexes": frame_indexes,
            "frames": [f"frame_{frame_idx:06d}.jpg" for frame_idx in frame_indexes],
        }

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
    )

    assert calls == ["tracking", "body"]
    assert report["body_runtime_ran"] is True
    assert report["body_failure_note"] == ""
    assert report["paths"]["frame_compute_plan"] == str(run_dir / "frame_compute_plan.json")
    assert report["summary"]["scheduled_player_frame_count"] == 1


def test_body_video_smoke_reports_quality_checked_outputs(tmp_path: Path, monkeypatch) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    _write_json(inputs / "frame_compute_plan.json", _frame_plan_payload())

    def fake_run_pipeline(**kwargs):
        stage = kwargs["stage"]
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            _write_json(out / "court_calibration.json", _calibration_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}
        frame = {
            "t": 1.0 / 30.0,
            "joints_world": [[0.01 * idx, 0.0, 0.2 + 0.03 * idx] for idx in range(17)],
            "joint_conf": [0.91] * 17,
            "mesh_vertices_world": [[0.0, 0.0, 0.1], [0.1, 0.0, 1.7]],
            "transl_world": [0.0, 0.0, 0.0],
        }
        _write_json(out / "smpl_motion.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(out / "skeleton3d.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(
            out / "contact_splice.json",
            {
                "schema_version": 1,
                "artifact_type": "racketsport_contact_splice",
                "summary": {
                    "scheduled_contact_count": 1,
                    "spliced_contact_count": 1,
                    "mesh_unavailable_count": 0,
                    "fallback_spliced_count": 0,
                    "overridden_joint_count": 2,
                },
            },
        )
        return {"status": "blocked", "stages": [{"stage": "body", "status": "ran"}]}

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        return {"extracted_frame_count": 1, "frame_indexes": [1], "frames": ["frame_000001.jpg"]}

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
    )

    assert report["status"] == "quality_checked_needs_accuracy_gate"
    assert report["body_runtime_ran"] is True
    assert report["summary"]["joint_frame_count"] == 1
    assert report["quality"]["quality_blockers"] == []
    assert report["quality"]["promotion_blockers"] == ["missing_world_mpjpe_gate"]
    assert report["quality"]["warnings"] == []
    assert report["full_clip_gate"]["passed"] is True
    assert report["full_clip_gate"]["coverage"] == 1.0
    assert report["full_clip_gate"]["contact_mesh_coverage"] == 1.0
    assert report["full_clip_gate"]["latency_seconds_per_video_minute"] is not None
    assert report["full_clip_gate"]["blockers"] == []
    assert report["label_packet"]["status"] == "needs_review"
    assert report["label_packet"]["sample_count"] == 1
    assert report["label_packet"]["required_review_sample_count"] == 1
    assert report["label_packet"]["selected_review_sample_count"] == 1
    assert report["label_packet"]["trusted_for_world_mpjpe"] is False
    assert report["label_review_bundle"]["status"] == "ready_for_review"
    assert report["label_review_bundle"]["selected_sample_count"] == 1
    assert report["label_review_bundle"]["missing_frame_count"] == 0
    assert report["paths"]["body_world_label_review_bundle"] == str(run_dir / "body_world_label_review_bundle")
    assert report["paths"]["body_world_label_review_overlay"] == str(
        run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json"
    )
    assert report["label_review_overlay"]["status"] == "blocked_missing_review_frames"
    assert report["label_review_overlay"]["sample_count"] == 1
    assert report["label_review_overlay"]["missing_frame_count"] == 1
    assert report["label_review_overlay"]["floor_anchor_projection_failed_count"] == 0
    assert report["label_review_overlay"]["floor_anchor_projection_warning_count"] == 0
    assert report["label_review_overlay"]["alignment_failed_count"] == 0
    assert report["label_review_overlay"]["alignment_warning_count"] == 0
    assert "missing_review_frame" in report["label_review_overlay"]["blockers"]
    overlay_index = json.loads(
        (run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert overlay_index["qualitative_status"] == "review_overlay_not_gate_verified"
    assert overlay_index["not_ground_truth"] is True
    full_gate = json.loads((run_dir / "body_full_clip_gate.json").read_text(encoding="utf-8"))
    assert full_gate["summary"]["tracked_player_frame_count"] == 1
    assert full_gate["summary"]["joint_player_frame_count"] == 1
    assert full_gate["summary"]["scheduled_contact_count"] == 1
    assert full_gate["summary"]["contact_mesh_frame_count"] == 1
    assert full_gate["contact_mesh_coverage"] == 1.0
    assert full_gate["latency_seconds_per_video_minute"] is not None
    quality = json.loads((run_dir / "body_joint_quality.json").read_text(encoding="utf-8"))
    assert quality["promotion_blockers"] == ["missing_world_mpjpe_gate"]
    label_packet = json.loads((run_dir / "body_world_label_packet.json").read_text(encoding="utf-8"))
    assert label_packet["not_ground_truth"] is True
    assert "joints_world" not in label_packet["samples"][0]
    assert label_packet["samples"][0]["predicted_joints_world"][0] == [0.0, 0.0, 0.2]
    packet_quality = json.loads((run_dir / "body_joint_quality_from_packet.json").read_text(encoding="utf-8"))
    assert packet_quality["summary"]["joint_source"] == "body_world_label_packet"
    assert packet_quality["summary"]["joint_frame_count"] == 1
    assert packet_quality["quality_blockers"] == []
    assert report["paths"]["body_joint_quality_from_packet"] == str(run_dir / "body_joint_quality_from_packet.json")


def test_body_video_smoke_diagnostic_full_track_removes_generated_frame_plan_before_body_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    inputs.mkdir()
    calls: list[dict] = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        stage = kwargs["stage"]
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            _write_json(out / "frame_compute_plan.json", _frame_plan_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}
        assert (out / "frame_compute_plan.json").exists()
        execution = json.loads((out / "body_compute_execution.json").read_text(encoding="utf-8"))
        assert execution["mode"] == "adaptive_frame_compute_plan"
        assert execution["summary"]["scheduled_player_frame_count"] == 1
        frame = {
            "t": 1.0 / 30.0,
            "joints_world": [[0.01 * idx, 0.0, 0.2 + 0.03 * idx] for idx in range(17)],
            "joint_conf": [0.91] * 17,
            "mesh_vertices_world": [[0.0, 0.0, 0.1]],
            "transl_world": [0.0, 0.0, 0.0],
        }
        _write_json(out / "smpl_motion.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(out / "skeleton3d.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(out / "frame_compute_plan.json", _frame_plan_payload())
        _write_json(
            out / "body_compute_execution.json",
            {
                "schema_version": 1,
                "artifact_type": "racketsport_body_compute_execution",
                "mode": "adaptive_frame_compute_plan",
                "scheduled_frames": [],
                "skipped_frames": [],
                "summary": {"scheduled_frame_count": 0, "scheduled_player_frame_count": 0},
            },
        )
        return {"status": "blocked", "stages": [{"stage": "body", "status": "ran"}]}

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        return {"extracted_frame_count": 1, "frame_indexes": [1], "frames": ["frame_000001.jpg"]}

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
        diagnostic_full_track=True,
    )

    assert calls[0]["stage"] == "tracking"
    assert calls[1]["stage"] == "body"
    assert report["paths"]["frame_compute_plan"] == str(run_dir / "frame_compute_plan.json")
    assert report["summary"]["scheduled_player_frame_count"] == 1
    assert report["diagnostic_full_track_mode"] is True
    assert any("bypasses the contact-aware tier rule" in note for note in report["frame_plan_notes"])
    assert (run_dir / "frame_compute_plan.json").exists()
    execution = json.loads((run_dir / "body_compute_execution.json").read_text(encoding="utf-8"))
    assert execution["mode"] == "adaptive_frame_compute_plan"
    assert execution["summary"]["scheduled_player_frame_count"] == 1


def test_body_video_smoke_default_path_ignores_stale_diagnostic_full_track_plan_in_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The default (non-diagnostic) path must never silently inherit a
    100%-coverage diagnostic_full_track plan that happens to already be
    sitting in inputs_dir (e.g. copied over from a prior diagnostic run
    sharing the same inputs bundle) -- that would bypass the contact-aware
    tier rule in what looks like a normal production run."""

    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    inputs.mkdir()
    # A stale diagnostic-tagged plan left in the shared inputs bundle.
    stale_plan = _frame_plan_payload()
    stale_plan["diagnostic_full_track"] = True
    _write_json(inputs / "frame_compute_plan.json", stale_plan)
    calls: list[dict] = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        stage = kwargs["stage"]
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}
        # The default path must not have pre-written frame_compute_plan.json
        # from the stale diagnostic plan; the orchestrator (mocked here) is
        # the one that would derive the real contact-aware plan.
        assert not (out / "frame_compute_plan.json").exists()
        frame = {
            "t": 1.0 / 30.0,
            "joints_world": [[0.0, 0.0, 0.2] for _ in range(17)],
            "joint_conf": [0.9] * 17,
            "mesh_vertices_world": [[0.0, 0.0, 0.1]],
            "transl_world": [0.0, 0.0, 0.0],
        }
        _write_json(out / "smpl_motion.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(out / "skeleton3d.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        return {"status": "blocked", "stages": [{"stage": "body", "status": "ran"}]}

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        return {"extracted_frame_count": 1, "frame_indexes": [1], "frames": ["frame_000001.jpg"]}

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
    )

    assert report["diagnostic_full_track_mode"] is False
    assert any("ignored diagnostic-tagged" in note for note in report["frame_plan_notes"])
    assert report["paths"]["frame_compute_plan"] == ""


def test_body_video_smoke_default_path_still_reuses_non_diagnostic_inputs_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A normal (non-diagnostic) contact-aware plan already present in
    inputs_dir must still be reused as before -- only diagnostic_full_track
    plans get rejected."""

    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video bytes")
    inputs.mkdir()
    _write_json(inputs / "frame_compute_plan.json", _frame_plan_payload())
    calls: list[dict] = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        stage = kwargs["stage"]
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}
        assert (out / "frame_compute_plan.json").exists()
        frame = {
            "t": 1.0 / 30.0,
            "joints_world": [[0.0, 0.0, 0.2] for _ in range(17)],
            "joint_conf": [0.9] * 17,
            "mesh_vertices_world": [[0.0, 0.0, 0.1]],
            "transl_world": [0.0, 0.0, 0.0],
        }
        _write_json(out / "smpl_motion.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        _write_json(out / "skeleton3d.json", {"schema_version": 1, "players": [{"id": 7, "frames": [frame]}]})
        return {"status": "blocked", "stages": [{"stage": "body", "status": "ran"}]}

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        return {"extracted_frame_count": 1, "frame_indexes": [1], "frames": ["frame_000001.jpg"]}

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
    )

    assert report["diagnostic_full_track_mode"] is False
    assert report["frame_plan_notes"] == []
    assert report["paths"]["frame_compute_plan"] == str(run_dir / "frame_compute_plan.json")


def test_body_video_smoke_forwards_runtime_overrides_to_body_stage(tmp_path: Path, monkeypatch) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    video = tmp_path / "source.mp4"
    fast_sam_repo = tmp_path / "fast-sam"
    video.write_bytes(b"fake video bytes")
    fast_sam_repo.mkdir()
    _write_json(inputs / "frame_compute_plan.json", _frame_plan_payload())
    calls: list[dict] = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        stage = kwargs["stage"]
        out = Path(kwargs["run_dir"])
        if stage == "tracking":
            _write_json(out / "tracks.json", _tracks_payload())
            return {"status": "blocked", "stages": [{"stage": "tracking", "status": "ran"}]}
        _write_json(
            out / "smpl_motion.json",
            {
                "schema_version": 1,
                "players": [
                    {
                        "id": 7,
                        "frames": [
                            {
                                "t": 1.0 / 30.0,
                                "joints_world": [[0.01 * idx, 0.0, 0.2 + 0.03 * idx] for idx in range(17)],
                                "joint_conf": [0.91] * 17,
                                "mesh_vertices_world": [[0.0, 0.0, 0.1]],
                                "transl_world": [0.0, 0.0, 0.0],
                            }
                        ],
                    }
                ],
            },
        )
        _write_json(
            out / "skeleton3d.json",
            {
                "schema_version": 1,
                "players": [
                    {
                        "id": 7,
                        "frames": [
                            {
                                "t": 1.0 / 30.0,
                                "joints_world": [[0.01 * idx, 0.0, 0.2 + 0.03 * idx] for idx in range(17)],
                                "joint_conf": [0.91] * 17,
                                "mesh_vertices_world": [[0.0, 0.0, 0.1]],
                                "transl_world": [0.0, 0.0, 0.0],
                            }
                        ],
                    }
                ],
            },
        )
        return {"status": "blocked", "stages": [{"stage": "body", "status": "ran"}]}

    def fake_materialize_body_frames(*, video_path, execution_path, out_dir, overwrite=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        return {"extracted_frame_count": 1, "frame_indexes": [1], "frames": ["frame_000001.jpg"]}

    monkeypatch.setattr("threed.racketsport.body_video_smoke.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("threed.racketsport.body_video_smoke.materialize_body_frames", fake_materialize_body_frames)

    report = run_body_video_smoke(
        clip="clip_001",
        inputs_dir=inputs,
        video_path=video,
        run_dir=run_dir,
        tracking_mode="precomputed",
        fast_sam_repo=fast_sam_repo,
        body_detector_name="",
        body_fov_name="",
    )

    assert report["body_runtime_ran"] is True
    assert calls[0].get("runners") is None
    body_runner = calls[1]["runners"]["body"]
    assert isinstance(body_runner, BodyStageRunner)
    assert body_runner.fast_sam_repo == fast_sam_repo
    assert body_runner.detector_name == ""
    assert body_runner.fov_name == ""


def test_run_body_video_smoke_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/run_body_video_smoke.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--video" in completed.stdout
    assert "--inputs" in completed.stdout
    assert "--fast-sam-repo" in completed.stdout
    assert "--body-detector-name" in completed.stdout
    assert "--body-fov-name" in completed.stdout
