from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.racketsport import process_video
from scripts.racketsport.remote_body_dispatch import RemoteConfig
from threed.racketsport import orchestrator
from threed.racketsport.best_stack import load_best_stack_manifest
from threed.racketsport.hmr_deep import verify_fast_sam_manifest_assets


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tracks_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {
                        "t": 0.0,
                        "bbox": [100.0, 100.0, 200.0, 300.0],
                        "world_xy": [1.0, 2.0],
                        "conf": 0.9,
                    }
                ],
            }
        ],
        "rally_spans": [],
    }


def _contact_windows_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 0.0,
                "frame": 0,
                "player_id": 1,
                "confidence": 0.95,
                "sources": {"wrist_vel": 0.95, "ball_inflection": 0.9},
                "window": {"t0": 0.0, "t1": 0.05, "importance": 0.95},
            }
        ],
    }


def _events_selected_payload() -> dict[str, Any]:
    return {
        "artifact_type": "racketsport_ball_arc_events_selected",
        "selected": [
            {
                "anchor_id": "contact_000_p1_left",
                "kind": "contact",
                "frame": 0,
                "t": 0.0,
                "player_id": 1,
                "candidate_confidence": 0.95,
                "selected": True,
            }
        ],
        "rejected": [],
        "selected_count": 1,
    }


def test_no_flag_invocation_resolves_wired_defaults_from_manifest(tmp_path: Path) -> None:
    manifest = load_best_stack_manifest()
    video = tmp_path / "clip.mp4"
    parser = process_video.build_arg_parser()
    options = process_video.build_options_from_args(
        parser.parse_args(["--video", str(video), "--out", str(tmp_path / "run")])
    )
    resolved = process_video.resolved_best_stack_config_from_options(options)

    assert resolved["ball.wasb_checkpoint"] == manifest.path_value("ball.wasb_checkpoint").as_posix()
    assert resolved["ball.wasb_repo"] == manifest.path_value("ball.wasb_repo").as_posix()
    assert resolved["tracking.reid_model"] == manifest.path_value("tracking.reid_model", must_exist=False).as_posix()
    assert resolved["confidence.calibration_curves"] == manifest.path_value("confidence.calibration_curves").as_posix()
    assert resolved["mesh.coverage_mode"] == manifest.string_value("mesh.coverage_mode")
    assert resolved["mesh.byte_budget_mib"] == manifest.number_value("mesh.byte_budget_mib")
    assert resolved["mesh.target_frame_budget"] == manifest.value("mesh.target_frame_budget")
    assert resolved["tracking.global_association_profile"] == manifest.string_value("tracking.global_association_profile")
    assert resolved["body.detector_fov"] == manifest.value("body.detector_fov")
    assert resolved["camera_motion.policy"] == manifest.value("camera_motion.policy")


def test_clip_keyed_association_no_flag_resolves_manifest_default(tmp_path: Path) -> None:
    manifest = load_best_stack_manifest()
    manifest_default = manifest.string_value("tracking.global_association_profile")
    parser = process_video.build_arg_parser()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    for clip in (
        "burlington_gold_0300_low_steep_corner",
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "wolverine_mixed_0200_mid_steep_corner",
    ):
        options = process_video.build_options_from_args(
            parser.parse_args(
                [
                    "--video",
                    str(video),
                    "--clip",
                    clip,
                    "--out",
                    str(tmp_path / clip),
                ]
            )
        )
        resolved = process_video.resolved_best_stack_config_from_options(options)

        assert resolved["tracking.global_association_profile"] == manifest_default
        assert "tracking.global_association_profile" not in process_video.best_stack_overrides_from_options(options)

    explicit = process_video.build_options_from_args(
        parser.parse_args(
            [
                "--video",
                str(video),
                "--clip",
                "burlington_gold_0300_low_steep_corner",
                "--global-association-profile",
                "burlington_internal_val_trk10_iter5_minconf05_appw2_margin2",
                "--out",
                str(tmp_path / "explicit"),
            ]
        )
    )
    explicit_resolved = process_video.resolved_best_stack_config_from_options(explicit)
    explicit_overrides = process_video.best_stack_overrides_from_options(explicit)

    assert explicit_resolved["tracking.global_association_profile"] == "burlington_internal_val_trk10_iter5_minconf05_appw2_margin2"
    assert (
        explicit_overrides["tracking.global_association_profile"]["resolved"]
        == "burlington_internal_val_trk10_iter5_minconf05_appw2_margin2"
    )
    assert explicit_overrides["tracking.global_association_profile"]["manifest"] == manifest_default


def test_no_flag_mesh_byte_budget_matches_explicit_300_and_fixed_frame_flag_wins(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    parser = process_video.build_arg_parser()
    no_flag = process_video.build_options_from_args(
        parser.parse_args(["--video", str(video), "--out", str(tmp_path / "run")])
    )
    explicit_budget = process_video.build_options_from_args(
        parser.parse_args(
            ["--video", str(video), "--out", str(tmp_path / "run2"), "--mesh-byte-budget-mib", "300"]
        )
    )
    fixed_frame = process_video.build_options_from_args(
        parser.parse_args(
            [
                "--video",
                str(video),
                "--out",
                str(tmp_path / "run3"),
                "--target-mesh-frame-budget",
                "250",
                "--mesh-byte-budget-mib",
                "200",
            ]
        )
    )

    assert no_flag.mesh_byte_budget_mib == 300.0
    assert no_flag.target_mesh_frame_budget is None
    assert explicit_budget.mesh_byte_budget_mib == no_flag.mesh_byte_budget_mib
    assert explicit_budget.target_mesh_frame_budget == no_flag.target_mesh_frame_budget
    assert fixed_frame.target_mesh_frame_budget == 250
    assert fixed_frame.mesh_byte_budget_mib is None
    assert fixed_frame.remote_config.target_mesh_frame_budget == 250
    assert fixed_frame.remote_config.mesh_byte_budget_mib is None


def test_body_detector_fov_defaults_match_manifest_for_local_and_remote() -> None:
    manifest = load_best_stack_manifest()
    expected = manifest.value("body.detector_fov")

    local = orchestrator.BodyStageRunner()
    remote = RemoteConfig()

    expected_identifiers = {
        "detector_name": expected["detector_name"],
        "fov_name": expected["fov_name"],
    }

    assert {"detector_name": local.detector_name, "fov_name": local.fov_name} == expected_identifiers
    assert {"detector_name": remote.body_detector_name, "fov_name": remote.body_fov_name} == expected_identifiers


def test_fov_checkpoint_repair_is_relative_and_absence_fails_loud() -> None:
    with pytest.raises(FileNotFoundError, match="moge_2_vitl_normal"):
        verify_fast_sam_manifest_assets(
            "models/MANIFEST.json",
            required_model_ids=("moge_2_vitl_normal",),
        )


def test_events_before_frames_makes_cold_mesh_plan_contact_dense(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    options = process_video.PipelineOptions(video=video, clip="clip", run_dir=tmp_path / "run", max_players=1, no_gpu=False)
    pipeline = process_video.ProcessVideoPipeline(options)
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "contact_windows.json", _contact_windows_payload())
    _write_json(options.clip_dir / "events_selected.json", _events_selected_payload())

    def _fake_materialize(**kwargs: Any) -> dict[str, Any]:
        plan = json.loads(Path(kwargs["frame_compute_plan_path"]).read_text(encoding="utf-8"))
        assert plan["mesh_coverage_policy"]["contact_selected_frame_count"] > 0
        return {
            "frame_count": 1,
            "total_bytes": 1024,
            "notes": ["fake extraction"],
            "schedule": {"capped": False, "source": "frame_compute_plan.json"},
        }

    monkeypatch.setattr(process_video, "materialize_process_video_frames", _fake_materialize)

    events_outcome = pipeline._stage_events()
    frames_outcome = pipeline._stage_frames()
    plan = json.loads((options.clip_dir / "frame_compute_plan.json").read_text(encoding="utf-8"))

    assert events_outcome.status == "ran"
    assert frames_outcome.status == "ran"
    assert plan["mesh_coverage_policy"]["mesh_budget_policy"] == "byte_budget"
    assert plan["mesh_coverage_policy"]["mesh_byte_budget_mib"] == 300.0
    assert plan["mesh_coverage_policy"]["contact_selected_frame_count"] > 0
    assert plan["mesh_coverage_policy"]["ball_aware_trigger_source_counts"]["events"] > 0
