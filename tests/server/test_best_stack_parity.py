from __future__ import annotations

from pathlib import Path

from scripts.racketsport import process_video
from server import gpu_runner
from server.worker import daemon
from threed.racketsport.best_stack import load_best_stack_manifest


def test_server_auto_court_preview_default_is_manifest_declared_override(tmp_path: Path) -> None:
    manifest = load_best_stack_manifest()
    assert manifest.server_override_value("allow_auto_court_corners_preview") is True

    request = gpu_runner.GpuRunRequest(
        job_id="job_1",
        clip="clip_1",
        input_dir=tmp_path / "input",
        video_path=tmp_path / "input" / "clip.mp4",
        artifacts_dir=tmp_path / "artifacts",
    )

    assert request.allow_auto_court_corners_preview is True
    assert daemon.default_allow_auto_court_corners_preview() is True


def test_cli_server_shared_defaults_match_manifest_except_declared_overrides(tmp_path: Path) -> None:
    manifest = load_best_stack_manifest()
    parser = process_video.build_arg_parser()
    options = process_video.build_options_from_args(
        parser.parse_args(["--video", str(tmp_path / "clip.mp4"), "--out", str(tmp_path / "run")])
    )
    resolved = process_video.resolved_best_stack_config_from_options(options)

    assert resolved["ball.wasb_checkpoint"] == manifest.path_value("ball.wasb_checkpoint").as_posix()
    assert resolved["ball.wasb_repo"] == manifest.path_value("ball.wasb_repo").as_posix()
    assert resolved["tracking.reid_model"] == manifest.path_value("tracking.reid_model", must_exist=False).as_posix()
    assert resolved["mesh.byte_budget_mib"] == 300.0

    assert options.allow_auto_court_corners_preview is False
    assert manifest.server_override_value("allow_auto_court_corners_preview") is True
