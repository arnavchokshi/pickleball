from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.racketsport import process_video
from tests.racketsport.test_paddle_pose_fused import _frame, _skeleton
from tests.racketsport.test_process_video import (
    _court_calibration_payload,
    _make_video,
    _tracks_payload,
    _write_json,
)
from threed.racketsport.best_stack import load_best_stack_manifest


def _patch_cpu_pipeline_inputs(monkeypatch, *, emit_skeleton: bool) -> None:
    def _fake_calibration(self: process_video.ProcessVideoPipeline) -> process_video.StageOutcome:
        payload = _court_calibration_payload()
        _write_json(self.clip_dir / "court_calibration.json", payload)
        self._set_court_trust_band(payload, self.clip_dir / "court_calibration.json")
        return process_video.StageOutcome(
            stage="calibration",
            status="ran",
            wall_seconds=0.0,
            artifacts=["court_calibration.json"],
        )

    def _fake_tracking(self: process_video.ProcessVideoPipeline) -> process_video.StageOutcome:
        _write_json(self.clip_dir / "tracks.json", _tracks_payload())
        return process_video.StageOutcome(
            stage="tracking",
            status="ran",
            wall_seconds=0.0,
            artifacts=["tracks.json"],
        )

    def _fake_frames(self: process_video.ProcessVideoPipeline) -> process_video.StageOutcome:
        return process_video.StageOutcome(stage="frames", status="ran", wall_seconds=0.0)

    def _fake_body(self: process_video.ProcessVideoPipeline) -> process_video.StageOutcome:
        if emit_skeleton:
            frames = [_frame(i / 30.0, frame_idx=i) for i in range(3)]
            _write_json(self.clip_dir / "skeleton3d.json", _skeleton(frames))
            return process_video.StageOutcome(
                stage="body",
                status="ran",
                wall_seconds=0.0,
                artifacts=["skeleton3d.json"],
                trust_badge="low_confidence",
            )
        return process_video.StageOutcome(
            stage="body",
            status="degraded",
            wall_seconds=0.0,
            notes=["test body emitted no SAM-3D skeleton evidence"],
        )

    monkeypatch.setattr(process_video.ProcessVideoPipeline, "_stage_calibration", _fake_calibration)
    monkeypatch.setattr(process_video.ProcessVideoPipeline, "_stage_tracking", _fake_tracking)
    monkeypatch.setattr(process_video.ProcessVideoPipeline, "_stage_frames", _fake_frames)
    monkeypatch.setattr(process_video.ProcessVideoPipeline, "_stage_body", _fake_body)


def _run_cpu_entrypoint(
    tmp_path: Path,
    monkeypatch,
    *,
    emit_skeleton: bool,
    extra_args: list[str] | None = None,
) -> tuple[dict[str, Any], process_video.PipelineOptions]:
    video = tmp_path / "clip.mp4"
    _make_video(video, frame_count=3)
    _patch_cpu_pipeline_inputs(monkeypatch, emit_skeleton=emit_skeleton)
    parser = process_video.build_arg_parser()
    args = parser.parse_args(
        [
            "--video",
            str(video),
            "--out",
            str(tmp_path / "run"),
            "--skip-ball",
            "--no-gpu",
            *(extra_args or []),
        ]
    )
    options = process_video.build_options_from_args(args)
    summary = process_video.ProcessVideoPipeline(options).run()
    return summary, options


def test_best_stack_declares_fused_paddle_wired_default_and_reflection_dormant() -> None:
    manifest = load_best_stack_manifest()

    fused = manifest.entry("paddle.fused_estimator")
    assert fused.status == "WIRED_DEFAULT"
    assert fused.value == {
        "enabled": True,
        "artifact": "racket_pose_estimate.json",
        "source": "wrist_palm_grip_fused",
        "trust_band": "estimated_preview",
    }
    assert any("acceptance_record_v2.json" in path for path in fused.provenance["evidence_paths"])
    assert fused.proven_against["wolverine_mean_iou"] == 0.235558
    assert fused.proven_against["burlington_mean_iou"] == 0.342387

    reflection = manifest.entry("paddle.reflection_cone_factor")
    assert reflection.status == "DORMANT"
    assert reflection.gate["metric_key"] == "p1_4_real_3d_ball_velocities"
    assert "reflection_channel_dormant_no_usable_ball_contacts" in reflection.notes


def test_process_video_default_run_produces_fused_paddle_artifact_from_sam3d_skeleton(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary, options = _run_cpu_entrypoint(tmp_path, monkeypatch, emit_skeleton=True)

    by_stage = {stage["stage"]: stage for stage in summary["stages"]}
    assert by_stage["paddle_pose"]["status"] == "ran"
    assert by_stage["paddle_pose"]["metrics"]["paddle_pose"]["status"] == "preview"
    assert by_stage["paddle_pose"]["metrics"]["paddle_pose"]["coverage"]["estimate_frame_count"] == 3
    artifact = json.loads((options.clip_dir / "racket_pose_estimate.json").read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "racketsport_racket_pose_estimate"
    assert artifact["source"] == "wrist_palm_grip_fused"
    assert artifact["render_only"] is True
    assert artifact["not_for_detection_metrics"] is True
    assert artifact["trusted_for_rkt_promotion"] is False
    assert artifact["rkt_gate_unscoreable"] is True
    assert summary["best_stack"]["resolved"]["paddle.fused_estimator"]["enabled"] is True


def test_process_video_paddle_stage_fail_closes_without_skeleton_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary, options = _run_cpu_entrypoint(tmp_path, monkeypatch, emit_skeleton=False)

    by_stage = {stage["stage"]: stage for stage in summary["stages"]}
    paddle = by_stage["paddle_pose"]
    assert paddle["status"] == "blocked"
    assert paddle["metrics"]["paddle_pose"] == {
        "status": "blocked",
        "reason": "missing_sam3d_skeleton3d",
        "coverage": {
            "estimate_frame_count": 0,
            "input_player_count": 0,
            "hidden_frame_count": 0,
        },
    }
    assert not (options.clip_dir / "racket_pose_estimate.json").exists()
    assert by_stage["world"]["status"] == "ran"


def test_no_paddle_pose_flag_skips_default_stage_and_records_best_stack_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary, options = _run_cpu_entrypoint(
        tmp_path,
        monkeypatch,
        emit_skeleton=True,
        extra_args=["--no-paddle-pose"],
    )

    by_stage = {stage["stage"]: stage for stage in summary["stages"]}
    assert by_stage["paddle_pose"]["status"] == "skipped"
    assert "--no-paddle-pose set" in " ".join(by_stage["paddle_pose"]["notes"])
    assert not (options.clip_dir / "racket_pose_estimate.json").exists()
    assert summary["best_stack"]["resolved"]["paddle.fused_estimator"]["enabled"] is False
    assert summary["best_stack"]["overrides"]["paddle.fused_estimator"]["manifest"]["enabled"] is True
    assert summary["best_stack"]["overrides"]["paddle.fused_estimator"]["resolved"]["enabled"] is False
