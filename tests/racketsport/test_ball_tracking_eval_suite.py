from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.racketsport.run_ball_tracking_eval_suite import (
    EvalSuiteConfig,
    ExternalCandidate,
    run_ball_tracking_eval_suite,
)


def _write_track(path: Path, *, fps: float = 30.0, visible: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": fps,
                "source": "tracknet",
                "frames": [
                    {"t": 0.0, "xy": [10.0, 10.0], "conf": 1.0 if visible else 0.0, "visible": visible},
                    {"t": 1.0 / fps, "xy": [12.0, 11.0], "conf": 1.0 if visible else 0.0, "visible": visible},
                ],
                "bounces": [],
            }
        ),
        encoding="utf-8",
    )


def _write_clicks(path: Path, *, clip: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_click_review",
                "status": "human_reviewed",
                "clip": clip,
                "target_file": "ball.json",
                "coordinate_frame": "image_pixels_video_space",
                "items": [
                    {
                        "review_id": "ball_frame_000000",
                        "frame_index": 0,
                        "t": 0.0,
                        "image": "frame_000000.jpg",
                        "ball_xy": [10.0, 10.0],
                        "visible": True,
                        "visibility": "visible",
                    },
                    {
                        "review_id": "ball_frame_000001",
                        "frame_index": 1,
                        "t": 1.0 / 30.0,
                        "image": "frame_000001.jpg",
                        "ball_xy": None,
                        "visible": False,
                        "visibility": "missing",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_video(path: Path, *, fps: float = 30.0, frame_count: int = 2) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (16, 16),
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()


def test_eval_suite_combines_generated_candidates_timings_and_benchmark(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    review_root = run_root / "ball_click_review_30"
    clip = "clip_a"
    base = run_root / clip / "tracknet_smoke_0000_0010"
    _write_track(base / "ball_track_0000_0010.json")
    _write_track(base / "ball_track_fusion_temporal_vball100.json")
    _write_track(base / "ball_track_fusion_temporal_vball100_localtraj.json")
    _write_track(base / "ball_track_target_court_120px.json")
    _write_track(base / "ball_track_target_court_temporal.json")
    _write_track(base / "vballnet_fast" / "ball_track.json")
    _write_track(base / "vballnet_v1" / "ball_track.json")
    _write_clicks(review_root / clip / "ball_points.json", clip=clip)
    (base / "input_0000_0010.mp4").write_bytes(b"unused")

    calls: list[str] = []

    def fake_temporal(*, ball_track_path, out_path, summary_path, mode, **_kwargs):
        calls.append(f"temporal:{mode}:{Path(out_path).name}")
        _write_track(Path(out_path), visible=True)
        Path(summary_path).write_text(json.dumps({"mode": mode}), encoding="utf-8")
        return {"mode": mode}

    def fake_fusion(*, out_path, summary_path, require_stable_verifier_support=False, **_kwargs):
        calls.append(f"fusion:{require_stable_verifier_support}:{Path(out_path).name}")
        _write_track(Path(out_path), visible=True)
        Path(summary_path).write_text(json.dumps({"stable_veto": require_stable_verifier_support}), encoding="utf-8")
        return {"stable_veto": require_stable_verifier_support}

    summary = run_ball_tracking_eval_suite(
        EvalSuiteConfig(
            run_root=run_root,
            review_root=review_root,
            out_root=tmp_path / "eval",
            clips=[clip],
            run_tracknet=False,
            render_overlays=False,
            selected_candidate="pbmat_v0_motion_composite",
        ),
        temporal_writer=fake_temporal,
        fusion_writer=fake_fusion,
    )

    assert summary["clip_count"] == 1
    assert summary["include_pbmat_v0"] is True
    assert summary["timings"]["total_seconds"] >= 0.0
    assert "pbmat_v0_motion_composite" in summary["benchmark"]["aggregate"]
    assert summary["benchmark"]["aggregate"]["pbmat_v0_motion_composite"]["category"] == "composite_alias_not_trained_model"
    assert "fusion_temporal_vball100_ballistic" in summary["benchmark"]["aggregate"]
    assert "fusion_temporal_vball100_localtraj_ballistic" in summary["benchmark"]["aggregate"]
    assert "fusion_temporal_vball100_stable_veto" in summary["benchmark"]["aggregate"]
    assert summary["selection"]["clips"][clip]["status"] == "selected"
    assert summary["selection"]["candidate_category"] == "composite_alias_not_trained_model"
    assert summary["selection"]["candidate_score"] is not None
    assert summary["selection"]["eligible_for_model_ranking"] is False
    selected_track = tmp_path / "eval" / "selected_tracks" / clip / "ball_track.json"
    assert json.loads(selected_track.read_text(encoding="utf-8"))["source"] == "tracknet"
    selection_sidecar = json.loads((selected_track.parent / "ball_track_selection.json").read_text(encoding="utf-8"))
    assert selection_sidecar["trained_pbmat_checkpoint"] is False
    assert any(call.startswith("temporal:ballistic") for call in calls)
    assert any(call.startswith("fusion:True") for call in calls)
    assert (tmp_path / "eval" / "benchmark.json").is_file()


def test_eval_suite_does_not_crash_when_no_ballistic_source_exists(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    review_root = run_root / "ball_click_review_30"
    clip = "clip_a"
    base = run_root / clip / "tracknet_smoke_0000_0010"
    _write_track(base / "ball_track_0000_0010.json")
    _write_clicks(review_root / clip / "ball_points.json", clip=clip)
    (base / "input_0000_0010.mp4").write_bytes(b"unused")

    summary = run_ball_tracking_eval_suite(
        EvalSuiteConfig(
            run_root=run_root,
            review_root=review_root,
            out_root=tmp_path / "eval",
            clips=[clip],
            run_tracknet=False,
        )
    )

    assert "tracknet_raw_existing" in summary["benchmark"]["aggregate"]
    assert not any(name.endswith("_ballistic") for name in summary["benchmark"]["aggregate"])


def test_eval_suite_adds_external_real_candidate_tracks_to_benchmark(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    review_root = run_root / "ball_click_review_30"
    clip = "clip_a"
    base = run_root / clip / "tracknet_smoke_0000_0010"
    external = tmp_path / "external" / clip / "tracknet_wasb_fusion" / "ball_track.json"
    _write_track(base / "ball_track_0000_0010.json")
    _write_track(external, visible=True)
    _write_clicks(review_root / clip / "ball_points.json", clip=clip)
    (base / "input_0000_0010.mp4").write_bytes(b"unused")

    summary = run_ball_tracking_eval_suite(
        EvalSuiteConfig(
            run_root=run_root,
            review_root=review_root,
            out_root=tmp_path / "eval",
            clips=[clip],
            run_tracknet=False,
            external_candidates=[
                ExternalCandidate(
                    clip=clip,
                    name="tracknet_wasb_fusion",
                    path=external,
                    category="m8_wasb_fusion",
                )
            ],
        )
    )

    assert summary["generated_candidates"][clip]["tracknet_wasb_fusion"] == str(external)
    aggregate = summary["benchmark"]["aggregate"]["tracknet_wasb_fusion"]
    assert aggregate["category"] == "m8_wasb_fusion"
    assert aggregate["clip_count"] == 1


def test_eval_suite_reruns_tracknet_with_repo_directory(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    review_root = run_root / "ball_click_review_30"
    clip = "clip_a"
    base = run_root / clip / "tracknet_smoke_0000_0010"
    _write_track(base / "ball_track_0000_0010.json")
    _write_track(base / "vballnet_fast" / "ball_track.json")
    _write_track(base / "vballnet_v1" / "ball_track.json")
    _write_clicks(review_root / clip / "ball_points.json", clip=clip)
    (base / "input_0000_0010.mp4").write_bytes(b"unused")
    tracknet_repo = tmp_path / "TrackNetV3"
    tracknet_repo.mkdir()
    (tracknet_repo / "predict.py").write_text("print('fake')\n", encoding="utf-8")
    tracknet_file = tmp_path / "TrackNet_best.pt"
    inpaintnet_file = tmp_path / "InpaintNet_best.pt"
    tracknet_file.write_bytes(b"tracknet")
    inpaintnet_file.write_bytes(b"inpaintnet")

    calls: list[str] = []

    def fake_tracknet(*, out, tracknet_repo, tracknet_file, inpaintnet_file, **_kwargs):
        calls.append(f"tracknet:{Path(tracknet_repo).name}:{Path(tracknet_file).name}:{Path(inpaintnet_file).name}")
        _write_track(Path(out), visible=True)
        return {"visible_frame_count": 2}

    def fake_court(*, out_path, summary_path, **_kwargs):
        calls.append(f"court:{Path(out_path).name}")
        _write_track(Path(out_path), visible=True)
        Path(summary_path).write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"ok": True}

    def fake_temporal(*, out_path, summary_path, mode, **_kwargs):
        calls.append(f"temporal:{mode}:{Path(out_path).name}")
        _write_track(Path(out_path), visible=True)
        Path(summary_path).write_text(json.dumps({"mode": mode}), encoding="utf-8")
        return {"mode": mode}

    def fake_fusion(*, out_path, summary_path, **_kwargs):
        calls.append(f"fusion:{Path(out_path).name}")
        _write_track(Path(out_path), visible=True)
        Path(summary_path).write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"ok": True}

    summary = run_ball_tracking_eval_suite(
        EvalSuiteConfig(
            run_root=run_root,
            review_root=review_root,
            out_root=tmp_path / "eval",
            clips=[clip],
            run_tracknet=True,
            tracknet_repo=tracknet_repo,
            tracknet_file=tracknet_file,
            inpaintnet_file=inpaintnet_file,
        ),
        tracknet_runner=fake_tracknet,
        court_writer=fake_court,
        temporal_writer=fake_temporal,
        fusion_writer=fake_fusion,
    )

    assert any(call == "tracknet:TrackNetV3:TrackNet_best.pt:InpaintNet_best.pt" for call in calls)
    assert "tracknet_pretrained_raw" in summary["benchmark"]["aggregate"]
    assert "tracknet_pretrained_fusion_vball100_localtraj" in summary["benchmark"]["aggregate"]


def test_eval_suite_rerun_uses_video_fps_when_existing_track_is_absent(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    review_root = run_root / "ball_click_review_30"
    clip = "clip_a"
    base = run_root / clip / "tracknet_smoke_0000_0010"
    base.mkdir(parents=True)
    _write_video(base / "input_0000_0010.mp4", fps=24.0)
    _write_track(base / "vballnet_fast" / "ball_track.json", fps=24.0)
    _write_track(base / "vballnet_v1" / "ball_track.json", fps=24.0)
    _write_clicks(review_root / clip / "ball_points.json", clip=clip)
    tracknet_repo = tmp_path / "TrackNetV3"
    tracknet_repo.mkdir()
    (tracknet_repo / "predict.py").write_text("print('fake')\n", encoding="utf-8")
    tracknet_file = tmp_path / "TrackNet_best.pt"
    inpaintnet_file = tmp_path / "InpaintNet_best.pt"
    tracknet_file.write_bytes(b"tracknet")
    inpaintnet_file.write_bytes(b"inpaintnet")
    seen_fps: list[float] = []

    def fake_tracknet(*, out, fps, **_kwargs):
        seen_fps.append(float(fps))
        _write_track(Path(out), fps=fps, visible=True)
        return {"visible_frame_count": 2}

    def fake_court(*, out_path, summary_path, ball_track_path, **_kwargs):
        _write_track(Path(out_path), fps=json.loads(Path(ball_track_path).read_text(encoding="utf-8"))["fps"], visible=True)
        Path(summary_path).write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"ok": True}

    def fake_temporal(*, out_path, summary_path, ball_track_path, mode, **_kwargs):
        _write_track(Path(out_path), fps=json.loads(Path(ball_track_path).read_text(encoding="utf-8"))["fps"], visible=True)
        Path(summary_path).write_text(json.dumps({"mode": mode}), encoding="utf-8")
        return {"mode": mode}

    def fake_fusion(*, out_path, summary_path, primary_ball_track_path, **_kwargs):
        _write_track(Path(out_path), fps=json.loads(Path(primary_ball_track_path).read_text(encoding="utf-8"))["fps"], visible=True)
        Path(summary_path).write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"ok": True}

    run_ball_tracking_eval_suite(
        EvalSuiteConfig(
            run_root=run_root,
            review_root=review_root,
            out_root=tmp_path / "eval",
            clips=[clip],
            run_tracknet=True,
            tracknet_repo=tracknet_repo,
            tracknet_file=tracknet_file,
            inpaintnet_file=inpaintnet_file,
        ),
        tracknet_runner=fake_tracknet,
        court_writer=fake_court,
        temporal_writer=fake_temporal,
        fusion_writer=fake_fusion,
    )

    assert seen_fps == [24.0]


def test_eval_suite_generates_localtraj_when_existing_artifact_is_missing(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    review_root = run_root / "ball_click_review_30"
    clip = "clip_a"
    base = run_root / clip / "tracknet_smoke_0000_0010"
    _write_track(base / "ball_track_0000_0010.json")
    _write_track(base / "ball_track_fusion_temporal_vball100.json")
    _write_clicks(review_root / clip / "ball_points.json", clip=clip)
    (base / "input_0000_0010.mp4").write_bytes(b"unused")

    calls: list[str] = []

    def fake_temporal(*, out_path, summary_path, mode, **_kwargs):
        calls.append(f"temporal:{mode}:{Path(out_path).name}")
        _write_track(Path(out_path), visible=True)
        Path(summary_path).write_text(json.dumps({"mode": mode}), encoding="utf-8")
        return {"mode": mode}

    summary = run_ball_tracking_eval_suite(
        EvalSuiteConfig(
            run_root=run_root,
            review_root=review_root,
            out_root=tmp_path / "eval",
            clips=[clip],
            run_tracknet=False,
        ),
        temporal_writer=fake_temporal,
    )

    assert "fusion_temporal_vball100_localtraj" in summary["benchmark"]["aggregate"]
    assert "pbmat_v0_motion_composite" in summary["benchmark"]["aggregate"]
    assert any(call.startswith("temporal:local_trajectory") for call in calls)
