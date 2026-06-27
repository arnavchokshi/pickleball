from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_overlay import render_ball_track_overlay


def _make_video(path: Path, *, frames: int = 6, fps: float = 30.0, size: tuple[int, int] = (80, 60)) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4")
    width, height = size
    for index in range(frames):
        frame = np.full((height, width, 3), 18 + index, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_ball_track(path: Path, frames: list[dict], *, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "fps": fps,
        "source": "tracknet",
        "frames": frames,
        "bounces": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _track_frame(index: int, *, xy: list[float], visible: bool = True, conf: float = 0.9) -> dict:
    return {
        "t": index / 30.0,
        "xy": xy,
        "conf": conf,
        "visible": visible,
    }


def _decoded_frames(path: Path) -> list:
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return frames
            frames.append(frame)
    finally:
        cap.release()


def _region_changed(frame, center: tuple[int, int], *, radius: int = 4) -> bool:
    import numpy as np

    x, y = center
    y0 = max(0, y - radius)
    y1 = min(frame.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(frame.shape[1], x + radius + 1)
    region = frame[y0:y1, x0:x1]
    return bool(np.max(region) - np.min(region) > 20)


def test_render_ball_track_overlay_writes_visible_centers_tail_and_status_labels(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    ball_track = tmp_path / "ball_track.json"
    out = tmp_path / "ball_overlay.mp4"
    _make_video(video, frames=5, fps=30.0)
    _write_ball_track(
        ball_track,
        [
            _track_frame(0, xy=[12.0, 20.0]),
            _track_frame(1, xy=[18.0, 20.0]),
            _track_frame(2, xy=[24.0, 22.0]),
            _track_frame(3, xy=[0.0, 0.0], visible=False, conf=0.0),
        ],
    )

    summary = render_ball_track_overlay(video_path=video, ball_track_path=ball_track, out_path=out, max_frames=5, tail=2)

    frames = _decoded_frames(out)
    assert summary["status"] == "rendered"
    assert summary["artifact_type"] == "racketsport_ball_track_overlay"
    assert summary["frame_count"] == 5
    assert summary["visible_frame_count"] == 3
    assert summary["invisible_frame_count"] == 1
    assert summary["missing_frame_count"] == 1
    assert summary["source_frame_indices"] == [0, 1, 2, 3, 4]
    assert out.stat().st_size > 0
    assert len(frames) == 5
    assert _region_changed(frames[2], (24, 22))
    assert _region_changed(frames[2], (18, 20))
    assert _region_changed(frames[3], (12, 14), radius=10)
    assert _region_changed(frames[4], (12, 14), radius=10)


def test_render_ball_track_overlay_respects_stride_max_frames_and_fps_out(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    ball_track = tmp_path / "ball_track.json"
    out = tmp_path / "ball_overlay.mp4"
    _make_video(video, frames=8, fps=30.0)
    _write_ball_track(ball_track, [_track_frame(index, xy=[10.0 + index, 15.0]) for index in range(8)])

    summary = render_ball_track_overlay(
        video_path=video,
        ball_track_path=ball_track,
        out_path=out,
        max_frames=3,
        stride=2,
        fps_out=12.5,
        tail=1,
    )

    assert summary["frame_count"] == 3
    assert summary["visible_frame_count"] == 3
    assert summary["source_frame_indices"] == [0, 2, 4]
    assert summary["stride"] == 2
    assert summary["fps_out"] == pytest.approx(12.5)
    assert len(_decoded_frames(out)) == 3


def test_render_ball_track_overlay_fails_closed_for_invalid_ball_track_schema(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    ball_track = tmp_path / "ball_track.json"
    _make_video(video, frames=1)
    ball_track.write_text(json.dumps({"schema_version": 1, "fps": 30.0, "source": "tracknet"}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid ball_track schema"):
        render_ball_track_overlay(video_path=video, ball_track_path=ball_track, out_path=tmp_path / "out.mp4")


def test_render_ball_track_overlay_cli_writes_overlay_and_json_summary(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    ball_track = tmp_path / "ball_track.json"
    out = tmp_path / "ball_overlay.mp4"
    _make_video(video, frames=2)
    _write_ball_track(ball_track, [_track_frame(0, xy=[12.0, 20.0]), _track_frame(1, xy=[16.0, 22.0])])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_ball_track_overlay.py",
            "--video",
            str(video),
            "--ball-track",
            str(ball_track),
            "--out",
            str(out),
            "--max-frames",
            "2",
            "--tail",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["status"] == "rendered"
    assert summary["frame_count"] == 2
    assert summary["visible_frame_count"] == 2
    assert out.is_file()


def test_render_ball_track_overlay_cli_fails_cleanly_when_inputs_are_missing(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_ball_track_overlay.py",
            "--video",
            str(tmp_path / "missing.mp4"),
            "--ball-track",
            str(tmp_path / "missing_ball_track.json"),
            "--out",
            str(tmp_path / "ball_overlay.mp4"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "missing video file" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "ball_overlay.mp4").exists()


def test_render_ball_track_overlay_cli_fails_cleanly_for_invalid_ball_track_schema(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    ball_track = tmp_path / "ball_track.json"
    out = tmp_path / "ball_overlay.mp4"
    _make_video(video, frames=1)
    ball_track.write_text(json.dumps({"schema_version": 1, "fps": 30.0, "source": "tracknet"}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_ball_track_overlay.py",
            "--video",
            str(video),
            "--ball-track",
            str(ball_track),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "invalid ball_track schema" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not out.exists()
