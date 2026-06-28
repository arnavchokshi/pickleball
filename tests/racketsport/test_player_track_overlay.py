from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.player_track_overlay import render_player_track_overlay
from threed.racketsport.schemas import PlayerTrack, TrackFrame, Tracks


cv2_available = importlib.util.find_spec("cv2") is not None


@pytest.mark.skipif(not cv2_available, reason="opencv-python is required for video overlay rendering")
def test_render_player_track_overlay_writes_video(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    finally:
        writer.release()

    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[
                    TrackFrame(t=0.0, bbox=(20.0, 30.0, 80.0, 130.0), world_xy=[-1.0, -2.0], conf=0.91),
                    TrackFrame(t=1.0 / 30.0, bbox=(24.0, 32.0, 84.0, 132.0), world_xy=[-0.9, -1.9], conf=0.92),
                ],
            )
        ],
        rally_spans=[],
    )
    out = tmp_path / "overlay.mp4"

    summary = render_player_track_overlay(video_path=source, tracks=tracks, output_path=out)

    assert summary["status"] == "rendered"
    assert summary["frame_count"] == 2
    assert summary["player_count"] == 1
    assert out.is_file()
    assert out.stat().st_size > 0


@pytest.mark.skipif(not cv2_available, reason="opencv-python is required for video overlay rendering")
def test_render_player_track_overlay_cli_writes_video_and_index(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    finally:
        writer.release()

    tracks_path = tmp_path / "tracks.json"
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[
                    TrackFrame(t=0.0, bbox=(20.0, 30.0, 80.0, 130.0), world_xy=[-1.0, -2.0], conf=0.91),
                    TrackFrame(t=1.0 / 30.0, bbox=(24.0, 32.0, 84.0, 132.0), world_xy=[-0.9, -1.9], conf=0.92),
                ],
            )
        ],
        rally_spans=[],
    )
    tracks_path.write_text(tracks.model_dump_json(), encoding="utf-8")
    out = tmp_path / "player_track_overlay.mp4"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_player_track_overlay.py",
            "--video",
            str(source),
            "--tracks",
            str(tracks_path),
            "--out",
            str(out),
            "--max-frames",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    summary = json.loads(completed.stdout)
    assert summary["artifact_type"] == "racketsport_player_track_overlay"
    assert summary["frame_count"] == 2
    assert out.is_file()
    assert (tmp_path / "player_track_overlay_index.json").is_file()


@pytest.mark.skipif(not cv2_available or shutil.which("ffmpeg") is None, reason="OpenCV and ffmpeg are required")
def test_render_player_track_overlay_cli_writes_h264_when_requested(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    finally:
        writer.release()

    tracks_path = tmp_path / "tracks.json"
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(20.0, 30.0, 80.0, 130.0), world_xy=[-1.0, -2.0], conf=0.91)],
            )
        ],
        rally_spans=[],
    )
    tracks_path.write_text(tracks.model_dump_json(), encoding="utf-8")
    raw_out = tmp_path / "player_track_overlay.mp4"
    h264_out = tmp_path / "player_track_overlay_h264.mp4"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_player_track_overlay.py",
            "--video",
            str(source),
            "--tracks",
            str(tracks_path),
            "--out",
            str(raw_out),
            "--h264-out",
            str(h264_out),
            "--max-frames",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    summary = json.loads(completed.stdout)
    assert summary["overlay_path"] == str(h264_out)
    assert summary["source_overlay_path"] == str(raw_out)
    assert summary["video_codec"] == "h264"
    assert h264_out.is_file()
    index = json.loads((tmp_path / "player_track_overlay_index.json").read_text(encoding="utf-8"))
    assert index["overlay_path"] == str(h264_out)
