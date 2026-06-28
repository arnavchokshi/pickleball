from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.racket_candidate_overlay import load_racket_candidates, render_racket_candidate_overlay
from threed.racketsport.schemas import RacketCandidateFrame, RacketCandidatePlayer, RacketCandidates


cv2_available = importlib.util.find_spec("cv2") is not None


def _candidates() -> RacketCandidates:
    return RacketCandidates(
        schema_version=1,
        artifact_type="racketsport_racket_candidates",
        fps=30.0,
        players=[
            RacketCandidatePlayer(
                id=7,
                paddle_dims_in={"length": 16.0, "width": 8.0},
                frames=[
                    RacketCandidateFrame(
                        t=0.0,
                        corners_px=[[20.0, 30.0], [80.0, 32.0], [78.0, 130.0], [18.0, 128.0]],
                        conf=0.91,
                        source="label_bbox:yolo26m_teacher",
                    ),
                    RacketCandidateFrame(
                        t=1.0 / 30.0,
                        corners_px=[[24.0, 34.0], [84.0, 36.0], [82.0, 134.0], [22.0, 132.0]],
                        conf=0.82,
                        source="label_bbox:yolo26m_teacher",
                    ),
                ],
            )
        ],
    )


@pytest.mark.skipif(not cv2_available, reason="opencv-python is required for video overlay rendering")
def test_render_racket_candidate_overlay_writes_video_and_index(tmp_path: Path) -> None:
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

    candidates_path = tmp_path / "racket_candidates.json"
    candidates_path.write_text(_candidates().model_dump_json(), encoding="utf-8")
    candidates = load_racket_candidates(candidates_path)
    out = tmp_path / "racket_candidate_overlay.mp4"

    summary = render_racket_candidate_overlay(video_path=source, candidates=candidates, output_path=out, max_frames=2)

    assert summary["artifact_type"] == "racketsport_racket_candidate_overlay"
    assert summary["status"] == "rendered"
    assert summary["frame_count"] == 2
    assert summary["candidate_frame_count"] == 2
    assert summary["candidate_player_count"] == 1
    assert summary["available_layers"] == ["paddle_candidates"]
    assert out.is_file()
    assert out.stat().st_size > 0
    index = json.loads((tmp_path / "racket_candidate_overlay_index.json").read_text(encoding="utf-8"))
    assert index["overlay_path"] == str(out)


@pytest.mark.skipif(not cv2_available, reason="opencv-python is required for video overlay rendering")
def test_render_racket_candidate_overlay_scales_candidate_coordinates(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    assert writer.isOpened()
    try:
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    finally:
        writer.release()

    summary = render_racket_candidate_overlay(
        video_path=source,
        candidates=_candidates(),
        output_path=tmp_path / "racket_candidate_overlay.mp4",
        max_frames=1,
        candidate_coord_width=160,
        candidate_coord_height=120,
    )

    assert summary["candidate_coord_width"] == 160
    assert summary["candidate_coord_height"] == 120
    assert summary["candidate_coord_scale_x"] == 2.0
    assert summary["candidate_coord_scale_y"] == 2.0


@pytest.mark.skipif(not cv2_available, reason="opencv-python is required for video overlay rendering")
def test_render_racket_candidate_overlay_warns_when_candidates_exceed_video_window(tmp_path: Path) -> None:
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

    candidates = RacketCandidates(
        schema_version=1,
        artifact_type="racketsport_racket_candidates",
        fps=30.0,
        players=[
            RacketCandidatePlayer(
                id=0,
                paddle_dims_in={"length": 16.0, "width": 8.0},
                frames=[
                    RacketCandidateFrame(
                        t=0.0,
                        corners_px=[[20.0, 30.0], [80.0, 32.0], [78.0, 130.0], [18.0, 128.0]],
                        conf=0.91,
                        source="label_bbox:yolo26m_teacher",
                    ),
                    RacketCandidateFrame(
                        t=1.0,
                        corners_px=[[24.0, 34.0], [84.0, 36.0], [82.0, 134.0], [22.0, 132.0]],
                        conf=0.82,
                        source="label_bbox:yolo26m_teacher",
                    ),
                ],
            )
        ],
    )

    summary = render_racket_candidate_overlay(
        video_path=source,
        candidates=candidates,
        output_path=tmp_path / "racket_candidate_overlay.mp4",
    )

    assert summary["candidate_frame_count"] == 2
    assert summary["rendered_candidate_count"] == 1
    assert summary["unrendered_candidate_count"] == 1
    assert summary["source_video_frame_count"] == 2
    assert summary["candidate_frame_index_max"] == 30
    assert summary["warnings"] == ["candidate_frames_outside_video_window"]


@pytest.mark.skipif(not cv2_available or shutil.which("ffmpeg") is None, reason="OpenCV and ffmpeg are required")
def test_render_racket_candidate_overlay_cli_writes_h264_when_requested(tmp_path: Path) -> None:
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

    candidates_path = tmp_path / "racket_candidates.json"
    candidates_path.write_text(_candidates().model_dump_json(), encoding="utf-8")
    raw_out = tmp_path / "racket_candidate_overlay.mp4"
    h264_out = tmp_path / "racket_candidate_overlay_h264.mp4"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_racket_candidate_overlay.py",
            "--video",
            str(source),
            "--racket-candidates",
            str(candidates_path),
            "--out",
            str(raw_out),
            "--h264-out",
            str(h264_out),
            "--candidate-coordinate-width",
            "160",
            "--candidate-coordinate-height",
            "120",
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
    assert summary["candidate_coord_scale_x"] == 2.0
    assert summary["candidate_coord_scale_y"] == 2.0
    assert h264_out.is_file()
    index = json.loads((tmp_path / "racket_candidate_overlay_index.json").read_text(encoding="utf-8"))
    assert index["overlay_path"] == str(h264_out)
