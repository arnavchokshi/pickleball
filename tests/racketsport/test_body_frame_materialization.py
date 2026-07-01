from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.body_frame_materialization import materialize_body_frames


def _make_tiny_clip(path: Path, *, rate: int = 10, duration_s: float = 1.0) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=96x64:rate={rate}:duration={duration_s}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        pytest.skip("ffmpeg is not installed")


def _write_execution(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_body_compute_execution",
                "scheduled_frames": [
                    {"frame_idx": 2, "target_player_ids": [1, 2]},
                    {"frame_idx": 5, "target_player_ids": [2]},
                    {"frame_idx": 2, "target_player_ids": [3]},
                ],
                "summary": {"scheduled_frame_count": 2, "scheduled_player_frame_count": 4},
            }
        ),
        encoding="utf-8",
    )


def test_materialize_body_frames_extracts_exact_scheduled_frames(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    execution = tmp_path / "body_compute_execution.json"
    out = tmp_path / "body_frames"
    _make_tiny_clip(video)
    _write_execution(execution)

    summary = materialize_body_frames(video_path=video, execution_path=execution, out_dir=out)

    assert summary["frame_indexes"] == [2, 5]
    assert summary["extracted_frame_count"] == 2
    assert summary["source_video"] == str(video)
    assert (out / "frame_000002.jpg").stat().st_size > 0
    assert (out / "frame_000005.jpg").stat().st_size > 0
    assert json.loads((out / "body_frame_manifest.json").read_text(encoding="utf-8")) == summary


def test_materialize_body_frames_batches_ffmpeg_extraction_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "source.mp4"
    execution = tmp_path / "body_compute_execution.json"
    out = tmp_path / "body_frames"
    video.write_bytes(b"fake-video")
    _write_execution(execution)
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        pattern = Path(command[-1])
        for frame_idx in (2, 5):
            (pattern.parent / f"frame_{frame_idx:06d}.jpg").write_bytes(b"jpg")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    summary = materialize_body_frames(video_path=video, execution_path=execution, out_dir=out)

    assert summary["frame_indexes"] == [2, 5]
    assert summary["extracted_frame_count"] == 2
    assert len(calls) == 1
    command = calls[0]
    assert "-frame_pts" in command
    assert "eq(n\\,2)" in " ".join(command)
    assert "eq(n\\,5)" in " ".join(command)
    assert (out / "frame_000002.jpg").is_file()
    assert (out / "frame_000005.jpg").is_file()


def test_materialize_body_frames_preserves_manifest_order_when_some_outputs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "source.mp4"
    execution = tmp_path / "body_compute_execution.json"
    out = tmp_path / "body_frames"
    video.write_bytes(b"fake-video")
    _write_execution(execution)
    out.mkdir()
    (out / "frame_000005.jpg").write_bytes(b"existing")

    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        pattern = Path(command[-1])
        (pattern.parent / "frame_000002.jpg").write_bytes(b"jpg")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    summary = materialize_body_frames(
        video_path=video,
        execution_path=execution,
        out_dir=out,
        overwrite=False,
    )

    assert summary["frames"] == ["frame_000002.jpg", "frame_000005.jpg"]
    assert json.loads((out / "body_frame_manifest.json").read_text(encoding="utf-8"))["frames"] == [
        "frame_000002.jpg",
        "frame_000005.jpg",
    ]


def test_materialize_body_frames_fails_when_no_frames_are_scheduled(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    execution = tmp_path / "body_compute_execution.json"
    _make_tiny_clip(video)
    execution.write_text(
        json.dumps({"artifact_type": "racketsport_body_compute_execution", "scheduled_frames": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no scheduled BODY frames"):
        materialize_body_frames(video_path=video, execution_path=execution, out_dir=tmp_path / "body_frames")


def test_materialize_body_frames_cli_extracts_scheduled_frames(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    execution = tmp_path / "body_compute_execution.json"
    out = tmp_path / "body_frames"
    _make_tiny_clip(video)
    _write_execution(execution)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/materialize_body_frames.py",
            "--video",
            str(video),
            "--body-compute-execution",
            str(execution),
            "--out-dir",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["frame_indexes"] == [2, 5]
    assert (out / "frame_000002.jpg").is_file()
    assert (out / "frame_000005.jpg").is_file()
