"""Materialize exact BODY runner frames from source video."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


MANIFEST_NAME = "body_frame_manifest.json"


def materialize_body_frames(
    *,
    video_path: str | Path,
    execution_path: str | Path,
    out_dir: str | Path,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Extract scheduled BODY frames into ``frame_XXXXXX.jpg`` files."""

    video = Path(video_path)
    execution = Path(execution_path)
    out = Path(out_dir)
    if not video.is_file():
        raise FileNotFoundError(f"missing source video: {video}")
    if not execution.is_file():
        raise FileNotFoundError(f"missing BODY execution manifest: {execution}")

    payload = json.loads(execution.read_text(encoding="utf-8"))
    frame_indexes = _scheduled_frame_indexes(payload)
    if not frame_indexes:
        raise ValueError("no scheduled BODY frames in execution manifest")

    out.mkdir(parents=True, exist_ok=True)
    missing_indexes: list[int] = []
    for frame_idx in frame_indexes:
        frame_path = out / f"frame_{frame_idx:06d}.jpg"
        if frame_path.exists() and not overwrite:
            continue
        missing_indexes.append(frame_idx)

    if missing_indexes:
        _extract_frames_batch(video, frame_indexes=missing_indexes, out_dir=out)

    extracted: list[str] = []
    for frame_idx in frame_indexes:
        frame_path = out / f"frame_{frame_idx:06d}.jpg"
        if not frame_path.is_file() or frame_path.stat().st_size <= 0:
            raise RuntimeError(f"ffmpeg did not produce BODY frame {frame_idx}: {frame_path}")
        extracted.append(frame_path.name)

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_frame_manifest",
        "source_video": str(video),
        "body_compute_execution": str(execution),
        "out_dir": str(out),
        "frame_indexes": frame_indexes,
        "extracted_frame_count": len(extracted),
        "frames": extracted,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _scheduled_frame_indexes(execution: Mapping[str, Any]) -> list[int]:
    indexes: set[int] = set()
    for frame in execution.get("scheduled_frames", []):
        indexes.add(int(frame["frame_idx"]))
    return sorted(indexes)


def _extract_frames_batch(video: Path, *, frame_indexes: list[int], out_dir: Path) -> None:
    select_expr = _select_expression(frame_indexes)
    # Do not use ``-frame_pts 1`` as the output filename. Some valid MP4 files
    # begin with a non-zero video-stream timestamp, so frame n=0 can have PTS=2
    # in the stream time base. Naming by that PTS silently shifts every BODY
    # JPEG even though the select filter correctly addresses frames by ``n``.
    # Extract an ordered, zero-based temporary sequence and map it back to the
    # requested source-frame identities explicitly.
    with tempfile.TemporaryDirectory(prefix="body_frame_extract_", dir=out_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        output_pattern = temp_dir / "selected_%06d.jpg"
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select='{select_expr}'",
            "-frames:v",
            str(len(frame_indexes)),
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(output_pattern),
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to materialize BODY frames") from exc
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to extract BODY frames: {completed.stderr.strip()}")

        extracted = sorted(temp_dir.glob("selected_*.jpg"))
        if len(extracted) != len(frame_indexes):
            raise RuntimeError(
                "ffmpeg produced the wrong number of BODY frames: "
                f"expected={len(frame_indexes)} actual={len(extracted)}"
            )
        for frame_idx, source_path in zip(frame_indexes, extracted, strict=True):
            source_path.replace(out_dir / f"frame_{frame_idx:06d}.jpg")


def _select_expression(frame_indexes: list[int]) -> str:
    ranges: list[tuple[int, int]] = []
    for frame_idx in sorted(set(frame_indexes)):
        if not ranges or frame_idx != ranges[-1][1] + 1:
            ranges.append((frame_idx, frame_idx))
        else:
            start, _end = ranges[-1]
            ranges[-1] = (start, frame_idx)

    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(f"eq(n\\,{start})")
        else:
            parts.append(f"between(n\\,{start}\\,{end})")
    return "+".join(parts)
