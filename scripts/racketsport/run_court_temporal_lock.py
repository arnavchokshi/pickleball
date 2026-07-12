#!/usr/bin/env python3
"""Emit one standalone ``court_temporal_lock.json`` candidate artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from threed.racketsport.court_temporal_lock import (  # noqa: E402
    TemporalCourtLock,
    TemporalCourtLockConfig,
    build_artifact,
    load_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--court-calibration", required=True, type=Path)
    parser.add_argument("--camera-motion", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--keyframe-interval-seconds", type=float, default=0.75)
    parser.add_argument("--max-coast-frames", type=int, default=30)
    parser.add_argument("--min-measurements", type=int, default=6)
    parser.add_argument("--static-motion-threshold-px", type=float, default=2.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_frames <= 0 or args.max_frames > 300:
        raise SystemExit("--max-frames must be within 1..300 for this bounded candidate CLI")
    output_path = args.output_dir / "court_temporal_lock.json"
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite immutable artifact: {output_path}")

    calibration = load_json(args.court_calibration)
    motion_payload = load_json(args.camera_motion) if args.camera_motion is not None else None
    drift_p95 = (motion_payload or {}).get("summary", {}).get("drift_px_p95")
    static_degenerate = drift_p95 is not None and float(drift_p95) <= args.static_motion_threshold_px
    motion_by_frame = {}
    for raw_frame in (motion_payload or {}).get("frames", []):
        frame = dict(raw_frame)
        if static_degenerate:
            frame["static_degenerate"] = True
        motion_by_frame[int(frame["frame_idx"])] = frame
    camera_motion_mode = (
        "static_degenerate_from_drift_p95"
        if static_degenerate
        else ("provided_per_frame" if motion_payload is not None else "identity_missing_artifact")
    )
    reference_frame_idx = int((motion_payload or {}).get("reference_frame_idx", 0))

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = float(args.fps or (source_fps if source_fps > 0.0 else 30.0))
    config = TemporalCourtLockConfig(
        fps=fps,
        keyframe_interval_seconds=args.keyframe_interval_seconds,
        max_coast_frames=args.max_coast_frames,
        min_measurements=args.min_measurements,
    )
    lock = TemporalCourtLock(calibration, config=config, reference_frame_idx=reference_frame_idx)
    frames: list[dict[str, object]] = []
    try:
        for frame_idx in range(args.max_frames):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(
                lock.step(
                    frame_idx,
                    frame_bgr=frame_bgr,
                    motion=motion_by_frame.get(frame_idx),
                )
            )
    finally:
        capture.release()
    if not frames:
        raise SystemExit(f"video has no readable frames: {args.video}")

    artifact = build_artifact(
        frames=frames,
        video_path=args.video,
        calibration_path=args.court_calibration,
        camera_motion_path=args.camera_motion,
        camera_motion_mode=camera_motion_mode,
        config=config,
        reference_frame_idx=reference_frame_idx,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output_path)
    print(json.dumps({"output": str(output_path), "summary": artifact["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
