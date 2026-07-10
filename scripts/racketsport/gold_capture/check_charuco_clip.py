#!/usr/bin/env python3
"""Check that recorded clips contain enough locked-contract ChArUco detections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport import calibrate_charuco_device  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modern_counts(*, cv2: Any, video: Path, board: Any, max_frames: int, stride: int) -> tuple[list[int], float]:
    detector = cv2.aruco.CharucoDetector(board)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    counts: list[int] = []
    source_frame = 0
    inspected = 0
    while inspected < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if source_frame % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _corners, ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
            counts.append(0 if ids is None else int(len(ids)))
            inspected += 1
        source_frame += 1
    capture.release()
    return counts, fps


def check_videos(
    videos: Sequence[Path], *, max_frames: int, stride: int, min_detection_frames: int, min_corners: int
) -> dict[str, Any]:
    import cv2  # type: ignore[import-not-found]

    if not videos:
        raise ValueError("provide at least one --video")
    if max_frames < 1 or stride < 1 or min_detection_frames < 1 or min_corners < 1:
        raise ValueError("frame and corner thresholds must be positive")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.03, dictionary)
    backend = "repo_legacy_collector"
    results: list[dict[str, Any]] = []
    for video in videos:
        if not video.is_file():
            raise ValueError(f"missing video: {video}")
        if hasattr(cv2.aruco, "detectMarkers"):
            observations = calibrate_charuco_device._collect_observations(
                cv2=cv2,
                videos=[video],
                dictionary=dictionary,
                board=board,
                max_frames_per_video=max_frames,
                min_corners=min_corners,
            )
            counts = [int(len(item.ids)) for item in observations]
            capture = cv2.VideoCapture(str(video))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
        else:
            backend = "opencv5_charuco_detector_compatibility"
            counts, fps = _modern_counts(cv2=cv2, video=video, board=board, max_frames=max_frames, stride=stride)
        detection_frames = sum(count >= min_corners for count in counts)
        results.append(
            {
                "video": video.as_posix(),
                "immutable_raw_reference": {"uri": video.as_posix(), "sha256": _sha256(video), "size_bytes": video.stat().st_size},
                "source_fps": fps,
                "inspected_frames": len(counts),
                "detection_frames": detection_frames,
                "max_charuco_corners": max(counts, default=0),
                "pass": detection_frames >= min_detection_frames,
            }
        )
    gate_pass = all(result["pass"] for result in results)
    return {
        "schema_version": 1,
        "artifact_type": "gold_capture_charuco_clip_check",
        "status": "pass" if gate_pass else "fail",
        "gate_pass": gate_pass,
        "backend": backend,
        "read_only_calibration_tool_imported": "scripts/racketsport/calibrate_charuco_device.py",
        "board_contract": {"squares_x": 5, "squares_y": 7, "square_length_m": 0.04, "marker_length_m": 0.03, "dictionary": "DICT_4X4_50"},
        "thresholds": {"min_detection_frames": min_detection_frames, "min_charuco_corners": min_corners, "max_inspected_frames": max_frames, "stride": stride},
        "videos": results,
        "warnings": [] if backend == "repo_legacy_collector" else ["existing calibration collector uses cv2.aruco.detectMarkers, which is absent in current OpenCV; used CharucoDetector without modifying the read-only tool"],
        "product_boundary": "The product remains monocular; extra cameras, markers, and surveys are GT-only.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-detection-frames", type=int, default=8)
    parser.add_argument("--min-corners", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = check_videos(
            args.video,
            max_frames=args.max_frames,
            stride=args.stride,
            min_detection_frames=args.min_detection_frames,
            min_corners=args.min_corners,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
