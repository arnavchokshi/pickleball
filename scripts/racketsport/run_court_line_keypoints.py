#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.train_court_keypoint_heatmap import (  # noqa: E402
    _error_summary,
    court_corner_keypoint_labels,
)
from threed.racketsport.court_line_keypoints import DetectedCourtKeypoints, detect_court_keypoints_from_image  # noqa: E402


def run_court_line_keypoint_video(
    *,
    video_path: Path,
    out_path: Path,
    overlay_out_path: Path,
    summary_out_path: Path,
    label_corners_path: Path | None = None,
    detect_frame_index: int | None = None,
) -> dict[str, Any]:
    cv2 = _cv2()
    label_row = _load_label_row(label_corners_path) if label_corners_path is not None else None
    resolved_detect_frame_index = (
        int(detect_frame_index)
        if detect_frame_index is not None
        else int(label_row["frame_index"])
        if label_row is not None
        else 0
    )
    if resolved_detect_frame_index < 0:
        raise ValueError("detect_frame_index must be non-negative")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"could not read video metadata: {video_path}")
        if resolved_detect_frame_index >= frame_count:
            raise ValueError("detect_frame_index is outside the video")
        capture.set(cv2.CAP_PROP_POS_FRAMES, resolved_detect_frame_index)
        ok, detect_frame = capture.read()
        if not ok:
            raise ValueError(f"could not read detection frame {resolved_detect_frame_index}: {video_path}")
        detected = detect_court_keypoints_from_image(detect_frame, cv2_module=cv2)
    finally:
        capture.release()

    frames, overlay_frame_count = _write_predictions_and_overlay(
        video_path=video_path,
        overlay_out_path=overlay_out_path,
        detected=detected,
        cv2_module=cv2,
    )
    prediction_payload = {
        "schema_version": 1,
        "artifact_type": "court_line_keypoint_predictions",
        "video": str(video_path),
        "coordinate_space": "source_video_pixels",
        "detect_frame_index": resolved_detect_frame_index,
        "source_size": [width, height],
        "frames": frames,
        "detector": {
            "name": "auto_white_line_near_strip_homography",
            "raw_segment_count": detected.raw_segment_count,
            "merged_line_count": detected.merged_line_count,
            "confidence": detected.confidence,
        },
        "verified": False,
        "not_cal3_verified": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prediction_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metric_summary = _score_predictions(detected, label_row) if label_row is not None else None
    summary = {
        "schema_version": 1,
        "artifact_type": "court_line_keypoint_run_summary",
        "status": "scored" if metric_summary is not None else "ran_unscored",
        "video": str(video_path),
        "prediction_artifact": str(out_path),
        "overlay_artifact": str(overlay_out_path),
        "detect_frame_index": resolved_detect_frame_index,
        "overlay_frame_count": overlay_frame_count,
        "raw_segment_count": detected.raw_segment_count,
        "merged_line_count": detected.merged_line_count,
        "confidence": detected.confidence,
        "gate": {
            "metric": "heldout_median_keypoint_reprojection_px",
            "threshold_px": 5.0,
            "passed": bool(metric_summary is not None and metric_summary["median"] < 5.0),
        },
        "median_keypoint_reprojection_px": metric_summary["median"] if metric_summary is not None else None,
        "p95_keypoint_reprojection_px": metric_summary["p95"] if metric_summary is not None else None,
        "max_keypoint_reprojection_px": metric_summary["max"] if metric_summary is not None else None,
        "keypoint_count": metric_summary["count"] if metric_summary is not None else 0,
    }
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_label_row(label_corners_path: Path | None) -> dict[str, Any] | None:
    if label_corners_path is None:
        return None
    payload = json.loads(label_corners_path.read_text(encoding="utf-8"))
    return court_corner_keypoint_labels(payload, clip_root=label_corners_path.parent.parent)


def _write_predictions_and_overlay(
    *,
    video_path: Path,
    overlay_out_path: Path,
    detected: DetectedCourtKeypoints,
    cv2_module: Any,
) -> tuple[list[dict[str, Any]], int]:
    cv2 = cv2_module
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        overlay_out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(overlay_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open overlay writer: {overlay_out_path}")
        frames: list[dict[str, Any]] = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            _draw_overlay(cv2, frame, detected)
            writer.write(frame)
            frames.append({"frame_index": frame_index, "keypoints": detected.keypoints})
            frame_index += 1
    finally:
        capture.release()
        if "writer" in locals():
            writer.release()
    return frames, len(frames)


def _draw_overlay(cv2: Any, frame: Any, detected: DetectedCourtKeypoints) -> None:
    line_type = getattr(cv2, "LINE_AA", 16)
    for segment in detected.semantic_lines.values():
        p0, p1 = segment
        cv2.line(frame, _point(p0), _point(p1), (0, 255, 255), 2, line_type)
    for keypoint in detected.keypoints.values():
        cv2.circle(frame, _point(keypoint["xy"]), 5, (0, 0, 0), -1, line_type)
        cv2.circle(frame, _point(keypoint["xy"]), 3, (0, 255, 0), -1, line_type)


def _score_predictions(detected: DetectedCourtKeypoints, label_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if label_row is None:
        return None
    errors: list[float] = []
    for name, label_xy in label_row["keypoints"].items():
        prediction = detected.keypoints.get(name)
        if prediction is None:
            continue
        errors.append(math.dist(prediction["xy"], label_xy))
    return _error_summary(errors)


def _point(xy: list[float]) -> tuple[int, int]:
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("court line keypoint runner requires opencv-python") from exc
    return cv2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run automatic court line keypoint detection on a video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlay-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--label-corners", type=Path, default=None)
    parser.add_argument("--detect-frame-index", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        summary = run_court_line_keypoint_video(
            video_path=args.video,
            out_path=args.out,
            overlay_out_path=args.overlay_out,
            summary_out_path=args.summary_out,
            label_corners_path=args.label_corners,
            detect_frame_index=args.detect_frame_index,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
