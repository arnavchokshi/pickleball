#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render provisional shot labels onto a review video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True, help="shot classifications JSON.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--marker-window-frames", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        shots_payload = _read_json_object(args.shots, "shots")
        shots = _shot_rows(shots_payload)
        render_overlay(
            video_path=args.video,
            shots=shots,
            out_path=args.out,
            marker_window_frames=args.marker_window_frames,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: shot-label overlay failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "shot_count": len(shots),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def render_overlay(
    *,
    video_path: Path,
    shots: list[dict[str, Any]],
    out_path: Path,
    marker_window_frames: int = 8,
) -> None:
    if marker_window_frames < 0:
        raise ValueError("marker_window_frames must be non-negative")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("video has invalid dimensions")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"could not open output video: {out_path}")

    by_frame = sorted(shots, key=lambda shot: int(shot["frame"]))
    current: dict[str, Any] | None = None
    next_index = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            while next_index < len(by_frame) and int(by_frame[next_index]["frame"]) <= frame_index:
                current = by_frame[next_index]
                next_index += 1

            _draw_header(frame, current=current, next_shot=by_frame[next_index] if next_index < len(by_frame) else None)
            for shot in by_frame:
                if abs(frame_index - int(shot["frame"])) <= marker_window_frames:
                    _draw_contact_marker(frame, shot)
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()


def _draw_header(frame: Any, *, current: Mapping[str, Any] | None, next_shot: Mapping[str, Any] | None) -> None:
    height, width = frame.shape[:2]
    panel_w = min(width - 20, 560)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 92), (12, 18, 28), thickness=-1)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 92), (255, 255, 255), thickness=1)

    current_text = "current: none"
    if current is not None:
        current_text = f"current: {current['type']}  conf={float(current['type_conf']):.2f}  t={float(current['t']):.2f}s"
    next_text = "next: none"
    if next_shot is not None:
        next_text = f"next: {next_shot['type']} @ {float(next_shot['t']):.2f}s"

    cv2.putText(frame, "Shot labels: provisional transfer baseline", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(frame, current_text, (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (70, 220, 255), 2)
    cv2.putText(frame, next_text, (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 220, 230), 1)


def _draw_contact_marker(frame: Any, shot: Mapping[str, Any]) -> None:
    height, width = frame.shape[:2]
    x = width - 220
    y = 44
    label = f"{shot['type']} {float(shot['type_conf']):.2f}"
    cv2.rectangle(frame, (x - 12, y - 28), (width - 12, y + 16), (20, 90, 120), thickness=-1)
    cv2.putText(frame, "CONTACT", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame, label, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _shot_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("shots", [])
    if not isinstance(raw, list):
        raise ValueError("shots must be an array")
    shots: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"shots/{index} must be an object")
        shots.append(
            {
                "id": str(item.get("id", f"shot_{index:04d}")),
                "t": float(item["t"]),
                "frame": int(item["frame"]),
                "type": str(item["type"]),
                "type_conf": float(item["type_conf"]),
            }
        )
    return shots


if __name__ == "__main__":
    raise SystemExit(main())
