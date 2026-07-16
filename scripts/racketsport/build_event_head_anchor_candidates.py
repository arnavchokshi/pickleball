#!/usr/bin/env python3
"""Build review-only typed contact-anchor candidates from an event-head checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Iterator

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import preprocess_rgb, sha256_file
from threed.racketsport.event_head.model import load_checkpoint


def _validated_device(name: str) -> None:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")


def _window_logits(
    model: torch.nn.Module, video: Path, *, image_size: int, window_frames: int,
    stride: int, device: str, max_seconds: float | None,
) -> tuple[Iterator[tuple[int, torch.Tensor]], float, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid FPS {fps}: {video}")
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = source_frames
    if max_seconds is not None:
        frame_limit = min(source_frames, max(1, int(math.ceil(max_seconds * fps))))

    def generate() -> Iterator[tuple[int, torch.Tensor]]:
        buffered: deque[torch.Tensor] = deque(maxlen=window_frames)
        emitted_starts: set[int] = set()
        decoded = 0
        try:
            while decoded < frame_limit:
                ok, bgr = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                buffered.append(preprocess_rgb(rgb, image_size=image_size))
                decoded += 1
                if len(buffered) == window_frames:
                    start = decoded - window_frames
                    if start % stride == 0:
                        frames = torch.stack(tuple(buffered)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            yield start, model(frames)[0].softmax(-1).cpu()
                        emitted_starts.add(start)
            if decoded == 0:
                raise RuntimeError(f"video decoded zero frames: {video}")
            tail_start = max(0, decoded - len(buffered))
            if tail_start not in emitted_starts:
                frames = torch.stack(tuple(buffered)).unsqueeze(0).to(device)
                with torch.no_grad():
                    yield tail_start, model(frames)[0].softmax(-1).cpu()
        finally:
            capture.release()

    return generate(), fps, frame_limit


def _peak_pick_scores(
    scores_by_frame: dict[int, dict[int, float]], *, threshold: float, nms_radius: int,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for class_id, class_name in ((1, "HIT"), (2, "BOUNCE")):
        candidates = [
            (frame, values[class_id])
            for frame, values in scores_by_frame.items()
            if values.get(class_id, 0.0) >= threshold
        ]
        kept: list[tuple[int, float]] = []
        for frame, score in sorted(candidates, key=lambda item: (-item[1], item[0])):
            if all(abs(frame - prior_frame) > nms_radius for prior_frame, _ in kept):
                kept.append((frame, score))
        events.extend({"frame_idx": frame, "class": class_name, "score": score} for frame, score in kept)
    return sorted(events, key=lambda event: (event["frame_idx"], event["class"]))


def build_candidates(
    *, checkpoint: Path, video: Path, out: Path, threshold: float,
    nms_radius_frames: int, device: str, stride: int, max_seconds: float | None,
    video_provenance: str,
) -> dict[str, object]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be in [0,1]")
    if nms_radius_frames < 0 or stride < 1:
        raise ValueError("--nms-radius-frames must be >=0 and --stride must be >=1")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("--max-seconds must be >0")
    _validated_device(device)
    model, payload = load_checkpoint(checkpoint, device=device)
    image_size = int(payload.get("image_size", 224))
    window_frames = int(payload.get("window_frames", 15))
    windows, fps, _ = _window_logits(
        model, video, image_size=image_size, window_frames=window_frames,
        stride=stride, device=device, max_seconds=max_seconds,
    )
    scores_by_frame: dict[int, dict[int, float]] = {}
    for start, probabilities in windows:
        for local_frame in range(probabilities.shape[0]):
            global_frame = start + local_frame
            values = scores_by_frame.setdefault(global_frame, {})
            for class_id in (1, 2):
                values[class_id] = max(values.get(class_id, 0.0), float(probabilities[local_frame, class_id]))
    events = _peak_pick_scores(
        scores_by_frame, threshold=threshold, nms_radius=nms_radius_frames,
    )
    for event in events:
        event["pts_s"] = int(event["frame_idx"]) / fps
        # Reinsert in frozen field order for readable artifacts.
        event.update({"class": event.pop("class"), "score": event.pop("score")})
    counts = {name: sum(event["class"] == name for event in events) for name in ("HIT", "BOUNCE")}
    pretrain_data = payload.get("pretrain_data", payload.get("data_manifest", "unspecified"))
    result = {
        "artifact_type": "event_head_contact_anchor_candidates",
        "schema_version": 1,
        "source_video": {"path": str(video), "sha256": sha256_file(video)},
        "video_provenance": video_provenance,
        "never_training": True, "review_only": True, "verified": False,
        "model": {
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "license_posture": str(payload.get("license_posture", "unspecified")),
            "pretrain_data": str(pretrain_data),
        },
        "config": {
            "threshold": threshold, "nms_radius_frames": nms_radius_frames,
            "stride": stride, "image_size": image_size, "window_frames": window_frames,
            "fps": fps, "pts_convention": "normalized_to_first_video_pts",
        },
        "events": events,
        "counts": counts,
        "honest_limits": [
            "Unpromoted event-head output; candidates require review and are never training labels.",
            "PTS uses decode-order frame_idx/fps normalized to the first video PTS, matching protected-seed evaluation.",
            "Overlapping-window class scores are max-merged before typed radius suppression.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-radius-frames", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--video-provenance", default="unspecified")
    args = parser.parse_args()
    try:
        result = build_candidates(
            checkpoint=args.checkpoint, video=args.video, out=args.out,
            threshold=args.threshold, nms_radius_frames=args.nms_radius_frames,
            device=args.device, stride=args.stride, max_seconds=args.max_seconds,
            video_provenance=args.video_provenance,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(3, f"event-head anchor candidate build failed: {exc}\n")
    print(json.dumps({"out": str(args.out), "counts": result["counts"], "verified": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
