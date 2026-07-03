#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.eval_guard import assert_not_training_on_eval_clip  # noqa: E402
from threed.racketsport.person_yolo_dataset import yolo_label_line  # noqa: E402
from threed.racketsport.schemas import PersonGroundTruth, validate_artifact_file  # noqa: E402


DEFAULT_CLIPS = (
    "task_2376761=runs/eval0/prototype_gate_h100_v2/wolverine_mixed_0200_mid_steep_corner/"
    "tracknet_smoke_0000_0010/input_0000_0010.mp4="
    "runs/phase2/iphone_person_tracking_eval/labels/task_2376761/person_ground_truth.json",
    "task_2376765=runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/"
    "tracknet_smoke_0000_0010/input_0000_0010.mp4="
    "runs/phase2/iphone_person_tracking_eval/labels/task_2376765/person_ground_truth.json",
)


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    video_path: Path
    ground_truth_path: Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export person_ground_truth.json clips as a YOLO detection dataset.")
    parser.add_argument("--clip", action="append", default=[], help="clip_id=video.mp4=person_ground_truth.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-mode", choices=("alternating", "by_clip"), default="alternating")
    parser.add_argument("--val-clip", action="append", default=[])
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    try:
        clips = _parse_clips(args.clip or list(DEFAULT_CLIPS), require_files=False)
        # Eval-clip integrity gate (fail closed): this builder writes YOLO
        # person-detector training data directly from clip ground truth, so it
        # counts as training-input creation. See threed/racketsport/eval_guard.py.
        # (DEFAULT_CLIPS itself currently points at Burlington/Wolverine source
        # clips -- this refuses that default until a non-eval clip is supplied.)
        assert_not_training_on_eval_clip(
            (value for clip in clips for value in (clip.clip_id, str(clip.video_path), str(clip.ground_truth_path))),
            allow_internal_val=False,
        )
        summary = export_yolo_dataset(
            clips=clips,
            out_dir=args.out_dir,
            split_mode=args.split_mode,
            val_clips=tuple(args.val_clip),
            val_every=int(args.val_every),
            frame_stride=int(args.frame_stride),
            jpeg_quality=int(args.jpeg_quality),
        )
    except Exception as exc:
        print(f"person YOLO dataset export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_compact_summary(summary), sort_keys=True))
    return 0


def export_yolo_dataset(
    *,
    clips: Sequence[ClipSpec],
    out_dir: Path,
    split_mode: str,
    val_clips: Sequence[str] = (),
    val_every: int = 5,
    frame_stride: int = 1,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for YOLO dataset export") from exc
    if val_every <= 1 and split_mode == "alternating":
        raise ValueError("val_every must be > 1 for alternating split")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    rows: list[dict[str, Any]] = []
    val_clip_set = set(val_clips)
    for clip in clips:
        gt = validate_artifact_file("person_ground_truth", clip.ground_truth_path)
        if not isinstance(gt, PersonGroundTruth):
            raise ValueError(f"ground truth artifact did not parse as PersonGroundTruth: {clip.ground_truth_path}")
        labels_by_frame = {frame.frame_index: [label for label in frame.labels if not label.ignored] for frame in gt.frames}
        cap = cv2.VideoCapture(str(clip.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {clip.video_path}")
        try:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                labels = labels_by_frame.get(frame_index, [])
                if labels and frame_index % frame_stride == 0:
                    height, width = frame.shape[:2]
                    split = _split_for_frame(
                        clip_id=clip.clip_id,
                        frame_index=frame_index,
                        split_mode=split_mode,
                        val_clips=val_clip_set,
                        val_every=val_every,
                    )
                    stem = f"{_safe_token(clip.clip_id)}_{frame_index:06d}"
                    image_path = out_dir / "images" / split / f"{stem}.jpg"
                    label_path = out_dir / "labels" / split / f"{stem}.txt"
                    lines: list[str] = []
                    for label in labels:
                        try:
                            lines.append(yolo_label_line(label, image_width=width, image_height=height))
                        except ValueError:
                            continue
                    if lines:
                        ok_write = cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                        if not ok_write:
                            raise RuntimeError(f"failed to write image: {image_path}")
                        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        image_count += 1
                        label_count += len(lines)
                        rows.append(
                            {
                                "clip_id": clip.clip_id,
                                "frame_index": frame_index,
                                "split": split,
                                "image_path": str(image_path),
                                "label_path": str(label_path),
                                "label_count": len(lines),
                            }
                        )
                frame_index += 1
        finally:
            cap.release()

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_yolo_dataset",
        "out_dir": str(out_dir),
        "data_yaml": str(data_yaml),
        "image_count": image_count,
        "label_count": label_count,
        "split_mode": split_mode,
        "val_clips": sorted(val_clip_set),
        "val_every": val_every,
        "frame_stride": frame_stride,
        "rows": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _split_for_frame(
    *,
    clip_id: str,
    frame_index: int,
    split_mode: str,
    val_clips: set[str],
    val_every: int,
) -> str:
    if split_mode == "alternating":
        return "val" if frame_index % val_every == 0 else "train"
    if split_mode == "by_clip":
        if not val_clips:
            raise ValueError("by_clip split requires at least one --val-clip")
        return "val" if clip_id in val_clips else "train"
    raise ValueError(f"unsupported split mode: {split_mode}")


def _parse_clips(specs: Sequence[str], *, require_files: bool = True) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for spec in specs:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"clip spec must be clip_id=video=ground_truth: {spec}")
        clip_id, video, gt = parts
        clip = ClipSpec(clip_id=clip_id, video_path=Path(video), ground_truth_path=Path(gt))
        if require_files:
            if not clip.video_path.is_file():
                raise FileNotFoundError(f"missing video for {clip_id}: {clip.video_path}")
            if not clip.ground_truth_path.is_file():
                raise FileNotFoundError(f"missing ground truth for {clip_id}: {clip.ground_truth_path}")
        clips.append(clip)
    return clips


def _safe_token(value: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in str(value).strip().lower())
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        raise ValueError("empty token")
    return token


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "rows"}


if __name__ == "__main__":
    raise SystemExit(main())
