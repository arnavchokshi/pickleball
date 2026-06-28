from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SPLITS = {
    "train": (
        "burlington_gold_0300_low_steep_corner",
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "indoor_doubles_fwuks_0500_long_mid_baseline",
    ),
    "val": ("wolverine_mixed_0200_mid_steep_corner",),
    "test": ("wolverine_mixed_0200_mid_steep_corner",),
}


@dataclass(frozen=True)
class TrackNetLabel:
    frame: int
    visibility: int
    x: float
    y: float
    source: str


def interpolated_tracknet_labels(
    items: Iterable[dict[str, Any]],
    *,
    frame_count: int,
    max_gap_frames: int,
) -> list[TrackNetLabel]:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")

    labels = [TrackNetLabel(frame=frame, visibility=0, x=0.0, y=0.0, source="unlabeled_hidden") for frame in range(frame_count)]
    explicit_visible: list[TrackNetLabel] = []
    explicit_hidden_frames: set[int] = set()

    for item in items:
        frame = _frame_index(item)
        if frame < 0 or frame >= frame_count:
            continue
        visible = item.get("visible") is True
        if visible:
            x, y = _xy_px(item)
            label = TrackNetLabel(frame=frame, visibility=1, x=x, y=y, source="human_click")
            labels[frame] = label
            explicit_visible.append(label)
        else:
            labels[frame] = TrackNetLabel(frame=frame, visibility=0, x=0.0, y=0.0, source="human_hidden")
            explicit_hidden_frames.add(frame)

    explicit_visible.sort(key=lambda label: label.frame)
    for left, right in zip(explicit_visible, explicit_visible[1:], strict=False):
        gap = right.frame - left.frame
        if gap <= 1 or gap > max_gap_frames:
            continue
        if any(frame in explicit_hidden_frames for frame in range(left.frame + 1, right.frame)):
            continue
        for frame in range(left.frame + 1, right.frame):
            alpha = (frame - left.frame) / gap
            labels[frame] = TrackNetLabel(
                frame=frame,
                visibility=1,
                x=left.x + (right.x - left.x) * alpha,
                y=left.y + (right.y - left.y) * alpha,
                source="interpolated",
            )

    return labels


def write_tracknet_csv(path: str | Path, labels: Iterable[TrackNetLabel]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frame", "Visibility", "X", "Y"])
        for label in labels:
            writer.writerow([label.frame, label.visibility, f"{label.x:.3f}", f"{label.y:.3f}"])


def _frame_index(item: dict[str, Any]) -> int:
    value = item.get("frame_index")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("TrackNet label item requires integer frame_index")
    return value


def _xy_px(item: dict[str, Any]) -> tuple[float, float]:
    value = item.get("xy_px")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("visible TrackNet label item requires xy_px")
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("xy_px values must be numeric")
    return float(x), float(y)


def build_tracknetv3_dataset(
    *,
    run_root: Path,
    review_root: Path,
    out: Path,
    splits: dict[str, tuple[str, ...]] | None = None,
    max_gap_frames: int = 45,
    overwrite: bool = False,
) -> dict[str, Any]:
    if splits is None:
        splits = DEFAULT_SPLITS
    if overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "tracknetv3_pickleball_finetune_dataset",
        "status": "prepared",
        "run_root": str(run_root),
        "review_root": str(review_root),
        "max_gap_frames": max_gap_frames,
        "splits": {},
    }
    match_index = 1
    for split, clips in splits.items():
        rows: list[dict[str, Any]] = []
        for clip in clips:
            rally_id = f"{match_index}_01_00"
            match_dir = out / split / f"match{match_index}"
            frame_dir = match_dir / "frame" / rally_id
            video_path = run_root / clip / "tracknet_smoke_0000_0010" / "input_0000_0010.mp4"
            review_path = review_root / clip / "ball_points.json"
            if not video_path.is_file():
                raise FileNotFoundError(f"missing TrackNet input video: {video_path}")
            if not review_path.is_file():
                raise FileNotFoundError(f"missing ball review labels: {review_path}")

            frame_count = _video_frame_count(video_path)
            _extract_frames(video_path, frame_dir)
            _write_median(frame_dir, frame_count)
            review = json.loads(review_path.read_text(encoding="utf-8"))
            labels = interpolated_tracknet_labels(
                review.get("items", []),
                frame_count=frame_count,
                max_gap_frames=max_gap_frames,
            )
            csv_dir = match_dir / ("corrected_csv" if split == "test" else "csv")
            csv_path = csv_dir / f"{rally_id}_ball.csv"
            write_tracknet_csv(csv_path, labels)
            rows.append(
                {
                    "clip": clip,
                    "match": f"match{match_index}",
                    "rally_id": rally_id,
                    "video": str(video_path),
                    "frame_dir": str(frame_dir),
                    "csv": str(csv_path),
                    "frame_count": frame_count,
                    "human_click_frames": sum(1 for label in labels if label.source == "human_click"),
                    "interpolated_frames": sum(1 for label in labels if label.source == "interpolated"),
                    "visible_label_frames": sum(label.visibility for label in labels),
                }
            )
            match_index += 1
        manifest["splits"][split] = rows

    manifest_path = out / "pickleball_tracknetv3_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _video_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value.isdigit():
        raise ValueError(f"could not determine frame count for {video_path}: {value!r}")
    return int(value)


def _extract_frames(video_path: Path, frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    if (frame_dir / "0.png").is_file():
        return
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-start_number",
            "0",
            str(frame_dir / "%d.png"),
        ],
        check=True,
    )


def _write_median(frame_dir: Path, frame_count: int) -> None:
    median_path = frame_dir / "median.npz"
    if median_path.is_file():
        return
    import numpy as np
    from PIL import Image

    samples = []
    stride = max(1, frame_count // 64)
    for frame in range(0, frame_count, stride):
        path = frame_dir / f"{frame}.png"
        if path.is_file():
            samples.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32))
    if not samples:
        raise FileNotFoundError(f"no extracted frames found for median: {frame_dir}")
    median = np.median(np.stack(samples, axis=0), axis=0).astype("uint8")
    np.savez(median_path, median=median)


def _parse_splits(args: argparse.Namespace) -> dict[str, tuple[str, ...]]:
    return {
        "train": tuple(args.train_clip) if args.train_clip else DEFAULT_SPLITS["train"],
        "val": tuple(args.val_clip) if args.val_clip else DEFAULT_SPLITS["val"],
        "test": tuple(args.test_clip) if args.test_clip else DEFAULT_SPLITS["test"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a TrackNetV3 fine-tune dataset from pickleball click labels.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-clip", action="append", default=[])
    parser.add_argument("--val-clip", action="append", default=[])
    parser.add_argument("--test-clip", action="append", default=[])
    parser.add_argument("--max-gap-frames", type=int, default=45)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        manifest = build_tracknetv3_dataset(
            run_root=args.run_root,
            review_root=args.review_root,
            out=args.out,
            splits=_parse_splits(args),
            max_gap_frames=args.max_gap_frames,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out), "splits": manifest["splits"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
