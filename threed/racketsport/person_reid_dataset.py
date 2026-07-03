"""Build labeled person ReID crop datasets from reviewed CVAT player boxes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .eval_guard import assert_not_training_on_eval_clip
from .schemas import PersonGroundTruth, PersonLabel, validate_artifact_file


@dataclass(frozen=True)
class PersonReIDClipSpec:
    clip_id: str
    video_path: Path
    ground_truth_path: Path


@dataclass(frozen=True)
class PersonReIDDatasetConfig:
    split_mode: str = "by_clip"
    val_clips: tuple[str, ...] = ()
    frame_stride: int = 1
    query_every: int = 10
    crop_padding_px: int = 12
    jpeg_quality: int = 95
    min_crop_width_px: int = 8
    min_crop_height_px: int = 16
    max_samples_per_identity: int | None = None

    def __post_init__(self) -> None:
        if self.split_mode not in {"by_clip"}:
            raise ValueError("split_mode must be by_clip")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.query_every <= 1:
            raise ValueError("query_every must be greater than 1")
        if self.crop_padding_px < 0:
            raise ValueError("crop_padding_px must be non-negative")
        if self.jpeg_quality < 1 or self.jpeg_quality > 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.min_crop_width_px <= 0 or self.min_crop_height_px <= 0:
            raise ValueError("minimum crop dimensions must be positive")
        if self.max_samples_per_identity is not None and self.max_samples_per_identity <= 0:
            raise ValueError("max_samples_per_identity must be positive when provided")
        if self.split_mode == "by_clip" and not self.val_clips:
            raise ValueError("by_clip ReID split requires at least one val clip")


def export_person_reid_crop_dataset(
    *,
    clips: Sequence[PersonReIDClipSpec],
    out_dir: str | Path,
    config: PersonReIDDatasetConfig,
) -> dict[str, Any]:
    """Crop reviewed player boxes into train/query/gallery ReID folders."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for person ReID crop dataset export") from exc

    if not clips:
        raise ValueError("at least one clip is required")

    val_clip_set = set(config.val_clips)
    clip_ids = {clip.clip_id for clip in clips}
    missing_val = sorted(val_clip_set - clip_ids)
    if missing_val:
        raise ValueError(f"val clips are not present in clip specs: {missing_val}")

    # W0 eval-clip guard: this MUST run before any output directory or crop file is
    # materialized (see review finding F3, 2026-07-02). Any clip not in val_clips lands
    # in the "train" split -- i.e. it is actual training data and is refused
    # unconditionally, with no override, exactly like `train_person_osnet_reid.py`'s
    # post-hoc `_assert_manifest_clips_are_not_protected` guard on the already-built
    # manifest. Clips in val_clips only ever contribute query/gallery crops (a
    # validation-during-fitting retrieval signal), so Burlington/Wolverine may be used
    # there via `allow_internal_val=True`; Outdoor/Indoor are strict holdouts and are
    # refused either way (see `threed/racketsport/eval_guard.py`).
    train_clip_ids = sorted(clip_id for clip_id in clip_ids if clip_id not in val_clip_set)
    val_only_clip_ids = sorted(clip_id for clip_id in clip_ids if clip_id in val_clip_set)
    assert_not_training_on_eval_clip(train_clip_ids, allow_internal_val=False)
    eval_guard_summary = assert_not_training_on_eval_clip(val_only_clip_ids, allow_internal_val=True)

    out = Path(out_dir)
    images_dir = out / "images"
    for split in ("train", "query", "gallery"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    pid_by_identity: dict[str, int] = {}
    query_counts: dict[str, int] = {}
    split_identity_counts: dict[tuple[str, str], int] = {}

    for clip in clips:
        gt = validate_artifact_file("person_ground_truth", clip.ground_truth_path)
        if not isinstance(gt, PersonGroundTruth):
            raise ValueError(f"ground truth artifact did not parse as PersonGroundTruth: {clip.ground_truth_path}")
        if gt.clip_id != clip.clip_id:
            raise ValueError(f"clip id mismatch: spec={clip.clip_id} gt={gt.clip_id}")

        cap = cv2.VideoCapture(str(clip.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {clip.video_path}")
        try:
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            if video_width <= 0 or video_height <= 0:
                raise ValueError(f"could not read video dimensions: {clip.video_path}")

            labels_by_frame = {
                frame.frame_index: frame
                for frame in sorted(gt.frames, key=lambda item: item.frame_index)
                if frame.frame_index % config.frame_stride == 0
                and any(not label.ignored and label.person_class for label in frame.labels)
            }
            if not labels_by_frame:
                continue

            last_labeled_frame = max(labels_by_frame)
            frame_index = 0
            while frame_index <= last_labeled_frame:
                ok, image = cap.read()
                if not ok:
                    warnings.append(f"{clip.clip_id}: could not read frame {frame_index}")
                    break
                frame = labels_by_frame.get(frame_index)
                if frame is None:
                    frame_index += 1
                    continue

                visible_labels = [label for label in frame.labels if not label.ignored and label.person_class]
                for label_index, label in enumerate(visible_labels):
                    identity_key = f"{clip.clip_id}:player_{label.track_id}"
                    pid = pid_by_identity.setdefault(identity_key, len(pid_by_identity))
                    split = _split_for_label(
                        clip_id=clip.clip_id,
                        identity_key=identity_key,
                        val_clip_set=val_clip_set,
                        query_every=config.query_every,
                        query_counts=query_counts,
                    )
                    if _sample_limit_reached(
                        split=split,
                        identity_key=identity_key,
                        counts=split_identity_counts,
                        max_samples=config.max_samples_per_identity,
                    ):
                        continue

                    crop_xyxy = _clamped_crop_xyxy(
                        label=label,
                        image_width=video_width,
                        image_height=video_height,
                        padding_px=config.crop_padding_px,
                    )
                    x1, y1, x2, y2 = crop_xyxy
                    if (x2 - x1) < config.min_crop_width_px or (y2 - y1) < config.min_crop_height_px:
                        warnings.append(f"{clip.clip_id}: skipped tiny crop track={label.track_id} frame={frame.frame_index}")
                        continue

                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        warnings.append(f"{clip.clip_id}: skipped empty crop track={label.track_id} frame={frame.frame_index}")
                        continue

                    identity_dir = images_dir / split / _safe_token(identity_key)
                    identity_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"{_safe_token(clip.clip_id)}_pid{pid:04d}_t{label.track_id:02d}_f{frame.frame_index:06d}_{label_index:02d}"
                    image_path = identity_dir / f"{stem}.jpg"
                    ok_write = cv2.imwrite(
                        str(image_path),
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_quality)],
                    )
                    if not ok_write:
                        raise RuntimeError(f"failed to write ReID crop: {image_path}")

                    split_identity_counts[(split, identity_key)] = split_identity_counts.get((split, identity_key), 0) + 1
                    camid = 0 if split != "gallery" else 1
                    rows.append(
                        {
                            "split": split,
                            "clip_id": clip.clip_id,
                            "frame_index": frame.frame_index,
                            "source_frame_id": frame.source_frame_id,
                            "track_id": label.track_id,
                            "identity_key": identity_key,
                            "pid": pid,
                            "camid": camid,
                            "image_path": str(image_path),
                            "relative_image_path": str(image_path.relative_to(out)),
                            "source_video": str(clip.video_path),
                            "ground_truth_path": str(clip.ground_truth_path),
                            "bbox_xywh": [round(float(value), 3) for value in label.bbox_xywh],
                            "crop_xyxy": [int(value) for value in crop_xyxy],
                            "label_index": label_index,
                        }
                    )
                frame_index += 1
        finally:
            cap.release()

    split_counts = _split_counts(rows)
    identity_counts = _identity_counts(rows)
    clip_counts = _clip_counts(rows)
    for split in ("train", "query", "gallery"):
        if split_counts.get(split, 0) == 0:
            warnings.append(f"split has no crops: {split}")

    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_reid_crop_dataset",
        "status": "label_based_reid_crop_dataset",
        "uses_cvat_labels": True,
        "source_only": False,
        "promote_trk": False,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "out_dir": str(out),
        "image_root": str(images_dir),
        "config": asdict(config),
        "clip_count": len(clips),
        "identity_count": len(pid_by_identity),
        "crop_count": len(rows),
        "split_counts": split_counts,
        "identity_counts": identity_counts,
        "clip_counts": clip_counts,
        "identity_pid_map": dict(sorted(pid_by_identity.items(), key=lambda item: item[1])),
        "eval_guard": eval_guard_summary,
        "warnings": warnings,
        "rows": rows,
        "notes": [
            "This dataset uses reviewed CVAT player IDs and is valid for ReID training/evaluation only.",
            "It is not a source-only TRK promotion artifact.",
            "For holdout scoring, train clips and val clips must be reported separately.",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def clip_specs_from_import_manifest(path: str | Path) -> list[PersonReIDClipSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = payload.get("clips")
    if not isinstance(clips, list):
        raise ValueError("CVAT import manifest must contain a clips list")
    specs: list[PersonReIDClipSpec] = []
    for row in clips:
        if not isinstance(row, Mapping):
            raise ValueError("manifest clip entries must be objects")
        specs.append(
            PersonReIDClipSpec(
                clip_id=str(row["clip_id"]),
                video_path=Path(str(row["source_video"])),
                ground_truth_path=Path(str(row["person_ground_truth"])),
            )
        )
    return specs


def parse_reid_clip_specs(specs: Sequence[str]) -> list[PersonReIDClipSpec]:
    clips: list[PersonReIDClipSpec] = []
    for spec in specs:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"clip spec must be clip_id=video=person_ground_truth: {spec}")
        clip_id, video, gt = parts
        clips.append(PersonReIDClipSpec(clip_id=clip_id, video_path=Path(video), ground_truth_path=Path(gt)))
    return clips


def _split_for_label(
    *,
    clip_id: str,
    identity_key: str,
    val_clip_set: set[str],
    query_every: int,
    query_counts: dict[str, int],
) -> str:
    if clip_id not in val_clip_set:
        return "train"
    seen = query_counts.get(identity_key, 0)
    query_counts[identity_key] = seen + 1
    return "query" if seen % query_every == 0 else "gallery"


def _sample_limit_reached(
    *,
    split: str,
    identity_key: str,
    counts: dict[tuple[str, str], int],
    max_samples: int | None,
) -> bool:
    if max_samples is None:
        return False
    return counts.get((split, identity_key), 0) >= max_samples


def _clamped_crop_xyxy(
    *,
    label: PersonLabel,
    image_width: int,
    image_height: int,
    padding_px: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = [float(value) for value in label.bbox_xywh]
    x1 = max(0, int(x) - padding_px)
    y1 = max(0, int(y) - padding_px)
    x2 = min(image_width, int(x + width + 0.999) + padding_px)
    y2 = min(image_height, int(y + height + 0.999) + padding_px)
    return (x1, y1, x2, y2)


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "query": 0, "gallery": 0}
    for row in rows:
        split = str(row["split"])
        counts[split] = counts.get(split, 0) + 1
    return counts


def _identity_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        identity = str(row["identity_key"])
        split = str(row["split"])
        counts.setdefault(identity, {"train": 0, "query": 0, "gallery": 0})
        counts[identity][split] = counts[identity].get(split, 0) + 1
    return dict(sorted(counts.items()))


def _clip_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        clip = str(row["clip_id"])
        split = str(row["split"])
        counts.setdefault(clip, {"train": 0, "query": 0, "gallery": 0})
        counts[clip][split] = counts[clip].get(split, 0) + 1
    return dict(sorted(counts.items()))


def _safe_token(value: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in str(value).strip().lower())
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        raise ValueError("empty token")
    return token


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
