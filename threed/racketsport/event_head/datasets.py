"""On-the-fly event datasets with explicit supervision and license masks.

License posture is deliberately attached at each loader boundary.  jhong93
labels/code are BSD-3 but broadcast pixels make trained weights RD_ONLY;
OpenTTGames/Extended OpenTTGames are CC BY-NC-SA and RD_ONLY_STRICT;
ShuttleSet labels are MIT but media is absent and broadcast rights unresolved.
No loader in this module reads ``data/event_bootstrap_20260713``.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

BACKGROUND, HIT, BOUNCE = 0, 1, 2
CLASS_NAMES = {BACKGROUND: "background", HIT: "HIT", BOUNCE: "BOUNCE"}
IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
EXPECTED_UNIVERSE = {"jhong93_spot": 33_791, "openttgames": 4_271, "shuttleset": 36_484}
DEFAULT_WINDOW_STRIDE = 32
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
_CLIP_RE = re.compile(r"^(?P<base>.+)_(?P<start>\d+)_(?P<end>\d+)$")
CURRENT_MANIFEST_CLASSES = {"0": "background", "1": "HIT", "2": "BOUNCE"}
CURRENT_ROW_FIELDS = frozenset({
    "source", "source_video", "video_path", "media_present", "split", "fps",
    "source_start_frame", "num_frames", "events", "loss_validity_mask",
    "license_posture",
})


class DatasetFormatError(ValueError):
    """Raised when public or reviewed event data violates its contract."""


@dataclass(frozen=True)
class WindowSpec:
    video_path: Path
    start_frame: int
    num_frames: int
    fps: float
    events: tuple[tuple[int, int], ...]
    validity_mask: tuple[bool, bool, bool]
    source: str
    license_posture: str
    unknown_frame_mask: tuple[bool, ...] = ()
    event_subframe_offsets: tuple[float, ...] = ()
    sample_weight: float = 1.0
    teacher_derived: bool = False
    row_index: int = -1
    source_video: str = ""


def validate_current_manifest(manifest: dict[str, Any]) -> None:
    """Validate the shared event-head manifest/row contract without decoding media."""

    artifact_type = str(manifest.get("artifact_type", ""))
    schema_version = manifest.get("schema_version")
    if (
        schema_version not in {1, 2}
        or not artifact_type.startswith("event_head_")
        or not artifact_type.endswith("dataset_manifest")
        or manifest.get("classes") != CURRENT_MANIFEST_CLASSES
        or not isinstance(manifest.get("rows"), list)
    ):
        raise DatasetFormatError("manifest does not use the current event-head dataset schema")
    for index, row in enumerate(manifest["rows"]):
        if not isinstance(row, dict):
            raise DatasetFormatError(f"row {index} must be an object")
        missing = sorted(CURRENT_ROW_FIELDS - set(row))
        if missing:
            raise DatasetFormatError(f"row {index} is missing current-schema fields: {missing}")
        if not isinstance(row["media_present"], bool):
            raise DatasetFormatError(f"row {index} media_present must be boolean")
        if row["split"] not in {"train", "val", "test"}:
            raise DatasetFormatError(f"row {index} has invalid split {row['split']!r}")
        mask = row["loss_validity_mask"]
        if (
            not isinstance(mask, list) or len(mask) != len(CLASS_NAMES)
            or not all(isinstance(value, bool) for value in mask) or not mask[BACKGROUND]
        ):
            raise DatasetFormatError(f"row {index} has invalid loss_validity_mask")
        if not isinstance(row["events"], list):
            raise DatasetFormatError(f"row {index} events must be an array")
        if not row["media_present"]:
            if row["video_path"] is not None:
                raise DatasetFormatError(f"row {index} absent media must have video_path=null")
            # Schema-v1 public inventories historically carry absent-media rows
            # with fps/num_frames=null. Preserve that contract while requiring
            # schema v2 to be fully frame-addressable for its per-frame mask.
            if schema_version == 1:
                continue
        elif not isinstance(row["video_path"], str) or not row["video_path"]:
            raise DatasetFormatError(f"row {index} media-present video_path must be nonempty")
        try:
            fps = float(row["fps"])
            source_start_frame = int(row["source_start_frame"])
            num_frames = int(row["num_frames"])
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(f"row {index} has invalid frame metadata") from exc
        if not np.isfinite(fps) or fps <= 0 or source_start_frame < 0 or num_frames < 1:
            raise DatasetFormatError(f"row {index} has invalid frame metadata")
        unknown = row.get("unknown_frame_mask")
        if schema_version == 2 and unknown is None:
            raise DatasetFormatError(f"row {index} schema v2 requires unknown_frame_mask")
        if unknown is not None and (
            not isinstance(unknown, list)
            or len(unknown) != num_frames
            or not all(isinstance(value, bool) for value in unknown)
        ):
            raise DatasetFormatError(
                f"row {index} unknown_frame_mask must be one bool per source frame"
            )
        seen: set[tuple[int, str]] = set()
        for event_index, event in enumerate(row["events"]):
            if not isinstance(event, dict) or event.get("class") not in {"HIT", "BOUNCE"}:
                raise DatasetFormatError(
                    f"row {index} event {event_index} must be HIT or BOUNCE"
                )
            frame = event.get("frame")
            if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < num_frames:
                raise DatasetFormatError(
                    f"row {index} event {event_index} frame is outside num_frames"
                )
            key = (frame, str(event["class"]))
            if key in seen:
                raise DatasetFormatError(f"row {index} contains duplicate event {key}")
            seen.add(key)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_jhong_clip_name(name: str) -> tuple[str, int, int]:
    """Return parent name and absolute [start,end) frames used by E2E-Spot.

    This mirrors ``third_party/spot/frames_as_jpg.py:get_tennis_tasks``: clip
    frame zero is decoded from source-video ``start`` and event labels remain
    clip-local.  The arithmetic check catches naming/provenance drift.
    """

    match = _CLIP_RE.match(name)
    if not match:
        raise DatasetFormatError(f"invalid jhong93 clip name: {name}")
    start, end = int(match.group("start")), int(match.group("end"))
    if end <= start:
        raise DatasetFormatError(f"invalid jhong93 frame range: {name}")
    return match.group("base"), start, end


def preprocess_rgb(frame_rgb: np.ndarray, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Deterministic train/eval shared preprocessing (RGB, short-side square)."""

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise DatasetFormatError(f"expected HxWx3 RGB frame, got {frame_rgb.shape}")
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div_(255.0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def decode_video_frames(
    video_path: Path, frame_indices: Sequence[int], *, image_size: int = IMAGE_SIZE
) -> torch.Tensor:
    """Decode requested frames without materializing or caching frame files."""

    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not frame_indices:
        raise DatasetFormatError("frame_indices must not be empty")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise DatasetFormatError(f"could not open video: {video_path}")
    frames: list[torch.Tensor] = []
    try:
        previous = -2
        for frame_index in frame_indices:
            if frame_index < 0:
                frame_index = 0
            if frame_index != previous + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok:
                raise DatasetFormatError(f"decode failed at frame {frame_index}: {video_path}")
            frames.append(preprocess_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_size))
            previous = frame_index
    finally:
        capture.release()
    return torch.stack(frames)


def event_subframe_offset_frames(event: dict[str, Any], *, fps: float) -> float:
    """Resolve an optional sub-frame label residual without inventing precision.

    Current owner labels are frame-addressed and therefore resolve to ``0``.
    Agreement-corpus rows preserve both the teacher timestamp and the selected
    encoded-frame PTS; their difference is a real, bounded fractional-frame
    target for the additive offset head.
    """

    for key in ("subframe_offset_frames", "frame_offset"):
        if key in event:
            value = float(event[key])
            if not math.isfinite(value) or abs(value) > 0.5:
                raise DatasetFormatError(f"{key} must be finite and within +/-0.5 frame")
            return value
    if "teacher_timestamp_s" in event and "source_pts_s" in event:
        value = (float(event["teacher_timestamp_s"]) - float(event["source_pts_s"])) * fps
        if not math.isfinite(value) or abs(value) > 0.5 + 1e-6:
            raise DatasetFormatError(
                "teacher/source PTS residual must be finite and within +/-0.5 frame"
            )
        return max(-0.5, min(0.5, value))
    return 0.0


def window_target_tensors(
    spec: WindowSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build hard labels, loss mask, and sub-frame offsets without decoding RGB."""

    if spec.unknown_frame_mask and len(spec.unknown_frame_mask) != spec.num_frames:
        raise DatasetFormatError("window unknown_frame_mask length mismatch")
    if spec.event_subframe_offsets and len(spec.event_subframe_offsets) != len(spec.events):
        raise DatasetFormatError("event_subframe_offsets must align with events")
    targets = torch.zeros(spec.num_frames, dtype=torch.long)
    subframe_offsets = torch.zeros(spec.num_frames, dtype=torch.float32)
    occupied: dict[int, int] = {}
    offsets = spec.event_subframe_offsets or (0.0,) * len(spec.events)
    for (local_frame, class_id), offset in zip(spec.events, offsets, strict=True):
        if not 0 <= local_frame < spec.num_frames:
            continue
        prior = occupied.get(local_frame)
        if prior is not None and prior != class_id:
            raise DatasetFormatError(
                "per-frame CE cannot encode two event classes at the same frame"
            )
        occupied[local_frame] = class_id
        targets[local_frame] = class_id
        subframe_offsets[local_frame] = float(offset)
    frame_loss_mask = ~torch.tensor(
        spec.unknown_frame_mask or (False,) * spec.num_frames,
        dtype=torch.bool,
    )
    return targets, frame_loss_mask, subframe_offsets


def dense_class_counts(
    windows: Sequence[WindowSpec],
    *,
    label_dilation_frames: int,
    neighbor_positive_weight: float,
) -> tuple[float, float, float]:
    """Count the actual loss-eligible dense target mass for a window set."""

    from .assignment import build_soft_dense_targets

    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    for spec in windows:
        targets, frame_loss_mask, _ = window_target_tensors(spec)
        validity = torch.tensor(spec.validity_mask, dtype=torch.bool)
        dense = build_soft_dense_targets(
            targets.unsqueeze(0),
            validity.unsqueeze(0),
            frame_loss_mask.unsqueeze(0),
            label_dilation_frames=label_dilation_frames,
            neighbor_positive_weight=neighbor_positive_weight,
        )[0]
        counts += dense[frame_loss_mask].to(torch.float64).sum(0)
    values = tuple(float(value) for value in counts.tolist())
    if len(values) != 3:
        raise AssertionError("event head must have exactly three classes")
    return values  # type: ignore[return-value]


def sqrt_frequency_class_weights(
    class_counts: Sequence[float],
) -> tuple[float, float, float]:
    """Return sqrt-inverse-frequency weights normalized to background=1.

    Normalizing the common scale leaves the weighted CE objective unchanged:
    ``w_c = sqrt(n_background / n_c)``.  A missing class fails closed rather
    than turning into an infinite rare-class weight.
    """

    if len(class_counts) != len(CLASS_NAMES):
        raise DatasetFormatError("class_counts must contain background, HIT, BOUNCE")
    counts = tuple(float(value) for value in class_counts)
    if any(not math.isfinite(value) or value <= 0 for value in counts):
        raise DatasetFormatError("sqrt-frequency weighting requires every class count > 0")
    background = counts[BACKGROUND]
    values = tuple(math.sqrt(background / value) for value in counts)
    return values  # type: ignore[return-value]


def deterministic_source_video_holdout(
    manifest: dict[str, Any], *, seed: int, holdout_source_count: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Create a source-video-disjoint internal split from an all-train corpus."""

    if isinstance(seed, bool) or seed < 0 or holdout_source_count < 1:
        raise DatasetFormatError("seed must be nonnegative and holdout_source_count positive")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise DatasetFormatError("manifest has no rows to split")
    if any(row.get("split") != "train" for row in rows):
        raise DatasetFormatError("internal split requires an all-train input manifest")
    sources = sorted({str(row.get("source_video", "")) for row in rows})
    if "" in sources or holdout_source_count >= len(sources):
        raise DatasetFormatError("internal split must leave at least one training source")
    ordered = sorted(
        sources,
        key=lambda source: (
            hashlib.sha256(f"{seed}:{source}".encode()).hexdigest(), source,
        ),
    )
    held_out = tuple(ordered[:holdout_source_count])
    output = copy.deepcopy(manifest)
    for row in output["rows"]:
        row["split"] = "val" if str(row["source_video"]) in held_out else "train"
    output.setdefault("config", {})["internal_validation"] = {
        "policy": "sha256_seeded_source_video_holdout",
        "seed": seed,
        "holdout_source_count": holdout_source_count,
        "held_out_source_videos": list(held_out),
    }
    return output, held_out


class EventWindowDataset(Dataset[dict[str, Any]]):
    """Small window dataset that decodes source video on demand."""

    def __init__(self, windows: Sequence[WindowSpec], *, image_size: int = IMAGE_SIZE) -> None:
        if not windows:
            raise DatasetFormatError("at least one media-present window is required")
        self.windows = tuple(windows)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.windows[index]
        absolute = range(spec.start_frame, spec.start_frame + spec.num_frames)
        frames = decode_video_frames(spec.video_path, list(absolute), image_size=self.image_size)
        targets, frame_loss_mask, event_subframe_offsets = window_target_tensors(spec)
        return {
            "frames": frames,
            "targets": targets,
            "validity_mask": torch.tensor(spec.validity_mask, dtype=torch.bool),
            "frame_loss_mask": frame_loss_mask,
            "event_subframe_offsets": event_subframe_offsets,
            "sample_weight": torch.tensor(spec.sample_weight, dtype=torch.float32),
            "teacher_derived": torch.tensor(spec.teacher_derived, dtype=torch.bool),
            "row_index": torch.tensor(spec.row_index, dtype=torch.long),
            "source": spec.source,
            "source_video": spec.source_video,
            "license_posture": spec.license_posture,
        }


def _find_pilot_video(pilot_dir: Path, base: str) -> Path | None:
    matches = sorted(
        p for p in pilot_dir.glob(f"{base}.*")
        if p.is_file() and not p.name.startswith("._")
    )
    return matches[0] if matches else None


def _split_counts(group_count: int) -> dict[str, int]:
    """Allocate exact parent counts with deterministic largest remainders."""

    raw = {split: group_count * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = group_count - sum(counts.values())
    split_order = list(SPLIT_RATIOS)
    order = sorted(
        SPLIT_RATIOS,
        key=lambda split: (-(raw[split] - counts[split]), split_order.index(split)),
    )
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def _rebalance_parent_splits(rows: list[dict[str, Any]], *, seed: int) -> None:
    """Assign approximately 70/15/15 splits without splitting a parent video."""

    for source in sorted({str(row["source"]) for row in rows}):
        parents = sorted({str(row["source_video"]) for row in rows if row["source"] == source})
        parents.sort(
            key=lambda parent: (
                hashlib.sha256(f"{seed}:{parent}".encode()).hexdigest(),
                parent,
            )
        )
        counts = _split_counts(len(parents))
        assignment: dict[str, str] = {}
        offset = 0
        for split in ("train", "val", "test"):
            for parent in parents[offset:offset + counts[split]]:
                assignment[parent] = split
            offset += counts[split]
        for row in rows:
            if row["source"] == source:
                row["split"] = assignment[str(row["source_video"])]


def load_jhong_rows(root: Path, *, seed: int) -> list[dict[str, Any]]:
    """Load BSD-3 labels; broadcast pixels force RD_ONLY checkpoint posture."""

    label_root = root / "jhong93_spot" / "data" / "tennis"
    pilot_dir = root / "jhong93_spot" / "videos_pilot"
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for clip in json.loads((label_root / f"{split}.json").read_text()):
            base, start, end = parse_jhong_clip_name(clip["video"])
            if end - start != int(clip["num_frames"]):
                raise DatasetFormatError(f"clip range mismatch: {clip['video']}")
            counts = {"HIT": 0, "BOUNCE": 0, "background": 0}
            events: list[dict[str, Any]] = []
            for event in clip["events"]:
                class_name = "BOUNCE" if event["label"].endswith("_bounce") else "HIT"
                counts[class_name] += 1
                events.append({"frame": int(event["frame"]), "class": class_name})
            media = _find_pilot_video(pilot_dir, base)
            rows.append({
                "source": "jhong93_spot",
                "video": clip["video"],
                "source_video": base,
                "video_path": str(media) if media else None,
                "media_present": media is not None,
                "split": split,
                "canonical_split": split,
                "fps": float(clip["fps"]),
                "source_start_frame": start,
                "num_frames": int(clip["num_frames"]),
                "event_counts": counts,
                "inventory_event_count": len(events),
                "events": events,
                "loss_validity_mask": [True, True, True],
                "license_id": "BSD-3-Clause-labels; broadcast-pixels-uncleared",
                "license_posture": "RD_ONLY",
            })
    return rows


def _video_metadata(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def load_opentt_rows(root: Path) -> list[dict[str, Any]]:
    """Load CC BY-NC-SA base BOUNCE labels plus Extended CONTACT/HIT labels."""

    base = root / "openttgames"
    extended = root / "extended_openttgames" / "data" / "raw" / "game_data"
    rows: list[dict[str, Any]] = []
    for event_path in sorted((base / "markup" / "extracted").glob("*/events_markup.json")):
        if any(part.startswith("._") for part in event_path.parts):
            continue
        name = event_path.parent.name
        split = "train" if name.startswith("game_") else "test"
        video_path = base / "videos" / f"{name}.mp4"
        raw = json.loads(event_path.read_text())
        ext_path = extended / split / f"{name}.json"
        ext = json.loads(ext_path.read_text()) if ext_path.is_file() else {}
        hits = sorted(int(frame) for frame, label in ext.items() if "forehand" in label or "backhand" in label)
        bounces = sorted(int(frame) for frame, label in raw.items() if label == "bounce")
        background_annotations = len(raw) - len(bounces)
        if video_path.is_file():
            fps, num_frames = _video_metadata(video_path)
        else:
            fps, num_frames = 120.0, 0
        rows.append({
            "source": "openttgames",
            "video": name,
            "source_video": name,
            "video_path": str(video_path) if video_path.is_file() else None,
            "media_present": video_path.is_file(),
            "split": split,
            "fps": fps,
            "source_start_frame": 0,
            "num_frames": num_frames,
            "event_counts": {"HIT": len(hits), "BOUNCE": len(bounces), "background": background_annotations},
            "inventory_event_count": len(raw),
            "events": sorted(
                ([{"frame": frame, "class": "HIT"} for frame in hits]
                 + [{"frame": frame, "class": "BOUNCE"} for frame in bounces]),
                key=lambda item: (item["frame"], item["class"]),
            ),
            "loss_validity_mask": [True, True, True],
            "license_id": "CC-BY-NC-SA-4.0",
            "license_posture": "RD_ONLY_STRICT",
        })
    return rows


def load_shuttleset_rows(root: Path, *, seed: int) -> list[dict[str, Any]]:
    """Load MIT stroke labels only; no broadcast media is present or trained."""

    set_root = root / "coachai_shuttleset" / "ShuttleSet" / "set"
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(set_root.glob("*/*.csv")):
        if csv_path.name.startswith("._") or csv_path.parent.name.startswith("._"):
            continue
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            events = [row for row in csv.DictReader(handle) if row.get("frame_num", "").strip()]
        group = csv_path.parent.name
        split_bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 10
        split = "test" if split_bucket == 0 else "val" if split_bucket == 1 else "train"
        rows.append({
            "source": "shuttleset",
            "video": f"{group}/{csv_path.name}",
            "source_video": group,
            "video_path": None,
            "media_present": False,
            "media_absent": True,
            "split": split,
            "fps": None,
            "source_start_frame": 0,
            "num_frames": None,
            "event_counts": {"HIT": len(events), "BOUNCE": 0, "background": 0},
            "inventory_event_count": len(events),
            "events": [],
            "loss_validity_mask": [True, True, False],
            "license_id": "MIT-labels; broadcast-pixels-absent",
            "license_posture": "COMMERCIAL_CLEAN_LABELS_ONLY",
        })
    return rows


def build_public_manifest(public_root: Path, *, seed: int = 20260716) -> dict[str, Any]:
    rows = load_jhong_rows(public_root, seed=seed) + load_opentt_rows(public_root) + load_shuttleset_rows(public_root, seed=seed)
    _rebalance_parent_splits(rows, seed=seed)
    totals: dict[str, dict[str, int]] = {}
    for source in EXPECTED_UNIVERSE:
        source_rows = [row for row in rows if row["source"] == source]
        totals[source] = {
            "rows": len(source_rows),
            "inventory_events": sum(int(row["inventory_event_count"]) for row in source_rows),
            "media_present_rows": sum(bool(row["media_present"]) for row in source_rows),
            "HIT": sum(int(row["event_counts"]["HIT"]) for row in source_rows),
            "BOUNCE": sum(int(row["event_counts"]["BOUNCE"]) for row in source_rows),
            "background_annotations": sum(int(row["event_counts"]["background"]) for row in source_rows),
        }
        if totals[source]["inventory_events"] != EXPECTED_UNIVERSE[source]:
            raise DatasetFormatError(
                f"{source} inventory mismatch: {totals[source]['inventory_events']} != {EXPECTED_UNIVERSE[source]}"
            )
    groups: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        groups.setdefault((row["source"], row["source_video"]), set()).add(row["split"])
    straddlers = {f"{a}:{b}": sorted(v) for (a, b), v in groups.items() if len(v) != 1}
    if straddlers:
        raise DatasetFormatError(f"source-video split leakage: {straddlers}")
    return {
        "schema_version": 1,
        "artifact_type": "event_head_public_dataset_manifest",
        "verified": False,
        "seed": seed,
        "config": {
            "split_unit": "source_parent_video",
            "split_ratios": SPLIT_RATIOS,
            "window_stride_frames": DEFAULT_WINDOW_STRIDE,
        },
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "image_size": IMAGE_SIZE,
        "decode_policy": "on_the_fly_no_frame_cache",
        "totals": totals,
        "rows": rows,
    }


def manifest_windows(
    manifest: dict[str, Any], *, split: str, limit: int, window_frames: int,
    stride_frames: int = DEFAULT_WINDOW_STRIDE,
) -> list[WindowSpec]:
    if limit < 1 or window_frames < 1 or stride_frames < 1:
        raise DatasetFormatError("limit, window_frames, and stride_frames must be positive")
    windows: list[WindowSpec] = []
    included_rows = 0
    teacher_derived = bool(manifest.get("teacher_derived", False))
    for row_index, row in enumerate(manifest["rows"]):
        if row["split"] != split or not row["media_present"] or not row["events"]:
            continue
        if included_rows >= limit:
            break
        included_rows += 1
        row_frames = int(row["num_frames"])
        tail_start = max(0, row_frames - window_frames)
        local_starts = list(range(0, tail_start + 1, stride_frames))
        if not local_starts or local_starts[-1] != tail_start:
            local_starts.append(tail_start)
        for local_start in local_starts:
            source_start = int(row["source_start_frame"]) + local_start
            local_events = tuple(
                (int(item["frame"]) - local_start, HIT if item["class"] == "HIT" else BOUNCE)
                for item in row["events"]
                if local_start <= int(item["frame"]) < local_start + window_frames
            )
            selected_events = [
                item for item in row["events"]
                if local_start <= int(item["frame"]) < local_start + window_frames
            ]
            windows.append(WindowSpec(
                video_path=Path(row["video_path"]), start_frame=source_start,
                num_frames=window_frames, fps=float(row["fps"]), events=local_events,
                validity_mask=tuple(row["loss_validity_mask"]), source=row["source"],
                license_posture=row["license_posture"],
                unknown_frame_mask=tuple(
                    row.get("unknown_frame_mask", [False] * row_frames)[
                        local_start:local_start + window_frames
                    ]
                ),
                event_subframe_offsets=tuple(
                    event_subframe_offset_frames(item, fps=float(row["fps"]))
                    for item in selected_events
                ),
                sample_weight=float(row.get("sample_weight", 1.0)),
                teacher_derived=teacher_derived,
                row_index=row_index,
                source_video=str(row.get("source_video", row.get("video", ""))),
            ))
    return windows


def manifest_event_centered_windows(
    manifest: dict[str, Any], *, split: str, limit: int, window_frames: int,
) -> list[WindowSpec]:
    """Build the frozen held-out protocol: one first-event-centered window per row.

    Training uses :func:`manifest_windows` and its sliding coverage. This
    separate evaluator preserves the 2026-07-16 checkpoint's preregistered
    public-eval sample while allowing its context length to match training.
    """

    if limit < 1 or window_frames < 1:
        raise DatasetFormatError("limit and window_frames must be positive")
    windows: list[WindowSpec] = []
    teacher_derived = bool(manifest.get("teacher_derived", False))
    for row_index, row in enumerate(manifest["rows"]):
        if row["split"] != split or not row["media_present"] or not row["events"]:
            continue
        event = row["events"][0]
        center = int(event["frame"])
        local_start = max(0, center - window_frames // 2)
        local_events = tuple(
            (int(item["frame"]) - local_start, HIT if item["class"] == "HIT" else BOUNCE)
            for item in row["events"]
            if local_start <= int(item["frame"]) < local_start + window_frames
        )
        selected_events = [
            item for item in row["events"]
            if local_start <= int(item["frame"]) < local_start + window_frames
        ]
        windows.append(WindowSpec(
            video_path=Path(row["video_path"]),
            start_frame=int(row["source_start_frame"]) + local_start,
            num_frames=window_frames, fps=float(row["fps"]), events=local_events,
            validity_mask=tuple(row["loss_validity_mask"]), source=row["source"],
            license_posture=row["license_posture"],
            unknown_frame_mask=tuple(
                row.get("unknown_frame_mask", [False] * int(row["num_frames"]))[
                    local_start:local_start + window_frames
                ]
            ),
            event_subframe_offsets=tuple(
                event_subframe_offset_frames(item, fps=float(row["fps"]))
                for item in selected_events
            ),
            sample_weight=float(row.get("sample_weight", 1.0)),
            teacher_derived=teacher_derived,
            row_index=row_index,
            source_video=str(row.get("source_video", row.get("video", ""))),
        ))
        if len(windows) >= limit:
            break
    return windows


def rows_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetFormatError(f"invalid JSONL line {line_no}: {path}") from exc
