"""On-the-fly event datasets with explicit supervision and license masks.

License posture is deliberately attached at each loader boundary.  jhong93
labels/code are BSD-3 but broadcast pixels make trained weights RD_ONLY;
OpenTTGames/Extended OpenTTGames are CC BY-NC-SA and RD_ONLY_STRICT;
ShuttleSet labels are MIT but media is absent and broadcast rights unresolved.
No loader in this module reads ``data/event_bootstrap_20260713``.
"""

from __future__ import annotations

import csv
import hashlib
import json
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
_CLIP_RE = re.compile(r"^(?P<base>.+)_(?P<start>\d+)_(?P<end>\d+)$")


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
        targets = torch.zeros(spec.num_frames, dtype=torch.long)
        for local_frame, class_id in spec.events:
            if 0 <= local_frame < spec.num_frames:
                targets[local_frame] = class_id
        return {
            "frames": frames,
            "targets": targets,
            "validity_mask": torch.tensor(spec.validity_mask, dtype=torch.bool),
            "source": spec.source,
            "license_posture": spec.license_posture,
        }


def _find_pilot_video(pilot_dir: Path, base: str) -> Path | None:
    matches = sorted(p for p in pilot_dir.glob(f"{base}.*") if p.is_file())
    return matches[0] if matches else None


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
    # The upstream clip-level train/val files straddle some parent broadcasts.
    # Preserve upstream identity in canonical_split, keep every canonical test
    # parent in test, and deterministically group train/val parents from seed.
    # This is the minimum reconciliation that makes actual splits leak-free.
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(row["source_video"], set()).add(row["canonical_split"])
    assigned: dict[str, str] = {}
    for group, splits in grouped.items():
        if "test" in splits:
            assigned[group] = "test"
        elif splits == {"val"}:
            assigned[group] = "val"
        elif splits == {"train"}:
            assigned[group] = "train"
        else:
            bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 5
            assigned[group] = "val" if bucket == 0 else "train"
    for row in rows:
        row["split"] = assigned[row["source_video"]]
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
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "image_size": IMAGE_SIZE,
        "decode_policy": "on_the_fly_no_frame_cache",
        "totals": totals,
        "rows": rows,
    }


def manifest_windows(
    manifest: dict[str, Any], *, split: str, limit: int, window_frames: int
) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    for row in manifest["rows"]:
        if row["split"] != split or not row["media_present"] or not row["events"]:
            continue
        event = row["events"][0]
        center = int(event["frame"])
        local_start = max(0, center - window_frames // 2)
        source_start = int(row["source_start_frame"]) + local_start
        local_events = tuple(
            (int(item["frame"]) - local_start, HIT if item["class"] == "HIT" else BOUNCE)
            for item in row["events"]
            if local_start <= int(item["frame"]) < local_start + window_frames
        )
        windows.append(WindowSpec(
            video_path=Path(row["video_path"]), start_frame=source_start,
            num_frames=window_frames, fps=float(row["fps"]), events=local_events,
            validity_mask=tuple(row["loss_validity_mask"]), source=row["source"],
            license_posture=row["license_posture"],
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
