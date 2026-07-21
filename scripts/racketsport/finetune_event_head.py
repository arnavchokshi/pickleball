#!/usr/bin/env python3
"""Fine-tune the event head on fixed owner train/validation manifests.

Production inputs use the current event-head dataset row schema. Owner
validation is never mixed with pseudo labels, and the protected 50-row seed is
rejected before media decode or checkpoint load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (
    BACKGROUND, BOUNCE, HIT, DatasetFormatError, EventWindowDataset, WindowSpec,
    sha256_file, validate_current_manifest,
)
from threed.racketsport.event_head.matcher import Event, greedy_match, peak_pick
from threed.racketsport.event_head.model import checkpoint_payload, masked_cross_entropy
from scripts.racketsport.train_event_head import (
    _git_head, _initialize_training_state, _seed_everything, _validated_device,
    _validation_is_better,
)

SEED = ROOT / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json"
DEFAULT_WINDOW_FRAMES = 64
DEFAULT_CLASS_WEIGHTS = (1.0, 5.0, 5.0)
DEFAULT_PSEUDO_WEIGHT_CAP = 1.0
EXPECTED_OWNER_TRAIN_ROWS = 61
EXPECTED_OWNER_VAL_ROWS = 41
_REQUIRED_ROW_FIELDS = {
    "source", "source_video", "video_path", "media_present", "split", "fps",
    "source_start_frame", "num_frames", "events", "loss_validity_mask",
    "license_posture",
}


class FineTuneInputError(ValueError):
    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class WeightedWindow:
    spec: WindowSpec
    sample_weight: float
    is_pseudo: bool
    row_index: int


class WeightedEventWindowDataset(EventWindowDataset):
    def __init__(self, windows: Sequence[WeightedWindow], *, image_size: int) -> None:
        self.weighted_windows = tuple(windows)
        super().__init__([window.spec for window in windows], image_size=image_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        window = self.weighted_windows[index]
        sample.update({
            "sample_weight": torch.tensor(window.sample_weight, dtype=torch.float32),
            "is_pseudo": torch.tensor(window.is_pseudo, dtype=torch.bool),
            "row_index": torch.tensor(window.row_index, dtype=torch.long),
        })
        return sample


def _contains_bootstrap(value: Any, *, key: str = "") -> bool:
    if isinstance(value, dict):
        for child_key, child in value.items():
            normalized = child_key.lower()
            if normalized in {"tier", "label_tier", "bootstrap_tier"}:
                return True
            if _contains_bootstrap(child, key=normalized):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_bootstrap(item, key=key) for item in value)
    text = str(value).lower()
    return any(token in text for token in (
        "event_bootstrap_v0", "data/event_bootstrap_20260713",
        "spot_check_tier_a_50", "owner_spot_check_results",
    ))


def _protected_frames(seed_path: Path = SEED) -> list[dict[str, Any]]:
    if not seed_path.is_file():
        raise FineTuneInputError(
            f"protected seed inventory is absent, refusing training: {seed_path}", 2
        )
    protected: list[dict[str, Any]] = []
    for index, label in enumerate(json.loads(seed_path.read_text())["labels"]):
        source = label["source"]
        video_path = Path(source["video_path"])
        frame = label["anchor"].get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise FineTuneInputError(
                f"protected seed row {index} has invalid anchor.frame", 20
            )
        protected.append({
            "frame": frame,
            "video_path": str(video_path.resolve()),
            "video_sha256": source.get("video_sha256"),
            "clip_id": source.get("clip_id"),
        })
    return protected


def _read_manifest(path: Path, *, role: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FineTuneInputError(f"{role} manifest is absent: {path}", 2)
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FineTuneInputError(f"{role} manifest is invalid JSON: {path}", 20) from exc
    try:
        validate_current_manifest(manifest)
    except DatasetFormatError as exc:
        raise FineTuneInputError(
            f"{role} manifest must use the current event-head dataset manifest schema: {exc}",
            20,
        ) from exc
    if _contains_bootstrap(manifest):
        raise FineTuneInputError(
            "FORBIDDEN_BOOTSTRAP_PROVENANCE: rejected Tier-A/B label source", 21
        )
    return manifest, raw


def _finite_positive(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FineTuneInputError(f"{field} must be numeric", 20) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise FineTuneInputError(f"{field} must be finite and positive", 20)
    return parsed


def _sample_weight(row: dict[str, Any], *, role: str, index: int) -> float:
    if role == "owner":
        if "sample_weight" in row and float(row["sample_weight"]) != 1.0:
            raise FineTuneInputError(
                f"owner row {index} sample_weight must be exactly 1.0", 20
            )
        return 1.0
    if "sample_weight" not in row:
        raise FineTuneInputError(
            f"pseudo row {index} must carry explicit sample_weight (0.25 or 0.5)", 20
        )
    weight = _finite_positive(
        row["sample_weight"], field=f"pseudo.rows[{index}].sample_weight"
    )
    if weight not in {0.25, 0.5}:
        raise FineTuneInputError(
            f"pseudo row {index} sample_weight must be 0.25 or 0.5, got {weight}", 20
        )
    if "agreement_count" in row:
        count = row["agreement_count"]
        if isinstance(count, bool) or not isinstance(count, int):
            raise FineTuneInputError(
                f"pseudo row {index} agreement_count must be an integer", 20
            )
        expected = 0.25 if count == 1 else 0.5 if count >= 2 else None
        if expected is None or expected != weight:
            raise FineTuneInputError(
                f"pseudo row {index} agreement_count={count} conflicts with "
                f"sample_weight={weight}", 20
            )
    return weight


def _validate_row(row: dict[str, Any], *, role: str, index: int) -> None:
    missing = sorted(_REQUIRED_ROW_FIELDS - set(row))
    if missing:
        raise FineTuneInputError(
            f"{role} manifest row {index} is missing current-schema fields: {missing}", 20
        )
    if row["split"] not in {"train", "val", "test"}:
        raise FineTuneInputError(
            f"{role} manifest row {index} has invalid split {row['split']!r}", 20
        )
    _finite_positive(row["fps"], field=f"{role}.rows[{index}].fps")
    if isinstance(row["source_start_frame"], bool) or int(row["source_start_frame"]) < 0:
        raise FineTuneInputError(
            f"{role} row {index} source_start_frame must be nonnegative", 20
        )
    if isinstance(row["num_frames"], bool) or int(row["num_frames"]) < 1:
        raise FineTuneInputError(f"{role} row {index} num_frames must be positive", 20)
    mask = row["loss_validity_mask"]
    if (
        not isinstance(mask, list) or len(mask) != 3
        or not all(isinstance(value, bool) for value in mask)
        or not mask[BACKGROUND]
    ):
        raise FineTuneInputError(
            f"{role} row {index} has invalid loss_validity_mask", 20
        )
    if role == "owner" and mask != [True, True, True]:
        raise FineTuneInputError(
            f"owner row {index} must validate all classes for macro-F1", 20
        )
    if not isinstance(row["events"], list):
        raise FineTuneInputError(f"{role} row {index} events must be an array", 20)
    seen: set[tuple[int, str]] = set()
    for event_index, event in enumerate(row["events"]):
        if not isinstance(event, dict) or event.get("class") not in {"HIT", "BOUNCE"}:
            raise FineTuneInputError(
                f"{role} row {index} event {event_index} must be HIT or BOUNCE", 20
            )
        frame = event.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise FineTuneInputError(
                f"{role} row {index} event {event_index} frame must be an integer", 20
            )
        if not 0 <= frame < int(row["num_frames"]):
            raise FineTuneInputError(
                f"{role} row {index} event {event_index} is outside num_frames", 20
            )
        key = (frame, event["class"])
        if key in seen:
            raise FineTuneInputError(f"duplicate event in {role} row {index}: {key}", 20)
        seen.add(key)


def _reject_protected_rows(
    rows: Iterable[dict[str, Any]], *, seed_path: Path = SEED
) -> None:
    protected = _protected_frames(seed_path)
    for index, row in enumerate(rows):
        if row.get("split") != "train":
            continue
        video_path = Path(str(row.get("video_path", "")))
        resolved = str(video_path.resolve())
        start = int(row["source_start_frame"])
        end = start + int(row["num_frames"])
        for seed in protected:
            same_media = (
                bool(row.get("video_sha256"))
                and row.get("video_sha256") == seed["video_sha256"]
            ) or (
                bool(row.get("clip_id")) and row.get("clip_id") == seed["clip_id"]
            ) or resolved == seed["video_path"]
            if not same_media or not start <= int(seed["frame"]) < end:
                continue
            raise FineTuneInputError(
                "PROTECTED_SEED_WINDOW_OVERLAP: manifest train row "
                f"{index} interval [{start},{end}) contains a protected frame",
                22,
            )


def validate_manifests(
    owner_path: Path,
    pseudo_path: Path | None,
    *,
    window_frames: int,
    expected_owner_train_rows: int = EXPECTED_OWNER_TRAIN_ROWS,
    expected_owner_val_rows: int = EXPECTED_OWNER_VAL_ROWS,
    seed_path: Path = SEED,
) -> tuple[dict[str, Any], bytes, dict[str, Any] | None, bytes | None]:
    owner, owner_raw = _read_manifest(owner_path, role="owner")
    configured = owner.get("config", {}).get("window_frames")
    if configured is not None and int(configured) != window_frames:
        raise FineTuneInputError(
            f"owner manifest window_frames={configured} does not match "
            f"--window-frames={window_frames}", 20
        )
    for index, row in enumerate(owner["rows"]):
        _validate_row(row, role="owner", index=index)
        _sample_weight(row, role="owner", index=index)
    train_rows = sum(row["split"] == "train" for row in owner["rows"])
    val_rows = sum(row["split"] == "val" for row in owner["rows"])
    if (train_rows, val_rows) != (
        expected_owner_train_rows, expected_owner_val_rows
    ):
        raise FineTuneInputError(
            f"fixed owner split mismatch: got train={train_rows},val={val_rows}; "
            f"expected train={expected_owner_train_rows},val={expected_owner_val_rows}",
            24,
        )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in owner["rows"]:
        group_splits[str(row["source_video"])].add(str(row["split"]))
    leaking = {
        group: sorted(splits) for group, splits in group_splits.items()
        if len(splits & {"train", "val"}) > 1
    }
    if leaking:
        raise FineTuneInputError(f"SOURCE_SPLIT_LEAKAGE: {leaking}", 24)
    _reject_protected_rows(owner["rows"], seed_path=seed_path)

    pseudo = None
    pseudo_raw = None
    if pseudo_path is not None:
        pseudo, pseudo_raw = _read_manifest(pseudo_path, role="pseudo")
        if pseudo.get("teacher_derived") is not True or pseudo.get("ground_truth") is not False:
            raise FineTuneInputError(
                "pseudo manifest must declare teacher_derived=true and ground_truth=false", 20
            )
        configured = pseudo.get("config", {}).get("window_frames")
        if configured is not None and int(configured) != window_frames:
            raise FineTuneInputError(
                f"pseudo manifest window_frames={configured} does not match "
                f"--window-frames={window_frames}", 20
            )
        for index, row in enumerate(pseudo["rows"]):
            _validate_row(row, role="pseudo", index=index)
            _sample_weight(row, role="pseudo", index=index)
            if row["split"] != "train":
                raise FineTuneInputError(
                    f"pseudo row {index} must be train-only; pseudo validation is unused",
                    20,
                )
        _reject_protected_rows(pseudo["rows"], seed_path=seed_path)
    return owner, owner_raw, pseudo, pseudo_raw


def _window_starts(
    num_frames: int, *, window_frames: int, stride_frames: int
) -> list[int]:
    if num_frames < window_frames:
        raise FineTuneInputError(
            f"manifest row has {num_frames} frames, shorter than required "
            f"context {window_frames}", 25
        )
    tail = num_frames - window_frames
    starts = list(range(0, tail + 1, stride_frames))
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return starts


def _weighted_window(
    row: dict[str, Any],
    *,
    local_start: int,
    window_frames: int,
    sample_weight: float,
    is_pseudo: bool,
    row_index: int,
) -> WeightedWindow:
    events = tuple(
        (
            int(event["frame"]) - local_start,
            HIT if event["class"] == "HIT" else BOUNCE,
        )
        for event in row["events"]
        if local_start <= int(event["frame"]) < local_start + window_frames
    )
    return WeightedWindow(
        spec=WindowSpec(
            video_path=Path(row["video_path"]),
            start_frame=int(row["source_start_frame"]) + local_start,
            num_frames=window_frames,
            fps=float(row["fps"]),
            events=events,
            validity_mask=tuple(row["loss_validity_mask"]),
            source=str(row["source"]),
            license_posture=str(row["license_posture"]),
            unknown_frame_mask=tuple(
                (row.get("unknown_frame_mask") or ())[
                    local_start:local_start + window_frames
                ]
            ),
        ),
        sample_weight=sample_weight,
        is_pseudo=is_pseudo,
        row_index=row_index,
    )


def _training_windows(
    manifest: dict[str, Any],
    *,
    role: str,
    window_frames: int,
    stride_frames: int,
) -> list[WeightedWindow]:
    windows: list[WeightedWindow] = []
    for index, row in enumerate(manifest["rows"]):
        if row["split"] != "train":
            continue
        if not row["media_present"] or not Path(row["video_path"]).is_file():
            raise FineTuneInputError(
                f"{role} train row {index} media is absent: {row['video_path']}", 25
            )
        weight = _sample_weight(row, role=role, index=index)
        for start in _window_starts(
            int(row["num_frames"]),
            window_frames=window_frames,
            stride_frames=stride_frames,
        ):
            windows.append(_weighted_window(
                row,
                local_start=start,
                window_frames=window_frames,
                sample_weight=weight,
                is_pseudo=role == "pseudo",
                row_index=index,
            ))
    if not windows:
        raise FineTuneInputError(f"{role} manifest has no train windows", 25)
    return windows


def _validation_windows(
    owner: dict[str, Any], *, window_frames: int
) -> list[WeightedWindow]:
    windows: list[WeightedWindow] = []
    for index, row in enumerate(owner["rows"]):
        if row["split"] != "val":
            continue
        if not row["media_present"] or not Path(row["video_path"]).is_file():
            raise FineTuneInputError(
                f"owner val row {index} media is absent: {row['video_path']}", 25
            )
        num_frames = int(row["num_frames"])
        _window_starts(
            num_frames, window_frames=window_frames, stride_frames=window_frames
        )
        if "eval_window_start_frame" in row:
            start = int(row["eval_window_start_frame"])
        elif num_frames == window_frames:
            start = 0
        elif row["events"]:
            start = int(row["events"][0]["frame"]) - window_frames // 2
        elif "anchor_frame" in row:
            start = int(row["anchor_frame"]) - window_frames // 2
        else:
            raise FineTuneInputError(
                "owner negative validation rows longer than --window-frames must "
                f"carry eval_window_start_frame or anchor_frame (row {index})", 20
            )
        start = min(max(0, start), num_frames - window_frames)
        windows.append(_weighted_window(
            row,
            local_start=start,
            window_frames=window_frames,
            sample_weight=1.0,
            is_pseudo=False,
            row_index=index,
        ))
    if not windows:
        raise FineTuneInputError("owner manifest has no fixed validation windows", 25)
    return windows


def _cap_effective_pseudo_loss(
    losses: torch.Tensor,
    valid_target: torch.Tensor,
    sample_weights: torch.Tensor,
    is_pseudo: torch.Tensor,
    *,
    pseudo_weight_cap: float,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    if losses.ndim != 2 or valid_target.shape != losses.shape:
        raise ValueError("losses and valid_target must be aligned [B,T] tensors")
    if sample_weights.ndim != 1 or is_pseudo.shape != sample_weights.shape:
        raise ValueError("sample_weights and is_pseudo must be one-dimensional and aligned")
    if losses.shape[0] != sample_weights.shape[0]:
        raise ValueError("loss rows and sample weights must align")
    if not math.isfinite(pseudo_weight_cap) or pseudo_weight_cap <= 0:
        raise ValueError("pseudo_weight_cap must be finite and positive")
    if not bool(torch.isfinite(sample_weights).all()) or not bool(
        (sample_weights > 0).all()
    ):
        raise ValueError("sample weights must be finite and positive")
    pseudo_mask = is_pseudo.bool()
    if bool(pseudo_mask.any()) and not bool((~pseudo_mask).any()):
        raise ValueError("pseudo rows cannot form a batch without human rows")
    base_frame_weights = sample_weights.to(losses.device)[:, None].expand_as(losses)
    detached_contributions = (losses.detach() * base_frame_weights)[valid_target]
    expanded_pseudo_mask = pseudo_mask.to(losses.device)[:, None].expand_as(losses)
    valid_is_pseudo = expanded_pseudo_mask[valid_target]
    human_loss = detached_contributions[~valid_is_pseudo].sum()
    pseudo_loss = detached_contributions[valid_is_pseudo].sum()
    allowed = human_loss * pseudo_weight_cap
    scale = (
        torch.minimum(torch.ones_like(pseudo_loss), allowed / pseudo_loss)
        if bool(pseudo_mask.any()) and float(pseudo_loss) > 0
        else torch.ones_like(pseudo_loss)
    )
    effective = sample_weights.clone()
    effective[pseudo_mask] *= scale
    effective_pseudo_loss = pseudo_loss * scale
    effective_total_loss = human_loss + effective_pseudo_loss
    pseudo_loss_fraction = (
        effective_pseudo_loss / effective_total_loss
        if float(effective_total_loss) > 0
        else torch.zeros_like(effective_total_loss)
    )
    stats: dict[str, float | bool] = {
        "raw_human_loss": float(human_loss.cpu()),
        "raw_pseudo_loss": float(pseudo_loss.cpu()),
        "effective_pseudo_loss": float(effective_pseudo_loss.cpu()),
        "effective_pseudo_loss_fraction": float(pseudo_loss_fraction.cpu()),
        "pseudo_scale": float(scale.detach().cpu()),
        "capped": bool(float(scale) < 1.0),
    }
    return effective, stats


def weighted_masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    validity_mask: torch.Tensor,
    *,
    frame_loss_mask: torch.Tensor | None = None,
    class_weights: torch.Tensor,
    sample_weights: torch.Tensor,
    is_pseudo: torch.Tensor,
    pseudo_weight_cap: float,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("logits/targets shape mismatch")
    if validity_mask.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("validity_mask must be [B,C]")
    if class_weights.shape != (logits.shape[2],):
        raise ValueError("class_weights must have one entry per class")
    valid_target = validity_mask.gather(1, targets).bool()
    if frame_loss_mask is not None:
        if frame_loss_mask.shape != targets.shape:
            raise ValueError("frame_loss_mask must be [B,T]")
        valid_target = valid_target & frame_loss_mask.to(logits.device).bool()
    if not bool(valid_target.any()):
        raise ValueError("batch contains no loss-valid targets")
    masked_logits = logits.masked_fill(~validity_mask[:, None, :], -1e4)
    losses = F.cross_entropy(
        masked_logits.flatten(0, 1),
        targets.flatten(),
        weight=class_weights,
        reduction="none",
    ).reshape_as(targets)
    effective, stats = _cap_effective_pseudo_loss(
        losses,
        valid_target,
        sample_weights.to(logits.device),
        is_pseudo.to(logits.device),
        pseudo_weight_cap=pseudo_weight_cap,
    )
    frame_weights = effective[:, None].expand_as(losses)
    denominator = (class_weights[targets] * frame_weights)[valid_target].sum()
    if not bool(denominator > 0):
        raise ValueError("weighted loss denominator must be positive")
    return (losses * frame_weights)[valid_target].sum() / denominator, stats


def _validation_metrics_from_batches(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> dict[str, Any]:
    totals = {
        "HIT": {"tp": 0, "fp": 0, "fn": 0},
        "BOUNCE": {"tp": 0, "fp": 0, "fn": 0},
    }
    max_probability = 0.0
    row_count = 0
    for logits, targets, validity_mask in batches:
        probabilities = logits.softmax(-1)[..., 1:]
        valid = validity_mask[:, None, 1:].expand_as(probabilities)
        finite = probabilities[torch.isfinite(probabilities) & valid]
        if finite.numel():
            max_probability = max(max_probability, float(finite.max()))
        for sample_index in range(logits.shape[0]):
            row_count += 1
            predictions = peak_pick(
                logits[sample_index], threshold=0.5, nms_radius=2
            )
            truth = [
                Event(frame, int(class_id))
                for frame, class_id in enumerate(targets[sample_index].tolist())
                if class_id in (HIT, BOUNCE)
            ]
            for class_id, name in ((HIT, "HIT"), (BOUNCE, "BOUNCE")):
                matched = greedy_match(
                    [event for event in predictions if event.class_id == class_id],
                    [event for event in truth if event.class_id == class_id],
                    tolerance_frames=2,
                )
                for key in ("tp", "fp", "fn"):
                    totals[name][key] += int(matched[key])
    f1s: list[float] = []
    for counts in totals.values():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        counts.update({"precision": precision, "recall": recall, "f1": f1})
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)
    return {
        "metric": "HIT_BOUNCE_macro_F1_at_plus_minus_2_frames",
        "threshold": 0.5,
        "nms_radius_frames": 2,
        "tolerance_frames": 2,
        "validation_rows": row_count,
        "macro_f1_at_2": macro_f1,
        "f1": macro_f1,
        "per_class": totals,
        "max_positive_class_probability": max_probability,
    }


def _validation_metrics(
    model: torch.nn.Module, loader: DataLoader, *, device: torch.device
) -> dict[str, Any]:
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batches.append((
                model(batch["frames"].to(device)).cpu(),
                batch["targets"],
                batch["validity_mask"],
            ))
    return _validation_metrics_from_batches(batches)


def _assert_checkpoint_context(
    checkpoint: Path, *, window_frames: int, image_size: int
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FineTuneInputError(f"initial checkpoint is absent: {checkpoint}", 2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "event_head_scaffold":
        raise FineTuneInputError(f"not an event-head checkpoint: {checkpoint}", 20)
    top_level_context = payload.get("window_frames")
    config_context = (
        payload["config"].get("window_frames")
        if isinstance(payload.get("config"), dict) else None
    )
    if top_level_context is None and config_context is None:
        raise FineTuneInputError(
            "checkpoint has no explicit window_frames context; refusing ambiguous fine-tune",
            20,
        )
    if (
        top_level_context is not None and config_context is not None
        and int(top_level_context) != int(config_context)
    ):
        raise FineTuneInputError(
            "checkpoint carries conflicting window_frames contexts: "
            f"top-level={top_level_context}, config={config_context}", 20
        )
    configured = top_level_context if top_level_context is not None else config_context
    if int(configured) != window_frames:
        raise FineTuneInputError(
            f"checkpoint window_frames={configured} does not match "
            f"--window-frames={window_frames}", 20
        )
    top_level_image = payload.get("image_size")
    config_image = (
        payload["config"].get("image_size")
        if isinstance(payload.get("config"), dict) else None
    )
    if (
        top_level_image is not None and config_image is not None
        and int(top_level_image) != int(config_image)
    ):
        raise FineTuneInputError(
            "checkpoint carries conflicting image_size contexts: "
            f"top-level={top_level_image}, config={config_image}", 20
        )
    checkpoint_image = top_level_image if top_level_image is not None else config_image
    if checkpoint_image is not None and int(checkpoint_image) != image_size:
        raise FineTuneInputError(
            f"checkpoint image_size={checkpoint_image} does not match "
            f"--image-size={image_size}", 20
        )
    return payload


def _loader(
    windows: Sequence[WeightedWindow],
    *,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        WeightedEventWindowDataset(windows, image_size=image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        **({"prefetch_factor": 2} if num_workers > 0 else {}),
    )


def _next_batch(iterator: Any, loader: DataLoader) -> tuple[dict[str, Any], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _merge_batches(
    owner_batch: dict[str, Any], pseudo_batch: dict[str, Any] | None
) -> dict[str, Any]:
    if pseudo_batch is None:
        return owner_batch
    merged = dict(owner_batch)
    for key in (
        "frames", "targets", "validity_mask", "frame_loss_mask", "sample_weight",
        "is_pseudo", "row_index",
    ):
        merged[key] = torch.cat((owner_batch[key], pseudo_batch[key]), dim=0)
    return merged


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    best_f1: float,
    best_probability: float,
    config: dict[str, Any],
    provenance: dict[str, Any],
    license_posture: str,
    license_reason: str,
    role: str,
) -> None:
    torch.save(checkpoint_payload(
        model,  # type: ignore[arg-type]
        license_posture=license_posture,
        license_reason=license_reason,
        window_frames=config["window_frames"],
        image_size=config["image_size"],
        completed_steps=completed_steps,
        optimizer_steps=completed_steps,
        optimizer_state_dict=optimizer.state_dict(),
        best_val_macro_f1_at_2=best_f1,
        best_val_max_positive_class_probability=best_probability,
        config=config,
        fine_tune_provenance=provenance,
        resume_mode="model_only",
        optimizer_state_restored=False,
        checkpoint_role=role,
        verified=False,
    ), path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a completion manifest only after its complete bytes are durable."""

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_finetune(
    *,
    owner_manifest_path: Path,
    pseudo_manifest_path: Path | None,
    init_checkpoint_model_only: Path,
    out: Path,
    device_name: str,
    steps: int,
    image_size: int,
    window_frames: int,
    batch_size: int,
    lr: float,
    val_every: int,
    seed: int,
    stride_frames: int,
    num_workers: int,
    class_weights: Sequence[float],
    pseudo_weight_cap: float,
    expected_owner_train_rows: int = EXPECTED_OWNER_TRAIN_ROWS,
    expected_owner_val_rows: int = EXPECTED_OWNER_VAL_ROWS,
    max_wall_minutes: float | None = None,
) -> dict[str, Any]:
    if (
        steps < 1 or image_size < 16 or window_frames < 1 or batch_size < 1
        or lr <= 0 or val_every < 1 or stride_frames < 1 or num_workers < 0
    ):
        raise FineTuneInputError("invalid non-positive fine-tune configuration", 20)
    if len(class_weights) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in class_weights
    ):
        raise FineTuneInputError(
            "--class-weights requires three finite positive values", 20
        )
    if not math.isfinite(pseudo_weight_cap) or pseudo_weight_cap <= 0:
        raise FineTuneInputError("--pseudo-weight-cap must be finite and positive", 20)
    if max_wall_minutes is not None and max_wall_minutes <= 0:
        raise FineTuneInputError("--max-wall-minutes must be positive", 20)

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "finetune_manifest.json"
    manifest_temporary_path = out / ".finetune_manifest.json.tmp"
    # Reusing an output directory starts a new arm attempt. Remove the prior
    # completion claim before validation, decode, checkpoint load, or training
    # can be interrupted; only the final atomic replace may recreate it.
    manifest_path.unlink(missing_ok=True)
    manifest_temporary_path.unlink(missing_ok=True)

    owner, owner_raw, pseudo, pseudo_raw = validate_manifests(
        owner_manifest_path,
        pseudo_manifest_path,
        window_frames=window_frames,
        expected_owner_train_rows=expected_owner_train_rows,
        expected_owner_val_rows=expected_owner_val_rows,
    )
    pretrain_payload = _assert_checkpoint_context(
        init_checkpoint_model_only,
        window_frames=window_frames,
        image_size=image_size,
    )
    owner_train = _training_windows(
        owner, role="owner", window_frames=window_frames,
        stride_frames=stride_frames,
    )
    owner_val = _validation_windows(owner, window_frames=window_frames)
    pseudo_train = (
        _training_windows(
            pseudo, role="pseudo", window_frames=window_frames,
            stride_frames=stride_frames,
        )
        if pseudo is not None else []
    )
    if len(owner_val) != expected_owner_val_rows:
        raise FineTuneInputError(
            f"fixed validation produced {len(owner_val)} windows, "
            f"expected {expected_owner_val_rows}", 24
        )

    device = _validated_device(device_name)
    _seed_everything(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    owner_loader = _loader(
        owner_train, image_size=image_size, batch_size=batch_size, shuffle=True,
        seed=seed, num_workers=num_workers,
    )
    pseudo_loader = (
        _loader(
            pseudo_train, image_size=image_size, batch_size=batch_size, shuffle=True,
            seed=seed + 1, num_workers=num_workers,
        )
        if pseudo_train else None
    )
    val_loader = _loader(
        owner_val, image_size=image_size, batch_size=batch_size, shuffle=False,
        seed=seed, num_workers=num_workers,
    )
    state = _initialize_training_state(
        device=device,
        device_name=device_name,
        weights="none",
        lr=lr,
        init_checkpoint=None,
        init_checkpoint_model_only=init_checkpoint_model_only,
    )
    if state.resume_mode != "model_only" or state.optimizer_state_restored:
        raise RuntimeError("fine-tune must load model-only with a fresh optimizer")
    model, optimizer = state.model, state.optimizer
    loss_class_weights = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )
    config = {
        "device": device_name,
        "steps": steps,
        "image_size": image_size,
        "window_frames": window_frames,
        "batch_size_human": batch_size,
        "batch_size_pseudo_max": batch_size if pseudo_train else 0,
        "lr": lr,
        "val_every": val_every,
        "seed": seed,
        "stride_frames": stride_frames,
        "num_workers": num_workers,
        "class_weights": list(class_weights),
        "pseudo_weight_cap": pseudo_weight_cap,
        "expected_owner_train_rows": expected_owner_train_rows,
        "expected_owner_val_rows": expected_owner_val_rows,
        "validation_threshold": 0.5,
        "validation_tolerance_frames": 2,
        "max_wall_minutes": max_wall_minutes,
    }
    provenance = {
        "owner_manifest": str(owner_manifest_path),
        "owner_manifest_sha256": hashlib.sha256(owner_raw).hexdigest(),
        "pseudo_manifest": str(pseudo_manifest_path) if pseudo_manifest_path else None,
        "pseudo_manifest_sha256": (
            hashlib.sha256(pseudo_raw).hexdigest() if pseudo_raw is not None else None
        ),
        "init_checkpoint_model_only": str(init_checkpoint_model_only),
        "init_checkpoint_sha256": sha256_file(init_checkpoint_model_only),
        "init_checkpoint_completed_steps": int(
            pretrain_payload.get("completed_steps", 0)
        ),
        "git_head": _git_head(),
    }
    license_posture = str(pretrain_payload.get("license_posture", "RD_ONLY"))
    license_reason = str(
        pretrain_payload.get("license_reason", "inherits pretrain posture")
    )
    if pseudo is not None:
        license_reason += (
            "; pseudo manifest posture="
            + str(pseudo.get("license_posture", "declared per row"))
        )
    best_path = out / "best_event_head_finetuned.pt"
    last_path = out / "event_head_finetuned.pt"
    started = time.monotonic()
    losses: list[float] = []
    validations: list[dict[str, Any]] = []
    weight_batches: list[dict[str, float | bool]] = []
    completed_steps = 0
    wall_stopped = False
    owner_iterator = iter(owner_loader)
    pseudo_iterator = iter(pseudo_loader) if pseudo_loader is not None else None

    initial = {"step": 0, **_validation_metrics(model, val_loader, device=device)}
    validations.append(initial)
    best_f1 = float(initial["macro_f1_at_2"])
    best_probability = float(initial["max_positive_class_probability"])
    _save_checkpoint(
        best_path, model=model, optimizer=optimizer, completed_steps=0,
        best_f1=best_f1, best_probability=best_probability, config=config,
        provenance=provenance, license_posture=license_posture,
        license_reason=license_reason, role="best_by_owner_val_macro_f1_at_2",
    )

    while completed_steps < steps:
        if (
            max_wall_minutes is not None
            and time.monotonic() - started >= max_wall_minutes * 60
        ):
            wall_stopped = True
            break
        owner_batch, owner_iterator = _next_batch(owner_iterator, owner_loader)
        pseudo_batch = None
        if pseudo_loader is not None and pseudo_iterator is not None:
            pseudo_batch, pseudo_iterator = _next_batch(
                pseudo_iterator, pseudo_loader
            )
        batch = _merge_batches(owner_batch, pseudo_batch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, weighting = weighted_masked_cross_entropy(
            model(batch["frames"].to(device)),
            batch["targets"].to(device),
            batch["validity_mask"].to(device),
            frame_loss_mask=batch["frame_loss_mask"].to(device),
            class_weights=loss_class_weights,
            sample_weights=batch["sample_weight"].to(device),
            is_pseudo=batch["is_pseudo"].to(device),
            pseudo_weight_cap=pseudo_weight_cap,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"fine-tune produced non-finite loss at step {completed_steps + 1}"
            )
        loss.backward()
        optimizer.step()
        completed_steps += 1
        losses.append(float(loss.detach().cpu()))
        weight_batches.append(weighting)
        if completed_steps % val_every == 0 or completed_steps == steps:
            validation = {
                "step": completed_steps,
                **_validation_metrics(model, val_loader, device=device),
            }
            validations.append(validation)
            if _validation_is_better(
                validation,
                best_val_f1=best_f1,
                best_val_max_positive_class_probability=best_probability,
            ):
                best_f1 = float(validation["macro_f1_at_2"])
                best_probability = float(
                    validation["max_positive_class_probability"]
                )
                _save_checkpoint(
                    best_path, model=model, optimizer=optimizer,
                    completed_steps=completed_steps, best_f1=best_f1,
                    best_probability=best_probability, config=config,
                    provenance=provenance, license_posture=license_posture,
                    license_reason=license_reason,
                    role="best_by_owner_val_macro_f1_at_2",
                )

    if completed_steps != steps:
        manifest_path.unlink(missing_ok=True)
        manifest_temporary_path.unlink(missing_ok=True)
        best_path.unlink(missing_ok=True)
        last_path.unlink(missing_ok=True)
        failure = {
            "schema_version": 1,
            "artifact_type": "event_head_finetune_arm_failure",
            "verified": False,
            "status": "failed_incomplete_step_budget",
            "completed_steps": completed_steps,
            "target_steps": steps,
            "equal_step_eligible": False,
            "reason": "wall_time_exit_before_target_steps" if wall_stopped else "step_mismatch",
        }
        (out / "arm_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        raise FineTuneInputError(
            "INCOMPLETE_ARM_STEPS: refusing an unequal-step arm; "
            f"completed {completed_steps} of {steps} target steps",
            31,
        )

    _save_checkpoint(
        last_path, model=model, optimizer=optimizer,
        completed_steps=completed_steps, best_f1=best_f1,
        best_probability=best_probability, config=config,
        provenance=provenance, license_posture=license_posture,
        license_reason=license_reason, role="last",
    )
    elapsed = time.monotonic() - started
    result = {
        "schema_version": 2,
        "artifact_type": "event_head_finetune_manifest",
        "verified": False,
        "status": "complete",
        "honest_partial": False,
        "equal_step_eligible": True,
        "config": config,
        "owner_train_rows": expected_owner_train_rows,
        "owner_validation_rows": expected_owner_val_rows,
        "owner_train_windows": len(owner_train),
        "pseudo_train_rows": len(pseudo["rows"]) if pseudo is not None else 0,
        "pseudo_train_windows": len(pseudo_train),
        "validation_windows": len(owner_val),
        "validation_protocol": (
            "fixed_owner_val_only_macro_F1_at_plus_minus_2_frames"
        ),
        "validations": validations,
        "best_val_macro_f1_at_2": best_f1,
        "best_val_max_positive_class_probability": best_probability,
        "losses": losses,
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "batch_weighting": {
            "owner_row_weight": 1.0,
            "pseudo_manifest_field": "sample_weight",
            "pseudo_loss_cap_vs_human_per_batch": pseudo_weight_cap,
            "cap_basis": "post_class_and_frame_weighted_aggregate_loss",
            "raw_pseudo_loss_max": max(
                (float(item["raw_pseudo_loss"]) for item in weight_batches),
                default=0.0,
            ),
            "effective_pseudo_loss_max": max(
                (float(item["effective_pseudo_loss"]) for item in weight_batches),
                default=0.0,
            ),
            "effective_pseudo_loss_fraction_max": max(
                (
                    float(item["effective_pseudo_loss_fraction"])
                    for item in weight_batches
                ),
                default=0.0,
            ),
            "capped_batches": sum(
                bool(item["capped"]) for item in weight_batches
            ),
            "batches": len(weight_batches),
        },
        "completed_steps": completed_steps,
        "target_steps": steps,
        "elapsed_s": elapsed,
        "steps_per_s": completed_steps / elapsed if elapsed else 0.0,
        "best_checkpoint": str(best_path),
        "checkpoint": str(last_path),
        "resume_mode": state.resume_mode,
        "optimizer_state_restored": state.optimizer_state_restored,
        "provenance": provenance,
        "license_posture": license_posture,
    }
    _atomic_write_json(manifest_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-manifest", type=Path)
    parser.add_argument("--pseudo-manifest", type=Path)
    parser.add_argument("--init-checkpoint-model-only", type=Path)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reviewed", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pretrain", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--stride-frames", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--class-weights", type=float, nargs=3,
        default=list(DEFAULT_CLASS_WEIGHTS),
        metavar=("BACKGROUND", "HIT", "BOUNCE"),
    )
    parser.add_argument(
        "--pseudo-weight-cap", type=float, default=DEFAULT_PSEUDO_WEIGHT_CAP,
        help=(
            "Maximum aggregate pseudo/human loss ratio after class and frame "
            "weighting in each mixed batch"
        ),
    )
    parser.add_argument(
        "--expected-owner-train-rows", type=int,
        default=EXPECTED_OWNER_TRAIN_ROWS,
    )
    parser.add_argument(
        "--expected-owner-val-rows", type=int,
        default=EXPECTED_OWNER_VAL_ROWS,
    )
    parser.add_argument("--max-wall-minutes", type=float)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.reviewed is not None or args.manifest is not None or args.pretrain is not None:
        parser.error(
            "legacy --reviewed/--manifest/--pretrain input was removed because it uses "
            "a stale schema; use --owner-manifest and --init-checkpoint-model-only"
        )
    try:
        if args.owner_manifest is None or args.init_checkpoint_model_only is None:
            parser.error(
                "fine-tune requires --owner-manifest and --init-checkpoint-model-only"
            )
        result = run_finetune(
            owner_manifest_path=args.owner_manifest,
            pseudo_manifest_path=args.pseudo_manifest,
            init_checkpoint_model_only=args.init_checkpoint_model_only,
            out=args.out,
            device_name=args.device,
            steps=args.steps,
            image_size=args.image_size,
            window_frames=args.window_frames,
            batch_size=args.batch_size,
            lr=args.lr,
            val_every=args.val_every,
            seed=args.seed,
            stride_frames=args.stride_frames,
            num_workers=args.num_workers,
            class_weights=args.class_weights,
            pseudo_weight_cap=args.pseudo_weight_cap,
            expected_owner_train_rows=args.expected_owner_train_rows,
            expected_owner_val_rows=args.expected_owner_val_rows,
            max_wall_minutes=args.max_wall_minutes,
        )
    except FineTuneInputError as exc:
        parser.exit(exc.exit_code, f"fine-tune input rejected: {exc}\n")
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(30, f"fine-tune failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
