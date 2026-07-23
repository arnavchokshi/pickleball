#!/usr/bin/env python3
"""Fine-tune the event head on fixed owner train/validation manifests.

Production inputs use the current event-head dataset row schema. In the legacy
owner-val mode, validation is never mixed with pseudo labels and the protected
50-row inventory is checked before media decode or checkpoint load. E-v2's
``final-step`` mode never opens that sealed inventory or constructs owner-val;
it requires SHA-pinned inputs plus the owner manifest's frozen exclusion
attestation and selects only the terminal training step.
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
from typing import Any, Iterable, Literal, Mapping, Sequence

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (
    BACKGROUND, BOUNCE, HIT, DatasetFormatError, EventWindowDataset, WindowSpec,
    dense_class_counts, event_subframe_offset_frames, preprocess_rgb, sha256_file,
    sqrt_frequency_class_weights, validate_current_manifest,
)
from threed.racketsport.event_head.assignment import (
    dense_cross_entropy, dynamic_label_assignment, offset_smooth_l1,
)
from threed.racketsport.event_head.matcher import Event, greedy_match, peak_pick
from threed.racketsport.event_head.model import checkpoint_payload, masked_cross_entropy
from scripts.racketsport.train_event_head import (
    _git_head, _initialize_training_state, _seed_everything, _validated_device,
    _validation_is_better,
)
from scripts.racketsport.verify_training_inputs import GateProofError, assert_gate_proof

SEED = ROOT / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json"
DEFAULT_WINDOW_FRAMES = 64
DEFAULT_CLASS_WEIGHTS = (1.0, 5.0, 5.0)
DEFAULT_PSEUDO_WEIGHT_CAP = 1.0
DEFAULT_HARD_NEGATIVE_LOSS_CAP = 0.5
EXPECTED_OWNER_TRAIN_ROWS = 61
EXPECTED_OWNER_VAL_ROWS = 41
REGISTERED_STAGE_P_HELD_OUT_SOURCE = "st0epgnab7dr"
REGISTERED_HARD_NEGATIVE_CANDIDATES = 262
REGISTERED_HARD_NEGATIVE_TOP_K = 96
REGISTERED_OWNER_BATCH_SIZE = 8
REGISTERED_HARD_NEGATIVE_BATCH_SIZE = 4
REGISTERED_TRAIN_SOURCE_VIDEOS = (
    "73VurrTKCZ8", "Ezz6HDNHlnk", "_L0HVmAlCQI", "wBu8bC4OfUY",
)
REGISTERED_VALIDATION_SOURCE_VIDEOS = ("HyUqT7zFiwk", "zwCtH_i1_S4")
REGISTERED_TRAIN_MEDIA_PATHS = 38
REGISTERED_VALIDATION_MEDIA_PATHS = 2
REGISTERED_TRAIN_MEDIA_FRAMES = 57_025
REGISTERED_TRAIN_MEDIA_DURATION_S = 2063.1827083333333
REGISTERED_TRAIN_MEDIA_COUNTS = {
    "73VurrTKCZ8": 8,
    "Ezz6HDNHlnk": 8,
    "_L0HVmAlCQI": 19,
    "wBu8bC4OfUY": 3,
}
REGISTERED_STAGE_F_STEPS = 1000
REGISTERED_STAGE_F_PROBE_STEPS = 100
REGISTERED_STAGE_F_IMAGE_SIZE = 224
REGISTERED_STAGE_F_WINDOW_FRAMES = 64
REGISTERED_STAGE_F_LR = 0.001
REGISTERED_STAGE_F_VAL_EVERY = 100
REGISTERED_STAGE_F_SEED = 20260722
REGISTERED_STAGE_F_STRIDE_FRAMES = 32
REGISTERED_STAGE_F_NUM_WORKERS = 4
REGISTERED_STAGE_F_MAX_WALL_MINUTES = 180.0
REGISTERED_OWNER_TRAIN_NEGATIVE_ROWS = 21
REGISTERED_OWNER_NEGATIVE_MAX_FP = 2
REGISTERED_RATE_MIN_PER_S = 0.3
REGISTERED_RATE_MAX_PER_S = 1.0
REGISTERED_AUDIO_ONLY_MAX_FIRED_ROWS = 26
REGISTERED_THRESHOLD_GRID = tuple(round(0.20 + 0.05 * index, 2) for index in range(11))
REGISTERED_THRESHOLD_TIE_BREAK = (
    "macro_f1_at_2_desc", "fp_asc", "fn_asc", "threshold_asc",
    "checkpoint_step_asc_strict_tie",
)
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
    is_hard_negative: bool
    row_index: int


@dataclass(frozen=True)
class HardNegativeCandidate:
    """One excluded audio-only teacher event relabeled as background."""

    window: WeightedWindow
    focal_event_id: str
    excluded_event_frame: int
    excluded_event_class: int
    source_row_index: int
    source_video_id: str


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
            "is_hard_negative": torch.tensor(
                window.is_hard_negative, dtype=torch.bool
            ),
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


def _require_registered_sha256(
    raw: bytes, expected: str | None, *, role: str
) -> str:
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise FineTuneInputError(
            f"{role} requires an explicit lowercase SHA-256 pin", 20
        )
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise FineTuneInputError(
            f"{role} SHA-256 mismatch: expected {expected}, got {actual}", 23
        )
    return actual


def _require_registered_file_sha256(
    path: Path, expected: str | None, *, role: str
) -> str:
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise FineTuneInputError(
            f"{role} requires an explicit lowercase SHA-256 pin", 20
        )
    actual = sha256_file(path)
    if actual != expected:
        raise FineTuneInputError(
            f"{role} SHA-256 mismatch: expected {expected}, got {actual}", 23
        )
    return actual


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
    protected_policy: Literal["inventory", "registered_sha_only"] = "inventory",
    owner_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes, dict[str, Any] | None, bytes | None]:
    owner, owner_raw = _read_manifest(owner_path, role="owner")
    if protected_policy == "registered_sha_only":
        _require_registered_sha256(
            owner_raw, owner_manifest_sha256, role="owner manifest"
        )
    elif protected_policy != "inventory":
        raise FineTuneInputError(
            f"unsupported protected-data policy: {protected_policy}", 20
        )
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
    if protected_policy == "inventory":
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
        if protected_policy == "inventory":
            _reject_protected_rows(pseudo["rows"], seed_path=seed_path)
    return owner, owner_raw, pseudo, pseudo_raw


def validate_stage_f_owner_manifest(
    owner_path: Path,
    *,
    owner_manifest_sha256: str,
    window_frames: int,
    expected_owner_train_rows: int,
    expected_owner_val_rows: int,
) -> tuple[dict[str, Any], bytes]:
    """Validate only the SHA-pinned envelope and Stage-F train rows.

    Validation-row fields are intentionally never inspected.  The returned
    manifest is sanitized to contain train rows only, making an accidental
    downstream owner-validation construction impossible in final-step mode.
    """

    if not owner_path.is_file():
        raise FineTuneInputError(f"owner manifest is absent: {owner_path}", 2)
    # The SHA check is intentionally content-blind: it authenticates bytes but
    # does not branch on or inspect any owner-validation field.
    owner_raw = owner_path.read_bytes()
    _require_registered_sha256(
        owner_raw, owner_manifest_sha256, role="owner manifest"
    )
    try:
        envelope = json.loads(owner_raw)
    except json.JSONDecodeError as exc:
        raise FineTuneInputError(
            f"owner manifest is invalid JSON: {owner_path}", 20
        ) from exc
    if not isinstance(envelope, dict):
        raise FineTuneInputError("owner manifest root must be an object", 20)
    if (
        envelope.get("artifact_type") != "event_head_owner_reviewed_dataset_manifest"
        or envelope.get("schema_version") not in {1, 2}
        or envelope.get("classes")
        != {"0": "background", "1": "HIT", "2": "BOUNCE"}
        or envelope.get("teacher_derived") is not False
        or envelope.get("ground_truth") is not True
        or not isinstance(envelope.get("rows"), list)
    ):
        raise FineTuneInputError(
            "owner manifest does not match the registered reviewed-data envelope",
            20,
        )

    # Split-only projection is the first per-row operation.  Validation rows
    # are reduced to their split marker before provenance scans or any other
    # validation, so poisoned validation content cannot alter Stage-F control
    # flow.  Train rows retain their full reviewed fields.
    train_rows: list[dict[str, Any]] = []
    validation_count = 0
    for index, row in enumerate(envelope["rows"]):
        if not isinstance(row, dict):
            raise FineTuneInputError(f"owner row {index} must be an object", 20)
        split = row.get("split")
        if split == "train":
            train_rows.append(row)
        elif split == "val":
            validation_count += 1
        else:
            raise FineTuneInputError(
                f"owner row {index} has unsupported split marker {split!r}", 20
            )
    if (
        len(train_rows) != expected_owner_train_rows
        or validation_count != expected_owner_val_rows
    ):
        raise FineTuneInputError(
            "fixed owner split mismatch: "
            f"got train={len(train_rows)},val={validation_count}; expected "
            f"train={expected_owner_train_rows},val={expected_owner_val_rows}",
            24,
        )

    # Drop the raw row collection (and therefore every validation-only field)
    # before any recursive content/provenance scan.
    del envelope["rows"]
    top_level_metadata = {
        key: value for key, value in envelope.items()
    }
    if _contains_bootstrap(top_level_metadata) or any(
        _contains_bootstrap(row) for row in train_rows
    ):
        raise FineTuneInputError(
            "FORBIDDEN_BOOTSTRAP_PROVENANCE: rejected Tier-A/B label source",
            21,
        )
    config = envelope.get("config")
    if not isinstance(config, dict) or int(config.get("window_frames", -1)) != window_frames:
        raise FineTuneInputError(
            "owner manifest window context does not match Stage-F", 20
        )
    train_groups = config.get("train_source_groups")
    validation_groups = config.get("validation_source_groups")
    if (
        config.get("split_unit") != "original_source_video_id"
        or not isinstance(train_groups, list)
        or not train_groups
        or not isinstance(validation_groups, list)
        or not validation_groups
        or set(map(str, train_groups)) & set(map(str, validation_groups))
    ):
        raise FineTuneInputError(
            "owner manifest lacks a source-disjoint train/validation envelope",
            24,
        )
    protected_check = envelope.get("protected_seed_check")
    if (
        not isinstance(protected_check, dict)
        or protected_check.get("status") != "pass"
        or int(protected_check.get("overlap_rows", -1)) != 0
        or int(protected_check.get("checked_training_windows", -1))
        != expected_owner_train_rows
    ):
        raise FineTuneInputError(
            "SHA-pinned owner manifest does not carry a passing protected exclusion attestation",
            24,
        )

    for index, row in enumerate(train_rows):
        _validate_row(row, role="owner", index=index)
        _sample_weight(row, role="owner", index=index)
    observed_train_groups = {str(row["source_video"]) for row in train_rows}
    if observed_train_groups != set(map(str, train_groups)):
        raise FineTuneInputError(
            "owner train rows do not match registered train source groups", 24
        )
    sanitized = dict(envelope)
    sanitized["rows"] = train_rows
    sanitized["stage_f_split_envelope"] = {
        "owner_train_rows": len(train_rows),
        "owner_validation_rows_uninspected": validation_count,
        "validation_row_fields_accessed": ["split"],
        "protected_inventory_opened": False,
    }
    return sanitized, owner_raw


def _load_finetune_manifests(
    *,
    checkpoint_selection: Literal["owner-val", "final-step"],
    owner_manifest_path: Path,
    pseudo_manifest_path: Path | None,
    owner_manifest_sha256: str | None,
    window_frames: int,
    expected_owner_train_rows: int,
    expected_owner_val_rows: int,
) -> tuple[dict[str, Any], bytes, dict[str, Any] | None, bytes | None]:
    if checkpoint_selection == "final-step":
        if pseudo_manifest_path is not None:
            raise FineTuneInputError(
                "final-step Stage-F forbids pseudo positives; use hard-negative inputs",
                20,
            )
        if owner_manifest_sha256 is None:
            raise FineTuneInputError(
                "final-step Stage-F requires --owner-manifest-sha256", 20
            )
        owner, raw = validate_stage_f_owner_manifest(
            owner_manifest_path,
            owner_manifest_sha256=owner_manifest_sha256,
            window_frames=window_frames,
            expected_owner_train_rows=expected_owner_train_rows,
            expected_owner_val_rows=expected_owner_val_rows,
        )
        return owner, raw, None, None
    return validate_manifests(
        owner_manifest_path,
        pseudo_manifest_path,
        window_frames=window_frames,
        expected_owner_train_rows=expected_owner_train_rows,
        expected_owner_val_rows=expected_owner_val_rows,
    )


def _rows_by_focal_event_id(
    manifest: Mapping[str, Any], *, role: str
) -> dict[str, tuple[int, dict[str, Any]]]:
    indexed: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(manifest["rows"]):
        focal_event_id = row.get("focal_event_id")
        if not isinstance(focal_event_id, str) or not focal_event_id:
            raise FineTuneInputError(
                f"{role} row {index} lacks a nonempty focal_event_id", 20
            )
        if focal_event_id in indexed:
            raise FineTuneInputError(
                f"{role} has duplicate focal_event_id={focal_event_id}", 20
            )
        indexed[focal_event_id] = (index, row)
    return indexed


def derive_audio_only_hard_negative_pool(
    invalid_manifest_path: Path,
    repaired_manifest_path: Path,
    *,
    invalid_manifest_sha256: str,
    repaired_manifest_sha256: str,
    expected_candidates: int,
    window_frames: int,
    excluded_source_video_ids: Sequence[str] = (),
    expected_raw_candidates: int | None = None,
    expected_excluded_source_rows: int | None = None,
) -> tuple[list[HardNegativeCandidate], dict[str, Any]]:
    """Return the invalid-B minus repaired-B rows as zero-event negatives.

    The two manifests are registered inputs, not evaluation artifacts.  The
    repaired manifest declares the physical-cue requirement that removed the
    old audio-only family.  The delta is accepted only when it is a strict,
    identity-preserving subset and has the preregistered cardinality.
    """

    if expected_candidates < 1:
        raise FineTuneInputError(
            "--hard-negative-expected-candidates must be positive", 20
        )
    invalid, invalid_raw = _read_manifest(
        invalid_manifest_path, role="invalid hard-negative source"
    )
    repaired, repaired_raw = _read_manifest(
        repaired_manifest_path, role="repaired hard-negative reference"
    )
    invalid_sha = _require_registered_sha256(
        invalid_raw,
        invalid_manifest_sha256,
        role="invalid hard-negative source manifest",
    )
    repaired_sha = _require_registered_sha256(
        repaired_raw,
        repaired_manifest_sha256,
        role="repaired hard-negative reference manifest",
    )
    for role, manifest in (("invalid", invalid), ("repaired", repaired)):
        if manifest.get("teacher_derived") is not True or manifest.get("ground_truth") is not False:
            raise FineTuneInputError(
                f"{role} hard-negative manifest must be teacher-derived, never truth",
                20,
            )
        if any(row.get("split") != "train" for row in manifest["rows"]):
            raise FineTuneInputError(
                f"{role} hard-negative manifest must be train-only", 20
            )
        for index, row in enumerate(manifest["rows"]):
            _validate_row(row, role="pseudo", index=index)
            _sample_weight(row, role="pseudo", index=index)

    repaired_config = repaired.get("config")
    if not isinstance(repaired_config, dict) or (
        repaired_config.get("arm_b_required_agreement_family")
        != "ball_velocity_kink"
        or repaired_config.get("audio_only_rejection_reason")
        != "audio_only_no_physical_cue"
    ):
        raise FineTuneInputError(
            "repaired hard-negative manifest does not declare the registered "
            "ball-kink requirement/audio-only rejection contract",
            20,
        )

    invalid_by_id = _rows_by_focal_event_id(invalid, role="invalid manifest")
    repaired_by_id = _rows_by_focal_event_id(repaired, role="repaired manifest")
    extra_repaired = sorted(set(repaired_by_id) - set(invalid_by_id))
    if extra_repaired:
        raise FineTuneInputError(
            "repaired hard-negative manifest is not a subset of the invalid source",
            20,
        )
    identity_fields = (
        "source_video", "source_start_frame", "num_frames", "video_path",
        "unknown_frame_mask", "loss_validity_mask",
    )
    for focal_event_id, (_, repaired_row) in repaired_by_id.items():
        _, invalid_row = invalid_by_id[focal_event_id]
        invalid_events = [
            (event.get("event_id"), event.get("class"), event.get("frame"))
            for event in invalid_row.get("events", [])
        ]
        repaired_events = [
            (event.get("event_id"), event.get("class"), event.get("frame"))
            for event in repaired_row.get("events", [])
        ]
        if (
            any(
                invalid_row.get(field) != repaired_row.get(field)
                for field in identity_fields
            )
            or invalid_events != repaired_events
        ):
            raise FineTuneInputError(
                "repaired hard-negative manifest changed row identity/content for "
                f"{focal_event_id}",
                20,
            )

    raw_delta_ids = sorted(set(invalid_by_id) - set(repaired_by_id))
    excluded_sources = {str(value) for value in excluded_source_video_ids}
    if "" in excluded_sources:
        raise FineTuneInputError(
            "hard-negative excluded source-video IDs must be nonempty", 20
        )
    held_out_ids = [
        focal_event_id for focal_event_id in raw_delta_ids
        if str(invalid_by_id[focal_event_id][1].get("source_video"))
        in excluded_sources
    ]
    held_out_id_set = set(held_out_ids)
    excluded_ids = [
        focal_event_id for focal_event_id in raw_delta_ids
        if focal_event_id not in held_out_id_set
    ]
    if (
        expected_raw_candidates is not None
        and len(raw_delta_ids) != expected_raw_candidates
    ):
        raise FineTuneInputError(
            "audio-only hard-negative raw delta count mismatch: "
            f"got {len(raw_delta_ids)}, expected {expected_raw_candidates}",
            24,
        )
    if (
        expected_excluded_source_rows is not None
        and len(held_out_ids) != expected_excluded_source_rows
    ):
        raise FineTuneInputError(
            "held-out-source hard-negative exclusion count mismatch: "
            f"got {len(held_out_ids)}, expected {expected_excluded_source_rows}",
            24,
        )
    if len(excluded_ids) != expected_candidates:
        raise FineTuneInputError(
            "audio-only hard-negative post-source-exclusion count mismatch: "
            f"got {len(excluded_ids)}, expected {expected_candidates}; "
            f"raw_delta={len(raw_delta_ids)}, held_out_removed={len(held_out_ids)}",
            24,
        )

    candidates: list[HardNegativeCandidate] = []
    for focal_event_id in excluded_ids:
        source_row_index, row = invalid_by_id[focal_event_id]
        source_video_id = str(row.get("source_video"))
        if source_video_id in excluded_sources:
            raise AssertionError(
                "held-out Stage-P source entered the Stage-F hard-negative pool"
            )
        if row.get("agreement_count") != 1 or float(row.get("sample_weight", -1)) != 0.25:
            raise FineTuneInputError(
                f"excluded row {focal_event_id} is not a one-family 0.25-tier row",
                20,
            )
        if int(row["num_frames"]) != window_frames or len(row["events"]) != 1:
            raise FineTuneInputError(
                f"excluded row {focal_event_id} must be one exact {window_frames}-frame event window",
                20,
            )
        original_event = row["events"][0]
        agreements = original_event.get("independent_agreements")
        if not isinstance(agreements, list) or not agreements:
            raise FineTuneInputError(
                f"excluded row {focal_event_id} lacks independent agreement provenance",
                20,
            )
        agreement_families = {
            str(agreement.get("family"))
            for agreement in agreements
            if isinstance(agreement, dict)
        }
        if agreement_families != {"audio_onset"}:
            raise FineTuneInputError(
                f"excluded row {focal_event_id} is not audio-only: "
                f"families={sorted(agreement_families)}",
                20,
            )
        event_frame = int(original_event["frame"])
        event_class = HIT if original_event["class"] == "HIT" else BOUNCE
        negative_row = dict(row)
        negative_row["events"] = []
        negative_row["loss_validity_mask"] = [True, True, True]
        negative_window = _weighted_window(
            negative_row,
            local_start=0,
            window_frames=window_frames,
            sample_weight=1.0,
            is_pseudo=False,
            is_hard_negative=True,
            row_index=source_row_index,
        )
        if negative_window.spec.events or negative_window.spec.validity_mask != (
            True, True, True
        ):
            raise AssertionError("hard-negative relabeling must produce all-background truth")
        candidates.append(HardNegativeCandidate(
            window=negative_window,
            focal_event_id=focal_event_id,
            excluded_event_frame=event_frame,
            excluded_event_class=event_class,
            source_row_index=source_row_index,
            source_video_id=source_video_id,
        ))

    if any(
        candidate.source_video_id in excluded_sources for candidate in candidates
    ):
        raise AssertionError(
            "hard-negative source isolation failed after candidate construction"
        )

    report = {
        "policy": "invalid_arm_b_minus_repaired_arm_b_audio_only_family",
        "invalid_manifest": str(invalid_manifest_path),
        "invalid_manifest_sha256": invalid_sha,
        "invalid_rows": len(invalid_by_id),
        "repaired_manifest": str(repaired_manifest_path),
        "repaired_manifest_sha256": repaired_sha,
        "repaired_rows": len(repaired_by_id),
        "raw_audio_only_delta_rows": len(raw_delta_ids),
        "excluded_source_video_ids": sorted(excluded_sources),
        "excluded_held_out_source_rows": len(held_out_ids),
        "candidate_rows": len(candidates),
        "candidate_agreement_family_counts": {"audio_onset": len(candidates)},
        "identity_key": "focal_event_id",
        "training_relabel": "events=[]; loss_validity_mask=[true,true,true]",
        "frozen_validation_rows_consumed": 0,
        "stage_p_internal_validation_source_rows_consumed": 0,
    }
    return candidates, report


def _rank_hard_negative_score_rows(
    score_rows: Sequence[Mapping[str, Any]], *, top_k: int
) -> list[dict[str, Any]]:
    if top_k < 1 or top_k > len(score_rows):
        raise FineTuneInputError(
            f"hard-negative top-K must be in [1,{len(score_rows)}], got {top_k}",
            20,
        )
    ids = [str(row["focal_event_id"]) for row in score_rows]
    if len(set(ids)) != len(ids):
        raise FineTuneInputError("hard-negative score rows contain duplicate IDs", 20)
    for row in score_rows:
        score = float(row["max_positive_probability_at_excluded_frame"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise FineTuneInputError("hard-negative mining score is invalid", 20)
    ordered = sorted(
        (dict(row) for row in score_rows),
        key=lambda row: (
            -float(row["max_positive_probability_at_excluded_frame"]),
            str(row["focal_event_id"]),
        ),
    )
    return ordered[:top_k]


def mine_hard_negatives(
    model: torch.nn.Module,
    candidates: Sequence[HardNegativeCandidate],
    *,
    top_k: int,
    image_size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> tuple[list[WeightedWindow], dict[str, Any]]:
    """Select deterministic top-K Stage-P false-positive candidates."""

    if not candidates:
        raise FineTuneInputError("hard-negative candidate pool is empty", 25)
    by_row_index = {candidate.source_row_index: candidate for candidate in candidates}
    if len(by_row_index) != len(candidates):
        raise FineTuneInputError(
            "hard-negative source row indices must be unique", 20
        )
    loader = _loader(
        [candidate.window for candidate in candidates],
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )
    score_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["frames"].to(device))
            probabilities = logits.softmax(dim=-1).cpu()
            for sample_index, raw_row_index in enumerate(batch["row_index"].tolist()):
                candidate = by_row_index[int(raw_row_index)]
                frame = candidate.excluded_event_frame
                positive = probabilities[sample_index, frame, 1:]
                score_rows.append({
                    "focal_event_id": candidate.focal_event_id,
                    "source_row_index": candidate.source_row_index,
                    "excluded_event_frame": frame,
                    "excluded_event_class": (
                        "HIT" if candidate.excluded_event_class == HIT else "BOUNCE"
                    ),
                    "max_positive_probability_at_excluded_frame": float(
                        positive.max()
                    ),
                    "excluded_class_probability": float(
                        probabilities[
                            sample_index, frame, candidate.excluded_event_class
                        ]
                    ),
                })
    selected_rows = _rank_hard_negative_score_rows(score_rows, top_k=top_k)
    # Put training windows in registered rank order before seeded shuffling.
    selected_by_id = {
        candidate.focal_event_id: candidate.window for candidate in candidates
    }
    selected = [selected_by_id[str(row["focal_event_id"])] for row in selected_rows]
    return selected, {
        "policy": (
            "descending_max_positive_probability_at_excluded_teacher_frame; "
            "tie=focal_event_id_ascending"
        ),
        "candidate_count": len(candidates),
        "top_k": top_k,
        "selected": selected_rows,
        "all_scores": sorted(
            score_rows, key=lambda row: str(row["focal_event_id"])
        ),
    }


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
    is_hard_negative: bool,
    row_index: int,
) -> WeightedWindow:
    selected_events = [
        event for event in row["events"]
        if local_start <= int(event["frame"]) < local_start + window_frames
    ]
    events = tuple(
        (
            int(event["frame"]) - local_start,
            HIT if event["class"] == "HIT" else BOUNCE,
        )
        for event in selected_events
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
            event_subframe_offsets=tuple(
                event_subframe_offset_frames(event, fps=float(row["fps"]))
                for event in selected_events
            ),
            sample_weight=sample_weight,
            teacher_derived=is_pseudo,
            row_index=row_index,
            source_video=str(row["source_video"]),
        ),
        sample_weight=sample_weight,
        is_pseudo=is_pseudo,
        is_hard_negative=is_hard_negative,
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
                is_hard_negative=False,
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
            is_hard_negative=False,
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


def _capped_group_sample_weights(
    per_sample_numerator: torch.Tensor,
    sample_weights: torch.Tensor,
    group_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    cap: float,
    label: str,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    if per_sample_numerator.ndim != 1:
        raise ValueError("per_sample_numerator must be one-dimensional")
    if sample_weights.shape != per_sample_numerator.shape:
        raise ValueError("sample weights must align with per-sample losses")
    if group_mask.shape != per_sample_numerator.shape or reference_mask.shape != group_mask.shape:
        raise ValueError("sample group masks must align with per-sample losses")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError(f"{label} loss cap must be finite and positive")
    group = group_mask.to(per_sample_numerator.device, dtype=torch.bool)
    reference = reference_mask.to(per_sample_numerator.device, dtype=torch.bool)
    if bool((group & reference).any()):
        raise ValueError(f"{label} and reference sample masks overlap")
    if bool(group.any()) and not bool(reference.any()):
        raise ValueError(f"{label} samples require human reference samples")
    detached = per_sample_numerator.detach()
    reference_loss = detached[reference].sum()
    group_loss = detached[group].sum()
    allowed = reference_loss * cap
    scale = (
        torch.minimum(torch.ones_like(group_loss), allowed / group_loss)
        if bool(group.any()) and float(group_loss) > 0
        else torch.ones_like(group_loss)
    )
    effective = sample_weights.clone()
    effective[group] *= scale.to(effective.device)
    return effective, {
        f"raw_{label}_loss": float(group_loss.cpu()),
        f"reference_human_loss_for_{label}": float(reference_loss.cpu()),
        f"effective_{label}_loss": float((group_loss * scale).cpu()),
        f"{label}_scale": float(scale.cpu()),
        f"{label}_capped": bool(float(scale) < 1.0),
    }


def assignment_recipe_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    validity_mask: torch.Tensor,
    frame_loss_mask: torch.Tensor,
    event_subframe_offsets: torch.Tensor,
    *,
    predicted_offsets: torch.Tensor | None,
    class_weights: torch.Tensor,
    sample_weights: torch.Tensor,
    is_pseudo: torch.Tensor,
    is_hard_negative: torch.Tensor,
    pseudo_weight_cap: float,
    hard_negative_loss_cap: float,
    assignment_mode: Literal["fixed", "hungarian"],
    assignment_max_shift_frames: int,
    assignment_class_cost_weight: float,
    assignment_temporal_cost_weight: float,
    label_dilation_frames: int,
    neighbor_positive_weight: float,
    offset_loss_weight: float,
    offset_smooth_l1_beta: float,
) -> tuple[torch.Tensor, dict[str, float | bool | int]]:
    assignment = dynamic_label_assignment(
        logits,
        targets,
        validity_mask,
        frame_loss_mask,
        event_subframe_offsets,
        mode=assignment_mode,
        max_shift_frames=assignment_max_shift_frames,
        class_cost_weight=assignment_class_cost_weight,
        temporal_cost_weight=assignment_temporal_cost_weight,
        label_dilation_frames=label_dilation_frames,
        neighbor_positive_weight=neighbor_positive_weight,
    )
    weights = sample_weights.to(device=logits.device, dtype=logits.dtype)
    pseudo_mask = is_pseudo.to(device=logits.device, dtype=torch.bool)
    hard_mask = is_hard_negative.to(device=logits.device, dtype=torch.bool)
    if bool((pseudo_mask & hard_mask).any()):
        raise ValueError("a sample cannot be both pseudo and hard-negative")
    human_mask = ~(pseudo_mask | hard_mask)
    _, raw_numerator, _ = dense_cross_entropy(
        logits,
        assignment.dense_targets,
        validity_mask,
        frame_loss_mask,
        class_weights=class_weights,
        sample_weights=weights,
    )
    effective_weights, pseudo_stats = _capped_group_sample_weights(
        raw_numerator,
        weights,
        pseudo_mask,
        human_mask,
        cap=pseudo_weight_cap,
        label="pseudo",
    )
    # Recompute sufficient statistics after the pseudo scale so the hard-
    # negative cap remains relative to human rows, never to another weak pool.
    _, capped_pseudo_numerator, _ = dense_cross_entropy(
        logits,
        assignment.dense_targets,
        validity_mask,
        frame_loss_mask,
        class_weights=class_weights,
        sample_weights=effective_weights,
    )
    hard_scaled, hard_stats = _capped_group_sample_weights(
        capped_pseudo_numerator,
        effective_weights,
        hard_mask,
        human_mask,
        cap=hard_negative_loss_cap,
        label="hard_negative",
    )
    classification_loss, _, _ = dense_cross_entropy(
        logits,
        assignment.dense_targets,
        validity_mask,
        frame_loss_mask,
        class_weights=class_weights,
        sample_weights=hard_scaled,
    )
    if predicted_offsets is None:
        if offset_loss_weight != 0:
            raise ValueError("positive offset loss requires predicted offsets")
        offset_loss = logits.sum() * 0.0
    else:
        offset_loss, _, _ = offset_smooth_l1(
            predicted_offsets,
            assignment.offset_targets,
            assignment.offset_mask,
            sample_weights=hard_scaled,
            beta=offset_smooth_l1_beta,
        )
    total = classification_loss + offset_loss_weight * offset_loss
    stats: dict[str, float | bool | int] = {
        **pseudo_stats,
        **hard_stats,
        "classification_loss": float(classification_loss.detach().cpu()),
        "offset_loss": float(offset_loss.detach().cpu()),
        "assignment_event_count": assignment.event_count,
        "assignment_shifted_event_count": assignment.shifted_event_count,
        "assignment_total_abs_shift": assignment.total_abs_shift,
    }
    return total, stats


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


def _prediction_proxy_for_windows(
    model: torch.nn.Module,
    windows: Sequence[WeightedWindow],
    *,
    image_size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    loader = _loader(
        windows,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )
    predicted_events = 0
    rows_with_predictions = 0
    per_row: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["frames"].to(device)).cpu()
            for sample_index, row_index in enumerate(batch["row_index"].tolist()):
                predictions = peak_pick(
                    logits[sample_index], threshold=threshold, nms_radius=2
                )
                count = len(predictions)
                predicted_events += count
                rows_with_predictions += int(count > 0)
                per_row.append({
                    "row_index": int(row_index),
                    "prediction_count": count,
                })
    return {
        "rows": len(windows),
        "predicted_events": predicted_events,
        "rows_with_predictions": rows_with_predictions,
        "per_row": sorted(per_row, key=lambda row: int(row["row_index"])),
    }


def _owner_train_negative_windows(
    owner: Mapping[str, Any], *, window_frames: int, expected_rows: int
) -> list[WeightedWindow]:
    windows: list[WeightedWindow] = []
    for index, row in enumerate(owner["rows"]):
        if row["split"] != "train" or row["events"]:
            continue
        if int(row["num_frames"]) != window_frames:
            raise FineTuneInputError(
                "owner-train negative proxy requires exact registered windows; "
                f"row {index} has {row['num_frames']} frames",
                20,
            )
        if not row["media_present"] or not Path(row["video_path"]).is_file():
            raise FineTuneInputError(
                f"owner-train negative row {index} media is absent", 25
            )
        windows.append(_weighted_window(
            row,
            local_start=0,
            window_frames=window_frames,
            sample_weight=1.0,
            is_pseudo=False,
            is_hard_negative=False,
            row_index=index,
        ))
    if len(windows) != expected_rows:
        raise FineTuneInputError(
            "owner-train negative row count mismatch: "
            f"got {len(windows)}, expected {expected_rows}",
            24,
        )
    return windows


def validate_registered_rate_media_inventory(
    media_root: Path,
    inventory_path: Path,
    inventory_sha256: str,
) -> dict[str, Any]:
    """Validate the immutable, label-independent 38-train/2-val media set."""

    if not media_root.is_dir():
        raise FineTuneInputError(
            f"registered rate-media root is absent: {media_root}", 25
        )
    if not inventory_path.is_file():
        raise FineTuneInputError(
            f"registered rate-media inventory is absent: {inventory_path}", 25
        )
    raw = inventory_path.read_bytes()
    _require_registered_sha256(
        raw, inventory_sha256, role="registered rate-media inventory"
    )
    try:
        inventory = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FineTuneInputError(
            "registered rate-media inventory is invalid JSON", 20
        ) from exc
    if (
        inventory.get("artifact_type")
        != "event_ev2_registered_rate_media_inventory"
        or inventory.get("schema_version") != 1
        or inventory.get("train_source_video_ids")
        != list(REGISTERED_TRAIN_SOURCE_VIDEOS)
        or inventory.get("validation_source_video_ids")
        != list(REGISTERED_VALIDATION_SOURCE_VIDEOS)
        or inventory.get("train_media_count") != REGISTERED_TRAIN_MEDIA_PATHS
        or inventory.get("validation_media_count")
        != REGISTERED_VALIDATION_MEDIA_PATHS
        or inventory.get("train_per_source_counts")
        != REGISTERED_TRAIN_MEDIA_COUNTS
        or inventory.get("train_total_frames")
        != REGISTERED_TRAIN_MEDIA_FRAMES
        or not math.isclose(
            float(inventory.get("train_total_duration_s", -1.0)),
            REGISTERED_TRAIN_MEDIA_DURATION_S,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise FineTuneInputError(
            "registered rate-media inventory metadata diverges from E-v2", 24
        )
    entries = inventory.get("entries")
    if not isinstance(entries, list) or len(entries) != (
        REGISTERED_TRAIN_MEDIA_PATHS + REGISTERED_VALIDATION_MEDIA_PATHS
    ):
        raise FineTuneInputError(
            "registered rate-media inventory must contain exactly 40 entries", 24
        )

    root = media_root.resolve()
    expected_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [],
    }
    registered_paths: set[str] = set()
    per_source: dict[str, int] = defaultdict(int)
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise FineTuneInputError(
                f"registered rate-media entry {index} is not an object", 24
            )
        split = entry.get("split")
        source = entry.get("source_video_id")
        relative = entry.get("relative_path")
        digest = entry.get("sha256")
        frames = entry.get("frames")
        fps = entry.get("fps")
        duration_s = entry.get("duration_s")
        if split not in expected_by_split or not isinstance(source, str):
            raise FineTuneInputError(
                f"registered rate-media entry {index} has invalid split/source", 24
            )
        allowed_sources = (
            REGISTERED_TRAIN_SOURCE_VIDEOS
            if split == "train" else REGISTERED_VALIDATION_SOURCE_VIDEOS
        )
        if source not in allowed_sources:
            raise FineTuneInputError(
                f"registered rate-media entry {index} has unregistered source", 24
            )
        relative_path = Path(str(relative))
        if (
            relative_path.is_absolute()
            or relative_path.parts[:1] != (source,)
            or len(relative_path.parts) != 2
            or relative_path.suffix.lower() != ".mp4"
            or str(relative_path) in registered_paths
        ):
            raise FineTuneInputError(
                f"registered rate-media entry {index} has unsafe/duplicate path", 24
            )
        if (
            not isinstance(frames, int) or isinstance(frames, bool) or frames < 1
            or not isinstance(fps, (int, float)) or not math.isfinite(float(fps))
            or float(fps) <= 0
            or not isinstance(duration_s, (int, float))
            or not math.isclose(
                float(duration_s), frames / float(fps), rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise FineTuneInputError(
                f"registered rate-media entry {index} has invalid frame timing", 24
            )
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file():
            raise FineTuneInputError(
                f"registered rate-media object is absent: {relative_path}", 25
            )
        _require_registered_file_sha256(
            path, str(digest), role=f"registered rate-media object {relative_path}"
        )
        registered_paths.add(str(relative_path))
        per_source[source] += 1
        expected_by_split[split].append({**entry, "path": path})

    if per_source != {
        **REGISTERED_TRAIN_MEDIA_COUNTS,
        **{source: 1 for source in REGISTERED_VALIDATION_SOURCE_VIDEOS},
    }:
        raise FineTuneInputError(
            "registered rate-media per-source counts diverge from E-v2", 24
        )
    filesystem_paths = {
        str(path.relative_to(root))
        for source in (
            *REGISTERED_TRAIN_SOURCE_VIDEOS,
            *REGISTERED_VALIDATION_SOURCE_VIDEOS,
        )
        for path in (root / source).glob("*.mp4")
    }
    if filesystem_paths != registered_paths:
        raise FineTuneInputError(
            "registered rate-media inventory is not the complete six-source filesystem set",
            24,
        )
    return {
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_sha256,
        "train": sorted(
            expected_by_split["train"], key=lambda item: str(item["relative_path"])
        ),
        "validation": sorted(
            expected_by_split["validation"],
            key=lambda item: str(item["relative_path"]),
        ),
        "train_total_frames": REGISTERED_TRAIN_MEDIA_FRAMES,
        "train_total_duration_s": REGISTERED_TRAIN_MEDIA_DURATION_S,
    }


def _full_video_train_source_rate(
    model: torch.nn.Module,
    *,
    media_root: Path,
    train_source_video_ids: Sequence[str],
    validation_source_video_ids: Sequence[str],
    image_size: int,
    window_frames: int,
    device: torch.device,
    threshold: float,
    expected_media_paths: int,
    expected_source_videos: int,
    inventory_path: Path,
    inventory_sha256: str,
) -> dict[str, Any]:
    """Measure firing rate on the immutable label-independent inventory."""

    train_sources = tuple(sorted(map(str, train_source_video_ids)))
    validation_sources = tuple(sorted(map(str, validation_source_video_ids)))
    if (
        not media_root.is_dir()
        or len(set(train_sources)) != len(train_sources)
        or len(set(validation_sources)) != len(validation_sources)
        or set(train_sources) & set(validation_sources)
        or train_sources != tuple(sorted(REGISTERED_TRAIN_SOURCE_VIDEOS))
        or validation_sources != tuple(sorted(REGISTERED_VALIDATION_SOURCE_VIDEOS))
    ):
        raise FineTuneInputError(
            "internal full-video proxy requires disjoint, unique source directories",
            24,
        )
    inventory = validate_registered_rate_media_inventory(
        media_root, inventory_path, inventory_sha256
    )
    train_paths = {
        str(entry["path"]): str(entry["source_video_id"])
        for entry in inventory["train"]
    }
    expected_entry = {
        str(entry["path"]): entry for entry in inventory["train"]
    }
    validation_paths = {
        str(entry["path"]) for entry in inventory["validation"]
    }
    path_overlap = sorted(set(train_paths) & validation_paths)
    if path_overlap:
        raise FineTuneInputError(
            f"train/validation full-video inventory overlaps: {path_overlap}", 24
        )
    if (
        len(train_paths) != expected_media_paths
        or len(train_sources) != expected_source_videos
    ):
        raise FineTuneInputError(
            "owner-train full-video proxy inventory mismatch: "
            f"paths={len(train_paths)} expected={expected_media_paths}, "
            f"source_videos={len(train_sources)} "
            f"expected={expected_source_videos}",
            24,
        )

    total_events = 0
    total_duration_s = 0.0
    summaries: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for resolved_path in sorted(train_paths):
            video_path = Path(resolved_path)
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FineTuneInputError(
                    f"cannot open owner-train full video: {video_path}", 25
                )
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                capture.release()
                raise FineTuneInputError(
                    f"owner-train full video has invalid FPS: {video_path}", 25
                )
            frame_batch: list[torch.Tensor] = []
            frames = 0
            event_count = 0
            try:
                while True:
                    ok, frame_bgr = capture.read()
                    if not ok:
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_batch.append(preprocess_rgb(frame_rgb, image_size))
                    frames += 1
                    if len(frame_batch) == window_frames:
                        logits = model(torch.stack(frame_batch)[None].to(device))[0].cpu()
                        event_count += len(
                            peak_pick(logits, threshold=threshold, nms_radius=2)
                        )
                        frame_batch.clear()
                if frame_batch:
                    logits = model(torch.stack(frame_batch)[None].to(device))[0].cpu()
                    event_count += len(
                        peak_pick(logits, threshold=threshold, nms_radius=2)
                    )
            finally:
                capture.release()
            if frames == 0:
                raise FineTuneInputError(
                    f"owner-train full video decoded zero frames: {video_path}", 25
                )
            duration_s = frames / fps
            frozen = expected_entry[resolved_path]
            if (
                frames != int(frozen["frames"])
                or not math.isclose(
                    fps, float(frozen["fps"]), rel_tol=0.0, abs_tol=1e-9
                )
                or not math.isclose(
                    duration_s,
                    float(frozen["duration_s"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise FineTuneInputError(
                    f"registered rate-media decode facts changed: {video_path}", 24
                )
            total_events += event_count
            total_duration_s += duration_s
            summaries.append({
                "source_video_id": train_paths[resolved_path],
                "video_path": str(video_path),
                "frames": frames,
                "fps": fps,
                "duration_s": duration_s,
                "event_count": event_count,
                "events_per_second": event_count / duration_s,
            })
    if (
        sum(int(row["frames"]) for row in summaries)
        != REGISTERED_TRAIN_MEDIA_FRAMES
        or not math.isclose(
            total_duration_s,
            REGISTERED_TRAIN_MEDIA_DURATION_S,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise FineTuneInputError(
            "registered rate-media aggregate frame/duration facts changed", 24
        )
    return {
        "policy": "sha256_locked_complete_38_train_plus_2_validation_media_set",
        "inventory_independent_of_owner_manifest_rows": True,
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_sha256,
        "media_root": str(media_root.resolve()),
        "distinct_source_video_ids": list(train_sources),
        "distinct_source_video_count": len(train_sources),
        "validation_source_video_ids": list(validation_sources),
        "validation_media_path_count": len(validation_paths),
        "train_validation_path_overlap": path_overlap,
        "unique_media_paths": sorted(train_paths),
        "unique_media_path_count": len(train_paths),
        "event_count": total_events,
        "duration_s": total_duration_s,
        "events_per_second": total_events / total_duration_s,
        "summaries": summaries,
    }


def run_internal_stage_f_guards(
    model: torch.nn.Module,
    owner: Mapping[str, Any],
    audio_only_candidates: Sequence[HardNegativeCandidate],
    *,
    image_size: int,
    window_frames: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    threshold: float,
    owner_media_root: Path,
    train_source_video_ids: Sequence[str],
    validation_source_video_ids: Sequence[str],
    expected_owner_negative_rows: int,
    owner_negative_max_fp: int,
    audio_only_max_fired_rows: int,
    rate_min_per_s: float,
    rate_max_per_s: float,
    expected_train_media_paths: int,
    expected_train_source_videos: int,
    rate_media_inventory_path: Path,
    rate_media_inventory_sha256: str,
) -> dict[str, Any]:
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise FineTuneInputError(
            "--internal-decode-threshold must be finite and in (0,1)", 20
        )
    owner_negative_windows = _owner_train_negative_windows(
        owner,
        window_frames=window_frames,
        expected_rows=expected_owner_negative_rows,
    )
    owner_negative = _prediction_proxy_for_windows(
        model,
        owner_negative_windows,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        seed=seed,
        threshold=threshold,
    )
    audio_only = _prediction_proxy_for_windows(
        model,
        [candidate.window for candidate in audio_only_candidates],
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        seed=seed,
        threshold=threshold,
    )
    full_video = _full_video_train_source_rate(
        model,
        media_root=owner_media_root,
        train_source_video_ids=train_source_video_ids,
        validation_source_video_ids=validation_source_video_ids,
        image_size=image_size,
        window_frames=window_frames,
        device=device,
        threshold=threshold,
        expected_media_paths=expected_train_media_paths,
        expected_source_videos=expected_train_source_videos,
        inventory_path=rate_media_inventory_path,
        inventory_sha256=rate_media_inventory_sha256,
    )
    checks = {
        "owner_train_negative_fp": {
            "value": int(owner_negative["predicted_events"]),
            "maximum": owner_negative_max_fp,
            "denominator_rows": expected_owner_negative_rows,
            "pass": int(owner_negative["predicted_events"]) <= owner_negative_max_fp,
        },
        "audio_only_rows_with_predictions": {
            "value": int(audio_only["rows_with_predictions"]),
            "maximum": audio_only_max_fired_rows,
            "denominator_rows": len(audio_only_candidates),
            "pass": (
                int(audio_only["rows_with_predictions"])
                <= audio_only_max_fired_rows
            ),
        },
        "full_video_rate_per_s": {
            "value": float(full_video["events_per_second"]),
            "minimum": rate_min_per_s,
            "maximum": rate_max_per_s,
            "pass": (
                rate_min_per_s
                <= float(full_video["events_per_second"])
                <= rate_max_per_s
            ),
        },
    }
    return {
        "policy": "train_side_only_pre_owner41_scoring_stop_gate",
        "owner_validation_constructed": False,
        "owner_validation_scored": False,
        "protected_inventory_opened": False,
        "threshold": threshold,
        "threshold_source": "registered_stage_p_internal_validation_lock",
        "nms_radius_frames": 2,
        "owner_train_negative_proxy": owner_negative,
        "audio_only_proxy": audio_only,
        "full_video_train_source_proxy": full_video,
        "checks": checks,
        "pass": all(bool(check["pass"]) for check in checks.values()),
    }


def _assert_checkpoint_context(
    checkpoint: Path, *, window_frames: int, image_size: int,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FineTuneInputError(f"initial checkpoint is absent: {checkpoint}", 2)
    if expected_sha256 is not None:
        _require_registered_file_sha256(
            checkpoint, expected_sha256, role="initial checkpoint"
        )
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


def validate_stage_p_threshold_lock(
    threshold_lock_path: Path,
    stage_p_train_manifest_path: Path,
    init_checkpoint_path: Path,
    checkpoint_payload_data: Mapping[str, Any],
    *,
    internal_decode_threshold: float,
) -> dict[str, Any]:
    """Hard-validate the complete Stage-P decode-lock contract for Stage F."""

    for role, path in (
        ("Stage-P threshold lock", threshold_lock_path),
        ("Stage-P train manifest", stage_p_train_manifest_path),
    ):
        if not path.is_file():
            raise FineTuneInputError(f"{role} is absent: {path}", 2)
    try:
        lock = json.loads(threshold_lock_path.read_bytes())
        train_manifest = json.loads(stage_p_train_manifest_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise FineTuneInputError("Stage-P threshold-lock contract is invalid JSON", 20) from exc
    if not isinstance(lock, dict) or not isinstance(train_manifest, dict):
        raise FineTuneInputError("Stage-P threshold-lock artifacts must be objects", 20)

    lock_sha = sha256_file(threshold_lock_path)
    checkpoint_sha = sha256_file(init_checkpoint_path)
    expected_lock_path = train_manifest.get("decode_threshold_lock")
    if expected_lock_path is None or Path(str(expected_lock_path)).resolve() != threshold_lock_path.resolve():
        raise FineTuneInputError(
            "Stage-P train manifest does not name the supplied threshold lock", 23
        )
    if train_manifest.get("decode_threshold_lock_sha256") != lock_sha:
        raise FineTuneInputError(
            "Stage-P threshold-lock SHA does not match train_manifest.json", 23
        )
    if (
        lock.get("artifact_type") != "event_head_stage_p_decode_threshold_lock"
        or int(lock.get("schema_version", -1)) != 1
        or lock.get("status") != "locked_from_stage_p_internal_validation"
        or lock.get("owner_val_used") is not False
        or lock.get("internal_validation_policy")
        != "sha256_seeded_source_video_holdout"
        or list(lock.get("internal_validation_source_videos", []))
        != [REGISTERED_STAGE_P_HELD_OUT_SOURCE]
        or tuple(float(value) for value in lock.get("threshold_grid", []))
        != REGISTERED_THRESHOLD_GRID
        or tuple(lock.get("threshold_tie_break", []))
        != REGISTERED_THRESHOLD_TIE_BREAK
        or int(lock.get("nms_radius_frames", -1)) != 2
        or int(lock.get("match_tolerance_frames", -1)) != 2
    ):
        raise FineTuneInputError(
            "Stage-P threshold lock diverges from the registered grid/NMS/tie-break/split",
            23,
        )

    threshold = float(lock.get("threshold", math.nan))
    checkpoint_step = int(lock.get("checkpoint_step", -1))
    payload_step = int(checkpoint_payload_data.get("completed_steps", -1))
    train_best_path = Path(str(train_manifest.get("best_checkpoint", "")))
    if (
        not math.isfinite(threshold)
        or threshold != internal_decode_threshold
        or threshold != float(train_manifest.get("best_validation_threshold", math.nan))
        or threshold != float(train_manifest.get("locked_decode_threshold", math.nan))
        or threshold != float(checkpoint_payload_data.get("best_validation_threshold", math.nan))
        or checkpoint_step != int(train_manifest.get("best_validation_step", -2))
        or checkpoint_step != payload_step
        or lock.get("checkpoint_sha256") != checkpoint_sha
        or not train_best_path.is_file()
        or sha256_file(train_best_path) != checkpoint_sha
    ):
        raise FineTuneInputError(
            "Stage-P threshold/checkpoint/step cross-lock mismatch", 23
        )

    data_manifest_path = Path(str(train_manifest.get("data_manifest", "")))
    if (
        not data_manifest_path.is_file()
        or sha256_file(data_manifest_path)
        != train_manifest.get("data_manifest_sha256")
        or lock.get("data_manifest_sha256")
        != train_manifest.get("data_manifest_sha256")
    ):
        raise FineTuneInputError("Stage-P data-manifest cross-SHA mismatch", 23)
    return {
        "threshold_lock_path": str(threshold_lock_path),
        "threshold_lock_sha256": lock_sha,
        "stage_p_train_manifest": str(stage_p_train_manifest_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint_step,
        "threshold": threshold,
        "threshold_grid": list(REGISTERED_THRESHOLD_GRID),
        "threshold_tie_break": list(REGISTERED_THRESHOLD_TIE_BREAK),
        "nms_radius_frames": 2,
    }


class DeterministicWrappedBatchSampler(Sampler[list[int]]):
    """Seeded full-batch sampler with deterministic reshuffle/top-up.

    Every yielded batch has exactly ``batch_size`` indices.  When an epoch's
    permutation is exhausted, a fresh seeded permutation supplies the top-up;
    its unused suffix is carried into the next DataLoader iteration.
    """

    def __init__(self, item_count: int, batch_size: int, seed: int) -> None:
        if item_count < 1 or batch_size < 1:
            raise ValueError("wrapped sampler requires positive item count and batch size")
        self.item_count = item_count
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self._pending: list[int] = []

    def __len__(self) -> int:
        return math.ceil(self.item_count / self.batch_size)

    def __iter__(self) -> Iterable[list[int]]:
        for _ in range(len(self)):
            while len(self._pending) < self.batch_size:
                self._pending.extend(
                    torch.randperm(
                        self.item_count, generator=self.generator
                    ).tolist()
                )
            batch = self._pending[:self.batch_size]
            del self._pending[:self.batch_size]
            if len(batch) != self.batch_size:
                raise AssertionError("wrapped sampler emitted a partial batch")
            yield batch


def _loader(
    windows: Sequence[WeightedWindow],
    *,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    exact_batch_size: bool = False,
) -> DataLoader:
    dataset = WeightedEventWindowDataset(windows, image_size=image_size)
    if exact_batch_size:
        if shuffle is not True:
            raise ValueError("exact wrapped batches require seeded shuffling")
        return DataLoader(
            dataset,
            batch_sampler=DeterministicWrappedBatchSampler(
                len(windows), batch_size, seed
            ),
            num_workers=num_workers,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )
    return DataLoader(
        dataset,
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
        "frames", "targets", "validity_mask", "frame_loss_mask",
        "event_subframe_offsets", "sample_weight", "is_pseudo",
        "is_hard_negative", "row_index",
    ):
        merged[key] = torch.cat((owner_batch[key], pseudo_batch[key]), dim=0)
    return merged


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    best_f1: float | None,
    best_probability: float | None,
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


def _validate_registered_final_step_arguments(
    arguments: Mapping[str, Any],
) -> None:
    """Reject any nonregistered Stage-F invocation before touching inputs."""

    probe_only = arguments["probe_only"] is True
    registered = {
        "--steps": (
            arguments["steps"],
            REGISTERED_STAGE_F_PROBE_STEPS
            if probe_only else REGISTERED_STAGE_F_STEPS,
        ),
        "--image-size": (arguments["image_size"], REGISTERED_STAGE_F_IMAGE_SIZE),
        "--window-frames": (
            arguments["window_frames"], REGISTERED_STAGE_F_WINDOW_FRAMES,
        ),
        "--batch-size": (arguments["batch_size"], REGISTERED_OWNER_BATCH_SIZE),
        "--lr": (arguments["lr"], REGISTERED_STAGE_F_LR),
        "--val-every": (arguments["val_every"], REGISTERED_STAGE_F_VAL_EVERY),
        "--seed": (arguments["seed"], REGISTERED_STAGE_F_SEED),
        "--stride-frames": (
            arguments["stride_frames"], REGISTERED_STAGE_F_STRIDE_FRAMES,
        ),
        "--num-workers": (
            arguments["num_workers"], REGISTERED_STAGE_F_NUM_WORKERS,
        ),
        "--class-weights": (
            tuple(arguments["class_weights"]), DEFAULT_CLASS_WEIGHTS,
        ),
        "--pseudo-weight-cap": (
            arguments["pseudo_weight_cap"], DEFAULT_PSEUDO_WEIGHT_CAP,
        ),
        "--hard-negative-batch-size": (
            arguments["hard_negative_batch_size"],
            REGISTERED_HARD_NEGATIVE_BATCH_SIZE,
        ),
        "--hard-negative-expected-candidates": (
            arguments["hard_negative_expected_candidates"],
            REGISTERED_HARD_NEGATIVE_CANDIDATES,
        ),
        "--hard-negative-top-k": (
            arguments["hard_negative_top_k"], REGISTERED_HARD_NEGATIVE_TOP_K,
        ),
        "--class-weighting": (arguments["class_weighting"], "sqrt-frequency"),
        "--assignment-mode": (arguments["assignment_mode"], "fixed"),
        "--assignment-max-shift-frames": (
            arguments["assignment_max_shift_frames"], 0,
        ),
        "--assignment-class-cost-weight": (
            arguments["assignment_class_cost_weight"], 1.0,
        ),
        "--assignment-temporal-cost-weight": (
            arguments["assignment_temporal_cost_weight"], 0.25,
        ),
        "--label-dilation-frames": (arguments["label_dilation_frames"], 1),
        "--label-neighbor-positive-weight": (
            arguments["label_neighbor_positive_weight"], 0.5,
        ),
        "--offset-loss-weight": (arguments["offset_loss_weight"], 0.2),
        "--offset-smooth-l1-beta": (
            arguments["offset_smooth_l1_beta"], 1.0,
        ),
        "--hard-negative-loss-cap": (arguments["hard_negative_loss_cap"], 0.5),
        "--expected-owner-train-negative-rows": (
            arguments["expected_owner_train_negative_rows"],
            REGISTERED_OWNER_TRAIN_NEGATIVE_ROWS,
        ),
        "--internal-owner-negative-max-fp": (
            arguments["internal_owner_negative_max_fp"],
            REGISTERED_OWNER_NEGATIVE_MAX_FP,
        ),
        "--internal-audio-only-max-fired-rows": (
            arguments["internal_audio_only_max_fired_rows"],
            REGISTERED_AUDIO_ONLY_MAX_FIRED_ROWS,
        ),
        "--internal-rate-min-per-s": (
            arguments["internal_rate_min_per_s"], REGISTERED_RATE_MIN_PER_S,
        ),
        "--internal-rate-max-per-s": (
            arguments["internal_rate_max_per_s"], REGISTERED_RATE_MAX_PER_S,
        ),
        "--expected-owner-train-media-paths": (
            arguments["expected_owner_train_media_paths"],
            REGISTERED_TRAIN_MEDIA_PATHS,
        ),
        "--expected-owner-train-source-videos": (
            arguments["expected_owner_train_source_videos"],
            len(REGISTERED_TRAIN_SOURCE_VIDEOS),
        ),
        "--expected-owner-train-rows": (
            arguments["expected_owner_train_rows"], EXPECTED_OWNER_TRAIN_ROWS,
        ),
        "--expected-owner-val-rows": (
            arguments["expected_owner_val_rows"], EXPECTED_OWNER_VAL_ROWS,
        ),
    }
    divergent = [
        f"{name}={actual!r} (registered {expected!r})"
        for name, (actual, expected) in registered.items()
        if actual != expected
    ]
    if arguments["pseudo_manifest_path"] is not None:
        divergent.append("--pseudo-manifest is forbidden in the registered Stage-F arm")
    if tuple(sorted(map(str, arguments["hard_negative_excluded_source_video_ids"]))) != (
        REGISTERED_STAGE_P_HELD_OUT_SOURCE,
    ):
        divergent.append(
            "--hard-negative-excluded-source-video must be exactly "
            + REGISTERED_STAGE_P_HELD_OUT_SOURCE
        )
    if tuple(sorted(map(str, arguments["owner_train_source_video_ids"]))) != tuple(
        sorted(REGISTERED_TRAIN_SOURCE_VIDEOS)
    ):
        divergent.append("--owner-train-source-video set diverges from registration")
    if tuple(sorted(map(str, arguments["owner_validation_source_video_ids"]))) != tuple(
        sorted(REGISTERED_VALIDATION_SOURCE_VIDEOS)
    ):
        divergent.append(
            "--owner-validation-source-video set diverges from registration"
        )
    maximum_wall = arguments["max_wall_minutes"]
    if (
        not isinstance(maximum_wall, (int, float))
        or isinstance(maximum_wall, bool)
        or not math.isfinite(float(maximum_wall))
        or float(maximum_wall) <= 0
        or float(maximum_wall) > REGISTERED_STAGE_F_MAX_WALL_MINUTES
        or (probe_only and float(maximum_wall) != REGISTERED_STAGE_F_MAX_WALL_MINUTES)
    ):
        divergent.append(
            "--max-wall-minutes must be 180 for the probe and in (0,180] "
            "for the measured full-stage cap"
        )
    threshold = arguments["internal_decode_threshold"]
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not 0.0 < float(threshold) < 1.0
    ):
        divergent.append(
            "--internal-decode-threshold must be the finite Stage-P lock in (0,1)"
        )
    required_paths = {
        "--hard-negative-invalid-manifest": arguments[
            "hard_negative_invalid_manifest_path"
        ],
        "--hard-negative-repaired-manifest": arguments[
            "hard_negative_repaired_manifest_path"
        ],
        "--stage-p-threshold-lock": arguments["stage_p_threshold_lock_path"],
        "--stage-p-train-manifest": arguments["stage_p_train_manifest_path"],
        "--owner-media-root": arguments["owner_media_root"],
        "--rate-media-inventory": arguments["rate_media_inventory_path"],
    }
    divergent.extend(
        f"{name} is required" for name, value in required_paths.items() if value is None
    )
    required_pins = {
        "--owner-manifest-sha256": arguments["owner_manifest_sha256"],
        "--init-checkpoint-sha256": arguments["init_checkpoint_sha256"],
        "--hard-negative-invalid-manifest-sha256": arguments[
            "hard_negative_invalid_manifest_sha256"
        ],
        "--hard-negative-repaired-manifest-sha256": arguments[
            "hard_negative_repaired_manifest_sha256"
        ],
        "--rate-media-inventory-sha256": arguments[
            "rate_media_inventory_sha256"
        ],
    }
    divergent.extend(
        f"{name} is required" for name, value in required_pins.items() if value is None
    )
    if divergent:
        raise FineTuneInputError(
            "final-step Stage-F recipe diverges from registered E-v2 before input read: "
            + "; ".join(divergent),
            20,
        )


def _validate_parsed_final_step_arguments(args: argparse.Namespace) -> None:
    """Apply the complete recipe lock to CLI arguments before gate input reads."""

    _validate_registered_final_step_arguments({
        **vars(args),
        "pseudo_manifest_path": args.pseudo_manifest,
        "hard_negative_invalid_manifest_path": (
            args.hard_negative_invalid_manifest
        ),
        "hard_negative_repaired_manifest_path": (
            args.hard_negative_repaired_manifest
        ),
        "hard_negative_excluded_source_video_ids": (
            args.hard_negative_excluded_source_video
        ),
        "stage_p_threshold_lock_path": args.stage_p_threshold_lock,
        "stage_p_train_manifest_path": args.stage_p_train_manifest,
        "rate_media_inventory_path": args.rate_media_inventory,
        "owner_train_source_video_ids": args.owner_train_source_video,
        "owner_validation_source_video_ids": (
            args.owner_validation_source_video
        ),
    })


def _stage_f_wall_has_expired(
    total_started: float, max_wall_minutes: float | None
) -> bool:
    """Use one clock boundary for mining plus every optimizer update."""

    return (
        max_wall_minutes is not None
        and time.monotonic() - total_started >= max_wall_minutes * 60
    )


def _enforce_stage_f_post_optimizer_wall(
    total_started: float, max_wall_minutes: float | None
) -> None:
    """Hard-abort before any post-update validation or guard workload."""

    if _stage_f_wall_has_expired(total_started, max_wall_minutes):
        raise FineTuneInputError(
            "STAGE_F_OPTIMIZER_WALL_EXPIRED: cap reached immediately after "
            "optimizer.step(); terminal guards are forbidden",
            31,
        )


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
    checkpoint_selection: Literal["owner-val", "final-step"] = "owner-val",
    owner_manifest_sha256: str | None = None,
    init_checkpoint_sha256: str | None = None,
    hard_negative_invalid_manifest_path: Path | None = None,
    hard_negative_repaired_manifest_path: Path | None = None,
    hard_negative_invalid_manifest_sha256: str | None = None,
    hard_negative_repaired_manifest_sha256: str | None = None,
    hard_negative_expected_candidates: int = REGISTERED_HARD_NEGATIVE_CANDIDATES,
    hard_negative_top_k: int = REGISTERED_HARD_NEGATIVE_TOP_K,
    hard_negative_batch_size: int = REGISTERED_HARD_NEGATIVE_BATCH_SIZE,
    hard_negative_excluded_source_video_ids: Sequence[str] = (),
    hard_negative_loss_cap: float = DEFAULT_HARD_NEGATIVE_LOSS_CAP,
    class_weighting: Literal["fixed", "sqrt-frequency"] = "fixed",
    assignment_mode: Literal["legacy", "fixed", "hungarian"] = "legacy",
    assignment_max_shift_frames: int = 0,
    assignment_class_cost_weight: float = 1.0,
    assignment_temporal_cost_weight: float = 1.0,
    label_dilation_frames: int = 0,
    label_neighbor_positive_weight: float = 0.5,
    offset_loss_weight: float = 0.0,
    offset_smooth_l1_beta: float = 1.0,
    internal_decode_threshold: float | None = None,
    stage_p_threshold_lock_path: Path | None = None,
    stage_p_train_manifest_path: Path | None = None,
    owner_media_root: Path | None = None,
    rate_media_inventory_path: Path | None = None,
    rate_media_inventory_sha256: str | None = None,
    owner_train_source_video_ids: Sequence[str] = (),
    owner_validation_source_video_ids: Sequence[str] = (),
    expected_owner_train_negative_rows: int = 21,
    internal_owner_negative_max_fp: int = 2,
    internal_audio_only_max_fired_rows: int = REGISTERED_AUDIO_ONLY_MAX_FIRED_ROWS,
    internal_rate_min_per_s: float = 0.3,
    internal_rate_max_per_s: float = 1.0,
    expected_owner_train_media_paths: int = REGISTERED_TRAIN_MEDIA_PATHS,
    expected_owner_train_source_videos: int = 4,
    probe_only: bool = False,
    expected_owner_train_rows: int = EXPECTED_OWNER_TRAIN_ROWS,
    expected_owner_val_rows: int = EXPECTED_OWNER_VAL_ROWS,
    max_wall_minutes: float | None = None,
) -> dict[str, Any]:
    if checkpoint_selection == "final-step":
        # This is intentionally the first act of final-step mode. It must remain
        # ahead of mkdir/unlink, manifest/media/checkpoint reads, and lock reads.
        _validate_registered_final_step_arguments(locals())
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
    if checkpoint_selection not in {"owner-val", "final-step"}:
        raise FineTuneInputError("invalid --checkpoint-selection", 20)
    if probe_only and checkpoint_selection != "final-step":
        raise FineTuneInputError(
            "--probe-only is valid only with --checkpoint-selection final-step",
            20,
        )
    if class_weighting not in {"fixed", "sqrt-frequency"}:
        raise FineTuneInputError("invalid --class-weighting", 20)
    if assignment_mode not in {"legacy", "fixed", "hungarian"}:
        raise FineTuneInputError("invalid --assignment-mode", 20)
    if label_dilation_frames not in {0, 1}:
        raise FineTuneInputError("--label-dilation-frames must be 0 or 1", 20)
    if (
        not math.isfinite(label_neighbor_positive_weight)
        or not 0 < label_neighbor_positive_weight <= 1
    ):
        raise FineTuneInputError(
            "--label-neighbor-positive-weight must be finite and in (0,1]", 20
        )
    if assignment_max_shift_frames < 0:
        raise FineTuneInputError(
            "--assignment-max-shift-frames must be nonnegative", 20
        )
    for name, value in (
        ("assignment-class-cost-weight", assignment_class_cost_weight),
        ("assignment-temporal-cost-weight", assignment_temporal_cost_weight),
        ("offset-loss-weight", offset_loss_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise FineTuneInputError(f"--{name} must be finite and nonnegative", 20)
    if not math.isfinite(offset_smooth_l1_beta) or offset_smooth_l1_beta <= 0:
        raise FineTuneInputError(
            "--offset-smooth-l1-beta must be finite and positive", 20
        )
    if not math.isfinite(hard_negative_loss_cap) or hard_negative_loss_cap <= 0:
        raise FineTuneInputError(
            "--hard-negative-loss-cap must be finite and positive", 20
        )
    if checkpoint_selection != "final-step" and any((
        hard_negative_invalid_manifest_path,
        hard_negative_repaired_manifest_path,
        hard_negative_invalid_manifest_sha256,
        hard_negative_repaired_manifest_sha256,
    )):
        raise FineTuneInputError(
            "hard-negative mining is available only with --checkpoint-selection final-step",
            20,
        )

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "finetune_manifest.json"
    manifest_temporary_path = out / ".finetune_manifest.json.tmp"
    # Reusing an output directory starts a new arm attempt. Remove the prior
    # completion claim before validation, decode, checkpoint load, or training
    # can be interrupted; only the final atomic replace may recreate it.
    manifest_path.unlink(missing_ok=True)
    manifest_temporary_path.unlink(missing_ok=True)

    owner, owner_raw, pseudo, pseudo_raw = _load_finetune_manifests(
        checkpoint_selection=checkpoint_selection,
        owner_manifest_path=owner_manifest_path,
        pseudo_manifest_path=pseudo_manifest_path,
        owner_manifest_sha256=owner_manifest_sha256,
        window_frames=window_frames,
        expected_owner_train_rows=expected_owner_train_rows,
        expected_owner_val_rows=expected_owner_val_rows,
    )
    pretrain_payload = _assert_checkpoint_context(
        init_checkpoint_model_only,
        window_frames=window_frames,
        image_size=image_size,
        expected_sha256=init_checkpoint_sha256,
    )
    stage_p_locked_threshold: float | None = None
    stage_p_threshold_contract: dict[str, Any] | None = None
    if checkpoint_selection == "final-step":
        assert stage_p_threshold_lock_path is not None
        assert stage_p_train_manifest_path is not None
        assert internal_decode_threshold is not None
        stage_p_threshold_contract = validate_stage_p_threshold_lock(
            stage_p_threshold_lock_path,
            stage_p_train_manifest_path,
            init_checkpoint_model_only,
            pretrain_payload,
            internal_decode_threshold=internal_decode_threshold,
        )
        stage_p_locked_threshold = float(stage_p_threshold_contract["threshold"])
    owner_train = _training_windows(
        owner, role="owner", window_frames=window_frames,
        stride_frames=stride_frames,
    )
    owner_val = (
        _validation_windows(owner, window_frames=window_frames)
        if checkpoint_selection == "owner-val" else []
    )
    pseudo_train = (
        _training_windows(
            pseudo, role="pseudo", window_frames=window_frames,
            stride_frames=stride_frames,
        )
        if pseudo is not None else []
    )
    if (
        checkpoint_selection == "owner-val"
        and len(owner_val) != expected_owner_val_rows
    ):
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
        exact_batch_size=checkpoint_selection == "final-step",
    )
    pseudo_loader = (
        _loader(
            pseudo_train, image_size=image_size, batch_size=batch_size, shuffle=True,
            seed=seed + 1, num_workers=num_workers,
        )
        if pseudo_train else None
    )
    val_loader = (
        _loader(
            owner_val,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=num_workers,
        )
        if owner_val else None
    )
    state = _initialize_training_state(
        device=device,
        device_name=device_name,
        weights="none",
        lr=lr,
        init_checkpoint=None,
        init_checkpoint_model_only=init_checkpoint_model_only,
        offset_regression_head=(
            offset_loss_weight > 0
            or bool(
                (pretrain_payload.get("model_config") or {}).get(
                    "offset_regression_head", False
                )
            )
        ),
    )
    if state.resume_mode != "model_only" or state.optimizer_state_restored:
        raise RuntimeError("fine-tune must load model-only with a fresh optimizer")
    model, optimizer = state.model, state.optimizer

    total_started = time.monotonic()
    hard_negative_candidates: list[HardNegativeCandidate] = []
    hard_negative_pool_report: dict[str, Any] | None = None
    hard_negative_mining: dict[str, Any] | None = None
    hard_negative_train: list[WeightedWindow] = []
    if checkpoint_selection == "final-step":
        assert hard_negative_invalid_manifest_path is not None
        assert hard_negative_repaired_manifest_path is not None
        assert hard_negative_invalid_manifest_sha256 is not None
        assert hard_negative_repaired_manifest_sha256 is not None
        hard_negative_candidates, hard_negative_pool_report = (
            derive_audio_only_hard_negative_pool(
                hard_negative_invalid_manifest_path,
                hard_negative_repaired_manifest_path,
                invalid_manifest_sha256=hard_negative_invalid_manifest_sha256,
                repaired_manifest_sha256=hard_negative_repaired_manifest_sha256,
                expected_candidates=hard_negative_expected_candidates,
                window_frames=window_frames,
                excluded_source_video_ids=(
                    hard_negative_excluded_source_video_ids
                ),
                expected_raw_candidates=292,
                expected_excluded_source_rows=30,
            )
        )
        if any(
            candidate.source_video_id == REGISTERED_STAGE_P_HELD_OUT_SOURCE
            for candidate in hard_negative_candidates
        ):
            raise AssertionError(
                "Stage-P held-out source entered final-step hard-negative candidates"
            )
        hard_negative_train, hard_negative_mining = mine_hard_negatives(
            model,
            hard_negative_candidates,
            top_k=hard_negative_top_k,
            image_size=image_size,
            batch_size=hard_negative_batch_size,
            device=device,
            num_workers=num_workers,
            seed=seed,
        )
    hard_negative_loader = (
        _loader(
            hard_negative_train,
            image_size=image_size,
            batch_size=hard_negative_batch_size,
            shuffle=True,
            seed=seed + 2,
            num_workers=num_workers,
            exact_batch_size=checkpoint_selection == "final-step",
        )
        if hard_negative_train else None
    )
    class_counts: tuple[float, float, float] | None = None
    effective_class_weights = tuple(float(value) for value in class_weights)
    if class_weighting == "sqrt-frequency":
        class_counts = dense_class_counts(
            [window.spec for window in owner_train + pseudo_train + hard_negative_train],
            label_dilation_frames=label_dilation_frames,
            neighbor_positive_weight=label_neighbor_positive_weight,
        )
        effective_class_weights = sqrt_frequency_class_weights(class_counts)
    loss_class_weights = torch.tensor(
        effective_class_weights, dtype=torch.float32, device=device
    )
    config = {
        "device": device_name,
        "steps": steps,
        "image_size": image_size,
        "window_frames": window_frames,
        "batch_size_human": batch_size,
        "batch_size_pseudo_max": batch_size if pseudo_train else 0,
        "batch_size_hard_negative": (
            hard_negative_batch_size if hard_negative_train else 0
        ),
        "owner_batch_sampler_policy": (
            "seeded_permutation_with_deterministic_reshuffle_wrap_top_up_exact_8"
            if checkpoint_selection == "final-step" else "legacy_drop_last_false"
        ),
        "lr": lr,
        "val_every": val_every,
        "seed": seed,
        "stride_frames": stride_frames,
        "num_workers": num_workers,
        "class_weighting": class_weighting,
        "class_counts": list(class_counts) if class_counts is not None else None,
        "class_weights": list(effective_class_weights),
        "fixed_class_weights_argument": list(class_weights),
        "pseudo_weight_cap": pseudo_weight_cap,
        "hard_negative_loss_cap": hard_negative_loss_cap,
        "hard_negative_expected_candidates": hard_negative_expected_candidates,
        "hard_negative_top_k": hard_negative_top_k,
        "hard_negative_excluded_source_video_ids": sorted(
            map(str, hard_negative_excluded_source_video_ids)
        ),
        "checkpoint_selection": checkpoint_selection,
        "probe_only": probe_only,
        "assignment_mode": assignment_mode,
        "assignment_max_shift_frames": assignment_max_shift_frames,
        "assignment_class_cost_weight": assignment_class_cost_weight,
        "assignment_temporal_cost_weight": assignment_temporal_cost_weight,
        "label_dilation_frames": label_dilation_frames,
        "label_neighbor_positive_weight": label_neighbor_positive_weight,
        "offset_loss_weight": offset_loss_weight,
        "offset_smooth_l1_beta": offset_smooth_l1_beta,
        "expected_owner_train_rows": expected_owner_train_rows,
        "expected_owner_val_rows": expected_owner_val_rows,
        "validation_threshold": (
            0.5 if checkpoint_selection == "owner-val" else None
        ),
        "validation_tolerance_frames": (
            2 if checkpoint_selection == "owner-val" else None
        ),
        "internal_decode_threshold": internal_decode_threshold,
        "stage_p_locked_decode_threshold": stage_p_locked_threshold,
        "stage_p_threshold_contract": stage_p_threshold_contract,
        "internal_nms_radius_frames": 2 if checkpoint_selection == "final-step" else None,
        "internal_guard_bounds": (
            {
                "expected_owner_train_negative_rows": expected_owner_train_negative_rows,
                "owner_negative_max_fp": internal_owner_negative_max_fp,
                "audio_only_max_fired_rows": internal_audio_only_max_fired_rows,
                "rate_min_per_s": internal_rate_min_per_s,
                "rate_max_per_s": internal_rate_max_per_s,
                "expected_owner_train_media_paths": expected_owner_train_media_paths,
                "expected_owner_train_source_videos": expected_owner_train_source_videos,
                "owner_media_root": str(owner_media_root),
                "rate_media_inventory_path": str(rate_media_inventory_path),
                "rate_media_inventory_sha256": rate_media_inventory_sha256,
                "train_source_video_ids": sorted(
                    map(str, owner_train_source_video_ids)
                ),
                "validation_source_video_ids": sorted(
                    map(str, owner_validation_source_video_ids)
                ),
            }
            if checkpoint_selection == "final-step" else None
        ),
        "max_wall_minutes": max_wall_minutes,
        "max_wall_scope": (
            "hard_negative_mining_plus_optimizer"
            if checkpoint_selection == "final-step" else "optimizer"
        ),
    }
    provenance = {
        "owner_manifest": str(owner_manifest_path),
        "owner_manifest_sha256": hashlib.sha256(owner_raw).hexdigest(),
        "owner_manifest_registered_sha256": owner_manifest_sha256,
        "pseudo_manifest": str(pseudo_manifest_path) if pseudo_manifest_path else None,
        "pseudo_manifest_sha256": (
            hashlib.sha256(pseudo_raw).hexdigest() if pseudo_raw is not None else None
        ),
        "init_checkpoint_model_only": str(init_checkpoint_model_only),
        "init_checkpoint_sha256": sha256_file(init_checkpoint_model_only),
        "init_checkpoint_registered_sha256": init_checkpoint_sha256,
        "init_checkpoint_completed_steps": int(
            pretrain_payload.get("completed_steps", 0)
        ),
        "stage_p_locked_decode_threshold": stage_p_locked_threshold,
        "stage_p_threshold_contract": stage_p_threshold_contract,
        "git_head": _git_head(),
        "protected_inventory_opened": checkpoint_selection != "final-step",
        "owner_validation_row_fields_accessed": (
            ["split"] if checkpoint_selection == "final-step" else "legacy_validation"
        ),
    }
    if hard_negative_pool_report is not None:
        provenance["hard_negative_pool"] = hard_negative_pool_report
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
    weight_batches: list[dict[str, float | bool | int]] = []
    completed_steps = 0
    wall_stopped = False
    owner_iterator = iter(owner_loader)
    pseudo_iterator = iter(pseudo_loader) if pseudo_loader is not None else None
    hard_negative_iterator = (
        iter(hard_negative_loader) if hard_negative_loader is not None else None
    )

    best_f1: float | None = None
    best_probability: float | None = None
    if checkpoint_selection == "owner-val":
        if val_loader is None:
            raise AssertionError("owner-val selection requires a validation loader")
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
        if _stage_f_wall_has_expired(total_started, max_wall_minutes):
            wall_stopped = True
            break
        owner_batch, owner_iterator = _next_batch(owner_iterator, owner_loader)
        if (
            checkpoint_selection == "final-step"
            and int(owner_batch["frames"].shape[0]) != REGISTERED_OWNER_BATCH_SIZE
        ):
            raise RuntimeError(
                "registered Stage-F owner sampler emitted a non-8 batch"
            )
        pseudo_batch = None
        if pseudo_loader is not None and pseudo_iterator is not None:
            pseudo_batch, pseudo_iterator = _next_batch(
                pseudo_iterator, pseudo_loader
            )
        batch = _merge_batches(owner_batch, pseudo_batch)
        if hard_negative_loader is not None and hard_negative_iterator is not None:
            hard_negative_batch, hard_negative_iterator = _next_batch(
                hard_negative_iterator, hard_negative_loader
            )
            if (
                checkpoint_selection == "final-step"
                and int(hard_negative_batch["frames"].shape[0])
                != REGISTERED_HARD_NEGATIVE_BATCH_SIZE
            ):
                raise RuntimeError(
                    "registered Stage-F hard-negative sampler emitted a non-4 batch"
                )
            batch = _merge_batches(batch, hard_negative_batch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if assignment_mode == "legacy":
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
        else:
            frames = batch["frames"].to(device)
            predicted_offsets: torch.Tensor | None
            if offset_loss_weight > 0:
                logits, predicted_offsets = model.forward_with_aux(frames)  # type: ignore[attr-defined]
            else:
                logits = model(frames)
                predicted_offsets = None
            loss, weighting = assignment_recipe_loss(
                logits,
                batch["targets"].to(device),
                batch["validity_mask"].to(device),
                batch["frame_loss_mask"].to(device),
                batch["event_subframe_offsets"].to(device),
                predicted_offsets=predicted_offsets,
                class_weights=loss_class_weights,
                sample_weights=batch["sample_weight"].to(device),
                is_pseudo=batch["is_pseudo"].to(device),
                is_hard_negative=batch["is_hard_negative"].to(device),
                pseudo_weight_cap=pseudo_weight_cap,
                hard_negative_loss_cap=hard_negative_loss_cap,
                assignment_mode=assignment_mode,
                assignment_max_shift_frames=assignment_max_shift_frames,
                assignment_class_cost_weight=assignment_class_cost_weight,
                assignment_temporal_cost_weight=assignment_temporal_cost_weight,
                label_dilation_frames=label_dilation_frames,
                neighbor_positive_weight=label_neighbor_positive_weight,
                offset_loss_weight=offset_loss_weight,
                offset_smooth_l1_beta=offset_smooth_l1_beta,
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
        # The optimizer/mining cap is checked at the exact post-update boundary,
        # before validation or terminal train-side guards can begin.
        _enforce_stage_f_post_optimizer_wall(total_started, max_wall_minutes)
        if (
            checkpoint_selection == "owner-val"
            and (completed_steps % val_every == 0 or completed_steps == steps)
        ):
            if val_loader is None or best_f1 is None or best_probability is None:
                raise AssertionError("owner-val selection state is incomplete")
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

    elapsed_training_s = time.monotonic() - started
    internal_guards: dict[str, Any] | None = None
    owner_score_eligible: bool | None = None
    if checkpoint_selection == "final-step":
        if probe_only:
            internal_guards = {
                "policy": "skipped_for_registered_throughput_probe",
                "status": "not_run_probe_only",
                "pass": None,
                "owner_validation_constructed": False,
                "owner_validation_scored": False,
                "protected_inventory_opened": False,
            }
            owner_score_eligible = False
        else:
            if internal_decode_threshold is None:
                raise AssertionError("final-step guard threshold is missing")
            if owner_media_root is None:
                raise AssertionError("final-step owner-media root is missing")
            if rate_media_inventory_path is None or rate_media_inventory_sha256 is None:
                raise AssertionError("final-step rate-media inventory lock is missing")
            internal_guards = run_internal_stage_f_guards(
                model,
                owner,
                hard_negative_candidates,
                image_size=image_size,
                window_frames=window_frames,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
                seed=seed,
                threshold=internal_decode_threshold,
                owner_media_root=owner_media_root,
                train_source_video_ids=owner_train_source_video_ids,
                validation_source_video_ids=owner_validation_source_video_ids,
                expected_owner_negative_rows=expected_owner_train_negative_rows,
                owner_negative_max_fp=internal_owner_negative_max_fp,
                audio_only_max_fired_rows=internal_audio_only_max_fired_rows,
                rate_min_per_s=internal_rate_min_per_s,
                rate_max_per_s=internal_rate_max_per_s,
                expected_train_media_paths=expected_owner_train_media_paths,
                expected_train_source_videos=expected_owner_train_source_videos,
                rate_media_inventory_path=rate_media_inventory_path,
                rate_media_inventory_sha256=rate_media_inventory_sha256,
            )
            owner_score_eligible = bool(internal_guards["pass"])
        config["internal_guards_pass"] = internal_guards["pass"]
        config["owner_score_eligible"] = owner_score_eligible
        checkpoint_role = (
            "terminal_step_probe_only_not_owner_score_eligible"
            if probe_only else
            "terminal_step_internal_guards_pass"
            if owner_score_eligible else
            "terminal_step_internal_guards_fail_not_owner_score_eligible"
        )
    else:
        checkpoint_role = "last"

    _save_checkpoint(
        last_path, model=model, optimizer=optimizer,
        completed_steps=completed_steps, best_f1=best_f1,
        best_probability=best_probability, config=config,
        provenance=provenance, license_posture=license_posture,
        license_reason=license_reason, role=checkpoint_role,
    )
    if checkpoint_selection == "final-step":
        _save_checkpoint(
            best_path, model=model, optimizer=optimizer,
            completed_steps=completed_steps, best_f1=None,
            best_probability=None, config=config,
            provenance=provenance, license_posture=license_posture,
            license_reason=license_reason, role=checkpoint_role,
        )
    elapsed_total_s = time.monotonic() - total_started

    def _maximum_stat(name: str) -> float:
        return max(
            (float(item.get(name, 0.0)) for item in weight_batches),
            default=0.0,
        )

    effective_pseudo_fractions = []
    for item in weight_batches:
        if "effective_pseudo_loss_fraction" in item:
            effective_pseudo_fractions.append(
                float(item["effective_pseudo_loss_fraction"])
            )
            continue
        pseudo_loss = float(item.get("effective_pseudo_loss", 0.0))
        human_loss = float(item.get("reference_human_loss_for_pseudo", 0.0))
        denominator = pseudo_loss + human_loss
        effective_pseudo_fractions.append(
            pseudo_loss / denominator if denominator > 0 else 0.0
        )
    batch_weighting = {
        "owner_row_weight": 1.0,
        "pseudo_manifest_field": "sample_weight",
        "pseudo_loss_cap_vs_human_per_batch": pseudo_weight_cap,
        "hard_negative_row_weight": 1.0 if hard_negative_train else None,
        "hard_negative_loss_cap_vs_human_per_batch": (
            hard_negative_loss_cap if hard_negative_train else None
        ),
        "cap_basis": "post_class_and_frame_weighted_aggregate_loss",
        "raw_pseudo_loss_max": _maximum_stat("raw_pseudo_loss"),
        "effective_pseudo_loss_max": _maximum_stat("effective_pseudo_loss"),
        "effective_pseudo_loss_fraction_max": max(
            effective_pseudo_fractions, default=0.0
        ),
        "capped_batches": sum(
            bool(item.get("capped", item.get("pseudo_capped", False)))
            for item in weight_batches
        ),
        "raw_hard_negative_loss_max": _maximum_stat("raw_hard_negative_loss"),
        "effective_hard_negative_loss_max": _maximum_stat(
            "effective_hard_negative_loss"
        ),
        "hard_negative_capped_batches": sum(
            bool(item.get("hard_negative_capped", False))
            for item in weight_batches
        ),
        "assignment_event_count": sum(
            int(item.get("assignment_event_count", 0)) for item in weight_batches
        ),
        "assignment_shifted_event_count": sum(
            int(item.get("assignment_shifted_event_count", 0))
            for item in weight_batches
        ),
        "assignment_total_abs_shift": sum(
            int(item.get("assignment_total_abs_shift", 0))
            for item in weight_batches
        ),
        "batches": len(weight_batches),
    }
    result = {
        "schema_version": 2,
        "artifact_type": "event_head_finetune_manifest",
        "verified": False,
        "status": (
            "complete_probe_only" if probe_only else
            "complete" if owner_score_eligible is not False else
            "complete_internal_guard_fail"
        ),
        "honest_partial": False,
        "equal_step_eligible": True,
        "probe_only": probe_only,
        "owner_score_eligible": owner_score_eligible,
        "config": config,
        "owner_train_rows": expected_owner_train_rows,
        "owner_validation_rows": expected_owner_val_rows,
        "owner_validation_rows_uninspected": (
            expected_owner_val_rows if checkpoint_selection == "final-step" else 0
        ),
        "owner_train_windows": len(owner_train),
        "pseudo_train_rows": len(pseudo["rows"]) if pseudo is not None else 0,
        "pseudo_train_windows": len(pseudo_train),
        "hard_negative_candidate_rows": len(hard_negative_candidates),
        "hard_negative_train_windows": len(hard_negative_train),
        "hard_negative_pool": hard_negative_pool_report,
        "hard_negative_mining": hard_negative_mining,
        "validation_windows": len(owner_val),
        "validation_protocol": (
            "fixed_owner_val_only_macro_F1_at_plus_minus_2_frames"
            if checkpoint_selection == "owner-val" else
            "none_terminal_step_selection_owner_validation_unconstructed_unscored"
        ),
        "validations": validations,
        "best_val_macro_f1_at_2": best_f1,
        "best_val_max_positive_class_probability": best_probability,
        "internal_guards": internal_guards,
        "losses": losses,
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "batch_weighting": batch_weighting,
        "completed_steps": completed_steps,
        "target_steps": steps,
        "elapsed_s": elapsed_training_s,
        "elapsed_training_s": elapsed_training_s,
        "elapsed_total_s": elapsed_total_s,
        "steps_per_s": (
            completed_steps / elapsed_training_s if elapsed_training_s else 0.0
        ),
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
    parser.add_argument(
        "--gate-proof",
        type=Path,
        help=(
            "Passing, fresh gate_proof.json from verify_training_inputs.py; "
            "required before any training input read"
        ),
    )
    parser.add_argument("--owner-manifest", type=Path)
    parser.add_argument("--owner-manifest-sha256")
    parser.add_argument("--pseudo-manifest", type=Path)
    parser.add_argument("--init-checkpoint-model-only", type=Path)
    parser.add_argument("--init-checkpoint-sha256")
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
        "--checkpoint-selection",
        choices=("owner-val", "final-step"),
        default="owner-val",
    )
    parser.add_argument(
        "--probe-only", action="store_true",
        help=(
            "Run a non-score-eligible Stage-F throughput probe and skip terminal "
            "internal guards"
        ),
    )
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
        "--class-weighting",
        choices=("fixed", "sqrt-frequency"),
        default="fixed",
    )
    parser.add_argument(
        "--assignment-mode",
        choices=("legacy", "fixed", "hungarian"),
        default="legacy",
    )
    parser.add_argument("--assignment-max-shift-frames", type=int, default=0)
    parser.add_argument("--assignment-class-cost-weight", type=float, default=1.0)
    parser.add_argument("--assignment-temporal-cost-weight", type=float, default=1.0)
    parser.add_argument(
        "--label-dilation-frames", type=int, choices=(0, 1), default=0
    )
    parser.add_argument(
        "--label-neighbor-positive-weight", type=float, default=0.5
    )
    parser.add_argument("--offset-loss-weight", type=float, default=0.0)
    parser.add_argument("--offset-smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--hard-negative-invalid-manifest", type=Path)
    parser.add_argument("--hard-negative-invalid-manifest-sha256")
    parser.add_argument("--hard-negative-repaired-manifest", type=Path)
    parser.add_argument("--hard-negative-repaired-manifest-sha256")
    parser.add_argument(
        "--hard-negative-expected-candidates", type=int,
        default=REGISTERED_HARD_NEGATIVE_CANDIDATES,
    )
    parser.add_argument(
        "--hard-negative-top-k", type=int, default=REGISTERED_HARD_NEGATIVE_TOP_K
    )
    parser.add_argument(
        "--hard-negative-batch-size", type=int,
        default=REGISTERED_HARD_NEGATIVE_BATCH_SIZE,
    )
    parser.add_argument(
        "--hard-negative-excluded-source-video", action="append", default=[]
    )
    parser.add_argument(
        "--hard-negative-loss-cap",
        type=float,
        default=DEFAULT_HARD_NEGATIVE_LOSS_CAP,
    )
    parser.add_argument("--internal-decode-threshold", type=float)
    parser.add_argument("--stage-p-threshold-lock", type=Path)
    parser.add_argument("--stage-p-train-manifest", type=Path)
    parser.add_argument("--owner-media-root", type=Path)
    parser.add_argument("--rate-media-inventory", type=Path)
    parser.add_argument("--rate-media-inventory-sha256")
    parser.add_argument(
        "--owner-train-source-video", action="append", default=[]
    )
    parser.add_argument(
        "--owner-validation-source-video", action="append", default=[]
    )
    parser.add_argument(
        "--expected-owner-train-negative-rows", type=int, default=21
    )
    parser.add_argument(
        "--internal-owner-negative-max-fp", type=int, default=2
    )
    parser.add_argument(
        "--internal-audio-only-max-fired-rows", type=int,
        default=REGISTERED_AUDIO_ONLY_MAX_FIRED_ROWS,
    )
    parser.add_argument("--internal-rate-min-per-s", type=float, default=0.3)
    parser.add_argument("--internal-rate-max-per-s", type=float, default=1.0)
    parser.add_argument(
        "--expected-owner-train-media-paths", type=int,
        default=REGISTERED_TRAIN_MEDIA_PATHS,
    )
    parser.add_argument(
        "--expected-owner-train-source-videos", type=int, default=4
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
        if args.checkpoint_selection == "final-step":
            _validate_parsed_final_step_arguments(args)
        required_training_inputs = [
            path
            for path in (
                args.owner_manifest,
                args.pseudo_manifest,
                args.hard_negative_invalid_manifest,
                args.hard_negative_repaired_manifest,
                args.stage_p_train_manifest,
                args.owner_media_root,
                args.rate_media_inventory,
            )
            if path is not None
        ]
        assert_gate_proof(
            args.gate_proof,
            repo_root=ROOT,
            required_input_paths=required_training_inputs,
        )
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
            checkpoint_selection=args.checkpoint_selection,
            owner_manifest_sha256=args.owner_manifest_sha256,
            init_checkpoint_sha256=args.init_checkpoint_sha256,
            hard_negative_invalid_manifest_path=(
                args.hard_negative_invalid_manifest
            ),
            hard_negative_repaired_manifest_path=(
                args.hard_negative_repaired_manifest
            ),
            hard_negative_invalid_manifest_sha256=(
                args.hard_negative_invalid_manifest_sha256
            ),
            hard_negative_repaired_manifest_sha256=(
                args.hard_negative_repaired_manifest_sha256
            ),
            hard_negative_expected_candidates=(
                args.hard_negative_expected_candidates
            ),
            hard_negative_top_k=args.hard_negative_top_k,
            hard_negative_batch_size=args.hard_negative_batch_size,
            hard_negative_excluded_source_video_ids=(
                args.hard_negative_excluded_source_video
            ),
            hard_negative_loss_cap=args.hard_negative_loss_cap,
            class_weighting=args.class_weighting,
            assignment_mode=args.assignment_mode,
            assignment_max_shift_frames=args.assignment_max_shift_frames,
            assignment_class_cost_weight=args.assignment_class_cost_weight,
            assignment_temporal_cost_weight=(
                args.assignment_temporal_cost_weight
            ),
            label_dilation_frames=args.label_dilation_frames,
            label_neighbor_positive_weight=(
                args.label_neighbor_positive_weight
            ),
            offset_loss_weight=args.offset_loss_weight,
            offset_smooth_l1_beta=args.offset_smooth_l1_beta,
            internal_decode_threshold=args.internal_decode_threshold,
            stage_p_threshold_lock_path=args.stage_p_threshold_lock,
            stage_p_train_manifest_path=args.stage_p_train_manifest,
            owner_media_root=args.owner_media_root,
            rate_media_inventory_path=args.rate_media_inventory,
            rate_media_inventory_sha256=args.rate_media_inventory_sha256,
            owner_train_source_video_ids=args.owner_train_source_video,
            owner_validation_source_video_ids=(
                args.owner_validation_source_video
            ),
            expected_owner_train_negative_rows=(
                args.expected_owner_train_negative_rows
            ),
            internal_owner_negative_max_fp=(
                args.internal_owner_negative_max_fp
            ),
            internal_audio_only_max_fired_rows=(
                args.internal_audio_only_max_fired_rows
            ),
            internal_rate_min_per_s=args.internal_rate_min_per_s,
            internal_rate_max_per_s=args.internal_rate_max_per_s,
            expected_owner_train_media_paths=(
                args.expected_owner_train_media_paths
            ),
            expected_owner_train_source_videos=(
                args.expected_owner_train_source_videos
            ),
            probe_only=args.probe_only,
            expected_owner_train_rows=args.expected_owner_train_rows,
            expected_owner_val_rows=args.expected_owner_val_rows,
            max_wall_minutes=args.max_wall_minutes,
        )
    except GateProofError as exc:
        parser.exit(
            20,
            f"fine-tune input rejected: {exc}; refusal occurred before input read\n",
        )
    except FineTuneInputError as exc:
        parser.exit(exc.exit_code, f"fine-tune input rejected: {exc}\n")
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(30, f"fine-tune failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
