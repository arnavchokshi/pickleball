#!/usr/bin/env python3
"""Train the compact event head in bounded smoke or full-pretrain mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (
    BOUNCE, HIT, DatasetFormatError, EventWindowDataset, build_public_manifest,
    dense_class_counts, deterministic_source_video_holdout, manifest_windows,
    sqrt_frequency_class_weights, validate_current_manifest,
)
from threed.racketsport.event_head.assignment import (
    dense_cross_entropy, dynamic_label_assignment, offset_smooth_l1,
)
from threed.racketsport.event_head.matcher import Event, greedy_match, peak_pick
from threed.racketsport.event_head.model import (
    EventHead, checkpoint_payload, load_checkpoint, masked_cross_entropy,
    upgrade_event_head_with_offset,
)


CORPUS_PUBLIC = "public"
CORPUS_PBVISION_AGREEMENT = "pbvision-agreement"
STAGE_P_THRESHOLD_LOCK_FILENAME = "stage_p_decode_threshold_lock.json"
PUBLIC_PRETRAIN_LICENSE_REASON = (
    "RD_ONLY: full pretrain consumes public broadcast pixels and may include "
    "NC-licensed pixels"
)
PBVISION_STAGE_P_LICENSE_REASON = (
    "RD_ONLY: Stage-P agreement pixels are pbvision_signed_full_usage; the output "
    "conservatively preserves restrictions inherited from its initialization checkpoint"
)


def _git_head(root: Path = ROOT) -> str:
    """Return source provenance without requiring a shipped mirror to contain .git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return "unavailable:no_git_metadata"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoke_windows(manifest: dict[str, object], window_frames: int) -> list:
    candidates = manifest_windows(manifest, split="train", limit=4000, window_frames=window_frames)
    jhong = [item for item in candidates if item.source == "jhong93_spot"][:3]
    opentt = [item for item in candidates if item.source == "openttgames"][:1]
    if len(jhong) < 2 or len(opentt) < 1:
        raise RuntimeError(f"smoke requires >=2 jhong93 and >=1 OpenTT windows, got {len(jhong)}/{len(opentt)}")
    return jhong + opentt


def run_smoke(*, out: Path, weights: str, steps: int, image_size: int, window_frames: int) -> dict[str, object]:
    """Preserve the original phase-1 smoke path and artifact contract."""

    if steps < 30:
        raise ValueError("smoke requires at least 30 optimizer steps")
    torch.manual_seed(20260716)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest = build_public_manifest(ROOT / "data/event_public_20260713")
    dataset = EventWindowDataset(_smoke_windows(manifest, window_frames), image_size=image_size)
    # Decode exactly once; this is a tiny in-memory overfit batch, never a disk frame cache.
    samples = [dataset[index] for index in range(len(dataset))]
    frames = torch.stack([sample["frames"] for sample in samples])
    targets = torch.stack([sample["targets"] for sample in samples])
    masks = torch.stack([sample["validity_mask"] for sample in samples])
    model = EventHead(weights=weights, feature_dim=16, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses: list[float] = []
    started = time.monotonic()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_cross_entropy(model(frames), targets, masks)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite smoke loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    first5, last5 = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    if not last5 < first5:
        raise RuntimeError(f"tiny-overfit sanity failed: last5={last5} first5={first5}")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "smoke_event_head.pt"
    license_reason = "RD_ONLY: checkpoint trained on uncleared jhong93 broadcast pixels and CC-BY-NC-SA OpenTTGames pixels"
    torch.save(checkpoint_payload(
        model, license_posture="RD_ONLY", license_reason=license_reason,
        git_head=_git_head(), smoke=True, image_size=image_size,
        window_frames=window_frames, optimizer_steps=steps,
    ), checkpoint)
    report = {
        "schema_version": 1, "artifact_type": "event_head_train_manifest",
        "verified": False, "smoke_verified": True, "weights": weights,
        "optimizer_steps": steps, "all_losses_finite": all(math.isfinite(x) for x in losses),
        "first5_mean_loss": first5, "last5_mean_loss": last5,
        "losses": losses, "elapsed_s": time.monotonic() - started,
        "sources": [sample["source"] for sample in samples],
        "decode_policy": "on_the_fly_then_tiny_in_memory_overfit_batch",
        "image_size": image_size, "window_frames": window_frames,
        "checkpoint": str(checkpoint), "license_posture": "RD_ONLY",
        "license_reason": license_reason, "git_head": _git_head(),
    }
    (out / "train_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _validated_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")
    return torch.device(name)


def _manifest_windows(
    manifest: dict[str, Any], *, split: str, window_frames: int, limit_clips: int | None,
    stride_frames: int,
) -> list:
    limit = limit_clips if limit_clips is not None else max(1, len(manifest.get("rows", [])))
    windows = manifest_windows(
        manifest, split=split, limit=limit, window_frames=window_frames,
        stride_frames=stride_frames,
    )
    if not windows:
        raise RuntimeError(f"manifest has no media-present event windows in split={split!r}")
    return windows


def _validate_full_training_manifest(
    manifest: dict[str, Any], *, corpus_kind: str = CORPUS_PUBLIC,
) -> None:
    """Keep protected/bootstrap/owner pixels out even under a forged envelope."""

    allowed_public_sources = {"jhong93_spot", "openttgames", "shuttleset"}
    forbidden_tokens = (
        "data/event_bootstrap_20260713", "event_bootstrap_v0", "spot_check_tier_a_50",
        "owner_spot_check_results", "data/online_harvest_20260706", "tier_a",
    )
    if corpus_kind not in {CORPUS_PUBLIC, CORPUS_PBVISION_AGREEMENT}:
        raise ValueError(f"unsupported corpus kind: {corpus_kind}")
    if corpus_kind == CORPUS_PBVISION_AGREEMENT:
        try:
            validate_current_manifest(manifest)
        except DatasetFormatError as exc:
            raise ValueError(f"invalid pb.vision agreement manifest: {exc}") from exc
        if (
            manifest.get("teacher_derived") is not True
            or manifest.get("ground_truth") is not False
            or str(manifest.get("arm", "")).upper() != "B"
            or not str(manifest.get("artifact_type", "")).startswith(
                "event_head_pbvision_arm_b_"
            )
        ):
            raise ValueError(
                "pb.vision agreement corpus must be the teacher-derived Arm-B manifest"
            )
        denylist = set(map(str, manifest.get("permanent_compare_only_denylist", [])))
        if not denylist:
            raise ValueError("pb.vision agreement manifest must declare its compare denylist")
    for index, row in enumerate(manifest.get("rows", [])):
        source = str(row.get("source", ""))
        row_text = json.dumps(row, sort_keys=True).lower()
        matched = next((token for token in forbidden_tokens if token in row_text), None)
        if matched:
            raise ValueError(f"protected or owner training input forbidden at manifest row {index}: {matched}")
        if corpus_kind == CORPUS_PBVISION_AGREEMENT:
            if source != "pbvision_teacher_predictions":
                raise ValueError(
                    f"unexpected agreement-corpus source at row {index}: {source!r}"
                )
            if row.get("split") != "train" or row.get("training_eligible") is not True:
                raise ValueError(f"agreement-corpus row {index} is not train eligible")
            if str(row.get("source_video")) in denylist:
                raise ValueError(f"compare-only pb.vision source reached row {index}")
            if float(row.get("sample_weight", 0.0)) not in {0.25, 0.5}:
                raise ValueError(f"agreement-corpus row {index} has invalid sample weight")
            families = {
                str(agreement.get("family"))
                for event in row.get("events", [])
                for agreement in event.get("independent_agreements", [])
            }
            if "ball_velocity_kink" not in families:
                raise ValueError(
                    f"agreement-corpus row {index} lacks the required physical kink cue"
                )
            continue
        if source not in allowed_public_sources:
            fixture_path = Path(str(row.get("video_path", "")))
            fixture_root = ROOT / "tests/racketsport/fixtures/event_head"
            try:
                is_fixture = fixture_path.resolve().is_relative_to(fixture_root.resolve())
            except (OSError, RuntimeError):
                is_fixture = False
            if source != "synthetic_fixture" or not is_fixture:
                raise ValueError(f"non-public training source forbidden at manifest row {index}: {source!r}")


def _collect_validation_batches(
    model: EventHead, loader: DataLoader, *, device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Run internal validation inference once for a whole threshold sweep."""

    batches: list[dict[str, torch.Tensor]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].cpu()
            batches.append({
                "logits": model(batch["frames"].to(device)).cpu(),
                "targets": targets,
                "validity_mask": batch["validity_mask"].cpu().bool(),
                "frame_loss_mask": batch.get(
                    "frame_loss_mask", torch.ones_like(targets, dtype=torch.bool)
                ).cpu().bool(),
            })
    return batches


def _validation_metrics_from_batches(
    batches: Sequence[dict[str, torch.Tensor]], *, threshold: float,
    nms_radius: int, respect_frame_loss_mask: bool = False,
) -> dict[str, Any]:
    totals = {"HIT": {"tp": 0, "fp": 0, "fn": 0}, "BOUNCE": {"tp": 0, "fp": 0, "fn": 0}}
    max_positive_class_probability = 0.0
    for batch in batches:
        logits = batch["logits"]
        targets = batch["targets"]
        validity_mask = batch["validity_mask"]
        frame_loss_mask = batch["frame_loss_mask"]
        probabilities = logits.softmax(-1)
        positive_probabilities = probabilities[..., 1:]
        valid_positive = validity_mask[:, None, 1:].expand_as(positive_probabilities)
        if respect_frame_loss_mask:
            valid_positive = valid_positive & frame_loss_mask[..., None]
        finite_positive = positive_probabilities[
            torch.isfinite(positive_probabilities) & valid_positive
        ]
        if finite_positive.numel():
            max_positive_class_probability = max(
                max_positive_class_probability, float(finite_positive.max()),
            )
        for sample_index in range(logits.shape[0]):
            sample_logits = logits[sample_index].clone()
            if respect_frame_loss_mask:
                sample_logits[:, 1:].masked_fill_(
                    ~validity_mask[sample_index, 1:][None, :], -1.0e4
                )
                sample_logits[~frame_loss_mask[sample_index], 1:] = -1.0e4
            predictions = peak_pick(
                sample_logits, threshold=threshold, nms_radius=nms_radius,
            )
            ground_truth = [
                Event(frame, int(class_id))
                for frame, class_id in enumerate(targets[sample_index].tolist())
                if class_id in (HIT, BOUNCE) and (
                    not respect_frame_loss_mask
                    or (
                        bool(frame_loss_mask[sample_index, frame])
                        and bool(validity_mask[sample_index, int(class_id)])
                    )
                )
            ]
            for class_id, name in ((HIT, "HIT"), (BOUNCE, "BOUNCE")):
                matched = greedy_match(
                    [event for event in predictions if event.class_id == class_id],
                    [event for event in ground_truth if event.class_id == class_id],
                    tolerance_frames=2,
                )
                for key in ("tp", "fp", "fn"):
                    totals[name][key] += int(matched[key])
    per_class_f1: list[float] = []
    for values in totals.values():
        class_tp, class_fp, class_fn = values["tp"], values["fp"], values["fn"]
        class_f1 = (
            2 * class_tp / (2 * class_tp + class_fp + class_fn)
            if 2 * class_tp + class_fp + class_fn else 0.0
        )
        values["f1"] = class_f1
        per_class_f1.append(class_f1)
    tp = sum(value["tp"] for value in totals.values())
    fp = sum(value["fp"] for value in totals.values())
    fn = sum(value["fn"] for value in totals.values())
    f1 = (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    return {
        "tolerance_frames": 2, "threshold": threshold,
        "nms_radius_frames": nms_radius,
        "f1": f1, "macro_f1_at_2": sum(per_class_f1) / len(per_class_f1),
        "tp": tp, "fp": fp, "fn": fn,
        "per_class": totals,
        "max_positive_class_probability": max_positive_class_probability,
    }


def _validation_metrics(
    model: EventHead,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float = 0.5,
    nms_radius: int = 2,
) -> dict[str, Any]:
    return _validation_metrics_from_batches(
        _collect_validation_batches(model, loader, device=device),
        threshold=threshold,
        nms_radius=nms_radius,
        respect_frame_loss_mask=False,
    )


def _threshold_selection_key(row: dict[str, Any]) -> tuple[float, int, int, float]:
    """Frozen Stage-P order: macro-F1, FP, FN, then lower threshold."""

    return (
        float(row["macro_f1_at_2"]),
        -int(row["fp"]),
        -int(row["fn"]),
        -float(row["threshold"]),
    )


def _validation_threshold_sweep(
    model: EventHead,
    loader: DataLoader,
    *,
    device: torch.device,
    thresholds: Sequence[float],
    nms_radius: int,
) -> dict[str, Any]:
    batches = _collect_validation_batches(model, loader, device=device)
    rows = [
        _validation_metrics_from_batches(
            batches,
            threshold=threshold,
            nms_radius=nms_radius,
            respect_frame_loss_mask=True,
        )
        for threshold in thresholds
    ]
    selected = max(rows, key=_threshold_selection_key)
    return {**selected, "threshold_sweep": rows}


def _validation_is_better(
    validation: dict[str, Any], *, best_val_f1: float,
    best_val_max_positive_class_probability: float,
) -> bool:
    """Use confidence only to break ties during an all-zero-F1 phase."""

    candidate_f1 = float(validation["f1"])
    if candidate_f1 > best_val_f1:
        return True
    if candidate_f1 != 0.0 or best_val_f1 != 0.0:
        return False
    return (
        float(validation["max_positive_class_probability"])
        > best_val_max_positive_class_probability
    )


@dataclass
class _TrainingState:
    model: EventHead
    optimizer: torch.optim.Optimizer
    start_step: int
    best_val_f1: float
    best_val_max_positive_class_probability: float
    best_validation_threshold: float | None
    best_validation_fp: int | None
    best_validation_fn: int | None
    best_validation_step: int | None
    resume_mode: str
    optimizer_state_restored: bool


def _initialize_training_state(
    *, device: torch.device, device_name: str, weights: str, lr: float,
    init_checkpoint: Path | None, init_checkpoint_model_only: Path | None,
    offset_regression_head: bool = False,
) -> _TrainingState:
    if init_checkpoint is not None and init_checkpoint_model_only is not None:
        raise ValueError("--init-checkpoint and --init-checkpoint-model-only are mutually exclusive")
    selected_checkpoint = init_checkpoint or init_checkpoint_model_only
    if selected_checkpoint is None:
        model, initial_payload = EventHead(
            weights=weights, offset_regression_head=offset_regression_head,
        ), {}
        model.to(device)
        resume_mode = "fresh"
    else:
        model, initial_payload = load_checkpoint(selected_checkpoint, device=device_name)
        if offset_regression_head and model.offset_regressor is None:
            model = upgrade_event_head_with_offset(model)
        elif not offset_regression_head and model.offset_regressor is not None:
            raise ValueError(
                "checkpoint has an offset head but --offset-regression-head is disabled"
            )
        resume_mode = "model_only" if init_checkpoint_model_only is not None else "full_state"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_state_restored = False
    if resume_mode == "full_state" and "optimizer_state_dict" in initial_payload:
        optimizer.load_state_dict(initial_payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        optimizer_state_restored = True

    start_step = int(initial_payload.get("completed_steps", 0))
    if resume_mode == "model_only":
        best_val_f1 = -1.0
        best_val_max_positive_class_probability = 0.0
        best_validation_threshold = None
        best_validation_fp = None
        best_validation_fn = None
        best_validation_step = None
    else:
        best_val_f1 = float(initial_payload.get("best_val_f1", -1.0))
        best_val_max_positive_class_probability = float(
            initial_payload.get("best_val_max_positive_class_probability", 0.0)
        )
        best_validation_threshold = initial_payload.get("best_validation_threshold")
        if best_validation_threshold is not None:
            best_validation_threshold = float(best_validation_threshold)
        best_validation_fp = initial_payload.get("best_validation_fp")
        if best_validation_fp is not None:
            best_validation_fp = int(best_validation_fp)
        best_validation_fn = initial_payload.get("best_validation_fn")
        if best_validation_fn is not None:
            best_validation_fn = int(best_validation_fn)
        best_validation_step = initial_payload.get("best_validation_step")
        if best_validation_step is not None:
            best_validation_step = int(best_validation_step)
    return _TrainingState(
        model=model,
        optimizer=optimizer,
        start_step=start_step,
        best_val_f1=best_val_f1,
        best_val_max_positive_class_probability=best_val_max_positive_class_probability,
        best_validation_threshold=best_validation_threshold,
        best_validation_fp=best_validation_fp,
        best_validation_fn=best_validation_fn,
        best_validation_step=best_validation_step,
        resume_mode=resume_mode,
        optimizer_state_restored=optimizer_state_restored,
    )


def _save_full_checkpoint(
    path: Path, *, model: EventHead, optimizer: torch.optim.Optimizer, completed_steps: int,
    best_val_f1: float, best_val_max_positive_class_probability: float,
    config: dict[str, Any], data_manifest_path: Path,
    data_manifest_sha256: str, elapsed_s: float, checkpoint_role: str,
    best_validation_threshold: float | None = None,
    best_validation_fp: int | None = None,
    best_validation_fn: int | None = None,
    best_validation_step: int | None = None,
) -> None:
    torch.save(checkpoint_payload(
        model,
        license_posture="RD_ONLY",
        license_reason=config["license_reason"],
        git_head=_git_head(), smoke=False, full_pretrain=True,
        image_size=config["image_size"], window_frames=config["window_frames"],
        completed_steps=completed_steps, optimizer_steps=completed_steps,
        optimizer_state_dict=optimizer.state_dict(), best_val_f1=best_val_f1,
        best_val_max_positive_class_probability=best_val_max_positive_class_probability,
        data_manifest=str(data_manifest_path), data_manifest_sha256=data_manifest_sha256,
        pretrain_data=str(data_manifest_path), seed=config["seed"], config=config,
        elapsed_s=elapsed_s, checkpoint_role=checkpoint_role,
        best_validation_threshold=best_validation_threshold,
        best_validation_fp=best_validation_fp,
        best_validation_fn=best_validation_fn,
        best_validation_step=best_validation_step,
    ), path)


def _recipe_training_loss(
    model: EventHead,
    batch: dict[str, Any],
    *,
    device: torch.device,
    class_weights: torch.Tensor | None,
    label_assignment: str,
    assignment_max_shift_frames: int,
    assignment_class_cost_weight: float,
    assignment_temporal_cost_weight: float,
    label_dilation_frames: int,
    label_dilation_soft_weight: float,
    offset_regression_head: bool,
    offset_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Compute the opt-in E-v2 objective and its auditable sufficient stats."""

    frames = batch["frames"].to(device)
    if offset_regression_head:
        logits, predicted_offsets = model.forward_with_aux(frames)
    else:
        logits = model(frames)
        predicted_offsets = None
    hard_targets = batch["targets"].to(device)
    validity_mask = batch["validity_mask"].to(device)
    frame_loss_mask = batch["frame_loss_mask"].to(device)
    event_subframe_offsets = batch["event_subframe_offsets"].to(device)
    sample_weights = batch["sample_weight"].to(device)
    assignment = dynamic_label_assignment(
        logits,
        hard_targets,
        validity_mask,
        frame_loss_mask,
        event_subframe_offsets,
        mode=label_assignment,
        max_shift_frames=assignment_max_shift_frames,
        class_cost_weight=assignment_class_cost_weight,
        temporal_cost_weight=assignment_temporal_cost_weight,
        label_dilation_frames=label_dilation_frames,
        neighbor_positive_weight=label_dilation_soft_weight,
    )
    classification_loss, _, classification_normalizers = dense_cross_entropy(
        logits,
        assignment.dense_targets,
        validity_mask,
        frame_loss_mask,
        class_weights=class_weights,
        sample_weights=sample_weights,
    )
    if predicted_offsets is None:
        offset_loss = logits.sum() * 0.0
        offset_normalizer = torch.zeros((), dtype=logits.dtype, device=device)
    else:
        offset_loss, _, offset_normalizers = offset_smooth_l1(
            predicted_offsets,
            assignment.offset_targets,
            assignment.offset_mask,
            sample_weights=sample_weights,
        )
        offset_normalizer = offset_normalizers.sum()
    total_loss = classification_loss + offset_loss_weight * offset_loss
    return total_loss, {
        "classification_loss": float(classification_loss.detach().cpu()),
        "offset_loss": float(offset_loss.detach().cpu()),
        "classification_normalizer": float(
            classification_normalizers.detach().sum().cpu()
        ),
        "offset_normalizer": float(offset_normalizer.detach().cpu()),
        "batch_sample_weight_sum": float(sample_weights.detach().sum().cpu()),
        "event_count": assignment.event_count,
        "shifted_event_count": assignment.shifted_event_count,
        "total_abs_shift": assignment.total_abs_shift,
    }


def run_full(
    *, manifest_path: Path, device_name: str, out: Path, weights: str, steps: int,
    image_size: int, window_frames: int, batch_size: int, lr: float, val_every: int,
    seed: int, max_wall_minutes: float | None, init_checkpoint: Path | None,
    limit_clips: int | None, stride_frames: int = 32, num_workers: int = 4,
    prefetch_factor: int = 2, class_weights: list[float] | tuple[float, ...] | None = None,
    init_checkpoint_model_only: Path | None = None,
    corpus_kind: str = CORPUS_PUBLIC,
    internal_val_source_count: int | None = None,
    sqrt_frequency_weights: bool = False,
    label_dilation_frames: int = 0,
    label_dilation_soft_weight: float = 0.5,
    label_assignment: str = "fixed",
    assignment_max_shift_frames: int = 2,
    assignment_class_cost_weight: float = 1.0,
    assignment_temporal_cost_weight: float = 0.25,
    offset_regression_head: bool = False,
    offset_loss_weight: float = 0.2,
    validation_thresholds: Sequence[float] = (0.5,),
    validation_nms_radius: int = 2,
) -> dict[str, Any]:
    if steps < 1 or image_size < 16 or window_frames < 1 or batch_size < 1 or lr <= 0 or val_every < 1:
        raise ValueError("full mode requires positive steps/window-frames/batch-size/lr/val-every and image-size >=16")
    if limit_clips is not None and limit_clips < 1:
        raise ValueError("--limit-clips must be >=1")
    if stride_frames < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("--stride-frames and --prefetch-factor must be positive; --num-workers must be >=0")
    if max_wall_minutes is not None and max_wall_minutes <= 0:
        raise ValueError("--max-wall-minutes must be >0")
    if class_weights is not None and (
        len(class_weights) != 3
        or any(not math.isfinite(value) or value <= 0 for value in class_weights)
    ):
        raise ValueError("--class-weights requires three finite positive values")
    if sqrt_frequency_weights and class_weights is not None:
        raise ValueError("sqrt-frequency weighting and fixed --class-weights are exclusive")
    if label_dilation_frames not in {0, 1}:
        raise ValueError("--label-dilation-frames must be 0 or 1")
    if not 0.0 < label_dilation_soft_weight <= 1.0:
        raise ValueError("--label-dilation-soft-weight must be in (0,1]")
    if label_assignment not in {"fixed", "hungarian"}:
        raise ValueError("--label-assignment must be fixed or hungarian")
    if assignment_max_shift_frames < 0:
        raise ValueError("--assignment-max-shift-frames must be nonnegative")
    if assignment_class_cost_weight <= 0 or assignment_temporal_cost_weight < 0:
        raise ValueError("assignment cost weights must be positive/nonnegative")
    if offset_loss_weight < 0 or not math.isfinite(offset_loss_weight):
        raise ValueError("--offset-loss-weight must be finite and nonnegative")
    thresholds = tuple(float(value) for value in validation_thresholds)
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in thresholds)
        or validation_nms_radius < 0
    ):
        raise ValueError("validation thresholds must be unique in (0,1) and NMS nonnegative")
    if corpus_kind == CORPUS_PBVISION_AGREEMENT and internal_val_source_count is None:
        raise ValueError("pb.vision agreement pretrain requires an internal source holdout")
    if corpus_kind == CORPUS_PUBLIC and internal_val_source_count is not None:
        raise ValueError("--internal-val-source-count is only for pb.vision agreement pretrain")
    device = _validated_device(device_name)
    _seed_everything(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if (
        corpus_kind == CORPUS_PUBLIC
        and manifest.get("artifact_type") != "event_head_public_dataset_manifest"
    ):
        raise ValueError("--manifest is not an event-head public dataset manifest")
    _validate_full_training_manifest(manifest, corpus_kind=corpus_kind)
    internal_val_sources: tuple[str, ...] = ()
    training_manifest = manifest
    if internal_val_source_count is not None:
        training_manifest, internal_val_sources = deterministic_source_video_holdout(
            manifest, seed=seed, holdout_source_count=internal_val_source_count,
        )
    train_windows = _manifest_windows(
        training_manifest, split="train", window_frames=window_frames,
        limit_clips=limit_clips,
        stride_frames=stride_frames,
    )
    val_windows = _manifest_windows(
        training_manifest, split="val", window_frames=window_frames,
        limit_clips=limit_clips,
        stride_frames=stride_frames,
    )
    train_dataset = EventWindowDataset(train_windows, image_size=image_size)
    val_dataset = EventWindowDataset(val_windows, image_size=image_size)
    generator = torch.Generator().manual_seed(seed)
    loader_workers = {
        "num_workers": num_workers,
        **({"prefetch_factor": prefetch_factor} if num_workers > 0 else {}),
    }
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator,
        **loader_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_workers,
    )
    training_state = _initialize_training_state(
        device=device, device_name=device_name, weights=weights, lr=lr,
        init_checkpoint=init_checkpoint,
        init_checkpoint_model_only=init_checkpoint_model_only,
        offset_regression_head=offset_regression_head,
    )
    model, optimizer = training_state.model, training_state.optimizer
    # A curriculum stage initialized model-only is a new exposure budget. Keep
    # source-step provenance, but reset this stage's optimizer step to zero.
    init_checkpoint_completed_steps = training_state.start_step
    start_step = (
        0 if init_checkpoint_model_only is not None else training_state.start_step
    )
    if start_step >= steps:
        raise ValueError(f"checkpoint already completed {start_step} steps, not less than --steps {steps}")
    best_val_f1 = training_state.best_val_f1
    best_val_max_positive_class_probability = (
        training_state.best_val_max_positive_class_probability
    )
    best_validation_threshold = training_state.best_validation_threshold
    best_validation_fp = training_state.best_validation_fp
    best_validation_fn = training_state.best_validation_fn
    best_validation_step = training_state.best_validation_step
    best_stage_p_key: tuple[float, int, int, float] | None = None
    if (
        corpus_kind == CORPUS_PBVISION_AGREEMENT
        and best_val_f1 >= 0
        and best_validation_threshold is not None
        and best_validation_fp is not None
        and best_validation_fn is not None
    ):
        best_stage_p_key = (
            best_val_f1,
            -best_validation_fp,
            -best_validation_fn,
            -best_validation_threshold,
        )
    class_counts = dense_class_counts(
        train_windows,
        label_dilation_frames=label_dilation_frames,
        neighbor_positive_weight=label_dilation_soft_weight,
    )
    resolved_class_weights = (
        sqrt_frequency_class_weights(class_counts)
        if sqrt_frequency_weights else tuple(class_weights) if class_weights is not None else None
    )
    loss_class_weights = (
        torch.tensor(resolved_class_weights, dtype=torch.float32, device=device)
        if resolved_class_weights is not None else None
    )
    use_recipe_objective = (
        corpus_kind == CORPUS_PBVISION_AGREEMENT
        or sqrt_frequency_weights
        or label_dilation_frames > 0
        or label_assignment == "hungarian"
        or offset_regression_head
    )
    sample_weight_counts: dict[str, int] = {}
    sample_weight_total = 0.0
    for window in train_windows:
        key = format(float(window.sample_weight), ".12g")
        sample_weight_counts[key] = sample_weight_counts.get(key, 0) + 1
        sample_weight_total += float(window.sample_weight)
    threshold_lock_path = out / STAGE_P_THRESHOLD_LOCK_FILENAME
    license_reason = (
        PBVISION_STAGE_P_LICENSE_REASON
        if corpus_kind == CORPUS_PBVISION_AGREEMENT
        else PUBLIC_PRETRAIN_LICENSE_REASON
    )
    config = {
        "device": device_name, "weights": weights, "steps": steps, "image_size": image_size,
        "window_frames": window_frames, "batch_size": batch_size, "lr": lr,
        "val_every": val_every, "seed": seed, "max_wall_minutes": max_wall_minutes,
        "limit_clips": limit_clips, "stride_frames": stride_frames,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "corpus_kind": corpus_kind,
        "internal_val_source_count": internal_val_source_count,
        "internal_val_source_videos": list(internal_val_sources),
        "class_weighting": (
            "sqrt_frequency" if sqrt_frequency_weights
            else "fixed" if class_weights is not None else "none"
        ),
        "class_counts_loss_eligible_dense_mass": list(class_counts),
        "class_weights": (
            list(resolved_class_weights) if resolved_class_weights is not None else None
        ),
        "class_weight_formula": (
            "sqrt(n_background/n_class)" if sqrt_frequency_weights else None
        ),
        "label_dilation_frames": label_dilation_frames,
        "label_dilation_soft_weight": label_dilation_soft_weight,
        "label_assignment": label_assignment,
        "assignment_max_shift_frames": assignment_max_shift_frames,
        "assignment_class_cost_weight": assignment_class_cost_weight,
        "assignment_temporal_cost_weight": assignment_temporal_cost_weight,
        "offset_regression_head": offset_regression_head,
        "offset_loss_weight": offset_loss_weight,
        "validation_thresholds": list(thresholds),
        "validation_nms_radius": validation_nms_radius,
        "validation_threshold_tie_break": [
            "macro_f1_at_2_desc", "fp_asc", "fn_asc", "threshold_asc",
            "checkpoint_step_asc_strict_tie",
        ],
        "training_objective": (
            "ev2_dense_assignment" if use_recipe_objective else "legacy_masked_cross_entropy"
        ),
        "train_window_sample_weight_counts": sample_weight_counts,
        "train_window_sample_weight_total": sample_weight_total,
        "decode_threshold_lock_path": (
            str(threshold_lock_path)
            if corpus_kind == CORPUS_PBVISION_AGREEMENT else None
        ),
        "license_reason": license_reason,
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    last_path, best_path = out / "last_event_head.pt", out / "best_event_head.pt"
    started = time.monotonic()
    losses: list[float] = []
    recipe_loss_stats: list[dict[str, float | int]] = []
    assignment_totals = {
        "event_count": 0, "shifted_event_count": 0, "total_abs_shift": 0,
    }
    validations: list[dict[str, Any]] = []
    completed_steps = start_step
    wall_stopped = False
    iterator = iter(train_loader)
    while completed_steps < steps:
        if max_wall_minutes is not None and (time.monotonic() - started) >= max_wall_minutes * 60:
            wall_stopped = True
            break
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if use_recipe_objective:
            loss, loss_stats = _recipe_training_loss(
                model,
                batch,
                device=device,
                class_weights=loss_class_weights,
                label_assignment=label_assignment,
                assignment_max_shift_frames=assignment_max_shift_frames,
                assignment_class_cost_weight=assignment_class_cost_weight,
                assignment_temporal_cost_weight=assignment_temporal_cost_weight,
                label_dilation_frames=label_dilation_frames,
                label_dilation_soft_weight=label_dilation_soft_weight,
                offset_regression_head=offset_regression_head,
                offset_loss_weight=offset_loss_weight,
            )
            recipe_loss_stats.append(loss_stats)
            for key in assignment_totals:
                assignment_totals[key] += int(loss_stats[key])
        else:
            # Preserve the original public-pretrain objective at default flags.
            loss = masked_cross_entropy(
                model(batch["frames"].to(device)), batch["targets"].to(device),
                batch["validity_mask"].to(device),
                class_weights=loss_class_weights,
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite full-train loss at step {completed_steps + 1}: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        completed_steps += 1
        run_steps = completed_steps - start_step
        if run_steps == 100:
            print(f"steps/s: {run_steps / (time.monotonic() - started):.6f} after first 100 steps", flush=True)
        if completed_steps % val_every == 0 or completed_steps == steps:
            validation_metrics = (
                _validation_threshold_sweep(
                    model,
                    val_loader,
                    device=device,
                    thresholds=thresholds,
                    nms_radius=validation_nms_radius,
                )
                if corpus_kind == CORPUS_PBVISION_AGREEMENT
                else _validation_metrics(
                    model,
                    val_loader,
                    device=device,
                    threshold=thresholds[0],
                    nms_radius=validation_nms_radius,
                )
            )
            validation = {"step": completed_steps, **validation_metrics}
            validations.append(validation)
            elapsed = time.monotonic() - started
            if corpus_kind == CORPUS_PBVISION_AGREEMENT:
                candidate_stage_p_key = _threshold_selection_key(validation)
                validation_is_better = (
                    best_stage_p_key is None or candidate_stage_p_key > best_stage_p_key
                )
            else:
                candidate_stage_p_key = None
                validation_is_better = _validation_is_better(
                    validation, best_val_f1=best_val_f1,
                    best_val_max_positive_class_probability=(
                        best_val_max_positive_class_probability
                    ),
                )
            if validation_is_better:
                best_val_f1 = float(
                    validation[
                        "macro_f1_at_2"
                        if corpus_kind == CORPUS_PBVISION_AGREEMENT else "f1"
                    ]
                )
                best_val_max_positive_class_probability = float(
                    validation["max_positive_class_probability"]
                )
                if candidate_stage_p_key is not None:
                    best_stage_p_key = candidate_stage_p_key
                    best_validation_threshold = float(validation["threshold"])
                    best_validation_fp = int(validation["fp"])
                    best_validation_fn = int(validation["fn"])
                    best_validation_step = completed_steps
                _save_full_checkpoint(
                    best_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
                    best_val_f1=best_val_f1,
                    best_val_max_positive_class_probability=best_val_max_positive_class_probability,
                    config=config, data_manifest_path=manifest_path,
                    data_manifest_sha256=manifest_sha, elapsed_s=elapsed,
                    checkpoint_role=(
                        "best_by_internal_val_macro_f1_at_2_fp_fn_threshold"
                        if corpus_kind == CORPUS_PBVISION_AGREEMENT
                        else "best_by_val_f1_then_zero_f1_max_probability"
                    ),
                    best_validation_threshold=best_validation_threshold,
                    best_validation_fp=best_validation_fp,
                    best_validation_fn=best_validation_fn,
                    best_validation_step=best_validation_step,
                )
            _save_full_checkpoint(
                last_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
                best_val_f1=best_val_f1,
                best_val_max_positive_class_probability=best_val_max_positive_class_probability,
                config=config, data_manifest_path=manifest_path,
                data_manifest_sha256=manifest_sha, elapsed_s=elapsed, checkpoint_role="last",
                best_validation_threshold=best_validation_threshold,
                best_validation_fp=best_validation_fp,
                best_validation_fn=best_validation_fn,
                best_validation_step=best_validation_step,
            )
    elapsed = time.monotonic() - started
    # A wall cap may fire between validations; always preserve the exact latest state.
    _save_full_checkpoint(
        last_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
        best_val_f1=best_val_f1,
        best_val_max_positive_class_probability=best_val_max_positive_class_probability,
        config=config, data_manifest_path=manifest_path,
        data_manifest_sha256=manifest_sha, elapsed_s=elapsed, checkpoint_role="last",
        best_validation_threshold=best_validation_threshold,
        best_validation_fp=best_validation_fp,
        best_validation_fn=best_validation_fn,
        best_validation_step=best_validation_step,
    )
    if not best_path.is_file():
        validation_metrics = (
            _validation_threshold_sweep(
                model,
                val_loader,
                device=device,
                thresholds=thresholds,
                nms_radius=validation_nms_radius,
            )
            if corpus_kind == CORPUS_PBVISION_AGREEMENT
            else _validation_metrics(
                model,
                val_loader,
                device=device,
                threshold=thresholds[0],
                nms_radius=validation_nms_radius,
            )
        )
        validation = {"step": completed_steps, **validation_metrics}
        validations.append(validation)
        best_val_f1 = float(
            validation[
                "macro_f1_at_2"
                if corpus_kind == CORPUS_PBVISION_AGREEMENT else "f1"
            ]
        )
        best_val_max_positive_class_probability = float(
            validation["max_positive_class_probability"]
        )
        if corpus_kind == CORPUS_PBVISION_AGREEMENT:
            best_stage_p_key = _threshold_selection_key(validation)
            best_validation_threshold = float(validation["threshold"])
            best_validation_fp = int(validation["fp"])
            best_validation_fn = int(validation["fn"])
            best_validation_step = completed_steps
        _save_full_checkpoint(
            best_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
            best_val_f1=best_val_f1,
            best_val_max_positive_class_probability=best_val_max_positive_class_probability,
            config=config, data_manifest_path=manifest_path,
            data_manifest_sha256=manifest_sha, elapsed_s=elapsed,
            checkpoint_role=(
                "best_by_internal_val_macro_f1_at_2_fp_fn_threshold"
                if corpus_kind == CORPUS_PBVISION_AGREEMENT
                else "best_by_val_f1_then_zero_f1_max_probability"
            ),
            best_validation_threshold=best_validation_threshold,
            best_validation_fp=best_validation_fp,
            best_validation_fn=best_validation_fn,
            best_validation_step=best_validation_step,
        )
        _save_full_checkpoint(
            last_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
            best_val_f1=best_val_f1,
            best_val_max_positive_class_probability=best_val_max_positive_class_probability,
            config=config, data_manifest_path=manifest_path,
            data_manifest_sha256=manifest_sha, elapsed_s=elapsed, checkpoint_role="last",
            best_validation_threshold=best_validation_threshold,
            best_validation_fp=best_validation_fp,
            best_validation_fn=best_validation_fn,
            best_validation_step=best_validation_step,
        )
    decode_threshold_lock: dict[str, Any] | None = None
    decode_threshold_lock_sha256: str | None = None
    if (
        corpus_kind == CORPUS_PBVISION_AGREEMENT
        and not wall_stopped
        and completed_steps == steps
    ):
        if (
            best_validation_threshold is None
            or best_validation_fp is None
            or best_validation_fn is None
            or best_validation_step is None
            or not best_path.is_file()
        ):
            raise RuntimeError("completed Stage-P run has no internal-validation decode lock")
        selected_validation = next(
            (
                row for row in validations
                if int(row["step"]) == best_validation_step
                and float(row["threshold"]) == best_validation_threshold
            ),
            None,
        )
        if selected_validation is None:
            raise RuntimeError("best Stage-P validation record is missing")
        decode_threshold_lock = {
            "schema_version": 1,
            "artifact_type": "event_head_stage_p_decode_threshold_lock",
            "verified": False,
            "status": "locked_from_stage_p_internal_validation",
            "owner_val_used": False,
            "data_manifest": str(manifest_path),
            "data_manifest_sha256": manifest_sha,
            "internal_validation_policy": "sha256_seeded_source_video_holdout",
            "internal_validation_source_videos": list(internal_val_sources),
            "seed": seed,
            "checkpoint": str(best_path),
            "checkpoint_sha256": _sha256_file(best_path),
            "checkpoint_step": best_validation_step,
            "threshold": best_validation_threshold,
            "threshold_grid": list(thresholds),
            "threshold_tie_break": [
                "macro_f1_at_2_desc", "fp_asc", "fn_asc", "threshold_asc",
                "checkpoint_step_asc_strict_tie",
            ],
            "nms_radius_frames": validation_nms_radius,
            "match_tolerance_frames": 2,
            "selected_internal_validation": selected_validation,
            "consumer": (
                f"jq -r .threshold {threshold_lock_path}"
            ),
        }
        threshold_lock_path.write_text(
            json.dumps(decode_threshold_lock, indent=2, sort_keys=True) + "\n"
        )
        decode_threshold_lock_sha256 = _sha256_file(threshold_lock_path)
    report = {
        "schema_version": 1, "artifact_type": "event_head_train_manifest",
        "verified": False, "smoke_verified": False, "mode": "full",
        "status": "partial_wall_stop" if wall_stopped else "complete",
        "honest_partial": wall_stopped, "git_head": _git_head(),
        "data_manifest": str(manifest_path), "data_manifest_sha256": manifest_sha,
        "seed": seed, "config": config, "license_posture": "RD_ONLY",
        "license_reason": license_reason,
        "train_windows": len(train_windows), "val_windows": len(val_windows),
        "start_step": start_step, "completed_steps": completed_steps, "target_steps": steps,
        "init_checkpoint_completed_steps": init_checkpoint_completed_steps,
        "losses": losses, "all_losses_finite": all(math.isfinite(value) for value in losses),
        "recipe_loss_stats": recipe_loss_stats,
        "assignment_totals": assignment_totals,
        "validations": validations, "best_val_f1": best_val_f1,
        "best_val_metric": (
            "macro_f1_at_2_internal"
            if corpus_kind == CORPUS_PBVISION_AGREEMENT else "micro_f1_at_2"
        ),
        "best_val_max_positive_class_probability": best_val_max_positive_class_probability,
        "best_validation_threshold": best_validation_threshold,
        "best_validation_fp": best_validation_fp,
        "best_validation_fn": best_validation_fn,
        "best_validation_step": best_validation_step,
        "best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
        "locked_decode_threshold": (
            decode_threshold_lock["threshold"] if decode_threshold_lock else None
        ),
        "decode_threshold_lock": (
            str(threshold_lock_path) if decode_threshold_lock else None
        ),
        "decode_threshold_lock_sha256": decode_threshold_lock_sha256,
        "elapsed_s": elapsed, "steps_per_s": (completed_steps - start_step) / elapsed if elapsed else 0.0,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "init_checkpoint_model_only": (
            str(init_checkpoint_model_only) if init_checkpoint_model_only else None
        ),
        "resume_mode": training_state.resume_mode,
        "optimizer_state_restored": training_state.optimizer_state_restored,
        "decode_policy": "on_the_fly_no_frame_cache",
        "dataloader": {
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        },
    }
    (out / "train_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Original bounded CPU smoke mode")
    mode.add_argument("--full", action="store_true", help="Full manifest-backed pretrain mode")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--weights", choices=("none", "imagenet"), default="none")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-wall-minutes", type=float)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--init-checkpoint", type=Path)
    resume.add_argument(
        "--init-checkpoint-model-only", type=Path,
        help="Load model weights and step provenance but reset Adam and best-selection state",
    )
    weighting = parser.add_mutually_exclusive_group()
    weighting.add_argument(
        "--class-weights", type=float, nargs=3,
        metavar=("BACKGROUND", "HIT", "BOUNCE"),
        help="Optional CE weights in background,HIT,BOUNCE order; recommended: 1 5 5",
    )
    weighting.add_argument(
        "--sqrt-frequency-class-weights", action="store_true",
        help="Compute sqrt(n_background/n_class) from loss-eligible training targets",
    )
    parser.add_argument(
        "--corpus-kind", choices=(CORPUS_PUBLIC, CORPUS_PBVISION_AGREEMENT),
        default=CORPUS_PUBLIC,
    )
    parser.add_argument("--internal-val-source-count", type=int)
    parser.add_argument("--label-dilation-frames", type=int, choices=(0, 1), default=0)
    parser.add_argument("--label-dilation-soft-weight", type=float, default=0.5)
    parser.add_argument(
        "--label-assignment", choices=("fixed", "hungarian"), default="fixed",
    )
    parser.add_argument("--assignment-max-shift-frames", type=int, default=2)
    parser.add_argument("--assignment-class-cost-weight", type=float, default=1.0)
    parser.add_argument("--assignment-temporal-cost-weight", type=float, default=0.25)
    parser.add_argument("--offset-regression-head", action="store_true")
    parser.add_argument("--offset-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--validation-thresholds", type=float, nargs="+", default=[0.5],
    )
    parser.add_argument("--validation-nms-radius", type=int, default=2)
    parser.add_argument("--limit-clips", type=int)
    parser.add_argument("--stride-frames", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args()
    if args.smoke:
        out = args.out or ROOT / "runs/lanes/event_head_scaffold_20260716/train"
        try:
            report = run_smoke(
                out=out, weights=args.weights, steps=args.steps,
                image_size=args.image_size, window_frames=args.window_frames,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(3, f"event-head smoke failed: {exc}\n")
    else:
        missing = [flag for flag, value in (("--manifest", args.manifest), ("--device", args.device), ("--out", args.out)) if value is None]
        if missing:
            parser.error(f"--full requires {', '.join(missing)}")
        try:
            report = run_full(
                manifest_path=args.manifest, device_name=args.device, out=args.out,
                weights=args.weights, steps=args.steps, image_size=args.image_size,
                window_frames=args.window_frames, batch_size=args.batch_size, lr=args.lr,
                val_every=args.val_every, seed=args.seed,
                max_wall_minutes=args.max_wall_minutes, init_checkpoint=args.init_checkpoint,
                limit_clips=args.limit_clips, stride_frames=args.stride_frames,
                num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                class_weights=args.class_weights,
                init_checkpoint_model_only=args.init_checkpoint_model_only,
                corpus_kind=args.corpus_kind,
                internal_val_source_count=args.internal_val_source_count,
                sqrt_frequency_weights=args.sqrt_frequency_class_weights,
                label_dilation_frames=args.label_dilation_frames,
                label_dilation_soft_weight=args.label_dilation_soft_weight,
                label_assignment=args.label_assignment,
                assignment_max_shift_frames=args.assignment_max_shift_frames,
                assignment_class_cost_weight=args.assignment_class_cost_weight,
                assignment_temporal_cost_weight=args.assignment_temporal_cost_weight,
                offset_regression_head=args.offset_regression_head,
                offset_loss_weight=args.offset_loss_weight,
                validation_thresholds=args.validation_thresholds,
                validation_nms_radius=args.validation_nms_radius,
            )
        except (RuntimeError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            parser.exit(3, f"event-head full train failed: {exc}\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
