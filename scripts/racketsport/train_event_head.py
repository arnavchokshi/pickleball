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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (
    BOUNCE, HIT, EventWindowDataset, build_public_manifest, manifest_windows,
)
from threed.racketsport.event_head.matcher import Event, greedy_match, peak_pick
from threed.racketsport.event_head.model import (
    EventHead, checkpoint_payload, load_checkpoint, masked_cross_entropy,
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


def _validate_full_training_manifest(manifest: dict[str, Any]) -> None:
    """Keep protected/bootstrap/owner pixels out even under a forged envelope."""

    allowed_public_sources = {"jhong93_spot", "openttgames", "shuttleset"}
    forbidden_tokens = (
        "data/event_bootstrap_20260713", "event_bootstrap_v0", "spot_check_tier_a_50",
        "owner_spot_check_results", "data/online_harvest_20260706", "tier_a",
    )
    for index, row in enumerate(manifest.get("rows", [])):
        source = str(row.get("source", ""))
        row_text = json.dumps(row, sort_keys=True).lower()
        matched = next((token for token in forbidden_tokens if token in row_text), None)
        if matched:
            raise ValueError(f"protected or owner training input forbidden at manifest row {index}: {matched}")
        if source not in allowed_public_sources:
            fixture_path = Path(str(row.get("video_path", "")))
            fixture_root = ROOT / "tests/racketsport/fixtures/event_head"
            try:
                is_fixture = fixture_path.resolve().is_relative_to(fixture_root.resolve())
            except (OSError, RuntimeError):
                is_fixture = False
            if source != "synthetic_fixture" or not is_fixture:
                raise ValueError(f"non-public training source forbidden at manifest row {index}: {source!r}")


def _validation_metrics(model: EventHead, loader: DataLoader, *, device: torch.device) -> dict[str, Any]:
    totals = {"HIT": {"tp": 0, "fp": 0, "fn": 0}, "BOUNCE": {"tp": 0, "fp": 0, "fn": 0}}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["frames"].to(device))
            targets = batch["targets"]
            for sample_index in range(logits.shape[0]):
                predictions = peak_pick(logits[sample_index].cpu(), threshold=0.5, nms_radius=2)
                ground_truth = [
                    Event(frame, int(class_id))
                    for frame, class_id in enumerate(targets[sample_index].tolist())
                    if class_id in (HIT, BOUNCE)
                ]
                for class_id, name in ((HIT, "HIT"), (BOUNCE, "BOUNCE")):
                    matched = greedy_match(
                        [event for event in predictions if event.class_id == class_id],
                        [event for event in ground_truth if event.class_id == class_id],
                        tolerance_frames=2,
                    )
                    for key in ("tp", "fp", "fn"):
                        totals[name][key] += int(matched[key])
    tp = sum(value["tp"] for value in totals.values())
    fp = sum(value["fp"] for value in totals.values())
    fn = sum(value["fn"] for value in totals.values())
    f1 = (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    return {"tolerance_frames": 2, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "per_class": totals}


def _save_full_checkpoint(
    path: Path, *, model: EventHead, optimizer: torch.optim.Optimizer, completed_steps: int,
    best_val_f1: float, config: dict[str, Any], data_manifest_path: Path,
    data_manifest_sha256: str, elapsed_s: float, checkpoint_role: str,
) -> None:
    torch.save(checkpoint_payload(
        model,
        license_posture="RD_ONLY",
        license_reason="RD_ONLY: full pretrain consumes public broadcast pixels and may include NC-licensed pixels",
        git_head=_git_head(), smoke=False, full_pretrain=True,
        image_size=config["image_size"], window_frames=config["window_frames"],
        completed_steps=completed_steps, optimizer_steps=completed_steps,
        optimizer_state_dict=optimizer.state_dict(), best_val_f1=best_val_f1,
        data_manifest=str(data_manifest_path), data_manifest_sha256=data_manifest_sha256,
        pretrain_data=str(data_manifest_path), seed=config["seed"], config=config,
        elapsed_s=elapsed_s, checkpoint_role=checkpoint_role,
    ), path)


def run_full(
    *, manifest_path: Path, device_name: str, out: Path, weights: str, steps: int,
    image_size: int, window_frames: int, batch_size: int, lr: float, val_every: int,
    seed: int, max_wall_minutes: float | None, init_checkpoint: Path | None,
    limit_clips: int | None, stride_frames: int = 32, num_workers: int = 4,
    prefetch_factor: int = 2,
) -> dict[str, Any]:
    if steps < 1 or image_size < 16 or window_frames < 1 or batch_size < 1 or lr <= 0 or val_every < 1:
        raise ValueError("full mode requires positive steps/window-frames/batch-size/lr/val-every and image-size >=16")
    if limit_clips is not None and limit_clips < 1:
        raise ValueError("--limit-clips must be >=1")
    if stride_frames < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("--stride-frames and --prefetch-factor must be positive; --num-workers must be >=0")
    if max_wall_minutes is not None and max_wall_minutes <= 0:
        raise ValueError("--max-wall-minutes must be >0")
    device = _validated_device(device_name)
    _seed_everything(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get("artifact_type") != "event_head_public_dataset_manifest":
        raise ValueError("--manifest is not an event-head public dataset manifest")
    _validate_full_training_manifest(manifest)
    train_windows = _manifest_windows(
        manifest, split="train", window_frames=window_frames, limit_clips=limit_clips,
        stride_frames=stride_frames,
    )
    val_windows = _manifest_windows(
        manifest, split="val", window_frames=window_frames, limit_clips=limit_clips,
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
    if init_checkpoint is not None:
        model, initial_payload = load_checkpoint(init_checkpoint, device=device_name)
    else:
        model, initial_payload = EventHead(weights=weights), {}
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_step = int(initial_payload.get("completed_steps", 0))
    if "optimizer_state_dict" in initial_payload:
        optimizer.load_state_dict(initial_payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    if start_step >= steps:
        raise ValueError(f"checkpoint already completed {start_step} steps, not less than --steps {steps}")
    best_val_f1 = float(initial_payload.get("best_val_f1", -1.0))
    config = {
        "device": device_name, "weights": weights, "steps": steps, "image_size": image_size,
        "window_frames": window_frames, "batch_size": batch_size, "lr": lr,
        "val_every": val_every, "seed": seed, "max_wall_minutes": max_wall_minutes,
        "limit_clips": limit_clips, "stride_frames": stride_frames,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    last_path, best_path = out / "last_event_head.pt", out / "best_event_head.pt"
    started = time.monotonic()
    losses: list[float] = []
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
        loss = masked_cross_entropy(
            model(batch["frames"].to(device)), batch["targets"].to(device),
            batch["validity_mask"].to(device),
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
            validation = {"step": completed_steps, **_validation_metrics(model, val_loader, device=device)}
            validations.append(validation)
            elapsed = time.monotonic() - started
            if validation["f1"] > best_val_f1:
                best_val_f1 = float(validation["f1"])
                _save_full_checkpoint(
                    best_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
                    best_val_f1=best_val_f1, config=config, data_manifest_path=manifest_path,
                    data_manifest_sha256=manifest_sha, elapsed_s=elapsed, checkpoint_role="best_by_val_f1",
                )
            _save_full_checkpoint(
                last_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
                best_val_f1=best_val_f1, config=config, data_manifest_path=manifest_path,
                data_manifest_sha256=manifest_sha, elapsed_s=elapsed, checkpoint_role="last",
            )
    elapsed = time.monotonic() - started
    # A wall cap may fire between validations; always preserve the exact latest state.
    _save_full_checkpoint(
        last_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
        best_val_f1=best_val_f1, config=config, data_manifest_path=manifest_path,
        data_manifest_sha256=manifest_sha, elapsed_s=elapsed, checkpoint_role="last",
    )
    if not best_path.is_file():
        validation = {"step": completed_steps, **_validation_metrics(model, val_loader, device=device)}
        validations.append(validation)
        best_val_f1 = float(validation["f1"])
        _save_full_checkpoint(
            best_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
            best_val_f1=best_val_f1, config=config, data_manifest_path=manifest_path,
            data_manifest_sha256=manifest_sha, elapsed_s=elapsed, checkpoint_role="best_by_val_f1",
        )
        _save_full_checkpoint(
            last_path, model=model, optimizer=optimizer, completed_steps=completed_steps,
            best_val_f1=best_val_f1, config=config, data_manifest_path=manifest_path,
            data_manifest_sha256=manifest_sha, elapsed_s=elapsed, checkpoint_role="last",
        )
    report = {
        "schema_version": 1, "artifact_type": "event_head_train_manifest",
        "verified": False, "smoke_verified": False, "mode": "full",
        "status": "partial_wall_stop" if wall_stopped else "complete",
        "honest_partial": wall_stopped, "git_head": _git_head(),
        "data_manifest": str(manifest_path), "data_manifest_sha256": manifest_sha,
        "seed": seed, "config": config, "license_posture": "RD_ONLY",
        "license_reason": "RD_ONLY: full pretrain consumes public broadcast pixels and may include NC-licensed pixels",
        "train_windows": len(train_windows), "val_windows": len(val_windows),
        "start_step": start_step, "completed_steps": completed_steps, "target_steps": steps,
        "losses": losses, "all_losses_finite": all(math.isfinite(value) for value in losses),
        "validations": validations, "best_val_f1": best_val_f1,
        "best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
        "elapsed_s": elapsed, "steps_per_s": (completed_steps - start_step) / elapsed if elapsed else 0.0,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
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
    parser.add_argument("--init-checkpoint", type=Path)
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
            )
        except (RuntimeError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            parser.exit(3, f"event-head full train failed: {exc}\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
