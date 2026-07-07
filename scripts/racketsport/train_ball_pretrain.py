#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.roboflow_corpus import (  # noqa: E402
    DEFAULT_BALL_PRETRAIN_FRAMES_IN,
    DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
    DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
    DEFAULT_CORE_TO_AUX_RATIO,
    DEFAULT_EVAL_SAMPLE_EVERY_S,
    DEFAULT_PROTECTED_EVAL_HASH_COUNT,
    RoboflowBallPretrainDataset,
)


ARTIFACT_TYPE = "racketsport_ball_pretrain_run"
MODEL_FAMILIES = ("wasb_hrnet", "tiny_wasb", "tracknet_tiny")
DEFAULT_CORPUS_INDEX = Path("data/roboflow_universe_20260706/aggregated/corpus_index.json")
DEFAULT_CONFIG_PATH = Path("configs/racketsport/ball_pretrain_roboflow_wasb.json")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser(_config_defaults(argv))
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except ModuleNotFoundError as exc:
        print(f"torch-gated ball pretrain skipped: missing module {exc.name}", file=sys.stderr)
        return 5
    except Exception as exc:
        print(f"ball pretrain failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_cli_summary(summary), indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    image_size = _parse_image_size(args.image_size)
    protected_eval_hashes = _parse_protected_eval_hashes(args.protected_eval_hash)
    image_path_rewrites = _parse_image_path_rewrites(args.image_root_rewrite)
    device = _device(args.device, torch=torch)
    mode = str(args.mode)

    train_dataset = None
    train_loader = None
    if mode in {"train", "smoke"}:
        train_dataset = RoboflowBallPretrainDataset(
            args.corpus_index,
            split_role="train",
            image_size=image_size,
            frames_in=int(args.frames_in),
            heatmap_radius_px=float(args.heatmap_radius_px),
            core_to_aux_ratio=int(args.core_to_aux_ratio) if args.core_to_aux_ratio is not None else None,
            seed=int(args.seed),
            max_samples=args.max_train_samples,
            protected_eval_hashes=protected_eval_hashes,
            eval_root=args.eval_root,
            eval_sample_every_s=float(args.eval_sample_every_s),
            expected_protected_eval_hash_count=args.expected_protected_eval_hash_count,
            collision_hamming_threshold=int(args.collision_hamming_threshold),
            skip_list_path=out_dir / "skip_list.json",
            skip_policy=str(args.skip_policy),
            image_path_rewrites=image_path_rewrites,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            generator=_loader_generator(int(args.seed), torch=torch),
            collate_fn=_collate_batch,
        )

    val_dataset = RoboflowBallPretrainDataset(
        args.corpus_index,
        split_role="internal_val",
        image_size=image_size,
        frames_in=int(args.frames_in),
        heatmap_radius_px=float(args.heatmap_radius_px),
        core_to_aux_ratio=None,
        seed=int(args.seed),
        max_samples=args.max_val_samples,
        include_aux_in_internal_val=bool(args.include_aux_val),
        protected_eval_hashes=protected_eval_hashes,
        eval_root=args.eval_root,
        eval_sample_every_s=float(args.eval_sample_every_s),
        expected_protected_eval_hash_count=args.expected_protected_eval_hash_count,
        collision_hamming_threshold=int(args.collision_hamming_threshold),
        skip_list_path=out_dir / "internal_val_skip_list.json",
        skip_policy=str(args.skip_policy),
        image_path_rewrites=image_path_rewrites,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=_collate_batch,
    )

    model = build_model(
        model_family=str(args.model_family),
        frames_in=int(args.frames_in),
        output_channels=int(args.output_channels),
        image_size=image_size,
        wasb_repo=Path(args.wasb_repo),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    resume_summary = None
    global_step = 0
    if args.resume_checkpoint is not None:
        resume_summary = load_training_checkpoint(
            Path(args.resume_checkpoint),
            model=model,
            optimizer=optimizer,
            device=device,
        )
        global_step = int(resume_summary["step"])
    init_summary = None
    if args.init_checkpoint is not None:
        init_summary = load_model_weights(
            Path(args.init_checkpoint),
            model=model,
            device=device,
            strict=bool(args.strict_init),
        )

    zero_shot = None
    if bool(args.zero_shot_baseline):
        zero_shot = evaluate(model, val_loader, device=device, threshold=float(args.visible_threshold))

    losses: list[float] = []
    latest_checkpoint = None
    checkpoint_round_trip = None
    if mode in {"train", "smoke"}:
        if train_loader is None or train_dataset is None or len(train_dataset) == 0:
            raise ValueError("training mode requires at least one train sample")
        batches = cycle(train_loader)
        model.train()
        for local_step in range(1, int(args.steps) + 1):
            batch = next(batches)
            loss = train_one_batch(
                model,
                batch,
                optimizer=optimizer,
                device=device,
                torch=torch,
            )
            global_step += 1
            losses.append(loss)
            if int(args.checkpoint_every) > 0 and global_step % int(args.checkpoint_every) == 0:
                latest_checkpoint = save_training_checkpoint(
                    checkpoint_dir / f"checkpoint_step_{global_step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    args=args,
                    train_dataset_summary=train_dataset.summary,
                    val_dataset_summary=val_dataset.summary,
                )
        latest_checkpoint = save_training_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            step=global_step,
            args=args,
            train_dataset_summary=train_dataset.summary,
            val_dataset_summary=val_dataset.summary,
        )
        checkpoint_round_trip = checkpoint_round_trip_summary(
            latest_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    internal_val_metrics = evaluate(model, val_loader, device=device, threshold=float(args.visible_threshold))
    elapsed = time.perf_counter() - start
    loss_summary = _loss_summary(losses)
    status = "eval_complete"
    if mode == "train":
        status = "train_complete"
    if mode == "smoke":
        status = (
            "smoke_passed"
            if loss_summary.get("strictly_decreased") is True
            and checkpoint_round_trip is not None
            and checkpoint_round_trip.get("round_trip_state_sha256_match") is True
            else "smoke_failed"
        )
    summary = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        "mode": mode,
        "ball_verified": False,
        "promotion_claimed": False,
        "public_pretrain_only": True,
        "heldout_touched": False,
        "corpus_index": str(args.corpus_index),
        "out_dir": str(out_dir),
        "model": {
            "family": args.model_family,
            "frames_in": int(args.frames_in),
            "output_channels": int(args.output_channels),
            "image_size": list(image_size),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "init_summary": init_summary,
            "resume_summary": resume_summary,
        },
        "data": {
            "train": train_dataset.summary if train_dataset is not None else None,
            "internal_val": val_dataset.summary,
            "image_path_rewrites": image_path_rewrites,
            "mixing_decision": (
                f"default core_pickleball:adjacent_sport_aux ratio is {args.core_to_aux_ratio}:1; "
                "aux sport samples are a small regularizer, not pickleball diversity."
            ),
        },
        "loss": loss_summary,
        "zero_shot_baseline": zero_shot,
        "internal_val": {
            "metrics": internal_val_metrics,
            "bounded_subsample": {
                "max_val_samples": args.max_val_samples,
                "seed": int(args.seed),
                "justification": "CPU prep lane uses a deterministic bounded internal_val subsample when requested; GPU lane should lift the cap.",
            },
        },
        "checkpoint": {
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
            **(checkpoint_round_trip or {}),
            "atomic_policy": "torch.save to sibling .tmp then os.replace",
        },
        "runtime": {
            "wall_seconds": elapsed,
            "device": str(device),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "limitations": [
            "Internal-val public corpus metrics are not held-out owner proof.",
            "Public labels have visibility_level absent; WBCE weight defaults to 1 without inventing occlusion.",
            "This harness is prep for a later GPU run; do not claim BALL promotion from this lane.",
        ],
    }
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "internal_val_metrics.json", internal_val_metrics)
    if zero_shot is not None:
        _write_json(out_dir / "zero_shot_baseline.json", zero_shot)
    return summary


def build_model(
    *,
    model_family: str,
    frames_in: int,
    output_channels: int,
    image_size: tuple[int, int],
    wasb_repo: Path,
) -> Any:
    torch = _torch()
    if model_family == "tiny_wasb":
        return _make_tiny_wasb(in_channels=frames_in * 3, out_channels=output_channels, torch=torch)
    if model_family == "tracknet_tiny":
        return _make_tracknet_tiny(in_channels=frames_in * 3, out_channels=output_channels, torch=torch)
    if model_family != "wasb_hrnet":
        raise ValueError(f"unsupported model_family: {model_family}")
    src = wasb_repo / "src"
    if not (src / "models" / "__init__.py").is_file():
        raise FileNotFoundError(f"missing WASB-SBDT source tree: {src}")
    if str(src.resolve()) not in sys.path:
        sys.path.insert(0, str(src.resolve()))
    import yaml
    from omegaconf import OmegaConf
    from models import build_model as build_wasb_model

    cfg = yaml.safe_load((src / "configs" / "model" / "wasb.yaml").read_text(encoding="utf-8"))
    cfg["frames_in"] = int(frames_in)
    cfg["frames_out"] = int(output_channels)
    cfg["inp_width"] = int(image_size[0])
    cfg["inp_height"] = int(image_size[1])
    cfg["out_width"] = int(image_size[0])
    cfg["out_height"] = int(image_size[1])
    return build_wasb_model(OmegaConf.create({"model": cfg}))


def _make_tiny_wasb(*, in_channels: int, out_channels: int, torch: Any) -> Any:
    class TinyWasbNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, out_channels, kernel_size=1),
            )

        def forward(self, inputs: Any) -> Any:
            return self.net(inputs)

    return TinyWasbNet()


def _make_tracknet_tiny(*, in_channels: int, out_channels: int, torch: Any) -> Any:
    class TrackNetTiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                torch.nn.Conv2d(64, out_channels, kernel_size=1),
            )

        def forward(self, inputs: Any) -> Any:
            return self.net(inputs)

    return TrackNetTiny()


def train_one_batch(model: Any, batch: Mapping[str, Any], *, optimizer: Any, device: Any, torch: Any) -> float:
    inputs = batch["input"].to(device)
    target = batch["target"].to(device)
    weights = batch["wbce_weight"].to(device).view(-1)
    optimizer.zero_grad(set_to_none=True)
    logits = _primary_logits(model(inputs))
    if logits.shape[-2:] != target.shape[-2:]:
        logits = torch.nn.functional.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    target = target.repeat(1, logits.shape[1], 1, 1)
    loss_map = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    sample_loss = loss_map.flatten(1).mean(dim=1) * weights
    loss = sample_loss.mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _collate_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    collated: dict[str, Any] = {
        "input": torch.utils.data.default_collate([item["input"] for item in items]),
        "target": torch.utils.data.default_collate([item["target"] for item in items]),
        "target_xy_px": torch.utils.data.default_collate([item["target_xy_px"] for item in items]),
        "wbce_weight": torch.utils.data.default_collate([item["wbce_weight"] for item in items]),
        "ball_present": torch.utils.data.default_collate([item["ball_present"] for item in items]),
    }
    for key in (
        "sample_id",
        "source_slug",
        "bucket",
        "source_split",
        "image_path",
        "window_sample_ids",
        "temporal_sample_kind",
        "visibility_level",
    ):
        collated[key] = [item[key] for item in items]
    return collated


def _primary_logits(output: Any) -> Any:
    if isinstance(output, Mapping):
        for key in (0, "0", "out", "logits", "heatmap"):
            if key in output:
                return output[key]
        if not output:
            raise ValueError("model returned an empty mapping")
        first_key = sorted(output.keys(), key=lambda key: str(key))[0]
        return output[first_key]
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("model returned an empty sequence")
        return output[0]
    return output


def evaluate(model: Any, loader: Any, *, device: Any, threshold: float) -> dict[str, Any]:
    torch = _torch()
    radii = (5.0, 10.0, 20.0, 40.0)
    hits = {radius: 0 for radius in radii}
    pred_visible = 0
    distances: list[float] = []
    sample_count = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            target_xy = batch["target_xy_px"].to(device)
            logits = _primary_logits(model(inputs))
            if logits.ndim != 4:
                raise ValueError(f"model output must be BCHW, got shape {tuple(logits.shape)}")
            probs = torch.sigmoid(logits[:, 0])
            flat = probs.flatten(1)
            peak_conf, peak_index = flat.max(dim=1)
            width = probs.shape[-1]
            pred_x = (peak_index % width).to(torch.float32)
            pred_y = (peak_index // width).to(torch.float32)
            batch_dist = torch.sqrt((pred_x - target_xy[:, 0]) ** 2 + (pred_y - target_xy[:, 1]) ** 2)
            for conf, distance in zip(peak_conf.detach().cpu().tolist(), batch_dist.detach().cpu().tolist(), strict=True):
                sample_count += 1
                distances.append(float(distance))
                if float(conf) >= threshold:
                    pred_visible += 1
                    for radius in radii:
                        if float(distance) <= radius:
                            hits[radius] += 1
    metrics: dict[str, Any] = {
        "sample_count": sample_count,
        "visible_label_count": sample_count,
        "visible_prediction_count": pred_visible,
        "visible_threshold": threshold,
        "median_error_px": _percentile(distances, 50) if distances else None,
        "p90_error_px": _percentile(distances, 90) if distances else None,
    }
    for radius in radii:
        precision = _ratio(hits[radius], pred_visible)
        recall = _ratio(hits[radius], sample_count)
        metrics[f"precision_at_{int(radius)}px"] = precision
        metrics[f"recall_at_{int(radius)}px"] = recall
        metrics[f"f1_at_{int(radius)}px"] = _f1(precision, recall)
    return metrics


def save_training_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    step: int,
    args: argparse.Namespace,
    train_dataset_summary: Mapping[str, Any],
    val_dataset_summary: Mapping[str, Any],
) -> Path:
    torch = _torch()
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_pretrain_checkpoint",
        "step": int(step),
        "model_family": args.model_family,
        "frames_in": int(args.frames_in),
        "output_channels": int(args.output_channels),
        "image_size": list(_parse_image_size(args.image_size)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "train_dataset_summary": dict(train_dataset_summary),
        "internal_val_dataset_summary": dict(val_dataset_summary),
    }
    atomic_torch_save(payload, path, torch=torch)
    return path


def atomic_torch_save(payload: Mapping[str, Any], path: Path, *, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def checkpoint_round_trip_summary(path: Path, *, model: Any, optimizer: Any, device: Any) -> dict[str, Any]:
    before = state_dict_sha256(model.state_dict())
    payload = _torch().load(path, map_location=device, weights_only=False)
    loaded = payload["model_state_dict"]
    after = state_dict_sha256(loaded)
    return {
        "round_trip_state_sha256_match": before == after,
        "state_sha256": before,
        "loaded_state_sha256": after,
        "step": int(payload["step"]),
    }


def load_training_checkpoint(path: Path, *, model: Any, optimizer: Any, device: Any) -> dict[str, Any]:
    payload = _torch().load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {"checkpoint": str(path), "step": int(payload["step"])}


def load_model_weights(path: Path, *, model: Any, device: Any, strict: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing init checkpoint: {path}")
    payload = _torch().load(path, map_location=device, weights_only=False)
    state = _extract_state_dict(payload)
    result = model.load_state_dict(_strip_module_prefix(state), strict=strict)
    return {
        "checkpoint": str(path),
        "strict": strict,
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
        "loaded_state_sha256": state_dict_sha256(model.state_dict()),
    }


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        digest.update(str(key).encode("utf-8"))
        if hasattr(tensor, "detach"):
            cpu = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(cpu.shape)).encode("utf-8"))
            digest.update(str(cpu.dtype).encode("utf-8"))
            digest.update(cpu.numpy().tobytes())
        else:
            digest.update(repr(tensor).encode("utf-8"))
    return digest.hexdigest()


def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
    if isinstance(payload, Mapping):
        return payload
    raise ValueError("checkpoint does not contain a state dict")


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).removeprefix("module."): value for key, value in state_dict.items()}


def _cli_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": summary["status"],
        "summary_json": str(Path(summary["out_dir"]) / "summary.json"),
        "checkpoint": summary["checkpoint"],
        "loss": summary["loss"],
        "internal_val": summary["internal_val"],
        "zero_shot_baseline": summary.get("zero_shot_baseline"),
        "runtime": summary["runtime"],
    }


def _loss_summary(losses: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(losses),
        "first": float(losses[0]) if losses else None,
        "last": float(losses[-1]) if losses else None,
        "strictly_decreased": bool(losses and losses[-1] < losses[0]),
        "values": [float(value) for value in losses],
    }


def _parse_image_size(value: str | Sequence[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, str):
        normalized = value.lower().replace(",", "x")
        parts = [part for part in normalized.split("x") if part]
        if len(parts) != 2:
            raise ValueError("image size must be WIDTHxHEIGHT")
        width, height = int(parts[0]), int(parts[1])
    else:
        width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    return width, height


def _parse_protected_eval_hashes(items: Sequence[str] | None) -> dict[str, list[str]] | None:
    if not items:
        return None
    parsed: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--protected-eval-hash must be clip_id=hexhash")
        clip_id, value = item.split("=", 1)
        if not clip_id or not value:
            raise ValueError("--protected-eval-hash must include both clip_id and hash")
        parsed.setdefault(clip_id, []).append(value)
    return parsed


def _parse_image_path_rewrites(items: Sequence[str] | Mapping[str, str] | None) -> dict[str, str]:
    if not items:
        return {}
    if isinstance(items, Mapping):
        iterable = [f"{old}={new}" for old, new in items.items()]
    else:
        iterable = [str(item) for item in items]
    rewrites: dict[str, str] = {}
    for item in iterable:
        if "=" not in item:
            raise ValueError("--image-root-rewrite must be OLD_PREFIX=NEW_PREFIX")
        old_prefix, new_prefix = item.split("=", 1)
        old = old_prefix.rstrip("/")
        new = new_prefix.rstrip("/")
        if not old or not new:
            raise ValueError("--image-root-rewrite prefixes must be non-empty")
        rewrites[old] = new
    return dict(sorted(rewrites.items(), key=lambda pair: len(pair[0]), reverse=True))


def _loader_generator(seed: int, *, torch: Any) -> Any:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _device(value: str, *, torch: Any) -> Any:
    if value == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if value == "mps" and not getattr(torch.backends, "mps", None):
        return torch.device("cpu")
    if value == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(value)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / float(denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return 0.0 if precision is not None and recall is not None else None
    return 2.0 * precision * recall / (precision + recall)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch() -> Any:
    import torch

    return torch


def _config_defaults(argv: list[str] | None) -> dict[str, Any]:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path, default=None)
    known, _ = probe.parse_known_args(argv)
    if known.config is None:
        return {}
    payload = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "racketsport_ball_pretrain_config":
        raise ValueError(f"unexpected config artifact_type: {payload.get('artifact_type')}")
    return {key.replace("-", "_"): value for key, value in payload.get("defaults", {}).items()}


def _build_parser(defaults: Mapping[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain/evaluate the BALL warm-start harness on the index-based Roboflow corpus.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.is_file() else None)
    parser.add_argument("--mode", choices=("train", "eval", "smoke"), default=defaults.get("mode", "train"))
    parser.add_argument("--corpus-index", type=Path, default=Path(defaults.get("corpus_index", DEFAULT_CORPUS_INDEX)))
    parser.add_argument("--out-dir", type=Path, required="out_dir" not in defaults, default=defaults.get("out_dir"))
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, default=defaults.get("model_family", "wasb_hrnet"))
    parser.add_argument("--wasb-repo", type=Path, default=Path(defaults.get("wasb_repo", "third_party/WASB-SBDT")))
    parser.add_argument("--init-checkpoint", type=Path, default=defaults.get("init_checkpoint"))
    parser.add_argument("--resume-checkpoint", type=Path, default=defaults.get("resume_checkpoint"))
    parser.add_argument("--strict-init", action="store_true", default=bool(defaults.get("strict_init", False)))
    parser.add_argument("--zero-shot-baseline", action="store_true", default=bool(defaults.get("zero_shot_baseline", False)))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=defaults.get("device", "cuda"))
    parser.add_argument("--image-size", default=defaults.get("image_size", f"{DEFAULT_BALL_PRETRAIN_IMAGE_SIZE[0]}x{DEFAULT_BALL_PRETRAIN_IMAGE_SIZE[1]}"))
    parser.add_argument("--frames-in", type=int, default=int(defaults.get("frames_in", DEFAULT_BALL_PRETRAIN_FRAMES_IN)))
    parser.add_argument("--output-channels", type=int, default=int(defaults.get("output_channels", 1)))
    parser.add_argument("--heatmap-radius-px", type=float, default=float(defaults.get("heatmap_radius_px", DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX)))
    parser.add_argument("--core-to-aux-ratio", type=int, default=int(defaults.get("core_to_aux_ratio", DEFAULT_CORE_TO_AUX_RATIO)))
    parser.add_argument("--include-aux-val", action="store_true", default=bool(defaults.get("include_aux_val", False)))
    parser.add_argument("--steps", type=int, default=int(defaults.get("steps", 1000)))
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 8)))
    parser.add_argument("--learning-rate", type=float, default=float(defaults.get("learning_rate", 5e-4)))
    parser.add_argument("--weight-decay", type=float, default=float(defaults.get("weight_decay", 5e-5)))
    parser.add_argument("--checkpoint-every", type=int, default=int(defaults.get("checkpoint_every", 500)))
    parser.add_argument("--num-workers", type=int, default=int(defaults.get("num_workers", 0)))
    parser.add_argument("--max-train-samples", type=int, default=defaults.get("max_train_samples"))
    parser.add_argument("--max-val-samples", type=int, default=defaults.get("max_val_samples"))
    parser.add_argument("--visible-threshold", type=float, default=float(defaults.get("visible_threshold", 0.5)))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 1337)))
    parser.add_argument("--eval-root", type=Path, default=Path(defaults.get("eval_root", "eval_clips/ball")))
    parser.add_argument("--eval-sample-every-s", type=float, default=float(defaults.get("eval_sample_every_s", DEFAULT_EVAL_SAMPLE_EVERY_S)))
    parser.add_argument("--expected-protected-eval-hash-count", type=int, default=int(defaults.get("expected_protected_eval_hash_count", DEFAULT_PROTECTED_EVAL_HASH_COUNT)))
    parser.add_argument("--collision-hamming-threshold", type=int, default=int(defaults.get("collision_hamming_threshold", 3)))
    parser.add_argument("--protected-eval-hash", action="append", default=defaults.get("protected_eval_hash", []))
    parser.add_argument(
        "--image-root-rewrite",
        action="append",
        default=defaults.get("image_root_rewrite", []),
        help="Rewrite absolute image paths as OLD_PREFIX=NEW_PREFIX for VM/checkouts at a different root.",
    )
    parser.add_argument("--skip-policy", choices=("fail", "skip"), default=defaults.get("skip_policy", "fail"))
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
