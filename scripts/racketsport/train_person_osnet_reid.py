#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.eval_guard import assert_not_training_on_eval_clip  # noqa: E402
from threed.racketsport.person_reid_diagnostics import (  # noqa: E402
    _extract_state_dict,
    _filter_compatible_state_dict,
    _torch_load_checkpoint,
)


DATASET_NAME = "pickleball_person_reid"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune OSNet on a labeled pickleball person ReID crop dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing the ReID crop manifest.json.")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--weights", type=Path, default=None, help="Optional checkpoint to initialize from.")
    parser.add_argument("--loss", choices=("softmax", "triplet"), default="softmax")
    parser.add_argument("--max-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--lr-scheduler", default="single_step")
    parser.add_argument("--stepsize", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--eval-freq", type=int, default=-1)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    try:
        summary = train_osnet_reid(args)
    except Exception as exc:
        print(f"OSNet ReID training failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


def train_osnet_reid(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import torchreid  # noqa: F401  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("torch and torchreid are required for OSNet ReID fine-tuning") from exc

    # Deliberately outside the try/except above: torchreid 1.4.0 (the pinned,
    # installed version -- see runs/phase2/trk_offline_authority_20260701T205912Z/
    # COMMANDS.sh) exposes these under `torchreid.data.*` / `torchreid.engine`,
    # not `torchreid.reid.*`. A prior version of this script imported the
    # nonexistent `torchreid.reid.*` paths, which raise ModuleNotFoundError (an
    # ImportError subclass) on every real run; catching that in the same
    # except as "torch/torchreid aren't installed" silently misreported a real
    # code bug as a missing dependency even though both packages were present.
    # Letting these raise unguarded means any future path regression surfaces
    # its real error instead of being masked as "install torch and torchreid".
    from torchreid import data, models, optim  # type: ignore[import-not-found]
    from torchreid.data.datasets.dataset import ImageDataset  # type: ignore[import-not-found]
    from torchreid.engine import ImageSoftmaxEngine, ImageTripletEngine  # type: ignore[import-not-found]

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    _validate_manifest_for_training(manifest)
    eval_guard_summary = _assert_manifest_clips_are_not_protected(manifest)

    dataset_class = _dataset_class(manifest_path, ImageDataset)
    try:
        data.register_image_dataset(DATASET_NAME, dataset_class)
    except ValueError as exc:
        if "already exists" not in str(exc):
            raise

    use_gpu = bool(torch.cuda.is_available() and not args.cpu)
    # RandomIdentitySampler groups each training batch into num_instances
    # crops per identity, which is what triplet loss needs to mine
    # positive/negative pairs within a batch. RandomSampler (plain shuffling)
    # makes --num-instances a no-op and is only correct for softmax loss.
    train_sampler = "RandomIdentitySampler" if args.loss == "triplet" else "RandomSampler"
    datamanager = data.ImageDataManager(
        root=str(dataset_dir),
        sources=DATASET_NAME,
        targets=DATASET_NAME,
        height=args.height,
        width=args.width,
        transforms="random_flip",
        use_gpu=use_gpu,
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size_test,
        workers=args.workers,
        num_instances=args.num_instances,
        train_sampler=train_sampler,
    )

    model = models.build_model(
        name=args.model_name,
        num_classes=datamanager.num_train_pids,
        loss=args.loss,
        pretrained=args.weights is None,
        use_gpu=use_gpu,
    )
    loaded_weight_summary: dict[str, Any] | None = None
    if args.weights is not None:
        loaded_weight_summary = _load_compatible_weights(torch, model, args.weights)
    if use_gpu:
        model = model.cuda()

    optimizer = optim.build_optimizer(
        model,
        optim=args.optim,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.build_lr_scheduler(
        optimizer,
        lr_scheduler=args.lr_scheduler,
        stepsize=args.stepsize,
        gamma=args.gamma,
        max_epoch=args.max_epoch,
    )
    if args.loss == "triplet":
        engine = ImageTripletEngine(datamanager, model, optimizer, scheduler=scheduler, use_gpu=use_gpu)
    else:
        engine = ImageSoftmaxEngine(datamanager, model, optimizer, scheduler=scheduler, use_gpu=use_gpu)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    engine.run(
        save_dir=str(args.save_dir),
        max_epoch=args.max_epoch,
        print_freq=args.print_freq,
        eval_freq=args.eval_freq,
        test_only=args.test_only,
        normalize_feature=True,
        ranks=[1, 2, 4],
    )

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_osnet_reid_training",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "save_dir": str(args.save_dir),
        "model_name": args.model_name,
        "loss": args.loss,
        "use_gpu": use_gpu,
        "cuda_device_count": int(torch.cuda.device_count()) if hasattr(torch.cuda, "device_count") else 0,
        "train_identity_count": int(datamanager.num_train_pids),
        "train_image_count": int(manifest["split_counts"]["train"]),
        "query_image_count": int(manifest["split_counts"]["query"]),
        "gallery_image_count": int(manifest["split_counts"]["gallery"]),
        "base_weights": str(args.weights) if args.weights is not None else None,
        "loaded_weight_summary": loaded_weight_summary,
        "latest_checkpoint": _latest_checkpoint(args.save_dir),
        "eval_guard": eval_guard_summary,
        "args": _json_safe(vars(args)),
        "notes": [
            "This checkpoint is trained from reviewed CVAT player crops.",
            "Report whether downstream track scoring used train-set clips or held-out clips before promotion decisions.",
        ],
    }
    (args.save_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _dataset_class(manifest_path: Path, image_dataset_base: type) -> type:
    class PickleballPersonReIDDataset(image_dataset_base):  # type: ignore[misc, valid-type]
        dataset_dir = ""

        def __init__(self, root: str = "", **kwargs: Any) -> None:
            root_path = Path(root)
            manifest = _read_manifest(root_path / "manifest.json" if root_path else manifest_path)
            rows = manifest.get("rows", [])
            train = _rows_for_split(root_path, rows, "train")
            query = _rows_for_split(root_path, rows, "query")
            gallery = _rows_for_split(root_path, rows, "gallery")
            super().__init__(train, query, gallery, **kwargs)

    return PickleballPersonReIDDataset


def _rows_for_split(root: Path, rows: Any, split: str) -> list[tuple[str, int, int]]:
    if not isinstance(rows, list):
        raise ValueError("ReID dataset manifest rows must be a list")
    converted: list[tuple[str, int, int]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("split") != split:
            continue
        rel = row.get("relative_image_path")
        image_path = root / str(rel) if rel else Path(str(row["image_path"]))
        converted.append((str(image_path), int(row["pid"]), int(row["camid"])))
    return converted


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ReID dataset manifest must be a JSON object")
    return payload


def _validate_manifest_for_training(manifest: Mapping[str, Any]) -> None:
    if manifest.get("artifact_type") != "racketsport_person_reid_crop_dataset":
        raise ValueError("dataset manifest is not a person ReID crop dataset")
    if not bool(manifest.get("uses_cvat_labels", False)):
        raise ValueError("person ReID training requires reviewed CVAT labels")
    split_counts = manifest.get("split_counts")
    if not isinstance(split_counts, Mapping):
        raise ValueError("dataset manifest missing split_counts")
    for split in ("train", "query", "gallery"):
        if int(split_counts.get(split, 0)) <= 0:
            raise ValueError(f"dataset split has no images: {split}")


def _assert_manifest_clips_are_not_protected(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if this crop dataset trains/internally-validates on a protected eval clip.

    ``clip_counts`` (written by ``threed/racketsport/person_reid_dataset.py``)
    maps each source clip id to its per-split crop counts. A clip that
    contributed at least one ``train``-split crop is actual training data and
    is refused unconditionally. A clip that only contributed ``query``/
    ``gallery`` crops is used purely as a validation-during-fitting retrieval
    signal, so Burlington/Wolverine are allowed there (Outdoor/Indoor are
    still refused -- see threed/racketsport/eval_guard.py).
    """
    clip_counts = manifest.get("clip_counts")
    if not isinstance(clip_counts, Mapping):
        return {"status": "no_clip_counts_in_manifest"}

    train_clip_ids: list[str] = []
    val_only_clip_ids: list[str] = []
    for clip_id, counts in clip_counts.items():
        train_count = int(counts.get("train", 0) or 0) if isinstance(counts, Mapping) else 0
        (train_clip_ids if train_count > 0 else val_only_clip_ids).append(str(clip_id))

    assert_not_training_on_eval_clip(train_clip_ids, allow_internal_val=False)
    return assert_not_training_on_eval_clip(val_only_clip_ids, allow_internal_val=True)


def _load_compatible_weights(torch: Any, model: Any, path: Path) -> dict[str, Any]:
    checkpoint = _torch_load_checkpoint(torch, path)
    state_dict = _extract_state_dict(checkpoint)
    compatible, skipped = _filter_compatible_state_dict(state_dict, model.state_dict())
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "path": str(path),
        "loaded_tensor_count": len(compatible),
        "skipped_tensor_count": len(skipped),
        "skipped_tensors": skipped[:20],
        "missing_keys": list(missing)[:20],
        "unexpected_keys": list(unexpected)[:20],
    }


def _latest_checkpoint(save_dir: Path) -> str | None:
    candidates = [path for path in save_dir.rglob("*") if path.is_file() and (".pth" in path.name or ".pt" in path.name)]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(latest)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
