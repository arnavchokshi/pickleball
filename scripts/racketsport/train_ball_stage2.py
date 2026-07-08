#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.train_ball_pretrain import (  # noqa: E402
    MODEL_FAMILIES,
    _collate_batch,
    _device,
    _loader_generator,
    _parse_image_path_rewrites,
    _parse_image_size,
    _parse_protected_eval_hashes,
    _primary_logits,
    _seed_loader_worker,
    _seed_training_process,
    atomic_torch_save,
    build_model,
    checkpoint_round_trip_summary,
    load_model_weights,
    state_dict_sha256,
    train_one_batch,
)
from threed.racketsport.ball_sst_dataset import (  # noqa: E402
    build_sst_manifest,
    iter_sst_manifest_samples,
)
from threed.racketsport.ball_tracknet_cvat_dataset import TrackNetCvatLabel  # noqa: E402
from threed.racketsport.cvat_video import import_cvat_video_zip  # noqa: E402
from threed.racketsport.roboflow_corpus import (  # noqa: E402
    DEFAULT_BALL_PRETRAIN_FRAMES_IN,
    DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
    DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_EVAL_SAMPLE_EVERY_S,
    DEFAULT_PROTECTED_EVAL_HASH_COUNT,
)
from threed.racketsport.schemas import (  # noqa: E402
    BALL_VISIBILITY_WBCE_WEIGHTS,
    BallVisibilityLevel,
    CvatVideoAnnotations,
    CvatVideoFrame,
    validate_artifact_file,
)
from threed.racketsport.wasb_adapter import (  # noqa: E402
    _preprocess_wasb_window_official,
    _wasb_affine_transform_xy,
    _wasb_official_input_affine,
)


ARTIFACT_TYPE = "racketsport_ball_stage2_run"
DEFAULT_CVAT_EXPORT_ROOT = Path("cvat_upload/exports/harvest_review_20260707")
DEFAULT_RALLY_ROOT = Path("data/online_harvest_20260706/rallies")
DEFAULT_PRELABEL_ROOT = Path("data/online_harvest_20260706/prelabels")


@dataclass(frozen=True)
class Stage2SampleRecord:
    sample_id: str
    source_kind: str
    clip_id: str
    video_path: Path
    frame_index: int
    source_width: int
    source_height: int
    ball_present: bool
    source_xy_px: tuple[float, float]
    visibility_level: BallVisibilityLevel | None
    wbce_weight: float
    source_path: str


class CvatBallStage2Dataset:
    def __init__(
        self,
        records: Sequence[Stage2SampleRecord],
        *,
        image_size: tuple[int, int] = DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
        frames_in: int = DEFAULT_BALL_PRETRAIN_FRAMES_IN,
        heatmap_radius_px: float = DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
        image_path_rewrites: Mapping[str, str] | Sequence[str] | None = None,
    ) -> None:
        _validate_dataset_shape(image_size=image_size, frames_in=frames_in, heatmap_radius_px=heatmap_radius_px)
        self.records = tuple(records)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.frames_in = int(frames_in)
        self.heatmap_radius_px = float(heatmap_radius_px)
        self.image_path_rewrites = _parse_image_path_rewrites(image_path_rewrites)
        self.summary = _dataset_summary("cvat_owner_sparse", self.records, self.image_size, self.frames_in, self.heatmap_radius_px)

    @classmethod
    def from_export_root(
        cls,
        cvat_export_root: str | Path,
        *,
        rally_root: str | Path = DEFAULT_RALLY_ROOT,
        video_paths: Mapping[str, str | Path] | None = None,
        image_size: tuple[int, int] = DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
        frames_in: int = DEFAULT_BALL_PRETRAIN_FRAMES_IN,
        heatmap_radius_px: float = DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
        image_path_rewrites: Mapping[str, str] | Sequence[str] | None = None,
        max_samples: int | None = None,
    ) -> "CvatBallStage2Dataset":
        root = Path(cvat_export_root)
        if not root.is_dir():
            raise FileNotFoundError(f"missing CVAT export root: {root}")
        normalized_videos = {str(clip): Path(path) for clip, path in (video_paths or {}).items()}
        records: list[Stage2SampleRecord] = []
        for clip_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            annotations = load_cvat_annotations_from_export_clip(clip_dir)
            video_path = normalized_videos.get(annotations.clip_id) or _resolve_rally_video(Path(rally_root), annotations.clip_id)
            labels = sparse_tracknet_labels_from_annotations(annotations)
            for label in labels:
                if max_samples is not None and len(records) >= max_samples:
                    break
                records.append(_record_from_cvat_label(annotations, label, video_path=video_path, source_path=str(clip_dir)))
            if max_samples is not None and len(records) >= max_samples:
                break
        if not records:
            raise ValueError(f"no sparse reviewed CVAT training rows found under {root}")
        return cls(
            records,
            image_size=image_size,
            frames_in=frames_in,
            heatmap_radius_px=heatmap_radius_px,
            image_path_rewrites=image_path_rewrites,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _record_to_item(
            self.records[index],
            image_size=self.image_size,
            frames_in=self.frames_in,
            heatmap_radius_px=self.heatmap_radius_px,
            image_path_rewrites=self.image_path_rewrites,
        )


class SstBallStage2Dataset(CvatBallStage2Dataset):
    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        image_size: tuple[int, int] = DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
        frames_in: int = DEFAULT_BALL_PRETRAIN_FRAMES_IN,
        heatmap_radius_px: float = DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
        image_path_rewrites: Mapping[str, str] | Sequence[str] | None = None,
        max_samples: int | None = None,
    ) -> "SstBallStage2Dataset":
        records: list[Stage2SampleRecord] = []
        size_cache: dict[Path, tuple[int, int]] = {}
        for sample in iter_sst_manifest_samples(manifest_path):
            if max_samples is not None and len(records) >= max_samples:
                break
            frame_ref = sample.get("frame_ref")
            if not isinstance(frame_ref, Mapping):
                raise ValueError(f"SST sample missing frame_ref: {sample.get('sample_id')}")
            video_path = Path(str(frame_ref.get("video") or ""))
            if not video_path.is_file():
                raise FileNotFoundError(f"missing SST frame video: {video_path}")
            source_width, source_height = size_cache.setdefault(video_path, _video_size(video_path))
            xy = sample.get("teacher_xy")
            if not isinstance(xy, Sequence) or isinstance(xy, (str, bytes)) or len(xy) != 2:
                raise ValueError(f"SST sample teacher_xy must be [x, y]: {sample.get('sample_id')}")
            weight = _finite_float(sample.get("weight"), f"SST sample {sample.get('sample_id')} weight")
            records.append(
                Stage2SampleRecord(
                    sample_id=str(sample.get("sample_id") or f"{sample.get('clip_id')}:{sample.get('frame_index')}"),
                    source_kind="sst_pseudo_label",
                    clip_id=str(sample["clip_id"]),
                    video_path=video_path,
                    frame_index=int(sample["frame_index"]),
                    source_width=source_width,
                    source_height=source_height,
                    ball_present=True,
                    source_xy_px=(float(xy[0]), float(xy[1])),
                    visibility_level=None,
                    wbce_weight=max(0.0, min(1.0, weight)),
                    source_path=str(manifest_path),
                )
            )
        if not records:
            raise ValueError(f"SST manifest contains no student samples: {manifest_path}")
        dataset = cls(
            records,
            image_size=image_size,
            frames_in=frames_in,
            heatmap_radius_px=heatmap_radius_px,
            image_path_rewrites=image_path_rewrites,
        )
        dataset.summary["source_kind"] = "sst_pseudo_label"
        dataset.summary["weight_policy"] = "weight = clamp(score, 0.0, 1.0)"
        return dataset


class CombinedStage2Dataset:
    def __init__(self, datasets: Sequence[Any]) -> None:
        self.datasets = tuple(datasets)
        if not self.datasets:
            raise ValueError("at least one stage-2 data source is required")
        self.offsets: list[tuple[int, int, Any]] = []
        total = 0
        for dataset in self.datasets:
            length = len(dataset)
            self.offsets.append((total, total + length, dataset))
            total += length
        self.summary = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_stage2_combined_dataset_summary",
            "source_count": len(self.datasets),
            "selected_sample_count": total,
            "sources": [getattr(dataset, "summary", {}) for dataset in self.datasets],
        }

    def __len__(self) -> int:
        return self.offsets[-1][1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        for start, end, dataset in self.offsets:
            if start <= index < end:
                return dataset[index - start]
        raise IndexError(index)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "build-sst-manifest":
            summary = run_build_sst_manifest(args)
        else:
            summary = run(args)
    except ModuleNotFoundError as exc:
        print(f"torch-gated ball stage2 skipped: missing module {exc.name}", file=sys.stderr)
        return 5
    except Exception as exc:
        print(f"ball stage2 failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_cli_summary(summary), indent=2, sort_keys=True))
    return 0


def run_build_sst_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.sst_manifest_out is None:
        raise ValueError("--mode build-sst-manifest requires --sst-manifest-out")
    protected_eval_hashes = _parse_protected_eval_hashes(args.protected_eval_hash)
    return build_sst_manifest(
        prelabel_root=args.prelabel_root,
        rally_root=args.rally_root,
        out_path=args.sst_manifest_out,
        clips=args.clip,
        max_samples_per_clip=args.max_sst_samples_per_clip,
        protected_eval_hashes=protected_eval_hashes,
        expected_protected_eval_hash_count=args.expected_protected_eval_hash_count,
        eval_root=args.eval_root,
        eval_sample_every_s=args.eval_sample_every_s,
        collision_hamming_threshold=args.collision_hamming_threshold,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    seed_summary = _seed_training_process(int(args.seed), torch=torch)
    start = time.perf_counter()
    if args.out_dir is None:
        raise ValueError("--out-dir is required for training")
    if int(args.epochs) > 30:
        raise ValueError("--epochs must be <= 30 for bounded stage-2 training")
    image_size = _parse_image_size(args.image_size)
    if float(args.occluded_prob) > 0.0 and float(args.occluded_prob) != 0.25:
        raise ValueError("stage-2 occlusion augmentation must use the pinned occluded_prob=0.25 or be disabled with 0")
    device = _device(args.device, torch=torch)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[Any] = []
    image_path_rewrites = _parse_image_path_rewrites(args.image_root_rewrite)
    if args.cvat_export_root is not None:
        datasets.append(
            CvatBallStage2Dataset.from_export_root(
                args.cvat_export_root,
                rally_root=args.rally_root,
                image_size=image_size,
                frames_in=int(args.frames_in),
                heatmap_radius_px=float(args.heatmap_radius_px),
                image_path_rewrites=image_path_rewrites,
                max_samples=args.max_cvat_samples,
            )
        )
    for manifest_path in args.sst_manifest or []:
        datasets.append(
            SstBallStage2Dataset.from_manifest(
                manifest_path,
                image_size=image_size,
                frames_in=int(args.frames_in),
                heatmap_radius_px=float(args.heatmap_radius_px),
                image_path_rewrites=image_path_rewrites,
                max_samples=args.max_sst_samples,
            )
        )
    dataset = CombinedStage2Dataset(datasets)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=_loader_generator(int(args.seed), torch=torch),
        worker_init_fn=_seed_loader_worker,
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
        resume_summary = load_stage2_checkpoint(
            Path(args.resume_checkpoint),
            model=model,
            optimizer=optimizer,
            device=device,
            args=args,
        )
        global_step = int(resume_summary["step"])
    init_summary = None
    if args.init_checkpoint is not None:
        init_summary = load_required_init_checkpoint(
            Path(args.init_checkpoint),
            model=model,
            device=device,
            frames_in=int(args.frames_in),
        )
    steps = int(args.steps) if args.steps is not None else _steps_for_epochs(len(dataset), int(args.batch_size), int(args.epochs))
    if steps <= 0:
        raise ValueError("--steps must be positive")
    occlusion_generator = _loader_generator(int(args.seed) + 10_000, torch=torch)
    losses: list[float] = []
    latest_checkpoint: Path | None = None
    model.train()
    batches = _no_cache_cycle(loader)
    for _ in range(steps):
        batch = next(batches)
        loss = train_one_stage2_batch(
            model,
            batch,
            optimizer=optimizer,
            device=device,
            torch=torch,
            occluded_prob=float(args.occluded_prob),
            occlusion_generator=occlusion_generator,
        )
        global_step += 1
        losses.append(loss)
        if int(args.checkpoint_every) > 0 and global_step % int(args.checkpoint_every) == 0:
            latest_checkpoint = save_stage2_checkpoint(
                checkpoint_dir / f"checkpoint_step_{global_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=global_step,
                args=args,
                train_dataset_summary=dataset.summary,
            )
    latest_checkpoint = save_stage2_checkpoint(
        checkpoint_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        step=global_step,
        args=args,
        train_dataset_summary=dataset.summary,
    )
    checkpoint_round_trip = checkpoint_round_trip_summary(
        latest_checkpoint,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    loss_summary = _loss_summary(losses)
    summary = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "train_complete" if loss_summary["last"] is not None else "train_failed",
        "ball_verified": False,
        "promotion_claimed": False,
        "heldout_touched": False,
        "out_dir": str(out_dir),
        "model": {
            "family": args.model_family,
            "frames_in": int(args.frames_in),
            "output_channels": int(args.output_channels),
            "image_size": list(image_size),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "init_summary": init_summary,
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
            "resume_summary": resume_summary,
        },
        "recipe": {
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "lr_schedule": "constant",
            "heatmap_radius_px": float(args.heatmap_radius_px),
            "occluded_prob": float(args.occluded_prob),
            "occlusion_policy": "model-space patch after official WASB warp, centered on warped target_xy_px before weighted BCE",
            "epochs": int(args.epochs),
            "steps": steps,
            "checkpoint_every": int(args.checkpoint_every),
        },
        "data": dataset.summary,
        "loss": loss_summary,
        "checkpoint": {
            "latest_checkpoint": str(latest_checkpoint),
            **checkpoint_round_trip,
            "state_sha256": state_dict_sha256(model.state_dict()),
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - start,
            "device": str(device),
            "seed": int(args.seed),
            "seed_summary": seed_summary,
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "limitations": [
            "Owner labels and SST samples are internal-val/build inputs only; no held-out gate is touched.",
            "Sparse CVAT exports train only reviewed frames; unreviewed frames are not fabricated negatives.",
            "Harness metrics are not BALL product gates.",
        ],
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def sparse_tracknet_labels_from_cvat(path: str | Path) -> list[TrackNetCvatLabel]:
    parsed = validate_artifact_file("cvat_video_annotations", path)
    if not isinstance(parsed, CvatVideoAnnotations):
        raise ValueError(f"reviewed boxes artifact did not parse as CvatVideoAnnotations: {path}")
    return sparse_tracknet_labels_from_annotations(parsed)


def sparse_tracknet_labels_from_annotations(annotations: CvatVideoAnnotations) -> list[TrackNetCvatLabel]:
    reviewed_indices = _reviewed_indices(annotations)
    frames_by_index = {frame.frame_index: frame for frame in annotations.frames}
    labels: list[TrackNetCvatLabel] = []
    for frame_index in reviewed_indices:
        frame = frames_by_index.get(frame_index)
        if frame is None:
            raise ValueError(f"{annotations.clip_id} reviewed frame {frame_index} is missing from annotations.frames")
        ball_boxes = [box for box in frame.boxes if box.label == "ball"]
        if len(ball_boxes) > 1:
            raise ValueError(f"multiple ball boxes in {annotations.clip_id} frame {frame_index}")
        frame_visibility_level = _frame_ball_visibility_level(frame)
        if not ball_boxes:
            if frame_visibility_level in {"clear", "partial"}:
                raise ValueError(f"{annotations.clip_id} frame {frame_index} has {frame_visibility_level} without a ball box")
            labels.append(
                TrackNetCvatLabel(
                    frame=frame_index,
                    visibility=0,
                    x=0.0,
                    y=0.0,
                    source="reviewed_cvat_ball_visibility_level" if frame_visibility_level is not None else "reviewed_absent_ball",
                    visibility_level=frame_visibility_level,
                    wbce_weight=_visibility_weight_or_clear(frame_visibility_level),
                    legacy_visibility_state=None if frame_visibility_level is not None else "legacy_hidden",
                )
            )
            continue
        box = ball_boxes[0]
        visibility_level = _merge_box_and_frame_visibility_level(
            box.visibility_level,
            frame_visibility_level,
            clip_id=annotations.clip_id,
            frame_index=frame_index,
        )
        if visibility_level in {"full", "out_of_frame"}:
            labels.append(
                TrackNetCvatLabel(
                    frame=frame_index,
                    visibility=0,
                    x=0.0,
                    y=0.0,
                    source="reviewed_cvat_ball_visibility_level",
                    center_convention=box.center_convention,
                    blur_angle_deg=box.blur_angle_deg,
                    blur_length_px=box.blur_length_px,
                    blur_width_px=box.blur_width_px,
                    blur_label_quality=box.blur_label_quality,
                    visibility_level=visibility_level,
                    wbce_weight=_visibility_weight_or_clear(visibility_level),
                )
            )
            continue
        x, y, width, height = box.bbox_xywh
        labels.append(
            TrackNetCvatLabel(
                frame=frame_index,
                visibility=1,
                x=float(x) + float(width) * 0.5,
                y=float(y) + float(height) * 0.5,
                source="reviewed_cvat_ball_box",
                center_convention=box.center_convention,
                blur_angle_deg=box.blur_angle_deg,
                blur_length_px=box.blur_length_px,
                blur_width_px=box.blur_width_px,
                blur_label_quality=box.blur_label_quality,
                visibility_level=visibility_level,
                wbce_weight=_visibility_weight_or_clear(visibility_level),
                legacy_visibility_state=None if visibility_level is not None else "legacy_visible",
            )
        )
    return labels


def load_cvat_annotations_from_export_clip(clip_dir: str | Path) -> CvatVideoAnnotations:
    path = Path(clip_dir)
    reviewed_json = path / "reviewed_boxes.json"
    if reviewed_json.is_file():
        parsed = validate_artifact_file("cvat_video_annotations", reviewed_json)
        if not isinstance(parsed, CvatVideoAnnotations):
            raise ValueError(f"reviewed boxes artifact did not parse as CvatVideoAnnotations: {reviewed_json}")
        return parsed
    xml_path = path / "annotations.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"CVAT clip dir needs reviewed_boxes.json or annotations.xml: {path}")
    with tempfile.TemporaryDirectory(prefix="cvat_video_xml_") as tmp_dir:
        zip_path = Path(tmp_dir) / "annotations.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(xml_path, "annotations.xml")
        annotations, _ = import_cvat_video_zip(zip_path, clip_id=path.name)
    return annotations


def train_one_stage2_batch(
    model: Any,
    batch: Mapping[str, Any],
    *,
    optimizer: Any,
    device: Any,
    torch: Any,
    occluded_prob: float,
    occlusion_generator: Any | None,
) -> float:
    augmented = apply_occlusion_augmentation(
        batch,
        occluded_prob=occluded_prob,
        generator=occlusion_generator,
        torch=torch,
    )
    return train_one_batch(
        model,
        augmented,
        optimizer=optimizer,
        device=device,
        torch=torch,
    )


def apply_occlusion_augmentation(
    batch: Mapping[str, Any],
    *,
    occluded_prob: float,
    generator: Any | None,
    torch: Any,
) -> dict[str, Any]:
    if occluded_prob <= 0.0:
        return dict(batch)
    if "wbce_weight" not in batch:
        raise ValueError("occlusion augmentation is allowed only with visibility-weighted WBCE batch['wbce_weight']")
    inputs = batch["input"].clone()
    target_xy = batch["target_xy_px"]
    ball_present = batch["ball_present"]
    if inputs.ndim != 4:
        raise ValueError(f"batch input must be BCHW, got {tuple(inputs.shape)}")
    batch_size, _, height, width = inputs.shape
    selected = torch.rand(batch_size, generator=generator) < float(occluded_prob)
    patch = max(4, int(round(min(height, width) * 0.15)))
    half = max(1, patch // 2)
    for index in range(batch_size):
        if not bool(selected[index]) or float(ball_present[index]) <= 0.0:
            continue
        x = int(round(float(target_xy[index][0])))
        y = int(round(float(target_xy[index][1])))
        x0 = max(0, x - half)
        x1 = min(width, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(height, y + half + 1)
        inputs[index, :, y0:y1, x0:x1] = 0.0
    out = dict(batch)
    out["input"] = inputs
    return out


def load_required_init_checkpoint(path: Path, *, model: Any, device: Any, frames_in: int) -> dict[str, Any]:
    payload = _torch().load(path, map_location=device, weights_only=False)
    if isinstance(payload, Mapping) and payload.get("frames_in") is not None and int(payload["frames_in"]) != int(frames_in):
        raise RuntimeError(f"init checkpoint frames_in mismatch: checkpoint={payload['frames_in']} requested={frames_in}")
    summary = load_model_weights(path, model=model, device=device, strict=False)
    missing = list(summary.get("missing_keys", []))
    unexpected = list(summary.get("unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"init checkpoint key mismatch: missing_keys={missing} unexpected_keys={unexpected}")
    return summary


def save_stage2_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    step: int,
    args: argparse.Namespace,
    train_dataset_summary: Mapping[str, Any],
) -> Path:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_stage2_checkpoint",
        "step": int(step),
        "model_family": args.model_family,
        "frames_in": int(args.frames_in),
        "output_channels": int(args.output_channels),
        "image_size": list(_parse_image_size(args.image_size)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_dataset_summary": dict(train_dataset_summary),
    }
    atomic_torch_save(payload, path, torch=_torch())
    return path


def load_stage2_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    device: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _torch().load(path, map_location=device, weights_only=False)
    if int(payload.get("frames_in", -1)) != int(args.frames_in):
        raise RuntimeError(f"resume checkpoint frames_in mismatch: checkpoint={payload.get('frames_in')} requested={args.frames_in}")
    if int(payload.get("output_channels", -1)) != int(args.output_channels):
        raise RuntimeError(
            f"resume checkpoint output_channels mismatch: checkpoint={payload.get('output_channels')} requested={args.output_channels}"
        )
    if str(payload.get("model_family")) != str(args.model_family):
        raise RuntimeError(f"resume checkpoint model_family mismatch: checkpoint={payload.get('model_family')} requested={args.model_family}")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {"checkpoint": str(path), "step": int(payload["step"])}


def _record_from_cvat_label(
    annotations: CvatVideoAnnotations,
    label: TrackNetCvatLabel,
    *,
    video_path: Path,
    source_path: str,
) -> Stage2SampleRecord:
    return Stage2SampleRecord(
        sample_id=f"cvat:{annotations.clip_id}:{label.frame}",
        source_kind="cvat_owner_sparse",
        clip_id=annotations.clip_id,
        video_path=video_path,
        frame_index=int(label.frame),
        source_width=int(annotations.task.original_size[0]),
        source_height=int(annotations.task.original_size[1]),
        ball_present=bool(label.visibility == 1),
        source_xy_px=(float(label.x), float(label.y)),
        visibility_level=label.visibility_level,
        wbce_weight=float(label.wbce_weight if label.wbce_weight is not None else 1),
        source_path=source_path,
    )


def _record_to_item(
    record: Stage2SampleRecord,
    *,
    image_size: tuple[int, int],
    frames_in: int,
    heatmap_radius_px: float,
    image_path_rewrites: Mapping[str, str],
) -> dict[str, Any]:
    torch = _torch()
    np = _numpy()
    cv2 = _cv2()
    target_w, target_h = image_size
    video_path = _rewrite_path(record.video_path, image_path_rewrites)
    offsets = _window_offsets(frames_in)
    frames_rgb = [
        _read_video_frame_rgb(video_path, max(0, record.frame_index + offset))
        for offset in offsets
    ]
    trans_input = _wasb_official_input_affine(
        record.source_width,
        record.source_height,
        cv2=cv2,
        np=np,
        output_wh=image_size,
    )
    input_tensor = _preprocess_wasb_window_official(
        frames_rgb,
        trans_input,
        cv2=cv2,
        np=np,
        torch=torch,
        output_wh=image_size,
    )
    if record.ball_present:
        warped_xy = _wasb_affine_transform_xy(record.source_xy_px, trans_input, np=np)
        scaled_x = float(warped_xy[0])
        scaled_y = float(warped_xy[1])
        target = _gaussian_heatmap(scaled_x, scaled_y, width=target_w, height=target_h, radius=heatmap_radius_px, torch=torch)
        target_xy = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)
    else:
        target = torch.zeros((1, target_h, target_w), dtype=torch.float32)
        target_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)
    return {
        "sample_id": record.sample_id,
        "source_slug": record.clip_id,
        "bucket": record.source_kind,
        "source_split": "train",
        "image_path": str(video_path),
        "window_sample_ids": [f"{record.clip_id}:{max(0, record.frame_index + offset)}" for offset in offsets],
        "temporal_sample_kind": "video_window",
        "input": input_tensor,
        "target": target,
        "target_xy_px": target_xy,
        "source_xy_px": torch.tensor(record.source_xy_px, dtype=torch.float32),
        "ball_present": torch.tensor(1.0 if record.ball_present else 0.0, dtype=torch.float32),
        "wbce_weight": torch.tensor(float(record.wbce_weight), dtype=torch.float32),
        "visibility_level": record.visibility_level,
    }


def _read_video_frame_rgb(path: Path, frame_index: int) -> Any:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                raise ValueError(f"could not read frame {frame_index} from {path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise ValueError(f"could not read frame {frame_index} from {path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb
    finally:
        capture.release()


def _video_size(path: Path) -> tuple[int, int]:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"could not read video size: {path}")
    return width, height


def _gaussian_heatmap(x: float, y: float, *, width: int, height: int, radius: float, torch: Any) -> Any:
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    heatmap = torch.exp(-((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / (2.0 * float(radius) ** 2))
    return heatmap.unsqueeze(0).clamp(0.0, 1.0)


def _reviewed_indices(annotations: CvatVideoAnnotations) -> list[int]:
    if annotations.reviewed_frame_indices is not None:
        return list(annotations.reviewed_frame_indices)
    if annotations.task.frame_filter:
        raise ValueError(
            f"{annotations.clip_id} has frame_filter={annotations.task.frame_filter!r} but no reviewed_frame_indices; "
            "cannot distinguish reviewed-absent frames from never-reviewed frames"
        )
    return [frame.frame_index for frame in sorted(annotations.frames, key=lambda frame: frame.frame_index)]


def _frame_ball_visibility_level(frame: CvatVideoFrame | None) -> BallVisibilityLevel | None:
    if frame is None:
        return None
    return frame.visibility_levels_by_label.get("ball")


def _merge_box_and_frame_visibility_level(
    box_level: BallVisibilityLevel | None,
    frame_level: BallVisibilityLevel | None,
    *,
    clip_id: str,
    frame_index: int,
) -> BallVisibilityLevel | None:
    if box_level is not None and frame_level is not None and box_level != frame_level:
        raise ValueError(f"{clip_id} frame {frame_index} has conflicting ball visibility levels: {box_level} vs {frame_level}")
    return box_level or frame_level


def _visibility_weight_or_clear(level: BallVisibilityLevel | None) -> int:
    if level is None:
        return BALL_VISIBILITY_WBCE_WEIGHTS["clear"]
    return BALL_VISIBILITY_WBCE_WEIGHTS[level]


def _resolve_rally_video(rally_root: Path, clip_id: str) -> Path:
    if "_rally_" not in clip_id:
        raise ValueError(f"cannot infer rally source id from clip id: {clip_id}")
    source_id = clip_id.split("_rally_", 1)[0]
    path = rally_root / source_id / f"{clip_id}.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"missing rally video for {clip_id}: {path}")
    return path


def _rewrite_path(path: Path, rewrites: Mapping[str, str]) -> Path:
    text = str(path)
    for old_prefix, new_prefix in rewrites.items():
        if text == old_prefix or text.startswith(f"{old_prefix}/"):
            return Path(f"{new_prefix}{text[len(old_prefix):]}")
    return path


def _window_offsets(frames_in: int) -> list[int]:
    half = frames_in // 2
    return list(range(-half, half + 1))


def _dataset_summary(
    source_kind: str,
    records: Sequence[Stage2SampleRecord],
    image_size: tuple[int, int],
    frames_in: int,
    heatmap_radius_px: float,
) -> dict[str, Any]:
    weights: dict[str, int] = {}
    visibility: dict[str, int] = {}
    for record in records:
        weights[str(int(record.wbce_weight) if float(record.wbce_weight).is_integer() else record.wbce_weight)] = (
            weights.get(str(int(record.wbce_weight) if float(record.wbce_weight).is_integer() else record.wbce_weight), 0) + 1
        )
        key = record.visibility_level or "none"
        visibility[key] = visibility.get(key, 0) + 1
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_stage2_dataset_summary",
        "source_kind": source_kind,
        "selected_sample_count": len(records),
        "clip_count": len({record.clip_id for record in records}),
        "positive_sample_count": sum(1 for record in records if record.ball_present),
        "negative_sample_count": sum(1 for record in records if not record.ball_present),
        "visibility_level_counts": dict(sorted(visibility.items())),
        "wbce_weight_counts": dict(sorted(weights.items())),
        "image_size": list(image_size),
        "input_preprocessing": "wasb_official_affine_imagenet",
        "frames_in": int(frames_in),
        "heatmap_radius_px": float(heatmap_radius_px),
        "sparse_review_policy": "only reviewed_frame_indices are training rows; unreviewed frames are never fabricated negatives",
    }


def _validate_dataset_shape(*, image_size: tuple[int, int], frames_in: int, heatmap_radius_px: float) -> None:
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size must contain positive width,height")
    if frames_in <= 0 or frames_in % 2 == 0:
        raise ValueError("frames_in must be a positive odd integer")
    if heatmap_radius_px <= 0.0:
        raise ValueError("heatmap_radius_px must be positive")


def _steps_for_epochs(sample_count: int, batch_size: int, epochs: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return max(1, int(math.ceil(sample_count / float(batch_size))) * int(epochs))


def _loss_summary(losses: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(losses),
        "first": float(losses[0]) if losses else None,
        "last": float(losses[-1]) if losses else None,
        "strictly_decreased": bool(losses and losses[-1] < losses[0]),
        "values": [float(value) for value in losses],
    }


def _no_cache_cycle(loader: Any) -> Any:
    while True:
        for batch in loader:
            yield batch


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cli_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("artifact_type") == "racketsport_ball_sst_manifest":
        return {
            "status": "sst_manifest_written",
            "summary": summary.get("summary"),
            "weight_policy": summary.get("weight_policy"),
            "protected_eval_hash_check": summary.get("protected_eval_hash_check"),
        }
    return {
        "status": summary.get("status"),
        "summary_json": str(Path(str(summary["out_dir"])) / "summary.json") if summary.get("out_dir") else None,
        "checkpoint": summary.get("checkpoint"),
        "loss": summary.get("loss"),
        "runtime": summary.get("runtime"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train BALL2D stage-2 on sparse owner CVAT labels and/or SST pseudo-label manifests.",
    )
    parser.add_argument("--mode", choices=("train", "build-sst-manifest"), default="train")
    parser.add_argument("--cvat-export-root", type=Path, default=None)
    parser.add_argument("--sst-manifest", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, default="wasb_hrnet")
    parser.add_argument("--wasb-repo", type=Path, default=Path("third_party/WASB-SBDT"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--image-size", default=f"{DEFAULT_BALL_PRETRAIN_IMAGE_SIZE[0]}x{DEFAULT_BALL_PRETRAIN_IMAGE_SIZE[1]}")
    parser.add_argument("--frames-in", type=int, default=DEFAULT_BALL_PRETRAIN_FRAMES_IN)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument("--heatmap-radius-px", type=float, default=DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--occluded-prob", type=float, default=0.25)
    parser.add_argument("--max-cvat-samples", type=int, default=None)
    parser.add_argument("--max-sst-samples", type=int, default=None)
    parser.add_argument("--rally-root", type=Path, default=DEFAULT_RALLY_ROOT)
    parser.add_argument(
        "--image-root-rewrite",
        action="append",
        default=[],
        help="Rewrite absolute video paths as OLD_PREFIX=NEW_PREFIX for VM/checkouts at a different root.",
    )
    parser.add_argument("--prelabel-root", type=Path, default=DEFAULT_PRELABEL_ROOT)
    parser.add_argument("--sst-manifest-out", type=Path, default=None)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--max-sst-samples-per-clip", type=int, default=None)
    parser.add_argument("--eval-root", type=Path, default=Path("eval_clips/ball"))
    parser.add_argument("--eval-sample-every-s", type=float, default=DEFAULT_EVAL_SAMPLE_EVERY_S)
    parser.add_argument("--expected-protected-eval-hash-count", type=int, default=DEFAULT_PROTECTED_EVAL_HASH_COUNT)
    parser.add_argument("--collision-hamming-threshold", type=int, default=DEFAULT_DEDUP_THRESHOLD)
    parser.add_argument("--protected-eval-hash", action="append", default=[])
    return parser


def _torch() -> Any:
    import torch

    return torch


def _numpy() -> Any:
    import numpy as np

    return np


def _cv2() -> Any:
    import cv2  # type: ignore[import-not-found]

    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
