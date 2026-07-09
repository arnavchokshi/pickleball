#!/usr/bin/env python3
"""CAL-MODEL v2 trainer for the `court_unet_v2` architecture (2026-07-05).

The 2026-07-02 neural kill (`runs/cal_external_retrain_20260702T003120Z/REPORT.md`) was
architectural: `encoder_decoder_v1` ran at 160x90, so one heatmap pixel covered 8+ source pixels
and the PCK@5px gate was unreachable by construction. This trainer targets the real replacement
architecture (`threed.racketsport.court_keypoint_net.make_court_unet_v2_model`: a torchvision
resnet34 encoder + U-Net decoder at 640x360 input, stride-4 heatmap/segmentation heads,
10-35M params) with a from-scratch trainer rather than reusing
`scripts/racketsport/train_court_keypoint_heatmap.py`'s single-tensor-output assumptions (that
script and its checkpoint format are left completely untouched -- the CAL evidence scanner and
any consumer of `encoder_decoder_v1`/`local_conv_v1` checkpoints keeps working exactly as before).

Losses per training step:
  - keypoint heatmaps: existing `court_keypoint_net.court_keypoint_heatmap_loss` (per-channel
    spatial-softmax cross-entropy), reused unmodified.
  - line-family+surface segmentation: class-balanced cross-entropy over the 5-class combined head
    (`court_keypoint_net.COURT_UNET_V2_SEG_CLASS_NAMES`).
  - keypoint visibility: binary cross-entropy (1.0 iff cleanly visible, 0.0 for occluded/off-frame).
  - geometric-consistency regularizer: existing
    `court_keypoint_geometric_loss.court_geometric_consistency_loss`, reused unmodified. Enabled
    by default (`--geometric-loss-weight 0.05`) -- unlike the legacy trainer, which defaults this
    to 0.0 for backward compatibility with pre-CAL-R2 checkpoints, this is a new architecture with
    no such constraint. Pass `--geometric-loss-weight 0.0` to disable.

Data:
  - CAL-SYNTH streaming contract `threed.racketsport.court_synth_stream.iter_synthetic_court_samples`
    wrapped in `SyntheticCourtIterableDataset` (per-worker seeding via
    `torch.utils.data.get_worker_info()`). If the module or that attribute is missing at import
    time, only the integration test that requires it skips cleanly
    (`tests/racketsport/test_train_court_model_v2.py`); this script itself always has a working
    fallback via `--synthetic-fallback`, a tiny self-contained procedural sampler
    (`_fallback_synthetic_sample`) emitting the exact same per-sample dict shape, so CPU smoke
    runs and unit tests never depend on CAL-SYNTH landing first.
  - Optional on-disk corpora: existing `<clip>/labels/court_keypoints.json` tier format via
    `scripts.racketsport.train_court_keypoint_heatmap.load_real_court_keypoint_labels` (imported,
    not modified), eval-clip guarded via `threed.racketsport.eval_guard`. Real rows only supervise
    the keypoint-heatmap and visibility losses (they carry no line-family/surface ground truth);
    mixed in via `--real-weight` / `--synthetic-weight` sampling odds.

CPU smoke (see `tests/racketsport/test_train_court_model_v2.py`): 2 epochs on 64
synthetic-fallback samples must strictly decrease loss and beat random-init PCK@40px on a 16-sample
held-out synthetic set generated once from a separate, fixed validation seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_geometric_loss import court_geometric_consistency_loss
from threed.racketsport.court_keypoint_net import (
    COURT_UNET_V2_ARCHITECTURE,
    COURT_UNET_V2_HEATMAP_STRIDE,
    COURT_UNET_V2_SEG_CLASS_NAMES,
    PICKLEBALL_KEYPOINTS,
    court_keypoint_heatmap_loss,
    make_court_keypoint_heatmap_model,
    merge_line_family_and_surface_targets,
)
from threed.racketsport.eval_guard import assert_not_training_on_eval_clip

KEYPOINT_NAMES: list[str] = [point.name for point in PICKLEBALL_KEYPOINTS]
KEYPOINT_COUNT = len(KEYPOINT_NAMES)
SEG_CLASS_COUNT = len(COURT_UNET_V2_SEG_CLASS_NAMES)

# CAL-SYNTH's stable contract (threed/racketsport/court_synth_stream.py). Imported defensively:
# CAL-SYNTH lands in this same worktree in parallel, so the module (or this specific attribute)
# may legitimately be missing when this file is first exercised. Every caller below must go
# through `_iter_synthetic_samples`, never call this directly.
try:
    from threed.racketsport.court_synth_stream import iter_synthetic_court_samples as _iter_synthetic_court_samples
except Exception:  # pragma: no cover - exercised by test_train_court_model_v2's missing-module test
    _iter_synthetic_court_samples = None

try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except Exception:  # pragma: no cover - torch absence is handled by CLI/test importers
    _TorchIterableDataset = object


# ---------------------------------------------------------------------------------------------
# Self-contained synthetic-fallback sampler (never depends on CAL-SYNTH landing)
# ---------------------------------------------------------------------------------------------


def _fallback_world_bounds() -> tuple[float, float, float, float]:
    xs = [point.world_xyz_m[0] for point in PICKLEBALL_KEYPOINTS]
    ys = [point.world_xyz_m[1] for point in PICKLEBALL_KEYPOINTS]
    return min(xs), max(xs), min(ys), max(ys)


_FALLBACK_WORLD_BOUNDS = _fallback_world_bounds()

# (start_keypoint, end_keypoint, line_family_class) for the fallback renderer's drawn court lines.
# line_family_class values match threed.racketsport.court_synth_stream.LINE_FAMILY_CLASSES
# (0=other/bg, 1=pickleball_line, 2=tennis_line, 3=net) -- this fallback never draws a tennis
# overlay, so class 2 never appears here.
_FALLBACK_COURT_LINES: tuple[tuple[str, str, int], ...] = (
    ("near_left_corner", "near_right_corner", 1),
    ("near_right_corner", "far_right_corner", 1),
    ("far_right_corner", "far_left_corner", 1),
    ("far_left_corner", "near_left_corner", 1),
    ("near_nvz_left", "near_nvz_right", 1),
    ("far_nvz_left", "far_nvz_right", 1),
    ("near_baseline_center", "near_nvz_center", 1),
    ("far_nvz_center", "far_baseline_center", 1),
    ("net_left_sideline", "net_right_sideline", 3),
)


def _fallback_bilinear(u: float, v: float, quad: list[tuple[float, float]]) -> tuple[float, float]:
    near_left, near_right, far_right, far_left = quad
    x = (1 - u) * (1 - v) * near_left[0] + u * (1 - v) * near_right[0] + u * v * far_right[0] + (1 - u) * v * far_left[0]
    y = (1 - u) * (1 - v) * near_left[1] + u * (1 - v) * near_right[1] + u * v * far_right[1] + (1 - u) * v * far_left[1]
    return x, y


def _fallback_random_quad(width: int, height: int, rng: random.Random) -> list[tuple[float, float]]:
    margin_x = width * rng.uniform(0.05, 0.18)
    near_y = height * rng.uniform(0.72, 0.94)
    far_y = height * rng.uniform(0.10, 0.42)
    near_left = (margin_x + rng.uniform(-4, 4), near_y + rng.uniform(-4, 4))
    near_right = (width - margin_x + rng.uniform(-4, 4), near_y + rng.uniform(-4, 4))
    far_width = width * rng.uniform(0.28, 0.66)
    far_center = width * rng.uniform(0.42, 0.58)
    far_left = (far_center - far_width / 2 + rng.uniform(-4, 4), far_y + rng.uniform(-4, 4))
    far_right = (far_center + far_width / 2 + rng.uniform(-4, 4), far_y + rng.uniform(-4, 4))
    return [near_left, near_right, far_right, far_left]


def _fallback_synthetic_sample(rng: random.Random, image_size: tuple[int, int]) -> dict[str, Any]:
    """Tiny, fully self-contained procedural court render (PIL only): a random quadrilateral
    court with straight pickleball lines + net, no tennis overlay / occlusion / distortion. Not a
    replacement for CAL-SYNTH's richer generator -- this exists purely so architecture/loss/trainer
    plumbing (including the CPU smoke acceptance proof) never depends on that module landing
    first, per the CAL-MODEL spec's `--synthetic-fallback` requirement."""

    import numpy as np
    from PIL import Image, ImageDraw

    width, height = image_size
    min_x, max_x, min_y, max_y = _FALLBACK_WORLD_BOUNDS
    quad = _fallback_random_quad(width, height, rng)

    keypoint_uv = {
        point.name: (
            (point.world_xyz_m[0] - min_x) / (max_x - min_x),
            (point.world_xyz_m[1] - min_y) / (max_y - min_y),
        )
        for point in PICKLEBALL_KEYPOINTS
    }
    keypoint_xy = {name: _fallback_bilinear(u, v, quad) for name, (u, v) in keypoint_uv.items()}

    background = tuple(rng.randint(30, 90) for _ in range(3))
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    line_family_image = Image.new("L", (width, height), 0)
    line_family_draw = ImageDraw.Draw(line_family_image)
    surface_image = Image.new("L", (width, height), 0)
    surface_draw = ImageDraw.Draw(surface_image)

    surface_draw.polygon([keypoint_xy[name] for name in ("near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner")], fill=2)

    line_color = tuple(rng.randint(180, 255) for _ in range(3))
    line_width_px = rng.randint(2, 4)
    for start_name, end_name, family_class in _FALLBACK_COURT_LINES:
        segment = [keypoint_xy[start_name], keypoint_xy[end_name]]
        draw.line(segment, fill=line_color, width=line_width_px)
        line_family_draw.line(segment, fill=family_class, width=max(2, line_width_px))

    keypoints_vis = []
    for point in PICKLEBALL_KEYPOINTS:
        # Fallback sampler always keeps the drawn quad fully in-frame, so no keypoint is ever
        # genuinely off-frame; occasionally mark one "occluded" so the visibility BCE loss sees a
        # mix of classes rather than a trivially constant target.
        keypoints_vis.append(1 if rng.random() < 0.12 else 2)

    image_rgb = np.asarray(image, dtype=np.uint8)
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    return {
        "image_bgr": image_bgr,
        "keypoints_xy": np.asarray([keypoint_xy[name] for name in KEYPOINT_NAMES], dtype=np.float32),
        "keypoints_vis": np.asarray(keypoints_vis, dtype=np.int64),
        "line_family_mask": np.asarray(line_family_image, dtype=np.uint8),
        "surface_mask": np.asarray(surface_image, dtype=np.uint8),
        "meta": {"source": "cal_model_synthetic_fallback"},
    }


def _iter_synthetic_samples(
    config: dict[str, Any] | None,
    seed: int,
    *,
    force_fallback: bool,
) -> Iterator[dict[str, Any]]:
    """Single entry point for synthetic samples: CAL-SYNTH's real contract when available (and
    not forced off), else `_fallback_synthetic_sample` forever. Every sample -- real or fallback
    -- is shaped identically (see `court_synth_stream`'s module docstring)."""

    image_size = tuple((config or {}).get("image_size") or (640, 360))
    if not force_fallback and _iter_synthetic_court_samples is not None:
        yield from _iter_synthetic_court_samples(config, seed=seed)
        return
    rng = random.Random(seed)
    while True:
        yield _fallback_synthetic_sample(rng, image_size)


# ---------------------------------------------------------------------------------------------
# Sample -> training tensors
# ---------------------------------------------------------------------------------------------


def sample_to_training_arrays(
    sample: dict[str, Any],
    *,
    model_width: int,
    model_height: int,
    heatmap_stride: int = COURT_UNET_V2_HEATMAP_STRIDE,
    sigma_px: float = 1.5,
    has_seg_target: bool = True,
) -> dict[str, Any]:
    """Convert one CAL-SYNTH/fallback-shaped sample dict into the fixed-size arrays the trainer
    batches together: image (3,H,W) float32 RGB in [0,1], per-channel keypoint heatmap targets +
    mask at the head's stride, per-keypoint visibility BCE target, and the merged 5-class
    line-family+surface segmentation target (or an all-ignored target when `has_seg_target` is
    False, e.g. for on-disk real-corpus rows that carry no segmentation ground truth)."""

    import cv2
    import numpy as np

    image_bgr = sample["image_bgr"]
    source_height, source_width = image_bgr.shape[:2]
    if (source_width, source_height) != (model_width, model_height):
        image_bgr = cv2.resize(image_bgr, (model_width, model_height), interpolation=cv2.INTER_AREA)
    scale_x = model_width / float(source_width)
    scale_y = model_height / float(source_height)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_chw = np.transpose(image_rgb, (2, 0, 1))

    keypoints_xy = np.asarray(sample["keypoints_xy"], dtype=np.float32)
    keypoints_vis = np.asarray(sample["keypoints_vis"], dtype=np.int64)
    keypoints_xy_model = keypoints_xy * np.asarray([scale_x, scale_y], dtype=np.float32)

    head_width = max(1, model_width // heatmap_stride)
    head_height = max(1, model_height // heatmap_stride)
    yy, xx = np.mgrid[0:head_height, 0:head_width]
    heatmaps = np.zeros((KEYPOINT_COUNT, head_height, head_width), dtype=np.float32)
    heatmap_mask = np.zeros((KEYPOINT_COUNT,), dtype=np.float32)
    for index in range(KEYPOINT_COUNT):
        if keypoints_vis[index] == 0:
            continue  # off-frame: no valid heatmap location to supervise
        cx = keypoints_xy_model[index, 0] / heatmap_stride
        cy = keypoints_xy_model[index, 1] / heatmap_stride
        heatmaps[index] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_px**2))
        heatmap_mask[index] = 1.0

    vis_target = (keypoints_vis == 2).astype(np.float32)

    if has_seg_target:
        line_family_mask = sample["line_family_mask"]
        surface_mask = sample["surface_mask"]
        if line_family_mask.shape != (source_height, source_width):
            raise ValueError("line_family_mask must match image_bgr's shape")
        if (source_width, source_height) != (model_width, model_height):
            line_family_mask = cv2.resize(
                line_family_mask, (model_width, model_height), interpolation=cv2.INTER_NEAREST
            )
            surface_mask = cv2.resize(surface_mask, (model_width, model_height), interpolation=cv2.INTER_NEAREST)
        merged = merge_line_family_and_surface_targets(line_family_mask, surface_mask)
        seg_target = cv2.resize(merged, (head_width, head_height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        seg_mask = np.ones((), dtype=np.float32)
    else:
        seg_target = np.zeros((head_height, head_width), dtype=np.int64)
        seg_mask = np.zeros((), dtype=np.float32)

    return {
        "image": image_chw,
        "heatmaps": heatmaps,
        "heatmap_mask": heatmap_mask,
        "vis_target": vis_target,
        "seg_target": seg_target,
        "seg_mask": seg_mask,
    }


def _stack_arrays(rows: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    import numpy as np

    def stacked(key: str, dtype: Any) -> Any:
        return torch.from_numpy(np.stack([row[key] for row in rows]).astype(dtype))

    return {
        "image": stacked("image", "float32"),
        "heatmaps": stacked("heatmaps", "float32"),
        "heatmap_mask": stacked("heatmap_mask", "float32"),
        "vis_target": stacked("vis_target", "float32"),
        "seg_target": stacked("seg_target", "int64"),
        "seg_mask": stacked("seg_mask", "float32"),
    }


def _stack_arrays_numpy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    def stacked(key: str, dtype: Any) -> Any:
        return np.stack([row[key] for row in rows]).astype(dtype)

    return {
        "image": stacked("image", "float32"),
        "heatmaps": stacked("heatmaps", "float32"),
        "heatmap_mask": stacked("heatmap_mask", "float32"),
        "vis_target": stacked("vis_target", "float32"),
        "seg_target": stacked("seg_target", "int64"),
        "seg_mask": stacked("seg_mask", "float32"),
    }


def _tensorize_training_batch(batch: dict[str, Any], torch: Any) -> dict[str, Any]:
    for key in ("image", "heatmaps", "heatmap_mask", "vis_target", "seg_target", "seg_mask"):
        value = batch[key]
        if not hasattr(value, "to"):
            batch[key] = torch.from_numpy(value)
    return batch


def _identity_collate(sample: Any) -> Any:
    return sample


class SyntheticCourtIterableDataset(_TorchIterableDataset):
    """`torch.utils.data.IterableDataset` wrapper around `_iter_synthetic_samples` with
    per-worker seeding: each DataLoader worker (`torch.utils.data.get_worker_info()`) offsets
    `base_seed` by its worker id, so multi-worker loading never repeats the same stream and stays
    deterministic for a fixed `(base_seed, num_workers)` pair. Falls back to a single-process
    stream (worker id 0) when used outside a DataLoader or with `num_workers=0`.

    In ordinary sample mode it yields one converted sample at a time. In `batch_size` mode, used
    by `run_training`, workers materialize full step batches. That preserves exact old trainer
    sample parity: global training step `g` always uses synthetic seed `base_seed + g + 1`,
    independent of how many DataLoader workers are active; workers only decide who prepares each
    stamped step batch, and the training loop re-orders those batches before optimization."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None,
        base_seed: int,
        model_width: int,
        model_height: int,
        sigma_px: float,
        force_fallback: bool,
        samples_per_epoch: int | None = None,
        batch_size: int | None = None,
        steps_per_epoch: int | None = None,
        global_step_offset: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.base_seed = base_seed
        self.model_width = model_width
        self.model_height = model_height
        self.sigma_px = sigma_px
        self.force_fallback = force_fallback
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.global_step_offset = global_step_offset

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        worker_count = 1 if worker_info is None else int(getattr(worker_info, "num_workers", 1))
        if self.batch_size is not None:
            if self.steps_per_epoch is None:
                raise ValueError("steps_per_epoch is required when SyntheticCourtIterableDataset batch_size is set")
            for local_step_index in range(worker_id, self.steps_per_epoch, worker_count):
                global_step_index = self.global_step_offset + local_step_index
                synthetic_seed = self.base_seed + global_step_index + 1
                batch = materialize_synthetic_batch_arrays(
                    config=self.config,
                    seed=synthetic_seed,
                    count=self.batch_size,
                    model_width=self.model_width,
                    model_height=self.model_height,
                    sigma_px=self.sigma_px,
                    force_fallback=self.force_fallback,
                )
                batch["loader_meta"] = {
                    "global_step_index": local_step_index,
                    "absolute_step_index": global_step_index,
                    "synthetic_seed": synthetic_seed,
                    "worker_id": worker_id,
                    "worker_count": worker_count,
                    "batch_size": self.batch_size,
                }
                yield batch
            return

        seed = self.base_seed + worker_id
        produced = 0
        for sample in _iter_synthetic_samples(self.config, seed, force_fallback=self.force_fallback):
            yield sample_to_training_arrays(
                sample,
                model_width=self.model_width,
                model_height=self.model_height,
                sigma_px=self.sigma_px,
                has_seg_target=True,
            )
            produced += 1
            if self.samples_per_epoch is not None and produced >= self.samples_per_epoch:
                return


def materialize_synthetic_batch(
    *,
    config: dict[str, Any] | None,
    seed: int,
    count: int,
    model_width: int,
    model_height: int,
    sigma_px: float,
    force_fallback: bool,
    torch: Any,
) -> dict[str, Any]:
    """Materialize exactly `count` synthetic samples from a fixed `(config, seed)` stream into one
    stacked tensor batch -- used both for ordinary training steps and for building the ONE-TIME,
    reused-every-epoch held-out validation set (a fixed seed, generated once, never regenerated,
    so it is a valid stable validation set rather than freshly-sampled noise each epoch)."""

    return _tensorize_training_batch(
        materialize_synthetic_batch_arrays(
            config=config,
            seed=seed,
            count=count,
            model_width=model_width,
            model_height=model_height,
            sigma_px=sigma_px,
            force_fallback=force_fallback,
        ),
        torch,
    )


def materialize_synthetic_batch_arrays(
    *,
    config: dict[str, Any] | None,
    seed: int,
    count: int,
    model_width: int,
    model_height: int,
    sigma_px: float,
    force_fallback: bool,
) -> dict[str, Any]:
    """Numpy-array variant of `materialize_synthetic_batch`.

    DataLoader worker processes return numpy arrays instead of torch tensors so this lane does not
    depend on PyTorch's shared-memory manager, which is disabled in some local sandboxes."""

    rows = []
    for sample in _iter_synthetic_samples(config, seed, force_fallback=force_fallback):
        rows.append(
            sample_to_training_arrays(
                sample,
                model_width=model_width,
                model_height=model_height,
                sigma_px=sigma_px,
                has_seg_target=True,
            )
        )
        if len(rows) >= count:
            break
    return _stack_arrays_numpy(rows)


def _resolve_synthetic_workers(requested_workers: int | None) -> int:
    if requested_workers is None:
        return max(0, min(8, os.cpu_count() or 1))
    if isinstance(requested_workers, bool) or requested_workers < 0:
        raise ValueError("--synthetic-workers must be a non-negative integer")
    return int(requested_workers)


def _make_synthetic_training_dataloader(
    *,
    config: dict[str, Any] | None,
    base_seed: int,
    batch_size: int,
    steps_per_epoch: int,
    global_step_offset: int,
    model_width: int,
    model_height: int,
    sigma_px: float,
    force_fallback: bool,
    synthetic_workers: int,
    torch: Any,
) -> Any:
    from torch.utils.data import DataLoader

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if steps_per_epoch <= 0:
        raise ValueError("--steps-per-epoch must be positive")
    workers = _resolve_synthetic_workers(synthetic_workers)
    dataset = SyntheticCourtIterableDataset(
        config=config,
        base_seed=base_seed,
        model_width=model_width,
        model_height=model_height,
        sigma_px=sigma_px,
        force_fallback=force_fallback,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        global_step_offset=global_step_offset,
    )
    kwargs: dict[str, Any] = {
        "batch_size": None,
        "collate_fn": _identity_collate,
        "num_workers": workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "persistent_workers": False,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["in_order"] = True
    return DataLoader(dataset, **kwargs)


def _iter_ordered_synthetic_batches(loader: Any, *, expected_steps: int) -> Iterator[dict[str, Any]]:
    """Yield stamped step batches in local epoch order even when worker processes finish out of
    order. This keeps optimization sample order equivalent to the old single-process loop while
    allowing the expensive materialization work to happen inside DataLoader workers."""

    import torch

    next_step = 0
    buffered: dict[int, dict[str, Any]] = {}
    for batch in loader:
        meta = batch.get("loader_meta") or {}
        step_index = int(meta.get("global_step_index", -1))
        if step_index < 0:
            raise ValueError("synthetic DataLoader batch missing loader_meta.global_step_index")
        buffered[step_index] = batch
        while next_step in buffered:
            yield _tensorize_training_batch(buffered.pop(next_step), torch)
            next_step += 1
            if next_step >= expected_steps:
                return
    if next_step != expected_steps:
        raise RuntimeError(f"synthetic DataLoader produced {next_step}/{expected_steps} expected step batches")


# ---------------------------------------------------------------------------------------------
# On-disk real corpora (existing tier format), eval-clip guarded
# ---------------------------------------------------------------------------------------------


def real_row_to_sample_arrays(
    row: dict[str, Any],
    *,
    model_width: int,
    model_height: int,
    sigma_px: float,
) -> dict[str, Any]:
    """Convert one `<clip>/labels/court_keypoints.json` row (the existing on-disk tier format,
    loaded via `scripts.racketsport.train_court_keypoint_heatmap.load_real_court_keypoint_labels`)
    into the same training-array shape `sample_to_training_arrays` produces for synthetic rows.
    Real rows carry no line-family/surface ground truth (`has_seg_target=False` -- the seg loss
    skips these rows entirely) and no explicit visibility label (every labeled keypoint is
    assumed cleanly visible: a human reviewer would not confidently place a point that was not
    visible enough to see).

    Coordinates: `row["keypoints"]` are stored in the row's own `source_video_size` pixel space,
    but `load_label_image` may return a cached preview image at a *different* (often lower,
    label-annotation-resolution) pixel size than that -- see `evaluate_court_model_v2.py`'s module
    docstring for the concrete 1280x720-vs-1920x1080 case this guards against. This function
    always rescales by the ratio between the row's declared `source_video_size` and the actually
    loaded image's own pixel size, so the keypoint targets stay correct regardless of which of the
    two resolutions the loaded image happens to be.
    """

    import cv2
    import numpy as np
    from PIL import Image

    # Imported lazily (not at module import time) to avoid a hard dependency for callers that
    # never use on-disk corpora (e.g. every CPU-smoke/synthetic-only code path).
    from scripts.racketsport.train_court_keypoint_heatmap import load_label_image

    image = load_label_image(row, cv2=cv2, image_module=Image)
    loaded_width, loaded_height = image.size
    source_size = row.get("source_video_size") or [loaded_width, loaded_height]
    source_width, source_height = float(source_size[0]), float(source_size[1])
    rescale_x = loaded_width / source_width
    rescale_y = loaded_height / source_height

    image_rgb = np.asarray(image, dtype=np.uint8)
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])

    keypoints = row["keypoints"]
    keypoints_xy = np.asarray(
        [[keypoints[name][0] * rescale_x, keypoints[name][1] * rescale_y] for name in KEYPOINT_NAMES],
        dtype=np.float32,
    )
    keypoints_vis = np.full((KEYPOINT_COUNT,), 2, dtype=np.int64)  # all labeled points: visible

    sample = {
        "image_bgr": image_bgr,
        "keypoints_xy": keypoints_xy,
        "keypoints_vis": keypoints_vis,
    }
    return sample_to_training_arrays(
        sample,
        model_width=model_width,
        model_height=model_height,
        sigma_px=sigma_px,
        has_seg_target=False,
    )


def load_real_training_rows(real_roots: list[Path] | None) -> list[dict[str, Any]]:
    if not real_roots:
        return []
    from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels

    rows: list[dict[str, Any]] = []
    for root in real_roots:
        rows.extend(load_real_court_keypoint_labels(Path(root)))
    assert_not_training_on_eval_clip((row.get("clip") for row in rows), allow_internal_val=False)
    return rows


# ---------------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------------


def class_balanced_seg_loss(logits: Any, target: Any, sample_mask: Any, *, torch: Any) -> Any:
    """Class-balanced cross-entropy over the 5-class combined line-family+surface head.
    `sample_mask` (batch,) zeros out rows with no segmentation ground truth (on-disk real rows).
    Class weights are the inverse square root of each class's pixel frequency in this batch
    (clamped, renormalized to a mean of 1) -- a simple, standard class-balancing scheme that keeps
    the (tiny, rare) line/net classes from being swamped by the dominant background class without
    needing a fixed, corpus-wide prior."""

    import torch.nn.functional as F

    if float(sample_mask.sum()) <= 0.0:
        return logits.new_zeros(())
    counts = torch.stack(
        [(target == class_index).sum().to(logits.dtype) for class_index in range(logits.shape[1])]
    )
    weights = 1.0 / torch.sqrt(counts.clamp_min(1.0))
    weights = weights * (logits.shape[1] / weights.sum().clamp_min(1e-6))
    per_pixel = F.cross_entropy(logits, target, weight=weights, reduction="none")
    per_row = per_pixel.mean(dim=(-2, -1))
    return (per_row * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)


def visibility_bce_loss(logits: Any, target: Any, *, torch: Any) -> Any:
    import torch.nn.functional as F

    return F.binary_cross_entropy_with_logits(logits, target)


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------


def decode_batch_keypoints_px(heatmap_probabilities: Any, *, heatmap_stride: int) -> Any:
    """(batch, K, h, w) spatial-softmax probabilities -> (batch, K, 2) subpixel keypoint (x, y) in
    model-input pixel space, via the existing `decode_subpixel_heatmap` parabolic refinement (the
    same decode used for the legacy encoder_decoder_v1/local_conv_v1 architectures and by
    `court_model_infer.infer_court_model` at inference time)."""

    from threed.racketsport.court_keypoint_net import decode_subpixel_heatmap

    batch, keypoint_count = heatmap_probabilities.shape[:2]
    decoded = [[None] * keypoint_count for _ in range(batch)]
    numpy_probabilities = heatmap_probabilities.detach().cpu().numpy()
    for row in range(batch):
        for channel in range(keypoint_count):
            point = decode_subpixel_heatmap(numpy_probabilities[row, channel].tolist())
            decoded[row][channel] = (point.x * heatmap_stride, point.y * heatmap_stride)
    return decoded


def evaluate_on_batch(model: Any, batch: dict[str, Any], *, device: Any, heatmap_stride: int, torch: Any) -> dict[str, Any]:
    from threed.racketsport.court_keypoint_net import court_keypoint_probabilities

    model.eval()
    with torch.no_grad():
        outputs = model(batch["image"].to(device))
        probabilities = court_keypoint_probabilities(outputs["keypoint_heatmaps"])
        decoded_px = decode_batch_keypoints_px(probabilities, heatmap_stride=heatmap_stride)
        seg_pred = outputs["line_family_logits"].argmax(dim=1).detach().cpu().numpy()

    heatmap_mask = batch["heatmap_mask"].numpy()
    errors: list[float] = []
    for row_index, row in enumerate(decoded_px):
        for channel, (px, py) in enumerate(row):
            if heatmap_mask[row_index, channel] <= 0:
                continue
            cy, cx = _target_peak_yx(batch["heatmaps"][row_index, channel].numpy())
            errors.append(float(math.hypot(px - cx * heatmap_stride, py - cy * heatmap_stride)))

    seg_target = batch["seg_target"].numpy()
    seg_mask = batch["seg_mask"].numpy()
    intersection = [0.0] * SEG_CLASS_COUNT
    union = [0.0] * SEG_CLASS_COUNT
    for row_index in range(seg_pred.shape[0]):
        if seg_mask[row_index] <= 0:
            continue
        for class_index in range(SEG_CLASS_COUNT):
            pred_mask = seg_pred[row_index] == class_index
            true_mask = seg_target[row_index] == class_index
            intersection[class_index] += float((pred_mask & true_mask).sum())
            union[class_index] += float((pred_mask | true_mask).sum())
    per_class_iou = [
        (intersection[i] / union[i]) if union[i] > 0 else None for i in range(SEG_CLASS_COUNT)
    ]
    valid_iou = [iou for iou in per_class_iou if iou is not None]

    ordered = sorted(errors)
    return {
        "keypoint_count": len(ordered),
        "median_px": _percentile(ordered, 50.0),
        "mean_px": (sum(ordered) / len(ordered)) if ordered else None,
        "p95_px": _percentile(ordered, 95.0),
        "pck_at_5px": _pck(ordered, 5.0),
        "pck_at_10px": _pck(ordered, 10.0),
        "pck_at_40px": _pck(ordered, 40.0),
        "seg_mean_iou": (sum(valid_iou) / len(valid_iou)) if valid_iou else None,
        "seg_per_class_iou": dict(zip(COURT_UNET_V2_SEG_CLASS_NAMES, per_class_iou, strict=True)),
    }


def _target_peak_yx(heatmap: Any) -> tuple[float, float]:
    import numpy as np

    flat_index = int(np.argmax(heatmap))
    height, width = heatmap.shape
    y, x = divmod(flat_index, width)
    return float(y), float(x)


def _percentile(ordered_values: list[float], percentile: float) -> float | None:
    if not ordered_values:
        return None
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    rank = (len(ordered_values) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered_values[low])
    weight = rank - low
    return float(ordered_values[low] * (1.0 - weight) + ordered_values[high] * weight)


def _pck(ordered_values: list[float], threshold_px: float) -> float | None:
    if not ordered_values:
        return None
    return sum(1 for value in ordered_values if value <= threshold_px) / len(ordered_values)


# ---------------------------------------------------------------------------------------------
# Encoder weight validation (local-only, no downloads)
# ---------------------------------------------------------------------------------------------


def _torchvision_resnet34_cached_weight_path() -> Path:
    from torchvision.models import ResNet34_Weights

    filename = Path(urlparse(ResNet34_Weights.IMAGENET1K_V1.url).path).name
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")).expanduser()
    candidate = torch_home / "hub" / "checkpoints" / filename
    if not candidate.is_file():
        raise FileNotFoundError(
            "encoder weights were requested as 'imagenet', but torchvision's resnet34 "
            f"checkpoint is not locally cached at {candidate}. This lane does not download "
            "weights. GPU lane download step: run torchvision's resnet34 ImageNet weight "
            "materialization once in the target environment, then rerun this trainer with "
            "--encoder-weights-path imagenet or the explicit cached .pth path."
        )
    return candidate


def _extract_state_dict(payload: Any, *, path: Path) -> dict[str, Any]:
    if isinstance(payload, dict) and "state_dict" in payload and "conv1.weight" not in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"encoder weights were requested but {path} is not a PyTorch state dict")
    return payload


def _validate_resnet34_encoder_checkpoint(path: Path) -> None:
    import torch
    from torchvision.models import resnet34

    state = _extract_state_dict(torch.load(str(path), map_location="cpu", weights_only=True), path=path)
    expected = resnet34(weights=None).state_dict()
    expected_backbone = {key: value for key, value in expected.items() if not key.startswith("fc.")}
    provided_non_fc = {key: value for key, value in state.items() if not key.startswith("fc.")}
    missing = sorted(key for key in expected_backbone if key not in provided_non_fc)
    unexpected = sorted(key for key in provided_non_fc if key not in expected_backbone)
    shape_mismatch = sorted(
        key
        for key, expected_tensor in expected_backbone.items()
        if key in provided_non_fc and tuple(provided_non_fc[key].shape) != tuple(expected_tensor.shape)
    )
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "encoder weights were requested but "
            f"{path} does not match expected torchvision resnet34 backbone "
            f"(missing={missing}, unexpected={unexpected}, shape_mismatch={shape_mismatch})"
        )


def _resolve_encoder_weights_path(requested: str | Path | None) -> Path | None:
    if requested is None:
        return None
    text = str(requested)
    if text.strip().lower() == "imagenet":
        path = _torchvision_resnet34_cached_weight_path()
    else:
        path = Path(text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            "encoder weights were requested but "
            f"{path} does not exist. Omit --encoder-weights-path for explicit random init, pass "
            "a local torchvision resnet34 .pth checkpoint, or pass 'imagenet' only after the "
            "checkpoint is already cached locally."
        )
    _validate_resnet34_encoder_checkpoint(path)
    return path


def _checkpoint_path(out_dir: Path, epoch: int) -> Path:
    return out_dir / f"court_model_v2_epoch_{epoch:04d}.pt"


def _save_training_checkpoint(
    *,
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    args: argparse.Namespace,
    torch: Any,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "image_size": [args.image_width, args.image_height],
            "model_architecture": COURT_UNET_V2_ARCHITECTURE,
            "network_architecture": COURT_UNET_V2_ARCHITECTURE,
            "keypoint_names": KEYPOINT_NAMES,
            "seg_class_names": list(COURT_UNET_V2_SEG_CLASS_NAMES),
            "heatmap_stride": COURT_UNET_V2_HEATMAP_STRIDE,
            "args": {key: str(value) for key, value in vars(args).items()},
        },
        path,
    )


def _prune_epoch_checkpoints(out_dir: Path, *, keep_last: int) -> None:
    if keep_last < 0:
        raise ValueError("--keep-last-checkpoints must be non-negative")
    checkpoints = sorted(out_dir.glob("court_model_v2_epoch_*.pt"))
    stale = checkpoints if keep_last == 0 else checkpoints[:-keep_last]
    for path in stale:
        path.unlink()


# ---------------------------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    synthetic_workers = _resolve_synthetic_workers(args.synthetic_workers)
    encoder_weights_path = _resolve_encoder_weights_path(args.encoder_weights_path)
    args.out.mkdir(parents=True, exist_ok=True)

    synthetic_config: dict[str, Any] = {"image_size": [args.image_width, args.image_height]}
    if args.synthetic_scenario:
        synthetic_config["scenarios"] = list(args.synthetic_scenario)

    real_roots = args.real_root or []
    real_rows = load_real_training_rows(real_roots)
    # Deterministic, non-overlapping split: the first `real_val_samples` rows are held out for
    # `real_holdout_after` eval ONLY and are never drawn into a training mini-batch below -- an
    # on-disk corpus is a finite dataset (unlike the synthetic stream), so without this split a
    # real-corpus holdout eval would silently score rows the model had already been trained on.
    real_holdout_rows = real_rows[: args.real_val_samples]
    real_train_rows = real_rows[args.real_val_samples :]
    real_fraction = 0.0
    if real_train_rows and (args.real_weight + args.synthetic_weight) > 0:
        real_fraction = args.real_weight / (args.real_weight + args.synthetic_weight)

    model = make_court_keypoint_heatmap_model(
        KEYPOINT_COUNT,
        architecture=COURT_UNET_V2_ARCHITECTURE,
        encoder_weights_path=encoder_weights_path,
    ).to(device)
    param_count = int(sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    if args.resume is not None:
        if not Path(args.resume).is_file():
            raise FileNotFoundError(f"--resume checkpoint does not exist: {args.resume}")
        checkpoint = torch.load(str(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))

    # ONE-TIME, fixed-seed held-out set (never regenerated, never trained on): the CPU smoke
    # acceptance proof and every mid-training eval score against this exact same batch.
    val_batch = materialize_synthetic_batch(
        config=synthetic_config,
        seed=args.val_seed,
        count=args.val_samples,
        model_width=args.image_width,
        model_height=args.image_height,
        sigma_px=args.heatmap_sigma_px,
        force_fallback=args.synthetic_fallback,
        torch=torch,
    )

    before = evaluate_on_batch(
        model, val_batch, device=device, heatmap_stride=COURT_UNET_V2_HEATMAP_STRIDE, torch=torch
    )

    history: list[dict[str, Any]] = []
    epoch_wall_times: list[float] = []
    saved_epoch_checkpoints: list[str] = []
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        epoch_losses: list[float] = []
        synthetic_loader = _make_synthetic_training_dataloader(
            config=synthetic_config,
            base_seed=args.seed,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            global_step_offset=epoch * args.steps_per_epoch,
            model_width=args.image_width,
            model_height=args.image_height,
            sigma_px=args.heatmap_sigma_px,
            force_fallback=args.synthetic_fallback,
            synthetic_workers=synthetic_workers,
            torch=torch,
        )
        last_loader_meta: dict[str, Any] | None = None
        first_loader_meta: dict[str, Any] | None = None
        for train_batch in _iter_ordered_synthetic_batches(synthetic_loader, expected_steps=args.steps_per_epoch):
            loader_meta = dict(train_batch.get("loader_meta") or {})
            if first_loader_meta is None:
                first_loader_meta = loader_meta
            last_loader_meta = loader_meta
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss, components = _compute_batch_loss(
                    model, train_batch, device=device, args=args, torch=torch
                )
            if real_train_rows and rng.random() < real_fraction:
                real_batch = _stack_arrays(
                    [
                        real_row_to_sample_arrays(
                            row,
                            model_width=args.image_width,
                            model_height=args.image_height,
                            sigma_px=args.heatmap_sigma_px,
                        )
                        for row in rng.sample(real_train_rows, min(args.real_batch_size, len(real_train_rows)))
                    ],
                    torch,
                )
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    real_loss, real_components = _compute_batch_loss(
                        model, real_batch, device=device, args=args, torch=torch, skip_geometric=True
                    )
                loss = loss + real_loss
                components["real_heatmap_loss"] = real_components["heatmap_loss"]
                components["real_vis_loss"] = real_components["vis_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        epoch_wall_times.append(time.time() - epoch_start)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            eval_result = evaluate_on_batch(
                model, val_batch, device=device, heatmap_stride=COURT_UNET_V2_HEATMAP_STRIDE, torch=torch
            )
            avg_epoch_time = sum(epoch_wall_times) / len(epoch_wall_times)
            remaining_epochs = args.epochs - (epoch + 1)
            checkpoint_epoch_path: Path | None = None
            if args.checkpoint_every_eval:
                checkpoint_epoch_path = _checkpoint_path(args.out, epoch + 1)
                _save_training_checkpoint(
                    path=checkpoint_epoch_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    args=args,
                    torch=torch,
                )
                _prune_epoch_checkpoints(args.out, keep_last=args.keep_last_checkpoints)
                saved_epoch_checkpoints = [str(path) for path in sorted(args.out.glob("court_model_v2_epoch_*.pt"))]
            first_seed = None if first_loader_meta is None else first_loader_meta.get("synthetic_seed")
            last_seed = None if last_loader_meta is None else last_loader_meta.get("synthetic_seed")
            row = {
                "epoch": epoch + 1,
                "train_loss_mean": sum(epoch_losses) / len(epoch_losses),
                "train_loss_last": epoch_losses[-1],
                "lr": scheduler.get_last_lr()[0],
                "epoch_wall_time_s": epoch_wall_times[-1],
                "eta_s": avg_epoch_time * remaining_epochs,
                "synthetic_worker_count": 1 if last_loader_meta is None else last_loader_meta.get("worker_count", 1),
                "synthetic_seed_first": first_seed,
                "synthetic_seed_last": last_seed,
                "checkpoint": None if checkpoint_epoch_path is None else str(checkpoint_epoch_path),
                **eval_result,
            }
            history.append(row)
            # Per-epoch progress -> stderr, so stdout stays reserved for exactly one JSON
            # document (the final CLI summary printed by main()) that callers can parse directly.
            print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)

    after = evaluate_on_batch(
        model, val_batch, device=device, heatmap_stride=COURT_UNET_V2_HEATMAP_STRIDE, torch=torch
    )

    real_after: dict[str, Any] | None = None
    if real_holdout_rows:
        real_val_batch = _stack_arrays(
            [
                real_row_to_sample_arrays(
                    row, model_width=args.image_width, model_height=args.image_height, sigma_px=args.heatmap_sigma_px
                )
                for row in real_holdout_rows
            ],
            torch,
        )
        real_after = evaluate_on_batch(
            model, real_val_batch, device=device, heatmap_stride=COURT_UNET_V2_HEATMAP_STRIDE, torch=torch
        )

    checkpoint_path = args.out / "court_model_v2.pt"
    _save_training_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=args.epochs,
        args=args,
        torch=torch,
    )

    gate_value = after.get("pck_at_5px")
    summary = {
        "schema_version": 1,
        "artifact_type": "court_keypoint_pretraining_run",
        "status": "trained_not_phase_verified",
        "checkpoint": str(checkpoint_path),
        "gate": {
            "metric": "heldout_synthetic_pck_at_5px",
            "value": gate_value,
            "threshold": 0.95,
            "pck_threshold_px": 5.0,
            "passed": bool(gate_value is not None and float(gate_value) >= 0.95),
            "not_cal3_verified": True,
            "note": (
                "This gate is scored on synthetic held-out samples only; the CAL promotion gate "
                "is PCK@5px>=0.95 on the 32 reviewed real-label rows, scored by "
                "scripts/racketsport/evaluate_court_model_v2.py, never by this synthetic number."
            ),
        },
        "before": before,
        "after": after,
        "real_holdout_after": real_after,
        "real_holdout_count": len(real_holdout_rows),
        "history": history,
        "architecture": {
            "name": COURT_UNET_V2_ARCHITECTURE,
            "network_architecture": COURT_UNET_V2_ARCHITECTURE,
            "param_count": param_count,
            "heatmap_stride": COURT_UNET_V2_HEATMAP_STRIDE,
            "seg_class_names": list(COURT_UNET_V2_SEG_CLASS_NAMES),
            "image_size": [args.image_width, args.image_height],
        },
        "postprocess": {
            "prediction_mode": "keypoint_heatmap_subpixel_argmax",
            "decode": "parabolic_subpixel_refine",
        },
        "training": {
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "synthetic_loader": "SyntheticCourtIterableDataset+DataLoader",
            "synthetic_workers": synthetic_workers,
            "synthetic_stream_mapping": (
                "absolute_step_index g uses synthetic seed seed+g+1; worker w prepares local "
                "epoch steps w,w+num_workers,...; batches are consumed in global_step_index order"
            ),
            "amp": amp_enabled,
            "real_train_row_count": len(real_train_rows),
            "real_fraction": real_fraction,
            "synthetic_fallback": bool(args.synthetic_fallback or _iter_synthetic_court_samples is None),
            "geometric_loss_weight": args.geometric_loss_weight,
            "checkpoint_every_eval": bool(args.checkpoint_every_eval),
            "keep_last_checkpoints": args.keep_last_checkpoints,
            "epoch_checkpoints": saved_epoch_checkpoints,
            "encoder_weights_requested": None if args.encoder_weights_path is None else str(args.encoder_weights_path),
            "encoder_weights_path": None if encoder_weights_path is None else str(encoder_weights_path),
        },
        "holdout_artifacts": [],
        "note": "CAL-MODEL v2 court_unet_v2 trainer run; not a CAL3 promotion (see gate.note).",
    }
    metrics_path = args.out / "court_keypoint_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _compute_batch_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    device: Any,
    args: argparse.Namespace,
    torch: Any,
    skip_geometric: bool = False,
) -> tuple[Any, dict[str, float]]:
    outputs = model(batch["image"].to(device))
    heatmap_loss = court_keypoint_heatmap_loss(
        outputs["keypoint_heatmaps"],
        batch["heatmaps"].to(device),
        batch["heatmap_mask"].to(device).unsqueeze(-1).unsqueeze(-1).expand_as(batch["heatmaps"]).to(device),
    )
    vis_loss = visibility_bce_loss(outputs["keypoint_vis_logits"], batch["vis_target"].to(device), torch=torch)
    seg_loss = class_balanced_seg_loss(
        outputs["line_family_logits"], batch["seg_target"].to(device), batch["seg_mask"].to(device), torch=torch
    )
    loss = heatmap_loss + args.vis_loss_weight * vis_loss + args.seg_loss_weight * seg_loss
    components = {
        "heatmap_loss": float(heatmap_loss.detach().cpu()),
        "vis_loss": float(vis_loss.detach().cpu()),
        "seg_loss": float(seg_loss.detach().cpu()),
    }
    if args.geometric_loss_weight > 0.0 and not skip_geometric:
        head_height, head_width = outputs["keypoint_heatmaps"].shape[-2:]
        geometric = court_geometric_consistency_loss(
            outputs["keypoint_heatmaps"],
            keypoint_names=KEYPOINT_NAMES,
            image_width=float(head_width),
            image_height=float(head_height),
            colinearity_weight=args.geometric_colinearity_weight,
            homography_weight=args.geometric_homography_weight,
        )
        loss = loss + args.geometric_loss_weight * geometric["loss"]
        components["geometric_loss"] = float(geometric["loss"].detach().cpu())
    return loss, components


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument(
        "--encoder-weights-path",
        default=None,
        help=(
            "Optional local torchvision resnet34 state dict path, or 'imagenet' to use the "
            "locally cached torchvision ResNet34_Weights.IMAGENET1K_V1 file. This trainer never "
            "downloads weights; missing or shape-mismatched requested weights fail loudly."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-seed", type=int, default=999983)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--heatmap-sigma-px", type=float, default=1.5)
    parser.add_argument("--seg-loss-weight", type=float, default=1.0)
    parser.add_argument("--vis-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--geometric-loss-weight",
        type=float,
        default=0.05,
        help=(
            "Weight for the existing court_keypoint_geometric_loss point+line consistency "
            "regularizer (colinearity + homography self-consistency + degenerate-layout guard), "
            "computed from this model's own soft-argmax-decoded predictions every step. Enabled "
            "by default (unlike the legacy train_court_keypoint_heatmap.py trainer, which "
            "defaults this to 0.0 for backward compatibility with pre-CAL-R2 checkpoints -- this "
            "is a new architecture with no such constraint). Pass 0.0 to disable."
        ),
    )
    parser.add_argument("--geometric-colinearity-weight", type=float, default=1.0)
    parser.add_argument("--geometric-homography-weight", type=float, default=1.0)
    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Force the tiny self-contained procedural sampler even if court_synth_stream is importable.",
    )
    parser.add_argument("--synthetic-scenario", action="append", default=None)
    parser.add_argument(
        "--synthetic-workers",
        type=int,
        default=None,
        help="DataLoader workers for synthetic training batches. Defaults to min(8, os.cpu_count()). Use 0 for inline loading.",
    )
    parser.add_argument(
        "--real-root",
        type=Path,
        action="append",
        default=None,
        help="Root(s) containing <clip>/labels/court_keypoints.json rows (existing tier format).",
    )
    parser.add_argument("--real-weight", type=float, default=0.0)
    parser.add_argument("--synthetic-weight", type=float, default=1.0)
    parser.add_argument("--real-batch-size", type=int, default=4)
    parser.add_argument("--real-val-samples", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write court_model_v2_epoch_XXXX.pt at every eval epoch so preemption does not lose all progress.",
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=3,
        help="Keep only the newest N periodic epoch checkpoints. The final court_model_v2.pt is always written.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        summary = run_training(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"checkpoint": summary["checkpoint"], "gate": summary["gate"], "before": summary["before"], "after": summary["after"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
