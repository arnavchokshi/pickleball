"""Zero-disk streaming API for synthetic court training samples (CAL-SYNTH v2, 2026-07-05).

STABLE CONTRACT for CAL-MODEL: ``iter_synthetic_court_samples`` yields dicts shaped for a
dataloader, generated entirely in-memory (no disk reads or writes, no ``eval_clips/**`` access, no
torch import). Deterministic for a given ``(config, seed)`` pair.

Each yielded dict has exactly these keys:
    image_bgr        HxWx3 uint8, BGR channel order.
    keypoints_xy      float32 array, shape (15, 2), pixel coords in
                       ``threed.racketsport.court_keypoint_net.PICKLEBALL_KEYPOINTS`` order.
    keypoints_vis     int array, shape (15,), values in {0, 1, 2} =
                       {off_frame, occluded, visible} (see ``KEYPOINT_VIS_CLASSES``).
    line_family_mask  HxW uint8, values in {0..3} = {other, pickleball_line, tennis_line, net}
                       (see ``LINE_FAMILY_CLASSES``).
    surface_mask      HxW uint8, values in {0, 1, 2} = {background, apron, interior}
                       (see ``SURFACE_CLASSES``).
    meta              dict with (at least) ``homography``, ``distortion``, ``scenario``,
                       ``image_size`` keys; ``court_synth_scenes.reproject_canonical_keypoints(meta)``
                       reproduces ``keypoints_xy`` to floating-point precision (self-consistency
                       proof, well under the 0.5px acceptance bar) from ``meta`` alone.

``config`` (all keys optional; unset keys use the ``SceneRenderConfig`` defaults, which already
cover the pinned real-eval-geometry ranges -- height 1-12m, distance 5-40m, azimuth -75..75deg,
focal 500-2000px eq):
    count                 int | None. None (default) streams forever.
    image_size            [width, height]. Applied to every scenario (including portrait_phone,
                           which otherwise defaults to a 9:16 canvas) when explicitly set -- pass
                           this to keep dataloader batches a single fixed shape.
    scenarios             list[str] restricting which of ``SCENARIO_NAMES`` may be sampled.
    scenario_weights      dict[str, float] mixture weights (unnormalized) over ``SCENARIO_NAMES``.
    apply_jpeg_roundtrip  bool, default False (skipped by default for streaming throughput; the
                           disk-writing CLI generator always applies it since it saves real JPEGs).
    height_m_range / distance_m_range / azimuth_deg_range / tilt_deg_range / roll_deg_range /
    focal_px_range / distortion_k1_range / distortion_p_range / jpeg_quality_range /
    line_width_px_range   range overrides forwarded to ``SceneRenderConfig``.
"""

from __future__ import annotations

import random
from typing import Any, Iterator

from threed.racketsport.court_synth_scenes import (
    KEYPOINT_VIS_CLASSES,
    LINE_FAMILY_CLASSES,
    SCENARIO_NAMES,
    SURFACE_CLASSES,
    CANONICAL_KEYPOINT_NAMES,
    SceneRenderConfig,
    choose_scenario,
    render_synthetic_court_sample,
    reproject_canonical_keypoints,
)

__all__ = [
    "iter_synthetic_court_samples",
    "scene_config_from_stream_config",
    "SCENARIO_NAMES",
    "LINE_FAMILY_CLASSES",
    "SURFACE_CLASSES",
    "KEYPOINT_VIS_CLASSES",
    "reproject_canonical_keypoints",
]

_RANGE_KEYS = (
    "height_m_range",
    "distance_m_range",
    "azimuth_deg_range",
    "tilt_deg_range",
    "roll_deg_range",
    "focal_px_range",
    "distortion_k1_range",
    "distortion_p_range",
    "jpeg_quality_range",
    "line_width_px_range",
)


def scene_config_from_stream_config(config: dict[str, Any] | None) -> tuple[SceneRenderConfig, dict[str, Any]]:
    """Build a ``SceneRenderConfig`` plus stream-only options (``count``, ``apply_jpeg_roundtrip``)."""

    cfg = dict(config or {})
    kwargs: dict[str, Any] = {}
    if "image_size" in cfg and cfg["image_size"] is not None:
        kwargs["image_size"] = tuple(cfg["image_size"])
    for key in _RANGE_KEYS:
        if key in cfg and cfg[key] is not None:
            kwargs[key] = tuple(cfg[key])

    scenario_weights = {name: 1.0 for name in SCENARIO_NAMES}
    if cfg.get("scenarios"):
        allowed = set(cfg["scenarios"])
        unknown = allowed - set(SCENARIO_NAMES)
        if unknown:
            raise ValueError(f"unknown synthetic court scenarios: {sorted(unknown)}")
        scenario_weights = {name: (1.0 if name in allowed else 0.0) for name in SCENARIO_NAMES}
    if cfg.get("scenario_weights"):
        unknown = set(cfg["scenario_weights"]) - set(SCENARIO_NAMES)
        if unknown:
            raise ValueError(f"unknown synthetic court scenarios: {sorted(unknown)}")
        scenario_weights = {name: float(cfg["scenario_weights"].get(name, 0.0)) for name in SCENARIO_NAMES}
    kwargs["scenario_weights"] = scenario_weights

    scene_config = SceneRenderConfig(**kwargs)
    stream_options = {
        "count": cfg.get("count"),
        "apply_jpeg_roundtrip": bool(cfg.get("apply_jpeg_roundtrip", False)),
        "force_image_size": "image_size" in cfg and cfg["image_size"] is not None,
    }
    return scene_config, stream_options


def iter_synthetic_court_samples(config: dict[str, Any] | None = None, seed: int = 0) -> Iterator[dict[str, Any]]:
    """Yield zero-disk synthetic court training samples. Deterministic for a given (config, seed)."""

    import numpy as np

    scene_config, stream_options = scene_config_from_stream_config(config)
    count = stream_options["count"]
    apply_jpeg_roundtrip = stream_options["apply_jpeg_roundtrip"]
    force_image_size = stream_options["force_image_size"]
    forced_image_size = scene_config.image_size

    rng = random.Random(seed)
    produced = 0
    while count is None or produced < count:
        scenario = choose_scenario(rng, scene_config.scenario_weights)
        scene = render_synthetic_court_sample(
            rng,
            scene_config,
            scenario=scenario,
            apply_jpeg_roundtrip=apply_jpeg_roundtrip,
            force_image_size=forced_image_size if force_image_size else None,
        )

        image_rgb = np.asarray(scene.image, dtype=np.uint8)
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        keypoints_xy = np.asarray(
            [scene.keypoints_xy[name] for name in CANONICAL_KEYPOINT_NAMES], dtype=np.float32
        )
        keypoints_vis = np.asarray(
            [scene.keypoints_vis[name] for name in CANONICAL_KEYPOINT_NAMES], dtype=np.int64
        )
        yield {
            "image_bgr": image_bgr,
            "keypoints_xy": keypoints_xy,
            "keypoints_vis": keypoints_vis,
            "line_family_mask": scene.line_family_mask,
            "surface_mask": scene.surface_mask,
            "meta": scene.meta,
        }
        produced += 1
