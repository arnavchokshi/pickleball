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

A2 opt-in (all dormant by default):
    aux_keypoints                   bool. Emit canonical-first 33-point coordinates/visibility
                                    plus stride-4 ``keypoint_heatmaps`` and
                                    ``keypoint_heatmap_mask`` targets.
    paint_texture_randomization     bool. Defaults to ``aux_keypoints``; enables the explicit
                                    HSV/wear/fade/surface/decorrelation ranges below.
    aux_partial_visibility          bool. Defaults to ``aux_keypoints``; accepts owner-style
                                    crops with >=1 canonical and >=6 combined correspondences.
    paint_hue_jitter_deg_range / paint_saturation_scale_range /
    paint_value_scale_range / line_wear_range / line_fade_alpha_range /
    surface_hue_jitter_deg_range / surface_saturation_scale_range /
    surface_value_scale_range / surface_texture_strength_range /
    surface_texture_scale_px_range / apron_independent_palette_probability_range
                                    Explicit flag-on randomization knobs documented in the lane
                                    DESIGN.md. They are never sampled on the default path.
"""

from __future__ import annotations

import random
from typing import Any, Iterator

from threed.racketsport.court_keypoint_net import COURT_UNET_V2_HEATMAP_STRIDE
from threed.racketsport.court_synth_scenes import (
    ALL_KEYPOINT_NAMES,
    AUX_KEYPOINT_NAMES,
    KEYPOINT_VIS_CLASSES,
    LINE_FAMILY_CLASSES,
    SCENARIO_NAMES,
    SURFACE_CLASSES,
    CANONICAL_KEYPOINT_NAMES,
    SceneRenderConfig,
    choose_scenario,
    render_synthetic_court_sample,
    reproject_canonical_keypoints,
    reproject_scene_keypoints,
)

__all__ = [
    "iter_synthetic_court_samples",
    "scene_config_from_stream_config",
    "SCENARIO_NAMES",
    "LINE_FAMILY_CLASSES",
    "SURFACE_CLASSES",
    "KEYPOINT_VIS_CLASSES",
    "reproject_canonical_keypoints",
    "reproject_scene_keypoints",
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
    "paint_hue_jitter_deg_range",
    "paint_saturation_scale_range",
    "paint_value_scale_range",
    "line_wear_range",
    "line_fade_alpha_range",
    "surface_hue_jitter_deg_range",
    "surface_saturation_scale_range",
    "surface_value_scale_range",
    "surface_texture_strength_range",
    "surface_texture_scale_px_range",
    "apron_independent_palette_probability_range",
)

_ACTIVE_UNIT_INTERVAL_RANGES = (
    "line_wear_range",
    "line_fade_alpha_range",
    "surface_texture_strength_range",
    "apron_independent_palette_probability_range",
)


def _strict_bool(config: dict[str, Any], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _validate_active_randomization_ranges(scene_config: SceneRenderConfig) -> None:
    if not scene_config.paint_texture_randomization:
        return
    for name in _ACTIVE_UNIT_INTERVAL_RANGES:
        low, high = getattr(scene_config, name)
        if low > high or low < 0.0 or high > 1.0:
            raise ValueError(f"{name} must be an ascending range within [0, 1]")
    for name in (
        "paint_saturation_scale_range",
        "paint_value_scale_range",
        "surface_saturation_scale_range",
        "surface_value_scale_range",
        "surface_texture_scale_px_range",
    ):
        low, high = getattr(scene_config, name)
        if low > high or low < 0.0:
            raise ValueError(f"{name} must be an ascending non-negative range")


def scene_config_from_stream_config(config: dict[str, Any] | None) -> tuple[SceneRenderConfig, dict[str, Any]]:
    """Build a ``SceneRenderConfig`` plus stream-only options (``count``, ``apply_jpeg_roundtrip``)."""

    cfg = dict(config or {})
    aux_keypoints = _strict_bool(cfg, "aux_keypoints", False)
    kwargs: dict[str, Any] = {}
    if "image_size" in cfg and cfg["image_size"] is not None:
        kwargs["image_size"] = tuple(cfg["image_size"])
    for key in _RANGE_KEYS:
        if key in cfg and cfg[key] is not None:
            kwargs[key] = tuple(cfg[key])
    kwargs["paint_texture_randomization"] = _strict_bool(
        cfg,
        "paint_texture_randomization",
        aux_keypoints,
    )
    kwargs["aux_partial_visibility"] = _strict_bool(
        cfg,
        "aux_partial_visibility",
        aux_keypoints,
    )
    for name, default in (("aux_min_visible_canonical", 1), ("aux_min_visible_combined", 6)):
        value = cfg.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        kwargs[name] = value
    if kwargs["aux_partial_visibility"] and not aux_keypoints:
        raise ValueError("aux_partial_visibility requires aux_keypoints=true")

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
    _validate_active_randomization_ranges(scene_config)
    heatmap_stride = cfg.get("heatmap_stride", 4)
    if isinstance(heatmap_stride, bool) or not isinstance(heatmap_stride, int) or heatmap_stride <= 0:
        raise ValueError("heatmap_stride must be a positive integer")
    if aux_keypoints and heatmap_stride != COURT_UNET_V2_HEATMAP_STRIDE:
        raise ValueError(
            f"aux_keypoints requires heatmap_stride={COURT_UNET_V2_HEATMAP_STRIDE} "
            "for court_unet_v2_aux"
        )
    heatmap_sigma_px = cfg.get("heatmap_sigma_px", 1.5)
    if isinstance(heatmap_sigma_px, bool) or not isinstance(heatmap_sigma_px, (int, float)):
        raise ValueError("heatmap_sigma_px must be positive")
    heatmap_sigma_px = float(heatmap_sigma_px)
    if heatmap_sigma_px <= 0.0:
        raise ValueError("heatmap_sigma_px must be positive")
    stream_options = {
        "count": cfg.get("count"),
        "apply_jpeg_roundtrip": bool(cfg.get("apply_jpeg_roundtrip", False)),
        "force_image_size": "image_size" in cfg and cfg["image_size"] is not None,
        "aux_keypoints": aux_keypoints,
        "heatmap_stride": heatmap_stride,
        "heatmap_sigma_px": heatmap_sigma_px,
    }
    return scene_config, stream_options


def _heatmap_targets(
    keypoints_xy: Any,
    keypoints_vis: Any,
    *,
    image_size: tuple[int, int],
    stride: int,
    sigma_px: float,
    line_family_mask: Any | None = None,
    aux_start_index: int | None = None,
) -> tuple[Any, Any]:
    import numpy as np

    width, height = image_size
    head_width = int((width + stride - 1) // stride)
    head_height = int((height + stride - 1) // stride)
    yy, xx = np.mgrid[0:head_height, 0:head_width]
    heatmaps = np.zeros((len(keypoints_xy), head_height, head_width), dtype=np.float32)
    masks = np.zeros_like(heatmaps, dtype=np.float32)
    for index, ((x, y), visibility) in enumerate(zip(keypoints_xy, keypoints_vis, strict=True)):
        if int(visibility) == KEYPOINT_VIS_CLASSES["off_frame"]:
            continue
        if line_family_mask is not None and aux_start_index is not None and index >= aux_start_index:
            # Fractional aux points are identifiable only from the painted segment beneath them.
            # Coupled wear and foreground occluders remove that segment from the line-family mask;
            # do not ask the network to regress an arbitrary invisible fraction in those cases.
            center_x_px = int(round(float(x)))
            center_y_px = int(round(float(y)))
            support_radius_px = max(2, stride)
            x0 = max(0, center_x_px - support_radius_px)
            x1 = min(width, center_x_px + support_radius_px + 1)
            y0 = max(0, center_y_px - support_radius_px)
            y1 = min(height, center_y_px + support_radius_px + 1)
            local_mask = np.asarray(line_family_mask)[y0:y1, x0:x1]
            if not np.any(local_mask == LINE_FAMILY_CLASSES["pickleball_line"]):
                continue
        center_x = float(x) / stride
        center_y = float(y) / stride
        heatmaps[index] = np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma_px**2)
        )
        masks[index] = 1.0
    return heatmaps, masks


def iter_synthetic_court_samples(config: dict[str, Any] | None = None, seed: int = 0) -> Iterator[dict[str, Any]]:
    """Yield zero-disk synthetic court training samples. Deterministic for a given (config, seed)."""

    import numpy as np

    scene_config, stream_options = scene_config_from_stream_config(config)
    count = stream_options["count"]
    apply_jpeg_roundtrip = stream_options["apply_jpeg_roundtrip"]
    force_image_size = stream_options["force_image_size"]
    forced_image_size = scene_config.image_size
    aux_keypoints = stream_options["aux_keypoints"]

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
            include_aux_keypoints=aux_keypoints,
        )

        image_rgb = np.asarray(scene.image, dtype=np.uint8)
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        keypoint_names = ALL_KEYPOINT_NAMES if aux_keypoints else CANONICAL_KEYPOINT_NAMES
        keypoints_xy = np.asarray([scene.keypoints_xy[name] for name in keypoint_names], dtype=np.float32)
        keypoints_vis = np.asarray([scene.keypoints_vis[name] for name in keypoint_names], dtype=np.int64)
        sample = {
            "image_bgr": image_bgr,
            "keypoints_xy": keypoints_xy,
            "keypoints_vis": keypoints_vis,
            "line_family_mask": scene.line_family_mask,
            "surface_mask": scene.surface_mask,
            "meta": scene.meta,
        }
        if aux_keypoints:
            heatmaps, heatmap_masks = _heatmap_targets(
                keypoints_xy,
                keypoints_vis,
                image_size=(image_bgr.shape[1], image_bgr.shape[0]),
                stride=stream_options["heatmap_stride"],
                sigma_px=stream_options["heatmap_sigma_px"],
                line_family_mask=scene.line_family_mask,
                aux_start_index=len(CANONICAL_KEYPOINT_NAMES),
            )
            sample["keypoint_heatmaps"] = heatmaps
            sample["keypoint_heatmap_mask"] = heatmap_masks
            aux_visibility = keypoints_vis[len(CANONICAL_KEYPOINT_NAMES) :]
            aux_masks = heatmap_masks[len(CANONICAL_KEYPOINT_NAMES) :]
            unsupported_aux = sum(
                int(visibility) != KEYPOINT_VIS_CLASSES["off_frame"] and not bool(np.any(mask))
                for visibility, mask in zip(aux_visibility, aux_masks, strict=True)
            )
            sample["meta"]["aux_keypoints"].update(
                {
                    "target_mask_rule": "off_frame_or_no_local_pickleball_paint_support_is_unsupervised",
                    "local_paint_support_radius_px": max(2, stream_options["heatmap_stride"]),
                    "paint_unsupported_in_frame_count": unsupported_aux,
                }
            )
        yield sample
        produced += 1
