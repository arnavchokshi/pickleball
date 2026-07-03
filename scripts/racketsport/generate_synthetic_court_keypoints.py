#!/usr/bin/env python3
"""Generate synthetic pickleball court keypoint labels from regulation geometry.

The output intentionally uses the same ``court_keypoints.json`` envelope consumed by
``train_court_keypoint_heatmap.py``, but every label carries synthetic provenance and uses
the dedicated ``synthetic`` item status accepted by the loader (CAL-R2 provenance fix,
2026-07-02: this used to be ``reviewed_static_camera_copy``, an enum workaround that let
synthetic rows silently inflate a count meant only for owner-approved REAL human-review copies
-- see ``SYNTHETIC_STATUS`` in ``train_court_keypoint_heatmap.py``). These images are training
augmentation only, never gate evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_calibration import homography_from_planar_points
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_templates import COORDINATE_FRAME, get_court_template


DEFAULT_OUTPUT_DIR = Path("runs/training_corpora_20260701/court_synthetic")
DEFAULT_SEED = 20260701
DEFAULT_IMAGE_SIZE = (640, 360)
DEFAULT_COUNT = 2000
DEFAULT_SPOT_CHECK_COUNT = 20
SYNTHETIC_ITEM_STATUS = "synthetic"
NET_KEYPOINT_HEIGHT_CONVENTION = "regulation_net_top"


Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class SyntheticCourtGenerationConfig:
    out_dir: Path = DEFAULT_OUTPUT_DIR
    count: int = DEFAULT_COUNT
    seed: int = DEFAULT_SEED
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    spot_check_count: int = DEFAULT_SPOT_CHECK_COUNT
    generated_at_utc: str | None = None
    height_m_range: tuple[float, float] = (1.0, 8.0)
    distance_m_range: tuple[float, float] = (8.0, 34.0)
    azimuth_deg_range: tuple[float, float] = (-60.0, 60.0)
    tilt_deg_range: tuple[float, float] = (3.0, 62.0)
    focal_mm_eq_range: tuple[float, float] = (22.0, 80.0)
    roll_deg_range: tuple[float, float] = (-2.5, 2.5)
    distortion_k1_range: tuple[float, float] = (-0.045, 0.025)
    jpeg_quality_range: tuple[int, int] = (78, 96)
    line_width_px_range: tuple[int, int] = (2, 7)
    overwrite: bool = False


@dataclass(frozen=True)
class CameraPose:
    position_m: Vector3
    target_m: Vector3
    right: Vector3
    down: Vector3
    forward: Vector3
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    height_m: float
    distance_m: float
    azimuth_deg: float
    tilt_deg: float
    roll_deg: float
    focal_mm_eq: float
    distortion_k1: float


@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str
    image_rel_path: Path
    label_rel_path: Path
    overlay_rel_path: Path | None
    image_sha256: str
    label_sha256: str
    overlay_sha256: str | None
    keypoints: dict[str, list[float]]
    generation: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_synthetic_court_corpus(config: SyntheticCourtGenerationConfig) -> dict[str, Any]:
    _validate_config(config)
    out_dir = config.out_dir
    _prepare_output_dir(out_dir, overwrite=config.overwrite)

    generated_at = config.generated_at_utc or datetime.now(timezone.utc).isoformat()
    rng = random.Random(config.seed)
    overlay_indices = set(_spot_check_indices(config.count, config.spot_check_count, config.seed))

    samples: list[SyntheticSample] = []
    for sample_index in range(config.count):
        samples.append(
            _generate_sample(
                out_dir,
                sample_index=sample_index,
                config=config,
                rng=rng,
                generated_at_utc=generated_at,
                write_overlay=sample_index in overlay_indices,
            )
        )

    manifest = {
        "schema_version": 1,
        "artifact_type": "synthetic_court_keypoint_corpus_manifest",
        "status": "synthetic_training_ammunition_not_gate_evidence",
        "generated_at_utc": generated_at,
        "seed": config.seed,
        "sample_count": config.count,
        "output_dir": _path_text(out_dir),
        "schema_notes": [
            "Each sample is loadable by scripts/racketsport/train_court_keypoint_heatmap.py via root/*/labels/court_keypoints.json.",
            "review.status is set to reviewed only to satisfy the existing loader contract.",
            "item.status is synthetic (CAL-R2 provenance fix, 2026-07-02) so these rows count separately "
            "(labels_synthetic_frame_count) from both independent human reviews and owner-approved "
            "static-camera copies -- never as any form of human verification.",
            "provenance.synthetic=true marks the labels as synthetic training augmentation, never gate evidence.",
        ],
        "generation_config": {
            "image_size": list(config.image_size),
            "height_m_range": list(config.height_m_range),
            "distance_m_range": list(config.distance_m_range),
            "azimuth_deg_range": list(config.azimuth_deg_range),
            "tilt_deg_range": list(config.tilt_deg_range),
            "focal_mm_eq_range": list(config.focal_mm_eq_range),
            "roll_deg_range": list(config.roll_deg_range),
            "distortion_k1_range": list(config.distortion_k1_range),
            "jpeg_quality_range": list(config.jpeg_quality_range),
            "line_width_px_range": list(config.line_width_px_range),
        },
        "canonical_keypoint_names": [point.name for point in PICKLEBALL_KEYPOINTS],
        "court_template": _court_template_manifest(),
        "spot_check_overlays": [
            {
                "sample_id": sample.sample_id,
                "path": sample.overlay_rel_path.as_posix(),
                "sha256": sample.overlay_sha256,
            }
            for sample in samples
            if sample.overlay_rel_path is not None
        ],
        "samples": [
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_rel_path.as_posix(),
                "label_path": sample.label_rel_path.as_posix(),
                "image_sha256": sample.image_sha256,
                "label_sha256": sample.label_sha256,
                "overlay_path": sample.overlay_rel_path.as_posix() if sample.overlay_rel_path else None,
                "overlay_sha256": sample.overlay_sha256,
                "camera": sample.generation["camera"],
                "distortion_k1": sample.generation["camera"]["distortion_k1"],
                "occlusion_count": sample.generation["domain_randomization"]["occlusion_count"],
            }
            for sample in samples
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def _generate_sample(
    out_dir: Path,
    *,
    sample_index: int,
    config: SyntheticCourtGenerationConfig,
    rng: random.Random,
    generated_at_utc: str,
    write_overlay: bool,
) -> SyntheticSample:
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np

    width, height = config.image_size
    template = get_court_template("pickleball")
    sample_id = f"synthetic_court_{sample_index:06d}"
    sample_dir = out_dir / sample_id
    frames_dir = sample_dir / "frames"
    labels_dir = sample_dir / "labels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    pose, projected_keypoints = _sample_visible_camera(config, rng)
    corner_points = [
        point
        for point in PICKLEBALL_KEYPOINTS
        if point.name in {"near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner"}
    ]
    homography = homography_from_planar_points(
        [point.world_xyz_m[:2] for point in corner_points],
        [projected_keypoints[point.name] for point in corner_points],
    )

    colors = _sample_colors(rng)
    line_width = rng.randint(config.line_width_px_range[0], config.line_width_px_range[1])
    line_wear = rng.uniform(0.0, 0.26)
    image = Image.new("RGB", (width, height), colors["floor_base"])
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_background_clutter(draw, width, height, rng)
    _draw_court_surface(draw, pose, colors, rng)
    _draw_court_lines(draw, pose, colors["line"], line_width, line_wear, rng)
    _draw_soft_shadows(image, width, height, rng)
    occlusion_count = _draw_occlusions(draw, width, height, colors, rng)
    image = _apply_lighting_and_sensor_artifacts(image, rng, np)

    jpeg_quality = rng.randint(config.jpeg_quality_range[0], config.jpeg_quality_range[1])
    image = _jpeg_roundtrip(image, jpeg_quality)

    frame_name = "frame_000000.jpg"
    image_path = frames_dir / frame_name
    image.save(image_path, format="JPEG", quality=jpeg_quality, optimize=False, progressive=False)

    generation = _generation_payload(
        pose=pose,
        homography=homography,
        colors=colors,
        line_width=line_width,
        line_wear=line_wear,
        jpeg_quality=jpeg_quality,
        occlusion_count=occlusion_count,
    )
    label_payload = _label_payload(
        sample_id=sample_id,
        frame_name=frame_name,
        frames_dir=frames_dir,
        keypoints=projected_keypoints,
        image_size=config.image_size,
        generated_at_utc=generated_at_utc,
        seed=config.seed,
        sample_index=sample_index,
        generation=generation,
    )
    label_path = labels_dir / "court_keypoints.json"
    _write_json(label_path, label_payload)

    overlay_rel_path: Path | None = None
    overlay_sha256: str | None = None
    if write_overlay:
        overlay_dir = out_dir / "spot_check_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{sample_id}_overlay.jpg"
        overlay = image.copy()
        _draw_keypoint_overlay(overlay, projected_keypoints)
        overlay.save(overlay_path, format="JPEG", quality=94, optimize=False, progressive=False)
        overlay_rel_path = overlay_path.relative_to(out_dir)
        overlay_sha256 = sha256_file(overlay_path)

    return SyntheticSample(
        sample_id=sample_id,
        image_rel_path=image_path.relative_to(out_dir),
        label_rel_path=label_path.relative_to(out_dir),
        overlay_rel_path=overlay_rel_path,
        image_sha256=sha256_file(image_path),
        label_sha256=sha256_file(label_path),
        overlay_sha256=overlay_sha256,
        keypoints=projected_keypoints,
        generation=generation,
    )


def _sample_visible_camera(
    config: SyntheticCourtGenerationConfig,
    rng: random.Random,
) -> tuple[CameraPose, dict[str, list[float]]]:
    width, height = config.image_size
    for _ in range(800):
        pose = _sample_camera_pose(config, rng)
        projected = {
            point.name: list(_project_world_point(point.world_xyz_m, pose))
            for point in PICKLEBALL_KEYPOINTS
        }
        if _is_visible_projection(projected, width=width, height=height):
            return pose, projected
    raise RuntimeError("could not sample a visible synthetic court camera pose")


def _sample_camera_pose(config: SyntheticCourtGenerationConfig, rng: random.Random) -> CameraPose:
    width, height_px = config.image_size
    template = get_court_template("pickleball")
    court_half_length = template.length_m / 2.0
    camera_height = rng.uniform(*config.height_m_range)
    distance = rng.uniform(*config.distance_m_range)
    side = -1.0 if rng.random() < 0.5 else 1.0
    azimuth_deg = rng.uniform(*config.azimuth_deg_range)
    azimuth_rad = math.radians(azimuth_deg)
    camera_x = math.sin(azimuth_rad) * distance
    camera_y = side * (court_half_length + math.cos(azimuth_rad) * distance)
    target = (
        rng.uniform(-0.75, 0.75),
        rng.uniform(-1.8, 1.8),
        rng.uniform(0.0, 0.45),
    )
    position = (camera_x, camera_y, camera_height)
    forward = _normalize(_sub(target, position))
    horizontal_distance = math.hypot(target[0] - position[0], target[1] - position[1])
    tilt_deg = math.degrees(math.atan2(position[2] - target[2], horizontal_distance))
    if tilt_deg < config.tilt_deg_range[0] or tilt_deg > config.tilt_deg_range[1]:
        return _sample_camera_pose(config, rng)

    world_up = (0.0, 0.0, 1.0)
    right = _normalize(_cross(forward, world_up))
    up = _normalize(_cross(right, forward))
    roll_deg = rng.uniform(*config.roll_deg_range)
    roll_rad = math.radians(roll_deg)
    rolled_right = _add(_mul(right, math.cos(roll_rad)), _mul(up, math.sin(roll_rad)))
    rolled_up = _add(_mul(right, -math.sin(roll_rad)), _mul(up, math.cos(roll_rad)))
    down = _mul(rolled_up, -1.0)

    focal_mm_eq = rng.uniform(*config.focal_mm_eq_range)
    focal_px = width * focal_mm_eq / 36.0
    cx = width / 2.0 + rng.uniform(-0.08 * width, 0.08 * width)
    cy = height_px / 2.0 + rng.uniform(-0.05 * height_px, 0.10 * height_px)
    distortion_k1 = rng.uniform(*config.distortion_k1_range)
    return CameraPose(
        position_m=position,
        target_m=target,
        right=rolled_right,
        down=down,
        forward=forward,
        fx_px=focal_px,
        fy_px=focal_px,
        cx_px=cx,
        cy_px=cy,
        height_m=camera_height,
        distance_m=distance,
        azimuth_deg=azimuth_deg,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
        focal_mm_eq=focal_mm_eq,
        distortion_k1=distortion_k1,
    )


def _is_visible_projection(projected: dict[str, list[float]], *, width: int, height: int) -> bool:
    xs = [point[0] for point in projected.values()]
    ys = [point[1] for point in projected.values()]
    if min(xs) < 8.0 or max(xs) > width - 8.0 or min(ys) < 8.0 or max(ys) > height - 8.0:
        return False
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    if bbox_w < width * 0.34 or bbox_h < height * 0.24:
        return False
    if bbox_w > width * 0.98 or bbox_h > height * 0.98:
        return False
    return True


def _project_world_point(point: Sequence[float], pose: CameraPose) -> Vector2:
    rel = _sub((float(point[0]), float(point[1]), float(point[2])), pose.position_m)
    cam_x = _dot(rel, pose.right)
    cam_y = _dot(rel, pose.down)
    cam_z = _dot(rel, pose.forward)
    if cam_z <= 0.05:
        raise ValueError("world point is behind sampled camera")
    x = cam_x / cam_z
    y = cam_y / cam_z
    if pose.distortion_k1 != 0.0:
        r2 = x * x + y * y
        factor = 1.0 + pose.distortion_k1 * r2
        x *= factor
        y *= factor
    return (pose.fx_px * x + pose.cx_px, pose.fy_px * y + pose.cy_px)


def _draw_background_clutter(draw: Any, width: int, height: int, rng: random.Random) -> None:
    for _ in range(rng.randint(8, 22)):
        color = (
            rng.randint(20, 150),
            rng.randint(20, 150),
            rng.randint(20, 150),
            rng.randint(18, 70),
        )
        x0 = rng.uniform(-0.2 * width, width)
        y0 = rng.uniform(-0.2 * height, height)
        x1 = x0 + rng.uniform(0.03 * width, 0.28 * width)
        y1 = y0 + rng.uniform(0.02 * height, 0.18 * height)
        if rng.random() < 0.55:
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)


def _draw_court_surface(draw: Any, pose: CameraPose, colors: dict[str, Any], rng: random.Random) -> None:
    template = get_court_template("pickleball")
    polygon = [_project_world_point(corner, pose) for corner in template.corners_m]
    draw.polygon(polygon, fill=(*colors["court"], rng.randint(210, 245)))


def _draw_court_lines(
    draw: Any,
    pose: CameraPose,
    color: tuple[int, int, int],
    width: int,
    wear: float,
    rng: random.Random,
) -> None:
    template = get_court_template("pickleball")
    line_segments = dict(template.line_segments_m)
    for name, (start, end) in sorted(line_segments.items()):
        if name == "net":
            start, end = _net_top_segment_m(template)
        samples = _sample_line(start, end, 48)
        projected = [_project_world_point(point, pose) for point in samples]
        _draw_worn_polyline(draw, projected, color, width, wear, rng)


def _draw_worn_polyline(
    draw: Any,
    points: Sequence[Vector2],
    color: tuple[int, int, int],
    width: int,
    wear: float,
    rng: random.Random,
) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-1], points[1:], strict=True):
        if rng.random() < wear:
            continue
        jitter = rng.randint(-16, 12)
        alpha = max(110, min(255, 235 + jitter))
        draw.line([start, end], fill=(*color, alpha), width=width)


def _draw_soft_shadows(image: Any, width: int, height: int, rng: random.Random) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for _ in range(rng.randint(1, 4)):
        x = rng.uniform(-0.1 * width, 0.9 * width)
        y = rng.uniform(-0.1 * height, 0.9 * height)
        w = rng.uniform(0.18 * width, 0.55 * width)
        h = rng.uniform(0.08 * height, 0.35 * height)
        polygon = [
            (x, y),
            (x + w, y + rng.uniform(-0.1 * height, 0.1 * height)),
            (x + w * rng.uniform(0.65, 1.2), y + h),
            (x + rng.uniform(-0.1 * width, 0.1 * width), y + h * rng.uniform(0.7, 1.2)),
        ]
        draw.polygon(polygon, fill=(0, 0, 0, rng.randint(16, 56)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=rng.uniform(4.0, 14.0)))
    image.alpha_composite(overlay) if image.mode == "RGBA" else image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))


def _draw_occlusions(
    draw: Any,
    width: int,
    height: int,
    colors: dict[str, Any],
    rng: random.Random,
) -> int:
    count = rng.randint(1, 5)
    for _ in range(count):
        color_base = colors["floor_base"] if rng.random() < 0.65 else colors["court"]
        color = (
            _clamp_color(color_base[0] + rng.randint(-35, 35)),
            _clamp_color(color_base[1] + rng.randint(-35, 35)),
            _clamp_color(color_base[2] + rng.randint(-35, 35)),
            rng.randint(110, 225),
        )
        cx = rng.uniform(0.05 * width, 0.95 * width)
        cy = rng.uniform(0.10 * height, 0.95 * height)
        w = rng.uniform(0.04 * width, 0.18 * width)
        h = rng.uniform(0.025 * height, 0.16 * height)
        if rng.random() < 0.5:
            draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=color)
        else:
            draw.polygon(
                [
                    (cx - w / 2, cy - h * rng.uniform(0.2, 0.7)),
                    (cx + w * rng.uniform(0.1, 0.7), cy - h / 2),
                    (cx + w / 2, cy + h * rng.uniform(0.1, 0.7)),
                    (cx - w * rng.uniform(0.1, 0.7), cy + h / 2),
                ],
                fill=color,
            )
    return count


def _apply_lighting_and_sensor_artifacts(image: Any, rng: random.Random, np: Any) -> Any:
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape[:2]
    x_grad = np.linspace(rng.uniform(0.76, 1.03), rng.uniform(0.92, 1.22), width, dtype=np.float32)
    y_grad = np.linspace(rng.uniform(0.82, 1.08), rng.uniform(0.88, 1.18), height, dtype=np.float32)
    gradient = (x_grad[None, :] + y_grad[:, None]) / 2.0
    arr *= gradient[:, :, None]
    if rng.random() < 0.85:
        noise = rng.uniform(1.5, 7.5)
        arr += np.random.default_rng(rng.randrange(2**32)).normal(0.0, noise, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    from PIL import Image

    return Image.fromarray(arr)


def _jpeg_roundtrip(image: Any, quality: int) -> Any:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False, progressive=False)
    buffer.seek(0)
    from PIL import Image

    return Image.open(buffer).convert("RGB")


def _draw_keypoint_overlay(image: Any, keypoints: dict[str, list[float]]) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    palette = [
        (255, 64, 64, 230),
        (64, 180, 255, 230),
        (80, 230, 120, 230),
        (255, 220, 64, 230),
    ]
    for idx, point in enumerate(PICKLEBALL_KEYPOINTS):
        x, y = keypoints[point.name]
        color = palette[idx % len(palette)]
        radius = 4
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(0, 0, 0, 220))
        draw.text((x + 5, y - 6), point.name, fill=(255, 255, 255, 235), stroke_width=1, stroke_fill=(0, 0, 0, 220))


def _label_payload(
    *,
    sample_id: str,
    frame_name: str,
    frames_dir: Path,
    keypoints: dict[str, list[float]],
    image_size: tuple[int, int],
    generated_at_utc: str,
    seed: int,
    sample_index: int,
    generation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoint_labels",
        "clip": sample_id,
        "annotation": {
            "items": [
                {
                    "frame": frame_name,
                    "status": SYNTHETIC_ITEM_STATUS,
                    "net_keypoint_height_convention": NET_KEYPOINT_HEIGHT_CONVENTION,
                    "review_id": f"{sample_id}_synthetic_geometry_0000",
                    "keypoints": {name: [float(xy[0]), float(xy[1])] for name, xy in sorted(keypoints.items())},
                    "provenance": {
                        "synthetic": True,
                        "human_reviewed": False,
                        "generator": "scripts/racketsport/generate_synthetic_court_keypoints.py",
                    },
                }
            ]
        },
        "frames": {
            "available_review_frame_count": 1,
            "frame_count": 1,
            "frame_dir": _path_text(frames_dir),
            "label_coordinate_space": [image_size[0], image_size[1]],
            "sample_every_frames": 1,
            "source_resolution": [image_size[0], image_size[1]],
        },
        "review": {
            "status": "reviewed",
            "reviewer": "synthetic_geometry_generator",
            "reviewed_at_utc": generated_at_utc,
            "human_reviewed": False,
            "independent_reviewed_count": 0,
            "static_camera_copy_count": 0,
            "synthetic_count": 1,
        },
        "provenance": {
            "synthetic": True,
            "human_labels": False,
            "seed": seed,
            "sample_index": sample_index,
            "coordinate_frame": COORDINATE_FRAME,
            "generator": "scripts/racketsport/generate_synthetic_court_keypoints.py",
            "note": "Synthetic court geometry labels are training augmentation only, not CAL gate evidence.",
            "net_keypoint_height_convention": NET_KEYPOINT_HEIGHT_CONVENTION,
        },
        "generation": generation,
    }


def _generation_payload(
    *,
    pose: CameraPose,
    homography: list[list[float]],
    colors: dict[str, Any],
    line_width: int,
    line_wear: float,
    jpeg_quality: int,
    occlusion_count: int,
) -> dict[str, Any]:
    return {
        "camera": {
            "position_m": list(pose.position_m),
            "target_m": list(pose.target_m),
            "height_m": pose.height_m,
            "distance_m": pose.distance_m,
            "azimuth_deg": pose.azimuth_deg,
            "tilt_deg": pose.tilt_deg,
            "roll_deg": pose.roll_deg,
            "focal_mm_eq": pose.focal_mm_eq,
            "fx_px": pose.fx_px,
            "fy_px": pose.fy_px,
            "cx_px": pose.cx_px,
            "cy_px": pose.cy_px,
            "distortion_k1": pose.distortion_k1,
        },
        "world_to_image_homography": homography,
        "net_keypoint_height_convention": NET_KEYPOINT_HEIGHT_CONVENTION,
        "court_template": _court_template_manifest(),
        "keypoint_world_xyz_m": {
            point.name: [float(value) for value in point.world_xyz_m]
            for point in PICKLEBALL_KEYPOINTS
        },
        "domain_randomization": {
            "court_color_rgb": list(colors["court"]),
            "floor_base_rgb": list(colors["floor_base"]),
            "line_color_rgb": list(colors["line"]),
            "line_width_px": line_width,
            "line_wear_probability": line_wear,
            "occlusion_count": occlusion_count,
            "jpeg_quality": jpeg_quality,
            "features": [
                "tripod_camera_pose",
                "court_and_floor_color_jitter",
                "line_width_color_wear",
                "lighting_gradient",
                "soft_shadows",
                "background_clutter",
                "mild_radial_lens_distortion_k1",
                "sensor_noise",
                "jpeg_artifacts",
                "partial_line_occlusions",
                "regulation_top_net_keypoints",
            ],
        },
    }


def _court_template_manifest() -> dict[str, Any]:
    template = get_court_template("pickleball")
    return {
        "sport": template.sport,
        "coordinate_frame": template.coordinate_frame,
        "length_ft": template.length_ft,
        "width_ft": template.width_ft,
        "net_width_ft": template.net_width_ft,
        "non_volley_zone_ft": template.non_volley_zone_ft,
        "center_net_height_in": template.center_net_height_in,
        "post_net_height_in": template.post_net_height_in,
    }


def _net_top_segment_m(template: Any) -> tuple[Vector3, Vector3]:
    return (
        (-template.half_width_ft * 0.3048, 0.0, template.post_net_height_m),
        (template.half_width_ft * 0.3048, 0.0, template.post_net_height_m),
    )


def _sample_colors(rng: random.Random) -> dict[str, tuple[int, int, int]]:
    court_palettes = [
        ((60, 126, 106), (37, 92, 79)),
        ((42, 106, 153), (28, 78, 112)),
        ((92, 143, 68), (58, 102, 56)),
        ((146, 92, 76), (106, 76, 63)),
        ((79, 116, 133), (54, 82, 96)),
    ]
    court_base, court_alt = rng.choice(court_palettes)
    court = _jitter_color(court_base, rng, 24)
    floor = _jitter_color(court_alt, rng, 32)
    line = _jitter_color((rng.randint(205, 250), rng.randint(205, 250), rng.randint(195, 250)), rng, 12)
    return {"court": court, "floor_base": floor, "line": line}


def _jitter_color(color: tuple[int, int, int], rng: random.Random, amount: int) -> tuple[int, int, int]:
    return tuple(_clamp_color(channel + rng.randint(-amount, amount)) for channel in color)


def _clamp_color(value: int) -> int:
    return max(0, min(255, int(value)))


def _sample_line(start: Sequence[float], end: Sequence[float], count: int) -> list[Vector3]:
    points: list[Vector3] = []
    for idx in range(count):
        t = idx / float(count - 1)
        points.append(
            (
                float(start[0]) * (1.0 - t) + float(end[0]) * t,
                float(start[1]) * (1.0 - t) + float(end[1]) * t,
                float(start[2]) * (1.0 - t) + float(end[2]) * t,
            )
        )
    return points


def _spot_check_indices(count: int, spot_check_count: int, seed: int) -> list[int]:
    if spot_check_count <= 0 or count <= 0:
        return []
    rng = random.Random(seed ^ 0x5F3759DF)
    return sorted(rng.sample(range(count), min(count, spot_check_count)))


def _prepare_output_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and not overwrite:
        generated = [path for path in out_dir.glob("synthetic_court_*") if path.is_dir()]
        if generated or (out_dir / "manifest.json").exists():
            raise ValueError(f"output directory already contains generated synthetic data: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in out_dir.glob("synthetic_court_*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        if (out_dir / "spot_check_overlays").exists():
            shutil.rmtree(out_dir / "spot_check_overlays")
        if (out_dir / "manifest.json").exists():
            (out_dir / "manifest.json").unlink()


def _validate_config(config: SyntheticCourtGenerationConfig) -> None:
    if config.count <= 0:
        raise ValueError("count must be positive")
    width, height = config.image_size
    if width < 160 or height < 90:
        raise ValueError("image_size must be at least 160x90")
    if config.spot_check_count < 0:
        raise ValueError("spot_check_count must be non-negative")
    _validate_range(config.height_m_range, "height_m_range", minimum=1.0, maximum=8.0)
    _validate_range(config.distance_m_range, "distance_m_range", minimum=2.0)
    _validate_range(config.azimuth_deg_range, "azimuth_deg_range", minimum=-60.0, maximum=60.0)
    _validate_range(config.tilt_deg_range, "tilt_deg_range", minimum=0.0, maximum=89.0)
    _validate_range(config.focal_mm_eq_range, "focal_mm_eq_range", minimum=10.0)
    _validate_range(config.roll_deg_range, "roll_deg_range")
    _validate_range(config.distortion_k1_range, "distortion_k1_range")
    if config.jpeg_quality_range[0] < 1 or config.jpeg_quality_range[1] > 100:
        raise ValueError("jpeg_quality_range must stay inside [1, 100]")
    if config.jpeg_quality_range[0] > config.jpeg_quality_range[1]:
        raise ValueError("jpeg_quality_range min must be <= max")
    if config.line_width_px_range[0] <= 0 or config.line_width_px_range[0] > config.line_width_px_range[1]:
        raise ValueError("line_width_px_range must be positive and ordered")


def _validate_range(
    value: tuple[float, float],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if len(value) != 2 or value[0] > value[1]:
        raise ValueError(f"{name} must be a two-value ordered range")
    if minimum is not None and value[0] < minimum:
        raise ValueError(f"{name} minimum must be >= {minimum}")
    if maximum is not None and value[1] > maximum:
        raise ValueError(f"{name} maximum must be <= {maximum}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_text(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: Vector3, scale: float) -> Vector3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _normalize(value: Vector3) -> Vector3:
    norm = math.sqrt(_dot(value, value))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic pickleball court keypoint labels.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--spot-check-count", type=int, default=DEFAULT_SPOT_CHECK_COUNT)
    parser.add_argument("--generated-at-utc", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = generate_synthetic_court_corpus(
        SyntheticCourtGenerationConfig(
            out_dir=args.out,
            count=args.count,
            seed=args.seed,
            image_size=(args.image_width, args.image_height),
            spot_check_count=args.spot_check_count,
            generated_at_utc=args.generated_at_utc,
            overwrite=args.overwrite,
        )
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "manifest": str(args.out / "manifest.json"),
                "spot_check_overlays": len(manifest["spot_check_overlays"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
