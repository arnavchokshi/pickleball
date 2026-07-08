"""Domain-randomized procedural court-scene rendering engine (CAL-SYNTH v2, 2026-07-05).

Shared core behind:
  - ``scripts/racketsport/generate_synthetic_court_keypoints.py`` (disk corpus writer, CLI
    backward-compatible).
  - ``threed/racketsport/court_synth_stream.py`` (zero-disk streaming API for the trainer,
    STABLE CONTRACT for CAL-MODEL).

Every pixel is procedurally rendered from regulation court geometry (``court_templates.py``) plus
randomized camera/color/occluder parameters -- this module never reads ``eval_clips/**`` or any
other real-frame content. PIL + numpy only; no torch import (the trainer converts).

Scenario families (>=7, mixture-weighted, see ``SCENARIO_NAMES``):
  dedicated_indoor, dedicated_outdoor, tennis_overlay, adjacent_multi_court, portrait_phone,
  harsh_shadow, portable_net_clutter.

Every rendered sample carries a canonical 15-keypoint pickleball layout (``PICKLEBALL_KEYPOINTS``
order) whose world coordinates are always the *primary* court instance, centered at the regulation
net-center origin regardless of scenario (extra tennis/adjacent-court geometry is translated
relative to it). ``render_synthetic_court_sample`` emits, alongside the image, a ``meta`` dict
whose ``homography`` + ``distortion`` fields fully determine the pinhole+distortion projection used
to generate every keypoint -- ``reproject_canonical_keypoints`` replays that exact pipeline, so
self-consistency versus the emitted ``keypoints_xy`` is exact to floating-point precision (<< the
0.5px acceptance bar), including the elevated net keypoints (z > 0) and nonzero distortion samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence

from threed.racketsport.court_calibration import homography_from_planar_points
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_templates import CourtTemplate, get_court_template

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]

# ---------------------------------------------------------------------------------------------
# Stable vocab (CAL-MODEL depends on these exact names/ints -- do not renumber without a version
# bump communicated in the streaming contract docstring).
# ---------------------------------------------------------------------------------------------

SCENARIO_NAMES: tuple[str, ...] = (
    "dedicated_indoor",
    "dedicated_outdoor",
    "tennis_overlay",
    "adjacent_multi_court",
    "portrait_phone",
    "harsh_shadow",
    "portable_net_clutter",
)

LINE_FAMILY_CLASSES: dict[str, int] = {"other": 0, "pickleball_line": 1, "tennis_line": 2, "net": 3}
SURFACE_CLASSES: dict[str, int] = {"background": 0, "apron": 1, "interior": 2}
KEYPOINT_VIS_CLASSES: dict[str, int] = {"off_frame": 0, "occluded": 1, "visible": 2}

CANONICAL_KEYPOINT_NAMES: tuple[str, ...] = tuple(point.name for point in PICKLEBALL_KEYPOINTS)
_KEYPOINT_WORLD_XYZ_M: dict[str, Vector3] = {point.name: point.world_xyz_m for point in PICKLEBALL_KEYPOINTS}
_PRIMARY_CORNER_NAMES = ("near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner")


# ---------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneRenderConfig:
    image_size: tuple[int, int] = (640, 360)
    height_m_range: tuple[float, float] = (1.0, 12.0)
    distance_m_range: tuple[float, float] = (5.0, 40.0)
    azimuth_deg_range: tuple[float, float] = (-75.0, 75.0)
    tilt_deg_range: tuple[float, float] = (2.0, 80.0)
    roll_deg_range: tuple[float, float] = (-4.0, 4.0)
    focal_px_range: tuple[float, float] = (500.0, 2000.0)
    distortion_k1_range: tuple[float, float] = (-0.07, 0.04)
    distortion_p_range: tuple[float, float] = (0.0, 0.0)
    jpeg_quality_range: tuple[int, int] = (78, 96)
    line_width_px_range: tuple[int, int] = (2, 7)
    scenario_weights: dict[str, float] = field(default_factory=lambda: {name: 1.0 for name in SCENARIO_NAMES})

    def weight_for(self, scenario: str) -> float:
        return float(self.scenario_weights.get(scenario, 0.0))


# Per-scenario range overrides layered on top of the base ``SceneRenderConfig`` (only keys present
# here are overridden; everything else is inherited). Keeps the "REAL eval geometry" pinned ranges
# (height 1-12m, distance 5-40m, azimuth -75..75deg, focal 500-2000px eq) as the union envelope
# while biasing each family toward its characteristic viewpoints.
_SCENARIO_RANGE_OVERRIDES: dict[str, dict[str, Any]] = {
    "dedicated_indoor": {
        "height_m_range": (1.5, 8.0),
        "distance_m_range": (6.0, 24.0),
    },
    "dedicated_outdoor": {
        "height_m_range": (1.0, 10.0),
        "distance_m_range": (6.0, 34.0),
        "tilt_deg_range": (2.0, 60.0),
    },
    "tennis_overlay": {
        "height_m_range": (2.0, 12.0),
        "distance_m_range": (10.0, 40.0),
    },
    "adjacent_multi_court": {
        "height_m_range": (1.0, 9.0),
        "distance_m_range": (8.0, 40.0),
        "azimuth_deg_range": (-70.0, 70.0),
    },
    "portrait_phone": {
        "image_size": (360, 640),
        "height_m_range": (1.0, 3.2),
        "distance_m_range": (5.0, 26.0),
        "focal_px_range": (450.0, 1300.0),
        "roll_deg_range": (-10.0, 10.0),
        "distortion_k1_range": (-0.11, 0.06),
        "distortion_p_range": (-0.015, 0.015),
    },
    "harsh_shadow": {
        "height_m_range": (1.0, 7.0),
        "distance_m_range": (5.0, 22.0),
    },
    "portable_net_clutter": {
        "height_m_range": (1.0, 6.0),
        "distance_m_range": (5.0, 20.0),
    },
}

# Bbox-fraction visibility gate for the primary court's 15 keypoints: wider/zoomed-out families get
# a looser minimum so extra court/tennis geometry can also fit in frame.
_SCENARIO_BBOX_MIN_FRACTION: dict[str, tuple[float, float]] = {
    "tennis_overlay": (0.16, 0.12),
    "adjacent_multi_court": (0.14, 0.14),
    "portrait_phone": (0.20, 0.16),
}
_DEFAULT_BBOX_MIN_FRACTION = (0.30, 0.20)

# Minimum number of the 15 canonical keypoints that must land strictly inside the frame for a pose
# to be accepted. Handheld portrait shots realistically crop a corner or two -- that's exactly the
# off-frame visibility case the corpus needs to cover -- so it gets a looser minimum than the
# tripod-style families, which keep the full court in frame.
_SCENARIO_MIN_VISIBLE_KEYPOINTS: dict[str, int] = {
    "portrait_phone": 9,
}
_DEFAULT_MIN_VISIBLE_KEYPOINTS = 15


def resolve_scenario_config(
    base: SceneRenderConfig, scenario: str, *, force_image_size: tuple[int, int] | None = None
) -> SceneRenderConfig:
    """Layer per-scenario range overrides onto ``base``.

    ``force_image_size``, when given, wins over both ``base`` and any per-scenario override (e.g.
    portrait_phone's default 9:16 canvas) -- lets a streaming caller keep every sample a single
    fixed shape for dataloader batching.
    """

    overrides = dict(_SCENARIO_RANGE_OVERRIDES.get(scenario, {}))
    if force_image_size is not None:
        overrides.pop("image_size", None)
    if not overrides and force_image_size is None:
        return base
    fields = {
        "image_size": base.image_size,
        "height_m_range": base.height_m_range,
        "distance_m_range": base.distance_m_range,
        "azimuth_deg_range": base.azimuth_deg_range,
        "tilt_deg_range": base.tilt_deg_range,
        "roll_deg_range": base.roll_deg_range,
        "focal_px_range": base.focal_px_range,
        "distortion_k1_range": base.distortion_k1_range,
        "distortion_p_range": base.distortion_p_range,
        "jpeg_quality_range": base.jpeg_quality_range,
        "line_width_px_range": base.line_width_px_range,
        "scenario_weights": base.scenario_weights,
    }
    fields.update(overrides)
    if force_image_size is not None:
        fields["image_size"] = tuple(force_image_size)
    return SceneRenderConfig(**fields)


def choose_scenario(rng: random.Random, weights: dict[str, float]) -> str:
    names = [name for name in SCENARIO_NAMES if weights.get(name, 0.0) > 0.0]
    if not names:
        names = list(SCENARIO_NAMES)
    picked = rng.choices(names, weights=[weights.get(name, 1.0) for name in names], k=1)
    return picked[0]


# ---------------------------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------------------------


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


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


def _clamp_color(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _jitter_color(color: tuple[int, int, int], rng: random.Random, amount: int) -> tuple[int, int, int]:
    return tuple(_clamp_color(channel + rng.randint(-amount, amount)) for channel in color)  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------------
# Camera pose + projection (single source of truth: used for keypoints, lines, surfaces, masks)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraPose:
    position_m: Vector3
    right: Vector3
    down: Vector3
    forward: Vector3
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    distortion_k1: float = 0.0
    distortion_p1: float = 0.0
    distortion_p2: float = 0.0
    height_m: float = 0.0
    distance_m: float = 0.0
    azimuth_deg: float = 0.0
    tilt_deg: float = 0.0
    roll_deg: float = 0.0
    focal_px: float = 0.0


def project_world_point(point_xyz_m: Sequence[float], pose: CameraPose) -> Vector2:
    """Pinhole + Brown-Conrady (radial k1 + tangential p1/p2) projection, camera-plane normalized.

    This is the exact forward function used to generate every emitted keypoint/line/mask pixel;
    ``reproject_canonical_keypoints`` replays it from ``meta`` alone to prove self-consistency.
    """

    rel = _sub((float(point_xyz_m[0]), float(point_xyz_m[1]), float(point_xyz_m[2])), pose.position_m)
    cam_x = _dot(rel, pose.right)
    cam_y = _dot(rel, pose.down)
    cam_z = _dot(rel, pose.forward)
    if cam_z <= 0.05:
        raise ValueError("world point is behind sampled camera")
    x = cam_x / cam_z
    y = cam_y / cam_z
    r2 = x * x + y * y
    radial = 1.0 + pose.distortion_k1 * r2
    p1, p2 = pose.distortion_p1, pose.distortion_p2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return (pose.fx_px * x_d + pose.cx_px, pose.fy_px * y_d + pose.cy_px)


def _sample_camera_pose(config: SceneRenderConfig, rng: random.Random) -> CameraPose:
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
        rng.uniform(-0.9, 0.9),
        rng.uniform(-2.2, 2.2),
        rng.uniform(0.0, 0.5),
    )
    position = (camera_x, camera_y, camera_height)
    forward = _normalize(_sub(target, position))
    horizontal_distance = math.hypot(target[0] - position[0], target[1] - position[1])
    tilt_deg = math.degrees(math.atan2(position[2] - target[2], max(horizontal_distance, 1e-6)))
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

    focal_px = rng.uniform(*config.focal_px_range)
    cx = width / 2.0 + rng.uniform(-0.08 * width, 0.08 * width)
    cy = height_px / 2.0 + rng.uniform(-0.05 * height_px, 0.10 * height_px)
    distortion_k1 = rng.uniform(*config.distortion_k1_range)
    distortion_p1 = rng.uniform(*config.distortion_p_range)
    distortion_p2 = rng.uniform(*config.distortion_p_range)
    return CameraPose(
        position_m=position,
        right=rolled_right,
        down=down,
        forward=forward,
        fx_px=focal_px,
        fy_px=focal_px,
        cx_px=cx,
        cy_px=cy,
        distortion_k1=distortion_k1,
        distortion_p1=distortion_p1,
        distortion_p2=distortion_p2,
        height_m=camera_height,
        distance_m=distance,
        azimuth_deg=azimuth_deg,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
        focal_px=focal_px,
    )


def _is_visible_projection(projected: dict[str, Vector2], *, width: int, height: int, scenario: str) -> bool:
    in_frame = [
        (x, y) for x, y in projected.values() if 4.0 <= x <= width - 4.0 and 4.0 <= y <= height - 4.0
    ]
    min_count = _SCENARIO_MIN_VISIBLE_KEYPOINTS.get(scenario, _DEFAULT_MIN_VISIBLE_KEYPOINTS)
    if len(in_frame) < min_count:
        return False
    xs = [point[0] for point in in_frame]
    ys = [point[1] for point in in_frame]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    min_w_frac, min_h_frac = _SCENARIO_BBOX_MIN_FRACTION.get(scenario, _DEFAULT_BBOX_MIN_FRACTION)
    if bbox_w < width * min_w_frac or bbox_h < height * min_h_frac:
        return False
    return True


def sample_visible_camera_pose(
    config: SceneRenderConfig, rng: random.Random, *, scenario: str
) -> tuple[CameraPose, dict[str, Vector2]]:
    width, height = config.image_size
    for _ in range(1200):
        pose = _sample_camera_pose(config, rng)
        try:
            projected = {name: project_world_point(xyz, pose) for name, xyz in _KEYPOINT_WORLD_XYZ_M.items()}
        except ValueError:
            continue
        if _is_visible_projection(projected, width=width, height=height, scenario=scenario):
            return pose, projected
    raise RuntimeError(f"could not sample a visible synthetic court camera pose for scenario={scenario!r}")


def _safe_project(point_xyz_m: Sequence[float], pose: CameraPose) -> Vector2 | None:
    """``project_world_point`` but returns ``None`` instead of raising for behind-camera points.

    Only the *primary* court's 15 canonical keypoints are guaranteed in front of the camera (by
    ``sample_visible_camera_pose``'s retry loop). Secondary geometry -- extra adjacent courts, the
    apron, portable-net-stand hardware -- can legitimately fall (partially) behind a wide-azimuth
    or close-distance camera; skipping those points/segments is the correct behavior, not a bug.
    """

    try:
        return project_world_point(point_xyz_m, pose)
    except ValueError:
        return None


def reproject_canonical_keypoints(meta: dict[str, Any]) -> dict[str, Vector2]:
    """Replay the exact forward projection from ``meta`` alone (self-consistency proof)."""

    homography = meta["homography"]
    distortion = meta["distortion"]
    pose = CameraPose(
        position_m=tuple(homography["camera_position_m"]),  # type: ignore[arg-type]
        right=tuple(homography["camera_right"]),  # type: ignore[arg-type]
        down=tuple(homography["camera_down"]),  # type: ignore[arg-type]
        forward=tuple(homography["camera_forward"]),  # type: ignore[arg-type]
        fx_px=homography["fx_px"],
        fy_px=homography["fy_px"],
        cx_px=homography["cx_px"],
        cy_px=homography["cy_px"],
        distortion_k1=distortion["k1"],
        distortion_p1=distortion.get("p1", 0.0),
        distortion_p2=distortion.get("p2", 0.0),
    )
    return {name: project_world_point(xyz, pose) for name, xyz in _KEYPOINT_WORLD_XYZ_M.items()}


# ---------------------------------------------------------------------------------------------
# Court layout (scenario -> list of court instances)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CourtInstance:
    sport: str
    offset_m: Vector2
    line_family: str
    is_primary: bool
    wear: float
    kitchen_two_tone: bool
    fill_interior: bool = True
    line_width_scale: float = 1.0


def build_court_instances(scenario: str, rng: random.Random) -> list[CourtInstance]:
    if scenario == "tennis_overlay":
        return [
            CourtInstance(
                sport="tennis",
                offset_m=(0.0, 0.0),
                line_family="tennis_line",
                is_primary=False,
                wear=rng.uniform(0.18, 0.5),
                kitchen_two_tone=False,
                fill_interior=True,
                line_width_scale=0.7,
            ),
            CourtInstance(
                sport="pickleball",
                offset_m=(0.0, 0.0),
                line_family="pickleball_line",
                is_primary=True,
                wear=rng.uniform(0.0, 0.12),
                kitchen_two_tone=rng.random() < 0.35,
                fill_interior=False,  # same physical surface as the tennis instance above -- only
                                      # its (fresher, wider) lines get painted on top of it.
                line_width_scale=1.35,
            ),
        ]
    if scenario == "adjacent_multi_court":
        template = get_court_template("pickleball")
        count = rng.randint(2, 4)
        primary_index = rng.randrange(count)
        gap_m = rng.uniform(1.0, 3.2)
        span = template.width_m + gap_m
        instances = []
        for idx in range(count):
            instances.append(
                CourtInstance(
                    sport="pickleball",
                    offset_m=((idx - primary_index) * span, 0.0),
                    line_family="pickleball_line",
                    is_primary=(idx == primary_index),
                    wear=rng.uniform(0.0, 0.28),
                    kitchen_two_tone=rng.random() < 0.3,
                )
            )
        return instances
    return [
        CourtInstance(
            sport="pickleball",
            offset_m=(0.0, 0.0),
            line_family="pickleball_line",
            is_primary=True,
            wear=rng.uniform(0.0, 0.26),
            kitchen_two_tone=rng.random() < 0.3,
        )
    ]


def _offset_point(point_m: Sequence[float], offset_m: Vector2) -> Vector3:
    return (float(point_m[0]) + offset_m[0], float(point_m[1]) + offset_m[1], float(point_m[2]))


def _instance_corners_m(instance: CourtInstance) -> list[Vector3]:
    template = get_court_template(instance.sport)  # type: ignore[arg-type]
    return [_offset_point(corner, instance.offset_m) for corner in template.corners_m]


def _instance_line_segments_m(instance: CourtInstance) -> dict[str, tuple[Vector3, Vector3]]:
    template = get_court_template(instance.sport)  # type: ignore[arg-type]
    segments = {}
    for name, (start, end) in template.line_segments_m.items():
        if name == "net":
            start, end = _net_top_segment_m(template)
        segments[name] = (_offset_point(start, instance.offset_m), _offset_point(end, instance.offset_m))
    return segments


def _net_top_segment_m(template: CourtTemplate) -> tuple[Vector3, Vector3]:
    return (
        (-template.half_net_width_ft * 0.3048, 0.0, template.post_net_height_m),
        (template.half_net_width_ft * 0.3048, 0.0, template.post_net_height_m),
    )


def _kitchen_strip_polygon_m(instance: CourtInstance) -> list[Vector3] | None:
    template = get_court_template(instance.sport)
    if template.non_volley_zone_ft is None:
        return None
    half_w = template.half_width_ft
    nvz = template.non_volley_zone_ft
    corners_ft = [(-half_w, -nvz), (half_w, -nvz), (half_w, nvz), (-half_w, nvz)]
    return [
        _offset_point((x * 0.3048, y * 0.3048, 0.0), instance.offset_m) for x, y in corners_ft
    ]


def _sample_line_points(start: Vector3, end: Vector3, count: int) -> list[Vector3]:
    points: list[Vector3] = []
    for idx in range(count):
        t = idx / float(count - 1)
        points.append(
            (
                start[0] * (1.0 - t) + end[0] * t,
                start[1] * (1.0 - t) + end[1] * t,
                start[2] * (1.0 - t) + end[2] * t,
            )
        )
    return points


# ---------------------------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------------------------

_COURT_PALETTES: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((42, 106, 153), (28, 78, 112)),  # blue
    ((60, 126, 106), (37, 92, 79)),  # green
    ((92, 143, 68), (58, 102, 56)),  # green (outdoor)
    ((104, 74, 138), (74, 52, 104)),  # purple
    ((146, 92, 76), (106, 76, 63)),  # terracotta / red
    ((79, 116, 133), (54, 82, 96)),  # slate
)

_KITCHEN_ACCENT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (196, 168, 88),
    (200, 96, 88),
    (88, 150, 196),
    (222, 214, 200),
)


def _sample_palette(rng: random.Random, scenario: str) -> dict[str, Any]:
    court_base, court_alt = rng.choice(_COURT_PALETTES)
    court = _jitter_color(court_base, rng, 22)
    apron = _jitter_color(court_alt, rng, 26)
    kitchen = _jitter_color(rng.choice(_KITCHEN_ACCENT_PALETTE), rng, 18)
    line_base = (rng.randint(225, 255), rng.randint(225, 255), rng.randint(215, 255))
    pickleball_line = _jitter_color(line_base, rng, 8)
    tennis_line_base = (rng.randint(205, 235), rng.randint(185, 210), rng.randint(90, 135))
    tennis_line = _jitter_color(tennis_line_base, rng, 12)
    if scenario in {"dedicated_indoor", "portable_net_clutter"}:
        floor = _jitter_color((176, 150, 108), rng, 20)  # gym wood tone
    else:
        floor = _jitter_color((86, 118, 70), rng, 26)  # grass/ground tone
    return {
        "court": court,
        "apron": apron,
        "kitchen": kitchen,
        "pickleball_line": pickleball_line,
        "tennis_line": tennis_line,
        "floor_base": floor,
    }


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------


@dataclass
class RenderedScene:
    image: Any  # PIL.Image, RGB
    keypoints_xy: dict[str, Vector2]
    keypoints_vis: dict[str, int]
    line_family_mask: Any  # np.ndarray HxW uint8
    surface_mask: Any  # np.ndarray HxW uint8
    meta: dict[str, Any]
    scenario: str


def _draw_environment_backdrop(draw: Any, width: int, height: int, rng: random.Random, scenario: str) -> None:
    outdoor = scenario in {
        "dedicated_outdoor",
        "tennis_overlay",
        "adjacent_multi_court",
        "harsh_shadow",
        "portrait_phone",
    }
    horizon = rng.uniform(0.22, 0.42) * height
    if outdoor:
        sky_top = _jitter_color((146, 178, 205), rng, 16)
        sky_bottom = _jitter_color((198, 210, 216), rng, 14)
        for y in range(0, int(horizon), 2):
            t = y / max(horizon, 1.0)
            color = tuple(int(sky_top[c] * (1 - t) + sky_bottom[c] * t) for c in range(3))
            draw.line([(0, y), (width, y)], fill=(*color, 255), width=2)
        ground = _jitter_color((92, 110, 78), rng, 18)
        draw.rectangle([0, horizon, width, height], fill=(*ground, 255))
        if rng.random() < 0.7:
            fence_color = _jitter_color((60, 60, 64), rng, 10)
            draw.line([(0, horizon), (width, horizon)], fill=(*fence_color, 200), width=max(1, int(height * 0.01)))
        for _ in range(rng.randint(0, 4)):
            bush_color = _jitter_color((54, 92, 52), rng, 16)
            bx = rng.uniform(0, width)
            by = horizon - rng.uniform(0.0, 0.04 * height)
            bw = rng.uniform(0.05, 0.14) * width
            bh = rng.uniform(0.03, 0.09) * height
            draw.ellipse([bx - bw / 2, by - bh, bx + bw / 2, by], fill=(*bush_color, 235))
    else:
        wall_color = _jitter_color((150, 150, 156), rng, 14)
        draw.rectangle([0, 0, width, horizon], fill=(*wall_color, 255))
        floor_color = _jitter_color((120, 108, 92), rng, 16)
        draw.rectangle([0, horizon, width, height], fill=(*floor_color, 255))


def _draw_gear_clutter(draw: Any, width: int, height: int, rng: random.Random, count: int) -> None:
    for _ in range(count):
        cx = rng.uniform(0.05 * width, 0.95 * width)
        cy = rng.uniform(0.55 * height, 0.97 * height)
        if rng.random() < 0.5:
            # duffel bag: rounded rectangle body + small handle ellipse.
            bw = rng.uniform(0.05, 0.11) * width
            bh = rng.uniform(0.03, 0.06) * height
            color = _jitter_color((70, 66, 60), rng, 24)
            draw.rounded_rectangle(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], radius=bh * 0.3, fill=(*color, 235)
            )
            draw.ellipse([cx - bw * 0.18, cy - bh * 0.85, cx + bw * 0.18, cy - bh * 0.35], outline=(*color, 235), width=2)
        else:
            # folding chair: backrest rectangle + seat rectangle.
            bw = rng.uniform(0.035, 0.07) * width
            bh = rng.uniform(0.06, 0.12) * height
            color = _jitter_color((90, 96, 100), rng, 22)
            draw.rectangle([cx - bw / 2, cy - bh, cx + bw / 2, cy - bh * 0.35], fill=(*color, 220))
            draw.rectangle([cx - bw / 2, cy - bh * 0.35, cx + bw / 2, cy], fill=(*color, 235))


def _draw_person_silhouette(
    draw: Any, cx: float, cy_feet: float, scale: float, color: tuple[int, int, int], rng: random.Random
) -> list[Vector2]:
    head_r = 0.09 * scale
    body_w = 0.34 * scale
    body_h = 0.62 * scale
    head_cy = cy_feet - body_h - head_r
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=(*color, 235))
    body_top = head_cy + head_r * 0.6
    draw.rounded_rectangle(
        [cx - body_w / 2, body_top, cx + body_w / 2, cy_feet], radius=body_w * 0.28, fill=(*color, 235)
    )
    polygon = [
        (cx - head_r, head_cy - head_r),
        (cx + head_r, head_cy - head_r),
        (cx + body_w / 2, body_top),
        (cx + body_w / 2, cy_feet),
        (cx - body_w / 2, cy_feet),
        (cx - body_w / 2, body_top),
    ]
    return polygon


def _point_in_polygon(x: float, y: float, polygon: Sequence[Vector2]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def render_synthetic_court_sample(
    rng: random.Random,
    config: SceneRenderConfig,
    *,
    scenario: str | None = None,
    apply_jpeg_roundtrip: bool = False,
    force_image_size: tuple[int, int] | None = None,
) -> RenderedScene:
    from io import BytesIO

    import numpy as np
    from PIL import Image, ImageDraw

    scenario = scenario or choose_scenario(rng, config.scenario_weights)
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"unknown synthetic court scenario: {scenario!r}")
    scene_config = resolve_scenario_config(config, scenario, force_image_size=force_image_size)
    width, height = scene_config.image_size

    pose, keypoints_xy = sample_visible_camera_pose(scene_config, rng, scenario=scenario)
    instances = build_court_instances(scenario, rng)
    primary = next(instance for instance in instances if instance.is_primary)
    palette = _sample_palette(rng, scenario)
    line_width = rng.randint(*scene_config.line_width_px_range)

    image = Image.new("RGB", (width, height), palette["floor_base"])
    draw = ImageDraw.Draw(image, "RGBA")
    line_mask_img = Image.new("L", (width, height), LINE_FAMILY_CLASSES["other"])
    line_mask_draw = ImageDraw.Draw(line_mask_img)
    surface_mask_img = Image.new("L", (width, height), SURFACE_CLASSES["background"])
    surface_mask_draw = ImageDraw.Draw(surface_mask_img)

    _draw_environment_backdrop(draw, width, height, rng, scenario)

    # -- surfaces (apron behind, interiors on top) -----------------------------------------
    apron_polygon_px = _apron_polygon_px(instances, pose, margin_m=rng.uniform(2.0, 5.5))
    if apron_polygon_px is not None:
        draw.polygon(apron_polygon_px, fill=(*palette["apron"], 255))
        surface_mask_draw.polygon(apron_polygon_px, fill=SURFACE_CLASSES["apron"])

    for instance in instances:
        if not instance.fill_interior:
            continue
        corners_px = [_safe_project(corner, pose) for corner in _instance_corners_m(instance)]
        if any(corner is None for corner in corners_px):
            continue  # this instance's footprint falls (partially) behind the camera; skip it
        interior_color = _jitter_color(palette["court"], rng, 10) if instance.sport == "pickleball" else palette["court"]
        draw.polygon(corners_px, fill=(*interior_color, 250))
        surface_mask_draw.polygon(corners_px, fill=SURFACE_CLASSES["interior"])
    for instance in instances:
        if instance.kitchen_two_tone:
            kitchen_m = _kitchen_strip_polygon_m(instance)
            if kitchen_m is not None:
                kitchen_px = [_safe_project(point, pose) for point in kitchen_m]
                if any(point is None for point in kitchen_px):
                    continue
                draw.polygon(kitchen_px, fill=(*palette["kitchen"], 235))
                surface_mask_draw.polygon(kitchen_px, fill=SURFACE_CLASSES["interior"])

    # -- lines (tennis drawn first so pickleball's fresher paint sits visibly on top) --------
    for instance in sorted(instances, key=lambda inst: inst.line_family == "pickleball_line"):
        color = palette["pickleball_line"] if instance.line_family == "pickleball_line" else palette["tennis_line"]
        mask_class = LINE_FAMILY_CLASSES[instance.line_family]
        instance_line_width = max(1, round(line_width * instance.line_width_scale))
        for name, (start, end) in sorted(_instance_line_segments_m(instance).items()):
            samples = _sample_line_points(start, end, 48)
            projected = [_safe_project(point, pose) for point in samples]
            is_net = name == "net"
            draw_color = color if not is_net else _jitter_color((235, 235, 238), rng, 8)
            draw_family = mask_class if not is_net else LINE_FAMILY_CLASSES["net"]
            draw_width = instance_line_width if not is_net else line_width
            _draw_worn_polyline(draw, projected, draw_color, draw_width, instance.wear, rng)
            _draw_worn_polyline_mask(line_mask_draw, projected, draw_family, draw_width, instance.wear, rng)

    # -- portable net stand (visual clutter + naturally occludes net keypoints) --------------
    occluder_polygons: list[list[Vector2]] = []
    if scenario == "portable_net_clutter":
        occluder_polygons.extend(_draw_portable_net_stand(draw, primary, pose, rng))

    # -- background clutter / gear -------------------------------------------------------------
    clutter_count = rng.randint(2, 5) if scenario == "portable_net_clutter" else rng.randint(0, 3)
    _draw_gear_clutter(draw, width, height, rng, clutter_count)

    # -- occluders (people), each casting a shadow along one shared light direction ------------
    light_dir = _sample_light_direction(rng)
    person_count = rng.randint(1, 4) if scenario != "portable_net_clutter" else rng.randint(1, 3)
    cast_shadow_polygons: list[list[Vector2]] = []
    for _ in range(person_count):
        cx = rng.uniform(0.05 * width, 0.95 * width)
        cy_feet = rng.uniform(0.35 * height, 0.97 * height)
        scale = rng.uniform(0.10, 0.24) * height
        color = _jitter_color((45, 45, 50), rng, 30)
        polygon = _draw_person_silhouette(draw, cx, cy_feet, scale, color, rng)
        occluder_polygons.append(polygon)
        cast_shadow_polygons.append(
            _cast_shadow_polygon(
                cx,
                cy_feet,
                scale,
                light_dir,
                length_scale=rng.uniform(1.2, 2.6),
                half_width_scale=rng.uniform(0.16, 0.26),
            )
        )

    # -- shadows (visual only, no mask effect) -------------------------------------------------
    if scenario == "harsh_shadow":
        _draw_harsh_shadows(image, width, height, rng, keypoints_xy, light_dir, cast_shadow_polygons)
    else:
        _draw_soft_shadows(image, width, height, rng, light_dir, cast_shadow_polygons)

    # -- stamp occluders into masks (modal/visible-only ground truth) -------------------------
    if occluder_polygons:
        occ_draw_line = ImageDraw.Draw(line_mask_img)
        occ_draw_surface = ImageDraw.Draw(surface_mask_img)
        for polygon in occluder_polygons:
            occ_draw_line.polygon(polygon, fill=LINE_FAMILY_CLASSES["other"])
            occ_draw_surface.polygon(polygon, fill=SURFACE_CLASSES["background"])

    image = _apply_lighting_and_sensor_artifacts(image, rng, np)
    jpeg_quality = rng.randint(*scene_config.jpeg_quality_range)
    if apply_jpeg_roundtrip:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=False, progressive=False)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")

    keypoints_vis = _compute_keypoint_visibility(keypoints_xy, occluder_polygons, width=width, height=height)

    ground_plane_3x3 = _fit_ground_plane_homography(primary, pose)
    meta = {
        "scenario": scenario,
        "image_size": [width, height],
        "homography": {
            "camera_position_m": list(pose.position_m),
            "camera_right": list(pose.right),
            "camera_down": list(pose.down),
            "camera_forward": list(pose.forward),
            "fx_px": pose.fx_px,
            "fy_px": pose.fy_px,
            "cx_px": pose.cx_px,
            "cy_px": pose.cy_px,
            "ground_plane_3x3": ground_plane_3x3,
        },
        "distortion": {
            "k1": pose.distortion_k1,
            "k2": 0.0,
            "p1": pose.distortion_p1,
            "p2": pose.distortion_p2,
            "model": "brown_conrady_normalized_camera_plane",
        },
        "camera": {
            "height_m": pose.height_m,
            "distance_m": pose.distance_m,
            "azimuth_deg": pose.azimuth_deg,
            "tilt_deg": pose.tilt_deg,
            "roll_deg": pose.roll_deg,
            "focal_px": pose.focal_px,
        },
        "court_instances": [
            {
                "sport": instance.sport,
                "offset_m": list(instance.offset_m),
                "line_family": instance.line_family,
                "is_primary": instance.is_primary,
                "wear": instance.wear,
                "kitchen_two_tone": instance.kitchen_two_tone,
            }
            for instance in instances
        ],
        "keypoint_names": list(CANONICAL_KEYPOINT_NAMES),
        "keypoint_world_xyz_m": {name: list(xyz) for name, xyz in _KEYPOINT_WORLD_XYZ_M.items()},
        "line_family_classes": dict(LINE_FAMILY_CLASSES),
        "surface_classes": dict(SURFACE_CLASSES),
        "keypoint_vis_classes": dict(KEYPOINT_VIS_CLASSES),
        "jpeg_quality": jpeg_quality,
        "line_width_px": line_width,
        "occluder_count": len(occluder_polygons),
    }

    return RenderedScene(
        image=image,
        keypoints_xy=keypoints_xy,
        keypoints_vis=keypoints_vis,
        line_family_mask=np.asarray(line_mask_img, dtype=np.uint8),
        surface_mask=np.asarray(surface_mask_img, dtype=np.uint8),
        meta=meta,
        scenario=scenario,
    )


def _apron_polygon_px(instances: Sequence[CourtInstance], pose: CameraPose, *, margin_m: float) -> list[Vector2] | None:
    xs_m: list[float] = []
    ys_m: list[float] = []
    for instance in instances:
        template = get_court_template(instance.sport)  # type: ignore[arg-type]
        half_w = template.half_width_ft * 0.3048
        half_l = template.half_length_ft * 0.3048
        xs_m.extend([instance.offset_m[0] - half_w, instance.offset_m[0] + half_w])
        ys_m.extend([instance.offset_m[1] - half_l, instance.offset_m[1] + half_l])
    if not xs_m:
        return None
    x_min, x_max = min(xs_m) - margin_m, max(xs_m) + margin_m
    y_min, y_max = min(ys_m) - margin_m, max(ys_m) + margin_m
    corners_m = [(x_min, y_min, 0.0), (x_max, y_min, 0.0), (x_max, y_max, 0.0), (x_min, y_max, 0.0)]
    try:
        return [project_world_point(corner, pose) for corner in corners_m]
    except ValueError:
        return None


def _draw_worn_polyline(
    draw: Any,
    points: Sequence[Vector2 | None],
    color: tuple[int, int, int],
    width: int,
    wear: float,
    rng: random.Random,
) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-1], points[1:], strict=True):
        if start is None or end is None:
            continue
        if rng.random() < wear:
            continue
        jitter = rng.randint(-16, 12)
        alpha = max(110, min(255, 235 + jitter))
        draw.line([start, end], fill=(*color, alpha), width=width)


def _draw_worn_polyline_mask(
    draw: Any, points: Sequence[Vector2 | None], mask_class: int, width: int, wear: float, rng: random.Random
) -> None:
    if len(points) < 2:
        return
    mask_width = max(1, width)
    for start, end in zip(points[:-1], points[1:], strict=True):
        if start is None or end is None:
            continue
        if rng.random() < wear:
            continue
        draw.line([start, end], fill=mask_class, width=mask_width)


def _draw_portable_net_stand(
    draw: Any, primary: CourtInstance, pose: CameraPose, rng: random.Random
) -> list[list[Vector2]]:
    template = get_court_template("pickleball")
    half_net = template.half_net_width_ft * 0.3048
    post_height = template.post_net_height_m
    polygons: list[list[Vector2]] = []
    pole_color = _jitter_color((40, 40, 44), rng, 14)
    for side in (-1.0, 1.0):
        x = side * half_net + primary.offset_m[0]
        top_m = (x, primary.offset_m[1], post_height)
        base_m = (x, primary.offset_m[1], 0.0)
        wheel_m = (x + side * 0.28, primary.offset_m[1], 0.0)
        try:
            top_px = project_world_point(top_m, pose)
            base_px = project_world_point(base_m, pose)
            wheel_px = project_world_point(wheel_m, pose)
        except ValueError:
            continue
        draw.line([top_px, base_px], fill=(*pole_color, 245), width=4)
        wheel_r = max(2.0, abs(wheel_px[0] - base_px[0]) * 0.9 + 3.0)
        draw.ellipse(
            [wheel_px[0] - wheel_r, wheel_px[1] - wheel_r * 0.6, wheel_px[0] + wheel_r, wheel_px[1] + wheel_r * 0.6],
            fill=(*pole_color, 245),
        )
        pole_half_w = max(2.0, 0.04 * abs(top_px[1] - base_px[1]) + 2.0)
        polygons.append(
            [
                (top_px[0] - pole_half_w, top_px[1]),
                (top_px[0] + pole_half_w, top_px[1]),
                (wheel_px[0] + wheel_r, wheel_px[1] + wheel_r),
                (wheel_px[0] - wheel_r, wheel_px[1] + wheel_r),
            ]
        )
    return polygons


def _sample_light_direction(rng: random.Random) -> Vector2:
    """One shared 2D shadow-cast direction per sample (single light source, not per-shadow noise).

    Biased away from near-vertical so cast shadows read as diagonal streaks crossing the court
    lines rather than tall vertical bars.
    """

    light_azimuth = rng.uniform(0.0, 2.0 * math.pi)
    dir_x = math.cos(light_azimuth)
    dir_y = math.copysign(max(abs(math.sin(light_azimuth)), 0.35), math.sin(light_azimuth) or 1.0) * 0.6
    return (dir_x, dir_y)


def _cast_shadow_polygon(
    cx: float, cy_feet: float, scale: float, light_dir: Vector2, *, length_scale: float, half_width_scale: float
) -> list[Vector2]:
    dir_x, dir_y = light_dir
    length = length_scale * scale
    half_w = half_width_scale * scale
    x1, y1 = cx + dir_x * length, cy_feet + dir_y * length
    perp_x, perp_y = -dir_y, dir_x
    return [
        (cx - perp_x * half_w, cy_feet - perp_y * half_w),
        (cx + perp_x * half_w, cy_feet + perp_y * half_w),
        (x1 + perp_x * half_w * 0.6, y1 + perp_y * half_w * 0.6),
        (x1 - perp_x * half_w * 0.6, y1 - perp_y * half_w * 0.6),
    ]


def _draw_soft_shadows(
    image: Any,
    width: int,
    height: int,
    rng: random.Random,
    light_dir: Vector2,
    cast_polygons: Sequence[Sequence[Vector2]],
) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for polygon in cast_polygons:
        draw.polygon(list(polygon), fill=(0, 0, 0, rng.randint(28, 60)))
    # A couple of ambient structure/tree shadows along the same shared light direction, for
    # scenes with few or no occluders.
    ground_top = 0.4 * height
    dir_x, dir_y = light_dir
    for _ in range(rng.randint(0, 2)):
        x = rng.uniform(-0.1 * width, 0.9 * width)
        y = rng.uniform(ground_top, 0.75 * height)
        length = rng.uniform(0.18 * width, 0.5 * width)
        half_w = rng.uniform(0.05, 0.12) * height
        polygon = _cast_shadow_polygon(x, y, length, (dir_x, dir_y), length_scale=1.0, half_width_scale=half_w / max(length, 1e-6))
        draw.polygon(polygon, fill=(0, 0, 0, rng.randint(16, 44)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=rng.uniform(4.0, 14.0)))
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))


def _draw_harsh_shadows(
    image: Any,
    width: int,
    height: int,
    rng: random.Random,
    keypoints_xy: dict[str, Vector2],
    light_dir: Vector2,
    cast_polygons: Sequence[Sequence[Vector2]],
) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for polygon in cast_polygons:
        draw.polygon(list(polygon), fill=(0, 0, 0, rng.randint(100, 170)))
    dir_x, dir_y = light_dir
    anchors = list(keypoints_xy.values())
    for _ in range(rng.randint(2, 4)):
        ax, ay = rng.choice(anchors)
        length = rng.uniform(0.18, 0.4) * width
        skinny = rng.uniform(0.025, 0.06) * height
        x0, y0 = ax, ay
        x1, y1 = ax + dir_x * length, ay + dir_y * length
        perp_x, perp_y = -dir_y, dir_x
        polygon = [
            (x0 - perp_x * skinny, y0 - perp_y * skinny),
            (x0 + perp_x * skinny, y0 + perp_y * skinny),
            (x1 + perp_x * skinny * 0.55, y1 + perp_y * skinny * 0.55),
            (x1 - perp_x * skinny * 0.55, y1 - perp_y * skinny * 0.55),
        ]
        draw.polygon(polygon, fill=(0, 0, 0, rng.randint(90, 150)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.2, 3.2)))
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))


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


def _compute_keypoint_visibility(
    keypoints_xy: dict[str, Vector2], occluder_polygons: Sequence[Sequence[Vector2]], *, width: int, height: int
) -> dict[str, int]:
    visibility: dict[str, int] = {}
    for name, (x, y) in keypoints_xy.items():
        if x < 0.0 or x > width or y < 0.0 or y > height:
            visibility[name] = KEYPOINT_VIS_CLASSES["off_frame"]
            continue
        occluded = any(_point_in_polygon(x, y, polygon) for polygon in occluder_polygons)
        visibility[name] = KEYPOINT_VIS_CLASSES["occluded"] if occluded else KEYPOINT_VIS_CLASSES["visible"]
    return visibility


def _fit_ground_plane_homography(primary: CourtInstance, pose: CameraPose) -> list[list[float]]:
    """Convenience DLT fit of the primary court's outer corners (world XY, z=0 -> pixel).

    Exact for every z=0 point only when ``distortion.k1 == p1 == p2 == 0`` (a homography cannot
    represent nonlinear lens distortion); kept for legacy consumers / rendering reference. The
    authoritative self-consistency path is ``reproject_canonical_keypoints`` via the full
    ``homography``+``distortion`` camera fields, which holds exactly regardless of distortion.
    """

    corner_points = [
        point for point in PICKLEBALL_KEYPOINTS if point.name in _PRIMARY_CORNER_NAMES
    ]
    world_xy = [
        (point.world_xyz_m[0] + primary.offset_m[0], point.world_xyz_m[1] + primary.offset_m[1])
        for point in corner_points
    ]
    image_xy = [project_world_point((x, y, 0.0), pose) for x, y in world_xy]
    return homography_from_planar_points(world_xy, image_xy)
