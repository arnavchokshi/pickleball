"""Manual-corner court-plane projection and in/out fields for BALL bounces.

Pixel-space contract (Task #20, 2026-07-02)
--------------------------------------------
The court corners in a ``court_corners.json`` sidecar are tapped by a human
against *some* review image, which is not necessarily the same pixel space as
the BALL track / bounce ``contact_xy_img`` points they get projected against
(those are native source-video pixels). A 2026-07-02 audit found that for
every reviewed clip, the review image was a 960x540 downscaled preview while
the source video (and every BALL track) is native 1920x1080 -- so using the
raw tapped corner pixels directly against native-pixel ball contacts silently
misplaces the homography by 2x and was flipping "in" bounces to "out" by
0.9-2.3 m of false margin (see
runs/ball_m5_geometric_uncertainty_20260702T022544Z/geometric_uncertainty_validation_summary.json
and runs/ball_inout_pixelspace_fix_20260702T*Z/).

To make this contract explicit and fail closed instead of silently
mis-scaling:

  * Every ``court_corners.json`` annotation item MUST declare the pixel space
    its corners were tapped in via ``"image_size": [width, height]``.
    ``_manual_corner_item`` raises a clear migration error if this is
    missing -- there is no implicit/legacy fallback.
  * Callers that need the homography/pose expressed in a specific pixel
    space (almost always: whatever pixel space the ball track's
    ``contact_xy_img`` is in) pass ``target_image_size=(width, height)``.
    ``manual_court_projection_from_corners`` / ``manual_court_pose_from_corners``
    then rescale the declared corners onto that target space before solving.
    ``target_image_size=None`` means "use the corners exactly as declared,
    no rescale" -- this is only correct when the caller's points already
    live in the corners' own declared pixel space; callers with a specific
    known target (BALL bounce in/out, 2D bounce detection, CVAT-derived
    labels) always pass it explicitly rather than relying on this default,
    so the default is only exercised by generic/test callers that
    deliberately want the corners' own space.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_inout_uncertainty import (
    CameraPose,
    bounce_geometric_uncertainty_m,
    fixed_override_breakdown,
    physics_constants_manifest,
    solve_manual_corner_camera_pose,
)
from .ball_line_calls import classify_bounce
from .court_calibration import homography_from_planar_points, project_image_points_to_world, project_planar_points, reprojection_error
from .court_templates import Sport, get_court_template
from .schemas import BallTrack


STATUS_TESTED = "TESTED-ON-REAL-DATA"
CORNER_ORDER = ("near_left", "near_right", "far_right", "far_left")

# Relative tolerance for the x/y scale factors between a corner artifact's
# declared image_size and a caller's requested target_image_size to agree.
# A genuine uniform preview downscale (e.g. 960x540 from 1920x1080) has
# scale_x == scale_y exactly; anything outside this tolerance means the
# declared/target sizes do not share an aspect ratio and one of them is
# almost certainly wrong.
_SCALE_ASPECT_TOLERANCE = 0.01


def manual_court_projection_from_corners(
    court_corners: Mapping[str, Any] | str | Path,
    *,
    sport: Sport = "pickleball",
    target_image_size: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build a homography artifact from manual outer court corners.

    ``target_image_size``, if given, rescales the declared corner pixels onto
    that (width, height) before solving -- see the module docstring for the
    pixel-space contract this implements.
    """

    payload = _load_json_or_mapping(court_corners)
    item = _manual_corner_item(payload)
    corners = item["court_corners"]
    declared_image_size = _declared_image_size(item)
    raw_image_pts = [_point(corners[name], f"court_corners.{name}") for name in CORNER_ORDER]
    image_pts, scale = _rescale_image_points(raw_image_pts, declared_image_size, target_image_size)
    world_pts = [list(point) for point in get_court_template(sport).corners_m]
    homography = homography_from_planar_points(world_pts, image_pts)
    projected = project_planar_points(homography, world_pts)
    error = reprojection_error(image_pts, projected)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_manual_court_projection",
        "status": STATUS_TESTED,
        "sport": sport,
        "source": item.get("source"),
        "corner_status": item.get("status"),
        "corner_frame": item.get("frame"),
        "corner_order": list(CORNER_ORDER),
        "declared_image_size": list(declared_image_size),
        "target_image_size": list(target_image_size) if target_image_size is not None else None,
        "corner_pixel_scale_applied": list(scale),
        "image_pts": image_pts,
        "world_pts": world_pts,
        "homography": homography,
        "reprojection_error_px": error.model_dump(mode="json"),
        "reprojection_gate": {
            "median_px_lt": 8.0,
            "p95_px_lt": 15.0,
            "passed": bool(float(error.median) < 8.0 and float(error.p95) < 15.0),
        },
        "not_ground_truth": True,
    }


def manual_court_pose_from_corners(
    court_corners: Mapping[str, Any] | str | Path,
    *,
    sport: Sport = "pickleball",
    target_image_size: Sequence[int] | None = None,
) -> CameraPose:
    """Solve a full camera pose from the same 4 manual corners used for the homography.

    See ball_inout_uncertainty.solve_manual_corner_camera_pose for the method:
    principal point at the corner centroid, focal length found by minimizing
    reprojection error of these same 4 corners. No new human input, no
    reviewed label. ``target_image_size`` rescales the declared corner
    pixels the same way as ``manual_court_projection_from_corners`` -- pass
    the same value to both when deriving a homography and a pose for the
    same bounce projection, or the two will disagree about pixel space.
    """

    payload = _load_json_or_mapping(court_corners)
    item = _manual_corner_item(payload)
    corners = item["court_corners"]
    declared_image_size = _declared_image_size(item)
    raw_image_pts = [_point(corners[name], f"court_corners.{name}") for name in CORNER_ORDER]
    image_pts, _scale = _rescale_image_points(raw_image_pts, declared_image_size, target_image_size)
    world_pts = [list(point) for point in get_court_template(sport).corners_m]
    return solve_manual_corner_camera_pose(image_pts, world_pts)


def apply_manual_court_inout_to_ball_track(
    ball_track: Mapping[str, Any] | str | Path,
    court_corners: Mapping[str, Any] | str | Path,
    *,
    target_image_size: Sequence[int],
    sport: Sport = "pickleball",
    uncertainty_m: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project bounce contact pixels to court plane and fill in/out fields.

    ``target_image_size`` is the (width, height) pixel space the ball
    track's own ``contact_xy_img`` points live in -- almost always the
    native source-video resolution. It is required (not optional) because
    this is exactly the path a 2026-07-02 audit found silently mis-scaling
    court corners against native-pixel ball contacts (see the module
    docstring). Pass the ball track's/source video's real pixel size here;
    do not guess.

    uncertainty_m is an explicit override (fixed radius, old behavior). By
    default (uncertainty_m=None) the per-bounce uncertainty is derived from
    the clip's own camera geometry -- see ball_inout_uncertainty.py.
    """

    target_size = _positive_int_pair(target_image_size, "target_image_size")
    override_uncertainty = None if uncertainty_m is None else _nonnegative_finite(uncertainty_m, "uncertainty_m")
    track = BallTrack.model_validate(_load_json_or_mapping(ball_track))
    projection = manual_court_projection_from_corners(court_corners, sport=sport, target_image_size=target_size)
    pose: CameraPose | None = None
    pose_error: str | None = None
    if override_uncertainty is None:
        try:
            pose = manual_court_pose_from_corners(court_corners, sport=sport, target_image_size=target_size)
        except Exception as exc:  # pragma: no cover - defensive, exercised via tests when cv2 is present
            pose_error = str(exc)
    payload = track.model_dump(mode="json")
    projected_count = 0
    skipped: list[dict[str, Any]] = []
    updated_bounces: list[dict[str, Any]] = []

    for index, bounce in enumerate(payload["bounces"]):
        contact_xy = bounce.get("contact_xy_img")
        if contact_xy is None:
            skipped.append({"index": index, "reason": "missing_contact_xy_img"})
            updated_bounces.append(bounce)
            continue
        contact_point = _point(contact_xy, f"bounces/{index}/contact_xy_img")
        world_xy = project_image_points_to_world(projection["homography"], [contact_point])[0]

        if override_uncertainty is not None:
            uncertainty = override_uncertainty
            dominant_uncertainty_term = "manual_corner_homography_projection"
            uncertainty_breakdown = fixed_override_breakdown(uncertainty)
        elif pose is not None:
            geometric = bounce_geometric_uncertainty_m(
                contact_xy_img=contact_point,
                world_xy=world_xy,
                pose=pose,
                sport=sport,
                fps=float(track.fps),
                reprojection_error_px_p95=float(projection["reprojection_error_px"]["p95"]),
            )
            uncertainty = geometric["uncertainty_m"]
            dominant_uncertainty_term = geometric["dominant_uncertainty_term"]
            uncertainty_breakdown = geometric["breakdown"]
        else:
            raise ValueError(f"cannot derive geometric uncertainty for {court_corners}: {pose_error}")

        line_call = classify_bounce(
            float(bounce["t"]),
            world_xy,
            sport=sport,
            uncertainty_radius_m=uncertainty,
        )
        margin = float(line_call["boundary_margin_m"])
        bounce["world_xy"] = [float(world_xy[0]), float(world_xy[1])]
        bounce["margin_m"] = margin
        bounce["uncertainty_m"] = uncertainty
        bounce["confidence"] = _inout_confidence(margin, uncertainty)
        bounce["call"] = "too_close_to_call" if line_call["court_call"] == "unknown" else line_call["court_call"]
        bounce["nearest_line"] = line_call["nearest_boundary_line_id"]
        bounce["region"] = line_call["zone"]
        bounce["dominant_uncertainty_term"] = dominant_uncertainty_term
        bounce["uncertainty_breakdown"] = uncertainty_breakdown
        updated_bounces.append(bounce)
        projected_count += 1

    payload["bounces"] = updated_bounces
    parsed = BallTrack.model_validate(payload)
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_manual_court_inout_summary",
        "status": STATUS_TESTED,
        "sport": sport,
        "source_ball_track": str(ball_track) if isinstance(ball_track, str | Path) else None,
        "source_court_corners": str(court_corners) if isinstance(court_corners, str | Path) else None,
        "bounce_count": len(parsed.bounces),
        "projected_bounce_count": projected_count,
        "skipped_bounces": skipped,
        "uncertainty_m": override_uncertainty,
        "uncertainty_model": _uncertainty_model_summary(pose, override_uncertainty),
        "projection": projection,
        "blocked_reason": _blocked_reason(total=len(parsed.bounces), projected=projected_count),
        "not_ground_truth": True,
    }
    return parsed.model_dump(mode="json"), summary


def _uncertainty_model_summary(pose: CameraPose | None, override_uncertainty: float | None) -> dict[str, Any]:
    if override_uncertainty is not None:
        return {
            "method": "fixed_override",
            "uncertainty_m": override_uncertainty,
        }
    if pose is None:
        return {"method": "camera_geometry_elevation_parallax_v1", "pose": None}
    return {
        "method": "camera_geometry_elevation_parallax_v1",
        "pose": {
            "fx": pose.fx,
            "fy": pose.fy,
            "cx": pose.cx,
            "cy": pose.cy,
            "camera_height_m": pose.camera_height_m,
            "reprojection_error_px": {
                "median": pose.reprojection_error_px_median,
                "p95": pose.reprojection_error_px_p95,
            },
            "source": pose.source,
        },
        "physics_constants": physics_constants_manifest(),
    }


def write_manual_court_inout_ball_track(
    *,
    ball_track_path: str | Path,
    court_corners_path: str | Path,
    out: str | Path,
    target_image_size: Sequence[int],
    summary_out: str | Path | None = None,
    sport: Sport = "pickleball",
    uncertainty_m: float | None = None,
) -> dict[str, Any]:
    payload, summary = apply_manual_court_inout_to_ball_track(
        ball_track_path,
        court_corners_path,
        target_image_size=target_image_size,
        sport=sport,
        uncertainty_m=uncertainty_m,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary_out is not None:
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _manual_corner_item(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    annotation = payload.get("annotation")
    if not isinstance(annotation, Mapping):
        raise ValueError("court_corners payload missing annotation object")
    items = annotation.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("court_corners payload missing annotation.items")
    item = items[0]
    if not isinstance(item, Mapping):
        raise ValueError("court_corners annotation item must be an object")
    corners = item.get("court_corners")
    if not isinstance(corners, Mapping):
        raise ValueError("court_corners annotation item missing court_corners object")
    missing = [name for name in CORNER_ORDER if name not in corners]
    if missing:
        raise ValueError(f"court_corners missing {', '.join(missing)}")
    return item


def _declared_image_size(item: Mapping[str, Any]) -> tuple[int, int]:
    """The pixel space (width, height) a court_corners item's coordinates were tapped in.

    Required -- fails closed (no legacy/implicit fallback) so a corner
    artifact can never be silently consumed in the wrong pixel space again.
    See the module docstring for the incident this migration fixes.
    """

    value = item.get("image_size")
    if value is None:
        raise ValueError(
            "court_corners annotation item is missing a required 'image_size': [width, height] "
            "declaring the pixel space its court_corners coordinates were tapped in. This field "
            "was made mandatory after a 2026-07-02 audit found manual court corners tapped against "
            "960x540 preview frames being silently consumed as if they were native 1920x1080 "
            "coordinates, flipping several 'in' bounces to 'out' by 0.9-2.3 m "
            "(runs/ball_m5_geometric_uncertainty_20260702T022544Z/geometric_uncertainty_validation_summary.json). "
            "Migrate this court_corners.json by adding \"image_size\": [width, height] to the "
            "annotation item (the width/height of whatever frame image the corners were clicked "
            "against -- for these prototype clips, check the actual image file dimensions before "
            "guessing)."
        )
    return _positive_int_pair(value, "court_corners item.image_size")


def _positive_int_pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a [width, height] pair")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a [width, height] pair") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must be a [width, height] pair")
    try:
        width, height = int(values[0]), int(values[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer width/height") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must have positive width and height, got {(width, height)}")
    return width, height


def _rescale_image_points(
    image_pts: list[list[float]],
    declared_image_size: tuple[int, int],
    target_image_size: Sequence[int] | None,
) -> tuple[list[list[float]], tuple[float, float]]:
    """Rescale corner pixels from their declared space onto target_image_size.

    Returns the (possibly rescaled) points and the (scale_x, scale_y)
    actually applied (``(1.0, 1.0)`` when target_image_size is None). Raises
    if the requested target does not share the declared corners' aspect
    ratio -- a mismatched aspect almost always means one of the two sizes is
    wrong, and silently non-uniformly stretching a court is worse than
    failing loudly (this is the "double-scaling hazard" guard: it also
    catches a caller accidentally rescaling already-native corners against a
    second target).
    """

    if target_image_size is None:
        return image_pts, (1.0, 1.0)

    target_width, target_height = _positive_int_pair(target_image_size, "target_image_size")
    declared_width, declared_height = declared_image_size
    scale_x = target_width / declared_width
    scale_y = target_height / declared_height
    if scale_x != scale_y:
        relative_difference = abs(scale_x - scale_y) / max(scale_x, scale_y)
        if relative_difference > _SCALE_ASPECT_TOLERANCE:
            raise ValueError(
                f"target_image_size {(target_width, target_height)} does not share an aspect ratio "
                f"with the corners' declared image_size {declared_image_size} "
                f"(scale_x={scale_x:.6f}, scale_y={scale_y:.6f}); refusing to non-uniformly stretch "
                "the court. Check that both sizes describe the same frame orientation/crop."
            )
    scaled = [[x * scale_x, y * scale_y] for x, y in image_pts]
    return scaled, (scale_x, scale_y)


def _point(value: Any, name: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a 2D point")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a 2D point") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must be a 2D point")
    point = [float(values[0]), float(values[1])]
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"{name} must contain finite coordinates")
    return point


def _load_json_or_mapping(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _nonnegative_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


def _inout_confidence(margin_m: float, uncertainty_m: float) -> float:
    denominator = abs(margin_m) + uncertainty_m
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, abs(margin_m) / denominator))


def _blocked_reason(*, total: int, projected: int) -> str | None:
    if total <= 0:
        return "ball_track_has_no_bounces"
    if projected <= 0:
        return "no_bounces_with_contact_xy_img"
    if projected < total:
        return "some_bounces_missing_contact_xy_img"
    return None


__all__ = [
    "apply_manual_court_inout_to_ball_track",
    "manual_court_pose_from_corners",
    "manual_court_projection_from_corners",
    "write_manual_court_inout_ball_track",
]
