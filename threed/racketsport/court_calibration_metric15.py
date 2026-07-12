"""Full single-view metric camera calibration from reviewed 15-point court keypoints.

This is a *new*, independent calibration path that does not require an ARKit floor
plane/camera pose (unlike `metric_calibration_from_sidecar_and_keypoints`). It fits
real intrinsics (fx, fy, cx, cy, k1, k2) plus a `solvePnP` pose directly from the
human-reviewed 15-point court keypoint labels shipped with each eval clip
(`eval_clips/ball/<clip>/labels/court_keypoints.json`).

Why this exists: every prior calibration in the repo used *guessed* intrinsics
(fx=fy=max(w,h)*1.2, principal point at the image centroid, zero distortion) plus a
solvePnP fit over only 4 manually-tapped corners -- a near-degenerate correspondence
set. The diagnostic run `runs/cal_body_projection_bias_20260702T014121Z/` measured the
resulting PnP-vs-homography footpoint disagreement at 63-75px on Wolverine (no fisheye)
and a 0/20 pass rate on Burlington (fisheye compounds it ~1.6x), and showed the guessed
focal-length/degenerate-4-point defect is present on *both* clips, not just the
fisheye one.

Single-view planar calibration identifiability
-----------------------------------------------
Fitting a full pinhole+radial-distortion camera model (fx, fy, cx, cy, k1, k2) from a
*single* view of a *planar* target (all 15 pickleball court keypoints lie on the court
plane, z=0) is classically under-constrained: a homography alone (8 DOF) cannot
uniquely decompose into intrinsics + pose without extra assumptions (Hartley &
Zisserman ch. 7; Zhang 2000's method needs >= 2 views at different orientations to
solve the image-of-the-absolute-conic linear system -- one view gives too few
equations). We make this identifiability tradeoff explicit and honest rather than
silently let a nonlinear optimizer converge to an unstable basin:

- **Principal point is fixed at the geometric image center** (cx=W/2, cy=H/2). A
  single planar view cannot separate a principal-point offset from a compensating
  pose change.
- **fx is constrained equal to fy** (unit aspect ratio / square pixels). Standard for
  consumer/webcam sensors and removes one more unidentifiable DOF.
- **Focal length is found by a coarse-to-fine 1D grid search** (zero distortion,
  solvePnP reprojection RMSE) rather than seeding straight into `cv2.calibrateCamera`,
  because the joint fx/k1/k2/pose optimization from a single view is prone to bad
  local minima without a good starting point.
- **k1, k2 are only accepted if they clear an honesty gate**: a `cv2.calibrateCamera`
  refinement (still fx=fy, cx/cy fixed, zero tangential distortion, k3=0) is attempted
  seeded from the grid-search focal; it is only adopted over the zero-distortion
  baseline if it reduces median reprojection error by
  `DEFAULT_DISTORTION_IMPROVEMENT_THRESHOLD` (15%) or more. This avoids overfitting two
  extra parameters to 15 single-view points when the data does not support it.

Every choice above is recorded in the returned fit's `identifiability_notes` and
surfaces into the `capture_quality.reasons` of the emitted `CourtCalibration`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capture_quality import score_capture_quality
from .court_calibration import homography_from_planar_points, reprojection_error
from .court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME
from .court_positioning import (
    CameraFloorGeometry,
    estimate_ground_sample_distance,
    estimate_position_uncertainty,
)
from .court_templates import Sport
from .schemas import (
    PICKLEBALL_COURT_KEYPOINT_NAMES,
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
)

DEFAULT_DISTORTION_IMPROVEMENT_THRESHOLD = 0.15
MIN_REVIEWED_CORRESPONDENCES = 6
METRIC15_SOURCE_TAG = "metric_15pt_reviewed"


@dataclass(frozen=True)
class ReviewedKeypointFrame:
    frame: str
    status: str
    keypoints: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class ReviewedCourtKeypoints:
    clip: str
    label_coordinate_space: tuple[float, float]
    source_resolution: tuple[float, float]
    frames: list[ReviewedKeypointFrame]


@dataclass(frozen=True)
class SingleViewCameraFit:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    R: list[list[float]]
    t: list[float]
    distortion_model: str
    reprojection_error_px: ReprojectionError
    per_point_residual_px: list[float]
    identifiability_notes: list[str]


def load_reviewed_court_keypoints_15pt(path: str | Path) -> ReviewedCourtKeypoints:
    """Load the human-reviewed 15-point CVAT-style court keypoint export.

    Raises if the artifact does not declare `label_coordinate_space`/
    `source_resolution` -- these labels are known to sometimes be produced on a
    downscaled preview frame (exactly the 960x540-vs-native defect this discipline
    guards against), so an undeclared size is treated as untrustworthy rather than
    silently assumed to be native pixels.
    """

    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames_meta = payload.get("frames")
    if not isinstance(frames_meta, dict):
        legacy = _load_legacy_single_frame_court_keypoints(payload, Path(path))
        if legacy is not None:
            return legacy
        raise ValueError(f"{path}: missing 'frames' metadata block")
    label_space = frames_meta.get("label_coordinate_space")
    source_res = frames_meta.get("source_resolution")
    if not label_space or not source_res:
        raise ValueError(
            f"{path}: reviewed court keypoints must declare both label_coordinate_space "
            "and source_resolution before use -- labels may have been produced on a "
            "downscaled preview and must never be trusted at face value as native pixels "
            "(this is the same 960x540-corner-tap discipline the calibration pipeline "
            "already applies elsewhere)."
        )

    items = payload.get("annotation", {}).get("items", [])
    frames: list[ReviewedKeypointFrame] = []
    for item in items:
        keypoints_raw = item.get("keypoints", {})
        missing = sorted(set(PICKLEBALL_COURT_KEYPOINT_NAMES) - set(keypoints_raw))
        if missing:
            continue
        keypoints = {
            name: (float(keypoints_raw[name][0]), float(keypoints_raw[name][1]))
            for name in PICKLEBALL_COURT_KEYPOINT_NAMES
        }
        frames.append(
            ReviewedKeypointFrame(
                frame=str(item.get("frame", "")),
                status=str(item.get("status", "")),
                keypoints=keypoints,
            )
        )
    if not frames:
        raise ValueError(f"{path}: no reviewed frame contains all 15 canonical pickleball keypoints")

    return ReviewedCourtKeypoints(
        clip=str(payload.get("clip", "")),
        label_coordinate_space=(float(label_space[0]), float(label_space[1])),
        source_resolution=(float(source_res[0]), float(source_res[1])),
        frames=frames,
    )


def _load_legacy_single_frame_court_keypoints(
    payload: Mapping[str, Any],
    path: Path,
) -> ReviewedCourtKeypoints | None:
    """Accept the old single-frame court-keypoint artifact shape.

    The legacy IMG_1605 reviewed label predates the `frames` metadata wrapper
    but is already in source-video pixel coordinates. Treat it as one reviewed
    static frame instead of mutating the protected label file.
    """

    raw_keypoints = payload.get("keypoints")
    if not isinstance(raw_keypoints, Sequence) or isinstance(raw_keypoints, (str, bytes)):
        return None
    keypoints_by_name: dict[str, tuple[float, float]] = {}
    for item in raw_keypoints:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "")
        uv = item.get("uv")
        if name not in PICKLEBALL_COURT_KEYPOINT_NAMES:
            continue
        if not isinstance(uv, Sequence) or isinstance(uv, (str, bytes)) or len(uv) < 2:
            continue
        keypoints_by_name[name] = (float(uv[0]), float(uv[1]))
    missing = sorted(set(PICKLEBALL_COURT_KEYPOINT_NAMES) - set(keypoints_by_name))
    if missing:
        return None

    frame_indexes = payload.get("frame_indexes")
    frame_id = ""
    if isinstance(frame_indexes, Sequence) and not isinstance(frame_indexes, (str, bytes)) and frame_indexes:
        frame_id = str(frame_indexes[0])
    image_size = _legacy_single_frame_image_size(payload, path, frame_id=frame_id)
    return ReviewedCourtKeypoints(
        clip=str(payload.get("clip") or path.parent.parent.name),
        label_coordinate_space=image_size,
        source_resolution=image_size,
        frames=[
            ReviewedKeypointFrame(
                frame=frame_id,
                status="legacy_single_frame_no_frames_metadata",
                keypoints=keypoints_by_name,
            )
        ],
    )


def _legacy_single_frame_image_size(
    payload: Mapping[str, Any],
    path: Path,
    *,
    frame_id: str,
) -> tuple[float, float]:
    for key in ("label_coordinate_space", "source_resolution", "image_size"):
        value = payload.get(key)
        parsed = _size_pair(value)
        if parsed is not None:
            return parsed
    frame_path = _legacy_single_frame_image_path(path, frame_id=frame_id)
    if frame_path is not None:
        try:
            from PIL import Image  # type: ignore[import-not-found]

            with Image.open(frame_path) as image:
                width, height = image.size
            return (float(width), float(height))
        except Exception as exc:  # pragma: no cover - defensive fallback path
            raise ValueError(f"{path}: could not read legacy frame image size from {frame_path}: {exc}") from exc
    raise ValueError(
        f"{path}: legacy single-frame court keypoints need label_coordinate_space/source_resolution "
        "or a readable sibling court_keypoint_partial_frames image"
    )


def _legacy_single_frame_image_path(path: Path, *, frame_id: str) -> Path | None:
    frame_dir = path.parent / "court_keypoint_partial_frames"
    if not frame_dir.is_dir():
        return None
    candidates: list[Path] = []
    if frame_id:
        try:
            candidates.append(frame_dir / f"frame_{int(frame_id):06d}.jpg")
        except ValueError:
            candidates.append(frame_dir / f"frame_{frame_id}.jpg")
    candidates.extend(sorted(frame_dir.glob("*.jpg")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _size_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    width, height = float(value[0]), float(value[1])
    if width <= 0.0 or height <= 0.0:
        return None
    return (width, height)


def aggregate_reviewed_keypoints_native_px(
    reviewed: ReviewedCourtKeypoints,
    *,
    native_image_size: tuple[float, float] | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, float]], tuple[float, float]]:
    """Median-aggregate per-frame keypoints and rescale into native source-video pixels.

    All reviewed frames for a static-camera clip are copies of the same camera pose
    (one independently-reviewed frame plus `reviewed_static_camera_copy` duplicates), so
    median-aggregating across them is a robustness step, not a source of new
    information -- consistent with the CAL static-camera aggregation policy in
    `NORTH_STAR_ROADMAP.md`. Returns (native_points_by_name, per_point_frame_stdev_px,
    native_image_size).
    """

    native_size = tuple(float(v) for v in (native_image_size or reviewed.source_resolution))
    label_w, label_h = reviewed.label_coordinate_space
    if label_w <= 0.0 or label_h <= 0.0:
        raise ValueError("label_coordinate_space must be positive")
    scale_x = native_size[0] / label_w
    scale_y = native_size[1] / label_h

    aggregated: dict[str, tuple[float, float]] = {}
    stdev_by_name: dict[str, dict[str, float]] = {}
    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        xs = [frame.keypoints[name][0] * scale_x for frame in reviewed.frames]
        ys = [frame.keypoints[name][1] * scale_y for frame in reviewed.frames]
        aggregated[name] = (_median(xs), _median(ys))
        stdev_by_name[name] = {"x_stdev_px": _stdev(xs), "y_stdev_px": _stdev(ys)}
    return aggregated, stdev_by_name, native_size


def fit_single_view_metric_camera(
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    image_size: tuple[float, float],
    *,
    distortion_improvement_threshold: float = DEFAULT_DISTORTION_IMPROVEMENT_THRESHOLD,
) -> SingleViewCameraFit:
    """Fit fx(=fy), k1, k2, and solvePnP pose from a single planar-target view.

    See the module docstring for the full identifiability discussion. `object_points_m`
    must be coplanar (z=0) in the calibration target's own frame; `image_points_px` are
    the matching pixel observations in the same (native) pixel space as `image_size`.
    """

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("fit_single_view_metric_camera requires opencv-python and numpy") from exc

    obj = np.asarray([[float(v) for v in point] for point in object_points_m], dtype=np.float64)
    img = np.asarray([[float(v) for v in point] for point in image_points_px], dtype=np.float64)
    if obj.shape[0] != img.shape[0]:
        raise ValueError("object and image point counts must match")
    if obj.shape[0] < MIN_REVIEWED_CORRESPONDENCES:
        raise ValueError(f"single-view metric calibration requires at least {MIN_REVIEWED_CORRESPONDENCES} correspondences")

    width, height = float(image_size[0]), float(image_size[1])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("image_size must be positive")
    cx, cy = width / 2.0, height / 2.0

    base_notes = [
        "principal point fixed at the geometric image center (cx=W/2, cy=H/2): a single "
        "planar view cannot separate principal-point offset from a compensating pose "
        "change (classic single-image calibration degeneracy).",
        "fx constrained equal to fy (square-pixel / unit aspect-ratio assumption): a "
        "single view cannot independently resolve fx vs fy without this assumption.",
    ]

    def _residual_for_focal(focal: float) -> tuple[float, Any, Any]:
        k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return math.inf, None, None
        projected, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
        projected = projected.reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=1))))
        return err, rvec, tvec

    lo, hi = 0.3 * max(width, height), 6.0 * max(width, height)
    best_err, best_focal, best_rvec, best_tvec = math.inf, None, None, None
    for _ in range(5):
        candidates = np.linspace(lo, hi, 51)
        round_err, round_focal, round_rvec, round_tvec = math.inf, None, None, None
        for focal in candidates:
            err, rvec, tvec = _residual_for_focal(float(focal))
            if err < round_err:
                round_err, round_focal, round_rvec, round_tvec = err, float(focal), rvec, tvec
        if round_err < best_err:
            best_err, best_focal, best_rvec, best_tvec = round_err, round_focal, round_rvec, round_tvec
        span = (hi - lo) / 50.0
        lo = max(1.0, round_focal - 4.0 * span)
        hi = round_focal + 4.0 * span

    if best_focal is None:
        raise ValueError("focal-length grid search failed to solve a camera pose for any candidate")

    zero_dist_fit = _build_fit(
        cv2,
        np,
        obj,
        img,
        fx=best_focal,
        fy=best_focal,
        cx=cx,
        cy=cy,
        dist=[0.0, 0.0, 0.0, 0.0],
        rvec=best_rvec,
        tvec=best_tvec,
        distortion_model="zero_distortion_grid_search_focal",
        notes=[
            *base_notes,
            "focal length found by coarse-to-fine 1D grid search minimizing solvePnP "
            "reprojection RMSE (zero distortion) rather than seeding cv2.calibrateCamera "
            "cold, since the joint optimization is prone to bad local minima from a single "
            "view.",
            "radial distortion fixed at zero (k1=k2=0): this is the identifiability "
            "baseline before testing whether allowing k1,k2 meaningfully improves the fit.",
        ],
    )

    distorted_fit = _try_refine_with_distortion(
        cv2, np, obj, img, seed_focal=best_focal, cx=cx, cy=cy, width=width, height=height, base_notes=base_notes
    )

    # Below this floor the zero-distortion fit is already sub-hundredth-pixel and any
    # "improvement" ratio is floating-point noise, not signal -- keep zero distortion.
    negligible_error_floor_px = 1e-3
    if distorted_fit is not None and zero_dist_fit.reprojection_error_px.median > negligible_error_floor_px:
        improvement = 1.0 - (distorted_fit.reprojection_error_px.median / zero_dist_fit.reprojection_error_px.median)
        if improvement >= distortion_improvement_threshold:
            note = (
                f"k1,k2 accepted: reduced median reprojection error by {improvement:.1%}, "
                f">= the {distortion_improvement_threshold:.0%} gate required to justify 2 extra "
                "single-view degrees of freedom."
            )
            return replace(distorted_fit, identifiability_notes=[*distorted_fit.identifiability_notes, note])
        note = (
            f"k1,k2 rejected: only {improvement:.1%} median-reprojection improvement over "
            f"zero-distortion (< the {distortion_improvement_threshold:.0%} gate) -- falling back to "
            "zero distortion to avoid overfitting 2 extra parameters to 15 single-view points."
        )
        return replace(zero_dist_fit, identifiability_notes=[*zero_dist_fit.identifiability_notes, note])

    if distorted_fit is None:
        note = "k1,k2 refinement did not converge; zero distortion retained."
    else:
        note = (
            f"zero-distortion median reprojection error ({zero_dist_fit.reprojection_error_px.median:.4f}px) is "
            f"already below the {negligible_error_floor_px}px noise floor -- any k1,k2 'improvement' ratio would "
            "be floating-point noise, not signal; zero distortion retained."
        )
    return replace(zero_dist_fit, identifiability_notes=[*zero_dist_fit.identifiability_notes, note])


def _try_refine_with_distortion(
    cv2: Any,
    np: Any,
    obj: Any,
    img: Any,
    *,
    seed_focal: float,
    cx: float,
    cy: float,
    width: float,
    height: float,
    base_notes: list[str],
) -> SingleViewCameraFit | None:
    k_guess = np.array([[seed_focal, 0.0, cx], [0.0, seed_focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )
    try:
        _rms, k_fit, dist_fit, rvecs, tvecs = cv2.calibrateCamera(
            [obj.astype(np.float32)],
            [img.astype(np.float32)],
            (int(round(width)), int(round(height))),
            k_guess.copy(),
            None,
            flags=flags,
        )
    except cv2.error:
        return None

    f_fit = float(k_fit[0, 0])
    dist_flat = dist_fit.reshape(-1)
    k1 = float(dist_flat[0]) if dist_flat.size > 0 else 0.0
    k2 = float(dist_flat[1]) if dist_flat.size > 1 else 0.0
    return _build_fit(
        cv2,
        np,
        obj,
        img,
        fx=f_fit,
        fy=f_fit,
        cx=cx,
        cy=cy,
        dist=[k1, k2, 0.0, 0.0],
        rvec=rvecs[0],
        tvec=tvecs[0],
        distortion_model="k1_k2_calibrate_camera_seeded",
        notes=[
            *base_notes,
            "k1,k2 allowed to vary (tangential distortion fixed at zero, k3 fixed at zero) "
            "via cv2.calibrateCamera, seeded from the zero-distortion grid-search focal/pose.",
        ],
    )


def _build_fit(
    cv2: Any,
    np: Any,
    obj: Any,
    img: Any,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist: list[float],
    rvec: Any,
    tvec: Any,
    distortion_model: str,
    notes: list[str],
) -> SingleViewCameraFit:
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_arr = np.asarray(dist, dtype=np.float64)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist_arr)
    projected = projected.reshape(-1, 2)
    error = reprojection_error(img.tolist(), projected.tolist())
    residuals = [
        math.hypot(float(o[0]) - float(p[0]), float(o[1]) - float(p[1])) for o, p in zip(img.tolist(), projected.tolist())
    ]
    residuals = [0.0 if value < 1e-9 else value for value in residuals]
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    return SingleViewCameraFit(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        k1=float(dist_arr[0]) if dist_arr.size > 0 else 0.0,
        k2=float(dist_arr[1]) if dist_arr.size > 1 else 0.0,
        R=[[float(value) for value in row] for row in rotation.tolist()],
        t=[float(value) for value in translation.tolist()],
        distortion_model=distortion_model,
        reprojection_error_px=error,
        per_point_residual_px=residuals,
        identifiability_notes=list(notes),
    )


def metric_calibration_from_reviewed_keypoints_15pt(
    keypoints_path: str | Path,
    *,
    sport: Sport = "pickleball",
    native_image_size: tuple[float, float] | None = None,
    source_tag: str = METRIC15_SOURCE_TAG,
    distortion_improvement_threshold: float = DEFAULT_DISTORTION_IMPROVEMENT_THRESHOLD,
) -> CourtCalibration:
    """Build a `CourtCalibration` from human-reviewed 15-point court keypoint labels.

    This is the importable API for the metric-15pt calibration path: no ARKit sidecar
    is required, unlike `metric_calibration_from_sidecar_and_keypoints`. See the module
    docstring for the single-view identifiability tradeoffs this makes explicit.
    """

    if sport != "pickleball":
        raise ValueError("reviewed 15-point metric calibration currently supports pickleball only")

    reviewed = load_reviewed_court_keypoints_15pt(keypoints_path)
    native_points, point_stdev_px, native_size = aggregate_reviewed_keypoints_native_px(
        reviewed, native_image_size=native_image_size
    )

    object_points = [list(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m) for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
    image_points = [list(native_points[name]) for name in PICKLEBALL_COURT_KEYPOINT_NAMES]

    fit = fit_single_view_metric_camera(
        object_points,
        image_points,
        native_size,
        distortion_improvement_threshold=distortion_improvement_threshold,
    )

    homography = homography_from_planar_points(object_points, image_points)

    intrinsics = CameraIntrinsics(
        fx=fit.fx,
        fy=fit.fy,
        cx=fit.cx,
        cy=fit.cy,
        dist=[fit.k1, fit.k2, 0.0, 0.0],
        source=source_tag,
    )
    camera_center_world = _camera_center_from_pose(fit.R, fit.t)
    extrinsics = CourtExtrinsics(
        R=fit.R,
        t=fit.t,
        camera_height_m=max(abs(camera_center_world[2]), 1e-6),
    )

    geometry = CameraFloorGeometry(
        intrinsics={"fx": fit.fx, "fy": fit.fy, "cx": fit.cx, "cy": fit.cy},
        camera_origin_world=camera_center_world,
        R_world_camera=_transpose(fit.R),
        floor_plane_point=[0.0, 0.0, 0.0],
        floor_plane_normal=[0.0, 0.0, 1.0],
    )
    gsd_samples = []
    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        uv = native_points[name]
        gsd = estimate_ground_sample_distance(uv, geometry)
        sigma = estimate_position_uncertainty(
            pixel_error_px=fit.reprojection_error_px.median,
            gsd_m_per_px=gsd,
            plane_sigma_m=0.0,
            calibration_sigma_m=0.0,
        )
        canonical = PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m
        gsd_samples.append(
            {
                "court_xy": [float(canonical[0]), float(canonical[1])],
                "gsd_m_per_px": gsd,
                "sigma_p_m": sigma,
            }
        )

    metric_confidence = _confidence_from_reprojection(fit.reprojection_error_px)

    base_quality = score_capture_quality(
        corners_visible=len(PICKLEBALL_COURT_KEYPOINT_NAMES),
        reprojection_rmse_px=fit.reprojection_error_px.median,
    )
    reasons = list(
        dict.fromkeys(
            [
                *base_quality.reasons,
                "single_view_planar_full_calibration",
                f"distortion_model={fit.distortion_model}",
                "reviewed_15pt_correspondences",
            ]
        )
    )
    capture_quality = CaptureQuality(grade=base_quality.grade, reasons=reasons)

    solved_frames = sorted({_native_frame_index_from_label(frame.frame) for frame in reviewed.frames})

    payload = {
        "schema_version": 1,
        "sport": sport,
        "coordinate_frame": "court_netcenter_z_up_m",
        "T_world_court": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "homography": homography,
        "intrinsics": intrinsics.model_dump(mode="json"),
        "image_size": [int(round(native_size[0])), int(round(native_size[1]))],
        "extrinsics": extrinsics.model_dump(mode="json"),
        "reprojection_error_px": fit.reprojection_error_px.model_dump(mode="json"),
        "per_keypoint_residual_px": fit.per_point_residual_px,
        "metric_confidence": metric_confidence,
        "gsd_model": {
            "type": "analytic_ray_plane",
            "plane_sigma_m": 0.0,
            "calibration_sigma_m": 0.0,
            "samples": gsd_samples,
        },
        "capture_quality": capture_quality.model_dump(mode="json"),
        "image_pts": image_points,
        "world_pts": object_points,
        "source": source_tag,
        "solved_over_frames": solved_frames,
    }
    return CourtCalibration.model_validate(payload)


def _confidence_from_reprojection(error: ReprojectionError) -> str:
    if error.median <= 2.0 and error.p95 <= 5.0:
        return "high"
    if error.median <= 6.0 and error.p95 <= 15.0:
        return "med"
    return "low"


def _camera_center_from_pose(rotation: list[list[float]], translation: list[float]) -> list[float]:
    # X_cam = R @ X_world + t  =>  camera center in world frame C = -R^T @ t
    rotated = [sum(rotation[k][i] * translation[k] for k in range(3)) for i in range(3)]
    return [-value for value in rotated]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]


_FRAME_INDEX_RE = re.compile(r"(\d+)")


def _native_frame_index_from_label(label: str, *, sample_every_frames: int = 30) -> int:
    """Map a reviewed label filename (e.g. `frame_000001.jpg`, 1-based) to the native
    0-based video frame index it was extracted from (verified against `source.mp4` via
    pixel-diff cross-check: label frame_000001 == native frame 0, frame_000002 ==
    native frame 30, i.e. `(label_index - 1) * sample_every_frames`)."""

    match = _FRAME_INDEX_RE.search(label)
    if not match:
        raise ValueError(f"cannot parse a frame index from label {label!r}")
    label_index = int(match.group(1))
    return max(0, label_index - 1) * sample_every_frames


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median requires at least one value")
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return math.sqrt(variance)
