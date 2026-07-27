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
- **k1 (then k2) are only accepted if they clear an honesty gate on HELD-OUT residual**:
  see the model-selection section below. This avoids overfitting extra parameters to 15
  single-view points when the data does not support it.

Model selection is leave-one-out cross-validated (2026-07-26, CAL lane)
----------------------------------------------------------------------
The original gate compared the distorted and zero-distortion fits on the *training*
median -- the same 15 points both models were fit to. A lower residual on the points you
fit is not evidence: it is what more parameters do. Selection now runs on leave-one-out
cross-validated residual (15 folds; each fold refits focal, distortion and pose on 14
points and scores the 15th, which the fold never saw), and complexity is only added when
the held-out median improves by `distortion_improvement_threshold`.

Two things are selected this way, in this order:

1. **Net-keypoint label height.** `court_keypoint_net.PICKLEBALL_KEYPOINTS` declares the
   3 net keypoints at the net *top* (`post_net_height_m`, 0.9144 m). Every reviewed label
   set in this repo actually marks the net line where it meets the court, ~0 m: measured
   by back-projecting each net label through a floor-only-fit camera onto the vertical
   net plane, the implied label height is 0.008-0.130 m on all six reviewed clips, and
   ground-net-line beats net-top on leave-one-out residual by 2-10x everywhere. A 0.9 m
   world-model error on 3 of 15 correspondences is a 40-120 px systematic residual, which
   is exactly why no distortion coefficient could ever clear the old gate: distortion
   cannot explain a mis-specified object point. The height is therefore *selected* per
   clip rather than assumed, so a future clip genuinely labelled at net top still fits.
   `court_keypoint_net` is not modified -- it is the court/training lane's taxonomy, and
   the fix belongs to whoever consumes it for a metric fit.
2. **Radial distortion**, in increasing complexity: zero -> k1 -> k1+k2. Each candidate is
   compared against the currently accepted model, not against the previous candidate, so a
   k1+k2 pair that works when k1 alone does not is still reachable (real: radial terms
   trade off against focal length). Coefficients are bounded and, harder, required to be
   *radially invertible over this frame* -- a radial map that folds back before covering
   the image corner is not a camera, and unconstrained `cv2.calibrateCamera` returns such
   models from 15 single-view points (it produced k1=+6.96 on one clip here).

Every choice above is recorded in the returned fit's `identifiability_notes` and
surfaces into the `capture_quality.reasons` of the emitted `CourtCalibration`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .camera_distortion import is_radially_invertible, max_normalized_radius
from .capture_quality import score_capture_quality
from .coordinates import camera_matrix_from_intrinsics, invert_extrinsics
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

# The 3 keypoints whose declared world height is ambiguous between label sources:
# the court's ground net line (z=0) and the top of the physical net.
NET_KEYPOINT_NAMES: tuple[str, ...] = ("net_left_sideline", "net_center", "net_right_sideline")

# Radial coefficient box. Wide enough for the strong barrel of an action cam
# (k1 ~ -0.3 measured on Burlington) without letting a 15-point single-view fit
# wander into the unphysical values `cv2.calibrateCamera` returns unconstrained.
# The binding constraint is `is_radially_invertible`, not this box.
K1_BOUNDS = (-0.60, 0.35)
K2_BOUNDS = (-0.50, 0.50)

# Leave-one-out is the held-out protocol: 15 folds, each refitting focal,
# distortion and pose on 14 points and scoring the 1 point it never saw.
CROSS_VALIDATION = "leave_one_out"


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
    # Held-out (leave-one-out) median reprojection residual of the accepted
    # model. This -- not `reprojection_error_px.median` -- is what selected it.
    held_out_median_px: float | None = None
    # Every candidate scored, so the choice is auditable rather than asserted.
    model_selection: list[dict[str, Any]] | None = None
    # Which object-point variant won, and the points it used.
    object_point_variant: str = "as_given"
    object_points_m: list[list[float]] | None = None


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
    object_point_variants: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> SingleViewCameraFit:
    """Fit fx(=fy), k1, k2, and a solvePnP pose from a single view.

    See the module docstring for the identifiability discussion and the selection
    protocol. `image_points_px` are pixel observations in the same (native) pixel space
    as `image_size`.

    `object_point_variants` offers competing world-coordinate hypotheses for the same
    pixels -- used for the net-keypoint label height, which differs by label source and
    is not knowable from the pixels alone. Each variant is scored by leave-one-out
    cross-validated residual and the best one wins; omit it and `object_points_m` is the
    only candidate.
    """

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("fit_single_view_metric_camera requires opencv-python and numpy") from exc

    variants = dict(object_point_variants) if object_point_variants else {"as_given": object_points_m}
    img = np.asarray([[float(v) for v in point] for point in image_points_px], dtype=np.float64)
    parsed: dict[str, Any] = {}
    for name, points in variants.items():
        obj = np.asarray([[float(v) for v in point] for point in points], dtype=np.float64)
        if obj.shape[0] != img.shape[0]:
            raise ValueError("object and image point counts must match")
        if obj.shape[0] < MIN_REVIEWED_CORRESPONDENCES:
            raise ValueError(
                f"single-view metric calibration requires at least {MIN_REVIEWED_CORRESPONDENCES} correspondences"
            )
        parsed[name] = obj

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
        "focal length seeded by a coarse-to-fine 1D grid search minimizing solvePnP "
        "reprojection RMSE rather than seeding cv2.calibrateCamera cold, since the joint "
        "optimization is prone to bad local minima from a single view.",
        f"model selection scored on {CROSS_VALIDATION} cross-validated residual "
        "(each fold refits focal, distortion and pose on n-1 points and scores the held-out "
        "one). Training residual is NOT used to select: a lower residual on the points you "
        "fit is what extra parameters do, not evidence that they are real.",
    ]

    scoreboard: list[dict[str, Any]] = []

    # --- Step 1: object-point variant, scored with the zero-distortion model. --------
    variant_scores: dict[str, float] = {}
    for name, obj in parsed.items():
        held_out = _cross_validated_median_px(cv2, np, obj, img, width, height, cx, cy, n_radial=0)
        variant_scores[name] = held_out
        scoreboard.append(
            {
                "object_point_variant": name,
                "distortion_model": "zero_distortion",
                "held_out_median_px": held_out,
                "role": "object_point_variant_candidate",
            }
        )
    best_variant = min(variant_scores, key=lambda name: variant_scores[name])
    obj = parsed[best_variant]
    variant_notes: list[str] = []
    if len(parsed) > 1:
        runners = sorted((score, name) for name, score in variant_scores.items() if name != best_variant)
        variant_notes.append(
            f"object-point variant '{best_variant}' selected on {CROSS_VALIDATION} held-out median "
            f"{variant_scores[best_variant]:.3f}px, against "
            + ", ".join(f"'{name}' {score:.3f}px" for score, name in runners)
            + ". The pixels cannot say which world height a label source meant; the held-out "
            "residual can."
        )

    # --- Step 2: radial distortion, in increasing complexity. ------------------------
    candidates: list[tuple[str, int]] = [
        ("zero_distortion_grid_search_focal", 0),
        ("k1_radial_bounded_cv_selected", 1),
        ("k1_k2_radial_bounded_cv_selected", 2),
    ]
    fitted: dict[int, Any] = {}
    held_out_by_n: dict[int, float] = {}
    for model_name, n_radial in candidates:
        params = _fit_camera(cv2, np, obj, img, width, height, cx, cy, n_radial=n_radial)
        if params is None:
            scoreboard.append(
                {
                    "object_point_variant": best_variant,
                    "distortion_model": model_name,
                    "held_out_median_px": None,
                    "role": "distortion_candidate",
                    "note": "fit did not converge to a feasible camera",
                }
            )
            continue
        if n_radial == 0 and best_variant in variant_scores:
            held_out = variant_scores[best_variant]
        else:
            held_out = _cross_validated_median_px(cv2, np, obj, img, width, height, cx, cy, n_radial=n_radial)
        fitted[n_radial] = (model_name, params)
        held_out_by_n[n_radial] = held_out
        focal, k1, k2, _rvec, _tvec = params
        scoreboard.append(
            {
                "object_point_variant": best_variant,
                "distortion_model": model_name,
                "fx": focal,
                "k1": k1,
                "k2": k2,
                "held_out_median_px": held_out,
                "role": "distortion_candidate",
            }
        )

    if 0 not in fitted:
        raise ValueError("focal-length grid search failed to solve a camera pose for any candidate")

    accepted_n = 0
    decision_notes: list[str] = []
    for _model_name, n_radial in candidates:
        if n_radial == 0 or n_radial not in fitted:
            continue
        incumbent = held_out_by_n[accepted_n]
        challenger = held_out_by_n[n_radial]
        label = f"k1" if n_radial == 1 else "k1,k2"
        if incumbent <= _NEGLIGIBLE_ERROR_FLOOR_PX:
            decision_notes.append(
                f"{label} not tested: the incumbent's held-out median ({incumbent:.4f}px) is already "
                f"below the {_NEGLIGIBLE_ERROR_FLOOR_PX}px noise floor, so any improvement ratio would "
                "be floating-point noise, not signal."
            )
            continue
        improvement = 1.0 - (challenger / incumbent)
        if improvement >= distortion_improvement_threshold:
            decision_notes.append(
                f"{label} accepted: {CROSS_VALIDATION} held-out median {challenger:.3f}px vs "
                f"{incumbent:.3f}px for the incumbent, a {improvement:.1%} reduction, >= the "
                f"{distortion_improvement_threshold:.0%} gate required to justify the extra "
                "single-view degree(s) of freedom."
            )
            accepted_n = n_radial
        else:
            decision_notes.append(
                f"{label} rejected: {CROSS_VALIDATION} held-out median {challenger:.3f}px vs "
                f"{incumbent:.3f}px for the incumbent, only {improvement:.1%} (< the "
                f"{distortion_improvement_threshold:.0%} gate) -- not enough to justify the extra "
                "parameter(s) on 15 single-view points."
            )

    model_name, (focal, k1, k2, rvec, tvec) = fitted[accepted_n]
    max_radius = max_normalized_radius((width, height), focal, focal, cx, cy)
    if accepted_n == 0:
        model_notes = [
            "radial distortion fixed at zero (k1=k2=0): the identifiability baseline every "
            "distorted candidate has to beat on held-out residual.",
        ]
    else:
        model_notes = [
            f"radial distortion fit over bounded k1{'' if accepted_n == 1 else ',k2'} "
            f"(k1 in [{K1_BOUNDS[0]:g},{K1_BOUNDS[1]:g}]"
            + (f", k2 in [{K2_BOUNDS[0]:g},{K2_BOUNDS[1]:g}]" if accepted_n == 2 else "")
            + "), tangential and k3 fixed at zero, pose refit by solvePnP at every step.",
            f"the accepted radial model is invertible over this frame (max normalized radius "
            f"{max_radius:.3f}): it reaches every observed radius before folding back, which "
            "unconstrained cv2.calibrateCamera solutions from 15 single-view points do not.",
        ]

    for record in scoreboard:
        record["accepted"] = (
            record.get("role") == "distortion_candidate"
            and record.get("distortion_model") == model_name
        )

    fit = _build_fit(
        cv2,
        np,
        obj,
        img,
        fx=focal,
        fy=focal,
        cx=cx,
        cy=cy,
        dist=[k1, k2, 0.0, 0.0],
        rvec=rvec,
        tvec=tvec,
        distortion_model=model_name,
        notes=[*base_notes, *variant_notes, *model_notes, *decision_notes],
    )
    return replace(
        fit,
        held_out_median_px=held_out_by_n[accepted_n],
        model_selection=scoreboard,
        object_point_variant=best_variant,
        object_points_m=[[float(value) for value in point] for point in obj.tolist()],
    )


# Below this the fit is already sub-hundredth-pixel and any "improvement" ratio is
# floating-point noise, not signal.
_NEGLIGIBLE_ERROR_FLOOR_PX = 1e-3


def _pose_for(cv2: Any, np: Any, obj: Any, img: Any, focal: float, cx: float, cy: float, k1: float, k2: float):
    if not (focal > 0.0) or not math.isfinite(focal):
        return None
    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None
    if not ok:
        return None
    return k, dist, rvec, tvec


def _rms_px(cv2: Any, np: Any, obj: Any, img: Any, focal: float, cx: float, cy: float, k1: float, k2: float) -> float:
    solved = _pose_for(cv2, np, obj, img, focal, cx, cy, k1, k2)
    if solved is None:
        return math.inf
    k, dist, rvec, tvec = solved
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    residual = projected.reshape(-1, 2) - img
    value = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return value if math.isfinite(value) else math.inf


def _grid_search_focal(
    cv2: Any, np: Any, obj: Any, img: Any, width: float, height: float, cx: float, cy: float,
    *, k1: float = 0.0, k2: float = 0.0, rounds: int = 4, samples: int = 41,
) -> float | None:
    lo, hi = 0.3 * max(width, height), 6.0 * max(width, height)
    best_err, best_focal = math.inf, None
    for _ in range(rounds):
        round_err, round_focal = math.inf, None
        for focal in np.linspace(lo, hi, samples):
            err = _rms_px(cv2, np, obj, img, float(focal), cx, cy, k1, k2)
            if err < round_err:
                round_err, round_focal = err, float(focal)
        if round_focal is None:
            return best_focal
        if round_err < best_err:
            best_err, best_focal = round_err, round_focal
        span = (hi - lo) / (samples - 1)
        lo = max(1.0, round_focal - 4.0 * span)
        hi = round_focal + 4.0 * span
    return best_focal


def _feasible(k1: float, k2: float, focal: float, width: float, height: float, cx: float, cy: float) -> bool:
    if not math.isfinite(focal) or focal <= 0.0:
        return False
    if not (K1_BOUNDS[0] <= k1 <= K1_BOUNDS[1] and K2_BOUNDS[0] <= k2 <= K2_BOUNDS[1]):
        return False
    return is_radially_invertible(k1, k2, max_normalized_radius((width, height), focal, focal, cx, cy))


def _fit_camera(
    cv2: Any, np: Any, obj: Any, img: Any, width: float, height: float, cx: float, cy: float, *, n_radial: int
):
    """Return `(focal, k1, k2, rvec, tvec)` for a model with `n_radial` radial terms."""

    seed_focal = _grid_search_focal(cv2, np, obj, img, width, height, cx, cy)
    if seed_focal is None:
        return None
    if n_radial == 0:
        solved = _pose_for(cv2, np, obj, img, seed_focal, cx, cy, 0.0, 0.0)
        if solved is None:
            return None
        return (seed_focal, 0.0, 0.0, solved[2], solved[3])

    def cost(focal: float, k1: float, k2: float) -> float:
        if not _feasible(k1, k2, focal, width, height, cx, cy):
            return _INFEASIBLE_COST
        return _rms_px(cv2, np, obj, img, focal, cx, cy, k1, k2)

    # Coarse joint seed over (focal scale, k1): the (focal, k1) surface has a long
    # curved valley -- they trade off almost exactly -- so a 1D-then-1D search lands
    # in the wrong place and a cold nonlinear solve wanders out of it.
    best = (math.inf, seed_focal, 0.0)
    for scale in np.linspace(0.6, 1.8, 19):
        for k1 in np.linspace(K1_BOUNDS[0], K1_BOUNDS[1], 19):
            value = cost(float(seed_focal * scale), float(k1), 0.0)
            if value < best[0]:
                best = (value, float(seed_focal * scale), float(k1))
    if not math.isfinite(best[0]) or best[0] >= _INFEASIBLE_COST:
        return None

    start = [best[1], best[2]] + ([0.0] if n_radial == 2 else [])
    refined = _refine_parameters(start, cost, n_radial, width, height)
    focal, k1 = float(refined[0]), float(refined[1])
    k2 = float(refined[2]) if n_radial == 2 else 0.0
    if not _feasible(k1, k2, focal, width, height, cx, cy):
        focal, k1, k2 = best[1], best[2], 0.0
    solved = _pose_for(cv2, np, obj, img, focal, cx, cy, k1, k2)
    if solved is None:
        return None
    return (focal, k1, k2, solved[2], solved[3])


_INFEASIBLE_COST = 1e12


def _refine_parameters(start: list[float], cost: Any, n_radial: int, width: float, height: float) -> list[float]:
    """Bounded local refinement of (focal, k1[, k2]) around a coarse seed."""

    bounds = [(0.2 * max(width, height), 8.0 * max(width, height)), K1_BOUNDS]
    if n_radial == 2:
        bounds.append(K2_BOUNDS)

    def objective(vector: Sequence[float]) -> float:
        k2 = float(vector[2]) if n_radial == 2 else 0.0
        return cost(float(vector[0]), float(vector[1]), k2)

    try:
        from scipy.optimize import minimize  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return _coordinate_descent(start, objective, bounds)
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 4000, "maxfev": 4000},
    )
    if not math.isfinite(float(result.fun)) or float(result.fun) >= _INFEASIBLE_COST:
        return list(start)
    return [float(value) for value in result.x]


def _coordinate_descent(start: list[float], objective: Any, bounds: list[tuple[float, float]]) -> list[float]:
    """scipy-free fallback: shrinking coordinate-wise line search."""

    current = list(start)
    best = objective(current)
    steps = [max(1e-3, 0.05 * abs(value) if index == 0 else 0.05) for index, value in enumerate(current)]
    for _ in range(60):
        improved = False
        for index in range(len(current)):
            for direction in (1.0, -1.0):
                trial = list(current)
                trial[index] = min(bounds[index][1], max(bounds[index][0], trial[index] + direction * steps[index]))
                value = objective(trial)
                if value < best:
                    best, current, improved = value, trial, True
        if not improved:
            steps = [step * 0.5 for step in steps]
            if max(steps) < 1e-6:
                break
    return current


def _cross_validated_median_px(
    cv2: Any, np: Any, obj: Any, img: Any, width: float, height: float, cx: float, cy: float, *, n_radial: int
) -> float:
    """Leave-one-out held-out median reprojection residual, in pixels.

    Each fold refits focal, distortion and pose on the other n-1 correspondences and
    scores the one it never saw. This is the number model selection uses; the training
    residual reported in `reprojection_error_px` is descriptive only.
    """

    count = int(obj.shape[0])
    held_out: list[float] = []
    for index in range(count):
        keep = [row for row in range(count) if row != index]
        params = _fit_camera(cv2, np, obj[keep], img[keep], width, height, cx, cy, n_radial=n_radial)
        if params is None:
            held_out.append(math.inf)
            continue
        focal, k1, k2, rvec, tvec = params
        k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
        projected, _ = cv2.projectPoints(obj[index : index + 1], rvec, tvec, k, dist)
        held_out.append(float(np.linalg.norm(projected.reshape(2) - img[index])))
    return _median(held_out)


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

    declared_points = [list(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m) for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
    image_points = [list(native_points[name]) for name in PICKLEBALL_COURT_KEYPOINT_NAMES]

    fit = fit_single_view_metric_camera(
        declared_points,
        image_points,
        native_size,
        distortion_improvement_threshold=distortion_improvement_threshold,
        object_point_variants=_net_label_height_variants(declared_points),
    )
    object_points = fit.object_points_m or declared_points

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
    camera_to_world_R, _camera_center_canonical = invert_extrinsics(
        extrinsics.R,
        extrinsics.t,
    )

    geometry = CameraFloorGeometry(
        intrinsics={"fx": fit.fx, "fy": fit.fy, "cx": fit.cx, "cy": fit.cy},
        camera_origin_world=camera_center_world,
        R_world_camera=camera_to_world_R.tolist(),
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

    metric_confidence = _confidence_from_reprojection(
        fit.reprojection_error_px, held_out_median_px=fit.held_out_median_px
    )

    base_quality = score_capture_quality(
        corners_visible=len(PICKLEBALL_COURT_KEYPOINT_NAMES),
        reprojection_rmse_px=fit.reprojection_error_px.median,
    )
    net_height = _net_label_height_of(object_points)
    provenance_reasons = [
        f"distortion_model={fit.distortion_model}",
        f"net_keypoint_label_height_m={net_height:.4f}",
        f"net_keypoint_label_height_variant={fit.object_point_variant}",
    ]
    if fit.held_out_median_px is not None:
        provenance_reasons.append(
            f"model_selected_on_{CROSS_VALIDATION}_held_out_median_{fit.held_out_median_px:.3f}px"
        )
    reasons = list(
        dict.fromkeys(
            [
                *base_quality.reasons,
                "single_view_planar_full_calibration",
                *provenance_reasons,
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
        "coordinate_contract": {
            "camera_matrix_K": camera_matrix_from_intrinsics(intrinsics),
            "camera_matrix_input_space": "camera_m",
            "camera_matrix_output_space": "pixels_undistorted_native",
            "extrinsics_convention": "world_to_camera_opencv_column",
            "extrinsics_input_space": "world_court_netcenter_z_up_m",
            "extrinsics_output_space": "camera_m",
            "homography_convention": "world_xy_to_image_column",
            "homography_input_space": "world_xy_homography_m",
            "homography_output_space": "pixels_raw_native",
            "homography_pixel_convention": "raw_pixels",
        },
    }
    return CourtCalibration.model_validate(payload)


_CONFIDENCE_ORDER = ("low", "med", "high")


def _confidence_from_reprojection(
    error: ReprojectionError, *, held_out_median_px: float | None = None
) -> str:
    """Confidence from the training residual, capped by what held-out data supports.

    The training thresholds are unchanged, so before/after comparisons stay
    apples-to-apples. The cap can only ever lower the grade: a fit that looks tight
    on the 15 points it was fit to, but predicts a held-out point badly, has not
    earned the higher label.
    """

    if error.median <= 2.0 and error.p95 <= 5.0:
        grade = "high"
    elif error.median <= 6.0 and error.p95 <= 15.0:
        grade = "med"
    else:
        grade = "low"
    if held_out_median_px is None or not math.isfinite(held_out_median_px):
        return grade
    if held_out_median_px <= 2.0:
        cap = "high"
    elif held_out_median_px <= 6.0:
        cap = "med"
    else:
        cap = "low"
    return min(grade, cap, key=_CONFIDENCE_ORDER.index)


def _net_label_height_variants(
    declared_points: Sequence[Sequence[float]],
) -> dict[str, list[list[float]]]:
    """The competing world-height hypotheses for the 3 net keypoints.

    `court_keypoint_net` declares them at the net top; every reviewed label set in this
    repo marks the ground net line instead. Which one a given label source meant is not
    recoverable from the pixels, so both are offered and the held-out residual decides.
    """

    indexes = [
        index
        for index, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES)
        if name in NET_KEYPOINT_NAMES
    ]
    declared = [[float(value) for value in point] for point in declared_points]
    if not indexes:
        return {"as_declared": declared}
    declared_height = declared[indexes[0]][2]
    ground = [list(point) for point in declared]
    for index in indexes:
        ground[index][2] = 0.0
    if declared_height == 0.0:
        return {"ground_net_line": ground}
    return {"net_top_as_declared": declared, "ground_net_line": ground}


def _net_label_height_of(object_points: Sequence[Sequence[float]]) -> float:
    for index, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES):
        if name in NET_KEYPOINT_NAMES and index < len(object_points):
            return float(object_points[index][2])
    return 0.0


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
