"""BODY-inference input adapter for the ASPset-510 external-GT validation lane.

Scoped by `runs/body_external_gt_20260702T040048Z/BODY_INFERENCE_INTEGRATION_TODO.md`
("Concrete adapter plan" section). This module implements steps 1-2 of that plan: turning
ASPset-510's own real, measured camera calibration and real GT joints into the two
artifacts `scripts/racketsport/run_body_video_smoke.py` / `threed.racketsport.orchestrator`
need to run the BODY stage against non-court, single-subject footage:

1. `court_calibration.json` (`threed.racketsport.schemas.CourtCalibration`) -- built
   directly from ASPset-510's real per-(subject, camera) `intrinsic_matrix`/
   `extrinsic_matrix` (see `raw_provenance/cameras/<subject>-<camera>.json`). This is
   real, better-measured calibration than this project's own guessed-FOV pickleball
   `court_calibration.json` artifacts (see `DATASET_SELECTION.md`), and -- critically --
   it is expressed in **exactly the same world frame ASPset-510's own GT joints already
   use** (confirmed empirically: projecting a `mid`/`right`-camera clip's GT joints
   through that camera's own real extrinsics lands them in-frame, near image center, at a
   physically sensible ~19-20m depth; see the adapter's own test coverage). Feeding this
   real calibration into the pipeline's world-grounding step is what promotes the
   `external_gt_alignment.mpjpe` variant from "theoretical" (per `METHODOLOGY.md` §3) to
   a real, reportable number -- predicted and GT joints should now genuinely share one
   world frame, without requiring any data-fitted alignment.

2. `tracks.json` (`threed.racketsport.schemas.Tracks`) -- a single-player track (ASPset
   test-split clips are single-subject) with one `TrackFrame` per GT-sampled frame index
   (matching `build_external_gt_aspset510_labels.py`'s frame-stride-10 sampling), whose
   `bbox` is derived by projecting that frame's own 12 shared-core GT joints through the
   same real camera calibration and padding the resulting 2D extent. This is an honest,
   disclosed scheduling input, not prediction assistance: it tells the tracker/scheduler
   *where in the frame* the one real subject is (a real per-camera bounding box would do
   the same job; ASPset's own `test-boxes` archive provides exactly that, but this GT-
   joint-derived box avoids needing a second raw-data download and is provably no more
   informative than the boxes archive would be -- both come from the same real mocap/rig,
   neither from a BODY model prediction). It never feeds joint *positions* to the BODY
   model; the model only ever sees a bounding crop, exactly as it would from any other
   real detector/tracker.

Units: ASPset-510's raw `extrinsic_matrix` translation is in millimeters (confirmed in
`DATASET_SELECTION.md`); this module converts to meters immediately at load time so every
public function here operates in this project's metric convention (matching
`external_gt_aspset510.MILLIMETERS_PER_METER`).

Camera convention (confirmed against `threed.racketsport.court_calibration.
project_world_points`, the only other place in this codebase that consumes
`CourtExtrinsics.R`/`.t` for a full 3D world->image projection): ``camera_xyz = R @
world_xyz + t`` (standard extrinsic world-to-camera transform) -- i.e. ASPset-510's raw
`extrinsic_matrix` can be used directly as `CourtExtrinsics.R`/`.t` with no transpose or
inversion.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from threed.racketsport.external_gt_aspset510 import SHARED_CORE_JOINT_NAMES

ARTIFACT_TYPE_TRACKS = "racketsport_tracks"
ARTIFACT_TYPE_COURT_CALIBRATION = "racketsport_court_calibration"
SCHEMA_VERSION = 1
DEFAULT_FPS = 50.0
DEFAULT_IMAGE_SIZE: tuple[int, int] = (3840, 2160)
MILLIMETERS_PER_METER = 1000.0
# Bounding-box padding: the 12 shared-core joints exclude head/face and toe/heel extent,
# so the raw joint-extent box is strictly smaller than the subject's real silhouette.
# These paddings are generous, disclosed scheduling-input heuristics, not measurements.
BBOX_PAD_RATIO = 0.35
BBOX_PAD_PX_MIN = 40.0
BBOX_HEAD_PAD_RATIO = 0.55  # extra top padding for head/head_top, not in the shared set
_LEFT_ANKLE_INDEX = SHARED_CORE_JOINT_NAMES.index("left_ankle")
_RIGHT_ANKLE_INDEX = SHARED_CORE_JOINT_NAMES.index("right_ankle")


class BodyInputAdapterError(RuntimeError):
    """Raised when real ASPset-510 calibration/joint data cannot be adapted safely."""


def load_camera_calibration_raw(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ASPset-510's raw per-(subject, camera) calibration JSON.

    Returns ``(K, R, t_m)``: 3x3 intrinsic matrix, 3x3 world-to-camera rotation, and the
    3-vector world-to-camera translation **converted to meters**.
    """

    intrinsic = np.asarray(payload["intrinsic_matrix"], dtype=np.float64).reshape(3, 4)
    extrinsic = np.asarray(payload["extrinsic_matrix"], dtype=np.float64).reshape(4, 4)
    K = intrinsic[:, :3]
    R = extrinsic[:3, :3]
    t_mm = extrinsic[:3, 3]
    t_m = t_mm / MILLIMETERS_PER_METER
    return K, R, t_m


def project_world_points(points_m: np.ndarray, *, K: np.ndarray, R: np.ndarray, t_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world-frame meter points into pixel coordinates.

    Returns ``(uv, depth)``; ``uv`` is (N, 2) pixel coordinates, ``depth`` is the (N,)
    camera-space Z (must be > 0 for a point to be in front of the camera / a valid
    projection).
    """

    points_m = np.asarray(points_m, dtype=np.float64)
    camera = points_m @ R.T + t_m
    depth = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * camera[:, 0] / depth + K[0, 2]
        v = K[1, 1] * camera[:, 1] / depth + K[1, 2]
    return np.stack([u, v], axis=1), depth


def build_court_calibration_payload(
    *,
    K: np.ndarray,
    R: np.ndarray,
    t_m: np.ndarray,
    source_label: str,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    reference_plane_z_m: float = 0.0,
    reference_plane_center_xy_m: tuple[float, float] = (0.0, 0.0),
    reference_plane_half_extent_m: float = 1.0,
) -> dict[str, Any]:
    """Build a `court_calibration.json`-shaped payload from real ASPset-510 calibration.

    The homography is the standard planar homography for the world frame's own Z =
    ``reference_plane_z_m`` plane (``H = K @ [r1 | r2 | t']``, where ``r1``/``r2`` are the
    first two columns of ``R`` and ``t' = R @ [0,0,reference_plane_z_m] + t_m``) --
    mathematically well-defined for *any* such plane in this world frame, regardless of
    whether it corresponds to a physical "floor" in this rig (it does not necessarily,
    since ASPset-510's own world frame is camera-rig-fixed -- specifically the `left`
    camera's own frame for a given subject's rig -- not floor-referenced; see this
    module's docstring). ``image_pts``/``world_pts`` are 4 points synthesized directly on
    that same plane via the same projection equations, so ``reprojection_error_px`` is
    exactly (not approximately) zero by construction.

    ``reference_plane_z_m`` defaults to 0.0 (the world origin's own depth plane), which is
    fine for `mid`/`right` cameras (whose own centers sit tens of meters from Z=0 in this
    shared world frame) but is **degenerate for the `left` camera itself**: `left`'s own
    extrinsic is the identity (it *defines* this world frame; see
    `DATASET_SELECTION.md`), so Z=0 is the plane running exactly through `left`'s own
    camera center, which cannot support a finite planar homography. Callers building
    calibration for a `left`-camera clip must pass a non-zero ``reference_plane_z_m``
    (e.g. that clip's own real median GT-joint depth) so the plane is nowhere near any of
    the rig's three camera centers.
    """

    plane_origin = R @ np.array([0.0, 0.0, reference_plane_z_m]) + t_m
    r1, r2 = R[:, 0], R[:, 1]
    H = np.column_stack([r1, r2, plane_origin])
    H = K @ H
    if abs(H[2, 2]) < 1e-9:
        raise BodyInputAdapterError(
            f"{source_label}: degenerate homography (H[2,2] ~= 0) at reference_plane_z_m="
            f"{reference_plane_z_m}; this plane passes through (or too near) the camera "
            "center -- pass a different reference_plane_z_m"
        )
    H = H / H[2, 2]

    cx, cy = reference_plane_center_xy_m
    e = reference_plane_half_extent_m
    world_pts_ref_plane = np.array(
        [
            [cx - e, cy - e, reference_plane_z_m],
            [cx + e, cy - e, reference_plane_z_m],
            [cx + e, cy + e, reference_plane_z_m],
            [cx - e, cy + e, reference_plane_z_m],
        ]
    )
    image_pts_arr, depth = project_world_points(world_pts_ref_plane, K=K, R=R, t_m=t_m)
    if np.any(depth <= 0):
        raise BodyInputAdapterError(
            f"{source_label}: synthesized reference-plane calibration points project behind "
            "the camera (depth <= 0); pick a different reference_plane_z_m/center for this "
            "camera pose"
        )

    camera_origin_world = -(R.T @ t_m)
    camera_height_m = max(float(np.linalg.norm(camera_origin_world)), 0.01)

    # NOTE: `CourtCalibration.source` (and its sibling "metric calibration" fields
    # coordinate_frame/T_world_court/per_keypoint_residual_px/metric_confidence/
    # gsd_model/solved_over_frames) are deliberately left unset (all-None). Setting
    # any one of them makes `threed.racketsport.schemas.CourtCalibration`'s
    # `_point_lists_must_be_paired` validator require *all* of them, including
    # `coordinate_frame: Literal["court_netcenter_z_up_m"]` -- a claim that this
    # calibration is expressed in this project's pickleball net-centered court
    # frame, which would be actively false for ASPset-510's own rig-fixed world
    # frame. The real provenance is instead recorded in `capture_quality.reasons`
    # (free-form strings), which makes no structured schema claim.
    return {
        "schema_version": SCHEMA_VERSION,
        "sport": "pickleball",
        "capture_quality": {
            "grade": "warn",
            "reasons": [
                "external_ground_truth_non_pickleball_scene",
                f"real_measured_multiview_calibration_source={source_label}",
                "sport_field_forced_pickleball_enum_schema_compatibility_only",
                "coordinate_frame_is_aspset510_rig_world_frame_not_pickleball_court_netcenter",
            ],
        },
        "homography": H.tolist(),
        "intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "dist": [],
            "source": "aspset510_real_camera_calibration",
        },
        "image_size": list(image_size),
        "extrinsics": {
            "R": R.tolist(),
            "t": t_m.tolist(),
            "camera_height_m": camera_height_m,
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "image_pts": image_pts_arr.tolist(),
        "world_pts": world_pts_ref_plane.tolist(),
    }


def build_tracks_payload(
    *,
    frame_indices: Sequence[int],
    joints_world_m_by_frame: Sequence[Sequence[Sequence[float]]],
    K: np.ndarray,
    R: np.ndarray,
    t_m: np.ndarray,
    fps: float = DEFAULT_FPS,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    player_id: int = 1,
    side: str = "near",
    role: str = "single",
) -> tuple[dict[str, Any], list[str]]:
    """Build a single-player `tracks.json`-shaped payload from real per-frame GT joints.

    Each frame's bbox is the padded 2D extent of that frame's 12 shared-core-joint GT
    positions, projected through the real camera calibration -- see this module's
    docstring for why this is a legitimate scheduling input, not prediction assistance.
    Frames whose GT joints project behind the camera (depth <= 0, should not happen for
    real captured footage but is checked defensively) are dropped with a note rather than
    silently emitting a nonsense bbox.
    """

    if len(frame_indices) != len(joints_world_m_by_frame):
        raise BodyInputAdapterError(
            f"frame_indices length {len(frame_indices)} != joints_world_m_by_frame length "
            f"{len(joints_world_m_by_frame)}"
        )
    width, height = image_size
    frames: list[dict[str, Any]] = []
    notes: list[str] = []
    for frame_idx, joints in zip(frame_indices, joints_world_m_by_frame):
        joints_arr = np.asarray(joints, dtype=np.float64)
        if joints_arr.shape != (len(SHARED_CORE_JOINT_NAMES), 3):
            raise BodyInputAdapterError(
                f"frame {frame_idx}: expected ({len(SHARED_CORE_JOINT_NAMES)}, 3) joints, got {joints_arr.shape}"
            )
        uv, depth = project_world_points(joints_arr, K=K, R=R, t_m=t_m)
        if np.any(depth <= 0):
            notes.append(f"frame {frame_idx}: dropped (GT joints project behind camera, depth<=0)")
            continue

        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)
        w, h = x2 - x1, y2 - y1
        pad_x = max(w * BBOX_PAD_RATIO, BBOX_PAD_PX_MIN)
        pad_y = max(h * BBOX_PAD_RATIO, BBOX_PAD_PX_MIN)
        head_pad = max(h * BBOX_HEAD_PAD_RATIO, BBOX_PAD_PX_MIN)
        x1 = max(0.0, x1 - pad_x)
        x2 = min(float(width), x2 + pad_x)
        y1 = max(0.0, y1 - head_pad)  # extra headroom: head/head_top are not in the shared-core set
        y2 = min(float(height), y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            notes.append(f"frame {frame_idx}: dropped (degenerate padded bbox after clipping to image bounds)")
            continue

        ankle_xy = joints_arr[[_LEFT_ANKLE_INDEX, _RIGHT_ANKLE_INDEX], :2].mean(axis=0)
        frames.append(
            {
                "t": round(float(frame_idx) / fps, 9),
                "bbox": [round(float(x1), 3), round(float(y1), 3), round(float(x2), 3), round(float(y2), 3)],
                "world_xy": [round(float(ankle_xy[0]), 6), round(float(ankle_xy[1]), 6)],
                "conf": 1.0,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "fps": float(fps),
        "players": [
            {
                "id": int(player_id),
                "side": side,
                "role": role,
                "frames": frames,
            }
        ],
        "rally_spans": [],
    }
    return payload, notes


__all__ = [
    "BodyInputAdapterError",
    "DEFAULT_FPS",
    "DEFAULT_IMAGE_SIZE",
    "MILLIMETERS_PER_METER",
    "build_court_calibration_payload",
    "build_tracks_payload",
    "load_camera_calibration_raw",
    "project_world_points",
]
