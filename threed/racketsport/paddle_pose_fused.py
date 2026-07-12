"""Fused wrist+palm+grip-transform render-only paddle 6-DOF estimator.

This module implements the "grip-transform fused paddle 6-DOF estimator" design from
``runs/archive/root_docs_20260709/RACKET_6DOF_GOAL.md`` / ``runs/lanes/racket_6dof_20260705/i1_fused_estimator/spec.md``.
It is the successor evidence channel to :mod:`threed.racketsport.paddle_proxy` (which stays
untouched and remains the fallback): instead of a crude forearm-axis + world-up face normal,
this solves for a **hand frame** ``H(t)`` from the raw MHR70 wrist+finger joints every frame,
and a **grip transform** ``G`` (hand -> paddle SE3, held constant per grip segment) fit by a
robust, weighted vector-alignment (Wahba's problem) plus a translation smoothness term, with
optional evidence from ball-reflection contacts and wrist-gated detector boxes.

Like ``paddle_proxy.py`` this module does not estimate or claim true paddle 6DoF: outputs are
render-only, ``estimated``-class trust, and never promote the RKT gate. It matches the
``racket_pose_estimate.json`` artifact contract exactly so ``virtual_world.py`` (unmodified)
consumes it identically to the wrist proxy.

Design notes (why these choices; see spec.md THE DESIGN section for the required shape):

* **Hand frame H(t)** per frame: origin = selected wrist; Y axis (grip axis) = wrist -> middle
  finger proximal ("third") joint, falling back to wrist -> elbow when finger data is degenerate;
  Z-candidate (palm normal) = a chirality-tagged cross of (index_third - wrist) x (pinky_third -
  wrist), continuity sign-locked frame to frame, falling back to a world-up cross when finger data
  is degenerate. The frame is completed via Gram-Schmidt (X = normalize(Y x Z), then
  Z = X x Y) so H(t).R is always a proper (det=+1) orthonormal rotation.

* **Grip transform G = (R_g, t_g)**, hand -> paddle, held constant per grip segment. Composing a
  CONSTANT rotation with H(t) leaves the *magnitude* of frame-to-frame rotation jitter invariant
  (conjugation by a fixed rotation preserves rotation angle) -- so G's ROTATION cannot be
  identified from H(t)'s own smoothness alone. Instead:
    - R_g is solved via a weighted Wahba/Kabsch vector-alignment: a soft prior anchor pulls R_g
      toward the neutral continental-grip prior (paddle Y == hand Y, paddle Z == hand Z), and
      real evidence (ball-reflection face normals in world space, inverse-uncertainty weighted)
      pulls it away when available. This is genuinely identifying because the observed vectors
      come from an EXTERNAL reference (the world-frame ball reflection normal), not from H(t)
      itself.
    - t_g (the grip offset, hand-local) IS identifiable from H(t)'s own smoothness: a translation
      lever-arm amplifies rotational jitter in H(t) into positional jitter in the composed pose,
      so t_g is fit by a closed-form weighted least squares that prefers directions where H(t)'s
      rotation is quietest, anchored to the physical continental-grip offset prior.
    - Detector boxes (optional, wrist-gated) contribute a small additive correction to t_g via a
      coarse pixel-to-world reprojection nudge.

* **Grip segments**: the whole rally is one segment unless a *sustained* (>=2s hysteresis)
  deviation of the raw palm-normal candidate from the segment's reference direction is observed.
  This is a coarse residual-break monitor, not a rigorous swing-vs-regrip classifier (ordinary
  swing motion can itself exceed the break angle transiently); the >=2s hold requirement and a
  generous break angle keep it from flapping on ordinary strokes while still catching gross,
  sustained reorientation (e.g. a hand switch).

* **Temporal smoothing** is done in SO(3) properly: a SLERP-based one-euro filter (adaptive
  cutoff driven by angular speed, applied as a partial-rotation step composed onto the running
  smoothed rotation) rather than a naive per-axis EMA -- axis-EMA breaks unit-norm/orthonormality
  between steps and is exactly the "axis-EMA jitter disease" the baseline proxy suffers from.

* **Provenance bands** (existing schema fields, no new ones): ``contact_locked`` (frame near a
  ball-reflection contact that was actually used to fit R_g), ``palm_fitted`` (full finger-based
  hand frame available this frame), ``grip_extrapolated`` (fingers degenerate this frame; wrist +
  constant G only). Fail-closed: no wrist -> no frame for that player/frame; membership-excluded
  players -> no paddle frames at all.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .coordinates import (
    CANONICAL_COURT_WORLD_FRAME,
    LEGACY_COURT_WORLD_FRAME,
    CoordinateSpace,
    project_world_points as project_world_points_typed,
    resolve_world_coordinate_space,
)
from .external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from .paddle_proxy import (
    DEFAULT_GRIP_OFFSET_M,
    DEFAULT_MIN_JOINT_CONFIDENCE,
    TRUST as _WRIST_PROXY_TRUST,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_racket_pose_estimate"
SOURCE = "wrist_palm_grip_fused"
# Reuse the wrist-proxy TRUST literal (not just its value): virtual_world.py's
# PADDLE_PROXY_TRUST_STATUSES = {"estimated_from_wrist"} is the only "estimated"-tier status it
# recognizes; this estimator's output stays estimated-class per spec, so it deliberately rides the
# same recognized status rather than inventing a new one virtual_world would silently drop.
TRUST = _WRIST_PROXY_TRUST
WORLD_FRAME = LEGACY_COURT_WORLD_FRAME
WORLD_COORDINATE_FRAME = CANONICAL_COURT_WORLD_FRAME
WORLD_COORDINATE_SPACE = CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M
DEFAULT_HANDLE_LENGTH_IN = 5.25
# Regulation-common paddle dims (16" x 8" is the ubiquitous production shape; the proxy's
# 15.5" x 7.5" was conservative). Physical constants, applied identically to every clip.
DEFAULT_PADDLE_DIMS_IN = {"length": 16.0, "width": 8.0}

BAND_CONTACT_LOCKED = "contact_locked"
BAND_PALM_FITTED = "palm_fitted"
BAND_GRIP_EXTRAPOLATED = "grip_extrapolated"
_VALID_BANDS = (BAND_CONTACT_LOCKED, BAND_PALM_FITTED, BAND_GRIP_EXTRAPOLATED)
_BAND_BASE_CONF = {
    BAND_CONTACT_LOCKED: 0.75,
    BAND_PALM_FITTED: 0.55,
    BAND_GRIP_EXTRAPOLATED: 0.30,
}


def _project_world_points_for_raw_detector_reference(
    extrinsics: Any,
    intrinsics: Any,
    world_points: Any,
) -> list[list[float]]:
    """Project unchanged pinhole pixels while declaring raw detector reference evidence."""

    return project_world_points_typed(
        extrinsics,
        intrinsics,
        world_points,
        input_space=WORLD_COORDINATE_SPACE,
        output_space=CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
        reference_space=CoordinateSpace.PIXELS_RAW_NATIVE,
    )


DEFAULT_MIN_SEGMENT_DURATION_S = 2.0
DEFAULT_SEGMENT_BREAK_ANGLE_DEG = 60.0
DEFAULT_PRIOR_ROTATION_WEIGHT = 0.4
DEFAULT_PRIOR_TRANSLATION_WEIGHT = 6.0
DEFAULT_REFLECTION_WEIGHT_SCALE = 8.0
DEFAULT_REFLECTION_MAX_TIME_GAP_S = 0.12
DEFAULT_DETECTOR_BOX_WRIST_GATE_RADIUS_PX = 130.0
DEFAULT_DETECTOR_BOX_MAX_CORRECTION_M = 0.30
DEFAULT_DETECTOR_BOX_ROLL_SEARCH_DEG = 90.0
DEFAULT_DETECTOR_BOX_ROLL_STEPS = 25
DEFAULT_DETECTOR_BOX_ROLL_MIN_GAIN = 0.01
DEFAULT_DETECTOR_BOX_ROLL_MAX_FRAMES = 60
DEFAULT_PER_FRAME_DEVIATION_MAX_M = 0.25
DEFAULT_PER_FRAME_DEVIATION_DECAY = 0.85
DEFAULT_PER_FRAME_DEVIATION_SLEW_M = 0.01
DEFAULT_HAND_SWITCH_MIN_HOLD_S = 4.0
DEFAULT_HAND_SWITCH_MIN_VOTES = 20
DEFAULT_HAND_SWITCH_MAJORITY = 0.8
DEFAULT_MAX_POSITION_JUMP_M_PER_FRAME = 0.30
DEFAULT_ONE_EURO_MIN_CUTOFF = 0.6
DEFAULT_ONE_EURO_BETA = 0.15
DEFAULT_ONE_EURO_D_CUTOFF = 1.0
# Position gets a much heavier filter than rotation: SAM3D wrist world positions carry large
# frame-to-frame (mostly depth-axis) noise -- raw wrist |dpos| medians imply implausible 1.5-4 m/s
# sustained median speeds -- while the image-plane signal the scorer sees is far cleaner.
DEFAULT_POSITION_ONE_EURO_MIN_CUTOFF = 0.8
DEFAULT_POSITION_ONE_EURO_BETA = 0.2
DEFAULT_CONTACT_LOCK_WINDOW_S = 0.12


def _joint_index(name: str) -> int:
    return MHR70_JOINT_NAMES.index(name)


RAW_JOINT_COUNT = len(MHR70_JOINT_NAMES)

# Chirality tag: for a right hand, cross(index-wrist, pinky-wrist) points to one physical side of
# the palm; for a mirrored left hand the same construction points to the opposite physical side.
# Flipping the sign for the left hand keeps the RAW palm-normal candidate on a consistent physical
# side for both hands, so a single grip-transform prior (R_g ~ Identity) is plausible for either
# hand without baking a per-side sign flip into G itself.
_SIDE_JOINTS: dict[str, dict[str, Any]] = {
    "right": {
        "wrist": _joint_index("right_wrist"),
        "elbow": _joint_index("right_elbow"),
        "middle_third": _joint_index("right_middle_third_joint"),
        "index_third": _joint_index("right_index_third_joint"),
        "pinky_third": _joint_index("right_pinky_third_joint"),
        "chirality": 1.0,
    },
    "left": {
        "wrist": _joint_index("left_wrist"),
        "elbow": _joint_index("left_elbow"),
        "middle_third": _joint_index("left_middle_third_joint"),
        "index_third": _joint_index("left_index_third_joint"),
        "pinky_third": _joint_index("left_pinky_third_joint"),
        "chirality": -1.0,
    },
}


# --------------------------------------------------------------------------------------
# Internal per-frame representations
# --------------------------------------------------------------------------------------


@dataclass
class _HandFrame:
    t: float
    frame_idx: int | None
    wrist: np.ndarray
    rotation: np.ndarray  # 3x3 proper rotation; columns = hand X, Y (grip axis), Z (palm normal)
    z_candidate_raw: np.ndarray  # pre-orthogonalization palm-normal candidate, for segment QC
    joint_confidence: float
    used_finger_grip_axis: bool
    used_finger_palm_normal: bool


@dataclass
class _ComposedFrame:
    t: float
    frame_idx: int | None
    rotation: np.ndarray
    translation: np.ndarray
    used_finger_palm_normal: bool
    joint_confidence: float
    hand_side: str = "right"


# --------------------------------------------------------------------------------------
# Hand-frame construction
# --------------------------------------------------------------------------------------


def _build_hand_frames(
    frames_in: Sequence[Any],
    *,
    side: str,
    min_joint_confidence: float,
) -> tuple[list[_HandFrame], list[dict[str, Any]]]:
    cfg = _SIDE_JOINTS[side]
    samples: list[_HandFrame] = []
    hidden: list[dict[str, Any]] = []
    previous_z_candidate: np.ndarray | None = None

    ordered = sorted(
        (item for item in frames_in if isinstance(item, Mapping)),
        key=lambda item: float(item.get("t", 0.0)),
    )
    for frame in ordered:
        t = float(frame.get("t", 0.0))
        frame_idx = _frame_index(frame)
        hidden_base = {"frame_idx": frame_idx, "t": t, "side": side}
        joints = frame.get("joints_world")
        if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or len(joints) < RAW_JOINT_COUNT:
            hidden.append({**hidden_base, "reason": "insufficient_raw_joint_count", "joint_confidence": 0.0})
            previous_z_candidate = None
            continue
        wrist = _vec3(joints[cfg["wrist"]])
        conf = _joint_conf(frame.get("joint_conf"), cfg["wrist"])
        if wrist is None or conf < min_joint_confidence:
            hidden.append({**hidden_base, "reason": "missing_or_low_confidence_wrist", "joint_confidence": round(conf, 6)})
            previous_z_candidate = None
            continue
        elbow = _vec3(joints[cfg["elbow"]])
        if elbow is None:
            hidden.append({**hidden_base, "reason": "missing_elbow", "joint_confidence": round(conf, 6)})
            previous_z_candidate = None
            continue

        middle = _vec3(joints[cfg["middle_third"]])
        index_pt = _vec3(joints[cfg["index_third"]])
        pinky = _vec3(joints[cfg["pinky_third"]])

        used_finger_grip_axis = False
        y_axis: np.ndarray | None = None
        if middle is not None:
            candidate = middle - wrist
            norm = float(np.linalg.norm(candidate))
            if norm > 1e-4:
                y_axis = candidate / norm
                used_finger_grip_axis = True
        if y_axis is None:
            candidate = wrist - elbow
            norm = float(np.linalg.norm(candidate))
            if norm <= 1e-9:
                hidden.append({**hidden_base, "reason": "degenerate_grip_axis", "joint_confidence": round(conf, 6)})
                previous_z_candidate = None
                continue
            y_axis = candidate / norm

        used_finger_palm_normal = False
        z_candidate: np.ndarray | None = None
        if index_pt is not None and pinky is not None:
            raw = np.cross(index_pt - wrist, pinky - wrist) * cfg["chirality"]
            norm = float(np.linalg.norm(raw))
            if norm > 1e-6:
                z_candidate = raw / norm
                used_finger_palm_normal = True
        if z_candidate is None:
            z_candidate = _face_normal_fallback(y_axis)

        if previous_z_candidate is not None and float(np.dot(z_candidate, previous_z_candidate)) < 0.0:
            z_candidate = -z_candidate
        previous_z_candidate = z_candidate

        rotation = _orthonormal_frame_from_y_z(y_axis, z_candidate)

        samples.append(
            _HandFrame(
                t=t,
                frame_idx=frame_idx,
                wrist=wrist,
                rotation=rotation,
                z_candidate_raw=z_candidate,
                joint_confidence=conf,
                used_finger_grip_axis=used_finger_grip_axis,
                used_finger_palm_normal=used_finger_palm_normal,
            )
        )
    return samples, hidden


def _face_normal_fallback(y_axis: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    candidate = np.cross(y_axis, up)
    if float(np.linalg.norm(candidate)) <= 1e-6:
        candidate = np.cross(y_axis, np.array([0.0, 1.0, 0.0]))
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-9:
        candidate = _arbitrary_perpendicular(y_axis)
        return candidate
    return candidate / norm


def _orthonormal_frame_from_y_z(y_axis: np.ndarray, z_candidate: np.ndarray) -> np.ndarray:
    x_axis = np.cross(y_axis, z_candidate)
    norm_x = float(np.linalg.norm(x_axis))
    if norm_x <= 1e-9:
        x_axis = _arbitrary_perpendicular(y_axis)
    else:
        x_axis = x_axis / norm_x
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / float(np.linalg.norm(z_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _arbitrary_perpendicular(axis: np.ndarray) -> np.ndarray:
    reference = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    candidate = np.cross(axis, reference)
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-9:
        reference = np.array([0.0, 0.0, 1.0])
        candidate = np.cross(axis, reference)
        norm = float(np.linalg.norm(candidate))
    return candidate / norm


# --------------------------------------------------------------------------------------
# Hand-side (left/right) auto-selection -- self-contained, mirrors paddle_proxy's
# auto-side-score heuristic but reads the RAW 70-joint array directly (no semantic remap,
# since the semantic map used by paddle_proxy drops finger joints entirely).
# --------------------------------------------------------------------------------------


def _hand_side_scores(frames_in: Sequence[Any], *, min_joint_confidence: float) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for side in ("right", "left"):
        cfg = _SIDE_JOINTS[side]
        usable = 0
        confidence_sum = 0.0
        wrist_motion = 0.0
        previous: np.ndarray | None = None
        for frame in frames_in:
            if not isinstance(frame, Mapping):
                continue
            joints = frame.get("joints_world")
            if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or len(joints) < RAW_JOINT_COUNT:
                continue
            wrist = _vec3(joints[cfg["wrist"]])
            if wrist is None:
                continue
            conf = _joint_conf(frame.get("joint_conf"), cfg["wrist"])
            if conf < min_joint_confidence:
                continue
            usable += 1
            confidence_sum += conf
            if previous is not None:
                wrist_motion += float(np.linalg.norm(wrist - previous))
            previous = wrist
        mean_conf = confidence_sum / usable if usable else 0.0
        score = float(usable) * 100.0 + mean_conf + min(wrist_motion, 10.0)
        scores[side] = {
            "usable_frame_count": usable,
            "mean_joint_confidence": round(mean_conf, 6),
            "wrist_motion_m": round(wrist_motion, 6),
            "score": round(score, 6),
        }
    return scores


def _select_side(
    player_id: int | None,
    overrides: Mapping[int, str],
    dominant_hand: str,
    side_scores: Mapping[str, Mapping[str, float]],
) -> str:
    if player_id is not None and player_id in overrides:
        return overrides[player_id]
    if dominant_hand != "auto":
        return dominant_hand
    right = side_scores.get("right", {})
    left = side_scores.get("left", {})
    right_key = (
        right.get("usable_frame_count", 0),
        right.get("mean_joint_confidence", 0.0),
        right.get("wrist_motion_m", 0.0),
    )
    left_key = (
        left.get("usable_frame_count", 0),
        left.get("mean_joint_confidence", 0.0),
        left.get("wrist_motion_m", 0.0),
    )
    return "left" if left_key > right_key else "right"


def _verify_handedness_with_detector_boxes(
    frames_in: Sequence[Any],
    *,
    auto_side: str,
    detector_records: Sequence[tuple[int, tuple[float, float, float, float], float]],
    calibration_model: Any,
    min_joint_confidence: float,
    gate_radius_px: float,
    min_wins: int = 10,
    win_ratio: float = 1.5,
) -> tuple[str, dict[str, Any]]:
    """Verify (and possibly flip) the auto-selected hand side using detector-box proximity.

    Spec design item 1: start from the proxy-style auto-side score, then VERIFY with
    detector-box proximity. For every frame with wrist-gated paddle boxes, the side whose
    projected wrist is closer to the nearest box center "wins" the frame; the auto side is
    flipped only on a decisive margin (>= ``min_wins`` wins for the challenger AND
    challenger_wins >= ``win_ratio`` * auto_wins). CVAT rectangles are never used here --
    only external-trained detector predictions are legal solver input.
    """

    by_frame: dict[int, list[tuple[tuple[float, float, float, float], float]]] = {}
    for frame_idx, bbox, conf in detector_records:
        by_frame.setdefault(frame_idx, []).append((bbox, conf))
    wins = {"right": 0, "left": 0}
    considered = 0
    for frame in frames_in:
        if not isinstance(frame, Mapping):
            continue
        frame_idx = _frame_index(frame)
        if frame_idx is None or frame_idx not in by_frame:
            continue
        joints = frame.get("joints_world")
        if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or len(joints) < RAW_JOINT_COUNT:
            continue
        distances: dict[str, float] = {}
        for side in ("right", "left"):
            cfg = _SIDE_JOINTS[side]
            wrist = _vec3(joints[cfg["wrist"]])
            if wrist is None or _joint_conf(frame.get("joint_conf"), cfg["wrist"]) < min_joint_confidence:
                continue
            try:
                wrist_px = _project_world_points_for_raw_detector_reference(
                    calibration_model.extrinsics, calibration_model.intrinsics, [tuple(float(v) for v in wrist)]
                )[0]
            except (ValueError, ZeroDivisionError):
                continue
            best = None
            for bbox, _conf in by_frame[frame_idx]:
                center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                distance = math.hypot(center[0] - wrist_px[0], center[1] - wrist_px[1])
                if distance <= gate_radius_px and (best is None or distance < best):
                    best = distance
            if best is not None:
                distances[side] = best
        if not distances:
            continue
        considered += 1
        winner = min(distances, key=lambda side: distances[side])
        wins[winner] += 1
    challenger = "left" if auto_side == "right" else "right"
    flipped = wins[challenger] >= min_wins and wins[challenger] >= win_ratio * max(1, wins[auto_side])
    selected = challenger if flipped else auto_side
    return selected, {
        "auto_side": auto_side,
        "selected_side": selected,
        "flipped": flipped,
        "wrist_gated_frames_considered": considered,
        "proximity_wins": dict(wins),
        "min_wins": min_wins,
        "win_ratio": win_ratio,
        "gate_radius_px": round(float(gate_radius_px), 3),
    }


def _per_frame_side_votes(
    frames_in: Sequence[Any],
    *,
    detector_records: Sequence[tuple[int, tuple[float, float, float, float], float]],
    calibration_model: Any,
    min_joint_confidence: float,
    gate_radius_px: float,
) -> list[tuple[float, str]]:
    """Per-frame hand-side votes from detector-box proximity: for each frame with wrist-gated
    boxes, the side whose projected wrist is closest to the nearest gated box wins the frame."""

    by_frame: dict[int, list[tuple[tuple[float, float, float, float], float]]] = {}
    for frame_idx, bbox, conf in detector_records:
        by_frame.setdefault(frame_idx, []).append((bbox, conf))
    votes: list[tuple[float, str]] = []
    for frame in frames_in:
        if not isinstance(frame, Mapping):
            continue
        frame_idx = _frame_index(frame)
        if frame_idx is None or frame_idx not in by_frame:
            continue
        joints = frame.get("joints_world")
        if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or len(joints) < RAW_JOINT_COUNT:
            continue
        t = float(frame.get("t", 0.0))
        distances: dict[str, float] = {}
        for side in ("right", "left"):
            cfg = _SIDE_JOINTS[side]
            wrist = _vec3(joints[cfg["wrist"]])
            if wrist is None or _joint_conf(frame.get("joint_conf"), cfg["wrist"]) < min_joint_confidence:
                continue
            try:
                wrist_px = _project_world_points_for_raw_detector_reference(
                    calibration_model.extrinsics, calibration_model.intrinsics, [tuple(float(v) for v in wrist)]
                )[0]
            except (ValueError, ZeroDivisionError):
                continue
            best = None
            for bbox, _conf in by_frame[frame_idx]:
                center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                distance = math.hypot(center[0] - wrist_px[0], center[1] - wrist_px[1])
                if distance <= gate_radius_px and (best is None or distance < best):
                    best = distance
            if best is not None:
                distances[side] = best
        if distances:
            votes.append((t, min(distances, key=lambda side: distances[side])))
    votes.sort(key=lambda item: item[0])
    return votes


def _hand_intervals_from_box_votes(
    votes: Sequence[tuple[float, str]],
    *,
    initial_side: str,
    min_hold_s: float,
    min_votes: int = 8,
    majority: float = 0.7,
) -> tuple[list[tuple[float, float, str]], dict[str, Any]]:
    """Segment the clip into hand-side intervals with >= ``min_hold_s`` hysteresis.

    runs/archive/root_docs_20260709/RACKET_6DOF_GOAL.md requires two-handed/hand-switch handling and the spec forbids per-frame
    side flapping: a switch requires a sustained window (>= ``min_hold_s``) in which the
    challenger side wins >= ``majority`` of >= ``min_votes`` detector-box proximity votes.
    Between switches the current side carries frames with no votes (fail-soft).
    """

    switches: list[dict[str, Any]] = []
    current = initial_side
    # Seed from the opening window when it is decisive, so a clip that *starts* on the
    # non-majority hand is not mis-sided until the first switch.
    if votes:
        t_first = votes[0][0]
        opening = [side for t, side in votes if t <= t_first + min_hold_s]
        if len(opening) >= min_votes:
            for side in ("right", "left"):
                if opening.count(side) / len(opening) >= majority and side != current:
                    switches.append({"t": None, "side": side, "reason": "opening_window_majority"})
                    current = side
                    break
    intervals: list[tuple[float, float, str]] = []
    start_t = -math.inf
    i = 0
    ordered = list(votes)
    while i < len(ordered):
        t_i, side_i = ordered[i]
        if side_i == current:
            i += 1
            continue
        window = [item for item in ordered[i:] if item[0] <= t_i + min_hold_s]
        challenger_count = sum(1 for _t, side in window if side == side_i)
        if len(window) >= min_votes and challenger_count / len(window) >= majority:
            intervals.append((start_t, t_i, current))
            switches.append({"t": round(t_i, 6), "side": side_i, "reason": "sustained_majority", "window_votes": len(window), "challenger_votes": challenger_count})
            current = side_i
            start_t = t_i
            i += len(window)
        else:
            i += 1
    intervals.append((start_t, math.inf, current))
    evidence = {
        "vote_count": len(votes),
        "initial_side": initial_side,
        "min_hold_s": round(float(min_hold_s), 6),
        "min_votes": min_votes,
        "majority": majority,
        "switches": switches,
        "interval_count": len(intervals),
        "intervals": [
            {"start_t": None if not math.isfinite(a) else round(a, 6), "end_t": None if not math.isfinite(b) else round(b, 6), "side": s}
            for a, b, s in intervals
        ],
    }
    return intervals, evidence


# --------------------------------------------------------------------------------------
# Grip-segment detection (residual break + >=2s hysteresis)
# --------------------------------------------------------------------------------------


def _detect_segments(
    samples: Sequence[_HandFrame],
    *,
    min_duration_s: float,
    break_angle_deg: float,
) -> list[tuple[int, int]]:
    if not samples:
        return []
    n = len(samples)
    segments: list[tuple[int, int]] = []
    seg_start = 0
    reference = samples[0].z_candidate_raw
    breaking_start: int | None = None
    for i in range(1, n):
        angle = _angle_deg(reference, samples[i].z_candidate_raw)
        if angle > break_angle_deg:
            if breaking_start is None:
                breaking_start = i
            duration = samples[i].t - samples[breaking_start].t
            if duration >= min_duration_s:
                segments.append((seg_start, breaking_start - 1))
                seg_start = breaking_start
                reference = samples[breaking_start].z_candidate_raw
                breaking_start = None
        else:
            breaking_start = None
    segments.append((seg_start, n - 1))
    return segments


# --------------------------------------------------------------------------------------
# Grip transform G = (R_g, t_g) fitting
# --------------------------------------------------------------------------------------


def _solve_wahba(pairs: Sequence[tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
    """Weighted Wahba/Kabsch solve: R minimizing sum w*||v_body - R @ v_local||^2 over SO(3)."""

    b = np.zeros((3, 3))
    for v_body, v_local, weight in pairs:
        if weight <= 0.0:
            continue
        b += weight * np.outer(v_body, v_local)
    u, _, vt = np.linalg.svd(b)
    d = np.linalg.det(u @ vt)
    correction = np.diag([1.0, 1.0, 1.0 if d >= 0.0 else -1.0])
    return u @ correction @ vt


def _translation_jitter_weights(samples: Sequence[_HandFrame]) -> list[float]:
    weights = [0.0]
    for i in range(1, len(samples)):
        weights.append(min(samples[i].joint_confidence, samples[i - 1].joint_confidence))
    return weights


def _solve_grip_translation(
    rotations: Sequence[np.ndarray],
    weights: Sequence[float],
    prior_translation: np.ndarray,
    prior_weight: float,
) -> np.ndarray:
    """Closed-form weighted least squares for t_g.

    Minimizes sum_i w_i * ||R_i @ t_g - R_{i-1} @ t_g||^2 + prior_weight * ||t_g - prior||^2.
    The smoothness term penalizes lever-arm-amplified jitter (a real, well-posed effect distinct
    from the rotation-conjugation invariance that makes this term uninformative for R_g).
    """

    m = prior_weight * np.eye(3)
    rhs = prior_weight * prior_translation
    for i in range(1, len(rotations)):
        d = rotations[i] - rotations[i - 1]
        w = weights[i] if i < len(weights) else 1.0
        if w <= 0.0:
            continue
        m += w * (d.T @ d)
    return np.linalg.solve(m, rhs)


def _fit_grip_transform(
    samples: Sequence[_HandFrame],
    *,
    prior_rotation: np.ndarray,
    prior_translation: np.ndarray,
    prior_rotation_weight: float,
    prior_translation_weight: float,
    reflection_pairs: Sequence[tuple[np.ndarray, np.ndarray, float]],
    use_reflection: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pairs: list[tuple[np.ndarray, np.ndarray, float]] = [
        (prior_rotation @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), prior_rotation_weight),
        (prior_rotation @ np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), prior_rotation_weight),
    ]
    if use_reflection:
        pairs.extend(reflection_pairs)
    r_g = _solve_wahba(pairs)

    rotations = [s.rotation for s in samples]
    weights = _translation_jitter_weights(samples)
    t_g = _solve_grip_translation(rotations, weights, prior_translation, prior_translation_weight)
    return (
        r_g,
        t_g,
        {
            "reflection_observation_count": len(reflection_pairs) if use_reflection else 0,
            "sample_count": len(samples),
        },
    )


# --------------------------------------------------------------------------------------
# Optional evidence channel (b): ball-reflection cone factor
# --------------------------------------------------------------------------------------


def _reflection_records_for_player(
    physics_estimate: Mapping[str, Any] | None,
    samples: Sequence[_HandFrame],
    *,
    side: str,
    player_id: Any,
    weight_scale: float,
    max_time_gap_s: float = DEFAULT_REFLECTION_MAX_TIME_GAP_S,
) -> list[dict[str, Any]]:
    if not physics_estimate or not samples:
        return []
    estimates = physics_estimate.get("estimates")
    if not isinstance(estimates, list):
        return []
    records: list[dict[str, Any]] = []
    for estimate in estimates:
        if not isinstance(estimate, Mapping):
            continue
        selected = estimate.get("selected_wrist")
        selected = selected if isinstance(selected, Mapping) else {}
        if player_id is not None and selected.get("player_id") is not None:
            if str(selected.get("player_id")) != str(player_id):
                continue
        if selected.get("side") and str(selected.get("side")) != side:
            continue
        normal = _vec3(estimate.get("face_normal_world"))
        if normal is None:
            continue
        t = _maybe_float(estimate.get("t"))
        if t is None:
            continue
        nearest = _nearest_sample(samples, t, max_time_gap_s)
        if nearest is None:
            continue
        uncertainty = estimate.get("uncertainty")
        bound = _maybe_float(uncertainty.get("normal_angle_bound_deg")) if isinstance(uncertainty, Mapping) else None
        bound = bound if bound and bound > 1e-3 else 45.0
        weight = weight_scale / max(bound, 1e-3)
        v_body = nearest.rotation.T @ normal
        records.append(
            {
                "v_body": v_body,
                "v_local": np.array([0.0, 0.0, 1.0]),
                "weight": weight,
                "t": t,
                "normal_angle_bound_deg": bound,
            }
        )
    return records


def _nearest_sample(samples: Sequence[_HandFrame], t: float, max_gap: float) -> _HandFrame | None:
    if not samples:
        return None
    nearest = min(samples, key=lambda s: abs(s.t - t))
    if abs(nearest.t - t) > max_gap:
        return None
    return nearest


# --------------------------------------------------------------------------------------
# Optional evidence channel (c): wrist-gated detector-box reprojection correction
# --------------------------------------------------------------------------------------


def _detector_box_records(detector_boxes: Mapping[str, Any]) -> list[tuple[int, tuple[float, float, float, float], float]]:
    records: list[tuple[int, tuple[float, float, float, float], float]] = []
    items = None
    for key in ("records", "detections", "frames", "boxes"):
        candidate = detector_boxes.get(key)
        if isinstance(candidate, list):
            items = candidate
            break
    if items is None:
        return records
    for item in items:
        if not isinstance(item, Mapping):
            continue
        frame_idx = _maybe_int(item.get("frame", item.get("frame_idx", item.get("frame_index"))))
        bbox = item.get("bbox_xyxy") or item.get("bbox")
        if frame_idx is not None and isinstance(bbox, Sequence) and len(bbox) == 4:
            conf = _maybe_float(item.get("conf", item.get("confidence", item.get("score")))) or 0.5
            try:
                records.append((frame_idx, tuple(float(v) for v in bbox), conf))
            except (TypeError, ValueError):
                pass
            continue
        nested = item.get("detections")
        if frame_idx is not None and isinstance(nested, list):
            for det in nested:
                if not isinstance(det, Mapping):
                    continue
                nested_bbox = det.get("bbox_xyxy") or det.get("bbox")
                if not isinstance(nested_bbox, Sequence) or len(nested_bbox) != 4:
                    continue
                nested_conf = _maybe_float(det.get("conf", det.get("confidence", det.get("score")))) or 0.5
                try:
                    records.append((frame_idx, tuple(float(v) for v in nested_bbox), nested_conf))
                except (TypeError, ValueError):
                    continue
    return records


def _detector_box_correction(
    detector_boxes: Mapping[str, Any],
    calibration: Mapping[str, Any],
    samples: Sequence[_HandFrame],
    *,
    r_g: np.ndarray,
    t_g: np.ndarray,
    wrist_gate_radius_px: float,
    max_correction_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from .schemas import CourtCalibration

    try:
        calib = CourtCalibration.model_validate(calibration)
    except Exception as exc:  # noqa: BLE001 - report, do not raise; optional channel
        return np.zeros(3), {"used": False, "reason": f"invalid_calibration: {exc}"}

    records = _detector_box_records(detector_boxes)
    if not records:
        return np.zeros(3), {"used": False, "reason": "no_detector_box_records"}

    by_frame_idx = {s.frame_idx: s for s in samples if s.frame_idx is not None}
    corrections: list[np.ndarray] = []
    weights: list[float] = []
    for frame_idx, bbox, conf in records:
        sample = by_frame_idx.get(frame_idx)
        if sample is None:
            continue
        try:
            wrist_px = _project_world_points_for_raw_detector_reference(
                calib.extrinsics, calib.intrinsics, [tuple(float(v) for v in sample.wrist)]
            )[0]
        except (ValueError, ZeroDivisionError):
            continue
        box_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        if math.hypot(box_center[0] - wrist_px[0], box_center[1] - wrist_px[1]) > wrist_gate_radius_px:
            continue
        face_center_world = sample.wrist + sample.rotation @ t_g
        try:
            face_px = _project_world_points_for_raw_detector_reference(
                calib.extrinsics, calib.intrinsics, [tuple(float(v) for v in face_center_world)]
            )[0]
        except (ValueError, ZeroDivisionError):
            continue
        depth = _camera_depth(calib.extrinsics, face_center_world)
        if depth is None or abs(depth) <= 1e-6:
            continue
        pixel_error = np.array([box_center[0] - face_px[0], box_center[1] - face_px[1]])
        world_delta_camera = np.array(
            [pixel_error[0] * depth / calib.intrinsics.fx, pixel_error[1] * depth / calib.intrinsics.fy, 0.0]
        )
        rot_cam_to_world = np.array([[float(v) for v in row] for row in calib.extrinsics.R]).T
        world_delta = rot_cam_to_world @ world_delta_camera
        hand_local_delta = sample.rotation.T @ world_delta
        corrections.append(hand_local_delta * conf)
        weights.append(conf)

    if not corrections:
        return np.zeros(3), {"used": False, "reason": "no_wrist_gated_boxes_matched", "candidate_record_count": len(records)}

    # Robust IRLS (Huber) around the confidence-weighted mean: detector false positives inside the
    # wrist gate otherwise drag the constant offset. The correction is constant per segment, so
    # improving its accuracy costs zero temporal motion.
    samples_arr = np.array([c / max(w, 1e-9) for c, w in zip(corrections, weights)])
    base_weights = np.array(weights)
    est = np.sum(samples_arr * base_weights[:, None], axis=0) / max(float(base_weights.sum()), 1e-9)
    huber_delta = 0.05
    for _ in range(3):
        residuals = np.linalg.norm(samples_arr - est, axis=1)
        rw = np.where(residuals <= huber_delta, 1.0, huber_delta / np.maximum(residuals, 1e-9))
        combined = base_weights * rw
        total = float(combined.sum())
        if total <= 1e-9:
            break
        est = np.sum(samples_arr * combined[:, None], axis=0) / total
    avg = est
    norm = float(np.linalg.norm(avg))
    if norm > max_correction_m and norm > 0.0:
        avg = avg * (max_correction_m / norm)
    return avg, {
        "used": True,
        "matched_box_count": len(corrections),
        "candidate_record_count": len(records),
        "correction_norm_m": round(float(np.linalg.norm(avg)), 6),
        "robust": "huber_irls_3it_delta_0.05",
    }


def _paddle_footprint_local(dims: Mapping[str, float], *, samples: int = 12) -> list[np.ndarray]:
    """Paddle-local footprint points (face ellipse + handle rectangle) in meters, matching the
    review scorer's paddle model (face ellipse plus handle below the face along -Y)."""

    length_in = float(dims.get("length", dims.get("h", DEFAULT_PADDLE_DIMS_IN["length"])))
    width_in = float(dims.get("width", dims.get("w", DEFAULT_PADDLE_DIMS_IN["width"])))
    handle_in = float(dims.get("handle_length", dims.get("handle_length_in", DEFAULT_HANDLE_LENGTH_IN)))
    face_len_m = max(0.01, (length_in - handle_in) * 0.0254)
    width_m = width_in * 0.0254
    handle_len_m = handle_in * 0.0254
    handle_half_w = min(width_m * 0.35, 1.5 * 0.0254) / 2.0
    points = [
        np.array([0.5 * width_m * math.cos(2.0 * math.pi * i / samples), 0.5 * face_len_m * math.sin(2.0 * math.pi * i / samples), 0.0])
        for i in range(samples)
    ]
    y0 = -0.5 * face_len_m - handle_len_m
    y1 = -0.5 * face_len_m
    points.extend(
        np.array(p)
        for p in ((-handle_half_w, y0, 0.0), (handle_half_w, y0, 0.0), (handle_half_w, y1, 0.0), (-handle_half_w, y1, 0.0))
    )
    return points


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _project_paddle_bbox(
    rotation: np.ndarray,
    translation: np.ndarray,
    footprint: Sequence[np.ndarray],
    calibration_model: Any,
    project_world_points: Any,
) -> list[float] | None:
    world = [tuple(float(v) for v in (rotation @ p + translation)) for p in footprint]
    try:
        image = project_world_points(calibration_model.extrinsics, calibration_model.intrinsics, world)
    except (ValueError, ZeroDivisionError):
        return None
    xs = [pt[0] for pt in image if math.isfinite(pt[0]) and math.isfinite(pt[1])]
    ys = [pt[1] for pt in image if math.isfinite(pt[0]) and math.isfinite(pt[1])]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _gated_boxes_by_frame(
    detector_records: Sequence[tuple[int, tuple[float, float, float, float], float]],
    samples: Sequence[_HandFrame],
    calibration_model: Any,
    project_world_points: Any,
    *,
    wrist_gate_radius_px: float,
) -> dict[int, list[tuple[tuple[float, float, float, float], float]]]:
    """Wrist-gate detector boxes per frame: only boxes whose center is within the gate radius of
    the projected wrist count (spec factor (c) gating rule)."""

    by_frame_records: dict[int, list[tuple[tuple[float, float, float, float], float]]] = {}
    for frame_idx, bbox, conf in detector_records:
        by_frame_records.setdefault(frame_idx, []).append((bbox, conf))
    gated: dict[int, list[tuple[tuple[float, float, float, float], float]]] = {}
    for sample in samples:
        if sample.frame_idx is None or sample.frame_idx not in by_frame_records:
            continue
        try:
            wrist_px = project_world_points(
                calibration_model.extrinsics, calibration_model.intrinsics, [tuple(float(v) for v in sample.wrist)]
            )[0]
        except (ValueError, ZeroDivisionError):
            continue
        keep = []
        for bbox, conf in by_frame_records[sample.frame_idx]:
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            if math.hypot(center[0] - wrist_px[0], center[1] - wrist_px[1]) <= wrist_gate_radius_px:
                keep.append((bbox, conf))
        if keep:
            gated[sample.frame_idx] = keep
    return gated


def _solve_grip_roll_against_boxes(
    samples: Sequence[_HandFrame],
    *,
    r_g: np.ndarray,
    t_g: np.ndarray,
    gated_boxes: Mapping[int, list[tuple[tuple[float, float, float, float], float]]],
    dims: Mapping[str, float],
    calibration_model: Any,
    project_world_points: Any,
    search_deg: float = DEFAULT_DETECTOR_BOX_ROLL_SEARCH_DEG,
    steps: int = DEFAULT_DETECTOR_BOX_ROLL_STEPS,
    min_gain: float = DEFAULT_DETECTOR_BOX_ROLL_MIN_GAIN,
    max_frames: int = DEFAULT_DETECTOR_BOX_ROLL_MAX_FRAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-segment 1-DOF grip-roll solve: rotate G about the paddle grip axis (local Y) to
    maximize the mean IoU between the projected paddle footprint bbox and the nearest
    wrist-gated detector box. Detector-box reprojection residual (spec factor (c)), rotation
    component; a coarse grid search is robust and cheap for a single angle."""

    usable = [s for s in samples if s.frame_idx in gated_boxes]
    if not usable:
        return r_g, {"applied": False, "reason": "no_wrist_gated_boxes"}
    if len(usable) > max_frames:
        stride = max(1, len(usable) // max_frames)
        usable = usable[::stride][:max_frames]
    footprint = _paddle_footprint_local(dims)

    def score_for(rotation_g: np.ndarray) -> float:
        total = 0.0
        count = 0
        for sample in usable:
            rotation = sample.rotation @ rotation_g
            translation = sample.wrist + sample.rotation @ t_g
            pred = _project_paddle_bbox(rotation, translation, footprint, calibration_model, project_world_points)
            if pred is None:
                continue
            pred_center = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0)
            best = None
            for bbox, conf in gated_boxes[sample.frame_idx]:
                center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                distance = math.hypot(center[0] - pred_center[0], center[1] - pred_center[1])
                if best is None or distance < best[0]:
                    best = (distance, bbox, conf)
            if best is None:
                continue
            total += _bbox_iou_xyxy(pred, best[1]) * best[2]
            count += 1
        return total / count if count else 0.0

    base_score = score_for(r_g)
    current = r_g
    current_score = base_score
    angles = {"roll_deg": 0.0, "pitch_deg": 0.0}
    # Coordinate descent over two 1-DOF grids: roll about the grip axis (local Y) first --
    # it fixes the dominant width foreshortening -- then pitch about local X for the
    # face lean. Both remain constant per grip segment (G stays segment-constant).
    axis_schedule = (
        ("roll_deg", np.array([0.0, 1.0, 0.0]), search_deg),
        ("pitch_deg", np.array([1.0, 0.0, 0.0]), search_deg / 2.0),
        ("roll_deg", np.array([0.0, 1.0, 0.0]), search_deg / 2.0),
        ("pitch_deg", np.array([1.0, 0.0, 0.0]), search_deg / 4.0),
    )
    for axis_name, axis, half_range in axis_schedule:
        best_angle = 0.0
        best_score = current_score
        for angle_deg in np.linspace(-half_range, half_range, steps):
            if abs(angle_deg) < 1e-9:
                continue
            rotation_test = current @ Rotation.from_rotvec(np.radians(angle_deg) * axis).as_matrix()
            score = score_for(rotation_test)
            if score > best_score:
                best_score = score
                best_angle = float(angle_deg)
        if best_angle != 0.0 and best_score > current_score:
            current = current @ Rotation.from_rotvec(np.radians(best_angle) * axis).as_matrix()
            current_score = best_score
            angles[axis_name] += best_angle
    if current_score - base_score < min_gain:
        return r_g, {
            "applied": False,
            "reason": "no_meaningful_gain",
            "base_weighted_iou": round(base_score, 6),
            "best_weighted_iou": round(current_score, 6),
            "sampled_frames": len(usable),
        }
    return current, {
        "applied": True,
        "roll_deg": round(angles["roll_deg"], 3),
        "pitch_deg": round(angles["pitch_deg"], 3),
        "base_weighted_iou": round(base_score, 6),
        "best_weighted_iou": round(current_score, 6),
        "sampled_frames": len(usable),
    }


def _per_frame_box_deviation(
    samples: Sequence[_HandFrame],
    composed: Sequence["_ComposedFrame"],
    *,
    gated_boxes: Mapping[int, list[tuple[tuple[float, float, float, float], float]]],
    calibration_model: Any,
    project_world_points: Any,
    max_deviation_m: float = DEFAULT_PER_FRAME_DEVIATION_MAX_M,
    decay: float = DEFAULT_PER_FRAME_DEVIATION_DECAY,
    slew_m_per_frame: float = DEFAULT_PER_FRAME_DEVIATION_SLEW_M,
) -> dict[str, Any]:
    """Small per-frame position deviation toward the nearest wrist-gated detector box.

    runs/archive/root_docs_20260709/RACKET_6DOF_GOAL.md's parameterization is "G held constant per grip segment, plus a small
    per-frame deviation"; this implements the deviation for the translation component only,
    driven by the (legal) detector-box channel: bounded (``max_deviation_m``), confidence-weighted,
    slew-limited (``slew_m_per_frame`` caps how much translational motion the deviation itself may
    add per frame, so detector noise cannot inject visible jitter/teleports), and decaying
    geometrically on frames without boxes so it fails soft to the constant-G pose.
    Mutates ``composed`` translations in place; rotation/bands untouched.
    """

    deviation = np.zeros(3)
    applied = 0
    for frame in composed:
        previous_deviation = deviation.copy()
        boxes = gated_boxes.get(frame.frame_idx) if frame.frame_idx is not None else None
        if boxes:
            try:
                face_px = project_world_points(
                    calibration_model.extrinsics,
                    calibration_model.intrinsics,
                    [tuple(float(v) for v in frame.translation)],
                )[0]
            except (ValueError, ZeroDivisionError):
                face_px = None
            if face_px is not None:
                best = None
                for bbox, conf in boxes:
                    center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                    distance = math.hypot(center[0] - face_px[0], center[1] - face_px[1])
                    if best is None or distance < best[0]:
                        best = (distance, center, conf)
                if best is not None:
                    depth = _camera_depth(calibration_model.extrinsics, frame.translation)
                    if depth is not None and abs(depth) > 1e-6:
                        pixel_error = np.array([best[1][0] - face_px[0], best[1][1] - face_px[1]])
                        delta_camera = np.array(
                            [
                                pixel_error[0] * depth / calibration_model.intrinsics.fx,
                                pixel_error[1] * depth / calibration_model.intrinsics.fy,
                                0.0,
                            ]
                        )
                        rot_cam_to_world = np.array(
                            [[float(v) for v in row] for row in calibration_model.extrinsics.R]
                        ).T
                        target = rot_cam_to_world @ delta_camera
                        weight = max(0.0, min(1.0, best[2]))
                        deviation = deviation * (1.0 - weight) + target * weight
                        applied += 1
        else:
            deviation = deviation * decay
        norm = float(np.linalg.norm(deviation))
        if norm > max_deviation_m and norm > 0.0:
            deviation = deviation * (max_deviation_m / norm)
        step = deviation - previous_deviation
        step_norm = float(np.linalg.norm(step))
        if step_norm > slew_m_per_frame and step_norm > 0.0:
            deviation = previous_deviation + step * (slew_m_per_frame / step_norm)
        frame.translation = frame.translation + deviation
    return {
        "frames_with_box_deviation": applied,
        "max_deviation_m": round(float(max_deviation_m), 6),
        "decay": round(float(decay), 6),
        "slew_m_per_frame": round(float(slew_m_per_frame), 6),
    }


def _camera_depth(extrinsics: Any, world_point: np.ndarray) -> float | None:
    rotation = np.array([[float(v) for v in row] for row in extrinsics.R])
    translation = np.array([float(v) for v in extrinsics.t])
    camera = rotation @ world_point + translation
    return float(camera[2])


# --------------------------------------------------------------------------------------
# Proper SO(3)+R3 smoothing: SLERP-based one-euro filter
# --------------------------------------------------------------------------------------


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    cutoff = max(cutoff, 1e-6)
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class _OneEuroScalar:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0

    def apply(self, x: float, dt: float) -> float:
        if self._x_prev is None or dt <= 0.0:
            self._x_prev = x
            self._dx_prev = 0.0
            return x
        dx = (x - self._x_prev) / dt
        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class _RotationOneEuro:
    """SLERP-based one-euro filter operating on the rotation manifold.

    At each step the "innovation" (relative rotation from the running smoothed estimate to the
    raw sample) is measured as a rotation vector in the tangent space of the smoothed estimate;
    one-euro's adaptive cutoff is driven by the angular speed of that innovation, and the smoothed
    estimate is advanced by re-composing a *partial* rotation (a true SLERP step, since scaling a
    rotation vector by alpha in [0, 1] and re-exponentiating is exactly SLERP along that geodesic)
    rather than averaging raw axis components (which is not norm-preserving and is exactly what
    causes the baseline proxy's axis-EMA jitter).
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._prev_rotation: Rotation | None = None
        self._prev_speed = 0.0

    def apply(self, raw: Rotation, dt: float) -> Rotation:
        if self._prev_rotation is None:
            self._prev_rotation = raw
            self._prev_speed = 0.0
            return raw
        if dt <= 0.0:
            return self._prev_rotation
        relative = self._prev_rotation.inv() * raw
        angle = float(relative.magnitude())
        speed = angle / dt
        a_d = _smoothing_factor(dt, self.d_cutoff)
        speed_hat = a_d * speed + (1.0 - a_d) * self._prev_speed
        cutoff = self.min_cutoff + self.beta * speed_hat
        alpha = _smoothing_factor(dt, cutoff)
        alpha = min(1.0, max(0.0, alpha))
        rotvec = relative.as_rotvec()
        step = Rotation.from_rotvec(alpha * rotvec)
        new_rotation = self._prev_rotation * step
        self._prev_rotation = new_rotation
        self._prev_speed = speed_hat
        return new_rotation


def _smooth_poses(
    poses: Sequence[tuple[float, np.ndarray, np.ndarray]],
    *,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
    position_min_cutoff: float | None = None,
    position_beta: float | None = None,
    rotation_reset_times: set[float] | None = None,
    position_reset_times: set[float] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One-euro smooth an SO(3)+R3 pose sequence.

    ``rotation_reset_times``: segment starts (grip-transform changes) -- the rotation really is
    discontinuous there; smearing it across frames turns one honest outlier delta into a run of
    near-p95 transition frames, so the rotation filter restarts cleanly instead.
    ``position_reset_times``: hand-switch boundaries only -- position is continuous across
    same-hand grip-segment changes, so its filter keeps state there and only restarts when the
    paddle genuinely moves to the other hand.
    """

    pos_min_cutoff = position_min_cutoff if position_min_cutoff is not None else min_cutoff
    pos_beta = position_beta if position_beta is not None else beta
    rotation_filter = _RotationOneEuro(min_cutoff, beta, d_cutoff)
    position_filters = [_OneEuroScalar(pos_min_cutoff, pos_beta, d_cutoff) for _ in range(3)]
    rotation_prev_t: float | None = None
    position_prev_t: float | None = None
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for t, rotation_raw, translation_raw in poses:
        if rotation_reset_times and t in rotation_reset_times:
            rotation_filter = _RotationOneEuro(min_cutoff, beta, d_cutoff)
            rotation_prev_t = None
        if position_reset_times and t in position_reset_times:
            position_filters = [_OneEuroScalar(pos_min_cutoff, pos_beta, d_cutoff) for _ in range(3)]
            position_prev_t = None
        rotation_dt = (t - rotation_prev_t) if rotation_prev_t is not None else 0.0
        position_dt = (t - position_prev_t) if position_prev_t is not None else 0.0
        rotation_prev_t = t
        position_prev_t = t
        raw_rotation = Rotation.from_matrix(rotation_raw)
        smoothed_rotation = rotation_filter.apply(raw_rotation, rotation_dt)
        smoothed_position = np.array([position_filters[i].apply(float(translation_raw[i]), position_dt) for i in range(3)])
        outputs.append((_orthonormalize_strict(smoothed_rotation.as_matrix()), smoothed_position))
    return outputs


def _orthonormalize_strict(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        r = u @ vt
    return r


# --------------------------------------------------------------------------------------
# Membership filtering (optional; best-effort against a variety of membership payload shapes)
# --------------------------------------------------------------------------------------


def _excluded_player_ids(membership: Mapping[str, Any] | None) -> set[str]:
    if not membership:
        return set()
    excluded: set[str] = set()
    players = membership.get("players")
    if players is None and isinstance(membership.get("per_player"), Mapping):
        # player_court_membership.py artifact shape: {"per_player": {"<id>": {"verdict": ...}}}.
        # Mirror virtual_world's rule: only "adjacent_or_spectator" excludes; "uncertain" stays.
        players = membership.get("per_player")
    if isinstance(players, Mapping):
        for player_id, info in players.items():
            if _is_excluded_membership_entry(info):
                excluded.add(str(player_id))
    elif isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        for entry in players:
            if not isinstance(entry, Mapping):
                continue
            player_id = entry.get("id", entry.get("player_id"))
            if player_id is None:
                continue
            if _is_excluded_membership_entry(entry):
                excluded.add(str(player_id))
    return excluded


def _is_excluded_membership_entry(info: Any) -> bool:
    if not isinstance(info, Mapping):
        return False
    if "member" in info:
        return info.get("member") is False
    if "is_member" in info:
        return info.get("is_member") is False
    verdict = info.get("verdict") or info.get("membership") or info.get("status")
    if isinstance(verdict, str):
        return verdict.lower() in {
            "excluded",
            "non_member",
            "not_member",
            "outside_court",
            "reject",
            "excluded_non_member",
            "adjacent_or_spectator",
        }
    return False


# --------------------------------------------------------------------------------------
# Top-level build API
# --------------------------------------------------------------------------------------


def build_paddle_pose_fused_from_file(
    skeleton3d_path: str | Path,
    *,
    clip_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    path = Path(skeleton3d_path)
    payload = _read_json_object(path, "skeleton3d")
    return build_paddle_pose_fused_from_skeleton(
        payload,
        clip_id=clip_id or path.parent.name or path.stem,
        source_path=path,
        **kwargs,
    )


def build_paddle_pose_fused_from_skeleton(
    skeleton3d: Mapping[str, Any],
    *,
    clip_id: str,
    source_path: str | Path | None = None,
    dominant_hand: str = "auto",
    dominant_hand_by_player: Mapping[int | str, str] | None = None,
    paddle_dims_in: Mapping[str, float] | None = None,
    grip_offset_m: float = DEFAULT_GRIP_OFFSET_M,
    min_joint_confidence: float = DEFAULT_MIN_JOINT_CONFIDENCE,
    ball_track: Mapping[str, Any] | None = None,
    contact_windows: Mapping[str, Any] | None = None,
    physics_estimate: Mapping[str, Any] | None = None,
    detector_boxes: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    membership: Mapping[str, Any] | None = None,
    use_reflection: bool = True,
    use_detector_boxes: bool = False,
    use_detector_box_handedness: bool = True,
    min_segment_duration_s: float = DEFAULT_MIN_SEGMENT_DURATION_S,
    segment_break_angle_deg: float = DEFAULT_SEGMENT_BREAK_ANGLE_DEG,
    prior_rotation_weight: float = DEFAULT_PRIOR_ROTATION_WEIGHT,
    prior_translation_weight: float = DEFAULT_PRIOR_TRANSLATION_WEIGHT,
    reflection_weight_scale: float = DEFAULT_REFLECTION_WEIGHT_SCALE,
    detector_box_wrist_gate_radius_px: float = DEFAULT_DETECTOR_BOX_WRIST_GATE_RADIUS_PX,
    detector_box_max_correction_m: float = DEFAULT_DETECTOR_BOX_MAX_CORRECTION_M,
    detector_box_roll_search_deg: float = DEFAULT_DETECTOR_BOX_ROLL_SEARCH_DEG,
    per_frame_deviation_max_m: float = DEFAULT_PER_FRAME_DEVIATION_MAX_M,
    per_frame_deviation_decay: float = DEFAULT_PER_FRAME_DEVIATION_DECAY,
    per_frame_deviation_slew_m: float = DEFAULT_PER_FRAME_DEVIATION_SLEW_M,
    hand_switch_min_hold_s: float = DEFAULT_HAND_SWITCH_MIN_HOLD_S,
    hand_switch_min_votes: int = DEFAULT_HAND_SWITCH_MIN_VOTES,
    hand_switch_majority: float = DEFAULT_HAND_SWITCH_MAJORITY,
    max_position_jump_m_per_frame: float = DEFAULT_MAX_POSITION_JUMP_M_PER_FRAME,
    position_one_euro_min_cutoff: float = DEFAULT_POSITION_ONE_EURO_MIN_CUTOFF,
    position_one_euro_beta: float = DEFAULT_POSITION_ONE_EURO_BETA,
    one_euro_min_cutoff: float = DEFAULT_ONE_EURO_MIN_CUTOFF,
    one_euro_beta: float = DEFAULT_ONE_EURO_BETA,
    one_euro_d_cutoff: float = DEFAULT_ONE_EURO_D_CUTOFF,
    contact_lock_window_s: float = DEFAULT_CONTACT_LOCK_WINDOW_S,
) -> dict[str, Any]:
    """Build fused wrist+palm+grip-transform paddle pose estimate frames.

    ``ball_track``/``contact_windows`` are accepted for interface symmetry with the wider evidence
    chain (spec.md item 3) but are not required by this solver directly: reflection evidence is
    consumed pre-fused via ``physics_estimate`` (a ``racket_physics_estimate.json``-shaped
    payload, e.g. from :mod:`threed.racketsport.racket_physics_estimate`).
    """

    if dominant_hand not in {"right", "left", "auto"}:
        raise ValueError("dominant_hand must be one of: right, left, auto")
    if grip_offset_m < 0.0:
        raise ValueError("grip_offset_m must be non-negative")
    if not 0.0 <= min_joint_confidence <= 1.0:
        raise ValueError("min_joint_confidence must be in [0, 1]")
    if min_segment_duration_s <= 0.0:
        raise ValueError("min_segment_duration_s must be positive")
    if segment_break_angle_deg <= 0.0:
        raise ValueError("segment_break_angle_deg must be positive")
    if prior_rotation_weight < 0.0 or prior_translation_weight <= 0.0:
        raise ValueError("prior weights must be non-negative (translation weight must be positive)")
    if reflection_weight_scale < 0.0:
        raise ValueError("reflection_weight_scale must be non-negative")
    if one_euro_min_cutoff <= 0.0 or one_euro_d_cutoff <= 0.0 or one_euro_beta < 0.0:
        raise ValueError("one_euro parameters must be positive (beta must be non-negative)")

    input_coordinate_space = resolve_world_coordinate_space(skeleton3d)
    dims = _paddle_dims(paddle_dims_in)
    grip_to_face_center_m = _grip_to_face_center_m(dims)
    prior_translation = np.array([0.0, grip_offset_m + grip_to_face_center_m, 0.0])
    prior_rotation = np.eye(3)

    hand_overrides = _hand_overrides(dominant_hand_by_player)
    excluded_ids = _excluded_player_ids(membership)

    detector_records: list[tuple[int, tuple[float, float, float, float], float]] = []
    calibration_model: Any = None
    handedness_verification_ready = False
    if detector_boxes and calibration and (use_detector_box_handedness or use_detector_boxes):
        detector_records = _detector_box_records(detector_boxes)
        if detector_records:
            try:
                from .schemas import CourtCalibration

                calibration_model = CourtCalibration.model_validate(calibration)
                handedness_verification_ready = use_detector_box_handedness
            except Exception:  # noqa: BLE001 - optional channel; fall back to auto side
                calibration_model = None

    players_in = skeleton3d.get("players")
    if not isinstance(players_in, Sequence) or isinstance(players_in, (str, bytes)):
        raise ValueError("skeleton3d players must be a list")
    fps = float(skeleton3d.get("fps", 30.0) or 30.0)

    output_players: list[dict[str, Any]] = []
    hidden_frames_all: list[dict[str, Any]] = []
    blockers: list[str] = []
    band_counts_total = {band: 0 for band in _VALID_BANDS}
    selected_hands_by_player: dict[str, str] = {}
    any_reflection_available = bool(physics_estimate and isinstance(physics_estimate.get("estimates"), list) and physics_estimate.get("estimates"))

    for player in players_in:
        if not isinstance(player, Mapping):
            continue
        player_id = player.get("id")
        player_id_int = _maybe_int(player_id)
        frames_in = player.get("frames")
        if not isinstance(frames_in, Sequence) or isinstance(frames_in, (str, bytes)):
            continue

        if player_id_int is not None and str(player_id_int) in excluded_ids:
            output_players.append(
                {
                    "id": player_id,
                    "dominant_hand": None,
                    "hand_selection": None,
                    "paddle_dims_in": dict(dims),
                    "frames": [],
                    "hidden_frame_count": 0,
                    "band_distribution": {band: 0 for band in _VALID_BANDS},
                    "segments": [],
                    "membership_excluded": True,
                }
            )
            continue

        side_scores = _hand_side_scores(frames_in, min_joint_confidence=min_joint_confidence)
        side = _select_side(player_id_int, hand_overrides, dominant_hand, side_scores)
        handedness_evidence: dict[str, Any] | None = None
        explicit_side = dominant_hand != "auto" or (player_id_int is not None and player_id_int in hand_overrides)
        hand_intervals: list[tuple[float, float, str]] | None = None
        if handedness_verification_ready and not explicit_side:
            side, handedness_evidence = _verify_handedness_with_detector_boxes(
                frames_in,
                auto_side=side,
                detector_records=detector_records,
                calibration_model=calibration_model,
                min_joint_confidence=min_joint_confidence,
                gate_radius_px=detector_box_wrist_gate_radius_px,
            )
            votes = _per_frame_side_votes(
                frames_in,
                detector_records=detector_records,
                calibration_model=calibration_model,
                min_joint_confidence=min_joint_confidence,
                gate_radius_px=detector_box_wrist_gate_radius_px,
            )
            hand_intervals, interval_evidence = _hand_intervals_from_box_votes(
                votes,
                initial_side=side,
                min_hold_s=hand_switch_min_hold_s,
                min_votes=hand_switch_min_votes,
                majority=hand_switch_majority,
            )
            handedness_evidence["hand_intervals"] = interval_evidence
        if player_id_int is not None:
            selected_hands_by_player[str(player_id_int)] = side

        interval_specs = hand_intervals or [(-math.inf, math.inf, side)]
        samples_by_interval: list[tuple[str, list[_HandFrame]]] = []
        hidden: list[dict[str, Any]] = []
        for interval_start, interval_end, interval_side in interval_specs:
            interval_frames = [
                item
                for item in frames_in
                if isinstance(item, Mapping) and interval_start <= float(item.get("t", 0.0)) < interval_end
            ]
            interval_samples, interval_hidden = _build_hand_frames(
                interval_frames, side=interval_side, min_joint_confidence=min_joint_confidence
            )
            samples_by_interval.append((interval_side, interval_samples))
            hidden.extend(interval_hidden)
        raw_samples = [sample for _side_i, samples in samples_by_interval for sample in samples]
        for entry in hidden:
            hidden_frames_all.append({"player_id": player_id, **entry})

        if not raw_samples:
            blockers.append("no_usable_wrist_frames_for_player")
            output_players.append(
                {
                    "id": player_id,
                    "dominant_hand": side,
                    "hand_selection": {
                        "selected_side": side,
                        "side_scores": side_scores,
                        "detector_box_verification": handedness_evidence,
                    },
                    "paddle_dims_in": dict(dims),
                    "frames": [],
                    "hidden_frame_count": len(hidden),
                    "band_distribution": {band: 0 for band in _VALID_BANDS},
                    "segments": [],
                }
            )
            continue

        composed: list[_ComposedFrame] = []
        segment_reports: list[dict[str, Any]] = []
        contact_times_used: list[float] = []
        gated_boxes_all: dict[int, list[tuple[tuple[float, float, float, float], float]]] = {}

        segment_jobs: list[tuple[str, list[_HandFrame], bool]] = []
        interval_index = 0
        for interval_side, interval_samples in samples_by_interval:
            if not interval_samples:
                continue
            interval_segments = _detect_segments(
                interval_samples, min_duration_s=min_segment_duration_s, break_angle_deg=segment_break_angle_deg
            )
            for segment_index, (seg_start, seg_end) in enumerate(interval_segments):
                is_hand_switch_start = interval_index > 0 and segment_index == 0
                segment_jobs.append((interval_side, interval_samples[seg_start : seg_end + 1], is_hand_switch_start))
            interval_index += 1

        for segment_side, seg_samples, _is_switch_start in segment_jobs:
            reflection_records: list[dict[str, Any]] = []
            if use_reflection and physics_estimate:
                reflection_records = _reflection_records_for_player(
                    physics_estimate,
                    seg_samples,
                    side=segment_side,
                    player_id=player_id_int,
                    weight_scale=reflection_weight_scale,
                )
            reflection_pairs = [(record["v_body"], record["v_local"], record["weight"]) for record in reflection_records]
            contact_times_used.extend(record["t"] for record in reflection_records)

            r_g, t_g, fit_info = _fit_grip_transform(
                seg_samples,
                prior_rotation=prior_rotation,
                prior_translation=prior_translation,
                prior_rotation_weight=prior_rotation_weight,
                prior_translation_weight=prior_translation_weight,
                reflection_pairs=reflection_pairs,
                use_reflection=use_reflection,
            )

            box_info: dict[str, Any] = {"used": False, "reason": "disabled"}
            roll_info: dict[str, Any] = {"applied": False, "reason": "disabled"}
            seg_gated_boxes: dict[int, list[tuple[tuple[float, float, float, float], float]]] = {}
            if use_detector_boxes and detector_boxes and calibration:
                box_correction, box_info = _detector_box_correction(
                    detector_boxes,
                    calibration,
                    seg_samples,
                    r_g=r_g,
                    t_g=t_g,
                    wrist_gate_radius_px=detector_box_wrist_gate_radius_px,
                    max_correction_m=detector_box_max_correction_m,
                )
                t_g = t_g + box_correction
                if calibration_model is not None and detector_records:
                    seg_gated_boxes = _gated_boxes_by_frame(
                        detector_records,
                        seg_samples,
                        calibration_model,
                        _project_world_points_for_raw_detector_reference,
                        wrist_gate_radius_px=detector_box_wrist_gate_radius_px,
                    )
                    if seg_gated_boxes:
                        r_g, roll_info = _solve_grip_roll_against_boxes(
                            seg_samples,
                            r_g=r_g,
                            t_g=t_g,
                            gated_boxes=seg_gated_boxes,
                            dims=dims,
                            calibration_model=calibration_model,
                            project_world_points=_project_world_points_for_raw_detector_reference,
                            search_deg=detector_box_roll_search_deg,
                        )
                gated_boxes_all.update(seg_gated_boxes)

            for sample in seg_samples:
                r_world = sample.rotation @ r_g
                t_world = sample.wrist + sample.rotation @ t_g
                composed.append(
                    _ComposedFrame(
                        t=sample.t,
                        frame_idx=sample.frame_idx,
                        rotation=r_world,
                        translation=t_world,
                        used_finger_palm_normal=sample.used_finger_palm_normal,
                        joint_confidence=sample.joint_confidence,
                        hand_side=segment_side,
                    )
                )
            segment_reports.append(
                {
                    "start_t": round(seg_samples[0].t, 6),
                    "end_t": round(seg_samples[-1].t, 6),
                    "hand_side": segment_side,
                    "frame_count": len(seg_samples),
                    "grip_rotation": [[round(float(v), 12) for v in row] for row in r_g],
                    "grip_translation_m": [round(float(v), 9) for v in t_g],
                    "fit_info": fit_info,
                    "detector_box": box_info,
                    "detector_box_roll": roll_info,
                }
            )

        composed.sort(key=lambda c: c.t)
        deviation_info: dict[str, Any] | None = None
        if use_detector_boxes and gated_boxes_all and calibration_model is not None:
            deviation_info = _per_frame_box_deviation(
                raw_samples,
                composed,
                gated_boxes=gated_boxes_all,
                calibration_model=calibration_model,
                project_world_points=_project_world_points_for_raw_detector_reference,
                max_deviation_m=per_frame_deviation_max_m,
                decay=per_frame_deviation_decay,
                slew_m_per_frame=per_frame_deviation_slew_m,
            )
        rotation_reset_times = {seg_samples[0].t for _side, seg_samples, _sw in segment_jobs[1:] if seg_samples}
        hand_switch_times = {seg_samples[0].t for _side, seg_samples, is_switch in segment_jobs if is_switch and seg_samples}
        smoothed = _smooth_poses(
            [(c.t, c.rotation, c.translation) for c in composed],
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            d_cutoff=one_euro_d_cutoff,
            position_min_cutoff=position_one_euro_min_cutoff,
            position_beta=position_one_euro_beta,
            rotation_reset_times=rotation_reset_times,
            position_reset_times=hand_switch_times,
        )
        # Hard translational slew clamp: outside declared hand-switch boundaries a paddle must not
        # jump more than max_position_jump_m_per_frame between consecutive emitted frames
        # (skeleton wrist teleports and detector glitches otherwise leak through the speed-adaptive
        # filter). At a declared switch the discontinuity is real and exempt.
        clamped_positions: list[np.ndarray] = []
        clamp_count = 0
        for index, (composed_frame, (_rot, position)) in enumerate(zip(composed, smoothed)):
            if index == 0 or composed_frame.t in hand_switch_times:
                clamped_positions.append(position)
                continue
            previous = clamped_positions[-1]
            step = position - previous
            step_norm = float(np.linalg.norm(step))
            if step_norm > max_position_jump_m_per_frame and step_norm > 0.0:
                position = previous + step * (max_position_jump_m_per_frame / step_norm)
                clamp_count += 1
            clamped_positions.append(position)
        smoothed = [(rot, clamped_positions[i]) for i, (rot, _pos) in enumerate(smoothed)]

        frames_out: list[dict[str, Any]] = []
        band_counts_player = {band: 0 for band in _VALID_BANDS}
        for composed_frame, (rotation_smooth, translation_smooth) in zip(composed, smoothed):
            band = BAND_PALM_FITTED if composed_frame.used_finger_palm_normal else BAND_GRIP_EXTRAPOLATED
            if any(abs(composed_frame.t - ct) <= contact_lock_window_s for ct in contact_times_used):
                band = BAND_CONTACT_LOCKED
            conf = round(_BAND_BASE_CONF[band] * max(0.4, min(1.0, composed_frame.joint_confidence)), 6)
            band_counts_player[band] += 1
            band_counts_total[band] += 1
            frames_out.append(
                _frame_payload(
                    composed_frame,
                    rotation_smooth,
                    translation_smooth,
                    band=band,
                    conf=conf,
                    side=composed_frame.hand_side,
                )
            )

        output_players.append(
            {
                "id": player_id,
                "dominant_hand": side,
                "hand_selection": {
                    "selected_side": side,
                    "side_scores": side_scores,
                    "detector_box_verification": handedness_evidence,
                },
                "paddle_dims_in": dict(dims),
                "frames": frames_out,
                "hidden_frame_count": len(hidden),
                "band_distribution": band_counts_player,
                "segments": segment_reports,
                "per_frame_box_deviation": deviation_info,
                "declared_hand_switch_times": sorted(round(value, 6) for value in hand_switch_times),
                "position_jump_clamped_frame_count": clamp_count,
            }
        )

    estimate_frame_count = sum(len(player["frames"]) for player in output_players)
    hidden_count = len(hidden_frames_all)
    summary = {
        "input_player_count": len([player for player in players_in if isinstance(player, Mapping)]),
        "player_count": len(output_players),
        "estimate_frame_count": estimate_frame_count,
        "hidden_frame_count": hidden_count,
        "band_distribution": band_counts_total,
        "dominant_hand_by_player": selected_hands_by_player,
        "evidence_channels": {
            "reflection_enabled": use_reflection,
            "reflection_contacts_available": any_reflection_available,
            "detector_boxes_enabled": bool(use_detector_boxes and detector_boxes and calibration),
            "detector_box_handedness_enabled": handedness_verification_ready,
        },
        "render_only": True,
        "trust": TRUST,
        "rkt_gate_unscoreable": True,
    }

    warnings = ["fused_estimate_render_only_not_rkt_gate", "rkt_gate_unscoreable_without_true_corner_gt"]
    if not any_reflection_available:
        warnings.append("reflection_channel_dormant_no_usable_ball_contacts")
    if band_counts_total[BAND_CONTACT_LOCKED] == 0:
        warnings.append("no_contact_locked_frames_emitted")
    if hidden_frames_all:
        warnings.append("hidden_fused_paddle_pose_frames")

    status = "preview" if estimate_frame_count else "blocked"

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip_id": str(clip_id),
        "status": status,
        "source": SOURCE,
        "render_only": True,
        "not_for_detection_metrics": True,
        "trusted_for_rkt_promotion": False,
        "never_canonical_racket_pose": True,
        "canonical_output_forbidden": "racket_pose.json",
        "rkt_gate_unscoreable": True,
        "trust": TRUST,
        "trust_band": {
            "status": TRUST,
            "gate_status": TRUST,
            "stage": "RKT",
            "gate_id": "wrist_palm_grip_fused_estimated_paddle",
            "badge": "low_confidence",
            "reason": (
                "Fused wrist+palm+grip-transform paddle estimator: hand frames from MHR70 finger "
                "joints, constant-per-segment grip transform, optional wrist-gated external "
                "detector-box evidence and (when available) ball-reflection contacts. Render-only, "
                "estimated-class, never canonical racket_pose.json, never RKT promotion. "
                "Distinct from (and successor to) the legacy wrist_proxy estimator."
            ),
        },
        "world_frame": WORLD_FRAME,
        "coordinate_frame": WORLD_COORDINATE_FRAME,
        "coordinate_space": WORLD_COORDINATE_SPACE.value,
        "input_coordinate_space": input_coordinate_space.value,
        "translation_unit": "m",
        "fps": fps,
        "source_path": str(source_path or ""),
        "parameters": {
            "coordinate_contract": {
                "world_input": input_coordinate_space.value,
                "pinhole_output": CoordinateSpace.PIXELS_UNDISTORTED_NATIVE.value,
                "detector_reference": CoordinateSpace.PIXELS_RAW_NATIVE.value,
                "transform_applied": False,
                "parity_note": "declaration_only_no_distortion_resize_crop_or_reference_transform",
            },
            "dominant_hand": dominant_hand,
            "dominant_hand_by_player": {str(key): value for key, value in sorted(hand_overrides.items())},
            "grip_offset_m": round(float(grip_offset_m), 6),
            "grip_to_face_center_m": round(float(grip_to_face_center_m), 6),
            "min_joint_confidence": round(float(min_joint_confidence), 6),
            "paddle_dims_in": dict(dims),
            "min_segment_duration_s": round(float(min_segment_duration_s), 6),
            "segment_break_angle_deg": round(float(segment_break_angle_deg), 6),
            "prior_rotation_weight": round(float(prior_rotation_weight), 6),
            "prior_translation_weight": round(float(prior_translation_weight), 6),
            "reflection_weight_scale": round(float(reflection_weight_scale), 6),
            "use_reflection": use_reflection,
            "use_detector_boxes": use_detector_boxes,
            "use_detector_box_handedness": use_detector_box_handedness,
            "detector_box_roll_search_deg": round(float(detector_box_roll_search_deg), 6),
            "per_frame_deviation_max_m": round(float(per_frame_deviation_max_m), 6),
            "per_frame_deviation_decay": round(float(per_frame_deviation_decay), 6),
            "per_frame_deviation_slew_m": round(float(per_frame_deviation_slew_m), 6),
            "hand_switch_min_hold_s": round(float(hand_switch_min_hold_s), 6),
            "hand_switch_min_votes": int(hand_switch_min_votes),
            "hand_switch_majority": round(float(hand_switch_majority), 6),
            "max_position_jump_m_per_frame": round(float(max_position_jump_m_per_frame), 6),
            "one_euro": {
                "min_cutoff": round(float(one_euro_min_cutoff), 6),
                "beta": round(float(one_euro_beta), 6),
                "d_cutoff": round(float(one_euro_d_cutoff), 6),
                "position_min_cutoff": round(float(position_one_euro_min_cutoff), 6),
                "position_beta": round(float(position_one_euro_beta), 6),
            },
            "render_mesh_style": "paddle_face_with_handle",
            "orientation": "mhr70_finger_hand_frame_grip_transform_fused",
        },
        "summary": summary,
        "warnings": warnings,
        "blockers": sorted(set(blockers)),
        "players": output_players,
        "hidden_frames": hidden_frames_all,
        "notes": [
            "This is a fused wrist+palm+grip-transform estimate for render continuity only, not true paddle 6DoF.",
            "The RKT face-angle/contact gate remains unscoreable until true paddle-corner/reference GT exists.",
            "Legacy world_frame=court_Z0 is retained; coordinate_frame and coordinate_space are canonical additive fields.",
        ],
    }


def _frame_payload(
    frame: _ComposedFrame,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    band: str,
    conf: float,
    side: str,
) -> dict[str, Any]:
    rotation = _orthonormalize_strict(rotation)
    return {
        "t": float(frame.t),
        "frame": frame.frame_idx,
        "pose_se3": {
            "R": [[round(float(v), 12) for v in row] for row in rotation],
            "t": [round(float(v), 9) for v in translation],
        },
        "conf": conf,
        "world_frame": WORLD_FRAME,
        "coordinate_frame": WORLD_COORDINATE_FRAME,
        "coordinate_space": WORLD_COORDINATE_SPACE.value,
        "translation_unit": "m",
        "source": SOURCE,
        "reprojection_error_px": None,
        "ambiguous": False,
        "render_only": True,
        "not_for_detection_metrics": True,
        "trust": TRUST,
        "trust_band": {
            "status": TRUST,
            "gate_status": TRUST,
            "stage": "RKT",
            "gate_id": "wrist_palm_grip_fused_estimated_paddle",
            "badge": "low_confidence",
            "note": band,
            "band_detail": band,
            "reason": (
                "Fused wrist+palm+grip-transform estimator (MHR70 hand frame + per-segment grip "
                "transform + wrist-gated external detector boxes). Render-only estimate; not true "
                "paddle 6DoF; never promotes the RKT gate. Not the legacy wrist proxy."
            ),
        },
        "confidence_provenance": {
            "band": TRUST,
            "display_band": "low_confidence",
            "predictor": band,
            "horizon_frames": 0,
            "predicted_sigma_m": None,
        },
        "render_mesh": {
            "style": "paddle_face_with_handle",
            "face_vertex_count": 4,
            "handle_vertex_count": 4,
        },
        "evidence": {
            "band": band,
            "hand_side": side,
        },
    }


def write_paddle_pose_fused(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Small shared utilities
# --------------------------------------------------------------------------------------


def _paddle_dims(value: Mapping[str, float] | None) -> dict[str, float]:
    dims = dict(value or DEFAULT_PADDLE_DIMS_IN)
    if not ({"length", "width"}.issubset(dims) or {"h", "w"}.issubset(dims)):
        raise ValueError("paddle_dims_in must include length/width or h/w")
    if any(float(item) <= 0.0 for item in dims.values()):
        raise ValueError("paddle_dims_in values must be positive")
    return {str(key): float(val) for key, val in dims.items()}


def _grip_to_face_center_m(paddle_dims_in: Mapping[str, float]) -> float:
    face_length_in = float(paddle_dims_in.get("length", paddle_dims_in.get("h", DEFAULT_PADDLE_DIMS_IN["length"])))
    handle_length_in = float(
        paddle_dims_in.get("handle_length", paddle_dims_in.get("handle_length_in", DEFAULT_HANDLE_LENGTH_IN))
    )
    if face_length_in <= 0.0:
        raise ValueError("paddle face length must be positive")
    if handle_length_in <= 0.0:
        raise ValueError("paddle handle length must be positive")
    return (face_length_in + handle_length_in) * 0.0254 / 2.0


def _hand_overrides(value: Mapping[int | str, str] | None) -> dict[int, str]:
    overrides: dict[int, str] = {}
    if value is None:
        return overrides
    for raw_player_id, raw_side in value.items():
        player_id = _maybe_int(raw_player_id)
        if player_id is None:
            raise ValueError("dominant_hand_by_player keys must be player ids")
        side = str(raw_side)
        if side not in {"right", "left"}:
            raise ValueError("dominant_hand_by_player values must be right or left")
        overrides[player_id] = side
    return overrides


def _frame_index(frame: Mapping[str, Any]) -> int | None:
    for key in ("frame_idx", "frame_index", "frame"):
        value = frame.get(key)
        if value is not None:
            return _maybe_int(value)
    return None


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _vec3(value: Any) -> np.ndarray | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        arr = np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)
    except (TypeError, ValueError):
        return None
    if not bool(np.all(np.isfinite(arr))):
        return None
    return arr


def _joint_conf(conf: Any, index: int) -> float:
    if not isinstance(conf, Sequence) or isinstance(conf, (str, bytes)) or index >= len(conf):
        return 0.0
    try:
        return max(0.0, min(1.0, float(conf[index])))
    except (TypeError, ValueError):
        return 0.0


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    cos_value = max(-1.0, min(1.0, float(np.dot(a, b) / denom)))
    return math.degrees(math.acos(cos_value))


__all__ = [
    "ARTIFACT_TYPE",
    "BAND_CONTACT_LOCKED",
    "BAND_GRIP_EXTRAPOLATED",
    "BAND_PALM_FITTED",
    "DEFAULT_CONTACT_LOCK_WINDOW_S",
    "DEFAULT_DETECTOR_BOX_MAX_CORRECTION_M",
    "DEFAULT_DETECTOR_BOX_WRIST_GATE_RADIUS_PX",
    "DEFAULT_GRIP_OFFSET_M",
    "DEFAULT_MIN_JOINT_CONFIDENCE",
    "DEFAULT_MIN_SEGMENT_DURATION_S",
    "DEFAULT_ONE_EURO_BETA",
    "DEFAULT_ONE_EURO_D_CUTOFF",
    "DEFAULT_ONE_EURO_MIN_CUTOFF",
    "DEFAULT_PADDLE_DIMS_IN",
    "DEFAULT_PRIOR_ROTATION_WEIGHT",
    "DEFAULT_PRIOR_TRANSLATION_WEIGHT",
    "DEFAULT_REFLECTION_WEIGHT_SCALE",
    "DEFAULT_SEGMENT_BREAK_ANGLE_DEG",
    "RAW_JOINT_COUNT",
    "SCHEMA_VERSION",
    "SOURCE",
    "TRUST",
    "build_paddle_pose_fused_from_file",
    "build_paddle_pose_fused_from_skeleton",
    "write_paddle_pose_fused",
]
