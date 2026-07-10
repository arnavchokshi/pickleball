#!/usr/bin/env python3
"""Compare our BODY pipeline's per-frame 3D joints against an OpenCap (Stanford,
github.com/stanfordnmbl/opencap-core) two-iPhone markerless-mocap session -- the FIRST
harness in this project for an INDEPENDENT accuracy number for BODY (see
`runs/archive/root_docs_20260709/TECH_BLUEPRINTS.md` PILLAR: BODY, P2-6). No real OpenCap capture exists yet (the owner
will record one later); this module is validated here only against a SYNTHETIC fixture
with a known injected error (see `run_self_test`/`--self-test`).

WHY OPENCAP: OpenCap reports ~4.5 deg mean-absolute joint-angle error (range 1.7-10.3 deg)
vs. marker-based mocap + force plates, using two ordinary iPhones, a printed checkerboard,
and no hardware sync -- a genuinely independent, cheap instrument, distinct from (and
complementary to) the court-landmark and two-iPhone-triangulation protocols already
described in `runs/archive/root_docs_20260709/TECH_BLUEPRINTS.md` PILLAR BODY section 6 (P2-6a/b). Sources consulted (web,
2026-07-07): stanfordnmbl/opencap-core README + stanfordnmbl/opencap-processing docs;
PLOS Comp Biol 10.1371/journal.pcbi.1011462 (OpenCap validation paper, MAE 4.5deg, 2 iPhone
minimum, 720x540mm precision checkerboard for factory intrinsics + 210x175mm A4 checkerboard
for per-session extrinsics); CS230 report on the LSTM marker-augmenter (43 anatomical
markers predicted from 20 video keypoints, "_study" suffix marks augmented markers).

============================================================================
DATA FORMATS THIS MODULE INGESTS
============================================================================

(a) OpenCap OpenSim outputs, both STANDARD, well-documented biomechanics file formats
    this module parses directly (no OpenSim install/API dependency):
    - `.trc` (marker trajectories): a fixed-layout, tab-delimited Motion-Analysis-style
      table. Header line 2 (`DataRate CameraRate NumFrames NumMarkers Units ...`) carries
      the sampling rate and a `Units` field (`m`/`mm`/`cm`) this parser reads and converts
      to meters -- we do NOT assume a unit, we read it from the file every time.
      Marker names appear once per 3-column (X/Y/Z) block in the 4th header row.
    - `.mot`/`.sto` (OpenSim Storage joint angles from Inverse Kinematics): a header block
      ending in a literal `endheader` line, with an `inDegrees=yes|no` field this parser
      reads and normalizes to degrees (converting from radians if `inDegrees=no`) -- again,
      read from the file, never assumed.

(b) Our BODY pipeline's per-frame 3D joints, in EITHER of the two shapes this codebase
    already produces (see `threed.racketsport.virtual_world` / `body_world_joints.json`):
    - "world" shape (`virtual_world.json`/`confidence_gated_world.json`/`smpl_motion.json`):
      top-level `joint_names` (optional) + `fps` + `players[].frames[].joints_world`
      (`frame_idx`/`frame_index` and/or explicit `t`/`t_seconds`).
    - "labels" shape (`body_world_joints.json`): top-level `joint_names` + `samples[]`
      (`frame_index`, `joints_world`, optional `accepted` flag -- rejected samples skipped).
    If `joint_names` is absent and the joint count is exactly 70, this module assumes MHR70
    order (`threed.racketsport.external_gt_body_prediction_schema.MHR70_JOINT_NAMES`) --
    the same fallback convention already documented and used by
    `scripts/racketsport/score_external_gt_aspset510_body_results.py`. Any other joint
    count with no `joint_names` is a hard error (never silently guessed).

============================================================================
ALIGNMENT METHOD (spatial + temporal) -- see ALIGNMENT_METHOD_DESCRIPTION below
============================================================================
Spatial: OpenCap's "world" frame (defined by its own per-session checkerboard) has no
fixed relationship to our court-calibration world frame -- they are two independent
capture rigs. We therefore reuse this project's existing external-GT scoring machinery
(`threed.racketsport.external_gt_alignment`, built for the ASPset-510 lane) and fit ONE
similarity transform (rotation + uniform scale + translation, Umeyama/Kabsch, pooled over
every matched frame x joint) per clip -- `clip_level_rigid_aligned_mpjpe` -- rather than
assume the two frames coincide. We also report the zero-alignment `mpjpe`,
`root_relative_mpjpe`, `pa_mpjpe`, and (ankle-anchored, translation-only)
`grounding_consistent_mpjpe` variants for the same reasons documented in that module.

Temporal: OpenCap has no hardware sync with our own camera; the owner protocol
(`runs/lanes/opencap_body_gt_20260707/OWNER_OPENCAP_PROTOCOL.md`) specifies an audio-clap
onset at session start, aligned via this project's existing audio-onset tooling
(`scripts/racketsport/build_audio_onsets.py`/`build_audio_onsets_v2.py`). The resulting
scalar offset is passed here as `--sync-offset-seconds` (added to our own clip's frame
timestamps to land them on OpenCap's own timeline) -- this module does NOT compute that
offset itself. Once shifted, our predicted joint sequence is linearly interpolated onto
OpenCap's own per-file timestamps (its `.trc`/`.mot` "Time"/"time" columns); OpenCap
samples outside our clip's covered time range are dropped (never extrapolated).

============================================================================
JOINT CORRESPONDENCE -- THE HONEST LIMITATION (read before trusting any number here)
============================================================================
POSITION comparison: OpenCap's `.trc` markers are its LSTM-augmented anatomical SURFACE
markers (or, pre-augmentation, raw video 2D-pose keypoints triangulated to 3D) -- these
are placed to correspond to skin/anatomical landmarks (e.g. "RKnee"/"r_knee_study"), NOT
to our MHR (Momentum Human Rig) model's own RIGGED joint centers
(`threed.racketsport.external_gt_body_prediction_schema.MHR70_JOINT_NAMES`). Two markers
named "the same joint" by both systems are a close but NOT exact correspondence -- the
same caveat this project already accepted for the ASPset-510 lane
(`threed.racketsport.external_gt_aspset510`), and for the same reason: this is the only
practically available shared-name subset without hand-building a custom marker-to-joint
regressor. `OPENCAP_MARKER_ALIASES` below is a best-effort superset of plausible OpenCap
export names (OpenPose/HRNet pre-augmentation keypoint names AND `_study`-suffixed
augmented names) built from OpenCap's public docs/papers, NOT verified against a real
export (none exists yet) -- the CLI always echoes `trc_marker_names_found` so a real
capture's actual names can be checked/extended immediately, rather than silently mis-pair.
Only the 12-joint limb subset shared with our core-17 schema
(`threed.racketsport.external_gt_aspset510.SHARED_CORE_JOINT_NAMES` -- both shoulders,
elbows, wrists, hips, knees, ankles) is scored; head/face/hand/foot joints are out of
scope here (OpenCap's video-keypoint set does not reliably cover them either).

ANGLE comparison: OpenSim's Inverse-Kinematics joint angles (e.g. `knee_angle_r`) are
SIGNED, anatomically-defined DOFs computed relative to each segment's own model coordinate
frame (for `hip_flexion_*`, a true 3-DOF ball-joint decomposition relative to the pelvis
segment). This module instead computes a simple UNSIGNED "flexion-proxy" angle from three
raw joint positions (law of cosines at the vertex, remapped so 0deg = fully extended,
increasing with flexion -- matching OpenSim's *direction* of increase but not its full
decomposition): see `_three_point_flexion_deg`. This is a materially weaker proxy for
`hip_flexion_*` in particular (a genuine pelvis-relative 3-DOF quantity; our proxy is only
a trunk-thigh sagittal approximation using shoulder-hip-knee, entangled with trunk lean)
than for `knee_angle_*`/`elbow_flex_*` (near-hinge joints where the 3-point sagittal
approximation is much closer to the true 1-DOF value). It also cannot detect hyperextension
sign or any coronal/transverse-plane component. Angle comparison has ONE major advantage
over position comparison, though: it needs NO OpenCap-side marker-name matching or spatial
alignment at all (only OUR OWN joint names, plus by-name matching against the `.mot`
column names) -- it is immune to the position-side coordinate-frame/marker-identity
ambiguity above, which is why both metrics are reported side by side rather than one
standing in for the other.

============================================================================
WHAT THIS MODULE DOES NOT CLAIM
============================================================================
No real OpenCap data exists yet, so nothing produced by this module today is a passed
gate on real labels (per this project's `VERIFIED` discipline). Task 2 of the lane that
built this module validates the harness machinery itself (`run_self_test`/`--self-test`):
a synthetic OpenCap-shaped fixture with a KNOWN injected similarity transform + angle bias
is scored, and the recovered numbers are asserted against that known injection to a
numeric tolerance -- this proves the ARITHMETIC is correct, not that our BODY pipeline is
accurate. The first real number only exists once the owner completes
`OWNER_OPENCAP_PROTOCOL.md` and this CLI is re-run on the real export.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.eval.body_gate_report import DEFAULT_WORLD_MPJPE_THRESHOLD_M  # noqa: E402
from threed.racketsport.external_gt_alignment import (  # noqa: E402
    VARIANT_DESCRIPTIONS,
    per_joint_breakdown,
    score_external_gt_clip,
    score_grounding_consistent_variant,
)
from threed.racketsport.external_gt_aspset510 import SHARED_CORE_JOINT_NAMES  # noqa: E402
from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES  # noqa: E402

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_opencap_body_gt_comparison"

DEFAULT_ROOT_JOINT_NAMES: tuple[str, ...] = ("left_hip", "right_hip")
DEFAULT_FLOOR_JOINT_NAMES: tuple[str, ...] = ("left_ankle", "right_ankle")
DEFAULT_GATE_VARIANT = "clip_level_rigid_aligned_mpjpe"
MIN_SHARED_JOINT_COUNT = 4

ALIGNMENT_METHOD_DESCRIPTION = (
    "Spatial: one similarity transform (rotation+uniform scale+translation, Umeyama/Kabsch, "
    "pooled over all matched frames x joints) fit per clip via "
    "threed.racketsport.external_gt_alignment.clip_level_rigid_aligned_mpjpe -- reused "
    "unmodified from the ASPset-510 external-GT lane, since OpenCap's checkerboard-defined "
    "world frame and our court-calibration world frame are two independent rigs with no a "
    "priori shared origin/orientation/scale. Zero-alignment (mpjpe), root-relative, "
    "per-frame-Procrustes (pa_mpjpe), and ankle-anchored translation-only "
    "(grounding_consistent_mpjpe) variants are also reported for the same reasons "
    "documented in that module. Temporal: our clip's frame timestamps are shifted by a "
    "caller-supplied --sync-offset-seconds (computed upstream via an audio-clap onset match "
    "using this project's existing audio-onset tooling -- NOT computed by this script), "
    "then linearly interpolated onto OpenCap's own .trc/.mot timestamps; samples outside "
    "the overlapping time range are dropped, never extrapolated."
)

JOINT_CORRESPONDENCE_LIMITATIONS = (
    "POSITION: OpenCap .trc markers are anatomical surface markers (LSTM-augmented "
    "'_study' markers, or raw triangulated video-keypoints) placed near, but not "
    "identical to, our MHR model's rigged joint centers -- a name-matched approximation, "
    "not an exact correspondence (same caveat class as the ASPset-510 lane). Only the "
    "12-joint core-limb subset (SHARED_CORE_JOINT_NAMES) is scored; OPENCAP_MARKER_ALIASES "
    "is a best-effort superset built from OpenCap's public docs, UNVERIFIED against any "
    "real export (none exists yet) -- trc_marker_names_found is always echoed in the "
    "report so a real capture's actual names can be checked immediately. ANGLES: OpenSim's "
    "IK angles are signed, anatomically-decomposed DOFs relative to each segment's own "
    "model frame (hip_flexion_* is a true pelvis-relative 3-DOF quantity); this module's "
    "proxy is an UNSIGNED three-point 'flexion' angle (law of cosines at the joint vertex, "
    "remapped so 0deg=full-extension increasing with flexion) -- a much closer "
    "approximation for near-hinge joints (knee, elbow) than for hip_flexion_* (only a "
    "trunk-thigh sagittal approximation, entangled with trunk lean, no ab/adduction or "
    "rotation component, no hyperextension sign)."
)


class OpenCapCompareError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# OpenCap marker-name aliases (POSITION side only; see JOINT_CORRESPONDENCE_LIMITATIONS)
# ---------------------------------------------------------------------------

OPENCAP_MARKER_ALIASES: dict[str, tuple[str, ...]] = {
    "left_shoulder": ("LShoulder", "L_Shoulder", "LeftShoulder", "l_shoulder_study", "LShoulder_study"),
    "right_shoulder": ("RShoulder", "R_Shoulder", "RightShoulder", "r_shoulder_study", "RShoulder_study"),
    "left_elbow": ("LElbow", "L_Elbow", "LeftElbow", "l_elbow_study", "LElbow_study"),
    "right_elbow": ("RElbow", "R_Elbow", "RightElbow", "r_elbow_study", "RElbow_study"),
    "left_wrist": ("LWrist", "L_Wrist", "LeftWrist", "l_wrist_study", "LWrist_study"),
    "right_wrist": ("RWrist", "R_Wrist", "RightWrist", "r_wrist_study", "RWrist_study"),
    "left_hip": ("LHip", "L_Hip", "LeftHip", "l_hip_study", "LHip_study", "LHJC"),
    "right_hip": ("RHip", "R_Hip", "RightHip", "r_hip_study", "RHip_study", "RHJC"),
    "left_knee": ("LKnee", "L_Knee", "LeftKnee", "l_knee_study", "LKnee_study"),
    "right_knee": ("RKnee", "R_Knee", "RightKnee", "r_knee_study", "RKnee_study"),
    "left_ankle": ("LAnkle", "L_Ankle", "LeftAnkle", "l_ankle_study", "LAnkle_study"),
    "right_ankle": ("RAnkle", "R_Ankle", "RightAnkle", "r_ankle_study", "RAnkle_study"),
}

# canonical joint-angle name -> (our 3-joint triplet, candidate OpenSim .mot column names)
ANGLE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "knee_flexion_r": {
        "triplet": ("right_hip", "right_knee", "right_ankle"),
        "mot_columns": ("knee_angle_r",),
        "description": "Sagittal knee-bend proxy (0deg=full extension, increasing with flexion). "
        "Close approximation to OpenSim's knee_angle_r (near-hinge joint).",
    },
    "knee_flexion_l": {
        "triplet": ("left_hip", "left_knee", "left_ankle"),
        "mot_columns": ("knee_angle_l",),
        "description": "Sagittal knee-bend proxy, left leg. See knee_flexion_r.",
    },
    "elbow_flexion_r": {
        "triplet": ("right_shoulder", "right_elbow", "right_wrist"),
        "mot_columns": ("elbow_flex_r", "elbow_angle_r"),
        "description": "Sagittal elbow-bend proxy. Close approximation to OpenSim's "
        "elbow_flex_r (near-hinge joint).",
    },
    "elbow_flexion_l": {
        "triplet": ("left_shoulder", "left_elbow", "left_wrist"),
        "mot_columns": ("elbow_flex_l", "elbow_angle_l"),
        "description": "Sagittal elbow-bend proxy, left arm. See elbow_flexion_r.",
    },
    "hip_flexion_r": {
        "triplet": ("right_shoulder", "right_hip", "right_knee"),
        "mot_columns": ("hip_flexion_r",),
        "description": "WEAK PROXY: trunk-thigh sagittal angle, NOT OpenSim's true "
        "pelvis-relative 3-DOF hip_flexion_r -- entangled with trunk lean, see "
        "JOINT_CORRESPONDENCE_LIMITATIONS.",
    },
    "hip_flexion_l": {
        "triplet": ("left_shoulder", "left_hip", "left_knee"),
        "mot_columns": ("hip_flexion_l",),
        "description": "WEAK PROXY, left leg. See hip_flexion_r.",
    },
}


# ---------------------------------------------------------------------------
# .trc parsing (marker trajectories)
# ---------------------------------------------------------------------------


class TrcData:
    def __init__(self, *, marker_names: list[str], times: np.ndarray, positions_m: np.ndarray, data_rate: float):
        self.marker_names = marker_names
        self.times = times
        self.positions_m = positions_m
        self.data_rate = data_rate


_TRC_UNIT_SCALE_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 1.0 / 1000.0,
    "millimeter": 1.0 / 1000.0,
    "millimeters": 1.0 / 1000.0,
    "cm": 1.0 / 100.0,
    "centimeter": 1.0 / 100.0,
    "centimeters": 1.0 / 100.0,
}


def parse_trc(path: Path) -> TrcData:
    """Parse an OpenSim/Motion-Analysis-style .trc marker-trajectory file.

    Layout (standard, not this project's invention): line0 PathFileType header; line1
    column-name row (DataRate/CameraRate/NumFrames/NumMarkers/Units/...); line2 the
    matching values row; line3 `Frame#  Time  <Marker1>      <Marker2>      ...` (marker
    name once per 3-column X/Y/Z block, blank cells for the other two); line4 a
    `X1 Y1 Z1 X2 Y2 Z2 ...` axis-label row (ignored here); line5+ data rows.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise OpenCapCompareError(f"{path}: too few lines to be a valid .trc file")

    header_keys = [key.strip() for key in lines[1].split("\t")]
    header_vals = [val.strip() for val in lines[2].split("\t")]
    header = dict(zip(header_keys, header_vals))
    for required in ("NumMarkers", "DataRate", "Units"):
        if required not in header:
            raise OpenCapCompareError(f"{path}: missing required .trc header field {required!r}")

    num_markers = int(float(header["NumMarkers"]))
    data_rate = float(header["DataRate"])
    units = header["Units"].strip().lower()
    if units not in _TRC_UNIT_SCALE_TO_METERS:
        raise OpenCapCompareError(f"{path}: unrecognized .trc Units {header['Units']!r}")
    scale = _TRC_UNIT_SCALE_TO_METERS[units]

    marker_row = lines[3].split("\t")
    marker_names = [name.strip() for name in marker_row[2:] if name.strip()]
    if len(marker_names) != num_markers:
        raise OpenCapCompareError(
            f"{path}: header NumMarkers={num_markers} but found {len(marker_names)} marker "
            "names in the marker-name row"
        )

    times: list[float] = []
    positions: list[list[list[float]]] = []
    expected_cell_count = 2 + 3 * num_markers
    for line_number, raw_line in enumerate(lines[5:], start=6):
        if not raw_line.strip():
            continue
        cells = raw_line.split("\t")
        if len(cells) < expected_cell_count:
            raise OpenCapCompareError(
                f"{path}:{line_number}: data row has {len(cells)} cells, expected "
                f"{expected_cell_count} (2 + 3*{num_markers} markers)"
            )
        times.append(float(cells[1]))
        frame_positions = []
        for marker_index in range(num_markers):
            base = 2 + marker_index * 3
            frame_positions.append(
                [float(cells[base]) * scale, float(cells[base + 1]) * scale, float(cells[base + 2]) * scale]
            )
        positions.append(frame_positions)

    if not times:
        raise OpenCapCompareError(f"{path}: no data rows parsed")

    return TrcData(
        marker_names=marker_names,
        times=np.asarray(times, dtype=np.float64),
        positions_m=np.asarray(positions, dtype=np.float64),
        data_rate=data_rate,
    )


# ---------------------------------------------------------------------------
# .mot/.sto parsing (OpenSim Inverse Kinematics joint angles)
# ---------------------------------------------------------------------------


class MotData:
    def __init__(self, *, column_names: list[str], times: np.ndarray, values_deg: np.ndarray):
        self.column_names = column_names
        self.times = times
        self.values_deg = values_deg


def parse_mot(path: Path) -> MotData:
    """Parse an OpenSim Storage (.mot/.sto) file: header block ending in `endheader`,
    an `inDegrees=yes|no` field (normalized to degrees here), then a `time <col> <col> ...`
    table."""

    lines = path.read_text(encoding="utf-8").splitlines()
    in_degrees: bool | None = None
    header_end_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("indegrees") and "=" in stripped:
            in_degrees = stripped.split("=", 1)[1].strip().lower() == "yes"
        if stripped.lower() == "endheader":
            header_end_index = index
            break
    if header_end_index is None:
        raise OpenCapCompareError(f"{path}: no 'endheader' line found -- not a valid OpenSim .mot/.sto file")
    if in_degrees is None:
        raise OpenCapCompareError(f"{path}: header missing 'inDegrees' field -- cannot determine angle units safely")

    remaining = [line for line in lines[header_end_index + 1 :] if line.strip()]
    if not remaining:
        raise OpenCapCompareError(f"{path}: no data found after 'endheader'")
    columns = remaining[0].split()
    if not columns or columns[0].lower() != "time":
        raise OpenCapCompareError(f"{path}: expected first column name to be 'time', got {columns[:1]!r}")
    column_names = columns[1:]

    times: list[float] = []
    values: list[list[float]] = []
    for line_number, raw_line in enumerate(remaining[1:], start=1):
        cells = raw_line.split()
        if len(cells) != len(columns):
            raise OpenCapCompareError(
                f"{path}: data row {line_number} has {len(cells)} cells, expected {len(columns)}"
            )
        times.append(float(cells[0]))
        values.append([float(value) for value in cells[1:]])

    values_array = np.asarray(values, dtype=np.float64)
    if not in_degrees:
        values_array = np.degrees(values_array)

    return MotData(column_names=column_names, times=np.asarray(times, dtype=np.float64), values_deg=values_array)


# ---------------------------------------------------------------------------
# Our own pipeline's predicted joints loader
# ---------------------------------------------------------------------------


class PredictedSequence:
    def __init__(self, *, joint_names: list[str], times: np.ndarray, positions_m: np.ndarray):
        self.joint_names = joint_names
        self.times = times
        self.positions_m = positions_m


def load_predicted_sequence(
    path: Path, *, player_id: int | None = None, fps: float | None = None
) -> PredictedSequence:
    """Load our BODY pipeline's per-frame joints from either the 'world' shape
    (`players[].frames[].joints_world`) or the 'labels' shape (`samples[]`). See the
    module docstring's DATA FORMATS section for the exact fields recognized."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    joint_names = payload.get("joint_names")
    effective_fps = fps if fps is not None else payload.get("fps")

    times: list[float] = []
    positions: list[Any] = []

    if "players" in payload:
        players = payload["players"]
        if not players:
            raise OpenCapCompareError(f"{path}: 'players' is empty")
        if player_id is not None:
            matches = [player for player in players if int(player.get("id", -1)) == player_id]
            if not matches:
                raise OpenCapCompareError(f"{path}: no player with id={player_id}")
            player = matches[0]
        elif len(players) == 1:
            player = players[0]
        else:
            raise OpenCapCompareError(
                f"{path}: multiple players present ({[p.get('id') for p in players]}); pass --player-id"
            )
        frames = player.get("frames", [])
        if not frames:
            raise OpenCapCompareError(f"{path}: selected player has no frames")
        for frame in frames:
            joints = frame.get("joints_world")
            if not joints:
                continue
            t = frame.get("t", frame.get("t_seconds"))
            if t is None:
                frame_idx = frame.get("frame_idx", frame.get("frame_index"))
                if frame_idx is None or not effective_fps:
                    raise OpenCapCompareError(
                        f"{path}: frame has no explicit time and no frame_idx+fps to derive one "
                        "(pass --fps or ensure the payload has a top-level 'fps' field)"
                    )
                t = float(frame_idx) / float(effective_fps)
            times.append(float(t))
            positions.append(joints)
    elif "samples" in payload:
        samples = payload["samples"]
        if not samples:
            raise OpenCapCompareError(f"{path}: 'samples' is empty")
        for sample in samples:
            if sample.get("accepted") is False:
                continue
            joints = sample.get("joints_world")
            if not joints:
                continue
            t = sample.get("t", sample.get("t_seconds"))
            if t is None:
                frame_index = sample.get("frame_index")
                if frame_index is None or not effective_fps:
                    raise OpenCapCompareError(
                        f"{path}: sample has no explicit time and no frame_index+fps to derive one "
                        "(pass --fps; 'labels'-shape payloads have no top-level 'fps' field)"
                    )
                t = float(frame_index) / float(effective_fps)
            times.append(float(t))
            positions.append(joints)
    else:
        raise OpenCapCompareError(f"{path}: expected a top-level 'players' or 'samples' key")

    positions_array = np.asarray(positions, dtype=np.float64)
    if joint_names is None:
        if positions_array.ndim == 3 and positions_array.shape[1] == len(MHR70_JOINT_NAMES):
            joint_names = list(MHR70_JOINT_NAMES)
        else:
            raise OpenCapCompareError(
                f"{path}: no top-level 'joint_names' and joint count "
                f"{positions_array.shape[1] if positions_array.ndim == 3 else '?'} != "
                f"MHR70 ({len(MHR70_JOINT_NAMES)}) -- cannot infer joint identity, refusing to guess"
            )
    elif len(joint_names) != positions_array.shape[1]:
        raise OpenCapCompareError(
            f"{path}: joint_names length {len(joint_names)} != joint axis width {positions_array.shape[1]}"
        )

    order = np.argsort(times)
    return PredictedSequence(
        joint_names=list(joint_names),
        times=np.asarray(times, dtype=np.float64)[order],
        positions_m=positions_array[order],
    )


# ---------------------------------------------------------------------------
# Marker-name resolution + resampling + angle math
# ---------------------------------------------------------------------------


def _normalize_marker_name(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def resolve_shared_joints(
    *, canonical_names: Sequence[str], trc_marker_names: Sequence[str], our_joint_names: Sequence[str]
) -> dict[str, dict[str, str]]:
    """canonical joint name -> {"trc_marker": <literal trc marker name>, "our_joint": <literal
    our-schema joint name>} for every canonical joint resolvable on BOTH sides, via
    OPENCAP_MARKER_ALIASES on the OpenCap side (exact-name match tried first) and an exact
    (normalized) name match on our side (our schemas always use the same joint-name strings
    as CORE_BODY_JOINT_NAMES -- no alias table needed there)."""

    trc_lookup = {_normalize_marker_name(name): name for name in trc_marker_names}
    our_lookup = {_normalize_marker_name(name): name for name in our_joint_names}
    resolved: dict[str, dict[str, str]] = {}
    for canonical in canonical_names:
        our_match = our_lookup.get(_normalize_marker_name(canonical))
        if our_match is None:
            continue
        trc_match = None
        for alias in (canonical, *OPENCAP_MARKER_ALIASES.get(canonical, ())):
            candidate = trc_lookup.get(_normalize_marker_name(alias))
            if candidate is not None:
                trc_match = candidate
                break
        if trc_match is not None:
            resolved[canonical] = {"trc_marker": trc_match, "our_joint": our_match}
    return resolved


def _resample_positions(
    *, source_times: np.ndarray, source_positions: np.ndarray, target_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate (frames, joints, 3) source_positions onto target_times.
    Returns (resampled, in_range_mask); entries outside [min(source_times), max(source_times)]
    are marked False and never extrapolated."""

    lo, hi = float(source_times.min()), float(source_times.max())
    in_range = (target_times >= lo) & (target_times <= hi)
    num_joints = source_positions.shape[1]
    out = np.zeros((len(target_times), num_joints, 3), dtype=np.float64)
    for joint_index in range(num_joints):
        for axis in range(3):
            out[:, joint_index, axis] = np.interp(
                target_times, source_times, source_positions[:, joint_index, axis]
            )
    return out, in_range


def _three_point_flexion_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """a,b,c: (..., 3) arrays; angle measured AT b. Returns (...,) degrees: 0deg when a-b-c
    are colinear (fully extended), increasing toward 180deg as the joint folds -- matches
    the *direction of increase* of OpenSim's `*_angle_*`/`*_flex_*` conventions (0deg=full
    extension) but is UNSIGNED (see JOINT_CORRESPONDENCE_LIMITATIONS)."""

    ba = a - b
    bc = c - b
    ba_norm = ba / np.clip(np.linalg.norm(ba, axis=-1, keepdims=True), 1e-9, None)
    bc_norm = bc / np.clip(np.linalg.norm(bc, axis=-1, keepdims=True), 1e-9, None)
    cos_angle = np.clip(np.sum(ba_norm * bc_norm, axis=-1), -1.0, 1.0)
    included_deg = np.degrees(np.arccos(cos_angle))
    return 180.0 - included_deg


def compute_predicted_angles(
    *, joint_names: Sequence[str], positions_m: np.ndarray, angle_names: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    angle_names = angle_names or list(ANGLE_DEFINITIONS)
    out: dict[str, np.ndarray] = {}
    for angle_name in angle_names:
        a_name, b_name, c_name = ANGLE_DEFINITIONS[angle_name]["triplet"]
        if a_name not in name_to_index or b_name not in name_to_index or c_name not in name_to_index:
            continue
        out[angle_name] = _three_point_flexion_deg(
            positions_m[:, name_to_index[a_name], :],
            positions_m[:, name_to_index[b_name], :],
            positions_m[:, name_to_index[c_name], :],
        )
    return out


def match_mot_angle_columns(
    *, mot_column_names: Sequence[str], angle_names: Sequence[str] | None = None
) -> dict[str, str]:
    angle_names = angle_names or list(ANGLE_DEFINITIONS)
    lookup = {name.strip().lower(): name for name in mot_column_names}
    resolved: dict[str, str] = {}
    for angle_name in angle_names:
        for candidate in ANGLE_DEFINITIONS[angle_name]["mot_columns"]:
            if candidate.lower() in lookup:
                resolved[angle_name] = lookup[candidate.lower()]
                break
    return resolved


# ---------------------------------------------------------------------------
# Comparison assembly
# ---------------------------------------------------------------------------


def compare_clip(
    *,
    trc: TrcData,
    mot: MotData | None,
    predicted: PredictedSequence,
    sync_offset_seconds: float,
    root_joint_names: Sequence[str] = DEFAULT_ROOT_JOINT_NAMES,
    floor_joint_names: Sequence[str] = DEFAULT_FLOOR_JOINT_NAMES,
    gate_variant: str = DEFAULT_GATE_VARIANT,
) -> dict[str, Any]:
    predicted_times_aligned = predicted.times + sync_offset_seconds

    resolved_joints = resolve_shared_joints(
        canonical_names=SHARED_CORE_JOINT_NAMES,
        trc_marker_names=trc.marker_names,
        our_joint_names=predicted.joint_names,
    )
    missing = [name for name in SHARED_CORE_JOINT_NAMES if name not in resolved_joints]
    if len(resolved_joints) < MIN_SHARED_JOINT_COUNT:
        raise OpenCapCompareError(
            f"only {len(resolved_joints)}/{len(SHARED_CORE_JOINT_NAMES)} shared joints resolved "
            f"(need >= {MIN_SHARED_JOINT_COUNT}) between OpenCap markers {list(trc.marker_names)} "
            f"and our joints {list(predicted.joint_names)}; refusing to score with too few "
            "correspondences. Check OPENCAP_MARKER_ALIASES against the real marker names."
        )
    ordered_canonical = [name for name in SHARED_CORE_JOINT_NAMES if name in resolved_joints]

    resampled_positions, in_range = _resample_positions(
        source_times=predicted_times_aligned, source_positions=predicted.positions_m, target_times=trc.times
    )
    if not np.any(in_range):
        raise OpenCapCompareError(
            "no time overlap between our predicted sequence (after --sync-offset-seconds) and "
            f"the OpenCap .trc timeline -- ours=[{predicted_times_aligned.min():.3f},"
            f"{predicted_times_aligned.max():.3f}]s, trc=[{trc.times.min():.3f},{trc.times.max():.3f}]s"
        )

    our_index = {name: predicted.joint_names.index(resolved_joints[name]["our_joint"]) for name in ordered_canonical}
    trc_index = {name: trc.marker_names.index(resolved_joints[name]["trc_marker"]) for name in ordered_canonical}

    predicted_matched = resampled_positions[in_range][:, [our_index[name] for name in ordered_canonical], :]
    gt_matched = trc.positions_m[in_range][:, [trc_index[name] for name in ordered_canonical], :]

    resolved_root_names = [name for name in root_joint_names if name in ordered_canonical]
    if not resolved_root_names:
        raise OpenCapCompareError(f"none of root_joint_names={list(root_joint_names)} resolved on both sides")

    position_scored = score_external_gt_clip(
        predicted_joints=predicted_matched,
        gt_joints=gt_matched,
        joint_names=ordered_canonical,
        root_joint_names=resolved_root_names,
        clip_id="opencap_session",
        gate_variant=gate_variant,
    )
    position_per_joint = per_joint_breakdown(
        predicted_joints=predicted_matched,
        gt_joints=gt_matched,
        joint_names=ordered_canonical,
        root_joint_names=resolved_root_names,
    )
    resolved_floor_names = [name for name in floor_joint_names if name in ordered_canonical]
    grounding_consistent = (
        score_grounding_consistent_variant(
            predicted_joints=predicted_matched,
            gt_joints=gt_matched,
            joint_names=ordered_canonical,
            floor_joint_names=resolved_floor_names,
        )
        if resolved_floor_names
        else None
    )

    angle_section: dict[str, Any] = {}
    if mot is not None:
        resampled_full, mot_in_range = _resample_positions(
            source_times=predicted_times_aligned, source_positions=predicted.positions_m, target_times=mot.times
        )
        if not np.any(mot_in_range):
            angle_section["_error"] = (
                "no time overlap between our predicted sequence (after --sync-offset-seconds) "
                "and the OpenCap .mot timeline"
            )
        else:
            predicted_angles = compute_predicted_angles(
                joint_names=predicted.joint_names, positions_m=resampled_full[mot_in_range]
            )
            mot_column_map = match_mot_angle_columns(mot_column_names=mot.column_names)
            mot_matched_values = mot.values_deg[mot_in_range]
            for angle_name, our_series in predicted_angles.items():
                mot_column = mot_column_map.get(angle_name)
                if mot_column is None:
                    angle_section[angle_name] = {
                        "status": "unavailable",
                        "reason": (
                            "no matching OpenCap .mot column for candidates "
                            f"{ANGLE_DEFINITIONS[angle_name]['mot_columns']}"
                        ),
                    }
                    continue
                mot_series = mot_matched_values[:, mot.column_names.index(mot_column)]
                diff = our_series - mot_series
                angle_section[angle_name] = {
                    "status": "scored",
                    "mot_column": mot_column,
                    "frame_count": int(len(diff)),
                    "mae_deg": float(np.mean(np.abs(diff))),
                    "rmse_deg": float(np.sqrt(np.mean(diff**2))),
                    "max_abs_error_deg": float(np.max(np.abs(diff))),
                    "signed_bias_deg": float(np.mean(diff)),
                    "description": ANGLE_DEFINITIONS[angle_name]["description"],
                }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "alignment_method": ALIGNMENT_METHOD_DESCRIPTION,
        "joint_correspondence_limitations": JOINT_CORRESPONDENCE_LIMITATIONS,
        "sync_offset_seconds": sync_offset_seconds,
        "shared_joint_names": ordered_canonical,
        "missing_shared_joint_names": missing,
        "trc_marker_names_found": list(trc.marker_names),
        "matched_frame_count": int(predicted_matched.shape[0]),
        "trc_total_frame_count": int(trc.positions_m.shape[0]),
        "position": {
            "gate_variant": gate_variant,
            "gate_value_m": position_scored["gate_value_m"],
            "gate_threshold_m_reference_only": DEFAULT_WORLD_MPJPE_THRESHOLD_M,
            "variants_m": {name: value["value_m"] for name, value in position_scored["variants"].items()},
            "per_joint_m": position_per_joint,
            "grounding_consistent_mpjpe": grounding_consistent,
            "root_joint_names_used": resolved_root_names,
            "floor_joint_names_used": resolved_floor_names,
        },
        "angles": angle_section,
        "not_verified": (
            "no real OpenCap capture has been scored yet -- see OWNER_OPENCAP_PROTOCOL.md. "
            "This report is either a synthetic self-test or a real-data run whose numbers "
            "are honest but not yet a passed product gate."
        ),
    }


def run_comparison(
    *,
    opencap_trc: Path,
    opencap_mot: Path | None,
    our_joints: Path,
    player_id: int | None,
    fps: float | None,
    sync_offset_seconds: float,
    root_joint_names: Sequence[str] = DEFAULT_ROOT_JOINT_NAMES,
    floor_joint_names: Sequence[str] = DEFAULT_FLOOR_JOINT_NAMES,
    gate_variant: str = DEFAULT_GATE_VARIANT,
) -> dict[str, Any]:
    trc = parse_trc(opencap_trc)
    mot = parse_mot(opencap_mot) if opencap_mot is not None else None
    predicted = load_predicted_sequence(our_joints, player_id=player_id, fps=fps)
    report = compare_clip(
        trc=trc,
        mot=mot,
        predicted=predicted,
        sync_offset_seconds=sync_offset_seconds,
        root_joint_names=root_joint_names,
        floor_joint_names=floor_joint_names,
        gate_variant=gate_variant,
    )
    report["provenance"] = {
        "opencap_trc": str(opencap_trc),
        "opencap_mot": str(opencap_mot) if opencap_mot is not None else None,
        "our_joints": str(our_joints),
        "player_id": player_id,
    }
    return report


# ---------------------------------------------------------------------------
# Synthetic fixture (task 2: validate the harness against a KNOWN injected error)
# ---------------------------------------------------------------------------


def _write_trc(path: Path, *, marker_names: Sequence[str], times: np.ndarray, positions_m: np.ndarray, data_rate: float) -> None:
    num_markers = len(marker_names)
    num_frames = len(times)
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{data_rate:.6f}\t{data_rate:.6f}\t{num_frames}\t{num_markers}\tm\t{data_rate:.6f}\t1\t{num_frames}",
    ]
    marker_header_cells = ["Frame#", "Time"]
    for name in marker_names:
        marker_header_cells.extend([name, "", ""])
    lines.append("\t".join(marker_header_cells))
    axis_header_cells = ["", ""]
    for marker_index in range(num_markers):
        axis_header_cells.extend([f"X{marker_index + 1}", f"Y{marker_index + 1}", f"Z{marker_index + 1}"])
    lines.append("\t".join(axis_header_cells))
    for frame_index in range(num_frames):
        row_cells = [str(frame_index + 1), f"{times[frame_index]:.6f}"]
        for marker_index in range(num_markers):
            x, y, z = positions_m[frame_index, marker_index]
            row_cells.extend([f"{x:.8f}", f"{y:.8f}", f"{z:.8f}"])
        lines.append("\t".join(row_cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mot(path: Path, *, column_names: Sequence[str], times: np.ndarray, values_deg: np.ndarray) -> None:
    num_frames = len(times)
    num_columns = len(column_names) + 1
    lines = [
        "Coordinates",
        "version=1",
        f"nRows={num_frames}",
        f"nColumns={num_columns}",
        "inDegrees=yes",
        "",
        "endheader",
        "\t".join(["time", *column_names]),
    ]
    for frame_index in range(num_frames):
        row_cells = [f"{times[frame_index]:.6f}"]
        row_cells.extend(f"{values_deg[frame_index, col_index]:.6f}" for col_index in range(len(column_names)))
        lines.append("\t".join(row_cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_synthetic_fixture(
    *,
    fixture_dir: Path,
    injected_rotation_deg: float = 12.0,
    injected_translation_m: tuple[float, float, float] = (5.0, -2.0, 0.4),
    injected_scale: float = 1.02,
    injected_angle_bias_deg: float = 3.0,
    frame_count: int = 90,
    fps: float = 30.0,
    noise_std_m: float = 0.0005,
    seed: int = 20260707,
) -> dict[str, Any]:
    """Builds a synthetic OpenCap-shaped (.trc + .mot) + our-schema prediction JSON pair
    with a KNOWN injected similarity transform (rotation+translation+scale) simulating
    OpenCap's independent world frame, and a KNOWN angle bias simulating a real accuracy
    gap -- so `run_comparison`'s recovered `clip_level_rigid_aligned_mpjpe` (should recover
    to ~0, since it is exactly the fit family that removes an injected similarity
    transform) and per-angle MAE (should recover the injected bias) can be asserted against
    ground truth, proving the harness arithmetic, not our BODY pipeline's real accuracy."""

    rng = np.random.default_rng(seed)
    joint_names = list(SHARED_CORE_JOINT_NAMES)
    base_positions = {
        "left_shoulder": (0.20, 1.45, 0.00),
        "right_shoulder": (-0.20, 1.45, 0.00),
        "left_elbow": (0.45, 1.20, 0.05),
        "right_elbow": (-0.45, 1.20, 0.05),
        "left_wrist": (0.55, 0.95, 0.10),
        "right_wrist": (-0.55, 0.95, 0.10),
        "left_hip": (0.12, 0.90, 0.00),
        "right_hip": (-0.12, 0.90, 0.00),
        "left_knee": (0.14, 0.48, 0.02),
        "right_knee": (-0.14, 0.48, 0.02),
        "left_ankle": (0.15, 0.08, 0.00),
        "right_ankle": (-0.15, 0.08, 0.00),
    }
    times = np.arange(frame_count, dtype=np.float64) / fps
    predicted_positions = np.zeros((frame_count, len(joint_names), 3), dtype=np.float64)
    for frame_index, t in enumerate(times):
        bend = max(0.35 * np.sin(2 * np.pi * 0.25 * t), 0.0)
        for joint_index, name in enumerate(joint_names):
            x, y, z = base_positions[name]
            if name in ("left_knee", "right_knee"):
                y = y - 0.10 * bend
                z = z + 0.05 * bend
            if name in ("left_ankle", "right_ankle"):
                z = z + 0.10 * bend
            predicted_positions[frame_index, joint_index] = (x, y, z)
    predicted_positions += rng.normal(scale=noise_std_m, size=predicted_positions.shape)

    theta = np.radians(injected_rotation_deg)
    rotation = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    translation = np.asarray(injected_translation_m, dtype=np.float64)
    gt_positions = (
        injected_scale * (rotation @ predicted_positions.reshape(-1, 3).T).T
    ).reshape(predicted_positions.shape) + translation
    gt_positions += rng.normal(scale=noise_std_m, size=gt_positions.shape)

    fixture_dir.mkdir(parents=True, exist_ok=True)
    trc_path = fixture_dir / "synthetic_opencap.trc"
    mot_path = fixture_dir / "synthetic_opencap.mot"
    our_joints_path = fixture_dir / "synthetic_our_joints.json"

    _write_trc(trc_path, marker_names=joint_names, times=times, positions_m=gt_positions, data_rate=fps)

    true_angles = compute_predicted_angles(joint_names=joint_names, positions_m=predicted_positions)
    mot_columns: list[str] = []
    mot_values_deg: list[np.ndarray] = []
    for angle_name, series in true_angles.items():
        mot_columns.append(ANGLE_DEFINITIONS[angle_name]["mot_columns"][0])
        mot_values_deg.append(series + injected_angle_bias_deg)
    _write_mot(mot_path, column_names=mot_columns, times=times, values_deg=np.stack(mot_values_deg, axis=1))

    our_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "fps": fps,
        "joint_names": joint_names,
        "players": [
            {
                "id": 1,
                "frames": [
                    {"frame_idx": idx, "joints_world": predicted_positions[idx].tolist()}
                    for idx in range(frame_count)
                ],
            }
        ],
    }
    our_joints_path.write_text(json.dumps(our_payload), encoding="utf-8")

    return {
        "trc": trc_path,
        "mot": mot_path,
        "our_joints": our_joints_path,
        "injected": {
            "rotation_deg": injected_rotation_deg,
            "translation_m": list(injected_translation_m),
            "scale": injected_scale,
            "angle_bias_deg": injected_angle_bias_deg,
        },
    }


SELF_TEST_POSITION_TOLERANCE_M = 0.003
SELF_TEST_ANGLE_TOLERANCE_DEG = 0.5


def run_self_test(*, fixture_dir: Path) -> dict[str, Any]:
    fixture = build_synthetic_fixture(fixture_dir=fixture_dir)
    report = run_comparison(
        opencap_trc=fixture["trc"],
        opencap_mot=fixture["mot"],
        our_joints=fixture["our_joints"],
        player_id=1,
        fps=None,
        sync_offset_seconds=0.0,
    )

    checks: list[dict[str, Any]] = []
    rigid_aligned = report["position"]["variants_m"]["clip_level_rigid_aligned_mpjpe"]
    checks.append(
        {
            "name": "clip_level_rigid_aligned_mpjpe_recovers_near_zero_after_injected_similarity_transform",
            "value_m": rigid_aligned,
            "tolerance_m": SELF_TEST_POSITION_TOLERANCE_M,
            "passed": rigid_aligned <= SELF_TEST_POSITION_TOLERANCE_M,
        }
    )
    raw_mpjpe = report["position"]["variants_m"]["mpjpe"]
    injected_translation_norm = float(np.linalg.norm(fixture["injected"]["translation_m"]))
    checks.append(
        {
            "name": "raw_mpjpe_reflects_injected_translation_when_unaligned",
            "value_m": raw_mpjpe,
            "expected_at_least_m": injected_translation_norm * 0.5,
            "passed": raw_mpjpe >= injected_translation_norm * 0.5,
        }
    )
    for angle_name, entry in report["angles"].items():
        if not isinstance(entry, dict) or entry.get("status") != "scored":
            continue
        bias_error = abs(entry["mae_deg"] - fixture["injected"]["angle_bias_deg"])
        checks.append(
            {
                "name": f"angle_mae_recovers_injected_bias[{angle_name}]",
                "mae_deg": entry["mae_deg"],
                "injected_bias_deg": fixture["injected"]["angle_bias_deg"],
                "tolerance_deg": SELF_TEST_ANGLE_TOLERANCE_DEG,
                "passed": bias_error <= SELF_TEST_ANGLE_TOLERANCE_DEG,
            }
        )
    scored_angle_count = sum(
        1 for entry in report["angles"].values() if isinstance(entry, dict) and entry.get("status") == "scored"
    )
    checks.append(
        {
            "name": "all_six_angle_definitions_scored",
            "scored_angle_count": scored_angle_count,
            "expected": len(ANGLE_DEFINITIONS),
            "passed": scored_angle_count == len(ANGLE_DEFINITIONS),
        }
    )

    all_passed = all(check["passed"] for check in checks)
    return {
        "self_test_passed": all_passed,
        "checks": checks,
        "fixture_injected": fixture["injected"],
        "report": report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opencap-trc", type=Path, default=None, help="OpenCap OpenSim .trc marker file")
    parser.add_argument("--opencap-mot", type=Path, default=None, help="OpenCap OpenSim IK .mot/.sto file (optional)")
    parser.add_argument("--our-joints", type=Path, default=None, help="Our BODY pipeline's joints JSON for the same clip")
    parser.add_argument("--player-id", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Override/supply fps if the payload lacks one")
    parser.add_argument(
        "--sync-offset-seconds",
        type=float,
        default=0.0,
        help="Seconds to add to our clip's timestamps to land them on OpenCap's timeline "
        "(compute via audio-clap onset match upstream; not computed by this script)",
    )
    parser.add_argument("--root-joint-names", default=",".join(DEFAULT_ROOT_JOINT_NAMES))
    parser.add_argument("--floor-joint-names", default=",".join(DEFAULT_FLOOR_JOINT_NAMES))
    parser.add_argument(
        "--gate-variant", default=DEFAULT_GATE_VARIANT, choices=sorted(VARIANT_DESCRIPTIONS)
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Ignore --opencap-trc/--opencap-mot/--our-joints; build a synthetic fixture with a "
        "known injected error and assert the harness recovers it (exit 1 on any failed check)",
    )
    parser.add_argument(
        "--self-test-fixture-dir",
        type=Path,
        default=None,
        help="Persist the self-test's synthetic fixture files here instead of a temp dir",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        if args.self_test_fixture_dir is not None:
            result = run_self_test(fixture_dir=args.self_test_fixture_dir)
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = run_self_test(fixture_dir=Path(tmp_dir))
        _write_report(result, args.out)
        print(json.dumps({"self_test_passed": result["self_test_passed"], "checks": result["checks"]}, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
        return 0 if result["self_test_passed"] else 1

    if args.opencap_trc is None or args.our_joints is None:
        parser.error("--opencap-trc and --our-joints are required unless --self-test is passed")
    if not args.opencap_trc.is_file():
        print(f"ERROR: missing --opencap-trc file: {args.opencap_trc}", file=sys.stderr)
        return 1
    if not args.our_joints.is_file():
        print(f"ERROR: missing --our-joints file: {args.our_joints}", file=sys.stderr)
        return 1
    if args.opencap_mot is not None and not args.opencap_mot.is_file():
        print(f"ERROR: missing --opencap-mot file: {args.opencap_mot}", file=sys.stderr)
        return 1

    try:
        report = run_comparison(
            opencap_trc=args.opencap_trc,
            opencap_mot=args.opencap_mot,
            our_joints=args.our_joints,
            player_id=args.player_id,
            fps=args.fps,
            sync_offset_seconds=args.sync_offset_seconds,
            root_joint_names=tuple(args.root_joint_names.split(",")),
            floor_joint_names=tuple(args.floor_joint_names.split(",")),
            gate_variant=args.gate_variant,
        )
    except OpenCapCompareError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _write_report(report, args.out)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "trc_marker_names_found"},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
