"""Local ball-labelling studio: click the ray in 2D, set depth in 3D.

The interaction this serves, in one paragraph. A human cannot place a free
point in 3D from a single video -- that is the same unsolvable problem the
solver has. But clicking the ball in the video fixes the camera **ray**, two
of three degrees of freedom, leaving exactly one: depth. So the owner clicks
in the left pane and then fixes depth in the right pane, where the court, the
net and the tracked player skeletons are drawn at their known 3D positions.
Depth is never a free 3D placement; it is one number on a line.

Three kinds of label, with genuinely different accuracy:

* ``bounce`` -- the ball is on the court, so its height is known and depth is
  *solved* by ray-plane intersection. No human depth input at all.
* ``near_player`` -- mid-flight but close to a tracked player whose 3D
  position we know, so the owner judges depth against a real object.
* ``free_flight`` -- open space, no reference. An honest guess, stored with a
  much larger sigma and never flagged as a ground-truth candidate.

This module loads a run directory read-only, builds the per-frame payload the
page needs, and serves it. It never writes into the run directory: labels go
to a separate output directory, and a guard refuses any output path that
would land inside the run.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from .ball_label_geometry import (
    BALL_RADIUS_M,
    DEFAULT_CLICK_SIGMA_PX,
    DEFAULT_PHYSICS,
    FREE_FLIGHT_DEPTH_SIGMA_M,
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    NEAR_PLAYER_DEPTH_SIGMA_M,
    NEAR_PLAYER_MAX_OFFSET_M,
    GeometryError,
    PixelRay,
    ballistic_positions,
    ballistic_speed_mps,
    bounce_depth_sigma_m,
    bounce_world_point,
    calibration_plane_residuals,
    free_flight_depth_sigma_m,
    interpolation_extra_sigma_m,
    is_in_front_of_camera,
    near_player_depth_sigma_m,
    perpendicular_sigma_m,
    pixel_extrapolation,
    project_world_to_pixel,
    ray_for_pixel,
    sigma_xyz_from_ray,
)
from .ball_label_schema import (
    KIND_BOUNCE,
    KIND_FREE_FLIGHT,
    KIND_NEAR_PLAYER,
    LABEL_FILE_NAME,
    LABEL_KINDS,
    SESSION_FILE_NAME,
    BallLabel,
    BallLabelSet,
    LabelContractError,
    read_label_set,
    write_label_set,
)
from .ball_label_studio_page import HTML, SAVE_TOKEN_PLACEHOLDER
from .ball_position_plausibility import BallPlausibilityBounds, evaluate_position
from .court_templates import get_court_template
from .external_gt_body_prediction_schema import MHR70_JOINT_NAMES

MAX_REQUEST_BYTES = 4 * 1024 * 1024

# The MHR70 bones worth drawing. Enough to read a body's pose and, critically,
# to see where the paddle hand is -- the wrists are the depth reference that
# makes a near_player label better than a guess. Copied from the replay
# renderer's CORE_MHR70_BONES so both surfaces draw the same skeleton.
CORE_BONES: tuple[tuple[int, int], ...] = (
    (0, 69),
    (69, 5),
    (69, 6),
    (5, 6),
    (5, 7),
    (7, 62),
    (6, 8),
    (8, 41),
    (5, 9),
    (6, 10),
    (9, 10),
    (9, 11),
    (11, 13),
    (10, 12),
    (12, 14),
    (13, 15),
    (13, 16),
    (13, 17),
    (14, 18),
    (14, 19),
    (14, 20),
)
CORE_JOINTS: tuple[int, ...] = tuple(sorted({index for bone in CORE_BONES for index in bone}))
CORE_JOINT_NAMES: tuple[str, ...] = tuple(MHR70_JOINT_NAMES[index] for index in CORE_JOINTS)
# Bones re-indexed into the compact CORE_JOINTS ordering the page receives.
CORE_BONE_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (CORE_JOINTS.index(a), CORE_JOINTS.index(b)) for a, b in CORE_BONES
)

# Artifacts consumed. All read-only; none is rewritten.
CALIBRATION_FILE = "court_calibration.json"
BALL_TRACK_FILE = "ball_track.json"
BALL_CANDIDATES_FILE = "ball_candidates.json"
BALL_BOUNCE_CANDIDATES_FILE = "ball_bounce_candidates.json"
ARC_SOLVED_FILE = "ball_track_arc_solved.json"
SKELETON_FILE = "skeleton3d.json"
FRAME_TIMES_FILE = "frame_times.json"
NET_PLANE_FILE = "net_plane.json"
VIDEO_CANDIDATES = ("source.mkv", "source.mp4", "source.mov", "source.MP4")

KEYBOARD_MAP: tuple[tuple[str, str], ...] = (
    ("Left / Right", "step one frame"),
    ("Shift + Left / Right", "step ten frames"),
    ("Space", "play / pause"),
    ("B / Shift+B", "next / previous detected-bounce candidate"),
    ("N / Shift+N", "next / previous unlabelled frame with a ball detection"),
    ("L / Shift+L", "next / previous existing label"),
    ("1 / 2 / 3", "set kind: bounce / near-player / free-flight"),
    ("K", "cycle label kind"),
    ("Click on video", "set the ray (2 of 3 DOF)"),
    ("Up / Down", "depth -/+ 0.10 m along the ray"),
    ("Shift + Up / Down", "depth -/+ 0.01 m (fine)"),
    ("Drag in 3D view", "orbit the 3D camera"),
    ("Enter", "save the label at this frame (autosaves)"),
    ("P", "load the pipeline prefill (or detector pixel) to correct — Enter confirms"),
    ("C", "cycle human confidence low / medium / high"),
    ("Backspace or Delete", "delete the label at this frame"),
    ("I", "propose a ballistic arc between the surrounding labels"),
    ("Shift + I", "accept the proposed arc as free-flight labels"),
    ("Z / Shift+Z", "video zoom in / out"),
    ("M", "toggle the magnifier"),
    ("G", "go to frame"),
    ("?", "toggle this legend"),
)


class StudioError(RuntimeError):
    """Raised when a run directory cannot back a labelling session."""


# ---------------------------------------------------------------------------
# Loading a run directory (read-only)
# ---------------------------------------------------------------------------


@dataclass
class ClipBundle:
    """Everything the page needs about one clip, loaded once at startup."""

    run_dir: Path
    clip_id: str
    video_path: Path
    calibration: dict[str, Any]
    fps: float
    frame_count: int
    image_size: tuple[int, int]
    frame_times_s: list[float]
    detections: dict[int, dict[str, Any]] = field(default_factory=dict)
    candidates: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    prefill: dict[int, dict[str, Any]] = field(default_factory=dict)
    prefill_refused_implausible: int = 0
    players: list[dict[str, Any]] = field(default_factory=list)
    bounce_candidate_frames: list[int] = field(default_factory=list)
    physics: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PHYSICS))
    net_plane: dict[str, Any] = field(default_factory=dict)
    plane_residuals: dict[str, Any] = field(default_factory=dict)
    missing_artifacts: list[str] = field(default_factory=list)
    source_artifacts: dict[str, Any] = field(default_factory=dict)

    def timestamp(self, frame: int) -> float:
        index = max(0, min(int(frame), len(self.frame_times_s) - 1))
        if not self.frame_times_s:
            return round(int(frame) / max(1e-6, self.fps), 6)
        return float(self.frame_times_s[index])

    def joints_at(self, frame: int) -> list[dict[str, Any]]:
        """Core-joint world positions for every player tracked at ``frame``."""

        out: list[dict[str, Any]] = []
        for player in self.players:
            entry = player["frames"].get(int(frame))
            if entry is None:
                continue
            out.append({"player_id": player["id"], **entry})
        return out


def load_clip_bundle(
    run_dir: str | Path, *, clip_id: str | None = None, video_path: str | Path | None = None
) -> ClipBundle:
    """Load a pipeline run directory read-only. Nothing here writes."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise StudioError(f"run directory does not exist: {root}")

    calibration_path = root / CALIBRATION_FILE
    if not calibration_path.is_file():
        raise StudioError(
            f"{calibration_path} is required: without a calibration a clicked pixel "
            f"cannot be turned into a world ray"
        )
    calibration = _load_json(calibration_path)

    if video_path is not None:
        video = Path(video_path).expanduser().resolve()
        if not video.is_file():
            raise StudioError(f"--video does not exist: {video}")
    else:
        # Several archived runs symlink source.mp4 to a path on the GPU box that
        # produced them. That is a missing video here, not a usable one.
        video = _resolve_video(root)
    video_path = video
    missing: list[str] = []

    frame_times_payload = _load_json_or_none(root / FRAME_TIMES_FILE)
    if frame_times_payload is None:
        missing.append(FRAME_TIMES_FILE)
        frame_times: list[float] = []
        fps = 30.0
        frame_count = 0
    else:
        frame_times = [
            float(entry.get("pts_s", 0.0)) for entry in frame_times_payload.get("frames", [])
        ]
        fps = float(frame_times_payload.get("fps") or 30.0)
        frame_count = int(frame_times_payload.get("frame_count") or len(frame_times))

    image_size = _image_size(calibration)

    bundle = ClipBundle(
        run_dir=root,
        clip_id=clip_id or _infer_clip_id(root, frame_times_payload),
        video_path=video,
        calibration=calibration,
        fps=fps,
        frame_count=frame_count,
        image_size=image_size,
        frame_times_s=frame_times,
    )
    bundle.source_artifacts[CALIBRATION_FILE] = _artifact_ref(calibration_path, root)

    _load_detections(bundle, missing)
    _load_candidates(bundle, missing)
    _load_prefill(bundle, missing)
    _load_skeletons(bundle, missing)
    _load_bounce_candidates(bundle, missing)

    net_plane = _load_json_or_none(root / NET_PLANE_FILE)
    if net_plane is None:
        missing.append(NET_PLANE_FILE)
    else:
        bundle.net_plane = net_plane
        bundle.source_artifacts[NET_PLANE_FILE] = _artifact_ref(root / NET_PLANE_FILE, root)

    if bundle.frame_count <= 0:
        bundle.frame_count = _probe_frame_count(video)
        if bundle.frame_count <= 0:
            raise StudioError(f"could not determine a frame count for {video}")
    if not bundle.frame_times_s:
        bundle.frame_times_s = [
            round(index / max(1e-6, bundle.fps), 6) for index in range(bundle.frame_count)
        ]

    bundle.plane_residuals = calibration_plane_residuals(calibration)
    bundle.missing_artifacts = missing
    return bundle


def _load_detections(bundle: ClipBundle, missing: list[str]) -> None:
    path = bundle.run_dir / BALL_TRACK_FILE
    payload = _load_json_or_none(path)
    if payload is None:
        missing.append(BALL_TRACK_FILE)
        return
    bundle.source_artifacts[BALL_TRACK_FILE] = _artifact_ref(path, bundle.run_dir)
    for index, entry in enumerate(payload.get("frames") or []):
        if not isinstance(entry, Mapping):
            continue
        xy = entry.get("xy")
        if not entry.get("visible") or not isinstance(xy, Sequence) or len(xy) != 2:
            continue
        bundle.detections[index] = {
            "pixel_xy": [float(xy[0]), float(xy[1])],
            "conf": float(entry.get("conf") or 0.0),
            "approx": bool(entry.get("approx")),
        }


def _load_candidates(bundle: ClipBundle, missing: list[str]) -> None:
    path = bundle.run_dir / BALL_CANDIDATES_FILE
    payload = _load_json_or_none(path)
    if payload is None:
        missing.append(BALL_CANDIDATES_FILE)
        return
    bundle.source_artifacts[BALL_CANDIDATES_FILE] = _artifact_ref(path, bundle.run_dir)
    for entry in payload.get("frames") or []:
        if not isinstance(entry, Mapping):
            continue
        frame = int(entry.get("frame", -1))
        if frame < 0:
            continue
        items: list[dict[str, Any]] = []
        for candidate in entry.get("candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            xy = candidate.get("xy") or candidate.get("pixel_xy")
            if not isinstance(xy, Sequence) or len(xy) != 2:
                continue
            items.append(
                {
                    "pixel_xy": [round(float(xy[0]), 2), round(float(xy[1]), 2)],
                    "conf": round(float(candidate.get("conf") or candidate.get("score") or 0.0), 4),
                }
            )
        if items:
            bundle.candidates[frame] = items[:5]


def _load_prefill(bundle: ClipBundle, missing: list[str]) -> None:
    """Seed each frame with the solver's current 3D guess, clearly marked.

    Correcting a prefill is far faster than creating a label from nothing, and
    every correction is itself a measurement of the solver's error. Prefills
    are never labels: they carry ``source`` and stay visually distinct until
    the owner confirms one.

    Physically impossible prefills are REFUSED rather than offered. Stored
    ``ball_track_arc_solved.json`` artifacts predate the plausibility gate and
    can carry positions twenty metres airborne or below the court; seeding the
    labeller with one of those is worse than offering nothing, because the
    reviewer has to drag it back across the whole frame. The count is recorded
    so the refusal is visible rather than silent.
    """

    path = bundle.run_dir / ARC_SOLVED_FILE
    payload = _load_json_or_none(path)
    if payload is None:
        missing.append(ARC_SOLVED_FILE)
        return
    refused_implausible = 0
    bundle.source_artifacts[ARC_SOLVED_FILE] = _artifact_ref(path, bundle.run_dir)
    physics = payload.get("physics_parameters")
    if isinstance(physics, Mapping):
        bundle.physics = {**DEFAULT_PHYSICS, **{str(k): v for k, v in physics.items()}}
    for index, entry in enumerate(payload.get("frames") or []):
        if not isinstance(entry, Mapping):
            continue
        world = entry.get("world_xyz")
        if not isinstance(world, Sequence) or len(world) != 3:
            continue
        verdict = evaluate_position(
            (float(world[0]), float(world[1]), float(world[2])),
            bounds=BallPlausibilityBounds(),
        )
        if not verdict.get("plausible", True):
            refused_implausible += 1
            continue
        try:
            pixel = project_world_to_pixel(bundle.calibration, world)
        except (GeometryError, ValueError, KeyError):
            continue
        bundle.prefill[index] = {
            "source": ARC_SOLVED_FILE,
            "world_xyz_m": [round(float(v), 6) for v in world],
            "pixel_xy": [round(pixel[0], 2), round(pixel[1], 2)],
            "band": entry.get("band"),
            "sigma_m": entry.get("sigma_m"),
            "not_ground_truth": True,
        }
    bundle.prefill_refused_implausible = refused_implausible


def _load_skeletons(bundle: ClipBundle, missing: list[str]) -> None:
    """Player skeletons at their tracked 3D positions, in both views.

    These are the mechanism that makes a near-player label better than a
    guess: they are objects at known 3D positions, so the owner can judge the
    ball's depth against a shoulder, a head or a paddle hand.
    """

    path = bundle.run_dir / SKELETON_FILE
    payload = _load_json_or_none(path)
    if payload is None:
        missing.append(SKELETON_FILE)
        return
    bundle.source_artifacts[SKELETON_FILE] = _artifact_ref(path, bundle.run_dir)
    for player in payload.get("players") or []:
        if not isinstance(player, Mapping):
            continue
        frames: dict[int, dict[str, Any]] = {}
        for entry in player.get("frames") or []:
            if not isinstance(entry, Mapping):
                continue
            joints = entry.get("joints_world")
            if not isinstance(joints, Sequence) or len(joints) < max(CORE_JOINTS) + 1:
                continue
            confidences = entry.get("joint_conf") or []
            world: list[list[float]] = []
            pixels: list[list[float] | None] = []
            confs: list[float] = []
            for joint_index in CORE_JOINTS:
                point = joints[joint_index]
                world.append([round(float(point[0]), 4), round(float(point[1]), 4), round(float(point[2]), 4)])
                try:
                    if is_in_front_of_camera(bundle.calibration, point):
                        u, v = project_world_to_pixel(bundle.calibration, point)
                        pixels.append([round(u, 1), round(v, 1)])
                    else:
                        pixels.append(None)
                except (GeometryError, ValueError, KeyError):
                    pixels.append(None)
                confs.append(
                    round(float(confidences[joint_index]), 3)
                    if joint_index < len(confidences)
                    else 0.0
                )
            frames[int(entry.get("frame_idx", -1))] = {
                "world": world,
                "pixels": pixels,
                "conf": confs,
                "implausible": bool(entry.get("skeleton_implausible")),
            }
        if frames:
            bundle.players.append({"id": int(player.get("id", len(bundle.players) + 1)), "frames": frames})


def _load_bounce_candidates(bundle: ClipBundle, missing: list[str]) -> None:
    frames: set[int] = set()
    path = bundle.run_dir / BALL_BOUNCE_CANDIDATES_FILE
    payload = _load_json_or_none(path)
    if payload is None:
        missing.append(BALL_BOUNCE_CANDIDATES_FILE)
    else:
        bundle.source_artifacts[BALL_BOUNCE_CANDIDATES_FILE] = _artifact_ref(path, bundle.run_dir)
        for candidate in payload.get("candidates") or []:
            if isinstance(candidate, Mapping) and candidate.get("frame") is not None:
                frames.add(int(candidate["frame"]))
    arc = _load_json_or_none(bundle.run_dir / ARC_SOLVED_FILE)
    if arc is not None:
        for anchor in arc.get("anchors") or []:
            if isinstance(anchor, Mapping) and anchor.get("kind") == "bounce":
                frames.add(int(anchor.get("frame", -1)))
    bundle.bounce_candidate_frames = sorted(frame for frame in frames if frame >= 0)


# ---------------------------------------------------------------------------
# Frame extraction (into the OUTPUT directory, never the run directory)
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: str | Path,
    frames_dir: str | Path,
    *,
    expected_count: int,
    quality: int = 3,
    force: bool = False,
) -> dict[str, Any]:
    """Decode the clip to JPEGs once so the page can step frames exactly.

    HTML5 video seeking is not frame-accurate, and a labelling tool whose
    frame index can drift from the artifact index is worse than useless. One
    ffmpeg pass costs a few seconds and buys exact 2D/3D synchronisation.
    """

    out = Path(frames_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.jpg"))
    if not force and len(existing) >= expected_count > 0:
        return {"extracted": False, "reason": "cache_hit", "frame_count": len(existing), "dir": str(out)}
    for stale in existing:
        stale.unlink(missing_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise StudioError("ffmpeg is required to extract frames but was not found on PATH")
    started = time.monotonic()
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vsync",
            "0",
            "-q:v",
            str(int(quality)),
            "-start_number",
            "0",
            str(out / "%06d.jpg"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise StudioError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()[:800]}")
    produced = len(list(out.glob("*.jpg")))
    return {
        "extracted": True,
        "frame_count": produced,
        "elapsed_s": round(time.monotonic() - started, 2),
        "dir": str(out),
        "matches_expected": produced == expected_count,
    }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class StudioSession:
    """A labelling session: the clip, the labels so far, and where they live."""

    bundle: ClipBundle
    out_dir: Path
    label_path: Path
    session_path: Path
    frames_dir: Path
    label_set: BallLabelSet
    click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX
    near_player_sigma_m: float = NEAR_PLAYER_DEPTH_SIGMA_M
    free_flight_sigma_m: float = FREE_FLIGHT_DEPTH_SIGMA_M

    def save(self) -> dict[str, Any]:
        write_label_set(self.label_path, self.label_set)
        return self.label_set.summary()

    def read_session(self) -> dict[str, Any]:
        if not self.session_path.is_file():
            return {}
        try:
            payload = json.loads(self.session_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def write_session(self, **updates: Any) -> dict[str, Any]:
        payload = self.read_session()
        payload.update(updates)
        payload["clip_id"] = self.bundle.clip_id
        payload["updated_at_epoch_s"] = round(time.time(), 3)
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return payload


def open_session(
    bundle: ClipBundle,
    out_dir: str | Path,
    *,
    click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX,
    near_player_sigma_m: float = NEAR_PLAYER_DEPTH_SIGMA_M,
    free_flight_sigma_m: float = FREE_FLIGHT_DEPTH_SIGMA_M,
) -> StudioSession:
    """Open (or resume) a session, refusing to write inside the run directory."""

    out = Path(out_dir).expanduser().resolve()
    _guard_output_dir(out, bundle.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_path = out / LABEL_FILE_NAME

    if label_path.is_file():
        label_set = read_label_set(label_path)
        if label_set.clip_id != bundle.clip_id:
            raise StudioError(
                f"{label_path} holds labels for clip {label_set.clip_id!r}, not "
                f"{bundle.clip_id!r}; point --out at a different directory"
            )
    else:
        label_set = BallLabelSet(
            clip_id=bundle.clip_id,
            fps=bundle.fps,
            frame_count=bundle.frame_count,
            image_size=bundle.image_size,
            source_artifacts=bundle.source_artifacts,
            calibration_evidence=_calibration_evidence(bundle),
        )
    # Refresh the evidence block on resume so a re-solved calibration cannot
    # silently keep an old accuracy claim attached to new labels.
    label_set.calibration_evidence = _calibration_evidence(bundle)
    label_set.source_artifacts = bundle.source_artifacts
    label_set.frame_count = bundle.frame_count
    label_set.fps = bundle.fps
    label_set.image_size = bundle.image_size

    return StudioSession(
        bundle=bundle,
        out_dir=out,
        label_path=label_path,
        session_path=out / SESSION_FILE_NAME,
        frames_dir=out / "frames",
        label_set=label_set,
        click_sigma_px=click_sigma_px,
        near_player_sigma_m=near_player_sigma_m,
        free_flight_sigma_m=free_flight_sigma_m,
    )


def _guard_output_dir(out: Path, run_dir: Path) -> None:
    try:
        out.relative_to(run_dir)
    except ValueError:
        return
    raise StudioError(
        f"refusing to write labels inside the run directory ({run_dir}): raw pipeline "
        f"observations are immutable. Point --out somewhere else."
    )


def _calibration_evidence(bundle: ClipBundle) -> dict[str, Any]:
    calibration = bundle.calibration
    return {
        "source": calibration.get("source"),
        "metric_confidence": calibration.get("metric_confidence"),
        "reprojection_error_px": calibration.get("reprojection_error_px"),
        "sha256": _sha256(bundle.run_dir / CALIBRATION_FILE),
        "plane_residual_check": bundle.plane_residuals,
        "note": (
            "plane_residual_check is this clip's measured bounce accuracy floor: even a "
            "perfect click inherits it. It is the sigma floor for every bounce label."
        ),
    }


# ---------------------------------------------------------------------------
# Building labels
# ---------------------------------------------------------------------------


def nearest_player_reference(
    bundle: ClipBundle, frame: int, ray: PixelRay
) -> dict[str, Any] | None:
    """Closest tracked joint to the ray: the depth reference for kind 2."""

    best: dict[str, Any] | None = None
    for player in bundle.joints_at(frame):
        for local_index, world in enumerate(player["world"]):
            depth = ray.depth_of(world)
            if depth <= 0.0:
                continue
            offset = ray.offset_from(world)
            if best is None or offset < best["offset_from_ray_m"]:
                best = {
                    "player_id": player["player_id"],
                    "joint_index": CORE_JOINTS[local_index],
                    "joint_name": CORE_JOINT_NAMES[local_index],
                    "joint_world_m": [round(v, 4) for v in world],
                    "offset_from_ray_m": round(offset, 4),
                    "depth_along_ray_m": round(depth, 4),
                    "joint_confidence": player["conf"][local_index],
                }
    return best


def solve_click(session: StudioSession, frame: int, pixel_xy: Sequence[float]) -> dict[str, Any]:
    """Turn a click into a ray plus everything the page needs to fix depth.

    All camera math stays here. The page only ever computes
    ``origin + t * direction``, so a projection convention can never drift
    between the tested Python and the untested browser.
    """

    bundle = session.bundle
    ray = ray_for_pixel(bundle.calibration, pixel_xy)

    bounce: dict[str, Any]
    try:
        world, depth = bounce_world_point(bundle.calibration, pixel_xy)
        sigma, basis = bounce_depth_sigma_m(
            bundle.calibration,
            pixel_xy,
            click_sigma_px=session.click_sigma_px,
            plane_residual_m=_plane_floor(bundle),
        )
        bounce = {
            "available": True,
            "depth_along_ray_m": round(depth, 6),
            "world_xyz_m": [round(v, 6) for v in world],
            "sigma_along_ray_m": round(sigma, 6),
            "uncertainty_basis": basis,
        }
    except GeometryError as exc:
        bounce = {"available": False, "reason": str(exc)}

    reference = nearest_player_reference(bundle, frame, ray)
    prefill = bundle.prefill.get(int(frame))
    prefill_depth = None
    if prefill is not None:
        candidate = ray.depth_of(prefill["world_xyz_m"])
        if candidate > 0.0:
            prefill_depth = round(candidate, 6)

    if bounce.get("available"):
        suggested = bounce["depth_along_ray_m"]
        suggested_kind = KIND_BOUNCE
    elif reference is not None and reference["offset_from_ray_m"] <= NEAR_PLAYER_MAX_OFFSET_M:
        suggested = reference["depth_along_ray_m"]
        suggested_kind = KIND_NEAR_PLAYER
    elif prefill_depth is not None:
        suggested = prefill_depth
        suggested_kind = KIND_FREE_FLIGHT
    else:
        suggested = round(min(MAX_DEPTH_M, max(MIN_DEPTH_M, 10.0)), 6)
        suggested_kind = KIND_FREE_FLIGHT

    return {
        "frame": int(frame),
        "ray": ray.to_json_dict(),
        "bounce": bounce,
        "near_player": reference,
        "near_player_usable": bool(
            reference is not None and reference["offset_from_ray_m"] <= NEAR_PLAYER_MAX_OFFSET_M
        ),
        "prefill_depth_along_ray_m": prefill_depth,
        "suggested_depth_along_ray_m": suggested,
        "suggested_kind": suggested_kind,
        "depth_range_m": [MIN_DEPTH_M, MAX_DEPTH_M],
    }


def build_label(
    session: StudioSession,
    *,
    frame: int,
    pixel_xy: Sequence[float],
    kind: str,
    depth_along_ray_m: float | None,
    human_confidence: str,
    origin: str,
    notes: str = "",
    depth_source: str | None = None,
    extra_sigma_along_m: float = 0.0,
) -> BallLabel:
    """Assemble one validated label. Fail-closed on every honesty boundary."""

    bundle = session.bundle
    if kind not in LABEL_KINDS:
        raise LabelContractError(f"unknown kind {kind!r}; known: {sorted(LABEL_KINDS)}")
    ray = ray_for_pixel(bundle.calibration, pixel_xy)
    reference = nearest_player_reference(bundle, frame, ray)

    if kind == KIND_BOUNCE:
        world, depth = bounce_world_point(bundle.calibration, pixel_xy)
        sigma_along, basis = bounce_depth_sigma_m(
            bundle.calibration,
            pixel_xy,
            click_sigma_px=session.click_sigma_px,
            plane_residual_m=_plane_floor(bundle),
        )
        resolved_depth_source = "ray_plane_intersection"
        near_player_payload = None
    else:
        if depth_along_ray_m is None:
            raise LabelContractError(f"kind={kind!r} needs an explicit depth along the ray")
        depth = float(depth_along_ray_m)
        world = ray.at_depth(depth)
        resolved_depth_source = depth_source or "human_drag"
        if kind == KIND_NEAR_PLAYER:
            if reference is None or reference["offset_from_ray_m"] > NEAR_PLAYER_MAX_OFFSET_M:
                raise LabelContractError(
                    f"no tracked player joint lies within {NEAR_PLAYER_MAX_OFFSET_M} m of this "
                    f"ray at frame {frame}, so there is no depth reference to judge against. "
                    f"Record this as free_flight instead."
                )
            sigma_along, basis = near_player_depth_sigma_m(
                reference["offset_from_ray_m"], base_sigma_m=session.near_player_sigma_m
            )
            near_player_payload = reference
        else:
            sigma_along, basis = free_flight_depth_sigma_m(
                base_sigma_m=session.free_flight_sigma_m
            )
            near_player_payload = None

    if extra_sigma_along_m > 0.0:
        sigma_along = math.sqrt(sigma_along**2 + float(extra_sigma_along_m) ** 2)
        basis = f"{basis} Plus {float(extra_sigma_along_m):.3f} m for drag neglected by the interpolated arc."

    sigma_perp = perpendicular_sigma_m(
        bundle.calibration, depth, click_sigma_px=session.click_sigma_px
    )
    sigma_xyz = sigma_xyz_from_ray(
        ray.direction, sigma_along_m=sigma_along, sigma_perp_m=sigma_perp
    )

    prefill_payload = None
    stored_prefill = bundle.prefill.get(int(frame))
    if origin != "fresh":
        if stored_prefill is None:
            raise LabelContractError(
                f"origin={origin!r} at frame {frame} but no pipeline prefill exists there"
            )
        prefill_payload = {
            **stored_prefill,
            "delta_m": round(math.dist(world, stored_prefill["world_xyz_m"]), 6),
            "delta_px": round(math.dist(tuple(pixel_xy), tuple(stored_prefill["pixel_xy"])), 3),
        }

    label = BallLabel(
        frame=int(frame),
        timestamp_s=bundle.timestamp(frame),
        pixel_xy=(float(pixel_xy[0]), float(pixel_xy[1])),
        world_xyz_m=(float(world[0]), float(world[1]), float(world[2])),
        kind=kind,
        depth_along_ray_m=float(depth),
        ray_origin_m=ray.origin,
        ray_direction_unit=ray.direction,
        depth_source=resolved_depth_source,
        sigma_xyz_m=sigma_xyz,
        sigma_along_ray_m=float(sigma_along),
        sigma_perp_m=float(sigma_perp),
        uncertainty_basis=basis,
        human_confidence=str(human_confidence),
        origin=str(origin),
        prefill=prefill_payload,
        near_player=near_player_payload,
        notes=str(notes or ""),
        extrapolation=pixel_extrapolation(bundle.calibration, pixel_xy),
    )
    label.validate()
    return label


def propose_interpolation(session: StudioSession, frame: int) -> dict[str, Any]:
    """Fill the gap around ``frame`` with the ballistic arc between two labels.

    Labelling every frame by hand is what makes a labelling session collapse.
    Two labels define one drag-free parabola; the frames between them are
    proposals the owner accepts or corrects, never labels in their own right.
    """

    bundle = session.bundle
    labels = session.label_set.labels
    before = [item for item in labels if item.frame < int(frame)]
    after = [item for item in labels if item.frame > int(frame)]
    if not before or not after:
        return {
            "available": False,
            "reason": (
                "interpolation needs a saved label on both sides of this frame; "
                f"found {len(before)} before and {len(after)} after"
            ),
        }
    start = before[-1]
    end = after[0]
    frames = list(range(start.frame + 1, end.frame))
    if not frames:
        return {"available": False, "reason": "the surrounding labels are adjacent frames"}

    times = [bundle.timestamp(index) for index in frames]
    positions = ballistic_positions(
        start.world_xyz_m, start.timestamp_s, end.world_xyz_m, end.timestamp_s, times
    )
    span_s = end.timestamp_s - start.timestamp_s
    speed = ballistic_speed_mps(
        start.world_xyz_m, start.timestamp_s, end.world_xyz_m, end.timestamp_s
    )
    extra_sigma = interpolation_extra_sigma_m(span_s, speed, bundle.physics)

    samples: list[dict[str, Any]] = []
    below_court = 0
    for index, position in zip(frames, positions):
        try:
            pixel = project_world_to_pixel(bundle.calibration, position)
            visible = is_in_front_of_camera(bundle.calibration, position)
        except (GeometryError, ValueError, KeyError):
            pixel = (float("nan"), float("nan"))
            visible = False
        detection = bundle.detections.get(index)
        residual = None
        if detection is not None and visible and math.isfinite(pixel[0]):
            residual = round(math.dist(pixel, tuple(detection["pixel_xy"])), 2)
        if position[2] < 0.0:
            below_court += 1
        samples.append(
            {
                "frame": index,
                "timestamp_s": bundle.timestamp(index),
                "world_xyz_m": [round(v, 4) for v in position],
                "pixel_xy": [round(pixel[0], 2), round(pixel[1], 2)] if math.isfinite(pixel[0]) else None,
                "in_front_of_camera": visible,
                "below_court": position[2] < 0.0,
                "detector_residual_px": residual,
                "already_labelled": session.label_set.get(index) is not None,
            }
        )

    residuals = [item["detector_residual_px"] for item in samples if item["detector_residual_px"] is not None]
    return {
        "available": True,
        "status": "proposal",
        "not_a_label": True,
        "start_frame": start.frame,
        "end_frame": end.frame,
        "span_s": round(span_s, 4),
        "fitted_speed_mps": round(speed, 3),
        "extra_sigma_along_ray_m": round(extra_sigma, 4),
        "below_court_count": below_court,
        "physically_implausible": below_court > 0,
        "detector_residual_px": {
            "count": len(residuals),
            "median": round(sorted(residuals)[len(residuals) // 2], 2) if residuals else None,
            "max": round(max(residuals), 2) if residuals else None,
        },
        "samples": samples,
        "warning": (
            "Accepting this arc creates free_flight labels. A drag-free parabola is only a "
            "good approximation over a short span; sigma is inflated accordingly and these "
            "are never ground-truth candidates."
        ),
    }


def accept_interpolation(
    session: StudioSession, *, frames: Sequence[int], human_confidence: str = "low"
) -> dict[str, Any]:
    """Turn accepted proposals into honest, clearly-marked free-flight labels."""

    proposal = propose_interpolation(session, int(frames[0]) if frames else 0)
    if not proposal.get("available"):
        return {"accepted": 0, "reason": proposal.get("reason")}
    wanted = {int(index) for index in frames}
    accepted = 0
    skipped: list[dict[str, Any]] = []
    for sample in proposal["samples"]:
        if sample["frame"] not in wanted:
            continue
        if not sample["in_front_of_camera"] or sample["pixel_xy"] is None:
            skipped.append({"frame": sample["frame"], "reason": "not visible in the image"})
            continue
        try:
            label = build_label(
                session,
                frame=sample["frame"],
                pixel_xy=sample["pixel_xy"],
                kind=KIND_FREE_FLIGHT,
                depth_along_ray_m=ray_for_pixel(
                    session.bundle.calibration, sample["pixel_xy"]
                ).depth_of(sample["world_xyz_m"]),
                human_confidence=human_confidence,
                origin="fresh",
                depth_source="interpolated_arc",
                extra_sigma_along_m=proposal["extra_sigma_along_ray_m"],
                notes=(
                    f"accepted from a ballistic interpolation between frames "
                    f"{proposal['start_frame']} and {proposal['end_frame']}"
                ),
            )
        except (LabelContractError, GeometryError) as exc:
            skipped.append({"frame": sample["frame"], "reason": str(exc)})
            continue
        session.label_set.upsert(label)
        accepted += 1
    summary = session.save()
    return {"accepted": accepted, "skipped": skipped, "summary": summary}


def _plane_floor(bundle: ClipBundle) -> float:
    residuals = bundle.plane_residuals
    if residuals.get("available"):
        return float(residuals.get("median_m") or 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# The payload the page boots from
# ---------------------------------------------------------------------------


def build_page_state(session: StudioSession) -> dict[str, Any]:
    """One blob with everything static, so frame stepping never round-trips."""

    bundle = session.bundle
    template = get_court_template("pickleball")
    skeleton_frames: dict[str, list[dict[str, Any]]] = {}
    for player in bundle.players:
        for frame_index, entry in player["frames"].items():
            skeleton_frames.setdefault(str(frame_index), []).append(
                {
                    "player_id": player["id"],
                    "world": entry["world"],
                    "pixels": entry["pixels"],
                    "implausible": entry["implausible"],
                }
            )
    return {
        "clip_id": bundle.clip_id,
        "run_dir": str(bundle.run_dir),
        "out_dir": str(session.out_dir),
        "label_path": str(session.label_path),
        "fps": bundle.fps,
        "frame_count": bundle.frame_count,
        "image_size": list(bundle.image_size),
        "frame_times_s": [round(value, 4) for value in bundle.frame_times_s],
        "detections": {str(k): v for k, v in bundle.detections.items()},
        "candidates": {str(k): v for k, v in bundle.candidates.items()},
        "prefill": {str(k): v for k, v in bundle.prefill.items()},
        "skeletons": skeleton_frames,
        "bone_pairs": [list(pair) for pair in CORE_BONE_PAIRS],
        "joint_names": list(CORE_JOINT_NAMES),
        "bounce_candidate_frames": bundle.bounce_candidate_frames,
        "missing_artifacts": bundle.missing_artifacts,
        "camera_origin_m": [round(v, 4) for v in ray_for_pixel(bundle.calibration, [bundle.image_size[0] / 2, bundle.image_size[1] / 2]).origin],
        "court": {
            "line_segments_m": {
                name: [list(points[0]), list(points[1])]
                for name, points in template.line_segments_m.items()
            },
            "corners_m": [list(corner) for corner in template.corners_m],
            "width_m": template.width_m,
            "length_m": template.length_m,
            "net_center_height_m": template.center_net_height_m,
            "net_post_height_m": template.post_net_height_m,
            "net_width_m": template.net_width_m,
        },
        "ball_radius_m": BALL_RADIUS_M,
        "depth_range_m": [MIN_DEPTH_M, MAX_DEPTH_M],
        "near_player_max_offset_m": NEAR_PLAYER_MAX_OFFSET_M,
        "keyboard_map": [list(item) for item in KEYBOARD_MAP],
        "calibration_evidence": _calibration_evidence(bundle),
        "kind_help": {
            KIND_BOUNCE: (
                "Ball on the court. Height is known, so depth is SOLVED by ray-plane "
                "intersection. No depth judgement needed. Highest value label."
            ),
            KIND_NEAR_PLAYER: (
                "Mid-flight but close to a tracked player. Judge depth against the "
                "skeleton. Good, but a human estimate."
            ),
            KIND_FREE_FLIGHT: (
                "Open space, no reference. Honest guess only. Stored with a large sigma "
                "and never treated as ground truth."
            ),
        },
        "labels": [label.to_json_dict() for label in session.label_set.labels],
        "summary": session.label_set.summary(),
        "session": session.read_session(),
        "verified": 0,
    }


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def run_studio_server(
    session: StudioSession, *, host: str = "127.0.0.1", port: int = 8791
) -> ThreadingHTTPServer:
    """Serve the studio. Returns the server; caller drives ``serve_forever``."""

    handler = _make_handler()
    bound = _free_port(port) if host in {"127.0.0.1", "localhost"} else port
    server = ThreadingHTTPServer((host, bound), handler)
    server.session = session  # type: ignore[attr-defined]
    # Per-process token required on every mutating request. Labels are hours of
    # human work; a stray page in another tab must not be able to overwrite them.
    server.save_token = secrets.token_urlsafe(32)  # type: ignore[attr-defined]
    return server


def _make_handler() -> type[BaseHTTPRequestHandler]:
    class BallLabelStudioHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        @property
        def session(self) -> StudioSession:
            return self.server.session  # type: ignore[attr-defined]

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                token = getattr(self.server, "save_token", "")
                self._send_text(HTML.replace(SAVE_TOKEN_PLACEHOLDER, token))
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if parsed.path == "/api/state":
                self._send_json(build_page_state(self.session))
                return
            if parsed.path == "/api/interpolate":
                params = parse_qs(parsed.query)
                try:
                    frame = int(params.get("frame", ["0"])[0])
                except ValueError:
                    self._send_json({"error": "frame must be an integer"}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(propose_interpolation(self.session, frame))
                return
            if parsed.path.startswith("/frame/"):
                self._serve_frame(parsed.path[len("/frame/") :])
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            routes = {
                "/api/ray": self._post_ray,
                "/api/label": self._post_label,
                "/api/label/delete": self._post_delete,
                "/api/interpolate/accept": self._post_accept_interpolation,
                "/api/session": self._post_session,
            }
            route = routes.get(parsed.path)
            if route is None:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            if not self._require_token():
                return
            try:
                payload = self._read_json()
                result = route(payload)
            except (LabelContractError, GeometryError, StudioError, ValueError, KeyError) as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)

        # -- routes ---------------------------------------------------------

        def _post_ray(self, payload: Mapping[str, Any]) -> dict[str, Any]:
            return solve_click(
                self.session, int(payload["frame"]), _pixel(payload["pixel_xy"])
            )

        def _post_label(self, payload: Mapping[str, Any]) -> dict[str, Any]:
            label = build_label(
                self.session,
                frame=int(payload["frame"]),
                pixel_xy=_pixel(payload["pixel_xy"]),
                kind=str(payload["kind"]),
                depth_along_ray_m=(
                    None
                    if payload.get("depth_along_ray_m") is None
                    else float(payload["depth_along_ray_m"])
                ),
                human_confidence=str(payload.get("human_confidence") or "medium"),
                origin=str(payload.get("origin") or "fresh"),
                notes=str(payload.get("notes") or ""),
            )
            self.session.label_set.upsert(label)
            summary = self.session.save()  # autosave after every single label
            self.session.write_session(last_frame=label.frame)
            return {
                "saved": True,
                "label": label.to_json_dict(),
                "summary": summary,
                "label_path": str(self.session.label_path),
            }

        def _post_delete(self, payload: Mapping[str, Any]) -> dict[str, Any]:
            removed = self.session.label_set.remove(int(payload["frame"]))
            summary = self.session.save()
            return {"deleted": removed, "frame": int(payload["frame"]), "summary": summary}

        def _post_accept_interpolation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
            frames = [int(value) for value in payload.get("frames") or []]
            if not frames:
                raise ValueError("frames must be a non-empty list")
            return accept_interpolation(
                self.session,
                frames=frames,
                human_confidence=str(payload.get("human_confidence") or "low"),
            )

        def _post_session(self, payload: Mapping[str, Any]) -> dict[str, Any]:
            return self.session.write_session(
                last_frame=int(payload.get("last_frame") or 0),
                last_kind=str(payload.get("last_kind") or KIND_BOUNCE),
            )

        # -- plumbing -------------------------------------------------------

        def _serve_frame(self, name: str) -> None:
            try:
                index = int(Path(name).stem)
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "frame index must be an integer")
                return
            path = self.session.frames_dir / f"{index:06d}.jpg"
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "frame not extracted")
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> Mapping[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("request body is empty")
            if length > MAX_REQUEST_BYTES:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("payload must be a JSON object")
            return payload

        def _require_token(self) -> bool:
            expected = getattr(self.server, "save_token", "")
            provided = self.headers.get("X-Studio-Token", "")
            if expected and hmac.compare_digest(provided, expected):
                return True
            self._send_json(
                {"error": "missing or invalid X-Studio-Token; reload the page"},
                HTTPStatus.UNAUTHORIZED,
            )
            return False

        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, text: str) -> None:
            data = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return BallLabelStudioHandler


def _free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise StudioError(f"no free port in range {preferred}-{preferred + 49}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _pixel(value: Any) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"pixel_xy must be [x, y], got {value!r}")
    return (float(value[0]), float(value[1]))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StudioError(f"{path}: expected a JSON object")
    return payload


def _load_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _load_json(path)
    except (json.JSONDecodeError, StudioError, OSError):
        return None


def _resolve_video(root: Path) -> Path:
    for name in VIDEO_CANDIDATES:
        candidate = root / name
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
    for candidate in sorted(root.glob("*.mp4")) + sorted(root.glob("*.mkv")):
        if candidate.is_file():
            return candidate.resolve()
    raise StudioError(
        f"no readable source video in {root} (looked for {', '.join(VIDEO_CANDIDATES)}); "
        f"a broken symlink to a remote path counts as missing"
    )


def _image_size(calibration: Mapping[str, Any]) -> tuple[int, int]:
    size = calibration.get("image_size")
    if isinstance(size, Sequence) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    intrinsics = calibration.get("intrinsics") or {}
    return (int(float(intrinsics.get("cx", 960)) * 2), int(float(intrinsics.get("cy", 540)) * 2))


def _infer_clip_id(root: Path, frame_times: Mapping[str, Any] | None) -> str:
    if isinstance(frame_times, Mapping):
        clip_path = frame_times.get("clip_path")
        if isinstance(clip_path, str) and clip_path:
            return Path(clip_path).parent.name or root.name
    return root.name


def _probe_frame_count(video_path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def _artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    return {"path": relative, "sha256": _sha256(path)}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_session(session: StudioSession) -> dict[str, Any]:
    """Machine-readable session summary for report.json."""

    bundle = session.bundle
    return {
        "clip_id": bundle.clip_id,
        "run_dir": str(bundle.run_dir),
        "video": str(bundle.video_path),
        "frame_count": bundle.frame_count,
        "fps": bundle.fps,
        "image_size": list(bundle.image_size),
        "player_count": len(bundle.players),
        "skeleton_frame_count": sum(len(player["frames"]) for player in bundle.players),
        "detection_frame_count": len(bundle.detections),
        "candidate_frame_count": len(bundle.candidates),
        "prefill_frame_count": len(bundle.prefill),
        "bounce_candidate_frames": bundle.bounce_candidate_frames,
        "missing_artifacts": bundle.missing_artifacts,
        "calibration_plane_residuals": bundle.plane_residuals,
        "label_path": str(session.label_path),
        "labels": session.label_set.summary(),
    }


__all__ = [
    "CORE_BONE_PAIRS",
    "CORE_JOINTS",
    "CORE_JOINT_NAMES",
    "KEYBOARD_MAP",
    "BallLabelSet",
    "ClipBundle",
    "StudioError",
    "StudioSession",
    "accept_interpolation",
    "build_label",
    "build_page_state",
    "extract_frames",
    "load_clip_bundle",
    "nearest_player_reference",
    "open_session",
    "propose_interpolation",
    "run_studio_server",
    "solve_click",
    "summarize_session",
]
