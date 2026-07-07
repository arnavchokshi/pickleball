"""Assemble inspectable court_Z0 world-state artifacts for replay review."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import ValidationError

from .court_calibration import project_image_points_to_world
from .court_templates import get_court_template
from .eval_guard import assert_not_training_on_eval_clip
from .external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from .joint_schema import WHOLEBODY_133_JOINT_NAMES
from .racket6dof import SE3PoseConfidence, camera_paddle_pose_to_court_world, paddle_face_corners_object_cm
from .racket_true_corners import is_box_derived_source
from .schemas import (
    BallTrack,
    CourtCalibration,
    RacketPose,
    Skeleton3D,
    SmplMotion,
    Tracks,
    VirtualWorld,
    validate_artifact_file,
)
from .skeleton3d import SAM3D_BODY_MHR70_SEMANTIC_MAP
from .trust_band import build_trust_band, derive_court_trust_band


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_virtual_world"
WORLD_FRAME = "court_Z0"
PADDLE_HANDLE_LENGTH_M = 5.25 * 0.0254
PADDLE_HANDLE_WIDTH_M = 1.25 * 0.0254
PADDLE_THICKNESS_M = 0.55 * 0.0254
PADDLE_MESH_FACES = [
    [0, 1, 2],
    [0, 2, 3],
    [4, 6, 5],
    [4, 7, 6],
    [0, 4, 5],
    [0, 5, 1],
    [1, 5, 6],
    [1, 6, 2],
    [2, 6, 7],
    [2, 7, 3],
    [3, 7, 4],
    [3, 4, 0],
    [8, 9, 10],
    [8, 10, 11],
    [12, 14, 13],
    [12, 15, 14],
    [8, 12, 13],
    [8, 13, 9],
    [9, 13, 14],
    [9, 14, 10],
    [10, 14, 15],
    [10, 15, 11],
    [11, 15, 12],
    [11, 12, 8],
]
ROOT = Path(__file__).resolve().parents[2]
EVAL_CLIPS_ROOT = ROOT / "eval_clips" / "ball"
PHYSICS_TRUST_STATUSES = {"corrected", "interpolated", "physics_derived"}
PADDLE_PROXY_TRUST_STATUSES = {"estimated_from_wrist"}
BODY_MESH_REF_ARTIFACT = "body_mesh.json"
BallWorldPolicy = Literal["arc_required_for_midair", "court_plane_approx_for_review_only"]
BALL_WORLD_POLICIES: set[str] = {"arc_required_for_midair", "court_plane_approx_for_review_only"}
DEFAULT_BALL_WORLD_POLICY: BallWorldPolicy = "arc_required_for_midair"
KNOWN_EVAL_CLIP_HINTS = {
    "burlington": "burlington_gold_0300_low_steep_corner",
    "wolverine": "wolverine_mixed_0200_mid_steep_corner",
    "outdoor": "outdoor_webcam_iynbd_1500_long_high_baseline",
    "indoor": "indoor_doubles_fwuks_0500_long_mid_baseline",
}


def build_virtual_world_state(
    *,
    court_calibration: CourtCalibration | Mapping[str, Any],
    tracks: Tracks | Mapping[str, Any] | None = None,
    smpl_motion: SmplMotion | Mapping[str, Any] | None = None,
    skeleton3d: Skeleton3D | Mapping[str, Any] | None = None,
    ball_track: BallTrack | Mapping[str, Any] | None = None,
    racket_pose: RacketPose | Mapping[str, Any] | None = None,
    trust_bands: Mapping[str, Mapping[str, Any] | None] | None = None,
    physics_footlock: Mapping[str, Any] | None = None,
    ball_track_physics_filled: Mapping[str, Any] | None = None,
    ball_track_arc_solved: Mapping[str, Any] | None = None,
    racket_pose_estimate: Mapping[str, Any] | None = None,
    placement_calibration_path: str | Path | None = None,
    artifact_paths: Mapping[str, str | Path | None] | None = None,
    membership_path: str | Path | None = None,
    ball_world_policy: BallWorldPolicy | str = DEFAULT_BALL_WORLD_POLICY,
) -> dict[str, Any]:
    """Build one inspectable world artifact from already-produced stage outputs.

    ``trust_bands`` is an optional mapping of entity kind -> trust-band
    payload (see ``threed.racketsport.trust_band``), attached verbatim to
    the matching entities: ``"court"`` -> the court object, ``"body"`` ->
    players whose representation is ``mesh``/``joints``, ``"track"`` ->
    players whose representation is ``track_only``, ``"ball"`` -> the ball
    object (only when it has frames), ``"paddle"`` -> every paddle object.
    Omitted keys leave the corresponding entity's ``trust_band`` as
    ``null``; this function never computes a trust band itself.

    ``ball_track_arc_solved`` is the optional BALL-ARC-SOLVER output
    (``ball_track_arc_solved.json``). When supplied it is the sole authority
    for each ball frame's ``world_xyz``: frames the arc solver bounded
    between two confident events are overlaid with its analytic position,
    and frames it could not bound (``band == "hidden"``) are forced to
    ``world_xyz = null`` regardless of what ``ball_track_physics_filled``
    says. This is what keeps a raw/interpolated 2D sighting from ever being
    rendered as a 3D ball position between events -- see
    ``apply_ball_track_arc_solved_overlay``.
    """

    calibration = _court_calibration(court_calibration)
    normalized_ball_world_policy = _normalize_ball_world_policy(ball_world_policy)
    tracks_obj = _tracks(tracks)
    smpl_obj = _smpl_motion(smpl_motion)
    skeleton_obj = _skeleton3d(skeleton3d)
    ball_obj = _ball_track(ball_track)
    racket_obj = _racket_pose(racket_pose)
    footlock_obj = _raw_mapping(physics_footlock, artifact="physics_footlock")
    ball_physics_obj = apply_ball_track_arc_solved_overlay(
        _raw_mapping(ball_track_physics_filled, artifact="ball_track_physics_filled"),
        _raw_mapping(ball_track_arc_solved, artifact="ball_track_arc_solved"),
    )
    racket_estimate_obj = _raw_mapping(racket_pose_estimate, artifact="racket_pose_estimate")
    paths = {key: str(value) for key, value in (artifact_paths or {}).items() if value is not None}
    membership_preview = _load_player_membership_preview(
        membership_path,
        placement_calibration_path=placement_calibration_path,
        artifact_paths=paths,
    )

    fps = _world_fps(tracks_obj, smpl_obj, skeleton_obj, ball_physics_obj, ball_obj, racket_estimate_obj, racket_obj)
    players = _players(tracks_obj=tracks_obj, smpl_obj=smpl_obj, skeleton_obj=skeleton_obj)
    membership_summary = _apply_player_membership_preview(
        players,
        membership_preview=membership_preview,
        trust_bands=trust_bands,
    )
    joint_names = _world_joint_names_from_skeleton(skeleton_obj)
    _fill_no_data_player_frames(players)
    _apply_physics_footlock(players, footlock_obj, evidence_path=paths.get("physics_footlock"))
    ball = _ball(
        ball_physics_obj or ball_obj,
        calibration=calibration,
        ball_world_policy=normalized_ball_world_policy,
        physics_filled=ball_physics_obj is not None,
        evidence_path=paths.get("ball_track_physics_filled"),
    )
    paddles, paddle_warnings, paddle_metadata = _paddles(
        racket_estimate_obj or racket_obj,
        calibration=calibration,
        estimate=racket_estimate_obj is not None,
        evidence_path=paths.get("racket_pose_estimate"),
    )
    if membership_summary is not None:
        paddles = _filter_membership_excluded_paddles(paddles, membership_summary=membership_summary)
    mesh_index_present = _body_mesh_index_present(paths=paths, placement_calibration_path=placement_calibration_path)
    warnings = [
        *_warnings(players=players, ball=ball, paddles=paddles, mesh_index_present=mesh_index_present),
        *paddle_warnings,
        *_membership_preview_warnings(membership_summary),
    ]
    court = _court(calibration, placement_calibration_path=placement_calibration_path)
    _attach_trust_bands(court=court, players=players, ball=ball, paddles=paddles, trust_bands=trust_bands)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "world_frame": WORLD_FRAME,
        "fps": fps,
        "court": court,
        "players": players,
        "ball": ball,
        "paddles": paddles,
        "summary": _summary(
            players=players,
            ball=ball,
            paddles=paddles,
            warnings=warnings,
            fps=fps,
            paddle_metadata=paddle_metadata,
        ),
    }
    if joint_names:
        payload["joint_names"] = joint_names
    validation_payload = dict(payload)
    validation_payload["summary"] = _schema_summary(payload["summary"])
    validated = VirtualWorld.model_validate(validation_payload).model_dump(mode="json")
    validated["summary"]["temporal_coverage"] = payload["summary"]["temporal_coverage"]
    _drop_empty_optional_world_fields(validated)
    return validated


def build_virtual_world_state_from_files(
    *,
    court_calibration_path: str | Path,
    tracks_path: str | Path | None = None,
    smpl_motion_path: str | Path | None = None,
    skeleton3d_path: str | Path | None = None,
    ball_track_path: str | Path | None = None,
    racket_pose_path: str | Path | None = None,
    trust_bands: Mapping[str, Mapping[str, Any] | None] | None = None,
    physics_footlock_path: str | Path | None = None,
    ball_track_physics_filled_path: str | Path | None = None,
    ball_track_arc_solved_path: str | Path | None = None,
    racket_pose_estimate_path: str | Path | None = None,
    membership_path: str | Path | None = None,
    ball_world_policy: BallWorldPolicy | str = DEFAULT_BALL_WORLD_POLICY,
) -> dict[str, Any]:
    calibration = validate_artifact_file("court_calibration", Path(court_calibration_path))
    if not isinstance(calibration, CourtCalibration):
        raise ValueError("court calibration artifact did not parse as CourtCalibration")
    return build_virtual_world_state(
        court_calibration=calibration,
        tracks=_optional_artifact("tracks", tracks_path, Tracks),
        smpl_motion=_optional_artifact("smpl_motion", smpl_motion_path, SmplMotion),
        skeleton3d=_optional_skeleton3d_artifact(skeleton3d_path),
        ball_track=_optional_artifact("ball_track", ball_track_path, BallTrack),
        racket_pose=_optional_artifact("racket_pose", racket_pose_path, RacketPose),
        trust_bands=trust_bands,
        physics_footlock=_optional_json_mapping(physics_footlock_path),
        ball_track_physics_filled=_optional_json_mapping(ball_track_physics_filled_path),
        ball_track_arc_solved=_optional_json_mapping(ball_track_arc_solved_path),
        racket_pose_estimate=_optional_json_mapping(racket_pose_estimate_path),
        placement_calibration_path=court_calibration_path,
        membership_path=membership_path,
        ball_world_policy=ball_world_policy,
        artifact_paths={
            "tracks": tracks_path,
            "body_mesh_index": _existing_body_mesh_index_path(Path(court_calibration_path).parent),
            "physics_footlock": physics_footlock_path,
            "ball_track_physics_filled": ball_track_physics_filled_path,
            "ball_track_arc_solved": ball_track_arc_solved_path,
            "racket_pose_estimate": racket_pose_estimate_path,
        },
    )


def build_virtual_world_state_from_run_dir(
    run_dir: str | Path,
    *,
    court_calibration_path: str | Path | None = None,
    clip: str | None = None,
    allow_internal_val: bool = False,
    membership_path: str | Path | None = None,
    ball_world_policy: BallWorldPolicy | str = DEFAULT_BALL_WORLD_POLICY,
) -> dict[str, Any]:
    """Build a world by consuming the best available artifacts in a run directory.

    Physics-lane schemas are still landing in parallel. The interfaces consumed here
    are intentionally minimal and schema-confirmed by field names:
    `physics_footlock.json` has `players[].frames[]` keyed by `t` with
    `floor_world_xyz`, `foot_contact`, `contact_locked`, and `trust_band`;
    `ball_track_physics_filled.json` is ball-track-shaped with per-frame
    `trust_band`; `racket_pose_estimate.json` is racket-pose-shaped with per-frame
    `trust_band`. TODO-schema-confirm: replace this permissive adapter with strict
    Pydantic artifact schemas once PHYS-FOOT/PHYS-BALLFILL/PHYS-RACKET publish them.

    When `ball_track_arc_solved.json` (BALL-ARC-SOLVER output) is present
    alongside `ball_track_physics_filled.json`, it is consumed automatically
    and takes precedence for `world_xyz`: see
    `apply_ball_track_arc_solved_overlay`.
    """

    root = Path(run_dir)
    inferred_clip = clip or _infer_eval_clip(root)
    guard_inputs: list[Any] = [root]
    if inferred_clip:
        guard_inputs.append(inferred_clip)
    assert_not_training_on_eval_clip(guard_inputs, allow_internal_val=allow_internal_val)
    legacy_world_path = _legacy_pre_r3_world_path(root)
    if legacy_world_path is not None:
        return _load_legacy_pre_r3_world(legacy_world_path)
    court_path = resolve_best_court_calibration_path(root, explicit=court_calibration_path, clip=clip)
    trust_bands = _optional_json_mapping(_existing_file(root / "trust_bands.json"))
    calibration_payload = _optional_json_mapping(court_path)
    if calibration_payload is not None:
        refreshed = dict(trust_bands or {})
        refreshed["court"] = derive_court_trust_band(calibration_payload, evidence_path=str(court_path))
        trust_bands = refreshed
    return build_virtual_world_state_from_files(
        court_calibration_path=court_path,
        tracks_path=_existing_file(root / "tracks.json"),
        smpl_motion_path=_existing_file(root / "smpl_motion.json"),
        skeleton3d_path=_existing_file(root / "skeleton3d.json"),
        ball_track_path=_existing_file(root / "ball_track.json"),
        racket_pose_path=_existing_file(root / "racket_pose.json"),
        trust_bands=trust_bands,
        physics_footlock_path=_existing_file(root / "physics_footlock.json"),
        ball_track_physics_filled_path=_existing_file(root / "ball_track_physics_filled.json"),
        ball_track_arc_solved_path=_existing_file(root / "ball_track_arc_solved.json"),
        racket_pose_estimate_path=_existing_file(root / "racket_pose_estimate.json"),
        membership_path=membership_path,
        ball_world_policy=ball_world_policy,
    )


def resolve_best_court_calibration_path(
    run_dir: str | Path,
    *,
    explicit: str | Path | None = None,
    clip: str | None = None,
) -> Path:
    """Resolve the calibration artifact that should feed court-world placement.

    Precedence is explicit path, metric-15pt run artifact, metric-15pt eval-label
    artifact for the inferred clip, then the run's existing `court_calibration.json`.
    """

    root = Path(run_dir)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        candidates.extend(
            [
                root / "court_calibration_metric15pt.json",
                root / "labels" / "court_calibration_metric15pt.json",
            ]
        )
        inferred = clip or _infer_eval_clip(root)
        if inferred:
            candidates.append(EVAL_CLIPS_ROOT / inferred / "labels" / "court_calibration_metric15pt.json")
        candidates.append(root / "court_calibration.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"no court calibration artifact found; checked: {rendered}")


def write_virtual_world(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_ball_world_policy(policy: BallWorldPolicy | str) -> BallWorldPolicy:
    if policy in BALL_WORLD_POLICIES:
        return policy  # type: ignore[return-value]
    rendered = ", ".join(sorted(BALL_WORLD_POLICIES))
    raise ValueError(f"invalid ball_world_policy {policy!r}; expected one of: {rendered}")


def apply_ball_track_arc_solved_overlay(
    ball_track_physics_filled: Mapping[str, Any] | None,
    ball_track_arc_solved: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Make the BALL-ARC-SOLVER output authoritative for ``world_xyz``.

    The owner's law: between two confident ball positions there is exactly
    one parabola, and direction changes only happen at selected events. The
    arc solver (`threed.racketsport.ball_arc_solver`) already enforces this
    per-frame in `ball_track_arc_solved.json`: every frame it can bound
    between two confident events gets an analytic (parabola-evaluated)
    `world_xyz`, and every frame outside that coverage is honestly marked
    `band == "hidden"` with `world_xyz = null`.

    `ball_track_physics_filled.json` predates the arc solver and, for run
    directories where only part of the clip was re-composed by hand, can
    still carry raw/interpolated 2D-lifted `world_xyz` values for frames the
    arc solver marked hidden. Rendering those values in the 3D world is
    exactly how a non-analytic sighting sneaks into the ball trail and shows
    up as a mid-air direction change. This overlay closes that gap: it is
    the single place `world_xyz` is decided once an arc-solved artifact is
    available, so the rendered trail is always a sampled evaluation of the
    solver's segments, never a mix of arc and raw positions.

    Only `world_xyz` is touched. `xy`/`conf`/`visible`/`trust_band` stay
    exactly as `ball_track_physics_filled` authored them -- those describe
    the 2D detection and its own render-continuity styling, which this lane
    does not own.
    """

    if ball_track_physics_filled is None:
        return None
    if not isinstance(ball_track_arc_solved, Mapping):
        return ball_track_physics_filled
    if str(ball_track_arc_solved.get("status", "ran")) != "ran":
        return ball_track_physics_filled
    arc_frames = ball_track_arc_solved.get("frames")
    if not isinstance(arc_frames, list):
        return ball_track_physics_filled

    arc_by_key: dict[str, Mapping[str, Any]] = {}
    for arc_frame in arc_frames:
        if not isinstance(arc_frame, Mapping):
            continue
        t = _optional_float(arc_frame.get("t"))
        if t is None:
            continue
        arc_by_key[_time_key(t)] = arc_frame

    merged_frames: list[Any] = []
    overlaid_count = 0
    forced_hidden_count = 0
    for frame in _sequence(ball_track_physics_filled.get("frames")):
        if not isinstance(frame, Mapping):
            merged_frames.append(frame)
            continue
        t = _optional_float(frame.get("t"))
        arc_frame = arc_by_key.get(_time_key(t)) if t is not None else None
        if arc_frame is None:
            # No arc-solver opinion for this timestamp (e.g. the arc-solved
            # artifact does not cover this clip at all) -- leave it as-authored.
            merged_frames.append(frame)
            continue
        merged = dict(frame)
        arc_world_xyz = arc_frame.get("world_xyz")
        arc_band = str(arc_frame.get("band") or "")
        if arc_band == "hidden" or arc_world_xyz is None:
            # No confident second endpoint bounds this frame -- honesty over
            # continuity: no world position beats a fabricated one.
            merged["world_xyz"] = None
            forced_hidden_count += 1
        else:
            merged["world_xyz"] = [float(component) for component in arc_world_xyz]
            overlaid_count += 1
        merged_frames.append(merged)

    result = dict(ball_track_physics_filled)
    result["frames"] = merged_frames
    result["arc_solved_overlay"] = {
        "applied": True,
        "source_artifact_type": str(
            ball_track_arc_solved.get("artifact_type") or "racketsport_ball_track_arc_solved"
        ),
        "overlaid_frame_count": overlaid_count,
        "forced_hidden_frame_count": forced_hidden_count,
    }
    return result


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _optional_artifact(artifact: str, path: str | Path | None, model: type[Any]) -> Any | None:
    if path is None:
        return None
    parsed = validate_artifact_file(artifact, Path(path))
    if not isinstance(parsed, model):
        raise ValueError(f"{artifact} artifact did not parse as {model.__name__}")
    return parsed


def _optional_skeleton3d_artifact(path: str | Path | None) -> Skeleton3D | None:
    if path is None:
        return None
    payload = _optional_json_mapping(path)
    if payload is None:
        return None
    return _skeleton3d(_strip_legacy_skeleton3d_extras(payload))


def _legacy_pre_r3_world_path(run_dir: Path) -> Path | None:
    skeleton = _optional_json_mapping(_existing_file(run_dir / "skeleton3d.json"))
    if not isinstance(skeleton, Mapping) or "foot_pin" not in skeleton:
        return None
    smpl = _optional_json_mapping(_existing_file(run_dir / "smpl_motion.json"))
    if _payload_has_r3_grounding_provenance(skeleton) or _payload_has_r3_grounding_provenance(smpl):
        return None
    return _existing_file(run_dir / "confidence_gated_world.json") or _existing_file(run_dir / "virtual_world.json")


def _load_legacy_pre_r3_world(path: Path) -> dict[str, Any]:
    parsed = validate_artifact_file("virtual_world", path)
    if not isinstance(parsed, VirtualWorld):
        raise ValueError(f"{path} did not parse as VirtualWorld")
    payload = parsed.model_dump(mode="json")
    repaired_transl = _repair_legacy_world_transl_world(payload)
    summary = dict(payload.get("summary") or {})
    warnings = list(summary.get("warnings") or [])
    warnings.append(
        "legacy_pre_r3_world_reused: run-dir BODY artifacts lack placement_track_world_xy grounding provenance; "
        f"reused {path.name} instead of reconstructing stale skeleton joints. Fresh A100 R3 BODY is the acceptance gate."
    )
    if repaired_transl:
        warnings.append(f"legacy_pre_r3_transl_world_repaired: reset {repaired_transl} stale transl_world anchors to track_world_xy")
    summary["warnings"] = warnings
    payload["summary"] = summary
    _drop_empty_optional_world_fields(payload)
    return payload


def _repair_legacy_world_transl_world(payload: dict[str, Any]) -> int:
    repaired = 0
    for player in payload.get("players", []) or []:
        if not isinstance(player, dict):
            continue
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, dict):
                continue
            transl = _vector3(frame.get("transl_world"))
            track_xy = _vector2(frame.get("track_world_xy"))
            if transl is None or track_xy is None:
                continue
            if _distance2(transl[:2], track_xy) <= 0.02:
                continue
            frame["transl_world"] = [track_xy[0], track_xy[1], 0.0]
            repaired += 1
    return repaired


def _payload_has_r3_grounding_provenance(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("grounding_anchor_source") == "placement_track_world_xy":
        return True
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("grounding_anchor_source") == "placement_track_world_xy":
        return True
    grounding_metrics = payload.get("grounding_metrics")
    if isinstance(grounding_metrics, Mapping) and grounding_metrics.get("grounding_anchor_source") == "placement_track_world_xy":
        return True
    for player in payload.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping):
                continue
            confidence = frame.get("confidence_provenance")
            if isinstance(confidence, Mapping) and confidence.get("grounding_anchor_source") == "placement_track_world_xy":
                return True
    return False


def _optional_json_mapping(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _existing_file(path: Path | None) -> Path | None:
    return path if path is not None and path.is_file() else None


def _load_player_membership_preview(
    membership_path: str | Path | None,
    *,
    placement_calibration_path: str | Path | None,
    artifact_paths: Mapping[str, str | Path | None],
) -> dict[str, Any] | None:
    resolved = _resolve_player_membership_path(
        membership_path,
        placement_calibration_path=placement_calibration_path,
        artifact_paths=artifact_paths,
    )
    if resolved is None:
        return None
    payload = _optional_json_mapping(resolved)
    if payload is None:
        return None
    if payload.get("artifact_type") != "racketsport_player_court_membership":
        raise ValueError(f"{resolved} is not a player-court membership artifact")
    if payload.get("verified") is not False or payload.get("not_gate_verified") is not True:
        raise ValueError(f"{resolved} must remain verified=false and not_gate_verified=true")
    per_player = payload.get("per_player")
    if not isinstance(per_player, Mapping):
        raise ValueError(f"{resolved} must contain per_player membership verdicts")
    return {
        "path": str(resolved),
        "per_player": per_player,
    }


def _resolve_player_membership_path(
    membership_path: str | Path | None,
    *,
    placement_calibration_path: str | Path | None,
    artifact_paths: Mapping[str, str | Path | None],
) -> Path | None:
    if membership_path is not None:
        explicit = Path(membership_path)
        if not explicit.is_file():
            raise FileNotFoundError(f"membership artifact not found: {explicit}")
        return explicit

    candidates: list[Path] = []
    candidate_dirs: list[Path] = []
    for raw in [placement_calibration_path, *artifact_paths.values()]:
        if raw is None:
            continue
        path = Path(raw)
        parent = path.parent if path.suffix else path
        if parent not in candidate_dirs:
            candidate_dirs.append(parent)
    for directory in candidate_dirs:
        candidates.append(directory / "membership.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _apply_player_membership_preview(
    players: list[dict[str, Any]],
    *,
    membership_preview: Mapping[str, Any] | None,
    trust_bands: Mapping[str, Mapping[str, Any] | None] | None,
) -> dict[str, Any] | None:
    if membership_preview is None:
        return None
    per_player = membership_preview.get("per_player")
    if not isinstance(per_player, Mapping):
        return None

    excluded: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    for raw_player_id, raw_membership in sorted(per_player.items(), key=lambda item: int(item[0])):
        if not isinstance(raw_membership, Mapping):
            continue
        player_id = _maybe_int(raw_player_id)
        if player_id is None:
            continue
        verdict = str(raw_membership.get("verdict") or "uncertain")
        record = {
            "id": player_id,
            "verdict": verdict,
            "reasons": [str(reason) for reason in _sequence(raw_membership.get("reasons"))],
        }
        if verdict == "adjacent_or_spectator":
            excluded.append(record)
        elif verdict == "uncertain":
            uncertain.append(record)

    excluded_ids = {int(record["id"]) for record in excluded}
    if excluded_ids:
        players[:] = [player for player in players if int(player["id"]) not in excluded_ids]

    summary = {
        "path": str(membership_preview.get("path") or ""),
        "excluded_players": excluded,
        "uncertain_players": uncertain,
    }
    if isinstance(trust_bands, dict):
        trust_bands["player_membership"] = {
            "stage": "TRK",
            "gate_id": "player_court_membership_preview",
            "gate_status": "membership_preview_not_verified",
            "badge": "low_confidence",
            "reason": (
                "Preview target-court membership consumed by virtual_world; adjacent_or_spectator players are excluded, "
                "uncertain players remain rendered."
            ),
            "evidence_path": summary["path"] or None,
            "excluded_players": excluded,
            "uncertain_players": uncertain,
        }
    return summary


def _filter_membership_excluded_paddles(
    paddles: list[dict[str, Any]],
    *,
    membership_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    excluded_ids = {
        int(record["id"])
        for record in _sequence(membership_summary.get("excluded_players"))
        if isinstance(record, Mapping) and _maybe_int(record.get("id")) is not None
    }
    if not excluded_ids:
        return paddles
    return [paddle for paddle in paddles if int(paddle["player_id"]) not in excluded_ids]


def _membership_preview_warnings(membership_summary: Mapping[str, Any] | None) -> list[str]:
    if membership_summary is None:
        return []
    excluded = [
        int(record["id"])
        for record in _sequence(membership_summary.get("excluded_players"))
        if isinstance(record, Mapping) and _maybe_int(record.get("id")) is not None
    ]
    uncertain = [
        int(record["id"])
        for record in _sequence(membership_summary.get("uncertain_players"))
        if isinstance(record, Mapping) and _maybe_int(record.get("id")) is not None
    ]
    warnings = [
        "player_membership_preview_not_verified",
        f"player_membership_excluded_count_{len(excluded)}",
    ]
    if excluded:
        warnings.append("player_membership_excluded_ids_" + "_".join(str(player_id) for player_id in sorted(excluded)))
    if uncertain:
        warnings.append("player_membership_uncertain_ids_" + "_".join(str(player_id) for player_id in sorted(uncertain)))
    return warnings


def _infer_eval_clip(run_dir: Path) -> str | None:
    lowered_parts = [part.lower() for part in run_dir.parts]
    for slug in KNOWN_EVAL_CLIP_HINTS.values():
        if slug in run_dir.parts:
            return slug
    joined = "/".join(lowered_parts)
    for hint, slug in KNOWN_EVAL_CLIP_HINTS.items():
        if hint in joined:
            return slug
    return None


def _court(calibration: CourtCalibration, *, placement_calibration_path: str | Path | None = None) -> dict[str, Any]:
    template = get_court_template(calibration.sport)
    line_segments = {
        line_id: [list(start), list(end)]
        for line_id, (start, end) in template.line_segments_m.items()
    }
    net_start, net_end = template.line_segments_m["net"]
    return {
        "sport": calibration.sport,
        "coordinate_frame": template.coordinate_frame,
        "length_m": template.length_m,
        "width_m": template.width_m,
        "line_segments": line_segments,
        "net": {
            "endpoints": [list(net_start), list(net_end)],
            "center_height_m": template.center_net_height_m,
            "post_height_m": template.post_net_height_m,
        },
        "placement_calibration": _placement_calibration(calibration, path=placement_calibration_path),
    }


def _placement_calibration(calibration: CourtCalibration, *, path: str | Path | None) -> dict[str, Any]:
    return {
        "source": calibration.source,
        "intrinsics_source": calibration.intrinsics.source,
        "capture_quality_grade": calibration.capture_quality.grade,
        "metric_confidence": calibration.metric_confidence,
        "evidence_path": str(path) if path is not None else None,
    }


def _players(
    *,
    tracks_obj: Tracks | None,
    smpl_obj: SmplMotion | None,
    skeleton_obj: Skeleton3D | None,
) -> list[dict[str, Any]]:
    track_meta, track_frames = _track_lookup(tracks_obj)
    if smpl_obj is not None:
        smpl_players = {player.id: player for player in smpl_obj.players}
        skeleton_players = {player.id: player for player in skeleton_obj.players} if skeleton_obj is not None else {}
        player_ids = sorted(set(track_meta) | set(smpl_players) | set(skeleton_players))
        players = []
        for player_id in player_ids:
            player = smpl_players.get(player_id)
            physics = player.physics if player is not None else None
            smpl_frames = {_time_key(frame.t): frame for frame in player.frames} if player is not None else {}
            skeleton_player = skeleton_players.get(player_id)
            skeleton_frames = {_time_key(frame.t): frame for frame in skeleton_player.frames} if skeleton_player is not None else {}
            frames, joints_source = _combined_smpl_skeleton_player_frames(
                player_id=player_id,
                track_frames=track_frames,
                smpl_frames_by_key=smpl_frames,
                skeleton_frames_by_key=skeleton_frames,
                smpl_frame_builder=lambda track_frame, smpl_frame, physics=physics, fps=smpl_obj.fps, player_id=player_id: _player_frame_from_sources(
                    track_frame=track_frame,
                    smpl_frame=smpl_frame,
                    physics=physics,
                    player_id=player_id,
                    fps=fps,
                ),
            )
            players.append(
                _player_record(
                    player_id=player_id,
                    track_meta=track_meta,
                    frames=frames,
                    joints_source=joints_source,
                )
            )
        return players

    if skeleton_obj is not None:
        skeleton_players = {player.id: player for player in skeleton_obj.players}
        player_ids = sorted(set(track_meta) | set(skeleton_players))
        players = []
        for player_id in player_ids:
            player = skeleton_players.get(player_id)
            skeleton_frames = {_time_key(frame.t): frame for frame in player.frames} if player is not None else {}
            frames = _combined_player_frames(
                player_id=player_id,
                track_frames=track_frames,
                world_frames_by_key=skeleton_frames,
                frame_builder=_skeleton_player_frame_from_sources,
            )
            players.append(
                _player_record(
                    player_id=player_id,
                    track_meta=track_meta,
                    frames=frames,
                    joints_source={
                        "skeleton3d": sum(1 for frame in frames if int(frame["joint_count"]) > 0),
                        "smpl_fill": 0,
                    },
                )
            )
        return players

    if tracks_obj is None:
        return []
    return [
        {
            "id": player.id,
            "side": player.side,
            "role": player.role,
            "representation": "track_only",
            "frames": [
                {
                    "t": frame.t,
                    "track_world_xy": _track_world_xy(frame),
                    "track_conf": float(frame.conf),
                    "bbox": tuple(float(value) for value in frame.bbox),
                    "transl_world": None,
                    "joints_world": [],
                    "joint_conf": [],
                    "mesh_vertices_world": [],
                    "joint_count": 0,
                    "mesh_vertex_count": 0,
                    **_track_floor_fields(frame),
                }
                for frame in player.frames
            ],
        }
        for player in tracks_obj.players
    ]


def _world_joint_names_from_skeleton(skeleton_obj: Skeleton3D | None) -> list[str] | None:
    if skeleton_obj is None:
        return None
    joint_names = [str(name) for name in skeleton_obj.joint_names]
    joint_count = _skeleton_source_joint_count(skeleton_obj) or len(joint_names)
    if joint_count == len(MHR70_JOINT_NAMES) and _is_sam3d_skeleton_source(skeleton_obj, joint_names):
        _assert_mhr70_names_match_semantic_map()
        return list(MHR70_JOINT_NAMES)
    if joint_count == len(WHOLEBODY_133_JOINT_NAMES):
        return list(WHOLEBODY_133_JOINT_NAMES)
    return joint_names or None


def _skeleton_source_joint_count(skeleton_obj: Skeleton3D) -> int | None:
    for player in skeleton_obj.players:
        for frame in player.frames:
            return len(frame.joints_world)
    return None


def _is_sam3d_skeleton_source(skeleton_obj: Skeleton3D, joint_names: list[str]) -> bool:
    source_model = str(skeleton_obj.source_model or "").lower()
    if "sam3d" in source_model or "sam_3d" in source_model:
        return True
    return joint_names == [f"sam3dbody_joint_{index:03d}" for index in range(len(MHR70_JOINT_NAMES))]


def _assert_mhr70_names_match_semantic_map() -> None:
    for semantic_name, source_index in SAM3D_BODY_MHR70_SEMANTIC_MAP.joints.items():
        if MHR70_JOINT_NAMES[source_index] != semantic_name:
            raise ValueError(
                "MHR70_JOINT_NAMES does not match SAM3D_BODY_MHR70_SEMANTIC_MAP "
                f"at index {source_index}: expected {semantic_name!r}, got {MHR70_JOINT_NAMES[source_index]!r}"
            )


def _apply_physics_footlock(
    players: list[dict[str, Any]],
    footlock: Mapping[str, Any] | None,
    *,
    evidence_path: str | None,
) -> None:
    if footlock is None:
        return
    lookup: dict[tuple[int, str], Mapping[str, Any]] = {}
    for player in _sequence(footlock.get("players")):
        player_id = _maybe_int(_get(player, "id"))
        if player_id is None:
            continue
        for frame in _sequence(_get(player, "frames")):
            try:
                lookup[(player_id, _time_key(float(_get(frame, "t"))))] = frame
            except (TypeError, ValueError):
                continue

    for player in players:
        player_id = int(player["id"])
        for frame in player["frames"]:
            foot_frame = lookup.get((player_id, _time_key(float(frame["t"]))))
            if foot_frame is None:
                continue
            corrected_joints = _vectors3(_get(foot_frame, "joints_world"))
            if corrected_joints is not None:
                frame["joints_world"] = corrected_joints
                frame["joint_count"] = len(corrected_joints)
            corrected_conf = _float_sequence(_get(foot_frame, "joint_conf"))
            if corrected_conf is not None:
                frame["joint_conf"] = corrected_conf
            status = _trust_status(_get(foot_frame, "trust_band")) or "corrected"
            floor = _vector3(_first_present(foot_frame, "floor_world_xyz", "locked_floor_world_xyz", "root_world_xyz"))
            if floor is not None:
                frame["floor_world_xyz"] = [floor[0], floor[1], 0.0]
                frame["floor_source"] = f"physics_footlock_{status}"
                frame["floor_offset_m"] = 0.0
                frame["floor_penetration_m"] = 0.0
            contact = _foot_contact(_get(foot_frame, "foot_contact")) or _foot_lock_contact(_get(foot_frame, "foot_lock"))
            if contact is not None:
                frame["foot_contact"] = contact
            locked = _get(foot_frame, "contact_locked")
            frame["contact_locked"] = bool(locked) if locked is not None else bool(contact and (contact["left"] or contact["right"]))
            frame["physics"] = str(_get(foot_frame, "physics") or "physics_footlock")
            grf = _get(foot_frame, "grf")
            if grf is not None:
                frame["grf"] = [list(vector) for vector in grf]
            frame["trust_band"] = _physics_trust_band(
                stage="PHYS-FOOT",
                gate_id="physics_footlock",
                status=status,
                reason=f"Foot placement consumed from physics_footlock.json as {status}; locked stance samples are not smoothed by virtual_world.",
                evidence_path=evidence_path,
            )


def _combined_player_frames(
    *,
    player_id: int,
    track_frames: Mapping[tuple[int, str], Any],
    world_frames_by_key: Mapping[str, Any],
    frame_builder: Callable[[Any | None, Any | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union track-frame and world-frame (SMPL/skeleton) timestamps for one player."""

    frame_keys = sorted(
        {key for key_player_id, key in track_frames if key_player_id == player_id} | set(world_frames_by_key)
    )
    frames = []
    for frame_key in frame_keys:
        track_frame = track_frames.get((player_id, frame_key))
        world_frame = world_frames_by_key.get(frame_key)
        frames.append(frame_builder(track_frame, world_frame))
    return frames


def _combined_smpl_skeleton_player_frames(
    *,
    player_id: int,
    track_frames: Mapping[tuple[int, str], Any],
    smpl_frames_by_key: Mapping[str, Any],
    skeleton_frames_by_key: Mapping[str, Any],
    smpl_frame_builder: Callable[[Any | None, Any | None], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Union track, sparse mesh, and quality-gated skeleton timestamps for one player."""

    frame_keys = sorted(
        {key for key_player_id, key in track_frames if key_player_id == player_id}
        | set(smpl_frames_by_key)
        | set(skeleton_frames_by_key),
        key=float,
    )
    frames = []
    joints_source = {"skeleton3d": 0, "smpl_fill": 0}
    for frame_key in frame_keys:
        track_frame = track_frames.get((player_id, frame_key))
        smpl_frame = smpl_frames_by_key.get(frame_key)
        skeleton_frame = skeleton_frames_by_key.get(frame_key)
        if skeleton_frame is not None:
            if smpl_frame is not None:
                frame = smpl_frame_builder(track_frame, smpl_frame)
                _replace_frame_joints_from_skeleton(frame, skeleton_frame)
            else:
                frame = _skeleton_player_frame_from_sources(track_frame, skeleton_frame)
            if int(frame["joint_count"]) > 0:
                joints_source["skeleton3d"] += 1
            frames.append(frame)
            continue
        if smpl_frame is not None:
            frame = smpl_frame_builder(track_frame, smpl_frame)
            if int(frame["joint_count"]) > 0:
                joints_source["smpl_fill"] += 1
            frames.append(frame)
            continue
        frames.append(_skeleton_player_frame_from_sources(track_frame, None))
    return frames, joints_source


def _fill_no_data_player_frames(players: list[dict[str, Any]]) -> None:
    if len(players) <= 1:
        return
    time_by_key: dict[str, float] = {}
    for player in players:
        for frame in player["frames"]:
            key = _time_key(float(frame["t"]))
            time_by_key.setdefault(key, float(frame["t"]))
    if not time_by_key:
        return
    ordered_keys = sorted(time_by_key, key=lambda key: time_by_key[key])
    for player in players:
        existing = {_time_key(float(frame["t"])): frame for frame in player["frames"]}
        if set(existing) == set(ordered_keys):
            continue
        player["frames"] = [
            existing.get(frame_key) or _no_data_player_frame(time_by_key[frame_key])
            for frame_key in ordered_keys
        ]


def _no_data_player_frame(t: float) -> dict[str, Any]:
    return {
        "t": float(t),
        "track_world_xy": None,
        "track_conf": None,
        "bbox": None,
        "transl_world": None,
        "joints_world": [],
        "joint_conf": [],
        "mesh_vertices_world": [],
        "mesh_ref": None,
        "joint_count": 0,
        "mesh_vertex_count": 0,
        **_empty_floor_fields(),
        "trust_band": {
            "stage": "WORLD",
            "gate_id": "world_no_data_placeholder",
            "gate_status": "no_data",
            "badge": "low_confidence",
            "reason": "Timestamp is present for another player in the world, but this player has no track, skeleton, or mesh sample at this time.",
            "evidence_path": None,
        },
    }


def _player_record(
    *,
    player_id: int,
    track_meta: Mapping[int, Mapping[str, str]],
    frames: list[dict[str, Any]],
    joints_source: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    meta = track_meta.get(player_id, {})
    record = {
        "id": player_id,
        "side": meta.get("side"),
        "role": meta.get("role"),
        "representation": _player_representation(frames),
        "frames": frames,
    }
    if joints_source is not None:
        record["joints_source"] = {
            "skeleton3d": int(joints_source.get("skeleton3d", 0)),
            "smpl_fill": int(joints_source.get("smpl_fill", 0)),
        }
    return record


def _replace_frame_joints_from_skeleton(frame: dict[str, Any], skeleton_frame: Any) -> None:
    transl_world, joint_delta = _skeleton_alignment(
        skeleton_frame,
        track_world_xy=frame.get("track_world_xy"),
    )
    joints = _translated_vectors(skeleton_frame.joints_world, joint_delta)
    frame["joints_world"] = joints
    frame["joint_conf"] = [float(conf) for conf in skeleton_frame.joint_conf]
    frame["joint_count"] = len(joints)
    frame["transl_world"] = transl_world


def _skeleton_player_frame_from_sources(track_frame: Any | None, skeleton_frame: Any | None) -> dict[str, Any]:
    if skeleton_frame is None:
        if track_frame is None:
            raise ValueError("player frame requires track_frame or skeleton_frame")
        return {
            "t": track_frame.t,
            "track_world_xy": _track_world_xy(track_frame),
            "track_conf": float(track_frame.conf),
            "bbox": tuple(float(value) for value in track_frame.bbox),
            "transl_world": None,
            "joints_world": [],
            "joint_conf": [],
            "mesh_vertices_world": [],
            "joint_count": 0,
            "mesh_vertex_count": 0,
            **_track_floor_fields(track_frame),
        }
    track_world_xy = _track_world_xy(track_frame) if track_frame is not None else None
    transl_world, joint_delta = _skeleton_alignment(skeleton_frame, track_world_xy=track_world_xy)
    return {
        "t": skeleton_frame.t,
        "track_world_xy": track_world_xy,
        "track_conf": float(track_frame.conf) if track_frame is not None else None,
        "bbox": tuple(float(value) for value in track_frame.bbox) if track_frame is not None else None,
        "transl_world": transl_world,
        "joints_world": _translated_vectors(skeleton_frame.joints_world, joint_delta),
        "joint_conf": [float(conf) for conf in skeleton_frame.joint_conf],
        "mesh_vertices_world": [],
        "joint_count": len(skeleton_frame.joints_world),
        "mesh_vertex_count": 0,
        **_skeleton_floor_fields(track_frame=track_frame),
    }


def _skeleton_transl_world(skeleton_frame: Any, *, track_world_xy: Any) -> list[float] | None:
    transl, _joint_delta = _skeleton_alignment(skeleton_frame, track_world_xy=track_world_xy)
    return transl


def _skeleton_alignment(skeleton_frame: Any, *, track_world_xy: Any) -> tuple[list[float] | None, list[float]]:
    transl = _vector3(getattr(skeleton_frame, "transl_world", None))
    track_xy = _vector2(track_world_xy)
    if transl is not None:
        return [transl[0], transl[1], transl[2]], [0.0, 0.0, 0.0]
    if track_xy is not None:
        target = [track_xy[0], track_xy[1], 0.0]
        return target, [0.0, 0.0, 0.0]
    return None, [0.0, 0.0, 0.0]


def _translated_vectors(vectors: Sequence[Sequence[float]], delta: Sequence[float]) -> list[list[float]]:
    return [
        [float(vector[0]) + float(delta[0]), float(vector[1]) + float(delta[1]), float(vector[2]) + float(delta[2])]
        for vector in vectors
    ]


def _skeleton_floor_fields(*, track_frame: Any | None) -> dict[str, Any]:
    if track_frame is None:
        return _empty_floor_fields()
    return {
        **_track_floor_fields(track_frame),
        "floor_source": "track_footpoint+skeleton_world",
    }


def _player_frame_from_sources(
    *,
    track_frame: Any | None,
    smpl_frame: Any | None,
    physics: str | None = None,
    player_id: int | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    if smpl_frame is None:
        if track_frame is None:
            raise ValueError("player frame requires track_frame or smpl_frame")
        return {
            "t": track_frame.t,
            "track_world_xy": _track_world_xy(track_frame),
            "track_conf": float(track_frame.conf),
            "bbox": tuple(float(value) for value in track_frame.bbox),
            "transl_world": None,
            "joints_world": [],
            "joint_conf": [],
            "mesh_vertices_world": [],
            "joint_count": 0,
            "mesh_vertex_count": 0,
            **_track_floor_fields(track_frame),
        }
    mesh_vertices = [list(vertex) for vertex in smpl_frame.mesh_vertices_world]
    mesh_vertex_count = len(mesh_vertices)
    frame_idx = _mesh_frame_index(smpl_frame, fps=fps)
    return {
        "t": smpl_frame.t,
        "track_world_xy": _track_world_xy(track_frame) if track_frame is not None else None,
        "track_conf": float(track_frame.conf) if track_frame is not None else None,
        "bbox": tuple(float(value) for value in track_frame.bbox) if track_frame is not None else None,
        "transl_world": list(smpl_frame.transl_world),
        "joints_world": [list(joint) for joint in smpl_frame.joints_world],
        "joint_conf": [float(conf) for conf in smpl_frame.joint_conf],
        "mesh_vertices_world": [],
        "mesh_ref": _mesh_ref(player_id=player_id, frame_idx=frame_idx, t=float(smpl_frame.t), vertex_count=mesh_vertex_count),
        "joint_count": len(smpl_frame.joints_world),
        "mesh_vertex_count": mesh_vertex_count,
        **_smpl_floor_fields(track_frame=track_frame, smpl_frame=smpl_frame, mesh_vertices=mesh_vertices, physics=physics),
    }


def _mesh_frame_index(smpl_frame: Any, *, fps: float | None) -> int:
    frame_idx = getattr(smpl_frame, "frame_idx", None)
    if frame_idx is not None:
        return int(frame_idx)
    return int(round(float(smpl_frame.t) * float(fps or 30.0)))


def _mesh_ref(*, player_id: int | None, frame_idx: int, t: float, vertex_count: int) -> dict[str, Any] | None:
    if player_id is None or vertex_count <= 0:
        return None
    return {
        "artifact": BODY_MESH_REF_ARTIFACT,
        "player_id": int(player_id),
        "frame_idx": int(frame_idx),
        "t": float(t),
    }


def _track_floor_fields(track_frame: Any) -> dict[str, Any]:
    track_xy = _track_world_xy(track_frame)
    return {
        "floor_world_xyz": [track_xy[0], track_xy[1], 0.0],
        "floor_source": "track_footpoint",
        "floor_offset_m": 0.0,
        "min_mesh_z_m": None,
        "floor_penetration_m": 0.0,
        "foot_contact": None,
        "contact_locked": False,
        "physics": None,
        "grf": None,
    }


def _smpl_floor_fields(
    *,
    track_frame: Any | None,
    smpl_frame: Any,
    mesh_vertices: list[list[float]],
    physics: str | None,
) -> dict[str, Any]:
    floor_xy: list[float] | None = None
    source = "smpl_world"
    if track_frame is not None:
        floor_xy = _track_world_xy(track_frame)
        source = "track_footpoint+smpl_world"
    elif smpl_frame.transl_world is not None:
        floor_xy = [float(smpl_frame.transl_world[0]), float(smpl_frame.transl_world[1])]

    min_mesh_z = min((float(vertex[2]) for vertex in mesh_vertices), default=None)
    foot_contact = {"left": bool(smpl_frame.foot_contact.left), "right": bool(smpl_frame.foot_contact.right)}
    return {
        "floor_world_xyz": [floor_xy[0], floor_xy[1], 0.0] if floor_xy is not None else None,
        "floor_source": source if floor_xy is not None else None,
        "floor_offset_m": float(smpl_frame.transl_world[2]) if smpl_frame.transl_world is not None else None,
        "min_mesh_z_m": min_mesh_z,
        "floor_penetration_m": max(0.0, -min_mesh_z) if min_mesh_z is not None else 0.0,
        "foot_contact": foot_contact,
        "contact_locked": bool(foot_contact["left"] or foot_contact["right"]),
        "physics": physics,
        "grf": [list(vector) for vector in smpl_frame.grf] if smpl_frame.grf is not None else None,
    }


def _empty_floor_fields() -> dict[str, Any]:
    return {
        "floor_world_xyz": None,
        "floor_source": None,
        "floor_offset_m": None,
        "min_mesh_z_m": None,
        "floor_penetration_m": 0.0,
        "foot_contact": None,
        "contact_locked": False,
        "physics": None,
        "grf": None,
    }


def _player_representation(frames: list[Mapping[str, Any]]) -> str:
    if any(int(frame["mesh_vertex_count"]) > 0 for frame in frames):
        return "mesh"
    if any(int(frame["joint_count"]) > 0 for frame in frames):
        return "joints"
    return "track_only"


def _track_lookup(tracks_obj: Tracks | None) -> tuple[dict[int, dict[str, str]], dict[tuple[int, str], Any]]:
    if tracks_obj is None:
        return {}, {}
    meta = {player.id: {"side": player.side, "role": player.role} for player in tracks_obj.players}
    frames = {
        (player.id, _time_key(frame.t)): frame
        for player in tracks_obj.players
        for frame in player.frames
    }
    return meta, frames


def _ball(
    ball_obj: BallTrack | Mapping[str, Any] | None,
    *,
    calibration: CourtCalibration,
    ball_world_policy: BallWorldPolicy,
    physics_filled: bool = False,
    evidence_path: str | None = None,
) -> dict[str, Any]:
    if ball_obj is None:
        return {"source": None, "frames": []}
    raw_source = _get(ball_obj, "source")
    source = str(raw_source or ("physics_filled" if physics_filled else "fused"))
    if source not in {"tracknet", "wasb", "fused", "tap", "pbmat", "totnet", "vn_trajectories", "physics_filled", "sam31"}:
        source = "physics_filled" if physics_filled else "fused"
    frames = [
        _ball_frame(
            frame,
            calibration=calibration,
            ball_world_policy=ball_world_policy,
            physics_filled=physics_filled,
            evidence_path=evidence_path,
        )
        for frame in _sequence(_get(ball_obj, "frames"))
    ]
    payload = {"source": source, "frames": frames}
    if physics_filled and frames:
        payload["trust_band"] = _physics_trust_band(
            stage="PHYS-BALL",
            gate_id="ball_track_physics_filled",
            status="physics_derived",
            reason=(
                "Ball trajectory consumed from ball_track_physics_filled.json; virtual_world applies no temporal "
                "smoothing and therefore does not smooth across bounce events."
            ),
            evidence_path=evidence_path,
        )
    return payload


def _ball_frame(
    frame: Any,
    *,
    calibration: CourtCalibration,
    ball_world_policy: BallWorldPolicy = DEFAULT_BALL_WORLD_POLICY,
    physics_filled: bool = False,
    evidence_path: str | None = None,
) -> dict[str, Any]:
    world_xyz = _ball_world_xyz(frame, calibration=calibration, ball_world_policy=ball_world_policy)
    status = _physics_ball_frame_status(frame) if physics_filled else None
    approx = bool(_get(frame, "approx") or (_get(frame, "world_xyz") is None and world_xyz is not None))
    physics_fill_meta = _get(frame, "physics_fill")
    render_only = bool(_get(frame, "render_only")) or approx or isinstance(physics_fill_meta, Mapping)
    payload = {
        "t": float(_get(frame, "t")),
        "xy": list(_get(frame, "xy")),
        "conf": float(_get(frame, "conf")),
        "visible": bool(_get(frame, "visible")),
        "world_xyz": world_xyz,
        "approx": approx,
        "trust_band": _physics_trust_band(
            stage="PHYS-BALL",
            gate_id="ball_track_physics_filled",
            status=status or "physics_derived",
            reason=(
                f"Ball sample consumed from ball_track_physics_filled.json as {status or 'physics_derived'}; "
                "virtual_world preserves the sample and applies no smoothing across bounce events."
            ),
            evidence_path=evidence_path,
        )
        if physics_filled
        else None,
    }
    if render_only:
        payload["render_only"] = True
        payload["not_for_detection_metrics"] = True
    return payload


def _legacy_ball(ball_obj: BallTrack | None, *, calibration: CourtCalibration) -> dict[str, Any]:
    return {
        "source": ball_obj.source,
        "frames": [_ball_frame(frame, calibration=calibration) for frame in ball_obj.frames],
    }


def _ball_world_xyz(
    frame: Any,
    *,
    calibration: CourtCalibration,
    ball_world_policy: BallWorldPolicy = DEFAULT_BALL_WORLD_POLICY,
) -> list[float] | None:
    raw_world = _get(frame, "world_xyz")
    if raw_world is not None:
        return list(raw_world)
    if ball_world_policy != "court_plane_approx_for_review_only":
        return None
    if not bool(_get(frame, "visible")):
        return None
    try:
        world_xy = project_image_points_to_world(calibration.homography, [_get(frame, "xy")])[0]
    except ValueError:
        return None
    if not _world_xy_in_court(calibration, world_xy, margin_m=0.35):
        return None
    return [float(world_xy[0]), float(world_xy[1]), 0.0]


def _physics_ball_frame_status(frame: Any) -> str | None:
    if str(_get(frame, "source") or "") == "physics_interpolated":
        return "interpolated"
    return _trust_status(_get(frame, "trust_band"))


def _world_xy_in_court(calibration: CourtCalibration, world_xy: Any, *, margin_m: float) -> bool:
    template = get_court_template(calibration.sport)
    points = [point for segment in template.line_segments_m.values() for point in segment]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x = float(world_xy[0])
    y = float(world_xy[1])
    return min(xs) - margin_m <= x <= max(xs) + margin_m and min(ys) - margin_m <= y <= max(ys) + margin_m


def _paddles(
    racket_obj: RacketPose | Mapping[str, Any] | None,
    *,
    calibration: CourtCalibration,
    estimate: bool = False,
    evidence_path: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    if racket_obj is None:
        return [], [], {}
    paddles = []
    suppressed_box_derived_count = 0
    metadata = _paddle_metadata(racket_obj)
    for player in _sequence(_get(racket_obj, "players")):
        player_id = _maybe_int(_get(player, "id"))
        if player_id is None:
            continue
        paddle_dims = dict(_get(player, "paddle_dims_in") or {})
        frames = []
        for frame in _sequence(_get(player, "frames")):
            if is_box_derived_source(str(_get(frame, "source") or "")):
                suppressed_box_derived_count += 1
                continue
            frames.append(
                _paddle_frame(
                    frame,
                    calibration=calibration,
                    paddle_dims_in=paddle_dims,
                    estimate=estimate,
                    evidence_path=evidence_path,
                )
            )
        if frames:
            paddle = {"player_id": player_id, "paddle_dims_in": paddle_dims, "frames": frames}
            if estimate:
                paddle["trust_band"] = _paddle_estimate_trust_band(
                    source_kind=metadata.get("source_kind"),
                    status=_paddle_entity_status(frames),
                    evidence_path=evidence_path,
                )
            paddles.append(paddle)
    warnings = ["box_derived_paddle_pose_suppressed"] if suppressed_box_derived_count else []
    return paddles, warnings, metadata


def _paddle_frame(
    frame: Any,
    *,
    calibration: CourtCalibration,
    paddle_dims_in: Mapping[str, float],
    estimate: bool = False,
    evidence_path: str | None = None,
) -> dict[str, Any]:
    pose = _get(frame, "pose_se3")
    rotation = _get(pose, "R")
    translation = _get(pose, "t")
    source = str(_get(frame, "source") or "unspecified")
    translation_unit = str(_get(frame, "translation_unit") or "cm")
    conf = float(_get(frame, "conf"))
    if _get(frame, "world_frame") == "court_Z0":
        scale = 0.01 if translation_unit == "cm" else 1.0
        pose_se3 = {"R": [list(row) for row in rotation], "t": [float(value) * scale for value in translation]}
        source = source if source.endswith(":court_Z0") else f"{source}:court_Z0"
    else:
        converted = camera_paddle_pose_to_court_world(
            SE3PoseConfidence(R=rotation, t=translation, confidence=conf, source=source),
            calibration,
            input_translation_unit=translation_unit,
        )
        pose_se3 = {"R": [list(row) for row in converted.R], "t": list(converted.t)}
        source = converted.source
    mesh_vertices_world = _paddle_mesh_vertices_world(pose_se3, paddle_dims_in)
    status = _trust_status(_get(frame, "trust_band")) or _trust_status(_get(frame, "trust")) or "physics_derived"
    source_kind = "wrist_proxy_estimated" if _is_wrist_proxy_source(source) else "racket_pose_estimate"
    payload = {
        "t": float(_get(frame, "t")),
        "pose_se3": pose_se3,
        "mesh_vertices_world": mesh_vertices_world,
        "mesh_faces": PADDLE_MESH_FACES,
        "conf": conf,
        "world_frame": WORLD_FRAME,
        "translation_unit": "m",
        "source": source,
        "reprojection_error_px": _get(frame, "reprojection_error_px"),
        "ambiguous": bool(_get(frame, "ambiguous") or False),
        "trust_band": _paddle_frame_trust_band(
            source_kind=source_kind,
            status=status,
            evidence_path=evidence_path,
        )
        if estimate
        else None,
    }
    if _get(frame, "render_only") is not None or source_kind == "wrist_proxy_estimated":
        payload["render_only"] = bool(_get(frame, "render_only") if _get(frame, "render_only") is not None else True)
    if _get(frame, "not_for_detection_metrics") is not None or source_kind == "wrist_proxy_estimated":
        payload["not_for_detection_metrics"] = bool(
            _get(frame, "not_for_detection_metrics") if _get(frame, "not_for_detection_metrics") is not None else True
        )
    confidence_provenance = _get(frame, "confidence_provenance")
    if isinstance(confidence_provenance, Mapping):
        payload["confidence_provenance"] = dict(confidence_provenance)
    return payload


def _paddle_metadata(racket_obj: RacketPose | Mapping[str, Any]) -> dict[str, Any]:
    source_kind = None
    if _is_wrist_proxy_source(str(_get(racket_obj, "source") or "")):
        source_kind = "wrist_proxy_estimated"
    for player in _sequence(_get(racket_obj, "players")):
        for frame in _sequence(_get(player, "frames")):
            if _is_wrist_proxy_source(str(_get(frame, "source") or "")):
                source_kind = "wrist_proxy_estimated"
                break
        if source_kind == "wrist_proxy_estimated":
            break
    summary = _get(racket_obj, "summary") or {}
    hidden_count = _maybe_int(_get(summary, "hidden_frame_count")) or 0
    metadata: dict[str, Any] = {"hidden_frame_count": hidden_count}
    if source_kind is not None:
        metadata["source_kind"] = source_kind
    return metadata


def _paddle_entity_status(frames: list[Mapping[str, Any]]) -> str:
    statuses = [_trust_status(frame.get("trust_band")) for frame in frames]
    if any(status == "estimated_from_wrist" for status in statuses):
        return "estimated_from_wrist"
    if any(status == "physics_derived" for status in statuses):
        return "physics_derived"
    return "physics_derived"


def _is_wrist_proxy_source(source: str) -> bool:
    return source == "wrist_proxy" or source.startswith("wrist_proxy:")


def _paddle_estimate_trust_band(
    *,
    source_kind: Any,
    status: str,
    evidence_path: str | None,
) -> dict[str, Any]:
    if source_kind == "wrist_proxy_estimated" or status == "estimated_from_wrist":
        return build_trust_band(
            stage="RKT",
            gate_id="wrist_proxy_estimated_paddle",
            gate_status="estimated_from_wrist",
            badge="low_confidence",
            reason=(
                "Paddle is a wrist-anchored render-only proxy from skeleton joints. "
                "It is not true paddle 6DoF and does not score or promote the RKT gate."
            ),
            evidence_path=evidence_path,
        )
    return _physics_trust_band(
        stage="PHYS-RACKET",
        gate_id="racket_pose_estimate",
        status=status,
        reason="Paddle pose consumed from racket_pose_estimate.json as a physics-derived estimate; not a promoted RKT gate result.",
        evidence_path=evidence_path,
    )


def _paddle_frame_trust_band(
    *,
    source_kind: str,
    status: str,
    evidence_path: str | None,
) -> dict[str, Any]:
    if source_kind == "wrist_proxy_estimated" or status == "estimated_from_wrist":
        return _paddle_estimate_trust_band(
            source_kind="wrist_proxy_estimated",
            status="estimated_from_wrist",
            evidence_path=evidence_path,
        )
    return _physics_trust_band(
        stage="PHYS-RACKET",
        gate_id="racket_pose_estimate",
        status=status,
        reason=f"Paddle pose sample consumed from racket_pose_estimate.json as {status}; not a promoted RKT gate result.",
        evidence_path=evidence_path,
    )


def _paddle_mesh_vertices_world(pose_se3: Mapping[str, Any], paddle_dims_in: Mapping[str, float]) -> list[list[float]]:
    rotation = pose_se3["R"]
    translation = pose_se3["t"]
    vertices = []
    face_corners_cm = paddle_face_corners_object_cm(paddle_dims_in)
    face_points_m = [[float(value) / 100.0 for value in corner_cm] for corner_cm in face_corners_cm]
    face_bottom_y_m = min(float(corner[1]) / 100.0 for corner in face_corners_cm)
    handle_half_width_m = PADDLE_HANDLE_WIDTH_M / 2.0
    handle_bottom_y_m = face_bottom_y_m - PADDLE_HANDLE_LENGTH_M
    handle_points_m = [
        [-handle_half_width_m, face_bottom_y_m, 0.0],
        [handle_half_width_m, face_bottom_y_m, 0.0],
        [handle_half_width_m, handle_bottom_y_m, 0.0],
        [-handle_half_width_m, handle_bottom_y_m, 0.0],
    ]
    half_thickness_m = PADDLE_THICKNESS_M / 2.0
    for point_m in face_points_m:
        vertices.append(
            _transform_paddle_local_m(rotation=rotation, translation=translation, point_m=[point_m[0], point_m[1], half_thickness_m])
        )
    for point_m in face_points_m:
        vertices.append(
            _transform_paddle_local_m(rotation=rotation, translation=translation, point_m=[point_m[0], point_m[1], -half_thickness_m])
        )
    for handle_point_m in handle_points_m:
        vertices.append(
            _transform_paddle_local_m(
                rotation=rotation,
                translation=translation,
                point_m=[handle_point_m[0], handle_point_m[1], half_thickness_m],
            )
        )
    for handle_point_m in handle_points_m:
        vertices.append(
            _transform_paddle_local_m(
                rotation=rotation,
                translation=translation,
                point_m=[handle_point_m[0], handle_point_m[1], -half_thickness_m],
            )
        )
    return vertices


def _transform_paddle_local_m(
    *,
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
    point_m: Sequence[float],
) -> list[float]:
    return [
        sum(float(rotation[row][col]) * float(point_m[col]) for col in range(3)) + float(translation[row])
        for row in range(3)
    ]


def _attach_trust_bands(
    *,
    court: dict[str, Any],
    players: list[dict[str, Any]],
    ball: dict[str, Any],
    paddles: list[dict[str, Any]],
    trust_bands: Mapping[str, Mapping[str, Any] | None] | None,
) -> None:
    if not trust_bands:
        return
    court_band = trust_bands.get("court")
    if court_band is not None:
        court["trust_band"] = dict(court_band)

    body_band = trust_bands.get("body")
    track_band = trust_bands.get("track")
    for player in players:
        band = body_band if player["representation"] in ("mesh", "joints") else track_band
        if band is not None:
            player["trust_band"] = dict(band)

    ball_band = trust_bands.get("ball")
    if ball_band is not None and ball.get("frames"):
        ball["trust_band"] = dict(ball_band)

    paddle_band = trust_bands.get("paddle")
    if paddle_band is not None:
        for paddle in paddles:
            paddle["trust_band"] = dict(paddle_band)


def _warnings(
    *,
    players: list[dict[str, Any]],
    ball: dict[str, Any],
    paddles: list[dict[str, Any]],
    mesh_index_present: bool = False,
) -> list[str]:
    warnings = []
    if not players:
        warnings.append("missing_players")
    elif not any(player["representation"] == "mesh" for player in players):
        warnings.append("missing_embedded_mesh_vertices" if mesh_index_present else "missing_mesh_vertices")
    if not ball["frames"]:
        warnings.append("missing_ball_track")
    elif any(frame.get("visible") is True and frame.get("world_xyz") is None for frame in ball["frames"]):
        warnings.append("unprojected_visible_ball_frames")
    if not any(paddle["frames"] for paddle in paddles):
        warnings.append("missing_paddle_pose")
    if any(frame.get("ambiguous") for paddle in paddles for frame in paddle["frames"]):
        warnings.append("ambiguous_paddle_pose")
    return warnings


def _body_mesh_index_present(
    *,
    paths: Mapping[str, str],
    placement_calibration_path: str | Path | None,
) -> bool:
    candidates: list[Path] = []
    explicit = paths.get("body_mesh_index")
    if explicit:
        candidates.append(Path(explicit))
    if placement_calibration_path is not None:
        candidates.extend(_body_mesh_index_candidates(Path(placement_calibration_path).parent))
    return any(_body_mesh_index_file_is_nonempty(candidate) for candidate in candidates)


def _existing_body_mesh_index_path(root: Path) -> Path | None:
    for candidate in _body_mesh_index_candidates(root):
        if candidate.is_file():
            return candidate
    return None


def _body_mesh_index_candidates(root: Path) -> list[Path]:
    return [root / "body_mesh_index" / "body_mesh_index.json", root / "body_mesh_index.json"]


def _body_mesh_index_file_is_nonempty(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        mesh_frame_count = summary.get("mesh_frame_count")
        if isinstance(mesh_frame_count, int) and mesh_frame_count > 0:
            return True
    windows = payload.get("windows")
    return isinstance(windows, list) and len(windows) > 0


def _summary(
    *,
    players: list[dict[str, Any]],
    ball: dict[str, Any],
    paddles: list[dict[str, Any]],
    warnings: list[str],
    fps: float,
    paddle_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    player_frames = [
        frame
        for player in players
        for frame in player["frames"]
    ]
    paddle_frames = [
        frame
        for paddle in paddles
        for frame in paddle["frames"]
    ]
    ball_frames = ball["frames"]
    metadata = paddle_metadata or {}
    summary = {
        "player_count": len(players),
        "mesh_player_count": sum(1 for player in players if player["representation"] == "mesh"),
        "mesh_player_frame_count": sum(1 for frame in player_frames if int(frame["mesh_vertex_count"]) > 0),
        "joint_player_frame_count": sum(1 for frame in player_frames if int(frame["joint_count"]) > 0),
        "track_only_player_frame_count": sum(
            1
            for frame in player_frames
            if frame.get("track_world_xy") is not None
            and int(frame["mesh_vertex_count"]) == 0
            and int(frame["joint_count"]) == 0
        ),
        "floor_placed_player_frame_count": sum(1 for frame in player_frames if frame.get("floor_world_xyz") is not None),
        "floor_contact_player_frame_count": sum(1 for frame in player_frames if frame.get("contact_locked") is True),
        "max_floor_penetration_m": max((float(frame.get("floor_penetration_m") or 0.0) for frame in player_frames), default=0.0),
        "max_abs_floor_offset_m": max((abs(float(frame["floor_offset_m"])) for frame in player_frames if frame.get("floor_offset_m") is not None), default=0.0),
        "physics_modes": sorted({str(frame["physics"]) for frame in player_frames if frame.get("physics")}),
        "ball_frame_count": len(ball_frames),
        "approx_ball_frame_count": sum(1 for frame in ball_frames if frame.get("approx")),
        "paddle_player_count": sum(1 for paddle in paddles if paddle["frames"]),
        "paddle_frame_count": len(paddle_frames),
        "ambiguous_paddle_frame_count": sum(1 for frame in paddle_frames if frame.get("ambiguous")),
        "temporal_coverage": _temporal_coverage(players=players, ball=ball, paddles=paddles, fps=fps),
        "warnings": warnings,
    }
    source_kind = metadata.get("source_kind")
    if source_kind is not None and paddle_frames:
        summary["paddle_source"] = str(source_kind)
    hidden_count = _maybe_int(metadata.get("hidden_frame_count")) or 0
    if hidden_count:
        summary["hidden_paddle_frame_count"] = hidden_count
    return summary


def _schema_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "temporal_coverage"}


def _temporal_coverage(
    *,
    players: list[dict[str, Any]],
    ball: dict[str, Any],
    paddles: list[dict[str, Any]],
    fps: float,
) -> dict[str, Any]:
    frame_period = _frame_period(fps)
    all_frames = [
        frame
        for player in players
        for frame in player["frames"]
    ] + [
        frame
        for frame in ball["frames"]
    ] + [
        frame
        for paddle in paddles
        for frame in paddle["frames"]
    ]
    clip_times = _frame_times(all_frames)
    clip_min_t = min(clip_times) if clip_times else None
    clip_max_t = max(clip_times) if clip_times else None
    clip_frame_span = _span_seconds(clip_times, frame_period=frame_period)
    clip_duration = clip_frame_span if clip_frame_span > 0.0 else 0.0
    return {
        "clip_min_t": _maybe_round(clip_min_t),
        "clip_max_t": _maybe_round(clip_max_t),
        "clip_frame_span": _round_seconds(clip_frame_span),
        "ball": _coverage_entry(ball["frames"], clip_duration=clip_duration, frame_period=frame_period),
        "players": [
            {
                "player_id": int(player["id"]),
                **_coverage_entry(
                    [frame for frame in player["frames"] if not _is_no_data_player_frame(frame)],
                    clip_duration=clip_duration,
                    frame_period=frame_period,
                ),
                "observed_frame_count": sum(1 for frame in player["frames"] if not _is_no_data_player_frame(frame)),
                "no_data_frame_count": sum(1 for frame in player["frames"] if _is_no_data_player_frame(frame)),
            }
            for player in players
        ],
    }


def _coverage_entry(frames: list[Mapping[str, Any]], *, clip_duration: float, frame_period: float) -> dict[str, float | None]:
    times = _frame_times(frames)
    if not times or clip_duration <= 0.0:
        return {
            "min_t": None,
            "max_t": None,
            "frame_span": 0.0,
            "coverage_fraction": 0.0,
        }
    frame_span = min(_span_seconds(times, frame_period=frame_period), clip_duration)
    return {
        "min_t": _round_seconds(min(times)),
        "max_t": _round_seconds(max(times)),
        "frame_span": _round_seconds(frame_span),
        "coverage_fraction": _round_fraction(frame_span / clip_duration),
    }


def _frame_times(frames: list[Mapping[str, Any]]) -> list[float]:
    times = []
    for frame in frames:
        try:
            times.append(float(frame["t"]))
        except (KeyError, TypeError, ValueError):
            continue
    return times


def _span_seconds(times: list[float], *, frame_period: float) -> float:
    if not times:
        return 0.0
    return max(times) - min(times) + frame_period


def _frame_period(fps: float) -> float:
    return 1.0 / fps if fps > 0.0 else 0.0


def _is_no_data_player_frame(frame: Mapping[str, Any]) -> bool:
    trust_band = frame.get("trust_band")
    return isinstance(trust_band, Mapping) and trust_band.get("gate_id") == "world_no_data_placeholder"


def _maybe_round(value: float | None) -> float | None:
    return None if value is None else _round_seconds(value)


def _round_seconds(value: float) -> float:
    return round(float(value), 6)


def _round_fraction(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


def _world_fps(*artifacts: Any) -> float:
    for artifact in artifacts:
        fps = _get(artifact, "fps", 0.0) if artifact is not None else 0.0
        if fps:
            return float(fps)
    return 30.0


def _time_key(t: float) -> str:
    return f"{float(t):.6f}"


def _raw_mapping(value: Mapping[str, Any] | None, *, artifact: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{artifact} must be a JSON-object mapping until its strict schema lands")
    return value


def _get(value: Any, field: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def _first_present(value: Any, *fields: str) -> Any:
    for field in fields:
        candidate = _get(value, field)
        if candidate is not None:
            return candidate
    return None


def _vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    items = list(value)
    if len(items) != 3:
        return None
    vector = [float(items[0]), float(items[1]), float(items[2])]
    return vector if all(math.isfinite(item) for item in vector) else None


def _vector2(value: Any) -> list[float] | None:
    if value is None:
        return None
    items = list(value)
    if len(items) < 2:
        return None
    try:
        vector = [float(items[0]), float(items[1])]
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(item) for item in vector) else None


def _track_world_xy(track_frame: Any) -> list[float]:
    track_xy = _vector2(getattr(track_frame, "world_xy", None))
    if track_xy is None:
        raise ValueError("track world_xy must be a finite 2-vector")
    return track_xy


def _distance2(left: Sequence[float], right: Sequence[float]) -> float:
    return ((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2) ** 0.5


def _vectors3(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    vectors = [_vector3(item) for item in _sequence(value)]
    if any(vector is None for vector in vectors):
        return None
    return [vector for vector in vectors if vector is not None]


def _float_sequence(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(item) for item in _sequence(value)]
    except (TypeError, ValueError):
        return None


def _foot_contact(value: Any) -> dict[str, bool] | None:
    if value is None:
        return None
    left = _get(value, "left")
    right = _get(value, "right")
    if left is None or right is None:
        return None
    return {"left": bool(left), "right": bool(right)}


def _foot_lock_contact(value: Any) -> dict[str, bool] | None:
    contacts = _sequence(_get(value, "active_contacts"))
    if not contacts:
        return None
    feet = {str(_get(contact, "foot") or "").lower() for contact in contacts}
    return {"left": "left" in feet, "right": "right" in feet}


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trust_status(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("gate_status") or value.get("status") or value.get("band")
    if value is None:
        return None
    status = str(value)
    return status if status in PHYSICS_TRUST_STATUSES | PADDLE_PROXY_TRUST_STATUSES else None


def _drop_empty_optional_world_fields(payload: dict[str, Any]) -> None:
    if payload.get("joint_names") in (None, []):
        payload.pop("joint_names", None)
    if payload.get("foot_pin") is None:
        payload.pop("foot_pin", None)
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("paddle_source", "hidden_paddle_frame_count"):
            if summary.get(key) is None:
                summary.pop(key, None)
    for player in payload.get("players", []):
        if not isinstance(player, dict):
            continue
        if player.get("joints_source") is None:
            player.pop("joints_source", None)
    for paddle in payload.get("paddles", []):
        if not isinstance(paddle, dict):
            continue
        for frame in paddle.get("frames", []):
            if not isinstance(frame, dict):
                continue
            for key in ("confidence_provenance", "render_only", "not_for_detection_metrics"):
                if frame.get(key) is None:
                    frame.pop(key, None)


def _physics_trust_band(
    *,
    stage: str,
    gate_id: str,
    status: str,
    reason: str,
    evidence_path: str | None,
) -> dict[str, Any]:
    badge = "preview" if status in PHYSICS_TRUST_STATUSES else "low_confidence"
    return build_trust_band(
        stage=stage,
        gate_id=gate_id,
        gate_status=status,
        badge=badge,
        reason=reason,
        evidence_path=evidence_path,
    )


def _court_calibration(value: CourtCalibration | Mapping[str, Any]) -> CourtCalibration:
    if isinstance(value, CourtCalibration):
        return value
    try:
        return CourtCalibration.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"court_calibration failed validation: {exc}") from exc


def _tracks(value: Tracks | Mapping[str, Any] | None) -> Tracks | None:
    if value is None or isinstance(value, Tracks):
        return value
    try:
        return Tracks.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"tracks failed validation: {exc}") from exc


def _smpl_motion(value: SmplMotion | Mapping[str, Any] | None) -> SmplMotion | None:
    if value is None or isinstance(value, SmplMotion):
        return value
    try:
        return SmplMotion.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"smpl_motion failed validation: {exc}") from exc


def _skeleton3d(value: Skeleton3D | Mapping[str, Any] | None) -> Skeleton3D | None:
    if value is None or isinstance(value, Skeleton3D):
        return value
    try:
        return Skeleton3D.model_validate(_strip_legacy_skeleton3d_extras(value))
    except ValidationError as exc:
        raise ValueError(f"skeleton3d failed validation: {exc}") from exc


def _strip_legacy_skeleton3d_extras(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("foot_pin", None)
    return payload


def _ball_track(value: BallTrack | Mapping[str, Any] | None) -> BallTrack | None:
    if value is None or isinstance(value, BallTrack):
        return value
    try:
        return BallTrack.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"ball_track failed validation: {exc}") from exc


def _racket_pose(value: RacketPose | Mapping[str, Any] | None) -> RacketPose | None:
    if value is None or isinstance(value, RacketPose):
        return value
    try:
        return RacketPose.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"racket_pose failed validation: {exc}") from exc


__all__ = [
    "apply_ball_track_arc_solved_overlay",
    "build_virtual_world_state",
    "build_virtual_world_state_from_files",
    "build_virtual_world_state_from_run_dir",
    "resolve_best_court_calibration_path",
    "write_virtual_world",
]
