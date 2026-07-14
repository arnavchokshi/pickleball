"""Unwired whole-rally, piecewise-ballistic BALL track candidate.

The candidate associates *all supplied* 2D detections by trajectory consistency;
it never applies a detector-score floor.  Consecutive gravity-only segments share
one endpoint at every in-sequence event, so position is continuous while velocity
may change at a contact or bounce.  A rally is emitted only when every segment is
viable and every interior observation hole is at most six frames.

This module intentionally has no pipeline or CLI wiring.  Its output is preview
evidence: ``measured-candidate`` means a model candidate was a member of the
track, not that the sample has measured authority.  Every other emitted sample
is ``physics_predicted`` and every sample carries a posterior and covariance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


ARTIFACT_TYPE = "racketsport_ball_global_track_candidate"
SCHEMA_VERSION = 1
BALL_RADIUS_M = 0.0371
_BANDS = {"measured-candidate", "physics_predicted"}


@dataclass(frozen=True)
class GlobalTrackConfig:
    gravity_mps2: float = 9.80665
    ball_radius_m: float = BALL_RADIUS_M
    robust_pixel_sigma_px: float = 6.0
    membership_gate_px: float = 18.0
    min_members_per_segment: int = 3
    max_membership_iterations: int = 6
    max_hole_frames: int = 6
    max_speed_mps: float = 35.0
    min_height_m: float = -0.05
    max_height_m: float = 12.0
    anchor_sigma_m: float = 0.25
    weak_endpoint_sigma_m: float = 1.5
    covariance_floor_m2: float = 1.0e-4
    predicted_process_variance_m2_per_frame2: float = 2.5e-4
    pre_track_pad_s: float = 0.9
    post_track_pad_s: float = 1.1
    radius_confidence_min: float = 0.7
    radius_min_observations: int = 12
    radius_min_r2: float = 0.25
    radius_match_px: float = 12.0
    radius_log_sigma: float = 0.18
    radius_residual_weight: float = 0.35
    max_nfev: int = 800

    def __post_init__(self) -> None:
        positive = (
            self.gravity_mps2,
            self.ball_radius_m,
            self.robust_pixel_sigma_px,
            self.membership_gate_px,
            self.max_speed_mps,
            self.anchor_sigma_m,
            self.weak_endpoint_sigma_m,
            self.covariance_floor_m2,
            self.predicted_process_variance_m2_per_frame2,
            self.radius_match_px,
            self.radius_log_sigma,
            self.max_nfev,
        )
        if min(positive) <= 0:
            raise ValueError("positive global-track configuration values must be > 0")
        if self.min_members_per_segment < 2:
            raise ValueError("min_members_per_segment must be at least 2")
        if self.max_membership_iterations < 1:
            raise ValueError("max_membership_iterations must be positive")
        if self.max_hole_frames < 0 or self.max_hole_frames > 6:
            raise ValueError("max_hole_frames must be in [0, 6]")
        if not 0.0 < self.radius_confidence_min <= 1.0:
            raise ValueError("radius_confidence_min must be in (0, 1]")
        if self.radius_min_observations < 3:
            raise ValueError("radius_min_observations must be at least 3")
        if not 0.0 <= self.radius_min_r2 <= 1.0:
            raise ValueError("radius_min_r2 must be in [0, 1]")
        if self.pre_track_pad_s < 0 or self.post_track_pad_s < 0:
            raise ValueError("track-window pads must be non-negative")


@dataclass(frozen=True)
class _Candidate:
    frame: int
    xy: tuple[float, float]
    score: float | None
    source: str
    rank: int


@dataclass(frozen=True)
class _Boundary:
    frame: int
    t: float
    kind: str
    world_xyz: tuple[float, float, float] | None
    sigma_m: float
    source: str
    anchor_id: str


@dataclass(frozen=True)
class _RadiusObservation:
    frame: int
    xy: tuple[float, float]
    radius_px: float
    confidence: float


def build_global_ball_track(
    ball_candidates: Mapping[str, Any],
    *,
    calibration: Mapping[str, Any],
    event_boundaries: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ball_track: Mapping[str, Any] | None = None,
    ball_size_observations: Mapping[str, Any] | None = None,
    ball_apparent_radius: Mapping[str, Any] | None = None,
    rally_bounds: tuple[int, int] | None = None,
    config: GlobalTrackConfig | None = None,
    clip_id: str | None = None,
) -> dict[str, Any]:
    """Fit one fail-closed global track over a rally.

    Detector scores are retained for diagnostics but never used as a membership
    floor or as the deciding term.  The nearest candidate to the current robust
    ballistic projection wins if and only if it is inside ``membership_gate_px``.
    """

    cfg = config or GlobalTrackConfig()
    fps = _positive_float(ball_candidates.get("fps")) or _positive_float((ball_track or {}).get("fps")) or 30.0
    candidates, input_stats = _candidate_pool(ball_candidates, ball_track)
    raw_events = _event_items(event_boundaries)
    boundaries, quarantined, serve_policy = _boundaries(raw_events, fps=fps, cfg=cfg)
    refusal_context = {
        "clip_id": clip_id,
        "fps": fps,
        "input_stats": input_stats,
        "quarantined_events": quarantined,
        "serve_initialization": serve_policy,
    }
    if not candidates:
        return _refusal("no_2d_candidates", ["the rally contains no usable 2D candidates"], cfg, refusal_context)
    if len(boundaries) < 2:
        return _refusal(
            "insufficient_event_boundaries",
            ["at least two in-sequence event boundaries are required"],
            cfg,
            refusal_context,
        )

    candidate_min = min(candidates)
    candidate_max = max(candidates)
    if rally_bounds is None:
        rally_start, rally_end = min(boundaries[0].frame, candidate_min), max(boundaries[-1].frame, candidate_max)
    else:
        rally_start, rally_end = int(rally_bounds[0]), int(rally_bounds[1])
    if rally_start < 0 or rally_end < rally_start:
        return _refusal("invalid_rally_bounds", ["rally bounds are invalid"], cfg, refusal_context)
    boundaries = [item for item in boundaries if rally_start <= item.frame <= rally_end]
    if len(boundaries) < 2:
        return _refusal(
            "event_boundaries_outside_rally",
            ["fewer than two event boundaries remain inside the rally bounds"],
            cfg,
            refusal_context,
        )

    try:
        initial_points = _initial_boundary_points(boundaries, candidates, calibration, cfg)
        camera = _camera(calibration)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return _refusal("invalid_geometry_inputs", [str(exc)], cfg, refusal_context)

    radii = _radius_observations(ball_size_observations, ball_apparent_radius)
    fixed = {
        3 * index + 2: cfg.ball_radius_m
        for index, boundary in enumerate(boundaries)
        if boundary.kind == "bounce"
    }
    full_initial = np.asarray(initial_points, dtype=float).reshape(-1)
    for index, value in fixed.items():
        full_initial[index] = value
    free_indexes = np.asarray([index for index in range(len(full_initial)) if index not in fixed], dtype=int)
    if free_indexes.size == 0:
        return _refusal("no_free_fit_parameters", ["all boundary coordinates were fixed"], cfg, refusal_context)

    membership = _select_membership(
        full_initial, boundaries, candidates, camera, cfg, rally_start=rally_start, rally_end=rally_end
    )
    if len(membership) < cfg.min_members_per_segment:
        return _refusal(
            "insufficient_track_membership",
            [f"only {len(membership)} candidate frames passed the trajectory-consistency gate"],
            cfg,
            {**refusal_context, "initial_membership_count": len(membership)},
        )

    fit_result = None
    radius_calibration: dict[str, Any] = _radius_abstention("not_evaluated", 0)
    membership_iterations: list[dict[str, Any]] = []
    full_params = full_initial
    for iteration in range(1, cfg.max_membership_iterations + 1):
        radius_calibration = _calibrate_radius(
            full_params, boundaries, membership, radii, camera, cfg
        )
        fit_result, full_params = _fit_shared_boundaries(
            full_params,
            free_indexes,
            fixed,
            boundaries,
            membership,
            radii,
            radius_calibration,
            camera,
            cfg,
        )
        next_membership = _select_membership(
            full_params, boundaries, candidates, camera, cfg, rally_start=rally_start, rally_end=rally_end
        )
        membership_iterations.append(
            {
                "iteration": iteration,
                "member_count": len(next_membership),
                "changed_frame_count": len(set(next_membership) ^ set(membership)),
                "radius_status": radius_calibration["status"],
            }
        )
        if _membership_identity(next_membership) == _membership_identity(membership):
            membership = next_membership
            break
        membership = next_membership

    if fit_result is None or not bool(fit_result.success) or not np.all(np.isfinite(full_params)):
        return _refusal(
            "robust_fit_failed",
            [str(getattr(fit_result, "message", "least-squares fit did not run"))],
            cfg,
            {**refusal_context, "membership_iterations": membership_iterations},
        )
    # Refit once against the final membership so diagnostics and covariance are
    # not reported from the preceding assignment.
    radius_calibration = _calibrate_radius(full_params, boundaries, membership, radii, camera, cfg)
    fit_result, full_params = _fit_shared_boundaries(
        full_params,
        free_indexes,
        fixed,
        boundaries,
        membership,
        radii,
        radius_calibration,
        camera,
        cfg,
    )
    membership = _select_membership(
        full_params, boundaries, candidates, camera, cfg, rally_start=rally_start, rally_end=rally_end
    )

    endpoint_covariance = _endpoint_covariance(fit_result, free_indexes, len(full_params), cfg)
    segments = _segment_diagnostics(full_params, boundaries, membership, camera, cfg)
    fallback_segments = [item for item in segments if item["status"] != "fit"]

    member_frames = sorted(membership)
    first_member, last_member = member_frames[0], member_frames[-1]
    serve_frame = next((item.frame for item in boundaries if item.kind in {"serve", "contact", "shot"}), None)
    if serve_frame is not None:
        emission_start = serve_frame
        terminal_frame = boundaries[-1].frame
        emission_policy = "first_contact_to_terminal_event"
    else:
        emission_start = first_member
        terminal_frame = last_member
        emission_policy = "track_extent_no_contact_available"
    emission_start = max(rally_start, emission_start)
    terminal_frame = min(rally_end, terminal_frame)
    holes = _missing_runs(set(member_frames), emission_start, terminal_frame)
    long_holes = [item for item in holes if item["length_frames"] > cfg.max_hole_frames]

    points = np.asarray(full_params, dtype=float).reshape((-1, 3))
    speed_violations = _segment_speed_violations(points, boundaries, cfg)
    height_violations = _height_violations(points, boundaries, emission_start, terminal_frame, fps, cfg)
    reasons: list[str] = []
    refusal_code = "no_viable_track"
    if fallback_segments:
        reasons.append(f"{len(fallback_segments)} segment(s) failed the robust-fit/physics gates")
        refusal_code = "fallback_segments_present"
    if long_holes:
        longest = max(item["length_frames"] for item in long_holes)
        reasons.append(f"interior candidate-membership hole is {longest} frames; ceiling is {cfg.max_hole_frames}")
        refusal_code = "interior_hole_above_ceiling"
    if speed_violations:
        reasons.append(f"{len(speed_violations)} segment boundary speed(s) exceed {cfg.max_speed_mps:g} m/s")
        refusal_code = "speed_ceiling_exceeded"
    if height_violations:
        reasons.append(f"{len(height_violations)} sampled positions violate height bounds")
        refusal_code = "height_bounds_exceeded"

    window_start = max(rally_start, first_member - int(round(cfg.pre_track_pad_s * fps)))
    window_end = min(rally_end, last_member + int(round(cfg.post_track_pad_s * fps)))
    membership_reprojection = []
    for frame, candidate in membership.items():
        try:
            world, _, _ = _trajectory_point(full_params, boundaries, frame, cfg.gravity_mps2)
            projected, _ = _project(camera, world)
        except ValueError:
            continue
        membership_reprojection.append(math.dist(projected, candidate.xy))
    diagnostics = {
        "fit_method": "joint_shared_boundary_two_ended_huber",
        "membership_policy": "minimum_reprojection_residual_over_all_supplied_candidates_no_score_floor",
        "membership_iterations": membership_iterations,
        "candidate_member_count": len(membership),
        "candidate_member_min_score": _optional_min(item.score for item in membership.values()),
        "candidate_member_below_0_5_count": sum(
            1 for item in membership.values() if item.score is not None and item.score < 0.5
        ),
        "reprojection_error_px": _distribution(membership_reprojection),
        "segments": segments,
        "fallback_segment_count": len(fallback_segments),
        "speed_ceiling_violations": speed_violations,
        "height_violations": height_violations,
        "radius_residual": radius_calibration,
        "interior_holes": holes,
        "max_interior_hole_frames": max((item["length_frames"] for item in holes), default=0),
        "serve_to_terminal_emission_policy": emission_policy,
        "window": {
            "track_extent": [first_member, last_member],
            "frame_start": window_start,
            "frame_end": window_end,
            "pre_pad_s": cfg.pre_track_pad_s,
            "post_pad_s": cfg.post_track_pad_s,
            "clamped_to_rally_bounds": [rally_start, rally_end],
        },
    }
    common_context = {
        **refusal_context,
        "rally_bounds": [rally_start, rally_end],
        "boundaries": [_boundary_json(item, full_params, index) for index, item in enumerate(boundaries)],
        "diagnostics": diagnostics,
    }
    if reasons:
        return _refusal(refusal_code, reasons, cfg, common_context)

    samples = _emit_samples(
        full_params,
        endpoint_covariance,
        boundaries,
        membership,
        emission_start,
        terminal_frame,
        fps,
        camera,
        cfg,
    )
    audit = _sample_contract_audit(samples)
    if not audit["passes"]:
        return _refusal("sample_contract_violation", audit["violations"], cfg, common_context)
    physics = _physics_reintegration(samples, boundaries, cfg)
    if physics["pass_rate"] != 1.0:
        return _refusal(
            "physics_reintegration_failed",
            [f"physics re-integration pass rate is {physics['pass_rate']:.6f}, not 1.0"],
            cfg,
            {**common_context, "physics_reintegration": physics},
        )

    diagnostics["sample_contract_audit"] = audit
    diagnostics["physics_reintegration"] = physics
    diagnostics["reprojection_error_px"] = _distribution(
        [_candidate_residual(sample) for sample in samples if sample["band"] == "measured-candidate"]
    )
    diagnostics["step_speed"] = _step_speed_summary(samples, cfg)
    diagnostics["bounce_at_radius"] = _bounce_radius_summary(full_params, boundaries, cfg)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "candidate_track",
        "verified": False,
        "default_off": True,
        "unwired": True,
        "render_only": True,
        "not_for_detection_metrics": True,
        "clip_id": clip_id,
        "fps": fps,
        "config": asdict(cfg),
        "inputs": input_stats,
        "policy": {
            "gravity_only": True,
            "piecewise_ballistic": True,
            "shared_event_boundary_positions": True,
            "candidate_score_floor": None,
            "max_physics_predicted_hole_frames": cfg.max_hole_frames,
            "measured_authority_allowed": False,
        },
        "serve_initialization": serve_policy,
        "quarantined_events": quarantined,
        "rally_bounds": [rally_start, rally_end],
        "window": diagnostics["window"],
        "boundaries": common_context["boundaries"],
        "segments": segments,
        "frames": samples,
        "summary": {
            "emitted_sample_count": len(samples),
            "measured_candidate_count": sum(item["band"] == "measured-candidate" for item in samples),
            "physics_predicted_count": sum(item["band"] == "physics_predicted" for item in samples),
            "fallback_segment_count": len(fallback_segments),
            "interior_gap_count": len(holes),
            "max_interior_gap_frames": max((item["length_frames"] for item in holes), default=0),
            "serve_to_terminal_continuous": len(samples) == terminal_frame - emission_start + 1,
            "posterior_mislabeled_measured_count": audit["posterior_mislabeled_measured_count"],
        },
        "diagnostics": diagnostics,
        "refusal": None,
    }


def _candidate_pool(
    payload: Mapping[str, Any], ball_track: Mapping[str, Any] | None
) -> tuple[dict[int, tuple[_Candidate, ...]], dict[str, Any]]:
    by_frame: dict[int, list[_Candidate]] = {}
    raw_count = 0
    scores: list[float] = []
    frames = payload.get("frames")
    if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
        for default_frame, frame_item in enumerate(frames):
            if not isinstance(frame_item, Mapping):
                continue
            frame = _int_or_none(frame_item.get("frame"))
            if frame is None:
                frame = default_frame
            items = frame_item.get("candidates")
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                continue
            for rank, item in enumerate(items):
                if not isinstance(item, Mapping):
                    continue
                xy = _xy(item.get("xy") or item.get("center_xy_px"))
                if xy is None:
                    continue
                score = _finite_float(item.get("score"))
                if score is not None:
                    scores.append(score)
                raw_count += 1
                by_frame.setdefault(frame, []).append(
                    _Candidate(
                        frame=frame,
                        xy=xy,
                        score=score,
                        source=str(item.get("source_detector") or payload.get("source") or "candidate"),
                        rank=rank,
                    )
                )
    primary_added = 0
    if isinstance(ball_track, Mapping):
        track_frames = ball_track.get("frames")
        if isinstance(track_frames, Sequence) and not isinstance(track_frames, (str, bytes)):
            for frame, item in enumerate(track_frames):
                if not isinstance(item, Mapping) or not bool(item.get("visible")):
                    continue
                xy = _xy(item.get("xy"))
                if xy is None or xy == (0.0, 0.0):
                    continue
                score = _finite_float(item.get("conf"))
                candidate = _Candidate(
                    frame=frame,
                    xy=xy,
                    score=score,
                    source=f"primary:{ball_track.get('source') or 'ball_track'}",
                    rank=-1,
                )
                existing = by_frame.setdefault(frame, [])
                if not any(math.dist(candidate.xy, other.xy) <= 0.5 for other in existing):
                    existing.append(candidate)
                    primary_added += 1
                    if score is not None:
                        scores.append(score)
    output = {frame: tuple(items) for frame, items in sorted(by_frame.items()) if items}
    return output, {
        "supplied_candidate_count": raw_count,
        "primary_track_candidate_count_added": primary_added,
        "candidate_frame_count": len(output),
        "candidate_score_min": min(scores) if scores else None,
        "candidate_score_max": max(scores) if scores else None,
        "candidate_score_floor_applied": None,
        "all_supplied_scores_eligible": True,
        "source_candidate_generation_floor": payload.get("heatmap_threshold"),
    }


def _event_items(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        for key in ("selected", "events", "boundaries", "items"):
            items = value.get(key)
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                return [item for item in items if isinstance(item, Mapping)]
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _boundaries(
    events: Sequence[Mapping[str, Any]], *, fps: float, cfg: GlobalTrackConfig
) -> tuple[list[_Boundary], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[_Boundary] = []
    quarantined: list[dict[str, Any]] = []
    for index, item in enumerate(events):
        frame = _int_or_none(item.get("frame") if item.get("frame") is not None else item.get("frame_index"))
        if frame is None or frame < 0:
            continue
        kind = _event_kind(item)
        if _is_out_of_sequence(item):
            quarantined.append(
                {
                    "type": "out_of_sequence_event",
                    "frame": frame,
                    "kind": kind,
                    "source": str(item.get("source") or "unknown"),
                    "used_as_boundary": False,
                }
            )
            continue
        t = _finite_float(item.get("t"))
        if t is None:
            t = frame / fps
        world = _xyz(item.get("world_xyz") or item.get("position_m"))
        sigma = _positive_float(item.get("sigma_m"))
        if sigma is None:
            sigma = cfg.weak_endpoint_sigma_m if kind == "rally_endpoint" else cfg.anchor_sigma_m
        accepted.append(
            _Boundary(
                frame=frame,
                t=t,
                kind=kind,
                world_xyz=world,
                sigma_m=sigma,
                source=str(item.get("source") or "event_boundary"),
                anchor_id=str(item.get("anchor_id") or f"event_{index:04d}"),
            )
        )
    accepted.sort(key=lambda item: (item.frame, item.t, item.anchor_id))
    deduped: list[_Boundary] = []
    for item in accepted:
        if deduped and item.frame == deduped[-1].frame:
            prior = deduped[-1]
            # Prefer an explicit contact/serve and then a bounce over a weak
            # endpoint if multiple event proposals land on one frame.
            rank = {"serve": 4, "contact": 3, "shot": 3, "bounce": 2, "net": 1, "rally_endpoint": 0}
            if rank.get(item.kind, 1) > rank.get(prior.kind, 1):
                deduped[-1] = item
            continue
        deduped.append(item)
    first_contact_index = next(
        (index for index, item in enumerate(deduped) if item.kind in {"serve", "contact", "shot"}), None
    )
    if first_contact_index is not None:
        initialization = deduped[first_contact_index]
        # The serve is constructed from the first contact.  Earlier loose
        # rally-window endpoints are window context, not trajectory anchors.
        deduped = deduped[first_contact_index:]
        policy = {
            "type": "serve_from_first_contact",
            "frame": initialization.frame,
            "anchor_id": initialization.anchor_id,
            "fallback_used": False,
        }
    else:
        policy = {
            "type": "weak_endpoint_preview_no_contact_available",
            "frame": deduped[0].frame if deduped else None,
            "anchor_id": deduped[0].anchor_id if deduped else None,
            "fallback_used": True,
        }
    return deduped, quarantined, policy


def _event_kind(item: Mapping[str, Any]) -> str:
    raw = str(item.get("kind") or item.get("type") or item.get("event_type") or "event").lower()
    if "serve" in raw:
        return "serve"
    if "contact" in raw or "hit" in raw:
        return "contact"
    if "shot" in raw:
        return "shot"
    if "bounce" in raw:
        return "bounce"
    if "net" in raw:
        return "net"
    if "endpoint" in raw or raw in {"start", "end"}:
        return "rally_endpoint"
    return raw


def _is_out_of_sequence(item: Mapping[str, Any]) -> bool:
    if bool(item.get("out_of_sequence")):
        return True
    status = str(item.get("status") or "").lower()
    if status == "out_of_sequence":
        return True
    details = item.get("details")
    return isinstance(details, Mapping) and bool(details.get("out_of_sequence"))


def _initial_boundary_points(
    boundaries: Sequence[_Boundary],
    candidates: Mapping[int, Sequence[_Candidate]],
    calibration: Mapping[str, Any],
    cfg: GlobalTrackConfig,
) -> np.ndarray:
    points: list[np.ndarray | None] = [
        np.asarray(item.world_xyz, dtype=float) if item.world_xyz is not None else None for item in boundaries
    ]
    known = [index for index, point in enumerate(points) if point is not None]
    if not known:
        raise ValueError("at least one event boundary must provide world_xyz")
    for index, point in enumerate(points):
        if point is not None:
            continue
        left = max((known_index for known_index in known if known_index < index), default=None)
        right = min((known_index for known_index in known if known_index > index), default=None)
        if left is not None and right is not None:
            span = boundaries[right].t - boundaries[left].t
            alpha = 0.5 if span <= 0 else (boundaries[index].t - boundaries[left].t) / span
            points[index] = (1.0 - alpha) * points[left] + alpha * points[right]  # type: ignore[operator]
            continue
        nearest = _nearest_candidate(candidates, boundaries[index].frame, max_gap=3)
        if nearest is None:
            raise ValueError(f"boundary {boundaries[index].anchor_id} has no world_xyz or nearby 2D candidate")
        z = cfg.ball_radius_m if boundaries[index].kind == "bounce" else 0.8
        points[index] = np.asarray(_intersect_pixel_z(calibration, nearest.xy, z), dtype=float)
    output = np.stack([point for point in points if point is not None])
    if output.shape != (len(boundaries), 3) or not np.all(np.isfinite(output)):
        raise ValueError("boundary initialization produced non-finite geometry")
    return output


def _nearest_candidate(
    candidates: Mapping[int, Sequence[_Candidate]], frame: int, *, max_gap: int
) -> _Candidate | None:
    rows = [
        (abs(candidate_frame - frame), item.rank, item)
        for candidate_frame, items in candidates.items()
        if abs(candidate_frame - frame) <= max_gap
        for item in items
    ]
    return min(rows, default=(0, 0, None), key=lambda row: (row[0], row[1]))[2]


def _camera(calibration: Mapping[str, Any]) -> dict[str, np.ndarray | float]:
    intrinsics = calibration.get("intrinsics")
    extrinsics = calibration.get("extrinsics")
    if not isinstance(intrinsics, Mapping) or not isinstance(extrinsics, Mapping):
        raise ValueError("calibration requires intrinsics and extrinsics")
    rotation = np.asarray(extrinsics["R"], dtype=float)
    translation = np.asarray(extrinsics["t"], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("calibration extrinsics have invalid shapes")
    return {
        "rotation": rotation,
        "translation": translation,
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
    }


def _project(camera: Mapping[str, Any], world_xyz: Sequence[float]) -> tuple[tuple[float, float], float]:
    point = np.asarray(world_xyz, dtype=float)
    camera_point = np.asarray(camera["rotation"]) @ point + np.asarray(camera["translation"])
    depth = float(camera_point[2])
    if not math.isfinite(depth) or depth <= 1.0e-6:
        raise ValueError("world point is behind the camera")
    return (
        (
            float(camera["fx"]) * float(camera_point[0]) / depth + float(camera["cx"]),
            float(camera["fy"]) * float(camera_point[1]) / depth + float(camera["cy"]),
        ),
        depth,
    )


def _intersect_pixel_z(
    calibration: Mapping[str, Any], xy: tuple[float, float], z: float
) -> tuple[float, float, float]:
    camera = _camera(calibration)
    camera_ray = np.asarray(
        [
            (xy[0] - float(camera["cx"])) / float(camera["fx"]),
            (xy[1] - float(camera["cy"])) / float(camera["fy"]),
            1.0,
        ],
        dtype=float,
    )
    rotation = np.asarray(camera["rotation"])
    origin = rotation.T @ (-np.asarray(camera["translation"]))
    direction = rotation.T @ camera_ray
    if abs(float(direction[2])) <= 1.0e-9:
        raise ValueError("candidate ray is parallel to the requested z plane")
    scale = (z - float(origin[2])) / float(direction[2])
    point = origin + scale * direction
    return float(point[0]), float(point[1]), float(z)


def _trajectory_point(
    flat_points: Sequence[float], boundaries: Sequence[_Boundary], frame: int, gravity: float
) -> tuple[np.ndarray, int, float]:
    points = np.asarray(flat_points, dtype=float).reshape((-1, 3))
    segment_index = _segment_index(boundaries, frame)
    left, right = boundaries[segment_index], boundaries[segment_index + 1]
    duration = right.t - left.t
    if duration <= 0:
        raise ValueError("event boundary times must be strictly increasing")
    frame_span = right.frame - left.frame
    if frame_span <= 0:
        raise ValueError("event boundary frames must be strictly increasing")
    alpha = (frame - left.frame) / frame_span
    dt = alpha * duration
    gravity_vector = np.asarray([0.0, 0.0, -gravity], dtype=float)
    point = (
        (1.0 - alpha) * points[segment_index]
        + alpha * points[segment_index + 1]
        + 0.5 * gravity_vector * (dt * dt - alpha * duration * duration)
    )
    return point, segment_index, alpha


def _segment_index(boundaries: Sequence[_Boundary], frame: int) -> int:
    if frame <= boundaries[0].frame:
        return 0
    for index in range(len(boundaries) - 1):
        if boundaries[index].frame <= frame <= boundaries[index + 1].frame:
            return index
    return len(boundaries) - 2


def _select_membership(
    flat_points: Sequence[float],
    boundaries: Sequence[_Boundary],
    candidates: Mapping[int, Sequence[_Candidate]],
    camera: Mapping[str, Any],
    cfg: GlobalTrackConfig,
    *,
    rally_start: int,
    rally_end: int,
) -> dict[int, _Candidate]:
    selected: dict[int, _Candidate] = {}
    fit_start = max(rally_start, boundaries[0].frame)
    fit_end = min(rally_end, boundaries[-1].frame)
    for frame, items in candidates.items():
        if frame < fit_start or frame > fit_end:
            continue
        try:
            world, _, _ = _trajectory_point(flat_points, boundaries, frame, cfg.gravity_mps2)
            projected, _ = _project(camera, world)
        except ValueError:
            continue
        ranked = sorted(
            ((math.dist(projected, item.xy), item.rank, item.source, item) for item in items),
            key=lambda row: (row[0], row[1], row[2]),
        )
        if ranked and ranked[0][0] <= cfg.membership_gate_px:
            selected[frame] = ranked[0][3]
    return selected


def _membership_identity(value: Mapping[int, _Candidate]) -> tuple[tuple[int, tuple[float, float], str], ...]:
    return tuple((frame, item.xy, item.source) for frame, item in sorted(value.items()))


def _fit_shared_boundaries(
    full_initial: np.ndarray,
    free_indexes: np.ndarray,
    fixed: Mapping[int, float],
    boundaries: Sequence[_Boundary],
    membership: Mapping[int, _Candidate],
    radii: Mapping[int, Sequence[_RadiusObservation]],
    radius_calibration: Mapping[str, Any],
    camera: Mapping[str, Any],
    cfg: GlobalTrackConfig,
) -> tuple[Any, np.ndarray]:
    def expand(free: Sequence[float]) -> np.ndarray:
        full = np.asarray(full_initial, dtype=float).copy()
        full[free_indexes] = free
        for index, value in fixed.items():
            full[index] = value
        return full

    def residuals(free: Sequence[float]) -> np.ndarray:
        full = expand(free)
        rows: list[float] = []
        for frame, candidate in sorted(membership.items()):
            try:
                world, _, _ = _trajectory_point(full, boundaries, frame, cfg.gravity_mps2)
                projected, depth = _project(camera, world)
            except ValueError:
                rows.extend([1.0e3, 1.0e3])
                continue
            rows.extend(
                [
                    (projected[0] - candidate.xy[0]) / cfg.robust_pixel_sigma_px,
                    (projected[1] - candidate.xy[1]) / cfg.robust_pixel_sigma_px,
                ]
            )
            if radius_calibration.get("status") == "active":
                observation = _matching_radius(radii.get(frame, ()), candidate.xy, cfg)
                if observation is not None:
                    predicted_log_radius = float(radius_calibration["intercept"]) + float(
                        radius_calibration["slope"]
                    ) * math.log(depth)
                    rows.append(
                        cfg.radius_residual_weight
                        * (math.log(observation.radius_px) - predicted_log_radius)
                        / cfg.radius_log_sigma
                    )
        points = full.reshape((-1, 3))
        for index, boundary in enumerate(boundaries):
            if boundary.world_xyz is None:
                continue
            anchor = np.asarray(boundary.world_xyz, dtype=float)
            rows.extend(((points[index] - anchor) / boundary.sigma_m).tolist())
        return np.asarray(rows, dtype=float)

    result = least_squares(
        residuals,
        np.asarray(full_initial, dtype=float)[free_indexes],
        loss="huber",
        f_scale=1.0,
        max_nfev=cfg.max_nfev,
    )
    return result, expand(result.x)


def _radius_observations(*payloads: Mapping[str, Any] | None) -> dict[int, tuple[_RadiusObservation, ...]]:
    output: dict[int, list[_RadiusObservation]] = {}
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        frames = payload.get("frames") or payload.get("observations") or payload.get("samples")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            continue
        for default_frame, item in enumerate(frames):
            if not isinstance(item, Mapping):
                continue
            frame = _int_or_none(item.get("frame") if item.get("frame") is not None else item.get("frame_index"))
            if frame is None:
                frame = default_frame
            blobs = item.get("blobs")
            if isinstance(blobs, Sequence) and not isinstance(blobs, (str, bytes)):
                rows = [blob for blob in blobs if isinstance(blob, Mapping)]
            else:
                rows = [item]
            for row in rows:
                xy = _xy(row.get("center_xy_px") or row.get("xy"))
                radius = _positive_float(row.get("radius_proxy_px") or row.get("radius_px") or row.get("apparent_radius_px"))
                confidence = _finite_float(
                    row.get("radius_confidence")
                    or row.get("heatmap_peak")
                    or item.get("radius_confidence")
                    or item.get("heatmap_peak")
                )
                if xy is None or radius is None or confidence is None:
                    continue
                output.setdefault(frame, []).append(
                    _RadiusObservation(frame=frame, xy=xy, radius_px=radius, confidence=confidence)
                )
    return {frame: tuple(items) for frame, items in output.items()}


def _matching_radius(
    observations: Sequence[_RadiusObservation], candidate_xy: tuple[float, float], cfg: GlobalTrackConfig
) -> _RadiusObservation | None:
    eligible = [
        (math.dist(item.xy, candidate_xy), item)
        for item in observations
        if item.confidence >= cfg.radius_confidence_min
    ]
    if not eligible:
        return None
    distance, item = min(eligible, key=lambda row: row[0])
    return item if distance <= cfg.radius_match_px else None


def _calibrate_radius(
    flat_points: Sequence[float],
    boundaries: Sequence[_Boundary],
    membership: Mapping[int, _Candidate],
    radii: Mapping[int, Sequence[_RadiusObservation]],
    camera: Mapping[str, Any],
    cfg: GlobalTrackConfig,
) -> dict[str, Any]:
    rows: list[tuple[float, float]] = []
    for frame, candidate in membership.items():
        observation = _matching_radius(radii.get(frame, ()), candidate.xy, cfg)
        if observation is None:
            continue
        try:
            world, _, _ = _trajectory_point(flat_points, boundaries, frame, cfg.gravity_mps2)
            _, depth = _project(camera, world)
        except ValueError:
            continue
        rows.append((math.log(depth), math.log(observation.radius_px)))
    if len(rows) < cfg.radius_min_observations:
        return _radius_abstention("insufficient_confident_observations", len(rows))
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    if float(np.ptp(x)) <= 1.0e-6 or float(np.ptp(y)) <= 1.0e-6:
        return _radius_abstention("insufficient_depth_or_radius_variation", len(rows))
    fit = least_squares(
        lambda params: y - (params[0] + params[1] * x),
        np.asarray([float(np.median(y)), -0.7], dtype=float),
        bounds=([-math.inf, -1.5], [math.inf, -0.1]),
        loss="huber",
        f_scale=cfg.radius_log_sigma,
        max_nfev=200,
    )
    prediction = fit.x[0] + fit.x[1] * x
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_total = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_total if ss_total > 1.0e-12 else 0.0
    if not fit.success or r2 < cfg.radius_min_r2:
        result = _radius_abstention("per_rally_calibration_below_r2_gate", len(rows))
        result.update({"r2": r2, "slope": float(fit.x[1])})
        return result
    return {
        "status": "active",
        "observation_count": len(rows),
        "confidence_floor": cfg.radius_confidence_min,
        "intercept": float(fit.x[0]),
        "slope": float(fit.x[1]),
        "r2": r2,
        "per_rally_calibration": True,
        "universal_linear_law_assumed": False,
    }


def _radius_abstention(reason: str, count: int) -> dict[str, Any]:
    return {
        "status": "abstained",
        "reason": reason,
        "observation_count": count,
        "per_rally_calibration": True,
        "universal_linear_law_assumed": False,
    }


def _endpoint_covariance(result: Any, free_indexes: np.ndarray, size: int, cfg: GlobalTrackConfig) -> np.ndarray:
    covariance = np.zeros((size, size), dtype=float)
    jacobian = np.asarray(getattr(result, "jac", np.empty((0, free_indexes.size))), dtype=float)
    residual = np.asarray(getattr(result, "fun", np.empty((0,))), dtype=float)
    if jacobian.ndim == 2 and jacobian.shape[1] == free_indexes.size and jacobian.size:
        dof = max(1, jacobian.shape[0] - jacobian.shape[1])
        scale = float(np.dot(residual, residual) / dof) if residual.size else 1.0
        free_covariance = np.linalg.pinv(jacobian.T @ jacobian, rcond=1.0e-10) * max(scale, 1.0e-6)
        covariance[np.ix_(free_indexes, free_indexes)] = free_covariance
    for index in range(size):
        covariance[index, index] = max(covariance[index, index], cfg.covariance_floor_m2)
    return covariance


def _segment_diagnostics(
    flat_points: Sequence[float],
    boundaries: Sequence[_Boundary],
    membership: Mapping[int, _Candidate],
    camera: Mapping[str, Any],
    cfg: GlobalTrackConfig,
) -> list[dict[str, Any]]:
    points = np.asarray(flat_points, dtype=float).reshape((-1, 3))
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        members = [item for frame, item in membership.items() if left.frame <= frame <= right.frame]
        residuals: list[float] = []
        for item in members:
            world, _, _ = _trajectory_point(flat_points, boundaries, item.frame, cfg.gravity_mps2)
            try:
                projected, _ = _project(camera, world)
            except ValueError:
                residuals.append(math.inf)
            else:
                residuals.append(math.dist(projected, item.xy))
        duration = right.t - left.t
        initial_velocity = _segment_initial_velocity(points[index], points[index + 1], duration, cfg.gravity_mps2)
        terminal_velocity = initial_velocity + np.asarray([0.0, 0.0, -cfg.gravity_mps2]) * duration
        speeds = [float(np.linalg.norm(initial_velocity)), float(np.linalg.norm(terminal_velocity))]
        reasons: list[str] = []
        if len(members) < cfg.min_members_per_segment:
            reasons.append("insufficient_candidate_members")
        if duration <= 0:
            reasons.append("non_positive_duration")
        if max(speeds) > cfg.max_speed_mps + 1.0e-9:
            reasons.append("speed_ceiling_exceeded")
        if residuals and max(residuals) > cfg.membership_gate_px + 1.0e-6:
            reasons.append("membership_gate_inconsistency")
        rows.append(
            {
                "segment_id": index,
                "frame_start": left.frame,
                "frame_end": right.frame,
                "start_event_kind": left.kind,
                "end_event_kind": right.kind,
                "status": "fit" if not reasons else "fallback",
                "fallback_reasons": reasons,
                "member_count": len(members),
                "initial_position_m": points[index].tolist(),
                "initial_velocity_mps": initial_velocity.tolist(),
                "initial_speed_mps": speeds[0],
                "terminal_speed_mps": speeds[1],
                "reprojection_error_px": _distribution(residuals),
                "physics_model": "gravity_only",
                "two_ended_shared_boundaries": True,
            }
        )
    return rows


def _segment_initial_velocity(p0: np.ndarray, p1: np.ndarray, duration: float, gravity: float) -> np.ndarray:
    if duration <= 0:
        return np.asarray([math.inf, math.inf, math.inf], dtype=float)
    acceleration = np.asarray([0.0, 0.0, -gravity], dtype=float)
    return (p1 - p0 - 0.5 * acceleration * duration * duration) / duration


def _segment_speed_violations(
    points: np.ndarray, boundaries: Sequence[_Boundary], cfg: GlobalTrackConfig
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        duration = right.t - left.t
        initial = _segment_initial_velocity(points[index], points[index + 1], duration, cfg.gravity_mps2)
        terminal = initial + np.asarray([0.0, 0.0, -cfg.gravity_mps2]) * duration
        for endpoint, velocity in (("start", initial), ("end", terminal)):
            speed = float(np.linalg.norm(velocity))
            if speed > cfg.max_speed_mps + 1.0e-9:
                output.append({"segment_id": index, "endpoint": endpoint, "speed_mps": speed})
    return output


def _height_violations(
    points: np.ndarray,
    boundaries: Sequence[_Boundary],
    start: int,
    end: int,
    fps: float,
    cfg: GlobalTrackConfig,
) -> list[dict[str, Any]]:
    del fps
    output: list[dict[str, Any]] = []
    for frame in range(start, end + 1):
        point, segment, _ = _trajectory_point(points.reshape(-1), boundaries, frame, cfg.gravity_mps2)
        if float(point[2]) < cfg.min_height_m or float(point[2]) > cfg.max_height_m:
            output.append({"frame": frame, "segment_id": segment, "z_m": float(point[2])})
    return output


def _missing_runs(member_frames: set[int], start: int, end: int) -> list[dict[str, int]]:
    output: list[dict[str, int]] = []
    run_start: int | None = None
    for frame in range(start, end + 2):
        missing = frame <= end and frame not in member_frames
        if missing and run_start is None:
            run_start = frame
        if not missing and run_start is not None:
            output.append(
                {"frame_start": run_start, "frame_end": frame - 1, "length_frames": frame - run_start}
            )
            run_start = None
    return output


def _emit_samples(
    flat_points: Sequence[float],
    endpoint_covariance: np.ndarray,
    boundaries: Sequence[_Boundary],
    membership: Mapping[int, _Candidate],
    start: int,
    end: int,
    fps: float,
    camera: Mapping[str, Any],
    cfg: GlobalTrackConfig,
) -> list[dict[str, Any]]:
    member_posteriors: dict[int, float] = {}
    for frame, candidate in membership.items():
        world, _, _ = _trajectory_point(flat_points, boundaries, frame, cfg.gravity_mps2)
        projected, _ = _project(camera, world)
        residual = math.dist(projected, candidate.xy)
        member_posteriors[frame] = max(1.0e-6, min(1.0, math.exp(-0.5 * (residual / (0.5 * cfg.membership_gate_px)) ** 2)))
    samples: list[dict[str, Any]] = []
    member_frame_list = sorted(membership)
    for frame in range(start, end + 1):
        world, segment, alpha = _trajectory_point(flat_points, boundaries, frame, cfg.gravity_mps2)
        projected, _ = _project(camera, world)
        candidate = membership.get(frame)
        band = "measured-candidate" if candidate is not None else "physics_predicted"
        covariance = _position_covariance(endpoint_covariance, segment, alpha, cfg)
        if candidate is None:
            nearest_distance = min(abs(frame - member_frame) for member_frame in member_frame_list)
            nearest_posterior = max(
                posterior
                for member_frame, posterior in member_posteriors.items()
                if abs(member_frame - frame) == nearest_distance
            )
            posterior = max(1.0e-6, nearest_posterior * math.exp(-nearest_distance / max(1, cfg.max_hole_frames)))
            covariance = covariance + np.eye(3) * cfg.predicted_process_variance_m2_per_frame2 * nearest_distance**2
        else:
            posterior = member_posteriors[frame]
        samples.append(
            {
                "frame": frame,
                "t": frame / fps,
                "segment_id": segment,
                "world_xyz": world.tolist(),
                "projected_xy": list(projected),
                "candidate_xy": list(candidate.xy) if candidate is not None else None,
                "candidate_score": candidate.score if candidate is not None else None,
                "candidate_source": candidate.source if candidate is not None else None,
                "xy": list(candidate.xy) if candidate is not None else None,
                "conf": posterior,
                "visible": candidate is not None,
                "approx": candidate is None,
                "band": band,
                "provenance_band": band,
                "posterior": posterior,
                "covariance_position_m2": covariance.tolist(),
                "measurement_authority": False,
                "measured": False,
                "authority": "candidate_preview",
                "render_only": True,
                "arc_solver": {
                    "segment_id": segment,
                    "segment_status": "fit",
                    "gravity_only": True,
                    "global_track_candidate": True,
                },
            }
        )
    return samples


def _position_covariance(
    endpoint_covariance: np.ndarray, segment: int, alpha: float, cfg: GlobalTrackConfig
) -> np.ndarray:
    weights = np.zeros(endpoint_covariance.shape[0], dtype=float)
    weights[3 * segment : 3 * segment + 3] = 1.0 - alpha
    weights[3 * (segment + 1) : 3 * (segment + 1) + 3] = alpha
    transform = np.zeros((3, endpoint_covariance.shape[0]), dtype=float)
    for axis in range(3):
        transform[axis, :] = weights
        # Each output axis depends only on the same endpoint axis.
        transform[axis, [idx for idx in range(len(weights)) if idx % 3 != axis]] = 0.0
    covariance = transform @ endpoint_covariance @ transform.T
    covariance = 0.5 * (covariance + covariance.T)
    for axis in range(3):
        covariance[axis, axis] = max(covariance[axis, axis], cfg.covariance_floor_m2)
    return covariance


def _sample_contract_audit(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    mislabeled = 0
    for sample in samples:
        band = sample.get("band")
        if band not in _BANDS or sample.get("provenance_band") != band:
            violations.append(f"frame {sample.get('frame')}: invalid provenance band")
        if sample.get("measured") is not False or sample.get("measurement_authority") is not False:
            mislabeled += 1
            violations.append(f"frame {sample.get('frame')}: candidate posterior mislabeled measured")
        posterior = _finite_float(sample.get("posterior"))
        if posterior is None or not 0.0 < posterior <= 1.0:
            violations.append(f"frame {sample.get('frame')}: invalid posterior")
        covariance = np.asarray(sample.get("covariance_position_m2"), dtype=float)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            violations.append(f"frame {sample.get('frame')}: invalid covariance shape/value")
        elif float(np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T)))) < -1.0e-9:
            violations.append(f"frame {sample.get('frame')}: covariance is not positive semidefinite")
    return {
        "passes": not violations,
        "sample_count": len(samples),
        "posterior_mislabeled_measured_count": mislabeled,
        "violations": violations,
    }


def _physics_reintegration(
    samples: Sequence[Mapping[str, Any]], boundaries: Sequence[_Boundary], cfg: GlobalTrackConfig
) -> dict[str, Any]:
    by_segment: dict[int, list[Mapping[str, Any]]] = {}
    for sample in samples:
        by_segment.setdefault(int(sample["segment_id"]), []).append(sample)
    rows: list[dict[str, Any]] = []
    for segment, items in sorted(by_segment.items()):
        items.sort(key=lambda item: int(item["frame"]))
        if len(items) < 2:
            rows.append({"segment_id": segment, "sample_count": len(items), "passes": True, "max_error_m": 0.0})
            continue
        left = boundaries[segment]
        duration = boundaries[segment + 1].t - left.t
        start = np.asarray(items[0]["world_xyz"], dtype=float)
        first_alpha = (int(items[0]["frame"]) - left.frame) / (boundaries[segment + 1].frame - left.frame)
        first_t = first_alpha * duration
        points = np.asarray([item["world_xyz"] for item in items], dtype=float)
        times = np.asarray(
            [
                ((int(item["frame"]) - left.frame) / (boundaries[segment + 1].frame - left.frame)) * duration
                for item in items
            ],
            dtype=float,
        )
        if len(items) >= 2 and times[1] > times[0]:
            velocity = (points[1] - points[0]) / (times[1] - times[0]) - np.asarray(
                [0.0, 0.0, -0.5 * cfg.gravity_mps2 * (times[1] - times[0])]
            )
        else:
            velocity = np.zeros(3)
        acceleration = np.asarray([0.0, 0.0, -cfg.gravity_mps2])
        predicted = np.stack(
            [start + velocity * (time - first_t) + 0.5 * acceleration * (time - first_t) ** 2 for time in times]
        )
        errors = np.linalg.norm(predicted - points, axis=1)
        max_error = float(np.max(errors))
        rows.append(
            {
                "segment_id": segment,
                "sample_count": len(items),
                "max_error_m": max_error,
                "passes": max_error <= 1.0e-5,
            }
        )
    pass_count = sum(bool(item["passes"]) for item in rows)
    return {
        "segment_count": len(rows),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(rows) if rows else 0.0,
        "tolerance_m": 1.0e-5,
        "segments": rows,
    }


def _step_speed_summary(samples: Sequence[Mapping[str, Any]], cfg: GlobalTrackConfig) -> dict[str, Any]:
    speeds: list[float] = []
    violations = 0
    for left, right in zip(samples, samples[1:]):
        dt = float(right["t"]) - float(left["t"])
        if dt <= 0:
            continue
        speed = math.dist(left["world_xyz"], right["world_xyz"]) / dt
        speeds.append(speed)
        violations += speed > cfg.max_speed_mps + 1.0e-9
    return {**_distribution(speeds), "over_35_mps_count": violations}


def _bounce_radius_summary(
    flat_points: Sequence[float], boundaries: Sequence[_Boundary], cfg: GlobalTrackConfig
) -> dict[str, Any]:
    points = np.asarray(flat_points, dtype=float).reshape((-1, 3))
    errors = [abs(float(points[index, 2]) - cfg.ball_radius_m) for index, item in enumerate(boundaries) if item.kind == "bounce"]
    return {
        "bounce_count": len(errors),
        "consistent_count": sum(error <= 1.0e-9 for error in errors),
        "max_abs_z_error_m": max(errors, default=None),
        "radius_m": cfg.ball_radius_m,
    }


def _boundary_json(boundary: _Boundary, flat_points: Sequence[float], index: int) -> dict[str, Any]:
    points = np.asarray(flat_points, dtype=float).reshape((-1, 3))
    return {
        "frame": boundary.frame,
        "t": boundary.t,
        "kind": boundary.kind,
        "anchor_id": boundary.anchor_id,
        "source": boundary.source,
        "input_world_xyz": list(boundary.world_xyz) if boundary.world_xyz is not None else None,
        "fitted_world_xyz": points[index].tolist(),
        "shared_by_adjacent_segments": 0 < index < len(points) - 1,
        "measured_authority": False,
    }


def _candidate_residual(sample: Mapping[str, Any]) -> float:
    return math.dist(sample["projected_xy"], sample["candidate_xy"])


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "p50": _percentile(finite, 0.50),
        "p95": _percentile(finite, 0.95),
        "max": finite[-1],
    }


def _percentile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = q * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _optional_min(values: Sequence[float | None] | Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(finite) if finite else None


def _refusal(
    code: str, reasons: Sequence[str], cfg: GlobalTrackConfig, context: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "refused",
        "verified": False,
        "default_off": True,
        "unwired": True,
        "render_only": True,
        "not_for_detection_metrics": True,
        "clip_id": context.get("clip_id"),
        "fps": context.get("fps"),
        "config": asdict(cfg),
        "inputs": context.get("input_stats", {}),
        "serve_initialization": context.get("serve_initialization"),
        "quarantined_events": context.get("quarantined_events", []),
        "rally_bounds": context.get("rally_bounds"),
        "boundaries": context.get("boundaries", []),
        "segments": context.get("diagnostics", {}).get("segments", []),
        "frames": [],
        "summary": {
            "emitted_sample_count": 0,
            "posterior_mislabeled_measured_count": 0,
            "rally_level_fail_closed": True,
        },
        "diagnostics": context.get("diagnostics", {}),
        "refusal": {
            "type": "rally_track_refusal",
            "code": code,
            "reasons": list(reasons),
            "fail_closed": True,
            "samples_suppressed": True,
        },
    }


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    x, y = _finite_float(value[0]), _finite_float(value[1])
    return (x, y) if x is not None and y is not None else None


def _xyz(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    output = tuple(_finite_float(value[index]) for index in range(3))
    return output if all(value is not None for value in output) else None  # type: ignore[return-value]


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_float(value: Any) -> float | None:
    number = _finite_float(value)
    return number if number is not None and number > 0 else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


__all__ = ["ARTIFACT_TYPE", "BALL_RADIUS_M", "GlobalTrackConfig", "build_global_ball_track"]
