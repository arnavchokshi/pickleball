"""M6 3D/spin/speed/events gate report for BALL-only tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_decode import FrameSource, probe_clip
from .schemas import BallTrack, ContactWindows, validate_artifact_file


M6_STATUS_TESTED = "TESTED-ON-REAL-DATA"
M6_STATUS_SCAFFOLD = "SCAFFOLD"
MAX_WORLD_BALL_SPEED_MPS = 30.0
MAX_ARC_RESIDUAL_PX = 5.0
MAX_REVIEW_DELTA_FRAMES = 2.0
MAX_AUDIO_TIMING_MS = 40.0
MAX_CONTACT_FUSION_HALF_WINDOW_MS = 35.0
MIN_CONTACT_CUES = 2


def build_ball_3d_events_gate_report(
    *,
    ball_track_path: str | Path,
    video_path: str | Path | None = None,
    m4_bounce_report_path: str | Path | None = None,
    m5_inout_report_path: str | Path | None = None,
    physics_segments_path: str | Path | None = None,
    contact_windows_path: str | Path | None = None,
    reviewed_contacts_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed M6 report from already-produced BALL artifacts."""

    video = probe_clip(video_path) if video_path is not None else None
    track_path = Path(ball_track_path)
    track = validate_artifact_file("ball_track", track_path)
    if not isinstance(track, BallTrack):
        raise ValueError(f"{track_path} did not validate as BallTrack")

    m4_report = _load_optional_json(m4_bounce_report_path)
    m5_report = _load_optional_json(m5_inout_report_path)
    physics_payload = _load_optional_json(physics_segments_path)
    contact_windows = _load_optional_contact_windows(contact_windows_path)
    reviewed_contacts = _load_optional_json(reviewed_contacts_path)

    upstream_m4 = _gate_summary(
        m4_report,
        expected_artifact_type="racketsport_ball_bounce_gate_report",
        missing_violation="missing_m4_bounce_gate_report",
        invalid_violation="m4_bounce_gate_artifact_type_invalid",
        not_passed_violation="m4_bounce_gate_not_passed",
        ball_track_path=track_path,
        bounce_count=len(track.bounces),
        ball_track_missing_violation="m4_bounce_gate_ball_track_missing",
        ball_track_mismatch_violation="m4_bounce_gate_ball_track_mismatch",
        bounce_count_missing_violation="m4_bounce_gate_bounce_count_missing",
        bounce_count_mismatch_violation="m4_bounce_gate_bounce_count_mismatch",
        not_tested_violation="m4_bounce_gate_not_tested_on_real_data",
    )
    upstream_m5 = _gate_summary(
        m5_report,
        expected_artifact_type="racketsport_ball_inout_gate_report",
        missing_violation="missing_m5_inout_gate_report",
        invalid_violation="m5_inout_gate_artifact_type_invalid",
        not_passed_violation="m5_inout_gate_not_passed",
        ball_track_path=track_path,
        bounce_count=len(track.bounces),
        ball_track_missing_violation="m5_inout_gate_ball_track_missing",
        ball_track_mismatch_violation="m5_inout_gate_ball_track_mismatch",
        bounce_count_missing_violation="m5_inout_gate_bounce_count_missing",
        bounce_count_mismatch_violation="m5_inout_gate_bounce_count_mismatch",
        not_tested_violation="m5_inout_gate_not_tested_on_real_data",
    )
    trajectory_3d = _trajectory_3d_summary(track)
    spin_speed = _spin_speed_summary(track)
    physics_segments = _physics_segments_summary(physics_payload, ball_track_path=track_path)
    event_summary, contact_timing = _events_and_contact_timing_summary(
        contact_windows,
        reviewed=reviewed_contacts,
        fps=float(track.fps),
    )

    violations: list[str] = []
    _extend_unique(violations, upstream_m4["violations"])
    _extend_unique(violations, upstream_m5["violations"])
    _extend_unique(violations, trajectory_3d["violations"])
    _extend_unique(violations, spin_speed["violations"])
    _extend_unique(violations, physics_segments["violations"])
    _extend_unique(violations, event_summary["violations"])
    _extend_unique(violations, contact_timing["violations"])

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_3d_events_gate_report",
        "milestone": "M6 3D/spin/events",
        "status": M6_STATUS_TESTED if video is not None else M6_STATUS_SCAFFOLD,
        "gate_result": "fail" if violations else "pass",
        "blocked_reason": "ball_3d_events_gate_failed" if violations else None,
        "ball_track_path": str(track_path),
        "video": _video_summary(video) if video is not None else None,
        "m4_bounce_gate": upstream_m4["summary"],
        "m5_inout_gate": upstream_m5["summary"],
        "source": track.source,
        "fps": float(track.fps),
        "frame_count": len(track.frames),
        "required_thresholds": {
            "max_world_ball_speed_mps": MAX_WORLD_BALL_SPEED_MPS,
            "max_arc_residual_px": MAX_ARC_RESIDUAL_PX,
            "max_review_delta_frames": MAX_REVIEW_DELTA_FRAMES,
            "max_audio_timing_ms": MAX_AUDIO_TIMING_MS,
            "max_contact_fusion_half_window_ms": MAX_CONTACT_FUSION_HALF_WINDOW_MS,
            "min_contact_cues": MIN_CONTACT_CUES,
        },
        "trajectory_3d": trajectory_3d["summary"],
        "spin_speed": spin_speed["summary"],
        "physics_segments": physics_segments["summary"],
        "events": event_summary["summary"],
        "contact_timing": contact_timing["summary"],
        "violations": violations,
        "not_ground_truth": True,
    }


def write_ball_3d_events_gate_report(
    *,
    ball_track_path: str | Path,
    out: str | Path,
    video_path: str | Path | None = None,
    m4_bounce_report_path: str | Path | None = None,
    m5_inout_report_path: str | Path | None = None,
    physics_segments_path: str | Path | None = None,
    contact_windows_path: str | Path | None = None,
    reviewed_contacts_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_ball_3d_events_gate_report(
        ball_track_path=ball_track_path,
        video_path=video_path,
        m4_bounce_report_path=m4_bounce_report_path,
        m5_inout_report_path=m5_inout_report_path,
        physics_segments_path=physics_segments_path,
        contact_windows_path=contact_windows_path,
        reviewed_contacts_path=reviewed_contacts_path,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _gate_summary(
    report: Mapping[str, Any] | None,
    *,
    expected_artifact_type: str,
    missing_violation: str,
    invalid_violation: str,
    not_passed_violation: str,
    ball_track_path: Path,
    bounce_count: int,
    ball_track_missing_violation: str,
    ball_track_mismatch_violation: str,
    bounce_count_missing_violation: str,
    bounce_count_mismatch_violation: str,
    not_tested_violation: str,
) -> dict[str, Any]:
    if report is None:
        return {
            "summary": {"path_present": False, "gate_result": None, "blocked_reason": None, "status": None},
            "violations": [missing_violation],
        }
    violations: list[str] = []
    if report.get("artifact_type") != expected_artifact_type:
        violations.append(invalid_violation)
    if report.get("gate_result") != "pass":
        violations.append(not_passed_violation)
    upstream_ball_track_path = report.get("ball_track_path")
    if not isinstance(upstream_ball_track_path, str) or not upstream_ball_track_path:
        violations.append(ball_track_missing_violation)
    elif not _paths_match(upstream_ball_track_path, ball_track_path):
        violations.append(ball_track_mismatch_violation)
    upstream_bounce_count = report.get("bounce_count")
    if not isinstance(upstream_bounce_count, int):
        violations.append(bounce_count_missing_violation)
    elif upstream_bounce_count != bounce_count:
        violations.append(bounce_count_mismatch_violation)
    if report.get("status") != M6_STATUS_TESTED:
        violations.append(not_tested_violation)
    return {
        "summary": {
            "path_present": True,
            "artifact_type": report.get("artifact_type"),
            "gate_result": report.get("gate_result"),
            "blocked_reason": report.get("blocked_reason"),
            "status": report.get("status"),
            "ball_track_path": upstream_ball_track_path,
            "bounce_count": upstream_bounce_count,
        },
        "violations": sorted(set(violations)),
    }


def _trajectory_3d_summary(track: BallTrack) -> dict[str, Any]:
    visible_frames = [frame for frame in track.frames if frame.visible]
    world_xyz_count = sum(1 for frame in visible_frames if frame.world_xyz is not None)
    violations: list[str] = []
    if not track.frames:
        violations.append("ball_track_has_no_frames")
    if not visible_frames:
        violations.append("no_visible_ball_frames")
    if not track.bounces:
        violations.append("ball_track_has_no_bounces")
    if world_xyz_count == 0:
        violations.append("no_world_xyz_frames")
    elif world_xyz_count < len(visible_frames):
        violations.append("world_xyz_missing_for_visible_frames")
    return {
        "summary": {
            "visible_frame_count": len(visible_frames),
            "world_xyz_frame_count": world_xyz_count,
            "world_xyz_coverage": _ratio(world_xyz_count, len(visible_frames)),
            "bounce_count": len(track.bounces),
            "world_frame": "court_Z0_context",
            "meter_level_context_only": True,
        },
        "violations": violations,
    }


def _spin_speed_summary(track: BallTrack) -> dict[str, Any]:
    visible_frames = [frame for frame in track.frames if frame.visible]
    spin_values = [float(frame.spin_rpm) for frame in visible_frames if frame.spin_rpm is not None]
    speed_values = [float(frame.speed_mps) for frame in visible_frames if frame.speed_mps is not None]
    violations: list[str] = []
    if not spin_values:
        violations.append("no_spin_estimates")
    elif len(spin_values) < len(visible_frames):
        violations.append("spin_missing_for_visible_frames")
    if not speed_values:
        violations.append("no_speed_estimates")
    elif len(speed_values) < len(visible_frames):
        violations.append("speed_missing_for_visible_frames")
    max_speed_mps = max(speed_values, default=None)
    if max_speed_mps is not None and max_speed_mps > MAX_WORLD_BALL_SPEED_MPS:
        violations.append("world_ball_speed_over_30mps")
    return {
        "summary": {
            "spin_frame_count": len(spin_values),
            "frame_speed_count": len(speed_values),
            "max_abs_spin_rpm": max((abs(value) for value in spin_values), default=None),
            "max_speed_mps": max_speed_mps,
            "max_speed_mph": max_speed_mps * 2.2369362920544 if max_speed_mps is not None else None,
            "max_speed_kph": max_speed_mps * 3.6 if max_speed_mps is not None else None,
            "spin_low_confidence_context_only": True,
        },
        "violations": violations,
    }


def _physics_segments_summary(payload: Mapping[str, Any] | None, *, ball_track_path: Path) -> dict[str, Any]:
    if payload is None:
        return {
            "summary": {
                "path_present": False,
                "artifact_type": None,
                "model": None,
                "input_ball_track_path": None,
                "segment_count": 0,
                "max_residual_px": None,
                "shots": [],
            },
            "violations": ["missing_physics_segments"],
        }

    violations: list[str] = []
    artifact_type = payload.get("artifact_type")
    model = payload.get("model")
    input_ball_track_path = payload.get("input_ball_track_path")
    solver_command = payload.get("solver_command")
    if artifact_type != "racketsport_ball_physics_segments":
        violations.append("physics_segments_artifact_type_invalid")
    if not isinstance(model, str) or not all(token in model.lower() for token in ("gravity", "drag", "magnus")):
        violations.append("physics_model_not_gravity_drag_magnus")
    if not isinstance(input_ball_track_path, str) or not input_ball_track_path:
        violations.append("physics_segments_input_track_missing")
    elif not _paths_match(input_ball_track_path, ball_track_path):
        violations.append("physics_segments_input_track_mismatch")
    if not isinstance(solver_command, str) or not solver_command:
        violations.append("physics_segments_solver_command_missing")

    raw_segments = payload.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    if not segments:
        violations.append("no_physics_segments")

    shots: list[dict[str, Any]] = []
    residuals: list[float] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            violations.append("physics_segment_invalid")
            continue
        start_t = _finite_or_none(segment.get("start_t"))
        end_t = _finite_or_none(segment.get("end_t"))
        if start_t is None or end_t is None or end_t < start_t:
            violations.append("physics_segment_time_invalid")
        if segment.get("uses_drag") is not True:
            violations.append("physics_segment_missing_drag")
        if segment.get("uses_magnus") is not True:
            violations.append("physics_segment_missing_magnus")

        residual_px = _first_finite(
            segment.get("fit_residual_px"),
            segment.get("arc_residual_px"),
            segment.get("ransac_arc_residual_px"),
        )
        if residual_px is None:
            violations.append("physics_segment_residual_missing")
        else:
            residuals.append(residual_px)
            if residual_px > MAX_ARC_RESIDUAL_PX:
                violations.append("ransac_arc_residual_over_5px")

        constraints = segment.get("boundary_constraints")
        constraint_set = set(str(value) for value in constraints) if isinstance(constraints, list) else set()
        if not {"contact", "bounce_z0"}.issubset(constraint_set):
            violations.append("physics_segment_boundary_constraints_missing")

        peak_speed_mps = _finite_or_none(segment.get("peak_speed_mps"))
        avg_speed_mps = _finite_or_none(segment.get("avg_speed_mps"))
        peak_speed_mph = _finite_or_none(segment.get("peak_speed_mph"))
        avg_speed_mph = _finite_or_none(segment.get("avg_speed_mph"))
        if peak_speed_mps is None or avg_speed_mps is None or peak_speed_mph is None or avg_speed_mph is None:
            violations.append("shot_speed_missing")
        if peak_speed_mps is not None and peak_speed_mps > MAX_WORLD_BALL_SPEED_MPS:
            violations.append("shot_speed_over_30mps")
        if segment.get("spin_sign") not in {"negative", "positive", "zero"}:
            violations.append("spin_sign_missing")
        if _finite_or_none(segment.get("spin_rpm_estimate")) is None:
            violations.append("spin_rpm_estimate_missing")

        shots.append(
            {
                "start_t": start_t,
                "end_t": end_t,
                "peak_speed_mps": peak_speed_mps,
                "avg_speed_mps": avg_speed_mps,
                "peak_speed_mph": peak_speed_mph,
                "avg_speed_mph": avg_speed_mph,
                "fit_residual_px": residual_px,
                "uses_drag": segment.get("uses_drag"),
                "uses_magnus": segment.get("uses_magnus"),
                "boundary_constraints": sorted(constraint_set),
                "spin_sign": segment.get("spin_sign"),
                "spin_rpm_estimate": _finite_or_none(segment.get("spin_rpm_estimate")),
            }
        )

    return {
        "summary": {
            "path_present": True,
            "artifact_type": artifact_type,
            "model": model,
            "input_ball_track_path": input_ball_track_path,
            "solver_command": solver_command,
            "segment_count": len(segments),
            "max_residual_px": max(residuals, default=None),
            "shots": shots,
        },
        "violations": sorted(set(violations)),
    }


def _events_and_contact_timing_summary(
    contact_windows: ContactWindows | None,
    *,
    reviewed: Mapping[str, Any] | None,
    fps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    event_violations: list[str] = []
    contact_violations: list[str] = []
    if contact_windows is None:
        return (
            {
                "summary": {
                    "path_present": False,
                    "event_count": 0,
                    "contact_count": 0,
                    "bounce_count": 0,
                    "net_cross_count": 0,
                    "into_net_count": 0,
                },
                "violations": ["missing_contact_windows", "no_contact_events", "no_net_cross_events"],
            },
            {
                "summary": {
                    "reviewed_contact_count": 0,
                    "matched_contact_count": 0,
                    "missing_reviewed_contact_count": 0,
                    "extra_predicted_contact_count": 0,
                    "max_abs_delta_frames": None,
                    "max_abs_audio_delta_ms": None,
                    "matches": [],
                },
                "violations": ["missing_reviewed_contact_labels"],
            },
        )

    contacts = [event for event in contact_windows.events if event.type == "contact"]
    net_crosses = [event for event in contact_windows.events if event.type == "net_cross"]
    into_nets = [event for event in contact_windows.events if event.type == "into_net"]
    bounces = [event for event in contact_windows.events if event.type == "bounce"]

    if not contacts:
        event_violations.append("no_contact_events")
    if not net_crosses:
        event_violations.append("no_net_cross_events")

    for event in contacts:
        cue_count = _contact_cue_count(event)
        if cue_count < MIN_CONTACT_CUES:
            event_violations.append("contact_event_has_fewer_than_two_cues")
        left_ms = (float(event.t) - float(event.window.t0)) * 1000.0
        right_ms = (float(event.window.t1) - float(event.t)) * 1000.0
        if left_ms < -1e-6 or right_ms < -1e-6:
            event_violations.append("contact_window_excludes_event")
        elif max(left_ms, right_ms) > MAX_CONTACT_FUSION_HALF_WINDOW_MS + 1e-6:
            event_violations.append("contact_fusion_window_over_35ms")

    predicted = [
        {
            "frame": int(event.frame),
            "t": float(event.t),
            "audio_present": event.sources.audio is not None and float(event.sources.audio) > 0.0,
            "cue_count": _contact_cue_count(event),
        }
        for event in contacts
    ]
    if reviewed is None:
        contact_violations.append("missing_reviewed_contact_labels")
        reviewed_contacts: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        extra = predicted
    else:
        reviewed_contacts = _reviewed_contacts(reviewed, fps=fps)
        if reviewed.get("artifact_type") != "racketsport_reviewed_ball_contacts":
            contact_violations.append("reviewed_contacts_artifact_type_invalid")
        if not reviewed_contacts:
            contact_violations.append("reviewed_contacts_empty")
        matches, missing, extra = _match_contacts_by_frame(
            predicted,
            reviewed_contacts,
            max_delta_frames=MAX_REVIEW_DELTA_FRAMES,
        )
        if missing:
            contact_violations.append("reviewed_contacts_missing_predictions")
        if extra:
            contact_violations.append("predicted_contacts_extra")
        for match in matches:
            if match["audio_present"] and abs(float(match["signed_delta_ms"])) > MAX_AUDIO_TIMING_MS:
                contact_violations.append("contact_audio_timing_over_40ms")

    max_abs_delta_frames = max((abs(float(match["signed_delta_frames"])) for match in matches), default=None)
    max_abs_audio_delta_ms = max(
        (abs(float(match["signed_delta_ms"])) for match in matches if match["audio_present"]),
        default=None,
    )
    return (
        {
            "summary": {
                "path_present": True,
                "event_count": len(contact_windows.events),
                "contact_count": len(contacts),
                "bounce_count": len(bounces),
                "net_cross_count": len(net_crosses),
                "into_net_count": len(into_nets),
            },
            "violations": sorted(set(event_violations)),
        },
        {
            "summary": {
                "reviewed_contact_count": len(reviewed_contacts),
                "matched_contact_count": len(matches),
                "missing_reviewed_contact_count": len(missing),
                "extra_predicted_contact_count": len(extra),
                "max_abs_delta_frames": max_abs_delta_frames,
                "max_abs_audio_delta_ms": max_abs_audio_delta_ms,
                "matches": matches,
            },
            "violations": sorted(set(contact_violations)),
        },
    )


def _contact_cue_count(event: Any) -> int:
    return sum(
        1
        for value in (
            event.sources.audio,
            event.sources.wrist_vel,
            event.sources.ball_inflection,
        )
        if value is not None and float(value) > 0.0
    )


def _match_contacts_by_frame(
    predicted: Sequence[dict[str, Any]],
    reviewed: Sequence[dict[str, Any]],
    *,
    max_delta_frames: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_pairs: list[tuple[float, int, int, float]] = []
    for pred_idx, pred in enumerate(predicted):
        for review_idx, truth in enumerate(reviewed):
            signed_delta = float(pred["frame"]) - float(truth["frame"])
            abs_delta = abs(signed_delta)
            if abs_delta <= max_delta_frames + 1e-9:
                candidate_pairs.append((abs_delta, pred_idx, review_idx, signed_delta))

    used_predicted: set[int] = set()
    used_reviewed: set[int] = set()
    matches: list[dict[str, Any]] = []
    for _, pred_idx, review_idx, signed_delta_frames in sorted(candidate_pairs):
        if pred_idx in used_predicted or review_idx in used_reviewed:
            continue
        used_predicted.add(pred_idx)
        used_reviewed.add(review_idx)
        pred = predicted[pred_idx]
        truth = reviewed[review_idx]
        signed_delta_ms = (float(pred["t"]) - float(truth["t"])) * 1000.0
        matches.append(
            {
                "predicted_frame": pred["frame"],
                "reviewed_frame": truth["frame"],
                "signed_delta_frames": signed_delta_frames,
                "predicted_t": pred["t"],
                "reviewed_t": truth["t"],
                "signed_delta_ms": signed_delta_ms,
                "audio_present": pred["audio_present"],
                "cue_count": pred["cue_count"],
            }
        )

    missing = [truth for idx, truth in enumerate(reviewed) if idx not in used_reviewed]
    extra = [pred for idx, pred in enumerate(predicted) if idx not in used_predicted]
    return matches, missing, extra


def _reviewed_contacts(payload: Mapping[str, Any], *, fps: float) -> list[dict[str, Any]]:
    contacts = payload.get("contacts")
    if not isinstance(contacts, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in contacts:
        if not isinstance(item, Mapping):
            continue
        frame = item.get("frame")
        t = _finite_or_none(item.get("t"))
        frame_index = int(frame) if isinstance(frame, int) else round(float(t) * fps) if t is not None else None
        if frame_index is None or frame_index < 0:
            continue
        parsed.append({"frame": frame_index, "t": float(t) if t is not None else frame_index / fps})
    return parsed


def _load_optional_contact_windows(path: str | Path | None) -> ContactWindows | None:
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.is_file():
        return None
    artifact = validate_artifact_file("contact_windows", json_path)
    if not isinstance(artifact, ContactWindows):
        raise ValueError(f"{json_path} did not validate as ContactWindows")
    return artifact


def _load_optional_json(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.is_file():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{json_path} must contain a JSON object")
    return payload


def _video_summary(video: FrameSource) -> dict[str, Any]:
    return {
        "path": str(video.path),
        "resolution": [int(video.width), int(video.height)],
        "fps": float(video.fps),
        "duration_s": float(video.duration_s),
        "frame_count": video.frame_count,
        "audio_present": video.audio_sample_rate is not None,
        "audio_sample_rate": video.audio_sample_rate,
    }


def _first_finite(*values: Any) -> float | None:
    for value in values:
        numeric = _finite_or_none(value)
        if numeric is not None:
            return numeric
    return None


def _finite_like(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _finite_or_none(value: Any) -> float | None:
    return float(value) if _finite_like(value) else None


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _paths_match(left: str, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


__all__ = [
    "M6_STATUS_SCAFFOLD",
    "M6_STATUS_TESTED",
    "build_ball_3d_events_gate_report",
    "write_ball_3d_events_gate_report",
]
