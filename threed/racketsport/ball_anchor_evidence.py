"""Offline, review-only ball anchor evidence fusion.

This module ranks bounce/contact *candidates*.  It deliberately has no solver
integration and never consumes reviewed event timestamps.  Reviewed timestamps
belong in a separate evaluation harness so candidate generation cannot tune to
ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_arc_solver import BALL_RADIUS_M, intersect_ray_z, pixel_ray_world
from .contact_provenance import audio_provenance
from .io_decode import time_for_frame


ARTIFACT_TYPE = "racketsport_ball_anchor_evidence_candidates"
SOURCE_AUDIO = "audio_onset"
SOURCE_KINEMATICS = "trajectory_kinematics"
SOURCE_BLUR = "frame_difference_blur_transition"
SOURCE_COURT = "court_plane_ray_proximity"


@dataclass(frozen=True)
class AnchorEvidenceConfig:
    """Frozen, label-free thresholds for offline candidate generation."""

    cluster_delta_s: float = 0.075
    source_min_separation_s: float = 0.100
    min_vertical_speed_px_s: float = 30.0
    min_direction_break_deg: float = 35.0
    max_track_step_px: float = 180.0
    raw_candidate_link_radius_px: float = 70.0
    raw_candidate_max_gap_frames: int = 4
    accepted_visibility_threshold: float = 0.5
    court_margin_m: float = 2.0
    blur_min_relative_change: float = 0.20
    blur_min_angle_change_deg: float = 20.0
    max_candidates_per_rally: int = 64

    def __post_init__(self) -> None:
        positive = {
            "cluster_delta_s": self.cluster_delta_s,
            "source_min_separation_s": self.source_min_separation_s,
            "min_vertical_speed_px_s": self.min_vertical_speed_px_s,
            "min_direction_break_deg": self.min_direction_break_deg,
            "max_track_step_px": self.max_track_step_px,
            "raw_candidate_link_radius_px": self.raw_candidate_link_radius_px,
            "court_margin_m": self.court_margin_m,
            "max_candidates_per_rally": float(self.max_candidates_per_rally),
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.raw_candidate_max_gap_frames < 1:
            raise ValueError("raw_candidate_max_gap_frames must be positive")
        if not 0.0 <= self.accepted_visibility_threshold <= 1.0:
            raise ValueError("accepted_visibility_threshold must be in [0, 1]")
        if not 0.0 <= self.blur_min_relative_change <= 1.0:
            raise ValueError("blur_min_relative_change must be in [0, 1]")
        if not 0.0 <= self.blur_min_angle_change_deg <= 90.0:
            raise ValueError("blur_min_angle_change_deg must be in [0, 90]")


def build_anchor_evidence_payload(
    *,
    ball_track: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
    audio_onsets: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    raw_ball_candidates: Mapping[str, Any] | None = None,
    blur_sidecar: Mapping[str, Any] | None = None,
    rally_spans: Sequence[Mapping[str, Any]] | None = None,
    clip_id: str = "",
    config: AnchorEvidenceConfig | None = None,
    audio_source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a ranked, typed candidate artifact from independent cue payloads.

    The output is candidate evidence only.  It contains explicit review-only
    and not-solver-wired policy fields and refuses when no usable cue exists.
    """

    cfg = config or AnchorEvidenceConfig()
    track = ball_track if isinstance(ball_track, Mapping) else {}
    fps = _positive_float(track.get("fps")) or _positive_float(
        raw_ball_candidates.get("fps") if isinstance(raw_ball_candidates, Mapping) else None
    ) or 30.0
    frame_count = _frame_count(track, raw_ball_candidates)
    rallies = _normalize_rallies(rally_spans, frame_count=frame_count, fps=fps)

    cues: list[dict[str, Any]] = []
    cues.extend(_audio_cues(audio_onsets, fps=fps, source_path=audio_source_path))
    cues.extend(
        _kinematic_cues(
            track,
            raw_ball_candidates if isinstance(raw_ball_candidates, Mapping) else {},
            fps=fps,
            config=cfg,
        )
    )
    cues.extend(_blur_cues(blur_sidecar, ball_track=track, fps=fps, config=cfg))

    bounds = _court_bounds(calibration) if isinstance(calibration, Mapping) else None
    if isinstance(calibration, Mapping):
        for cue in cues:
            xy = _xy(cue.get("xy"))
            if xy is None:
                continue
            court = _court_plane_cue(
                cue=cue,
                xy=xy,
                calibration=calibration,
                bounds=bounds,
                margin_m=cfg.court_margin_m,
            )
            if court is not None:
                cue.setdefault("position_hypotheses", []).append(court["position_hypothesis"])
                cue.setdefault("support", []).append(court["source"])

    ranked = _fuse_and_rank(cues, rallies=rallies, fps=fps, config=cfg)
    source_summary = _source_summary(cues, ranked)
    status = "ranked_candidates" if ranked else "refused_no_evidence"
    refusal_reasons = [] if ranked else _refusal_reasons(
        track=track,
        raw_ball_candidates=raw_ball_candidates,
        audio_onsets=audio_onsets,
        blur_sidecar=blur_sidecar,
        calibration=calibration,
    )
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "clip_id": str(clip_id),
        "status": status,
        "verified": False,
        "not_ground_truth": True,
        "candidate_prediction": True,
        "review_only": True,
        "solver_wired": False,
        "policy": {
            "candidate_evidence_only": True,
            "reviewed_timestamps_consumed": False,
            "pb_events_consumed": False,
            "solver_wiring": "forbidden_in_this_artifact",
            "notes": [
                "Ranks hypotheses from immutable cue artifacts; it does not alter raw observations.",
                "Court-plane positions are ray/plane hypotheses, not measured 3D ball locations.",
                "Audio uses corrected_time_s when present while preserving raw timing in provenance.",
            ],
        },
        "fps": fps,
        "frame_count": frame_count,
        "rallies": rallies,
        "config": {
            name: getattr(cfg, name)
            for name in AnchorEvidenceConfig.__dataclass_fields__
        },
        "source_summary": source_summary,
        "refusal_reasons": refusal_reasons,
        "candidate_count": len(ranked),
        "candidates": ranked,
    }


def _audio_cues(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    fps: float,
    source_path: str | Path | None,
) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        raw_items = payload.get("onsets", payload.get("audio_onsets", payload.get("items", [])))
        provenance_payload: Any = payload
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        raw_items = payload
        provenance_payload = {"onsets": list(payload)}
    else:
        raw_items = []
        provenance_payload = {}
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return []
    source = Path(source_path) if source_path is not None else None
    spine_provenance = audio_provenance(provenance_payload, source_path=source)
    normalized: list[tuple[int, Mapping[str, Any], float, float | None, float]] = []
    for input_index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        corrected = _nonnegative_float(item.get("corrected_time_s"))
        raw = _nonnegative_float(item.get("raw_time_s", item.get("time_s", item.get("t"))))
        time_s = corrected if corrected is not None else raw
        if time_s is None:
            frame = _int_or_none(item.get("frame", item.get("frame_index")))
            if frame is None or frame < 0:
                continue
            time_s = frame / fps
        confidence = _confidence(item.get("confidence", item.get("score", item.get("conf", 0.0))))
        if confidence <= 0.0:
            continue
        normalized.append((input_index, item, time_s, raw, confidence))
    normalized.sort(
        key=lambda value: (
            float(value[2]),
            _int_or_none(value[1].get("corrected_order", value[1].get("onset_order"))) or 0,
            value[0],
        )
    )
    cues: list[dict[str, Any]] = []
    for corrected_order, (input_index, item, time_s, raw, confidence) in enumerate(normalized):
        corrected = _nonnegative_float(item.get("corrected_time_s"))
        classified_type = str(
            item.get("event_type", item.get("class_label", item.get("onset_class", "pop_transient")))
        ).lower()
        if classified_type in {"bounce", "ball_bounce"}:
            type_scores = {"bounce": confidence, "contact": 0.25 * confidence}
            cue_type = "classified_audio_bounce_onset"
        else:
            type_scores = {"contact": confidence, "bounce": 0.25 * confidence}
            cue_type = (
                "classified_audio_contact_onset"
                if classified_type in {"contact", "shot", "ball_contact", "pop"}
                else "heuristic_audio_pop_onset"
            )
        cues.append(
            {
                "cue_id": f"audio_{corrected_order:04d}",
                "source_type": SOURCE_AUDIO,
                "cue_type": cue_type,
                "t": time_s,
                "frame": int(round(time_s * fps)),
                "confidence": confidence,
                "type_scores": type_scores,
                "source": {
                    "source_type": SOURCE_AUDIO,
                    "cue_type": cue_type,
                    "confidence": confidence,
                    "raw_time_s": raw,
                    "corrected_time_s": corrected if corrected is not None else time_s,
                    "timing_used": "corrected_time_s" if corrected is not None else "raw_time_s",
                    "input_order": input_index,
                    "raw_order": _int_or_none(item.get("raw_order")),
                    "corrected_order": corrected_order,
                    "artifact_corrected_order": _int_or_none(
                        item.get("corrected_order", item.get("onset_order"))
                    ),
                    "classification": classified_type,
                    "timing_provenance": (
                        dict(item["timing_provenance"])
                        if isinstance(item.get("timing_provenance"), Mapping)
                        else None
                    ),
                    "spine_audio_provenance": spine_provenance,
                },
                "support": [],
                "position_hypotheses": [],
            }
        )
    return cues


def _kinematic_cues(
    track: Mapping[str, Any],
    raw_candidates: Mapping[str, Any],
    *,
    fps: float,
    config: AnchorEvidenceConfig,
) -> list[dict[str, Any]]:
    samples = _trajectory_samples(track, raw_candidates, fps=fps, config=config)
    if len(samples) < 3:
        return []
    raw_cues: list[dict[str, Any]] = []
    for previous, current, following in zip(samples, samples[1:], samples[2:]):
        before_dt = float(current["t"]) - float(previous["t"])
        after_dt = float(following["t"]) - float(current["t"])
        if before_dt <= 1e-9 or after_dt <= 1e-9:
            continue
        if int(current["frame"]) - int(previous["frame"]) > config.raw_candidate_max_gap_frames:
            continue
        if int(following["frame"]) - int(current["frame"]) > config.raw_candidate_max_gap_frames:
            continue
        before = (
            (current["xy"][0] - previous["xy"][0]) / before_dt,
            (current["xy"][1] - previous["xy"][1]) / before_dt,
        )
        after = (
            (following["xy"][0] - current["xy"][0]) / after_dt,
            (following["xy"][1] - current["xy"][1]) / after_dt,
        )
        before_speed = math.hypot(*before)
        after_speed = math.hypot(*after)
        if before_speed > config.max_track_step_px * fps or after_speed > config.max_track_step_px * fps:
            continue
        base_confidence = min(
            _confidence(previous.get("confidence")),
            _confidence(current.get("confidence")),
            _confidence(following.get("confidence")),
        )
        raw_support = any(not bool(item.get("accepted")) for item in (previous, current, following))
        provenance = {
            "source_type": SOURCE_KINEMATICS,
            "trajectory_inputs": [previous["provenance"], current["provenance"], following["provenance"]],
            "raw_candidate_support": raw_support,
            "accepted_visibility_threshold": config.accepted_visibility_threshold,
            "below_accepted_confidence_support": any(
                float(item.get("confidence", 0.0)) < config.accepted_visibility_threshold
                for item in (previous, current, following)
            ),
        }
        vertical_flip_strength = min(abs(before[1]), abs(after[1])) / max(
            config.min_vertical_speed_px_s, 1e-9
        )
        if before[1] >= config.min_vertical_speed_px_s and after[1] <= -config.min_vertical_speed_px_s:
            confidence = min(1.0, base_confidence * min(1.0, vertical_flip_strength / 3.0 + 0.35))
            raw_cues.append(
                _kinematic_cue(
                    current,
                    cue_type="vertical_velocity_sign_flip",
                    confidence=confidence,
                    type_scores={"bounce": confidence, "contact": 0.35 * confidence},
                    source={
                        **provenance,
                        "cue_type": "vertical_velocity_sign_flip",
                        "velocity_before_px_s": [round(before[0], 6), round(before[1], 6)],
                        "velocity_after_px_s": [round(after[0], 6), round(after[1], 6)],
                    },
                )
            )
        turn = _direction_break_deg(before, after)
        if turn >= config.min_direction_break_deg:
            turn_strength = min(1.0, (turn - config.min_direction_break_deg) / 90.0 + 0.25)
            confidence = min(1.0, base_confidence * turn_strength)
            raw_cues.append(
                _kinematic_cue(
                    current,
                    cue_type="direction_break",
                    confidence=confidence,
                    type_scores={"contact": confidence, "bounce": 0.45 * confidence},
                    source={
                        **provenance,
                        "cue_type": "direction_break",
                        "direction_break_deg": round(turn, 6),
                        "velocity_before_px_s": [round(before[0], 6), round(before[1], 6)],
                        "velocity_after_px_s": [round(after[0], 6), round(after[1], 6)],
                    },
                )
            )
    return _suppress_nearby_source_cues(raw_cues, min_separation_s=config.source_min_separation_s)


def _trajectory_samples(
    track: Mapping[str, Any],
    raw_candidates: Mapping[str, Any],
    *,
    fps: float,
    config: AnchorEvidenceConfig,
) -> list[dict[str, Any]]:
    frames = track.get("frames")
    accepted: dict[int, dict[str, Any]] = {}
    if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping) or frame.get("visible") is not True:
                continue
            xy = _xy(frame.get("xy"))
            confidence = _confidence(frame.get("conf", frame.get("confidence", 0.0)))
            if xy is None or confidence < config.accepted_visibility_threshold:
                continue
            t = _nonnegative_float(frame.get("t"))
            accepted[frame_index] = {
                "frame": frame_index,
                "t": t if t is not None else frame_index / fps,
                "xy": xy,
                "confidence": confidence,
                "accepted": True,
                "provenance": {
                    "artifact": "ball_track",
                    "frame": frame_index,
                    "accepted": True,
                    "confidence": confidence,
                },
            }

    raw_by_frame = _raw_candidates_by_frame(raw_candidates)
    samples = dict(accepted)
    accepted_indices = sorted(accepted)
    for frame_index, candidates in sorted(raw_by_frame.items()):
        if frame_index in samples or not candidates:
            continue
        expectation = _interpolated_track_xy(
            frame_index,
            accepted,
            accepted_indices,
            max_gap_frames=config.raw_candidate_max_gap_frames,
        )
        if expectation is None:
            continue
        ranked = sorted(
            candidates,
            key=lambda item: (_distance2(item["xy"], expectation), -float(item["confidence"])),
        )
        chosen = ranked[0]
        distance = math.sqrt(_distance2(chosen["xy"], expectation))
        if distance > config.raw_candidate_link_radius_px:
            continue
        confidence = _confidence(chosen["confidence"])
        samples[frame_index] = {
            "frame": frame_index,
            "t": float(chosen["t"]) if chosen.get("t") is not None else frame_index / fps,
            "xy": chosen["xy"],
            "confidence": confidence,
            "accepted": False,
            "provenance": {
                "artifact": "ball_candidates",
                "frame": frame_index,
                "accepted": False,
                "confidence": confidence,
                "source_detector": chosen.get("source_detector"),
                "pts_seconds": chosen.get("t"),
                "source_artifact_type": chosen.get("artifact_type"),
                "distance_to_interpolated_primary_px": round(distance, 6),
                "below_accepted_confidence": confidence < config.accepted_visibility_threshold,
                "below_primary_selection": True,
            },
        }
    return [samples[index] for index in sorted(samples)]


def _blur_cues(
    payload: Mapping[str, Any] | None,
    *,
    ball_track: Mapping[str, Any],
    fps: float,
    config: AnchorEvidenceConfig,
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    records = payload.get("frames")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    normalized: list[dict[str, Any]] = []
    track_times = _track_times_by_frame(ball_track, fps=fps)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        frame = _int_or_none(record.get("frame_index", record.get("frame")))
        length = _nonnegative_float(record.get("blur_length_px"))
        angle = _nonnegative_float(record.get("blur_angle_deg"))
        xy = _xy(record.get("center_xy", record.get("track_xy")))
        if frame is None or frame < 0 or length is None or xy is None:
            continue
        normalized.append(
            {
                "frame": frame,
                "t": track_times.get(frame, frame / fps),
                "length": length,
                "angle": angle,
                "xy": xy,
                "quality": str(record.get("quality", "unknown")),
                "velocity_angle": _nonnegative_float(record.get("velocity_angle_deg")),
                "angle_delta_to_velocity": _nonnegative_float(
                    record.get("angle_delta_to_track_velocity_deg")
                ),
                "record": record,
            }
        )
    normalized.sort(key=lambda item: item["frame"])
    cues: list[dict[str, Any]] = []
    for index, current in enumerate(normalized):
        previous = normalized[index - 1] if index > 0 else None
        following = normalized[index + 1] if index + 1 < len(normalized) else None
        if previous is None:
            continue
        if int(current["frame"]) - int(previous["frame"]) > 2:
            continue
        if following is not None and int(following["frame"]) - int(current["frame"]) > 2:
            following = None
        comparison = following if following is not None else current
        scale = max(float(previous["length"]), float(comparison["length"]), 1.0)
        signed_length_change = float(comparison["length"]) - float(previous["length"])
        relative_change = abs(signed_length_change) / scale
        blur_angle_change = _angle_delta_180(previous.get("angle"), comparison.get("angle"))
        center_direction_change = None
        vertical_velocity_sign_flip = False
        velocity_before: tuple[float, float] | None = None
        velocity_after: tuple[float, float] | None = None
        if following is not None:
            before_dt = max(float(current["t"]) - float(previous["t"]), 1e-9)
            after_dt = max(float(following["t"]) - float(current["t"]), 1e-9)
            velocity_before = (
                (float(current["xy"][0]) - float(previous["xy"][0])) / before_dt,
                (float(current["xy"][1]) - float(previous["xy"][1])) / before_dt,
            )
            velocity_after = (
                (float(following["xy"][0]) - float(current["xy"][0])) / after_dt,
                (float(following["xy"][1]) - float(current["xy"][1])) / after_dt,
            )
            center_direction_change = _direction_break_deg(velocity_before, velocity_after)
            vertical_velocity_sign_flip = (
                velocity_before[1] >= config.min_vertical_speed_px_s
                and velocity_after[1] <= -config.min_vertical_speed_px_s
            )
        transition_angle = max(
            value
            for value in (blur_angle_change, center_direction_change, 0.0)
            if value is not None
        )
        if relative_change < config.blur_min_relative_change and (
            transition_angle < config.blur_min_angle_change_deg
        ):
            continue
        quality_factor = 1.0 if current["quality"] == "clear" else 0.65
        strength = max(
            relative_change / max(config.blur_min_relative_change, 1e-9),
            transition_angle / max(config.blur_min_angle_change_deg, 1e-9),
        )
        confidence = min(1.0, quality_factor * (0.3 + 0.35 * min(strength, 2.0)))
        proposal_type = "bounce" if vertical_velocity_sign_flip else "shot"
        cue_type = f"motion_blur_{proposal_type}_transition"
        type_scores = (
            {"bounce": confidence, "contact": 0.25 * confidence}
            if proposal_type == "bounce"
            else {"contact": confidence, "bounce": 0.30 * confidence}
        )
        cues.append(
            {
                "cue_id": f"blur_{int(current['frame']):06d}",
                "source_type": SOURCE_BLUR,
                "cue_type": cue_type,
                "t": float(current["t"]),
                "frame": int(current["frame"]),
                "xy": list(current["xy"]),
                "confidence": confidence,
                "type_scores": type_scores,
                "source": {
                    "source_type": SOURCE_BLUR,
                    "cue_type": cue_type,
                    "proposal_type": proposal_type,
                    "confidence": confidence,
                    "frame": int(current["frame"]),
                    "relative_length_change": round(relative_change, 6),
                    "signed_length_change_px": round(signed_length_change, 6),
                    "blur_angle_change_deg": blur_angle_change,
                    "center_direction_change_deg": (
                        None if center_direction_change is None else round(center_direction_change, 6)
                    ),
                    "vertical_velocity_sign_flip": vertical_velocity_sign_flip,
                    "center_velocity_before_px_s": (
                        None if velocity_before is None else [round(value, 6) for value in velocity_before]
                    ),
                    "center_velocity_after_px_s": (
                        None if velocity_after is None else [round(value, 6) for value in velocity_after]
                    ),
                    "signature_frames": [
                        int(item["frame"])
                        for item in (previous, current, following)
                        if item is not None
                    ],
                    "quality": current["quality"],
                    "artifact_type": payload.get("artifact_type"),
                    "extraction": "frame_difference_ball_crop_principal_axis",
                },
                "support": [],
                "position_hypotheses": [
                    {
                        "space": "image_px",
                        "xy": [round(float(current["xy"][0]), 6), round(float(current["xy"][1]), 6)],
                        "source_type": SOURCE_BLUR,
                    }
                ],
            }
        )
    return _suppress_nearby_source_cues(cues, min_separation_s=config.source_min_separation_s)


def _track_times_by_frame(track: Mapping[str, Any], *, fps: float) -> dict[int, float]:
    frames = track.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return {}
    output: dict[int, float] = {}
    for frame_index, item in enumerate(frames):
        if not isinstance(item, Mapping):
            continue
        time_s = _nonnegative_float(item.get("t"))
        output[frame_index] = time_s if time_s is not None else frame_index / fps
    return output


def _court_plane_cue(
    *,
    cue: Mapping[str, Any],
    xy: tuple[float, float],
    calibration: Mapping[str, Any],
    bounds: tuple[float, float, float, float] | None,
    margin_m: float,
) -> dict[str, Any] | None:
    try:
        origin, direction = pixel_ray_world(calibration, xy)
        xyz = intersect_ray_z(origin, direction, BALL_RADIUS_M)
    except (KeyError, TypeError, ValueError):
        return None
    in_bounds = _in_court_bounds(xyz, bounds, margin_m=margin_m)
    if not in_bounds:
        return None
    confidence = min(1.0, 0.45 + 0.55 * _confidence(cue.get("confidence")))
    return {
        "source": {
            "source_type": SOURCE_COURT,
            "cue_type": "ray_intersection_ball_radius_plane",
            "confidence": confidence,
            "in_extended_court_bounds": True,
            "ball_radius_m": BALL_RADIUS_M,
            "semantics": "bounce_position_hypothesis_not_measured_depth",
        },
        "position_hypothesis": {
            "space": "court_world_m",
            "xyz": [round(float(value), 6) for value in xyz],
            "source_type": SOURCE_COURT,
            "semantics": "ray_intersection_with_ball_radius_plane",
        },
    }


def _fuse_and_rank(
    cues: Sequence[Mapping[str, Any]],
    *,
    rallies: Sequence[Mapping[str, Any]],
    fps: float,
    config: AnchorEvidenceConfig,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for rally in rallies:
        start_t = float(rally["start_t"])
        end_t = float(rally["end_t"])
        relevant = [dict(cue) for cue in cues if start_t <= float(cue["t"]) <= end_t]
        relevant.sort(key=lambda item: (float(item["t"]), str(item.get("source_type")), str(item.get("cue_id"))))
        clusters: list[list[dict[str, Any]]] = []
        for cue in relevant:
            if not clusters:
                clusters.append([cue])
                continue
            center = _weighted_time(clusters[-1])
            if float(cue["t"]) - center <= config.cluster_delta_s:
                clusters[-1].append(cue)
            else:
                clusters.append([cue])
        rally_candidates = [
            _candidate_from_cluster(cluster, rally_id=str(rally["rally_id"]), fps=fps)
            for cluster in clusters
        ]
        rally_candidates.sort(key=lambda item: (-float(item["confidence"]), float(item["t"]), item["anchor_type"]))
        for rank, candidate in enumerate(rally_candidates[: config.max_candidates_per_rally], start=1):
            candidate["rank_in_rally"] = rank
            candidate["anchor_id"] = f"{rally['rally_id']}_anchor_{rank:03d}"
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: (str(item["rally_id"]), int(item["rank_in_rally"])))


def _candidate_from_cluster(cluster: Sequence[Mapping[str, Any]], *, rally_id: str, fps: float) -> dict[str, Any]:
    source_weights = {
        SOURCE_AUDIO: 0.50,
        SOURCE_KINEMATICS: 0.55,
        SOURCE_BLUR: 0.35,
        SOURCE_COURT: 0.25,
    }
    time_s = _weighted_time(cluster)
    all_sources: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    type_evidence: dict[str, float] = {"bounce": 0.0, "contact": 0.0}
    source_best: dict[str, float] = {}
    for cue in cluster:
        source_type = str(cue["source_type"])
        source_confidence = _confidence(cue.get("confidence"))
        source_best[source_type] = max(source_best.get(source_type, 0.0), source_confidence)
        source = cue.get("source")
        if isinstance(source, Mapping):
            all_sources.append(dict(source))
        support = cue.get("support")
        if isinstance(support, Sequence) and not isinstance(support, (str, bytes)):
            for item in support:
                if not isinstance(item, Mapping):
                    continue
                item_source = str(item.get("source_type", ""))
                item_confidence = _confidence(item.get("confidence"))
                source_best[item_source] = max(source_best.get(item_source, 0.0), item_confidence)
                all_sources.append(dict(item))
        raw_positions = cue.get("position_hypotheses")
        if isinstance(raw_positions, Sequence) and not isinstance(raw_positions, (str, bytes)):
            positions.extend(dict(item) for item in raw_positions if isinstance(item, Mapping))
        raw_type_scores = cue.get("type_scores")
        if isinstance(raw_type_scores, Mapping):
            for anchor_type in type_evidence:
                value = _confidence(raw_type_scores.get(anchor_type))
                type_evidence[anchor_type] = 1.0 - (1.0 - type_evidence[anchor_type]) * (
                    1.0 - source_weights.get(source_type, 0.2) * value
                )

    confidence = 0.0
    for source_type, value in source_best.items():
        confidence = 1.0 - (1.0 - confidence) * (1.0 - source_weights.get(source_type, 0.2) * value)
    unique_source_types = sorted(source_type for source_type in source_best if source_type)
    confidence = min(0.99, confidence + 0.06 * max(0, len(unique_source_types) - 1))
    anchor_type = max(type_evidence, key=lambda name: (type_evidence[name], name == "contact"))
    typed = [
        {"type": name, "confidence": round(score, 6)}
        for name, score in sorted(type_evidence.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "rally_id": rally_id,
        "frame": int(round(time_s * fps)),
        "t": round(time_s, 9),
        "anchor_type": anchor_type,
        "type_hypotheses": typed,
        "confidence": round(confidence, 6),
        "source_types": unique_source_types,
        "sources": _dedupe_mappings(all_sources),
        "position_hypotheses": _dedupe_mappings(positions),
        "not_ground_truth": True,
        "candidate_prediction": True,
        "review_only": True,
        "solver_wired": False,
    }


def _source_summary(cues: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_types = (SOURCE_AUDIO, SOURCE_KINEMATICS, SOURCE_BLUR, SOURCE_COURT)
    rows: dict[str, Any] = {}
    for source_type in source_types:
        cue_count = sum(1 for cue in cues if cue.get("source_type") == source_type)
        if source_type == SOURCE_COURT:
            cue_count = sum(
                1
                for cue in cues
                for support in cue.get("support", [])
                if isinstance(support, Mapping) and support.get("source_type") == SOURCE_COURT
            )
        rows[source_type] = {
            "cue_count": cue_count,
            "ranked_candidate_count": sum(
                1 for candidate in candidates if source_type in candidate.get("source_types", [])
            ),
        }
    low_confidence_raw = 0
    raw_primary_rejected = 0
    for cue in cues:
        if cue.get("source_type") != SOURCE_KINEMATICS:
            continue
        source = cue.get("source")
        if not isinstance(source, Mapping):
            continue
        if source.get("below_accepted_confidence_support"):
            low_confidence_raw += 1
        if source.get("raw_candidate_support"):
            raw_primary_rejected += 1
    rows[SOURCE_KINEMATICS]["cues_with_raw_candidate_support"] = raw_primary_rejected
    rows[SOURCE_KINEMATICS]["cues_below_accepted_confidence_threshold"] = low_confidence_raw
    return rows


def _normalize_rallies(
    rally_spans: Sequence[Mapping[str, Any]] | None,
    *,
    frame_count: int,
    fps: float,
) -> list[dict[str, Any]]:
    if not rally_spans:
        end_frame = max(0, frame_count - 1)
        return [{"rally_id": "rally_000", "start_frame": 0, "end_frame": end_frame, "start_t": 0.0, "end_t": end_frame / fps}]
    normalized = []
    for index, raw in enumerate(rally_spans):
        if not isinstance(raw, Mapping):
            continue
        start_frame = _int_or_none(raw.get("start_frame"))
        end_frame = _int_or_none(raw.get("end_frame"))
        start_t = _nonnegative_float(raw.get("start_t"))
        end_t = _nonnegative_float(raw.get("end_t"))
        if start_t is None and start_frame is not None:
            start_t = start_frame / fps
        if end_t is None and end_frame is not None:
            end_t = end_frame / fps
        if start_t is None or end_t is None or end_t < start_t:
            continue
        if start_frame is None:
            start_frame = int(round(start_t * fps))
        if end_frame is None:
            end_frame = int(round(end_t * fps))
        normalized.append(
            {
                "rally_id": str(raw.get("rally_id", f"rally_{index:03d}")),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_t": start_t,
                "end_t": end_t,
            }
        )
    if not normalized:
        raise ValueError("rally_spans contains no valid spans")
    return normalized


def _raw_candidates_by_frame(payload: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    frames = payload.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return {}
    output: dict[int, list[dict[str, Any]]] = {}
    for raw_frame in frames:
        if not isinstance(raw_frame, Mapping):
            continue
        frame = _int_or_none(raw_frame.get("frame", raw_frame.get("frame_index")))
        pts_seconds = _nonnegative_float(
            raw_frame.get("pts_seconds", raw_frame.get("pts_s", raw_frame.get("time_s")))
        )
        candidates = raw_frame.get("candidates")
        if frame is None or not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            continue
        items = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            xy = _xy(candidate.get("xy"))
            if xy is None:
                continue
            items.append(
                {
                    "xy": xy,
                    "confidence": _confidence(candidate.get("score", candidate.get("confidence", 0.0))),
                    "source_detector": candidate.get("source_detector"),
                    "t": pts_seconds,
                    "artifact_type": payload.get("artifact_type"),
                }
            )
        output[frame] = items
    return output


def _interpolated_track_xy(
    frame: int,
    accepted: Mapping[int, Mapping[str, Any]],
    accepted_indices: Sequence[int],
    *,
    max_gap_frames: int,
) -> tuple[float, float] | None:
    left = max((index for index in accepted_indices if index < frame), default=None)
    right = min((index for index in accepted_indices if index > frame), default=None)
    if left is None or right is None or frame - left > max_gap_frames or right - frame > max_gap_frames:
        return None
    left_xy = accepted[left]["xy"]
    right_xy = accepted[right]["xy"]
    alpha = (frame - left) / (right - left)
    return (
        float(left_xy[0]) + alpha * (float(right_xy[0]) - float(left_xy[0])),
        float(left_xy[1]) + alpha * (float(right_xy[1]) - float(left_xy[1])),
    )


def _kinematic_cue(
    sample: Mapping[str, Any],
    *,
    cue_type: str,
    confidence: float,
    type_scores: Mapping[str, float],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    xy = sample["xy"]
    return {
        "cue_id": f"kin_{cue_type}_{int(sample['frame']):06d}",
        "source_type": SOURCE_KINEMATICS,
        "cue_type": cue_type,
        "t": float(sample["t"]),
        "frame": int(sample["frame"]),
        "xy": list(xy),
        "confidence": confidence,
        "type_scores": dict(type_scores),
        "source": {**dict(source), "confidence": confidence, "frame": int(sample["frame"])},
        "support": [],
        "position_hypotheses": [
            {
                "space": "image_px",
                "xy": [round(float(xy[0]), 6), round(float(xy[1]), 6)],
                "source_type": SOURCE_KINEMATICS,
            }
        ],
    }


def _suppress_nearby_source_cues(
    cues: Sequence[Mapping[str, Any]],
    *,
    min_separation_s: float,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(cue) for cue in cues),
        key=lambda cue: (-float(cue["confidence"]), float(cue["t"]), str(cue["cue_id"])),
    )
    kept: list[dict[str, Any]] = []
    for cue in ranked:
        if any(
            cue.get("source_type") == existing.get("source_type")
            and cue.get("cue_type") == existing.get("cue_type")
            and abs(float(cue["t"]) - float(existing["t"])) < min_separation_s
            for existing in kept
        ):
            continue
        kept.append(cue)
    return sorted(kept, key=lambda cue: (float(cue["t"]), str(cue["cue_id"])))


def _court_bounds(calibration: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    points = calibration.get("world_pts")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return None
    parsed = []
    for point in points:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            continue
        try:
            parsed.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None
    return (
        min(point[0] for point in parsed),
        max(point[0] for point in parsed),
        min(point[1] for point in parsed),
        max(point[1] for point in parsed),
    )


def _in_court_bounds(
    xyz: Sequence[float],
    bounds: tuple[float, float, float, float] | None,
    *,
    margin_m: float,
) -> bool:
    if bounds is None:
        return False
    x_min, x_max, y_min, y_max = bounds
    x, y = float(xyz[0]), float(xyz[1])
    return x_min - margin_m <= x <= x_max + margin_m and y_min - margin_m <= y <= y_max + margin_m


def _refusal_reasons(
    *,
    track: Mapping[str, Any],
    raw_ball_candidates: Mapping[str, Any] | None,
    audio_onsets: Any,
    blur_sidecar: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
) -> list[str]:
    reasons = []
    if not track.get("frames"):
        reasons.append("no_ball_track_frames")
    if not isinstance(raw_ball_candidates, Mapping) or not raw_ball_candidates.get("frames"):
        reasons.append("no_raw_ball_candidates")
    if not audio_onsets:
        reasons.append("no_audio_onsets")
    if not isinstance(blur_sidecar, Mapping) or not blur_sidecar.get("frames"):
        reasons.append("no_blur_records")
    if not isinstance(calibration, Mapping):
        reasons.append("no_calibration")
    if not reasons:
        reasons.append("all_sources_present_but_no_thresholded_evidence")
    return reasons


def _frame_count(track: Mapping[str, Any], raw_candidates: Mapping[str, Any] | None) -> int:
    frames = track.get("frames")
    if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
        return len(frames)
    if isinstance(raw_candidates, Mapping):
        raw_frames = raw_candidates.get("frames")
        if isinstance(raw_frames, Sequence) and not isinstance(raw_frames, (str, bytes)):
            indices = [
                _int_or_none(item.get("frame", item.get("frame_index")))
                for item in raw_frames
                if isinstance(item, Mapping)
            ]
            valid = [index for index in indices if index is not None and index >= 0]
            return max(valid) + 1 if valid else 0
    return 0


def _weighted_time(cues: Sequence[Mapping[str, Any]]) -> float:
    weights = [max(_confidence(cue.get("confidence")), 1e-6) for cue in cues]
    return sum(float(cue["t"]) * weight for cue, weight in zip(cues, weights)) / sum(weights)


def _direction_break_deg(before: Sequence[float], after: Sequence[float]) -> float:
    before_norm = math.hypot(float(before[0]), float(before[1]))
    after_norm = math.hypot(float(after[0]), float(after[1]))
    if before_norm <= 1e-9 or after_norm <= 1e-9:
        return 0.0
    cosine = (
        float(before[0]) * float(after[0]) + float(before[1]) * float(after[1])
    ) / (before_norm * after_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _angle_delta_180(left: Any, right: Any) -> float | None:
    left_value = _nonnegative_float(left)
    right_value = _nonnegative_float(right)
    if left_value is None or right_value is None:
        return None
    delta = abs((left_value - right_value) % 180.0)
    return round(min(delta, 180.0 - delta), 6)


def _dedupe_mappings(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        normalized = dict(item)
        key = repr(sorted(normalized.items(), key=lambda pair: pair[0]))
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def _distance2(left: Sequence[float], right: Sequence[float]) -> float:
    return (float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        output = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return output if all(math.isfinite(component) for component in output) else None


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _nonnegative_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ARTIFACT_TYPE",
    "AnchorEvidenceConfig",
    "SOURCE_AUDIO",
    "SOURCE_BLUR",
    "SOURCE_COURT",
    "SOURCE_KINEMATICS",
    "build_anchor_evidence_payload",
]
