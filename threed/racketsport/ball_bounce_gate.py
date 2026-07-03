"""M4 bounce gate report for BALL-only tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib

from .io_decode import FrameSource, probe_clip
from .schemas import BallTrack, validate_artifact_file


M4_STATUS_TESTED = "TESTED-ON-REAL-DATA"
M4_STATUS_SCAFFOLD = "SCAFFOLD"
BOUNCE_PROBABILITY_GATE = 0.50
MIN_BOUNCE_SEPARATION_S = 0.10
MAX_AUDIO_ALIGNMENT_MS = 40.0
MAX_REVIEW_DELTA_FRAMES = 2.0
ALLOWED_MODEL_FAMILIES = {"catboost", "gbm", "temporal_cnn"}


def build_ball_bounce_gate_report(
    *,
    ball_track_path: str | Path,
    video_path: str | Path | None = None,
    classifier_path: str | Path | None = None,
    audio_onsets_path: str | Path | None = None,
    reviewed_bounces_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed M4 report from real BALL artifacts."""

    video = probe_clip(video_path) if video_path is not None else None
    track_path = Path(ball_track_path)
    track = validate_artifact_file("ball_track", track_path)
    if not isinstance(track, BallTrack):
        raise ValueError(f"{track_path} did not validate as BallTrack")

    violations: list[str] = []

    classifier = _load_optional_json(classifier_path)
    audio = _load_optional_json(audio_onsets_path)
    reviewed = _load_optional_json(reviewed_bounces_path)

    _extend_unique(violations, _bounce_payload_violations(track))
    classifier_summary = _classifier_summary(
        classifier,
        classifier_path=Path(classifier_path) if classifier_path is not None else None,
        ball_track_path=track_path,
        track=track,
    )
    _extend_unique(violations, classifier_summary["violations"])
    audio_alignment = _audio_alignment_summary(track, audio=audio, video=video)
    _extend_unique(violations, audio_alignment["violations"])
    review_timing = _review_timing_summary(track, reviewed=reviewed)
    _extend_unique(violations, review_timing["violations"])

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_bounce_gate_report",
        "milestone": "M4 Bounce",
        "status": M4_STATUS_TESTED if video is not None else M4_STATUS_SCAFFOLD,
        "gate_result": "fail" if violations else "pass",
        "blocked_reason": "ball_bounce_gate_failed" if violations else None,
        "ball_track_path": str(track_path),
        "video": _video_summary(video) if video is not None else None,
        "source": track.source,
        "fps": float(track.fps),
        "frame_count": len(track.frames),
        "bounce_count": len(track.bounces),
        "required_thresholds": {
            "p_bounce_min": BOUNCE_PROBABILITY_GATE,
            "min_bounce_separation_s": MIN_BOUNCE_SEPARATION_S,
            "max_audio_alignment_ms": MAX_AUDIO_ALIGNMENT_MS,
            "max_review_delta_frames": MAX_REVIEW_DELTA_FRAMES,
        },
        "bounces": [_bounce_summary(bounce, fps=float(track.fps)) for bounce in track.bounces],
        "classifier": classifier_summary["summary"],
        "audio_alignment": audio_alignment["summary"],
        "review_timing": review_timing["summary"],
        "violations": violations,
        "not_ground_truth": True,
    }


def write_ball_bounce_gate_report(
    *,
    ball_track_path: str | Path,
    out: str | Path,
    video_path: str | Path | None = None,
    classifier_path: str | Path | None = None,
    audio_onsets_path: str | Path | None = None,
    reviewed_bounces_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_ball_bounce_gate_report(
        ball_track_path=ball_track_path,
        video_path=video_path,
        classifier_path=classifier_path,
        audio_onsets_path=audio_onsets_path,
        reviewed_bounces_path=reviewed_bounces_path,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _bounce_payload_violations(track: BallTrack) -> list[str]:
    violations: list[str] = []
    if not track.bounces:
        violations.append("ball_track_has_no_bounces")
        return violations

    times = sorted(float(bounce.t) for bounce in track.bounces)
    if any((right - left) < MIN_BOUNCE_SEPARATION_S for left, right in zip(times, times[1:], strict=False)):
        violations.append("bounce_separation_below_0_10s")

    for bounce in track.bounces:
        if bounce.frame is None:
            violations.append("bounce_frame_missing")
        if bounce.contact_xy_img is None:
            violations.append("bounce_contact_xy_img_missing")
        if bounce.p_bounce is None:
            violations.append("bounce_probability_missing")
        elif float(bounce.p_bounce) < BOUNCE_PROBABILITY_GATE:
            violations.append("bounce_probability_below_0_50")
        if not bounce.source:
            violations.append("bounce_source_missing")
    return sorted(set(violations))


def _classifier_summary(
    classifier: Mapping[str, Any] | None,
    *,
    classifier_path: Path | None,
    ball_track_path: Path,
    track: BallTrack,
) -> dict[str, Any]:
    violations: list[str] = []
    if classifier is None:
        return {
            "summary": {
                "path_present": False,
                "artifact_type": None,
                "model_family": None,
                "trained_on_real_labels": None,
                "probability_threshold": None,
                "accepted_bounce_count": 0,
                "model_path": None,
                "model_sha256": None,
                "model_sha256_verified": False,
                "feature_window_frames": None,
                "training_label_count": None,
                "validation_label_count": None,
                "input_ball_track_path": None,
            },
            "violations": ["missing_bounce_classifier_output"],
        }

    artifact_type = classifier.get("artifact_type")
    if artifact_type == "racketsport_ball_bounce_2d_output":
        return _bounce_2d_summary(
            classifier,
            classifier_path=classifier_path,
            ball_track_path=ball_track_path,
            track=track,
        )

    model_family = classifier.get("model_family")
    trained_on_real_labels = classifier.get("trained_on_real_labels")
    probability_threshold = _finite_or_none(classifier.get("probability_threshold"))
    accepted = classifier.get("accepted_bounces")
    accepted_bounces = accepted if isinstance(accepted, list) else []
    model_path_value = classifier.get("model_path")
    model_sha256 = classifier.get("model_sha256")
    feature_window_frames = _int_or_none(classifier.get("feature_window_frames"))
    training_label_count = _int_or_none(classifier.get("training_label_count"))
    validation_label_count = _int_or_none(classifier.get("validation_label_count"))
    training_command = classifier.get("training_command")
    inference_command = classifier.get("inference_command")
    input_ball_track_path = classifier.get("input_ball_track_path")
    resolved_model_path = _resolve_artifact_path(model_path_value, base=classifier_path.parent if classifier_path else None)
    model_sha256_verified = False

    if artifact_type != "racketsport_ball_bounce_classifier_output":
        violations.append("bounce_classifier_artifact_type_invalid")
    if model_family not in ALLOWED_MODEL_FAMILIES:
        violations.append("bounce_classifier_model_family_not_allowed")
    if trained_on_real_labels is not True:
        violations.append("bounce_classifier_not_trained_on_real_labels")
    if probability_threshold is None:
        violations.append("bounce_classifier_threshold_missing")
    elif probability_threshold < BOUNCE_PROBABILITY_GATE:
        violations.append("bounce_classifier_threshold_below_0_50")
    if not accepted_bounces:
        violations.append("bounce_classifier_has_no_accepted_bounces")
    if not isinstance(model_path_value, str) or not model_path_value:
        violations.append("bounce_classifier_model_path_missing")
    elif resolved_model_path is None or not resolved_model_path.is_file():
        violations.append("bounce_classifier_model_file_missing")
    if not isinstance(model_sha256, str) or not model_sha256:
        violations.append("bounce_classifier_model_sha256_missing")
    elif resolved_model_path is not None and resolved_model_path.is_file():
        actual_sha256 = _sha256_file(resolved_model_path)
        model_sha256_verified = actual_sha256 == model_sha256
        if not model_sha256_verified:
            violations.append("bounce_classifier_model_sha256_mismatch")
    if feature_window_frames is None:
        violations.append("bounce_classifier_feature_window_frames_missing")
    elif feature_window_frames != 20:
        violations.append("bounce_classifier_feature_window_frames_not_20")
    if training_label_count is None:
        violations.append("bounce_classifier_training_label_count_missing")
    elif training_label_count <= 0:
        violations.append("bounce_classifier_training_label_count_nonpositive")
    if validation_label_count is None:
        violations.append("bounce_classifier_validation_label_count_missing")
    elif validation_label_count <= 0:
        violations.append("bounce_classifier_validation_label_count_nonpositive")
    if not isinstance(training_command, str) or not training_command:
        violations.append("bounce_classifier_training_command_missing")
    if not isinstance(inference_command, str) or not inference_command:
        violations.append("bounce_classifier_inference_command_missing")
    if not isinstance(input_ball_track_path, str) or not input_ball_track_path:
        violations.append("bounce_classifier_input_track_missing")
    elif not _paths_match(input_ball_track_path, ball_track_path):
        violations.append("bounce_classifier_input_track_mismatch")
    for candidate in accepted_bounces:
        if not isinstance(candidate, Mapping):
            violations.append("bounce_classifier_candidate_invalid")
            continue
        p_bounce = _finite_or_none(candidate.get("p_bounce"))
        if p_bounce is None:
            violations.append("bounce_classifier_candidate_probability_missing")
        elif p_bounce < BOUNCE_PROBABILITY_GATE:
            violations.append("bounce_classifier_candidate_probability_below_0_50")

    return {
        "summary": {
            "path_present": True,
            "artifact_type": artifact_type,
            "model_family": model_family,
            "trained_on_real_labels": trained_on_real_labels,
            "probability_threshold": probability_threshold,
            "candidate_count": classifier.get("candidate_count"),
            "accepted_bounce_count": len(accepted_bounces),
            "model_path": str(resolved_model_path) if resolved_model_path is not None else model_path_value,
            "model_sha256": model_sha256,
            "model_sha256_verified": model_sha256_verified,
            "feature_window_frames": feature_window_frames,
            "training_label_count": training_label_count,
            "validation_label_count": validation_label_count,
            "input_ball_track_path": input_ball_track_path,
        },
        "violations": sorted(set(violations)),
    }


def _bounce_2d_summary(
    detector: Mapping[str, Any],
    *,
    classifier_path: Path | None,
    ball_track_path: Path,
    track: BallTrack,
) -> dict[str, Any]:
    violations: list[str] = []
    artifact_type = detector.get("artifact_type")
    status = detector.get("status")
    algorithm = detector.get("algorithm")
    probability_threshold = _finite_or_none(detector.get("probability_threshold"))
    accepted = detector.get("accepted_bounces")
    accepted_bounces = accepted if isinstance(accepted, list) else []
    input_ball_track_path = detector.get("input_ball_track_path")
    command = detector.get("command")
    court_corners_path = detector.get("court_corners_path")
    input_track_matches = isinstance(input_ball_track_path, str) and _paths_match(input_ball_track_path, ball_track_path)
    accepted_bounces_match_track = _accepted_2d_bounces_match_track(accepted_bounces, track)

    if artifact_type != "racketsport_ball_bounce_2d_output":
        violations.append("bounce_2d_artifact_type_invalid")
    if status != M4_STATUS_TESTED:
        violations.append("bounce_2d_not_tested_on_real_data")
    if algorithm != "image_velocity_inflection_court_plane_2d_v1":
        violations.append("bounce_2d_algorithm_not_allowed")
    if probability_threshold is None:
        violations.append("bounce_2d_threshold_missing")
    elif probability_threshold < BOUNCE_PROBABILITY_GATE:
        violations.append("bounce_2d_threshold_below_0_50")
    if not accepted_bounces:
        violations.append("bounce_2d_has_no_accepted_bounces")
    if not isinstance(input_ball_track_path, str) or not input_ball_track_path:
        violations.append("bounce_2d_input_track_missing")
    elif not input_track_matches and not accepted_bounces_match_track:
        violations.append("bounce_2d_input_track_mismatch")
    if not accepted_bounces_match_track:
        violations.append("bounce_2d_accepted_bounces_mismatch_ball_track")
    if not isinstance(command, str) or not command:
        violations.append("bounce_2d_command_missing")
    if not isinstance(court_corners_path, str) or not court_corners_path:
        violations.append("bounce_2d_court_corners_missing")

    for candidate in accepted_bounces:
        if not isinstance(candidate, Mapping):
            violations.append("bounce_2d_candidate_invalid")
            continue
        p_bounce = _finite_or_none(candidate.get("p_bounce"))
        if p_bounce is None:
            violations.append("bounce_2d_candidate_probability_missing")
        elif p_bounce < BOUNCE_PROBABILITY_GATE:
            violations.append("bounce_2d_candidate_probability_below_0_50")
        if candidate.get("source") != "image_velocity_inflection_court_plane_2d_v1":
            violations.append("bounce_2d_candidate_source_invalid")

    return {
        "summary": {
            "path_present": True,
            "artifact_type": artifact_type,
            "status": status,
            "algorithm": algorithm,
            "probability_threshold": probability_threshold,
            "candidate_count": detector.get("candidate_count"),
            "accepted_bounce_count": len(accepted_bounces),
            "input_ball_track_path": input_ball_track_path,
            "input_track_matches_ball_track": input_track_matches,
            "accepted_bounces_match_ball_track": accepted_bounces_match_track,
            "court_corners_path": court_corners_path,
            "command": command,
            "artifact_path": str(classifier_path) if classifier_path is not None else None,
        },
        "violations": sorted(set(violations)),
    }


def _accepted_2d_bounces_match_track(accepted_bounces: Sequence[Any], track: BallTrack) -> bool:
    if len(accepted_bounces) != len(track.bounces):
        return False
    accepted_sorted = sorted(accepted_bounces, key=_accepted_2d_sort_key)
    track_sorted = sorted(track.bounces, key=lambda bounce: (bounce.frame if bounce.frame is not None else -1, float(bounce.t)))
    return all(_accepted_2d_bounce_matches_track_bounce(candidate, bounce) for candidate, bounce in zip(accepted_sorted, track_sorted, strict=False))


def _accepted_2d_sort_key(candidate: Any) -> tuple[int, float]:
    if not isinstance(candidate, Mapping):
        return (-1, -1.0)
    frame = candidate.get("frame")
    t = _finite_or_none(candidate.get("t"))
    return (int(frame) if isinstance(frame, int) else -1, float(t) if t is not None else -1.0)


def _accepted_2d_bounce_matches_track_bounce(candidate: Any, bounce: Any) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    if candidate.get("source") != bounce.source:
        return False
    candidate_frame = candidate.get("frame")
    if not isinstance(candidate_frame, int) or bounce.frame != candidate_frame:
        return False
    candidate_t = _finite_or_none(candidate.get("t"))
    if candidate_t is None or not math.isclose(candidate_t, float(bounce.t), rel_tol=1e-6, abs_tol=1e-6):
        return False
    candidate_p_bounce = _finite_or_none(candidate.get("p_bounce"))
    if candidate_p_bounce is None or bounce.p_bounce is None:
        return False
    if not math.isclose(candidate_p_bounce, float(bounce.p_bounce), rel_tol=1e-6, abs_tol=1e-6):
        return False
    return _point_matches(candidate.get("world_xy"), bounce.world_xy) and _point_matches(candidate.get("contact_xy_img"), bounce.contact_xy_img)


def _point_matches(candidate: Any, value: Any) -> bool:
    if value is None or not isinstance(candidate, list | tuple) or len(candidate) != 2:
        return False
    return all(
        _finite_or_none(left) is not None and math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)
        for left, right in zip(candidate, value, strict=True)
    )


def _audio_alignment_summary(
    track: BallTrack,
    *,
    audio: Mapping[str, Any] | None,
    video: FrameSource | None,
) -> dict[str, Any]:
    violations: list[str] = []
    if video is not None and video.audio_sample_rate is None:
        violations.append("video_audio_missing")
    if audio is None:
        return {
            "summary": {
                "audio_present": video.audio_sample_rate is not None if video is not None else None,
                "audio_onset_count": 0,
                "matched_bounce_count": 0,
                "max_abs_delta_ms": None,
                "matches": [],
            },
            "violations": sorted(set([*violations, "missing_audio_onsets"])),
        }

    onsets = _audio_onsets(audio)
    if not onsets:
        violations.append("audio_onsets_empty")
    matches: list[dict[str, Any]] = []
    for bounce in track.bounces:
        if not onsets:
            continue
        best = min(onsets, key=lambda onset: abs(float(onset["time_s"]) - float(bounce.t)))
        delta_ms = (float(best["time_s"]) - float(bounce.t)) * 1000.0
        matches.append(
            {
                "bounce_t": float(bounce.t),
                "audio_time_s": float(best["time_s"]),
                "signed_delta_ms": delta_ms,
                "abs_delta_ms": abs(delta_ms),
                "score": best.get("score"),
            }
        )
        if abs(delta_ms) > MAX_AUDIO_ALIGNMENT_MS:
            violations.append("bounce_audio_alignment_over_40ms")
    max_abs_delta_ms = max((float(match["abs_delta_ms"]) for match in matches), default=None)
    return {
        "summary": {
            "audio_present": video.audio_sample_rate is not None if video is not None else None,
            "audio_onset_count": len(onsets),
            "matched_bounce_count": len(matches),
            "max_abs_delta_ms": max_abs_delta_ms,
            "matches": matches,
        },
        "violations": sorted(set(violations)),
    }


def _review_timing_summary(track: BallTrack, *, reviewed: Mapping[str, Any] | None) -> dict[str, Any]:
    if reviewed is None:
        return {
            "summary": {
                "reviewed_bounce_count": 0,
                "matched_bounce_count": 0,
                "missing_reviewed_bounce_count": 0,
                "extra_predicted_bounce_count": len(track.bounces),
                "precision": None,
                "recall": None,
                "max_abs_delta_frames": None,
                "matches": [],
            },
            "violations": ["missing_reviewed_bounce_labels"],
        }

    reviewed_bounces = _reviewed_bounces(reviewed, fps=float(track.fps))
    predicted = [_predicted_bounce(bounce, fps=float(track.fps)) for bounce in track.bounces]
    violations: list[str] = []
    if not reviewed_bounces:
        violations.append("reviewed_bounces_empty")
    matches, missing, extra = _match_by_frame(
        predicted,
        reviewed_bounces,
        max_delta_frames=MAX_REVIEW_DELTA_FRAMES,
    )
    if missing:
        violations.append("reviewed_bounces_missing_predictions")
    if extra:
        violations.append("predicted_bounces_extra")
    max_abs_delta_frames = max((abs(float(match["signed_delta_frames"])) for match in matches), default=None)
    return {
        "summary": {
            "reviewed_bounce_count": len(reviewed_bounces),
            "matched_bounce_count": len(matches),
            "missing_reviewed_bounce_count": len(missing),
            "extra_predicted_bounce_count": len(extra),
            "precision": _ratio(len(matches), len(predicted)),
            "recall": _ratio(len(matches), len(reviewed_bounces)),
            "max_abs_delta_frames": max_abs_delta_frames,
            "matches": matches,
            "missing_reviewed_bounces": missing,
            "extra_predicted_bounces": extra,
        },
        "violations": sorted(set(violations)),
    }


def _match_by_frame(
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
    for _, pred_idx, review_idx, signed_delta in sorted(candidate_pairs):
        if pred_idx in used_predicted or review_idx in used_reviewed:
            continue
        used_predicted.add(pred_idx)
        used_reviewed.add(review_idx)
        pred = predicted[pred_idx]
        truth = reviewed[review_idx]
        matches.append(
            {
                "predicted_frame": pred["frame"],
                "reviewed_frame": truth["frame"],
                "signed_delta_frames": signed_delta,
                "predicted_t": pred["t"],
                "reviewed_t": truth["t"],
            }
        )

    missing = [truth for idx, truth in enumerate(reviewed) if idx not in used_reviewed]
    extra = [pred for idx, pred in enumerate(predicted) if idx not in used_predicted]
    return matches, missing, extra


def _bounce_summary(bounce: Any, *, fps: float) -> dict[str, Any]:
    return {
        "t": float(bounce.t),
        "frame": bounce.frame if bounce.frame is not None else round(float(bounce.t) * fps),
        "world_xy": list(bounce.world_xy),
        "contact_xy_img": list(bounce.contact_xy_img) if bounce.contact_xy_img is not None else None,
        "p_bounce": float(bounce.p_bounce) if bounce.p_bounce is not None else None,
        "audio_delta_ms": float(bounce.audio_delta_ms) if bounce.audio_delta_ms is not None else None,
        "source": bounce.source,
    }


def _predicted_bounce(bounce: Any, *, fps: float) -> dict[str, Any]:
    return {
        "frame": int(bounce.frame) if bounce.frame is not None else round(float(bounce.t) * fps),
        "t": float(bounce.t),
    }


def _reviewed_bounces(payload: Mapping[str, Any], *, fps: float) -> list[dict[str, Any]]:
    bounces = payload.get("bounces")
    if not isinstance(bounces, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in bounces:
        if not isinstance(item, Mapping):
            continue
        frame = item.get("frame")
        t = item.get("t")
        if frame is None and t is None:
            continue
        time_s = _finite_or_none(t)
        frame_index = int(frame) if isinstance(frame, int) else round(float(time_s) * fps) if time_s is not None else None
        if frame_index is None or frame_index < 0:
            continue
        parsed.append({"frame": frame_index, "t": float(time_s) if time_s is not None else frame_index / fps})
    return parsed


def _audio_onsets(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    onsets = payload.get("onsets")
    if not isinstance(onsets, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in onsets:
        if not isinstance(item, Mapping):
            continue
        time_s = _finite_or_none(item.get("time_s"))
        if time_s is None or time_s < 0.0:
            continue
        parsed.append({"time_s": time_s, "score": item.get("score"), "source": item.get("source")})
    return parsed


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


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _resolve_artifact_path(value: Any, *, base: Path | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or base is None:
        return path
    return base / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths_match(left: str, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


__all__ = [
    "BOUNCE_PROBABILITY_GATE",
    "M4_STATUS_SCAFFOLD",
    "M4_STATUS_TESTED",
    "build_ball_bounce_gate_report",
    "write_ball_bounce_gate_report",
]
