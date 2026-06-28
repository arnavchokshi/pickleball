"""Compare promoted contact windows against saved human review contact times."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.schemas import ContactWindows


ARTIFACT_TYPE = "racketsport_contact_timing_review_alignment"
SCHEMA_VERSION = 1
DEFAULT_MAX_MATCH_DELTA_FRAMES = 2.0


def evaluate_review_alignment(
    *,
    review_input_path: str | Path,
    run_root: str | Path,
    clips: Sequence[str] | None = None,
    fps: float = 60.0,
    max_match_delta_frames: float = DEFAULT_MAX_MATCH_DELTA_FRAMES,
) -> dict[str, Any]:
    """Evaluate whether promoted contact windows preserve saved review UI timing.

    This is intentionally a review-alignment report. It does not evaluate the
    machine BALL detector, cue fusion, or the Phase 5 acceptance gate.
    """

    fps = _positive_finite(fps, "fps")
    max_match_delta_frames = _positive_finite(max_match_delta_frames, "max_match_delta_frames")
    review_path = Path(review_input_path)
    root_path = Path(run_root)
    review_input = _read_json_object(review_path)
    clip_names = list(clips) if clips is not None else _default_clips(review_input, root_path)
    if not clip_names:
        raise ValueError("no clips to evaluate")

    clip_reports = [
        _evaluate_clip(
            clip=clip,
            review_input=review_input,
            run_root=root_path,
            fps=fps,
            max_match_delta_frames=max_match_delta_frames,
        )
        for clip in clip_names
    ]
    summary = _summary(clip_reports, max_match_delta_frames=max_match_delta_frames)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "verification_scope": "human_review_alignment_only",
        "ball_verified": False,
        "status": _overall_status(clip_reports),
        "review_input_path": str(review_path),
        "run_root": str(root_path),
        "fps": fps,
        "max_match_delta_frames": max_match_delta_frames,
        "summary": summary,
        "clips": clip_reports,
        "notes": [
            "This compares promoted contact_windows.json events back to saved human review UI contact timestamps; it does not verify machine BALL contact detection, label F1, cue fusion, audio timing, or 3D ball physics."
        ],
    }


def write_review_alignment_report(out: str | Path, report: Mapping[str, Any]) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate_clip(
    *,
    clip: str,
    review_input: Mapping[str, Any],
    run_root: Path,
    fps: float,
    max_match_delta_frames: float,
) -> dict[str, Any]:
    reviewed_contacts = _reviewed_contacts(review_input, clip=clip, fps=fps)
    contact_windows_path = run_root / clip / "contact_windows.json"
    if not contact_windows_path.is_file():
        return {
            "clip": clip,
            "status": "blocked",
            "contact_windows_path": str(contact_windows_path),
            "missing_artifacts": ["contact_windows.json"],
            "reviewed_contact_count": len(reviewed_contacts),
            "promoted_contact_count": 0,
            "matched_contact_count": 0,
            "missing_reviewed_contact_count": len(reviewed_contacts),
            "extra_promoted_contact_count": 0,
            "matches": [],
            "missing_reviewed_contacts": reviewed_contacts,
            "extra_promoted_contacts": [],
            "metrics": _timing_metrics([]),
            "notes": ["missing promoted contact_windows.json"],
        }

    contact_windows = ContactWindows.model_validate(_read_json_object(contact_windows_path))
    promoted_contacts = _promoted_contacts(contact_windows)
    matches, missing, extra = _match_contacts(
        reviewed_contacts,
        promoted_contacts,
        fps=fps,
        max_match_delta_frames=max_match_delta_frames,
    )
    status = (
        "review_alignment_ok"
        if reviewed_contacts and len(matches) == len(reviewed_contacts) and not missing and not extra
        else "review_alignment_needs_attention"
    )
    return {
        "clip": clip,
        "status": status,
        "contact_windows_path": str(contact_windows_path),
        "missing_artifacts": [],
        "reviewed_contact_count": len(reviewed_contacts),
        "promoted_contact_count": len(promoted_contacts),
        "matched_contact_count": len(matches),
        "missing_reviewed_contact_count": len(missing),
        "extra_promoted_contact_count": len(extra),
        "matches": matches,
        "missing_reviewed_contacts": missing,
        "extra_promoted_contacts": extra,
        "metrics": _timing_metrics(matches),
        "notes": [],
    }


def _match_contacts(
    reviewed_contacts: list[dict[str, Any]],
    promoted_contacts: list[dict[str, Any]],
    *,
    fps: float,
    max_match_delta_frames: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_pairs: list[tuple[float, int, int, float, int]] = []
    for reviewed_idx, reviewed in enumerate(reviewed_contacts):
        for promoted_idx, promoted in enumerate(promoted_contacts):
            signed_delta_frames = (float(promoted["t"]) - float(reviewed["time_s"])) * fps
            abs_delta_frames = abs(signed_delta_frames)
            if abs_delta_frames <= max_match_delta_frames + 1e-9:
                signed_frame_index_delta = int(promoted["frame"]) - int(reviewed["frame"])
                candidate_pairs.append(
                    (abs_delta_frames, reviewed_idx, promoted_idx, signed_delta_frames, signed_frame_index_delta)
                )

    used_reviewed: set[int] = set()
    used_promoted: set[int] = set()
    pair_by_reviewed: dict[int, tuple[float, int, int, float, int]] = {}
    for pair in sorted(candidate_pairs):
        _, reviewed_idx, promoted_idx, _, _ = pair
        if reviewed_idx in used_reviewed or promoted_idx in used_promoted:
            continue
        used_reviewed.add(reviewed_idx)
        used_promoted.add(promoted_idx)
        pair_by_reviewed[reviewed_idx] = pair

    matches: list[dict[str, Any]] = []
    for reviewed_idx in sorted(pair_by_reviewed):
        abs_delta_frames, _, promoted_idx, signed_delta_frames, signed_frame_index_delta = pair_by_reviewed[reviewed_idx]
        reviewed = reviewed_contacts[reviewed_idx]
        promoted = promoted_contacts[promoted_idx]
        matches.append(
            {
                "reviewed_time_s": reviewed["time_s"],
                "reviewed_frame": reviewed["frame"],
                "promoted_t": promoted["t"],
                "promoted_frame": promoted["frame"],
                "signed_delta_frames": signed_delta_frames,
                "abs_delta_frames": abs_delta_frames,
                "signed_frame_index_delta": signed_frame_index_delta,
            }
        )

    missing = [contact for idx, contact in enumerate(reviewed_contacts) if idx not in used_reviewed]
    extra = [contact for idx, contact in enumerate(promoted_contacts) if idx not in used_promoted]
    return matches, missing, extra


def _reviewed_contacts(review_input: Mapping[str, Any], *, clip: str, fps: float) -> list[dict[str, Any]]:
    clips = review_input.get("clips")
    if not isinstance(clips, Mapping):
        raise ValueError("review input must contain a clips object")
    clip_payload = clips.get(clip)
    if not isinstance(clip_payload, Mapping):
        return []
    contacts = clip_payload.get("contacts")
    if contacts is None:
        return []
    if not isinstance(contacts, list):
        raise ValueError(f"review input contacts for {clip} must be a list")

    reviewed = []
    for contact in contacts:
        if not isinstance(contact, Mapping):
            continue
        time_s = _nonnegative_finite(contact.get("time_s"), "contact time_s")
        reviewed.append({"time_s": time_s, "frame": max(0, int(round(time_s * fps)))})
    return sorted(reviewed, key=lambda item: (item["time_s"], item["frame"]))


def _promoted_contacts(contact_windows: ContactWindows) -> list[dict[str, Any]]:
    contacts = [
        {"t": float(event.t), "frame": int(event.frame)}
        for event in contact_windows.events
        if event.type == "contact"
    ]
    return sorted(contacts, key=lambda item: (item["t"], item["frame"]))


def _timing_metrics(matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signed = [float(match["signed_delta_frames"]) for match in matches]
    absolute = [abs(value) for value in signed]
    return {
        "mean_signed_delta_frames": _mean(signed),
        "mean_abs_delta_frames": _mean(absolute),
        "p50_abs_delta_frames": _percentile(absolute, 50.0),
        "p90_abs_delta_frames": _percentile(absolute, 90.0),
        "max_abs_delta_frames": max(absolute) if absolute else None,
        "within_1_frame_count": sum(1 for value in absolute if value <= 1.0 + 1e-9),
        "within_2_frames_count": sum(1 for value in absolute if value <= 2.0 + 1e-9),
    }


def _summary(clip_reports: Sequence[Mapping[str, Any]], *, max_match_delta_frames: float) -> dict[str, Any]:
    reviewed_count = sum(int(report["reviewed_contact_count"]) for report in clip_reports)
    promoted_count = sum(int(report["promoted_contact_count"]) for report in clip_reports)
    matched_count = sum(int(report["matched_contact_count"]) for report in clip_reports)
    missing_count = sum(int(report["missing_reviewed_contact_count"]) for report in clip_reports)
    extra_count = sum(int(report["extra_promoted_contact_count"]) for report in clip_reports)
    matches = [match for report in clip_reports for match in report["matches"]]
    timing_metrics = _timing_metrics(matches)
    tolerance_label = _frame_tolerance_label(max_match_delta_frames)
    return {
        "clip_count": len(clip_reports),
        "reviewed_contact_count": reviewed_count,
        "promoted_contact_count": promoted_count,
        "matched_contact_count": matched_count,
        "missing_reviewed_contact_count": missing_count,
        "extra_promoted_contact_count": extra_count,
        f"reviewed_contacts_within_{tolerance_label}_frames_rate": (matched_count / reviewed_count)
        if reviewed_count
        else None,
        "all_reviewed_contacts_promoted_within_tolerance": bool(reviewed_count)
        and matched_count == reviewed_count
        and extra_count == 0,
        **timing_metrics,
    }


def _overall_status(clip_reports: Sequence[Mapping[str, Any]]) -> str:
    if any(report["status"] == "blocked" for report in clip_reports):
        return "blocked"
    if all(report["status"] == "review_alignment_ok" for report in clip_reports):
        return "review_alignment_ok"
    return "review_alignment_needs_attention"


def _default_clips(review_input: Mapping[str, Any], run_root: Path) -> list[str]:
    clips = review_input.get("clips")
    if not isinstance(clips, Mapping):
        raise ValueError("review input must contain a clips object")
    return sorted(
        clip
        for clip, payload in clips.items()
        if isinstance(clip, str)
        and isinstance(payload, Mapping)
        and isinstance(payload.get("contacts"), list)
        and (run_root / clip / "contact_windows.json").is_file()
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _positive_finite(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_finite(value: Any, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _frame_tolerance_label(max_match_delta_frames: float) -> str:
    if float(max_match_delta_frames).is_integer():
        return str(int(max_match_delta_frames))
    return str(max_match_delta_frames).replace(".", "_")
