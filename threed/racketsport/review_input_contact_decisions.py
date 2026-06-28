"""Bridge browser review-input contacts into contact-window review decisions."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .schemas import ContactWindowCandidates, ContactWindowReview, ContactWindows


def apply_review_input_contacts_to_review(
    candidates: ContactWindowCandidates | Mapping[str, Any],
    review: ContactWindowReview | Mapping[str, Any],
    review_input: Mapping[str, Any],
    *,
    clip: str,
    reviewer: str = "review-ui",
    max_delta_s: float = 0.25,
    player_map: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Accept nearest contact candidates from explicit browser review UI marks."""

    if max_delta_s <= 0.0 or not math.isfinite(max_delta_s):
        raise ValueError("max_delta_s must be a positive finite number")

    candidate_artifact = ContactWindowCandidates.model_validate(candidates)
    review_artifact = ContactWindowReview.model_validate(review)
    if candidate_artifact.clip != clip:
        raise ValueError(f"candidate clip {candidate_artifact.clip} does not match requested clip {clip}")
    if review_artifact.clip != clip:
        raise ValueError(f"review clip {review_artifact.clip} does not match requested clip {clip}")

    contacts = _review_input_contacts(review_input, clip)
    if not contacts:
        raise ValueError(f"review input clip {clip} has no contact marks")

    payload = review_artifact.model_dump(mode="json")
    decisions = payload["decisions"]
    decisions_by_id = {str(decision["review_id"]): decision for decision in decisions}
    candidate_by_id = {candidate.review_id: candidate for candidate in candidate_artifact.candidates}
    used_review_ids: set[str] = set()
    player_ids = _player_map(player_map)

    for contact in contacts:
        time_s = _finite_time(contact.get("time_s"))
        candidate, delta_s = _nearest_contact_candidate(candidate_artifact, time_s, used_review_ids)
        if candidate is None or delta_s > max_delta_s:
            raise ValueError(f"no contact candidate within {max_delta_s:.3f}s for review input contact at {time_s:.3f}s")
        decision = decisions_by_id.get(candidate.review_id)
        if decision is None:
            raise ValueError(f"candidate {candidate.review_id} is missing from review decisions")
        if decision.get("decision") == "accepted":
            raise ValueError(f"candidate {candidate.review_id} is already accepted")
        decision.update(
            {
                "decision": "accepted",
                "reviewer": reviewer,
                "reason": _contact_reason(contact, delta_s),
                "player_id": _contact_player_id(contact.get("player"), player_ids),
            }
        )
        used_review_ids.add(candidate.review_id)

    _refresh_summary_and_status(payload)
    ContactWindowReview.model_validate(payload)
    return payload


def build_contact_windows_from_review_input_contacts(
    review_input: Mapping[str, Any],
    *,
    clip: str,
    fps: float = 60.0,
    reviewer: str = "review-ui",
    window_radius_s: float = 0.08,
    trust_player_labels: bool = False,
    player_map: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build ContactWindows directly from explicit browser review UI contact marks.

    This is for human-clicked contact timestamps that are more complete than the
    current machine candidate list. Player labels default to untrusted because
    the review UI can collect timing before player identities are mapped.
    """

    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError("fps must be a positive finite number")
    if window_radius_s <= 0.0 or not math.isfinite(window_radius_s):
        raise ValueError("window_radius_s must be a positive finite number")

    contacts = _review_input_contacts(review_input, clip)
    if not contacts:
        raise ValueError(f"review input clip {clip} has no contact marks")

    player_ids = _player_map(player_map)
    events: list[dict[str, Any]] = []
    for contact in sorted(contacts, key=lambda item: _finite_time(item.get("time_s"))):
        time_s = _finite_time(contact.get("time_s"))
        player_id = _contact_player_id(contact.get("player"), player_ids) if trust_player_labels else None
        events.append(
            {
                "type": "contact",
                "t": time_s,
                "frame": max(0, int(round(time_s * fps))),
                "player_id": player_id,
                "confidence": 1.0,
                "sources": {
                    "audio": 0.0,
                    "wrist_vel": 0.0,
                    "ball_inflection": 0.0,
                    "human_review": 1.0,
                },
                "window": {
                    "t0": max(0.0, time_s - window_radius_s),
                    "t1": time_s + window_radius_s,
                    "importance": 1.0,
                },
            }
        )

    payload = {"schema_version": 1, "events": events}
    ContactWindows.model_validate(payload)
    return payload


def _review_input_contacts(review_input: Mapping[str, Any], clip: str) -> list[Mapping[str, Any]]:
    clips = review_input.get("clips")
    if not isinstance(clips, Mapping):
        return []
    clip_payload = clips.get(clip)
    if not isinstance(clip_payload, Mapping):
        return []
    contacts = clip_payload.get("contacts")
    if not isinstance(contacts, list):
        return []
    return [contact for contact in contacts if isinstance(contact, Mapping)]


def _nearest_contact_candidate(
    candidates: ContactWindowCandidates,
    time_s: float,
    used_review_ids: set[str],
) -> tuple[Any | None, float]:
    nearest = None
    nearest_delta = math.inf
    for candidate in candidates.candidates:
        if candidate.type != "contact" or candidate.review_id in used_review_ids:
            continue
        delta = abs(candidate.t - time_s)
        if delta < nearest_delta:
            nearest = candidate
            nearest_delta = delta
    return nearest, nearest_delta


def _finite_time(value: Any) -> float:
    try:
        time_s = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"contact time_s must be finite, got {value!r}") from exc
    if not math.isfinite(time_s):
        raise ValueError(f"contact time_s must be finite, got {value!r}")
    return time_s


def _contact_reason(contact: Mapping[str, Any], delta_s: float) -> str:
    note = str(contact.get("note", "")).strip()
    if note:
        return f"{note} Matched review UI contact mark within {delta_s:.3f}s."
    return f"Human marked contact in review UI; matched nearest contact candidate within {delta_s:.3f}s."


def _contact_player_id(value: Any, player_map: Mapping[str, int]) -> int | None:
    key = str(value or "").strip()
    if not key or key == "unknown":
        return None
    if key in player_map:
        return player_map[key]
    try:
        return int(key)
    except ValueError as exc:
        raise ValueError(f"review input player {key!r} is not in player_map") from exc


def _player_map(player_map: Mapping[str, int] | None) -> dict[str, int]:
    mapping = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    if player_map:
        mapping.update({str(key): int(value) for key, value in player_map.items()})
    return mapping


def _refresh_summary_and_status(payload: dict[str, Any]) -> None:
    decisions = payload["decisions"]
    accepted_count = sum(1 for decision in decisions if decision["decision"] == "accepted")
    rejected_count = sum(1 for decision in decisions if decision["decision"] == "rejected")
    pending_count = sum(1 for decision in decisions if decision["decision"] == "pending")
    payload["summary"] = {
        "candidate_count": len(decisions),
        "pending_count": pending_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
    }
    payload["status"] = "reviewed" if pending_count == 0 else "partially_reviewed" if accepted_count or rejected_count else "pending_review"
