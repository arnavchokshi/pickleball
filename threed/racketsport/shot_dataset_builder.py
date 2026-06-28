"""Build scaffold shot-classification datasets from reviewed event labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.shot_classifier import ALLOWED_SHOT_LABELS


DEFAULT_MAX_CONTACT_DT_S = 0.300
DEFAULT_WINDOW_MS = 900.0


def build_shot_dataset(
    *,
    dataset_id: str,
    clip_id: str,
    truth_events_payload: Mapping[str, Any],
    contact_windows_payload: Mapping[str, Any],
    out_dir: str | Path,
    split: str,
    fps: float,
    window_ms: float = DEFAULT_WINDOW_MS,
    max_contact_dt_s: float = DEFAULT_MAX_CONTACT_DT_S,
) -> dict[str, Any]:
    """Write a DATA-5 manifest and feature windows from reviewed truth labels."""

    dataset_id = _require_id(dataset_id, "dataset_id")
    clip_id = _require_id(clip_id, "clip_id")
    split = _require_split(split)
    fps = _require_positive_float(fps, "fps")
    window_ms = _require_positive_float(window_ms, "window_ms")
    max_contact_dt_s = _require_nonnegative_float(max_contact_dt_s, "max_contact_dt_s")
    truth_events = _truth_events(truth_events_payload)
    contact_events = _contact_events(contact_windows_payload)
    output_dir = Path(out_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    half_window_s = window_ms / 2000.0
    for truth in truth_events:
        contact, dt_s = _nearest_contact(truth, contact_events, max_contact_dt_s=max_contact_dt_s)
        entry_id = f"{clip_id}_{truth['id']}"
        feature_relpath = Path("features") / f"{entry_id}.json"
        feature = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "clip_id": clip_id,
            "truth": truth,
            "contact": contact,
            "features": _scaffold_features(truth=truth, contact=contact, dt_s=dt_s),
            "window": {
                "center_t": _round_time(truth["t"]),
                "start_t": _round_time(max(0.0, truth["t"] - half_window_s)),
                "end_t": _round_time(truth["t"] + half_window_s),
                "center_frame": _frame_index(truth, fps),
                "start_frame": max(0, round((truth["t"] - half_window_s) * fps)),
                "end_frame": round((truth["t"] + half_window_s) * fps),
                "fps": float(fps),
                "window_ms": float(window_ms),
            },
        }
        (output_dir / feature_relpath).write_text(json.dumps(feature, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        entries.append(
            {
                "id": entry_id,
                "path": feature_relpath.as_posix(),
                "split": split,
                "shot_label": truth["shot_label"],
                "source_type": "manual_review",
                "fps": float(fps),
                "contact_time_ms": round(truth["t"] * 1000.0, 3),
                "window_ms": float(window_ms),
                "player_id": str(truth["player_id"]),
                "notes": f"matched_contact_dt_s={dt_s:.3f}",
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "description": "Scaffold shot-classification dataset built from human-reviewed event labels.",
        "entries": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shot_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _truth_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("artifact_type") == "racketsport_shot_classification" or "shots" in payload:
        raise ValueError("prediction output cannot be used as DATA-5 truth")
    if payload.get("not_ground_truth") is True:
        raise ValueError("truth events marked not_ground_truth cannot build DATA-5")
    if payload.get("status") not in ("human_reviewed", "accepted", "reviewed"):
        raise ValueError("truth events must have explicit human_reviewed or accepted status")

    annotation = payload.get("annotation")
    if not isinstance(annotation, Mapping):
        raise ValueError("truth events payload must contain annotation.items")
    items = annotation.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("truth events annotation.items must be an array")

    truth_events: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"truth events/{index} must be an object")
        if item.get("status") != "accepted":
            continue
        shot_label = str(item.get("shot_label", ""))
        if shot_label not in ALLOWED_SHOT_LABELS:
            raise ValueError(f"truth events/{index}/shot_label is not a known pickleball shot label")
        t = _require_nonnegative_float(item.get("t"), f"truth events/{index}/t")
        player_id = item.get("player_id")
        if player_id is None or str(player_id) == "":
            raise ValueError(f"truth events/{index}/player_id is required")
        truth_events.append(
            {
                "id": _require_id(item.get("id", f"truth_{index:03d}"), f"truth events/{index}/id"),
                "t": t,
                "frame_index": _optional_nonnegative_int(item.get("frame_index")),
                "player_id": str(player_id),
                "shot_label": shot_label,
            }
        )

    if not truth_events:
        raise ValueError("no accepted DATA-5 truth events found")
    return truth_events


def _scaffold_features(*, truth: Mapping[str, Any], contact: Mapping[str, Any], dt_s: float) -> dict[str, float]:
    confidence = contact.get("confidence")
    frame_index = truth.get("frame_index")
    player_id = truth.get("player_id")
    try:
        player_value = float(player_id)
    except (TypeError, ValueError):
        player_value = 0.0
    return {
        "contact_confidence": float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 0.0,
        "contact_dt_s": round(float(dt_s), 6),
        "contact_time_s": round(float(truth["t"]), 6),
        "frame_index": float(frame_index) if isinstance(frame_index, int) else 0.0,
        "player_id": player_value,
    }


def _contact_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise ValueError("contact windows payload must contain events")
    events: list[dict[str, Any]] = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            raise ValueError(f"contact windows/{index} must be an object")
        event_type = event.get("type", "contact")
        if event_type != "contact":
            continue
        t = _require_nonnegative_float(event.get("t"), f"contact windows/{index}/t")
        events.append(
            {
                "id": str(event.get("id", f"contact_{index:03d}")),
                "t": t,
                "frame": _optional_nonnegative_int(event.get("frame")),
                "player_id": str(event.get("player_id", "")),
                "confidence": event.get("confidence"),
                "window": event.get("window", {}),
            }
        )
    if not events:
        raise ValueError("no contact events found")
    return events


def _nearest_contact(
    truth: Mapping[str, Any],
    contacts: Sequence[Mapping[str, Any]],
    *,
    max_contact_dt_s: float,
) -> tuple[dict[str, Any], float]:
    best = min(contacts, key=lambda contact: abs(float(contact["t"]) - float(truth["t"])))
    dt_s = abs(float(best["t"]) - float(truth["t"]))
    if dt_s > max_contact_dt_s:
        raise ValueError(f"no contact within {max_contact_dt_s:.3f}s for {truth['id']}")
    return dict(best), dt_s


def _frame_index(truth: Mapping[str, Any], fps: float) -> int:
    frame_index = truth.get("frame_index")
    if isinstance(frame_index, int):
        return frame_index
    return round(float(truth["t"]) * fps)


def _require_id(value: Any, label: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{label} is required")
    if "/" in text or "\\" in text or text in (".", ".."):
        raise ValueError(f"{label} must be a safe identifier")
    return text


def _require_split(value: str) -> str:
    if value not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, test")
    return value


def _require_positive_float(value: Any, label: str) -> float:
    number = _float(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _require_nonnegative_float(value: Any, label: str) -> float:
    number = _float(value, label)
    if number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _round_time(value: float) -> float:
    return round(float(value), 6)


__all__ = [
    "DEFAULT_MAX_CONTACT_DT_S",
    "DEFAULT_WINDOW_MS",
    "build_shot_dataset",
]
