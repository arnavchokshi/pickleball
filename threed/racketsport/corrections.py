from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
CORRECTION_STATUSES = ("accepted", "pending", "rejected")
DEFAULT_CORRECTION_STATUS = "pending"
UNKNOWN_GROUP = "unknown"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PHASE_PATTERN = re.compile(r"^(?:phase|eval)\d+$")


def enriched_queue_item(manifest_id: str, correction: Mapping[str, Any], manifest_created_at: str) -> dict[str, Any]:
    target = correction["target"]
    artifact = target["artifact"]
    phase = target.get("phase") or infer_phase(artifact)
    metric = target.get("metric") or infer_metric(artifact)
    return {
        "manifest_id": manifest_id,
        "correction_id": correction["id"],
        "status": correction.get("status", DEFAULT_CORRECTION_STATUS),
        "phase": phase,
        "metric": metric,
        "operation": correction["operation"],
        "artifact": artifact,
        "clip_id": target.get("clip_id"),
        "frame_index": target.get("frame_index"),
        "t_s": target.get("t_s"),
        "path": target["path"],
        "value": correction.get("value"),
        "confidence": correction.get("confidence"),
        "reason": correction["reason"],
        "annotator": correction["annotator"],
        "created_at": correction.get("created_at", manifest_created_at),
    }


def summarize_corrections_queue(corrections: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(corrections)
    return {
        "by_operation": _counter_dict(item["operation"] for item in items),
        "by_artifact": _counter_dict(item["artifact"] for item in items),
        "by_clip": _counter_dict(item.get("clip_id") or UNKNOWN_GROUP for item in items),
        "by_status": _counter_dict(item.get("status", DEFAULT_CORRECTION_STATUS) for item in items),
        "by_phase": _counter_dict(item.get("phase") or infer_phase(item.get("artifact")) for item in items),
        "by_metric": _counter_dict(item.get("metric") or infer_metric(item.get("artifact")) for item in items),
        "by_phase_metric_clip": _counter_dict(_phase_metric_clip_key(item) for item in items),
    }


def build_training_manifest_candidates(
    queue_payload: Mapping[str, Any], *, candidate_root: str | Path = "training/corrections"
) -> dict[str, Any]:
    root = _safe_relative_path_text(candidate_root, "candidate_root")
    raw_corrections = queue_payload.get("corrections")
    if not isinstance(raw_corrections, list):
        raise ValueError("corrections queue must contain a corrections array")

    entries: list[dict[str, Any]] = []
    for index, correction in enumerate(raw_corrections):
        if not isinstance(correction, Mapping):
            raise ValueError(f"corrections/{index}: must be an object")
        status = correction.get("status", DEFAULT_CORRECTION_STATUS)
        if status not in CORRECTION_STATUSES:
            raise ValueError(f"corrections/{index}/status: must be one of accepted, pending, rejected")
        if status != "accepted":
            continue

        manifest_id = _required_id(correction.get("manifest_id"), f"corrections/{index}/manifest_id")
        correction_id = _required_id(correction.get("correction_id"), f"corrections/{index}/correction_id")
        phase = _required_group_id(
            correction.get("phase") or infer_phase(correction.get("artifact")), f"corrections/{index}/phase"
        )
        metric = _required_group_id(
            correction.get("metric") or infer_metric(correction.get("artifact")), f"corrections/{index}/metric"
        )
        clip_id = _required_group_id(correction.get("clip_id") or UNKNOWN_GROUP, f"corrections/{index}/clip_id")
        artifact = _required_safe_artifact(correction.get("artifact"), f"corrections/{index}/artifact")
        source_path = correction.get("path")
        if not isinstance(source_path, str) or not source_path.startswith("/"):
            raise ValueError(f"corrections/{index}/path: must be a JSON pointer")

        entry_id = f"{manifest_id}__{correction_id}"
        entries.append(
            {
                "id": entry_id,
                "manifest_id": manifest_id,
                "correction_id": correction_id,
                "clip_id": clip_id,
                "phase": phase,
                "metric": metric,
                "source_artifact": artifact,
                "source_path": source_path,
                "candidate_path": f"{root}/{phase}/{metric}/{clip_id}/{entry_id}.json",
                "operation": correction.get("operation"),
                "value": correction.get("value"),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "racketsport_corrections_training_candidates",
        "candidate_root": root,
        "accepted_correction_count": len(entries),
        "entries": entries,
    }


def infer_phase(artifact: Any) -> str:
    if not isinstance(artifact, str):
        return UNKNOWN_GROUP
    for part in Path(artifact).parts:
        if PHASE_PATTERN.match(part):
            return part
    return UNKNOWN_GROUP


def infer_metric(artifact: Any) -> str:
    if not isinstance(artifact, str) or not artifact:
        return UNKNOWN_GROUP
    stem = Path(artifact).stem
    return stem if stem else UNKNOWN_GROUP


def is_unsafe_relative_path(value: str | Path) -> bool:
    path = Path(value)
    return path.is_absolute() or ".." in path.parts


def _phase_metric_clip_key(item: Mapping[str, Any]) -> str:
    phase = item.get("phase") or infer_phase(item.get("artifact"))
    metric = item.get("metric") or infer_metric(item.get("artifact"))
    clip_id = item.get("clip_id") or UNKNOWN_GROUP
    return f"{phase}/{metric}/{clip_id}"


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _safe_relative_path_text(value: str | Path, field: str) -> str:
    if is_unsafe_relative_path(value):
        raise ValueError(f"{field} must be relative and stay within the workspace")
    text = Path(value).as_posix().strip("/")
    if not text:
        raise ValueError(f"{field} must be a non-empty relative path")
    return text


def _required_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise ValueError(f"{field}: must match ^[A-Za-z0-9][A-Za-z0-9._-]*$")
    return value


def _required_group_id(value: Any, field: str) -> str:
    if value == UNKNOWN_GROUP:
        return UNKNOWN_GROUP
    return _required_id(value, field)


def _required_safe_artifact(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: must be a non-empty string")
    if is_unsafe_relative_path(value):
        raise ValueError(f"{field}: must be relative and stay within the workspace")
    return value


__all__ = [
    "CORRECTION_STATUSES",
    "DEFAULT_CORRECTION_STATUS",
    "SCHEMA_VERSION",
    "build_training_manifest_candidates",
    "enriched_queue_item",
    "infer_metric",
    "infer_phase",
    "is_unsafe_relative_path",
    "summarize_corrections_queue",
]
