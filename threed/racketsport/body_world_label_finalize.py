"""Finalize reviewed BODY world-joint labels for world-MPJPE gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_world_label_finalization"
LABEL_ARTIFACT_TYPE = "racketsport_body_world_joints_labels"
OVERLAY_INDEX_RELATIVE_PATH = Path("overlays/body_world_label_review_overlay_index.json")
RESOLVED_WARNING_REVIEW_STATUSES = {"accepted", "resolved", "human_reviewed"}
INDEPENDENT_LABEL_SOURCES = {
    "manual_3d_annotation",
    "trusted_teacher_3d",
    "multi_view_triangulation",
    "motion_capture",
    "external_ground_truth",
}


def finalize_body_world_labels(
    *,
    template_path: str | Path,
    out_path: str | Path,
    report_out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and write reviewed BODY world-joint labels.

    Draft templates are intentionally blocked. The world-MPJPE gate should only
    see labels after a reviewer has explicitly flipped the template out of
    draft/not-ground-truth state and accepted every selected sample.
    """

    template = _read_json(template_path)
    selected_ids = _selected_sample_ids(template)
    sample_index = _sample_index(template)
    accepted_samples, blockers, missing_selected_ids, candidate_label_ids = _accepted_samples(
        selected_ids=selected_ids,
        sample_index=sample_index,
        template=template,
    )
    overlay_warning_selected_ids = _overlay_warning_selected_sample_ids(
        template_path=Path(template_path),
        selected_ids=selected_ids,
    )
    if overlay_warning_selected_ids:
        blockers.append("selected_samples_have_overlay_warnings")
    blockers.extend(_template_state_blockers(template))
    blockers = _dedupe(blockers)
    status = "blocked" if blockers else "finalized"
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        "template_path": str(template_path),
        "out_path": str(out_path),
        "clip": str(template.get("clip", "")),
        "selected_sample_count": len(selected_ids),
        "accepted_sample_count": len(accepted_samples),
        "missing_selected_sample_count": len(missing_selected_ids),
        "missing_selected_sample_ids": missing_selected_ids,
        "candidate_label_sample_count": len(candidate_label_ids),
        "candidate_label_sample_ids": candidate_label_ids,
        "overlay_warning_selected_sample_count": len(overlay_warning_selected_ids),
        "overlay_warning_selected_sample_ids": overlay_warning_selected_ids,
        "blockers": blockers,
    }

    if blockers:
        if report_out_path is not None:
            _write_json(report_out_path, report)
        return report

    final_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": LABEL_ARTIFACT_TYPE,
        "status": "human_reviewed",
        "not_ground_truth": False,
        "trusted_for_world_mpjpe": True,
        "clip": str(template.get("clip", "")),
        "source_template": str(template_path),
        "source_packet": str(template.get("source_packet", "")),
        "source_video": str(template.get("source_video", "")),
        "joint_names": list(template.get("joint_names", [])) if isinstance(template.get("joint_names"), list) else [],
        "samples": accepted_samples,
    }
    _write_json(out_path, final_payload)
    if report_out_path is not None:
        _write_json(report_out_path, report)
    return report


def _template_state_blockers(template: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    status = str(template.get("status", ""))
    if status != "human_reviewed":
        blockers.append("template_not_reviewed")
    if template.get("not_ground_truth") is not False:
        blockers.append("template_marked_not_ground_truth")
    if template.get("trusted_for_world_mpjpe") is not True:
        blockers.append("template_not_trusted_for_world_mpjpe")
    return blockers


def _accepted_samples(
    *,
    selected_ids: list[str],
    sample_index: Mapping[str, Mapping[str, Any]],
    template: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    accepted_samples: list[dict[str, Any]] = []
    blockers: list[str] = []
    missing_selected_ids: list[str] = []
    rejected_selected_ids: list[str] = []
    candidate_label_ids: list[str] = []

    for sample_id in selected_ids:
        sample = sample_index.get(sample_id)
        if sample is None:
            missing_selected_ids.append(sample_id)
            continue
        if sample.get("accepted") is not True:
            rejected_selected_ids.append(sample_id)
            continue
        frame_index = _maybe_int(sample.get("frame_index"))
        player_id = _maybe_int(sample.get("player_id"))
        joints = _vectors(sample.get("joints_world"))
        if frame_index is None or player_id is None or not joints:
            rejected_selected_ids.append(sample_id)
            continue
        label_source = _label_source(sample=sample, template=template)
        if not _is_independent_label_source(label_source):
            candidate_label_ids.append(sample_id)
        accepted_samples.append(
            {
                "sample_id": sample_id,
                "frame_index": frame_index,
                "t": sample.get("t"),
                "player_id": player_id,
                "accepted": True,
                "label_source": label_source,
                "joints_world": joints,
            }
        )

    if missing_selected_ids:
        blockers.append("missing_selected_samples")
    if rejected_selected_ids or len(accepted_samples) != len(selected_ids):
        blockers.append("selected_samples_not_all_accepted")
    if not selected_ids:
        blockers.append("no_selected_samples")
    if candidate_label_ids:
        blockers.append("accepted_candidate_labels_not_independent_ground_truth")
    return accepted_samples, blockers, missing_selected_ids, _dedupe(candidate_label_ids)


def _label_source(*, sample: Mapping[str, Any], template: Mapping[str, Any]) -> str:
    for key in ("label_source", "ground_truth_source", "independent_label_source"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("label_source", "ground_truth_source", "independent_label_source"):
        value = template.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "accepted_candidate_prediction"


def _is_independent_label_source(label_source: str) -> bool:
    return label_source.strip().lower() in INDEPENDENT_LABEL_SOURCES


def _selected_sample_ids(template: Mapping[str, Any]) -> list[str]:
    selected = template.get("selected_sample_ids")
    if not isinstance(selected, list):
        return []
    return [str(sample_id) for sample_id in selected if str(sample_id)]


def _overlay_warning_selected_sample_ids(*, template_path: Path, selected_ids: list[str]) -> list[str]:
    overlay_index_path = template_path.parent / OVERLAY_INDEX_RELATIVE_PATH
    if not overlay_index_path.is_file():
        return []
    payload = _read_json(overlay_index_path)
    overlays = payload.get("overlays")
    if not isinstance(overlays, list):
        return []
    selected = set(selected_ids)
    warning_ids: list[str] = []
    for item in overlays:
        if not isinstance(item, Mapping):
            continue
        sample_id = str(item.get("sample_id", ""))
        warnings = item.get("warnings")
        if (
            sample_id in selected
            and isinstance(warnings, list)
            and any(str(warning) for warning in warnings)
            and not _overlay_warnings_human_resolved(item)
        ):
            warning_ids.append(sample_id)
    return _dedupe(warning_ids)


def _overlay_warnings_human_resolved(item: Mapping[str, Any]) -> bool:
    return str(item.get("warning_review_status", "")).strip().lower() in RESOLVED_WARNING_REVIEW_STATUSES


def _sample_index(template: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    samples = template.get("samples")
    if not isinstance(samples, list):
        return {}
    index: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        sample_id = str(sample.get("sample_id", ""))
        if sample_id:
            index[sample_id] = sample
    return index


def _vectors(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    vectors: list[list[float]] = []
    for item in value:
        vector = _vector(item)
        if not vector:
            return []
        vectors.append(vector)
    return vectors


def _vector(value: Any) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return []
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["ARTIFACT_TYPE", "finalize_body_world_labels"]
