"""Apply accepted BODY world-label review corrections to review bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from threed.racketsport.body_world_label_finalize import finalize_body_world_labels


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_world_label_review_corrections_apply"
TEMPLATE_ARTIFACT_TYPE = "racketsport_body_world_label_review_corrections_template"
PIPELINE_ARTIFACT_TYPE = "racketsport_body_world_label_review_decision_pipeline"
TEMPLATE_ARTIFACT_SUFFIX = "body_world_label_review_bundle/body_world_joints.template.json"
OVERLAY_INDEX_ARTIFACT_SUFFIX = "body_world_label_review_bundle/overlays/body_world_label_review_overlay_index.json"
TEMPLATE_TOP_LEVEL_FIELDS = {"status", "not_ground_truth", "trusted_for_world_mpjpe"}
SAMPLE_FIELDS = {"accepted", "review_status", "notes", "joints_world"}
OVERLAY_FIELDS = {"warning_review_status", "warning_review_note"}
DEFAULT_ANNOTATOR = "human_reviewer"
BODY_OVERLAY_ACCEPT_DECISIONS = {"overlay_ok"}
BODY_OVERLAY_REJECT_DECISIONS = {"bad_alignment", "wrong_player"}
BODY_OVERLAY_PENDING_DECISIONS = {"", "unsure"}
BODY_LABEL_ACCEPT_DECISIONS = {"accept_candidate_label"}
BODY_LABEL_REJECT_DECISIONS = {"reject_candidate_label", "bad_alignment", "wrong_player"}
BODY_LABEL_PENDING_DECISIONS = {"", "unsure", "overlay_ok"}


def apply_body_world_label_review_corrections(
    *,
    template_path: str | Path,
    corrections_path: str | Path,
    out_template_path: str | Path,
    overlay_index_path: str | Path | None = None,
    out_overlay_index_path: str | Path | None = None,
) -> dict[str, Any]:
    template = _read_json(template_path)
    overlay_index = _read_json(overlay_index_path) if overlay_index_path is not None else None
    corrections = _read_json(corrections_path)
    raw_corrections = corrections.get("corrections")
    if not isinstance(raw_corrections, list):
        raise ValueError("corrections manifest must contain a corrections array")

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    blockers: list[str] = []
    sample_index = _sample_index(template)
    overlay_item_index = _overlay_item_index(overlay_index)

    for index, correction in enumerate(raw_corrections):
        if not isinstance(correction, Mapping):
            blockers.append(f"corrections/{index}: must be an object")
            continue
        status = str(correction.get("status", "pending"))
        correction_id = str(correction.get("id", f"correction_{index}"))
        if status != "accepted":
            skipped.append({"id": correction_id, "reason": f"status_{status}"})
            continue
        try:
            applied_item = _apply_correction(
                template=template,
                overlay_index=overlay_index,
                sample_index=sample_index,
                overlay_item_index=overlay_item_index,
                correction=correction,
            )
        except ValueError as exc:
            blockers.append(f"{correction_id}: {exc}")
            continue
        applied.append({"id": correction_id, **applied_item})

    status = "blocked" if blockers else ("applied" if applied else "no_accepted_corrections")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        "template_path": str(template_path),
        "overlay_index_path": str(overlay_index_path or ""),
        "corrections_path": str(corrections_path),
        "out_template_path": str(out_template_path),
        "out_overlay_index_path": str(out_overlay_index_path or ""),
        "accepted_correction_count": len(applied) + len(blockers),
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "applied": applied,
        "skipped": skipped,
        "blockers": blockers,
    }
    if blockers or not applied:
        return summary

    _write_json(out_template_path, template)
    if overlay_index is not None and out_overlay_index_path is not None:
        _write_json(out_overlay_index_path, overlay_index)
    return summary


def build_body_world_label_review_corrections_template(
    *,
    template_path: str | Path,
    out_path: str | Path,
    manifest_id: str,
    created_at: str,
    overlay_index_path: str | Path | None = None,
    annotator: str = DEFAULT_ANNOTATOR,
) -> dict[str, Any]:
    template = _read_json(template_path)
    overlay_index = _read_json(overlay_index_path) if overlay_index_path is not None else None
    corrections = _pending_template_corrections(
        template=template,
        template_path=Path(template_path),
        overlay_index=overlay_index,
        overlay_index_path=Path(overlay_index_path) if overlay_index_path is not None else None,
        manifest_id=manifest_id,
        created_at=created_at,
        annotator=annotator,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "created_at": created_at,
        "description": (
            "Pending BODY world-label review corrections. Review overlay images first; "
            "change selected correction statuses to accepted only after human review."
        ),
        "corrections": corrections,
    }
    _write_json(out_path, manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": TEMPLATE_ARTIFACT_TYPE,
        "status": "written",
        "manifest_id": manifest_id,
        "template_path": str(template_path),
        "overlay_index_path": str(overlay_index_path or ""),
        "out_path": str(out_path),
        "correction_count": len(corrections),
        "pending_correction_count": sum(1 for correction in corrections if correction.get("status") == "pending"),
    }


def merge_body_world_label_review_decisions_into_corrections(
    *,
    corrections_path: str | Path,
    review_input_path: str | Path,
    run_id: str,
    out_path: str | Path,
) -> dict[str, Any]:
    corrections_manifest = _read_json(corrections_path)
    review_input = _read_json(review_input_path)
    raw_corrections = corrections_manifest.get("corrections")
    if not isinstance(raw_corrections, list):
        raise ValueError("corrections manifest must contain a corrections array")
    run_decisions = _body_world_label_review_decisions(review_input, run_id)
    label_sample_ids = _label_sample_ids_from_corrections(raw_corrections)
    label_decision_states = {
        sample_id: _body_label_review_state(run_decisions.get(sample_id)) for sample_id in label_sample_ids
    }
    top_level_label_state = _top_level_label_state(label_decision_states)

    blockers: list[str] = []
    accepted_overlay_samples: list[str] = []
    rejected_overlay_samples: list[str] = []
    pending_overlay_samples: list[str] = []
    missing_overlay_decision_samples: list[str] = []
    accepted_label_samples: list[str] = []
    rejected_label_samples: list[str] = []
    pending_label_samples: list[str] = []
    missing_label_decision_samples: list[str] = []
    updated_corrections = 0
    overlay_corrections = 0
    label_corrections = 0
    top_level_label_corrections = 0

    for index, correction in enumerate(raw_corrections):
        if not isinstance(correction, dict):
            blockers.append(f"corrections/{index}: must be an object")
            continue
        target = correction.get("target")
        if not isinstance(target, Mapping):
            continue
        path = str(target.get("path", ""))
        overlay_path = _overlay_correction_path(path)
        if overlay_path is not None:
            overlay_corrections += 1
            sample_id, field = overlay_path
            decision_payload = run_decisions.get(sample_id)
            if decision_payload is None:
                missing_overlay_decision_samples.append(sample_id)
                continue
            decision = str(decision_payload.get("decision", "")).strip()
            if decision in BODY_OVERLAY_ACCEPT_DECISIONS:
                correction["status"] = "accepted"
                if field == "warning_review_status":
                    correction["value"] = "accepted"
                if field == "warning_review_note":
                    correction["value"] = _body_overlay_review_note(decision_payload, decision)
                accepted_overlay_samples.append(sample_id)
                updated_corrections += 1
            elif decision in BODY_OVERLAY_REJECT_DECISIONS:
                correction["status"] = "rejected"
                if field == "warning_review_note":
                    correction["value"] = _body_overlay_review_note(decision_payload, decision)
                rejected_overlay_samples.append(sample_id)
                updated_corrections += 1
            elif decision in BODY_OVERLAY_PENDING_DECISIONS or decision in BODY_LABEL_ACCEPT_DECISIONS:
                correction["status"] = "pending"
                pending_overlay_samples.append(sample_id)
            else:
                blockers.append(f"{sample_id}: unsupported BODY overlay review decision {decision!r}")
            continue

        sample_path = _sample_correction_path(path)
        if sample_path is not None:
            label_corrections += 1
            sample_id, field = sample_path
            decision_payload = run_decisions.get(sample_id)
            state = label_decision_states.get(sample_id, "missing")
            if state == "missing":
                missing_label_decision_samples.append(sample_id)
                continue
            if state == "accepted":
                correction["status"] = "accepted"
                if field == "notes":
                    correction["value"] = _body_label_review_note(decision_payload)
                accepted_label_samples.append(sample_id)
                updated_corrections += 1
            elif state == "rejected":
                correction["status"] = "rejected"
                rejected_label_samples.append(sample_id)
                updated_corrections += 1
            elif state == "pending":
                correction["status"] = "pending"
                pending_label_samples.append(sample_id)
            else:
                blockers.append(f"{sample_id}: unsupported BODY label review decision {state.removeprefix('unsupported:')}")
            continue

        if _is_template_top_level_correction_path(path):
            top_level_label_corrections += 1
            if top_level_label_state == "accepted":
                correction["status"] = "accepted"
                updated_corrections += 1
            elif top_level_label_state == "rejected":
                correction["status"] = "rejected"
                updated_corrections += 1
            elif top_level_label_state == "pending":
                correction["status"] = "pending"
            else:
                blockers.append(f"BODY label review has unsupported decisions: {top_level_label_state}")
            continue

    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "racketsport_body_world_label_review_decisions_merge",
        "status": "blocked" if blockers else "written",
        "corrections_path": str(corrections_path),
        "review_input_path": str(review_input_path),
        "run_id": run_id,
        "out_path": str(out_path),
        "overlay_correction_count": overlay_corrections,
        "label_correction_count": label_corrections,
        "top_level_label_correction_count": top_level_label_corrections,
        "updated_correction_count": updated_corrections,
        "accepted_overlay_warning_sample_count": len(_dedupe(accepted_overlay_samples)),
        "accepted_overlay_warning_sample_ids": _dedupe(accepted_overlay_samples),
        "rejected_overlay_warning_sample_count": len(_dedupe(rejected_overlay_samples)),
        "rejected_overlay_warning_sample_ids": _dedupe(rejected_overlay_samples),
        "pending_overlay_warning_sample_count": len(_dedupe(pending_overlay_samples)),
        "pending_overlay_warning_sample_ids": _dedupe(pending_overlay_samples),
        "missing_decision_sample_count": len(_dedupe(missing_overlay_decision_samples)),
        "missing_decision_sample_ids": _dedupe(missing_overlay_decision_samples),
        "accepted_label_sample_count": len(_dedupe(accepted_label_samples)),
        "accepted_label_sample_ids": _dedupe(accepted_label_samples),
        "rejected_label_sample_count": len(_dedupe(rejected_label_samples)),
        "rejected_label_sample_ids": _dedupe(rejected_label_samples),
        "pending_label_sample_count": len(_dedupe(pending_label_samples)),
        "pending_label_sample_ids": _dedupe(pending_label_samples),
        "missing_label_decision_sample_count": len(_dedupe(missing_label_decision_samples)),
        "missing_label_decision_sample_ids": _dedupe(missing_label_decision_samples),
        "blockers": blockers,
    }
    if blockers:
        return summary
    _write_json(out_path, corrections_manifest)
    return summary


def run_body_world_label_review_decision_pipeline(
    *,
    template_path: str | Path,
    overlay_index_path: str | Path,
    corrections_path: str | Path,
    review_input_path: str | Path,
    run_id: str,
    out_dir: str | Path,
    final_labels_path: str | Path | None = None,
    finalization_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge saved BODY review decisions, apply accepted corrections, and finalize if safe."""

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    merged_corrections_path = out_root / "body_world_label_review_corrections.merged.json"
    write_canonical_review_bundle = final_labels_path is not None or finalization_report_path is not None
    if write_canonical_review_bundle:
        reviewed_template_path = Path(template_path)
        reviewed_overlay_index_path = Path(overlay_index_path)
    else:
        reviewed_bundle_dir = out_root / "body_world_label_review_bundle"
        reviewed_template_path = reviewed_bundle_dir / "body_world_joints.template.json"
        reviewed_overlay_index_path = reviewed_bundle_dir / "overlays" / "body_world_label_review_overlay_index.json"
    final_labels_out = Path(final_labels_path) if final_labels_path is not None else out_root / "labels" / "body_world_joints.json"
    finalization_report_out = (
        Path(finalization_report_path)
        if finalization_report_path is not None
        else out_root / "body_world_label_finalization.json"
    )

    merge_summary = merge_body_world_label_review_decisions_into_corrections(
        corrections_path=corrections_path,
        review_input_path=review_input_path,
        run_id=run_id,
        out_path=merged_corrections_path,
    )
    apply_summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "not_run",
        "reason": "merge_blocked",
    }
    finalization_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "not_run",
        "reason": "merge_blocked",
    }
    finalization_template_path = str(template_path)

    if merge_summary["status"] == "written":
        apply_summary = apply_body_world_label_review_corrections(
            template_path=template_path,
            overlay_index_path=overlay_index_path,
            corrections_path=merged_corrections_path,
            out_template_path=reviewed_template_path,
            out_overlay_index_path=reviewed_overlay_index_path,
        )
        if apply_summary["status"] == "blocked":
            finalization_report = {
                "schema_version": SCHEMA_VERSION,
                "status": "not_run",
                "reason": "apply_blocked",
                "blockers": apply_summary.get("blockers", []),
            }
        else:
            finalization_input_path = reviewed_template_path if reviewed_template_path.is_file() else Path(template_path)
            finalization_template_path = str(finalization_input_path)
            finalization_report = finalize_body_world_labels(
                template_path=finalization_input_path,
                out_path=final_labels_out,
                report_out_path=finalization_report_out,
            )

    status = "finalized" if finalization_report.get("status") == "finalized" else "blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PIPELINE_ARTIFACT_TYPE,
        "status": status,
        "run_id": run_id,
        "template_path": str(template_path),
        "overlay_index_path": str(overlay_index_path),
        "corrections_path": str(corrections_path),
        "review_input_path": str(review_input_path),
        "out_dir": str(out_root),
        "merged_corrections_path": str(merged_corrections_path),
        "reviewed_template_path": str(reviewed_template_path),
        "reviewed_overlay_index_path": str(reviewed_overlay_index_path),
        "finalization_template_path": finalization_template_path,
        "final_labels_path": str(final_labels_out),
        "finalization_report_path": str(finalization_report_out),
        "merge": merge_summary,
        "apply": apply_summary,
        "finalization": finalization_report,
    }


def _pending_template_corrections(
    *,
    template: Mapping[str, Any],
    template_path: Path,
    overlay_index: Mapping[str, Any] | None,
    overlay_index_path: Path | None,
    manifest_id: str,
    created_at: str,
    annotator: str,
) -> list[dict[str, Any]]:
    corrections: list[dict[str, Any]] = []
    template_artifact = _artifact_for_manifest(template_path)
    clip_id = _correction_clip_id(template)
    _append_pending_correction(
        corrections,
        manifest_id=manifest_id,
        correction_id="set_template_status",
        artifact=template_artifact,
        clip_id=clip_id,
        path="/status",
        value="human_reviewed",
        reason="Mark template reviewed only after all selected BODY samples have been reviewed.",
        annotator=annotator,
        created_at=created_at,
    )
    _append_pending_correction(
        corrections,
        manifest_id=manifest_id,
        correction_id="set_template_not_ground_truth_false",
        artifact=template_artifact,
        clip_id=clip_id,
        path="/not_ground_truth",
        value=False,
        reason="Clear draft/not-ground-truth flag only after human review accepts selected samples.",
        annotator=annotator,
        created_at=created_at,
    )
    _append_pending_correction(
        corrections,
        manifest_id=manifest_id,
        correction_id="set_template_trusted_for_world_mpjpe",
        artifact=template_artifact,
        clip_id=clip_id,
        path="/trusted_for_world_mpjpe",
        value=True,
        reason="Trust template for world-MPJPE only after human review accepts selected samples.",
        annotator=annotator,
        created_at=created_at,
    )

    sample_index = _sample_index(template)
    for sample_id in _selected_sample_ids(template):
        sample = sample_index.get(sample_id)
        if sample is None:
            continue
        predicted_joints = _vectors(sample.get("predicted_joints_world"))
        if not predicted_joints:
            continue
        sample_id_fragment = _id_fragment(sample_id)
        _append_pending_correction(
            corrections,
            manifest_id=manifest_id,
            correction_id=f"sample_{sample_id_fragment}_joints_world",
            artifact=template_artifact,
            clip_id=clip_id,
            path=f"/samples/{sample_id}/joints_world",
            value=predicted_joints,
            reason="Use predicted world joints as candidate label only if human review accepts this sample.",
            annotator=annotator,
            created_at=created_at,
        )
        _append_pending_correction(
            corrections,
            manifest_id=manifest_id,
            correction_id=f"sample_{sample_id_fragment}_accepted",
            artifact=template_artifact,
            clip_id=clip_id,
            path=f"/samples/{sample_id}/accepted",
            value=True,
            reason="Accept selected BODY sample only after visual review.",
            annotator=annotator,
            created_at=created_at,
        )
        _append_pending_correction(
            corrections,
            manifest_id=manifest_id,
            correction_id=f"sample_{sample_id_fragment}_review_status",
            artifact=template_artifact,
            clip_id=clip_id,
            path=f"/samples/{sample_id}/review_status",
            value="reviewed",
            reason="Mark selected BODY sample reviewed only after visual review.",
            annotator=annotator,
            created_at=created_at,
        )
        _append_pending_correction(
            corrections,
            manifest_id=manifest_id,
            correction_id=f"sample_{sample_id_fragment}_notes",
            artifact=template_artifact,
            clip_id=clip_id,
            path=f"/samples/{sample_id}/notes",
            value=_sample_note_value(sample.get("notes")),
            reason="Preserve reviewer free-text notes with the accepted BODY sample.",
            annotator=annotator,
            created_at=created_at,
        )

    if overlay_index is not None and overlay_index_path is not None:
        overlay_artifact = _artifact_for_manifest(overlay_index_path)
        for sample_id in _overlay_warning_sample_ids(overlay_index):
            sample_id_fragment = _id_fragment(sample_id)
            _append_pending_correction(
                corrections,
                manifest_id=manifest_id,
                correction_id=f"overlay_{sample_id_fragment}_warning_review_status",
                artifact=overlay_artifact,
                clip_id=clip_id,
                path=f"/overlays/{sample_id}/warning_review_status",
                value="accepted",
                reason="Resolve overlay warning only after reviewer confirms it does not invalidate the sample.",
                annotator=annotator,
                created_at=created_at,
            )
            _append_pending_correction(
                corrections,
                manifest_id=manifest_id,
                correction_id=f"overlay_{sample_id_fragment}_warning_review_note",
                artifact=overlay_artifact,
                clip_id=clip_id,
                path=f"/overlays/{sample_id}/warning_review_note",
                value="Reviewer accepted this warning after visual overlay inspection.",
                reason="Record why the overlay warning was accepted for finalization.",
                annotator=annotator,
                created_at=created_at,
            )
    return corrections


def _body_world_label_review_decisions(review_input: Mapping[str, Any], run_id: str) -> dict[str, Mapping[str, Any]]:
    all_reviews = review_input.get("body_world_label_review")
    if not isinstance(all_reviews, Mapping):
        return {}
    run_reviews = all_reviews.get(run_id)
    if not isinstance(run_reviews, Mapping):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for sample_id, decision in run_reviews.items():
        if isinstance(decision, Mapping):
            out[str(sample_id)] = decision
    return out


def _overlay_correction_path(path: str) -> tuple[str, str] | None:
    try:
        parts = _logical_path_parts(path)
    except ValueError:
        return None
    if len(parts) == 3 and parts[0] == "overlays" and parts[2] in OVERLAY_FIELDS:
        return parts[1], parts[2]
    return None


def _sample_correction_path(path: str) -> tuple[str, str] | None:
    try:
        parts = _logical_path_parts(path)
    except ValueError:
        return None
    if len(parts) == 3 and parts[0] == "samples" and parts[2] in SAMPLE_FIELDS:
        return parts[1], parts[2]
    return None


def _is_template_top_level_correction_path(path: str) -> bool:
    try:
        parts = _logical_path_parts(path)
    except ValueError:
        return False
    return len(parts) == 1 and parts[0] in TEMPLATE_TOP_LEVEL_FIELDS


def _label_sample_ids_from_corrections(corrections: list[Any]) -> list[str]:
    sample_ids: list[str] = []
    for correction in corrections:
        if not isinstance(correction, Mapping):
            continue
        target = correction.get("target")
        if not isinstance(target, Mapping):
            continue
        sample_path = _sample_correction_path(str(target.get("path", "")))
        if sample_path is not None:
            sample_ids.append(sample_path[0])
    return _dedupe(sample_ids)


def _body_label_review_state(decision_payload: Mapping[str, Any] | None) -> str:
    if decision_payload is None:
        return "missing"
    decision = str(decision_payload.get("decision", "")).strip()
    if decision in BODY_LABEL_ACCEPT_DECISIONS:
        return "accepted"
    if decision in BODY_LABEL_REJECT_DECISIONS:
        return "rejected"
    if decision in BODY_LABEL_PENDING_DECISIONS:
        return "pending"
    return f"unsupported:{decision!r}"


def _top_level_label_state(label_decision_states: Mapping[str, str]) -> str:
    states = list(label_decision_states.values())
    if states and all(state == "accepted" for state in states):
        return "accepted"
    if any(state == "rejected" for state in states):
        return "rejected"
    unsupported = sorted(state for state in states if state.startswith("unsupported:"))
    if unsupported:
        return ", ".join(unsupported)
    return "pending"


def _body_overlay_review_note(decision_payload: Mapping[str, Any], decision: str) -> str:
    notes = str(decision_payload.get("notes", "")).strip()
    if notes:
        return notes
    if decision in BODY_OVERLAY_ACCEPT_DECISIONS:
        return "Reviewer accepted this BODY overlay warning via body_world_label_review."
    return f"Reviewer marked this BODY overlay warning as {decision} via body_world_label_review."


def _body_label_review_note(decision_payload: Mapping[str, Any] | None) -> str:
    if decision_payload is None:
        return ""
    return _sample_note_value(decision_payload.get("notes"))


def _sample_note_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _append_pending_correction(
    corrections: list[dict[str, Any]],
    *,
    manifest_id: str,
    correction_id: str,
    artifact: str,
    clip_id: str,
    path: str,
    value: Any,
    reason: str,
    annotator: str,
    created_at: str,
) -> None:
    corrections.append(
        {
            "id": correction_id,
            "target": {
                "artifact": artifact,
                "clip_id": clip_id,
                "phase": "phase3",
                "metric": "body_world_labels",
                "path": path,
            },
            "operation": "replace",
            "value": value,
            "reason": reason,
            "annotator": annotator,
            "created_at": created_at,
            "status": "pending",
        }
    )


def _apply_correction(
    *,
    template: dict[str, Any],
    overlay_index: dict[str, Any] | None,
    sample_index: Mapping[str, dict[str, Any]],
    overlay_item_index: Mapping[str, dict[str, Any]],
    correction: Mapping[str, Any],
) -> dict[str, Any]:
    target = correction.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("target must be an object")
    artifact = str(target.get("artifact", ""))
    path = str(target.get("path", ""))
    operation = str(correction.get("operation", ""))
    if operation not in {"set", "replace"}:
        raise ValueError(f"unsupported operation for BODY review correction: {operation}")
    if "value" not in correction:
        raise ValueError("accepted set/replace correction requires value")
    value = correction.get("value")

    if _is_template_artifact(artifact):
        return _apply_template_correction(template=template, sample_index=sample_index, path=path, value=value)
    if _is_overlay_artifact(artifact):
        if overlay_index is None:
            raise ValueError("overlay correction requires overlay index input")
        return _apply_overlay_correction(overlay_item_index=overlay_item_index, path=path, value=value)
    raise ValueError(f"unsupported BODY correction artifact: {artifact}")


def _apply_template_correction(
    *,
    template: dict[str, Any],
    sample_index: Mapping[str, dict[str, Any]],
    path: str,
    value: Any,
) -> dict[str, Any]:
    parts = _logical_path_parts(path)
    if len(parts) == 1 and parts[0] in TEMPLATE_TOP_LEVEL_FIELDS:
        _validate_template_value(parts[0], value)
        template[parts[0]] = value
        return {"artifact": "template", "path": path}
    if len(parts) == 3 and parts[0] == "samples" and parts[2] in SAMPLE_FIELDS:
        sample_id = parts[1]
        sample = sample_index.get(sample_id)
        if sample is None:
            raise ValueError(f"unknown sample id: {sample_id}")
        _validate_sample_value(parts[2], value)
        sample[parts[2]] = value
        return {"artifact": "template", "sample_id": sample_id, "path": path}
    raise ValueError(f"unsupported template correction path: {path}")


def _apply_overlay_correction(
    *,
    overlay_item_index: Mapping[str, dict[str, Any]],
    path: str,
    value: Any,
) -> dict[str, Any]:
    parts = _logical_path_parts(path)
    if len(parts) != 3 or parts[0] != "overlays" or parts[2] not in OVERLAY_FIELDS:
        raise ValueError(f"unsupported overlay correction path: {path}")
    sample_id = parts[1]
    overlay = overlay_item_index.get(sample_id)
    if overlay is None:
        raise ValueError(f"unknown overlay sample id: {sample_id}")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{parts[2]} must be a non-empty string")
    overlay[parts[2]] = value
    return {"artifact": "overlay_index", "sample_id": sample_id, "path": path}


def _is_template_artifact(artifact: str) -> bool:
    return artifact == "body_world_joints.template.json" or artifact.endswith(TEMPLATE_ARTIFACT_SUFFIX)


def _is_overlay_artifact(artifact: str) -> bool:
    return artifact == "body_world_label_review_overlay_index.json" or artifact.endswith(OVERLAY_INDEX_ARTIFACT_SUFFIX)


def _logical_path_parts(path: str) -> list[str]:
    if not path.startswith("/"):
        raise ValueError(f"path must be an absolute JSON-pointer-like path: {path}")
    return [part.replace("~1", "/").replace("~0", "~") for part in path.strip("/").split("/") if part]


def _validate_template_value(field: str, value: Any) -> None:
    if field == "status" and not isinstance(value, str):
        raise ValueError("status must be a string")
    if field in {"not_ground_truth", "trusted_for_world_mpjpe"} and not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")


def _validate_sample_value(field: str, value: Any) -> None:
    if field == "accepted" and not isinstance(value, bool):
        raise ValueError("accepted must be a boolean")
    if field in {"review_status", "notes"} and not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if field == "joints_world" and not _vectors(value):
        raise ValueError("joints_world must be a non-empty list of 3D vectors")


def _sample_index(template: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    samples = template.get("samples")
    if not isinstance(samples, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id", ""))
        if sample_id:
            out[sample_id] = sample
    return out


def _selected_sample_ids(template: Mapping[str, Any]) -> list[str]:
    selected = template.get("selected_sample_ids")
    if not isinstance(selected, list):
        return []
    return [str(sample_id) for sample_id in selected if str(sample_id)]


def _overlay_item_index(payload: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    overlays = payload.get("overlays") if isinstance(payload, Mapping) else None
    if not isinstance(overlays, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in overlays:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("sample_id", ""))
        if sample_id:
            out[sample_id] = item
    return out


def _overlay_warning_sample_ids(payload: Mapping[str, Any]) -> list[str]:
    overlays = payload.get("overlays")
    if not isinstance(overlays, list):
        return []
    sample_ids: list[str] = []
    for item in overlays:
        if not isinstance(item, Mapping):
            continue
        sample_id = str(item.get("sample_id", ""))
        warnings = item.get("warnings")
        if sample_id and isinstance(warnings, list) and any(str(warning) for warning in warnings):
            sample_ids.append(sample_id)
    return _dedupe(sample_ids)


def _correction_clip_id(template: Mapping[str, Any]) -> str:
    clip = str(template.get("clip", ""))
    return _id_fragment(clip) if clip else "unknown"


def _artifact_for_manifest(path: Path) -> str:
    parts = path.parts
    if "body_world_label_review_bundle" in parts:
        index = parts.index("body_world_label_review_bundle")
        return Path(*parts[index:]).as_posix()
    return path.name


def _id_fragment(value: str) -> str:
    out = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return out[:128] or "unknown"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _vectors(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    vectors: list[list[float]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 3:
            return []
        vector: list[float] = []
        for component in item:
            if isinstance(component, bool):
                return []
            try:
                vector.append(float(component))
            except (TypeError, ValueError):
                return []
        vectors.append(vector)
    return vectors


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
