from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from threed.racketsport.body_grounding_quality import build_body_grounding_quality


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_gate_report"
DEFAULT_WORLD_MPJPE_THRESHOLD_M = 0.05
DEFAULT_WORLD_WRIST_MPJPE_THRESHOLD_M = 0.03
DEFAULT_WORLD_MPJPE_MIN_LABEL_SAMPLES = 20
DEFAULT_WORLD_MPJPE_MIN_LABEL_COVERAGE_RATIO = 0.10
BODY_WORLD_LABEL_FILENAMES = ("body_world_joints.json", "body_world_mpjpe.json")
BODY_WORLD_LABEL_PACKET_FILENAME = "body_world_label_packet.json"
BODY_PACKET_QUALITY_FILENAME = "body_joint_quality_from_packet.json"
BODY_GROUNDING_QUALITY_FILENAME = "body_grounding_quality.json"
PERSON_TRACK_GT_SCORE_FILENAME = "person_track_gt_score.json"
PERSON_TRACK_GT_SCORING_REPORT_FILENAME = "person_track_gt_scoring_report.json"
FULL_CLIP_GATE_FILENAME = "body_full_clip_gate.json"
FULL_CLIP_GATE_REQUIRED_FIELDS = ("passed", "coverage", "evaluated_frame_count")
BODY_LABEL_REVIEW_BUNDLE_PATH = Path("body_world_label_review_bundle/body_world_label_review_bundle.json")
BODY_LABEL_REVIEW_TEMPLATE_PATH = Path("body_world_label_review_bundle/body_world_joints.template.json")
BODY_LABEL_FINALIZATION_REPORT_PATH = Path("body_world_label_review_bundle/body_world_label_finalization.json")
BODY_REVIEW_OVERLAY_INDEX_PATH = Path(
    "body_world_label_review_bundle/overlays/body_world_label_review_overlay_index.json"
)
RESOLVED_WARNING_REVIEW_STATUSES = {"accepted", "resolved", "human_reviewed"}
INDEPENDENT_BODY_LABEL_SOURCES = {
    "manual_3d_annotation",
    "trusted_teacher_3d",
    "multi_view_triangulation",
    "motion_capture",
    "external_ground_truth",
}
INSPECTABLE_OUTPUTS = (
    "virtual_world_paddle_preview.html",
    "virtual_world_review_index.json",
    "virtual_world.json",
    BODY_PACKET_QUALITY_FILENAME,
    BODY_GROUNDING_QUALITY_FILENAME,
    PERSON_TRACK_GT_SCORE_FILENAME,
    PERSON_TRACK_GT_SCORING_REPORT_FILENAME,
    str(BODY_LABEL_REVIEW_BUNDLE_PATH),
    str(BODY_LABEL_REVIEW_TEMPLATE_PATH),
    str(BODY_LABEL_FINALIZATION_REPORT_PATH),
    str(BODY_REVIEW_OVERLAY_INDEX_PATH),
)


def build_body_gate_report(
    *,
    root: str | Path,
    clips: list[str] | tuple[str, ...] | None = None,
    labels_root: str | Path | None = None,
    world_mpjpe_threshold_m: float = DEFAULT_WORLD_MPJPE_THRESHOLD_M,
    world_wrist_mpjpe_threshold_m: float = DEFAULT_WORLD_WRIST_MPJPE_THRESHOLD_M,
    world_mpjpe_min_label_samples: int = DEFAULT_WORLD_MPJPE_MIN_LABEL_SAMPLES,
    world_mpjpe_min_label_coverage_ratio: float = DEFAULT_WORLD_MPJPE_MIN_LABEL_COVERAGE_RATIO,
) -> dict[str, Any]:
    root_path = Path(root)
    labels_path = Path(labels_root) if labels_root is not None else root_path
    clip_names = list(clips) if clips else _discover_clips(root_path)
    clip_reports = [
        _build_clip_report(
            root_path=root_path,
            labels_root=labels_path,
            clip=clip,
            world_mpjpe_threshold_m=world_mpjpe_threshold_m,
            world_wrist_mpjpe_threshold_m=world_wrist_mpjpe_threshold_m,
            world_mpjpe_min_label_samples=world_mpjpe_min_label_samples,
            world_mpjpe_min_label_coverage_ratio=world_mpjpe_min_label_coverage_ratio,
        )
        for clip in clip_names
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "root": str(root_path),
        "labels_root": str(labels_path),
        "status": _aggregate_status(clip_reports),
        "world_mpjpe_threshold_m": world_mpjpe_threshold_m,
        "world_wrist_mpjpe_threshold_m": world_wrist_mpjpe_threshold_m,
        "world_mpjpe_min_label_samples": world_mpjpe_min_label_samples,
        "world_mpjpe_min_label_coverage_ratio": world_mpjpe_min_label_coverage_ratio,
        "summary": {
            "clip_count": len(clip_reports),
            "pass_count": sum(1 for clip in clip_reports if clip["status"] == "pass"),
            "fail_count": sum(1 for clip in clip_reports if clip["status"] == "fail"),
            "blocked_count": sum(1 for clip in clip_reports if clip["status"] == "blocked"),
        },
        "clips": clip_reports,
    }


def write_body_gate_report(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_body_gate_markdown(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_body_gate_markdown(payload), encoding="utf-8")


def write_clip_body_gate_reports(root: str | Path, payload: Mapping[str, Any]) -> None:
    root_path = Path(root)
    for clip in payload.get("clips", []):
        if not isinstance(clip, Mapping):
            continue
        clip_name = str(clip.get("clip", ""))
        if not clip_name:
            continue
        clip_payload = {
            **dict(payload),
            "status": clip.get("status", "blocked"),
            "summary": {
                "clip_count": 1,
                "pass_count": 1 if clip.get("status") == "pass" else 0,
                "fail_count": 1 if clip.get("status") == "fail" else 0,
                "blocked_count": 1 if clip.get("status") == "blocked" else 0,
            },
            "clips": [dict(clip)],
        }
        clip_dir = root_path / clip_name
        write_body_gate_report(clip_dir / "body_gate_report.json", clip_payload)
        write_body_gate_markdown(clip_dir / "body_gate_report.md", clip_payload)


def render_body_gate_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# BODY Gate Report",
        "",
        f"- status: `{payload.get('status', 'blocked')}`",
        f"- root: `{payload.get('root', '')}`",
        f"- labels_root: `{payload.get('labels_root', '')}`",
        f"- world_mpjpe_threshold_m: `{payload.get('world_mpjpe_threshold_m', DEFAULT_WORLD_MPJPE_THRESHOLD_M)}`",
        f"- world_wrist_mpjpe_threshold_m: `{payload.get('world_wrist_mpjpe_threshold_m', DEFAULT_WORLD_WRIST_MPJPE_THRESHOLD_M)}`",
        f"- world_mpjpe_min_label_samples: `{payload.get('world_mpjpe_min_label_samples', DEFAULT_WORLD_MPJPE_MIN_LABEL_SAMPLES)}`",
        f"- world_mpjpe_min_label_coverage_ratio: `{payload.get('world_mpjpe_min_label_coverage_ratio', DEFAULT_WORLD_MPJPE_MIN_LABEL_COVERAGE_RATIO)}`",
        "",
        "| Clip | Status | Mesh smoke | Packet quality | World MPJPE | Grounding | Identity | Full clip BODY | BODY overlay | Label review | Scheduled frames | Mesh player-frames | Blockers | Inspectable outputs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for clip in payload.get("clips", []):
        if not isinstance(clip, Mapping):
            continue
        mesh = clip.get("mesh_smoke") if isinstance(clip.get("mesh_smoke"), Mapping) else {}
        packet_quality = (
            clip.get("body_packet_quality") if isinstance(clip.get("body_packet_quality"), Mapping) else {}
        )
        world = clip.get("world_mpjpe") if isinstance(clip.get("world_mpjpe"), Mapping) else {}
        grounding = (
            clip.get("body_grounding_quality")
            if isinstance(clip.get("body_grounding_quality"), Mapping)
            else {}
        )
        identity = (
            clip.get("tracking_identity_quality")
            if isinstance(clip.get("tracking_identity_quality"), Mapping)
            else {}
        )
        full = clip.get("full_clip_body_gate") if isinstance(clip.get("full_clip_body_gate"), Mapping) else {}
        overlay = (
            clip.get("body_review_overlay_alignment")
            if isinstance(clip.get("body_review_overlay_alignment"), Mapping)
            else {}
        )
        label_review = clip.get("body_label_review") if isinstance(clip.get("body_label_review"), Mapping) else {}
        blockers = ", ".join(str(item) for item in clip.get("blockers", [])) or "-"
        outputs = ", ".join(str(item) for item in clip.get("inspectable_outputs", [])) or "-"
        lines.append(
            "| {clip} | {status} | {mesh_status} | {packet_quality_status} | {world_status} | {grounding_status} | {identity_status} | {full_status} | {overlay_status} | {label_review_status} | {scheduled} | {mesh_frames} | {blockers} | {outputs} |".format(
                clip=clip.get("clip", ""),
                status=clip.get("status", ""),
                mesh_status=mesh.get("status", ""),
                packet_quality_status=packet_quality.get("status", ""),
                world_status=world.get("status", ""),
                grounding_status=grounding.get("status", ""),
                identity_status=identity.get("status", ""),
                full_status=full.get("status", ""),
                overlay_status=overlay.get("status", ""),
                label_review_status=label_review.get("status", ""),
                scheduled=mesh.get("scheduled_frame_count", 0),
                mesh_frames=mesh.get("mesh_player_frame_count", 0),
                blockers=blockers,
                outputs=outputs,
            )
        )
    _append_label_finalization_markdown(lines, payload)
    _append_overlay_warning_markdown(lines, payload)
    lines.append("")
    return "\n".join(lines)


def _append_label_finalization_markdown(lines: list[str], payload: Mapping[str, Any]) -> None:
    rows: list[str] = []
    for clip in payload.get("clips", []):
        if not isinstance(clip, Mapping):
            continue
        label_review = clip.get("body_label_review") if isinstance(clip.get("body_label_review"), Mapping) else {}
        if label_review.get("status") != "blocked_finalization":
            continue
        blockers = _markdown_list(label_review.get("finalization_blockers"))
        warning_ids = _markdown_list(label_review.get("finalization_overlay_warning_selected_sample_ids"))
        rows.append(
            "| {clip} | {status} | {selected} | {accepted} | {blockers} | {warning_ids} |".format(
                clip=clip.get("clip", ""),
                status=label_review.get("status", ""),
                selected=label_review.get(
                    "finalization_selected_sample_count",
                    label_review.get("selected_sample_count", 0),
                ),
                accepted=label_review.get("finalization_accepted_sample_count", 0),
                blockers=blockers,
                warning_ids=warning_ids,
            )
        )
    if not rows:
        return
    lines.extend(
        [
            "",
            "## BODY Label Finalization Blockers",
            "",
            "| Clip | Label review | Selected samples | Accepted samples | Finalization blockers | Overlay-warning selected samples |",
            "| --- | --- | ---: | ---: | --- | --- |",
            *rows,
        ]
    )


def _markdown_list(value: Any) -> str:
    if not isinstance(value, list):
        return "-"
    items = [str(item) for item in value if str(item)]
    return ", ".join(items) if items else "-"


def _append_overlay_warning_markdown(lines: list[str], payload: Mapping[str, Any]) -> None:
    rows: list[str] = []
    for clip in payload.get("clips", []):
        if not isinstance(clip, Mapping):
            continue
        overlay = (
            clip.get("body_review_overlay_alignment")
            if isinstance(clip.get("body_review_overlay_alignment"), Mapping)
            else {}
        )
        warning_samples = overlay.get("warning_samples")
        if not isinstance(warning_samples, list):
            continue
        for sample in warning_samples:
            if not isinstance(sample, Mapping):
                continue
            rows.append(
                "| {clip} | {sample_id} | {warnings} | {containment} | {center_delta} | {competing_player_id} | {overlay_path} |".format(
                    clip=clip.get("clip", ""),
                    sample_id=sample.get("sample_id", ""),
                    warnings=_markdown_list(sample.get("warnings")),
                    containment=_markdown_scalar(sample.get("containment_ratio")),
                    center_delta=_markdown_scalar(sample.get("center_delta_px")),
                    competing_player_id=_markdown_scalar(sample.get("competing_player_id")),
                    overlay_path=_markdown_scalar(sample.get("overlay_path")),
                )
            )
    if not rows:
        return
    lines.extend(
        [
            "",
            "## BODY Overlay Warning Samples",
            "",
            "| Clip | Sample | Warnings | Containment ratio | Center delta px | Competing player | Overlay |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
            *rows,
        ]
    )


def _markdown_scalar(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _build_clip_report(
    *,
    root_path: Path,
    labels_root: Path,
    clip: str,
    world_mpjpe_threshold_m: float,
    world_wrist_mpjpe_threshold_m: float,
    world_mpjpe_min_label_samples: int,
    world_mpjpe_min_label_coverage_ratio: float,
) -> dict[str, Any]:
    run_dir = root_path / clip
    labels_dir = labels_root / clip
    smpl_motion = _read_optional_json(run_dir / "smpl_motion.json")
    skeleton3d = _read_optional_json(run_dir / "skeleton3d.json")
    body_compute_execution = _read_optional_json(run_dir / "body_compute_execution.json")
    body_mesh_readiness = _read_optional_json(run_dir / "body_mesh_readiness.json")

    mesh_smoke = _mesh_smoke_status(
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution=body_compute_execution,
        body_mesh_readiness=body_mesh_readiness,
    )
    world_mpjpe = _world_mpjpe_status(
        run_dir=run_dir,
        labels_dir=labels_dir,
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution=body_compute_execution,
        threshold_m=world_mpjpe_threshold_m,
        wrist_threshold_m=world_wrist_mpjpe_threshold_m,
        min_label_samples=world_mpjpe_min_label_samples,
        min_label_coverage_ratio=world_mpjpe_min_label_coverage_ratio,
    )
    body_grounding_quality = _body_grounding_quality_status(run_dir=run_dir)
    tracking_identity_quality = _tracking_identity_quality_status(run_dir=run_dir, labels_dir=labels_dir, clip=clip)
    full_clip_gate = _full_clip_gate_status(run_dir=run_dir, labels_dir=labels_dir)
    body_packet_quality = _body_packet_quality_status(run_dir=run_dir)
    body_review_overlay_alignment = _body_review_overlay_alignment_status(run_dir=run_dir)
    body_label_review = _body_label_review_status(run_dir=run_dir, labels_dir=labels_dir)
    blockers = _dedupe(
        [
            *mesh_smoke.get("blockers", []),
            *body_packet_quality.get("blockers", []),
            *world_mpjpe.get("blockers", []),
            *body_grounding_quality.get("blockers", []),
            *tracking_identity_quality.get("blockers", []),
            *full_clip_gate.get("blockers", []),
            *body_review_overlay_alignment.get("blockers", []),
            *body_label_review.get("blockers", []),
        ]
    )
    status = _clip_status(mesh_smoke, world_mpjpe, body_grounding_quality, tracking_identity_quality, full_clip_gate, blockers)
    return {
        "clip": clip,
        "run_dir": str(run_dir),
        "labels_dir": str(labels_dir),
        "status": status,
        "mesh_smoke": mesh_smoke,
        "body_packet_quality": body_packet_quality,
        "world_mpjpe": world_mpjpe,
        "body_grounding_quality": body_grounding_quality,
        "tracking_identity_quality": tracking_identity_quality,
        "full_clip_body_gate": full_clip_gate,
        "body_review_overlay_alignment": body_review_overlay_alignment,
        "body_label_review": body_label_review,
        "blockers": blockers,
        "inspectable_outputs": [name for name in INSPECTABLE_OUTPUTS if (run_dir / name).is_file()],
    }


def _mesh_smoke_status(
    *,
    smpl_motion: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
    body_compute_execution: Mapping[str, Any] | None,
    body_mesh_readiness: Mapping[str, Any] | None,
) -> dict[str, Any]:
    execution_summary = _mapping(body_compute_execution, "summary")
    readiness_summary = _mapping(body_mesh_readiness, "summary")
    representation_plan = _mapping(body_mesh_readiness, "representation_plan")
    scheduled_by_reason = _non_negative_int_mapping(execution_summary.get("scheduled_by_reason"))
    scheduled_by_target = _non_negative_int_mapping(execution_summary.get("scheduled_by_target_representation"))
    skipped_by_reason = _non_negative_int_mapping(execution_summary.get("skipped_by_reason"))
    skipped_by_target = _non_negative_int_mapping(execution_summary.get("skipped_by_target_representation"))
    skipped_by_tier = _non_negative_int_mapping(execution_summary.get("skipped_by_tier"))
    scheduled_frame_count = _non_negative_int(execution_summary.get("scheduled_frame_count"))
    scheduled_player_frame_count = _non_negative_int(execution_summary.get("scheduled_player_frame_count"))
    if scheduled_frame_count == 0:
        scheduled_frame_count = _non_negative_int(representation_plan.get("scheduled_world_mesh_frame_count"))
    if scheduled_player_frame_count == 0:
        scheduled_player_frame_count = _non_negative_int(representation_plan.get("scheduled_world_mesh_player_frame_count"))
    mesh_player_frame_count = _non_negative_int(readiness_summary.get("mesh_frame_count"))
    skeleton_player_frame_count = _skeleton_player_frame_count(skeleton3d)
    smpl_player_frame_count = _smpl_player_frame_count(smpl_motion)
    blockers: list[str] = []
    notes: list[str] = []

    if body_compute_execution is None:
        blockers.append("missing_body_compute_execution")
    if body_mesh_readiness is None:
        blockers.append("missing_body_mesh_readiness")
    if scheduled_frame_count > 0 and mesh_player_frame_count > 0:
        status = "pass"
        notes.append("scheduled-frame mesh smoke available")
    elif scheduled_frame_count == 0:
        status = "not_measured"
        blockers.append("no_scheduled_body_mesh_smoke")
        notes.append("no scheduled BODY world-mesh frames")
    else:
        status = "blocked"
        blockers.append("missing_scheduled_body_mesh_output")

    if smpl_motion is None and mesh_player_frame_count > 0:
        notes.append("smpl_motion.json not present in this compact report copy")
    if skeleton3d is None and mesh_player_frame_count > 0:
        notes.append("skeleton3d.json not present in this compact report copy")

    return {
        "status": status,
        "scheduled_frame_count": scheduled_frame_count,
        "scheduled_player_frame_count": scheduled_player_frame_count,
        "scheduled_by_reason": scheduled_by_reason,
        "scheduled_by_target_representation": scheduled_by_target,
        "skipped_frame_count": _non_negative_int(execution_summary.get("skipped_frame_count")),
        "skipped_by_reason": skipped_by_reason,
        "skipped_by_target_representation": skipped_by_target,
        "skipped_by_tier": skipped_by_tier,
        "mesh_player_frame_count": mesh_player_frame_count,
        "skeleton_player_frame_count": skeleton_player_frame_count,
        "smpl_player_frame_count": smpl_player_frame_count,
        "body_mesh_readiness_status": str(body_mesh_readiness.get("status", "")) if isinstance(body_mesh_readiness, Mapping) else "",
        "blockers": _dedupe(blockers),
        "notes": notes,
    }


def _world_mpjpe_status(
    *,
    run_dir: Path,
    labels_dir: Path,
    smpl_motion: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
    body_compute_execution: Mapping[str, Any] | None,
    threshold_m: float,
    wrist_threshold_m: float,
    min_label_samples: int,
    min_label_coverage_ratio: float,
) -> dict[str, Any]:
    label_path = _find_body_label_path(labels_dir)
    if label_path is None:
        label_import = _missing_body_label_import_status(labels_dir)
        return {
            "status": "not_measured",
            "label_path": "",
            "mean_error_m": None,
            "threshold_m": threshold_m,
            "wrist_threshold_m": wrist_threshold_m,
            "wrist_mean_error_m": None,
            "wrist_joint_count": 0,
            "sample_count": 0,
            "joint_count": 0,
            "blockers": ["missing_world_mpjpe_gate"],
            "notes": ["no BODY world-joint labels found"],
            "label_import": label_import,
        }

    payload = _read_json(label_path)
    samples = _label_samples(payload)
    label_import = _body_label_import_status(
        labels_dir=labels_dir,
        label_path=label_path,
        payload=payload,
        accepted_sample_count=len(samples),
    )
    if label_import["status"] != "present_reviewed":
        return {
            "status": "not_measured",
            "label_path": str(label_path),
            "mean_error_m": None,
            "threshold_m": threshold_m,
            "wrist_threshold_m": wrist_threshold_m,
            "wrist_mean_error_m": None,
            "wrist_joint_count": 0,
            "sample_count": 0,
            "joint_count": 0,
            "blockers": _dedupe(["missing_world_mpjpe_gate", *label_import.get("blockers", [])]),
            "notes": list(label_import.get("notes", [])),
            "label_import": label_import,
        }
    if not samples:
        return {
            "status": "not_measured",
            "label_path": str(label_path),
            "mean_error_m": None,
            "threshold_m": threshold_m,
            "wrist_threshold_m": wrist_threshold_m,
            "wrist_mean_error_m": None,
            "wrist_joint_count": 0,
            "sample_count": 0,
            "joint_count": 0,
            "blockers": ["missing_world_mpjpe_gate"],
            "notes": ["BODY world-joint label file has no accepted samples"],
            "label_import": label_import,
        }

    prediction_index, prediction_source = _prediction_index(
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_world_label_packet=_read_optional_json(run_dir / BODY_WORLD_LABEL_PACKET_FILENAME),
    )
    label_coverage = _world_label_coverage(
        samples=samples,
        prediction_index=prediction_index,
        body_compute_execution=body_compute_execution,
        min_label_samples=min_label_samples,
        min_label_coverage_ratio=min_label_coverage_ratio,
    )
    if label_coverage["accepted_sample_count"] < label_coverage["required_sample_count"]:
        return {
            "status": "not_measured",
            "label_path": str(label_path),
            "mean_error_m": None,
            "threshold_m": threshold_m,
            "wrist_threshold_m": wrist_threshold_m,
            "wrist_mean_error_m": None,
            "wrist_joint_count": 0,
            "sample_count": 0,
            "joint_count": 0,
            "prediction_source": prediction_source,
            "blockers": ["world_mpjpe_label_coverage_too_low"],
            "notes": [
                "BODY world-joint labels are reviewed but too sparse for the scheduled BODY player-frames",
            ],
            "label_import": label_import,
            "label_coverage": label_coverage,
        }

    errors: list[float] = []
    body_feet_errors: list[float] = []
    wrist_errors: list[float] = []
    unmatched_samples = 0
    for sample in samples:
        prediction = prediction_index.get((sample["frame_index"], sample["player_id"]))
        if prediction is None:
            unmatched_samples += 1
            continue
        sample_errors = _joint_errors(prediction, sample["joints_world"])
        errors.extend(sample_errors)
        wrist_indices = _wrist_joint_indices(sample.get("joint_names"))
        if wrist_indices:
            wrist_error_indices = {index for index in wrist_indices if index < len(sample_errors)}
            wrist_errors.extend(sample_errors[index] for index in sorted(wrist_error_indices))
            body_feet_errors.extend(error for index, error in enumerate(sample_errors) if index not in wrist_error_indices)
        else:
            body_feet_errors.extend(sample_errors)

    if not errors:
        return {
            "status": "fail",
            "label_path": str(label_path),
            "mean_error_m": None,
            "threshold_m": threshold_m,
            "body_feet_mean_error_m": None,
            "body_feet_threshold_m": threshold_m,
            "wrist_threshold_m": wrist_threshold_m,
            "wrist_mean_error_m": None,
            "wrist_joint_count": 0,
            "sample_count": 0,
            "joint_count": 0,
            "prediction_source": prediction_source,
            "blockers": ["world_mpjpe_no_matching_predictions"],
            "notes": [f"{unmatched_samples} label samples had no matching BODY prediction"],
            "label_import": label_import,
            "label_coverage": label_coverage,
        }

    mean_error = round(sum(errors) / len(errors), 6)
    body_feet_mean_error = round(sum(body_feet_errors) / len(body_feet_errors), 6) if body_feet_errors else None
    wrist_mean_error = round(sum(wrist_errors) / len(wrist_errors), 6) if wrist_errors else None
    body_feet_passed = body_feet_mean_error is not None and body_feet_mean_error <= threshold_m
    wrist_passed = wrist_mean_error is None or wrist_mean_error <= wrist_threshold_m
    blockers = []
    if not body_feet_passed:
        blockers.append("world_mpjpe_gate_failed")
    if not wrist_passed:
        blockers.append("wrist_mpjpe_gate_failed")
    return {
        "status": "pass" if not blockers else "fail",
        "label_path": str(label_path),
        "mean_error_m": mean_error,
        "threshold_m": threshold_m,
        "body_feet_mean_error_m": body_feet_mean_error,
        "body_feet_threshold_m": threshold_m,
        "wrist_threshold_m": wrist_threshold_m,
        "wrist_mean_error_m": wrist_mean_error,
        "wrist_joint_count": len(wrist_errors),
        "sample_count": len(samples) - unmatched_samples,
        "joint_count": len(errors),
        "prediction_source": prediction_source,
        "blockers": blockers,
        "notes": [f"{unmatched_samples} label samples had no matching BODY prediction"] if unmatched_samples else [],
        "label_import": label_import,
        "label_coverage": {**label_coverage, "matched_sample_count": len(samples) - unmatched_samples},
    }


def _body_grounding_quality_status(*, run_dir: Path) -> dict[str, Any]:
    path = run_dir / BODY_GROUNDING_QUALITY_FILENAME
    if not path.is_file():
        pipeline_run_path = run_dir / "pipeline_run.json"
        pipeline_run = _read_optional_json(pipeline_run_path)
        body_metrics = _pipeline_run_body_stage_metrics(pipeline_run)
        if "max_foot_lock_slide_m" in body_metrics:
            payload = build_body_grounding_quality(
                clip=str(pipeline_run.get("clip", "")) if isinstance(pipeline_run, Mapping) else "",
                grounding_metrics=body_metrics,
            )
            return _body_grounding_quality_payload_status(
                payload,
                path=pipeline_run_path,
                source="pipeline_run_body_stage_metrics",
            )
        return {
            "status": "not_measured",
            "path": "",
            "source": "",
            "foot_slide_gate": {
                "name": "foot_slide_max_m",
                "threshold_m": None,
                "value_m": None,
                "passed": False,
            },
            "blockers": ["missing_body_grounding_quality_gate"],
            "notes": ["no BODY grounding-quality artifact found"],
        }

    payload = _read_json(path)
    return _body_grounding_quality_payload_status(
        payload,
        path=path,
        source="body_grounding_quality_json",
    )


def _body_grounding_quality_payload_status(
    payload: Mapping[str, Any],
    *,
    path: Path,
    source: str,
) -> dict[str, Any]:
    status = str(payload.get("status", ""))
    foot_slide_gate = _mapping(payload, "foot_slide_gate")
    blockers = _string_list(payload.get("blockers"))
    if not blockers:
        if status == "fail" or foot_slide_gate.get("passed") is False:
            blockers = ["foot_slide_gate_failed"]
        elif status == "blocked":
            blockers = ["missing_foot_slide_metric"]
    gate_status = status if status in {"pass", "fail", "blocked"} else "not_measured"
    return {
        "status": gate_status,
        "path": str(path),
        "source": source,
        "artifact_type": str(payload.get("artifact_type", "")),
        "schema_version": payload.get("schema_version"),
        "foot_slide_gate": {
            "name": str(foot_slide_gate.get("name", "foot_slide_max_m")),
            "threshold_m": _maybe_float(foot_slide_gate.get("threshold_m")),
            "value_m": _maybe_float(foot_slide_gate.get("value_m")),
            "passed": foot_slide_gate.get("passed") is True,
        },
        "blockers": _dedupe(blockers),
        "notes": _string_list(payload.get("notes")),
    }


def _pipeline_run_body_stage_metrics(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    stages = payload.get("stages") if isinstance(payload, Mapping) else None
    if not isinstance(stages, list):
        return {}
    for stage in stages:
        if not isinstance(stage, Mapping) or stage.get("stage") != "body":
            continue
        metrics = stage.get("metrics")
        return dict(metrics) if isinstance(metrics, Mapping) else {}
    return {}


def _tracking_identity_quality_status(*, run_dir: Path, labels_dir: Path, clip: str) -> dict[str, Any]:
    candidate_paths = _tracking_identity_candidate_paths(run_dir=run_dir, labels_dir=labels_dir)
    expected_paths = [str(path) for path in candidate_paths]
    for path in candidate_paths:
        if not path.is_file():
            continue
        payload = _read_json(path)
        artifact_type = str(payload.get("artifact_type", ""))
        if artifact_type == "racketsport_person_track_gt_score":
            return _tracking_identity_score_status(
                payload,
                path=path,
                expected_paths=expected_paths,
                run_dir=run_dir,
                source="person_track_gt_score",
            )
        if artifact_type == "racketsport_person_track_gt_scoring_report":
            return _tracking_identity_scoring_report_status(
                payload,
                path=path,
                expected_paths=expected_paths,
                run_dir=run_dir,
                clip=clip,
            )
        return {
            "status": "blocked",
            "path": str(path),
            "expected_paths": expected_paths,
            "source": "",
            "artifact_type": artifact_type,
            "id_switches": None,
            "identity_switch_event_count": None,
            "blockers": ["invalid_person_track_identity_artifact"],
            "notes": [f"unrecognized person-track identity artifact type {artifact_type!r}"],
        }
    return {
        "status": "not_measured",
        "path": "",
        "expected_paths": expected_paths,
        "source": "",
        "artifact_type": "",
        "id_switches": None,
        "identity_switch_event_count": None,
        "blockers": ["missing_person_track_identity_gate"],
        "notes": ["no person-track GT identity score artifact found"],
    }


def _tracking_identity_candidate_paths(*, run_dir: Path, labels_dir: Path) -> list[Path]:
    return _unique_paths(
        [
            run_dir / PERSON_TRACK_GT_SCORE_FILENAME,
            run_dir / PERSON_TRACK_GT_SCORING_REPORT_FILENAME,
            labels_dir / PERSON_TRACK_GT_SCORE_FILENAME,
            labels_dir / PERSON_TRACK_GT_SCORING_REPORT_FILENAME,
            labels_dir / "labels" / PERSON_TRACK_GT_SCORE_FILENAME,
            labels_dir / "labels" / PERSON_TRACK_GT_SCORING_REPORT_FILENAME,
        ]
    )


def _tracking_identity_score_status(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_paths: list[str],
    run_dir: Path,
    source: str,
) -> dict[str, Any]:
    score_tracks_path = str(payload.get("tracks_path", ""))
    body_tracks_path = run_dir / "tracks.json"
    if not _same_artifact_path(score_tracks_path, body_tracks_path):
        return {
            "status": "blocked",
            "path": str(path),
            "expected_paths": expected_paths,
            "source": source,
            "artifact_type": str(payload.get("artifact_type", "")),
            "id_switches": None,
            "identity_switch_event_count": None,
            "score_tracks_path": score_tracks_path,
            "body_tracks_path": str(body_tracks_path),
            "blockers": ["person_track_identity_score_not_for_body_tracks"],
            "notes": ["person-track identity score does not reference this BODY run's tracks.json"],
        }
    id_switches = _maybe_int(payload.get("id_switches"))
    if id_switches is None:
        return {
            "status": "blocked",
            "path": str(path),
            "expected_paths": expected_paths,
            "source": source,
            "artifact_type": str(payload.get("artifact_type", "")),
            "id_switches": None,
            "identity_switch_event_count": _maybe_int(payload.get("identity_switch_event_count")),
            "score_tracks_path": score_tracks_path,
            "body_tracks_path": str(body_tracks_path),
            "blockers": ["person_track_identity_score_missing_id_switches"],
            "notes": ["person-track identity score lacks id_switches"],
        }
    blockers = ["person_track_identity_switches_present"] if id_switches > 0 else []
    return {
        "status": "pass" if not blockers else "fail",
        "path": str(path),
        "expected_paths": expected_paths,
        "source": source,
        "artifact_type": str(payload.get("artifact_type", "")),
        "id_switches": id_switches,
        "identity_switch_event_count": _non_negative_int(payload.get("identity_switch_event_count")),
        "score_tracks_path": score_tracks_path,
        "body_tracks_path": str(body_tracks_path),
        "identity_switch_events": payload.get("identity_switch_events", []) if isinstance(payload.get("identity_switch_events"), list) else [],
        "blockers": blockers,
        "notes": [],
    }


def _tracking_identity_scoring_report_status(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_paths: list[str],
    run_dir: Path,
    clip: str,
) -> dict[str, Any]:
    rows = _matching_tracking_score_rows(payload, run_dir=run_dir, clip=clip)
    if not rows:
        return {
            "status": "blocked",
            "path": str(path),
            "expected_paths": expected_paths,
            "source": "person_track_gt_scoring_report",
            "artifact_type": str(payload.get("artifact_type", "")),
            "id_switches": None,
            "identity_switch_event_count": None,
            "blockers": ["person_track_identity_score_not_for_body_tracks"],
            "notes": ["person-track GT scoring report has no row for this BODY run's tracks.json"],
        }
    id_switches = sum(_non_negative_int(row.get("id_switches")) for row in rows)
    event_count = sum(_non_negative_int(row.get("identity_switch_event_count")) for row in rows)
    blockers = ["person_track_identity_switches_present"] if id_switches > 0 else []
    return {
        "status": "pass" if not blockers else "fail",
        "path": str(path),
        "expected_paths": expected_paths,
        "source": "person_track_gt_scoring_report",
        "artifact_type": str(payload.get("artifact_type", "")),
        "id_switches": id_switches,
        "identity_switch_event_count": event_count,
        "matched_row_count": len(rows),
        "score_tracks_path": str(rows[0].get("tracks_path", "")),
        "body_tracks_path": str(run_dir / "tracks.json"),
        "blockers": blockers,
        "notes": [],
    }


def _matching_tracking_score_rows(
    payload: Mapping[str, Any],
    *,
    run_dir: Path,
    clip: str,
) -> list[Mapping[str, Any]]:
    sources = payload.get("sources")
    if not isinstance(sources, list):
        return []
    body_tracks_path = run_dir / "tracks.json"
    rows: list[Mapping[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        source_rows = source.get("rows")
        if not isinstance(source_rows, list):
            continue
        for row in source_rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("clip_id", "")) != clip:
                continue
            if _same_artifact_path(row.get("tracks_path"), body_tracks_path):
                rows.append(row)
    return rows


def _same_artifact_path(value: Any, target: Path) -> bool:
    if not value:
        return False
    candidate = Path(str(value))
    if candidate == target:
        return True
    try:
        if candidate.resolve() == target.resolve():
            return True
    except OSError:
        pass
    candidate_text = str(candidate).strip()
    target_text = str(target).strip()
    return candidate_text == target_text or candidate_text.endswith(target_text) or target_text.endswith(candidate_text)


def _full_clip_gate_status(*, run_dir: Path, labels_dir: Path) -> dict[str, Any]:
    candidate_paths = _full_clip_gate_candidate_paths(run_dir=run_dir, labels_dir=labels_dir)
    expected_paths = [str(path) for path in candidate_paths]
    path = next((candidate for candidate in candidate_paths if candidate.is_file()), candidate_paths[0])
    if not path.is_file():
        return {
            "status": "not_measured",
            "path": "",
            "expected_paths": expected_paths,
            "required_fields": list(FULL_CLIP_GATE_REQUIRED_FIELDS),
            "passed": None,
            "coverage": None,
            "min_coverage": None,
            "contact_mesh_coverage": None,
            "latency_seconds_per_video_minute": None,
            "evaluated_frame_count": 0,
            "scheduled_contact_count": 0,
            "contact_mesh_frame_count": 0,
            "mesh_unavailable_contact_count": 0,
            "fallback_spliced_contact_count": 0,
            "contact_mesh_accounted_count": 0,
            "clip_duration_s": None,
            "body_runtime_seconds": None,
            "blockers": ["missing_full_clip_body_gate"],
            "notes": ["no full-clip BODY gate artifact found"],
        }

    payload = _read_json(path)
    summary = _mapping(payload, "summary")
    passed = payload.get("passed")
    if not isinstance(passed, bool):
        return {
            "status": "fail",
            "path": str(path),
            "expected_paths": expected_paths,
            "required_fields": list(FULL_CLIP_GATE_REQUIRED_FIELDS),
            "artifact_type": str(payload.get("artifact_type", "")),
            "schema_version": payload.get("schema_version"),
            "passed": None,
            "coverage": payload.get("coverage"),
            "min_coverage": _maybe_float(payload.get("min_coverage")),
            "contact_mesh_coverage": _maybe_float(payload.get("contact_mesh_coverage")),
            "latency_seconds_per_video_minute": _maybe_float(payload.get("latency_seconds_per_video_minute")),
            "evaluated_frame_count": _non_negative_int(payload.get("evaluated_frame_count")),
            "scheduled_contact_count": _non_negative_int(summary.get("scheduled_contact_count")),
            "contact_mesh_frame_count": _non_negative_int(summary.get("contact_mesh_frame_count")),
            "mesh_unavailable_contact_count": _non_negative_int(summary.get("mesh_unavailable_contact_count")),
            "fallback_spliced_contact_count": _non_negative_int(summary.get("fallback_spliced_contact_count")),
            "contact_mesh_accounted_count": _non_negative_int(summary.get("contact_mesh_accounted_count")),
            "clip_duration_s": _maybe_float(summary.get("clip_duration_s")),
            "body_runtime_seconds": _maybe_float(summary.get("body_runtime_seconds")),
            "blockers": ["invalid_full_clip_body_gate"],
            "notes": ["body_full_clip_gate.json must contain boolean passed"],
        }
    return {
        "status": "pass" if passed else "fail",
        "path": str(path),
        "expected_paths": expected_paths,
        "required_fields": list(FULL_CLIP_GATE_REQUIRED_FIELDS),
        "artifact_type": str(payload.get("artifact_type", "")),
        "schema_version": payload.get("schema_version"),
        "passed": passed,
        "coverage": payload.get("coverage"),
        "min_coverage": _maybe_float(payload.get("min_coverage")),
        "contact_mesh_coverage": _maybe_float(payload.get("contact_mesh_coverage")),
        "latency_seconds_per_video_minute": _maybe_float(payload.get("latency_seconds_per_video_minute")),
        "evaluated_frame_count": _non_negative_int(payload.get("evaluated_frame_count")),
        "scheduled_contact_count": _non_negative_int(summary.get("scheduled_contact_count")),
        "contact_mesh_frame_count": _non_negative_int(summary.get("contact_mesh_frame_count")),
        "mesh_unavailable_contact_count": _non_negative_int(summary.get("mesh_unavailable_contact_count")),
        "fallback_spliced_contact_count": _non_negative_int(summary.get("fallback_spliced_contact_count")),
        "contact_mesh_accounted_count": _non_negative_int(summary.get("contact_mesh_accounted_count")),
        "clip_duration_s": _maybe_float(summary.get("clip_duration_s")),
        "body_runtime_seconds": _maybe_float(summary.get("body_runtime_seconds")),
        "blockers": [] if passed else ["full_clip_body_gate_failed"],
        "notes": [],
    }


def _body_packet_quality_status(*, run_dir: Path) -> dict[str, Any]:
    path = run_dir / BODY_PACKET_QUALITY_FILENAME
    if not path.is_file():
        return {
            "status": "not_measured",
            "path": "",
            "payload_status": "",
            "usable_for_review": False,
            "joint_source": "",
            "joint_frame_count": 0,
            "scheduled_player_frame_count": 0,
            "schedule_coverage_ratio": None,
            "quality_blockers": [],
            "warnings": [],
            "blockers": [],
            "notes": ["no compact BODY packet-quality sidecar found"],
        }

    payload = _read_json(path)
    summary = _mapping(payload, "summary")
    quality_blockers = _string_list(payload.get("quality_blockers"))
    warnings = _string_list(payload.get("warnings"))
    usable_for_review = bool(payload.get("usable_for_review"))
    payload_status = str(payload.get("status", ""))
    notes: list[str] = []
    blockers: list[str] = []
    if quality_blockers:
        blockers.extend(quality_blockers)
        notes.append("compact BODY packet quality has structural blockers")
        status = "blocked"
    elif payload_status == "quality_blocked" or not usable_for_review:
        blockers.append("body_packet_quality_blocked")
        notes.append(f"compact BODY packet quality status is {payload_status or 'unknown'}")
        status = "blocked"
    elif warnings:
        notes.append("compact BODY packet quality has warnings")
        status = "warning"
    elif payload_status == "quality_checked_needs_accuracy_gate" or usable_for_review:
        status = "pass"
    else:
        notes.append(f"compact BODY packet quality status is {payload_status or 'unknown'}")
        status = "not_measured"

    return {
        "status": status,
        "path": str(path),
        "payload_status": payload_status,
        "usable_for_review": usable_for_review,
        "joint_source": str(summary.get("joint_source", "")),
        "joint_frame_count": _non_negative_int(summary.get("joint_frame_count")),
        "scheduled_player_frame_count": _non_negative_int(summary.get("scheduled_player_frame_count")),
        "schedule_coverage_ratio": _maybe_float(summary.get("schedule_coverage_ratio")),
        "quality_blockers": quality_blockers,
        "warnings": warnings,
        "blockers": _dedupe(blockers),
        "notes": notes,
    }


def _body_review_overlay_alignment_status(*, run_dir: Path) -> dict[str, Any]:
    path = run_dir / BODY_REVIEW_OVERLAY_INDEX_PATH
    if not path.is_file():
        return {
            "status": "not_measured",
            "path": "",
            "floor_anchor_projection_failed_count": 0,
            "floor_anchor_projection_warning_count": 0,
            "alignment_failed_count": 0,
            "alignment_warning_count": 0,
            "rendered_count": 0,
            "sample_count": 0,
            "blockers": [],
            "notes": ["no BODY selected-sample overlay index found"],
        }

    payload = _read_json(path)
    status = str(payload.get("status", ""))
    warning_samples = _overlay_warning_samples(payload)
    unresolved_warning_samples = [
        sample for sample in warning_samples if not _overlay_warning_review_resolved(sample)
    ]
    resolved_warning_sample_count = len(warning_samples) - len(unresolved_warning_samples)
    floor_anchor_projection_failed_count = _non_negative_int(payload.get("floor_anchor_projection_failed_count"))
    floor_anchor_projection_warning_count = _non_negative_int(payload.get("floor_anchor_projection_warning_count"))
    alignment_failed_count = _non_negative_int(payload.get("alignment_failed_count"))
    alignment_warning_count = _non_negative_int(payload.get("alignment_warning_count"))
    competing_player_warning_count = _non_negative_int(payload.get("competing_player_warning_count"))
    has_warning_status = status in {"ready_for_review_with_alignment_warnings", "ready_for_review_with_overlay_warnings"}
    has_warning_count = (
        floor_anchor_projection_warning_count > 0
        or alignment_warning_count > 0
        or competing_player_warning_count > 0
    )
    has_unresolved_warnings = bool(unresolved_warning_samples) or (
        (has_warning_status or has_warning_count) and not warning_samples
    )
    blockers = []
    notes = []
    if status == "rendered_floor_anchor_projection_failed" or floor_anchor_projection_failed_count > 0:
        blockers.append("body_floor_anchor_projection_failed")
        notes.append("tracked player floor anchors do not project back to their image-space person boxes")
        gate_status = "blocked"
    if status == "rendered_alignment_failed" or alignment_failed_count > 0:
        blockers.append("body_joint_overlay_alignment_failed")
        notes.append("selected-sample projected BODY joints do not align with tracked person boxes")
        gate_status = "blocked"
    if blockers:
        gate_status = "blocked"
    elif status.startswith("blocked_"):
        blockers.append("body_joint_overlay_not_ready")
        notes.append(f"BODY selected-sample overlay status is {status}")
        gate_status = "blocked"
    elif has_unresolved_warnings:
        blockers.append("body_joint_overlay_warning_review_required")
        notes.append("selected-sample overlay has floor-anchor, joint-vs-box, or competing-player warnings")
        gate_status = "warning"
    elif status == "ready_for_review" or ((has_warning_status or has_warning_count) and warning_samples):
        gate_status = "pass"
    else:
        gate_status = "not_measured"
        notes.append(f"BODY selected-sample overlay status is {status or 'unknown'}")

    return {
        "status": gate_status,
        "path": str(path),
        "payload_status": status,
        "floor_anchor_projection_failed_count": floor_anchor_projection_failed_count,
        "floor_anchor_projection_warning_count": floor_anchor_projection_warning_count,
        "alignment_failed_count": alignment_failed_count,
        "alignment_warning_count": alignment_warning_count,
        "competing_player_warning_count": competing_player_warning_count,
        "rendered_count": _non_negative_int(payload.get("rendered_count")),
        "sample_count": _non_negative_int(payload.get("sample_count")),
        "warning_sample_count": len(warning_samples),
        "resolved_warning_sample_count": resolved_warning_sample_count,
        "unresolved_warning_sample_count": len(unresolved_warning_samples),
        "warning_samples": warning_samples,
        "unresolved_warning_samples": unresolved_warning_samples,
        "blockers": blockers,
        "notes": notes,
    }


def _overlay_warning_samples(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    overlays = payload.get("overlays")
    if not isinstance(overlays, list):
        return []
    samples: list[dict[str, Any]] = []
    for item in overlays:
        if not isinstance(item, Mapping):
            continue
        warnings = _string_list(item.get("warnings"))
        if not warnings:
            continue
        alignment = _mapping(item, "joint_bbox_alignment")
        sample = {
            "sample_id": str(item.get("sample_id", "")),
            "frame_index": _maybe_int(item.get("frame_index")),
            "player_id": _maybe_int(item.get("player_id")),
            "overlay_path": str(item.get("overlay_path", "")),
            "warnings": warnings,
            "warning_review_status": str(item.get("warning_review_status", "")),
            "warning_review_note": str(item.get("warning_review_note", "")),
            "joint_bbox_alignment_status": str(alignment.get("status", "")),
            "center_delta_px": _maybe_float(alignment.get("center_delta_px")),
            "center_delta_bbox_diag": _maybe_float(alignment.get("center_delta_bbox_diag")),
            "containment_ratio": _maybe_float(alignment.get("containment_ratio")),
        }
        competing_alignment = _mapping(item, "competing_player_alignment")
        if competing_alignment:
            sample.update(
                {
                    "competing_player_alignment_status": str(competing_alignment.get("status", "")),
                    "competing_player_id": _maybe_int(competing_alignment.get("best_player_id")),
                    "competing_player_containment_ratio": _maybe_float(
                        competing_alignment.get("best_player_containment_ratio")
                    ),
                    "competing_player_score_margin": _maybe_float(competing_alignment.get("score_margin")),
                }
            )
        samples.append(sample)
    return sorted(samples, key=_overlay_warning_sort_key)


def _overlay_warning_review_resolved(sample: Mapping[str, Any]) -> bool:
    return str(sample.get("warning_review_status", "")).strip().lower() in RESOLVED_WARNING_REVIEW_STATUSES


def _overlay_warning_sort_key(sample: Mapping[str, Any]) -> tuple[float, float, int, int, str]:
    containment = _maybe_float(sample.get("containment_ratio"))
    center_delta = _maybe_float(sample.get("center_delta_px"))
    frame_index = _maybe_int(sample.get("frame_index"))
    player_id = _maybe_int(sample.get("player_id"))
    return (
        containment if containment is not None else math.inf,
        -(center_delta if center_delta is not None else -math.inf),
        frame_index if frame_index is not None else 10**12,
        player_id if player_id is not None else 10**12,
        str(sample.get("sample_id", "")),
    )


def _body_label_review_status(*, run_dir: Path, labels_dir: Path) -> dict[str, Any]:
    bundle_path = run_dir / BODY_LABEL_REVIEW_BUNDLE_PATH
    template_path = run_dir / BODY_LABEL_REVIEW_TEMPLATE_PATH
    finalization_report_path = run_dir / BODY_LABEL_FINALIZATION_REPORT_PATH
    bundle = _read_optional_json(bundle_path)
    template = _read_optional_json(template_path)
    finalization = _read_optional_json(finalization_report_path)
    label_path = _find_body_label_path(labels_dir)

    bundle_status = str(bundle.get("status", "")) if isinstance(bundle, Mapping) else ""
    finalization_status = str(finalization.get("status", "")) if isinstance(finalization, Mapping) else ""
    finalization_blockers = _string_list(finalization.get("blockers")) if isinstance(finalization, Mapping) else []
    blockers: list[str] = []
    notes: list[str] = []

    if finalization is not None and finalization_status != "finalized":
        status = "blocked_finalization"
        blockers.append("body_world_label_finalization_blocked")
        notes.append("BODY world-label finalization report is blocked")
    elif finalization_status == "finalized":
        status = "finalized"
        notes.append("BODY world-label finalization report is finalized")
    elif bundle is not None:
        if bundle_status == "ready_for_review":
            status = "ready_for_review"
            notes.append("BODY world-label review bundle is ready; final labels are still required for world-MPJPE")
        elif bundle_status.startswith("blocked_") or bundle_status in {"no_selected_samples"}:
            status = "blocked_review_bundle"
            blockers.append("body_world_label_review_not_ready")
            notes.append(f"BODY world-label review bundle status is {bundle_status}")
        else:
            status = bundle_status or "review_bundle_present"
            notes.append(f"BODY world-label review bundle status is {status}")
    elif template is not None:
        status = "template_only"
        notes.append("BODY world-label template exists without a review bundle manifest")
    elif label_path is not None:
        status = "final_label_candidate_present"
        notes.append("BODY world-label candidate exists; world-MPJPE import status decides whether it is trusted")
    else:
        status = "not_started"
        notes.append("no BODY world-label review bundle found")

    return {
        "status": status,
        "bundle_path": str(bundle_path) if bundle_path.is_file() else "",
        "label_template_path": str(template_path) if template_path.is_file() else "",
        "finalization_report_path": str(finalization_report_path) if finalization_report_path.is_file() else "",
        "final_label_path": _final_label_path(
            labels_dir=labels_dir,
            label_path=label_path,
            bundle=bundle,
        ),
        "finalize_command": str(bundle.get("finalize_command", "")) if isinstance(bundle, Mapping) else "",
        "selected_sample_count": _non_negative_int(bundle.get("selected_sample_count")) if isinstance(bundle, Mapping) else 0,
        "required_sample_count": _non_negative_int(bundle.get("required_sample_count")) if isinstance(bundle, Mapping) else 0,
        "missing_frame_count": _non_negative_int(bundle.get("missing_frame_count")) if isinstance(bundle, Mapping) else 0,
        "missing_selected_sample_count": _non_negative_int(bundle.get("missing_selected_sample_count"))
        if isinstance(bundle, Mapping)
        else 0,
        "missing_selected_sample_ids": _string_list(bundle.get("missing_selected_sample_ids"))
        if isinstance(bundle, Mapping)
        else [],
        "template_status": str(template.get("status", "")) if isinstance(template, Mapping) else "",
        "template_selected_sample_count": _template_selected_sample_count(template),
        "template_accepted_sample_count": _template_accepted_sample_count(template),
        "template_not_ground_truth": template.get("not_ground_truth") if isinstance(template, Mapping) else None,
        "template_trusted_for_world_mpjpe": template.get("trusted_for_world_mpjpe")
        if isinstance(template, Mapping)
        else None,
        "finalization_status": finalization_status,
        "finalization_selected_sample_count": _non_negative_int(finalization.get("selected_sample_count"))
        if isinstance(finalization, Mapping)
        else 0,
        "finalization_accepted_sample_count": _non_negative_int(finalization.get("accepted_sample_count"))
        if isinstance(finalization, Mapping)
        else 0,
        "finalization_overlay_warning_selected_sample_count": _non_negative_int(
            finalization.get("overlay_warning_selected_sample_count")
        )
        if isinstance(finalization, Mapping)
        else 0,
        "finalization_overlay_warning_selected_sample_ids": _string_list(
            finalization.get("overlay_warning_selected_sample_ids")
        )
        if isinstance(finalization, Mapping)
        else [],
        "finalization_blockers": finalization_blockers,
        "blockers": _dedupe(blockers),
        "notes": notes,
    }


def _final_label_path(
    *,
    labels_dir: Path,
    label_path: Path | None,
    bundle: Mapping[str, Any] | None,
) -> str:
    if label_path is not None:
        return str(label_path)
    if isinstance(bundle, Mapping):
        final_label_path = str(bundle.get("final_label_path", ""))
        if final_label_path:
            return final_label_path
    return str(labels_dir / "labels" / "body_world_joints.json")


def _template_selected_sample_count(template: Mapping[str, Any] | None) -> int:
    selected = template.get("selected_sample_ids") if isinstance(template, Mapping) else None
    if not isinstance(selected, list):
        return 0
    return sum(1 for sample_id in selected if str(sample_id))


def _template_accepted_sample_count(template: Mapping[str, Any] | None) -> int:
    samples = template.get("samples") if isinstance(template, Mapping) else None
    if not isinstance(samples, list):
        return 0
    return sum(1 for sample in samples if isinstance(sample, Mapping) and sample.get("accepted") is True)


def _discover_clips(root_path: Path) -> list[str]:
    if not root_path.is_dir():
        return []
    return sorted(path.name for path in root_path.iterdir() if path.is_dir() and _looks_like_clip_run_dir(path))


def _looks_like_clip_run_dir(path: Path) -> bool:
    return any(
        (path / filename).is_file()
        for filename in (
            "body_mesh_readiness.json",
            "body_compute_execution.json",
            "pipeline_readiness_e2e.json",
            "smpl_motion.json",
            "skeleton3d.json",
        )
    )


def _aggregate_status(clips: list[Mapping[str, Any]]) -> str:
    if any(clip.get("status") == "fail" for clip in clips):
        return "fail"
    if any(clip.get("status") == "blocked" for clip in clips):
        return "blocked"
    if clips and all(clip.get("status") == "pass" for clip in clips):
        return "pass"
    return "not_measured"


def _clip_status(
    mesh_smoke: Mapping[str, Any],
    world_mpjpe: Mapping[str, Any],
    body_grounding_quality: Mapping[str, Any],
    tracking_identity_quality: Mapping[str, Any],
    full_clip_gate: Mapping[str, Any],
    blockers: list[str],
) -> str:
    if (
        mesh_smoke.get("status") == "fail"
        or world_mpjpe.get("status") == "fail"
        or body_grounding_quality.get("status") == "fail"
        or tracking_identity_quality.get("status") == "fail"
        or full_clip_gate.get("status") == "fail"
    ):
        return "fail"
    if blockers:
        return "blocked"
    if (
        mesh_smoke.get("status") == "pass"
        and world_mpjpe.get("status") == "pass"
        and body_grounding_quality.get("status") == "pass"
        and tracking_identity_quality.get("status") == "pass"
        and full_clip_gate.get("status") == "pass"
    ):
        return "pass"
    return "not_measured"


def _find_body_label_path(labels_dir: Path) -> Path | None:
    for path in _body_label_candidate_paths(labels_dir):
        if path.is_file():
            return path
    return None


def _body_label_candidate_paths(labels_dir: Path) -> list[Path]:
    labels_subdir = labels_dir / "labels"
    return [
        *[labels_dir / filename for filename in BODY_WORLD_LABEL_FILENAMES],
        *[labels_subdir / filename for filename in BODY_WORLD_LABEL_FILENAMES],
    ]


def _missing_body_label_import_status(labels_dir: Path) -> dict[str, Any]:
    return {
        "status": "missing",
        "path": "",
        "expected_paths": [str(path) for path in _body_label_candidate_paths(labels_dir)],
        "artifact_type": "",
        "payload_status": "",
        "not_ground_truth": None,
        "accepted_sample_count": 0,
        "blockers": ["missing_world_mpjpe_gate"],
        "notes": ["no BODY world-joint labels found"],
    }


def _body_label_import_status(
    *,
    labels_dir: Path,
    label_path: Path,
    payload: Mapping[str, Any],
    accepted_sample_count: int,
) -> dict[str, Any]:
    payload_status = str(payload.get("status", ""))
    not_ground_truth = payload.get("not_ground_truth") is True
    result = {
        "status": "present_reviewed",
        "path": str(label_path),
        "expected_paths": [str(path) for path in _body_label_candidate_paths(labels_dir)],
        "artifact_type": str(payload.get("artifact_type", "")),
        "payload_status": payload_status,
        "not_ground_truth": not_ground_truth,
        "accepted_sample_count": accepted_sample_count,
        "blockers": [],
        "notes": [],
    }
    if not_ground_truth:
        result["status"] = "rejected_not_ground_truth"
        result["blockers"] = ["body_world_labels_not_ground_truth"]
        result["notes"] = ["BODY world-joint labels have not_ground_truth=true; world-MPJPE not measured"]
        return result
    if _body_label_payload_requires_independent_source(payload):
        if payload.get("trusted_for_world_mpjpe") is not True:
            result["status"] = "rejected_not_trusted_for_world_mpjpe"
            result["blockers"] = ["body_world_labels_not_trusted_for_world_mpjpe"]
            result["notes"] = [
                "BODY packet-derived labels must be explicitly trusted for world-MPJPE; world-MPJPE not measured"
            ]
            return result
        non_independent_sample_ids = _non_independent_body_label_sample_ids(payload)
        if non_independent_sample_ids:
            result["status"] = "rejected_not_independent_ground_truth"
            result["blockers"] = ["accepted_candidate_labels_not_independent_ground_truth"]
            result["notes"] = [
                "BODY packet-derived labels still use accepted candidate predictions instead of independent ground truth"
            ]
            result["non_independent_sample_ids"] = non_independent_sample_ids
            return result
    if any(token in payload_status.lower() for token in ("draft", "unverified", "teacher")):
        result["status"] = "rejected_unreviewed_status"
        result["blockers"] = ["body_world_labels_not_reviewed"]
        result["notes"] = [f"BODY world-joint label status is {payload_status!r}; world-MPJPE not measured"]
    return result


def _body_label_payload_requires_independent_source(payload: Mapping[str, Any]) -> bool:
    return bool(str(payload.get("source_packet", "")).strip() or str(payload.get("source_template", "")).strip())


def _non_independent_body_label_sample_ids(payload: Mapping[str, Any]) -> list[str]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        return ["<missing_samples>"]
    out: list[str] = []
    for index, sample in enumerate(raw_samples):
        if not isinstance(sample, Mapping) or sample.get("accepted", True) is False:
            continue
        label_source = _body_label_source(sample=sample, payload=payload)
        if label_source.strip().lower() not in INDEPENDENT_BODY_LABEL_SOURCES:
            out.append(_body_label_sample_id(sample, index=index))
    return _dedupe(out)


def _body_label_source(*, sample: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    for key in ("label_source", "ground_truth_source", "independent_label_source"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("label_source", "ground_truth_source", "independent_label_source"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "accepted_candidate_prediction"


def _body_label_sample_id(sample: Mapping[str, Any], *, index: int) -> str:
    sample_id = str(sample.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    frame_index = _maybe_int(sample.get("frame_index"))
    player_id = _maybe_int(sample.get("player_id"))
    if frame_index is not None and player_id is not None:
        return f"frame_{frame_index:06d}_player_{player_id}"
    return f"sample_{index}"


def _full_clip_gate_candidate_paths(*, run_dir: Path, labels_dir: Path) -> list[Path]:
    return _unique_paths(
        [
            run_dir / FULL_CLIP_GATE_FILENAME,
            labels_dir / FULL_CLIP_GATE_FILENAME,
            labels_dir / "labels" / FULL_CLIP_GATE_FILENAME,
        ]
    )


def _unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _label_samples(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        return []
    top_level_joint_names = _joint_names(payload.get("joint_names"))
    samples: list[dict[str, Any]] = []
    for item in raw_samples:
        if not isinstance(item, Mapping) or item.get("accepted", True) is False:
            continue
        frame_index = _maybe_int(item.get("frame_index"))
        player_id = _maybe_int(item.get("player_id"))
        joints = _vectors(item.get("joints_world"))
        if frame_index is None or player_id is None or not joints:
            continue
        samples.append(
            {
                "frame_index": frame_index,
                "player_id": player_id,
                "joints_world": joints,
                "joint_names": _joint_names(item.get("joint_names")) or top_level_joint_names,
            }
        )
    return samples


def _prediction_index(
    *,
    smpl_motion: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
    body_world_label_packet: Mapping[str, Any] | None,
) -> tuple[dict[tuple[int, int], list[tuple[float, float, float]]], str]:
    smpl_index = _players_prediction_index(smpl_motion, fps=_fps(smpl_motion))
    if smpl_index:
        return smpl_index, "smpl_motion"
    skeleton_index = _players_prediction_index(skeleton3d, fps=_fps(smpl_motion) or 30.0)
    if skeleton_index:
        return skeleton_index, "skeleton3d"
    packet_index = _packet_prediction_index(body_world_label_packet)
    if packet_index:
        return packet_index, "body_world_label_packet"
    return {}, ""


def _packet_prediction_index(payload: Mapping[str, Any] | None) -> dict[tuple[int, int], list[tuple[float, float, float]]]:
    raw_samples = payload.get("samples") if isinstance(payload, Mapping) else None
    if not isinstance(raw_samples, list):
        return {}
    out: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for sample in raw_samples:
        if not isinstance(sample, Mapping):
            continue
        frame_index = _maybe_int(sample.get("frame_index"))
        player_id = _maybe_int(sample.get("player_id"))
        joints = _vectors(sample.get("predicted_joints_world"))
        if frame_index is None or player_id is None or not joints:
            continue
        out[(frame_index, player_id)] = joints
    return out


def _world_label_coverage(
    *,
    samples: list[dict[str, Any]],
    prediction_index: Mapping[tuple[int, int], list[tuple[float, float, float]]],
    body_compute_execution: Mapping[str, Any] | None,
    min_label_samples: int,
    min_label_coverage_ratio: float,
) -> dict[str, Any]:
    expected_sample_count = _expected_world_label_sample_count(
        prediction_index=prediction_index,
        body_compute_execution=body_compute_execution,
    )
    required_sample_count = _required_world_label_sample_count(
        expected_sample_count=expected_sample_count,
        min_label_samples=min_label_samples,
        min_label_coverage_ratio=min_label_coverage_ratio,
    )
    accepted_sample_count = len(samples)
    accepted_frame_count = len({sample["frame_index"] for sample in samples})
    accepted_player_count = len({sample["player_id"] for sample in samples})
    return {
        "accepted_sample_count": accepted_sample_count,
        "accepted_frame_count": accepted_frame_count,
        "accepted_player_count": accepted_player_count,
        "expected_sample_count": expected_sample_count,
        "required_sample_count": required_sample_count,
        "min_sample_count": max(0, int(min_label_samples)),
        "min_coverage_ratio": max(0.0, float(min_label_coverage_ratio)),
        "accepted_coverage_ratio": round(accepted_sample_count / expected_sample_count, 6)
        if expected_sample_count > 0
        else None,
    }


def _expected_world_label_sample_count(
    *,
    prediction_index: Mapping[tuple[int, int], list[tuple[float, float, float]]],
    body_compute_execution: Mapping[str, Any] | None,
) -> int:
    execution_summary = _mapping(body_compute_execution, "summary")
    scheduled_player_frames = _non_negative_int(execution_summary.get("scheduled_player_frame_count"))
    if scheduled_player_frames > 0:
        return scheduled_player_frames
    return len(prediction_index)


def _required_world_label_sample_count(
    *,
    expected_sample_count: int,
    min_label_samples: int,
    min_label_coverage_ratio: float,
) -> int:
    if expected_sample_count <= 0:
        return max(0, int(min_label_samples))
    min_samples = max(1, int(min_label_samples))
    ratio_samples = max(1, math.ceil(expected_sample_count * max(0.0, float(min_label_coverage_ratio))))
    return min(expected_sample_count, max(min_samples, ratio_samples))


def _players_prediction_index(payload: Mapping[str, Any] | None, *, fps: float) -> dict[tuple[int, int], list[tuple[float, float, float]]]:
    players = payload.get("players") if isinstance(payload, Mapping) else None
    if not isinstance(players, list):
        return {}
    out: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = _maybe_int(player.get("id"))
        frames = player.get("frames")
        if player_id is None or not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            joints = _vectors(frame.get("joints_world"))
            if not joints:
                continue
            frame_index = _maybe_int(frame.get("frame_index"))
            if frame_index is None:
                t = _maybe_float(frame.get("t"))
                if t is None:
                    continue
                frame_index = int(round(t * fps))
            out[(frame_index, player_id)] = joints
    return out


def _joint_errors(
    prediction: list[tuple[float, float, float]],
    label: list[tuple[float, float, float]],
) -> list[float]:
    count = min(len(prediction), len(label))
    return [_distance(prediction[index], label[index]) for index in range(count)]


def _wrist_joint_indices(joint_names: Any) -> list[int]:
    names = _joint_names(joint_names)
    return [index for index, name in enumerate(names) if _is_wrist_joint_name(name)]


def _is_wrist_joint_name(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    return normalized in {"left_wrist", "right_wrist", "lwrist", "rwrist", "wrist"} or normalized.endswith("_wrist")


def _joint_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _smpl_player_frame_count(payload: Mapping[str, Any] | None) -> int:
    return _player_frame_count(payload)


def _skeleton_player_frame_count(payload: Mapping[str, Any] | None) -> int:
    return _player_frame_count(payload)


def _player_frame_count(payload: Mapping[str, Any] | None) -> int:
    players = payload.get("players") if isinstance(payload, Mapping) else None
    if not isinstance(players, list):
        return 0
    count = 0
    for player in players:
        frames = player.get("frames") if isinstance(player, Mapping) else None
        if isinstance(frames, list):
            count += len(frames)
    return count


def _mapping(payload: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    value = payload.get(key) if isinstance(payload, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _read_optional_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _fps(payload: Mapping[str, Any] | None) -> float:
    value = payload.get("fps") if isinstance(payload, Mapping) else None
    fps = _maybe_float(value)
    return fps if fps and fps > 0 else 30.0


def _vectors(value: Any) -> list[tuple[float, float, float]]:
    if not isinstance(value, list):
        return []
    vectors: list[tuple[float, float, float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 3:
            return []
        vector = tuple(_maybe_float(component) for component in item)
        if any(component is None for component in vector):
            return []
        vectors.append((float(vector[0]), float(vector[1]), float(vector[2])))
    return vectors


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _non_negative_int(value: Any) -> int:
    number = _maybe_int(value)
    return max(0, number) if number is not None else 0


def _non_negative_int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        count = _non_negative_int(raw)
        if count > 0:
            out[str(key)] = count
    return out


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
