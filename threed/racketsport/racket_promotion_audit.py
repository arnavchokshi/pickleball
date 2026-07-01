"""Audit whether preview paddle poses leaked into canonical racket_pose.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import RacketCandidates, RacketPose


ARTIFACT_TYPE = "racketsport_racket_promotion_audit"
SCHEMA_VERSION = 1


def build_racket_promotion_audit(
    *,
    clip: str,
    racket_candidates: RacketCandidates | Mapping[str, Any],
    racket_pose_preview: RacketPose | Mapping[str, Any] | None = None,
    racket_pose: RacketPose | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report whether canonical paddle pose promotion is safe to trust."""

    candidates = _candidates(racket_candidates)
    preview = _pose(racket_pose_preview)
    promoted = _pose(racket_pose)

    source_counts = _candidate_source_counts(candidates)
    source_evidence_counts = _source_evidence_counts(source_counts)
    pose_source_counts = _pose_source_counts(promoted)
    unsafe_promoted_sources = {
        source: count
        for source, count in pose_source_counts.items()
        if _is_box_derived_source(source) or _is_preview_pose_source(source)
    }

    candidate_frame_count = sum(source_counts.values())
    box_derived_candidate_frame_count = source_evidence_counts["box_derived"]
    true_corner_frame_count = source_evidence_counts["true_corners_or_pose"]
    reference_gt_frame_count = source_evidence_counts["reference_gt"]
    preview_pose_frame_count = _pose_frame_count(preview)
    promoted_pose_frame_count = _pose_frame_count(promoted)
    unsafe_promoted_frame_count = sum(unsafe_promoted_sources.values())
    canonical_racket_pose_present = promoted_pose_frame_count > 0

    blockers: list[str] = []
    if unsafe_promoted_frame_count:
        blockers.append("box_derived_racket_pose_promoted")
    if box_derived_candidate_frame_count:
        blockers.append("box_derived_candidate_corners")
    if true_corner_frame_count == 0:
        blockers.append("missing_true_paddle_keypoints_or_cad_pose")
    if promoted_pose_frame_count == 0:
        blockers.append("missing_promoted_racket_pose_json")
    if reference_gt_frame_count == 0:
        blockers.append("missing_reference_pose_gt")
    blockers.append("missing_racket_pose_evaluation")

    status = "safe_preview_only"
    if unsafe_promoted_frame_count:
        status = "unsafe_box_derived_promoted"
    elif promoted_pose_frame_count > 0 and true_corner_frame_count > 0 and reference_gt_frame_count == 0:
        status = "pose_present_needs_reference_and_eval"
    elif promoted_pose_frame_count > 0 and true_corner_frame_count > 0 and reference_gt_frame_count > 0:
        status = "pose_present_needs_eval"
    elif promoted_pose_frame_count > 0:
        status = "pose_present_missing_true_evidence"

    warnings = _warnings(
        unsafe_promoted_frame_count=unsafe_promoted_frame_count,
        promoted_pose_frame_count=promoted_pose_frame_count,
        preview_pose_frame_count=preview_pose_frame_count,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": status,
        "not_gate_verified": True,
        "trusted_for_rkt_promotion": False,
        "canonical_racket_pose_present": canonical_racket_pose_present,
        "source_counts": source_counts,
        "pose_source_counts": pose_source_counts,
        "unsafe_promoted_sources": unsafe_promoted_sources,
        "source_evidence_counts": source_evidence_counts,
        "candidate_frame_count": candidate_frame_count,
        "box_derived_candidate_frame_count": box_derived_candidate_frame_count,
        "true_corner_frame_count": true_corner_frame_count,
        "reference_gt_frame_count": reference_gt_frame_count,
        "preview_pose_frame_count": preview_pose_frame_count,
        "promoted_pose_frame_count": promoted_pose_frame_count,
        "unsafe_promoted_frame_count": unsafe_promoted_frame_count,
        "blockers": blockers,
        "warnings": warnings,
        "recommended_next_actions": _recommended_next_actions(blockers),
        "summary": {
            "candidate_player_count": len(candidates.players),
            "candidate_frame_count": candidate_frame_count,
            "box_derived_candidate_frame_count": box_derived_candidate_frame_count,
            "true_corner_frame_count": true_corner_frame_count,
            "reference_gt_frame_count": reference_gt_frame_count,
            "preview_pose_frame_count": preview_pose_frame_count,
            "promoted_pose_frame_count": promoted_pose_frame_count,
            "unsafe_promoted_frame_count": unsafe_promoted_frame_count,
        },
    }


def build_racket_promotion_audit_from_files(
    *,
    clip: str,
    racket_candidates_path: str | Path,
    racket_pose_preview_path: str | Path | None = None,
    racket_pose_path: str | Path | None = None,
) -> dict[str, Any]:
    candidates = RacketCandidates.model_validate(_read_json(racket_candidates_path))
    preview = _read_optional_pose(racket_pose_preview_path)
    promoted = _read_optional_pose(racket_pose_path)
    return build_racket_promotion_audit(
        clip=clip,
        racket_candidates=candidates,
        racket_pose_preview=preview,
        racket_pose=promoted,
    )


def write_racket_promotion_audit(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_source_counts(candidates: RacketCandidates) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in candidates.players:
        for frame in player.frames:
            counts[frame.source] = counts.get(frame.source, 0) + 1
    return dict(sorted(counts.items()))


def _pose_source_counts(pose: RacketPose | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if pose is None:
        return counts
    for player in pose.players:
        for frame in player.frames:
            counts[frame.source] = counts.get(frame.source, 0) + 1
    return dict(sorted(counts.items()))


def _source_evidence_counts(source_counts: Mapping[str, int]) -> dict[str, int]:
    evidence = {
        "box_derived": 0,
        "keypoint_or_mask": 0,
        "reference_gt": 0,
        "synthetic_or_cad": 0,
        "true_corners_or_pose": 0,
    }
    for source, count in source_counts.items():
        kind = _source_evidence_kind(source)
        if kind == "box_derived":
            evidence["box_derived"] += count
            continue
        evidence["true_corners_or_pose"] += count
        evidence[kind] += count
    return evidence


def _source_evidence_kind(source: str) -> str:
    normalized = source.lower()
    if _is_box_derived_source(normalized):
        return "box_derived"
    if any(token in normalized for token in ("synthetic", "blenderproc", "cad")):
        return "synthetic_or_cad"
    if any(token in normalized for token in ("aruco", "april", "tag", "gt", "ground_truth", "reference")):
        return "reference_gt"
    return "keypoint_or_mask"


def _is_box_derived_source(source: str) -> bool:
    normalized = source.lower()
    return normalized.startswith("label_bbox:") or ":label_bbox:" in normalized or "box_corner" in normalized


def _is_preview_pose_source(source: str) -> bool:
    normalized = source.lower()
    return normalized.endswith(":pnp_ippe_preview") or "preview" in normalized


def _pose_frame_count(pose: RacketPose | None) -> int:
    if pose is None:
        return 0
    return sum(len(player.frames) for player in pose.players)


def _warnings(
    *,
    unsafe_promoted_frame_count: int,
    promoted_pose_frame_count: int,
    preview_pose_frame_count: int,
) -> list[str]:
    if unsafe_promoted_frame_count:
        return ["box_derived_racket_pose_promoted", "not_trusted_for_rkt_promotion"]
    if promoted_pose_frame_count:
        return ["not_trusted_for_rkt_promotion"]
    warnings = ["canonical_racket_pose_missing"]
    if preview_pose_frame_count:
        warnings.append("preview_only_not_gate_verified")
    return warnings


def _recommended_next_actions(blockers: list[str]) -> list[str]:
    actions: list[str] = []
    if "box_derived_racket_pose_promoted" in blockers:
        actions.append("remove or quarantine canonical racket_pose.json built from label-box or preview sources")
    if "box_derived_candidate_corners" in blockers:
        actions.append("keep label-box candidates review-only; do not write canonical racket_pose.json from them")
    if "missing_true_paddle_keypoints_or_cad_pose" in blockers:
        actions.append("collect true paddle corners, masks/keypoints, CAD, or reference pose evidence before promotion")
    if "missing_promoted_racket_pose_json" in blockers:
        actions.append("write canonical racket_pose.json only from non-box evidence that passes fail-closed pose checks")
    if "missing_reference_pose_gt" in blockers:
        actions.append("collect ArUco/AprilTag or measured reference-pose clips for paddle pose evaluation")
    if "missing_racket_pose_evaluation" in blockers:
        actions.append("run face-angle, translation, reprojection, and temporal stability evaluation before RKT promotion")
    return actions


def _candidates(value: RacketCandidates | Mapping[str, Any]) -> RacketCandidates:
    if isinstance(value, RacketCandidates):
        return value
    return RacketCandidates.model_validate(value)


def _pose(value: RacketPose | Mapping[str, Any] | None) -> RacketPose | None:
    if value is None or isinstance(value, RacketPose):
        return value
    return RacketPose.model_validate(value)


def _read_optional_pose(path: str | Path | None) -> RacketPose | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return RacketPose.model_validate(_read_json(candidate))


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ARTIFACT_TYPE",
    "build_racket_promotion_audit",
    "build_racket_promotion_audit_from_files",
    "write_racket_promotion_audit",
]
