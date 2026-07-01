"""Racket/paddle pose readiness diagnostics for human review packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import RacketCandidates, RacketPose


ARTIFACT_TYPE = "racketsport_racket_pose_readiness"
SCHEMA_VERSION = 1


def build_racket_pose_readiness(
    *,
    clip: str,
    racket_candidates: RacketCandidates | Mapping[str, Any],
    racket_pose_preview: RacketPose | Mapping[str, Any] | None = None,
    racket_pose: RacketPose | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize whether paddle pose evidence is preview-only or gate-ready."""

    candidates = _candidates(racket_candidates)
    preview = _pose(racket_pose_preview)
    promoted = _pose(racket_pose)
    source_counts = _source_counts(candidates)
    source_evidence_counts = _source_evidence_counts(source_counts)
    candidate_frame_count = sum(source_counts.values())
    box_derived_frame_count = source_evidence_counts["box_derived"]
    true_corner_frame_count = source_evidence_counts["true_corners_or_pose"]
    reference_gt_frame_count = source_evidence_counts["reference_gt"]
    preview_pose_frame_count = _pose_frame_count(preview)
    promoted_pose_frame_count = _pose_frame_count(promoted)
    blockers: list[str] = []

    if box_derived_frame_count:
        blockers.append("box_derived_candidate_corners")
    if true_corner_frame_count == 0:
        blockers.append("missing_true_paddle_keypoints_or_cad_pose")
    if promoted_pose_frame_count == 0:
        blockers.append("missing_promoted_racket_pose_json")
    if reference_gt_frame_count == 0:
        blockers.append("missing_reference_pose_gt")
    blockers.append("missing_racket_pose_evaluation")

    status = "blocked_preview_only"
    if promoted_pose_frame_count > 0 and true_corner_frame_count > 0 and reference_gt_frame_count == 0:
        status = "pose_present_needs_reference_and_eval"
    if promoted_pose_frame_count > 0 and true_corner_frame_count > 0 and reference_gt_frame_count > 0:
        status = "pose_present_needs_eval"

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": status,
        "source_counts": source_counts,
        "source_evidence_counts": source_evidence_counts,
        "candidate_frame_count": candidate_frame_count,
        "box_derived_frame_count": box_derived_frame_count,
        "true_corner_frame_count": true_corner_frame_count,
        "reference_gt_frame_count": reference_gt_frame_count,
        "preview_pose_frame_count": preview_pose_frame_count,
        "promoted_pose_frame_count": promoted_pose_frame_count,
        "local_readiness": _local_readiness(
            candidate_frame_count=candidate_frame_count,
            box_derived_frame_count=box_derived_frame_count,
            true_corner_frame_count=true_corner_frame_count,
        ),
        "missing_label_or_asset_state": _missing_label_or_asset_state(
            source_evidence_counts=source_evidence_counts,
            true_corner_frame_count=true_corner_frame_count,
            reference_gt_frame_count=reference_gt_frame_count,
        ),
        "runnable_now": _runnable_now(candidate_frame_count=candidate_frame_count),
        "blocked_until": _blocked_until(blockers),
        "blockers": blockers,
        "recommended_next_actions": _recommended_next_actions(blockers),
        "summary": {
            "candidate_player_count": len(candidates.players),
            "candidate_frame_count": candidate_frame_count,
            "box_derived_frame_count": box_derived_frame_count,
            "true_corner_frame_count": true_corner_frame_count,
            "reference_gt_frame_count": reference_gt_frame_count,
            "preview_pose_frame_count": preview_pose_frame_count,
            "promoted_pose_frame_count": promoted_pose_frame_count,
        },
    }


def _local_readiness(
    *,
    candidate_frame_count: int,
    box_derived_frame_count: int,
    true_corner_frame_count: int,
) -> dict[str, bool]:
    has_candidates = candidate_frame_count > 0
    non_box_candidate_only = has_candidates and box_derived_frame_count == 0
    can_attempt_canonical_pose = non_box_candidate_only and true_corner_frame_count > 0
    return {
        "can_convert_box_labels_to_review_candidates": has_candidates and box_derived_frame_count > 0,
        "can_build_true_corner_review": has_candidates,
        "can_build_preview_pose": has_candidates,
        "can_run_promotion_audit": has_candidates,
        "can_run_fail_closed_stage_smoke": has_candidates,
        "can_write_canonical_racket_pose": can_attempt_canonical_pose,
        "can_claim_paddle_6dof": False,
    }


def _missing_label_or_asset_state(
    *,
    source_evidence_counts: Mapping[str, int],
    true_corner_frame_count: int,
    reference_gt_frame_count: int,
) -> dict[str, str]:
    cad_or_reference_count = int(source_evidence_counts.get("synthetic_or_cad", 0)) + reference_gt_frame_count
    return {
        "true_paddle_face_corner_labels": "present" if true_corner_frame_count > 0 else "missing",
        "paddle_cad_or_reference_asset": "present" if cad_or_reference_count > 0 else "missing",
        "aruco_apriltag_or_reference_pose_gt": "present" if reference_gt_frame_count > 0 else "missing",
        "racket_pose_evaluation": "missing",
    }


def _runnable_now(*, candidate_frame_count: int) -> list[str]:
    if candidate_frame_count <= 0:
        return []
    return [
        "convert CVAT or draft paddle boxes into review-only racket_candidates.json",
        "render candidate overlays and paddle true-corner crop sheets",
        "build racket_pose_preview.json for visualization only when calibration is present",
        "build racket_promotion_audit.json to prove canonical racket_pose.json is absent or safe",
        "run RacketStageRunner as a fail-closed smoke; box-derived candidates must be rejected",
    ]


def _blocked_until(blockers: list[str]) -> list[str]:
    blocked: list[str] = []
    if "missing_true_paddle_keypoints_or_cad_pose" in blockers or "missing_reference_pose_gt" in blockers:
        blocked.append("true paddle-face corner labels, mask/keypoint corners, CAD/reference pose, or ArUco/AprilTag GT")
    if "missing_promoted_racket_pose_json" in blockers:
        blocked.append("canonical racket_pose.json from non-box evidence")
    if "missing_racket_pose_evaluation" in blockers:
        blocked.append("RKT face-angle/contact-point evaluation against reference labels")
    return blocked


def build_racket_pose_readiness_from_files(
    *,
    clip: str,
    racket_candidates_path: str | Path,
    racket_pose_preview_path: str | Path | None = None,
    racket_pose_path: str | Path | None = None,
) -> dict[str, Any]:
    candidates = RacketCandidates.model_validate(_read_json(racket_candidates_path))
    preview = _read_optional_pose(racket_pose_preview_path)
    promoted = _read_optional_pose(racket_pose_path)
    return build_racket_pose_readiness(
        clip=clip,
        racket_candidates=candidates,
        racket_pose_preview=preview,
        racket_pose=promoted,
    )


def write_racket_pose_readiness(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_counts(candidates: RacketCandidates) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in candidates.players:
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
    if normalized.startswith("label_bbox:"):
        return "box_derived"
    if any(token in normalized for token in ("synthetic", "blenderproc", "cad")):
        return "synthetic_or_cad"
    if any(token in normalized for token in ("aruco", "april", "tag", "gt", "ground_truth", "reference")):
        return "reference_gt"
    return "keypoint_or_mask"


def _pose_frame_count(pose: RacketPose | None) -> int:
    if pose is None:
        return 0
    return sum(len(player.frames) for player in pose.players)


def _recommended_next_actions(blockers: list[str]) -> list[str]:
    actions: list[str] = []
    if "missing_true_paddle_keypoints_or_cad_pose" in blockers:
        actions.append("collect true paddle corner labels or CAD/reference pose evidence")
    if "missing_promoted_racket_pose_json" in blockers:
        actions.append("run fail-closed RacketStageRunner only after non-box candidate evidence exists")
    if "missing_reference_pose_gt" in blockers:
        actions.append("collect ArUco/AprilTag/reference pose clips for face-angle and translation error evaluation")
    if "missing_racket_pose_evaluation" in blockers:
        actions.append("run ArUco/AprilTag or held-out pose evaluation before RKT promotion")
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
    "build_racket_pose_readiness",
    "build_racket_pose_readiness_from_files",
    "write_racket_pose_readiness",
]
