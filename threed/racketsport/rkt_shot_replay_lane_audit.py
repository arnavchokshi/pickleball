"""Fail-closed RKT/SHOT/RPL lane audit from local artifacts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_rkt_shot_replay_lane_audit"


def build_rkt_shot_replay_lane_audit(
    *,
    cvat_import_root: str | Path,
    replay_readiness_path: str | Path,
    shot_review_root: str | Path | None = None,
    shot_external_eval_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build a lane report without promoting review artifacts to production truth."""

    cvat_root = Path(cvat_import_root)
    manifest = _read_json_object(cvat_root / "manifest.json")
    cvat = _build_cvat_section(cvat_root, manifest)
    rkt = _build_rkt_section(cvat["clips"])
    replay = _build_replay_section(Path(replay_readiness_path))
    shot = _build_shot_section(
        Path(shot_review_root) if shot_review_root is not None else None,
        [Path(path) for path in shot_external_eval_paths],
    )
    blockers = _production_blockers(cvat=cvat, rkt=rkt, replay=replay, shot=shot)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "blocked_not_production_ready" if blockers else "ready_for_review",
        "cvat": cvat,
        "rkt": rkt,
        "replay": replay,
        "shot": shot,
        "production_blockers": blockers,
        "next_best_action": _next_best_action(rkt=rkt, replay=replay, shot=shot),
        "notes": [
            "Do not claim paddle 6DoF from rectangle boxes.",
            "Do not claim production replay from static review GLBs.",
            "Do not claim SHOT-1 without reviewed pickleball shot labels and held-out metrics.",
        ],
    }


def write_rkt_shot_replay_lane_audit(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rkt_shot_replay_lane_audit_markdown(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_rkt_shot_replay_lane_audit_markdown(payload), encoding="utf-8")


def build_rkt_shot_replay_lane_audit_markdown(payload: Mapping[str, Any]) -> str:
    cvat = _mapping(payload.get("cvat"))
    cvat_summary = _mapping(cvat.get("summary"))
    rkt = _mapping(payload.get("rkt"))
    rkt_summary = _mapping(rkt.get("summary"))
    replay = _mapping(payload.get("replay"))
    replay_summary = _mapping(replay.get("summary"))
    shot = _mapping(payload.get("shot"))
    shot_summary = _mapping(shot.get("summary"))
    blockers = [str(item) for item in payload.get("production_blockers", []) if item]

    lines = [
        "# RKT/SHOT/RPL Lane Audit",
        "",
        f"Status: {payload.get('status', 'unknown')}",
        f"Active CVAT clips: {cvat_summary.get('active_clip_count', 0)}",
        f"Player boxes: {cvat_summary.get('player_box_count', 0)}",
        f"Ball boxes: {cvat_summary.get('ball_box_count', 0)}",
        f"Paddle rectangles: {cvat_summary.get('paddle_rectangle_count', 0)}",
        "",
        "## RKT",
        "",
        (
            "RKT candidate frames: "
            f"{rkt_summary.get('candidate_frame_count', 0)} "
            f"(box-derived: {rkt_summary.get('box_derived_frame_count', 0)}, "
            "true-corner/reference: "
            f"{int(rkt_summary.get('true_corner_frame_count', 0)) + int(rkt_summary.get('reference_gt_frame_count', 0))})"
        ),
        f"Promoted canonical pose frames: {rkt_summary.get('promoted_pose_frame_count', 0)}",
        f"Unsafe promoted pose frames: {rkt_summary.get('unsafe_promoted_frame_count', 0)}",
        f"May promote RKT: {rkt.get('may_promote_rkt', False)}",
        "",
        "## RPL",
        "",
        (
            "Production replay-ready clips: "
            f"{replay_summary.get('production_replay_ready_clips', 0)}/"
            f"{replay_summary.get('clip_count', 0)}"
        ),
        f"Review-visual-ready clips: {replay_summary.get('review_visual_ready_clips', 0)}",
        f"Metrics-gate-ready clips: {replay_summary.get('metrics_gate_ready_clips', 0)}",
        "",
        "## SHOT",
        "",
        f"Review prediction artifacts: {shot_summary.get('review_prediction_file_count', 0)}",
        f"Review predictions: {shot_summary.get('review_prediction_count', 0)}",
        f"Reviewed pickleball shot truth labels: {shot_summary.get('reviewed_truth_label_count', 0)}",
        f"Trained model artifacts counted: {shot_summary.get('trained_model_count', 0)}",
        "",
        "## Production Blockers",
        "",
    ]
    lines.extend(f"- {blocker}" for blocker in blockers)
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Do not claim paddle 6DoF from rectangle boxes.",
            "- Do not claim production replay from static review GLBs.",
            "- Do not claim SHOT-1 without reviewed pickleball shot labels and held-out metrics.",
            "",
            f"Next best action: {payload.get('next_best_action', '')}",
            "",
        ]
    )
    return "\n".join(lines)


def _build_cvat_section(cvat_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    clips = []
    totals = Counter({"player": 0, "ball": 0, "paddle": 0})
    for item in _sequence(manifest.get("clips")):
        if not isinstance(item, Mapping):
            continue
        clip_id = str(item.get("clip_id", ""))
        if not clip_id:
            continue
        counts = _mapping(item.get("visible_box_count_by_label"))
        player_count = int(counts.get("player", 0) or 0)
        ball_count = int(counts.get("ball", 0) or 0)
        paddle_count = int(counts.get("paddle", 0) or 0)
        totals.update({"player": player_count, "ball": ball_count, "paddle": paddle_count})
        import_dir = Path(str(item.get("import_dir") or cvat_root / clip_id))
        clips.append(
            {
                "clip": clip_id,
                "import_dir": str(import_dir),
                "visible_box_count_by_label": {
                    "player": player_count,
                    "ball": ball_count,
                    "paddle": paddle_count,
                },
                "racket_candidates_path": str(import_dir / "racket_candidates_from_cvat_paddle_boxes.json"),
                "racket_readiness_path": str(import_dir / "racket_pose_readiness_from_cvat_paddle_boxes.json"),
                "racket_promotion_audit_path": str(import_dir / "racket_promotion_audit_from_cvat_paddle_boxes.json"),
            }
        )
    pending = [str(item) for item in _sequence(manifest.get("pending_clips"))]
    return {
        "manifest_path": str(cvat_root / "manifest.json"),
        "status": str(manifest.get("status", "unknown")),
        "summary": {
            "active_clip_count": len(clips),
            "pending_clip_count": len(pending),
            "player_box_count": int(totals["player"]),
            "ball_box_count": int(totals["ball"]),
            "paddle_rectangle_count": int(totals["paddle"]),
        },
        "pending_clips": pending,
        "clips": clips,
    }


def _build_rkt_section(cvat_clips: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clips = []
    blocker_frequency: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    totals = Counter(
        {
            "candidate_frame_count": 0,
            "box_derived_frame_count": 0,
            "true_corner_frame_count": 0,
            "reference_gt_frame_count": 0,
            "promoted_pose_frame_count": 0,
            "unsafe_promoted_frame_count": 0,
        }
    )
    for item in cvat_clips:
        clip = str(item.get("clip", ""))
        readiness_path = Path(str(item.get("racket_readiness_path", "")))
        audit_path = Path(str(item.get("racket_promotion_audit_path", "")))
        candidates_path = Path(str(item.get("racket_candidates_path", "")))
        readiness = _read_optional_json_object(readiness_path)
        audit = _read_optional_json_object(audit_path)
        candidates = _read_optional_json_object(candidates_path)
        candidate_count = _int_first(
            readiness.get("candidate_frame_count") if readiness else None,
            audit.get("candidate_frame_count") if audit else None,
            _candidate_frame_count(candidates) if candidates else None,
        )
        box_count = _int_first(
            readiness.get("box_derived_frame_count") if readiness else None,
            audit.get("box_derived_candidate_frame_count") if audit else None,
            candidate_count,
        )
        true_corner_count = _int_first(
            readiness.get("true_corner_frame_count") if readiness else None,
            audit.get("true_corner_frame_count") if audit else None,
            0,
        )
        reference_gt_count = _int_first(
            readiness.get("reference_gt_frame_count") if readiness else None,
            audit.get("reference_gt_frame_count") if audit else None,
            0,
        )
        promoted_count = _int_first(
            readiness.get("promoted_pose_frame_count") if readiness else None,
            audit.get("promoted_pose_frame_count") if audit else None,
            0,
        )
        unsafe_count = _int_first(audit.get("unsafe_promoted_frame_count") if audit else None, 0)
        source_counts.update(_mapping(readiness.get("source_counts") if readiness else {}))
        blockers = sorted(
            set(
                str(blocker)
                for blocker in list(_sequence(readiness.get("blockers") if readiness else []))
                + list(_sequence(audit.get("blockers") if audit else []))
                if blocker
            )
        )
        blocker_frequency.update(blockers)
        totals.update(
            {
                "candidate_frame_count": candidate_count,
                "box_derived_frame_count": box_count,
                "true_corner_frame_count": true_corner_count,
                "reference_gt_frame_count": reference_gt_count,
                "promoted_pose_frame_count": promoted_count,
                "unsafe_promoted_frame_count": unsafe_count,
            }
        )
        clips.append(
            {
                "clip": clip,
                "candidate_frame_count": candidate_count,
                "box_derived_frame_count": box_count,
                "true_corner_frame_count": true_corner_count,
                "reference_gt_frame_count": reference_gt_count,
                "promoted_pose_frame_count": promoted_count,
                "unsafe_promoted_frame_count": unsafe_count,
                "readiness_status": str(readiness.get("status", "missing") if readiness else "missing"),
                "promotion_status": str(audit.get("status", "missing") if audit else "missing"),
                "trusted_for_rkt_promotion": bool(audit.get("trusted_for_rkt_promotion", False)) if audit else False,
                "blockers": blockers,
            }
        )

    may_promote = (
        totals["candidate_frame_count"] > 0
        and totals["box_derived_frame_count"] == 0
        and totals["true_corner_frame_count"] > 0
        and totals["reference_gt_frame_count"] > 0
        and totals["promoted_pose_frame_count"] > 0
        and totals["unsafe_promoted_frame_count"] == 0
        and not blocker_frequency
    )
    return {
        "status": "ready_for_promotion_review" if may_promote else "blocked_preview_only",
        "summary": {key: int(totals[key]) for key in sorted(totals)},
        "source_counts": dict(sorted((str(key), int(value)) for key, value in source_counts.items())),
        "blocker_frequency": dict(sorted(blocker_frequency.items())),
        "may_promote_rkt": may_promote,
        "clips": clips,
    }


def _build_replay_section(path: Path) -> dict[str, Any]:
    report = _read_json_object(path)
    summary = dict(_mapping(report.get("summary")))
    clips = []
    blocker_frequency: Counter[str] = Counter()
    artifact_classes: Counter[str] = Counter()
    for item in _sequence(report.get("clips")):
        if not isinstance(item, Mapping):
            continue
        blockers = [str(blocker) for blocker in _sequence(item.get("blockers")) if blocker]
        blocker_frequency.update(sorted(set(blockers)))
        glb_report = _mapping(item.get("glb_report"))
        artifact_class = str(glb_report.get("artifact_class", "unknown"))
        artifact_classes.update([artifact_class])
        clips.append(
            {
                "clip": str(item.get("clip", "")),
                "review_visual_ready": bool(item.get("review_visual_ready", False)),
                "production_replay_ready": bool(item.get("production_replay_ready", False)),
                "metrics_gate_ready": bool(item.get("metrics_gate_ready", False)),
                "counts": dict(_mapping(item.get("counts"))),
                "artifact_class": artifact_class,
                "blockers": blockers,
            }
        )
    summary.setdefault("clip_count", len(clips))
    summary.setdefault("review_visual_ready_clips", sum(1 for clip in clips if clip["review_visual_ready"]))
    summary.setdefault("production_replay_ready_clips", sum(1 for clip in clips if clip["production_replay_ready"]))
    summary.setdefault("metrics_gate_ready_clips", sum(1 for clip in clips if clip["metrics_gate_ready"]))
    return {
        "report_path": str(path),
        "status": str(report.get("status", "unknown")),
        "summary": summary,
        "artifact_classes": dict(sorted(artifact_classes.items())),
        "blocker_frequency": dict(sorted(blocker_frequency.items())),
        "clips": clips,
    }


def _build_shot_section(shot_review_root: Path | None, external_eval_paths: Sequence[Path]) -> dict[str, Any]:
    predictions = []
    trained_model_count = 0
    prediction_count = 0
    gated_count = 0
    reviewed_truth_count = 0
    if shot_review_root is not None and shot_review_root.exists():
        for path in sorted(shot_review_root.glob("*/shot_classification.json")):
            payload = _read_json_object(path)
            shots = [item for item in _sequence(payload.get("shots")) if isinstance(item, Mapping)]
            classifier = _mapping(payload.get("classifier"))
            if classifier.get("trained_model"):
                trained_model_count += 1
            prediction_count += len(shots)
            gated_count += sum(1 for shot in shots if bool(shot.get("gated", False)))
            predictions.append(
                {
                    "path": str(path),
                    "clip": str(payload.get("clip_id", path.parent.name)),
                    "classifier_family": str(classifier.get("family", "unknown")),
                    "not_gate_verified": bool(classifier.get("not_gate_verified", True)),
                    "shot_count": len(shots),
                    "gated_count": sum(1 for shot in shots if bool(shot.get("gated", False))),
                }
            )
        reviewed_truth_count = _reviewed_truth_label_count(shot_review_root)
    external = [_external_eval_summary(path) for path in external_eval_paths if path.exists()]
    may_train = reviewed_truth_count > 0 and trained_model_count > 0
    return {
        "status": "reviewed_truth_present_needs_training" if reviewed_truth_count else "scaffold_transfer_only",
        "summary": {
            "review_prediction_file_count": len(predictions),
            "review_prediction_count": prediction_count,
            "review_prediction_gated_count": gated_count,
            "reviewed_truth_label_count": reviewed_truth_count,
            "trained_model_count": trained_model_count,
            "external_eval_count": len(external),
        },
        "may_train_poseconv3d_or_bst": may_train,
        "review_predictions": predictions,
        "external_eval_summaries": external,
    }


def _production_blockers(*, cvat: Mapping[str, Any], rkt: Mapping[str, Any], replay: Mapping[str, Any], shot: Mapping[str, Any]) -> list[str]:
    rkt_summary = _mapping(rkt.get("summary"))
    replay_summary = _mapping(replay.get("summary"))
    shot_summary = _mapping(shot.get("summary"))
    blockers = []
    candidate_count = int(rkt_summary.get("candidate_frame_count", 0) or 0)
    box_count = int(rkt_summary.get("box_derived_frame_count", 0) or 0)
    true_or_ref_count = int(rkt_summary.get("true_corner_frame_count", 0) or 0) + int(
        rkt_summary.get("reference_gt_frame_count", 0) or 0
    )
    if not rkt.get("may_promote_rkt", False):
        blockers.append(
            f"RKT blocked: {box_count}/{candidate_count} candidate frames are box-derived rectangles; "
            f"{true_or_ref_count} true-corner/reference frames."
        )
    production_ready = int(replay_summary.get("production_replay_ready_clips", 0) or 0)
    clip_count = int(replay_summary.get("clip_count", 0) or 0)
    if production_ready < clip_count or clip_count == 0:
        blockers.append(
            f"RPL blocked: {production_ready}/{clip_count} clips are production replay ready; "
            "static/review GLBs remain non-production."
        )
    truth_count = int(shot_summary.get("reviewed_truth_label_count", 0) or 0)
    if not shot.get("may_train_poseconv3d_or_bst", False):
        blockers.append(
            f"SHOT blocked: {truth_count} reviewed pickleball shot truth labels; "
            "transfer/heuristic predictions cannot train or verify SHOT-1."
        )
    return blockers


def _next_best_action(*, rkt: Mapping[str, Any], replay: Mapping[str, Any], shot: Mapping[str, Any]) -> str:
    rkt_summary = _mapping(rkt.get("summary"))
    if int(rkt_summary.get("true_corner_frame_count", 0) or 0) == 0:
        return (
            "RKT: import true paddle-face corner/keypoint/CAD/reference labels, then rerun the "
            "fail-closed RKT promotion audit before any replay promotion."
        )
    shot_summary = _mapping(shot.get("summary"))
    if int(shot_summary.get("reviewed_truth_label_count", 0) or 0) == 0:
        return "SHOT: create reviewed pickleball shot truth labels before DATA-5 or PoseConv3D/BST training claims."
    replay_summary = _mapping(replay.get("summary"))
    if int(replay_summary.get("production_replay_ready_clips", 0) or 0) == 0:
        return "RPL: build animated GLB/USDZ only after upstream BODY/BALL/RKT artifacts are trustworthy."
    return "Run focused gates and update the lane report."


def _external_eval_summary(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    return {
        "path": str(path),
        "dataset": str(payload.get("dataset", path.stem)),
        "accuracy": payload.get("accuracy"),
        "family_accuracy": payload.get("family_accuracy"),
        "top2_family_accuracy": payload.get("top2_family_accuracy"),
        "macro_f1": payload.get("macro_f1") or _mapping(payload.get("test")).get("macro_f1"),
        "by_label": dict(_mapping(payload.get("by_label"))),
        "by_truth": dict(_mapping(payload.get("by_truth"))),
        "status": str(payload.get("status", "external_eval_only")),
        "not_pickleball_gate": True,
    }


def _reviewed_truth_label_count(root: Path) -> int:
    count = 0
    for path in sorted(root.rglob("*.json")):
        if path.name == "shot_classification.json":
            continue
        try:
            payload = _read_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if payload.get("artifact_type") == "racketsport_shot_classification":
            continue
        if payload.get("not_ground_truth") is True:
            continue
        if payload.get("status") not in {"human_reviewed", "reviewed", "accepted"}:
            continue
        annotation = _mapping(payload.get("annotation"))
        for item in _sequence(annotation.get("items")):
            if isinstance(item, Mapping) and item.get("status") == "accepted" and item.get("shot_label"):
                count += 1
    return count


def _candidate_frame_count(payload: Mapping[str, Any] | None) -> int:
    if not payload:
        return 0
    total = 0
    for player in _sequence(payload.get("players")):
        if isinstance(player, Mapping):
            total += len(_sequence(player.get("frames")))
    return total


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json_object(path)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _int_first(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


__all__ = [
    "ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "build_rkt_shot_replay_lane_audit",
    "build_rkt_shot_replay_lane_audit_markdown",
    "write_rkt_shot_replay_lane_audit",
    "write_rkt_shot_replay_lane_audit_markdown",
]
