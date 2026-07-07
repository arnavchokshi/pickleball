"""Benchmark existing ball tracks against reviewed CVAT ball boxes.

This module scores already-materialized tracks only. It does not run model
inference, train TrackNet/PB-MAT, or promote BALL gate status.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_overlay import load_ball_track
from .schemas import BallFrame, CvatVideoAnnotations


ARTIFACT_TYPE = "racketsport_cvat_ball_tracker_benchmark"
CONTACT_CUE_ARTIFACT_TYPE = "racketsport_ball_contact_cue_coverage"
STATUS_TESTED = "TESTED-ON-REAL-DATA"
STATUS_NOT_STARTED = "NOT-STARTED"
DEFAULT_HIT_RADIUS_PX = 36.0
DEFAULT_F1_RADIUS_PX = 20.0
DEFAULT_TELEPORT_PX_PER_FRAME = 160.0
DEFAULT_MAX_JUMP_GAP_FRAMES = 3
DEFAULT_MAX_CUE_DELTA_FRAMES = 6.0


@dataclass(frozen=True)
class CvatBallCandidate:
    clip: str
    name: str
    path: Path
    category: str = "generalizable"


def benchmark_cvat_ball_track_candidate(
    *,
    ball_track_path: str | Path,
    cvat_labels_path: str | Path,
    candidate_name: str,
    category: str = "generalizable",
    hit_radius_px: float = DEFAULT_HIT_RADIUS_PX,
    f1_radius_px: float = DEFAULT_F1_RADIUS_PX,
    teleport_px_per_frame: float = DEFAULT_TELEPORT_PX_PER_FRAME,
    max_jump_gap_frames: int = DEFAULT_MAX_JUMP_GAP_FRAMES,
    include_approx: bool = True,
) -> dict[str, Any]:
    """Score one existing ball-track artifact against reviewed CVAT labels."""

    hit_radius_px = _nonnegative_finite(hit_radius_px, "hit_radius_px")
    f1_radius_px = _positive_finite(f1_radius_px, "f1_radius_px")
    teleport_px_per_frame = _positive_finite(teleport_px_per_frame, "teleport_px_per_frame")
    if max_jump_gap_frames < 1:
        raise ValueError("max_jump_gap_frames must be >= 1")

    track = load_ball_track(ball_track_path)
    labels = _load_cvat_labels(cvat_labels_path)
    excluded_candidate_approx_frame_count = _visible_approx_sample_count(track) if not include_approx else 0
    samples_by_index = _track_samples_by_frame_index(track, include_approx=include_approx)
    track_frame_count = _track_frame_count(track)
    label_frame_count = len(labels.frames)
    evaluated_frame_count = min(track_frame_count, label_frame_count)
    reviewed_frame_indices = _reviewed_frame_indices(labels, label_frame_count=label_frame_count)
    evaluated_reviewed_frame_indices = [index for index in reviewed_frame_indices if index < evaluated_frame_count]
    excluded_reviewed_frame_indices = [index for index in reviewed_frame_indices if index >= evaluated_frame_count]
    evaluated_frames = _frames_for_indices(labels, evaluated_reviewed_frame_indices)
    excluded_frames = _frames_for_indices(labels, excluded_reviewed_frame_indices)
    centers_by_frame = _ball_centers_for_frame_indices(labels, evaluated_reviewed_frame_indices)
    all_centers_by_frame = _ball_centers_for_frame_indices(labels, reviewed_frame_indices)
    excluded_centers_by_frame = _ball_centers_for_frame_indices(labels, excluded_reviewed_frame_indices)

    label_metrics = _label_metrics(
        samples_by_index=samples_by_index,
        centers_by_frame=centers_by_frame,
        evaluated_reviewed_frame_indices=evaluated_reviewed_frame_indices,
        hit_radius_px=hit_radius_px,
        f1_radius_px=f1_radius_px,
        fps=float(track.fps),
    )
    jitter_metrics = _jitter_metrics(
        samples_by_index=samples_by_index,
        evaluated_frame_count=evaluated_frame_count,
        teleport_px_per_frame=teleport_px_per_frame,
        max_jump_gap_frames=max_jump_gap_frames,
    )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_ball_tracker_benchmark_candidate",
        "status": STATUS_TESTED,
        "ball_verified": False,
        "clip": labels.clip_id,
        "candidate": candidate_name,
        "category": category,
        "ball_track_path": str(ball_track_path),
        "cvat_labels_path": str(cvat_labels_path),
        "hit_radius_px": hit_radius_px,
        "f1_radius_px": f1_radius_px,
        "teleport_px_per_frame": teleport_px_per_frame,
        "max_jump_gap_frames": int(max_jump_gap_frames),
        "include_approx": bool(include_approx),
        "excluded_candidate_approx_frame_count": excluded_candidate_approx_frame_count,
        "track_frame_count": track_frame_count,
        "cvat_frame_count": label_frame_count,
        "evaluated_frame_count": evaluated_frame_count,
        "reviewed_frame_count": len(reviewed_frame_indices),
        "evaluated_reviewed_frame_count": len(evaluated_reviewed_frame_indices),
        "excluded_reviewed_frame_count": len(excluded_reviewed_frame_indices),
        "reviewed_frame_indices_source": labels.reviewed_frame_indices_source or "dense_all_frames",
        "track_frame_range": _index_range(samples_by_index.keys()),
        "evaluated_cvat_frame_range": _frame_range(evaluated_frames),
        "excluded_cvat_frame_range": _frame_range(excluded_frames),
        "excluded_track_frame_count": max(0, track_frame_count - evaluated_frame_count),
        "excluded_cvat_frame_count": len(excluded_reviewed_frame_indices),
        "cvat_visible_label_count": len(all_centers_by_frame),
        "evaluated_cvat_visible_label_count": len(centers_by_frame),
        "excluded_cvat_visible_label_count": len(excluded_centers_by_frame),
        "excluded_cvat_hidden_frame_count": max(0, len(excluded_reviewed_frame_indices) - len(excluded_centers_by_frame)),
        "track_fps": float(track.fps),
        "cvat_original_size": list(labels.task.original_size),
        "visible_frame_count": sum(
            1 for index, frame in samples_by_index.items() if index < evaluated_frame_count and frame.visible
        ),
        "label_metrics": label_metrics,
        "jitter_metrics": jitter_metrics,
        "quality_score": _quality_score(label_metrics, jitter_metrics),
        "notes": [
            "CVAT ball boxes are held-out reviewed labels. The track is scored as an existing artifact only; this report does not run inference or train a model.",
            "BALL is not verified by this report unless the downstream numeric acceptance gates are defined and passed on reviewed labels.",
        ],
    }


def benchmark_cvat_ball_tracker_candidates(
    *,
    candidates: list[CvatBallCandidate],
    cvat_root: str | Path,
    hit_radius_px: float = DEFAULT_HIT_RADIUS_PX,
    f1_radius_px: float = DEFAULT_F1_RADIUS_PX,
    teleport_px_per_frame: float = DEFAULT_TELEPORT_PX_PER_FRAME,
    max_jump_gap_frames: int = DEFAULT_MAX_JUMP_GAP_FRAMES,
    include_approx: bool = True,
    review_input_path: str | Path | None = None,
    cue_root: str | Path | None = None,
    contact_fps: float = 60.0,
    max_cue_delta_frames: float = DEFAULT_MAX_CUE_DELTA_FRAMES,
) -> dict[str, Any]:
    """Score many candidates and optionally attach ball-inflection cue coverage."""

    cvat_base = Path(cvat_root)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        labels_path = cvat_base / candidate.clip / "reviewed_boxes.json"
        rows.append(
            benchmark_cvat_ball_track_candidate(
                ball_track_path=candidate.path,
                cvat_labels_path=labels_path,
                candidate_name=candidate.name,
                category=candidate.category,
                hit_radius_px=hit_radius_px,
                f1_radius_px=f1_radius_px,
                teleport_px_per_frame=teleport_px_per_frame,
                max_jump_gap_frames=max_jump_gap_frames,
                include_approx=include_approx,
            )
        )

    clip_names = sorted({candidate.clip for candidate in candidates})
    contact_cue_coverage = None
    if review_input_path is not None and cue_root is not None:
        contact_cue_coverage = evaluate_ball_contact_cue_coverage(
            review_input_path=review_input_path,
            cue_root=cue_root,
            clips=clip_names,
            fps=contact_fps,
            max_match_delta_frames=max_cue_delta_frames,
        )

    aggregate = _aggregate(rows)
    summary = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": STATUS_TESTED,
        "ball_verified": False,
        "candidate_count": len(rows),
        "clip_count": len(clip_names),
        "clips": clip_names,
        "cvat_root": str(cvat_base),
        "hit_radius_px": float(hit_radius_px),
        "f1_radius_px": float(f1_radius_px),
        "teleport_px_per_frame": float(teleport_px_per_frame),
        "max_jump_gap_frames": int(max_jump_gap_frames),
        "include_approx": bool(include_approx),
        "results": rows,
        "aggregate": aggregate,
        "candidate_paths": _candidate_paths(rows),
        "contact_cue_coverage": contact_cue_coverage,
        "notes": [
            "This is a local benchmark over existing artifacts and reviewed labels only.",
            "BALL is not verified; strict no-click artifacts remain scored candidates until numeric acceptance gates pass.",
        ],
    }
    summary["full_horizon"] = _full_horizon_coverage(summary)
    summary["next_training_eval_recommendation"] = _next_training_eval_recommendation(summary)
    summary["verification_blockers"] = _verification_blockers(summary)
    return summary


def write_cvat_ball_tracker_benchmark(
    *,
    candidates: list[CvatBallCandidate],
    cvat_root: str | Path,
    out_json: str | Path,
    out_markdown: str | Path | None = None,
    hit_radius_px: float = DEFAULT_HIT_RADIUS_PX,
    f1_radius_px: float = DEFAULT_F1_RADIUS_PX,
    teleport_px_per_frame: float = DEFAULT_TELEPORT_PX_PER_FRAME,
    max_jump_gap_frames: int = DEFAULT_MAX_JUMP_GAP_FRAMES,
    include_approx: bool = True,
    review_input_path: str | Path | None = None,
    cue_root: str | Path | None = None,
    contact_fps: float = 60.0,
    max_cue_delta_frames: float = DEFAULT_MAX_CUE_DELTA_FRAMES,
) -> dict[str, Any]:
    summary = benchmark_cvat_ball_tracker_candidates(
        candidates=candidates,
        cvat_root=cvat_root,
        hit_radius_px=hit_radius_px,
        f1_radius_px=f1_radius_px,
        teleport_px_per_frame=teleport_px_per_frame,
        max_jump_gap_frames=max_jump_gap_frames,
        include_approx=include_approx,
        review_input_path=review_input_path,
        cue_root=cue_root,
        contact_fps=contact_fps,
        max_cue_delta_frames=max_cue_delta_frames,
    )
    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_markdown is not None:
        out_md_path = Path(out_markdown)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text(render_cvat_ball_tracker_benchmark_markdown(summary), encoding="utf-8")
    return summary


def evaluate_ball_contact_cue_coverage(
    *,
    review_input_path: str | Path,
    cue_root: str | Path,
    clips: Sequence[str] | None = None,
    fps: float = 60.0,
    max_match_delta_frames: float = DEFAULT_MAX_CUE_DELTA_FRAMES,
) -> dict[str, Any]:
    """Compare ball-inflection cue timing to reviewed contact timestamps."""

    fps = _positive_finite(fps, "fps")
    max_match_delta_frames = _positive_finite(max_match_delta_frames, "max_match_delta_frames")
    review_path = Path(review_input_path)
    cue_base = Path(cue_root)
    review_input = _read_json_object(review_path)
    clip_names = list(clips) if clips is not None else _default_review_clips(review_input)
    clip_reports = [
        _evaluate_contact_cues_for_clip(
            clip=clip,
            review_input=review_input,
            cue_root=cue_base,
            fps=fps,
            max_match_delta_frames=max_match_delta_frames,
        )
        for clip in clip_names
    ]
    summary = _contact_cue_summary(clip_reports, max_match_delta_frames=max_match_delta_frames)
    return {
        "schema_version": 1,
        "artifact_type": CONTACT_CUE_ARTIFACT_TYPE,
        "status": STATUS_TESTED,
        "verification_scope": "ball_inflection_cue_vs_reviewed_contacts",
        "ball_verified": False,
        "review_input_path": str(review_path),
        "cue_root": str(cue_base),
        "fps": fps,
        "max_match_delta_frames": max_match_delta_frames,
        "summary": summary,
        "clips": clip_reports,
        "notes": [
            "Ball-inflection cues are support signals only. They are not CONTACT or BALL gate verification without the rest of the reviewed acceptance suite."
        ],
    }


def render_cvat_ball_tracker_benchmark_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# CVAT Ball Tracker Benchmark",
        "",
        "BALL is not verified by this report. It scores existing artifacts against reviewed CVAT labels and contact review timestamps only.",
        "",
        "## Verification Blockers",
        "",
    ]
    for blocker in summary.get("verification_blockers", []):
        lines.append(f"- {blocker}")

    full_horizon = summary.get("full_horizon")
    if isinstance(full_horizon, Mapping):
        lines.extend(
            [
                "",
                "## Full-Horizon Coverage",
                "",
                "| CVAT labels | Evaluated labels | Excluded labels | All labels evaluated |",
                "| ---: | ---: | ---: | --- |",
                "| {total} | {evaluated} | {excluded} | {all_done} |".format(
                    total=full_horizon["total_cvat_visible_label_count"],
                    evaluated=full_horizon["total_evaluated_cvat_visible_label_count"],
                    excluded=full_horizon["total_excluded_cvat_visible_label_count"],
                    all_done="yes" if full_horizon["all_cvat_visible_labels_evaluated"] else "no",
                ),
            ]
        )
        blockers = full_horizon.get("blockers")
        if isinstance(blockers, Sequence) and blockers:
            lines.extend(["", "Full-horizon blockers:"])
            for blocker in blockers:
                lines.append(f"- {blocker}")

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Candidate | Category | Clips | Eval labels | Excl labels | F1@20 | Precision@20 | Recall@20 | Hit recall | P90 px | P95 px | Hidden FP | Hidden FP/min | Coverage | P95 step px | Teleports | Score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in sorted(summary["aggregate"].items()):
        lines.append(
            "| {name} | {category} | {clips} | {eval_labels} | {excluded_labels} | {f1} | {precision} | {recall20} | {hit_recall} | {p90} | {p95} | {hidden_fp} | {hidden_fp_min} | {coverage} | {p95_step} | {teleports} | {score} |".format(
                name=name,
                category=row["category"],
                clips=row["clip_count"],
                eval_labels=_fmt(row.get("total_visible_label_count"), digits=0),
                excluded_labels=_fmt(row.get("total_excluded_cvat_visible_label_count"), digits=0),
                f1=_fmt(row.get("micro_label_f1_at_20px")),
                precision=_fmt(row.get("micro_precision_at_20px")),
                recall20=_fmt(row.get("micro_visible_recall_at_20px")),
                hit_recall=_fmt(row.get("micro_visible_hit_recall")),
                p90=_fmt(row.get("mean_p90_error_px")),
                p95=_fmt(row.get("mean_p95_error_px")),
                hidden_fp=_fmt(row.get("micro_hidden_false_positive_rate")),
                hidden_fp_min=_fmt(row.get("mean_hidden_false_positives_per_minute")),
                coverage=_fmt(row.get("mean_visible_coverage_rate")),
                p95_step=_fmt(row.get("mean_p95_step_px")),
                teleports=_fmt(row.get("total_teleport_count"), digits=0),
                score=_fmt(row.get("mean_quality_score")),
            )
        )

    lines.extend(
        [
            "",
            "## Per Clip",
            "",
            "| Clip | Candidate | CVAT frames | Evaluated frames | Excl frames | Eval labels | Excl labels | F1@20 | Hit recall | P90 px | P95 px | Hidden FP | Coverage | Teleports | Score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summary["results"], key=lambda item: (item["clip"], item["candidate"])):
        label = row["label_metrics"]
        jitter = row["jitter_metrics"]
        lines.append(
            "| {clip} | {candidate} | {cvat_frames} | {eval_frames} | {excluded_frames} | {visible} | {excluded_labels} | {f1} | {hit_recall} | {p90} | {p95} | {hidden_fp} | {coverage} | {teleports} | {score} |".format(
                clip=row["clip"],
                candidate=row["candidate"],
                cvat_frames=row["cvat_frame_count"],
                eval_frames=row["evaluated_frame_count"],
                excluded_frames=row["excluded_cvat_frame_count"],
                visible=label["visible_label_count"],
                excluded_labels=row["excluded_cvat_visible_label_count"],
                f1=_fmt(label.get("label_f1_at_20px")),
                hit_recall=_fmt(label.get("visible_hit_recall")),
                p90=_fmt(label.get("p90_error_px")),
                p95=_fmt(label.get("p95_error_px")),
                hidden_fp=_fmt(label.get("hidden_false_positive_rate")),
                coverage=_fmt(jitter.get("visible_coverage_rate")),
                teleports=_fmt(jitter.get("teleport_count"), digits=0),
                score=_fmt(row.get("quality_score")),
            )
        )

    contact = summary.get("contact_cue_coverage")
    if isinstance(contact, Mapping):
        contact_summary = contact["summary"]
        lines.extend(
            [
                "",
                "## Contact Cue Coverage",
                "",
                "| Clips | Reviewed contacts | Matched cues | Missing contacts | Cue coverage | Within 2f | P90 timing delta frames | Extra cues |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| {clips} | {reviewed} | {matched} | {missing} | {coverage} | {within2} | {p90} | {extra} |".format(
                    clips=contact_summary["clip_count"],
                    reviewed=contact_summary["reviewed_contact_count"],
                    matched=contact_summary["matched_contact_count"],
                    missing=contact_summary["missing_reviewed_contact_count"],
                    coverage=_fmt(contact_summary.get("cue_coverage_rate")),
                    within2=contact_summary["within_2_frames_count"],
                    p90=_fmt(contact_summary.get("p90_abs_delta_frames")),
                    extra=contact_summary["extra_cue_count"],
                ),
                "",
                "Ball-inflection cue coverage is support-signal coverage only; promoted contact windows still need a separate reviewed-contact alignment check.",
            ]
        )

    recommendation = summary.get("next_training_eval_recommendation")
    if isinstance(recommendation, str) and recommendation:
        lines.extend(
            [
                "",
                "## Next Training/Eval Recommendation",
                "",
                recommendation,
            ]
        )

    lines.extend(
        [
            "",
            "## Candidate Paths",
            "",
            "| Clip | Candidate | Category | Track frames | Excluded labels | Path |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in sorted(summary.get("candidate_paths", []), key=lambda item: (item["clip"], item["candidate"])):
        lines.append(
            "| {clip} | {candidate} | {category} | {track_frames} | {excluded_labels} | `{path}` |".format(
                clip=row["clip"],
                candidate=row["candidate"],
                category=row["category"],
                track_frames=row["track_frame_count"],
                excluded_labels=row["excluded_cvat_visible_label_count"],
                path=row["ball_track_path"],
            )
        )

    lines.append("")
    return "\n".join(lines)


def _load_cvat_labels(path: str | Path) -> CvatVideoAnnotations:
    labels_path = Path(path)
    if not labels_path.is_file():
        raise ValueError(f"missing CVAT labels file: {labels_path}")
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid CVAT labels JSON: {labels_path}: {exc}") from exc
    try:
        return CvatVideoAnnotations.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"invalid CVAT labels schema: {labels_path}: {exc}") from exc


def _track_samples_by_frame_index(track: Any, *, include_approx: bool = True) -> dict[int, BallFrame]:
    samples: dict[int, BallFrame] = {}
    for frame in track.frames:
        if frame.visible and frame.approx and not include_approx:
            continue
        samples[int(round(float(frame.t) * float(track.fps)))] = frame
    return samples


def _visible_approx_sample_count(track: Any) -> int:
    return sum(1 for frame in track.frames if frame.visible and frame.approx)


def _track_frame_count(track: Any) -> int:
    indices = [int(round(float(frame.t) * float(track.fps))) for frame in track.frames]
    return max(indices, default=-1) + 1


def _reviewed_frame_indices(labels: CvatVideoAnnotations, *, label_frame_count: int) -> list[int]:
    if labels.reviewed_frame_indices is None:
        return list(range(label_frame_count))
    return [int(index) for index in labels.reviewed_frame_indices if int(index) < label_frame_count]


def _frames_for_indices(labels: CvatVideoAnnotations, frame_indices: Sequence[int]) -> list[Any]:
    by_index = {int(frame.frame_index): frame for frame in labels.frames}
    return [by_index[index] for index in frame_indices if index in by_index]


def _ball_centers_for_frame_indices(
    labels: CvatVideoAnnotations, frame_indices: Sequence[int]
) -> dict[int, list[tuple[float, float]]]:
    return _ball_centers_for_frames(_frames_for_indices(labels, frame_indices))


def _ball_centers_for_frames(frames: Sequence[Any]) -> dict[int, list[tuple[float, float]]]:
    centers: dict[int, list[tuple[float, float]]] = {}
    for frame in frames:
        ball_centers: list[tuple[float, float]] = []
        for box in frame.boxes:
            if box.label.strip().lower() != "ball":
                continue
            if box.visibility_level in {"full", "out_of_frame"}:
                continue
            x1, y1, x2, y2 = box.bbox_xyxy
            ball_centers.append(((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0))
        if ball_centers:
            centers[int(frame.frame_index)] = ball_centers
    return centers


def _label_metrics(
    *,
    samples_by_index: dict[int, BallFrame],
    centers_by_frame: Mapping[int, Sequence[tuple[float, float]]],
    evaluated_reviewed_frame_indices: Sequence[int],
    hit_radius_px: float,
    f1_radius_px: float,
    fps: float,
) -> dict[str, Any]:
    radii = tuple(sorted({5.0, 10.0, 20.0, 40.0, f1_radius_px}))
    distances: list[float] = []
    visible_prediction_count = 0
    visible_hit_count = 0
    hit_counts_by_radius = {radius: 0 for radius in radii}
    visible_mislocalized_at_f1 = 0

    for frame_index, centers in centers_by_frame.items():
        frame = samples_by_index.get(frame_index)
        if frame is None or not frame.visible:
            continue
        visible_prediction_count += 1
        distance_px = min(_distance(frame.xy, center) for center in centers)
        distances.append(distance_px)
        if distance_px <= hit_radius_px:
            visible_hit_count += 1
        if distance_px > f1_radius_px:
            visible_mislocalized_at_f1 += 1
        for radius in radii:
            if distance_px <= radius:
                hit_counts_by_radius[radius] += 1

    hidden_indices = [index for index in evaluated_reviewed_frame_indices if index not in centers_by_frame]
    hidden_false_positive_count = sum(
        1 for index in hidden_indices if (frame := samples_by_index.get(index)) is not None and frame.visible
    )
    hidden_true_negative_count = len(hidden_indices) - hidden_false_positive_count
    hidden_reviewed_minutes = len(hidden_indices) / float(fps) / 60.0 if fps > 0.0 else 0.0
    hidden_false_positives_per_minute = (
        hidden_false_positive_count / hidden_reviewed_minutes if hidden_reviewed_minutes > 0.0 else None
    )

    visible_label_count = len(centers_by_frame)
    f1_tp = hit_counts_by_radius[f1_radius_px]
    f1_fp = hidden_false_positive_count + visible_mislocalized_at_f1
    f1_fn = visible_label_count - f1_tp
    ten_px_tp = hit_counts_by_radius[10.0]
    ten_px_mislocalized = sum(1 for distance in distances if distance > 10.0)
    ten_px_fp = hidden_false_positive_count + ten_px_mislocalized
    ten_px_fn = visible_label_count - ten_px_tp
    precision_at_20 = _ratio(f1_tp, f1_tp + f1_fp)
    recall_at_20 = _ratio(f1_tp, f1_tp + f1_fn)
    precision_at_10 = _ratio(ten_px_tp, ten_px_tp + ten_px_fp)
    recall_at_10 = _ratio(ten_px_tp, ten_px_tp + ten_px_fn)
    hidden_true_negative_rate = _ratio(hidden_true_negative_count, len(hidden_indices))
    return {
        "visible_label_count": visible_label_count,
        "visible_prediction_count": visible_prediction_count,
        "visible_hit_count": visible_hit_count,
        "visible_presence_recall": _ratio(visible_prediction_count, visible_label_count),
        "visible_hit_recall": _ratio(visible_hit_count, visible_label_count),
        "visible_recall_at_5px": _ratio(hit_counts_by_radius[5.0], visible_label_count),
        "visible_recall_at_10px": _ratio(hit_counts_by_radius[10.0], visible_label_count),
        "visible_recall_at_20px": recall_at_20,
        "visible_recall_at_40px": _ratio(hit_counts_by_radius[40.0], visible_label_count),
        "visible_mislocalized_at_20px_count": visible_mislocalized_at_f1,
        "median_error_px": _percentile(distances, 50) if distances else None,
        "p90_error_px": _percentile(distances, 90) if distances else None,
        "p95_error_px": _percentile(distances, 95) if distances else None,
        "hit_radius_px": hit_radius_px,
        "f1_radius_px": f1_radius_px,
        "hidden_label_count": len(hidden_indices),
        "hidden_false_positive_count": hidden_false_positive_count,
        "hidden_false_positive_rate": _ratio(hidden_false_positive_count, len(hidden_indices)),
        "hidden_false_positives_per_minute": hidden_false_positives_per_minute,
        "hidden_true_negative_count": hidden_true_negative_count,
        "hidden_true_negative_rate": hidden_true_negative_rate,
        "precision_at_20px": precision_at_20,
        "label_f1_at_20px": _f1(precision_at_20, recall_at_20),
        "precision_at_10px": precision_at_10,
        "label_f1_at_10px": _f1(precision_at_10, recall_at_10),
        "balanced_accuracy_at_20px": _mean(value for value in (recall_at_20, hidden_true_negative_rate)),
        "f1_false_positive_count": f1_fp,
        "f1_false_negative_count": f1_fn,
        "f1_true_positive_count": f1_tp,
        "f1_true_positive_count_at_10px": ten_px_tp,
        "f1_false_positive_count_at_10px": ten_px_fp,
        "f1_false_negative_count_at_10px": ten_px_fn,
    }


def _jitter_metrics(
    *,
    samples_by_index: dict[int, BallFrame],
    evaluated_frame_count: int,
    teleport_px_per_frame: float,
    max_jump_gap_frames: int,
) -> dict[str, Any]:
    visible = sorted(
        (index, frame) for index, frame in samples_by_index.items() if index < evaluated_frame_count and frame.visible
    )
    steps: list[float] = []
    speeds: list[float] = []
    visible_gaps = [index - prev_index for (prev_index, _prev), (index, _frame) in zip(visible, visible[1:])]
    teleport_count = 0
    max_step_px = 0.0
    for (prev_index, prev_frame), (index, frame) in zip(visible, visible[1:]):
        gap = index - prev_index
        if gap <= 0:
            continue
        distance_px = _distance(prev_frame.xy, frame.xy)
        speed = distance_px / float(gap)
        steps.append(distance_px)
        speeds.append(speed)
        max_step_px = max(max_step_px, distance_px)
        if gap <= max_jump_gap_frames and speed > teleport_px_per_frame:
            teleport_count += 1
    return {
        "visible_segment_count": len(visible),
        "visible_coverage_rate": _ratio(len(visible), evaluated_frame_count),
        "max_visible_gap_frames": max(visible_gaps) if visible_gaps else None,
        "p95_visible_gap_frames": _percentile([float(gap) for gap in visible_gaps], 95) if visible_gaps else None,
        "long_gap_count_10f": sum(1 for gap in visible_gaps if gap > 10),
        "long_gap_count_30f": sum(1 for gap in visible_gaps if gap > 30),
        "step_count": len(steps),
        "median_step_px": _percentile(steps, 50) if steps else None,
        "p95_step_px": _percentile(steps, 95) if steps else None,
        "max_step_px": max_step_px if steps else None,
        "median_speed_px_per_frame": _percentile(speeds, 50) if speeds else None,
        "p95_speed_px_per_frame": _percentile(speeds, 95) if speeds else None,
        "teleport_count": teleport_count,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)

    aggregate: dict[str, Any] = {}
    for name, group in grouped.items():
        label_rows = [row["label_metrics"] for row in group]
        jitter_rows = [row["jitter_metrics"] for row in group]
        total_visible_labels = sum(int(row.get("visible_label_count") or 0) for row in label_rows)
        total_visible_hits = sum(int(row.get("visible_hit_count") or 0) for row in label_rows)
        total_hidden_labels = sum(int(row.get("hidden_label_count") or 0) for row in label_rows)
        total_hidden_false_positives = sum(int(row.get("hidden_false_positive_count") or 0) for row in label_rows)
        total_f1_tp = sum(int(row.get("f1_true_positive_count") or 0) for row in label_rows)
        total_f1_fp = sum(int(row.get("f1_false_positive_count") or 0) for row in label_rows)
        total_f1_fn = sum(int(row.get("f1_false_negative_count") or 0) for row in label_rows)
        total_f1_tp_10 = sum(int(row.get("f1_true_positive_count_at_10px") or 0) for row in label_rows)
        total_f1_fp_10 = sum(int(row.get("f1_false_positive_count_at_10px") or 0) for row in label_rows)
        total_f1_fn_10 = sum(int(row.get("f1_false_negative_count_at_10px") or 0) for row in label_rows)
        precision_at_20 = _ratio(total_f1_tp, total_f1_tp + total_f1_fp)
        recall_at_20 = _ratio(total_f1_tp, total_f1_tp + total_f1_fn)
        precision_at_10 = _ratio(total_f1_tp_10, total_f1_tp_10 + total_f1_fp_10)
        recall_at_10 = _ratio(total_f1_tp_10, total_f1_tp_10 + total_f1_fn_10)
        aggregate[name] = {
            "category": _first_category(group),
            "clip_count": len({row["clip"] for row in group}),
            "total_visible_label_count": total_visible_labels,
            "total_cvat_visible_label_count": sum(int(row.get("cvat_visible_label_count") or 0) for row in group),
            "total_evaluated_cvat_visible_label_count": sum(
                int(row.get("evaluated_cvat_visible_label_count") or 0) for row in group
            ),
            "total_excluded_cvat_visible_label_count": sum(
                int(row.get("excluded_cvat_visible_label_count") or 0) for row in group
            ),
            "total_excluded_cvat_frame_count": sum(int(row.get("excluded_cvat_frame_count") or 0) for row in group),
            "total_reviewed_frame_count": sum(int(row.get("reviewed_frame_count") or 0) for row in group),
            "total_evaluated_reviewed_frame_count": sum(
                int(row.get("evaluated_reviewed_frame_count") or 0) for row in group
            ),
            "total_excluded_reviewed_frame_count": sum(
                int(row.get("excluded_reviewed_frame_count") or 0) for row in group
            ),
            "total_excluded_cvat_hidden_frame_count": sum(
                int(row.get("excluded_cvat_hidden_frame_count") or 0) for row in group
            ),
            "total_visible_hit_count": total_visible_hits,
            "total_f1_tp": total_f1_tp,
            "total_f1_fp": total_f1_fp,
            "total_f1_fn": total_f1_fn,
            "micro_visible_hit_recall": _ratio(total_visible_hits, total_visible_labels),
            "micro_visible_recall_at_20px": recall_at_20,
            "micro_precision_at_20px": precision_at_20,
            "micro_label_f1_at_20px": _f1(precision_at_20, recall_at_20),
            "micro_visible_recall_at_10px": recall_at_10,
            "micro_recall_at_10px": recall_at_10,
            "micro_precision_at_10px": precision_at_10,
            "micro_label_f1_at_10px": _f1(precision_at_10, recall_at_10),
            "mean_visible_hit_recall": _mean(row.get("visible_hit_recall") for row in label_rows),
            "mean_median_error_px": _mean(row.get("median_error_px") for row in label_rows),
            "mean_p90_error_px": _mean(row.get("p90_error_px") for row in label_rows),
            "mean_p95_error_px": _mean(row.get("p95_error_px") for row in label_rows),
            "total_hidden_label_count": total_hidden_labels,
            "total_hidden_false_positive_count": total_hidden_false_positives,
            "micro_hidden_false_positive_rate": _ratio(total_hidden_false_positives, total_hidden_labels),
            "mean_hidden_false_positive_rate": _mean(row.get("hidden_false_positive_rate") for row in label_rows),
            "mean_hidden_false_positives_per_minute": _mean(row.get("hidden_false_positives_per_minute") for row in label_rows),
            "mean_visible_coverage_rate": _mean(row.get("visible_coverage_rate") for row in jitter_rows),
            "mean_max_visible_gap_frames": _mean(row.get("max_visible_gap_frames") for row in jitter_rows),
            "mean_p95_visible_gap_frames": _mean(row.get("p95_visible_gap_frames") for row in jitter_rows),
            "total_long_gap_count_10f": sum(int(row.get("long_gap_count_10f") or 0) for row in jitter_rows),
            "total_long_gap_count_30f": sum(int(row.get("long_gap_count_30f") or 0) for row in jitter_rows),
            "mean_p95_step_px": _mean(row.get("p95_step_px") for row in jitter_rows),
            "total_teleport_count": sum(int(row.get("teleport_count") or 0) for row in jitter_rows),
            "mean_quality_score": _mean(row.get("quality_score") for row in group),
        }
    return aggregate


def _candidate_paths(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "clip": str(row["clip"]),
            "candidate": str(row["candidate"]),
            "category": str(row["category"]),
            "ball_track_path": str(row["ball_track_path"]),
            "track_frame_count": int(row["track_frame_count"]),
            "track_frame_range": row.get("track_frame_range"),
            "cvat_frame_count": int(row["cvat_frame_count"]),
            "evaluated_frame_count": int(row["evaluated_frame_count"]),
            "reviewed_frame_count": int(row.get("reviewed_frame_count") or 0),
            "evaluated_reviewed_frame_count": int(row.get("evaluated_reviewed_frame_count") or 0),
            "reviewed_frame_indices_source": str(row.get("reviewed_frame_indices_source") or "dense_all_frames"),
            "evaluated_cvat_frame_range": row.get("evaluated_cvat_frame_range"),
            "excluded_cvat_frame_count": int(row["excluded_cvat_frame_count"]),
            "excluded_cvat_frame_range": row.get("excluded_cvat_frame_range"),
            "excluded_cvat_visible_label_count": int(row.get("excluded_cvat_visible_label_count") or 0),
        }
        for row in rows
    ]


def _full_horizon_coverage(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in summary.get("results", []) if isinstance(row, Mapping)]
    best_by_clip: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        clip = str(row.get("clip"))
        current = best_by_clip.get(clip)
        if current is None or _coverage_sort_key(row) > _coverage_sort_key(current):
            best_by_clip[clip] = row

    total_cvat_visible = sum(int(row.get("cvat_visible_label_count") or 0) for row in best_by_clip.values())
    total_evaluated_visible = sum(
        int(row.get("evaluated_cvat_visible_label_count") or 0) for row in best_by_clip.values()
    )
    total_excluded_visible = sum(
        int(row.get("excluded_cvat_visible_label_count") or 0) for row in best_by_clip.values()
    )
    total_cvat_frames = sum(int(row.get("cvat_frame_count") or 0) for row in best_by_clip.values())
    total_evaluated_frames = sum(int(row.get("evaluated_frame_count") or 0) for row in best_by_clip.values())
    blockers = [
        _full_horizon_blocker(clip, row)
        for clip, row in sorted(best_by_clip.items())
        if int(row.get("excluded_cvat_visible_label_count") or 0) > 0
    ]
    return {
        "clip_count": len(best_by_clip),
        "total_cvat_frame_count": total_cvat_frames,
        "total_evaluated_frame_count": total_evaluated_frames,
        "total_cvat_visible_label_count": total_cvat_visible,
        "total_evaluated_cvat_visible_label_count": total_evaluated_visible,
        "total_excluded_cvat_visible_label_count": total_excluded_visible,
        "all_cvat_visible_labels_evaluated": total_excluded_visible == 0,
        "blockers": blockers,
    }


def _coverage_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    excluded_visible = int(row.get("excluded_cvat_visible_label_count") or 0)
    evaluated_visible = int(row.get("evaluated_cvat_visible_label_count") or 0)
    evaluated_frames = int(row.get("evaluated_frame_count") or 0)
    return (-excluded_visible, evaluated_visible, evaluated_frames)


def _full_horizon_blocker(clip: str, row: Mapping[str, Any]) -> str:
    frame_range = row.get("excluded_cvat_frame_range")
    range_text = "unknown frames"
    if isinstance(frame_range, list) and len(frame_range) == 2:
        range_text = f"frames {frame_range[0]}-{frame_range[1]}"
    return (
        f"{clip} has {int(row.get('excluded_cvat_visible_label_count') or 0)} "
        f"reviewed visible ball labels beyond scored track spans over {range_text}"
    )


def _next_training_eval_recommendation(summary: Mapping[str, Any]) -> str:
    full_horizon = summary.get("full_horizon")
    if isinstance(full_horizon, Mapping) and not full_horizon.get("all_cvat_visible_labels_evaluated"):
        return (
            "Next: generate full-span candidate tracks before training or promotion, then rerun this reviewed-label "
            "benchmark over every CVAT ball box. Do not repeat the sparse TrackNetV3 fine-tune blindly; only train after "
            "the full-horizon inference baseline is scored and the remaining failure modes are clear."
        )
    return (
        "Next: keep the best full-span candidate as a benchmark candidate only, inspect its false positives and misses, "
        "and train only against dense reviewed labels or hard negatives if the held-out CVAT metrics still miss the BALL gate."
    )


def _verification_blockers(summary: Mapping[str, Any]) -> list[str]:
    blockers = [
        "No reviewed-label BALL acceptance gate is defined and passed here; every row remains a scored candidate, not a verification result."
    ]
    results = summary.get("results", [])
    if isinstance(results, Sequence):
        excluded_by_clip: dict[str, dict[str, Any]] = {}
        for row in results:
            if not isinstance(row, Mapping):
                continue
            clip = str(row.get("clip"))
            excluded_labels = int(row.get("excluded_cvat_visible_label_count") or 0)
            current = excluded_by_clip.get(clip)
            if current is None or excluded_labels > int(current.get("excluded_cvat_visible_label_count") or 0):
                excluded_by_clip[clip] = dict(row)
        excluded_details = [
            _excluded_detail_for_clip(clip, row)
            for clip, row in sorted(excluded_by_clip.items())
            if int(row.get("excluded_cvat_visible_label_count") or 0) > 0
        ]
        if excluded_details:
            blockers.append(
                "Some reviewed CVAT ball labels are outside scored track spans: " + "; ".join(excluded_details) + "."
            )

    aggregate = summary.get("aggregate")
    if isinstance(aggregate, Mapping) and aggregate:
        best_name, best_row = _best_aggregate_row(aggregate)
        if best_row is not None:
            blockers.append(
                "Best scored candidate is {name} with score {score}, F1@20 {f1}, hit recall {hit_recall}, hidden FP {hidden_fp}, and {teleports} teleports; this is not a BALL gate pass.".format(
                    name=best_name,
                    score=_fmt(best_row.get("mean_quality_score")),
                    f1=_fmt(best_row.get("micro_label_f1_at_20px")),
                    hit_recall=_fmt(best_row.get("micro_visible_hit_recall")),
                    hidden_fp=_fmt(best_row.get("micro_hidden_false_positive_rate")),
                    teleports=_fmt(best_row.get("total_teleport_count"), digits=0),
                )
            )

    contact = summary.get("contact_cue_coverage")
    if isinstance(contact, Mapping):
        contact_summary = contact.get("summary")
        if isinstance(contact_summary, Mapping):
            blockers.append(
                "Ball-contact cue coverage is incomplete: matched {matched}/{reviewed} reviewed contacts, missing {missing}, within 2 frames {within2}, p90 delta {p90} frames, with {extra} extra cues.".format(
                    matched=contact_summary.get("matched_contact_count"),
                    reviewed=contact_summary.get("reviewed_contact_count"),
                    missing=contact_summary.get("missing_reviewed_contact_count"),
                    within2=contact_summary.get("within_2_frames_count"),
                    p90=_fmt(contact_summary.get("p90_abs_delta_frames")),
                    extra=contact_summary.get("extra_cue_count"),
                )
            )
    return blockers


def _excluded_detail_for_clip(clip: str, row: Mapping[str, Any]) -> str:
    frame_range = row.get("excluded_cvat_frame_range")
    range_text = "unknown frames"
    if isinstance(frame_range, list) and len(frame_range) == 2:
        range_text = f"frames {frame_range[0]}-{frame_range[1]}"
    return f"{clip} excludes {int(row.get('excluded_cvat_visible_label_count') or 0)} visible labels over {range_text}"


def _best_aggregate_row(aggregate: Mapping[str, Any]) -> tuple[str, Mapping[str, Any] | None]:
    best_name = ""
    best_row: Mapping[str, Any] | None = None
    best_score = float("-inf")
    for name, row in aggregate.items():
        if not isinstance(row, Mapping):
            continue
        score = row.get("mean_quality_score")
        if score is None:
            continue
        numeric_score = float(score)
        if numeric_score > best_score:
            best_name = str(name)
            best_row = row
            best_score = numeric_score
    return best_name, best_row


def _evaluate_contact_cues_for_clip(
    *,
    clip: str,
    review_input: Mapping[str, Any],
    cue_root: Path,
    fps: float,
    max_match_delta_frames: float,
) -> dict[str, Any]:
    reviewed_contacts = _reviewed_contacts(review_input, clip=clip, fps=fps)
    cue_path = cue_root / clip / "ball_inflections.json"
    if not cue_path.is_file():
        return {
            "clip": clip,
            "status": STATUS_NOT_STARTED,
            "ball_inflections_path": str(cue_path),
            "reviewed_contact_count": len(reviewed_contacts),
            "cue_candidate_count": 0,
            "matched_contact_count": 0,
            "missing_reviewed_contact_count": len(reviewed_contacts),
            "extra_cue_count": 0,
            "matches": [],
            "missing_reviewed_contacts": reviewed_contacts,
            "extra_cues": [],
            "metrics": _timing_metrics([]),
            "notes": ["missing ball_inflections.json"],
        }
    cues = _ball_inflection_cues(cue_path, fps=fps)
    matches, missing, extra = _match_timed_events(
        reviewed_contacts,
        cues,
        fps=fps,
        max_match_delta_frames=max_match_delta_frames,
        right_prefix="cue",
    )
    return {
        "clip": clip,
        "status": STATUS_TESTED,
        "ball_inflections_path": str(cue_path),
        "reviewed_contact_count": len(reviewed_contacts),
        "cue_candidate_count": len(cues),
        "matched_contact_count": len(matches),
        "missing_reviewed_contact_count": len(missing),
        "extra_cue_count": len(extra),
        "cue_coverage_rate": _ratio(len(matches), len(reviewed_contacts)),
        "matches": matches,
        "missing_reviewed_contacts": missing,
        "extra_cues": extra,
        "metrics": _timing_metrics(matches),
        "notes": [],
    }


def _match_timed_events(
    reviewed_contacts: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    *,
    fps: float,
    max_match_delta_frames: float,
    right_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_pairs: list[tuple[float, int, int, float, int]] = []
    for reviewed_idx, reviewed in enumerate(reviewed_contacts):
        for cue_idx, cue in enumerate(cues):
            signed_delta_frames = (float(cue["time_s"]) - float(reviewed["time_s"])) * fps
            abs_delta_frames = abs(signed_delta_frames)
            if abs_delta_frames <= max_match_delta_frames + 1e-9:
                signed_frame_index_delta = int(cue["frame"]) - int(reviewed["frame"])
                candidate_pairs.append((abs_delta_frames, reviewed_idx, cue_idx, signed_delta_frames, signed_frame_index_delta))

    used_reviewed: set[int] = set()
    used_cues: set[int] = set()
    pair_by_reviewed: dict[int, tuple[float, int, int, float, int]] = {}
    for pair in sorted(candidate_pairs):
        _, reviewed_idx, cue_idx, _, _ = pair
        if reviewed_idx in used_reviewed or cue_idx in used_cues:
            continue
        used_reviewed.add(reviewed_idx)
        used_cues.add(cue_idx)
        pair_by_reviewed[reviewed_idx] = pair

    matches: list[dict[str, Any]] = []
    for reviewed_idx in sorted(pair_by_reviewed):
        abs_delta_frames, _, cue_idx, signed_delta_frames, signed_frame_index_delta = pair_by_reviewed[reviewed_idx]
        reviewed = reviewed_contacts[reviewed_idx]
        cue = cues[cue_idx]
        matches.append(
            {
                "reviewed_time_s": reviewed["time_s"],
                "reviewed_frame": reviewed["frame"],
                f"{right_prefix}_time_s": cue["time_s"],
                f"{right_prefix}_frame": cue["frame"],
                f"{right_prefix}_confidence": cue.get("confidence"),
                "signed_delta_frames": signed_delta_frames,
                "abs_delta_frames": abs_delta_frames,
                "signed_frame_index_delta": signed_frame_index_delta,
            }
        )
    missing = [contact for idx, contact in enumerate(reviewed_contacts) if idx not in used_reviewed]
    extra = [cue for idx, cue in enumerate(cues) if idx not in used_cues]
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
    reviewed: list[dict[str, Any]] = []
    for contact in contacts:
        if not isinstance(contact, Mapping):
            continue
        time_s = _nonnegative_finite(contact.get("time_s"), "contact time_s")
        reviewed.append({"time_s": time_s, "frame": max(0, int(round(time_s * fps)))})
    return sorted(reviewed, key=lambda item: (item["time_s"], item["frame"]))


def _ball_inflection_cues(path: Path, *, fps: float) -> list[dict[str, Any]]:
    payload = _read_json_object(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"ball inflections must contain candidates: {path}")
    cues: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        time_s = _nonnegative_finite(candidate.get("time_s"), "cue time_s")
        frame = candidate.get("frame")
        cue_frame = int(frame) if isinstance(frame, int) else max(0, int(round(time_s * fps)))
        cues.append(
            {
                "time_s": time_s,
                "frame": cue_frame,
                "confidence": candidate.get("confidence"),
            }
        )
    return sorted(cues, key=lambda item: (item["time_s"], item["frame"]))


def _contact_cue_summary(clip_reports: Sequence[Mapping[str, Any]], *, max_match_delta_frames: float) -> dict[str, Any]:
    reviewed_count = sum(int(report["reviewed_contact_count"]) for report in clip_reports)
    matched_count = sum(int(report["matched_contact_count"]) for report in clip_reports)
    missing_count = sum(int(report["missing_reviewed_contact_count"]) for report in clip_reports)
    extra_count = sum(int(report["extra_cue_count"]) for report in clip_reports)
    cue_count = sum(int(report["cue_candidate_count"]) for report in clip_reports)
    matches = [match for report in clip_reports for match in report["matches"]]
    return {
        "clip_count": len(clip_reports),
        "reviewed_contact_count": reviewed_count,
        "cue_candidate_count": cue_count,
        "matched_contact_count": matched_count,
        "missing_reviewed_contact_count": missing_count,
        "extra_cue_count": extra_count,
        "cue_coverage_rate": _ratio(matched_count, reviewed_count),
        "max_match_delta_frames": max_match_delta_frames,
        **_timing_metrics(matches),
    }


def _timing_metrics(matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signed = [float(match["signed_delta_frames"]) for match in matches]
    absolute = [abs(value) for value in signed]
    return {
        "mean_signed_delta_frames": _mean(signed),
        "mean_abs_delta_frames": _mean(absolute),
        "p50_abs_delta_frames": _percentile(absolute, 50.0) if absolute else None,
        "p90_abs_delta_frames": _percentile(absolute, 90.0) if absolute else None,
        "max_abs_delta_frames": max(absolute) if absolute else None,
        "within_1_frame_count": sum(1 for value in absolute if value <= 1.0 + 1e-9),
        "within_2_frames_count": sum(1 for value in absolute if value <= 2.0 + 1e-9),
    }


def _quality_score(label_metrics: Mapping[str, Any], jitter_metrics: Mapping[str, Any]) -> float:
    f1 = float(label_metrics.get("label_f1_at_20px") or 0.0)
    hit_recall = float(label_metrics.get("visible_hit_recall") or 0.0)
    hidden_fp = float(label_metrics.get("hidden_false_positive_rate") or 0.0)
    teleport_penalty = min(0.35, float(jitter_metrics.get("teleport_count") or 0) * 0.01)
    p90_error = label_metrics.get("p90_error_px")
    error_penalty = min(0.35, float(p90_error) / 400.0) if p90_error is not None else 0.35
    return round((0.65 * f1) + (0.35 * hit_recall) - (0.45 * hidden_fp) - teleport_penalty - error_penalty, 6)


def _read_json_object(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{json_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{json_path} must contain a JSON object")
    return payload


def _default_review_clips(review_input: Mapping[str, Any]) -> list[str]:
    clips = review_input.get("clips")
    if not isinstance(clips, Mapping):
        raise ValueError("review input must contain a clips object")
    return sorted(str(clip) for clip in clips)


def _frame_range(frames: Sequence[Any]) -> list[int] | None:
    if not frames:
        return None
    indices = [int(frame.frame_index) for frame in frames]
    return [min(indices), max(indices)]


def _index_range(indices: Any) -> list[int] | None:
    numbers = [int(index) for index in indices]
    if not numbers:
        return None
    return [min(numbers), max(numbers)]


def _first_category(rows: list[dict[str, Any]]) -> str:
    categories = {str(row["category"]) for row in rows}
    if len(categories) == 1:
        return next(iter(categories))
    return "mixed"


def _nonnegative_finite(value: Any, name: str) -> float:
    value = _positive_or_zero_finite(value, name)
    return value


def _positive_or_zero_finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def _positive_finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / float(denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _distance(a: Any, b: Any) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


__all__ = [
    "CvatBallCandidate",
    "benchmark_cvat_ball_track_candidate",
    "benchmark_cvat_ball_tracker_candidates",
    "evaluate_ball_contact_cue_coverage",
    "render_cvat_ball_tracker_benchmark_markdown",
    "write_cvat_ball_tracker_benchmark",
]
