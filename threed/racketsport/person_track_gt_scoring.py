"""Score existing player tracks against reviewed person ground truth."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .court_templates import Sport, get_court_template
from .mobile_person_eval import _match_frame, _overlaps_ignored, score_mobile_person_tracks
from .schemas import (
    OnDevicePersonFrame,
    OnDevicePersonTracks,
    OnDevicePersonTracksSummary,
    PersonGroundTruth,
    Tracks,
)


DEFAULT_IDF1_THRESHOLD = 0.85
DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD = 0.95
FAILURE_MODE_ORDER = (
    "missing_gt_detections",
    "spectator_or_background_false_positives",
    "off_court_false_positives",
    "four_player_coverage_gap",
    "id_switches",
)
FAILURE_MODE_DESCRIPTIONS = {
    "missing_gt_detections": "reviewed player boxes with no matching prediction",
    "spectator_or_background_false_positives": "predictions that did not match reviewed or ignored GT",
    "off_court_false_positives": "unmatched predictions whose world point is outside the court template",
    "four_player_coverage_gap": "GT frames without exactly the expected player count predicted",
    "id_switches": "GT identities matched to a different predicted track id",
}


def derive_track_source_id(path: str | Path, *, clip_ids: list[str]) -> str:
    parts = list(Path(path).parts)
    if parts and parts[-1] == "tracks.json":
        parts = parts[:-1]

    if len(parts) >= 3 and parts[0] == "runs" and parts[1] == "eval0":
        rest = parts[2:]
        if len(rest) >= 2 and rest[1] in clip_ids:
            if len(rest) == 2:
                return f"eval0/{rest[0]}/canonical_tracks"
            return "eval0/" + "/".join([rest[0], *rest[2:]])
        return "eval0/" + "/".join(_without_clip_ids(rest, clip_ids=clip_ids))

    if len(parts) >= 3 and parts[0] == "runs" and parts[1] == "phase2":
        rest = _without_clip_ids(parts[2:], clip_ids=clip_ids)
        while len(rest) >= 2 and rest[-1] == rest[-2]:
            rest.pop()
        return "phase2/" + "/".join(rest)

    return "/".join(_without_clip_ids(parts, clip_ids=clip_ids))


def score_tracks_against_person_ground_truth(
    *,
    ground_truth: PersonGroundTruth,
    tracks: Tracks,
    candidate: str,
    tracks_path: str | Path,
    iou_threshold: float = 0.5,
    expected_players: int | None = None,
    bbox_scale_x: float = 1.0,
    bbox_scale_y: float = 1.0,
    sport: Sport = "pickleball",
) -> dict[str, Any]:
    predictions, prediction_world, outside_gt = _tracks_to_predictions(
        ground_truth=ground_truth,
        tracks=tracks,
        candidate=candidate,
        bbox_scale_x=bbox_scale_x,
        bbox_scale_y=bbox_scale_y,
    )
    expected = expected_players if expected_players is not None else ground_truth.summary.max_valid_players_per_frame
    metrics = score_mobile_person_tracks(
        ground_truth,
        predictions,
        iou_threshold=iou_threshold,
        expected_players=expected,
    )
    false_positive_details = _false_positive_details(
        ground_truth=ground_truth,
        predictions=predictions,
        prediction_world=prediction_world,
        iou_threshold=iou_threshold,
        sport=sport,
    )
    switch_diagnostics = _identity_switch_diagnostics(
        ground_truth=ground_truth,
        predictions=predictions,
        iou_threshold=iou_threshold,
    )

    track_frame_count = sum(len(player.frames) for player in tracks.players)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_track_gt_score",
        "clip_id": ground_truth.clip_id,
        "candidate": candidate,
        "tracks_path": str(tracks_path),
        "iou_threshold": iou_threshold,
        "bbox_scale_x": bbox_scale_x,
        "bbox_scale_y": bbox_scale_y,
        "gt_frame_count": ground_truth.summary.frame_count,
        "gt_detections": metrics.gt_detections,
        "pred_detections": metrics.pred_detections,
        "matches": metrics.matches,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "spectator_or_background_false_positives": metrics.false_positives,
        "id_switches": metrics.id_switches,
        "idf1": metrics.idf1,
        "mota": metrics.mota,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "expected_players": metrics.expected_players,
        "four_player_coverage": metrics.expected_player_coverage,
        "expected_four_player_frames": metrics.expected_player_frames,
        "exact_four_player_frames": metrics.exact_expected_player_frames,
        "track_count": len(tracks.players),
        "track_frame_count": track_frame_count,
        "tracks_fps": tracks.fps,
        "outside_gt_prediction_count": outside_gt["prediction_count"],
        "outside_gt_prediction_track_ids": outside_gt["track_ids"],
        "identity_switch_event_count": switch_diagnostics["event_count"],
        "identity_switch_events": switch_diagnostics["events"],
        "identity_switch_transitions": switch_diagnostics["transitions"],
        "temporal_coverage": _temporal_coverage_diagnostics(ground_truth, predictions),
        **false_positive_details,
    }


def build_scoring_report(
    rows: list[dict[str, Any]],
    *,
    required_clip_ids: list[str],
    iou_threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["track_source_id"])].append(row)

    sources = []
    for source_id in sorted(grouped):
        source_rows = [
            _annotate_failure_modes(row)
            for row in sorted(grouped[source_id], key=lambda row: str(row.get("clip_id")))
        ]
        decision = build_source_promotion_decision(source_rows, required_clip_ids=required_clip_ids)
        sources.append(
            {
                "track_source_id": source_id,
                "clip_count": len({row.get("clip_id") for row in source_rows}),
                "clips": [row.get("clip_id") for row in source_rows],
                "decision": decision,
                "aggregate": _aggregate_source_rows(source_rows),
                "failure_analysis": _aggregate_failure_analysis(source_rows),
                "rows": source_rows,
            }
        )

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_track_gt_scoring_report",
        "status": "scored_existing_tracks_only",
        "iou_threshold": iou_threshold,
        "required_clip_ids": required_clip_ids,
        "track_source_count": len(sources),
        "track_file_count": len(rows),
        "promotion_policy": {
            "idf1_threshold": DEFAULT_IDF1_THRESHOLD,
            "requires_zero_id_switches": True,
            "requires_zero_spectator_or_background_false_positives": True,
            "requires_zero_off_court_false_positive_frames": True,
            "four_player_coverage_threshold": DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD,
        },
        "sources": sources,
    }


def render_scoring_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Person Track GT Scoring",
        "",
        "- status: `scored_existing_tracks_only`",
        f"- IoU threshold: `{report['iou_threshold']}`",
        f"- track sources: `{report['track_source_count']}`",
        f"- track files: `{report['track_file_count']}`",
        "- inference: not run",
        "",
        "Promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero spectator/background false positives, zero off-court false-positive frames, and four-player coverage >= 0.95.",
        "",
        "## Source Decisions",
        "",
        "| Source | Decision | Clips | Mean IDF1 | Worst IDF1 | Switches | FP | Off-court FP | Mean cov4 | Worst cov4 | FPS | Primary failure | Blockers |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for source in report["sources"]:
        aggregate = source["aggregate"]
        decision = source["decision"]
        fps = aggregate.get("mean_effective_fps")
        lines.append(
            "| `{source}` | `{decision}` | {clips} | {mean_idf1} | {worst_idf1} | {switches} | {fp} | {offcourt} | {mean_cov} | {worst_cov} | {fps} | {primary_failure} | {blockers} |".format(
                source=source["track_source_id"],
                decision=decision["status"],
                clips=source["clip_count"],
                mean_idf1=_fmt(aggregate.get("mean_idf1")),
                worst_idf1=_fmt(aggregate.get("worst_idf1")),
                switches=aggregate.get("total_id_switches"),
                fp=aggregate.get("total_spectator_or_background_false_positives"),
                offcourt=aggregate.get("total_off_court_false_positive_frames"),
                mean_cov=_fmt(aggregate.get("mean_four_player_coverage")),
                worst_cov=_fmt(aggregate.get("worst_four_player_coverage")),
                fps=_fmt(fps) if fps is not None else "n/a",
                primary_failure=source.get("failure_analysis", {}).get("primary_failure_mode", "none"),
                blockers=", ".join(decision["blockers"]) if decision["blockers"] else "none",
            )
        )

    lines.extend(
        [
            "",
            "## Clip Scores",
            "",
            "| Source | Clip | IDF1 | MOTA | Switches | FP | FN | Off-court FP | cov4 | exact/expected cov4 frames | FPS | Tracks | Primary failure | Path |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for source in report["sources"]:
        for row in source["rows"]:
            fps = _timing_value(row, "effective_fps")
            lines.append(
                "| `{source}` | {clip} | {idf1} | {mota} | {switches} | {fp} | {fn} | {offcourt} | {cov} | {frames} | {fps} | {tracks} | {primary_failure} | `{path}` |".format(
                    source=source["track_source_id"],
                    clip=row["clip_id"],
                    idf1=_fmt(row["idf1"]),
                    mota=_fmt(row["mota"]),
                    switches=row["id_switches"],
                    fp=row["spectator_or_background_false_positives"],
                    fn=row["false_negatives"],
                    offcourt=row["off_court_false_positive_frames"],
                    cov=_fmt(row["four_player_coverage"]),
                    frames=f"{row['exact_four_player_frames']}/{row['expected_four_player_frames']}",
                    fps=_fmt(fps) if fps is not None else "n/a",
                    tracks=row["track_count"],
                    primary_failure=row.get("primary_failure_mode", "none"),
                    path=row["tracks_path"],
                )
            )
    lines.append("")

    lines.extend(
        [
            "## Temporal Coverage Diagnostics",
            "",
            "| Source | Clip | GT range | Prediction range | GT frames after last prediction | GT detections after last prediction | GT frames without predictions |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for source in report["sources"]:
        for row in source["rows"]:
            temporal = row.get("temporal_coverage") if isinstance(row.get("temporal_coverage"), dict) else {}
            gt_range = temporal.get("gt_frame_range") if isinstance(temporal.get("gt_frame_range"), dict) else {}
            pred_range = (
                temporal.get("prediction_frame_range")
                if isinstance(temporal.get("prediction_frame_range"), dict)
                else {}
            )
            lines.append(
                "| `{source}` | {clip} | {gt_range} | {pred_range} | {gt_after} | {det_after} | {without_pred} |".format(
                    source=source["track_source_id"],
                    clip=row["clip_id"],
                    gt_range=_range_text(gt_range),
                    pred_range=_range_text(pred_range),
                    gt_after=temporal.get("gt_frames_after_last_prediction", "n/a"),
                    det_after=temporal.get("gt_detections_after_last_prediction", "n/a"),
                    without_pred=temporal.get("gt_frames_without_predictions", "n/a"),
                )
            )

    lines.extend(
        [
            "",
            "## Identity Switch Events",
            "",
            "Full per-row switch event lists are in the JSON report. Markdown shows the first 10 events per scored clip.",
            "",
            "| Source | Clip | Frame | GT id | Previous pred id | New pred id | Previous match frame | Gap frames | IoU |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source in report["sources"]:
        for row in source["rows"]:
            for event in row.get("identity_switch_events", [])[:10]:
                lines.append(
                    "| `{source}` | {clip} | {frame} | {gt_id} | {prev_pred} | {new_pred} | {prev_frame} | {gap} | {iou} |".format(
                        source=source["track_source_id"],
                        clip=row["clip_id"],
                        frame=event["frame_index"],
                        gt_id=event["gt_track_id"],
                        prev_pred=event["previous_pred_track_id"],
                        new_pred=event["new_pred_track_id"],
                        prev_frame=event["previous_match_frame_index"],
                        gap=event["frames_since_previous_match"],
                        iou=_fmt(event.get("iou")),
                    )
                )
    lines.append("")
    return "\n".join(lines)


def build_source_promotion_decision(
    rows: list[dict[str, Any]],
    *,
    required_clip_ids: list[str],
    idf1_threshold: float = DEFAULT_IDF1_THRESHOLD,
    four_player_coverage_threshold: float = DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    blockers: list[str] = []
    rows_by_clip = {str(row.get("clip_id")): row for row in rows}
    missing = [clip_id for clip_id in required_clip_ids if clip_id not in rows_by_clip]
    if missing:
        blockers.append("missing_required_clips:" + ",".join(missing))

    for clip_id in sorted(rows_by_clip):
        row = rows_by_clip[clip_id]
        if float(row.get("idf1", 0.0)) < idf1_threshold:
            blockers.append(f"{clip_id}:idf1_below_{idf1_threshold:.2f}")
        if int(row.get("id_switches", 0)) > 0:
            blockers.append(f"{clip_id}:id_switches_present")
        if int(row.get("spectator_or_background_false_positives", 0)) > 0:
            blockers.append(f"{clip_id}:spectator_or_background_false_positives_present")
        if int(row.get("off_court_false_positive_frames", 0)) > 0:
            blockers.append(f"{clip_id}:off_court_false_positives_present")
        if float(row.get("four_player_coverage", 0.0)) < four_player_coverage_threshold:
            blockers.append(f"{clip_id}:four_player_coverage_below_{four_player_coverage_threshold:.2f}")

    return {
        "promote": not blockers,
        "status": "promote" if not blockers else "do_not_promote",
        "blockers": blockers,
        "policy": {
            "required_clip_ids": required_clip_ids,
            "idf1_threshold": idf1_threshold,
            "requires_zero_id_switches": True,
            "requires_zero_spectator_or_background_false_positives": True,
            "requires_zero_off_court_false_positive_frames": True,
            "four_player_coverage_threshold": four_player_coverage_threshold,
        },
    }


def summarize_score_failure_modes(row: dict[str, Any]) -> list[dict[str, Any]]:
    modes = [
        _failure_mode_record(mode, count, denominator)
        for mode, (count, denominator) in _failure_mode_counts(row).items()
        if count > 0
    ]
    return sorted(modes, key=_failure_mode_sort_key)


def _annotate_failure_modes(row: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(row)
    modes = summarize_score_failure_modes(annotated)
    annotated["failure_modes"] = modes
    annotated["primary_failure_mode"] = modes[0]["mode"] if modes else "none"
    return annotated


def _aggregate_failure_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {mode: {"count": 0, "denominator": 0} for mode in FAILURE_MODE_ORDER}
    for row in rows:
        for mode, (count, denominator) in _failure_mode_counts(row).items():
            totals[mode]["count"] += count
            if denominator is not None:
                totals[mode]["denominator"] += denominator

    modes = [
        _failure_mode_record(
            mode,
            values["count"],
            values["denominator"] if values["denominator"] > 0 else None,
        )
        for mode, values in totals.items()
        if values["count"] > 0
    ]
    modes = sorted(modes, key=_failure_mode_sort_key)
    return {
        "primary_failure_mode": modes[0]["mode"] if modes else "none",
        "modes": modes,
    }


def _failure_mode_counts(row: dict[str, Any]) -> dict[str, tuple[int, int | None]]:
    expected_four_player_frames = _nonnegative_int(row.get("expected_four_player_frames"))
    exact_four_player_frames = _nonnegative_int(row.get("exact_four_player_frames"))
    return {
        "missing_gt_detections": (
            _nonnegative_int(row.get("false_negatives")),
            _positive_int(row.get("gt_detections")),
        ),
        "spectator_or_background_false_positives": (
            _nonnegative_int(row.get("spectator_or_background_false_positives")),
            _positive_int(row.get("pred_detections")),
        ),
        "off_court_false_positives": (
            _nonnegative_int(row.get("off_court_false_positive_frames")),
            _positive_int(row.get("pred_detections")),
        ),
        "four_player_coverage_gap": (
            max(0, expected_four_player_frames - exact_four_player_frames),
            expected_four_player_frames if expected_four_player_frames > 0 else None,
        ),
        "id_switches": (
            _nonnegative_int(row.get("id_switches")),
            _positive_int(row.get("gt_detections")),
        ),
    }


def _failure_mode_record(mode: str, count: int, denominator: int | None) -> dict[str, Any]:
    return {
        "mode": mode,
        "count": count,
        "rate": (count / denominator) if denominator else None,
        "denominator": denominator,
        "description": FAILURE_MODE_DESCRIPTIONS[mode],
    }


def _failure_mode_sort_key(mode: dict[str, Any]) -> tuple[int, int]:
    order = FAILURE_MODE_ORDER.index(str(mode["mode"]))
    return (-int(mode["count"]), order)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _positive_int(value: Any) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed > 0 else None


def _aggregate_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    idf1_values = [float(row["idf1"]) for row in rows]
    coverage_values = [float(row["four_player_coverage"]) for row in rows]
    fps_values = [
        value
        for row in rows
        for value in [_timing_value(row, "effective_fps")]
        if value is not None
    ]
    return {
        "mean_idf1": sum(idf1_values) / len(idf1_values) if idf1_values else 0.0,
        "worst_idf1": min(idf1_values) if idf1_values else 0.0,
        "mean_four_player_coverage": sum(coverage_values) / len(coverage_values) if coverage_values else 0.0,
        "worst_four_player_coverage": min(coverage_values) if coverage_values else 0.0,
        "total_id_switches": sum(int(row["id_switches"]) for row in rows),
        "total_spectator_or_background_false_positives": sum(
            int(row["spectator_or_background_false_positives"]) for row in rows
        ),
        "total_off_court_false_positive_frames": sum(int(row["off_court_false_positive_frames"]) for row in rows),
        "total_false_negatives": sum(int(row["false_negatives"]) for row in rows),
        "mean_effective_fps": sum(fps_values) / len(fps_values) if fps_values else None,
    }


def _timing_value(row: dict[str, Any], key: str) -> float | None:
    timing = row.get("timing")
    if not isinstance(timing, dict):
        return None
    value = timing.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _without_clip_ids(parts: list[str], *, clip_ids: list[str]) -> list[str]:
    return [part for part in parts if part not in set(clip_ids)]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _range_text(value: dict[str, Any]) -> str:
    first = value.get("first")
    last = value.get("last")
    if first is None or last is None:
        return "n/a"
    return f"{first}-{last}"


def _identity_switch_diagnostics(
    *,
    ground_truth: PersonGroundTruth,
    predictions: OnDevicePersonTracks,
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_frame = {frame.frame_index: [label for label in frame.labels if not label.ignored] for frame in ground_truth.frames}
    pred_by_frame = {frame.frame_index: frame.detections for frame in predictions.frames}
    last_match_for_gt: dict[int, tuple[int, int]] = {}
    events: list[dict[str, Any]] = []
    transition_counts: dict[tuple[int, int, int], dict[str, int]] = {}

    for frame_index in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_labels = gt_by_frame.get(frame_index, [])
        pred_labels = pred_by_frame.get(frame_index, [])
        for gt_index, pred_index, iou in sorted(_match_frame(gt_labels, pred_labels, iou_threshold=iou_threshold)):
            gt_id = int(gt_labels[gt_index].track_id)
            pred_id = int(pred_labels[pred_index].track_id)
            previous = last_match_for_gt.get(gt_id)
            if previous is not None and previous[0] != pred_id:
                previous_pred_id, previous_frame_index = previous
                events.append(
                    {
                        "frame_index": frame_index,
                        "gt_track_id": gt_id,
                        "previous_pred_track_id": previous_pred_id,
                        "new_pred_track_id": pred_id,
                        "previous_match_frame_index": previous_frame_index,
                        "frames_since_previous_match": frame_index - previous_frame_index,
                        "iou": iou,
                    }
                )
                key = (gt_id, previous_pred_id, pred_id)
                transition = transition_counts.setdefault(
                    key,
                    {
                        "gt_track_id": gt_id,
                        "previous_pred_track_id": previous_pred_id,
                        "new_pred_track_id": pred_id,
                        "count": 0,
                        "first_frame_index": frame_index,
                        "last_frame_index": frame_index,
                    },
                )
                transition["count"] += 1
                transition["last_frame_index"] = frame_index
            last_match_for_gt[gt_id] = (pred_id, frame_index)

    transitions = sorted(
        transition_counts.values(),
        key=lambda transition: (
            -transition["count"],
            transition["first_frame_index"],
            transition["gt_track_id"],
            transition["previous_pred_track_id"],
            transition["new_pred_track_id"],
        ),
    )
    return {"event_count": len(events), "events": events, "transitions": transitions}


def _temporal_coverage_diagnostics(
    ground_truth: PersonGroundTruth,
    predictions: OnDevicePersonTracks,
) -> dict[str, Any]:
    gt_by_frame = {frame.frame_index: [label for label in frame.labels if not label.ignored] for frame in ground_truth.frames}
    pred_by_frame = {frame.frame_index: frame.detections for frame in predictions.frames}
    gt_indexes = sorted(gt_by_frame)
    pred_indexes = sorted(pred_by_frame)
    last_prediction = pred_indexes[-1] if pred_indexes else None

    if last_prediction is None:
        gt_after_last_prediction = gt_indexes
    else:
        gt_after_last_prediction = [frame_index for frame_index in gt_indexes if frame_index > last_prediction]
    gt_without_predictions = [frame_index for frame_index in gt_indexes if frame_index not in pred_by_frame]

    return {
        "gt_frame_range": _frame_range(gt_indexes),
        "prediction_frame_range": _frame_range(pred_indexes),
        "gt_frame_count": len(gt_indexes),
        "prediction_frame_count": len(pred_indexes),
        "gt_frames_after_last_prediction": len(gt_after_last_prediction),
        "gt_detections_after_last_prediction": sum(len(gt_by_frame[frame_index]) for frame_index in gt_after_last_prediction),
        "gt_frames_without_predictions": len(gt_without_predictions),
        "gt_detections_without_predictions": sum(len(gt_by_frame[frame_index]) for frame_index in gt_without_predictions),
    }


def _frame_range(frame_indexes: list[int]) -> dict[str, int | None]:
    if not frame_indexes:
        return {"first": None, "last": None}
    return {"first": frame_indexes[0], "last": frame_indexes[-1]}


def _tracks_to_predictions(
    *,
    ground_truth: PersonGroundTruth,
    tracks: Tracks,
    candidate: str,
    bbox_scale_x: float,
    bbox_scale_y: float,
) -> tuple[OnDevicePersonTracks, dict[int, list[tuple[int, list[float]]]], dict[str, Any]]:
    total_frames = ground_truth.summary.frame_count
    detections_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    world_by_frame: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    outside_count = 0
    outside_track_ids: set[int] = set()

    for player in tracks.players:
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(tracks.fps)))
            if frame_index < 0:
                continue
            if frame_index >= total_frames:
                outside_count += 1
                outside_track_ids.add(int(player.id))
                continue
            x1, y1, x2, y2 = [float(value) for value in frame.bbox]
            x1 *= bbox_scale_x
            x2 *= bbox_scale_x
            y1 *= bbox_scale_y
            y2 *= bbox_scale_y
            detection = {
                "track_id": int(player.id),
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                "confidence": float(frame.conf),
                "source": "tracks_json",
                "role": player.role,
            }
            detections_by_frame[frame_index].append(detection)
            world_by_frame[frame_index].append((int(player.id), [float(value) for value in frame.world_xy]))

    frames = [
        OnDevicePersonFrame.model_validate({"frame_index": frame_index, "detections": detections_by_frame[frame_index]})
        for frame_index in sorted(detections_by_frame)
    ]
    track_ids = sorted(
        {
            int(detection["track_id"])
            for detections in detections_by_frame.values()
            for detection in detections
        }
    )
    predictions = OnDevicePersonTracks(
        schema_version=1,
        artifact_type="racketsport_on_device_person_tracks",
        clip_id=ground_truth.clip_id,
        candidate=candidate,
        fps=tracks.fps,
        frames=frames,
        summary=OnDevicePersonTracksSummary(
            frame_count=total_frames,
            detection_count=sum(len(frame.detections) for frame in frames),
            track_ids=track_ids,
        ),
    )
    return predictions, world_by_frame, {"prediction_count": outside_count, "track_ids": sorted(outside_track_ids)}


def _false_positive_details(
    *,
    ground_truth: PersonGroundTruth,
    predictions: OnDevicePersonTracks,
    prediction_world: dict[int, list[tuple[int, list[float]]]],
    iou_threshold: float,
    sport: Sport,
) -> dict[str, Any]:
    gt_by_frame = {frame.frame_index: [label for label in frame.labels if not label.ignored] for frame in ground_truth.frames}
    ignored_by_frame = {frame.frame_index: [label for label in frame.labels if label.ignored] for frame in ground_truth.frames}
    pred_by_frame = {frame.frame_index: frame.detections for frame in predictions.frames}
    template = get_court_template(sport)
    half_width_m = template.width_m / 2.0
    half_length_m = template.length_m / 2.0

    off_court_count = 0
    off_court_track_ids: set[int] = set()
    false_positive_frames: list[int] = []
    for frame_index in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_labels = gt_by_frame.get(frame_index, [])
        ignored_labels = ignored_by_frame.get(frame_index, [])
        pred_labels = pred_by_frame.get(frame_index, [])
        frame_matches = _match_frame(gt_labels, pred_labels, iou_threshold=iou_threshold)
        matched_pred_indexes = {pred_index for _, pred_index, _ in frame_matches}
        world_entries = prediction_world.get(frame_index, [])
        for pred_index, pred in enumerate(pred_labels):
            if pred_index in matched_pred_indexes:
                continue
            if _overlaps_ignored(pred.bbox_xywh, ignored_labels, threshold=iou_threshold):
                continue
            false_positive_frames.append(frame_index)
            if pred_index >= len(world_entries):
                continue
            track_id, world_xy = world_entries[pred_index]
            x, y = world_xy
            if x < -half_width_m or x > half_width_m or y < -half_length_m or y > half_length_m:
                off_court_count += 1
                off_court_track_ids.add(track_id)

    return {
        "false_positive_frame_count": len(set(false_positive_frames)),
        "off_court_false_positive_frames": off_court_count,
        "off_court_false_positive_track_ids": sorted(off_court_track_ids),
    }


__all__ = [
    "DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD",
    "DEFAULT_IDF1_THRESHOLD",
    "build_scoring_report",
    "build_source_promotion_decision",
    "derive_track_source_id",
    "render_scoring_report_markdown",
    "score_tracks_against_person_ground_truth",
    "summarize_score_failure_modes",
]
