"""Run one preregistered audio-soft-split preset and write honest metrics."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from threed.racketsport import ball_arc_solver as solver
from threed.racketsport.ball_arc_chain import run_default_ball_arc_chain
from threed.racketsport.ball_arc_solver import SoftSegmentBoundary


LANE_DIR = Path(__file__).resolve().parent
REGISTRATION_PATH = LANE_DIR / "PRESET_REGISTRATION.json"
BASELINE_METRICS_PATH = ROOT / "runs/lanes/ballarc_scale_guard_20260715/full_guard5_r4_metrics.json"
BASELINE_ARTIFACT_PATH = ROOT / "runs/lanes/ballarc_scale_guard_20260715/full_guard5_r4/ball_track_arc_solved.json"


def emit(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, "monotonic_s": time.monotonic(), **payload}, sort_keys=True), flush=True)


def read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def frame_rows(ball_track: Mapping[str, Any]) -> list[dict[str, Any]]:
    fps = float(ball_track.get("fps") or 30.0)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(list(ball_track.get("frames") or [])):
        frame = dict(raw) if isinstance(raw, Mapping) else {}
        frame_index = int(frame.get("frame", frame.get("frame_index", index)))
        frame["frame"] = frame_index
        frame["t"] = float(frame.get("t", frame.get("time_s", frame_index / fps)))
        rows.append(frame)
    return rows


def rally_active_spans(
    ball_track: Mapping[str, Any],
    *,
    max_visible_gap_s: float,
    minimum_span_duration_s: float,
) -> list[tuple[float, float]]:
    visible_times = [
        float(frame["t"])
        for frame in frame_rows(ball_track)
        if frame.get("visible") is True and frame.get("xy") is not None
    ]
    spans: list[list[float]] = []
    for onset_time in sorted(visible_times):
        if not spans or onset_time - spans[-1][1] > max_visible_gap_s:
            spans.append([onset_time, onset_time])
        else:
            spans[-1][1] = onset_time
    return [
        (start, end)
        for start, end in spans
        if end - start >= minimum_span_duration_s
    ]


def in_spans(value: float, spans: Sequence[tuple[float, float]]) -> bool:
    return any(start - 1e-9 <= value <= end + 1e-9 for start, end in spans)


def select_boundaries(
    onset_payload: Mapping[str, Any],
    *,
    spans: Sequence[tuple[float, float]],
    score_floor: float,
    minimum_spacing_s: float,
    selection_rule_id: str,
    source_artifact: str,
) -> tuple[list[SoftSegmentBoundary], dict[str, Any]]:
    raw_onsets = [item for item in list(onset_payload.get("onsets") or []) if isinstance(item, Mapping)]
    passing: list[Mapping[str, Any]] = []
    for onset in raw_onsets:
        corrected = onset.get("corrected_time_s")
        score = onset.get("score")
        try:
            corrected_time_s = float(corrected)
            onset_score = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(corrected_time_s) or not math.isfinite(onset_score):
            continue
        if onset_score < score_floor or not in_spans(corrected_time_s, spans):
            continue
        passing.append(onset)
    ranked = sorted(
        passing,
        key=lambda onset: (
            -float(onset["score"]),
            float(onset["corrected_time_s"]),
            int(onset.get("onset_order", onset.get("corrected_order", 0))),
        ),
    )
    selected: list[Mapping[str, Any]] = []
    selected_times: list[float] = []
    for onset in ranked:
        corrected_time_s = float(onset["corrected_time_s"])
        insert_at = bisect.bisect_left(selected_times, corrected_time_s)
        neighbor_times = selected_times[max(0, insert_at - 1) : insert_at + 1]
        if any(abs(corrected_time_s - other) < minimum_spacing_s - 1e-12 for other in neighbor_times):
            continue
        selected_times.insert(insert_at, corrected_time_s)
        selected.insert(insert_at, onset)
    boundaries = []
    for onset in selected:
        onset_order = int(onset.get("onset_order", onset.get("corrected_order", 0)))
        boundaries.append(
            SoftSegmentBoundary(
                boundary_id=f"audio_onset_soft_{onset_order:04d}",
                corrected_time_s=float(onset["corrected_time_s"]),
                frame=int(onset.get("nearest_frame", round(float(onset["corrected_time_s"]) * 30.0))),
                onset_ids=(f"onset_{onset_order:04d}",),
                selection_rule_id=selection_rule_id,
                source_artifact=source_artifact,
            )
        )
    return boundaries, {
        "raw_onset_count": len(raw_onsets),
        "passing_floor_and_rally_count": len(passing),
        "selected_count": len(boundaries),
        "score_floor": score_floor,
        "minimum_spacing_s": minimum_spacing_s,
        "selection_rule_id": selection_rule_id,
    }


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "min": None, "median": None, "p90": None, "p95": None, "max": None}

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def segment_durations(artifact: Mapping[str, Any]) -> list[float]:
    output = []
    for segment in list(artifact.get("segments") or []):
        if not isinstance(segment, Mapping):
            continue
        try:
            output.append(float(segment["t1"]) - float(segment["t0"]))
        except (KeyError, TypeError, ValueError):
            continue
    return output


def coverage_metrics(
    artifact: Mapping[str, Any],
    *,
    track_frames: Sequence[Mapping[str, Any]],
    spans: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    segments = [item for item in list(artifact.get("segments") or []) if isinstance(item, Mapping)]
    fitted = [item for item in segments if str(item.get("status") or "").startswith("fit")]
    in_rally_frames = [frame for frame in track_frames if in_spans(float(frame["t"]), spans)]
    covered_frames = [
        frame
        for frame in in_rally_frames
        if any(float(segment["t0"]) - 1e-9 <= float(frame["t"]) <= float(segment["t1"]) + 1e-9 for segment in fitted)
    ]
    return {
        "definition_in_rally_frame_coverage": "number of input frames whose timestamp is inside a preregistered rally-active span and inside any emitted status=fit* segment, divided by all input frames in those rally-active spans",
        "definition_total_segment_fit_fraction": "number of emitted solver segments with status beginning fit divided by all emitted solver segments (confident plus weak)",
        "in_rally_frame_count": len(in_rally_frames),
        "in_rally_frame_inside_fitted_segment_count": len(covered_frames),
        "in_rally_frame_coverage_fraction": len(covered_frames) / len(in_rally_frames) if in_rally_frames else 0.0,
        "in_rally_frame_coverage_percent": 100.0 * len(covered_frames) / len(in_rally_frames) if in_rally_frames else 0.0,
        "segment_count": len(segments),
        "segments_fit_count": len(fitted),
        "total_segment_fit_fraction": len(fitted) / len(segments) if segments else 0.0,
        "total_segment_fit_percent": 100.0 * len(fitted) / len(segments) if segments else 0.0,
    }


def provenance_audit(artifact: Mapping[str, Any]) -> dict[str, Any]:
    missing: list[int] = []
    invalid: list[int] = []
    soft_segment_count = 0
    for raw in list(artifact.get("segments") or []):
        if not isinstance(raw, Mapping):
            continue
        anchors = [anchor for anchor in list(raw.get("anchors_used") or []) if isinstance(anchor, Mapping)]
        has_soft_endpoint = any(anchor.get("kind") == "audio_onset_soft" for anchor in anchors)
        if not has_soft_endpoint:
            continue
        soft_segment_count += 1
        provenance = [item for item in list(raw.get("soft_split_provenance") or []) if isinstance(item, Mapping)]
        if not provenance:
            missing.append(int(raw.get("segment_id", -1)))
            continue
        required = {"anchor_class", "onset_ids", "corrected_time_s", "selection_rule_id"}
        if any(
            not required.issubset(item)
            or item.get("anchor_class") != "audio_onset_soft"
            or item.get("event_type") is not None
            or item.get("world_constraint") is not None
            or item.get("counts_as_bounce_evidence") is not False
            or item.get("counts_as_flight_sanity_anchor") is not False
            for item in provenance
        ):
            invalid.append(int(raw.get("segment_id", -1)))
    return {
        "soft_split_segment_count": soft_segment_count,
        "missing_provenance_segment_ids": missing,
        "invalid_provenance_segment_ids": invalid,
        "pass": not missing and not invalid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--onsets", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, default=5.0)
    args = parser.parse_args()

    registration = read_object(REGISTRATION_PATH)
    preset = next(
        (item for item in registration["presets"] if item["selection_rule_id"] == args.preset),
        None,
    )
    if preset is None:
        raise ValueError(f"preset is not preregistered: {args.preset}")
    input_dir = args.input_dir.resolve()
    ball_track = read_object(input_dir / "ball_track.json")
    track_frames = frame_rows(ball_track)
    onsets = read_object(args.onsets.resolve())
    span_rule = registration["shared_rule"]["rally_active_span_rule"]
    spans = rally_active_spans(
        ball_track,
        max_visible_gap_s=1.0,
        minimum_span_duration_s=float(span_rule["minimum_span_duration_s"]),
    )
    boundaries, selection = select_boundaries(
        onsets,
        spans=spans,
        score_floor=float(preset["score_floor"]),
        minimum_spacing_s=float(preset["min_inter_split_spacing_s"]),
        selection_rule_id=str(preset["selection_rule_id"]),
        source_artifact=str(args.onsets.resolve()),
    )
    solver.SEGMENT_WALL_CLOCK_BUDGET_S = float(args.budget_s)
    original_pair = solver._fit_anchor_pair
    original_weak = solver.fit_weak_flight_segment

    def timed_pair(segment_id: int, start: solver.AnchorEvent, end: solver.AnchorEvent, **kwargs: object):
        started = time.monotonic()
        result = original_pair(segment_id, start, end, **kwargs)
        emit(
            "segment_end",
            segment_id=segment_id,
            t0=start.t,
            t1=end.t,
            duration_s=end.t - start.t,
            start_kind=start.kind,
            end_kind=end.kind,
            wall_s=time.monotonic() - started,
            status=None if result is None else result.status,
            degradation=None if result is None else result.degradation,
        )
        return result

    def timed_weak(**kwargs: object):
        started = time.monotonic()
        result = original_weak(**kwargs)
        emit(
            "weak_segment_end",
            segment_id=kwargs["segment_id"],
            wall_s=time.monotonic() - started,
            status=result.status,
            degradation=result.degradation,
        )
        return result

    solver._fit_anchor_pair = timed_pair
    solver.fit_weak_flight_segment = timed_weak
    emit(
        "run_start",
        clip=args.clip,
        preset=preset,
        selection=selection,
        rally_span_count=len(spans),
        rally_span_duration_s=sum(end - start for start, end in spans),
        guard_budget_s=args.budget_s,
    )
    started = time.monotonic()
    result = run_default_ball_arc_chain(
        clip=args.clip,
        ball_track_path=input_dir / "ball_track.json",
        court_calibration_path=input_dir / "court_calibration.json",
        ball_candidate_paths=[input_dir / "ball_candidates.json"],
        frame_times_path=input_dir / "frame_times.json",
        net_plane_path=input_dir / "net_plane.json",
        out_dir=args.out_dir.resolve(),
        soft_split_boundaries=boundaries,
        generated_at="2026-07-16T00:00:00Z",
    )
    wall_s = time.monotonic() - started
    artifact = read_object(Path(result["outputs"]["ball_track_arc_solved"]))
    sanity = read_object(Path(result["outputs"]["ball_flight_sanity"]))
    baseline_artifact = read_object(BASELINE_ARTIFACT_PATH)
    baseline_metrics = read_object(BASELINE_METRICS_PATH)
    coverage = coverage_metrics(artifact, track_frames=track_frames, spans=spans)
    provenance = provenance_audit(artifact)
    sanity_summary = dict(sanity.get("summary") or {})
    failed_sanity = int(sanity_summary.get("failed_segment_count") or 0)
    metrics = {
        "schema_version": 1,
        "lane": "ballarc_anchorfusion_20260716",
        "verified": False,
        "authority": "preview",
        "fit_is_not_accuracy": True,
        "preset": dict(preset),
        "selection": selection,
        "rally_active_spans": {
            "definition": registration["shared_rule"]["rally_active_span_rule"],
            "count": len(spans),
            "duration_s": sum(end - start for start, end in spans),
            "spans": [[start, end] for start, end in spans],
        },
        "baseline": {
            "metrics_path": str(BASELINE_METRICS_PATH),
            "segments_fit_count": int(baseline_metrics["artifact_audit"]["fit_segment_count"]),
            "segment_count": int(baseline_metrics["artifact_audit"]["segment_count"]),
            "segment_duration_s": distribution(segment_durations(baseline_artifact)),
        },
        "after": {
            **coverage,
            "segment_duration_s": distribution(segment_durations(artifact)),
            "wall_s": wall_s,
            "wall_min": wall_s / 60.0,
            "guard_budget_s": float(args.budget_s),
            "segment_budget_exceeded_count": int(
                dict(artifact.get("summary") or {}).get("segment_budget_exceeded_count") or 0
            ),
            "physics_sanity": sanity_summary,
            "physics_violation_count": failed_sanity,
            "provenance_audit": provenance,
        },
        "kill_rule": {
            "killed": failed_sanity > 0 or not provenance["pass"],
            "reasons": [
                *(["flight_sanity_violation"] if failed_sanity > 0 else []),
                *(["soft_split_provenance_failure"] if not provenance["pass"] else []),
            ],
        },
        "source_policy": {
            "pbvision_demo": "R&D reference only; not ground truth; not training; not redistributed",
            "audio_onsets": "review-only; not_gate_verified; trusted_for_contact=false",
            "soft_anchor_role": "segment split boundary only",
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    emit("run_end", wall_s=wall_s, result=result, metrics=str(args.metrics), kill_rule=metrics["kill_rule"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
