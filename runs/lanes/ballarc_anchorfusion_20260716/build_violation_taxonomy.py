#!/usr/bin/env python3
"""Build the locked cross-preset flight-sanity failure taxonomy."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRESETS = ("conservative", "balanced", "broad")
NEAR_SPLIT_MAX_FRAMES = 5


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def soft_boundaries(artifact: dict) -> list[dict]:
    by_id: dict[str, dict] = {}
    for segment in artifact["segments"]:
        for boundary in segment.get("soft_split_provenance") or []:
            by_id[str(boundary["boundary_id"])] = boundary
    return sorted(by_id.values(), key=lambda item: (item["frame"], item["boundary_id"]))


def violating_frame_evidence(artifact: dict, gate: dict) -> tuple[list[int], list[int], list[str]]:
    reason_frames = sorted(int(frame) for frame in (gate.get("frame_reasons") or {}))
    solver_ids: set[int] = set()
    weak_frames: list[int] = []
    statuses: set[str] = set()
    for frame in reason_frames:
        arc = (artifact["frames"][frame].get("arc_solver") or {})
        if arc.get("segment_id") is not None:
            solver_ids.add(int(arc["segment_id"]))
        if arc.get("weak_segment"):
            weak_frames.append(frame)
        if arc.get("segment_status"):
            statuses.add(str(arc["segment_status"]))
    return sorted(solver_ids), weak_frames, sorted(statuses)


def motion_events(gate: dict) -> list[dict]:
    events: list[dict] = []
    reasons = set(gate["reasons"])
    horizontal = gate.get("horizontal") or {}
    speed = gate.get("speed_continuity") or {}
    if "horizontal_direction_reversal" in reasons:
        events.append(
            {
                "kind": "horizontal_direction_reversal",
                "frame": int(horizontal["max_heading_change_frame"]),
                "measured": float(horizontal["max_heading_change_deg"]),
                "threshold": float(horizontal["threshold_deg"]),
                "unit": "deg",
            }
        )
    if "speed_jump" in reasons:
        events.append(
            {
                "kind": "speed_jump",
                "frame": int(speed["max_speed_jump_frame"]),
                "measured": float(speed["max_speed_jump_mps"]),
                "threshold": float(speed["limit_mps"]),
                "unit": "mps",
            }
        )
    return events


def classify(gate: dict, interior: list[dict], weak_frames: list[int]) -> tuple[str, list[dict]]:
    events = motion_events(gate)
    nearest: list[dict] = []
    for event in events:
        if not interior:
            continue
        boundary = min(interior, key=lambda item: abs(int(item["frame"]) - event["frame"]))
        nearest.append(
            {
                **event,
                "nearest_soft_boundary_id": boundary["boundary_id"],
                "nearest_soft_boundary_frame": int(boundary["frame"]),
                "signed_delta_frames": event["frame"] - int(boundary["frame"]),
                "absolute_delta_frames": abs(event["frame"] - int(boundary["frame"])),
            }
        )
    if weak_frames:
        return "weak-fit-passed-through", nearest
    if nearest and min(item["absolute_delta_frames"] for item in nearest) <= NEAR_SPLIT_MAX_FRAMES:
        return "split-landed-mid-flight", nearest
    if any(
        reason in gate["reasons"]
        for reason in ("vertical_multi_apex", "horizontal_direction_reversal", "speed_jump")
    ):
        return "bridged-unmarked-direction-change", nearest
    return "anchor-semantics-structural", nearest


def build_preset(name: str) -> dict:
    output = ROOT / f"preset_{name}"
    artifact = read(output / "ball_track_arc_solved.json")
    sanity = read(output / "ball_flight_sanity.json")
    metrics = read(ROOT / f"preset_{name}_metrics.json")
    boundaries = soft_boundaries(artifact)
    segment_by_id = {int(item["segment_id"]): item for item in artifact["segments"]}
    failures: list[dict] = []
    for gate in sanity["segments"]:
        if gate["verdict"] != "fail":
            continue
        interior = [
            item
            for item in boundaries
            if int(gate["frame_start"]) < int(item["frame"]) < int(gate["frame_end"])
        ]
        solver_ids, weak_frames, statuses = violating_frame_evidence(artifact, gate)
        category, nearest = classify(gate, interior, weak_frames)
        solver_evidence = []
        for segment_id in solver_ids:
            segment = segment_by_id[segment_id]
            solver_evidence.append(
                {
                    "solver_segment_id": segment_id,
                    "status": segment["status"],
                    "start_anchor": segment["start_anchor"],
                    "end_anchor": segment["end_anchor"],
                    "soft_split_provenance_count": len(segment.get("soft_split_provenance") or []),
                }
            )
        court = gate.get("court_volume") or {}
        failures.append(
            {
                "flight_sanity_segment_id": int(gate["segment_id"]),
                "frame_start": int(gate["frame_start"]),
                "frame_end": int(gate["frame_end"]),
                "t_start": gate.get("t_start"),
                "t_end": gate.get("t_end"),
                "classification": category,
                "reasons": gate["reasons"],
                "interior_soft_boundary_count": len(interior),
                "interior_soft_boundary_ids": [item["boundary_id"] for item in interior],
                "nearest_boundary_to_failed_motion_check": nearest,
                "violating_frame_count": len(gate.get("frame_reasons") or {}),
                "outside_court_frame_count": int(court.get("outside_frame_count") or 0),
                "max_court_overage_m": court.get("max_overage_m"),
                "violating_solver_segment_ids": solver_ids,
                "violating_solver_statuses": statuses,
                "weak_violating_frames": weak_frames,
                "solver_segment_evidence": solver_evidence,
                "evidence_interpretation": (
                    "A failed motion check occurs within five frames of an untyped audio split; "
                    "the split landed inside a physical flight and independent fits disagree."
                    if category == "split-landed-mid-flight"
                    else "The failed direction/velocity topology is not marked by a nearby typed event anchor."
                    if category == "bridged-unmarked-direction-change"
                    else "A weak solver segment passed into the flight gate."
                    if category == "weak-fit-passed-through"
                    else "Only untyped soft endpoints/free-depth fits are available inside the hard "
                    "bounce-to-bounce gate span; the unchanged gate cannot treat audio splits as physics anchors."
                ),
            }
        )
    counts = Counter(item["classification"] for item in failures)
    return {
        "preset": name,
        "selection_rule_id": metrics["preset"]["selection_rule_id"],
        "selected_onset_count": metrics["selection"]["selected_count"],
        "segments_fit_count": metrics["after"]["segments_fit_count"],
        "in_rally_frame_coverage_percent": metrics["after"]["in_rally_frame_coverage_percent"],
        "physics_violation_count": metrics["after"]["physics_violation_count"],
        "killed": metrics["kill_rule"]["killed"],
        "classification_counts": {
            key: counts.get(key, 0)
            for key in (
                "split-landed-mid-flight",
                "bridged-unmarked-direction-change",
                "weak-fit-passed-through",
                "anchor-semantics-structural",
            )
        },
        "failures": failures,
    }


def main() -> None:
    presets = [build_preset(name) for name in PRESETS]
    total = Counter()
    for preset in presets:
        total.update(preset["classification_counts"])
    result = {
        "schema_version": 1,
        "lane": "ballarc_anchorfusion_20260716",
        "verified": False,
        "authority": "preview",
        "classification_rule": {
            "weak-fit-passed-through": "Any gate-failing frame is marked weak_segment=true.",
            "split-landed-mid-flight": (
                "A failed direction/speed check is within five frames of an interior audio soft split."
            ),
            "bridged-unmarked-direction-change": (
                "A direction/speed/vertical-topology check fails without a split within five frames."
            ),
            "anchor-semantics-structural": (
                "The remaining failure is outside-court geometry from non-weak free-depth/BVP fallback "
                "fits whose audio endpoints cannot legally act as physics anchors in the unchanged gate."
            ),
            "precedence": [
                "weak-fit-passed-through",
                "split-landed-mid-flight",
                "bridged-unmarked-direction-change",
                "anchor-semantics-structural",
            ],
        },
        "presets": presets,
        "aggregate_classification_counts": dict(total),
        "trend_conservative_to_balanced": {
            "selected_onset_delta": presets[1]["selected_onset_count"] - presets[0]["selected_onset_count"],
            "segments_fit_delta": presets[1]["segments_fit_count"] - presets[0]["segments_fit_count"],
            "in_rally_coverage_percentage_point_delta": (
                presets[1]["in_rally_frame_coverage_percent"]
                - presets[0]["in_rally_frame_coverage_percent"]
            ),
            "physics_violation_delta": (
                presets[1]["physics_violation_count"] - presets[0]["physics_violation_count"]
            ),
            "new_failed_flight_sanity_segment_ids": sorted(
                {item["flight_sanity_segment_id"] for item in presets[1]["failures"]}
                - {item["flight_sanity_segment_id"] for item in presets[0]["failures"]}
            ),
            "explanation": (
                "Balanced selected 34 additional untyped boundaries, so more short pools fit and frame "
                "coverage rose. The same additional free-depth fits exposed two previously unevaluated hard "
                "gate spans (0 and 15) as outside-court failures, raising violations from 16 to 18."
            ),
        },
        "taxonomy_verdict": "needs-typed-anchors",
        "taxonomy_verdict_reason": (
            "Threshold/spacing changes cannot give review-only audio onsets contact semantics. Across all "
            "presets, failures are dominated by anchor-semantics-structural free-depth/BVP fallback geometry; "
            "none are weak-fit pass-throughs. Denser presets recover more fits but do not reduce the unchanged "
            "gate's violation count to zero. Zero-violation recovery therefore requires Track G typed event "
            "anchors (or equivalent physics-bearing contact evidence), not another audio preset search."
        ),
    }
    (ROOT / "violation_taxonomy.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
