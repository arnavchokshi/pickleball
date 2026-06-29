#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_line_evidence import (  # noqa: E402
    aggregate_court_line_evidence,
    required_court_line_ids,
    required_court_net_ids,
)
from threed.racketsport.schemas import CourtLineObservation, NetLineObservation, validate_artifact_file  # noqa: E402


def build_court_line_evidence_from_review_inputs(
    review_input: Mapping[str, Any],
    *,
    clip: str,
    sport: str = "pickleball",
) -> Any:
    clips = review_input.get("clips")
    if not isinstance(clips, Mapping):
        raise ValueError("review input must contain a clips object")
    clip_payload = clips.get(clip)
    if not isinstance(clip_payload, Mapping):
        raise ValueError(f"review input does not contain clip {clip}")
    court = clip_payload.get("court_evidence")
    if not isinstance(court, Mapping):
        raise ValueError(f"review input clip {clip} does not contain court_evidence")

    points = court.get("points")
    if not isinstance(points, Mapping):
        points = {}

    required_line_ids = required_court_line_ids(sport)  # type: ignore[arg-type]
    required_net_ids = required_court_net_ids(sport)  # type: ignore[arg-type]
    line_observations: list[CourtLineObservation] = []
    for line_id in required_line_ids:
        observation = _line_observation_from_points(line_id, points)
        if observation is not None:
            line_observations.append(observation)

    net_observations: list[NetLineObservation] = []
    top_net = _top_net_observation_from_points(points)
    if top_net is not None:
        net_observations.append(top_net)

    return aggregate_court_line_evidence(
        sport=sport,  # type: ignore[arg-type]
        line_observations=line_observations,
        net_observations=net_observations,
        required_line_ids=required_line_ids,
        required_net_ids=required_net_ids,
        max_mean_residual_px=8.0,
        max_p95_residual_px=16.0,
    )


def _line_observation_from_points(line_id: str, points: Mapping[str, Any]) -> CourtLineObservation | None:
    first = _clicked_point(points.get(f"{line_id}:a"))
    second = _clicked_point(points.get(f"{line_id}:b"))
    if first is None or second is None:
        return None
    return CourtLineObservation(
        line_id=line_id,
        image_segment=[list(first), list(second)],
        confidence=1.0,
        frame_indexes=sorted({_frame_index_from_time(points.get(f"{line_id}:a")), _frame_index_from_time(points.get(f"{line_id}:b"))}),
        residual_px={"mean": 0.0, "p95": 0.0},
        visible_fraction=1.0,
        source="human_review_clicks",
    )


def _top_net_observation_from_points(points: Mapping[str, Any]) -> NetLineObservation | None:
    first = _clicked_point(points.get("top_net:a"))
    second = _clicked_point(points.get("top_net:b"))
    if first is None or second is None:
        return None
    left, right = sorted([first, second], key=lambda item: item[0])
    midpoint = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    return NetLineObservation(
        net_id="top_net",
        image_points=[list(left), list(midpoint), list(right)],
        confidence=1.0,
        frame_indexes=sorted({_frame_index_from_time(points.get("top_net:a")), _frame_index_from_time(points.get("top_net:b"))}),
        residual_px={"mean": 0.0, "p95": 0.0},
        source="human_review_clicks_midpoint",
    )


def _clicked_point(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, Mapping):
        return None
    if raw.get("status") != "clicked":
        return None
    x = raw.get("x")
    y = raw.get("y")
    if not isinstance(x, int | float) or not isinstance(y, int | float):
        return None
    return float(x), float(y)


def _frame_index_from_time(raw: Any, *, fps: float = 30.0) -> int:
    if not isinstance(raw, Mapping):
        return 0
    value = raw.get("time_s")
    if not isinstance(value, int | float):
        return 0
    return max(0, int(round(float(value) * fps)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote browser-clicked court evidence into court_line_evidence.json.")
    parser.add_argument("--review-input", type=Path, required=True, help="Saved review input JSON.")
    parser.add_argument("--clip", required=True, help="Clip id to promote.")
    parser.add_argument("--sport", default="pickleball", choices=("pickleball", "tennis"))
    parser.add_argument("--out", type=Path, required=True, help="Output court_line_evidence.json path.")
    args = parser.parse_args(argv)

    try:
        review_input = json.loads(args.review_input.read_text(encoding="utf-8"))
        if not isinstance(review_input, Mapping):
            raise ValueError("review input must be a JSON object")
        evidence = build_court_line_evidence_from_review_inputs(review_input, clip=args.clip, sport=args.sport)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_artifact_file("court_line_evidence", args.out)
    except Exception as exc:
        print(f"ERROR: court line review promotion failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "auto_calibration_ready": evidence.aggregate.auto_calibration_ready,
                "accepted_line_ids": evidence.aggregate.accepted_line_ids,
                "missing_required_line_ids": evidence.aggregate.missing_required_line_ids,
                "missing_required_net_ids": evidence.aggregate.missing_required_net_ids,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
