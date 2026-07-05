"""Regulation court proposals from normalized line-bank evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .court_proposals import CourtProposal
from .court_template_competition import score_template_competition


def propose_regulation_courts_from_line_bank(
    line_bank: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
) -> list[CourtProposal]:
    """Build conservative review-only court proposals from line-bank segments."""

    segments: list[dict[str, Any]] = []
    for item in line_bank.get("segments", []):
        segment = _normalize_segment(item)
        if segment is not None:
            segments.append(segment)
    horizontal = sorted(
        [segment for segment in segments if segment["orientation"] == "horizontal"],
        key=lambda item: (item["mid_y"], -item["length"]),
    )
    vertical = sorted(
        [segment for segment in segments if segment["orientation"] == "vertical"],
        key=lambda item: (item["mid_x"], -item["length"]),
    )
    if len(horizontal) < 4 or len(vertical) < 2:
        return []

    cross = _dedupe_parallel(horizontal, axis="y")[:]
    long = _dedupe_parallel(vertical, axis="x")[:]
    if len(cross) < 4 or len(long) < 2:
        return []
    cross = sorted(cross, key=lambda item: item["mid_y"])
    long = sorted(long, key=lambda item: item["mid_x"])
    top, second, third, bottom = cross[0], cross[1], cross[-2], cross[-1]
    left, right = long[0], long[-1]

    height_span = max(1.0, bottom["mid_y"] - top["mid_y"])
    semantic_lines = {
        "near_baseline": {"court_y_ft": 0.0, "support": bottom["length"]},
        "near_nvz": {"court_y_ft": (bottom["mid_y"] - third["mid_y"]) / height_span * 44.0, "support": third["length"]},
        "net": {"court_y_ft": 22.0, "support": min(second["length"], third["length"])},
        "far_nvz": {"court_y_ft": 44.0 - (second["mid_y"] - top["mid_y"]) / height_span * 44.0, "support": second["length"]},
        "far_baseline": {"court_y_ft": 44.0, "support": top["length"]},
    }
    competition = score_template_competition(semantic_lines)
    failed = ["not_verified"]
    if competition["winner"] != "pickleball":
        failed.append("tennis_template_wins")
    if float(competition["margin"]) < 0.2:
        failed.append("template_margin_too_small")
    if len(cross) < 5:
        failed.append("missing_required_pickleball_lines")

    court_keypoints = {
        "near_left_corner": (left["mid_x"], bottom["mid_y"]),
        "near_right_corner": (right["mid_x"], bottom["mid_y"]),
        "far_left_corner": (left["mid_x"], top["mid_y"]),
        "far_right_corner": (right["mid_x"], top["mid_y"]),
        "near_nvz_left": (left["mid_x"], third["mid_y"]),
        "near_nvz_right": (right["mid_x"], third["mid_y"]),
        "far_nvz_left": (left["mid_x"], second["mid_y"]),
        "far_nvz_right": (right["mid_x"], second["mid_y"]),
    }
    return [
        CourtProposal(
            proposal_id="proposal_regulation_0001",
            source="line_bank_regulation",
            court_keypoints=court_keypoints,
            scores={
                "overall": max(0.0, float(competition["pickleball"]["score"]) - float(competition["tennis"]["score"])),
                "pickleball_template": float(competition["pickleball"]["score"]),
                "tennis_template": float(competition["tennis"]["score"]),
                "template_margin": float(competition["margin"]),
                "line_support": min(1.0, (len(cross) + len(long)) / 8.0),
                "mask_support": None,
                "net_consistency": None,
                "temporal_jitter_px_p95": None,
                "reprojection_px_median": None,
                "reprojection_px_p95": None,
                "worst_corner_px": None,
            },
            gate={
                "auto_usable": False,
                "review_usable": True,
                "failed": failed,
                "warnings": [],
            },
            evidence={
                "template_competition": competition,
                "semantic_lines": semantic_lines,
                "line_segments": [
                    item["raw"]
                    for item in (top, second, third, bottom, left, right)
                ],
                "image_size": [int(image_size[0]), int(image_size[1])],
            },
        )
    ]


def _normalize_segment(item: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = item.get("xyxy")
    if raw is None and "p1" in item and "p2" in item:
        p1 = item["p1"]
        p2 = item["p2"]
        raw = [p1[0], p1[1], p2[0], p2[1]]
    if not isinstance(raw, Sequence) or len(raw) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in raw]
    length = float(item.get("length") or item.get("length_px") or math.hypot(x2 - x1, y2 - y1))
    if length <= 1e-6:
        return None
    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
    if angle <= 18.0 or angle >= 162.0:
        orientation = "horizontal"
    elif 72.0 <= angle <= 108.0:
        orientation = "vertical"
    else:
        return None
    return {
        "raw": {"xyxy": [x1, y1, x2, y2], "detector": item.get("detector", item.get("source", "unknown")), "length": length},
        "orientation": orientation,
        "length": length,
        "mid_x": (x1 + x2) / 2.0,
        "mid_y": (y1 + y2) / 2.0,
    }


def _dedupe_parallel(segments: list[dict[str, Any]], *, axis: str) -> list[dict[str, Any]]:
    key = "mid_y" if axis == "y" else "mid_x"
    ordered = sorted(segments, key=lambda item: (item[key], -item["length"]))
    out: list[dict[str, Any]] = []
    for segment in ordered:
        if out and abs(float(segment[key]) - float(out[-1][key])) < 5.0:
            if float(segment["length"]) > float(out[-1]["length"]):
                out[-1] = segment
            continue
        out.append(segment)
    return out
