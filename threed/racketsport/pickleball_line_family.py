"""Evidence-only pickleball line-family diagnostics.

This module scores line color families and centerline topology for clips where
pickleball and tennis paint can overlap. It is diagnostic by construction:
reports stay unverified and must not promote CAL gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .court_calibration import project_image_points_to_world, project_planar_points


PICKLEBALL_HALF_WIDTH_M = 3.048
PICKLEBALL_HALF_LENGTH_M = 6.7056
PICKLEBALL_NVZ_Y_M = 2.1336


@dataclass(frozen=True)
class LineFamilyConfig:
    max_families: int = 4
    max_segments: int = 160
    min_segment_length_px: float = 18.0
    centerline_x_tolerance_m: float = 0.32
    boundary_position_tolerance_m: float = 0.42
    orientation_ratio: float = 1.25
    min_boundary_support_px: float = 120.0
    centerline_ready_support_fraction: float = 0.55
    kitchen_violation_fraction: float = 0.12
    endpoint_ready_residual_m: float = 0.45


EXPECTED_LINES: dict[str, dict[str, Any]] = {
    "near_baseline": {
        "axis": "x",
        "constant_axis": "y",
        "constant": -PICKLEBALL_HALF_LENGTH_M,
        "span": (-PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_HALF_WIDTH_M),
        "endpoints": ((-PICKLEBALL_HALF_WIDTH_M, -PICKLEBALL_HALF_LENGTH_M), (PICKLEBALL_HALF_WIDTH_M, -PICKLEBALL_HALF_LENGTH_M)),
        "keypoints": ("near_left_corner", "near_right_corner"),
    },
    "far_baseline": {
        "axis": "x",
        "constant_axis": "y",
        "constant": PICKLEBALL_HALF_LENGTH_M,
        "span": (-PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_HALF_WIDTH_M),
        "endpoints": ((-PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_HALF_LENGTH_M), (PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_HALF_LENGTH_M)),
        "keypoints": ("far_left_corner", "far_right_corner"),
    },
    "near_nvz": {
        "axis": "x",
        "constant_axis": "y",
        "constant": -PICKLEBALL_NVZ_Y_M,
        "span": (-PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_HALF_WIDTH_M),
        "endpoints": ((-PICKLEBALL_HALF_WIDTH_M, -PICKLEBALL_NVZ_Y_M), (PICKLEBALL_HALF_WIDTH_M, -PICKLEBALL_NVZ_Y_M)),
        "keypoints": ("near_nvz_left", "near_nvz_right"),
    },
    "far_nvz": {
        "axis": "x",
        "constant_axis": "y",
        "constant": PICKLEBALL_NVZ_Y_M,
        "span": (-PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_HALF_WIDTH_M),
        "endpoints": ((-PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_NVZ_Y_M), (PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_NVZ_Y_M)),
        "keypoints": ("far_nvz_left", "far_nvz_right"),
    },
    "left_sideline": {
        "axis": "y",
        "constant_axis": "x",
        "constant": -PICKLEBALL_HALF_WIDTH_M,
        "span": (-PICKLEBALL_HALF_LENGTH_M, PICKLEBALL_HALF_LENGTH_M),
        "endpoints": ((-PICKLEBALL_HALF_WIDTH_M, -PICKLEBALL_HALF_LENGTH_M), (-PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_HALF_LENGTH_M)),
        "keypoints": ("near_left_corner", "far_left_corner"),
    },
    "right_sideline": {
        "axis": "y",
        "constant_axis": "x",
        "constant": PICKLEBALL_HALF_WIDTH_M,
        "span": (-PICKLEBALL_HALF_LENGTH_M, PICKLEBALL_HALF_LENGTH_M),
        "endpoints": ((PICKLEBALL_HALF_WIDTH_M, -PICKLEBALL_HALF_LENGTH_M), (PICKLEBALL_HALF_WIDTH_M, PICKLEBALL_HALF_LENGTH_M)),
        "keypoints": ("near_right_corner", "far_right_corner"),
    },
    "near_centerline": {
        "axis": "y",
        "constant_axis": "x",
        "constant": 0.0,
        "span": (-PICKLEBALL_HALF_LENGTH_M, -PICKLEBALL_NVZ_Y_M),
        "endpoints": ((0.0, -PICKLEBALL_HALF_LENGTH_M), (0.0, -PICKLEBALL_NVZ_Y_M)),
        "keypoints": ("near_baseline_center", "near_nvz_center"),
    },
    "far_centerline": {
        "axis": "y",
        "constant_axis": "x",
        "constant": 0.0,
        "span": (PICKLEBALL_NVZ_Y_M, PICKLEBALL_HALF_LENGTH_M),
        "endpoints": ((0.0, PICKLEBALL_NVZ_Y_M), (0.0, PICKLEBALL_HALF_LENGTH_M)),
        "keypoints": ("far_nvz_center", "far_baseline_center"),
    },
}

CENTERLINE_ROLES = ("near_centerline", "far_centerline")
BOUNDARY_FAMILY_ROLES = ("near_baseline", "far_baseline", "near_nvz", "far_nvz", "left_sideline", "right_sideline")


def analyze_frame_line_family(
    image_bgr: Any,
    calibration: Mapping[str, Any],
    *,
    frame_id: str,
    keypoints: Mapping[str, Any] | None = None,
    config: LineFamilyConfig | None = None,
) -> dict[str, Any]:
    """Analyze one BGR frame and return a JSON-safe diagnostic payload."""

    cfg = config or LineFamilyConfig()
    homography = _homography(calibration)
    height, width = _image_size(image_bgr)
    raw_segments = _detect_segments(image_bgr, cfg)
    segments = _enrich_segments(raw_segments, image_bgr, homography, cfg)
    families = _cluster_color_families(segments, max_families=cfg.max_families)
    _assign_color_family(segments, families)
    dominant = _dominant_pickleball_family(segments, families, cfg)
    dominant_id = dominant.get("family_id")
    reviewed = _load_keypoint_map(keypoints)

    candidate_lines = _candidate_lines(segments, dominant_id=dominant_id, homography=homography, reviewed=reviewed, cfg=cfg)
    centerline_verdicts = {
        role: _centerline_verdict(
            role,
            segments,
            dominant_id=dominant_id,
            homography=homography,
            reviewed=reviewed,
            cfg=cfg,
        )
        for role in CENTERLINE_ROLES
    }
    selected_lines = _selected_lines(
        segments,
        centerline_verdicts=centerline_verdicts,
        dominant_id=dominant_id,
        homography=homography,
        reviewed=reviewed,
        cfg=cfg,
    )
    mixed_family_penalty = _mixed_family_penalty(segments, dominant_id=dominant_id, cfg=cfg)
    reasons = _readiness_reasons(dominant, centerline_verdicts)
    auto_ready = not reasons

    return {
        "artifact_type": "racketsport_pickleball_line_family_frame_diagnostic",
        "schema_version": 1,
        "frame_id": str(frame_id),
        "image_size_px": {"width": int(width), "height": int(height)},
        "verified": False,
        "not_gate_verified": True,
        "not_cal3_verified": True,
        "detected_segment_count": len(segments),
        "color_families": families,
        "dominant_pickleball_color_family": dominant,
        "segments": [_public_segment(segment) for segment in segments],
        "candidate_lines": candidate_lines,
        "selected_lines": selected_lines,
        "centerline_verdicts": centerline_verdicts,
        "mixed_family_penalty_for_current_calibration_assumed_lines": mixed_family_penalty,
        "auto_centerline_evidence_ready": auto_ready,
        "reasons": reasons,
    }


def run_pickleball_line_family_diagnostic(
    *,
    video: str | Path | None,
    frame: str | Path | None,
    calibration_path: str | Path,
    keypoints_path: str | Path | None,
    out_dir: str | Path,
    frame_count: int = 5,
    config: LineFamilyConfig | None = None,
) -> dict[str, Any]:
    """Run the diagnostic on a frame or sampled video and write JSON/MD/overlays."""

    if (video is None) == (frame is None):
        raise ValueError("provide exactly one of video or frame")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration = _read_json(calibration_path)
    keypoints = _read_json(keypoints_path) if keypoints_path is not None else None
    frames = _load_frames(video=video, frame=frame, count=frame_count)
    frame_results: list[dict[str, Any]] = []
    for frame_item in frames:
        result = analyze_frame_line_family(
            frame_item["image_bgr"],
            calibration,
            frame_id=frame_item["frame_id"],
            keypoints=keypoints,
            config=config,
        )
        overlay_path = out / f"overlay_{_safe_stem(frame_item['frame_id'])}.png"
        render_line_family_overlay(frame_item["image_bgr"], result, calibration, overlay_path)
        public = dict(result)
        public["overlay_png"] = overlay_path.name
        frame_results.append(public)

    aggregate = _aggregate_frame_results(
        frame_results,
        video=video,
        frame=frame,
        calibration_path=calibration_path,
        keypoints_path=keypoints_path,
    )
    json_path = out / "line_family_diagnostic.json"
    json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out / "line_family_diagnostic.md"
    md_path.write_text(_markdown_report(aggregate), encoding="utf-8")
    return aggregate


def render_line_family_overlay(
    image_bgr: Any,
    frame_result: Mapping[str, Any],
    calibration: Mapping[str, Any],
    out_path: str | Path,
) -> None:
    cv2 = _cv2()
    np = _np()
    overlay = np.array(image_bgr, copy=True)
    family_colors = [
        (0, 255, 255),
        (255, 0, 255),
        (255, 180, 0),
        (0, 180, 255),
        (180, 255, 0),
    ]
    for segment in frame_result.get("segments", []):
        p1 = _point2(segment.get("p1_px"))
        p2 = _point2(segment.get("p2_px"))
        if p1 is None or p2 is None:
            continue
        family_index = _family_index(str(segment.get("color_family_id", "family_0")))
        color = family_colors[family_index % len(family_colors)]
        role = str(segment.get("best_role") or "")
        thickness = 3 if "centerline" in role or role == "tennis_artifact" else 1
        cv2.line(overlay, _int_point(p1), _int_point(p2), color, thickness, cv2.LINE_AA)

    homography = _homography(calibration)
    for role, spec in EXPECTED_LINES.items():
        endpoints = project_planar_points(homography, spec["endpoints"])
        color = (80, 255, 80) if "centerline" not in role else (0, 80, 255)
        cv2.line(overlay, _int_point(endpoints[0]), _int_point(endpoints[1]), color, 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), overlay)


def _detect_segments(image_bgr: Any, cfg: LineFamilyConfig) -> list[dict[str, Any]]:
    cv2 = _cv2()
    np = _np()
    gray = cv2.cvtColor(_as_uint8_bgr(image_bgr), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
    raw_hough = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=28,
        minLineLength=max(16, int(round(width * 0.03))),
        maxLineGap=max(8, int(round(width * 0.015))),
    )
    segments: list[dict[str, Any]] = []
    if raw_hough is not None:
        for x1, y1, x2, y2 in raw_hough.reshape(-1, 4):
            segments.append(_segment_item(float(x1), float(y1), float(x2), float(y2), source="opencv_hough"))
    try:
        detector = cv2.createLineSegmentDetector()
        raw_lsd = detector.detect(gray)[0]
    except Exception:
        raw_lsd = None
    if raw_lsd is not None:
        for x1, y1, x2, y2 in raw_lsd.reshape(-1, 4):
            segments.append(_segment_item(float(x1), float(y1), float(x2), float(y2), source="opencv_lsd"))

    selected: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: float(item["length_px"]), reverse=True):
        if float(segment["length_px"]) < cfg.min_segment_length_px:
            continue
        if _is_duplicate_segment(segment, selected):
            continue
        selected.append(segment)
        if len(selected) >= cfg.max_segments:
            break
    return selected


def _enrich_segments(
    segments: Sequence[Mapping[str, Any]],
    image_bgr: Any,
    homography: Sequence[Sequence[float]],
    cfg: LineFamilyConfig,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        p1 = _point2(segment.get("p1"))
        p2 = _point2(segment.get("p2"))
        if p1 is None or p2 is None:
            continue
        try:
            world = project_image_points_to_world(homography, [p1, p2])
        except Exception:
            continue
        world_p1 = (float(world[0][0]), float(world[0][1]))
        world_p2 = (float(world[1][0]), float(world[1][1]))
        hsv_payload = _sample_segment_hsv(image_bgr, p1, p2)
        dx = world_p2[0] - world_p1[0]
        dy = world_p2[1] - world_p1[1]
        item = dict(segment)
        item.update(
            {
                "segment_id": f"seg_{index:03d}",
                "p1_world_m": [round(world_p1[0], 6), round(world_p1[1], 6)],
                "p2_world_m": [round(world_p2[0], 6), round(world_p2[1], 6)],
                "world_length_m": round(math.hypot(dx, dy), 6),
                "world_angle_deg": round(math.degrees(math.atan2(dy, dx)), 6),
                "hsv_median": hsv_payload["hsv_median"],
                "hsv_background_median": hsv_payload["hsv_background_median"],
                "paint_delta": hsv_payload["paint_delta"],
                "orientation": _segment_orientation(world_p1, world_p2, cfg),
            }
        )
        enriched.append(item)
    enriched.sort(
        key=lambda item: (
            str(item["source"]),
            -float(item["length_px"]),
            round(float(item["p1"][0]), 3),
            round(float(item["p1"][1]), 3),
            round(float(item["p2"][0]), 3),
            round(float(item["p2"][1]), 3),
        )
    )
    return enriched


def _cluster_color_families(segments: Sequence[Mapping[str, Any]], *, max_families: int) -> list[dict[str, Any]]:
    if not segments:
        return []
    clusters: list[dict[str, Any]] = []
    for segment in sorted(
        segments,
        key=lambda item: (
            _hsv_tuple(item)[0],
            _hsv_tuple(item)[1],
            _hsv_tuple(item)[2],
            -float(item.get("length_px") or 0.0),
        ),
    ):
        hsv = _hsv_tuple(segment)
        length = float(segment.get("length_px") or 0.0)
        if not clusters:
            clusters.append(_new_cluster(hsv, length))
            continue
        distances = [_hsv_distance(hsv, tuple(cluster["centroid_hsv"])) for cluster in clusters]
        best_index = min(range(len(distances)), key=lambda index: (distances[index], index))
        if distances[best_index] > 34.0 and len(clusters) < max_families:
            clusters.append(_new_cluster(hsv, length))
        else:
            cluster = clusters[best_index]
            cluster["samples"].append(hsv)
            cluster["support_length_px"] += length
            cluster["centroid_hsv"] = _median_hsv(cluster["samples"])

    clusters.sort(key=lambda item: (-float(item["support_length_px"]), tuple(item["centroid_hsv"])))
    public: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters):
        samples = cluster["samples"]
        h_values = [int(value[0]) for value in samples]
        s_values = [int(value[1]) for value in samples]
        v_values = [int(value[2]) for value in samples]
        public.append(
            {
                "family_id": f"family_{index}",
                "median_hsv": [int(value) for value in cluster["centroid_hsv"]],
                "hsv_range": {
                    "h": [min(h_values), max(h_values)],
                    "s": [min(s_values), max(s_values)],
                    "v": [min(v_values), max(v_values)],
                },
                "segment_count": len(samples),
                "support_length_px": round(float(cluster["support_length_px"]), 3),
            }
        )
    return public


def _assign_color_family(segments: list[dict[str, Any]], families: Sequence[Mapping[str, Any]]) -> None:
    if not families:
        for segment in segments:
            segment["color_family_id"] = None
            segment["color_family_distance"] = None
        return
    for segment in segments:
        hsv = _hsv_tuple(segment)
        distances = [
            _hsv_distance(hsv, tuple(int(value) for value in family["median_hsv"]))
            for family in families
        ]
        best_index = min(range(len(distances)), key=lambda index: (distances[index], index))
        segment["color_family_id"] = str(families[best_index]["family_id"])
        segment["color_family_distance"] = round(float(distances[best_index]), 3)


def _dominant_pickleball_family(
    segments: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
    cfg: LineFamilyConfig,
) -> dict[str, Any]:
    if not families:
        return {
            "family_id": None,
            "coherent": False,
            "median_hsv": None,
            "hsv_range": None,
            "boundary_support_length_px": 0.0,
            "reason": "no_detected_line_color_families",
        }
    support_by_family: dict[str, float] = {str(family["family_id"]): 0.0 for family in families}
    support_roles_by_family: dict[str, set[str]] = {str(family["family_id"]): set() for family in families}
    for segment in segments:
        family_id = segment.get("color_family_id")
        if not isinstance(family_id, str):
            continue
        role = _best_expected_role(segment, BOUNDARY_FAMILY_ROLES, cfg)
        if role is None:
            continue
        support_by_family[family_id] = support_by_family.get(family_id, 0.0) + float(segment.get("length_px") or 0.0)
        support_roles_by_family.setdefault(family_id, set()).add(role)
    dominant_id, dominant_support = max(
        support_by_family.items(),
        key=lambda item: (item[1], len(support_roles_by_family.get(item[0], set())), item[0]),
    )
    total_support = sum(support_by_family.values())
    family = next((item for item in families if item["family_id"] == dominant_id), None)
    coherent = dominant_support >= cfg.min_boundary_support_px and len(support_roles_by_family.get(dominant_id, set())) >= 2
    if total_support > 0.0 and dominant_support / total_support < 0.38:
        coherent = False
    reason = None
    if not coherent:
        reason = "insufficient_coherent_boundary_or_nvz_color_support"
    return {
        "family_id": dominant_id if coherent else (dominant_id if dominant_support > 0.0 else None),
        "coherent": bool(coherent),
        "median_hsv": family.get("median_hsv") if family is not None else None,
        "hsv_range": family.get("hsv_range") if family is not None else None,
        "boundary_support_length_px": round(float(dominant_support), 3),
        "boundary_support_roles": sorted(support_roles_by_family.get(dominant_id, set())),
        "support_fraction_among_boundary_candidates": round(dominant_support / total_support, 6) if total_support > 0 else 0.0,
        "reason": reason,
    }


def _candidate_lines(
    segments: Sequence[Mapping[str, Any]],
    *,
    dominant_id: Any,
    homography: Sequence[Sequence[float]],
    reviewed: Mapping[str, tuple[float, float]],
    cfg: LineFamilyConfig,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for segment in segments:
        candidate = _centerline_like_candidate(segment, dominant_id=dominant_id, homography=homography, reviewed=reviewed, cfg=cfg)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: (str(item["role"]), -float(item["score"]), str(item["segment_id"])))
    return candidates


def _centerline_like_candidate(
    segment: Mapping[str, Any],
    *,
    dominant_id: Any,
    homography: Sequence[Sequence[float]],
    reviewed: Mapping[str, tuple[float, float]],
    cfg: LineFamilyConfig,
) -> dict[str, Any] | None:
    if segment.get("orientation") != "lengthwise":
        return None
    p1 = _point2(segment.get("p1_world_m"))
    p2 = _point2(segment.get("p2_world_m"))
    if p1 is None or p2 is None:
        return None
    mean_x = (p1[0] + p2[0]) / 2.0
    if abs(mean_x) > cfg.centerline_x_tolerance_m:
        return None
    y0, y1 = sorted((p1[1], p2[1]))
    full_span = (-PICKLEBALL_HALF_LENGTH_M, PICKLEBALL_HALF_LENGTH_M)
    raw_support = _interval_overlap_fraction((y0, y1), full_span)
    kitchen_fraction = _interval_overlap_fraction((y0, y1), (-PICKLEBALL_NVZ_Y_M, PICKLEBALL_NVZ_Y_M))
    kitchen_violation = kitchen_fraction >= cfg.kitchen_violation_fraction
    overlaps_near = _interval_overlap_fraction((y0, y1), EXPECTED_LINES["near_centerline"]["span"]) > 0.12
    overlaps_far = _interval_overlap_fraction((y0, y1), EXPECTED_LINES["far_centerline"]["span"]) > 0.12
    if kitchen_violation:
        role = "tennis_artifact"
    elif overlaps_near and not overlaps_far:
        role = "near_centerline"
    elif overlaps_far and not overlaps_near:
        role = "far_centerline"
    elif overlaps_near and overlaps_far:
        role = "tennis_artifact"
    else:
        return None
    same_family = dominant_id is not None and segment.get("color_family_id") == dominant_id
    endpoint_residual_m = _candidate_endpoint_residual_m((y0, y1), role)
    score = _candidate_score(
        raw_support=raw_support,
        same_family=same_family,
        kitchen_violation=kitchen_violation,
        endpoint_residual_m=endpoint_residual_m,
        mean_abs_x=abs(mean_x),
        cfg=cfg,
    )
    payload = {
        "segment_id": segment["segment_id"],
        "role": role,
        "p1_px": segment["p1"],
        "p2_px": segment["p2"],
        "court_frame_endpoints_m": [segment["p1_world_m"], segment["p2_world_m"]],
        "color_family": segment.get("color_family_id"),
        "color_family_match": bool(same_family),
        "raw_support_fraction": round(float(raw_support), 6),
        "support": round(float(raw_support), 6),
        "kitchen_violation": bool(kitchen_violation),
        "kitchen_coverage_fraction": round(float(kitchen_fraction), 6),
        "residual_to_expected_m": None if endpoint_residual_m is None else round(float(endpoint_residual_m), 6),
        "score": round(float(score), 6),
    }
    _add_reviewed_line_residual(payload, role, reviewed)
    return payload


def _centerline_verdict(
    role: str,
    segments: Sequence[Mapping[str, Any]],
    *,
    dominant_id: Any,
    homography: Sequence[Sequence[float]],
    reviewed: Mapping[str, tuple[float, float]],
    cfg: LineFamilyConfig,
) -> dict[str, Any]:
    spec = EXPECTED_LINES[role]
    expected_span = tuple(float(value) for value in spec["span"])
    same_family_intervals: list[tuple[float, float]] = []
    same_family_kitchen_intervals: list[tuple[float, float]] = []
    all_matching_intervals: list[tuple[float, float]] = []
    endpoint_residuals: list[float] = []
    support_segment_ids: list[str] = []
    best_family: str | None = None
    family_support: dict[str, float] = {}
    for segment in segments:
        if segment.get("orientation") != "lengthwise":
            continue
        p1 = _point2(segment.get("p1_world_m"))
        p2 = _point2(segment.get("p2_world_m"))
        if p1 is None or p2 is None:
            continue
        mean_x = (p1[0] + p2[0]) / 2.0
        if abs(mean_x) > cfg.centerline_x_tolerance_m:
            continue
        interval = tuple(sorted((p1[1], p2[1])))
        overlap = _interval_overlap(interval, expected_span)
        if overlap > 0.0:
            family_id = segment.get("color_family_id")
            if isinstance(family_id, str):
                family_support[family_id] = family_support.get(family_id, 0.0) + overlap
            all_matching_intervals.append((max(interval[0], expected_span[0]), min(interval[1], expected_span[1])))
        if dominant_id is not None and segment.get("color_family_id") == dominant_id:
            kitchen_overlap = _interval_overlap(interval, (-PICKLEBALL_NVZ_Y_M, PICKLEBALL_NVZ_Y_M))
            if kitchen_overlap > 0.0:
                same_family_kitchen_intervals.append((max(interval[0], -PICKLEBALL_NVZ_Y_M), min(interval[1], PICKLEBALL_NVZ_Y_M)))
            if overlap > 0.0:
                clipped = (max(interval[0], expected_span[0]), min(interval[1], expected_span[1]))
                same_family_intervals.append(clipped)
                support_segment_ids.append(str(segment["segment_id"]))
                endpoint = expected_span[1] if role == "near_centerline" else expected_span[0]
                nearest = min((abs(interval[0] - endpoint), abs(interval[1] - endpoint)))
                endpoint_residuals.append(float(nearest))
    if family_support:
        best_family = max(family_support.items(), key=lambda item: (item[1], item[0]))[0]
    support_fraction = _union_length(same_family_intervals) / abs(expected_span[1] - expected_span[0])
    raw_support_fraction = _union_length(all_matching_intervals) / abs(expected_span[1] - expected_span[0])
    kitchen_fraction = _union_length(same_family_kitchen_intervals) / (PICKLEBALL_NVZ_Y_M * 2.0)
    kitchen_violation = kitchen_fraction >= cfg.kitchen_violation_fraction
    endpoint_residual_m = min(endpoint_residuals) if endpoint_residuals else None
    endpoint_residual_px = _endpoint_residual_px(role, endpoint_residual_m, homography)
    color_family_match = dominant_id is not None and support_fraction >= cfg.centerline_ready_support_fraction
    termination_score = 0.0 if endpoint_residual_m is None else max(0.0, 1.0 - endpoint_residual_m / cfg.endpoint_ready_residual_m)
    score = (
        0.42 * min(1.0, support_fraction)
        + 0.25 * (1.0 if color_family_match else 0.0)
        + 0.23 * termination_score
        + 0.10 * (0.0 if kitchen_violation else 1.0)
        - (0.65 if kitchen_violation else 0.0)
    )
    ready = (
        support_fraction >= cfg.centerline_ready_support_fraction
        and color_family_match
        and not kitchen_violation
        and endpoint_residual_m is not None
        and endpoint_residual_m <= cfg.endpoint_ready_residual_m
    )
    payload = {
        "role": role,
        "expected_court_frame_endpoints_m": _round_points(spec["endpoints"]),
        "support_fraction": round(float(support_fraction), 6),
        "raw_any_family_support_fraction": round(float(raw_support_fraction), 6),
        "kitchen_crossing_violation": bool(kitchen_violation),
        "same_color_kitchen_coverage_fraction": round(float(kitchen_fraction), 6),
        "color_family": dominant_id if color_family_match else best_family,
        "color_family_match": bool(color_family_match),
        "endpoint_at_nvz_residual_m": None if endpoint_residual_m is None else round(float(endpoint_residual_m), 6),
        "endpoint_at_nvz_residual_px": None if endpoint_residual_px is None else round(float(endpoint_residual_px), 3),
        "support_segment_ids": sorted(support_segment_ids),
        "score": round(float(score), 6),
        "ready": bool(ready),
    }
    if reviewed:
        expected_line = project_planar_points(homography, spec["endpoints"])
        keypoints = spec["keypoints"]
        if keypoints[0] in reviewed and keypoints[1] in reviewed:
            residuals = [
                _point_line_distance_px(reviewed[keypoints[0]], _point_tuple(expected_line[0]), _point_tuple(expected_line[1])),
                _point_line_distance_px(reviewed[keypoints[1]], _point_tuple(expected_line[0]), _point_tuple(expected_line[1])),
            ]
            payload["reviewed_keypoint_line_residual_px"] = round(sum(residuals) / len(residuals), 3)
            payload["reviewed_keypoint_names"] = list(keypoints)
    return payload


def _selected_lines(
    segments: Sequence[Mapping[str, Any]],
    *,
    centerline_verdicts: Mapping[str, Mapping[str, Any]],
    dominant_id: Any,
    homography: Sequence[Sequence[float]],
    reviewed: Mapping[str, tuple[float, float]],
    cfg: LineFamilyConfig,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for role in BOUNDARY_FAMILY_ROLES:
        best = _best_role_segment(role, segments, dominant_id=dominant_id, cfg=cfg)
        if best is None:
            continue
        payload = _selected_payload_from_segment(role, best, homography=homography, reviewed=reviewed)
        selected.append(payload)
    for role in CENTERLINE_ROLES:
        verdict = centerline_verdicts[role]
        if not verdict.get("support_segment_ids"):
            continue
        selected.append(
            {
                "role": role,
                "court_frame_endpoints": verdict["expected_court_frame_endpoints_m"],
                "color_family": verdict["color_family"],
                "support": verdict["support_fraction"],
                "kitchen_violation": verdict["kitchen_crossing_violation"],
                "residual_to_expected_m": verdict["endpoint_at_nvz_residual_m"],
                "endpoint_at_nvz_residual_px": verdict["endpoint_at_nvz_residual_px"],
                "score": verdict["score"],
                "reviewed_keypoint_line_residual_px": verdict.get("reviewed_keypoint_line_residual_px"),
            }
        )
    selected.sort(key=lambda item: str(item["role"]))
    return selected


def _best_role_segment(
    role: str,
    segments: Sequence[Mapping[str, Any]],
    *,
    dominant_id: Any,
    cfg: LineFamilyConfig,
) -> Mapping[str, Any] | None:
    candidates: list[tuple[float, Mapping[str, Any]]] = []
    for segment in segments:
        residual = _role_residual_m(segment, role, cfg)
        if residual is None:
            continue
        same = dominant_id is not None and segment.get("color_family_id") == dominant_id
        score = float(segment.get("world_length_m") or 0.0) - residual * 3.0 + (1.0 if same else 0.0)
        candidates.append((score, segment))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], str(item[1]["segment_id"])))
    return candidates[0][1]


def _selected_payload_from_segment(
    role: str,
    segment: Mapping[str, Any],
    *,
    homography: Sequence[Sequence[float]],
    reviewed: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    residual = _distance_to_role_constant(segment, role)
    support = _role_support_fraction(segment, role)
    payload = {
        "role": role,
        "court_frame_endpoints": [segment["p1_world_m"], segment["p2_world_m"]],
        "color_family": segment.get("color_family_id"),
        "support": round(float(support), 6),
        "kitchen_violation": False,
        "residual_to_expected_m": None if residual is None else round(float(residual), 6),
        "score": round(max(0.0, min(1.0, support)) + (0.2 if residual is not None and residual < 0.20 else 0.0), 6),
    }
    _add_reviewed_line_residual(payload, role, reviewed)
    return payload


def _mixed_family_penalty(
    segments: Sequence[Mapping[str, Any]],
    *,
    dominant_id: Any,
    cfg: LineFamilyConfig,
) -> dict[str, Any]:
    if dominant_id is None:
        return {
            "penalty": 1.0,
            "mixed_role_count": 0,
            "assessed_role_count": 0,
            "roles": [],
            "reason": "no_dominant_pickleball_family",
        }
    roles: list[dict[str, Any]] = []
    mixed = 0
    for role in EXPECTED_LINES:
        family_lengths: dict[str, float] = {}
        for segment in segments:
            residual = _role_residual_m(segment, role, cfg)
            if residual is None:
                continue
            family_id = segment.get("color_family_id")
            if not isinstance(family_id, str):
                continue
            family_lengths[family_id] = family_lengths.get(family_id, 0.0) + float(segment.get("length_px") or 0.0)
        if not family_lengths:
            continue
        best_family = max(family_lengths.items(), key=lambda item: (item[1], item[0]))[0]
        is_mixed = best_family != dominant_id
        mixed += int(is_mixed)
        roles.append({"role": role, "best_family": best_family, "mixed_family": is_mixed})
    assessed = len(roles)
    return {
        "penalty": round(mixed / assessed, 6) if assessed else 0.0,
        "mixed_role_count": mixed,
        "assessed_role_count": assessed,
        "roles": roles,
    }


def _aggregate_frame_results(
    frame_results: Sequence[Mapping[str, Any]],
    *,
    video: str | Path | None,
    frame: str | Path | None,
    calibration_path: str | Path,
    keypoints_path: str | Path | None,
) -> dict[str, Any]:
    if not frame_results:
        raise ValueError("at least one frame result is required")
    best = max(
        frame_results,
        key=lambda item: (
            bool(item["dominant_pickleball_color_family"].get("coherent")),
            sum(float(item["centerline_verdicts"][role]["score"]) for role in CENTERLINE_ROLES),
            -len(str(item["frame_id"])),
        ),
    )
    aggregate_verdicts: dict[str, Any] = {}
    for role in CENTERLINE_ROLES:
        role_items = [item["centerline_verdicts"][role] for item in frame_results]
        best_role = max(role_items, key=lambda item: (float(item["score"]), float(item["support_fraction"])))
        aggregate_verdicts[role] = dict(best_role)
        aggregate_verdicts[role]["mean_support_fraction_across_frames"] = round(
            sum(float(item["support_fraction"]) for item in role_items) / len(role_items), 6
        )
        aggregate_verdicts[role]["any_kitchen_crossing_violation_across_frames"] = any(
            bool(item["kitchen_crossing_violation"]) for item in role_items
        )
    reasons = _readiness_reasons(best["dominant_pickleball_color_family"], aggregate_verdicts)
    return {
        "artifact_type": "racketsport_pickleball_line_family_diagnostic",
        "schema_version": 1,
        "input": {
            "video": None if video is None else str(video),
            "frame": None if frame is None else str(frame),
            "calibration": str(calibration_path),
            "keypoints": None if keypoints_path is None else str(keypoints_path),
        },
        "sampled_frame_count": len(frame_results),
        "sampled_frame_ids": [str(item["frame_id"]) for item in frame_results],
        "verified": False,
        "not_gate_verified": True,
        "not_cal3_verified": True,
        "dominant_pickleball_color_family": best["dominant_pickleball_color_family"],
        "selected_lines": best["selected_lines"],
        "candidate_lines": best["candidate_lines"],
        "centerline_verdicts": aggregate_verdicts,
        "mixed_family_penalty_for_current_calibration_assumed_lines": best[
            "mixed_family_penalty_for_current_calibration_assumed_lines"
        ],
        "auto_centerline_evidence_ready": not reasons,
        "reasons": reasons,
        "frames": frame_results,
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    dominant = payload["dominant_pickleball_color_family"]
    lines = [
        "# Pickleball Line Family Diagnostic",
        "",
        f"- verified: `{str(payload['verified']).lower()}`",
        f"- not_gate_verified: `{str(payload['not_gate_verified']).lower()}`",
        f"- not_cal3_verified: `{str(payload['not_cal3_verified']).lower()}`",
        f"- auto_centerline_evidence_ready: `{str(payload['auto_centerline_evidence_ready']).lower()}`",
        f"- dominant family: `{dominant.get('family_id')}` median_hsv=`{dominant.get('median_hsv')}` coherent=`{dominant.get('coherent')}`",
        "",
        "| role | color family | support | kitchen violation | residual m | endpoint NVZ px | reviewed residual px | score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for line in payload.get("selected_lines", []):
        lines.append(
            "| {role} | {family} | {support} | {violation} | {residual} | {endpoint_px} | {reviewed_px} | {score} |".format(
                role=line.get("role"),
                family=line.get("color_family"),
                support=line.get("support"),
                violation=line.get("kitchen_violation"),
                residual=line.get("residual_to_expected_m"),
                endpoint_px=line.get("endpoint_at_nvz_residual_px"),
                reviewed_px=line.get("reviewed_keypoint_line_residual_px"),
                score=line.get("score"),
            )
        )
    lines.extend(["", "## Centerline Verdicts", ""])
    lines.append("| role | support | mean support | kitchen crossing | color match | endpoint residual m | endpoint residual px | ready | score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for role, verdict in payload.get("centerline_verdicts", {}).items():
        lines.append(
            "| {role} | {support} | {mean_support} | {kitchen} | {color} | {res_m} | {res_px} | {ready} | {score} |".format(
                role=role,
                support=verdict.get("support_fraction"),
                mean_support=verdict.get("mean_support_fraction_across_frames"),
                kitchen=verdict.get("kitchen_crossing_violation"),
                color=verdict.get("color_family_match"),
                res_m=verdict.get("endpoint_at_nvz_residual_m"),
                res_px=verdict.get("endpoint_at_nvz_residual_px"),
                ready=verdict.get("ready"),
                score=verdict.get("score"),
            )
        )
    if payload.get("reasons"):
        lines.extend(["", "## Reasons", ""])
        for reason in payload["reasons"]:
            lines.append(f"- `{reason}`")
    lines.append("")
    return "\n".join(lines)


def _load_frames(*, video: str | Path | None, frame: str | Path | None, count: int) -> list[dict[str, Any]]:
    cv2 = _cv2()
    if frame is not None:
        path = Path(frame)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read frame image: {path}")
        return [{"frame_id": path.stem, "frame_index": None, "image_bgr": image}]
    assert video is not None
    path = Path(video)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        positions = _linspace_int(0, max(0, total - 1), max(1, min(int(count), total)))
    else:
        positions = list(range(max(1, int(count))))
    frames: list[dict[str, Any]] = []
    for frame_index in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image = cap.read()
        if ok and image is not None:
            frames.append({"frame_id": f"frame_{int(frame_index):06d}", "frame_index": int(frame_index), "image_bgr": image})
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from video: {path}")
    return frames


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _homography(calibration: Mapping[str, Any]) -> list[list[float]]:
    value = calibration.get("homography")
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError("calibration must contain homography")
    rows = [[float(item) for item in row] for row in value]
    if any(len(row) != 3 for row in rows):
        raise ValueError("homography must be 3x3")
    return rows


def _image_size(image_bgr: Any) -> tuple[int, int]:
    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        raise ValueError("image_bgr must be a BGR image array")
    height, width = image_bgr.shape[:2]
    return int(height), int(width)


def _as_uint8_bgr(image_bgr: Any) -> Any:
    np = _np()
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 BGR")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _segment_item(x1: float, y1: float, x2: float, y2: float, *, source: str) -> dict[str, Any]:
    length = math.hypot(x2 - x1, y2 - y1)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return {
        "p1": [round(float(x1), 3), round(float(y1), 3)],
        "p2": [round(float(x2), 3), round(float(y2), 3)],
        "length_px": round(float(length), 3),
        "angle_deg": round(float(angle), 3),
        "source": source,
    }


def _is_duplicate_segment(segment: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> bool:
    p1 = _point2(segment.get("p1"))
    p2 = _point2(segment.get("p2"))
    if p1 is None or p2 is None:
        return True
    mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    angle = float(segment["angle_deg"])
    for existing in selected:
        q1 = _point2(existing.get("p1"))
        q2 = _point2(existing.get("p2"))
        if q1 is None or q2 is None:
            continue
        existing_mid = ((q1[0] + q2[0]) / 2.0, (q1[1] + q2[1]) / 2.0)
        if math.dist(mid, existing_mid) <= 5.5 and _angle_diff_deg(angle, float(existing["angle_deg"])) <= 3.0:
            return True
    return False


def _sample_segment_hsv(image_bgr: Any, p1: tuple[float, float], p2: tuple[float, float]) -> dict[str, Any]:
    cv2 = _cv2()
    np = _np()
    image = _as_uint8_bgr(image_bgr)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    length = max(1.0, math.dist(p1, p2))
    sample_count = max(9, min(80, int(round(length / 5.0))))
    dx = (p2[0] - p1[0]) / length
    dy = (p2[1] - p1[1]) / length
    normal = (-dy, dx)
    line_samples: list[Any] = []
    bg_samples: list[Any] = []
    for index in range(sample_count):
        t = 0.0 if sample_count == 1 else index / (sample_count - 1)
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        line_samples.append(hsv[_clamp_int(y, 0, height - 1), _clamp_int(x, 0, width - 1)])
        for offset in (-7.0, 7.0):
            bx = x + normal[0] * offset
            by = y + normal[1] * offset
            bg_samples.append(hsv[_clamp_int(by, 0, height - 1), _clamp_int(bx, 0, width - 1)])
    line_arr = np.asarray(line_samples, dtype=np.float32)
    bg_arr = np.asarray(bg_samples, dtype=np.float32)
    line_med = [int(round(float(value))) for value in np.median(line_arr, axis=0)]
    bg_med = [int(round(float(value))) for value in np.median(bg_arr, axis=0)]
    return {
        "hsv_median": line_med,
        "hsv_background_median": bg_med,
        "paint_delta": {
            "s": int(line_med[1] - bg_med[1]),
            "v": int(line_med[2] - bg_med[2]),
            "h_circular": int(_hue_diff(line_med[0], bg_med[0])),
        },
    }


def _segment_orientation(p1: tuple[float, float], p2: tuple[float, float], cfg: LineFamilyConfig) -> str:
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dy >= dx * cfg.orientation_ratio:
        return "lengthwise"
    if dx >= dy * cfg.orientation_ratio:
        return "crosswise"
    return "diagonal"


def _best_expected_role(segment: Mapping[str, Any], roles: Sequence[str], cfg: LineFamilyConfig) -> str | None:
    scored: list[tuple[float, str]] = []
    for role in roles:
        residual = _role_residual_m(segment, role, cfg)
        if residual is None:
            continue
        support = _role_support_fraction(segment, role)
        scored.append((residual - support * 0.08, role))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][1]


def _role_residual_m(segment: Mapping[str, Any], role: str, cfg: LineFamilyConfig) -> float | None:
    residual = _distance_to_role_constant(segment, role)
    if residual is None or residual > cfg.boundary_position_tolerance_m:
        return None
    spec = EXPECTED_LINES[role]
    expected_orientation = "lengthwise" if spec["axis"] == "y" else "crosswise"
    if segment.get("orientation") != expected_orientation:
        return None
    if _role_support_fraction(segment, role) <= 0.04:
        return None
    return residual


def _distance_to_role_constant(segment: Mapping[str, Any], role: str) -> float | None:
    spec = EXPECTED_LINES[role]
    p1 = _point2(segment.get("p1_world_m"))
    p2 = _point2(segment.get("p2_world_m"))
    if p1 is None or p2 is None:
        return None
    if spec["constant_axis"] == "x":
        return (abs(p1[0] - float(spec["constant"])) + abs(p2[0] - float(spec["constant"]))) / 2.0
    return (abs(p1[1] - float(spec["constant"])) + abs(p2[1] - float(spec["constant"]))) / 2.0


def _role_support_fraction(segment: Mapping[str, Any], role: str) -> float:
    spec = EXPECTED_LINES[role]
    p1 = _point2(segment.get("p1_world_m"))
    p2 = _point2(segment.get("p2_world_m"))
    if p1 is None or p2 is None:
        return 0.0
    if spec["axis"] == "y":
        interval = tuple(sorted((p1[1], p2[1])))
    else:
        interval = tuple(sorted((p1[0], p2[0])))
    return _interval_overlap_fraction(interval, tuple(float(value) for value in spec["span"]))


def _candidate_endpoint_residual_m(interval: tuple[float, float], role: str) -> float | None:
    if role == "near_centerline":
        return min(abs(interval[0] + PICKLEBALL_NVZ_Y_M), abs(interval[1] + PICKLEBALL_NVZ_Y_M))
    if role == "far_centerline":
        return min(abs(interval[0] - PICKLEBALL_NVZ_Y_M), abs(interval[1] - PICKLEBALL_NVZ_Y_M))
    return min(
        abs(interval[0] + PICKLEBALL_NVZ_Y_M),
        abs(interval[1] + PICKLEBALL_NVZ_Y_M),
        abs(interval[0] - PICKLEBALL_NVZ_Y_M),
        abs(interval[1] - PICKLEBALL_NVZ_Y_M),
    )


def _candidate_score(
    *,
    raw_support: float,
    same_family: bool,
    kitchen_violation: bool,
    endpoint_residual_m: float | None,
    mean_abs_x: float,
    cfg: LineFamilyConfig,
) -> float:
    termination = 0.0 if endpoint_residual_m is None else max(0.0, 1.0 - endpoint_residual_m / cfg.endpoint_ready_residual_m)
    axis = max(0.0, 1.0 - mean_abs_x / cfg.centerline_x_tolerance_m)
    score = 0.22 * min(1.0, raw_support) + 0.38 * (1.0 if same_family else 0.0) + 0.25 * termination + 0.15 * axis
    if kitchen_violation:
        score -= 0.75
    if not same_family:
        score -= 0.20
    return score


def _endpoint_residual_px(
    role: str,
    residual_m: float | None,
    homography: Sequence[Sequence[float]],
) -> float | None:
    if residual_m is None:
        return None
    expected_y = -PICKLEBALL_NVZ_Y_M if role == "near_centerline" else PICKLEBALL_NVZ_Y_M
    p0, p1 = project_planar_points(homography, [(0.0, expected_y), (0.0, expected_y + residual_m)])
    return math.dist(_point_tuple(p0), _point_tuple(p1))


def _add_reviewed_line_residual(payload: dict[str, Any], role: str, reviewed: Mapping[str, tuple[float, float]]) -> None:
    spec = EXPECTED_LINES.get(role)
    if spec is None or not reviewed:
        return
    keypoint_names = spec["keypoints"]
    p1 = _point2(payload.get("p1_px"))
    p2 = _point2(payload.get("p2_px"))
    if p1 is None or p2 is None:
        return
    values = [
        _point_line_distance_px(reviewed[name], p1, p2)
        for name in keypoint_names
        if name in reviewed
    ]
    if values:
        payload["reviewed_keypoint_line_residual_px"] = round(sum(values) / len(values), 3)
        payload["reviewed_keypoint_names"] = [name for name in keypoint_names if name in reviewed]


def _load_keypoint_map(keypoints: Mapping[str, Any] | None) -> dict[str, tuple[float, float]]:
    if not keypoints:
        return {}
    raw_items = keypoints.get("keypoints")
    if isinstance(raw_items, Mapping):
        iterable = [{"name": name, "uv": value} for name, value in raw_items.items()]
    elif isinstance(raw_items, Sequence):
        iterable = raw_items
    else:
        iterable = []
    parsed: dict[str, tuple[float, float]] = {}
    for item in iterable:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        uv = item.get("uv") or item.get("point") or item.get("image_px")
        point = _point2(uv)
        if isinstance(name, str) and point is not None:
            parsed[name] = point
    return parsed


def _readiness_reasons(
    dominant: Mapping[str, Any],
    verdicts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if not dominant.get("coherent"):
        reasons.append("no_coherent_pickleball_paint_family")
    for role in CENTERLINE_ROLES:
        verdict = verdicts[role]
        if float(verdict.get("support_fraction") or 0.0) < LineFamilyConfig().centerline_ready_support_fraction:
            reasons.append(f"{role}_low_support")
        if not verdict.get("color_family_match"):
            reasons.append(f"{role}_color_family_mismatch")
        if verdict.get("kitchen_crossing_violation"):
            reasons.append("same_color_centerline_crosses_kitchen")
        residual = verdict.get("endpoint_at_nvz_residual_m")
        if residual is None:
            reasons.append(f"{role}_missing_nvz_endpoint")
        elif float(residual) > LineFamilyConfig().endpoint_ready_residual_m:
            reasons.append(f"{role}_nvz_endpoint_residual_high")
    return sorted(set(reasons))


def _public_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    role = _public_best_role(segment)
    return {
        "segment_id": segment["segment_id"],
        "p1_px": segment["p1"],
        "p2_px": segment["p2"],
        "court_frame_endpoints_m": [segment["p1_world_m"], segment["p2_world_m"]],
        "length_px": segment["length_px"],
        "world_length_m": segment["world_length_m"],
        "source": segment["source"],
        "orientation": segment["orientation"],
        "hsv_median": segment["hsv_median"],
        "hsv_background_median": segment["hsv_background_median"],
        "paint_delta": segment["paint_delta"],
        "color_family_id": segment.get("color_family_id"),
        "color_family_distance": segment.get("color_family_distance"),
        "best_role": role,
    }


def _public_best_role(segment: Mapping[str, Any]) -> str | None:
    if segment.get("orientation") == "lengthwise":
        p1 = _point2(segment.get("p1_world_m"))
        p2 = _point2(segment.get("p2_world_m"))
        if p1 is not None and p2 is not None:
            mean_x = (p1[0] + p2[0]) / 2.0
            interval = tuple(sorted((p1[1], p2[1])))
            if abs(mean_x) <= LineFamilyConfig().centerline_x_tolerance_m:
                if _interval_overlap(interval, (-PICKLEBALL_NVZ_Y_M, PICKLEBALL_NVZ_Y_M)) > 0.0:
                    return "tennis_artifact"
                if _interval_overlap(interval, EXPECTED_LINES["near_centerline"]["span"]) > 0.0:
                    return "near_centerline"
                if _interval_overlap(interval, EXPECTED_LINES["far_centerline"]["span"]) > 0.0:
                    return "far_centerline"
    return _best_expected_role(segment, tuple(EXPECTED_LINES), LineFamilyConfig())


def _interval_overlap(interval: tuple[float, float], span: tuple[float, float]) -> float:
    lo = max(float(interval[0]), float(span[0]))
    hi = min(float(interval[1]), float(span[1]))
    return max(0.0, hi - lo)


def _interval_overlap_fraction(interval: tuple[float, float], span: tuple[float, float]) -> float:
    denom = abs(float(span[1]) - float(span[0]))
    if denom <= 1e-9:
        return 0.0
    return _interval_overlap(interval, span) / denom


def _union_length(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    normalized = sorted((min(a, b), max(a, b)) for a, b in intervals if max(a, b) > min(a, b))
    if not normalized:
        return 0.0
    total = 0.0
    cur_start, cur_end = normalized[0]
    for start, end in normalized[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def _hsv_tuple(segment: Mapping[str, Any]) -> tuple[int, int, int]:
    value = segment.get("hsv_median")
    if not isinstance(value, Sequence) or len(value) != 3:
        return (0, 0, 0)
    return (int(value[0]), int(value[1]), int(value[2]))


def _new_cluster(hsv: tuple[int, int, int], length: float) -> dict[str, Any]:
    return {"samples": [hsv], "centroid_hsv": list(hsv), "support_length_px": float(length)}


def _median_hsv(samples: Sequence[tuple[int, int, int]]) -> list[int]:
    np = _np()
    arr = np.asarray(samples, dtype=np.float32)
    return [int(round(float(value))) for value in np.median(arr, axis=0)]


def _hsv_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt((_hue_diff(a[0], b[0]) * 2.2) ** 2 + ((a[1] - b[1]) * 0.35) ** 2 + ((a[2] - b[2]) * 0.15) ** 2)


def _hue_diff(a: int | float, b: int | float) -> float:
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def _point_line_distance_px(point: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    denom = math.hypot(dx, dy)
    if denom <= 1e-9:
        return math.dist(point, p1)
    return abs(dy * point[0] - dx * point[1] + p2[0] * p1[1] - p2[1] * p1[0]) / denom


def _point2(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)


def _point_tuple(value: Sequence[float]) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _int_point(value: Sequence[float]) -> tuple[int, int]:
    return (int(round(float(value[0]))), int(round(float(value[1]))))


def _round_points(points: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[round(float(point[0]), 6), round(float(point[1]), 6)] for point in points]


def _clamp_int(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(float(value)))))


def _linspace_int(start: int, stop: int, count: int) -> list[int]:
    if count <= 1:
        return [int(start)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [int(round(float(start) + step * index)) for index in range(count)]


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))[:80]


def _family_index(family_id: str) -> int:
    try:
        return int(family_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _cv2() -> Any:
    import cv2  # type: ignore[import-not-found]

    return cv2


def _np() -> Any:
    import numpy as np

    return np
