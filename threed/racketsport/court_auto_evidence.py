"""Automatic court-line evidence extraction from video frames."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .court_calibration import calibration_image_size, project_planar_points
from .court_line_evidence import (
    Segment2,
    aggregate_court_line_evidence,
    score_line_candidate,
    select_best_line_observation,
)
from .court_templates import get_court_template
from .net_plane import build_net_plane, project_net_plane
from .schemas import CourtCalibration, CourtLineEvidence, CourtLineObservation, NetLineObservation, NetPlane


DEFAULT_REQUIRED_PICKLEBALL_LINES = ("near_nvz", "far_nvz", "near_centerline", "far_centerline")
DEFAULT_REQUIRED_NET_IDS = ("top_net",)
MAX_TRUSTED_TOP_NET_ANGLE_DELTA_DEG = 6.0
MAX_TRUSTED_TOP_NET_LENGTH_RATIO = 4.0
UNTRUSTED_TOP_NET_INTRINSIC_SOURCES = {"estimated_from_review_frame"}


def build_auto_court_line_evidence_from_frame(
    frame_path: str | Path,
    calibration: CourtCalibration,
    *,
    net_plane: NetPlane | None = None,
    frame_index: int = 0,
    cv2_module: Any | None = None,
) -> CourtLineEvidence:
    cv2 = cv2_module or _cv2()
    frame_path = Path(frame_path)
    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError(f"cannot open frame image: {frame_path}")
    return build_auto_court_line_evidence_from_image(
        image,
        calibration,
        net_plane=net_plane,
        frame_indexes=[frame_index],
        cv2_module=cv2,
    )


def build_auto_court_line_evidence_from_video(
    video_path: str | Path,
    calibration: CourtCalibration,
    *,
    net_plane: NetPlane | None = None,
    sample_count: int = 7,
    cv2_module: Any | None = None,
    required_line_ids: Sequence[str] = DEFAULT_REQUIRED_PICKLEBALL_LINES,
    required_net_ids: Sequence[str] = DEFAULT_REQUIRED_NET_IDS,
) -> CourtLineEvidence:
    """Sample a video and aggregate semantic court-line evidence across frames."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    cv2 = cv2_module or _cv2()
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_indexes = _sample_frame_indexes(frame_count, sample_count)
        frame_evidence: list[CourtLineEvidence] = []
        for frame_index in sample_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_evidence.append(
                build_auto_court_line_evidence_from_image(
                    frame,
                    calibration,
                    net_plane=net_plane,
                    frame_indexes=[frame_index],
                    cv2_module=cv2,
                    required_line_ids=required_line_ids,
                    required_net_ids=required_net_ids,
                )
            )
    finally:
        cap.release()

    if not frame_evidence:
        raise ValueError(f"no readable frames in video: {video_path}")

    lines = _merge_line_observations(
        observation for evidence in frame_evidence for observation in evidence.line_observations
    )
    nets = _merge_net_observations(
        observation for evidence in frame_evidence for observation in evidence.net_observations
    )
    evidence = aggregate_court_line_evidence(
        sport=calibration.sport,
        line_observations=lines,
        net_observations=nets,
        required_line_ids=required_line_ids,
        required_net_ids=required_net_ids,
    )
    evidence.source = "auto_hough_template_video"
    return evidence


def build_auto_court_line_evidence_from_image(
    image: Any,
    calibration: CourtCalibration,
    *,
    net_plane: NetPlane | None = None,
    frame_indexes: Sequence[int] = (0,),
    cv2_module: Any | None = None,
    required_line_ids: Sequence[str] = DEFAULT_REQUIRED_PICKLEBALL_LINES,
    required_net_ids: Sequence[str] = DEFAULT_REQUIRED_NET_IDS,
) -> CourtLineEvidence:
    """Detect line candidates and aggregate semantic court evidence."""

    cv2 = cv2_module or _cv2()
    height, width = image.shape[:2]
    calibration = calibration_for_image_size(calibration, width=int(width), height=int(height))
    candidates = detect_image_line_segments(image, cv2_module=cv2)
    expected_lines = projected_template_line_segments(calibration)
    observations = [
        select_best_line_observation(
            line_id=line_id,
            expected_segment=expected,
            candidate_segments=candidates,
            frame_indexes=frame_indexes,
            source="auto_hough_template",
            min_confidence=0.5,
        )
        for line_id, expected in expected_lines.items()
    ]
    net = net_plane or build_net_plane(calibration.sport)
    top_net_observation = select_top_net_observation(
        calibration,
        net,
        candidates,
        frame_indexes=frame_indexes,
        min_confidence=0.5,
    )
    evidence = aggregate_court_line_evidence(
        sport=calibration.sport,
        line_observations=observations,
        net_observations=[top_net_observation] if top_net_observation is not None else [],
        required_line_ids=required_line_ids,
        required_net_ids=required_net_ids,
    )
    evidence.source = "auto_hough_template"
    return evidence


def write_auto_court_line_evidence(
    out_path: str | Path,
    evidence: CourtLineEvidence,
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def calibration_for_image_size(calibration: CourtCalibration, *, width: int, height: int) -> CourtCalibration:
    """Scale calibration image coordinates when applying a calibration to a different frame resolution."""

    try:
        base_width, base_height = calibration_image_size(calibration, fallback_target=(float(width), float(height)))
    except ValueError:
        return calibration
    scale_x = float(width) / base_width
    scale_y = float(height) / base_height
    if math.isclose(scale_x, 1.0, rel_tol=1e-6) and math.isclose(scale_y, 1.0, rel_tol=1e-6):
        return calibration

    homography = [
        [float(value) * scale_x for value in calibration.homography[0]],
        [float(value) * scale_y for value in calibration.homography[1]],
        [float(value) for value in calibration.homography[2]],
    ]
    intrinsics = calibration.intrinsics.model_copy(
        update={
            "fx": float(calibration.intrinsics.fx) * scale_x,
            "fy": float(calibration.intrinsics.fy) * scale_y,
            "cx": float(calibration.intrinsics.cx) * scale_x,
            "cy": float(calibration.intrinsics.cy) * scale_y,
        }
    )
    reprojection_scale = (abs(scale_x) + abs(scale_y)) / 2.0
    reprojection_error = calibration.reprojection_error_px.model_copy(
        update={
            "median": float(calibration.reprojection_error_px.median) * reprojection_scale,
            "p95": float(calibration.reprojection_error_px.p95) * reprojection_scale,
        }
    )
    return calibration.model_copy(
        deep=True,
        update={
            "homography": homography,
            "intrinsics": intrinsics,
            "reprojection_error_px": reprojection_error,
            "image_pts": [
                [float(point[0]) * scale_x, float(point[1]) * scale_y] for point in calibration.image_pts
            ],
        },
    )


def projected_template_line_segments(calibration: CourtCalibration) -> dict[str, Segment2]:
    template = get_court_template(calibration.sport)
    projected: dict[str, Segment2] = {}
    for line_id, endpoints in template.line_segments_m.items():
        image_points = project_planar_points(calibration.homography, endpoints)
        projected[line_id] = _segment(image_points)
    return projected


def detect_image_line_segments(
    image: Any,
    *,
    cv2_module: Any | None = None,
    min_line_length_px: int = 25,
    max_line_gap_px: int = 12,
) -> list[Segment2]:
    """Return raw image-space line segments from edge/Hough detection."""

    cv2 = cv2_module or _cv2()
    if image is None:
        raise ValueError("image is required")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180.0,
        threshold=35,
        minLineLength=min_line_length_px,
        maxLineGap=max_line_gap_px,
    )
    if raw_lines is None:
        return []

    segments: list[Segment2] = []
    for raw in raw_lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in raw]
        if math.hypot(x2 - x1, y2 - y1) >= min_line_length_px:
            segments.append(((x1, y1), (x2, y2)))
    return segments


def _sample_frame_indexes(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 1:
        return [0]
    count = min(sample_count, frame_count)
    if count == 1:
        return [0]
    return sorted({round(index * (frame_count - 1) / (count - 1)) for index in range(count)})


def _merge_line_observations(observations: Iterable[CourtLineObservation]) -> list[CourtLineObservation]:
    by_line: dict[str, list[CourtLineObservation]] = {}
    for observation in observations:
        by_line.setdefault(observation.line_id, []).append(observation)
    return [_merge_one_line(line_id, group) for line_id, group in sorted(by_line.items())]


def _merge_one_line(line_id: str, observations: list[CourtLineObservation]) -> CourtLineObservation:
    count = len(observations)
    return CourtLineObservation(
        line_id=line_id,
        image_segment=[
            [
                sum(observation.image_segment[point_idx][axis_idx] for observation in observations) / count
                for axis_idx in range(2)
            ]
            for point_idx in range(2)
        ],
        confidence=sum(observation.confidence for observation in observations) / count,
        frame_indexes=sorted({index for observation in observations for index in observation.frame_indexes}),
        residual_px={
            "mean": sum(observation.residual_px.mean for observation in observations) / count,
            "p95": max(observation.residual_px.p95 for observation in observations),
        },
        visible_fraction=sum(observation.visible_fraction for observation in observations) / count,
        source="auto_hough_template_video",
    )


def _merge_net_observations(observations: Iterable[NetLineObservation]) -> list[NetLineObservation]:
    by_net: dict[str, list[NetLineObservation]] = {}
    for observation in observations:
        by_net.setdefault(observation.net_id, []).append(observation)
    return [_merge_one_net(net_id, group) for net_id, group in sorted(by_net.items())]


def _merge_one_net(net_id: str, observations: list[NetLineObservation]) -> NetLineObservation:
    count = len(observations)
    return NetLineObservation(
        net_id=net_id,
        image_points=[
            [
                sum(observation.image_points[point_idx][axis_idx] for observation in observations) / count
                for axis_idx in range(2)
            ]
            for point_idx in range(3)
        ],
        confidence=sum(observation.confidence for observation in observations) / count,
        frame_indexes=sorted({index for observation in observations for index in observation.frame_indexes}),
        residual_px={
            "mean": sum(observation.residual_px.mean for observation in observations) / count,
            "p95": max(observation.residual_px.p95 for observation in observations),
        },
        source="auto_hough_net_top_video",
    )


def select_top_net_observation(
    calibration: CourtCalibration,
    net_plane: NetPlane,
    candidate_segments: Sequence[Segment2],
    *,
    frame_indexes: Sequence[int] = (0,),
    min_confidence: float = 0.5,
    max_distance_px: float = 24.0,
    min_visible_fraction: float = 0.2,
) -> NetLineObservation | None:
    if calibration.intrinsics.source in UNTRUSTED_TOP_NET_INTRINSIC_SOURCES:
        return None

    net_points = project_net_plane(calibration, net_plane)
    expected_triplet = [net_points["left_post"], net_points["center"], net_points["right_post"]]
    expected_segment = _segment([expected_triplet[0], expected_triplet[2]])
    ground_net_segment = projected_template_line_segments(calibration).get("net")
    if ground_net_segment is not None:
        angle_delta = score_line_candidate(ground_net_segment, expected_segment).angle_delta_deg
        ground_length = _segment_length(ground_net_segment)
        top_length = _segment_length(expected_segment)
        length_ratio = (
            max(top_length / ground_length, ground_length / top_length)
            if ground_length > 0.0 and top_length > 0.0
            else float("inf")
        )
        if angle_delta > MAX_TRUSTED_TOP_NET_ANGLE_DELTA_DEG or length_ratio > MAX_TRUSTED_TOP_NET_LENGTH_RATIO:
            return None

    scored = [(score_line_candidate(expected_segment, segment), segment) for segment in candidate_segments]
    if not scored:
        return None
    score, segment = max(scored, key=lambda item: item[0].score)
    if (
        score.confidence < min_confidence
        or score.distance_px > max_distance_px
        or score.visible_fraction < min_visible_fraction
    ):
        return None

    observed_triplet = _candidate_triplet_for_expected_points(expected_triplet, expected_segment, segment)
    return NetLineObservation(
        net_id="top_net",
        image_points=[[float(x), float(y)] for x, y in observed_triplet],
        confidence=score.confidence,
        frame_indexes=[int(index) for index in frame_indexes],
        residual_px={"mean": score.distance_px, "p95": score.p95_distance_px},
        source="auto_hough_net_top",
    )


def _candidate_triplet_for_expected_points(
    expected_triplet: Sequence[Sequence[float]],
    expected_segment: Segment2,
    candidate_segment: Segment2,
) -> list[tuple[float, float]]:
    ex0, ex1 = expected_segment
    ca0, ca1 = candidate_segment
    expected_length = math.hypot(ex1[0] - ex0[0], ex1[1] - ex0[1])
    if expected_length <= 0.0:
        return [ca0, ((ca0[0] + ca1[0]) / 2.0, (ca0[1] + ca1[1]) / 2.0), ca1]
    ux = (ex1[0] - ex0[0]) / expected_length
    uy = (ex1[1] - ex0[1]) / expected_length
    points: list[tuple[float, float]] = []
    for point in expected_triplet:
        distance_along = ((float(point[0]) - ex0[0]) * ux + (float(point[1]) - ex0[1]) * uy) / expected_length
        points.append(
            (
                ca0[0] + (ca1[0] - ca0[0]) * distance_along,
                ca0[1] + (ca1[1] - ca0[1]) * distance_along,
            )
        )
    return points


def _segment(points: Sequence[Sequence[float]]) -> Segment2:
    if len(points) != 2:
        raise ValueError("segment requires exactly two points")
    return (
        (float(points[0][0]), float(points[0][1])),
        (float(points[1][0]), float(points[1][1])),
    )


def _segment_length(segment: Segment2) -> float:
    return math.hypot(segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("automatic court evidence extraction requires opencv-python") from exc
    return cv2
