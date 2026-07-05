"""Benchmark court-finding technologies against reviewed court labels.

This module is deliberately evaluation-oriented. It compares proposal-producing
technologies against reviewed keypoints, and records evidence-only technologies
without pretending they have solved calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from scripts.racketsport.compare_court_proposals_to_reviewed_keypoints import (
    _load_reviewed_points_native_px,
    compare_proposal_to_reviewed_keypoints,
)

from .court_line_keypoints import detect_court_keypoints_from_image
from .court_line_bank import normalize_hough_lines_p
from .overlapping_court_calibration import (
    detect_hsv_paint_hough_segments,
    pretrained_shadow_removal_preprocess,
    shadow_removal_preprocess,
)
from .net_anchor_court import load_player_suppressed_frame, solve_net_anchor_court_from_frame


BENCHMARK_ARTIFACT_TYPE = "racketsport_court_finding_technology_benchmark"
DEFAULT_TECHNOLOGIES = ("net_anchor", "hough_keypoints", "opencv_lsd")
LINE_CANDIDATE_ONLY_TECHNOLOGIES = {
    "opencv_lsd",
    "opencv_hough",
    "opencv_hough_shadow_normalized",
    "opencv_hough_pretrained_shadow_removed",
    "opencv_hough_lsd",
    "opencv_fast_line_detector",
    "skimage_probabilistic_hough",
    "opencv_hough_lsd_skimage",
    "elsed",
    "opencv_hsv_paint_hough",
    "opencv_hsv_paint_net_crop_hough",
    "opencv_hough_lsd_temporal",
    "opencv_hough_lsd_temporal_persistent",
}
FLOOR_LINE_KEYPOINT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("near_baseline", "near_left_corner", "near_right_corner"),
    ("far_baseline", "far_left_corner", "far_right_corner"),
    ("left_sideline", "near_left_corner", "far_left_corner"),
    ("right_sideline", "near_right_corner", "far_right_corner"),
    ("near_nvz", "near_nvz_left", "near_nvz_right"),
    ("far_nvz", "far_nvz_left", "far_nvz_right"),
    ("near_centerline", "near_baseline_center", "near_nvz_center"),
    ("far_centerline", "far_baseline_center", "far_nvz_center"),
)


@dataclass(frozen=True)
class CourtFindingSample:
    clip: str
    label_kind: str
    label_path: Path
    frame_input: Path


@dataclass(frozen=True)
class _CandidateLine:
    line: tuple[float, float, float]
    segment: dict[str, Any]
    support_length_px: float
    source_segment_count: int
    angle_deg: float


def resolve_sample_frame_input(sample_root: Path, candidate: Path) -> Path:
    """Resolve sample media while refusing missing label-frame directories."""

    if candidate.exists():
        if candidate.is_dir() and not any(candidate.glob("*.jpg")):
            return _fallback_sample_media(sample_root, candidate)
        return candidate
    if candidate.name in {"court_keypoint_frames", "court_keypoint_partial_frames"}:
        return _fallback_sample_media(sample_root, candidate)
    return candidate


def _fallback_sample_media(sample_root: Path, candidate: Path) -> Path:
    for name in ("source.mp4", "source.mov", "video.mp4", "frame.jpg", "source.jpg"):
        media = sample_root / name
        if media.exists():
            return media
    labels = sample_root / "labels"
    for name in ("court_keypoint_frames", "court_keypoint_partial_frames"):
        frame_dir = labels / name
        if frame_dir.exists() and any(frame_dir.glob("*.jpg")):
            return frame_dir
    raise ValueError(f"sample {sample_root.name} has labels but no readable media file for {candidate}")


def _review_label_has_usable_frames(label_path: Path) -> bool:
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    frames = payload.get("frames")
    if not isinstance(frames, Mapping):
        return False
    frame_count = frames.get("frame_count")
    coordinate_space = frames.get("label_coordinate_space")
    if not isinstance(coordinate_space, Sequence) or len(coordinate_space) != 2:
        return False
    try:
        return int(frame_count) > 0
    except (TypeError, ValueError):
        return False


_FLOOR_WORLD_XY: dict[str, tuple[float, float]] = {
    "near_left_corner": (-10.0, -22.0),
    "near_baseline_center": (0.0, -22.0),
    "near_right_corner": (10.0, -22.0),
    "far_right_corner": (10.0, 22.0),
    "far_baseline_center": (0.0, 22.0),
    "far_left_corner": (-10.0, 22.0),
    "near_nvz_left": (-10.0, -7.0),
    "near_nvz_center": (0.0, -7.0),
    "near_nvz_right": (10.0, -7.0),
    "net_left_sideline": (-10.0, 0.0),
    "net_center": (0.0, 0.0),
    "net_right_sideline": (10.0, 0.0),
    "far_nvz_left": (-10.0, 7.0),
    "far_nvz_center": (0.0, 7.0),
    "far_nvz_right": (10.0, 7.0),
}

_LINE_WORLD_COORDS: dict[str, tuple[float, float, str]] = {
    "near_baseline": (-22.0, 0.0, "y"),
    "near_nvz": (-7.0, 0.0, "y"),
    "net": (0.0, 0.0, "y"),
    "far_nvz": (7.0, 0.0, "y"),
    "far_baseline": (22.0, 0.0, "y"),
    "left_sideline": (-10.0, 0.0, "x"),
    "centerline": (0.0, 0.0, "x"),
    "right_sideline": (10.0, 0.0, "x"),
}

_TENNIS_CROSS_WORLD_Y: dict[str, float] = {
    "near_baseline": -39.0,
    "near_nvz": -21.0,
    "net": 0.0,
    "far_nvz": 21.0,
    "far_baseline": 39.0,
}

_CROSS_LINE_TEMPLATE_ORDER: tuple[str, ...] = (
    "far_baseline",
    "far_nvz",
    "net",
    "near_nvz",
    "near_baseline",
)

_FLOOR_LINE_ENDPOINT_KEYPOINTS: dict[str, tuple[str, str]] = {
    "near_baseline": ("near_left_corner", "near_right_corner"),
    "far_baseline": ("far_left_corner", "far_right_corner"),
    "near_nvz": ("near_nvz_left", "near_nvz_right"),
    "far_nvz": ("far_nvz_left", "far_nvz_right"),
    "left_sideline": ("near_left_corner", "far_left_corner"),
    "right_sideline": ("near_right_corner", "far_right_corner"),
    "centerline": ("near_baseline_center", "far_baseline_center"),
}

_IMAGE_EVIDENCE_LINE_ENDPOINT_KEYPOINTS: dict[str, tuple[str, str]] = {
    "near_baseline": ("near_left_corner", "near_right_corner"),
    "far_baseline": ("far_left_corner", "far_right_corner"),
    "near_nvz": ("near_nvz_left", "near_nvz_right"),
    "far_nvz": ("far_nvz_left", "far_nvz_right"),
    "left_sideline": ("near_left_corner", "far_left_corner"),
    "right_sideline": ("near_right_corner", "far_right_corner"),
    "near_centerline": ("near_baseline_center", "near_nvz_center"),
    "far_centerline": ("far_baseline_center", "far_nvz_center"),
}


def discover_court_finding_samples(eval_root: str | Path) -> list[CourtFindingSample]:
    """Return the four full-label court clips plus owner IMG_1605 partial labels."""

    root = Path(eval_root)
    samples: list[CourtFindingSample] = []
    for clip_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        labels = clip_dir / "labels"
        full_label = labels / "court_keypoints.json"
        partial_label = labels / "court_keypoints_partial.json"
        if full_label.exists() and _review_label_has_usable_frames(full_label):
            samples.append(
                CourtFindingSample(
                    clip=clip_dir.name,
                    label_kind="full_15pt",
                    label_path=full_label,
                    frame_input=resolve_sample_frame_input(clip_dir, labels / "court_keypoint_frames"),
                )
            )
        elif partial_label.exists():
            samples.append(
                CourtFindingSample(
                    clip=clip_dir.name,
                    label_kind="partial_visible",
                    label_path=partial_label,
                    frame_input=resolve_sample_frame_input(clip_dir, labels / "court_keypoint_partial_frames"),
                )
            )
    return samples


def _load_temporal_frames(input_path: str | Path, *, max_frames: int) -> list[tuple[str, Any]]:
    import cv2  # type: ignore[import-not-found]

    path = Path(input_path)
    frames: list[tuple[str, Any]] = []
    if path.is_dir():
        frame_paths = sorted(path.glob("*.jpg"))[: max(1, int(max_frames))]
        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is not None:
                frames.append((frame_path.name, image))
        if not frames:
            raise ValueError(f"no readable .jpg frames in {path}")
        return frames

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return [(path.name, image)]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open input as image/video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        positions = [
            int(round(value))
            for value in _linspace_int(0, max(0, total - 1), max(1, min(int(max_frames), total)))
        ]
    else:
        positions = list(range(max(1, int(max_frames))))
    for frame_index in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((f"frame_{frame_index:06d}", frame))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from video: {path}")
    return frames


def _linspace_int(start: int, stop: int, count: int) -> list[int]:
    if count <= 1:
        return [int(start)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [int(round(float(start) + step * index)) for index in range(count)]


def detect_line_candidates_for_technology(image_bgr: Any, technology_id: str) -> dict[str, Any]:
    """Run a line-candidate adapter and return a JSON-safe evidence payload."""

    if technology_id == "opencv_lsd":
        segments = _opencv_lsd_segments(image_bgr)
    elif technology_id == "opencv_hough":
        segments = _opencv_hough_segments(image_bgr)
    elif technology_id == "opencv_hough_shadow_normalized":
        preprocessed, shadow_evidence = shadow_removal_preprocess(image_bgr)
        segments = _retag_segments_source(_opencv_hough_segments(preprocessed), technology_id)
        return {
            "technology_id": technology_id,
            "available": True,
            "candidate_count": len(segments),
            "segments": segments,
            "shadow_preprocess": shadow_evidence,
        }
    elif technology_id == "opencv_hough_pretrained_shadow_removed":
        preprocessed, shadow_evidence = pretrained_shadow_removal_preprocess(image_bgr)
        if not shadow_evidence.get("available"):
            return {
                "technology_id": technology_id,
                "available": False,
                "candidate_count": 0,
                "segments": [],
                "shadow_preprocess": shadow_evidence,
                "reason": shadow_evidence.get("reason", "pretrained_shadow_removal_unavailable"),
            }
        segments = _retag_segments_source(_opencv_hough_segments(preprocessed), technology_id)
        return {
            "technology_id": technology_id,
            "available": True,
            "candidate_count": len(segments),
            "segments": segments,
            "shadow_preprocess": shadow_evidence,
        }
    elif technology_id == "opencv_hough_lsd":
        segments = _dedupe_segments(_opencv_hough_segments(image_bgr) + _opencv_lsd_segments(image_bgr))
    elif technology_id == "opencv_fast_line_detector":
        try:
            segments = _opencv_fast_line_detector_segments(image_bgr)
        except AttributeError as exc:
            return {
                "technology_id": technology_id,
                "available": False,
                "candidate_count": 0,
                "segments": [],
                "reason": "opencv_ximgproc_fast_line_detector_unavailable",
                "error": str(exc),
            }
    elif technology_id == "skimage_probabilistic_hough":
        try:
            segments = _skimage_probabilistic_hough_segments(image_bgr)
        except ImportError as exc:
            return {
                "technology_id": technology_id,
                "available": False,
                "candidate_count": 0,
                "segments": [],
                "reason": "skimage_unavailable",
                "error": str(exc),
            }
    elif technology_id == "opencv_hough_lsd_skimage":
        hough_segments = _opencv_hough_segments(image_bgr)
        lsd_segments = _opencv_lsd_segments(image_bgr)
        skimage_evidence = detect_line_candidates_for_technology(image_bgr, "skimage_probabilistic_hough")
        skimage_segments = skimage_evidence["segments"] if skimage_evidence.get("available") else []
        segments = _dedupe_segments(hough_segments + lsd_segments + list(skimage_segments))
        return {
            "technology_id": technology_id,
            "base_technology_ids": ["opencv_hough", "opencv_lsd", "skimage_probabilistic_hough"],
            "available": True,
            "candidate_count": len(segments),
            "segments": segments,
            "component_candidate_counts": {
                "opencv_hough": len(hough_segments),
                "opencv_lsd": len(lsd_segments),
                "skimage_probabilistic_hough": int(skimage_evidence.get("candidate_count") or 0),
            },
            "component_availability": {
                "skimage_probabilistic_hough": bool(skimage_evidence.get("available")),
            },
        }
    elif technology_id == "elsed":
        return _elsed_line_candidate_evidence(image_bgr)
    elif technology_id == "opencv_hsv_paint_hough":
        return detect_hsv_paint_hough_segments(image_bgr, technology_id=technology_id)
    elif technology_id == "opencv_hsv_paint_net_crop_hough":
        from .court_detector_v2_net import detect_court_net_evidence

        net_evidence = detect_court_net_evidence(image_bgr)
        return detect_hsv_paint_hough_segments(
            image_bgr,
            net_evidence=net_evidence,
            use_near_side_crop=True,
            technology_id=technology_id,
        )
    else:
        return {
            "technology_id": technology_id,
            "available": False,
            "candidate_count": 0,
            "segments": [],
            "reason": "unsupported_line_candidate_technology",
        }

    return {
        "technology_id": technology_id,
        "available": True,
        "candidate_count": len(segments),
        "segments": segments,
    }


def detect_temporal_line_candidates_for_input(
    input_path: str | Path,
    *,
    technology_id: str = "opencv_hough_lsd_temporal",
    max_frames: int = 12,
) -> dict[str, Any]:
    """Union Hough+LSD line candidates across sampled frames for fixed cameras."""

    if technology_id not in {"opencv_hough_lsd_temporal", "opencv_hough_lsd_temporal_persistent"}:
        return {
            "technology_id": technology_id,
            "available": False,
            "candidate_count": 0,
            "segments": [],
            "reason": "unsupported_temporal_line_candidate_technology",
        }
    frames = _load_temporal_frames(input_path, max_frames=max_frames)
    segments: list[dict[str, Any]] = []
    frame_names: list[str] = []
    image_size: tuple[int, int] | None = None
    for frame_index, (frame_name, frame) in enumerate(frames):
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            continue
        image_size = (int(frame.shape[1]), int(frame.shape[0]))
        frame_names.append(frame_name)
        evidence = detect_line_candidates_for_technology(frame, "opencv_hough_lsd")
        for segment in evidence["segments"]:
            item = dict(segment)
            item["source"] = f"opencv_hough_lsd_temporal:{segment.get('source', 'unknown')}"
            item["temporal_frame_index"] = int(frame_index)
            item["temporal_frame_name"] = frame_name
            segments.append(item)
    if technology_id == "opencv_hough_lsd_temporal_persistent":
        segments, persistence_min_frame_count = _persistent_temporal_segments(
            segments,
            temporal_frame_count=len(frame_names),
            image_size=image_size,
        )
    else:
        segments = _dedupe_segments(segments)
        persistence_min_frame_count = None
    payload: dict[str, Any] = {
        "technology_id": technology_id,
        "base_technology_id": "opencv_hough_lsd",
        "available": bool(frames),
        "temporal_frame_count": len(frame_names),
        "frame_names": frame_names,
        "candidate_count": len(segments),
        "segments": segments,
    }
    if persistence_min_frame_count is not None:
        payload["persistence_min_frame_count"] = int(persistence_min_frame_count)
    if image_size is not None:
        payload["image_size"] = [image_size[0], image_size[1]]
    return payload


def _persistent_temporal_segments(
    segments: Sequence[dict[str, Any]],
    *,
    temporal_frame_count: int,
    image_size: tuple[int, int] | None,
) -> tuple[list[dict[str, Any]], int]:
    if temporal_frame_count <= 1:
        selected = _dedupe_segments(segments)
        for segment in selected:
            segment["temporal_support_frame_count"] = 1
            segment["temporal_persistence_ratio"] = 1.0
            segment["temporal_source_segment_count"] = 1
        return selected, 1

    width, height = image_size if image_size is not None else (1280, 720)
    grouping_distance_px = max(12.0, min(float(width), float(height)) * 0.022)
    groups: list[list[dict[str, Any]]] = []
    for segment in sorted(segments, key=lambda item: float(item.get("length_px") or 0.0), reverse=True):
        try:
            line = _line_from_segment_mapping(segment)
        except ValueError:
            continue
        assigned = False
        for group in groups:
            reference = group[0]
            reference_line = _line_from_segment_mapping(reference)
            if (
                _angle_diff_mod_180(float(segment["angle_deg"]), float(reference["angle_deg"])) <= 4.5
                and _line_segment_distance(line, reference) <= grouping_distance_px
            ):
                group.append(dict(segment))
                assigned = True
                break
        if not assigned:
            groups.append([dict(segment)])

    min_frame_count = 2
    persistent: list[dict[str, Any]] = []
    for group in groups:
        frame_indexes = {
            int(item["temporal_frame_index"])
            for item in group
            if item.get("temporal_frame_index") is not None
        }
        if len(frame_indexes) < min_frame_count:
            continue
        representative = dict(max(group, key=lambda item: float(item.get("length_px") or 0.0)))
        support_length = sum(float(item.get("length_px") or 0.0) for item in group)
        representative["source"] = "opencv_hough_lsd_temporal_persistent"
        representative["temporal_support_frame_count"] = len(frame_indexes)
        representative["temporal_persistence_ratio"] = round(len(frame_indexes) / float(temporal_frame_count), 4)
        representative["temporal_source_segment_count"] = len(group)
        representative["temporal_support_length_px"] = round(float(support_length), 3)
        representative["temporal_support_frame_indexes"] = sorted(frame_indexes)
        representative["temporal_support_frame_names"] = sorted(
            {
                str(item["temporal_frame_name"])
                for item in group
                if item.get("temporal_frame_name") is not None
            }
        )
        persistent.append(representative)

    persistent.sort(
        key=lambda item: (
            int(item.get("temporal_support_frame_count") or 0),
            float(item.get("temporal_support_length_px") or 0.0),
            float(item.get("length_px") or 0.0),
        ),
        reverse=True,
    )
    return persistent[:128], min_frame_count


def solve_regulation_court_from_line_candidates(
    image_bgr: Any,
    *,
    technology_id: str = "opencv_hough_lsd_regulation",
    clip_id: str = "",
    line_segments: Sequence[Mapping[str, Any]] | None = None,
    line_evidence: Mapping[str, Any] | None = None,
    image_evidence_mode: str = "sparse_pixel",
    line_refinement: bool = False,
) -> dict[str, Any]:
    """Search Hough/LSD line assignments against the regulation pickleball template.

    This is still an experimental proposal generator. It never promotes
    calibration, and its confidence is capped until the real verification gate is
    much stronger than this benchmark scorer.
    """

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        raise ValueError("image_bgr must be an image array")
    height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    if line_segments is None:
        evidence = detect_line_candidates_for_technology(image_bgr, "opencv_hough_lsd")
        segments = list(evidence["segments"])
    else:
        evidence = dict(line_evidence or {})
        segments = [dict(segment) for segment in line_segments]
    groups = _candidate_line_groups(segments, width=float(width), height=float(height))
    hypotheses = _regulation_hypotheses_from_groups(
        groups,
        segments=segments,
        image_bgr=image_bgr,
        image_size=(width, height),
        image_evidence_mode=image_evidence_mode,
        line_refinement=line_refinement,
    )
    if not hypotheses:
        raise ValueError("no regulation line assignment hypothesis could be built")
    best = hypotheses[0]
    keypoints = {
        name: {
            "xy": [round(float(xy[0]), 3), round(float(xy[1]), 3)],
            "confidence": round(min(0.62, max(0.05, 0.12 + 0.08 * best["supported_line_count"])), 4),
            "source": technology_id,
        }
        for name, xy in best["keypoints"].items()
    }
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_finding_technology_proposal",
        "source": {
            "clip_id": clip_id,
            "image_size": [width, height],
        },
        "solver": {
            "name": technology_id,
            "version": 1,
            "writes_court_calibration": False,
        },
        "keypoints": keypoints,
        "solver_confidence": round(min(0.62, max(0.05, 0.12 + 0.08 * best["supported_line_count"])), 4),
        "needs_user_input": ["court_keypoints"],
        "needs_user_confirmation": True,
        "line_assignment": best["line_assignment"],
        "line_evidence": _line_evidence_summary(evidence=evidence, segments=segments),
        "score_components": best["score_components"],
        "hypotheses": [
            {
                "rank": index + 1,
                "score": round(float(item["score"]), 4),
                "supported_line_count": int(item["supported_line_count"]),
                "line_assignment": item["line_assignment"],
                "score_components": item["score_components"],
            }
            for index, item in enumerate(hypotheses[:8])
        ],
        "notes": [
            "benchmark_proposal_only_never_writes_court_calibration_json",
            "top_net_points_projected_from_floor_net_line_for_comparison_only",
            "confidence_capped_until_reviewed_residual_gates_pass",
        ],
    }


def _line_evidence_summary(*, evidence: Mapping[str, Any], segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {
        "technology_id": str(evidence.get("technology_id") or "opencv_hough_lsd"),
        "candidate_count": int(evidence.get("candidate_count") or len(segments)),
    }
    if evidence.get("base_technology_id") is not None:
        summary["base_technology_id"] = str(evidence["base_technology_id"])
    if evidence.get("temporal_frame_count") is not None:
        summary["temporal_frame_count"] = int(evidence["temporal_frame_count"])
    if evidence.get("persistence_min_frame_count") is not None:
        summary["persistence_min_frame_count"] = int(evidence["persistence_min_frame_count"])
    if evidence.get("image_size") is not None:
        summary["image_size"] = list(evidence["image_size"])
    return summary


def score_template_competition_for_line_assignment(
    line_assignment: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Compare cross-line spacing against pickleball and tennis service templates."""

    width, _height = image_size
    x_ref = float(width) * 0.5
    entries: list[tuple[str, float]] = []
    for label in _CROSS_LINE_TEMPLATE_ORDER:
        raw = line_assignment.get(label)
        if raw is None:
            continue
        try:
            if isinstance(raw, _CandidateLine):
                line = raw.line
                fallback = _segment_midpoint(raw.segment)[1]
            elif isinstance(raw, Mapping):
                segment = _segment_from_assignment_mapping(raw)
                line = _line_from_segment_mapping(segment)
                fallback = _segment_midpoint(segment)[1]
            else:
                continue
            entries.append((label, _line_y_at_x(line, x_ref, fallback=fallback)))
        except (KeyError, TypeError, ValueError):
            continue
    if len(entries) < 3:
        return {
            "available": False,
            "reason": "fewer_than_three_cross_lines",
            "tennis_template_penalty": 0.0,
        }

    observed_gaps = [entries[index + 1][1] - entries[index][1] for index in range(len(entries) - 1)]
    ordering_violation_count = sum(1 for gap in observed_gaps if gap <= 1.0)
    positive_gaps = [max(1.0, float(gap)) for gap in observed_gaps]
    labels = [label for label, _y in entries]
    pickleball_expected = [
        abs(float(_LINE_WORLD_COORDS[labels[index + 1]][0]) - float(_LINE_WORLD_COORDS[labels[index]][0]))
        for index in range(len(labels) - 1)
    ]
    tennis_expected = [
        abs(float(_TENNIS_CROSS_WORLD_Y[labels[index + 1]]) - float(_TENNIS_CROSS_WORLD_Y[labels[index]]))
        for index in range(len(labels) - 1)
    ]
    pickleball_error = _spacing_error(positive_gaps, pickleball_expected)
    tennis_error = _spacing_error(positive_gaps, tennis_expected)
    tennis_margin = float(pickleball_error) - float(tennis_error)
    tennis_better = tennis_margin > 0.05
    penalty = max(0.0, tennis_margin - 0.05) * 120.0 + ordering_violation_count * 25.0
    return {
        "available": True,
        "tennis_template_penalty": round(float(penalty), 4),
        "cross_template": {
            "labels": labels,
            "observed_y_at_x_ref": [round(float(y), 3) for _label, y in entries],
            "observed_gaps_px": [round(float(gap), 3) for gap in observed_gaps],
            "pickleball_expected_gaps_ft": [round(float(value), 3) for value in pickleball_expected],
            "tennis_service_expected_gaps_ft": [round(float(value), 3) for value in tennis_expected],
            "pickleball_spacing_error": round(float(pickleball_error), 6),
            "tennis_service_spacing_error": round(float(tennis_error), 6),
            "tennis_margin": round(float(tennis_margin), 6),
            "tennis_better_than_pickleball": bool(tennis_better),
            "ordering_violation_count": int(ordering_violation_count),
        },
    }


def _segment_from_assignment_mapping(item: Mapping[str, Any]) -> dict[str, Any]:
    p1 = item.get("p1")
    p2 = item.get("p2")
    if not isinstance(p1, Sequence) or not isinstance(p2, Sequence) or len(p1) != 2 or len(p2) != 2:
        raise ValueError("assignment item must contain p1 and p2")
    return {
        "p1": [float(p1[0]), float(p1[1])],
        "p2": [float(p2[0]), float(p2[1])],
        "length_px": float(item.get("length_px") or math.dist((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))),
        "angle_deg": float(item.get("angle_deg") or math.degrees(math.atan2(float(p2[1]) - float(p1[1]), float(p2[0]) - float(p1[0])))),
        "source": item.get("source", "line_assignment"),
    }


def score_line_candidates_against_reviewed_keypoints(
    *,
    reviewed_keypoints_path: str | Path,
    line_candidates: Sequence[Mapping[str, Any]],
    candidate_image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Score candidate segments against reviewed floor-line pairs.

    This is a support metric, not a homography/calibration pass. It asks whether
    a detector produced segments close to already-reviewed court lines.
    """

    reviewed = _load_reviewed_points_native_px(Path(reviewed_keypoints_path))
    scaled_candidates = _scale_line_candidates_to_native(
        line_candidates=line_candidates,
        candidate_image_size=candidate_image_size,
        native_image_size=reviewed.native_size,
    )
    per_line: dict[str, dict[str, Any]] = {}
    evaluated = 0
    supported = 0
    for line_name, p1_name, p2_name in FLOOR_LINE_KEYPOINT_PAIRS:
        p1 = reviewed.points.get(p1_name)
        p2 = reviewed.points.get(p2_name)
        if p1 is None or p2 is None:
            per_line[line_name] = {
                "status": "missing_reviewed_endpoint",
                "endpoint_names": [p1_name, p2_name],
            }
            continue
        evaluated += 1
        best = _best_segment_support_for_line(p1, p2, scaled_candidates)
        is_supported = bool(
            best is not None
            and best["angle_diff_deg"] <= 12.0
            and best["mean_perpendicular_distance_px"] <= 16.0
            and best["overlap_fraction"] >= 0.08
        )
        if is_supported:
            supported += 1
        per_line[line_name] = {
            "status": "supported" if is_supported else "unsupported",
            "endpoint_names": [p1_name, p2_name],
            "best_segment": best,
        }
    return {
        "evaluated_line_count": evaluated,
        "supported_line_count": supported,
        "support_ratio": None if evaluated == 0 else round(supported / evaluated, 4),
        "per_line": per_line,
    }


def score_projected_line_pixels_against_image(
    image_bgr: Any,
    keypoints: Mapping[str, Sequence[float]],
    *,
    line_width_px: int = 5,
    line_pixel_mask: Any | None = None,
) -> dict[str, Any]:
    """Score projected regulation floor lines against local high-contrast line pixels."""

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        return {"available": False, "reason": "invalid_image"}
    mask = line_pixel_mask if line_pixel_mask is not None else _court_line_pixel_mask(image_bgr, dilation_px=max(1, int(line_width_px)))
    height, width = int(mask.shape[0]), int(mask.shape[1])
    per_line: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    supported_count = 0
    evaluated_count = 0
    for line_name, (p1_name, p2_name) in _IMAGE_EVIDENCE_LINE_ENDPOINT_KEYPOINTS.items():
        raw_p1 = keypoints.get(p1_name)
        raw_p2 = keypoints.get(p2_name)
        if not _is_xy(raw_p1) or not _is_xy(raw_p2):
            per_line[line_name] = {
                "status": "missing_projected_endpoint",
                "endpoint_names": [p1_name, p2_name],
            }
            continue
        p1 = (float(raw_p1[0]), float(raw_p1[1]))
        p2 = (float(raw_p2[0]), float(raw_p2[1]))
        samples = _sample_points_on_segment(p1, p2, spacing_px=5.0, min_count=16, max_count=96)
        supported_samples = 0
        inside_samples = 0
        for x, y in samples:
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < width and 0 <= iy < height:
                inside_samples += 1
                if int(mask[iy, ix]) > 0:
                    supported_samples += 1
        ratio = supported_samples / float(len(samples)) if samples else 0.0
        inside_ratio = inside_samples / float(len(samples)) if samples else 0.0
        supported = ratio >= 0.34 and inside_ratio >= 0.45
        evaluated_count += 1
        if supported:
            supported_count += 1
        ratios.append(ratio)
        per_line[line_name] = {
            "status": "supported" if supported else "unsupported",
            "endpoint_names": [p1_name, p2_name],
            "sample_count": len(samples),
            "inside_image_sample_count": int(inside_samples),
            "line_pixel_sample_count": int(supported_samples),
            "line_pixel_support_ratio": round(float(ratio), 4),
            "inside_image_ratio": round(float(inside_ratio), 4),
        }
    return {
        "available": evaluated_count > 0,
        "mode": "local_high_contrast_value_mask",
        "evaluated_line_count": int(evaluated_count),
        "supported_line_pixel_count": int(supported_count),
        "mean_line_pixel_support_ratio": round(float(sum(ratios) / len(ratios)), 4) if ratios else 0.0,
        "mask_support_ratio": round(float((mask > 0).sum()) / float(mask.size), 6) if mask.size else 0.0,
        "per_line": per_line,
    }


def score_projected_line_distance_transform_against_image(
    image_bgr: Any,
    keypoints: Mapping[str, Sequence[float]],
    *,
    line_pixel_mask: Any | None = None,
    distance_map: Any | None = None,
) -> dict[str, Any]:
    """Score projected regulation lines by distance to the nearest line-like pixel."""

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        return {"available": False, "reason": "invalid_image"}
    mask = line_pixel_mask if line_pixel_mask is not None else _court_line_pixel_mask(image_bgr, dilation_px=3)
    distances = distance_map if distance_map is not None else _line_pixel_distance_transform(mask)
    height, width = int(mask.shape[0]), int(mask.shape[1])
    per_line: dict[str, dict[str, Any]] = {}
    line_means: list[float] = []
    line_p95s: list[float] = []
    supported_count = 0
    evaluated_count = 0
    for line_name, (p1_name, p2_name) in _IMAGE_EVIDENCE_LINE_ENDPOINT_KEYPOINTS.items():
        raw_p1 = keypoints.get(p1_name)
        raw_p2 = keypoints.get(p2_name)
        if not _is_xy(raw_p1) or not _is_xy(raw_p2):
            per_line[line_name] = {
                "status": "missing_projected_endpoint",
                "endpoint_names": [p1_name, p2_name],
            }
            continue
        p1 = (float(raw_p1[0]), float(raw_p1[1]))
        p2 = (float(raw_p2[0]), float(raw_p2[1]))
        samples = _sample_points_on_segment(p1, p2, spacing_px=5.0, min_count=16, max_count=96)
        sample_distances: list[float] = []
        inside_samples = 0
        for x, y in samples:
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < width and 0 <= iy < height:
                inside_samples += 1
                sample_distances.append(float(distances[iy, ix]))
        if sample_distances:
            mean_distance = sum(sample_distances) / float(len(sample_distances))
            p95_distance = _percentile(sample_distances, 95.0)
        else:
            mean_distance = float(max(width, height))
            p95_distance = float(max(width, height))
        inside_ratio = inside_samples / float(len(samples)) if samples else 0.0
        supported = mean_distance <= 5.0 and p95_distance <= 12.0 and inside_ratio >= 0.45
        evaluated_count += 1
        if supported:
            supported_count += 1
        line_means.append(mean_distance)
        line_p95s.append(p95_distance)
        per_line[line_name] = {
            "status": "supported" if supported else "unsupported",
            "endpoint_names": [p1_name, p2_name],
            "sample_count": len(samples),
            "inside_image_sample_count": int(inside_samples),
            "inside_image_ratio": round(float(inside_ratio), 4),
            "mean_distance_px": round(float(mean_distance), 4),
            "p95_distance_px": round(float(p95_distance), 4),
        }
    return {
        "available": evaluated_count > 0,
        "mode": "distance_transform_local_high_contrast_mask",
        "evaluated_line_count": int(evaluated_count),
        "distance_supported_line_count": int(supported_count),
        "mean_projected_line_distance_px": round(float(sum(line_means) / len(line_means)), 4) if line_means else 0.0,
        "p95_projected_line_distance_px": round(_percentile(line_p95s, 95.0), 4) if line_p95s else 0.0,
        "mask_support_ratio": round(float((mask > 0).sum()) / float(mask.size), 6) if mask.size else 0.0,
        "per_line": per_line,
    }


def score_line_color_consistency_for_assignment(
    image_bgr: Any,
    line_assignment: Mapping[str, Any],
    *,
    line_pixel_mask: Any | None = None,
) -> dict[str, Any]:
    """Estimate whether assigned pickleball lines belong to one local color layer."""

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        return {"available": False, "reason": "invalid_image"}
    mask = line_pixel_mask if line_pixel_mask is not None else _court_line_pixel_mask(image_bgr, dilation_px=2)
    per_line: dict[str, dict[str, Any]] = {}
    clusters: dict[str, int] = {}
    sampled_count = 0
    for name, raw in sorted(line_assignment.items()):
        try:
            segment = raw.segment if isinstance(raw, _CandidateLine) else _segment_from_assignment_mapping(raw)
        except (AttributeError, TypeError, ValueError):
            per_line[str(name)] = {"status": "invalid_segment"}
            continue
        p1 = tuple(float(value) for value in segment["p1"])
        p2 = tuple(float(value) for value in segment["p2"])
        pixels = _sample_line_pixels(image_bgr, mask, p1, p2)
        if not pixels:
            per_line[str(name)] = {
                "status": "no_line_pixels",
                "sample_count": 0,
            }
            continue
        mean_bgr = [
            sum(float(pixel[index]) for pixel in pixels) / float(len(pixels))
            for index in range(3)
        ]
        cluster = _line_color_cluster(mean_bgr)
        clusters[cluster] = clusters.get(cluster, 0) + 1
        sampled_count += 1
        per_line[str(name)] = {
            "status": "sampled",
            "sample_count": len(pixels),
            "mean_bgr": [round(float(value), 3) for value in mean_bgr],
            "color_cluster": cluster,
        }
    if sampled_count == 0:
        return {
            "available": False,
            "reason": "no_line_pixels_sampled",
            "sampled_line_count": 0,
            "per_line": per_line,
        }
    dominant_cluster, dominant_count = max(clusters.items(), key=lambda item: item[1])
    dominant_fraction = dominant_count / float(sampled_count)
    distinct_count = len(clusters)
    mixed_penalty = max(0.0, (distinct_count - 1) * 18.0 + (1.0 - dominant_fraction) * 42.0)
    return {
        "available": True,
        "sampled_line_count": int(sampled_count),
        "distinct_color_cluster_count": int(distinct_count),
        "dominant_color_cluster": dominant_cluster,
        "dominant_color_cluster_fraction": round(float(dominant_fraction), 4),
        "cluster_counts": dict(sorted(clusters.items())),
        "mixed_layer_penalty": round(float(mixed_penalty), 4),
        "per_line": per_line,
    }


def build_court_finding_technology_report(
    *,
    eval_root: str | Path,
    technologies: Sequence[str] = DEFAULT_TECHNOLOGIES,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate technology adapters on all available reviewed court samples."""

    out_root = Path(out_dir)
    samples = discover_court_finding_samples(eval_root)
    results: list[dict[str, Any]] = []
    for sample in samples:
        frame, frame_meta = load_player_suppressed_frame(sample.frame_input)
        proposal_cache: dict[str, dict[str, Any]] = {}
        for technology_id in technologies:
            result_dir = out_root / sample.clip / technology_id
            result_dir.mkdir(parents=True, exist_ok=True)
            try:
                if technology_id in LINE_CANDIDATE_ONLY_TECHNOLOGIES:
                    if technology_id in {"opencv_hough_lsd_temporal", "opencv_hough_lsd_temporal_persistent"}:
                        evidence = detect_temporal_line_candidates_for_input(
                            sample.frame_input,
                            technology_id=technology_id,
                        )
                    else:
                        evidence = detect_line_candidates_for_technology(frame, technology_id)
                    line_support = score_line_candidates_against_reviewed_keypoints(
                        reviewed_keypoints_path=sample.label_path,
                        line_candidates=evidence["segments"],
                        candidate_image_size=(int(frame.shape[1]), int(frame.shape[0])),
                    )
                    evidence_path = result_dir / "line_candidates.json"
                    _write_json(evidence_path, evidence)
                    line_support_path = result_dir / "line_support.json"
                    _write_json(line_support_path, line_support)
                    status = "line_candidates_only" if int(evidence["candidate_count"]) > 0 else "no_line_candidates"
                    results.append(
                        {
                            "clip": sample.clip,
                            "label_kind": sample.label_kind,
                            "technology_id": technology_id,
                            "status": status,
                            "frame_meta": dict(frame_meta),
                            "line_candidate_path": str(evidence_path),
                            "line_candidate_count": int(evidence["candidate_count"]),
                            "line_support_path": str(line_support_path),
                            "line_support_ratio": line_support.get("support_ratio"),
                            "supported_line_count": line_support.get("supported_line_count"),
                            "evaluated_line_count": line_support.get("evaluated_line_count"),
                            "scored": False,
                        }
                    )
                    continue

                if technology_id in proposal_cache:
                    proposal = proposal_cache[technology_id]
                else:
                    proposal = _proposal_for_technology(
                        frame,
                        sample=sample,
                        technology_id=technology_id,
                        proposal_cache=proposal_cache,
                    )
                    proposal_cache[technology_id] = proposal
                proposal_path = result_dir / "court_proposal.json"
                _write_json(proposal_path, proposal)
                comparison = compare_proposal_to_reviewed_keypoints(
                    reviewed_keypoints_path=sample.label_path,
                    proposal_path=proposal_path,
                )
                comparison_path = result_dir / "comparison.json"
                _write_json(comparison_path, comparison)
                floor_visible = comparison["groups"]["floor_visible"]
                all_visible = comparison["groups"]["all_visible"]
                results.append(
                    {
                        "clip": sample.clip,
                        "label_kind": sample.label_kind,
                        "technology_id": technology_id,
                        "status": "scored",
                        "frame_meta": dict(frame_meta),
                        "proposal_path": str(proposal_path),
                        "comparison_path": str(comparison_path),
                        "scored": True,
                        "verdict": comparison.get("verdict"),
                        "needs_user_input": comparison.get("needs_user_input"),
                        "all_visible_median_px": all_visible.get("median_px"),
                        "floor_visible_median_px": floor_visible.get("median_px"),
                        "floor_visible_p95_px": floor_visible.get("p95_px"),
                        "visible_keypoint_count": int(all_visible.get("count") or 0),
                        "rejection_reasons": _result_rejection_reasons(
                            technology_id=technology_id,
                            comparison=comparison,
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - exercised by real experiments.
                results.append(
                    {
                        "clip": sample.clip,
                        "label_kind": sample.label_kind,
                        "technology_id": technology_id,
                        "status": "error",
                        "frame_meta": dict(frame_meta),
                        "scored": False,
                        "error": str(exc),
                    }
                )

    report = {
        "schema_version": 1,
        "artifact_type": BENCHMARK_ARTIFACT_TYPE,
        "status": "ran_not_verified",
        "verified": False,
        "not_cal3_verified": True,
        "summary": _summary(samples=samples, technologies=technologies, results=results),
        "results": results,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "court_finding_technology_benchmark.json", report)
    return report


def _proposal_for_technology(
    image_bgr: Any,
    *,
    sample: CourtFindingSample,
    technology_id: str,
    proposal_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if technology_id == "net_anchor":
        return solve_net_anchor_court_from_frame(image_bgr, clip_id=sample.clip)
    if technology_id == "opencv_hough_lsd_regulation":
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
        )
    if technology_id == "opencv_hough_lsd_regulation_distance_mask":
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
            image_evidence_mode="distance_mask",
        )
    if technology_id == "opencv_hough_lsd_regulation_line_refined":
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
            line_refinement=True,
        )
    if technology_id == "opencv_hough_lsd_skimage_regulation":
        line_evidence = detect_line_candidates_for_technology(image_bgr, "opencv_hough_lsd_skimage")
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
            line_segments=line_evidence["segments"],
            line_evidence=line_evidence,
        )
    if technology_id == "opencv_hough_lsd_temporal_regulation":
        line_evidence = detect_temporal_line_candidates_for_input(sample.frame_input)
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
            line_segments=line_evidence["segments"],
            line_evidence=line_evidence,
        )
    if technology_id == "opencv_hough_lsd_temporal_regulation_distance_mask":
        line_evidence = detect_temporal_line_candidates_for_input(sample.frame_input)
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
            line_segments=line_evidence["segments"],
            line_evidence=line_evidence,
            image_evidence_mode="distance_mask",
        )
    if technology_id == "opencv_hough_lsd_temporal_persistent_regulation":
        line_evidence = detect_temporal_line_candidates_for_input(
            sample.frame_input,
            technology_id="opencv_hough_lsd_temporal_persistent",
        )
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
            line_segments=line_evidence["segments"],
            line_evidence=line_evidence,
        )
    if technology_id == "opencv_hough_lsd_temporal_persistent_regulation_distance_mask":
        line_evidence = detect_temporal_line_candidates_for_input(
            sample.frame_input,
            technology_id="opencv_hough_lsd_temporal_persistent",
        )
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
            line_segments=line_evidence["segments"],
            line_evidence=line_evidence,
            image_evidence_mode="distance_mask",
        )
    if technology_id == "opencv_hsv_paint_hough_regulation":
        line_evidence = detect_line_candidates_for_technology(image_bgr, "opencv_hsv_paint_hough")
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
            line_segments=line_evidence["segments"],
            line_evidence=line_evidence,
        )
    if technology_id == "opencv_hsv_paint_net_crop_hough_regulation":
        line_evidence = detect_line_candidates_for_technology(image_bgr, "opencv_hsv_paint_net_crop_hough")
        return solve_regulation_court_from_line_candidates(
            image_bgr,
            technology_id=technology_id,
            clip_id=sample.clip,
            line_segments=line_evidence["segments"],
            line_evidence=line_evidence,
        )
    if technology_id == "hough_or_regulation_line_selector":
        return _select_hough_or_regulation_proposal(
            image_bgr,
            sample=sample,
            proposal_cache=proposal_cache if proposal_cache is not None else {},
        )
    if technology_id == "hough_or_regulation_distance_mask_selector":
        return _select_hough_or_regulation_distance_mask_proposal(
            image_bgr,
            sample=sample,
            proposal_cache=proposal_cache if proposal_cache is not None else {},
        )
    if technology_id == "hough_or_refined_regulation_line_selector":
        return _select_hough_or_refined_regulation_proposal(
            image_bgr,
            sample=sample,
            proposal_cache=proposal_cache if proposal_cache is not None else {},
        )
    if technology_id == "hough_regulation_temporal_line_selector":
        return _select_hough_regulation_temporal_proposal(
            image_bgr,
            sample=sample,
            proposal_cache=proposal_cache if proposal_cache is not None else {},
        )
    if technology_id == "hough_regulation_temporal_balanced_selector":
        return _select_hough_regulation_temporal_balanced_proposal(
            image_bgr,
            sample=sample,
            proposal_cache=proposal_cache if proposal_cache is not None else {},
        )
    if technology_id == "hough_regulation_temporal_persistent_tail_selector":
        return _select_hough_regulation_temporal_persistent_tail_proposal(
            image_bgr,
            sample=sample,
            proposal_cache=proposal_cache if proposal_cache is not None else {},
        )
    if technology_id == "reviewed_oracle_hough_or_regulation":
        return _select_reviewed_oracle_hough_or_regulation_proposal(
            image_bgr,
            sample=sample,
            proposal_cache=proposal_cache if proposal_cache is not None else {},
        )
    if technology_id == "reviewed_oracle_hough_regulation_temporal":
        return _select_reviewed_oracle_hough_regulation_temporal_proposal(
            image_bgr,
            sample=sample,
            proposal_cache=proposal_cache if proposal_cache is not None else {},
        )
    if technology_id == "hough_keypoints":
        detected = detect_court_keypoints_from_image(image_bgr)
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        keypoints = {}
        confidences: list[float] = []
        for name, item in detected.keypoints.items():
            xy = item.get("xy") if isinstance(item, Mapping) else None
            if not isinstance(xy, Sequence) or len(xy) != 2:
                continue
            confidence = float(item.get("confidence", detected.confidence)) if isinstance(item, Mapping) else float(detected.confidence)
            confidences.append(confidence)
            keypoints[str(name)] = {
                "xy": [float(xy[0]), float(xy[1])],
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "source": "court_line_keypoints_hough",
            }
        solver_confidence = _median(confidences) if confidences else 0.0
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_court_finding_technology_proposal",
            "source": {
                "clip_id": sample.clip,
                "image_size": [width, height],
            },
            "solver": {
                "name": "hough_keypoints",
                "version": 1,
                "writes_court_calibration": False,
            },
            "keypoints": keypoints,
            "solver_confidence": round(float(solver_confidence), 4),
            "needs_user_input": [] if solver_confidence >= 0.70 else ["court_keypoints"],
            "needs_user_confirmation": solver_confidence < 0.70,
            "notes": ["benchmark_proposal_only_never_writes_court_calibration_json"],
        }
    raise ValueError(f"unsupported proposal technology: {technology_id}")


def _result_rejection_reasons(*, technology_id: str, comparison: Mapping[str, Any]) -> list[str]:
    reasons = list(comparison.get("rejection_reasons") or [])
    if technology_id in {"reviewed_oracle_hough_or_regulation", "reviewed_oracle_hough_regulation_temporal"}:
        reasons.append("reviewed_label_oracle_not_deployable")
    return reasons


def _select_hough_or_regulation_proposal(
    image_bgr: Any,
    *,
    sample: CourtFindingSample,
    proposal_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    hough = proposal_cache.get("hough_keypoints")
    if hough is None:
        hough = _proposal_for_technology(
            image_bgr,
            sample=sample,
            technology_id="hough_keypoints",
            proposal_cache=proposal_cache,
        )
        proposal_cache["hough_keypoints"] = hough
    regulation = proposal_cache.get("opencv_hough_lsd_regulation")
    if regulation is None:
        regulation = _proposal_for_technology(
            image_bgr,
            sample=sample,
            technology_id="opencv_hough_lsd_regulation",
            proposal_cache=proposal_cache,
        )
        proposal_cache["opencv_hough_lsd_regulation"] = regulation

    segments = detect_line_candidates_for_technology(image_bgr, "opencv_hough_lsd")["segments"]
    hough_score = _proposal_internal_line_score(hough, segments)
    regulation_score = _proposal_internal_line_score(regulation, segments)
    hough_score["geometry_risk"] = _proposal_geometry_risk_score(hough)
    regulation_score["geometry_risk"] = _proposal_geometry_risk_score(regulation)
    if _should_prefer_hough_over_collapsed_regulation(hough_score, regulation_score):
        selected_name = "hough_keypoints"
    else:
        selected_name = "hough_keypoints" if hough_score["selector_cost"] <= regulation_score["selector_cost"] else "opencv_hough_lsd_regulation"
    selected = hough if selected_name == "hough_keypoints" else regulation
    payload = json.loads(json.dumps(selected))
    payload["solver"] = {
        "name": "hough_or_regulation_line_selector",
        "version": 1,
        "writes_court_calibration": False,
    }
    payload["solver_confidence"] = min(float(payload.get("solver_confidence") or 0.0), 0.62)
    payload["needs_user_input"] = ["court_keypoints"]
    payload["needs_user_confirmation"] = True
    payload["selector"] = {
        "selected_technology_id": selected_name,
        "candidate_scores": {
            "hough_keypoints": hough_score,
            "opencv_hough_lsd_regulation": regulation_score,
        },
        "rule": "choose_lowest_projected_line_selector_cost",
    }
    payload.setdefault("notes", [])
    payload["notes"].append("selector_proposal_only_never_writes_court_calibration_json")
    for item in payload.get("keypoints", {}).values():
        if isinstance(item, dict):
            item["source"] = f"hough_or_regulation_line_selector:{selected_name}"
    return payload


def _select_hough_or_regulation_distance_mask_proposal(
    image_bgr: Any,
    *,
    sample: CourtFindingSample,
    proposal_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for technology_id in (
        "hough_keypoints",
        "opencv_hough_lsd_regulation",
        "opencv_hough_lsd_regulation_distance_mask",
    ):
        proposal = proposal_cache.get(technology_id)
        if proposal is None:
            proposal = _proposal_for_technology(
                image_bgr,
                sample=sample,
                technology_id=technology_id,
                proposal_cache=proposal_cache,
            )
            proposal_cache[technology_id] = proposal
        candidates[technology_id] = proposal

    segments = detect_line_candidates_for_technology(image_bgr, "opencv_hough_lsd")["segments"]
    candidate_scores = {
        technology_id: _proposal_internal_line_score(proposal, segments)
        for technology_id, proposal in candidates.items()
    }
    for technology_id, score in candidate_scores.items():
        score["geometry_risk"] = _proposal_geometry_risk_score(candidates[technology_id])
    noncollapsed_candidates = {
        name: score
        for name, score in candidate_scores.items()
        if not (
            name != "hough_keypoints"
            and _should_prefer_hough_over_collapsed_regulation(candidate_scores["hough_keypoints"], score)
        )
    }
    selected_name = min(noncollapsed_candidates, key=lambda name: float(noncollapsed_candidates[name]["selector_cost"]))
    selected = candidates[selected_name]
    payload = json.loads(json.dumps(selected))
    payload["solver"] = {
        "name": "hough_or_regulation_distance_mask_selector",
        "version": 1,
        "writes_court_calibration": False,
    }
    payload["solver_confidence"] = min(float(payload.get("solver_confidence") or 0.0), 0.62)
    payload["needs_user_input"] = ["court_keypoints"]
    payload["needs_user_confirmation"] = True
    payload["selector"] = {
        "selected_technology_id": selected_name,
        "candidate_scores": candidate_scores,
        "rule": "choose_lowest_projected_line_cost_across_hough_sparse_regulation_and_distance_mask_regulation_with_collapsed_geometry_guard",
    }
    payload.setdefault("notes", [])
    payload["notes"].append("distance_mask_selector_proposal_only_never_writes_court_calibration_json")
    for item in payload.get("keypoints", {}).values():
        if isinstance(item, dict):
            item["source"] = f"hough_or_regulation_distance_mask_selector:{selected_name}"
    return payload


def _select_hough_or_refined_regulation_proposal(
    image_bgr: Any,
    *,
    sample: CourtFindingSample,
    proposal_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for technology_id in (
        "hough_keypoints",
        "opencv_hough_lsd_regulation",
        "opencv_hough_lsd_regulation_line_refined",
    ):
        proposal = proposal_cache.get(technology_id)
        if proposal is None:
            proposal = _proposal_for_technology(
                image_bgr,
                sample=sample,
                technology_id=technology_id,
                proposal_cache=proposal_cache,
            )
            proposal_cache[technology_id] = proposal
        candidates[technology_id] = proposal

    segments = detect_line_candidates_for_technology(image_bgr, "opencv_hough_lsd")["segments"]
    candidate_scores = {
        technology_id: _proposal_internal_line_score(proposal, segments)
        for technology_id, proposal in candidates.items()
    }
    for technology_id, score in candidate_scores.items():
        score["geometry_risk"] = _proposal_geometry_risk_score(candidates[technology_id])

    if _should_prefer_hough_over_collapsed_regulation(
        candidate_scores["hough_keypoints"],
        candidate_scores["opencv_hough_lsd_regulation"],
    ):
        selected_name = "hough_keypoints"
    else:
        hough_cost = float(candidate_scores["hough_keypoints"]["selector_cost"])
        regulation_cost = float(candidate_scores["opencv_hough_lsd_regulation"]["selector_cost"])
        selected_name = "hough_keypoints" if hough_cost <= regulation_cost else "opencv_hough_lsd_regulation"

    refinement = (
        candidates["opencv_hough_lsd_regulation_line_refined"]
        .get("score_components", {})
        .get("line_refinement", {})
    )
    pixel_guard = refinement.get("pixel_guard") if isinstance(refinement, Mapping) else None
    refined_risk = candidate_scores["opencv_hough_lsd_regulation_line_refined"]["geometry_risk"]
    regulation_risk = candidate_scores["opencv_hough_lsd_regulation"]["geometry_risk"]
    refined_upgrade = bool(
        isinstance(refinement, Mapping)
        and refinement.get("accepted") is True
        and isinstance(pixel_guard, Mapping)
        and float(pixel_guard.get("mean_line_pixel_support_delta") or 0.0) >= 0.015
        and int(pixel_guard.get("supported_line_pixel_delta") or 0) >= 0
        and float(refined_risk.get("risk_score") or 0.0) <= float(regulation_risk.get("risk_score") or 0.0) + 2.0
    )
    if selected_name == "opencv_hough_lsd_regulation" and refined_upgrade:
        selected_name = "opencv_hough_lsd_regulation_line_refined"

    selected = candidates[selected_name]
    payload = json.loads(json.dumps(selected))
    payload["solver"] = {
        "name": "hough_or_refined_regulation_line_selector",
        "version": 1,
        "writes_court_calibration": False,
    }
    payload["solver_confidence"] = min(float(payload.get("solver_confidence") or 0.0), 0.62)
    payload["needs_user_input"] = ["court_keypoints"]
    payload["needs_user_confirmation"] = True
    payload["selector"] = {
        "selected_technology_id": selected_name,
        "candidate_scores": candidate_scores,
        "refined_upgrade_eligible": bool(refined_upgrade),
        "rule": "start_from_hough_or_single_frame_regulation_then_upgrade_to_guarded_line_refinement_when_self_verification_improves_pixel_support",
    }
    payload.setdefault("notes", [])
    payload["notes"].append("refined_selector_proposal_only_never_writes_court_calibration_json")
    payload["notes"].append("line_refinement_upgrade_requires_optimizer_acceptance_and_projected_pixel_support_gain")
    for item in payload.get("keypoints", {}).values():
        if isinstance(item, dict):
            item["source"] = f"hough_or_refined_regulation_line_selector:{selected_name}"
    return payload


def _select_hough_regulation_temporal_proposal(
    image_bgr: Any,
    *,
    sample: CourtFindingSample,
    proposal_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for technology_id in (
        "hough_keypoints",
        "opencv_hough_lsd_regulation",
        "opencv_hough_lsd_temporal_regulation",
    ):
        proposal = proposal_cache.get(technology_id)
        if proposal is None:
            proposal = _proposal_for_technology(
                image_bgr,
                sample=sample,
                technology_id=technology_id,
                proposal_cache=proposal_cache,
            )
            proposal_cache[technology_id] = proposal
        candidates[technology_id] = proposal

    median_segments = detect_line_candidates_for_technology(image_bgr, "opencv_hough_lsd")["segments"]
    temporal_evidence = detect_temporal_line_candidates_for_input(sample.frame_input)
    temporal_segments = temporal_evidence["segments"]
    candidate_scores = {
        "hough_keypoints": _proposal_internal_line_score(candidates["hough_keypoints"], median_segments),
        "opencv_hough_lsd_regulation": _proposal_internal_line_score(
            candidates["opencv_hough_lsd_regulation"],
            median_segments,
        ),
        "opencv_hough_lsd_temporal_regulation": _proposal_internal_line_score(
            candidates["opencv_hough_lsd_temporal_regulation"],
            temporal_segments,
        ),
    }
    for technology_id, score in candidate_scores.items():
        score["geometry_risk"] = _proposal_geometry_risk_score(candidates[technology_id])
    noncollapsed_candidates = {
        name: score
        for name, score in candidate_scores.items()
        if not _score_has_geometry_reason(score, "projected_floor_height_too_collapsed")
        and not _score_has_geometry_reason(score, "projected_floor_width_too_collapsed")
    }
    if (
        "hough_keypoints" in noncollapsed_candidates
        and len(noncollapsed_candidates) < len(candidate_scores)
        and int(candidate_scores["hough_keypoints"].get("supported_line_count") or 0) >= 3
    ):
        selected_name = min(noncollapsed_candidates, key=lambda name: float(noncollapsed_candidates[name]["selector_cost"]))
    else:
        selected_name = min(candidate_scores, key=lambda name: float(candidate_scores[name]["selector_cost"]))
    selected = candidates[selected_name]
    payload = json.loads(json.dumps(selected))
    payload["solver"] = {
        "name": "hough_regulation_temporal_line_selector",
        "version": 1,
        "writes_court_calibration": False,
    }
    payload["solver_confidence"] = min(float(payload.get("solver_confidence") or 0.0), 0.62)
    payload["needs_user_input"] = ["court_keypoints"]
    payload["needs_user_confirmation"] = True
    payload["selector"] = {
        "selected_technology_id": selected_name,
        "candidate_scores": candidate_scores,
        "rule": "choose_lowest_projected_line_selector_cost_with_temporal_pool_for_temporal_candidate",
    }
    payload.setdefault("notes", [])
    payload["notes"].append("selector_proposal_only_never_writes_court_calibration_json")
    payload["notes"].append("temporal_candidate_uses_multiframe_line_pool_but_still_requires_review")
    for item in payload.get("keypoints", {}).values():
        if isinstance(item, dict):
            item["source"] = f"hough_regulation_temporal_line_selector:{selected_name}"
    return payload


def _select_hough_regulation_temporal_balanced_proposal(
    image_bgr: Any,
    *,
    sample: CourtFindingSample,
    proposal_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for technology_id in (
        "hough_keypoints",
        "opencv_hough_lsd_regulation",
        "opencv_hough_lsd_temporal_regulation",
    ):
        proposal = proposal_cache.get(technology_id)
        if proposal is None:
            proposal = _proposal_for_technology(
                image_bgr,
                sample=sample,
                technology_id=technology_id,
                proposal_cache=proposal_cache,
            )
            proposal_cache[technology_id] = proposal
        candidates[technology_id] = proposal

    median_segments = detect_line_candidates_for_technology(image_bgr, "opencv_hough_lsd")["segments"]
    temporal_evidence = detect_temporal_line_candidates_for_input(sample.frame_input)
    temporal_segments = temporal_evidence["segments"]
    candidate_scores = {
        "hough_keypoints": _proposal_internal_line_score(candidates["hough_keypoints"], median_segments),
        "opencv_hough_lsd_regulation": _proposal_internal_line_score(
            candidates["opencv_hough_lsd_regulation"],
            median_segments,
        ),
        "opencv_hough_lsd_temporal_regulation": _proposal_internal_line_score(
            candidates["opencv_hough_lsd_temporal_regulation"],
            temporal_segments,
        ),
    }
    for technology_id, score in candidate_scores.items():
        score["geometry_risk"] = _proposal_geometry_risk_score(candidates[technology_id])

    hough_cost = float(candidate_scores["hough_keypoints"]["selector_cost"])
    single_cost = float(candidate_scores["opencv_hough_lsd_regulation"]["selector_cost"])
    temporal_cost = float(candidate_scores["opencv_hough_lsd_temporal_regulation"]["selector_cost"])
    temporal_risk = candidate_scores["opencv_hough_lsd_temporal_regulation"]["geometry_risk"]
    temporal_eligible = (
        bool(temporal_risk["eligible_for_temporal_selection"])
        and temporal_cost <= single_cost - 8.0
        and hough_cost > -60.0
    )
    if _should_prefer_hough_over_collapsed_regulation(
        candidate_scores["hough_keypoints"],
        candidate_scores["opencv_hough_lsd_regulation"],
    ):
        selected_name = "hough_keypoints"
    elif temporal_eligible:
        selected_name = "opencv_hough_lsd_temporal_regulation"
    else:
        selected_name = (
            "hough_keypoints"
            if hough_cost <= single_cost
            else "opencv_hough_lsd_regulation"
        )

    selected = candidates[selected_name]
    payload = json.loads(json.dumps(selected))
    payload["solver"] = {
        "name": "hough_regulation_temporal_balanced_selector",
        "version": 1,
        "writes_court_calibration": False,
    }
    payload["solver_confidence"] = min(float(payload.get("solver_confidence") or 0.0), 0.62)
    payload["needs_user_input"] = ["court_keypoints"]
    payload["needs_user_confirmation"] = True
    payload["selector"] = {
        "selected_technology_id": selected_name,
        "candidate_scores": candidate_scores,
        "rule": "prefer_low_risk_temporal_regulation_only_when_it_clearly_beats_single_frame_and_hough_is_not_already_strong",
    }
    payload.setdefault("notes", [])
    payload["notes"].append("balanced_selector_proposal_only_never_writes_court_calibration_json")
    payload["notes"].append("temporal_candidate_rejected_when_projected_geometry_is_collapsed_or_unstable")
    for item in payload.get("keypoints", {}).values():
        if isinstance(item, dict):
            item["source"] = f"hough_regulation_temporal_balanced_selector:{selected_name}"
    return payload


def _select_hough_regulation_temporal_persistent_tail_proposal(
    image_bgr: Any,
    *,
    sample: CourtFindingSample,
    proposal_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for technology_id in (
        "hough_keypoints",
        "opencv_hough_lsd_regulation",
        "opencv_hough_lsd_temporal_persistent_regulation",
    ):
        proposal = proposal_cache.get(technology_id)
        if proposal is None:
            proposal = _proposal_for_technology(
                image_bgr,
                sample=sample,
                technology_id=technology_id,
                proposal_cache=proposal_cache,
            )
            proposal_cache[technology_id] = proposal
        candidates[technology_id] = proposal

    median_segments = detect_line_candidates_for_technology(image_bgr, "opencv_hough_lsd")["segments"]
    persistent_evidence = detect_temporal_line_candidates_for_input(
        sample.frame_input,
        technology_id="opencv_hough_lsd_temporal_persistent",
    )
    persistent_segments = persistent_evidence["segments"]
    candidate_scores = {
        "hough_keypoints": _proposal_internal_line_score(candidates["hough_keypoints"], median_segments),
        "opencv_hough_lsd_regulation": _proposal_internal_line_score(
            candidates["opencv_hough_lsd_regulation"],
            median_segments,
        ),
        "opencv_hough_lsd_temporal_persistent_regulation": _proposal_internal_line_score(
            candidates["opencv_hough_lsd_temporal_persistent_regulation"],
            persistent_segments,
        ),
    }
    for technology_id, score in candidate_scores.items():
        score["geometry_risk"] = _proposal_geometry_risk_score(candidates[technology_id])

    hough_cost = float(candidate_scores["hough_keypoints"]["selector_cost"])
    single_cost = float(candidate_scores["opencv_hough_lsd_regulation"]["selector_cost"])
    if _should_prefer_hough_over_collapsed_regulation(
        candidate_scores["hough_keypoints"],
        candidate_scores["opencv_hough_lsd_regulation"],
    ):
        selected_name = "hough_keypoints"
    else:
        selected_name = "hough_keypoints" if hough_cost <= single_cost else "opencv_hough_lsd_regulation"

    persistent_score = candidate_scores["opencv_hough_lsd_temporal_persistent_regulation"]
    persistent_risk = persistent_score["geometry_risk"]
    persistent_ratio = persistent_risk.get("far_to_near_width_ratio")
    persistent_supported = int(persistent_score.get("supported_line_count") or 0)
    single_supported = int(candidate_scores["opencv_hough_lsd_regulation"].get("supported_line_count") or 0)
    persistent_upgrade = (
        hough_cost > -60.0
        and persistent_supported >= single_supported
        and float(persistent_score["selector_cost"]) <= single_cost + 15.0
        and float(persistent_risk["risk_score"]) <= 5.0
        and (persistent_ratio is None or float(persistent_ratio) <= 1.08)
    )
    if persistent_upgrade:
        selected_name = "opencv_hough_lsd_temporal_persistent_regulation"

    selected = candidates[selected_name]
    payload = json.loads(json.dumps(selected))
    payload["solver"] = {
        "name": "hough_regulation_temporal_persistent_tail_selector",
        "version": 1,
        "writes_court_calibration": False,
    }
    payload["solver_confidence"] = min(float(payload.get("solver_confidence") or 0.0), 0.62)
    payload["needs_user_input"] = ["court_keypoints"]
    payload["needs_user_confirmation"] = True
    payload["selector"] = {
        "selected_technology_id": selected_name,
        "candidate_scores": candidate_scores,
        "rule": "start_from_tail_safe_hough_or_single_frame_then_upgrade_to_persistent_temporal_when_support_is_equal_and_geometry_risk_is_low",
    }
    payload.setdefault("notes", [])
    payload["notes"].append("persistent_tail_selector_proposal_only_never_writes_court_calibration_json")
    payload["notes"].append("persistent_temporal_candidate_requires_repeated_frame_support_and_low_projected_geometry_risk")
    for item in payload.get("keypoints", {}).values():
        if isinstance(item, dict):
            item["source"] = f"hough_regulation_temporal_persistent_tail_selector:{selected_name}"
    return payload


def _select_reviewed_oracle_hough_or_regulation_proposal(
    image_bgr: Any,
    *,
    sample: CourtFindingSample,
    proposal_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    hough = proposal_cache.get("hough_keypoints")
    if hough is None:
        hough = _proposal_for_technology(
            image_bgr,
            sample=sample,
            technology_id="hough_keypoints",
            proposal_cache=proposal_cache,
        )
        proposal_cache["hough_keypoints"] = hough
    regulation = proposal_cache.get("opencv_hough_lsd_regulation")
    if regulation is None:
        regulation = _proposal_for_technology(
            image_bgr,
            sample=sample,
            technology_id="opencv_hough_lsd_regulation",
            proposal_cache=proposal_cache,
        )
        proposal_cache["opencv_hough_lsd_regulation"] = regulation

    hough_score = _reviewed_floor_median_for_payload(sample=sample, proposal=hough)
    regulation_score = _reviewed_floor_median_for_payload(sample=sample, proposal=regulation)
    selected_name = "hough_keypoints" if hough_score <= regulation_score else "opencv_hough_lsd_regulation"
    selected = hough if selected_name == "hough_keypoints" else regulation
    payload = json.loads(json.dumps(selected))
    payload["solver"] = {
        "name": "reviewed_oracle_hough_or_regulation",
        "version": 1,
        "writes_court_calibration": False,
    }
    payload["solver_confidence"] = min(float(payload.get("solver_confidence") or 0.0), 0.62)
    payload["needs_user_input"] = ["court_keypoints"]
    payload["needs_user_confirmation"] = True
    payload["selector"] = {
        "selected_technology_id": selected_name,
        "candidate_scores": {
            "hough_keypoints": {"floor_visible_median_px": round(float(hough_score), 4)},
            "opencv_hough_lsd_regulation": {"floor_visible_median_px": round(float(regulation_score), 4)},
        },
        "rule": "reviewed_label_oracle_choose_lowest_floor_visible_median",
        "deployable": False,
    }
    payload.setdefault("notes", [])
    payload["notes"].append("reviewed_label_oracle_not_deployable")
    payload["notes"].append("benchmark_upper_bound_only_never_writes_court_calibration_json")
    for item in payload.get("keypoints", {}).values():
        if isinstance(item, dict):
            item["source"] = f"reviewed_oracle_hough_or_regulation:{selected_name}"
    return payload


def _select_reviewed_oracle_hough_regulation_temporal_proposal(
    image_bgr: Any,
    *,
    sample: CourtFindingSample,
    proposal_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for technology_id in (
        "hough_keypoints",
        "opencv_hough_lsd_regulation",
        "opencv_hough_lsd_temporal_regulation",
    ):
        proposal = proposal_cache.get(technology_id)
        if proposal is None:
            proposal = _proposal_for_technology(
                image_bgr,
                sample=sample,
                technology_id=technology_id,
                proposal_cache=proposal_cache,
            )
            proposal_cache[technology_id] = proposal
        candidates[technology_id] = proposal

    reviewed_scores = {
        technology_id: _reviewed_floor_median_for_payload(sample=sample, proposal=proposal)
        for technology_id, proposal in candidates.items()
    }
    selected_name = min(reviewed_scores, key=lambda name: float(reviewed_scores[name]))
    selected = candidates[selected_name]
    payload = json.loads(json.dumps(selected))
    payload["solver"] = {
        "name": "reviewed_oracle_hough_regulation_temporal",
        "version": 1,
        "writes_court_calibration": False,
    }
    payload["solver_confidence"] = min(float(payload.get("solver_confidence") or 0.0), 0.62)
    payload["needs_user_input"] = ["court_keypoints"]
    payload["needs_user_confirmation"] = True
    payload["selector"] = {
        "selected_technology_id": selected_name,
        "candidate_scores": {
            technology_id: {"floor_visible_median_px": round(float(value), 4)}
            for technology_id, value in reviewed_scores.items()
        },
        "rule": "reviewed_label_oracle_choose_lowest_floor_visible_median",
        "deployable": False,
    }
    payload.setdefault("notes", [])
    payload["notes"].append("reviewed_label_oracle_not_deployable")
    payload["notes"].append("benchmark_upper_bound_only_never_writes_court_calibration_json")
    for item in payload.get("keypoints", {}).values():
        if isinstance(item, dict):
            item["source"] = f"reviewed_oracle_hough_regulation_temporal:{selected_name}"
    return payload


def _reviewed_floor_median_for_payload(*, sample: CourtFindingSample, proposal: Mapping[str, Any]) -> float:
    with tempfile.TemporaryDirectory(prefix="court_finding_oracle_") as tmp:
        proposal_path = Path(tmp) / "proposal.json"
        _write_json(proposal_path, proposal)
        comparison = compare_proposal_to_reviewed_keypoints(
            reviewed_keypoints_path=sample.label_path,
            proposal_path=proposal_path,
        )
    floor = comparison.get("groups", {}).get("floor_visible", {})
    value = floor.get("median_px")
    return float(value) if value is not None else float("inf")


def _proposal_internal_line_score(
    proposal: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    keypoints = {
        str(name): tuple(float(value) for value in item["xy"])
        for name, item in (proposal.get("keypoints") or {}).items()
        if isinstance(item, Mapping) and isinstance(item.get("xy"), Sequence) and len(item["xy"]) == 2
    }
    line_scores = _score_projected_court_lines(keypoints, segments) if keypoints else {}
    supported = sum(1 for item in line_scores.values() if item.get("supported"))
    mean_cost = (
        999.0
        if not line_scores
        else sum(float(item["support_cost"]) for item in line_scores.values()) / len(line_scores)
    )
    collapsed_ladder_penalty = _collapsed_cross_ladder_penalty(proposal)
    selector_cost = mean_cost - supported * 8.0 + collapsed_ladder_penalty
    return {
        "supported_line_count": int(supported),
        "mean_projected_line_cost": round(float(mean_cost), 4),
        "collapsed_cross_ladder_penalty": round(float(collapsed_ladder_penalty), 4),
        "selector_cost": round(float(selector_cost), 4),
    }


def _proposal_geometry_risk_score(proposal: Mapping[str, Any]) -> dict[str, Any]:
    source = proposal.get("source") if isinstance(proposal.get("source"), Mapping) else {}
    image_size = source.get("image_size") if isinstance(source, Mapping) else None
    width = float(image_size[0]) if isinstance(image_size, Sequence) and len(image_size) == 2 else 0.0
    height = float(image_size[1]) if isinstance(image_size, Sequence) and len(image_size) == 2 else 0.0
    keypoints = {
        str(name): tuple(float(value) for value in item["xy"])
        for name, item in (proposal.get("keypoints") or {}).items()
        if isinstance(item, Mapping) and isinstance(item.get("xy"), Sequence) and len(item["xy"]) == 2
    }
    floor_names = (
        "near_left_corner",
        "near_right_corner",
        "far_left_corner",
        "far_right_corner",
        "near_nvz_left",
        "near_nvz_right",
        "far_nvz_left",
        "far_nvz_right",
    )
    floor_points = [keypoints[name] for name in floor_names if name in keypoints]
    if width <= 0.0 or height <= 0.0 or len(floor_points) < 4:
        return {
            "risk_score": 999.0,
            "eligible_for_temporal_selection": False,
            "reason": "insufficient_projected_floor_geometry",
        }
    xs = [point[0] for point in floor_points]
    ys = [point[1] for point in floor_points]
    max_outside = 0.0
    for x, y in floor_points:
        outside_x = max(0.0, -x, x - width)
        outside_y = max(0.0, -y, y - height)
        max_outside = max(max_outside, math.hypot(outside_x, outside_y))
    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)
    near_width = (
        math.dist(keypoints["near_left_corner"], keypoints["near_right_corner"])
        if "near_left_corner" in keypoints and "near_right_corner" in keypoints
        else 0.0
    )
    far_width = (
        math.dist(keypoints["far_left_corner"], keypoints["far_right_corner"])
        if "far_left_corner" in keypoints and "far_right_corner" in keypoints
        else 0.0
    )
    far_to_near = far_width / near_width if near_width > 1e-6 else float("inf")
    reasons: list[str] = []
    min_visible_width = max(140.0, width * 0.07)
    if bbox_height < max(55.0, height * 0.07):
        reasons.append("projected_floor_height_too_collapsed")
    if bbox_width < min_visible_width or near_width < min_visible_width:
        reasons.append("projected_floor_width_too_collapsed")
    if far_to_near > 1.05:
        reasons.append("far_width_larger_than_near_width")
    if max_outside > max(width, height) * 0.20:
        reasons.append("projected_floor_too_far_outside_image")
    risk_score = 0.0
    risk_score += max(0.0, max(55.0, height * 0.07) - bbox_height) * 1.5
    risk_score += max(0.0, min_visible_width - bbox_width) * 0.8
    risk_score += max(0.0, min_visible_width - near_width) * 0.8
    risk_score += max(0.0, far_to_near - 1.05) * 80.0
    risk_score += max(0.0, max_outside - max(width, height) * 0.20) * 0.25
    return {
        "risk_score": round(float(risk_score), 4),
        "eligible_for_temporal_selection": not reasons,
        "reasons": reasons,
        "bbox_width_px": round(float(bbox_width), 3),
        "bbox_height_px": round(float(bbox_height), 3),
        "max_outside_image_px": round(float(max_outside), 3),
        "near_width_px": round(float(near_width), 3),
        "far_width_px": round(float(far_width), 3),
        "far_to_near_width_ratio": round(float(far_to_near), 4) if math.isfinite(far_to_near) else None,
    }


def _score_has_geometry_reason(score: Mapping[str, Any], reason: str) -> bool:
    risk = score.get("geometry_risk")
    if not isinstance(risk, Mapping):
        return False
    reasons = risk.get("reasons")
    return isinstance(reasons, Sequence) and reason in reasons


def _should_prefer_hough_over_collapsed_regulation(
    hough_score: Mapping[str, Any],
    regulation_score: Mapping[str, Any],
) -> bool:
    if not (
        _score_has_geometry_reason(regulation_score, "projected_floor_height_too_collapsed")
        or _score_has_geometry_reason(regulation_score, "projected_floor_width_too_collapsed")
    ):
        return False
    if (
        _score_has_geometry_reason(hough_score, "projected_floor_height_too_collapsed")
        or _score_has_geometry_reason(hough_score, "projected_floor_width_too_collapsed")
    ):
        return False
    if int(hough_score.get("supported_line_count") or 0) < 3:
        return False
    hough_risk = hough_score.get("geometry_risk")
    regulation_risk = regulation_score.get("geometry_risk")
    hough_value = float(hough_risk.get("risk_score") or 0.0) if isinstance(hough_risk, Mapping) else 0.0
    regulation_value = (
        float(regulation_risk.get("risk_score") or 0.0)
        if isinstance(regulation_risk, Mapping)
        else float("inf")
    )
    return hough_value <= regulation_value


def _collapsed_cross_ladder_penalty(proposal: Mapping[str, Any]) -> float:
    assignment = proposal.get("line_assignment")
    if not isinstance(assignment, Mapping):
        return 0.0
    cross_names = ["far_baseline", "far_nvz", "net", "near_nvz", "near_baseline"]
    cross: list[tuple[str, float]] = []
    for name in cross_names:
        item = assignment.get(name)
        if not isinstance(item, Mapping):
            continue
        p1 = item.get("p1")
        p2 = item.get("p2")
        if not isinstance(p1, Sequence) or not isinstance(p2, Sequence) or len(p1) != 2 or len(p2) != 2:
            continue
        cross.append((name, (float(p1[1]) + float(p2[1])) / 2.0))
    if len(cross) < 5:
        return 0.0
    ordered = sorted(cross, key=lambda item: item[1])
    gaps = [ordered[index + 1][1] - ordered[index][1] for index in range(len(ordered) - 1)]
    min_gap = min(gaps) if gaps else float("inf")
    if min_gap >= 6.0:
        return 0.0
    # A full 5-line ladder with near-coincident observed cross lines usually
    # means tennis/gym/mesh evidence was forced into pickleball semantics.
    return 140.0 + (6.0 - min_gap) * 12.0


def _regulation_hypotheses_from_groups(
    groups: Sequence[_CandidateLine],
    *,
    segments: Sequence[Mapping[str, Any]],
    image_bgr: Any,
    image_size: tuple[int, int],
    image_evidence_mode: str,
    line_refinement: bool,
) -> list[dict[str, Any]]:
    width, height = image_size
    cross_pool = [
        group
        for group in groups
        if _angle_diff_mod_180(group.angle_deg, 0.0) <= 12.0 and group.support_length_px >= max(28.0, width * 0.04)
    ][:14]
    long_pool = [
        group
        for group in groups
        if _angle_diff_mod_180(group.angle_deg, 0.0) >= 18.0 and group.support_length_px >= max(28.0, min(width, height) * 0.045)
    ][:14]
    if len(cross_pool) < 3 or len(long_pool) < 2:
        return []

    cross_sets = (
        ("far_baseline", "far_nvz", "near_nvz", "near_baseline"),
        ("far_baseline", "far_nvz", "near_nvz"),
        ("far_nvz", "near_nvz", "near_baseline"),
        ("far_baseline", "far_nvz", "net", "near_nvz", "near_baseline"),
    )
    long_sets = (
        ("left_sideline", "centerline", "right_sideline"),
        ("left_sideline", "right_sideline"),
        ("centerline", "right_sideline"),
        ("left_sideline", "centerline"),
    )
    hypotheses: list[dict[str, Any]] = []
    y_ref = float(height) * 0.55
    x_ref = float(width) * 0.50
    for cross_labels in cross_sets:
        if len(cross_pool) < len(cross_labels):
            continue
        cross_assignments = _ranked_cross_assignments(cross_pool, cross_labels, x_ref=x_ref, limit=36)
        for cross_assignment in cross_assignments:
            for long_labels in long_sets:
                if len(long_pool) < len(long_labels):
                    continue
                long_assignments = _ranked_longitudinal_assignments(long_pool, long_labels, y_ref=y_ref, limit=18)
                for long_assignment in long_assignments:
                    line_assignment = {**cross_assignment, **long_assignment}
                    hypothesis = _hypothesis_from_line_assignment(
                        line_assignment,
                        segments=segments,
                        image_size=image_size,
                    )
                    if hypothesis is not None:
                        hypotheses.append(hypothesis)
    hypotheses.sort(key=lambda item: float(item["score"]))
    shortlist = hypotheses[:96]
    if line_refinement:
        refinement_line_mask = _court_line_pixel_mask(image_bgr, dilation_px=5)
        shortlist = [
            refine_regulation_homography_for_line_assignment(
                hypothesis,
                image_size=image_size,
                image_bgr=image_bgr,
                line_pixel_mask=refinement_line_mask,
            )
            for hypothesis in shortlist
        ]
    projected_line_mask = _court_line_pixel_mask(image_bgr, dilation_px=5)
    color_line_mask = _court_line_pixel_mask(image_bgr, dilation_px=2)
    if image_evidence_mode == "distance_mask":
        distance_line_mask = _court_line_pixel_mask(image_bgr, dilation_px=3)
        distance_map = _line_pixel_distance_transform(distance_line_mask)
    else:
        distance_line_mask = None
        distance_map = None
    enriched = [
        _hypothesis_with_image_evidence(
            hypothesis,
            image_bgr=image_bgr,
            projected_line_mask=projected_line_mask,
            color_line_mask=color_line_mask,
            distance_line_mask=distance_line_mask,
            distance_map=distance_map,
            image_evidence_mode=image_evidence_mode,
        )
        for hypothesis in shortlist
    ]
    enriched.sort(key=lambda item: float(item["score"]))
    return enriched[:32]


def _hypothesis_from_line_assignment(
    line_assignment: Mapping[str, _CandidateLine],
    *,
    segments: Sequence[Mapping[str, Any]],
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    width, height = image_size
    if len({id(group) for group in line_assignment.values()}) != len(line_assignment):
        return None
    world_points: list[list[float]] = []
    image_points: list[list[float]] = []
    for cross_name, cross_line in line_assignment.items():
        if _LINE_WORLD_COORDS.get(cross_name, (0.0, 0.0, ""))[2] != "y":
            continue
        world_y = _LINE_WORLD_COORDS[cross_name][0]
        for long_name, long_line in line_assignment.items():
            if _LINE_WORLD_COORDS.get(long_name, (0.0, 0.0, ""))[2] != "x":
                continue
            world_x = _LINE_WORLD_COORDS[long_name][0]
            try:
                xy = _line_intersection(cross_line.line, long_line.line)
            except ValueError:
                continue
            if not (_point_is_finite(xy) and _point_inside_loose_bounds(xy, width=width, height=height)):
                continue
            world_points.append([float(world_x), float(world_y), 0.0])
            image_points.append([float(xy[0]), float(xy[1])])
    if len(world_points) < 4:
        return None
    try:
        from .court_calibration import homography_from_planar_points, project_planar_points

        homography = homography_from_planar_points(world_points, image_points)
        projected = project_planar_points(
            homography,
            [[_FLOOR_WORLD_XY[name][0], _FLOOR_WORLD_XY[name][1], 0.0] for name in _FLOOR_WORLD_XY],
        )
    except Exception:
        return None
    keypoints = {
        name: (float(xy[0]), float(xy[1]))
        for name, xy in zip(_FLOOR_WORLD_XY, projected, strict=True)
    }
    if not _projected_court_is_plausible(keypoints, width=width, height=height):
        return None
    line_scores = _score_projected_court_lines(keypoints, segments)
    supported = sum(1 for item in line_scores.values() if item["supported"])
    mean_cost = sum(float(item["support_cost"]) for item in line_scores.values()) / max(1, len(line_scores))
    assignment_support = sum(group.support_length_px for group in line_assignment.values())
    correspondence_bonus = len(image_points) * 2.5
    support_bonus = supported * 28.0
    weak_line_penalty = (len(_FLOOR_LINE_ENDPOINT_KEYPOINTS) - supported) * 32.0
    template_competition = score_template_competition_for_line_assignment(
        line_assignment,
        image_size=image_size,
    )
    tennis_template_penalty = float(template_competition.get("tennis_template_penalty") or 0.0)
    score = (
        mean_cost
        + weak_line_penalty
        + tennis_template_penalty
        - support_bonus
        - correspondence_bonus
        - min(80.0, assignment_support / 40.0)
    )
    return {
        "score": float(score),
        "keypoints": keypoints,
        "supported_line_count": supported,
        "line_assignment": {
            name: {
                "p1": group.segment["p1"],
                "p2": group.segment["p2"],
                "angle_deg": round(float(group.angle_deg), 3),
                "support_length_px": round(float(group.support_length_px), 3),
                "source_segment_count": int(group.source_segment_count),
            }
            for name, group in sorted(line_assignment.items())
        },
        "score_components": {
            "mean_projected_line_cost": round(float(mean_cost), 4),
            "supported_line_count": int(supported),
            "assignment_support_px": round(float(assignment_support), 3),
            "correspondence_count": len(image_points),
            "weak_line_penalty": round(float(weak_line_penalty), 4),
            "tennis_template_penalty": round(float(tennis_template_penalty), 4),
            "template_competition": template_competition,
            "support_bonus": round(float(support_bonus), 4),
        },
    }


def refine_regulation_homography_for_line_assignment(
    hypothesis: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
    image_bgr: Any | None = None,
    line_pixel_mask: Any | None = None,
) -> dict[str, Any]:
    """Refine a regulation homography against assigned point and line evidence.

    This is benchmark/proposal-only. It does not promote calibration; it only
    updates the hypothesis when the optimized homography improves assigned-line
    residuals without making correspondence residuals or geometry plausibility
    worse.
    """

    payload = json.loads(json.dumps(hypothesis))
    line_assignment = payload.get("line_assignment")
    if not isinstance(line_assignment, Mapping):
        _set_line_refinement_metadata(payload, {"available": False, "reason": "missing_line_assignment"})
        return payload
    initial_keypoints = {
        str(name): tuple(float(value) for value in item)
        for name, item in (payload.get("keypoints") or {}).items()
        if _is_xy(item)
    }
    if len(initial_keypoints) < 4:
        _set_line_refinement_metadata(payload, {"available": False, "reason": "insufficient_initial_keypoints"})
        return payload

    observed_lines: dict[str, tuple[float, float, float]] = {}
    line_endpoints: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for label, raw_segment in line_assignment.items():
        if not isinstance(raw_segment, Mapping):
            continue
        endpoints = _world_line_endpoints_for_assignment(str(label))
        if endpoints is None:
            continue
        try:
            observed_lines[str(label)] = _line_from_segment_mapping(raw_segment)
        except ValueError:
            continue
        line_endpoints[str(label)] = endpoints
    if len(observed_lines) < 4:
        _set_line_refinement_metadata(payload, {"available": False, "reason": "insufficient_assigned_lines"})
        return payload

    correspondences = _line_assignment_world_image_correspondences(line_assignment, image_size=image_size)
    if len(correspondences) < 4:
        _set_line_refinement_metadata(payload, {"available": False, "reason": "insufficient_line_intersections"})
        return payload

    try:
        from scipy.optimize import least_squares

        from .court_calibration import homography_from_planar_points
    except Exception as exc:
        _set_line_refinement_metadata(
            payload,
            {"available": False, "reason": "scipy_unavailable", "error": str(exc)},
        )
        return payload

    try:
        world_points = [[_FLOOR_WORLD_XY[name][0], _FLOOR_WORLD_XY[name][1], 0.0] for name in initial_keypoints if name in _FLOOR_WORLD_XY]
        image_points = [initial_keypoints[name] for name in initial_keypoints if name in _FLOOR_WORLD_XY]
        initial_h = homography_from_planar_points(world_points, image_points)
    except Exception as exc:
        _set_line_refinement_metadata(
            payload,
            {"available": False, "reason": "initial_homography_failed", "error": str(exc)},
        )
        return payload

    initial_params = _homography_params(initial_h)

    def residuals(params: Sequence[float]) -> list[float]:
        values: list[float] = []
        for world_xy, observed_xy in correspondences:
            projected = _project_world_xy_with_params(params, world_xy)
            values.append((projected[0] - observed_xy[0]) / 18.0)
            values.append((projected[1] - observed_xy[1]) / 18.0)
        for label, observed_line in observed_lines.items():
            endpoints = line_endpoints[label]
            for world_xy in endpoints:
                projected = _project_world_xy_with_params(params, world_xy)
                values.append(_signed_point_line_distance(projected, observed_line) / 16.0)
        for name, initial_xy in initial_keypoints.items():
            world_xy = _FLOOR_WORLD_XY.get(name)
            if world_xy is None:
                continue
            projected = _project_world_xy_with_params(params, (world_xy[0], world_xy[1], 0.0))
            values.append((projected[0] - initial_xy[0]) / 80.0)
            values.append((projected[1] - initial_xy[1]) / 80.0)
        return values

    try:
        result = least_squares(
            residuals,
            initial_params,
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=160,
        )
    except Exception as exc:
        _set_line_refinement_metadata(
            payload,
            {"available": False, "reason": "least_squares_failed", "error": str(exc)},
        )
        return payload

    optimized_params = [float(value) for value in result.x]
    initial_line_rmse = _line_assignment_rmse_px(initial_params, observed_lines, line_endpoints)
    optimized_line_rmse = _line_assignment_rmse_px(optimized_params, observed_lines, line_endpoints)
    initial_point_rmse = _line_assignment_point_rmse_px(initial_params, correspondences)
    optimized_point_rmse = _line_assignment_point_rmse_px(optimized_params, correspondences)
    optimized_keypoints = {
        name: _project_world_xy_with_params(optimized_params, (xy[0], xy[1], 0.0))
        for name, xy in _FLOOR_WORLD_XY.items()
    }
    width, height = image_size
    plausible = _projected_court_is_plausible(optimized_keypoints, width=width, height=height)
    pixel_guard = _line_refinement_pixel_guard(
        image_bgr=image_bgr,
        line_pixel_mask=line_pixel_mask,
        initial_keypoints=initial_keypoints,
        optimized_keypoints=optimized_keypoints,
    )
    accepted = bool(
        result.success
        and plausible
        and optimized_line_rmse <= initial_line_rmse - 0.25
        and optimized_point_rmse <= max(initial_point_rmse * 1.15, initial_point_rmse + 4.0)
        and bool(pixel_guard["acceptable"])
    )
    improvement = max(0.0, initial_line_rmse - optimized_line_rmse)
    metadata = {
        "available": True,
        "method": "scipy_least_squares_point_line_homography",
        "accepted": accepted,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "assigned_line_count": int(len(observed_lines)),
        "intersection_correspondence_count": int(len(correspondences)),
        "initial_line_rmse_px": round(float(initial_line_rmse), 4),
        "optimized_line_rmse_px": round(float(optimized_line_rmse), 4),
        "initial_point_rmse_px": round(float(initial_point_rmse), 4),
        "optimized_point_rmse_px": round(float(optimized_point_rmse), 4),
        "line_rmse_improvement_px": round(float(improvement), 4),
        "geometry_plausible": bool(plausible),
        "pixel_guard": pixel_guard,
        "max_nfev": 160,
    }
    if accepted:
        payload["keypoints"] = {
            name: (float(xy[0]), float(xy[1]))
            for name, xy in optimized_keypoints.items()
        }
        bonus = min(45.0, improvement * 0.65)
        payload["score"] = float(payload.get("score") or 0.0) - bonus
        metadata["score_bonus"] = round(float(bonus), 4)
    else:
        metadata["rejection_reason"] = _line_refinement_rejection_reason(
            success=bool(result.success),
            plausible=plausible,
            initial_line_rmse=initial_line_rmse,
            optimized_line_rmse=optimized_line_rmse,
            initial_point_rmse=initial_point_rmse,
            optimized_point_rmse=optimized_point_rmse,
            pixel_guard=pixel_guard,
        )
        metadata["score_bonus"] = 0.0
    _set_line_refinement_metadata(payload, metadata)
    return payload


def _set_line_refinement_metadata(payload: dict[str, Any], metadata: Mapping[str, Any]) -> None:
    components = payload.setdefault("score_components", {})
    components["line_refinement"] = dict(metadata)


def _line_refinement_rejection_reason(
    *,
    success: bool,
    plausible: bool,
    initial_line_rmse: float,
    optimized_line_rmse: float,
    initial_point_rmse: float,
    optimized_point_rmse: float,
    pixel_guard: Mapping[str, Any],
) -> str:
    if not success:
        return "optimizer_unsuccessful"
    if not plausible:
        return "optimized_geometry_implausible"
    if optimized_line_rmse > initial_line_rmse - 0.25:
        return "line_residual_not_improved"
    if optimized_point_rmse > max(initial_point_rmse * 1.15, initial_point_rmse + 4.0):
        return "intersection_residual_too_much_worse"
    if not bool(pixel_guard.get("acceptable", True)):
        return "projected_pixel_support_worse"
    return "not_accepted"


def _line_refinement_pixel_guard(
    *,
    image_bgr: Any | None,
    line_pixel_mask: Any | None,
    initial_keypoints: Mapping[str, tuple[float, float]],
    optimized_keypoints: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    if image_bgr is None:
        return {
            "available": False,
            "acceptable": True,
            "reason": "image_not_provided",
        }
    initial_support = score_projected_line_pixels_against_image(
        image_bgr,
        initial_keypoints,
        line_pixel_mask=line_pixel_mask,
    )
    optimized_support = score_projected_line_pixels_against_image(
        image_bgr,
        optimized_keypoints,
        line_pixel_mask=line_pixel_mask,
    )
    initial_mean = float(initial_support.get("mean_line_pixel_support_ratio") or 0.0)
    optimized_mean = float(optimized_support.get("mean_line_pixel_support_ratio") or 0.0)
    initial_supported = int(initial_support.get("supported_line_pixel_count") or 0)
    optimized_supported = int(optimized_support.get("supported_line_pixel_count") or 0)
    mean_delta = optimized_mean - initial_mean
    supported_delta = optimized_supported - initial_supported
    acceptable = mean_delta >= -0.03 and supported_delta >= 0
    return {
        "available": True,
        "acceptable": bool(acceptable),
        "initial_mean_line_pixel_support_ratio": round(float(initial_mean), 4),
        "optimized_mean_line_pixel_support_ratio": round(float(optimized_mean), 4),
        "mean_line_pixel_support_delta": round(float(mean_delta), 4),
        "initial_supported_line_pixel_count": int(initial_supported),
        "optimized_supported_line_pixel_count": int(optimized_supported),
        "supported_line_pixel_delta": int(supported_delta),
    }


def _world_line_endpoints_for_assignment(
    label: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if label in {"near_baseline", "near_nvz", "net", "far_nvz", "far_baseline"}:
        world_y = float(_LINE_WORLD_COORDS[label][0])
        return ((-10.0, world_y, 0.0), (10.0, world_y, 0.0))
    if label in {"left_sideline", "centerline", "right_sideline"}:
        world_x = float(_LINE_WORLD_COORDS[label][0])
        return ((world_x, -22.0, 0.0), (world_x, 22.0, 0.0))
    return None


def _line_assignment_world_image_correspondences(
    line_assignment: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
) -> list[tuple[tuple[float, float, float], tuple[float, float]]]:
    width, height = image_size
    cross_lines: dict[str, tuple[float, float, float]] = {}
    longitudinal_lines: dict[str, tuple[float, float, float]] = {}
    for label, raw_segment in line_assignment.items():
        if not isinstance(raw_segment, Mapping) or label not in _LINE_WORLD_COORDS:
            continue
        try:
            line = _line_from_segment_mapping(raw_segment)
        except ValueError:
            continue
        axis = _LINE_WORLD_COORDS[str(label)][2]
        if axis == "y":
            cross_lines[str(label)] = line
        elif axis == "x":
            longitudinal_lines[str(label)] = line
    correspondences: list[tuple[tuple[float, float, float], tuple[float, float]]] = []
    for cross_name, cross_line in cross_lines.items():
        world_y = float(_LINE_WORLD_COORDS[cross_name][0])
        for long_name, long_line in longitudinal_lines.items():
            world_x = float(_LINE_WORLD_COORDS[long_name][0])
            try:
                xy = _line_intersection(cross_line, long_line)
            except ValueError:
                continue
            if _point_is_finite(xy) and _point_inside_loose_bounds(xy, width=width, height=height):
                correspondences.append(((world_x, world_y, 0.0), (float(xy[0]), float(xy[1]))))
    return correspondences


def _homography_params(homography: Sequence[Sequence[float]]) -> list[float]:
    return [
        float(homography[0][0]),
        float(homography[0][1]),
        float(homography[0][2]),
        float(homography[1][0]),
        float(homography[1][1]),
        float(homography[1][2]),
        float(homography[2][0]),
        float(homography[2][1]),
    ]


def _project_world_xy_with_params(
    params: Sequence[float],
    world_xy: Sequence[float],
) -> tuple[float, float]:
    x = float(world_xy[0])
    y = float(world_xy[1])
    u_num = float(params[0]) * x + float(params[1]) * y + float(params[2])
    v_num = float(params[3]) * x + float(params[4]) * y + float(params[5])
    scale = float(params[6]) * x + float(params[7]) * y + 1.0
    if abs(scale) <= 1e-9:
        return (float("inf"), float("inf"))
    return (float(u_num / scale), float(v_num / scale))


def _signed_point_line_distance(point: Sequence[float], line: tuple[float, float, float]) -> float:
    return float(line[0] * float(point[0]) + line[1] * float(point[1]) + line[2])


def _line_assignment_rmse_px(
    params: Sequence[float],
    observed_lines: Mapping[str, tuple[float, float, float]],
    line_endpoints: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> float:
    distances: list[float] = []
    for label, line in observed_lines.items():
        endpoints = line_endpoints[label]
        for world_xy in endpoints:
            projected = _project_world_xy_with_params(params, world_xy)
            distances.append(_signed_point_line_distance(projected, line))
    return _rmse(distances)


def _line_assignment_point_rmse_px(
    params: Sequence[float],
    correspondences: Sequence[tuple[tuple[float, float, float], tuple[float, float]]],
) -> float:
    residuals: list[float] = []
    for world_xy, observed_xy in correspondences:
        projected = _project_world_xy_with_params(params, world_xy)
        residuals.append(projected[0] - observed_xy[0])
        residuals.append(projected[1] - observed_xy[1])
    return _rmse(residuals)


def _hypothesis_with_image_evidence(
    hypothesis: Mapping[str, Any],
    *,
    image_bgr: Any,
    projected_line_mask: Any,
    color_line_mask: Any,
    distance_line_mask: Any | None,
    distance_map: Any | None,
    image_evidence_mode: str,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(hypothesis))
    keypoints = {
        str(name): tuple(float(value) for value in xy)
        for name, xy in payload.get("keypoints", {}).items()
        if _is_xy(xy)
    }
    projected_pixel_support = score_projected_line_pixels_against_image(
        image_bgr,
        keypoints,
        line_pixel_mask=projected_line_mask,
    )
    pixel_mean_support = float(projected_pixel_support.get("mean_line_pixel_support_ratio") or 0.0)
    pixel_supported = int(projected_pixel_support.get("supported_line_pixel_count") or 0)
    projected_pixel_penalty = max(0.0, 0.38 - pixel_mean_support) * 145.0
    projected_pixel_penalty += max(0, 5 - pixel_supported) * 7.0
    line_color_consistency = score_line_color_consistency_for_assignment(
        image_bgr,
        payload.get("line_assignment") or {},
        line_pixel_mask=color_line_mask,
    )
    color_layer_penalty = min(45.0, float(line_color_consistency.get("mixed_layer_penalty") or 0.0) * 0.35)
    pixel_support_bonus = pixel_supported * 7.5 + pixel_mean_support * 26.0
    base_score = float(payload.get("score") or 0.0)
    distance_support: dict[str, Any] | None = None
    distance_penalty = 0.0
    distance_support_bonus = 0.0
    if image_evidence_mode == "distance_mask":
        if distance_line_mask is None or distance_map is None:
            raise ValueError("distance_mask scoring requires a line mask and distance map")
        distance_support = score_projected_line_distance_transform_against_image(
            image_bgr,
            keypoints,
            line_pixel_mask=distance_line_mask,
            distance_map=distance_map,
        )
        distance_mean = float(distance_support.get("mean_projected_line_distance_px") or 0.0)
        distance_p95 = float(distance_support.get("p95_projected_line_distance_px") or 0.0)
        distance_supported = int(distance_support.get("distance_supported_line_count") or 0)
        distance_penalty = max(0.0, distance_mean - 4.0) * 5.5 + max(0.0, distance_p95 - 12.0) * 1.2
        distance_penalty += max(0, 5 - distance_supported) * 9.0
        distance_support_bonus = distance_supported * 9.0 + max(0.0, 8.0 - distance_mean) * 2.5
    payload["score"] = float(
        base_score
        + projected_pixel_penalty
        + color_layer_penalty
        + distance_penalty
        - pixel_support_bonus
        - distance_support_bonus
    )
    components = payload.setdefault("score_components", {})
    components["image_evidence_mode"] = image_evidence_mode
    components["base_score_before_image_evidence"] = round(float(base_score), 4)
    components["projected_pixel_penalty"] = round(float(projected_pixel_penalty), 4)
    components["pixel_support_bonus"] = round(float(pixel_support_bonus), 4)
    components["color_layer_penalty"] = round(float(color_layer_penalty), 4)
    components["distance_penalty"] = round(float(distance_penalty), 4)
    components["distance_support_bonus"] = round(float(distance_support_bonus), 4)
    components["projected_pixel_support"] = projected_pixel_support
    components["line_color_consistency"] = line_color_consistency
    if distance_support is not None:
        components["projected_distance_support"] = distance_support
    return payload


def _score_projected_court_lines(
    keypoints: Mapping[str, tuple[float, float]],
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for line_name, (p1_name, p2_name) in _FLOOR_LINE_ENDPOINT_KEYPOINTS.items():
        p1 = keypoints[p1_name]
        p2 = keypoints[p2_name]
        best = _best_segment_support_for_line(p1, p2, segments)
        supported = bool(
            best is not None
            and best["angle_diff_deg"] <= 13.0
            and best["mean_perpendicular_distance_px"] <= 18.0
            and best["overlap_fraction"] >= 0.06
        )
        scores[line_name] = {
            "supported": supported,
            "best_segment": best,
            "support_cost": 250.0 if best is None else float(best["support_cost"]),
        }
    return scores


def _candidate_line_groups(
    segments: Sequence[Mapping[str, Any]],
    *,
    width: float,
    height: float,
) -> list[_CandidateLine]:
    groups: list[list[Mapping[str, Any]]] = []
    for segment in sorted(segments, key=lambda item: float(item.get("length_px") or 0.0), reverse=True):
        try:
            line = _line_from_segment_mapping(segment)
        except ValueError:
            continue
        assigned = False
        for group in groups:
            reference = _line_from_segment_mapping(group[0])
            if (
                _angle_diff_mod_180(float(segment["angle_deg"]), float(group[0]["angle_deg"])) <= 4.0
                and _line_segment_distance(line, group[0]) <= max(10.0, min(width, height) * 0.018)
            ):
                group.append(segment)
                assigned = True
                break
        if not assigned:
            groups.append([segment])
    merged: list[_CandidateLine] = []
    for group in groups:
        reference = max(group, key=lambda item: float(item.get("length_px") or 0.0))
        line = _line_from_segment_mapping(reference)
        support = sum(float(item.get("length_px") or 0.0) for item in group)
        merged.append(
            _CandidateLine(
                line=line,
                segment=dict(reference),
                support_length_px=float(support),
                source_segment_count=len(group),
                angle_deg=float(reference["angle_deg"]),
            )
        )
    return sorted(merged, key=lambda item: item.support_length_px, reverse=True)


def _ranked_cross_assignments(
    values: Sequence[_CandidateLine],
    labels: Sequence[str],
    *,
    x_ref: float,
    limit: int,
) -> list[dict[str, _CandidateLine]]:
    from itertools import combinations

    ranked: list[tuple[float, dict[str, _CandidateLine]]] = []
    expected_positions = [_LINE_WORLD_COORDS[label][0] for label in labels]
    expected_distances = [abs(expected_positions[index + 1] - expected_positions[index]) for index in range(len(labels) - 1)]
    for combo in combinations(values, len(labels)):
        ordered = sorted(combo, key=lambda group: _line_y_at_x(group.line, x_ref, fallback=_segment_midpoint(group.segment)[1]))
        positions = [_line_y_at_x(group.line, x_ref, fallback=_segment_midpoint(group.segment)[1]) for group in ordered]
        distances = [positions[index + 1] - positions[index] for index in range(len(positions) - 1)]
        if any(distance <= 10.0 for distance in distances):
            continue
        spacing_error = _spacing_error(distances, expected_distances)
        support = sum(group.support_length_px for group in ordered)
        score = spacing_error * 100.0 - min(60.0, support / 45.0)
        ranked.append((score, dict(zip(labels, ordered, strict=True))))
    ranked.sort(key=lambda item: item[0])
    return [assignment for _, assignment in ranked[:limit]]


def _ranked_longitudinal_assignments(
    values: Sequence[_CandidateLine],
    labels: Sequence[str],
    *,
    y_ref: float,
    limit: int,
) -> list[dict[str, _CandidateLine]]:
    from itertools import combinations

    ranked: list[tuple[float, dict[str, _CandidateLine]]] = []
    expected_positions = [_LINE_WORLD_COORDS[label][0] for label in labels]
    expected_distances = [abs(expected_positions[index + 1] - expected_positions[index]) for index in range(len(labels) - 1)]
    for combo in combinations(values, len(labels)):
        ordered = sorted(combo, key=lambda group: _line_x_at_y(group.line, y_ref, fallback=_segment_midpoint(group.segment)[0]))
        positions = [_line_x_at_y(group.line, y_ref, fallback=_segment_midpoint(group.segment)[0]) for group in ordered]
        distances = [positions[index + 1] - positions[index] for index in range(len(positions) - 1)]
        if any(distance <= 12.0 for distance in distances):
            continue
        spacing_error = 0.0 if not expected_distances else _spacing_error(distances, expected_distances)
        support = sum(group.support_length_px for group in ordered)
        separation_bonus = min(50.0, sum(distances) / 18.0)
        score = spacing_error * 80.0 - min(60.0, support / 45.0) - separation_bonus
        ranked.append((score, dict(zip(labels, ordered, strict=True))))
    ranked.sort(key=lambda item: item[0])
    return [assignment for _, assignment in ranked[:limit]]


def _spacing_error(observed: Sequence[float], expected: Sequence[float]) -> float:
    if len(observed) != len(expected) or not observed:
        return 0.0
    denom = sum(value * value for value in expected)
    if denom <= 1e-6:
        return 0.0
    scale = sum(float(obs) * float(exp) for obs, exp in zip(observed, expected, strict=True)) / denom
    if scale <= 1e-6:
        return float("inf")
    return sum(abs(float(obs) - scale * float(exp)) / max(1.0, scale * float(exp)) for obs, exp in zip(observed, expected, strict=True)) / len(observed)


def _projected_court_is_plausible(
    keypoints: Mapping[str, tuple[float, float]],
    *,
    width: int,
    height: int,
) -> bool:
    margin = max(width, height) * 0.40
    for xy in keypoints.values():
        if not _point_is_finite(xy):
            return False
        if xy[0] < -margin or xy[0] > width + margin or xy[1] < -margin or xy[1] > height + margin:
            return False
    far_y = (keypoints["far_left_corner"][1] + keypoints["far_right_corner"][1]) / 2.0
    near_nvz_y = (keypoints["near_nvz_left"][1] + keypoints["near_nvz_right"][1]) / 2.0
    near_y = (keypoints["near_left_corner"][1] + keypoints["near_right_corner"][1]) / 2.0
    if not far_y < near_nvz_y < near_y:
        return False
    near_width = math.dist(keypoints["near_left_corner"], keypoints["near_right_corner"])
    far_width = math.dist(keypoints["far_left_corner"], keypoints["far_right_corner"])
    if near_width < max(40.0, width * 0.08) or far_width < max(20.0, width * 0.03):
        return False
    return True


def _opencv_lsd_segments(image_bgr: Any) -> list[dict[str, Any]]:
    import cv2  # type: ignore[import-not-found]

    gray = _gray(image_bgr)
    detector = cv2.createLineSegmentDetector()
    raw = detector.detect(gray)[0]
    if raw is None:
        return []
    segments: list[dict[str, Any]] = []
    for line in raw.reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in line]
        item = _segment_item(x1, y1, x2, y2, source="opencv_lsd")
        if item["length_px"] >= 20.0:
            segments.append(item)
    segments.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return segments[:96]


def _retag_segments_source(segments: Sequence[Mapping[str, Any]], source: str) -> list[dict[str, Any]]:
    retagged: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        item["source"] = source
        retagged.append(item)
    return retagged


def _opencv_hough_segments(image_bgr: Any) -> list[dict[str, Any]]:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    gray = _gray(image_bgr)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 145)
    _height, width = gray.shape[:2]
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=38,
        minLineLength=max(24, int(round(width * 0.045))),
        maxLineGap=max(10, int(round(width * 0.018))),
    )
    segments: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in normalize_hough_lines_p(raw):
        item = _segment_item(x1, y1, x2, y2, source="opencv_hough")
        if item["length_px"] >= 20.0:
            segments.append(item)
    segments.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return segments[:96]


def _opencv_fast_line_detector_segments(image_bgr: Any) -> list[dict[str, Any]]:
    import cv2  # type: ignore[import-not-found]

    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "createFastLineDetector"):
        raise AttributeError("cv2.ximgproc.createFastLineDetector is unavailable")
    gray = _gray(image_bgr)
    height, width = gray.shape[:2]
    detector = cv2.ximgproc.createFastLineDetector(
        length_threshold=max(24, int(round(width * 0.04))),
        distance_threshold=1.414213562,
        canny_th1=45.0,
        canny_th2=145.0,
        canny_aperture_size=3,
        do_merge=True,
    )
    raw = detector.detect(gray)
    if raw is None:
        return []
    segments: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        item = _segment_item(float(x1), float(y1), float(x2), float(y2), source="opencv_fast_line_detector")
        if item["length_px"] >= 20.0:
            segments.append(item)
    segments.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return segments[:96]


def _skimage_probabilistic_hough_segments(image_bgr: Any) -> list[dict[str, Any]]:
    import numpy as np
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    gray = _gray(image_bgr)
    height, width = gray.shape[:2]
    edges = canny(
        gray.astype(np.float32) / 255.0,
        sigma=1.6,
        low_threshold=0.08,
        high_threshold=0.24,
    )
    raw = probabilistic_hough_line(
        edges,
        threshold=12,
        line_length=max(24, int(round(width * 0.04))),
        line_gap=max(8, int(round(width * 0.016))),
        rng=0,
    )
    segments: list[dict[str, Any]] = []
    for (x1, y1), (x2, y2) in raw:
        item = _segment_item(float(x1), float(y1), float(x2), float(y2), source="skimage_probabilistic_hough")
        if item["length_px"] >= 20.0:
            segments.append(item)
    segments.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return segments[:96]


def _elsed_line_candidate_evidence(image_bgr: Any) -> dict[str, Any]:
    try:
        import pyelsed  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:
        return {
            "technology_id": "elsed",
            "available": False,
            "candidate_count": 0,
            "segments": [],
            "reason": "pyelsed_unavailable",
            "error": str(exc),
            "install_hint": "pip install git+https://github.com/iago-suarez/ELSED.git",
        }

    gray = _gray(image_bgr)
    detected = pyelsed.detect(gray)
    if not isinstance(detected, tuple) or not detected:
        raw_segments = detected
        scores = []
    else:
        raw_segments = detected[0]
        scores = detected[1] if len(detected) > 1 else []
    array = np.asarray(raw_segments, dtype=float)
    if array.size == 0:
        return {
            "technology_id": "elsed",
            "available": True,
            "candidate_count": 0,
            "segments": [],
        }
    segments: list[dict[str, Any]] = []
    score_array = np.asarray([], dtype=float)
    if scores is not None:
        raw_scores = np.asarray(scores, dtype=float)
        if raw_scores.size:
            score_array = raw_scores.reshape(-1)
    for index, line in enumerate(array.reshape(-1, 4)):
        x1, y1, x2, y2 = [float(value) for value in line]
        item = _segment_item(x1, y1, x2, y2, source="elsed")
        if item["length_px"] < 20.0:
            continue
        if index < len(score_array):
            item["score"] = round(float(score_array[index]), 6)
        segments.append(item)
    segments.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return {
        "technology_id": "elsed",
        "available": True,
        "candidate_count": len(segments[:128]),
        "segments": segments[:128],
    }


def _gray(image_bgr: Any) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        raise ValueError("image_bgr must be an image array")
    if len(image_bgr.shape) == 2:
        return image_bgr.astype(np.uint8)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _segment_item(x1: float, y1: float, x2: float, y2: float, *, source: str) -> dict[str, Any]:
    dx = x2 - x1
    dy = y2 - y1
    return {
        "p1": [round(float(x1), 3), round(float(y1), 3)],
        "p2": [round(float(x2), 3), round(float(y2), 3)],
        "length_px": round(float(math.hypot(dx, dy)), 3),
        "angle_deg": round(float(math.degrees(math.atan2(dy, dx))), 3),
        "source": source,
    }


def _dedupe_segments(segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: float(item["length_px"]), reverse=True):
        mid = _segment_midpoint(segment)
        angle = float(segment["angle_deg"])
        duplicate = False
        for existing in selected:
            existing_mid = _segment_midpoint(existing)
            if math.dist(mid, existing_mid) <= 8.0 and abs(angle - float(existing["angle_deg"])) <= 4.0:
                duplicate = True
                break
        if not duplicate:
            selected.append(dict(segment))
        if len(selected) >= 128:
            break
    return selected


def _segment_midpoint(segment: Mapping[str, Any]) -> tuple[float, float]:
    p1 = segment["p1"]
    p2 = segment["p2"]
    return ((float(p1[0]) + float(p2[0])) / 2.0, (float(p1[1]) + float(p2[1])) / 2.0)


def _best_segment_support_for_line(
    reviewed_p1: tuple[float, float],
    reviewed_p2: tuple[float, float],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    rx1, ry1 = reviewed_p1
    rx2, ry2 = reviewed_p2
    rdx = rx2 - rx1
    rdy = ry2 - ry1
    r_len = math.hypot(rdx, rdy)
    if r_len <= 1e-6:
        return None
    r_angle = math.degrees(math.atan2(rdy, rdx))
    best: dict[str, Any] | None = None
    best_cost = float("inf")
    for candidate in candidates:
        p1 = candidate.get("p1")
        p2 = candidate.get("p2")
        if not isinstance(p1, Sequence) or not isinstance(p2, Sequence) or len(p1) != 2 or len(p2) != 2:
            continue
        c1 = (float(p1[0]), float(p1[1]))
        c2 = (float(p2[0]), float(p2[1]))
        c_len = math.dist(c1, c2)
        if c_len <= 1e-6:
            continue
        c_angle = math.degrees(math.atan2(c2[1] - c1[1], c2[0] - c1[0]))
        angle_diff = _angle_diff_mod_180(r_angle, c_angle)
        mean_distance = (_point_line_distance(c1, reviewed_p1, reviewed_p2) + _point_line_distance(c2, reviewed_p1, reviewed_p2)) / 2.0
        overlap_fraction = _segment_overlap_fraction_along_line(c1, c2, reviewed_p1, reviewed_p2)
        midpoint = ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)
        midpoint_t = _projection_t(midpoint, reviewed_p1, reviewed_p2)
        outside_penalty = max(0.0, -midpoint_t, midpoint_t - 1.0) * 100.0
        cost = mean_distance + angle_diff * 2.0 - overlap_fraction * 20.0 + outside_penalty
        if cost < best_cost:
            best_cost = cost
            best = {
                "p1": [round(c1[0], 3), round(c1[1], 3)],
                "p2": [round(c2[0], 3), round(c2[1], 3)],
                "source": candidate.get("source"),
                "length_px": candidate.get("length_px"),
                "angle_diff_deg": round(angle_diff, 3),
                "mean_perpendicular_distance_px": round(mean_distance, 3),
                "overlap_fraction": round(overlap_fraction, 4),
                "midpoint_t": round(midpoint_t, 4),
                "support_cost": round(cost, 3),
            }
    return best


def _line_from_segment_mapping(segment: Mapping[str, Any]) -> tuple[float, float, float]:
    p1 = segment.get("p1")
    p2 = segment.get("p2")
    if not isinstance(p1, Sequence) or not isinstance(p2, Sequence) or len(p1) != 2 or len(p2) != 2:
        raise ValueError("segment must contain p1 and p2")
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)
    if norm <= 1e-6:
        raise ValueError("zero-length segment")
    a, b, c = a / norm, b / norm, c / norm
    if a < 0.0 or (abs(a) <= 1e-9 and b < 0.0):
        a, b, c = -a, -b, -c
    return (float(a), float(b), float(c))


def _line_segment_distance(line: tuple[float, float, float], segment: Mapping[str, Any]) -> float:
    p1 = segment.get("p1")
    p2 = segment.get("p2")
    if not isinstance(p1, Sequence) or not isinstance(p2, Sequence) or len(p1) != 2 or len(p2) != 2:
        return float("inf")
    a, b, c = line
    return (
        abs(a * float(p1[0]) + b * float(p1[1]) + c)
        + abs(a * float(p2[0]) + b * float(p2[1]) + c)
    ) / 2.0


def _line_intersection(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float]:
    a1, b1, c1 = first
    a2, b2, c2 = second
    det = a1 * b2 - a2 * b1
    if abs(det) <= 1e-6:
        raise ValueError("parallel lines do not intersect")
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return (float(x), float(y))


def _line_y_at_x(line: tuple[float, float, float], x: float, *, fallback: float) -> float:
    a, b, c = line
    if abs(b) <= 1e-6:
        return float(fallback)
    return float((-a * x - c) / b)


def _line_x_at_y(line: tuple[float, float, float], y: float, *, fallback: float) -> float:
    a, b, c = line
    if abs(a) <= 1e-6:
        return float(fallback)
    return float((-b * y - c) / a)


def _point_is_finite(point: tuple[float, float]) -> bool:
    return math.isfinite(float(point[0])) and math.isfinite(float(point[1]))


def _point_inside_loose_bounds(point: tuple[float, float], *, width: int, height: int) -> bool:
    margin = max(width, height) * 0.50
    return -margin <= point[0] <= width + margin and -margin <= point[1] <= height + margin


def _is_xy(value: Any) -> bool:
    return isinstance(value, Sequence) and len(value) == 2


def _court_line_pixel_mask(image_bgr: Any, *, dilation_px: int) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if len(image_bgr.shape) == 2:
        bgr = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        bgr = image_bgr.astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    lightness = lab[:, :, 0]
    local_value = cv2.GaussianBlur(value, (0, 0), 7.0)
    local_lightness = cv2.GaussianBlur(lightness, (0, 0), 7.0)
    bright_or_colored = (
        ((value.astype(np.int16) - local_value.astype(np.int16)) >= 24)
        | ((lightness.astype(np.int16) - local_lightness.astype(np.int16)) >= 22)
        | ((value >= 165) & (saturation >= 45))
        | ((value >= 205) & (saturation <= 70))
    )
    mask = bright_or_colored.astype(np.uint8) * 255
    if dilation_px > 1:
        kernel_size = max(3, int(dilation_px) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _line_pixel_distance_transform(mask: Any) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    line_pixels = (mask > 0).astype(np.uint8)
    inverse = (1 - line_pixels).astype(np.uint8) * 255
    return cv2.distanceTransform(inverse, cv2.DIST_L2, 3)


def _sample_points_on_segment(
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    spacing_px: float,
    min_count: int,
    max_count: int,
) -> list[tuple[float, float]]:
    length = math.dist(p1, p2)
    if length <= 1e-6:
        return [p1]
    count = max(min_count, min(max_count, int(math.ceil(length / max(1.0, spacing_px))) + 1))
    return [
        (
            p1[0] + (p2[0] - p1[0]) * index / float(count - 1),
            p1[1] + (p2[1] - p1[1]) * index / float(count - 1),
        )
        for index in range(count)
    ]


def _sample_line_pixels(
    image_bgr: Any,
    mask: Any,
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> list[tuple[float, float, float]]:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    pixels: list[tuple[float, float, float]] = []
    for x, y in _sample_points_on_segment(p1, p2, spacing_px=7.0, min_count=10, max_count=72):
        ix = int(round(x))
        iy = int(round(y))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                px = ix + dx
                py = iy + dy
                if 0 <= px < width and 0 <= py < height and int(mask[py, px]) > 0:
                    value = image_bgr[py, px]
                    if len(image_bgr.shape) == 2:
                        scalar = float(value)
                        pixels.append((scalar, scalar, scalar))
                    else:
                        pixels.append((float(value[0]), float(value[1]), float(value[2])))
    return pixels[:512]


def _line_color_cluster(mean_bgr: Sequence[float]) -> str:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    bgr = np.array([[[int(round(max(0.0, min(255.0, float(value))))) for value in mean_bgr]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    hue, saturation, value = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if value >= 175 and saturation <= 42:
        return "white"
    if value < 85:
        return "dark"
    hue_bin = int(round(hue / 10.0) * 10) % 180
    sat_bin = "high_sat" if saturation >= 70 else "low_sat"
    return f"hue_{hue_bin:03d}_{sat_bin}"


def _scale_line_candidates_to_native(
    *,
    line_candidates: Sequence[Mapping[str, Any]],
    candidate_image_size: tuple[int, int] | None,
    native_image_size: tuple[float, float],
) -> list[dict[str, Any]]:
    if candidate_image_size is None:
        scale_x = 1.0
        scale_y = 1.0
    else:
        width, height = candidate_image_size
        native_w, native_h = native_image_size
        scale_x = native_w / float(width) if width > 0 else 1.0
        scale_y = native_h / float(height) if height > 0 else 1.0
    scaled: list[dict[str, Any]] = []
    for candidate in line_candidates:
        p1 = candidate.get("p1")
        p2 = candidate.get("p2")
        if not isinstance(p1, Sequence) or not isinstance(p2, Sequence) or len(p1) != 2 or len(p2) != 2:
            continue
        item = dict(candidate)
        item["p1"] = [float(p1[0]) * scale_x, float(p1[1]) * scale_y]
        item["p2"] = [float(p2[0]) * scale_x, float(p2[1]) * scale_y]
        item["length_px"] = round(math.dist(tuple(item["p1"]), tuple(item["p2"])), 3)
        item["coordinate_space"] = "native_reviewed_px"
        item["candidate_to_native_scale"] = [round(scale_x, 6), round(scale_y, 6)]
        scaled.append(item)
    return scaled


def _point_line_distance(point: tuple[float, float], line_p1: tuple[float, float], line_p2: tuple[float, float]) -> float:
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    return numerator / denominator if denominator > 1e-6 else float("inf")


def _projection_t(point: tuple[float, float], line_p1: tuple[float, float], line_p2: tuple[float, float]) -> float:
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-6:
        return 0.0
    return ((px - x1) * dx + (py - y1) * dy) / denom


def _segment_overlap_fraction_along_line(
    c1: tuple[float, float],
    c2: tuple[float, float],
    reviewed_p1: tuple[float, float],
    reviewed_p2: tuple[float, float],
) -> float:
    t1 = _projection_t(c1, reviewed_p1, reviewed_p2)
    t2 = _projection_t(c2, reviewed_p1, reviewed_p2)
    low = max(0.0, min(t1, t2))
    high = min(1.0, max(t1, t2))
    return max(0.0, high - low)


def _angle_diff_mod_180(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def _summary(
    *,
    samples: Sequence[CourtFindingSample],
    technologies: Sequence[str],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scored = [result for result in results if result.get("scored")]
    by_technology: dict[str, dict[str, Any]] = {}
    for technology_id in technologies:
        tech_scored = [result for result in scored if result.get("technology_id") == technology_id]
        medians = [
            float(result["floor_visible_median_px"])
            for result in tech_scored
            if result.get("floor_visible_median_px") is not None
        ]
        p95s = [
            float(result["floor_visible_p95_px"])
            for result in tech_scored
            if result.get("floor_visible_p95_px") is not None
        ]
        line_support_ratios = [
            float(result["line_support_ratio"])
            for result in results
            if result.get("technology_id") == technology_id and result.get("line_support_ratio") is not None
        ]
        by_technology[str(technology_id)] = {
            "scored_clip_count": len(tech_scored),
            "floor_visible_median_px_mean": None if not medians else round(sum(medians) / len(medians), 4),
            "floor_visible_median_px_median": None if not medians else round(_median(medians), 4),
            "floor_visible_p95_px_mean": None if not p95s else round(sum(p95s) / len(p95s), 4),
            "floor_visible_p95_px_max": None if not p95s else round(max(p95s), 4),
            "line_support_ratio_mean": (
                None
                if not line_support_ratios
                else round(sum(line_support_ratios) / len(line_support_ratios), 4)
            ),
        }
    ranked = sorted(
        (
            {"technology_id": tech, **metrics}
            for tech, metrics in by_technology.items()
            if metrics["floor_visible_median_px_mean"] is not None
        ),
        key=lambda item: float(item["floor_visible_median_px_mean"]),
    )
    return {
        "sample_count": len(samples),
        "full_15pt_sample_count": sum(1 for sample in samples if sample.label_kind == "full_15pt"),
        "partial_visible_sample_count": sum(1 for sample in samples if sample.label_kind == "partial_visible"),
        "technology_ids": [str(value) for value in technologies],
        "result_count": len(results),
        "scored_result_count": len(scored),
        "by_technology": by_technology,
        "ranked_scored_technologies": ranked,
        "best_scored_technology_id": ranked[0]["technology_id"] if ranked else None,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _rmse(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(value) * float(value) for value in values) / float(len(values)))


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (max(0.0, min(100.0, float(percentile))) / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
