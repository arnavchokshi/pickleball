#!/usr/bin/env python3
"""Extract quarantined pb.vision court pseudo-labels and geometry evidence.

The exporter contains twelve anonymous, normalized court points.  This tool
discovers their 4-by-3 projective grid from repeated collinearity, assigns the
twelve planar pickleball keypoints, measures full-set and leave-one-out
homography residuals, and emits only the manager-approved corpus videos.

The resulting labels are intentionally ``PENDING_SPOTCHECK`` and are not a
training authorization.  The shared row loader now rejects the top-level
pending status, ``training_eligibility.queued=false``, and pending item status
unless an explicit diagnostic-only Python opt-in is supplied.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_templates import get_court_template


PROVENANCE = "pbvision_cv_export_rev190"
PENDING_STATUS = "PENDING_SPOTCHECK"
LOCAL_VIDEO_ID = "83gyqyc10y8f"
COMPARE_ONLY_IDS = frozenset({LOCAL_VIDEO_ID, "iottnc0h3ekn", "o4dee9dn0ccr"})
DROPPED_CORPUS_IDS = frozenset({"bewqc0glhgpq"})
CANONICAL_NAMES = tuple(point.name for point in PICKLEBALL_KEYPOINTS)
CANONICAL_WORLD_XY = {point.name: tuple(float(v) for v in point.world_xyz_m[:2]) for point in PICKLEBALL_KEYPOINTS}

# The semantic row order is fixed only after structured assignment has found
# the anonymous index grid.  The two planar reflections have identical
# reprojection residuals; the declared convention resolves them by increasing
# image u (left -> right) and the majority end-court depth order (far -> near).
GROUND_NAME_GRID: tuple[tuple[str, str, str], ...] = (
    ("far_left_corner", "far_baseline_center", "far_right_corner"),
    ("far_nvz_left", "far_nvz_center", "far_nvz_right"),
    ("near_nvz_left", "near_nvz_center", "near_nvz_right"),
    ("near_left_corner", "near_baseline_center", "near_right_corner"),
)
GROUND_NAMES = tuple(name for row in GROUND_NAME_GRID for name in row)


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    title: str
    cv_export_path: Path
    width: int
    height: int
    fps: float
    pbv_tick_rate: float
    duration_s: float
    frame_count: int
    video_sha256: str
    origin: str
    source_video: Path | None


@dataclass(frozen=True)
class MappingCandidate:
    groups: tuple[tuple[int, int, int], ...]
    index_to_name: tuple[str, ...]
    score: float


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ffprobe(path: Path, ffprobe_bin: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream in {path}")
    stream = streams[0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
    fps = float(numerator) / float(denominator)
    duration = float(stream.get("duration") or 0.0)
    frame_count = int(stream.get("nb_frames") or round(duration * fps))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration_s": duration,
        "frame_count": frame_count,
    }


def _manifest_rows(manifest_path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(manifest_path)
    rows = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("manifest.videos must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("video_id"), str):
            raise ValueError("every manifest video needs a video_id")
        video_id = row["video_id"]
        if video_id in result:
            raise ValueError(f"duplicate manifest video_id {video_id}")
        result[video_id] = row
    return result


def _pbv_tick_rate(export_path: Path) -> float:
    payload = _read_json(export_path)
    camera = payload.get("camera") if isinstance(payload, dict) else None
    raw_rate = camera.get("fps") if isinstance(camera, dict) else None
    if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float)) or float(raw_rate) <= 0:
        raise ValueError(f"camera.fps must be a positive PBV segment tick rate in {export_path}")
    return float(raw_rate)


def load_records(
    *,
    gallery_root: Path,
    local_root: Path,
    manifest_path: Path,
    local_video_id: str,
    ffprobe_bin: str,
) -> list[VideoRecord]:
    """Load only permitted manifest identities.

    The local arguments remain accepted for command compatibility, but they do not determine
    eligibility and are never accessed. Compare-only identity is fixed by provenance policy and
    rejected from the already-open manifest before any source path is constructed, stated,
    opened, parsed, hashed, or probed.
    """

    del local_root, local_video_id, ffprobe_bin
    manifest = _manifest_rows(manifest_path)
    records: list[VideoRecord] = []
    for video_id, manifest_row in sorted(manifest.items()):
        if video_id in COMPARE_ONLY_IDS:
            continue
        video_root = gallery_root / video_id
        export_path = video_root / "cv_export.json"
        metadata_path = video_root / "api_get_metadata.json"
        metadata_payload = _read_json(metadata_path)
        metadata = metadata_payload.get("metadata") if isinstance(metadata_payload, dict) else None
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata block missing for {video_id}")
        fps = float(metadata["fps"])
        duration_s = float(metadata["secs"])
        records.append(
            VideoRecord(
                video_id=video_id,
                title=str(manifest_row["title"]),
                cv_export_path=export_path,
                width=int(metadata["width"]),
                height=int(metadata["height"]),
                fps=fps,
                pbv_tick_rate=_pbv_tick_rate(export_path),
                duration_s=duration_s,
                frame_count=int(round(duration_s * fps)),
                video_sha256=str(manifest_row["video_sha256"]),
                origin="gallery",
                source_video=None,
            )
        )
    return records


def _camera_segments(record: VideoRecord) -> list[dict[str, Any]]:
    if record.video_id in COMPARE_ONLY_IDS:
        raise ValueError(
            f"compare-only identity {record.video_id!r} is rejected before source access"
        )
    payload = _read_json(record.cv_export_path)
    camera = payload.get("camera") if isinstance(payload, dict) else None
    segments = camera.get("cameraSegments") if isinstance(camera, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{record.video_id} has no cameraSegments")
    for segment_index, segment in enumerate(segments):
        points = segment.get("court_points") if isinstance(segment, dict) else None
        if not isinstance(points, list) or len(points) != 12:
            raise ValueError(f"{record.video_id} segment {segment_index} must contain 12 court_points slots")
        if sum(point is not None for point in points) < 8:
            raise ValueError(f"{record.video_id} segment {segment_index} has fewer than 8 usable court points")
    return segments


def _normalized_xy(point: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(point["u"]), float(point["v"])], dtype=np.float64)


def _tls_line_rms(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    distances = centered @ vh[-1]
    return float(np.sqrt(np.mean(np.square(distances))))


def _triple_line_score(triple: tuple[int, int, int], segments: list[list[dict[str, Any] | None]]) -> float:
    scores: list[float] = []
    for points in segments:
        if all(points[index] is not None for index in triple):
            scores.append(_tls_line_rms(np.stack([_normalized_xy(points[index]) for index in triple])))  # type: ignore[arg-type]
    return float(np.median(scores)) if scores else math.inf


def _exact_covers(triples: list[tuple[int, int, int]]) -> list[tuple[tuple[int, int, int], ...]]:
    covers: list[tuple[tuple[int, int, int], ...]] = []

    def visit(start: int, chosen: list[tuple[int, int, int]], used: frozenset[int]) -> None:
        if len(chosen) == 4:
            if len(used) == 12:
                covers.append(tuple(chosen))
            return
        for index in range(start, len(triples)):
            triple = triples[index]
            if used.isdisjoint(triple):
                visit(index + 1, chosen + [triple], used | frozenset(triple))

    visit(0, [], frozenset())
    return covers


def _median_coordinate(
    point_index: int,
    axis: int,
    segments: list[list[dict[str, Any] | None]],
) -> float:
    values = [_normalized_xy(points[point_index])[axis] for points in segments if points[point_index] is not None]
    if not values:
        raise ValueError(f"court point index {point_index} is null in every segment")
    return float(np.median(values))


def _mapping_from_cover(
    cover: tuple[tuple[int, int, int], ...],
    segments: list[list[dict[str, Any] | None]],
) -> tuple[tuple[tuple[int, int, int], ...], tuple[str, ...]]:
    # Projective rows remain collinear.  Across the 13-video batch, increasing
    # median image v is the majority far-to-near order; within each row,
    # increasing median u is the declared image-left-to-right convention.
    ordered_groups = sorted(
        cover,
        key=lambda triple: float(np.median([_median_coordinate(index, 1, segments) for index in triple])),
    )
    groups = tuple(
        tuple(sorted(triple, key=lambda index: _median_coordinate(index, 0, segments)))
        for triple in ordered_groups
    )
    names: list[str | None] = [None] * 12
    for group_index, group in enumerate(groups):
        for column_index, point_index in enumerate(group):
            names[point_index] = GROUND_NAME_GRID[group_index][column_index]
    if any(name is None for name in names):
        raise AssertionError("structured cover did not assign all twelve indices")
    return groups, tuple(str(name) for name in names)


def _fit_homography(world: np.ndarray, image: np.ndarray, method: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    homography, mask = cv2.findHomography(world, image, method, 3.0)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("homography fit failed")
    return homography.astype(np.float64), mask


def _project(homography: np.ndarray, world: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(world.reshape(-1, 1, 2).astype(np.float64), homography).reshape(-1, 2)


def _candidate_score_for_segments(
    index_to_name: tuple[str, ...],
    segments: list[list[dict[str, Any] | None]],
) -> float:
    medians: list[float] = []
    for points in segments:
        indices = [index for index, point in enumerate(points) if point is not None]
        world = np.asarray([CANONICAL_WORLD_XY[index_to_name[index]] for index in indices], dtype=np.float64)
        image = np.asarray([_normalized_xy(points[index]) for index in indices], dtype=np.float64)  # type: ignore[arg-type]
        homography, _ = _fit_homography(world, image)
        medians.append(float(np.median(np.linalg.norm(_project(homography, world) - image, axis=1))))
    return float(sum(medians))


def _candidate_score(index_to_name: tuple[str, ...], records: list[VideoRecord]) -> float:
    return _candidate_score_for_segments(
        index_to_name,
        [segment["court_points"] for record in records for segment in _camera_segments(record)],
    )


def discover_mapping(records: list[VideoRecord]) -> tuple[MappingCandidate, dict[str, Any]]:
    segments = [segment["court_points"] for record in records for segment in _camera_segments(record)]
    ranked_triples = sorted(
        (_triple_line_score(triple, segments), triple)
        for triple in itertools.combinations(range(12), 3)
    )
    covers: list[tuple[tuple[int, int, int], ...]] = []
    candidates: list[MappingCandidate] = []
    shortlist_size = 0
    for shortlist_size in (20, 32, 48, 80, 140, len(ranked_triples)):
        covers = _exact_covers([triple for _, triple in ranked_triples[:shortlist_size]])
        candidates = []
        for cover in covers:
            groups, mapping = _mapping_from_cover(cover, segments)
            candidates.append(
                MappingCandidate(groups=groups, index_to_name=mapping, score=_candidate_score_for_segments(mapping, segments))
            )
        candidates.sort(key=lambda candidate: (candidate.score, candidate.index_to_name))
        # Do not stop merely because a few top-ranked line triples happen to
        # form an exact cover.  The side-view export has several misleading
        # near-collinear triples; expand until one structured grid explains
        # the points to within 3% of normalized image coordinates.
        if candidates and candidates[0].score / len(segments) <= 0.03:
            break
    if not candidates:
        raise ValueError("no four-row exact cover found for the twelve court point indices")
    winner = candidates[0]
    details = {
        "method": "aggregate TLS collinearity shortlist -> four disjoint triples -> structured homography assignment",
        "full_permutation_space_avoided": "12!",
        "triple_combinations_scored": len(ranked_triples),
        "collinearity_shortlist_size": shortlist_size,
        "exact_covers_scored": len(covers),
        "candidate_score_sum_normalized_median_residual": winner.score,
        "reflection_ambiguity": (
            "A planar homography cannot distinguish global left-right or near-far reflection by residual. "
            "Convention: increasing aggregate image u is left-to-right; increasing aggregate image v is far-to-near."
        ),
        "groups_by_far_to_near": [list(group) for group in winner.groups],
        "ranked_line_triples": [
            {"indices": list(triple), "aggregate_tls_rms_normalized": score}
            for score, triple in ranked_triples[:shortlist_size]
        ],
    }
    return winner, details


def validate_per_video_assignments(
    records: list[VideoRecord],
    batch_mapping: MappingCandidate,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    disagreements: list[str] = []
    complete_independent = 0
    consistent_count = 0
    for record in records:
        segments = _camera_segments(record)
        point_sets = [segment["court_points"] for segment in segments]
        missing_indices = sorted(
            index
            for index in range(12)
            if all(segment["court_points"][index] is None for segment in segments)
        )
        if not missing_indices:
            complete_independent += 1

        # The batch step discovers the four anonymous collinear index groups.
        # Per video, score every admissible semantic assignment of those four
        # rows (4!) and the common left/right orientation (2): 48 candidates,
        # rather than re-opening an unbounded point-partition search or 12!.
        candidates: list[tuple[float, tuple[str, ...], tuple[int, ...], bool]] = []
        for row_permutation in itertools.permutations(range(4)):
            for reverse_columns in (False, True):
                names: list[str | None] = [None] * 12
                for observed_row, group in enumerate(batch_mapping.groups):
                    canonical_row = GROUND_NAME_GRID[row_permutation[observed_row]]
                    if reverse_columns:
                        canonical_row = tuple(reversed(canonical_row))
                    for column, point_index in enumerate(group):
                        names[point_index] = canonical_row[column]
                candidate_mapping = tuple(str(name) for name in names)
                candidates.append(
                    (
                        _candidate_score_for_segments(candidate_mapping, point_sets),
                        candidate_mapping,
                        row_permutation,
                        reverse_columns,
                    )
                )
        candidates.sort(key=lambda row: (row[0], row[1]))
        minimum_score = candidates[0][0]
        batch_score = _candidate_score_for_segments(batch_mapping.index_to_name, point_sets)
        # Reflections of a planar rectangle are mathematically tied.  The
        # declared batch convention wins a tie; disagreement means a genuinely
        # different row permutation lowers residual beyond numerical noise.
        tolerance = max(1e-10, minimum_score * 1e-8)
        consistent = batch_score <= minimum_score + tolerance
        status = (
            "supports_batch_mapping_with_missing_export_slot"
            if consistent and missing_indices
            else "identical_up_to_planar_reflection"
            if consistent
            else "disagrees"
        )
        local_mapping = list(batch_mapping.index_to_name if consistent else candidates[0][1])
        groups = [list(group) for group in batch_mapping.groups]
        tied_count = sum(candidate[0] <= minimum_score + tolerance for candidate in candidates)
        if consistent:
            consistent_count += 1
        else:
            disagreements.append(record.video_id)
        rows.append(
            {
                "video_id": record.video_id,
                "status": status,
                "consistent_with_batch_mapping": consistent,
                "missing_indices": missing_indices,
                "groups_by_far_to_near": groups,
                "index_to_name": local_mapping,
                "batch_candidate_score_sum_normalized_median_residual": batch_score,
                "minimum_candidate_score_sum_normalized_median_residual": minimum_score,
                "reflection_tied_minimum_count": tied_count,
                "structured_candidates_scored": len(candidates),
            }
        )
    return {
        "consistent_video_count": consistent_count,
        "video_count": len(records),
        "complete_independent_assignment_count": complete_independent,
        "consistency_fraction": consistent_count / len(records),
        "disagreements": disagreements,
        "videos": rows,
    }


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def analyze_segment(
    record: VideoRecord,
    segment_index: int,
    segment: dict[str, Any],
    index_to_name: tuple[str, ...],
) -> dict[str, Any]:
    points = segment["court_points"]
    indices = [index for index, point in enumerate(points) if point is not None]
    world = np.asarray([CANONICAL_WORLD_XY[index_to_name[index]] for index in indices], dtype=np.float64)
    image = np.asarray(
        [[float(points[index]["u"]) * record.width, float(points[index]["v"]) * record.height] for index in indices],
        dtype=np.float64,
    )
    full_h, _ = _fit_homography(world, image)
    projected = _project(full_h, world)
    residuals = np.linalg.norm(projected - image, axis=1)
    _, ransac_mask = _fit_homography(world, image, cv2.RANSAC)

    loo_by_index: dict[int, float] = {}
    for held_position, held_index in enumerate(indices):
        keep = np.ones(len(indices), dtype=bool)
        keep[held_position] = False
        loo_h, _ = _fit_homography(world[keep], image[keep])
        loo_prediction = _project(loo_h, world[held_position : held_position + 1])[0]
        loo_by_index[held_index] = float(np.linalg.norm(loo_prediction - image[held_position]))

    residual_by_index = {point_index: float(residuals[position]) for position, point_index in enumerate(indices)}
    rows: list[dict[str, Any]] = []
    for point_index, name in enumerate(index_to_name):
        point = points[point_index]
        rows.append(
            {
                "index": point_index,
                "canonical_name": name,
                "uv_px": None
                if point is None
                else [float(point["u"]) * record.width, float(point["v"]) * record.height],
                "confidence": None if point is None else float(point["confidence"]),
                "spread": None if point is None else float(point["spread"]),
                "full_set_residual_px": residual_by_index.get(point_index),
                "leave_one_out_residual_px": loo_by_index.get(point_index),
            }
        )
    valid_rows = [row for row in rows if row["uv_px"] is not None]
    worst_loo = max(valid_rows, key=lambda row: float(row["leave_one_out_residual_px"]))
    return {
        "segment_index": segment_index,
        "frame_start": int(segment["s"]),
        "frame_end": int(segment["e"]),
        "usable_point_count": len(indices),
        "missing_indices": [index for index, point in enumerate(points) if point is None],
        "homography_world_xy_to_native_px": full_h.tolist(),
        "opencv_method0_all_point_residual_px": _summary(float(value) for value in residuals),
        "all_point_residual_estimator": {
            "implementation": "cv2.findHomography",
            "method": 0,
            "fit": "all usable points with OpenCV method-0 refinement",
            "accuracy_claim": "geometric self-consistency only; not independent ground truth",
        },
        "ransac_threshold_px": 3.0,
        "ransac_inlier_count": None if ransac_mask is None else int(ransac_mask.sum()),
        "leave_one_out_residual_px": _summary(loo_by_index.values()),
        "leave_one_out_worst_point": {
            "index": worst_loo["index"],
            "canonical_name": worst_loo["canonical_name"],
            "residual_px": worst_loo["leave_one_out_residual_px"],
        },
        "confidence": _summary(float(row["confidence"]) for row in valid_rows),
        "spread_normalized": _summary(float(row["spread"]) for row in valid_rows),
        "points": rows,
    }


def analyze_records(records: list[VideoRecord], index_to_name: tuple[str, ...]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        segments = [
            analyze_segment(record, segment_index, segment, index_to_name)
            for segment_index, segment in enumerate(_camera_segments(record))
        ]
        all_residuals = [
            float(point["full_set_residual_px"])
            for segment in segments
            for point in segment["points"]
            if point["full_set_residual_px"] is not None
        ]
        all_loo = [
            float(point["leave_one_out_residual_px"])
            for segment in segments
            for point in segment["points"]
            if point["leave_one_out_residual_px"] is not None
        ]
        results.append(
            {
                "video_id": record.video_id,
                "title": record.title,
                "origin": record.origin,
                "width": record.width,
                "height": record.height,
                "source_fps": record.fps,
                "pbv_tick_rate": record.pbv_tick_rate,
                "segment_count": len(segments),
                "full_set_residual_px": _summary(all_residuals),
                "leave_one_out_residual_px": _summary(all_loo),
                "leave_one_out_worst_point": max(
                    (
                        segment["leave_one_out_worst_point"] | {"segment_index": segment["segment_index"]}
                        for segment in segments
                    ),
                    key=lambda row: float(row["residual_px"]),
                ),
                "confidence": _summary(
                    float(point["confidence"])
                    for segment in segments
                    for point in segment["points"]
                    if point["confidence"] is not None
                ),
                "spread_normalized": _summary(
                    float(point["spread"])
                    for segment in segments
                    for point in segment["points"]
                    if point["spread"] is not None
                ),
                "segments": segments,
            }
        )
    return results


def pbv_tick_to_native_frame(
    tick: float,
    *,
    pbv_tick_rate: float,
    source_fps: float,
    frame_count: int | None = None,
) -> int:
    """Convert a PBV camera-segment tick to a native source-frame index."""

    if pbv_tick_rate <= 0 or source_fps <= 0:
        raise ValueError("pbv_tick_rate and source_fps must be positive")
    native = int(round(float(tick) * float(source_fps) / float(pbv_tick_rate)))
    if frame_count is not None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        native = min(native, frame_count - 1)
    return max(native, 0)


def choose_target_frames(
    start: int,
    end: int,
    pbv_tick_rate: float,
    count: int,
    *,
    source_fps: float | None = None,
    frame_count: int | None = None,
) -> list[int]:
    """Sample PBV segment ticks, then explicitly convert them to native frame indices."""

    source_fps = float(pbv_tick_rate if source_fps is None else source_fps)
    low = int(math.ceil(start + 2.0 * pbv_tick_rate))
    high = int(math.floor(end - 2.0 * pbv_tick_rate))
    if high < low:
        low, high = int(start), int(end)
    available = high - low + 1
    target_count = min(count, available)
    sampled_ticks = [int(round(value)) for value in np.linspace(low, high, target_count)]
    native_frames = sorted(
        {
            pbv_tick_to_native_frame(
                tick,
                pbv_tick_rate=pbv_tick_rate,
                source_fps=source_fps,
                frame_count=frame_count,
            )
            for tick in sampled_ticks
        }
    )
    if len(native_frames) != target_count:
        raise ValueError(
            f"PBV tick conversion collapsed {target_count} requested samples to {len(native_frames)} native frames"
        )
    return native_frames


def build_timebase_audit(
    records: list[VideoRecord],
    *,
    ffprobe_bin: str,
) -> dict[str, Any]:
    """Audit the ten original corpus candidates with one PBV-tick/source-fps probe."""

    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.video_id):
        if record.video_id in COMPARE_ONLY_IDS:
            raise ValueError(
                f"compare-only identity {record.video_id!r} reached the permitted timebase audit"
            )
        if record.origin != "gallery":
            continue
        segments = _camera_segments(record)
        max_tick = max(int(segment["e"]) for segment in segments)
        conversion_applied = not math.isclose(record.pbv_tick_rate, record.fps, rel_tol=0.0, abs_tol=1e-9)
        row: dict[str, Any] = {
            "video": record.video_id,
            "pbv_tick_rate": record.pbv_tick_rate,
            "pbv_endpoint_tick_rate": max_tick / record.duration_s,
            "source_fps": record.fps,
            "source_duration_s_api": record.duration_s,
            "conversion_factor": record.fps / record.pbv_tick_rate,
            "conversion_applied": conversion_applied,
            "source_probe": "api_get_metadata.json",
        }
        local_max_mp4 = ROOT / "data/pbv_replay_20260720" / record.video_id / "max.mp4"
        if local_max_mp4.is_file():
            local_probe = _ffprobe(local_max_mp4, ffprobe_bin)
            row["source_probe"] = "local max.mp4 ffprobe + api_get_metadata.json cap"
            row["local_max_mp4"] = str(local_max_mp4)
            row["local_ffprobe"] = local_probe
        rows.append(row)
    expected_count = sum(record.origin == "gallery" for record in records)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} original corpus candidates in timebase audit, found {len(rows)}")
    return {
        "schema_version": 1,
        "artifact_type": "pbvision_court_source_timebase_audit",
        "native_frame_formula": "round(pbv_tick * source_fps / pbv_tick_rate), clamped to API metadata frame_count - 1",
        "videos": rows,
    }


def write_timebase_audit(lane_dir: Path, audit: dict[str, Any]) -> None:
    _write_json(lane_dir / "fps_timebase_audit.json", audit)
    lines = [
        "| video | pbv_tick_rate | source_fps | conversion applied | factor | source probe |",
        "|---|---:|---:|:---:|---:|---|",
    ]
    for row in audit["videos"]:
        lines.append(
            f"| {row['video']} | {row['pbv_tick_rate']:.6f} | {row['source_fps']:.6f} | "
            f"{'yes' if row['conversion_applied'] else 'no'} | {row['conversion_factor']:.6f} | "
            f"{row['source_probe']} |"
        )
    (lane_dir / "fps_timebase_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (lane_dir / "fps_timebase_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _keypoint_payload(
    record: VideoRecord,
    segment: dict[str, Any],
    index_to_name: tuple[str, ...],
) -> tuple[dict[str, list[float] | None], dict[str, float | None], dict[str, float | None]]:
    keypoints: dict[str, list[float] | None] = {name: None for name in CANONICAL_NAMES}
    confidence: dict[str, float | None] = {name: None for name in CANONICAL_NAMES}
    spread: dict[str, float | None] = {name: None for name in CANONICAL_NAMES}
    for point_index, name in enumerate(index_to_name):
        point = segment["court_points"][point_index]
        if point is not None:
            keypoints[name] = [float(point["u"]) * record.width, float(point["v"]) * record.height]
            confidence[name] = float(point["confidence"])
            spread[name] = float(point["spread"])
    return keypoints, confidence, spread


def emit_corpus(
    *,
    records: list[VideoRecord],
    index_to_name: tuple[str, ...],
    output_root: Path,
    dropped_corpus_ids: frozenset[str],
    frames_per_segment: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    emitted_videos = 0
    emitted_segments = 0
    emitted_rows = 0
    target_frames = 0
    by_video: dict[str, Any] = {}
    for record in records:
        if record.video_id in COMPARE_ONLY_IDS:
            raise ValueError(
                f"compare-only identity {record.video_id!r} reached corpus emission"
            )
        eligible = (
            record.origin == "gallery"
            and record.video_id not in dropped_corpus_ids
        )
        if not eligible:
            if record.video_id in dropped_corpus_ids:
                stale_video_root = output_root / record.video_id
                if stale_video_root.exists():
                    shutil.rmtree(stale_video_root)
            continue
        segments = _camera_segments(record)
        frame_indices: list[int] = []
        items: list[dict[str, Any]] = []
        for segment_index, segment in enumerate(segments):
            targets = choose_target_frames(
                int(segment["s"]),
                int(segment["e"]),
                record.pbv_tick_rate,
                frames_per_segment,
                source_fps=record.fps,
                frame_count=record.frame_count,
            )
            frame_indices.extend(targets)
            representative = targets[len(targets) // 2]
            keypoints, confidence, spread = _keypoint_payload(record, segment, index_to_name)
            items.append(
                {
                    "frame": f"frame_{representative:06d}.jpg",
                    # Legacy loader enum only.  The shared eligibility gate separately
                    # enforces all pending pseudo-label markers below.
                    "status": "reviewed_external_dataset",
                    "pseudo_label_status": PENDING_STATUS,
                    "keypoints": keypoints,
                    "keypoint_confidence": confidence,
                    "keypoint_spread_normalized": spread,
                    "camera_segment": {
                        "index": segment_index,
                        "pbv_tick_start": int(segment["s"]),
                        "pbv_tick_end": int(segment["e"]),
                        "pbv_tick_rate": record.pbv_tick_rate,
                        "native_frame_start": pbv_tick_to_native_frame(
                            int(segment["s"]),
                            pbv_tick_rate=record.pbv_tick_rate,
                            source_fps=record.fps,
                            frame_count=record.frame_count,
                        ),
                        "native_frame_end": pbv_tick_to_native_frame(
                            int(segment["e"]),
                            pbv_tick_rate=record.pbv_tick_rate,
                            source_fps=record.fps,
                            frame_count=record.frame_count,
                        ),
                        "source_fps": record.fps,
                        "conversion_applied": not math.isclose(
                            record.pbv_tick_rate,
                            record.fps,
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        ),
                    },
                    "provenance": PROVENANCE,
                }
            )
        frame_indices = sorted(set(frame_indices))
        relative_frame_dir = Path(record.video_id) / "frames"
        labels_payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_court_keypoint_labels",
            "clip": record.video_id,
            "status": PENDING_STATUS,
            "provenance": PROVENANCE,
            "training_eligibility": {
                "queued": False,
                "reason": "mandatory owner/Fable spot-check has not ruled these pseudo-labels usable",
            },
            "annotation": {"items": items},
            "frames": {
                "available_review_frame_count": 0,
                "frame_count": len(items),
                "frame_dir": relative_frame_dir.as_posix(),
                "path_base": "corpus_root",
                "label_coordinate_space": [record.width, record.height],
                "source_resolution": [record.width, record.height],
            },
            "review": {
                "status": "reviewed",
                "reviewer": "LOADER_COMPATIBILITY_SENTINEL_NOT_A_HUMAN_REVIEW",
                "independent_reviewed_count": 0,
                "pending_spotcheck_count": len(items),
                "note": (
                    "Structural sentinel required by the legacy review enum. The shared loader default-deny "
                    "gate enforces top-level status=PENDING_SPOTCHECK, training_eligibility.queued=false, "
                    "and item pseudo_label_status=PENDING_SPOTCHECK."
                ),
            },
        }
        video_root = output_root / record.video_id
        _write_json(video_root / "labels" / "court_keypoints.json", labels_payload)
        _write_json(
            video_root / "frames_needed.json",
            {
                "video_id": record.video_id,
                "gcs_url": f"https://storage.googleapis.com/pbv-pro/{record.video_id}/max.mp4",
                "video_sha256": record.video_sha256,
                "fps": record.fps,
                "pbv_tick_rate": record.pbv_tick_rate,
                "timebase_conversion_applied": not math.isclose(
                    record.pbv_tick_rate,
                    record.fps,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                "timebase_conversion": "native_frame=round(pbv_tick*source_fps/pbv_tick_rate)",
                "width": record.width,
                "height": record.height,
                "frame_indices": frame_indices,
                "frame_dir": relative_frame_dir.as_posix(),
            },
        )
        emitted_videos += 1
        emitted_segments += len(segments)
        emitted_rows += len(items)
        target_frames += len(frame_indices)
        by_video[record.video_id] = {
            "segments": len(segments),
            "rows": len(items),
            "target_frames": len(frame_indices),
        }

    families: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: item.video_id):
        corpus_eligible = (
            record.origin == "gallery"
            and record.video_id not in dropped_corpus_ids
        )
        family: dict[str, Any] = {
            "family": f"pbv_{record.video_id}",
            "title": record.title,
            "corpus_eligible": corpus_eligible,
        }
        if record.video_id in dropped_corpus_ids:
            family["reason"] = (
                "dropped_entire_video: bad multi-point planar solve; no single-point filtering; "
                "re-admission requires rectification or manual reannotation plus stable robust evidence"
            )
        families[record.video_id] = family
    _write_json(output_root / "families.json", families)
    for video_id in COMPARE_ONLY_IDS | dropped_corpus_ids:
        if (output_root / video_id).exists():
            raise AssertionError(f"ineligible video leaked into corpus directory: {video_id}")
    return {
        "videos_emitted": emitted_videos,
        "segments_emitted": emitted_segments,
        "rows_emitted": emitted_rows,
        "target_frames_listed": target_frames,
        "compare_only_rows_emitted": 0,
        "dropped_corpus_rows_emitted": 0,
        "dropped_corpus_ids": sorted(dropped_corpus_ids),
        "families": len(families),
        "by_video": by_video,
    }


def _extract_spotcheck_frames(
    *,
    video_path: Path,
    frame_count: int,
    frame_indices: list[int],
    raw_dir: Path,
    ffmpeg_bin: str,
) -> list[Path]:
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)
    expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select={expression}",
            "-fps_mode",
            "vfr",
            "-start_number",
            "0",
            str(raw_dir / "raw_%02d.jpg"),
        ],
        check=True,
    )
    paths = sorted(raw_dir.glob("raw_*.jpg"))
    if len(paths) != len(frame_indices):
        raise ValueError(
            f"ffmpeg extracted {len(paths)} spot-check frames, expected {len(frame_indices)} from {frame_count} frames"
        )
    return paths


def _project_one(homography: np.ndarray, xy: tuple[float, float]) -> tuple[int, int]:
    value = homography @ np.asarray([xy[0], xy[1], 1.0], dtype=np.float64)
    return int(round(value[0] / value[2])), int(round(value[1] / value[2]))


def write_spotcheck(
    *,
    record: VideoRecord,
    analysis: dict[str, Any],
    lane_dir: Path,
    count: int,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    if record.video_id in COMPARE_ONLY_IDS:
        raise ValueError(
            f"compare-only identity {record.video_id!r} is rejected before source access"
        )
    if record.source_video is None:
        raise ValueError("spot-check record has no local source video")
    frame_indices = [int(round(value)) for value in np.linspace(0, record.frame_count - 1, count)]
    spotcheck_dir = lane_dir / "spotcheck_11min"
    spotcheck_dir.mkdir(parents=True, exist_ok=True)
    for old_path in spotcheck_dir.glob("*.jpg"):
        old_path.unlink()
    raw_dir = lane_dir / "_spotcheck_raw"
    raw_paths = _extract_spotcheck_frames(
        video_path=record.source_video,
        frame_count=record.frame_count,
        frame_indices=frame_indices,
        raw_dir=raw_dir,
        ffmpeg_bin=ffmpeg_bin,
    )
    segment = analysis["segments"][0]
    homography = np.asarray(segment["homography_world_xy_to_native_px"], dtype=np.float64)
    template = get_court_template("pickleball")
    output_paths: list[str] = []
    for order, (frame_index, raw_path) in enumerate(zip(frame_indices, raw_paths, strict=True), start=1):
        frame = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"could not read ffmpeg frame {raw_path}")
        for line_name, (start_xyz, end_xyz) in template.line_segments_m.items():
            start = _project_one(homography, (float(start_xyz[0]), float(start_xyz[1])))
            end = _project_one(homography, (float(end_xyz[0]), float(end_xyz[1])))
            cv2.line(frame, start, end, (60, 220, 60), 2, cv2.LINE_AA)
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(frame, line_name, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 220, 60), 1, cv2.LINE_AA)
        for point in segment["points"]:
            if point["uv_px"] is None:
                continue
            xy = tuple(int(round(value)) for value in point["uv_px"])
            cv2.circle(frame, xy, 5, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(point["canonical_name"]),
                (xy[0] + 6, xy[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"pb.vision pseudo-labels | frame {frame_index} | PENDING_SPOTCHECK",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 240),
            2,
            cv2.LINE_AA,
        )
        output_path = spotcheck_dir / f"overlay_{order:02d}_frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise ValueError(f"could not write {output_path}")
        output_paths.append(str(output_path))
    shutil.rmtree(raw_dir)
    summary = {
        "video_id": record.video_id,
        "status": PENDING_STATUS,
        "provenance": PROVENANCE,
        "frame_indices": frame_indices,
        "overlay_count": len(output_paths),
        "overlays": output_paths,
        "full_set_residual_px": analysis["full_set_residual_px"],
        "leave_one_out_residual_px": analysis["leave_one_out_residual_px"],
        "leave_one_out_worst_point": analysis["leave_one_out_worst_point"],
        "confidence": analysis["confidence"],
        "spread_normalized": analysis["spread_normalized"],
    }
    _write_json(spotcheck_dir / "summary.json", summary)
    return summary


def find_existing_solver_cross_reference(
    *,
    runs_root: Path,
    record: VideoRecord,
    index_to_name: tuple[str, ...],
) -> dict[str, Any]:
    if record.video_id in COMPARE_ONLY_IDS:
        raise ValueError(
            f"compare-only identity {record.video_id!r} is rejected before source access"
        )
    candidates: list[tuple[tuple[int, int, str], Path, dict[str, Any]]] = []
    for path in runs_root.rglob("court_calibration.json"):
        if "pbvision_11min" not in path.as_posix() and record.video_id not in path.as_posix():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        homography = payload.get("homography") if isinstance(payload, dict) else None
        if not isinstance(homography, list) or np.asarray(homography).shape != (3, 3):
            continue
        source = str(payload.get("source") or payload.get("intrinsics", {}).get("source") or "")
        priority = (
            0 if source == "metric_15pt_reviewed" else 1,
            0 if "metric15_preflight2" in path.as_posix() else 1,
            path.as_posix(),
        )
        candidates.append((priority, path, payload))
    if not candidates:
        return {"status": "not_found", "note": "No prior own-pipeline keypoint or homography artifact found."}
    _, selected_path, payload = sorted(candidates, key=lambda row: row[0])[0]
    homography = np.asarray(payload["homography"], dtype=np.float64)
    segment = _camera_segments(record)[0]
    points: list[dict[str, Any]] = []
    deltas: list[float] = []
    for point_index, name in enumerate(index_to_name):
        point = segment["court_points"][point_index]
        if point is None:
            continue
        ours = _project_one(homography, CANONICAL_WORLD_XY[name])
        pb = [float(point["u"]) * record.width, float(point["v"]) * record.height]
        delta = float(math.hypot(ours[0] - pb[0], ours[1] - pb[1]))
        deltas.append(delta)
        points.append(
            {
                "index": point_index,
                "canonical_name": name,
                "pbvision_uv_px": pb,
                "ours_uv_px": list(ours),
                "delta_px": delta,
            }
        )
    return {
        "status": "found",
        "selected_artifact": str(selected_path),
        "candidate_artifact_count": len(candidates),
        "source": payload.get("source") or payload.get("intrinsics", {}).get("source"),
        "delta_px": _summary(deltas),
        "points": points,
        "note": "Reference-only comparison; neither solver is independent accuracy truth.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-root", type=Path, default=ROOT / "data/pbvision_gallery_20260719")
    parser.add_argument("--local-root", type=Path, default=ROOT / "data/pbvision_11min_20260713")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/pbvision_gallery_20260719/MANIFEST.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/court_real_pbvision_20260722")
    parser.add_argument("--lane-dir", type=Path, default=ROOT / "runs/lanes/court_pbv_extract_20260722")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--frames-per-segment", type=int, default=24)
    parser.add_argument("--expected-video-count", type=int, default=10)
    parser.add_argument("--expected-corpus-videos", type=int, default=9)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--json", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(
        gallery_root=args.gallery_root,
        local_root=args.local_root,
        manifest_path=args.manifest,
        local_video_id=LOCAL_VIDEO_ID,
        ffprobe_bin=args.ffprobe,
    )
    if len(records) != args.expected_video_count:
        raise ValueError(f"expected {args.expected_video_count} videos, found {len(records)}")
    mapping, assignment_details = discover_mapping(records)
    analyses = analyze_records(records, mapping.index_to_name)
    timebase_audit = build_timebase_audit(
        records,
        ffprobe_bin=args.ffprobe,
    )
    write_timebase_audit(args.lane_dir, timebase_audit)

    # Kill only if the ground-template hypothesis fails broadly.  A single bad
    # view is surfaced, not used to force or suppress the other twelve.
    medians = [float(row["full_set_residual_px"]["median"]) for row in analyses]
    if sum(value > 30.0 for value in medians) >= math.ceil(0.8 * len(medians)):
        kill = {
            "status": "PBV_COURT_POINTS_NOT_TEMPLATE",
            "median_residuals_px": dict(zip((record.video_id for record in records), medians, strict=True)),
        }
        _write_json(args.lane_dir / "kill_criterion.json", kill)
        raise RuntimeError("PBV_COURT_POINTS_NOT_TEMPLATE")

    corpus = emit_corpus(
        records=records,
        index_to_name=mapping.index_to_name,
        output_root=args.output_root,
        dropped_corpus_ids=DROPPED_CORPUS_IDS,
        frames_per_segment=args.frames_per_segment,
    )
    if corpus["videos_emitted"] != args.expected_corpus_videos:
        raise ValueError(
            f"expected {args.expected_corpus_videos} eligible corpus videos, emitted {corpus['videos_emitted']}"
        )

    spotcheck = {"status": "not_run_no_permitted_local_source", "overlay_count": 0}
    cross_reference = {"status": "not_run_no_permitted_local_source"}

    mapping_rows = [
        {"index": index, "canonical_name": name, "world_xy_m": list(CANONICAL_WORLD_XY[name])}
        for index, name in enumerate(mapping.index_to_name)
    ]
    per_video_assignment = validate_per_video_assignments(records, mapping)
    assignment = {
        **assignment_details,
        "mapping": mapping_rows,
        **per_video_assignment,
        "side_view_note": (
            "bewqc0glhgpq supports the batch mapping only under the constrained check that freezes the "
            "aggregate four index groups and scores 48 row-semantic/reflection assignments. Its "
            "unconstrained single-video 4x3 partition differs, so it is not an independent partition win."
        ),
    }
    corpus_medians = [
        float(row["full_set_residual_px"]["median"])
        for row in analyses
        if row["video_id"] not in DROPPED_CORPUS_IDS and row["origin"] == "gallery"
    ]
    summary = {
        "schema_version": 1,
        "artifact_type": "pbvision_court_keypoint_extraction_summary",
        "status": PENDING_STATUS,
        "provenance": PROVENANCE,
        "owner_directive": (
            "Owner signed-agreement ruling authorizes complete internal use; PROGRAM.md Track A action 1 "
            "requires pb.vision court pseudo-label extraction plus mandatory spot-check."
        ),
        "compare_only_ids": sorted(COMPARE_ONLY_IDS),
        "assignment": assignment,
        "videos": analyses,
        "corpus": corpus,
        "timebase_audit": timebase_audit,
        "quality_bar": {
            "corpus_video_count": len(corpus_medians),
            "median_residual_le_3px_count": sum(value <= 3.0 for value in corpus_medians),
            "target_count": 8,
            "passed": sum(value <= 3.0 for value in corpus_medians) >= 8,
        },
        "spotcheck": spotcheck,
        "existing_solver_cross_reference": cross_reference,
        "cross_signal": {
            "consumes": (
                "permitted pb.vision cv_export court calibrations under owner full-use ruling; "
                "compare-only identities rejected before source access"
            ),
            "feeds": [
                "Track A court detector retrain",
                "C0 family-held split",
                "fail-closed residual-statistics baseline",
                "Track B gallery domain audit",
            ],
        },
        "best_stack_delta": "none",
    }
    _write_json(args.lane_dir / "assignment_analysis.json", assignment)
    _write_json(args.lane_dir / "residuals_10_permitted_videos.json", {"videos": analyses})
    _write_json(args.lane_dir / "corpus_summary.json", corpus)
    _write_json(args.lane_dir / "existing_solver_cross_reference.json", cross_reference)
    _write_json(args.lane_dir / "extraction_summary.json", summary)
    return {
        "status": PENDING_STATUS,
        "mapping_consistent": f"{per_video_assignment['consistent_video_count']}/{len(records)}",
        "corpus_videos": corpus["videos_emitted"],
        "corpus_rows": corpus["rows_emitted"],
        "target_frames": corpus["target_frames_listed"],
        "spotcheck_overlays": 0,
        "quality_bar_count": summary["quality_bar"]["median_residual_le_3px_count"],
        "summary": str(args.lane_dir / "extraction_summary.json"),
    }


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
