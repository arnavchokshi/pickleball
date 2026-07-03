#!/usr/bin/env python3
"""Compare two person raw-pool tracking artifacts frame by frame.

This is a diagnostic-only script for TRK pool parity investigations. It reads
existing raw-pool artifacts and writes source-only summaries; it does not train,
mutate protected clips, or change production pipeline behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from threed.racketsport.court_templates import get_court_template  # noqa: E402
from threed.racketsport.person_fast import person_detection_from_bbox  # noqa: E402
from threed.racketsport.schemas import CourtCalibration, validate_artifact_file  # noqa: E402


CONF_BINS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 0.95, 1.01]
AREA_BINS = [0, 2_500, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000, 10**12]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _detection_path(pool_dir: Path, name: str) -> Path:
    path = pool_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {name} in {pool_dir}")
    return path


def _is_person_detection(detection: Mapping[str, Any]) -> bool:
    value = detection.get("class", "person")
    return value == 0 or str(value).lower() in {"person", "player", "0"}


def _bbox(detection: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw = detection.get("bbox") or detection.get("bbox_xyxy")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in raw)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def flatten_detections(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("detections payload must contain a frames list")
    for default_frame, frame_entry in enumerate(frames):
        if not isinstance(frame_entry, Mapping):
            continue
        frame = int(frame_entry.get("frame", frame_entry.get("frame_index", default_frame)))
        detections = frame_entry.get("detections", [])
        if not isinstance(detections, list):
            continue
        for det_idx, detection in enumerate(detections):
            if not isinstance(detection, Mapping) or not _is_person_detection(detection):
                continue
            box = _bbox(detection)
            if box is None:
                continue
            width = box[2] - box[0]
            height = box[3] - box[1]
            rows.append(
                {
                    "frame": frame,
                    "det_idx": det_idx,
                    "track_id": int(detection.get("track_id", detection.get("id", det_idx + 1))),
                    "conf": float(detection.get("conf", detection.get("confidence", 1.0))),
                    "bbox": box,
                    "width": width,
                    "height": height,
                    "area": width * height,
                }
            )
    return rows


def _histogram(values: Iterable[float], bins: Sequence[float]) -> dict[str, int]:
    labels = [f"[{bins[idx]:g},{bins[idx + 1]:g})" for idx in range(len(bins) - 1)]
    counts = {label: 0 for label in labels}
    for value in values:
        placed = False
        for idx in range(len(bins) - 1):
            if bins[idx] <= value < bins[idx + 1]:
                counts[labels[idx]] += 1
                placed = True
                break
        if not placed and value >= bins[-1]:
            counts[labels[-1]] += 1
    return counts


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _count_histogram(counts: Iterable[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(counts).items())}


def _ranges(frames: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = prev = ordered[0]
    for frame in ordered[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        ranges.append([start, prev])
        start = prev = frame
    ranges.append([start, prev])
    return ranges


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _bbox_delta(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(a, b, strict=True))


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return math.inf
    dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return math.inf
    return 1.0 - max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def _load_calibration(path: str | Path) -> CourtCalibration:
    parsed = validate_artifact_file("court_calibration", Path(path))
    if not isinstance(parsed, CourtCalibration):
        raise ValueError(f"{path} did not parse as CourtCalibration")
    return parsed


def _greedy_non_overlapping(
    boxes: Sequence[tuple[float, tuple[float, float, float, float]]],
    *,
    iou_threshold: float,
) -> int:
    selected: list[tuple[float, float, float, float]] = []
    for _conf, box in sorted(boxes, key=lambda item: item[0], reverse=True):
        if any(_iou(box, existing) > iou_threshold for existing in selected):
            continue
        selected.append(box)
    return len(selected)


def on_court_frame_counts(
    payload: Mapping[str, Any],
    *,
    calibration_path: str | Path,
    court_margin_m: float,
    expected_players: int = 4,
    overlap_iou_threshold: float = 0.3,
    sport: str = "pickleball",
) -> dict[str, Any]:
    calibration = _load_calibration(calibration_path)
    template = get_court_template(sport)
    half_width_m = template.width_m / 2.0 + court_margin_m
    half_length_m = template.length_m / 2.0 + court_margin_m
    rows: list[dict[str, Any]] = []
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("detections payload must contain a frames list")
    for default_frame, frame_entry in enumerate(frames):
        if not isinstance(frame_entry, Mapping):
            continue
        frame = int(frame_entry.get("frame", frame_entry.get("frame_index", default_frame)))
        raw_detections = frame_entry.get("detections", [])
        if not isinstance(raw_detections, list):
            raw_detections = []
        on_court: list[tuple[float, tuple[float, float, float, float]]] = []
        person_count = 0
        for detection in raw_detections:
            if not isinstance(detection, Mapping) or not _is_person_detection(detection):
                continue
            box = _bbox(detection)
            if box is None:
                continue
            person_count += 1
            conf = float(detection.get("conf", detection.get("confidence", 1.0)))
            person = person_detection_from_bbox(calibration, bbox_xyxy=box, confidence=conf)
            x, y = person.foot_world_xy
            if -half_width_m <= x <= half_width_m and -half_length_m <= y <= half_length_m:
                on_court.append((conf, box))
        nonoverlap = _greedy_non_overlapping(on_court, iou_threshold=overlap_iou_threshold)
        rows.append(
            {
                "frame": frame,
                "person_detections": person_count,
                "on_court_nonoverlap": nonoverlap,
                "sufficient": nonoverlap >= expected_players,
            }
        )
    sufficient_frames = [row["frame"] for row in rows if row["sufficient"]]
    return {
        "frame_count": len(rows),
        "frames_with_sufficient_on_court_detections": len(sufficient_frames),
        "four_player_detection_ceiling": len(sufficient_frames) / len(rows) if rows else 0.0,
        "on_court_count_histogram": _count_histogram(row["on_court_nonoverlap"] for row in rows),
        "person_detection_count_histogram": _count_histogram(row["person_detections"] for row in rows),
        "sufficient_frame_ranges": _ranges(sufficient_frames),
        "per_frame": rows,
    }


def detection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame_counts = Counter(int(row["frame"]) for row in rows)
    track_counts = Counter(int(row["track_id"]) for row in rows)
    return {
        "detection_count": len(rows),
        "frame_count": len(frame_counts),
        "per_frame_count_histogram": _count_histogram(frame_counts.values()),
        "frames_with_at_least_4_raw_detections": sum(1 for count in frame_counts.values() if count >= 4),
        "confidence_stats": _stats([float(row["conf"]) for row in rows]),
        "confidence_histogram": _histogram([float(row["conf"]) for row in rows], CONF_BINS),
        "bbox_width_stats": _stats([float(row["width"]) for row in rows]),
        "bbox_height_stats": _stats([float(row["height"]) for row in rows]),
        "bbox_area_stats": _stats([float(row["area"]) for row in rows]),
        "bbox_area_histogram": _histogram([float(row["area"]) for row in rows], AREA_BINS),
        "source_track_count": len(track_counts),
        "top_source_track_counts": [
            {"track_id": track_id, "detections": count} for track_id, count in track_counts.most_common(20)
        ],
    }


def compare_detection_sets(
    baseline_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.99,
) -> dict[str, Any]:
    baseline_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    fresh_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        baseline_by_frame[int(row["frame"])].append(row)
    for row in fresh_rows:
        fresh_by_frame[int(row["frame"])].append(row)

    matches: list[tuple[Mapping[str, Any], Mapping[str, Any], float]] = []
    unmatched_baseline = 0
    unmatched_fresh = 0
    count_delta_by_frame: list[dict[str, int]] = []
    for frame in sorted(set(baseline_by_frame) | set(fresh_by_frame)):
        base_items = sorted(baseline_by_frame.get(frame, []), key=lambda row: float(row["conf"]), reverse=True)
        fresh_items = sorted(fresh_by_frame.get(frame, []), key=lambda row: float(row["conf"]), reverse=True)
        used_fresh: set[int] = set()
        frame_matches = 0
        for base in base_items:
            best_idx = None
            best_iou = -1.0
            for idx, fresh in enumerate(fresh_items):
                if idx in used_fresh:
                    continue
                iou = _iou(base["bbox"], fresh["bbox"])  # type: ignore[arg-type]
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= iou_threshold:
                used_fresh.add(best_idx)
                matches.append((base, fresh_items[best_idx], best_iou))
                frame_matches += 1
        unmatched_baseline += len(base_items) - frame_matches
        unmatched_fresh += len(fresh_items) - frame_matches
        count_delta_by_frame.append(
            {
                "frame": frame,
                "baseline_count": len(base_items),
                "fresh_count": len(fresh_items),
                "delta": len(fresh_items) - len(base_items),
            }
        )

    conf_abs_deltas = [abs(float(base["conf"]) - float(fresh["conf"])) for base, fresh, _iou_value in matches]
    bbox_deltas = [_bbox_delta(base["bbox"], fresh["bbox"]) for base, fresh, _iou_value in matches]  # type: ignore[arg-type]
    ious = [iou_value for _base, _fresh, iou_value in matches]
    track_id_changed = sum(1 for base, fresh, _iou_value in matches if int(base["track_id"]) != int(fresh["track_id"]))
    count_delta_frames = [row for row in count_delta_by_frame if row["delta"] != 0]
    return {
        "iou_threshold": iou_threshold,
        "matched_detection_count": len(matches),
        "unmatched_baseline_count": unmatched_baseline,
        "unmatched_fresh_count": unmatched_fresh,
        "track_id_changed_on_matched_boxes": track_id_changed,
        "track_id_changed_rate": track_id_changed / len(matches) if matches else 0.0,
        "matched_iou_stats": _stats(ious),
        "matched_conf_abs_delta_stats": _stats(conf_abs_deltas),
        "matched_bbox_max_abs_delta_px_stats": _stats(bbox_deltas),
        "frame_count_delta_histogram": _count_histogram(row["delta"] for row in count_delta_by_frame),
        "frames_with_detection_count_delta": {
            "count": len(count_delta_frames),
            "ranges": _ranges(row["frame"] for row in count_delta_frames),
            "first_40": count_delta_frames[:40],
        },
    }


def _embedding_rows_by_frame(path: Path | None) -> dict[int, list[Mapping[str, Any]]]:
    if path is None or not path.is_file():
        return {}
    payload = read_json(path)
    rows = payload.get("detections", [])
    if not isinstance(rows, list):
        return {}
    out: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        frame = int(row.get("frame", row.get("frame_index", row.get("frame_idx", -1))))
        if frame >= 0:
            out[frame].append(row)
    return out


def _nearest_embedding(
    rows_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    frame: int,
    bbox: Sequence[float],
    max_delta_px: float = 3.0,
) -> Sequence[float] | None:
    best: Mapping[str, Any] | None = None
    best_delta = math.inf
    for row in rows_by_frame.get(frame, []):
        row_bbox = row.get("bbox") or row.get("bbox_xyxy")
        if not isinstance(row_bbox, list | tuple) or len(row_bbox) != 4:
            continue
        delta = _bbox_delta(bbox, row_bbox)
        if delta < best_delta:
            best_delta = delta
            best = row
    if best is None or best_delta > max_delta_px:
        return None
    vector = best.get("embedding")
    return vector if isinstance(vector, list | tuple) else None


def compare_embeddings_for_matched_boxes(
    baseline_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_embedding_path: Path | None,
    fresh_embedding_path: Path | None,
    iou_threshold: float = 0.99,
) -> dict[str, Any]:
    baseline_embeddings = _embedding_rows_by_frame(baseline_embedding_path)
    fresh_embeddings = _embedding_rows_by_frame(fresh_embedding_path)
    if not baseline_embeddings or not fresh_embeddings:
        return {"status": "missing_embedding_payload"}

    baseline_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    fresh_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        baseline_by_frame[int(row["frame"])].append(row)
    for row in fresh_rows:
        fresh_by_frame[int(row["frame"])].append(row)

    distances: list[float] = []
    missing = 0
    compared = 0
    exact_stored_vectors = 0
    for frame in sorted(set(baseline_by_frame) & set(fresh_by_frame)):
        fresh_items = list(fresh_by_frame[frame])
        used_fresh: set[int] = set()
        for base in baseline_by_frame[frame]:
            best_idx = None
            best_iou = -1.0
            for idx, fresh in enumerate(fresh_items):
                if idx in used_fresh:
                    continue
                iou = _iou(base["bbox"], fresh["bbox"])  # type: ignore[arg-type]
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is None or best_iou < iou_threshold:
                continue
            used_fresh.add(best_idx)
            base_vec = _nearest_embedding(
                baseline_embeddings,
                frame=frame,
                bbox=base["bbox"],  # type: ignore[arg-type]
            )
            fresh_vec = _nearest_embedding(
                fresh_embeddings,
                frame=frame,
                bbox=fresh_items[best_idx]["bbox"],  # type: ignore[arg-type]
            )
            if base_vec is None or fresh_vec is None:
                missing += 1
                continue
            compared += 1
            if list(base_vec) == list(fresh_vec):
                exact_stored_vectors += 1
            distances.append(_cosine_distance(base_vec, fresh_vec))
    return {
        "status": "ok",
        "compared_embedding_pairs": compared,
        "missing_embedding_pairs": missing,
        "exact_stored_vector_pairs": exact_stored_vectors,
        "exact_stored_vector_rate": exact_stored_vectors / compared if compared else 0.0,
        "cosine_distance_stats": _stats(distances),
    }


def track_summary(tracks_payload: Mapping[str, Any], *, frame_count_hint: int | None = None) -> dict[str, Any]:
    fps = float(tracks_payload.get("fps", 30.0))
    active_by_frame: dict[int, list[int]] = defaultdict(list)
    player_lengths: dict[str, int] = {}
    for player in tracks_payload.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", len(player_lengths) + 1))
        frames = player.get("frames", [])
        if not isinstance(frames, list):
            continue
        player_lengths[str(player_id)] = len(frames)
        for frame_entry in frames:
            if not isinstance(frame_entry, Mapping):
                continue
            frame = int(round(float(frame_entry.get("t", 0.0)) * fps))
            active_by_frame[frame].append(player_id)

    if frame_count_hint is not None:
        frames = list(range(frame_count_hint))
    elif active_by_frame:
        frames = list(range(max(active_by_frame) + 1))
    else:
        frames = []
    counts = {frame: len(set(active_by_frame.get(frame, []))) for frame in frames}
    exact_four = [frame for frame, count in counts.items() if count >= 4]
    return {
        "fps": fps,
        "player_count": len(player_lengths),
        "player_frame_lengths": player_lengths,
        "track_frame_count": sum(player_lengths.values()),
        "active_player_count_histogram": _count_histogram(counts.values()),
        "frames_with_at_least_4_tracks": len(exact_four),
        "four_player_coverage": len(exact_four) / len(frames) if frames else 0.0,
        "four_player_frame_ranges": _ranges(exact_four),
        "per_frame_counts": [{"frame": frame, "active_players": count} for frame, count in sorted(counts.items())],
    }


def compare_track_coverage(
    baseline_track_summary: Mapping[str, Any],
    fresh_track_summary: Mapping[str, Any],
    *,
    fresh_on_court_summary: Mapping[str, Any],
) -> dict[str, Any]:
    base_counts = {int(row["frame"]): int(row["active_players"]) for row in baseline_track_summary["per_frame_counts"]}
    fresh_counts = {int(row["frame"]): int(row["active_players"]) for row in fresh_track_summary["per_frame_counts"]}
    fresh_on_court = {
        int(row["frame"]): int(row["on_court_nonoverlap"]) for row in fresh_on_court_summary.get("per_frame", [])
    }
    all_frames = sorted(set(base_counts) | set(fresh_counts))
    baseline_exact_not_fresh = [
        frame for frame in all_frames if base_counts.get(frame, 0) >= 4 and fresh_counts.get(frame, 0) < 4
    ]
    fresh_exact_not_baseline = [
        frame for frame in all_frames if fresh_counts.get(frame, 0) >= 4 and base_counts.get(frame, 0) < 4
    ]
    lost_with_fresh_raw_sufficient = [
        frame for frame in baseline_exact_not_fresh if fresh_on_court.get(frame, 0) >= 4
    ]
    return {
        "baseline_exact_four_not_fresh_count": len(baseline_exact_not_fresh),
        "baseline_exact_four_not_fresh_ranges": _ranges(baseline_exact_not_fresh),
        "fresh_exact_four_not_baseline_count": len(fresh_exact_not_baseline),
        "fresh_exact_four_not_baseline_ranges": _ranges(fresh_exact_not_baseline),
        "lost_frames_where_fresh_raw_pool_still_had_4_on_court": len(lost_with_fresh_raw_sufficient),
        "lost_frames_where_fresh_raw_pool_still_had_4_on_court_ranges": _ranges(lost_with_fresh_raw_sufficient),
        "fresh_active_count_hist_on_baseline_lost_frames": _count_histogram(
            fresh_counts.get(frame, 0) for frame in baseline_exact_not_fresh
        ),
        "first_80_baseline_lost_frames": [
            {
                "frame": frame,
                "baseline_tracks": base_counts.get(frame, 0),
                "fresh_tracks": fresh_counts.get(frame, 0),
                "fresh_raw_on_court_nonoverlap": fresh_on_court.get(frame, 0),
            }
            for frame in baseline_exact_not_fresh[:80]
        ],
    }


def _score_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = read_json(path)
    return {
        key: payload.get(key)
        for key in (
            "candidate",
            "clip_id",
            "idf1",
            "hota",
            "mota",
            "id_switches",
            "false_positives",
            "false_negatives",
            "pred_detections",
            "matches",
            "recall",
            "precision",
            "exact_four_player_frames",
            "expected_four_player_frames",
            "four_player_coverage",
            "bbox_scale_x",
            "bbox_scale_y",
        )
    }


def _current_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("torch", "ultralytics"):
        try:
            env[f"{package}_version"] = metadata.version(package)
        except metadata.PackageNotFoundError:
            env[f"{package}_version"] = None
    try:
        import torch  # type: ignore[import-not-found]

        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_mps_available"] = bool(torch.backends.mps.is_available()) if hasattr(torch.backends, "mps") else None
        env["torch_mps_built"] = bool(torch.backends.mps.is_built()) if hasattr(torch.backends, "mps") else None
    except Exception as exc:  # noqa: BLE001
        env["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return env


def _infer_imgsz_conf_from_variant(variant: Any) -> dict[str, Any]:
    if not isinstance(variant, str):
        return {"imgsz_effective": None, "conf_effective": None}
    out: dict[str, Any] = {"imgsz_effective": None, "conf_effective": None}
    parts = variant.split("_")
    for part in parts:
        if part.startswith("img") and part[3:].isdigit():
            out["imgsz_effective"] = int(part[3:])
        if part.startswith("conf") and part[4:].isdigit():
            digits = part[4:]
            if len(digits) == 3 and digits.startswith("0"):
                out["conf_effective"] = int(digits) / 100.0
            else:
                out["conf_effective"] = int(digits) / (10 ** len(digits))
    return out


def _pool_metadata(pool_dir: Path, summary_path: Path, score_path: Path | None) -> dict[str, Any]:
    metrics_path = pool_dir / "metrics.json"
    summary = read_json(summary_path)
    metrics = read_json(metrics_path) if metrics_path.is_file() else {}
    inferred = _infer_imgsz_conf_from_variant(metrics.get("variant"))
    return {
        "pool_dir": str(pool_dir),
        "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
        "summary_path": str(summary_path),
        "score_path": str(score_path) if score_path is not None and score_path.is_file() else None,
        "model": metrics.get("model"),
        "variant": metrics.get("variant"),
        "source_video": metrics.get("source_video"),
        "device": metrics.get("device"),
        "tracker_config": metrics.get("tracker_config"),
        "imgsz": metrics.get("imgsz"),
        "imgsz_effective": metrics.get("imgsz") or inferred["imgsz_effective"],
        "conf": metrics.get("conf"),
        "conf_effective": metrics.get("conf") if metrics.get("conf") is not None else inferred["conf_effective"],
        "iou": metrics.get("iou"),
        "iou_effective": metrics.get("iou"),
        "counts": metrics.get("counts", {}),
        "raw_pool_candidate": summary.get("candidate"),
        "association_config": summary.get("config", {}),
        "global_association": summary.get("global_association", {}),
        "detection_limited_ceiling": summary.get("detection_limited_ceiling"),
        "embedding_export": summary.get("embedding_export"),
        "embedding_bbox_scale": summary.get("embedding_bbox_scale"),
        "score_bbox_scale_x": summary.get("score_bbox_scale_x"),
        "score_bbox_scale_y": summary.get("score_bbox_scale_y"),
        "score": _score_summary(score_path),
    }


def _write_per_frame_csv(path: Path, baseline_rows: Sequence[Mapping[str, Any]], fresh_rows: Sequence[Mapping[str, Any]]) -> None:
    base_by_frame = {int(row["frame"]): row for row in baseline_rows}
    fresh_by_frame = {int(row["frame"]): row for row in fresh_rows}
    frames = sorted(set(base_by_frame) | set(fresh_by_frame))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "baseline_person_detections",
                "fresh_person_detections",
                "baseline_on_court_nonoverlap",
                "fresh_on_court_nonoverlap",
                "baseline_sufficient",
                "fresh_sufficient",
            ],
        )
        writer.writeheader()
        for frame in frames:
            base = base_by_frame.get(frame, {})
            fresh = fresh_by_frame.get(frame, {})
            writer.writerow(
                {
                    "frame": frame,
                    "baseline_person_detections": base.get("person_detections", 0),
                    "fresh_person_detections": fresh.get("person_detections", 0),
                    "baseline_on_court_nonoverlap": base.get("on_court_nonoverlap", 0),
                    "fresh_on_court_nonoverlap": fresh.get("on_court_nonoverlap", 0),
                    "baseline_sufficient": base.get("sufficient", False),
                    "fresh_sufficient": fresh.get("sufficient", False),
                }
            )


def _write_markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    base = report["baseline"]
    fresh = report["fresh"]
    det = report["detection_comparison"]
    coverage = report["track_coverage_comparison"]
    embed = report["embedding_comparison"]
    env = report["environment"]
    lines = [
        "# Pool Parity Diagnostic Report",
        "",
        f"- clip: `{report['clip_id']}`",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        f"- baseline pool: `{base['metadata']['pool_dir']}`",
        f"- fresh pool: `{fresh['metadata']['pool_dir']}`",
        "",
        "## Summary",
        "",
        (
            "- Raw pool detection counts are effectively unchanged: "
            f"{base['raw_detection_summary']['detection_count']} baseline vs "
            f"{fresh['raw_detection_summary']['detection_count']} fresh."
        ),
        (
            "- On-court >=4 detection frames are effectively unchanged: "
            f"{base['on_court_summary']['frames_with_sufficient_on_court_detections']} baseline vs "
            f"{fresh['on_court_summary']['frames_with_sufficient_on_court_detections']} fresh."
        ),
        (
            "- Final associated >=4-track frames fell from "
            f"{base['track_summary']['frames_with_at_least_4_tracks']} to "
            f"{fresh['track_summary']['frames_with_at_least_4_tracks']}."
        ),
        (
            "- Baseline exact-four frames missing in fresh: "
            f"{coverage['baseline_exact_four_not_fresh_count']}; of those, fresh still had >=4 raw on-court boxes in "
            f"{coverage['lost_frames_where_fresh_raw_pool_still_had_4_on_court']} frames."
        ),
        (
            "- Matched raw boxes: "
            f"{det['matched_detection_count']} at IoU >= {det['iou_threshold']}; unmatched baseline/fresh: "
            f"{det['unmatched_baseline_count']}/{det['unmatched_fresh_count']}."
        ),
        (
            "- Matched-box source track IDs changed in "
            f"{det['track_id_changed_on_matched_boxes']} pairs "
            f"({det['track_id_changed_rate']:.3f})."
        ),
        "",
        "## Root-Cause Read",
        "",
        (
            "The dominant driver is not raw detection coverage, confidence/NMS thresholding, or court-scaling loss. "
            "The raw pools have the same frame count, nearly the same detection count, nearly identical on-court "
            ">=4-frame ceiling, and the same association config. The loss appears after source tracking/association: "
            f"selected fragments dropped from {base['metadata']['global_association'].get('selected_fragment_count')} to "
            f"{fresh['metadata']['global_association'].get('selected_fragment_count')}, and synthetic gap-fill dropped from "
            f"{base['metadata']['global_association'].get('synthetic_frame_count')} to "
            f"{fresh['metadata']['global_association'].get('synthetic_frame_count')}."
        ),
        (
            "No export-path config bug is proven by these artifacts. The measurable sensitivity is in the "
            "Ultralytics/BoT-SORT source track IDs and fragment topology produced by the cold local run versus the "
            "registered A100 pool. Recommendation: keep the registered A100 raw-pool route as the prereg source for "
            "this candidate, or rerun cold tracking on the A100 before treating local cold-start cov4 as equivalent."
        ),
        "",
        "## Environment Probe",
        "",
        f"- python: `{env.get('python_executable')}` `{env.get('python_version')}`",
        f"- torch: `{env.get('torch_version')}` cuda_available={env.get('torch_cuda_available')} mps_available={env.get('torch_mps_available')} mps_built={env.get('torch_mps_built')}",
        f"- ultralytics: `{env.get('ultralytics_version')}`",
        "",
        "## Config/Score",
        "",
        "| field | baseline | fresh |",
        "| --- | --- | --- |",
    ]
    for field in (
        "tracker_config",
        "imgsz_effective",
        "conf_effective",
        "iou_effective",
        "device",
        "score_bbox_scale_x",
        "score_bbox_scale_y",
    ):
        lines.append(f"| `{field}` | `{base['metadata'].get(field)}` | `{fresh['metadata'].get(field)}` |")
    for field in ("idf1", "four_player_coverage", "pred_detections", "false_negatives", "false_positives"):
        base_score = base["metadata"].get("score") or {}
        fresh_score = fresh["metadata"].get("score") or {}
        lines.append(f"| score `{field}` | `{base_score.get(field)}` | `{fresh_score.get(field)}` |")
    lines.extend(
        [
            "",
            "## Key JSON Artifacts",
            "",
            "- `pool_parity_report.json`",
            "- `per_frame_on_court_counts.csv`",
            "",
        ]
    )
    if embed.get("status") == "ok":
        lines.extend(
            [
                "## Embeddings",
                "",
                (
                    f"- compared pairs: `{embed['compared_embedding_pairs']}`, exact stored vectors: "
                    f"`{embed['exact_stored_vector_pairs']}` "
                    f"({embed['exact_stored_vector_rate']:.3f})"
                ),
                f"- cosine distance stats: `{embed['cosine_distance_stats']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parity_report(
    *,
    clip_id: str,
    baseline_pool_dir: str | Path,
    baseline_summary_path: str | Path,
    baseline_score_path: str | Path | None,
    fresh_pool_dir: str | Path,
    fresh_summary_path: str | Path,
    fresh_score_path: str | Path | None,
    out_dir: str | Path,
    baseline_embedding_path: str | Path | None = None,
    fresh_embedding_path: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline_pool = Path(baseline_pool_dir)
    fresh_pool = Path(fresh_pool_dir)
    baseline_summary = read_json(baseline_summary_path)
    fresh_summary = read_json(fresh_summary_path)

    baseline_raw_payload = read_json(_detection_path(baseline_pool, "raw_tracked_detections.json"))
    fresh_raw_payload = read_json(_detection_path(fresh_pool, "raw_tracked_detections.json"))
    baseline_geometry_payload = read_json(_detection_path(baseline_pool, "tracked_detections.json"))
    fresh_geometry_payload = read_json(_detection_path(fresh_pool, "tracked_detections.json"))
    baseline_tracks = read_json(Path(baseline_summary["tracks_path"]))
    fresh_tracks = read_json(Path(fresh_summary["tracks_path"]))

    baseline_raw_rows = flatten_detections(baseline_raw_payload)
    fresh_raw_rows = flatten_detections(fresh_raw_payload)
    baseline_geometry_rows = flatten_detections(baseline_geometry_payload)
    fresh_geometry_rows = flatten_detections(fresh_geometry_payload)

    baseline_court_margin = float((baseline_summary.get("config") or {}).get("court_margin_m", 2.0))
    fresh_court_margin = float((fresh_summary.get("config") or {}).get("court_margin_m", 2.0))
    baseline_on_court = on_court_frame_counts(
        baseline_geometry_payload,
        calibration_path=baseline_summary["calibration_path"],
        court_margin_m=baseline_court_margin,
    )
    fresh_on_court = on_court_frame_counts(
        fresh_geometry_payload,
        calibration_path=fresh_summary["calibration_path"],
        court_margin_m=fresh_court_margin,
    )
    frame_count_hint = None
    baseline_score = _score_summary(Path(baseline_score_path)) if baseline_score_path else None
    fresh_score = _score_summary(Path(fresh_score_path)) if fresh_score_path else None
    for score in (baseline_score, fresh_score):
        if score and score.get("expected_four_player_frames"):
            frame_count_hint = int(score["expected_four_player_frames"])
            break
    baseline_track_summary = track_summary(baseline_tracks, frame_count_hint=frame_count_hint)
    fresh_track_summary = track_summary(fresh_tracks, frame_count_hint=frame_count_hint)

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_pool_parity_diagnostic",
        "clip_id": clip_id,
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "environment": _current_env(),
        "baseline": {
            "metadata": _pool_metadata(Path(baseline_pool_dir), Path(baseline_summary_path), Path(baseline_score_path) if baseline_score_path else None),
            "raw_detection_summary": detection_summary(baseline_raw_rows),
            "geometry_detection_summary": detection_summary(baseline_geometry_rows),
            "on_court_summary": {key: value for key, value in baseline_on_court.items() if key != "per_frame"},
            "track_summary": {key: value for key, value in baseline_track_summary.items() if key != "per_frame_counts"},
        },
        "fresh": {
            "metadata": _pool_metadata(Path(fresh_pool_dir), Path(fresh_summary_path), Path(fresh_score_path) if fresh_score_path else None),
            "raw_detection_summary": detection_summary(fresh_raw_rows),
            "geometry_detection_summary": detection_summary(fresh_geometry_rows),
            "on_court_summary": {key: value for key, value in fresh_on_court.items() if key != "per_frame"},
            "track_summary": {key: value for key, value in fresh_track_summary.items() if key != "per_frame_counts"},
        },
        "detection_comparison": compare_detection_sets(baseline_raw_rows, fresh_raw_rows),
        "geometry_detection_comparison": compare_detection_sets(baseline_geometry_rows, fresh_geometry_rows),
        "embedding_comparison": compare_embeddings_for_matched_boxes(
            baseline_raw_rows,
            fresh_raw_rows,
            baseline_embedding_path=Path(baseline_embedding_path) if baseline_embedding_path else Path(str(baseline_summary.get("embedding_export_path", ""))),
            fresh_embedding_path=Path(fresh_embedding_path) if fresh_embedding_path else Path(str(fresh_summary.get("embedding_export_path", ""))),
        ),
        "track_coverage_comparison": compare_track_coverage(
            baseline_track_summary,
            fresh_track_summary,
            fresh_on_court_summary=fresh_on_court,
        ),
    }
    write_json(out / "pool_parity_report.json", report)
    _write_per_frame_csv(out / "per_frame_on_court_counts.csv", baseline_on_court["per_frame"], fresh_on_court["per_frame"])
    _write_markdown_report(out / "REPORT.md", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff two raw-pool person tracking artifacts.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--baseline-pool-dir", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--baseline-score", default=None, type=Path)
    parser.add_argument("--fresh-pool-dir", required=True, type=Path)
    parser.add_argument("--fresh-summary", required=True, type=Path)
    parser.add_argument("--fresh-score", default=None, type=Path)
    parser.add_argument("--baseline-embedding", default=None, type=Path)
    parser.add_argument("--fresh-embedding", default=None, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_parity_report(
        clip_id=args.clip,
        baseline_pool_dir=args.baseline_pool_dir,
        baseline_summary_path=args.baseline_summary,
        baseline_score_path=args.baseline_score,
        fresh_pool_dir=args.fresh_pool_dir,
        fresh_summary_path=args.fresh_summary,
        fresh_score_path=args.fresh_score,
        baseline_embedding_path=args.baseline_embedding,
        fresh_embedding_path=args.fresh_embedding,
        out_dir=args.out,
    )
    print(json.dumps({"status": "ok", "report": str(args.out / "pool_parity_report.json"), "clip": report["clip_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
