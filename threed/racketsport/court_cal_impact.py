"""Offline downstream-impact scoring for court-calibration candidates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .court_calibration import project_image_points_to_world


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_court_calibration_impact_report"
DEFAULT_OUT_DIR = Path("runs/lanes/calv1_impact_20260708")
PICKLEBALL_HALF_WIDTH_M = 3.048
PICKLEBALL_HALF_LENGTH_M = 6.7056
PICKLEBALL_NVZ_M = 2.1336
GROUNDING_METRIC_KEYS = (
    "max_foot_lock_slide_m",
    "foot_lock_slide_p95_m",
    "max_candidate_phase_slide_m",
    "max_track_anchor_residual_m",
    "max_pre_reset_track_anchor_residual_m",
    "transition_anchor_lag_p95_m",
)
BUILD_CHECKLIST_BULLET = (
    "- [CALV1 IMPACT 2026-07-08, Codex] Court calibration impact harness scores "
    "baseline-vs-candidate CAL artifacts in downstream placement deltas and marks BODY/ball-3D "
    "metrics deferred unless a cheap existing offline path is provided; advisory only, ledger row "
    "+ manager go still mandatory. VERIFIED=0 unchanged."
)


def build_impact_report(
    *,
    baseline_calibration_path: str | Path,
    candidate_calibration_path: str | Path,
    tracks_path: str | Path | None = None,
    placement_path: str | Path | None = None,
    body_grounding_quality_path: str | Path | None = None,
    ball_track_path: str | Path | None = None,
    ball_track_arc_solved_path: str | Path | None = None,
    video_path: str | Path | None = None,
    clip: str | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a pure comparison report; no heavy pipeline stages are run."""

    baseline_path = Path(baseline_calibration_path)
    candidate_path = Path(candidate_calibration_path)
    tracks = _read_json_object(tracks_path) if tracks_path is not None else None
    placement = _read_json_object(placement_path) if placement_path is not None else None
    grounding = _read_json_object(body_grounding_quality_path) if body_grounding_quality_path is not None else None
    ball_track = _read_json_object(ball_track_path) if ball_track_path is not None else None
    ball_arc = _read_json_object(ball_track_arc_solved_path) if ball_track_arc_solved_path is not None else None
    baseline = _read_json_object(baseline_path)
    candidate = _read_json_object(candidate_path)
    clip_id = clip or _infer_clip(tracks, placement, baseline_path)

    live_metrics: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    if tracks is None:
        notes.append("placement metrics deferred: no tracks.json was supplied")
    else:
        track_samples = _track_anchor_samples(tracks)
        if not track_samples:
            notes.append("placement metrics deferred: tracks.json had no bbox bottom-center samples")
        else:
            baseline_points = _project_samples(track_samples, baseline)
            candidate_points = _project_samples(track_samples, candidate)
            _add_placement_metrics(
                live_metrics,
                baseline_points=baseline_points,
                candidate_points=candidate_points,
                placement=placement,
            )

    if ball_track is not None:
        ball_samples = _ball_pixel_samples(ball_track)
        if ball_samples:
            baseline_ball = _project_samples(ball_samples, baseline)
            candidate_ball = _project_samples(ball_samples, candidate)
            _add_ball_court_plane_approx_metrics(
                live_metrics,
                baseline_points=baseline_ball,
                candidate_points=candidate_ball,
            )
        else:
            notes.append("ball court-plane approximation skipped: ball_track.json had no visible xy samples")

    deferred = _deferred_metrics(
        grounding=grounding,
        ball_arc=ball_arc,
        clip=clip_id,
        video_path=Path(video_path) if video_path is not None else None,
        candidate_calibration_path=candidate_path,
        tracks_path=Path(tracks_path) if tracks_path is not None else None,
        ball_track_path=Path(ball_track_path) if ball_track_path is not None else None,
        out_dir=Path(out_dir),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip_id,
        "generated_at": generated_at or _utc_now(),
        "inputs": {
            "baseline_calibration": str(baseline_path),
            "candidate_calibration": str(candidate_path),
            "tracks": str(tracks_path) if tracks_path is not None else None,
            "placement": str(placement_path) if placement_path is not None else None,
            "body_grounding_quality": str(body_grounding_quality_path) if body_grounding_quality_path is not None else None,
            "ball_track": str(ball_track_path) if ball_track_path is not None else None,
            "ball_track_arc_solved": str(ball_track_arc_solved_path) if ball_track_arc_solved_path is not None else None,
            "video": str(video_path) if video_path is not None else None,
        },
        "live_metrics": live_metrics,
        "deferred_requires_pipeline": deferred,
        "downstream_metrics_status": {
            "live": sorted(live_metrics),
            "deferred_requires_pipeline": sorted(deferred),
        },
        "promotion_recommendation": {
            "recommendation": "advisory_only_no_auto_promotion",
            "never_auto_promotes": True,
            "ledger_row_required": True,
            "manager_go_required": True,
            "message": (
                "This report can recommend follow-up, but it never promotes CAL. A CAL promotion "
                "still needs a ledger row plus explicit manager go."
            ),
        },
        "build_checklist_bullet": BUILD_CHECKLIST_BULLET,
        "notes": notes,
    }
    return report


def write_impact_report(report: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _add_placement_metrics(
    live_metrics: dict[str, dict[str, Any]],
    *,
    baseline_points: Sequence[dict[str, Any]],
    candidate_points: Sequence[dict[str, Any]],
    placement: Mapping[str, Any] | None,
) -> None:
    baseline_xy = [item["world_xy"] for item in baseline_points]
    candidate_xy = [item["world_xy"] for item in candidate_points]
    deltas = [_distance(a, b) for a, b in zip(baseline_xy, candidate_xy, strict=True)]
    baseline_stats = _placement_class_stats(baseline_xy)
    candidate_stats = _placement_class_stats(candidate_xy)

    _put_metric(live_metrics, "placement_sample_count", len(baseline_xy), len(candidate_xy), "count")
    _put_metric(live_metrics, "placement_mean_world_x_m", _mean_axis(baseline_xy, 0), _mean_axis(candidate_xy, 0), "m")
    _put_metric(live_metrics, "placement_mean_world_y_m", _mean_axis(baseline_xy, 1), _mean_axis(candidate_xy, 1), "m")
    _put_metric(live_metrics, "placement_step_p95_m", _step_p95(baseline_points), _step_p95(candidate_points), "m")
    _put_metric(live_metrics, "placement_step_max_m", _step_max(baseline_points), _step_max(candidate_points), "m")
    _put_metric(live_metrics, "placement_world_delta_p50_m", 0.0, _percentile(deltas, 50), "m")
    _put_metric(live_metrics, "placement_world_delta_p95_m", 0.0, _percentile(deltas, 95), "m")
    _put_metric(live_metrics, "placement_world_delta_max_m", 0.0, max(deltas, default=0.0), "m")
    for key in (
        "near_fraction",
        "far_fraction",
        "kitchen_fraction",
        "out_of_court_fraction",
    ):
        _put_metric(live_metrics, f"placement_{key}", baseline_stats[key], candidate_stats[key], "fraction")
    _put_metric(
        live_metrics,
        "placement_near_far_flip_count",
        0.0,
        _near_far_flip_count(baseline_xy, candidate_xy),
        "count",
    )
    _put_metric(
        live_metrics,
        "placement_kitchen_flip_count",
        0.0,
        _kitchen_flip_count(baseline_xy, candidate_xy),
        "count",
    )

    placement_index = _placement_index(placement) if placement is not None else {}
    if not placement_index:
        return
    baseline_residuals, baseline_dx, baseline_dy = _residuals_to_existing(baseline_points, placement_index)
    candidate_residuals, candidate_dx, candidate_dy = _residuals_to_existing(candidate_points, placement_index)
    if baseline_residuals and candidate_residuals:
        _put_metric(
            live_metrics,
            "placement_residual_to_existing_p50_m",
            _percentile(baseline_residuals, 50),
            _percentile(candidate_residuals, 50),
            "m",
        )
        _put_metric(
            live_metrics,
            "placement_residual_to_existing_p95_m",
            _percentile(baseline_residuals, 95),
            _percentile(candidate_residuals, 95),
            "m",
        )
        _put_metric(
            live_metrics,
            "placement_residual_to_existing_mean_dx_m",
            _mean(baseline_dx),
            _mean(candidate_dx),
            "m",
        )
        _put_metric(
            live_metrics,
            "placement_residual_to_existing_mean_dy_m",
            _mean(baseline_dy),
            _mean(candidate_dy),
            "m",
        )


def _add_ball_court_plane_approx_metrics(
    live_metrics: dict[str, dict[str, Any]],
    *,
    baseline_points: Sequence[dict[str, Any]],
    candidate_points: Sequence[dict[str, Any]],
) -> None:
    baseline_xy = [item["world_xy"] for item in baseline_points]
    candidate_xy = [item["world_xy"] for item in candidate_points]
    deltas = [_distance(a, b) for a, b in zip(baseline_xy, candidate_xy, strict=True)]
    _put_metric(live_metrics, "ball_court_plane_approx_sample_count", len(baseline_xy), len(candidate_xy), "count")
    _put_metric(live_metrics, "ball_court_plane_approx_mean_world_x_m", _mean_axis(baseline_xy, 0), _mean_axis(candidate_xy, 0), "m")
    _put_metric(live_metrics, "ball_court_plane_approx_mean_world_y_m", _mean_axis(baseline_xy, 1), _mean_axis(candidate_xy, 1), "m")
    _put_metric(live_metrics, "ball_court_plane_approx_world_delta_p95_m", 0.0, _percentile(deltas, 95), "m")


def _deferred_metrics(
    *,
    grounding: Mapping[str, Any] | None,
    ball_arc: Mapping[str, Any] | None,
    clip: str,
    video_path: Path | None,
    candidate_calibration_path: Path,
    tracks_path: Path | None,
    ball_track_path: Path | None,
    out_dir: Path,
) -> dict[str, dict[str, Any]]:
    deferred: dict[str, dict[str, Any]] = {}
    grounding_metrics = grounding.get("grounding_metrics") if isinstance(grounding, Mapping) else None
    grounding_metrics = grounding_metrics if isinstance(grounding_metrics, Mapping) else {}
    for key in GROUNDING_METRIC_KEYS:
        current_value = _maybe_number(grounding_metrics.get(key))
        if current_value is None and key not in grounding_metrics and key != "max_foot_lock_slide_m":
            continue
        deferred[f"grounding.{key}"] = {
            "status": "deferred_requires_pipeline",
            "reason": (
                "BODY grounding metrics are baked after BODY/grounding consumes placement_track_world_xy. "
                "A candidate calibration needs a process_video rerun; this harness does not mutate baked BODY world coordinates."
            ),
            "current_artifact_value": current_value,
            "required_command": _process_video_command(
                clip=clip,
                video_path=video_path,
                candidate_calibration_path=candidate_calibration_path,
                tracks_path=tracks_path,
                ball_track_path=ball_track_path,
                out_dir=out_dir,
            ),
        }
    arc_frame_count = None
    arc_status = None
    if isinstance(ball_arc, Mapping):
        frames = ball_arc.get("frames")
        arc_frame_count = len(frames) if isinstance(frames, list) else None
        arc_status = str(ball_arc.get("status") or "")
    deferred["ball_3d.arc_solved_world_xyz"] = {
        "status": "deferred_requires_pipeline",
        "reason": (
            "True BALL-3D world_xyz is authored by solve_ball_arcs.py from image detections, events, net plane, "
            "and calibration. Existing arc-solved world_xyz is already baked; compare candidate CAL by rerunning the solver."
        ),
        "current_artifact_value": {"status": arc_status, "frame_count": arc_frame_count} if ball_arc is not None else None,
        "required_command": _ball_arc_command(
            clip=clip,
            candidate_calibration_path=candidate_calibration_path,
            ball_track_path=ball_track_path,
            out_dir=out_dir,
        ),
    }
    return deferred


def _track_anchor_samples(tracks: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    fps = float(tracks.get("fps") or 30.0)
    for player in tracks.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        player_id = player.get("id")
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping):
                continue
            bbox = _bbox(frame.get("bbox"))
            if bbox is None:
                continue
            frame_idx = _frame_index(frame, fps)
            bottom_center = [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
            samples.append(
                {
                    "kind": "track_bbox_bottom_center",
                    "player_id": str(player_id),
                    "frame_idx": int(frame_idx),
                    "t": _maybe_number(frame.get("t")) if _maybe_number(frame.get("t")) is not None else frame_idx / fps,
                    "pixel_xy": bottom_center,
                }
            )
    return samples


def _ball_pixel_samples(ball_track: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    frames = ball_track.get("frames")
    if not isinstance(frames, list):
        return samples
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        if frame.get("visible") is False:
            continue
        xy = _xy(frame.get("xy"))
        if xy is None:
            continue
        samples.append(
            {
                "kind": "ball_xy_court_plane_approx",
                "player_id": "ball",
                "frame_idx": int(_maybe_number(frame.get("frame_idx")) or index),
                "t": _maybe_number(frame.get("t")),
                "pixel_xy": xy,
            }
        )
    return samples


def _project_samples(samples: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]) -> list[dict[str, Any]]:
    homography = calibration.get("homography")
    if not isinstance(homography, Sequence):
        raise ValueError("calibration missing homography")
    pixels = [sample["pixel_xy"] for sample in samples]
    world_points = project_image_points_to_world(homography, pixels)
    result: list[dict[str, Any]] = []
    for sample, world_xy in zip(samples, world_points, strict=True):
        result.append({**dict(sample), "world_xy": [float(world_xy[0]), float(world_xy[1])]})
    return result


def _placement_index(placement: Mapping[str, Any] | None) -> dict[tuple[str, int], list[float]]:
    if not isinstance(placement, Mapping):
        return {}
    index: dict[tuple[str, int], list[float]] = {}
    fps = float(placement.get("fps") or 30.0)
    for player in placement.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id"))
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping):
                continue
            xy = _xy(frame.get("smoothed_world_xy") or frame.get("track_world_xy") or frame.get("fused_world_xy"))
            if xy is None:
                continue
            index[(player_id, _frame_index(frame, fps))] = xy
    return index


def _residuals_to_existing(
    samples: Sequence[Mapping[str, Any]],
    placement_index: Mapping[tuple[str, int], Sequence[float]],
) -> tuple[list[float], list[float], list[float]]:
    residuals: list[float] = []
    dx_values: list[float] = []
    dy_values: list[float] = []
    for sample in samples:
        key = (str(sample["player_id"]), int(sample["frame_idx"]))
        reference = placement_index.get(key)
        if reference is None:
            continue
        xy = sample["world_xy"]
        dx = float(xy[0]) - float(reference[0])
        dy = float(xy[1]) - float(reference[1])
        dx_values.append(dx)
        dy_values.append(dy)
        residuals.append(math.hypot(dx, dy))
    return residuals, dx_values, dy_values


def _placement_class_stats(points: Sequence[Sequence[float]]) -> dict[str, float]:
    if not points:
        return {
            "near_fraction": 0.0,
            "far_fraction": 0.0,
            "kitchen_fraction": 0.0,
            "out_of_court_fraction": 0.0,
        }
    near = 0
    far = 0
    kitchen = 0
    out = 0
    for point in points:
        x = float(point[0])
        y = float(point[1])
        if y < 0.0:
            near += 1
        elif y > 0.0:
            far += 1
        if abs(y) <= PICKLEBALL_NVZ_M:
            kitchen += 1
        if abs(x) > PICKLEBALL_HALF_WIDTH_M or abs(y) > PICKLEBALL_HALF_LENGTH_M:
            out += 1
    count = float(len(points))
    return {
        "near_fraction": near / count,
        "far_fraction": far / count,
        "kitchen_fraction": kitchen / count,
        "out_of_court_fraction": out / count,
    }


def _near_far_flip_count(baseline: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> int:
    count = 0
    for a, b in zip(baseline, candidate, strict=True):
        if _side_label(float(a[1])) != _side_label(float(b[1])):
            count += 1
    return count


def _kitchen_flip_count(baseline: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> int:
    count = 0
    for a, b in zip(baseline, candidate, strict=True):
        if (abs(float(a[1])) <= PICKLEBALL_NVZ_M) != (abs(float(b[1])) <= PICKLEBALL_NVZ_M):
            count += 1
    return count


def _side_label(y: float) -> str:
    if y < 0.0:
        return "near"
    if y > 0.0:
        return "far"
    return "net"


def _step_p95(samples: Sequence[Mapping[str, Any]]) -> float:
    return _percentile(_step_distances(samples), 95)


def _step_max(samples: Sequence[Mapping[str, Any]]) -> float:
    return max(_step_distances(samples), default=0.0)


def _step_distances(samples: Sequence[Mapping[str, Any]]) -> list[float]:
    by_player: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_player[str(sample.get("player_id"))].append(sample)
    distances: list[float] = []
    for rows in by_player.values():
        rows_sorted = sorted(rows, key=lambda row: (int(row.get("frame_idx", 0)), float(row.get("t") or 0.0)))
        for prev, cur in zip(rows_sorted, rows_sorted[1:]):
            distances.append(_distance(prev["world_xy"], cur["world_xy"]))
    return distances


def _put_metric(
    live_metrics: dict[str, dict[str, Any]],
    name: str,
    baseline: float | int,
    candidate: float | int,
    unit: str,
) -> None:
    baseline_value = float(baseline)
    candidate_value = float(candidate)
    live_metrics[name] = {
        "status": "live",
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": candidate_value - baseline_value,
        "unit": unit,
    }


def _process_video_command(
    *,
    clip: str,
    video_path: Path | None,
    candidate_calibration_path: Path,
    tracks_path: Path | None,
    ball_track_path: Path | None,
    out_dir: Path,
) -> str:
    run_out = out_dir / f"{_safe_name(clip)}_candidate_pipeline"
    parts = [
        ".venv/bin/python",
        "scripts/racketsport/process_video.py",
        "--video",
        str(video_path) if video_path is not None else "<same_clip_video>",
        "--clip",
        clip,
        "--court-calibration",
        str(candidate_calibration_path),
    ]
    if tracks_path is not None:
        parts.extend(["--tracks", str(tracks_path)])
    if ball_track_path is not None:
        parts.extend(["--ball-track", str(ball_track_path)])
    parts.extend(["--out", str(run_out), "--force"])
    return " ".join(parts)


def _ball_arc_command(
    *,
    clip: str,
    candidate_calibration_path: Path,
    ball_track_path: Path | None,
    out_dir: Path,
) -> str:
    ball_track_arg = str(ball_track_path) if ball_track_path is not None else "<ball_track.json>"
    return " ".join(
        [
            ".venv/bin/python",
            "scripts/racketsport/solve_ball_arcs.py",
            "--clip",
            clip,
            "--ball-track",
            ball_track_arg,
            "--court-calibration",
            str(candidate_calibration_path),
            "--out-dir",
            str(out_dir / f"{_safe_name(clip)}_candidate_ball_arc"),
        ]
    )


def _read_json_object(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _infer_clip(tracks: Mapping[str, Any] | None, placement: Mapping[str, Any] | None, baseline_path: Path) -> str:
    for payload in (tracks, placement):
        if isinstance(payload, Mapping):
            for key in ("clip", "clip_id"):
                value = payload.get(key)
                if value:
                    return str(value)
    return baseline_path.parent.name or "unknown_clip"


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value[idx]) for idx in range(4)]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _xy(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]


def _frame_index(frame: Mapping[str, Any], fps: float) -> int:
    raw = _maybe_number(frame.get("frame_idx"))
    if raw is not None:
        return int(round(raw))
    t = _maybe_number(frame.get("t"))
    if t is None:
        return 0
    return int(round(t * fps))


def _maybe_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mean_axis(points: Sequence[Sequence[float]], axis: int) -> float:
    if not points:
        return 0.0
    return sum(float(point[axis]) for point in points) / float(len(points))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / float(len(values))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _percentile(values: Iterable[float], percentile: float) -> float:
    vals = sorted(float(value) for value in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return vals[int(rank)]
    frac = rank - low
    return vals[low] * (1.0 - frac) + vals[high] * frac


def _safe_name(value: str) -> str:
    safe = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value]
    return "".join(safe).strip("_") or "unknown_clip"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
