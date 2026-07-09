"""BODY+COURT-only post-hoc match stats from banked pipeline artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_match_stats"
HEATMAP_COLUMNS = 6
HEATMAP_ROWS = 12
BASELINE_BAND_M = 1.5
TELEPORT_SPEED_MPS = 9.0
TELEPORT_DISTANCE_M = 3.0
MAX_SEGMENT_GAP_MULTIPLIER = 2.5
ZONE_NAMES = ("baseline", "kitchen", "out_of_court", "transition")
EXCLUDED_STATS = [
    "shot_counts",
    "rally_stats",
    "contact_stats",
    "ball_speed",
    "paddle_contact",
]


def compute_match_stats_for_run_dir(run_dir: str | Path) -> dict[str, Any]:
    """Compute post-hoc stats from placement/skeleton/court artifacts.

    Consumed source shapes are the repo schemas for `placement.json`
    (`racketsport_placement` players[].frames[].smoothed_world_xy), `skeleton3d.json`
    (`racketsport_skeleton3d` BODY presence/trust), `court_zones.json` (metric
    court polygons), `trust_bands.json`, and optional `frame_times.json`.
    """

    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise NotADirectoryError(f"run dir is not a directory: {run_path}")

    placement_path = run_path / "placement.json"
    court_zones_path = run_path / "court_zones.json"
    if not placement_path.is_file():
        raise FileNotFoundError(f"placement.json not found: {placement_path}")
    if not court_zones_path.is_file():
        raise FileNotFoundError(f"court_zones.json not found: {court_zones_path}")

    placement = _read_json_object(placement_path)
    court_zones_payload = _read_json_object(court_zones_path)
    court_zones = _zones(court_zones_payload)
    skeleton_path = run_path / "skeleton3d.json"
    skeleton = _read_json_object(skeleton_path) if skeleton_path.is_file() else None
    trust_bands_path = run_path / "trust_bands.json"
    trust_bands = _read_json_object(trust_bands_path) if trust_bands_path.is_file() else {}
    frame_times_path = run_path / "frame_times.json"
    frame_times = _read_json_object(frame_times_path) if frame_times_path.is_file() else None

    fps = _positive_float(placement.get("fps")) or _positive_float((skeleton or {}).get("fps")) or 30.0
    total_frames = _total_frames(frame_times, placement=placement)
    position_trust = _position_trust_band(trust_bands, skeleton_present=skeleton is not None)
    court_trust = _trust_band_or_missing(trust_bands.get("court"), stage="CAL")
    stats_trust_bands = {"position": position_trust, "court": court_trust}

    players = [
        _player_stats(
            player,
            fps=fps,
            total_frames=total_frames,
            court_zones=court_zones,
            trust_bands=stats_trust_bands,
        )
        for player in _players(placement)
    ]
    players = [player for player in players if player is not None]

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "source_run_dir": str(run_path),
        "inputs": {
            "placement": str(placement_path),
            "skeleton3d": str(skeleton_path) if skeleton_path.is_file() else None,
            "court_zones": str(court_zones_path),
            "trust_bands": str(trust_bands_path) if trust_bands_path.is_file() else None,
            "frame_times": str(frame_times_path) if frame_times_path.is_file() else None,
            "ball": None,
            "paddle": None,
        },
        "policy": {
            "body_court_only": True,
            "ball_paddle_stats_excluded": True,
            "post_hoc_consumer_only": True,
        },
        "alignment": {
            "shot_rules_v0_zone_vocabulary": ["kitchen", "transition", "baseline", "out_of_court"],
            "zone_geometry_source": "rally_metrics.py: kitchen=near/far_nvz, baseline=within 1.5m of baseline, transition=remaining in-court",
        },
        "fps": fps,
        "frame_count_total": total_frames,
        "player_count": len(players),
        "excluded_stats": list(EXCLUDED_STATS),
        "players": players,
        "summary": _summary(players, total_frames=total_frames),
    }


def write_match_stats_json(payload: Mapping[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _player_stats(
    player: Mapping[str, Any],
    *,
    fps: float,
    total_frames: int,
    court_zones: Mapping[str, Any],
    trust_bands: Mapping[str, Any],
) -> dict[str, Any] | None:
    player_id = player.get("id")
    if player_id is None:
        return None
    frames = sorted(_valid_frames(player), key=lambda frame: (frame["t"], frame["frame_idx"]))
    frames_used = len(frames)
    coverage = _coverage(frames_used, total_frames)
    distance, speeds, speed_segment_count, world_jumps = _distance_speeds_and_jumps(frames, fps=fps)
    zone_counts = {name: 0 for name in ZONE_NAMES}
    for frame in frames:
        zone_counts[_zone_for_point(frame["xy"], court_zones)] += 1
    zone_seconds = {zone: round(count / fps, 6) for zone, count in zone_counts.items()}
    zone_fractions = {zone: round(count / frames_used, 6) if frames_used else 0.0 for zone, count in zone_counts.items()}

    heatmap_points = [frame["xy"] for frame in frames if _point_in_named_zone(frame["xy"], court_zones, "court")]
    balance_points = heatmap_points

    speed_total_segments = max(total_frames - 1, 0)
    distance_metric = _metric(
        round(distance, 6),
        "m",
        frames_used,
        total_frames,
        trust_bands,
    )
    speed_metric = _metric(
        {
            "p50": round(_percentile(speeds, 50.0), 6) if speeds else 0.0,
            "p95": round(_percentile(speeds, 95.0), 6) if speeds else 0.0,
        },
        "mps",
        speed_segment_count,
        speed_total_segments,
        trust_bands,
    )
    zones_metric = _metric(zone_seconds, "s", frames_used, total_frames, trust_bands)
    zones_metric["fractions"] = zone_fractions
    heatmap_metric = _metric(
        _heatmap(heatmap_points, court_zones),
        "count",
        len(heatmap_points),
        total_frames,
        trust_bands,
    )
    balance_metric = _metric(
        _left_right_balance(balance_points),
        "fraction",
        len(balance_points),
        total_frames,
        trust_bands,
    )

    duration_s = total_frames / fps if fps > 0 else 0.0
    plausible_distance_limit = TELEPORT_SPEED_MPS * duration_s
    return {
        "player_id": int(player_id),
        "source": "placement.json",
        "source_frames_used": frames_used,
        "source_frames_total": total_frames,
        "coverage_fraction": coverage,
        "stats": {
            "distance_covered_m": distance_metric,
            "movement_speed_distribution_mps": speed_metric,
            "court_coverage_heatmap": heatmap_metric,
            "time_in_zone_s": zones_metric,
            "left_right_court_balance": balance_metric,
        },
        "sanity": {
            "distance_plausible_for_clip_duration": distance <= plausible_distance_limit,
            "distance_plausible_limit_m": round(plausible_distance_limit, 6),
            "world_jump_count": len(world_jumps),
            "world_jumps": world_jumps,
            "zone_fraction_sum": round(sum(zone_counts.values()) / frames_used, 6) if frames_used else 0.0,
        },
    }


def _valid_frames(player: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = player.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return []
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(frames):
        if not isinstance(raw, Mapping):
            continue
        xy = _xy_from_frame(raw)
        t = _nonnegative_float(raw.get("t"))
        if xy is None or t is None:
            continue
        frame_idx = _int_or_default(raw.get("frame_idx"), idx)
        out.append({"frame_idx": frame_idx, "t": t, "xy": xy})
    return out


def _xy_from_frame(frame: Mapping[str, Any]) -> tuple[float, float] | None:
    for key in ("smoothed_world_xy", "fused_world_xy", "world_xy", "original_world_xy", "track_world_xy"):
        value = frame.get(key)
        xy = _xy(value)
        if xy is not None:
            return xy
    transl = frame.get("transl_world")
    if isinstance(transl, Sequence) and not isinstance(transl, (str, bytes)) and len(transl) >= 2:
        return _xy(transl[:2])
    return None


def _distance_speeds_and_jumps(
    frames: Sequence[Mapping[str, Any]],
    *,
    fps: float,
) -> tuple[float, list[float], int, list[dict[str, Any]]]:
    distance = 0.0
    speeds: list[float] = []
    speed_segment_count = 0
    jumps: list[dict[str, Any]] = []
    max_gap_s = MAX_SEGMENT_GAP_MULTIPLIER / fps if fps > 0 else 0.0
    for previous, current in zip(frames, frames[1:]):
        dt = float(current["t"]) - float(previous["t"])
        if dt <= 0.0 or (max_gap_s > 0.0 and dt > max_gap_s):
            continue
        step = math.dist(previous["xy"], current["xy"])
        speed = step / dt
        if step > TELEPORT_DISTANCE_M or speed > TELEPORT_SPEED_MPS:
            jumps.append(
                {
                    "from_frame_idx": int(previous["frame_idx"]),
                    "to_frame_idx": int(current["frame_idx"]),
                    "dt_s": round(dt, 6),
                    "distance_m": round(step, 6),
                    "speed_mps": round(speed, 6),
                    "threshold_speed_mps": TELEPORT_SPEED_MPS,
                    "threshold_distance_m": TELEPORT_DISTANCE_M,
                }
            )
            continue
        distance += step
        speeds.append(speed)
        speed_segment_count += 1
    return distance, speeds, speed_segment_count, jumps


def _heatmap(points: Sequence[tuple[float, float]], court_zones: Mapping[str, Any]) -> dict[str, Any]:
    bounds = _court_bounds(court_zones) or (-3.048, 3.048, -6.7056, 6.7056)
    min_x, max_x, min_y, max_y = bounds
    x_edges = _edges(min_x, max_x, HEATMAP_COLUMNS)
    y_edges = _edges(min_y, max_y, HEATMAP_ROWS)
    counts = [[0 for _ in range(HEATMAP_COLUMNS)] for _ in range(HEATMAP_ROWS)]
    for x, y in points:
        col = _bin_index(x, x_edges)
        row = _bin_index(y, y_edges)
        if row is not None and col is not None:
            counts[row][col] += 1
    total = sum(sum(row) for row in counts)
    fractions = [
        [round(count / total, 6) if total else 0.0 for count in row]
        for row in counts
    ]
    return {
        "coordinate_frame": "court_Z0_metric_xy_m",
        "columns": HEATMAP_COLUMNS,
        "rows": HEATMAP_ROWS,
        "x_edges_m": [round(value, 6) for value in x_edges],
        "y_edges_m": [round(value, 6) for value in y_edges],
        "counts": counts,
        "fractions": fractions,
        "total_count": total,
    }


def _left_right_balance(points: Sequence[tuple[float, float]]) -> dict[str, float]:
    if not points:
        return {"left_fraction": 0.0, "right_fraction": 0.0}
    left = sum(1 for x, _y in points if x < 0.0)
    right = len(points) - left
    return {
        "left_fraction": round(left / len(points), 6),
        "right_fraction": round(right / len(points), 6),
    }


def _zone_for_point(point: tuple[float, float], court_zones: Mapping[str, Any]) -> str:
    if not _point_in_named_zone(point, court_zones, "court"):
        return "out_of_court"
    if _point_in_named_zone(point, court_zones, "near_nvz") or _point_in_named_zone(point, court_zones, "far_nvz"):
        return "kitchen"
    bounds = _court_bounds(court_zones)
    if bounds is not None:
        _min_x, _max_x, min_y, max_y = bounds
        if point[1] <= min_y + BASELINE_BAND_M or point[1] >= max_y - BASELINE_BAND_M:
            return "baseline"
    return "transition"


def _metric(
    value: Any,
    unit: str,
    frames_used: int,
    frames_total: int,
    trust_bands: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "frames_used": frames_used,
        "frames_total": frames_total,
        "coverage_fraction": _coverage(frames_used, frames_total),
        "trust_bands": {
            "position": dict(trust_bands["position"]),
            "court": dict(trust_bands["court"]),
        },
    }


def _summary(players: Sequence[Mapping[str, Any]], *, total_frames: int) -> dict[str, Any]:
    distances = [
        float(((player.get("stats") or {}).get("distance_covered_m") or {}).get("value") or 0.0)
        for player in players
    ]
    world_jump_count = sum(int((player.get("sanity") or {}).get("world_jump_count") or 0) for player in players)
    return {
        "player_count": len(players),
        "frame_count_total": total_frames,
        "distance_covered_m_by_player": {
            str(player.get("player_id")): ((player.get("stats") or {}).get("distance_covered_m") or {}).get("value")
            for player in players
        },
        "max_distance_covered_m": round(max(distances), 6) if distances else 0.0,
        "world_jump_count": world_jump_count,
        "excluded_ball_paddle_stats": list(EXCLUDED_STATS),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _coverage(frames_used: int, frames_total: int) -> float:
    return round(frames_used / frames_total, 6) if frames_total > 0 else 0.0


def _position_trust_band(trust_bands: Mapping[str, Any], *, skeleton_present: bool) -> dict[str, Any]:
    key = "body" if skeleton_present else "track"
    stage = "BODY" if skeleton_present else "TRK"
    return _trust_band_or_missing(trust_bands.get(key), stage=stage)


def _trust_band_or_missing(value: Any, *, stage: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "stage": stage,
        "gate_id": "missing_trust_band",
        "gate_status": "missing",
        "badge": "low_confidence",
        "reason": f"{stage} trust band was missing from source artifacts; stats remain low confidence.",
        "evidence_path": None,
    }


def _total_frames(frame_times: Mapping[str, Any] | None, *, placement: Mapping[str, Any]) -> int:
    max_frame = -1
    max_player_frame_count = 0
    total_observed = 0
    for player in _players(placement):
        frames = _valid_frames(player)
        max_player_frame_count = max(max_player_frame_count, len(frames))
        for frame in frames:
            total_observed += 1
            max_frame = max(max_frame, int(frame["frame_idx"]))
    observed_total = max(max_frame + 1 if max_frame >= 0 else 0, max_player_frame_count, total_observed if not _players(placement) else 0)
    if frame_times is not None:
        frame_count = _int_or_none(frame_times.get("frame_count"))
        if frame_count is not None and frame_count > 0:
            return max(frame_count, observed_total)
    return observed_total


def _zones(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    zones = payload.get("zones")
    if not isinstance(zones, Mapping):
        raise ValueError("court_zones.json missing zones object")
    return zones


def _players(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    players = payload.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return []
    return [player for player in players if isinstance(player, Mapping)]


def _court_bounds(court_zones: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    polygon = court_zones.get("court")
    if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)):
        return None
    points = [_xy(point) for point in polygon]
    points = [point for point in points if point is not None]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def _point_in_named_zone(point: tuple[float, float], court_zones: Mapping[str, Any], name: str) -> bool:
    polygon = court_zones.get(name)
    return isinstance(polygon, Sequence) and not isinstance(polygon, (str, bytes)) and _point_in_polygon(point, polygon)


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[Any]) -> bool:
    vertices = [_xy(vertex) for vertex in polygon]
    vertices = [vertex for vertex in vertices if vertex is not None]
    if len(vertices) < 3:
        return False
    x, y = point
    inside = False
    count = len(vertices)
    for idx in range(count):
        x1, y1 = vertices[idx]
        x2, y2 = vertices[(idx + 1) % count]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
    return inside


def _point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9


def _edges(start: float, stop: float, bins: int) -> list[float]:
    width = (stop - start) / bins
    return [start + idx * width for idx in range(bins + 1)]


def _bin_index(value: float, edges: Sequence[float]) -> int | None:
    if len(edges) < 2 or value < edges[0] or value > edges[-1]:
        return None
    if value == edges[-1]:
        return len(edges) - 2
    for idx in range(len(edges) - 1):
        if edges[idx] <= value < edges[idx + 1]:
            return idx
    return None


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    x = _finite_float(value[0])
    y = _finite_float(value[1])
    if x is None or y is None:
        return None
    return x, y


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _positive_float(value: Any) -> float | None:
    numeric = _finite_float(value)
    if numeric is None or numeric <= 0.0:
        return None
    return numeric


def _nonnegative_float(value: Any) -> float | None:
    numeric = _finite_float(value)
    if numeric is None or numeric < 0.0:
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed is not None and parsed >= 0 else default
