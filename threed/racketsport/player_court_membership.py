"""Preview-only target-court membership scoring for tracked players."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .court_calibration import project_image_points_to_world, project_planar_points
from .court_templates import get_court_template


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_player_court_membership"
COURT_HALF_WIDTH_M = 3.048
COURT_BASELINE_Y_M = 6.7056
COURT_KITCHEN_Y_M = 2.1336


@dataclass(frozen=True)
class PlayerCourtMembershipConfig:
    lateral_apron_m: float = 0.6
    longitudinal_apron_m: float = 2.5
    far_median_y_margin_m: float = 0.15
    median_x_margin_m: float = 0.30
    inside_strict_adjacent_threshold: float = 0.20
    inside_asym_on_target_threshold: float = 0.80
    far_boundary_baseline_band_m: float = 0.75
    kitchen_approach_buffer_m: float = 1.5
    stationary_speed_mps: float = 0.25
    min_frames_for_on_target: int = 3


def compute_player_court_membership(
    tracks_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    camera_motion_payload: Mapping[str, Any] | None = None,
    config: PlayerCourtMembershipConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify each tracked person as target-court, adjacent/spectator, or uncertain.

    This is a preview/advisory filter. It uses bbox-bottom image points, optionally
    compensates camera motion into the calibration reference frame, and fails closed
    in the output vocabulary with ``verified=false``.
    """

    cfg = _config(config)
    homography = _homography(calibration_payload)
    fps = _fps(tracks_payload)
    motion_by_frame = _camera_motion_by_frame(camera_motion_payload)
    per_player: dict[str, dict[str, Any]] = {}
    total_compensated = 0
    total_uncompensated = 0

    for player in _players(tracks_payload):
        player_id = _player_id(player)
        if player_id is None:
            continue
        samples = _world_samples(
            player,
            homography=homography,
            fps=fps,
            motion_by_frame=motion_by_frame,
        )
        metrics = _player_metrics(samples, fps=fps, cfg=cfg)
        per_player[str(player_id)] = metrics
        total_compensated += int(metrics["n_compensated_frames_used"])
        total_uncompensated += int(metrics["n_uncompensated_frames_used"])

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": {
            "id": _clip_id(tracks_payload),
            "fps": fps,
        },
        "calibration": {
            "sport": str(calibration_payload.get("sport") or "pickleball"),
            "source": calibration_payload.get("source"),
            "homography_used": True,
        },
        "camera_motion_used": bool(motion_by_frame),
        "n_compensated_frames_used": total_compensated,
        "n_uncompensated_frames_used": total_uncompensated,
        "per_player": dict(sorted(per_player.items(), key=lambda item: int(item[0]))),
        "thresholds": _thresholds(cfg),
        "verified": False,
        "not_gate_verified": True,
    }


def write_membership_json(payload: Mapping[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_membership_evidence(
    *,
    membership_payload: Mapping[str, Any],
    tracks_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    camera_motion_payload: Mapping[str, Any] | None,
    video_path: str | Path,
    evidence_dir: str | Path,
) -> dict[str, Any]:
    """Write deterministic preview evidence images for non-target players."""

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment-specific guard
        raise RuntimeError("evidence writing requires opencv-python and numpy") from exc

    out_dir = Path(evidence_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = _fps(tracks_payload)
    motion_by_frame = _camera_motion_by_frame(camera_motion_payload)
    homography = _homography(calibration_payload)
    written: list[str] = []
    players_by_id = {
        str(player_id): player
        for player in _players(tracks_payload)
        if (player_id := _player_id(player)) is not None
    }

    for player_id, player_metrics in sorted(
        _per_player(membership_payload).items(),
        key=lambda item: int(item[0]),
    ):
        if player_metrics.get("verdict") == "on_target_court":
            continue
        player = players_by_id.get(str(player_id))
        if player is None:
            continue
        frame_by_idx = {
            _frame_idx(frame, fps=fps): frame
            for frame in _frames(player)
            if _frame_idx(frame, fps=fps) is not None
        }
        representative = [
            int(frame_idx)
            for frame_idx in player_metrics.get("representative_frame_indices", [])
            if int(frame_idx) in frame_by_idx
        ][:2]
        for frame_idx in representative:
            image = _read_frame(cap, frame_idx)
            if image is None:
                continue
            bbox = _bbox(frame_by_idx[frame_idx])
            if bbox is None:
                continue
            crop, offset = _crop_with_padding(image, bbox, padding_px=24)
            x1, y1, x2, y2 = bbox
            ox, oy = offset
            color = _verdict_color_bgr(str(player_metrics.get("verdict")))
            cv2.rectangle(
                crop,
                (int(round(x1 - ox)), int(round(y1 - oy))),
                (int(round(x2 - ox)), int(round(y2 - oy))),
                color,
                2,
            )
            cv2.putText(
                crop,
                f"p{player_id} {player_metrics.get('verdict')}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            out = out_dir / f"player_{int(player_id):03d}_{player_metrics.get('verdict')}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out), crop)
            written.append(out.as_posix())

    overlay = _read_frame(cap, _reference_frame_idx(camera_motion_payload) or 0)
    if overlay is None:
        overlay = _read_frame(cap, 0)
    if overlay is not None:
        _draw_reference_court_overlay(overlay, homography, calibration_payload, cv2=cv2, np=np)
        for player in _players(tracks_payload):
            player_id = _player_id(player)
            if player_id is None:
                continue
            verdict = str(_per_player(membership_payload).get(str(player_id), {}).get("verdict") or "uncertain")
            for frame in _frames(player):
                bbox = _bbox(frame)
                frame_idx = _frame_idx(frame, fps=fps)
                if bbox is None or frame_idx is None:
                    continue
                point = _bbox_bottom(bbox)
                motion = motion_by_frame.get(frame_idx)
                if motion is not None and motion["compensated"]:
                    point = _apply_homography(motion["M"], point)
                cv2.circle(
                    overlay,
                    (int(round(point[0])), int(round(point[1]))),
                    3,
                    _verdict_color_bgr(verdict),
                    -1,
                )
        out = out_dir / "player_membership_compensated_court_overlay.jpg"
        cv2.imwrite(str(out), overlay)
        written.append(out.as_posix())

    cap.release()
    return {"evidence_dir": out_dir.as_posix(), "files": written}


def _config(value: PlayerCourtMembershipConfig | Mapping[str, Any] | None) -> PlayerCourtMembershipConfig:
    if value is None:
        return PlayerCourtMembershipConfig()
    if isinstance(value, PlayerCourtMembershipConfig):
        return value
    defaults = asdict(PlayerCourtMembershipConfig())
    for key, raw in value.items():
        if key in defaults:
            defaults[key] = raw
    return PlayerCourtMembershipConfig(**defaults)


def _thresholds(cfg: PlayerCourtMembershipConfig) -> dict[str, Any]:
    return {
        "half_width_x_m": COURT_HALF_WIDTH_M,
        "baseline_y_m": COURT_BASELINE_Y_M,
        "kitchen_y_m": COURT_KITCHEN_Y_M,
        **asdict(cfg),
    }


def _homography(calibration_payload: Mapping[str, Any]) -> list[list[float]]:
    raw = calibration_payload.get("homography")
    if not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError("calibration payload must contain a 3x3 homography")
    matrix = [[float(value) for value in row] for row in raw]
    if any(len(row) != 3 for row in matrix):
        raise ValueError("calibration payload must contain a 3x3 homography")
    return matrix


def _fps(tracks_payload: Mapping[str, Any]) -> float:
    try:
        fps = float(tracks_payload.get("fps") or 30.0)
    except (TypeError, ValueError):
        fps = 30.0
    return fps if fps > 0.0 else 30.0


def _clip_id(tracks_payload: Mapping[str, Any]) -> str | None:
    for key in ("clip", "clip_id", "video", "source_video"):
        value = tracks_payload.get(key)
        if value is not None:
            return str(value)
    return None


def _players(tracks_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    players = tracks_payload.get("players")
    return [player for player in players if isinstance(player, Mapping)] if isinstance(players, list) else []


def _player_id(player: Mapping[str, Any]) -> int | None:
    try:
        return int(player.get("id"))
    except (TypeError, ValueError):
        return None


def _frames(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = player.get("frames")
    return [frame for frame in frames if isinstance(frame, Mapping)] if isinstance(frames, list) else []


def _world_samples(
    player: Mapping[str, Any],
    *,
    homography: Sequence[Sequence[float]],
    fps: float,
    motion_by_frame: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for frame in _frames(player):
        bbox = _bbox(frame)
        frame_idx = _frame_idx(frame, fps=fps)
        if bbox is None or frame_idx is None:
            continue
        bottom = _bbox_bottom(bbox)
        motion = motion_by_frame.get(frame_idx)
        compensated = bool(motion is not None and motion["compensated"])
        reference_point = _apply_homography(motion["M"], bottom) if compensated else bottom
        world = project_image_points_to_world(homography, [reference_point])[0]
        samples.append(
            {
                "frame_idx": int(frame_idx),
                "t": _frame_time(frame, frame_idx=frame_idx, fps=fps),
                "bbox": bbox,
                "image_xy": [_round_metric(bottom[0]), _round_metric(bottom[1])],
                "reference_xy": [_round_metric(reference_point[0]), _round_metric(reference_point[1])],
                "x": float(world[0]),
                "y": float(world[1]),
                "compensated": compensated,
            }
        )
    return sorted(samples, key=lambda sample: (sample["frame_idx"], sample["t"]))


def _player_metrics(samples: list[Mapping[str, Any]], *, fps: float, cfg: PlayerCourtMembershipConfig) -> dict[str, Any]:
    xs = [float(sample["x"]) for sample in samples]
    ys = [float(sample["y"]) for sample in samples]
    abs_ys = [abs(value) for value in ys]
    speeds = _speeds(samples, fps=fps)
    n_frames = len(samples)
    median_x = _percentile(xs, 50)
    median_y = _percentile(ys, 50)
    own_side = -1 if median_y < 0.0 else 1
    inside_strict = [
        abs(x) <= COURT_HALF_WIDTH_M and abs(y) <= COURT_BASELINE_Y_M
        for x, y in zip(xs, ys, strict=True)
    ]
    inside_asym = [_inside_asym(x, y, own_side=own_side, cfg=cfg) for x, y in zip(xs, ys, strict=True)]
    inside_strict_frac = _fraction(inside_strict)
    inside_asym_frac = _fraction(inside_asym)
    abs_y_p10 = _percentile(abs_ys, 10)
    min_abs_y = min(abs_ys) if abs_ys else 0.0
    far_boundary_camper = (
        median_y > 0.0
        and abs_y_p10 > COURT_BASELINE_Y_M - cfg.far_boundary_baseline_band_m
        and min_abs_y > COURT_KITCHEN_Y_M + cfg.kitchen_approach_buffer_m
    )
    median_inside_asym = _inside_asym(median_x, median_y, own_side=own_side, cfg=cfg)
    near_asym_exempt = own_side < 0 and inside_asym_frac >= cfg.inside_asym_on_target_threshold and median_inside_asym

    reasons: list[str] = []
    if inside_strict_frac < cfg.inside_strict_adjacent_threshold and not near_asym_exempt:
        reasons.append("inside_strict_frac_below_threshold")
    if median_y > COURT_BASELINE_Y_M + cfg.far_median_y_margin_m:
        reasons.append("median_y_beyond_far_baseline_margin")
    if median_y < -COURT_BASELINE_Y_M - cfg.longitudinal_apron_m:
        reasons.append("median_y_beyond_near_asym_apron")
    if abs(median_x) > COURT_HALF_WIDTH_M + cfg.median_x_margin_m:
        reasons.append("median_x_beyond_sideline_margin")
    if far_boundary_camper:
        reasons.append("far_boundary_camper")

    if reasons:
        verdict = "adjacent_or_spectator"
    elif n_frames < cfg.min_frames_for_on_target:
        verdict = "uncertain"
        reasons.append("too_few_frames_for_on_target")
    elif inside_asym_frac >= cfg.inside_asym_on_target_threshold and median_inside_asym and not far_boundary_camper:
        verdict = "on_target_court"
    else:
        verdict = "uncertain"
        if inside_asym_frac < cfg.inside_asym_on_target_threshold:
            reasons.append("inside_asym_frac_below_on_target_threshold")
        if not median_inside_asym:
            reasons.append("median_outside_asym_court")

    frame_indices = [int(sample["frame_idx"]) for sample in samples]
    representative = _representative_frames(frame_indices)
    return {
        "verdict": verdict,
        "reasons": reasons,
        "n_frames": n_frames,
        "n_compensated_frames_used": sum(1 for sample in samples if sample.get("compensated") is True),
        "n_uncompensated_frames_used": sum(1 for sample in samples if sample.get("compensated") is not True),
        "inside_strict_frac": _round_fraction(inside_strict_frac),
        "inside_asym_frac": _round_fraction(inside_asym_frac),
        "median_x_m": _round_metric(median_x),
        "x_p10_m": _round_metric(_percentile(xs, 10)),
        "x_p90_m": _round_metric(_percentile(xs, 90)),
        "median_y_m": _round_metric(median_y),
        "y_p10_m": _round_metric(_percentile(ys, 10)),
        "y_p90_m": _round_metric(_percentile(ys, 90)),
        "min_abs_y_m": _round_metric(min_abs_y),
        "abs_y_p10_m": _round_metric(abs_y_p10),
        "speed_p50_mps": _round_metric(_percentile(speeds, 50)),
        "speed_p90_mps": _round_metric(_percentile(speeds, 90)),
        "stationary_frac": _round_fraction(_fraction([speed < cfg.stationary_speed_mps for speed in speeds])),
        "own_side": "near" if own_side < 0 else "far",
        "far_boundary_camper": far_boundary_camper,
        "representative_frame_indices": representative,
    }


def _inside_asym(x: float, y: float, *, own_side: int, cfg: PlayerCourtMembershipConfig) -> bool:
    if abs(x) > COURT_HALF_WIDTH_M + cfg.lateral_apron_m:
        return False
    if own_side < 0:
        return -COURT_BASELINE_Y_M - cfg.longitudinal_apron_m <= y <= COURT_BASELINE_Y_M
    return -COURT_BASELINE_Y_M <= y <= COURT_BASELINE_Y_M + cfg.longitudinal_apron_m


def _speeds(samples: list[Mapping[str, Any]], *, fps: float) -> list[float]:
    speeds: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        previous_t = float(previous["t"])
        current_t = float(current["t"])
        dt = current_t - previous_t
        if dt <= 0.0:
            dt = (int(current["frame_idx"]) - int(previous["frame_idx"])) / fps
        if dt <= 0.0:
            continue
        speeds.append(math.hypot(float(current["x"]) - float(previous["x"]), float(current["y"]) - float(previous["y"])) / dt)
    return speeds


def _camera_motion_by_frame(camera_motion_payload: Mapping[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not isinstance(camera_motion_payload, Mapping):
        return {}
    frames = camera_motion_payload.get("frames")
    if not isinstance(frames, list):
        return {}
    by_frame: dict[int, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        try:
            frame_idx = int(frame.get("frame_idx"))
        except (TypeError, ValueError):
            continue
        matrix = _motion_matrix(frame.get("M"))
        if matrix is None:
            continue
        by_frame[frame_idx] = {"M": matrix, "compensated": frame.get("compensated") is True}
    return by_frame


def _motion_matrix(raw: Any) -> list[list[float]] | None:
    if not isinstance(raw, Sequence) or len(raw) != 3:
        return None
    matrix = [[float(value) for value in row] for row in raw]
    return matrix if all(len(row) == 3 for row in matrix) else None


def _bbox(frame: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw = frame.get("bbox")
    if not isinstance(raw, Sequence) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_bottom(bbox: Sequence[float]) -> tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[3]))


def _frame_idx(frame: Mapping[str, Any], *, fps: float) -> int | None:
    for key in ("frame_idx", "frame_index", "frame"):
        if key in frame:
            try:
                return int(frame[key])
            except (TypeError, ValueError):
                return None
    if "t" in frame:
        try:
            return int(round(float(frame["t"]) * fps))
        except (TypeError, ValueError):
            return None
    return None


def _frame_time(frame: Mapping[str, Any], *, frame_idx: int, fps: float) -> float:
    try:
        return float(frame.get("t"))
    except (TypeError, ValueError):
        return float(frame_idx) / fps


def _apply_homography(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    denom = float(matrix[2][0]) * x + float(matrix[2][1]) * y + float(matrix[2][2])
    if math.isclose(denom, 0.0):
        raise ValueError("camera motion homography has zero scale for point")
    out_x = (float(matrix[0][0]) * x + float(matrix[0][1]) * y + float(matrix[0][2])) / denom
    out_y = (float(matrix[1][0]) * x + float(matrix[1][1]) * y + float(matrix[1][2])) / denom
    return out_x, out_y


def _percentile(values: Sequence[float], percentile: float) -> float:
    cleaned = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not cleaned:
        return 0.0
    if len(cleaned) == 1:
        return cleaned[0]
    rank = (len(cleaned) - 1) * (float(percentile) / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return cleaned[lower]
    weight = rank - lower
    return cleaned[lower] * (1.0 - weight) + cleaned[upper] * weight


def _fraction(values: Sequence[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _round_metric(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if abs(rounded) < 1e-9 else rounded


def _round_fraction(value: float) -> float:
    return round(float(value), 6)


def _representative_frames(frame_indices: Sequence[int]) -> list[int]:
    unique = sorted(set(int(frame_idx) for frame_idx in frame_indices))
    if not unique:
        return []
    median = unique[len(unique) // 2]
    return [unique[0]] if median == unique[0] else [unique[0], median]


def _per_player(payload: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    per_player = payload.get("per_player")
    return per_player if isinstance(per_player, Mapping) else {}


def _reference_frame_idx(camera_motion_payload: Mapping[str, Any] | None) -> int | None:
    if not isinstance(camera_motion_payload, Mapping):
        return None
    try:
        return int(camera_motion_payload.get("reference_frame_idx"))
    except (TypeError, ValueError):
        return None


def _read_frame(cap: Any, frame_idx: int) -> Any | None:
    cap.set(1, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _crop_with_padding(image: Any, bbox: Sequence[float], *, padding_px: int) -> tuple[Any, tuple[int, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, int(math.floor(x1)) - padding_px)
    top = max(0, int(math.floor(y1)) - padding_px)
    right = min(width, int(math.ceil(x2)) + padding_px)
    bottom = min(height, int(math.ceil(y2)) + padding_px)
    return image[top:bottom, left:right].copy(), (left, top)


def _draw_reference_court_overlay(
    image: Any,
    homography: Sequence[Sequence[float]],
    calibration_payload: Mapping[str, Any],
    *,
    cv2: Any,
    np: Any,
) -> None:
    sport = str(calibration_payload.get("sport") or "pickleball")
    template = get_court_template(sport)
    for start, end in template.line_segments_m.values():
        p0, p1 = project_planar_points(homography, [start, end])
        cv2.line(
            image,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (255, 255, 255),
            2,
        )
    image[:] = np.clip(image.astype(np.float32) * 0.92, 0, 255).astype(image.dtype)


def _verdict_color_bgr(verdict: str) -> tuple[int, int, int]:
    if verdict == "on_target_court":
        return (40, 180, 40)
    if verdict == "adjacent_or_spectator":
        return (40, 40, 230)
    return (20, 170, 230)


__all__ = [
    "ARTIFACT_TYPE",
    "PlayerCourtMembershipConfig",
    "compute_player_court_membership",
    "write_membership_evidence",
    "write_membership_json",
]
