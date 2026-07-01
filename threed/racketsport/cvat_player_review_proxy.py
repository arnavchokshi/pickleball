"""Build review-only player track proxies from CVAT person ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .court_calibration import calibration_image_size, project_image_points_to_world
from .doubles_id import assign_doubles_roles
from .schemas import CourtCalibration, PersonGroundTruth, PlayerTrack, TrackFrame, Tracks
from .track_lock import TrackCandidate

PROXY_TRACKS_FILENAME = "cvat_player_review_proxy_tracks.json"
JSON_REPORT_FILENAME = "cvat_player_review_proxy_report.json"
MARKDOWN_REPORT_FILENAME = "CVAT_PLAYER_REVIEW_PROXY_REPORT.md"


@dataclass(frozen=True)
class CvatPlayerReviewProxyResult:
    tracks: Tracks
    report: dict[str, Any]
    markdown: str


def build_cvat_player_review_proxy(
    *,
    ground_truth: PersonGroundTruth,
    calibration: CourtCalibration,
    source_ground_truth_path: str | Path,
    source_calibration_path: str | Path,
    output_tracks_path: str | Path,
    expected_players: int | None = None,
    source_image_size: tuple[float, float] | None = None,
) -> CvatPlayerReviewProxyResult:
    """Convert reviewed CVAT player boxes into a non-promotable review proxy."""

    fps = float(ground_truth.fps or 0.0)
    if fps <= 0.0:
        raise ValueError("person ground truth must include a positive fps")

    coordinate_mapping = _coordinate_mapping(calibration, source_image_size=source_image_size)
    frames_by_track: dict[int, list[TrackFrame]] = {}
    for frame in ground_truth.frames:
        for label in frame.labels:
            if label.ignored or not label.person_class:
                continue
            bbox_xyxy = _bbox_xywh_to_xyxy(label.bbox_xywh)
            world_xy = _bbox_foot_world_xy(calibration, bbox_xyxy, coordinate_mapping=coordinate_mapping)
            frames_by_track.setdefault(int(label.track_id), []).append(
                TrackFrame(
                    t=float(frame.frame_index) / fps,
                    bbox=bbox_xyxy,
                    world_xy=world_xy,
                    conf=float(label.confidence if label.confidence is not None else 1.0),
                )
            )

    players = [
        PlayerTrack(id=track_id, side="unknown", role="unknown", frames=sorted(frames, key=lambda item: item.t))
        for track_id, frames in sorted(frames_by_track.items())
    ]
    players = _with_identity_labels(players)
    tracks = Tracks(schema_version=1, fps=fps, players=players, rally_spans=[])

    report = _build_report(
        ground_truth=ground_truth,
        tracks=tracks,
        source_ground_truth_path=source_ground_truth_path,
        source_calibration_path=source_calibration_path,
        output_tracks_path=output_tracks_path,
        expected_players=expected_players or ground_truth.summary.max_valid_players_per_frame,
        coordinate_mapping=coordinate_mapping,
    )
    return CvatPlayerReviewProxyResult(tracks=tracks, report=report, markdown=render_cvat_player_review_proxy_markdown(report))


def write_cvat_player_review_proxy(
    *,
    out_dir: str | Path,
    result: CvatPlayerReviewProxyResult,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracks_path = out / PROXY_TRACKS_FILENAME
    report_path = out / JSON_REPORT_FILENAME
    markdown_path = out / MARKDOWN_REPORT_FILENAME
    tracks_path.write_text(json.dumps(result.tracks.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(result.markdown, encoding="utf-8")
    return {"tracks": tracks_path, "report": report_path, "markdown": markdown_path}


def render_cvat_player_review_proxy_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    safe = report["safe_outputs"]
    lines = [
        "# CVAT Player Review Proxy",
        "",
        "- status: `review_only_not_ground_truth`",
        "- review only: `true`",
        "- not ground truth: `true`",
        "- not gate verified: `true`",
        "- promote TRK: `false`",
        "- note: this is not TRK promotion and not a detector/tracker output.",
        "",
        "## Coverage",
        "",
        f"- clip: `{report['clip_id']}`",
        f"- CVAT frame range: `{_range_text(coverage['gt_frame_range'])}`",
        f"- labeled frame range: `{_range_text(coverage['gt_labeled_frame_range'])}`",
        f"- proxy prediction frame range: `{_range_text(coverage['proxy_prediction_frame_range'])}`",
        f"- frame count: `{coverage['frame_count']}`",
        f"- labeled frames: `{coverage['gt_labeled_frame_count']}`",
        f"- expected-player frames: `{coverage['expected_player_frame_count']}`",
        f"- proxy detections: `{coverage['proxy_detection_count']}`",
        f"- full-horizon label span: `{str(coverage['full_horizon_label_span']).lower()}`",
        "",
        "## Safety",
        "",
        f"- can overwrite canonical tracks: `{str(safe['can_overwrite_canonical_tracks']).lower()}`",
        f"- can promote TRK: `{str(safe['can_promote_trk']).lower()}`",
        f"- blocker: {report['blocker']}",
        f"- next action: {report['next_action']}",
        "",
        "## Outputs",
        "",
        f"- proxy tracks: `{report['output_tracks_path']}`",
        f"- source GT: `{report['source_person_ground_truth_path']}`",
        f"- source calibration: `{report['source_court_calibration_path']}`",
        "",
    ]
    return "\n".join(lines)


def _build_report(
    *,
    ground_truth: PersonGroundTruth,
    tracks: Tracks,
    source_ground_truth_path: str | Path,
    source_calibration_path: str | Path,
    output_tracks_path: str | Path,
    expected_players: int,
    coordinate_mapping: dict[str, Any],
) -> dict[str, Any]:
    label_counts = {
        frame.frame_index: sum(1 for label in frame.labels if not label.ignored and label.person_class)
        for frame in ground_truth.frames
    }
    gt_frame_indexes = sorted(frame.frame_index for frame in ground_truth.frames)
    labeled_frame_indexes = sorted(frame for frame, count in label_counts.items() if count > 0)
    expected_frame_indexes = sorted(frame for frame, count in label_counts.items() if count == expected_players)
    proxy_frame_indexes = sorted(
        {
            int(round(float(frame.t) * float(tracks.fps)))
            for player in tracks.players
            for frame in player.frames
        }
    )
    frame_count = int(ground_truth.summary.frame_count)
    missing_any_label = [frame for frame, count in sorted(label_counts.items()) if count == 0]
    missing_expected = [frame for frame, count in sorted(label_counts.items()) if count != expected_players]

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_player_review_proxy_report",
        "status": "review_only_not_ground_truth",
        "clip_id": ground_truth.clip_id,
        "review_only": True,
        "not_ground_truth": True,
        "not_gate_verified": True,
        "model_prediction": False,
        "promote_trk": False,
        "source_person_ground_truth_path": str(source_ground_truth_path),
        "source_court_calibration_path": str(source_calibration_path),
        "output_tracks_path": str(output_tracks_path),
        "safe_outputs": {
            "can_write_proxy_tracks": True,
            "proxy_tracks_filename": PROXY_TRACKS_FILENAME,
            "can_write_canonical_tracks": False,
            "can_overwrite_canonical_tracks": False,
            "can_promote_trk": False,
        },
        "coordinate_mapping": coordinate_mapping,
        "coverage": {
            "frame_count": frame_count,
            "fps": tracks.fps,
            "expected_players": expected_players,
            "gt_frame_range": _frame_range(gt_frame_indexes),
            "gt_labeled_frame_range": _frame_range(labeled_frame_indexes),
            "proxy_prediction_frame_range": _frame_range(proxy_frame_indexes),
            "gt_labeled_frame_count": len(labeled_frame_indexes),
            "expected_player_frame_count": len(expected_frame_indexes),
            "missing_any_label_frame_count": len(missing_any_label),
            "missing_expected_player_frame_count": len(missing_expected),
            "missing_any_label_frame_sample": missing_any_label[:20],
            "missing_expected_player_frame_sample": missing_expected[:20],
            "proxy_prediction_frame_count": len(proxy_frame_indexes),
            "proxy_detection_count": sum(len(player.frames) for player in tracks.players),
            "proxy_track_count": len(tracks.players),
            "proxy_track_lengths": {str(player.id): len(player.frames) for player in tracks.players},
            "gt_valid_label_count": ground_truth.summary.valid_label_count,
            "gt_track_ids": ground_truth.summary.track_ids,
            "full_horizon_label_span": bool(labeled_frame_indexes)
            and labeled_frame_indexes[0] == (gt_frame_indexes[0] if gt_frame_indexes else 0)
            and labeled_frame_indexes[-1] == frame_count - 1,
        },
        "blocker": (
            "CVAT label-derived proxy tracks are useful for coverage diagnostics and review-only downstream probes, "
            "but they are not model predictions and cannot satisfy TRK IDF1/spectator/ID-switch/throughput promotion gates."
        ),
        "next_action": (
            "Run or repair a real detector/tracker source across the full Outdoor CVAT 0-1150 horizon, "
            "then score that source against CVAT labels before any TRK promotion review."
        ),
    }


def _bbox_xywh_to_xyxy(bbox_xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) for value in bbox_xywh]
    return (x, y, x + width, y + height)


def _coordinate_mapping(
    calibration: CourtCalibration,
    *,
    source_image_size: tuple[float, float] | None,
) -> dict[str, Any]:
    calibration_width, calibration_height = calibration_image_size(calibration)
    source_width = float(source_image_size[0]) if source_image_size is not None else float(calibration_width)
    source_height = float(source_image_size[1]) if source_image_size is not None else float(calibration_height)
    if source_width <= 0.0 or source_height <= 0.0:
        raise ValueError("source_image_size must be positive when provided")
    return {
        "source_image_size": [float(source_width), float(source_height)],
        "calibration_image_size": [float(calibration_width), float(calibration_height)],
        "image_to_calibration_scale_x": float(calibration_width) / source_width,
        "image_to_calibration_scale_y": float(calibration_height) / source_height,
    }


def _bbox_foot_world_xy(
    calibration: CourtCalibration,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    coordinate_mapping: dict[str, Any],
) -> list[float]:
    x1, _, x2, y2 = bbox_xyxy
    scale_x = float(coordinate_mapping["image_to_calibration_scale_x"])
    scale_y = float(coordinate_mapping["image_to_calibration_scale_y"])
    foot = [((x1 + x2) / 2.0) * scale_x, y2 * scale_y]
    return project_image_points_to_world(calibration.homography, [foot])[0]


def _with_identity_labels(players: list[PlayerTrack]) -> list[PlayerTrack]:
    candidates = [
        TrackCandidate(track_id=player.id, world_xy=_median_world_xy(player), confidence=_mean_conf(player))
        for player in players
        if player.frames
    ]
    if len(candidates) == 4:
        identities = assign_doubles_roles(candidates)
        return [
            player.model_copy(
                update={
                    "side": identities.get(player.id).side if player.id in identities else "unknown",
                    "role": identities.get(player.id).role if player.id in identities else "unknown",
                }
            )
            for player in players
        ]
    return [
        player.model_copy(
            update={
                "side": "near" if _median_world_xy(player)[1] <= 0.0 else "far",
                "role": "singles" if len(players) == 2 else "unknown",
            }
        )
        for player in players
    ]


def _median_world_xy(player: PlayerTrack) -> list[float]:
    if not player.frames:
        return [0.0, 0.0]
    xs = sorted(float(frame.world_xy[0]) for frame in player.frames)
    ys = sorted(float(frame.world_xy[1]) for frame in player.frames)
    mid = len(xs) // 2
    return [xs[mid], ys[mid]]


def _mean_conf(player: PlayerTrack) -> float:
    return sum(float(frame.conf) for frame in player.frames) / len(player.frames) if player.frames else 0.0


def _frame_range(frame_indexes: list[int]) -> dict[str, int | None]:
    if not frame_indexes:
        return {"first": None, "last": None}
    return {"first": frame_indexes[0], "last": frame_indexes[-1]}


def _range_text(value: dict[str, Any]) -> str:
    first = value.get("first")
    last = value.get("last")
    if first is None or last is None:
        return "n/a"
    return f"{first}-{last}"


__all__ = [
    "CvatPlayerReviewProxyResult",
    "JSON_REPORT_FILENAME",
    "MARKDOWN_REPORT_FILENAME",
    "PROXY_TRACKS_FILENAME",
    "build_cvat_player_review_proxy",
    "render_cvat_player_review_proxy_markdown",
    "write_cvat_player_review_proxy",
]
