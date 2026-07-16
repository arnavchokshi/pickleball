from __future__ import annotations

import subprocess
import sys

from threed.racketsport.player_id_repair import RepairDetection, RepairConfig, repair_detections_to_tracks


def _det(
    frame: int,
    tid: int,
    x: float,
    y: float,
    *,
    conf: float = 0.9,
) -> RepairDetection:
    return RepairDetection(
        frame_idx=frame,
        source_track_id=tid,
        bbox=(x, y, x + 1.0, y + 2.0),
        world_xy=(x, y),
        conf=conf,
    )


def _player_world_series(tracks, player_id: int) -> list[tuple[int, tuple[float, float]]]:
    player = next(player for player in tracks.players if player.id == player_id)
    return [(int(round(frame.t * tracks.fps)), tuple(frame.world_xy)) for frame in player.frames]


def test_repair_reconnects_motion_continuous_fragments_and_fills_safe_gap() -> None:
    detections = [_det(frame, 10, float(frame), 0.0) for frame in range(0, 5)]
    detections += [_det(frame, 20, float(frame), 0.0) for frame in range(6, 11)]
    detections += [_det(frame, 30, 0.0, 10.0) for frame in range(0, 11)]

    tracks, summary = repair_detections_to_tracks(
        detections,
        fps=10.0,
        config=RepairConfig(expected_players=2, max_merge_gap_frames=12, max_gap_fill_frames=4),
    )

    assert summary.output_player_count == 2
    assert summary.merged_fragment_count >= 1
    connected = [
        _player_world_series(tracks, player.id)
        for player in tracks.players
        if len(player.frames) == 11
    ]
    assert connected
    assert any(frame_idx == 5 and world_xy == (5.0, 0.0) for frame_idx, world_xy in connected[0])
    assert summary.confidence_repairs == (
        {
            "player_id": 1,
            "frame_index": 5,
            "t": 0.5,
            "conf": 0.35,
            "conf_source": "interpolated_endpoint_min_capped_0_35",
            "repaired": True,
        },
    )
    assert all("conf_source" not in frame for player in tracks.model_dump()["players"] for frame in player["frames"])


def test_repair_refuses_impossible_teleport_merge() -> None:
    detections = [_det(frame, 10, float(frame), 0.0) for frame in range(0, 4)]
    detections += [_det(frame, 20, 100.0 + float(frame), 0.0) for frame in range(4, 8)]
    detections += [_det(frame, 30, 0.0, 10.0) for frame in range(0, 8)]

    tracks, summary = repair_detections_to_tracks(
        detections,
        fps=10.0,
        config=RepairConfig(
            expected_players=2,
            max_merge_gap_frames=12,
            max_merge_speed_m_s=5.0,
            merge_distance_slack_m=0.25,
        ),
    )

    assert summary.dropped_fragment_count >= 1
    for player in tracks.players:
        series = _player_world_series(tracks, player.id)
        for (_, prev), (_, nxt) in zip(series, series[1:], strict=False):
            assert ((nxt[0] - prev[0]) ** 2 + (nxt[1] - prev[1]) ** 2) ** 0.5 < 20.0


def test_gap_fill_fails_closed_when_interpolation_overlaps_other_player() -> None:
    detections = [_det(0, 10, 0.0, 0.0), _det(2, 10, 2.0, 0.0)]
    detections += [_det(frame, 30, 1.0, 0.0) for frame in range(0, 3)]

    tracks, summary = repair_detections_to_tracks(
        detections,
        fps=10.0,
        config=RepairConfig(expected_players=2, max_gap_fill_frames=3, gap_fill_iou_threshold=0.1),
    )

    assert summary.gap_fill_skipped_overlap_count >= 1
    overlapping_frame_predictions = sum(
        1
        for player in tracks.players
        for frame in player.frames
        if int(round(frame.t * tracks.fps)) == 1 and abs(frame.world_xy[0] - 1.0) < 0.01
    )
    assert overlapping_frame_predictions == 1


def test_repair_person_tracks_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/repair_person_tracks.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Offline exactly-4 player ID repair" in completed.stdout
