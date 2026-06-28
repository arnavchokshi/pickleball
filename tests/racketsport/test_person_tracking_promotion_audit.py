from __future__ import annotations

from threed.racketsport.person_tracking_promotion_audit import build_person_tracking_promotion_audit
from threed.racketsport.schemas import PlayerTrack, TrackFrame, Tracks


def _tracks_with_world_points(*, off_court: bool = False) -> Tracks:
    frames_by_player = {
        1: [(-2.0, -5.0), (-1.8, -4.8)],
        2: [(2.0, -5.0), (1.8, -4.8)],
        3: [(-2.0, 5.0), (-1.8, 4.8)],
        4: [(2.0, 5.0), (1.8, 4.8)],
    }
    if off_court:
        frames_by_player[4][1] = (7.0, 6.0)

    return Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=player_id,
                side="near" if player_id <= 2 else "far",
                role="left" if player_id % 2 else "right",
                frames=[
                    TrackFrame(
                        t=index / 30.0,
                        bbox=(10.0 * player_id, 10.0, 10.0 * player_id + 20.0, 80.0),
                        world_xy=[x, y],
                        conf=0.91,
                    )
                    for index, (x, y) in enumerate(world_points)
                ],
            )
            for player_id, world_points in frames_by_player.items()
        ],
        rally_spans=[],
    )


def test_person_tracking_audit_distinguishes_strict_candidate_from_widened_diagnostic() -> None:
    strict = build_person_tracking_promotion_audit(
        tracks=_tracks_with_world_points(),
        clip="indoor_doubles",
        variant="yolo26m_source30_rolelock",
        court_margin_m=0.0,
        max_players=4,
        total_frames=2,
    )

    assert strict["safe_for_canonical_review"] is True
    assert strict["diagnostic_only"] is False
    assert strict["trusted_for_trk_promotion"] is False
    assert strict["promotion_blockers"] == ["labeled_idf1_spectator_gate_missing"]
    assert strict["safety_blockers"] == []
    assert strict["track_safety"]["outside_court_frame_count"] == 0

    widened = build_person_tracking_promotion_audit(
        tracks=_tracks_with_world_points(off_court=True),
        clip="indoor_doubles",
        variant="yolo26m_source30_rolelock_margin4_diagnostic",
        court_margin_m=4.0,
        max_players=4,
        total_frames=2,
    )

    assert widened["safe_for_canonical_review"] is False
    assert widened["diagnostic_only"] is True
    assert widened["trusted_for_trk_promotion"] is False
    assert widened["safety_blockers"] == [
        "widened_court_margin_diagnostic_only",
        "off_court_player_frames_present",
    ]
    assert widened["track_safety"]["outside_court_frame_count"] == 1
    assert widened["track_safety"]["outside_court_player_ids"] == [4]
