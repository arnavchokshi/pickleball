from __future__ import annotations

import subprocess
import sys

import threed.racketsport.player_global_association as player_global_association
from threed.racketsport.court_templates import get_court_template
from threed.racketsport.player_global_association import (
    GlobalAssociationConfig,
    GlobalAssociationDetection,
    associate_global_identities,
    raw_pool_to_global_detections,
    tracks_to_global_detections,
)
from threed.racketsport.schemas import CourtCalibration, PlayerTrack, TrackFrame, Tracks


def _identity_calibration() -> CourtCalibration:
    return CourtCalibration.model_validate(
        {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 480.0, "cy": 270.0, "dist": [], "source": "test"},
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 5.0],
                "camera_height_m": 5.0,
            },
            "reprojection_error_px": {"median": 0.0, "p95": 0.0},
            "capture_quality": {"grade": "good", "reasons": []},
            "image_pts": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "world_pts": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        }
    )


def _det(
    frame: int,
    source: int,
    x: float,
    y: float,
    emb: tuple[float, float],
    *,
    conf: float = 0.9,
) -> GlobalAssociationDetection:
    return GlobalAssociationDetection(
        frame_idx=frame,
        source_track_id=source,
        bbox=(x, y, x + 1.0, y + 2.0),
        world_xy=(x, y),
        conf=conf,
        embedding=emb,
    )


def _player_frames(tracks) -> list[list[int]]:
    return [
        [int(round(frame.t * tracks.fps)) for frame in player.frames]
        for player in sorted(tracks.players, key=lambda item: item.id)
    ]


def _player_xs(tracks) -> list[list[float]]:
    return [
        [round(float(frame.world_xy[0]), 1) for frame in player.frames]
        for player in sorted(tracks.players, key=lambda item: item.id)
    ]


def test_global_association_splits_embedding_mixed_tracklets_and_reconnects_fragments() -> None:
    detections = [
        _det(0, 10, 0.0, -1.0, (1.0, 0.0)),
        _det(1, 10, 1.0, -1.0, (1.0, 0.0)),
        _det(2, 10, 10.0, 1.0, (0.0, 1.0)),
        _det(3, 10, 10.0, 2.0, (0.0, 1.0)),
        _det(4, 20, 4.0, -1.0, (1.0, 0.0)),
        _det(5, 20, 5.0, -1.0, (1.0, 0.0)),
        _det(4, 30, 10.0, 3.0, (0.0, 1.0)),
        _det(5, 30, 10.0, 4.0, (0.0, 1.0)),
    ]

    tracks, summary = associate_global_identities(
        detections,
        fps=1.0,
        config=GlobalAssociationConfig(
            expected_players=2,
            embedding_split_eps=0.25,
            appearance_weight=2.0,
            max_merge_gap_frames=6,
            max_merge_speed_m_s=3.0,
            max_gap_fill_frames=0,
        ),
    )

    xs = _player_xs(tracks)

    assert summary.status == "ok"
    assert summary.output_player_count == 2
    assert summary.embedding_split_count >= 1
    assert summary.merged_fragment_count >= 2
    assert any(series == [0.0, 1.0, 4.0, 5.0] for series in xs)
    assert any(series == [10.0, 10.0, 10.0, 10.0] for series in xs)
    assert summary.source_only is True
    assert summary.uses_cvat_labels is False


def test_global_association_drops_extra_low_persistence_fragment_instead_of_fabricating_identity() -> None:
    detections = [
        *[_det(frame, 10, float(frame), -2.0, (1.0, 0.0)) for frame in range(4)],
        *[_det(frame, 20, float(frame), 2.0, (0.0, 1.0)) for frame in range(4)],
        _det(1, 99, 50.0, 50.0, (0.7, 0.7), conf=0.2),
    ]

    tracks, summary = associate_global_identities(
        detections,
        fps=1.0,
        config=GlobalAssociationConfig(expected_players=2, min_fragment_frames=1),
    )

    assert summary.status == "ok"
    assert summary.output_player_count == 2
    assert summary.dropped_fragment_count >= 1
    assert all(abs(x) < 10.0 for series in _player_xs(tracks) for x in series)


def test_global_association_interpolates_short_safe_gaps_but_leaves_long_gaps_missing() -> None:
    detections = [
        _det(0, 10, 0.0, 0.0, (1.0, 0.0)),
        _det(2, 10, 2.0, 0.0, (1.0, 0.0)),
        _det(8, 10, 8.0, 0.0, (1.0, 0.0)),
    ]

    tracks, summary = associate_global_identities(
        detections,
        fps=1.0,
        config=GlobalAssociationConfig(
            expected_players=1,
            split_gap_frames=10,
            max_gap_fill_frames=3,
            max_gap_fill_speed_m_s=3.0,
        ),
    )

    assert summary.status == "ok"
    assert summary.synthetic_frame_count == 1
    assert _player_frames(tracks) == [[0, 1, 2, 8]]


def test_global_association_backfills_missing_cardinality_from_source_detections() -> None:
    detections = [
        _det(0, 10, 0.0, -1.0, (1.0, 0.0)),
        _det(1, 10, 0.1, -1.0, (1.0, 0.0)),
        _det(2, 10, 0.2, -1.0, (1.0, 0.0)),
        _det(1, 20, 4.1, 1.0, (0.0, 1.0)),
        _det(2, 20, 4.2, 1.0, (0.0, 1.0)),
        _det(0, 30, 4.0, 1.0, (0.0, 1.0)),
    ]

    tracks, summary = associate_global_identities(
        detections,
        fps=10.0,
        config=GlobalAssociationConfig(
            expected_players=2,
            cardinality_backfill=True,
            max_merge_cost=0.0,
            backfill_max_cost=10.0,
            max_gap_fill_frames=0,
        ),
    )

    assert summary.cardinality_backfilled_detection_count == 1
    assert summary.synthetic_frame_count == 0
    assert _player_frames(tracks) == [[0, 1, 2], [0, 1, 2]]


def test_global_association_fails_closed_when_expected_identities_are_missing() -> None:
    tracks, summary = associate_global_identities(
        [_det(0, 10, 0.0, 0.0, (1.0, 0.0))],
        fps=1.0,
        config=GlobalAssociationConfig(expected_players=2),
    )

    assert summary.status == "insufficient_players"
    assert summary.output_player_count == 1
    assert len(tracks.players) == 1


def test_global_association_can_drop_outside_court_detections_fail_closed() -> None:
    detections = [
        _det(0, 10, 0.0, 0.0, (1.0, 0.0)),
        _det(1, 10, 200.0, 200.0, (1.0, 0.0)),
    ]

    tracks, summary = associate_global_identities(
        detections,
        fps=1.0,
        config=GlobalAssociationConfig(expected_players=1, drop_outside_court=True),
    )

    assert summary.status == "ok"
    assert summary.court_rejected_detection_count == 1
    assert _player_frames(tracks) == [[0]]


def test_global_association_court_gate_allows_configured_apron_margin() -> None:
    court = get_court_template("pickleball")
    near_boundary_x = court.width_m / 2.0 + 0.4
    far_outside_x = court.width_m / 2.0 + 1.6
    detections = [
        _det(0, 10, near_boundary_x, 0.0, (1.0, 0.0)),
        _det(1, 10, far_outside_x, 0.0, (1.0, 0.0)),
    ]

    tracks, summary = associate_global_identities(
        detections,
        fps=1.0,
        config=GlobalAssociationConfig(expected_players=1, drop_outside_court=True, court_margin_m=1.0),
    )

    assert summary.court_rejected_detection_count == 1
    assert _player_frames(tracks) == [[0]]


def test_global_association_court_gate_reuses_person_fast_court_polygon_filter(monkeypatch) -> None:
    """The candidate-construction court gate must delegate to the same
    ``court_polygon_filter`` reused elsewhere (track.py, player_source_selection.py),
    not a private re-derivation of the rectangle math."""

    calls: list[float] = []
    real_filter = player_global_association.court_polygon_filter

    def spy(detections, *, sport, margin_m):
        calls.append(margin_m)
        return real_filter(detections, sport=sport, margin_m=margin_m)

    monkeypatch.setattr(player_global_association, "court_polygon_filter", spy)

    detections = [
        _det(0, 10, 0.0, 0.0, (1.0, 0.0)),
        _det(1, 10, 200.0, 200.0, (1.0, 0.0)),
    ]
    tracks, summary = associate_global_identities(
        detections,
        fps=1.0,
        config=GlobalAssociationConfig(expected_players=1, drop_outside_court=True, court_margin_m=1.0),
    )

    assert calls, "expected court_polygon_filter to be invoked for candidate construction"
    assert all(margin == 1.0 for margin in calls)
    assert summary.court_rejected_detection_count == 1
    assert summary.court_filter_skipped_reason == ""
    assert _player_frames(tracks) == [[0]]


def test_global_association_court_gate_fails_open_on_unsupported_sport() -> None:
    """An unavailable/unsupported court template must never crash the whole
    association run -- the filter is skipped (fail-open) and a reason is
    surfaced in the summary for the caller to log."""

    detections = [
        _det(0, 10, 0.0, 0.0, (1.0, 0.0)),
        _det(1, 10, 200.0, 200.0, (1.0, 0.0)),
    ]

    tracks, summary = associate_global_identities(
        detections,
        fps=1.0,
        config=GlobalAssociationConfig(expected_players=2, drop_outside_court=True, sport="not_a_real_sport"),
    )

    assert summary.court_rejected_detection_count == 0
    assert "court_template_unavailable" in summary.court_filter_skipped_reason
    assert summary.status == "ok"
    assert summary.output_player_count == 2


def test_global_association_post_association_filter_drops_only_off_court_output_frames() -> None:
    """A candidate-construction apron margin (e.g. 2m) intentionally lets a
    real boundary-crossing player through; a tighter post-association margin
    should be able to trim just the frames that are still off the strict
    court polygon from the *final* selected track, without re-running
    fragment/identity selection (so it cannot introduce new ID switches)."""

    court = get_court_template("pickleball")
    near_boundary_x = court.width_m / 2.0 + 0.5
    detections = [
        _det(0, 10, 0.0, 0.0, (1.0, 0.0)),
        _det(1, 10, near_boundary_x, 0.0, (1.0, 0.0)),
        _det(2, 10, 0.0, 0.0, (1.0, 0.0)),
    ]

    baseline_tracks, baseline_summary = associate_global_identities(
        detections,
        fps=1.0,
        config=GlobalAssociationConfig(expected_players=1, drop_outside_court=True, court_margin_m=1.0),
    )
    assert _player_frames(baseline_tracks) == [[0, 1, 2]]
    assert baseline_summary.post_association_court_rejected_frame_count == 0

    filtered_tracks, filtered_summary = associate_global_identities(
        detections,
        fps=1.0,
        config=GlobalAssociationConfig(
            expected_players=1,
            drop_outside_court=True,
            court_margin_m=1.0,
            post_association_court_margin_m=0.0,
        ),
    )

    assert filtered_summary.post_association_court_rejected_frame_count == 1
    assert _player_frames(filtered_tracks) == [[0, 2]]
    assert filtered_summary.court_filter_skipped_reason == ""


def test_embedding_splitter_uses_bounded_comparisons_for_long_tracklets(monkeypatch) -> None:
    detections = [
        _det(frame, 10, float(frame), 0.0, (1.0, 0.0) if frame % 2 == 0 else (0.0, 1.0))
        for frame in range(120)
    ]
    calls = 0
    original = player_global_association._cosine_distance

    def counted_cosine_distance(left, right):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(player_global_association, "_cosine_distance", counted_cosine_distance)

    labels = player_global_association._embedding_cluster_labels(
        detections,
        config=GlobalAssociationConfig(embedding_split_eps=0.25),
    )

    assigned = {label for label in labels if label is not None}
    assert len(assigned) == 2
    assert calls < 500


def test_embedding_splitter_caps_runaway_cluster_growth(monkeypatch) -> None:
    detections = []
    for frame in range(80):
        embedding = tuple(1.0 if idx == frame else 0.0 for idx in range(80))
        detections.append(_det(frame, 10, float(frame), 0.0, embedding))  # type: ignore[arg-type]
    calls = 0
    original = player_global_association._cosine_distance

    def counted_cosine_distance(left, right):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(player_global_association, "_cosine_distance", counted_cosine_distance)

    labels = player_global_association._embedding_cluster_labels(
        detections,
        config=GlobalAssociationConfig(embedding_split_eps=0.0, embedding_split_max_clusters=8),
    )

    assigned = {label for label in labels if label is not None}
    assert len(assigned) <= 8
    assert calls <= 80 * 8


def test_embedding_splitter_updates_centroids_incrementally(monkeypatch) -> None:
    detections = [
        _det(frame, 10, float(frame), 0.0, (1.0, 0.0) if frame % 2 == 0 else (0.0, 1.0))
        for frame in range(120)
    ]
    calls = 0
    original = player_global_association._mean_embedding

    def counted_mean_embedding(embeddings):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(embeddings)

    monkeypatch.setattr(player_global_association, "_mean_embedding", counted_mean_embedding)

    labels = player_global_association._embedding_cluster_labels(
        detections,
        config=GlobalAssociationConfig(embedding_split_eps=0.25),
    )

    assigned = {label for label in labels if label is not None}
    assert len(assigned) == 2
    assert calls == 0


def test_fragment_builder_splits_local_motion_and_appearance_handoff() -> None:
    detections = [
        _det(0, 10, 0.0, 4.5, (1.0, 0.0)),
        _det(1, 10, 0.1, 4.5, (1.0, 0.0)),
        _det(14, 10, 0.3, 4.6, (0.98, 0.02)),
        _det(19, 10, 1.2, 2.7, (0.0, 1.0)),
        _det(20, 10, 1.2, 2.7, (0.0, 1.0)),
    ]

    fragments, _split_count = player_global_association._build_fragments(
        detections,
        fps=60.0,
        config=GlobalAssociationConfig(
            split_gap_frames=24,
            max_fragment_speed_m_s=12.0,
            local_switch_split_distance_m=1.5,
            local_switch_split_embedding_distance=0.10,
            local_switch_split_max_gap_frames=15,
        ),
    )

    assert [(fragment.start_frame, fragment.end_frame) for fragment in fragments] == [(0, 14), (19, 20)]


def test_fragment_connector_scores_each_fragment_pair_once(monkeypatch) -> None:
    fragments = [
        player_global_association._make_fragment(
            idx + 1,
            idx % 4,
            [_det(idx * 3, idx % 4, float(idx % 4), 0.0, (1.0, 0.0))],
        )
        for idx in range(40)
    ]
    calls = 0
    original = player_global_association._fragment_link_cost

    def counted_fragment_link_cost(left, right, *, fps, config):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(left, right, fps=fps, config=config)

    monkeypatch.setattr(player_global_association, "_fragment_link_cost", counted_fragment_link_cost)

    clusters, merged = player_global_association._connect_fragments(
        fragments,
        fps=30.0,
        config=GlobalAssociationConfig(
            expected_players=4,
            max_merge_gap_frames=1000,
            max_merge_speed_m_s=100.0,
            max_merge_cost=10.0,
            appearance_weight=0.0,
            motion_weight=1.0,
            side_prior_weight=0.0,
        ),
    )

    assert len(clusters) == 4
    assert merged == 36
    assert calls <= len(fragments) * (len(fragments) - 1) // 2


def test_raw_pool_to_global_detections_reads_every_raw_detection_including_spectators() -> None:
    detections_payload = {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [10.0, 10.0, 20.0, 30.0], "class": "person", "conf": 0.9, "track_id": 1},
                    {"bbox": [100.0, 100.0, 110.0, 120.0], "class": "person", "conf": 0.4, "track_id": 42},
                    {"bbox": [1.0, 1.0, 5.0, 5.0], "class": "ball", "conf": 0.9, "track_id": 3},
                ],
            }
        ],
    }

    detections = raw_pool_to_global_detections(detections_payload, calibration=_identity_calibration())

    assert len(detections) == 2
    assert {detection.source_track_id for detection in detections} == {1, 42}
    foot_point = next(detection for detection in detections if detection.source_track_id == 1)
    assert foot_point.world_xy == (15.0, 30.0)
    assert foot_point.bbox == (10.0, 10.0, 20.0, 30.0)
    assert foot_point.conf == 0.9


def test_raw_pool_to_global_detections_joins_embeddings_and_filters_low_conf() -> None:
    detections_payload = {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [10.0, 10.0, 20.0, 30.0], "class": "person", "conf": 0.9, "track_id": 1},
                    {"bbox": [200.0, 200.0, 210.0, 220.0], "class": "person", "conf": 0.05, "track_id": 2},
                ],
            }
        ],
    }
    embedding_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_reid_embedding_export",
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "feature_dim": 2,
        "l2_normalized": True,
        "detections": [
            {"frame": 0, "source_track_id": 1, "bbox": [10.0, 10.0, 20.0, 30.0], "embedding": [1.0, 0.0]},
        ],
    }

    detections = raw_pool_to_global_detections(
        detections_payload,
        calibration=_identity_calibration(),
        embedding_payload=embedding_payload,
        min_conf=0.1,
    )

    assert len(detections) == 1
    assert detections[0].source_track_id == 1
    assert detections[0].embedding == (1.0, 0.0)


def test_tracks_to_global_detections_joins_source_only_osnet_embedding_export() -> None:
    tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[
                    TrackFrame(t=0.0, bbox=(2.0, 2.0, 12.0, 18.0), world_xy=(1.0, -1.0), conf=0.9),
                    TrackFrame(t=0.1, bbox=(3.0, 2.0, 13.0, 18.0), world_xy=(1.2, -1.0), conf=0.8),
                ],
            )
        ],
        rally_spans=[],
    )
    embedding_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_reid_embedding_export",
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "feature_type": "osnet_reid_embedding",
        "feature_dim": 2,
        "l2_normalized": True,
        "detections": [
            {"frame": 0, "source_track_id": 7, "bbox": [2.0, 2.0, 12.0, 18.0], "embedding": [1.0, 0.0]},
            {"frame": 1, "source_track_id": 7, "bbox": [3.0, 2.0, 13.0, 18.0], "embedding": [0.0, 1.0]},
        ],
    }

    detections = tracks_to_global_detections(tracks, embedding_payload=embedding_payload)

    assert [detection.embedding for detection in detections] == [(1.0, 0.0), (0.0, 1.0)]


def test_tracks_to_global_detections_falls_back_to_frame_bbox_when_role_locked_ids_differ() -> None:
    tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(431.875, 185.3125, 495.9375, 336.875), world_xy=(0.0, -1.0), conf=0.9)],
            )
        ],
        rally_spans=[],
    )
    embedding_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_reid_embedding_export",
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "feature_type": "osnet_reid_embedding",
        "feature_dim": 2,
        "l2_normalized": True,
        "detections": [
            {
                "frame": 0,
                "source_track_id": 2,
                "bbox": [863.75, 370.625, 991.875, 673.75],
                "embedding": [0.0, 1.0],
            },
            {
                "frame": 0,
                "source_track_id": 99,
                "bbox": [100.0, 100.0, 140.0, 200.0],
                "embedding": [1.0, 0.0],
            },
        ],
    }

    detections = tracks_to_global_detections(
        tracks,
        embedding_payload=embedding_payload,
        embedding_bbox_scale=2.0,
    )

    assert detections[0].source_track_id == 1
    assert detections[0].embedding == (0.0, 1.0)


def test_tracks_to_global_detections_leaves_repaired_gap_frames_without_bad_embedding() -> None:
    tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(50.0, 50.0, 80.0, 120.0), world_xy=(0.0, -1.0), conf=0.5)],
            )
        ],
        rally_spans=[],
    )
    embedding_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_reid_embedding_export",
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "feature_type": "osnet_reid_embedding",
        "feature_dim": 2,
        "l2_normalized": True,
        "detections": [
            {
                "frame": 0,
                "source_track_id": 99,
                "bbox": [500.0, 500.0, 560.0, 640.0],
                "embedding": [1.0, 0.0],
            }
        ],
    }

    detections = tracks_to_global_detections(
        tracks,
        embedding_payload=embedding_payload,
        max_embedding_bbox_delta_px=2.5,
    )

    assert detections[0].embedding is None


def test_tracks_to_global_detections_rejects_label_derived_embedding_export() -> None:
    tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(2.0, 2.0, 12.0, 18.0), world_xy=(1.0, -1.0), conf=0.9)],
            )
        ],
        rally_spans=[],
    )

    try:
        tracks_to_global_detections(
            tracks,
            embedding_payload={"source_only": False, "uses_cvat_labels": True, "detections": []},
        )
    except ValueError as exc:
        assert "source-only" in str(exc)
    else:
        raise AssertionError("expected label-derived embedding payload rejection")


def test_global_associate_person_tracks_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/global_associate_person_tracks.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Global exactly-N player identity association" in completed.stdout
    assert "--embedding-export" in completed.stdout
