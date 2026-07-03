from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.court_templates import get_court_template
from threed.racketsport.person_track_gt_scoring import (
    DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2,
    DEFAULT_OFF_COURT_APRON_MARGIN_M,
    build_source_promotion_decision,
    build_source_promotion_decision_v2,
    build_source_promotion_decision_v2_1,
    build_scoring_report,
    derive_track_source_id,
    render_scoring_report_markdown,
    score_tracks_against_person_ground_truth,
    summarize_score_failure_modes_v2,
)
from threed.racketsport.schemas import PersonGroundTruth, PlayerTrack, TrackFrame, Tracks


def _person_label(track_id: int, frame_index: int, x: float) -> dict:
    return {
        "track_id": track_id,
        "bbox_xywh": [x, 0.0, 10.0, 10.0],
        "ignored": False,
        "visibility": 1.0,
        "confidence": 1.0,
        "class_id": None,
        "class_name": "player",
        "person_class": True,
    }


def _ground_truth() -> PersonGroundTruth:
    return PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_a",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [
                {"frame_index": 0, "source_frame_id": 1, "labels": [_person_label(1, 0, 0.0)]},
                {"frame_index": 1, "source_frame_id": 2, "labels": [_person_label(1, 1, 1.0)]},
            ],
            "summary": {
                "frame_count": 2,
                "valid_label_count": 2,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )


def _track_frame(frame_index: int, bbox: tuple[float, float, float, float], world_xy: list[float]) -> TrackFrame:
    return TrackFrame(t=frame_index / 30.0, bbox=bbox, world_xy=world_xy, conf=0.9)


def _tracks_with_switch_fp_and_tail() -> Tracks:
    return Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                frames=[_track_frame(0, (0.0, 0.0, 10.0, 10.0), [0.0, 0.0])],
            ),
            PlayerTrack(
                id=2,
                side="near",
                role="right",
                frames=[_track_frame(1, (1.0, 0.0, 11.0, 10.0), [0.0, 0.0])],
            ),
            PlayerTrack(
                id=3,
                side="far",
                role="left",
                frames=[_track_frame(1, (50.0, 50.0, 60.0, 60.0), [8.0, 8.0])],
            ),
            PlayerTrack(
                id=4,
                side="far",
                role="right",
                frames=[_track_frame(3, (0.0, 0.0, 10.0, 10.0), [0.0, 0.0])],
            ),
        ],
        rally_spans=[],
    )


def test_score_tracks_against_person_ground_truth_catches_switches_false_positives_and_caps_tail() -> None:
    score = score_tracks_against_person_ground_truth(
        ground_truth=_ground_truth(),
        tracks=_tracks_with_switch_fp_and_tail(),
        candidate="candidate_a",
        tracks_path="runs/example/clip_a/candidate_a/tracks.json",
        iou_threshold=0.5,
    )

    assert score["idf1"] == pytest.approx(0.4)
    assert score["id_switches"] == 1
    assert score["false_positives"] == 1
    assert score["false_negatives"] == 0
    assert score["four_player_coverage"] == pytest.approx(0.5)
    assert score["spectator_or_background_false_positives"] == 1
    assert score["off_court_false_positive_frames"] == 1
    assert score["off_court_false_positive_track_ids"] == [3]
    assert score["outside_gt_prediction_count"] == 1
    assert score["deta"] == pytest.approx(2.0 / 3.0)
    assert score["assa"] == pytest.approx(0.5)
    assert score["hota"] == pytest.approx(math.sqrt(1.0 / 3.0))
    assert score["hota_iou_threshold"] == pytest.approx(0.5)


def test_score_tracks_against_person_ground_truth_reports_switch_events_and_temporal_coverage() -> None:
    score = score_tracks_against_person_ground_truth(
        ground_truth=_ground_truth(),
        tracks=_tracks_with_switch_fp_and_tail(),
        candidate="candidate_a",
        tracks_path="runs/example/clip_a/candidate_a/tracks.json",
        iou_threshold=0.5,
    )

    assert score["identity_switch_event_count"] == 1
    assert score["identity_switch_events"] == [
        {
            "frame_index": 1,
            "gt_track_id": 1,
            "previous_pred_track_id": 1,
            "new_pred_track_id": 2,
            "previous_match_frame_index": 0,
            "frames_since_previous_match": 1,
            "iou": pytest.approx(1.0),
        }
    ]
    assert score["identity_switch_transitions"] == [
        {
            "gt_track_id": 1,
            "previous_pred_track_id": 1,
            "new_pred_track_id": 2,
            "count": 1,
            "first_frame_index": 1,
            "last_frame_index": 1,
        }
    ]
    assert score["temporal_coverage"]["gt_frame_range"] == {"first": 0, "last": 1}
    assert score["temporal_coverage"]["prediction_frame_range"] == {"first": 0, "last": 1}
    assert score["temporal_coverage"]["gt_detections_after_last_prediction"] == 0


def test_score_tracks_against_person_ground_truth_scales_track_boxes_to_gt_pixels() -> None:
    gt = PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_scaled",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [
                {"frame_index": 0, "source_frame_id": 1, "labels": [_person_label(1, 0, 20.0)]},
            ],
            "summary": {
                "frame_count": 1,
                "valid_label_count": 1,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[_track_frame(0, (10.0, 0.0, 20.0, 5.0), [0.0, 0.0])],
            )
        ],
        rally_spans=[],
    )

    score = score_tracks_against_person_ground_truth(
        ground_truth=gt,
        tracks=tracks,
        candidate="half_scale_tracks",
        tracks_path="runs/example/clip_scaled/half_scale_tracks/tracks.json",
        bbox_scale_x=2.0,
        bbox_scale_y=2.0,
    )

    assert score["idf1"] == pytest.approx(1.0)
    assert score["deta"] == pytest.approx(1.0)
    assert score["assa"] == pytest.approx(1.0)
    assert score["hota"] == pytest.approx(1.0)
    assert score["false_positives"] == 0
    assert score["false_negatives"] == 0


def _fp_decomposition_label(track_id: int, x: float) -> dict:
    return {
        "track_id": track_id,
        "bbox_xywh": [x, 0.0, 10.0, 10.0],
        "ignored": False,
        "visibility": 1.0,
        "confidence": 1.0,
        "class_id": None,
        "class_name": "player",
        "person_class": True,
    }


def _fp_decomposition_ground_truth() -> PersonGroundTruth:
    four_players = [_fp_decomposition_label(1, 0.0), _fp_decomposition_label(2, 20.0), _fp_decomposition_label(3, 40.0), _fp_decomposition_label(4, 60.0)]
    two_players = [_fp_decomposition_label(1, 0.0), _fp_decomposition_label(2, 20.0)]
    return PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_fp",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [
                {"frame_index": 0, "source_frame_id": 1, "labels": four_players},
                {"frame_index": 1, "source_frame_id": 2, "labels": two_players},
                {"frame_index": 2, "source_frame_id": 3, "labels": four_players},
            ],
            "summary": {
                "frame_count": 3,
                "valid_label_count": 10,
                "ignored_label_count": 0,
                "track_ids": [1, 2, 3, 4],
                "max_valid_players_per_frame": 4,
            },
        }
    )


def _fp_pred_frame(frame_index: int, bbox_xyxy: tuple[float, float, float, float]) -> TrackFrame:
    return TrackFrame(t=frame_index / 30.0, bbox=bbox_xyxy, world_xy=[0.0, 0.0], conf=0.9)


def _fp_decomposition_tracks() -> Tracks:
    # Frame 0 (full 4-player GT): 4 exact matches + a near-miss (id 5, IoU
    # ~0.333 against GT id 1) + a true spectator (id 6, zero IoU with any
    # real player).
    # Frame 1 (GT has only 2 players -- below expected_players=4): 2 exact
    # matches + a zero-IoU extra (id 7) that is a cardinality artifact, not a
    # spectator.
    # Frame 2 (full 4-player GT again): 4 exact matches + a zero-IoU box
    # (id 8) far outside any reasonable image bound, to exercise the
    # inside-image split.
    return Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(id=1, side="near", role="left", frames=[_fp_pred_frame(0, (0.0, 0.0, 10.0, 10.0)), _fp_pred_frame(1, (0.0, 0.0, 10.0, 10.0)), _fp_pred_frame(2, (0.0, 0.0, 10.0, 10.0))]),
            PlayerTrack(id=2, side="near", role="right", frames=[_fp_pred_frame(0, (20.0, 0.0, 30.0, 10.0)), _fp_pred_frame(1, (20.0, 0.0, 30.0, 10.0)), _fp_pred_frame(2, (20.0, 0.0, 30.0, 10.0))]),
            PlayerTrack(id=3, side="far", role="left", frames=[_fp_pred_frame(0, (40.0, 0.0, 50.0, 10.0)), _fp_pred_frame(2, (40.0, 0.0, 50.0, 10.0))]),
            PlayerTrack(id=4, side="far", role="right", frames=[_fp_pred_frame(0, (60.0, 0.0, 70.0, 10.0)), _fp_pred_frame(2, (60.0, 0.0, 70.0, 10.0))]),
            PlayerTrack(id=5, side="near", role="left", frames=[_fp_pred_frame(0, (5.0, 0.0, 15.0, 10.0))]),
            PlayerTrack(id=6, side="near", role="left", frames=[_fp_pred_frame(0, (200.0, 0.0, 210.0, 10.0))]),
            PlayerTrack(id=7, side="near", role="left", frames=[_fp_pred_frame(1, (500.0, 500.0, 510.0, 510.0))]),
            PlayerTrack(id=8, side="near", role="left", frames=[_fp_pred_frame(2, (-50.0, -50.0, -40.0, -40.0))]),
        ],
        rally_spans=[],
    )


def test_score_tracks_v2_decomposes_near_miss_no_gt_frame_and_true_spectator_without_image_bounds() -> None:
    score = score_tracks_against_person_ground_truth(
        ground_truth=_fp_decomposition_ground_truth(),
        tracks=_fp_decomposition_tracks(),
        candidate="fp_decomposition",
        tracks_path="runs/example/clip_fp/fp_decomposition/tracks.json",
        iou_threshold=0.5,
        expected_players=4,
    )

    # Aggregate stays exactly the v1 semantics for backward compatibility.
    assert score["spectator_or_background_false_positives"] == 4
    assert score["false_positives"] == 4

    # v2 decomposition: id5 near-miss, id7 no-GT-frame, id6+id8 true spectator
    # (no image bounds supplied, so "inside image" defaults to True).
    assert score["near_miss_false_positives"] == 1
    assert score["no_gt_frame_false_positives"] == 1
    assert score["true_spectator_or_background_false_positives"] == 2
    assert score["outside_image_false_positives"] == 0
    assert (
        score["near_miss_false_positives"]
        + score["no_gt_frame_false_positives"]
        + score["true_spectator_or_background_false_positives"]
        + score["outside_image_false_positives"]
        == score["spectator_or_background_false_positives"]
    )

    localization = score["near_miss_localization"]
    assert localization["count"] == 1
    assert localization["median_iou"] == pytest.approx(1.0 / 3.0)
    assert localization["p90_iou"] == pytest.approx(1.0 / 3.0)
    assert localization["rate"] == pytest.approx(score["near_miss_false_positive_rate"])
    assert score["near_miss_false_positive_rate"] == pytest.approx(1 / score["pred_detections"])


def test_score_tracks_v2_splits_outside_image_false_positives_when_bounds_supplied() -> None:
    score = score_tracks_against_person_ground_truth(
        ground_truth=_fp_decomposition_ground_truth(),
        tracks=_fp_decomposition_tracks(),
        candidate="fp_decomposition_bounded",
        tracks_path="runs/example/clip_fp/fp_decomposition_bounded/tracks.json",
        iou_threshold=0.5,
        expected_players=4,
        image_width=1000.0,
        image_height=1000.0,
    )

    # id6 (center ~205,5) stays inside [0,1000]x[0,1000]; id8 (center
    # ~-45,-45) falls outside -- reclassified out of true-spectator.
    assert score["near_miss_false_positives"] == 1
    assert score["no_gt_frame_false_positives"] == 1
    assert score["true_spectator_or_background_false_positives"] == 1
    assert score["outside_image_false_positives"] == 1
    assert score["spectator_or_background_false_positives"] == 4


def test_score_tracks_v2_1_splits_apron_vs_far_off_court_and_includes_matched_predictions() -> None:
    """Gate v2.1 classification test (see module docstring "Gate v2.1"
    section). Modeled on the real evidence
    (`runs/synergy_wirings_20260702T043529Z/w4_court_polygon_filter/w4_gate_v2_before_after.json`):
    a single real player excursion contains both correctly-matched (TP)
    frames and unmatched (FP) frames, all within a small distance of the
    sideline. Three predictions on one frame:

    - id 1: matched to GT (exact bbox overlap), world point 0.5m past the
      sideline -- an apron excursion by a *real* player. Must show up in
      the apron diagnostic (matched) but never as any kind of off-court
      false positive.
    - id 2: unmatched, world point also 0.5m past the sideline (apron) --
      the FP-labeled slice of the same kind of excursion. Counts toward
      the legacy (v1/v2) `off_court_false_positive_frames` axis (unchanged
      behavior) but NOT toward the v2.1 `far_off_court_false_positive_frames`
      gate-blocking axis.
    - id 3: unmatched, world point ~5.1m past the sideline (far) -- a
      genuine far-off-court false positive. Counts toward both the legacy
      axis and the v2.1 `far_off_court_false_positive_frames` axis.
    """
    half_width_m = get_court_template("pickleball").width_m / 2.0
    apron_x = half_width_m + 0.5  # 0.5m past the sideline -- within the 1.0m apron
    far_x = 8.0  # far outside the court in both axes (matches the pre-existing off-court test fixture)

    gt = PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_apron",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [{"frame_index": 0, "source_frame_id": 1, "labels": [_person_label(1, 0, 0.0)]}],
            "summary": {
                "frame_count": 1,
                "valid_label_count": 1,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                # Matches GT exactly (IoU 1.0) but its world point is in the apron.
                frames=[_track_frame(0, (0.0, 0.0, 10.0, 10.0), [apron_x, 0.0])],
            ),
            PlayerTrack(
                id=2,
                side="near",
                role="right",
                # Unmatched (non-overlapping bbox), apron world point.
                frames=[_track_frame(0, (500.0, 0.0, 510.0, 10.0), [apron_x, 0.0])],
            ),
            PlayerTrack(
                id=3,
                side="far",
                role="left",
                # Unmatched, far off-court world point.
                frames=[_track_frame(0, (700.0, 0.0, 710.0, 10.0), [far_x, far_x])],
            ),
        ],
        rally_spans=[],
    )

    score = score_tracks_against_person_ground_truth(
        ground_truth=gt,
        tracks=tracks,
        candidate="apron_classification",
        tracks_path="runs/example/clip_apron/apron_classification/tracks.json",
        iou_threshold=0.5,
        expected_players=1,
    )

    # v1/v2 legacy field is unchanged: both unmatched off-court predictions
    # count, the matched one (id 1) never does.
    assert score["off_court_false_positive_frames"] == 2
    assert score["off_court_false_positive_track_ids"] == [2, 3]

    # v2.1 apron diagnostic: both id 1 (matched) and id 2 (unmatched) show up,
    # id 3 (far) does not.
    assert score["off_court_apron_margin_m"] == pytest.approx(DEFAULT_OFF_COURT_APRON_MARGIN_M)
    assert score["apron_off_court_excursion_prediction_count"] == 2
    assert score["apron_off_court_excursion_frame_count"] == 1
    assert score["apron_off_court_excursion_track_ids"] == [1, 2]
    assert score["apron_off_court_excursion_matched_prediction_count"] == 1
    assert score["apron_off_court_excursion_unmatched_prediction_count"] == 1

    # v2.1 gate-blocking axis: only id 3 (far, unmatched) counts.
    assert score["far_off_court_false_positive_frames"] == 1
    assert score["far_off_court_false_positive_track_ids"] == [3]


def test_build_source_promotion_decision_v2_narrows_fp_axis_and_gates_near_miss_rate() -> None:
    clean_row = {
        "clip_id": "clip_clean",
        "idf1": 0.90,
        "id_switches": 0,
        "true_spectator_or_background_false_positives": 0,
        "spectator_or_background_false_positives": 2,
        "near_miss_false_positives": 2,
        "pred_detections": 100,
        "off_court_false_positive_frames": 0,
        "four_player_coverage": 0.97,
    }
    decision = build_source_promotion_decision_v2([clean_row], required_clip_ids=["clip_clean"])
    assert decision["promote"] is True
    assert decision["blockers"] == []

    true_spectator_row = {**clean_row, "clip_id": "clip_spectator", "true_spectator_or_background_false_positives": 1}
    decision = build_source_promotion_decision_v2([true_spectator_row], required_clip_ids=["clip_spectator"])
    assert decision["promote"] is False
    assert "clip_spectator:true_spectator_or_background_false_positives_present" in decision["blockers"]

    near_miss_heavy_row = {**clean_row, "clip_id": "clip_near_miss", "near_miss_false_positives": 20}
    decision = build_source_promotion_decision_v2([near_miss_heavy_row], required_clip_ids=["clip_near_miss"])
    assert decision["promote"] is False
    assert f"clip_near_miss:near_miss_false_positive_rate_above_{DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2:.2f}" in decision["blockers"]

    # A pre-v2 row (no true_spectator_or_background_false_positives key) must
    # fall back to the v1 aggregate rather than silently passing.
    legacy_row = {
        "clip_id": "clip_legacy",
        "idf1": 0.90,
        "id_switches": 0,
        "spectator_or_background_false_positives": 3,
        "off_court_false_positive_frames": 0,
        "four_player_coverage": 0.97,
    }
    decision = build_source_promotion_decision_v2([legacy_row], required_clip_ids=["clip_legacy"])
    assert decision["promote"] is False
    assert "clip_legacy:true_spectator_or_background_false_positives_present" in decision["blockers"]


def test_build_source_promotion_decision_v2_1_apron_clears_off_court_but_v2_still_blocks_on_it() -> None:
    """v2 vs v2.1 verdict coexistence, modeled directly on the Burlington
    evidence (`w4_gate_v2_before_after.json`): 30 off-court FP frames, all
    within the 1.0m apron (max observed excess 0.82m), spanning a single
    ~1.2s / 73-frame excursion. v2's off-court axis is unchanged and still
    blocks on the legacy field; v2.1's narrower axis clears it. Both still
    fail overall on cov4 (0.8867 < 0.95) -- the module docstring's no-motive
    proof: this refinement clears the off-court axis, not cov4, so it could
    not have been motivated by wanting to flip Burlington's promotion
    verdict.
    """
    burlington_like_row = {
        "clip_id": "clip_burlington_like",
        "idf1": 0.9112,
        "id_switches": 0,
        "true_spectator_or_background_false_positives": 0,
        "spectator_or_background_false_positives": 30,
        "near_miss_false_positives": 176,
        "pred_detections": 1000,
        "off_court_false_positive_frames": 30,
        "far_off_court_false_positive_frames": 0,
        "apron_off_court_excursion_frame_count": 73,
        "apron_off_court_excursion_prediction_count": 73,
        "four_player_coverage": 0.8867,
    }

    decision_v2 = build_source_promotion_decision_v2([burlington_like_row], required_clip_ids=["clip_burlington_like"])
    assert decision_v2["promote"] is False
    assert "clip_burlington_like:off_court_false_positives_present" in decision_v2["blockers"]

    decision_v2_1 = build_source_promotion_decision_v2_1(
        [burlington_like_row], required_clip_ids=["clip_burlington_like"]
    )
    assert decision_v2_1["promote"] is False
    assert not any("off_court" in blocker for blocker in decision_v2_1["blockers"])
    assert "clip_burlington_like:four_player_coverage_below_0.95" in decision_v2_1["blockers"]


def test_build_source_promotion_decision_v2_1_apron_excursion_never_blocks_regardless_of_sustained_duration() -> None:
    """Sustained-time criterion test. v2.1 deliberately uses a pure
    distance-based apron with no duration/sustained-time term (see the
    module docstring's "Gate v2.1" section for why a duration criterion was
    considered, per the trk_r2_tiled REPORT.md's "excursion-duration-aware"
    suggestion, and rejected as unnecessary complexity given the evidence).
    This locks in that decision from one side: an apron excursion spanning
    120 frames (2.0s @ 60fps -- at least as long as the illustrative "under
    2s" duration this task considered) never blocks, exactly like a
    single-frame apron excursion would not.
    """
    row = {
        "clip_id": "clip_sustained_apron",
        "idf1": 0.90,
        "id_switches": 0,
        "true_spectator_or_background_false_positives": 0,
        "near_miss_false_positives": 0,
        "pred_detections": 100,
        "far_off_court_false_positive_frames": 0,
        "apron_off_court_excursion_frame_count": 120,
        "apron_off_court_excursion_prediction_count": 120,
        "four_player_coverage": 0.97,
    }
    decision = build_source_promotion_decision_v2_1([row], required_clip_ids=["clip_sustained_apron"])
    assert decision["promote"] is True
    assert decision["blockers"] == []


def test_build_source_promotion_decision_v2_1_brief_far_excursion_still_blocks() -> None:
    """The other side of the sustained-duration decision above: v2.1 does
    NOT exempt brief far-off-court excursions either -- a single-frame
    (<< 2s) unmatched prediction beyond the apron still blocks. Duration is
    irrelevant in both directions under the pure distance-based rule this
    task chose as "the simplest defensible formulation".
    """
    row = {
        "clip_id": "clip_brief_far",
        "idf1": 0.90,
        "id_switches": 0,
        "true_spectator_or_background_false_positives": 0,
        "near_miss_false_positives": 0,
        "pred_detections": 100,
        "far_off_court_false_positive_frames": 1,
        "apron_off_court_excursion_frame_count": 0,
        "four_player_coverage": 0.97,
    }
    decision = build_source_promotion_decision_v2_1([row], required_clip_ids=["clip_brief_far"])
    assert decision["promote"] is False
    assert "clip_brief_far:far_off_court_false_positives_present" in decision["blockers"]


def test_build_source_promotion_decision_v2_1_legacy_row_falls_back_to_v1_off_court_field() -> None:
    # A pre-v2.1 row (no far_off_court_false_positive_frames key) must fall
    # back to the legacy off_court_false_positive_frames value rather than
    # silently passing just because it predates v2.1 -- same discipline as
    # v2's true-spectator fallback above.
    legacy_row = {
        "clip_id": "clip_legacy",
        "idf1": 0.90,
        "id_switches": 0,
        "true_spectator_or_background_false_positives": 0,
        "off_court_false_positive_frames": 3,
        "four_player_coverage": 0.97,
    }
    decision = build_source_promotion_decision_v2_1([legacy_row], required_clip_ids=["clip_legacy"])
    assert decision["promote"] is False
    assert "clip_legacy:far_off_court_false_positives_present" in decision["blockers"]


def test_summarize_score_failure_modes_v2_orders_by_count_and_uses_v2_descriptions() -> None:
    row = {
        "false_negatives": 5,
        "gt_detections": 50,
        "true_spectator_or_background_false_positives": 2,
        "near_miss_false_positives": 20,
        "no_gt_frame_false_positives": 1,
        "pred_detections": 40,
        "off_court_false_positive_frames": 0,
        "id_switches": 0,
        "expected_four_player_frames": 10,
        "exact_four_player_frames": 8,
    }
    modes = summarize_score_failure_modes_v2(row)
    modes_by_name = {mode["mode"]: mode for mode in modes}
    assert modes[0]["mode"] == "near_miss_localization"
    assert modes_by_name["near_miss_localization"]["count"] == 20
    assert modes_by_name["true_spectator_or_background_false_positives"]["count"] == 2
    assert modes_by_name["no_gt_frame_false_positives"]["count"] == 1
    assert "detector localization noise" in modes_by_name["near_miss_localization"]["description"]


def test_build_source_promotion_decision_requires_all_clips_and_clean_identity_gate() -> None:
    rows = [
        {
            "clip_id": "clip_a",
            "idf1": 0.91,
            "id_switches": 0,
            "spectator_or_background_false_positives": 0,
            "off_court_false_positive_frames": 0,
            "four_player_coverage": 0.96,
        },
        {
            "clip_id": "clip_b",
            "idf1": 0.84,
            "id_switches": 0,
            "spectator_or_background_false_positives": 0,
            "off_court_false_positive_frames": 0,
            "four_player_coverage": 0.97,
        },
    ]

    decision = build_source_promotion_decision(rows, required_clip_ids=["clip_a", "clip_b", "clip_c"])

    assert decision["promote"] is False
    assert "missing_required_clips:clip_c" in decision["blockers"]
    assert "clip_b:idf1_below_0.85" in decision["blockers"]


def test_scoring_report_adds_failure_mode_summary_per_row_and_source() -> None:
    rows = [
        {
            "clip_id": "clip_a",
            "track_source_id": "source_a",
            "idf1": 0.5,
            "mota": 0.2,
            "id_switches": 3,
            "spectator_or_background_false_positives": 8,
            "off_court_false_positive_frames": 2,
            "false_positives": 8,
            "false_negatives": 40,
            "gt_detections": 100,
            "pred_detections": 68,
            "matches": 60,
            "four_player_coverage": 0.75,
            "expected_four_player_frames": 20,
            "exact_four_player_frames": 15,
            "track_count": 4,
            "tracks_path": "runs/source_a/clip_a/tracks.json",
        },
        {
            "clip_id": "clip_b",
            "track_source_id": "source_a",
            "idf1": 0.7,
            "mota": 0.3,
            "id_switches": 1,
            "spectator_or_background_false_positives": 4,
            "off_court_false_positive_frames": 3,
            "false_positives": 4,
            "false_negatives": 6,
            "gt_detections": 50,
            "pred_detections": 48,
            "matches": 44,
            "four_player_coverage": 0.9,
            "expected_four_player_frames": 10,
            "exact_four_player_frames": 9,
            "track_count": 4,
            "tracks_path": "runs/source_a/clip_b/tracks.json",
        },
    ]

    report = build_scoring_report(rows, required_clip_ids=["clip_a", "clip_b"], iou_threshold=0.5)
    source = report["sources"][0]

    assert source["failure_analysis"]["primary_failure_mode"] == "missing_gt_detections"
    assert source["failure_analysis"]["modes"][0]["count"] == 46
    assert source["rows"][0]["primary_failure_mode"] == "missing_gt_detections"
    assert {mode["mode"] for mode in source["rows"][0]["failure_modes"]} >= {
        "id_switches",
        "spectator_or_background_false_positives",
        "off_court_false_positives",
        "four_player_coverage_gap",
    }
    markdown = render_scoring_report_markdown(report)
    assert "Mean HOTA" in markdown
    assert "Worst HOTA" in markdown
    source_line = next(line for line in markdown.splitlines() if line.startswith("| `source_a` |"))
    assert source_line.count("|") == 16
    assert "| missing_gt_detections |" in source_line

    # Gate v2: both rows predate the v2 decomposition, so the narrower
    # true-spectator field falls back to the v1 aggregate -- neither clip
    # silently passes just because it wasn't scored under v2.
    assert report["schema_version"] == 2
    decision_v2 = source["decision_v2"]
    assert decision_v2["status"] == "do_not_promote"
    assert "clip_a:true_spectator_or_background_false_positives_present" in decision_v2["blockers"]
    assert "clip_b:true_spectator_or_background_false_positives_present" in decision_v2["blockers"]
    failure_modes_v2 = {mode["mode"]: mode for mode in source["failure_analysis_v2"]["modes"]}
    assert failure_modes_v2["true_spectator_or_background_false_positives"]["count"] == 12
    assert "near_miss_localization" not in failure_modes_v2
    assert "## Source Decisions (Gate v2)" in markdown
    assert "Gate v2 promotion policy" in markdown

    # Gate v2.1 is reported alongside v1/v2, never replacing them. Both rows
    # predate the v2.1 apron split too, so the narrower
    # far_off_court_false_positive_frames field falls back to the legacy
    # off_court_false_positive_frames value -- same never-silently-pass
    # discipline as the v2 true-spectator fallback above.
    assert "decision_v2_1" in source
    assert "failure_analysis_v2_1" in source
    assert "promotion_policy_v2_1" in report
    assert report["promotion_policy_v2_1"]["prospective_only"] is True
    decision_v2_1 = source["decision_v2_1"]
    assert decision_v2_1["status"] == "do_not_promote"
    assert "clip_a:far_off_court_false_positives_present" in decision_v2_1["blockers"]
    assert "clip_b:far_off_court_false_positives_present" in decision_v2_1["blockers"]
    assert "## Source Decisions (Gate v2.1)" in markdown
    assert "Gate v2.1 promotion policy" in markdown
    assert "PROSPECTIVE ONLY" in markdown


def test_derive_track_source_id_groups_phase2_and_canonical_paths() -> None:
    clip_ids = ["burlington_gold_0300_low_steep_corner", "wolverine_mixed_0200_mid_steep_corner"]

    phase2 = derive_track_source_id(
        "runs/phase2/person_tracking_h100_final_modes_fullclips/yolo26n_fulltb3/"
        "burlington_gold_0300_low_steep_corner/yolo26n_fulltb3/tracks.json",
        clip_ids=clip_ids,
    )
    canonical = derive_track_source_id(
        "runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/tracks.json",
        clip_ids=clip_ids,
    )

    assert phase2 == "phase2/person_tracking_h100_final_modes_fullclips/yolo26n_fulltb3"
    assert canonical == "eval0/prototype_gate_h100_v2/canonical_tracks"


def test_score_person_track_sources_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    gt_dir = cvat_root / "clip_a"
    gt_dir.mkdir(parents=True)
    (gt_dir / "person_ground_truth.json").write_text(
        json.dumps(_ground_truth().model_dump(mode="json")),
        encoding="utf-8",
    )
    runs_root = tmp_path / "runs"
    track_dir = runs_root / "phase2" / "source_a" / "clip_a" / "candidate_a"
    track_dir.mkdir(parents=True)
    (track_dir / "tracks.json").write_text(
        json.dumps(_tracks_with_switch_fp_and_tail().model_dump(mode="json")),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/score_person_track_sources.py",
            "--cvat-root",
            str(cvat_root),
            "--runs-root",
            str(runs_root),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "person_track_gt_scoring_report.json").read_text(encoding="utf-8"))
    assert payload["track_file_count"] == 1
    assert payload["sources"][0]["decision"]["status"] == "do_not_promote"
    assert (out_dir / "PERSON_TRACK_GT_SCORING_REPORT.md").exists()


def test_score_person_track_sources_cli_applies_metrics_coordinate_scale(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    gt_dir = cvat_root / "clip_scaled"
    gt_dir.mkdir(parents=True)
    gt = PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_scaled",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [{"frame_index": 0, "source_frame_id": 1, "labels": [_person_label(1, 0, 20.0)]}],
            "summary": {
                "frame_count": 1,
                "valid_label_count": 1,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )
    (gt_dir / "person_ground_truth.json").write_text(json.dumps(gt.model_dump(mode="json")), encoding="utf-8")
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[_track_frame(0, (10.0, 0.0, 20.0, 5.0), [0.0, 0.0])],
            )
        ],
        rally_spans=[],
    )
    track_dir = tmp_path / "runs" / "phase2" / "source_scaled" / "clip_scaled" / "candidate_scaled"
    track_dir.mkdir(parents=True)
    (track_dir / "tracks.json").write_text(json.dumps(tracks.model_dump(mode="json")), encoding="utf-8")
    (track_dir / "metrics.json").write_text(
        json.dumps(
            {
                "counts": {
                    "source_width": 1920,
                    "source_height": 1080,
                    "calibration_width": 960,
                    "calibration_height": 540,
                }
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/score_person_track_sources.py",
            "--cvat-root",
            str(cvat_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "person_track_gt_scoring_report.json").read_text(encoding="utf-8"))
    row = payload["sources"][0]["rows"][0]
    assert row["bbox_scale_x"] == pytest.approx(2.0)
    assert row["bbox_scale_y"] == pytest.approx(2.0)
    assert row["idf1"] == pytest.approx(1.0)


def test_score_person_track_sources_cli_applies_offline_authority_score_scale(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    gt_dir = cvat_root / "clip_scaled"
    gt_dir.mkdir(parents=True)
    gt = PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_scaled",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [{"frame_index": 0, "source_frame_id": 1, "labels": [_person_label(1, 0, 20.0)]}],
            "summary": {
                "frame_count": 1,
                "valid_label_count": 1,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )
    (gt_dir / "person_ground_truth.json").write_text(json.dumps(gt.model_dump(mode="json")), encoding="utf-8")
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[_track_frame(0, (10.0, 0.0, 20.0, 5.0), [0.0, 0.0])],
            )
        ],
        rally_spans=[],
    )
    track_dir = tmp_path / "runs" / "phase2" / "offline_authority" / "clip_scaled" / "candidate_scaled"
    track_dir.mkdir(parents=True)
    (track_dir / "tracks.json").write_text(json.dumps(tracks.model_dump(mode="json")), encoding="utf-8")
    (track_dir / "metrics.json").write_text(
        json.dumps({"counts": {"score_bbox_scale_x": 2.0, "score_bbox_scale_y": 2.0}}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/score_person_track_sources.py",
            "--cvat-root",
            str(cvat_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "person_track_gt_scoring_report.json").read_text(encoding="utf-8"))
    row = payload["sources"][0]["rows"][0]
    assert row["bbox_scale_x"] == pytest.approx(2.0)
    assert row["bbox_scale_y"] == pytest.approx(2.0)
    assert row["bbox_scale_source"] == "metrics_score_bbox_scale"
    assert row["idf1"] == pytest.approx(1.0)


def test_score_person_track_sources_cli_wires_image_dims_into_outside_image_bucket(tmp_path: Path) -> None:
    """Regression test for review finding F4 (2026-07-02, MEDIUM-HIGH).

    The batch CLI already loads each clip's native resolution from the CVAT import
    manifest (`_load_clip_resolutions`) but previously never passed it into
    `score_tracks_against_person_ground_truth`, so gate v2's `outside_image_false_positives`
    bucket was always 0 out of this CLI and a genuinely out-of-frame prediction box was
    silently miscounted as `true_spectator_or_background_false_positives` instead --
    sending debugging toward detector/spectator filtering instead of the real
    coordinate-space failure.
    """

    cvat_root = tmp_path / "cvat"
    gt_dir = cvat_root / "clip_outside_image"
    gt_dir.mkdir(parents=True)
    gt = PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_outside_image",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [{"frame_index": 0, "source_frame_id": 1, "labels": [_person_label(1, 0, 0.0)]}],
            "summary": {
                "frame_count": 1,
                "valid_label_count": 1,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )
    (gt_dir / "person_ground_truth.json").write_text(json.dumps(gt.model_dump(mode="json")), encoding="utf-8")
    # The CVAT import manifest declares this clip's native (image-space) resolution --
    # exactly what `_load_clip_resolutions` reads and what the fix must thread through.
    (cvat_root / "manifest.json").write_text(
        json.dumps({"clips": [{"clip_id": "clip_outside_image", "resolution": [100, 100]}]}),
        encoding="utf-8",
    )
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                # Matches the single GT box exactly -- a true positive.
                frames=[_track_frame(0, (0.0, 0.0, 10.0, 10.0), [0.0, 0.0])],
            ),
            PlayerTrack(
                id=2,
                side="near",
                role="right",
                # No overlap with any GT box, and its center (505, 505) is well outside
                # the clip's declared 100x100 resolution -- must land in
                # outside_image_false_positives, not true_spectator_or_background.
                frames=[_track_frame(0, (500.0, 500.0, 510.0, 510.0), [5.0, 5.0])],
            ),
        ],
        rally_spans=[],
    )
    track_dir = tmp_path / "runs" / "phase2" / "source_outside" / "clip_outside_image" / "candidate_outside"
    track_dir.mkdir(parents=True)
    (track_dir / "tracks.json").write_text(json.dumps(tracks.model_dump(mode="json")), encoding="utf-8")
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/score_person_track_sources.py",
            "--cvat-root",
            str(cvat_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "person_track_gt_scoring_report.json").read_text(encoding="utf-8"))
    row = payload["sources"][0]["rows"][0]
    assert row["outside_image_false_positives"] == 1
    assert row["true_spectator_or_background_false_positives"] == 0
