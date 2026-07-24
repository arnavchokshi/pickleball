from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.racketsport import select_players_from_pool as selection_cli
from tests.racketsport.json_schema_assertions import assert_matches_json_schema
from threed.racketsport import player_selection as player_selection_module
from threed.racketsport.player_selection import (
    COURT_REGION_HARD_BOUND_M,
    DROP_TRIGGER_REASON_VOCABULARY,
    EXACTLY_FOUR_HARD_CAP,
    HardConstraintConfig,
    OpenSetDecision,
    PlayerSelectionConfig,
    SelectionDetection,
    SelectionSlot,
    TrackFragment,
    destructive_action_allowed,
    enroll_slots,
    evaluate_stitch,
    fusion_score,
    infer_active_player_count,
    mark_micro_fill_provenance,
    median_real_footpoint_court_excess_m,
    micro_fill_allowed,
    open_set_decision,
    recover_identity_conditioned_pool,
    registered_hard_constraint_exclusions,
    select_players_payload,
)
from threed.racketsport.schemas import CourtCalibration, PlayerTrack, TrackFrame, Tracks


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSIS = ROOT / "runs/lanes/trkL_selection_20260717/diagnosis"


def _vector_at_cosine_distance(distance: float) -> tuple[float, float]:
    cosine = 1.0 - distance
    return (cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)))


def _detection(
    frame: int,
    source: int,
    xy: tuple[float, float],
    embedding: tuple[float, ...] | None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    conf: float = 0.9,
    interpolated: bool = False,
    raw_detection_uid: str | None = None,
) -> SelectionDetection:
    return SelectionDetection(
        frame_idx=frame,
        source_track_id=source,
        bbox=bbox or (xy[0], xy[1], xy[0] + 0.5, xy[1] + 1.0),
        world_xy=xy,
        conf=conf,
        embedding=embedding,
        interpolated=interpolated,
        raw_detection_uid=raw_detection_uid,
    )


def test_auto_player_count_selects_singles_or_doubles_without_interpolation_backfill() -> None:
    singles = [
        _detection(frame, source, (-1.0 if source == 1 else 1.0, -2.0), (1.0, 0.0))
        for frame in range(10)
        for source in (1, 2)
    ]
    assert infer_active_player_count(singles, fps=30.0) == 2

    doubles = [
        _detection(
            frame,
            source,
            (-1.0 if source % 2 else 1.0, -2.0 if source <= 2 else 2.0),
            (1.0, 0.0),
        )
        for frame in range(10)
        for source in (1, 2, 3, 4)
    ]
    doubles.extend(
        _detection(frame, source, (0.0, 0.0), None, interpolated=True)
        for frame in range(10)
        for source in (5, 6)
    )
    assert infer_active_player_count(doubles, fps=30.0) == 4


def test_measured_association_fallback_preserves_ids_and_excludes_interpolation() -> None:
    fps = 30.0
    players = []
    detections: list[SelectionDetection] = []
    fragments: list[TrackFragment] = []
    for source in (1, 2, 3, 4):
        x = -1.0 if source % 2 else 1.0
        y = -2.0 if source <= 2 else 2.0
        measured = tuple(
            _detection(
                frame,
                source,
                (x, y),
                (1.0, 0.0),
                bbox=(source * 10.0, 20.0, source * 10.0 + 5.0, 40.0),
                raw_detection_uid=f"raw:{frame}:{source}",
            )
            for frame in range(10)
        )
        detections.extend(measured)
        fragments.append(
            TrackFragment(
                fragment_id=f"pool-{source}-1-0-9",
                source_track_id=source,
                detections=measured,
            )
        )
        frames = [
            {
                "t": detection.frame_idx / fps,
                "bbox": list(detection.bbox),
                "world_xy": list(detection.world_xy),
                "conf": detection.conf,
            }
            for detection in measured
        ]
        frames.append(
            {
                "t": 10 / fps,
                "bbox": [999.0, 999.0, 1000.0, 1000.0],
                "world_xy": [x, y],
                "conf": 0.1,
                "interpolated": True,
            }
        )
        players.append(
            {
                "id": source,
                "side": "near" if y < 0.0 else "far",
                "role": "left" if x < 0.0 else "right",
                "frames": frames,
            }
        )
    real_by_frame: dict[int, list[SelectionDetection]] = {}
    for detection in detections:
        real_by_frame.setdefault(detection.frame_idx, []).append(detection)

    result = player_selection_module._select_measured_association_fallback(
        players,
        real_by_frame=real_by_frame,
        pool_fragments=fragments,
        fps=fps,
        config=PlayerSelectionConfig(expected_players=4),
    )

    assert result is not None
    selected, unbound, decisions, _rows, counts = result
    assert [player["id"] for player in selected] == [1, 2, 3, 4]
    assert all(len(player["frames"]) == 10 for player in selected)
    assert not any(
        frame.get("interpolated") is True
        for player in selected
        for frame in player["frames"]
    )
    assert unbound == []
    assert counts["interpolated_frames"] == 0
    assert len(decisions) == 4


def test_measured_association_fallback_recovers_one_identity_supported_baseline_candidate() -> None:
    fps = 30.0
    basis = {
        1: (1.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0),
    }
    players: list[dict[str, object]] = []
    detections: list[SelectionDetection] = []
    for source in (1, 2, 3, 4):
        x = -1.0 if source % 2 else 1.0
        y = -2.0 if source <= 2 else 2.0
        frames = []
        for frame_idx in range(10):
            detection_y = 9.5 if source == 4 and frame_idx == 5 else y
            detection = _detection(
                frame_idx,
                source,
                (x, detection_y),
                basis[source],
                bbox=(source * 10.0, 20.0, source * 10.0 + 5.0, 40.0),
                raw_detection_uid=f"raw:{frame_idx}:{source}",
            )
            detections.append(detection)
            if source == 4 and frame_idx == 5:
                continue
            frames.append(
                {
                    "t": frame_idx / fps,
                    "bbox": list(detection.bbox),
                    "world_xy": list(detection.world_xy),
                    "conf": detection.conf,
                }
            )
        players.append(
            {
                "id": source,
                "side": "near" if y < 0.0 else "far",
                "role": "left" if x < 0.0 else "right",
                "frames": frames,
            }
        )
    real_by_frame: dict[int, list[SelectionDetection]] = defaultdict(list)
    for detection in detections:
        real_by_frame[detection.frame_idx].append(detection)
    fragments = [
        TrackFragment(
            fragment_id=f"pool-{source}-1-0-9",
            source_track_id=source,
            detections=tuple(detection for detection in detections if detection.source_track_id == source),
        )
        for source in (1, 2, 3, 4)
    ]

    result = player_selection_module._select_measured_association_fallback(
        players,
        real_by_frame=real_by_frame,
        pool_fragments=fragments,
        fps=fps,
        config=PlayerSelectionConfig(expected_players=4),
    )

    assert result is not None
    selected, _unbound, decisions, _rows, counts = result
    player_four = next(player for player in selected if player["id"] == 4)
    recovered = next(frame for frame in player_four["frames"] if frame["frame_idx"] == 5)
    assert recovered["world_xy"] == [1.0, 9.5]
    assert recovered.get("interpolated") is not True
    assert counts["recovered_real_detections"] == 1
    assert any(
        decision.get("action") == "recover_measured_enrolled_player"
        and decision.get("player_id") == 4
        and decision.get("frame_idx") == 5
        for decision in decisions
    )


def _fragment(
    fragment_id: str,
    source: int,
    start: int,
    end: int,
    xy: tuple[float, float],
    embedding: tuple[float, ...] | None,
    *,
    bbox_x: float = 0.0,
) -> TrackFragment:
    return TrackFragment(
        fragment_id=fragment_id,
        source_track_id=source,
        detections=tuple(
            _detection(
                frame,
                source,
                xy,
                embedding,
                bbox=(bbox_x, 0.0, bbox_x + 1.0, 2.0),
            )
            for frame in range(start, end + 1)
        ),
    )


def _identity_calibration() -> CourtCalibration:
    return CourtCalibration.model_validate(
        {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "intrinsics": {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 0.0,
                "cy": 0.0,
                "dist": [],
                "source": "test",
            },
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 5.0],
                "camera_height_m": 5.0,
            },
            "reprojection_error_px": {"median": 0.0, "p95": 0.0},
            "capture_quality": {"grade": "good", "reasons": []},
            "image_pts": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "world_pts": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        }
    )


def _raw_selection_inputs(
    records: list[dict[str, object]],
    *,
    feature_dim: int = 2,
) -> tuple[dict[str, object], dict[str, object]]:
    by_frame: dict[int, list[dict[str, object]]] = {}
    for record in records:
        frame_idx = int(record["frame"])
        by_frame.setdefault(frame_idx, []).append(record)
    raw_frames: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    for frame_idx in sorted(by_frame):
        raw_detections: list[dict[str, object]] = []
        for detection_index, record in enumerate(by_frame[frame_idx]):
            source = int(record["source"])
            x, y = record["xy"]  # type: ignore[misc]
            bbox = record.get(
                "bbox",
                (float(x) - 0.25, float(y) - 1.0, float(x) + 0.25, float(y)),
            )
            raw_detections.append(
                {
                    "bbox": list(bbox),  # type: ignore[arg-type]
                    "class": "person",
                    "conf": float(record.get("conf", 0.9)),
                    "track_id": source,
                }
            )
            embedding = record.get("embedding")
            if embedding is not None:
                embedding_rows.append(
                    {
                        "frame": frame_idx,
                        "source_track_id": source,
                        "detection_index": detection_index,
                        "bbox": list(bbox),  # type: ignore[arg-type]
                        "embedding": list(embedding),  # type: ignore[arg-type]
                    }
                )
        raw_frames.append({"frame": frame_idx, "detections": raw_detections})
    return (
        {"schema_version": 1, "fps": 30.0, "frames": raw_frames},
        {
            "schema_version": 1,
            "source_only": True,
            "uses_cvat_labels": False,
            "promote_trk": False,
            "feature_dim": feature_dim,
            "l2_normalized": True,
            "detections": embedding_rows,
        },
    )


def _empty_tracks() -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [],
        "rally_spans": [],
    }


def _four_slot_records(
    *,
    end_frame: int = 29,
) -> tuple[
    list[dict[str, object]],
    dict[int, tuple[float, float]],
    dict[int, tuple[float, float]],
]:
    positions = {
        1: (-1.5, -3.0),
        2: (1.5, -3.0),
        3: (-1.5, 3.0),
        4: (1.5, 3.0),
    }
    embeddings = {
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (-1.0, 0.0),
        4: (0.0, -1.0),
    }
    records = [
        {
            "frame": frame_idx,
            "source": source,
            "xy": positions[source],
            "embedding": embeddings[source],
        }
        for frame_idx in range(end_frame + 1)
        for source in sorted(positions)
    ]
    return records, positions, embeddings


def test_registered_defaults_are_frozen_values() -> None:
    cfg = PlayerSelectionConfig()
    assert cfg.sigma_court_m == 0.5
    assert cfg.court_ema_half_life_s == 2.0
    assert cfg.identity_accept_distance == 0.35
    assert cfg.identity_reject_distance == 0.42
    assert cfg.max_displacement_m == 2.5
    assert cfg.max_micro_fill_frames == 12
    assert cfg.fusion_weights == (0.4, 0.4, 0.2)
    assert cfg.selection_score_min == 0.5
    assert cfg.recovery_max_speed_m_s == 7.0
    assert EXACTLY_FOUR_HARD_CAP == 4
    assert COURT_REGION_HARD_BOUND_M == 1.0
    assert HardConstraintConfig().exactly_four_hard_cap == 4
    assert HardConstraintConfig().court_region_hard_bound_m == 1.0
    with pytest.raises(Exception):
        cfg.identity_accept_distance = 0.36  # type: ignore[misc]


def test_hard_court_median_uses_real_detections_only() -> None:
    detections = (
        _detection(0, 1, (0.0, -2.0), (1.0, 0.0)),
        _detection(1, 1, (0.0, -2.0), (1.0, 0.0)),
        _detection(2, 1, (20.0, -2.0), None, interpolated=True),
    )
    assert median_real_footpoint_court_excess_m(detections) == 0.0
    excluded, decisions = registered_hard_constraint_exclusions(detections)
    assert excluded == frozenset()
    assert decisions[0]["real_detection_count"] == 2


def _run_mutation_sensitive_six_track_fixture() -> tuple[dict[str, object], dict[str, object]]:
    positions = {
        1: (-1.5, -3.0),
        2: (1.5, -3.0),
        3: (-1.5, 3.0),
        4: (1.5, 3.0),
    }
    basis = {
        1: (1.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0, 0.0),
    }
    records = [
        {
            "frame": frame_idx,
            "source": source,
            "xy": positions[source],
            "embedding": basis[source],
        }
        for frame_idx in range(31)
        for source in sorted(positions)
    ]
    for frame_idx in range(31, 81):
        records.append(
            {
                "frame": frame_idx,
                "source": 51,
                "xy": (-1.5 - 0.2 * (frame_idx - 31), -3.0),
                "embedding": basis[1],
            }
        )
    for frame_idx in range(45, 48):
        records.append(
            {
                "frame": frame_idx,
                "source": 52,
                "xy": (0.0, -2.0),
                "embedding": (0.0, 0.0, 0.0, 0.0, 1.0),
            }
        )
    raw_pool, embedding_payload = _raw_selection_inputs(records, feature_dim=5)
    return select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )


def _assert_mutation_sensitive_six_track_fixture(
    selected: dict[str, object], report: dict[str, object]
) -> None:
    players = selected["players"]
    assert isinstance(players, list)
    assert len(players) == EXACTLY_FOUR_HARD_CAP
    assert all(
        len(
            [
                player
                for player in players
                if isinstance(player, dict)
                and any(
                    isinstance(frame, dict) and frame.get("frame_idx") == frame_idx
                    for frame in player.get("frames", [])
                )
            ]
        )
        <= EXACTLY_FOUR_HARD_CAP
        for frame_idx in range(81)
    )
    decisions = report["decisions"]
    assert isinstance(decisions, list)
    hard_rows = [
        row
        for row in decisions
        if isinstance(row, dict)
        and row.get("action") == "hard_exclude_court_region"
        and row.get("source_track_id") == 51
    ]
    assert hard_rows
    hard = hard_rows[0]
    assert hard["median_real_footpoint_court_excess_m"] > COURT_REGION_HARD_BOUND_M
    assert any(
        isinstance(row, dict)
        and row.get("action") == "hard_drop_bound_detection"
        and row.get("source_track_id") == 51
        for row in decisions
    )
    bound_uids = {
        uid
        for row in report["tracks"]  # type: ignore[index]
        if isinstance(row, dict) and row.get("selection_state") == "bound_slot"
        for uid in row.get("raw_detection_uids", [])
    }
    walk_through_uids = {
        uid
        for row in decisions
        if isinstance(row, dict)
        and str(row.get("fragment_id", "")).startswith("pool-52-")
        for uid in row.get("raw_detection_uids", [])
    }
    assert walk_through_uids
    assert not (walk_through_uids & bound_uids)


def test_hard_cap_six_track_fixture_is_caused_by_registered_post_filter() -> None:
    selected, report = _run_mutation_sensitive_six_track_fixture()
    _assert_mutation_sensitive_six_track_fixture(selected, report)


def test_hard_cap_fixture_rejects_disabled_filter_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        player_selection_module,
        "registered_hard_constraint_exclusions",
        lambda *args, **kwargs: (frozenset(), []),
    )
    selected, report = _run_mutation_sensitive_six_track_fixture()
    with pytest.raises(AssertionError):
        _assert_mutation_sensitive_six_track_fixture(selected, report)


def _selective_reid_summary(report: dict[str, object]) -> dict[str, object]:
    decisions = report["decisions"]
    assert isinstance(decisions, list)
    summaries = [
        row
        for row in decisions
        if isinstance(row, dict)
        and row.get("action") == "selective_reid_policy_summary"
    ]
    assert len(summaries) == 1
    return summaries[0]


def test_exact_four_unambiguous_frames_never_call_cosine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, _positions, _embeddings = _four_slot_records()
    raw_pool, embedding_payload = _raw_selection_inputs(records)
    calls = 0
    original = player_selection_module.cosine_distance

    def instrumented(left: object, right: object) -> float | None:
        nonlocal calls
        calls += 1
        return original(left, right)  # type: ignore[arg-type]

    monkeypatch.setattr(player_selection_module, "cosine_distance", instrumented)
    _selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    summary = _selective_reid_summary(report)
    assert calls == 0
    assert summary["frames_ambiguous"] == 0
    assert summary["reid_invoked"] == 0
    assert summary["reid_skipped_unambiguous"] == 30
    assert summary["reid_unavailable"] == 0


def test_provider_absence_is_loud_on_empty_input() -> None:
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload={"schema_version": 1, "fps": 30.0, "frames": []},
        embedding_payload=None,
        calibration=_identity_calibration(),
        enabled=True,
        reid_provider_available=False,
        reid_provider_reason="embedding_provider_checkpoint_absent",
    )
    assert selected["players"] == []
    summary = _selective_reid_summary(report)
    assert summary["reid_unavailable"] == 1
    assert summary["provider_available"] is False
    assert summary["warning_codes"] == ["reid_unavailable"]


def test_provider_absence_is_loud_on_exactly_four_unambiguous_frames() -> None:
    records, _positions, _embeddings = _four_slot_records()
    for record in records:
        record["embedding"] = None
    raw_pool, _embedding_payload = _raw_selection_inputs(records)
    _selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=None,
        calibration=_identity_calibration(),
        enabled=True,
        reid_provider_available=False,
        reid_provider_reason="embedding_provider_checkpoint_absent",
    )
    summary = _selective_reid_summary(report)
    assert summary["frames_ambiguous"] == 0
    assert summary["reid_invoked"] == 0
    assert summary["reid_skipped_unambiguous"] == 30
    assert summary["reid_unavailable"] == 1


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, OpenSetDecision.ACCEPT),
        (0.35, OpenSetDecision.ACCEPT),
        (0.3500001, OpenSetDecision.DEFER),
        (0.4199999, OpenSetDecision.DEFER),
        (0.42, OpenSetDecision.REJECT),
        (0.8, OpenSetDecision.REJECT),
        (None, OpenSetDecision.DEFER),
    ],
)
def test_open_set_band_semantics(
    distance: float | None, expected: OpenSetDecision
) -> None:
    assert open_set_decision(distance) is expected


def test_fusion_uses_all_registered_weights() -> None:
    assert fusion_score(
        court_presence=1.0, identity_match=0.0, persistence=0.5
    ) == pytest.approx(0.5)
    assert fusion_score(
        court_presence=0.0, identity_match=1.0, persistence=0.5
    ) == pytest.approx(0.5)
    assert not destructive_action_allowed(
        {"appearance": True, "geometry": False}
    )
    assert not destructive_action_allowed({"appearance": False, "geometry": True})
    assert destructive_action_allowed({"appearance": True, "geometry": True})
    with pytest.raises(ValueError, match="non-independent"):
        destructive_action_allowed({"role": True, "kinematics": True})


def test_wolverine_bridge_fixture_refused_and_gt1_rebind_accepted() -> None:
    diag = json.loads(
        (DIAGNOSIS / "wolverine_rfdetr_l_p_diag.json").read_text(encoding="utf-8")
    )
    probe = json.loads(
        (DIAGNOSIS / "osnet_stitch_probe.json").read_text(encoding="utf-8")
    )
    ghost = diag["true_spectator_events"]
    assert [row["frame"] for row in ghost] == [59, 60, 61, 62]
    assert {row["conf"] for row in ghost} == {0.35}

    # Recover the bridge line's registered endpoints from committed f59-62 world points.
    step_x = (ghost[-1]["world_xy"][0] - ghost[0]["world_xy"][0]) / 3.0
    step_y = (ghost[-1]["world_xy"][1] - ghost[0]["world_xy"][1]) / 3.0
    left_xy = (
        ghost[0]["world_xy"][0] - 15.0 * step_x,
        ghost[0]["world_xy"][1] - 15.0 * step_y,
    )
    right_xy = (
        ghost[-1]["world_xy"][0] + 25.0 * step_x,
        ghost[-1]["world_xy"][1] + 25.0 * step_y,
    )

    anchor = (1.0, 0.0)
    bad_distance = probe["T4_preStitch_GT1|T4_postStitch_GT4"]
    left = _fragment("wolverine-t4-pre", 4, 44, 44, left_xy, anchor)
    right = _fragment(
        "wolverine-t4-post",
        4,
        87,
        87,
        right_xy,
        _vector_at_cosine_distance(bad_distance),
    )
    decision = evaluate_stitch(left, right)
    assert decision.refused
    assert decision.open_set is OpenSetDecision.REJECT
    assert decision.embedding_distance == pytest.approx(0.448, abs=1e-12)
    assert decision.displacement_m > 2.5
    assert decision.net_crossing
    assert set(decision.evidence_classes) == {"appearance", "geometry"}

    gt1_distance = probe["T4_preStitch_GT1|T1_GT1_continuation"]
    assert open_set_decision(gt1_distance) is OpenSetDecision.ACCEPT


def test_burlington_registered_three_frame_fill_is_untouched_except_provenance() -> (
    None
):
    diag = json.loads(
        (DIAGNOSIS / "burlington_rfdetr_l_p_diag.json").read_text(encoding="utf-8")
    )
    assert diag["counts_by_category"]["TRUE_SPECTATOR"] == 0
    left = _detection(10, 4, (-1.0, 3.0), (1.0, 0.0))
    right = _detection(14, 4, (0.14, 3.0), _vector_at_cosine_distance(0.10))
    allowed, reasons = micro_fill_allowed(left, right, identity_distance=0.10)
    assert allowed and reasons == ()
    original = [
        {
            "frame_idx": frame,
            "t": frame / 30.0,
            "bbox": [frame, 1, frame + 2, 5],
            "world_xy": [0, 3],
            "conf": 0.35,
        }
        for frame in (11, 12, 13)
    ]
    marked = mark_micro_fill_provenance(
        original, left=left, right=right, identity_distance=0.10
    )
    assert [
        {key: value for key, value in row.items() if key != "interpolated"}
        for row in marked
    ] == original
    assert all(row["interpolated"] is True for row in marked)


def test_identity_ambiguous_or_large_gap_is_never_micro_filled() -> None:
    left = _detection(0, 1, (0.0, -1.0), (1.0, 0.0))
    right = _detection(13, 1, (0.1, -1.0), _vector_at_cosine_distance(0.36))
    allowed, reasons = micro_fill_allowed(left, right, identity_distance=0.36)
    assert not allowed
    assert "same_identity_not_accepted" in reasons
    far = _detection(2, 1, (3.0, -1.0), (1.0, 0.0))
    assert not micro_fill_allowed(left, far, identity_distance=0.1)[0]
    across_net = _detection(2, 1, (0.0, 1.0), (1.0, 0.0))
    assert not micro_fill_allowed(left, across_net, identity_distance=0.1)[0]


def test_enrollment_is_deterministic_and_role_ordered() -> None:
    fragments = [
        _fragment("far-right", 40, 0, 29, (1.5, 3.0), (0.0, 1.0), bbox_x=30.0),
        _fragment("near-left", 10, 0, 29, (-1.5, -3.0), (1.0, 0.0), bbox_x=0.0),
        _fragment("far-left", 30, 0, 29, (-1.5, 3.0), (-1.0, 0.0), bbox_x=20.0),
        _fragment("near-right", 20, 0, 29, (1.5, -3.0), (0.0, -1.0), bbox_x=10.0),
    ]
    first = enroll_slots(fragments, fps=30.0)
    second = enroll_slots(list(reversed(fragments)), fps=30.0)
    assert first == second
    assert [(slot.side, slot.role) for slot in first] == [
        ("near", "left"),
        ("near", "right"),
        ("far", "left"),
        ("far", "right"),
    ]
    assert all(slot.enrolled_frames == tuple(range(30)) for slot in first)


def test_layer_c_recovery_uses_real_accept_band_detections_inside_motion_envelope() -> (
    None
):
    slot = SelectionSlot(1, "near", "left", (1.0, 0.0), (0,), ("seed",), 1.0)
    last = _detection(0, 1, (0.0, -2.0), (1.0, 0.0))
    pool = [
        _detection(1, 50, (0.1, -2.0), _vector_at_cosine_distance(0.30)),
        _detection(2, 51, (0.2, -2.0), _vector_at_cosine_distance(0.42)),
        _detection(3, 52, (100.0, -2.0), _vector_at_cosine_distance(0.10)),
        _detection(
            4, 53, (0.4, -2.0), _vector_at_cosine_distance(0.10), interpolated=True
        ),
    ]
    recovered = recover_identity_conditioned_pool(
        slot, last_detection=last, pool=pool, fps=30.0
    )
    assert [detection.frame_idx for detection in recovered] == [1]
    assert all(not detection.interpolated for detection in recovered)


def test_selection_off_payload_is_semantic_noop_and_report_is_honest() -> None:
    payload = {
        "schema_version": 1,
        "fps": 30.0,
        "players": [{"id": 1, "side": "near", "role": "left", "frames": []}],
        "rally_spans": [],
    }
    selected, report = select_players_payload(payload, enabled=False)
    assert selected == payload
    assert selected is not payload
    assert report["status"] == "disabled_noop"
    assert report["VERIFIED"] == 0
    assert report["preview_only"] is True
    assert report["input_counts"]["players"] == 1
    assert report["input_counts"]["track_frames"] == 0
    assert report["output_counts"]["players"] == 1
    assert report["output_counts"]["track_frames"] == 0


def test_enabled_selection_recovers_real_pool_and_marks_only_micro_fill() -> None:
    positions = {1: (-1.5, -3.0), 2: (1.5, -3.0), 3: (-1.5, 3.0), 4: (1.5, 3.0)}
    embeddings = {
        1: (1.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0),
    }
    raw_frames = []
    embedding_rows = []
    players = {track_id: [] for track_id in positions}
    for frame_idx in range(60):
        detections = []
        for detection_index, track_id in enumerate(sorted(positions)):
            if track_id == 1 and frame_idx in {40, 41, 42}:
                continue
            x, y = positions[track_id]
            bbox = [x - 0.25, y - 1.0, x + 0.25, y]
            detection = {
                "bbox": bbox,
                "class": "person",
                "conf": 0.9,
                "track_id": track_id,
            }
            detections.append(detection)
            embedding_rows.append(
                {
                    "frame": frame_idx,
                    "source_track_id": track_id,
                    "detection_index": len(detections) - 1,
                    "bbox": bbox,
                    "embedding": list(embeddings[track_id]),
                }
            )
            players[track_id].append(
                {"t": frame_idx / 30.0, "bbox": bbox, "world_xy": [x, y], "conf": 0.9}
            )
        raw_frames.append({"frame": frame_idx, "detections": detections})
    # Association's old three-frame fill is present but absent from the real pool.
    for frame_idx in (40, 41, 42):
        players[1].append(
            {
                "t": frame_idx / 30.0,
                "bbox": [-1.75, -4.0, -1.25, -3.0],
                "world_xy": [-1.5, -3.0],
                "conf": 0.35,
            }
        )
    tracks = {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": track_id,
                "side": "near" if positions[track_id][1] < 0 else "far",
                "role": "left" if positions[track_id][0] < 0 else "right",
                "frames": sorted(players[track_id], key=lambda row: row["t"]),
            }
            for track_id in sorted(players)
        ],
        "rally_spans": [],
    }
    raw_pool = {"schema_version": 1, "fps": 30.0, "frames": raw_frames}
    embedding_payload = {
        "schema_version": 1,
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "feature_dim": 4,
        "l2_normalized": True,
        "detections": embedding_rows,
    }
    selected, report = select_players_payload(
        tracks,
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    assert report["enrollment"]["slot_count"] == 4
    assert len(selected["players"]) == 4
    player_one = next(
        player
        for player in selected["players"]
        if player["side"] == "near" and player["role"] == "left"
    )
    assert len(player_one["frames"]) == 60
    assert [
        frame["frame_idx"] for frame in player_one["frames"] if frame["interpolated"]
    ] == [40, 41, 42]
    assert all(
        frame["interpolated"] is False
        for frame in player_one["frames"]
        if frame["frame_idx"] not in {40, 41, 42}
    )
    assert any(
        decision["action"] in {"bind", "rebind"} for decision in report["decisions"]
    )
    schema = json.loads(
        (ROOT / "docs/racketsport/player_selection_report_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert_matches_json_schema(report, schema)


def test_selection_off_cli_is_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "tracks.input.json"
    output = tmp_path / "tracks.output.json"
    report = tmp_path / "selection_report.json"
    original = b'{ "schema_version":1, "fps":30.0, "players":[], "rally_spans":[] }\n\n'
    source.write_bytes(original)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/racketsport/select_players_from_pool.py"),
            "--tracks",
            str(source),
            "--out-tracks",
            str(output),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes() == original
    assert not (tmp_path / "out.scoring_projection.json").exists()
    assert not (tmp_path / "out.scoring_projection.json.sha256").exists()


def test_report_schema_accepts_disabled_report() -> None:
    _, report = select_players_payload(
        {"schema_version": 1, "fps": 30.0, "players": [], "rally_spans": []}
    )
    schema = json.loads(
        (ROOT / "docs/racketsport/player_selection_report_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["type"] == "object"
    assert set(schema["required"]) <= set(report)
    assert report["artifact_type"] == schema["properties"]["artifact_type"]["const"]
    assert report["VERIFIED"] == schema["properties"]["VERIFIED"]["const"]
    assert report["status"] in schema["properties"]["status"]["enum"]


def test_cli_help_exposes_explicit_default_off_switch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/racketsport/select_players_from_pool.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--enable-selection" in completed.stdout
    assert "--raw-pool" in completed.stdout
    assert "--scoring-projection" in completed.stdout


def test_track_frame_interpolated_is_additive_and_survives_tracks_round_trip() -> None:
    legacy = TrackFrame(
        t=0.0,
        bbox=(0.0, 0.0, 1.0, 2.0),
        world_xy=(0.5, 2.0),
        conf=0.9,
    )
    assert legacy.interpolated is False
    assert "interpolated" not in legacy.model_dump(mode="json")
    with pytest.raises(ValueError):
        TrackFrame(
            t=0.0,
            bbox=(0.0, 0.0, 1.0, 2.0),
            world_xy=(0.5, 2.0),
            conf=0.9,
            interpolated=1,
        )
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                frames=[
                    TrackFrame(
                        frame_idx=1,
                        t=1.0 / 30.0,
                        bbox=(0.0, 0.0, 1.0, 2.0),
                        world_xy=(0.5, 2.0),
                        conf=0.35,
                        interpolated=True,
                    )
                ],
            )
        ],
        rally_spans=[],
    )
    dumped = tracks.model_dump(mode="json")
    assert dumped["players"][0]["frames"][0]["interpolated"] is True
    assert Tracks.model_validate(dumped).players[0].frames[0].interpolated is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_only", None),
        ("source_only", False),
        ("uses_cvat_labels", None),
        ("uses_cvat_labels", True),
        ("promote_trk", None),
        ("promote_trk", True),
    ],
)
def test_enabled_selection_requires_explicit_literal_source_attestations(
    field: str,
    replacement: object,
) -> None:
    embedding_payload: dict[str, object] = {
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "feature_dim": 2,
        "detections": [],
    }
    if replacement is None:
        del embedding_payload[field]
    else:
        embedding_payload[field] = replacement
    with pytest.raises(ValueError, match="attestation"):
        select_players_payload(
            _empty_tracks(),
            raw_pool_payload={"schema_version": 1, "fps": 30.0, "frames": []},
            embedding_payload=embedding_payload,
            calibration=_identity_calibration(),
            enabled=True,
        )


def test_no_enrollment_is_typed_partial_and_removes_wolverine_bridge_end_to_end() -> (
    None
):
    diag = json.loads(
        (DIAGNOSIS / "wolverine_rfdetr_l_p_diag.json").read_text(encoding="utf-8")
    )
    probe = json.loads(
        (DIAGNOSIS / "osnet_stitch_probe.json").read_text(encoding="utf-8")
    )
    ghost = diag["true_spectator_events"]
    step_x = (ghost[-1]["world_xy"][0] - ghost[0]["world_xy"][0]) / 3.0
    step_y = (ghost[-1]["world_xy"][1] - ghost[0]["world_xy"][1]) / 3.0
    left_xy = (
        ghost[0]["world_xy"][0] - 15.0 * step_x,
        ghost[0]["world_xy"][1] - 15.0 * step_y,
    )
    right_xy = (
        ghost[-1]["world_xy"][0] + 25.0 * step_x,
        ghost[-1]["world_xy"][1] + 25.0 * step_y,
    )
    bad_distance = probe["T4_preStitch_GT1|T4_postStitch_GT4"]
    records = [
        {"frame": 44, "source": 4, "xy": left_xy, "embedding": (1.0, 0.0)},
        {
            "frame": 87,
            "source": 4,
            "xy": right_xy,
            "embedding": _vector_at_cosine_distance(bad_distance),
        },
    ]
    raw_pool, embeddings = _raw_selection_inputs(records)
    associated_frames = []
    for frame_idx in range(44, 88):
        alpha = (frame_idx - 44) / (87 - 44)
        x = (1.0 - alpha) * left_xy[0] + alpha * right_xy[0]
        y = (1.0 - alpha) * left_xy[1] + alpha * right_xy[1]
        associated_frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "bbox": [x - 0.25, y - 1.0, x + 0.25, y],
                "world_xy": [x, y],
                "conf": 0.35 if 44 < frame_idx < 87 else 0.9,
            }
        )
    tracks = {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 4,
                "side": "near",
                "role": "left",
                "frames": associated_frames,
            }
        ],
        "rally_spans": [],
    }
    selected, report = select_players_payload(
        tracks,
        raw_pool_payload=raw_pool,
        embedding_payload=embeddings,
        calibration=_identity_calibration(),
        enabled=True,
    )
    assert report["status"] == "preview_selection_partial_no_enrollment"
    assert report["selection_mode"] == "partial_unbound_real_only"
    assert report["enrollment"]["slot_count"] == 0
    assert selected["players"] == []
    assert len(selected["unbound_observations"]) == 2
    output_frames = [
        frame["frame_idx"]
        for observation in selected["unbound_observations"]
        for frame in observation["frames"]
    ]
    assert output_frames == [44, 87]
    assert all(
        not ({44, 87} <= {frame["frame_idx"] for frame in observation["frames"]})
        for observation in selected["unbound_observations"]
    )
    assert not set(range(45, 87)) & set(output_frames)
    assert report["output_counts"]["synthetic_frames_removed"] == 42
    assert any(
        decision["action"] == "stitch_refused"
        and decision["left_fragment_id"] != decision["right_fragment_id"]
        for decision in report["decisions"]
    )
    assert all(
        set(decision.get("evidence_classes", ())) <= {"appearance", "geometry"}
        for decision in report["decisions"]
    )
    canonical_tracks = dict(selected)
    del canonical_tracks["unbound_observations"]
    Tracks.model_validate(canonical_tracks)


def test_one_raw_uid_is_joined_once_and_exports_only_raw_geometry() -> None:
    raw_pool, embeddings = _raw_selection_inputs(
        [
            {
                "frame": 0,
                "source": 7,
                "xy": (1.0, 2.0),
                "embedding": (1.0, 0.0),
                "conf": 0.73,
            }
        ]
    )
    raw_bbox = raw_pool["frames"][0]["detections"][0]["bbox"]  # type: ignore[index]
    tracks = {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "bbox": raw_bbox,
                        "world_xy": [99.0, 99.0],
                        "conf": 0.9,
                    }
                ],
            },
            {
                "id": 2,
                "side": "far",
                "role": "right",
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "bbox": raw_bbox,
                        "world_xy": [-99.0, -99.0],
                        "conf": 0.9,
                    }
                ],
            },
        ],
        "rally_spans": [],
    }
    selected, report = select_players_payload(
        tracks,
        raw_pool_payload=raw_pool,
        embedding_payload=embeddings,
        calibration=_identity_calibration(),
        enabled=True,
    )
    frames = [
        frame
        for observation in selected["unbound_observations"]
        for frame in observation["frames"]
    ]
    assert len(frames) == 1
    assert frames[0]["bbox"] == raw_bbox
    assert frames[0]["world_xy"] == pytest.approx([1.0, 2.0])
    assert frames[0]["conf"] == pytest.approx(0.73)
    assert frames[0]["interpolated"] is False
    assert report["input_counts"]["association_frames_joined_to_raw"] == 1
    exported_uids = [
        uid for track in report["tracks"] for uid in track["raw_detection_uids"]
    ]
    assert exported_uids == ["raw:0:0"]
    assert len(exported_uids) == len(set(exported_uids))


def test_enrollment_owners_bind_before_tied_slot_ranking() -> None:
    records, positions, _embeddings = _four_slot_records()
    identical_embedding = (1.0, 0.0)
    for record in records:
        record["embedding"] = identical_embedding
    raw_pool, embedding_payload = _raw_selection_inputs(records)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    assert report["enrollment"]["slot_count"] == 4
    assert [len(player["frames"]) for player in selected["players"]] == [30, 30, 30, 30]
    bind_decisions = [
        decision for decision in report["decisions"] if decision["action"] == "bind"
    ]
    assert len(bind_decisions) == 4
    assert len({decision["fragment_id"] for decision in bind_decisions}) == 4
    assert len({decision["slot_id"] for decision in bind_decisions}) == 4
    assert {(player["side"], player["role"]) for player in selected["players"]} == {
        ("near", "left"),
        ("near", "right"),
        ("far", "left"),
        ("far", "right"),
    }
    assert set(positions) == {1, 2, 3, 4}


def test_missing_enrollment_centroid_returns_typed_partial() -> None:
    records, _positions, _embeddings = _four_slot_records()
    for record in records:
        record["embedding"] = None
    raw_pool, embedding_payload = _raw_selection_inputs(records)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    assert report["status"] == "preview_selection_partial_no_enrollment"
    assert report["enrollment"]["slot_count"] == 0
    assert selected["players"] == []
    assert sum(
        len(observation["frames"])
        for observation in selected["unbound_observations"]
    ) == 120


def test_defer_band_real_fragment_survives_unbound_and_role_motion_do_not_double_count() -> (
    None
):
    records, positions, embeddings = _four_slot_records()
    basis = {
        1: (1.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0, 0.0),
    }
    for record in records:
        record["embedding"] = basis[int(record["source"])]
    cosine = 1.0 - 0.36
    defer_embedding = (cosine, 0.0, 0.0, 0.0, math.sqrt(1.0 - cosine * cosine))
    for frame_idx in range(40, 43):
        records.append(
            {
                "frame": frame_idx,
                "source": 50,
                "xy": (1.5, -3.0),
                "embedding": defer_embedding,
            }
        )
    raw_pool, embedding_payload = _raw_selection_inputs(records, feature_dim=5)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    decision = next(
        row
        for row in report["decisions"]
        if row.get("fragment_id", "").startswith("pool-50-")
    )
    assert decision["action"] == "leave_unbound"
    assert decision["fusion_score"] == pytest.approx(0.9428571428571428)
    assert set(decision["evidence_classes"]) <= {"appearance", "geometry"}
    assert "role" not in decision["evidence_classes"]
    unbound = selected["unbound_observations"]
    assert len(unbound) == 1
    assert unbound[0]["selection_state"] == "unbound_abstention"
    assert [frame["frame_idx"] for frame in unbound[0]["frames"]] == [40, 41, 42]
    assert decision["output_unbound_observation_ids"] == [
        unbound[0]["observation_id"]
    ]
    assert positions[1] == (-1.5, -3.0)
    assert embeddings[1] == (1.0, 0.0)


def test_enabled_export_claims_only_bound_slots_and_preserves_unbound_observations() -> (
    None
):
    records, _positions, _embeddings = _four_slot_records()
    records.append(
        {
            "frame": 40,
            "source": 50,
            "xy": (0.0, -2.5),
            "embedding": None,
        }
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )

    bound_rows = [
        row for row in report["tracks"] if row["selection_state"] == "bound_slot"
    ]
    assert len(selected["players"]) == len(bound_rows) == 4
    assert len(selected["players"]) <= 4
    assert all(player["side"] != "unbound" for player in selected["players"])

    observations = selected["unbound_observations"]
    assert len(observations) == 1
    observation = observations[0]
    assert observation["selection_state"] == "unbound_abstention"
    assert observation["abstention_reasons"][0] == "not_bound_to_claimed_slot"
    assert observation["raw_detection_uids"] == ["raw:40:0"]
    assert [frame["frame_idx"] for frame in observation["frames"]] == [40]
    assert observation["frames"][0] not in [
        frame for player in selected["players"] for frame in player["frames"]
    ]
    assert report["output_counts"]["players"] == 4
    assert report["output_counts"]["unbound_observations"] == 1


def test_wolverine_f44_f87_refusal_separates_final_bound_ids() -> None:
    probe = json.loads(
        (DIAGNOSIS / "osnet_stitch_probe.json").read_text(encoding="utf-8")
    )
    bad_distance = probe["T4_preStitch_GT1|T4_postStitch_GT4"]
    records, positions, embeddings = _four_slot_records(end_frame=44)
    embeddings[4] = _vector_at_cosine_distance(bad_distance)
    for record in records:
        if record["source"] == 4:
            record["embedding"] = embeddings[4]
    records.append(
        {
            "frame": 87,
            "source": 1,
            "xy": positions[4],
            "embedding": embeddings[4],
        }
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    owner_f44 = next(
        player
        for player in selected["players"]
        if any(
            frame["frame_idx"] == 44 and frame["world_xy"] == list(positions[1])
            for frame in player["frames"]
        )
    )
    owner_f87 = next(
        player
        for player in selected["players"]
        if any(frame["frame_idx"] == 87 for frame in player["frames"])
    )
    assert owner_f44["id"] != owner_f87["id"]
    assert (owner_f87["side"], owner_f87["role"]) == ("far", "right")
    assert not any(
        45 <= frame["frame_idx"] <= 86
        for player in selected["players"]
        for frame in player["frames"]
    )
    assert any(
        decision["action"] == "stitch_refused"
        and decision["left_fragment_id"].startswith("pool-1-")
        and decision["right_fragment_id"].startswith("pool-1-")
        for decision in report["decisions"]
    )


def test_continuous_same_tracker_appearance_switch_splits_final_output() -> None:
    raw_pool, embedding_payload = _raw_selection_inputs(
        [
            {
                "frame": 0,
                "source": 9,
                "xy": (0.0, -2.0),
                "embedding": (1.0, 0.0),
            },
            {
                "frame": 1,
                "source": 9,
                "xy": (0.05, -2.0),
                "embedding": (-1.0, 0.0),
            },
        ]
    )
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    assert report["status"] == "preview_selection_partial_no_enrollment"
    assert selected["players"] == []
    assert len(selected["unbound_observations"]) == 2
    assert [
        [frame["frame_idx"] for frame in observation["frames"]]
        for observation in selected["unbound_observations"]
    ] == [[0], [1]]
    fragment_decisions = [
        decision
        for decision in report["decisions"]
        if decision.get("action") == "leave_unbound"
    ]
    assert len(fragment_decisions) == 2
    assert len({decision["fragment_id"] for decision in fragment_decisions}) == 2


def test_enrollment_ownership_is_limited_to_registered_window_during_identity_switch() -> (
    None
):
    records, positions, embeddings = _four_slot_records(end_frame=33)
    for record in records:
        if int(record["source"]) != 1 or int(record["frame"]) < 30:
            continue
        record["xy"] = positions[4]
        record["embedding"] = None if int(record["frame"]) < 33 else embeddings[4]
    raw_pool, embedding_payload = _raw_selection_inputs(records)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    near_left = next(
        player
        for player in selected["players"]
        if (player["side"], player["role"]) == ("near", "left")
    )
    assert [frame["frame_idx"] for frame in near_left["frames"]] == list(range(30))
    far_right = next(
        player
        for player in selected["players"]
        if (player["side"], player["role"]) == ("far", "right")
    )
    assert [frame["frame_idx"] for frame in far_right["frames"]] == list(range(34))
    switched = next(
        decision
        for decision in report["decisions"]
        if str(decision.get("fragment_id", "")).startswith("pool-1-1-0-32:residual-")
    )
    assert switched["action"] == "leave_unbound"
    assert switched["output_unbound_observation_ids"]
    switched_output = next(
        observation
        for observation in selected["unbound_observations"]
        if observation["observation_id"]
        == switched["output_unbound_observation_ids"][0]
    )
    assert switched_output["selection_state"] == "unbound_abstention"
    assert [frame["frame_idx"] for frame in switched_output["frames"]] == [
        30,
        31,
        32,
    ]


def test_pool_fragment_appearance_split_uses_incremental_centroid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = tuple([1.0] + [0.0] * 127)
    detections = [
        _detection(frame, 1, (0.0, -2.0), embedding) for frame in range(2_400)
    ]

    def reject_prefix_recomputation(
        _detections: object,
    ) -> tuple[float, ...] | None:
        raise AssertionError("pool splitting recomputed a full prefix centroid")

    monkeypatch.setattr(
        player_selection_module,
        "embedding_centroid",
        reject_prefix_recomputation,
    )
    fragments = player_selection_module._pool_fragments(
        detections,
        config=PlayerSelectionConfig(),
    )
    assert len(fragments) == 1
    assert len(fragments[0].detections) == 2_400


def test_one_to_one_assignment_scales_for_crowded_same_start_pool() -> None:
    fragments = [
        _fragment(
            f"crowd-{index:03d}",
            index + 1,
            40,
            40,
            (0.0, -2.0),
            (1.0, 0.0),
        )
        for index in range(100)
    ]
    candidates = {
        fragment.fragment_id: [
            player_selection_module._SlotCandidate(
                fragment_id=fragment.fragment_id,
                slot_id=slot_id,
                fusion_score=0.9,
                embedding_distance=0.1,
                open_set=OpenSetDecision.ACCEPT,
                role_consistent=True,
                temporal_motion_consistent=True,
                stitch_vetoed=False,
                registered_owner_continuity=(
                    fragment.fragment_id == f"crowd-{slot_id - 1:03d}"
                ),
                feasible=True,
            )
            for slot_id in range(1, 5)
        ]
        for fragment in fragments
    }
    chosen = player_selection_module._choose_one_to_one_bindings(
        fragments,
        candidates,
    )
    assert len(chosen) == 4
    assert {candidate.slot_id for candidate in chosen.values()} == {1, 2, 3, 4}
    assert all(candidate.registered_owner_continuity for candidate in chosen.values())


def test_boundary_player_binds_by_registered_soft_fusion_below_half_court_score() -> (
    None
):
    records, positions, embeddings = _four_slot_records()
    boundary_xy = (-3.45, positions[1][1])
    records.append(
        {
            "frame": 60,
            "source": 1,
            "xy": boundary_xy,
            "embedding": embeddings[1],
        }
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records)

    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )

    decision = next(
        row
        for row in report["decisions"]
        if row.get("action") == "rebind"
        and str(row.get("fragment_id", "")).startswith("pool-1-2-")
    )
    assert decision["court_presence"] < 0.5
    assert decision["fusion_score"] >= PlayerSelectionConfig().selection_score_min
    assert decision["registered_owner_continuity"] is True
    assert "fusion_at_or_above_0_5" in decision["reasons"]
    assert not any("court_presence_at_or_above_0_5" in reason for reason in decision["reasons"])
    near_left = next(
        player
        for player in selected["players"]
        if (player["side"], player["role"]) == ("near", "left")
    )
    assert next(
        frame for frame in near_left["frames"] if frame["frame_idx"] == 60
    )["world_xy"] == pytest.approx(list(boundary_xy))


def test_source_id_reuse_follows_appearance_instead_of_enrollment_owner() -> None:
    records, positions, embeddings = _four_slot_records()
    records.append(
        {
            "frame": 60,
            "source": 1,
            "xy": positions[4],
            "embedding": embeddings[4],
        }
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records)

    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )

    assert any(
        decision["action"] == "stitch_refused"
        and decision.get("source_track_id") == 1
        for decision in report["decisions"]
    )
    rebind = next(
        decision
        for decision in report["decisions"]
        if decision.get("action") == "rebind"
        and str(decision.get("fragment_id", "")).startswith("pool-1-2-")
    )
    assert rebind["slot_id"] == 4
    assert rebind["registered_owner_continuity"] is False
    assert rebind["open_set"] == OpenSetDecision.ACCEPT.value
    far_right = next(
        player
        for player in selected["players"]
        if (player["side"], player["role"]) == ("far", "right")
    )
    assert next(
        frame for frame in far_right["frames"] if frame["frame_idx"] == 60
    )["world_xy"] == pytest.approx(list(positions[4]))


def test_registered_owner_never_bypasses_generic_stitch_veto() -> None:
    slot = SelectionSlot(
        1,
        "near",
        "left",
        (1.0, 0.0),
        (0,),
        ("arbitrary-owner-seed",),
        1.0,
    )
    registered = _fragment(
        "arbitrary-owner-seed",
        77,
        0,
        0,
        (-1.5, -3.0),
        (1.0, 0.0),
    )
    reentry = _fragment(
        "arbitrary-owner-reentry",
        77,
        30,
        30,
        (-1.5, -3.0),
        (1.0, 0.0),
    )

    candidate = player_selection_module._binding_candidates(
        reentry,
        slots=(slot,),
        assignments={1: (registered,)},
        stitch_vetoes={(registered.fragment_id, reentry.fragment_id)},
        fps=30.0,
        config=PlayerSelectionConfig(),
        owner_source_ids_by_slot={1: frozenset({77})},
    )[0]

    assert candidate.registered_owner_continuity
    assert candidate.open_set is OpenSetDecision.ACCEPT
    assert candidate.role_consistent
    assert candidate.temporal_motion_consistent
    assert candidate.fusion_score >= PlayerSelectionConfig().selection_score_min
    assert candidate.stitch_vetoed
    assert not candidate.feasible
    incident_candidate = player_selection_module._binding_candidates(
        reentry,
        slots=(slot,),
        assignments={1: (registered,)},
        stitch_vetoes={("unassigned-veto-counterpart", reentry.fragment_id)},
        fps=30.0,
        config=PlayerSelectionConfig(),
        owner_source_ids_by_slot={1: frozenset({77})},
    )[0]
    assert incident_candidate.registered_owner_continuity
    assert incident_candidate.stitch_vetoed
    assert not incident_candidate.feasible


def test_layer_c_cannot_recover_uid_vetoed_for_same_owner_slot() -> None:
    def measured(
        frame_idx: int,
        uid: str,
    ) -> SelectionDetection:
        return SelectionDetection(
            frame_idx=frame_idx,
            source_track_id=77,
            bbox=(0.0, 0.0, 1.0, 2.0),
            world_xy=(-1.5, -3.0),
            conf=0.9,
            embedding=(1.0, 0.0),
            raw_detection_uid=uid,
        )

    seed_detection = measured(0, "raw:0:0")
    vetoed_detection = measured(1, "raw:1:0")
    future_detection = measured(2, "raw:2:0")
    seed = TrackFragment("arbitrary-seed", 77, (seed_detection,))
    vetoed = TrackFragment("arbitrary-vetoed", 77, (vetoed_detection,))
    future = TrackFragment("arbitrary-future", 77, (future_detection,))
    slot = SelectionSlot(
        1,
        "near",
        "left",
        (1.0, 0.0),
        (0,),
        (seed.fragment_id,),
        1.0,
    )

    players, unbound, decisions, _tracks, _counts = (
        player_selection_module._select_slot_players(
            (slot,),
            pool_fragments=(seed, vetoed, future),
            real_pool=(seed_detection, vetoed_detection, future_detection),
            stitch_vetoes={(seed.fragment_id, vetoed.fragment_id)},
            fps=30.0,
            config=PlayerSelectionConfig(),
        )
    )

    assert [frame["frame_idx"] for frame in players[0]["frames"]] == [0, 2]
    assert not any(
        decision.get("action") == "recover_real"
        and decision.get("raw_detection_uid") == vetoed_detection.raw_detection_uid
        for decision in decisions
    )
    assert len(unbound) == 1
    assert unbound[0]["raw_detection_uids"] == [vetoed_detection.raw_detection_uid]
    veto_decision = next(
        decision
        for decision in decisions
        if decision.get("fragment_id") == vetoed.fragment_id
    )
    assert veto_decision["action"] == "leave_unbound"
    assert veto_decision["stitch_vetoed"] is True


def test_layer_c_recomputes_veto_against_later_final_assignment() -> None:
    def measured(
        frame_idx: int,
        source_track_id: int,
        xy: tuple[float, float],
        embedding: tuple[float, float],
        uid: str,
    ) -> SelectionDetection:
        return SelectionDetection(
            frame_idx=frame_idx,
            source_track_id=source_track_id,
            bbox=(xy[0], 0.0, xy[0] + 1.0, 2.0),
            world_xy=xy,
            conf=0.9,
            embedding=embedding,
            raw_detection_uid=uid,
        )

    positive_accept = _vector_at_cosine_distance(0.30)
    negative_accept = (positive_accept[0], -positive_accept[1])
    seed_detection = measured(0, 77, (0.1, 0.1), (1.0, 0.0), "raw:0:0")
    early_detection = measured(
        1,
        50,
        (-0.01, -0.01),
        positive_accept,
        "raw:1:0",
    )
    future_detection = measured(
        2,
        50,
        (0.1, 0.01),
        negative_accept,
        "raw:2:0",
    )
    seed = TrackFragment("reverse-seed", 77, (seed_detection,))
    early_vetoed = TrackFragment("reverse-early-vetoed", 50, (early_detection,))
    future_counterpart = TrackFragment(
        "reverse-future-counterpart", 50, (future_detection,)
    )
    slot = SelectionSlot(
        1,
        "far",
        "right",
        (1.0, 0.0),
        (0,),
        (seed.fragment_id,),
        1.0,
    )
    stitch_decisions, stitch_vetoes = player_selection_module._fragment_stitch_audit(
        (seed, early_vetoed, future_counterpart),
        real_by_frame={
            0: (seed_detection,),
            1: (early_detection,),
            2: (future_detection,),
        },
        config=PlayerSelectionConfig(),
    )
    generated_veto = next(
        decision
        for decision in stitch_decisions
        if decision.get("left_fragment_id") == early_vetoed.fragment_id
        and decision.get("right_fragment_id") == future_counterpart.fragment_id
    )
    assert generated_veto["action"] == "stitch_refused"
    assert generated_veto["open_set"] == OpenSetDecision.REJECT.value
    assert generated_veto["net_crossing"] is True

    players, unbound, decisions, _tracks, _counts = (
        player_selection_module._select_slot_players(
            (slot,),
            pool_fragments=(seed, early_vetoed, future_counterpart),
            real_pool=(seed_detection, early_detection, future_detection),
            stitch_vetoes=stitch_vetoes,
            fps=30.0,
            config=PlayerSelectionConfig(),
        )
    )

    assert [frame["frame_idx"] for frame in players[0]["frames"]] == [0, 2]
    assert not any(
        decision.get("action") == "recover_real"
        and decision.get("raw_detection_uid") == early_detection.raw_detection_uid
        for decision in decisions
    )
    assert len(unbound) == 1
    assert unbound[0]["raw_detection_uids"] == [early_detection.raw_detection_uid]
    early_decision = next(
        decision
        for decision in decisions
        if decision.get("fragment_id") == early_vetoed.fragment_id
    )
    assert early_decision["action"] == "leave_unbound"
    assert early_decision["stitch_vetoed"] is False
    assert any(
        decision.get("fragment_id") == future_counterpart.fragment_id
        and decision.get("action") == "rebind"
        for decision in decisions
    )


def test_layer_c_never_recovers_both_unbound_endpoints_of_generated_veto() -> None:
    def measured(
        frame_idx: int,
        source_track_id: int,
        x: float,
        embedding: tuple[float, float],
        uid: str,
    ) -> SelectionDetection:
        return SelectionDetection(
            frame_idx=frame_idx,
            source_track_id=source_track_id,
            bbox=(x, 0.0, x + 1.0, 2.0),
            world_xy=(x, -3.0),
            conf=0.9,
            embedding=embedding,
            raw_detection_uid=uid,
        )

    positive_accept = _vector_at_cosine_distance(0.30)
    negative_accept = (positive_accept[0], -positive_accept[1])
    seed_detection = measured(0, 77, 0.1, (1.0, 0.0), "raw:0:0")
    left_detection = measured(1, 50, -0.1, positive_accept, "raw:1:0")
    right_detection = measured(13, 50, -2.65, negative_accept, "raw:13:0")
    future_detection = measured(25, 77, 0.1, (1.0, 0.0), "raw:25:0")
    seed = TrackFragment("pair-seed", 77, (seed_detection,))
    left = TrackFragment("pair-left-unbound", 50, (left_detection,))
    right = TrackFragment("pair-right-unbound", 50, (right_detection,))
    future = TrackFragment("pair-future", 77, (future_detection,))
    slot = SelectionSlot(
        1,
        "near",
        "right",
        (1.0, 0.0),
        (0,),
        (seed.fragment_id,),
        1.0,
    )
    stitch_decisions, stitch_vetoes = player_selection_module._fragment_stitch_audit(
        (seed, left, right, future),
        real_by_frame={
            0: (seed_detection,),
            1: (left_detection,),
            13: (right_detection,),
            25: (future_detection,),
        },
        config=PlayerSelectionConfig(),
    )
    generated_veto = next(
        decision
        for decision in stitch_decisions
        if decision.get("left_fragment_id") == left.fragment_id
        and decision.get("right_fragment_id") == right.fragment_id
    )
    assert generated_veto["action"] == "stitch_refused"
    assert generated_veto["open_set"] == OpenSetDecision.REJECT.value
    assert generated_veto["displacement_m"] > 2.5

    players, unbound, decisions, _tracks, _counts = (
        player_selection_module._select_slot_players(
            (slot,),
            pool_fragments=(seed, left, right, future),
            real_pool=(
                seed_detection,
                left_detection,
                right_detection,
                future_detection,
            ),
            stitch_vetoes=stitch_vetoes,
            fps=30.0,
            config=PlayerSelectionConfig(),
        )
    )

    assert [frame["frame_idx"] for frame in players[0]["frames"]] == [0, 25]
    assert not any(decision.get("action") == "recover_real" for decision in decisions)
    assert {uid for observation in unbound for uid in observation["raw_detection_uids"]} == {
        left_detection.raw_detection_uid,
        right_detection.raw_detection_uid,
    }
    assert any(decision.get("action") == "micro_fill_refused" for decision in decisions)


def test_registered_owner_wins_same_frame_binding_without_hard_court_gate() -> None:
    records, positions, embeddings = _four_slot_records()
    records.extend(
        [
            {
                "frame": 40,
                "source": 1,
                "xy": positions[2],
                "embedding": embeddings[1],
            },
            {
                "frame": 40,
                "source": 50,
                "xy": (-6.0, -3.0),
                "embedding": embeddings[1],
            },
        ]
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records)

    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )

    near_left = next(
        player
        for player in selected["players"]
        if (player["side"], player["role"]) == ("near", "left")
    )
    assert next(
        frame for frame in near_left["frames"] if frame["frame_idx"] == 40
    )["world_xy"] == pytest.approx(list(positions[2]))
    owner_decision = next(
        decision
        for decision in report["decisions"]
        if str(decision.get("fragment_id", "")).startswith("pool-1-2-")
    )
    assert owner_decision["action"] == "rebind"
    assert owner_decision["registered_owner_continuity"] is True
    assert owner_decision["role_consistent"] is False
    spectator_decision = next(
        decision
        for decision in report["decisions"]
        if str(decision.get("fragment_id", "")).startswith("pool-50-")
    )
    assert spectator_decision["action"] == "leave_unbound"


def test_temporal_binding_compares_only_nearest_predecessor_and_successor() -> None:
    stale_bad_predecessor = _fragment(
        "stale-bad-predecessor", 1, 0, 0, (100.0, -3.0), (1.0, 0.0)
    )
    nearest_predecessor = _fragment(
        "nearest-predecessor", 1, 90, 90, (0.0, -3.0), (1.0, 0.0)
    )
    candidate = _fragment(
        "candidate", 1, 100, 100, (0.0, -3.0), (1.0, 0.0)
    )
    nearest_successor = _fragment(
        "nearest-successor", 1, 110, 110, (0.0, -3.0), (1.0, 0.0)
    )
    stale_bad_successor = _fragment(
        "stale-bad-successor", 1, 200, 200, (100.0, -3.0), (1.0, 0.0)
    )

    assert player_selection_module._fragment_inside_slot_motion_envelope(
        candidate,
        (
            stale_bad_predecessor,
            nearest_predecessor,
            nearest_successor,
            stale_bad_successor,
        ),
        fps=30.0,
        max_speed_m_s=7.0,
    )


def test_layer_c_prefers_appearance_accepted_owner_hint_over_generic_accept() -> None:
    slot = SelectionSlot(1, "near", "left", (1.0, 0.0), (0,), ("seed",), 1.0)
    last = _detection(0, 10, (-0.2, -3.0), (1.0, 0.0))
    owner_hint = _detection(
        1,
        10,
        (-0.2, -3.0),
        _vector_at_cosine_distance(0.30),
    )
    generic = _detection(
        1,
        50,
        (-0.2, -3.0),
        _vector_at_cosine_distance(0.10),
    )

    recovered = recover_identity_conditioned_pool(
        slot,
        last_detection=last,
        pool=(generic, owner_hint),
        fps=30.0,
        preferred_source_track_ids=(10,),
    )

    assert recovered == (owner_hint,)


def test_layer_c_recovery_runs_end_to_end_and_consumes_uid_once() -> None:
    records, positions, embeddings = _four_slot_records(end_frame=30)
    records.append(
        {
            "frame": 34,
            "source": 1,
            "xy": positions[1],
            "embedding": embeddings[1],
        }
    )
    records.extend(
        [
            {
                "frame": 31,
                "source": 50,
                "xy": positions[2],
                "embedding": embeddings[1],
            },
            {
                "frame": 32,
                "source": 50,
                "xy": positions[1],
                "embedding": embeddings[1],
            },
            {
                "frame": 33,
                "source": 50,
                "xy": positions[2],
                "embedding": embeddings[1],
            },
        ]
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    recovered = [
        decision
        for decision in report["decisions"]
        if decision["action"] == "recover_real"
    ]
    assert [(row["frame_idx"], row["interpolated"]) for row in recovered] == [
        (32, False)
    ]
    slot_one = next(
        player
        for player in selected["players"]
        if (player["side"], player["role"]) == ("near", "left")
    )
    frame_32 = next(frame for frame in slot_one["frames"] if frame["frame_idx"] == 32)
    assert frame_32["interpolated"] is False
    assert frame_32["world_xy"] == pytest.approx(list(positions[1]))
    all_report_uids = [
        uid for track in report["tracks"] for uid in track["raw_detection_uids"]
    ]
    assert all_report_uids.count(recovered[0]["raw_detection_uid"]) == 1
    assert report["output_counts"]["recovered_real_detections"] == 1


def test_ambiguous_real_gap_observation_is_preserved_and_vetoes_synthesis() -> None:
    records, positions, embeddings = _four_slot_records(end_frame=30)
    records.append(
        {
            "frame": 34,
            "source": 1,
            "xy": positions[1],
            "embedding": embeddings[1],
        }
    )
    records.append(
        {
            "frame": 32,
            "source": 50,
            "xy": positions[1],
            "embedding": _vector_at_cosine_distance(0.36),
        }
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    slot_one = next(
        player
        for player in selected["players"]
        if (player["side"], player["role"]) == ("near", "left")
    )
    assert not {31, 32, 33} & {frame["frame_idx"] for frame in slot_one["frames"]}
    unbound = selected["unbound_observations"]
    assert len(unbound) == 1
    assert [frame["frame_idx"] for frame in unbound[0]["frames"]] == [32]
    assert unbound[0]["frames"][0]["interpolated"] is False
    assert any(
        decision["action"] == "micro_fill_refused"
        and "ambiguous_real_observation_in_gap" in decision["reasons"]
        for decision in report["decisions"]
    )


def test_real_observation_bound_to_other_slot_still_vetoes_synthesis() -> None:
    cosine = 0.64
    embeddings = {
        1: (1.0, 0.0, 0.0, 0.0, 0.0),
        2: (cosine, math.sqrt(1.0 - cosine * cosine), 0.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0, 0.0),
    }
    positions = {
        1: (-0.2, -3.0),
        2: (0.2, -3.0),
        3: (-1.5, 3.0),
        4: (1.5, 3.0),
    }
    records = [
        {
            "frame": frame_idx,
            "source": source,
            "xy": positions[source],
            "embedding": embeddings[source],
            "bbox": (
                positions[source][0] - 0.05,
                positions[source][1] - 1.0,
                positions[source][0] + 0.05,
                positions[source][1],
            ),
        }
        for frame_idx in range(31)
        for source in sorted(positions)
    ]
    records.extend(
        [
            {
                "frame": 32,
                "source": 2,
                "xy": positions[2],
                "embedding": embeddings[2],
                "bbox": (0.15, -4.0, 0.25, -3.0),
            },
            {
                "frame": 34,
                "source": 1,
                "xy": positions[1],
                "embedding": embeddings[1],
                "bbox": (-0.25, -4.0, -0.15, -3.0),
            },
        ]
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records, feature_dim=5)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    near_left = next(
        player
        for player in selected["players"]
        if (player["side"], player["role"]) == ("near", "left")
    )
    near_right = next(
        player
        for player in selected["players"]
        if (player["side"], player["role"]) == ("near", "right")
    )
    assert not {31, 32, 33} & {frame["frame_idx"] for frame in near_left["frames"]}
    bound_frame_32 = next(
        frame for frame in near_right["frames"] if frame["frame_idx"] == 32
    )
    assert bound_frame_32["interpolated"] is False
    refused = next(
        decision
        for decision in report["decisions"]
        if decision["action"] == "micro_fill_refused"
        and decision.get("slot_id") == near_left["id"]
        and decision.get("left_frame") == 30
        and decision.get("right_frame") == 34
    )
    assert refused["reasons"] == ["ambiguous_real_observation_in_gap"]
    assert refused["ambiguous_raw_detection_uids"] == ["raw:32:0"]


def test_end_to_end_drop_requires_appearance_plus_persistence_or_motion() -> None:
    positions = {
        1: (-1.5, -3.0),
        2: (1.5, -3.0),
        3: (-1.5, 3.0),
        4: (1.5, 3.0),
    }
    embeddings = {
        1: (1.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0, 0.0),
    }
    enrollment_records = [
        {
            "frame": frame_idx,
            "source": source,
            "xy": positions[source],
            "embedding": embeddings[source],
        }
        for frame_idx in range(30)
        for source in sorted(positions)
    ]
    unknown_embedding = (0.0, 0.0, 0.0, 0.0, 1.0)

    dropped_raw, dropped_embeddings = _raw_selection_inputs(
        enrollment_records
        + [
            {
                "frame": 31,
                "source": 50,
                "xy": (6.0, -3.0),
                "embedding": unknown_embedding,
            }
        ],
        feature_dim=5,
    )
    dropped_selected, dropped_report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=dropped_raw,
        embedding_payload=dropped_embeddings,
        calibration=_identity_calibration(),
        enabled=True,
    )
    drop_decision = next(
        decision
        for decision in dropped_report["decisions"]
        if str(decision.get("fragment_id", "")).startswith("pool-50-")
    )
    assert drop_decision["action"] == "drop"
    assert drop_decision["evidence_classes"] == ["appearance", "geometry"]
    assert drop_decision["reasons"] == [
        "appearance_all_detections_all_slots_at_or_above_identity_reject_distance_0_42",
        "geometry_all_slots_temporal_overlap_or_speed_above_7_0_m_s",
    ]
    assert not {
        "identity_accept",
        "identity_reject",
        "fusion_at_or_above_0_5",
    } & set(drop_decision["reasons"])
    assert dropped_report["output_counts"]["dropped_real_detections"] == 1
    assert not any(
        frame["frame_idx"] == 31
        for player in dropped_selected["players"]
        for frame in player["frames"]
    )

    unbound_raw, unbound_embeddings = _raw_selection_inputs(
        enrollment_records
        + [
            {
                "frame": 100,
                "source": 50,
                "xy": (0.0, -3.0),
                "embedding": unknown_embedding,
            }
        ],
        feature_dim=5,
    )
    unbound_selected, unbound_report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=unbound_raw,
        embedding_payload=unbound_embeddings,
        calibration=_identity_calibration(),
        enabled=True,
    )
    unbound_decision = next(
        decision
        for decision in unbound_report["decisions"]
        if str(decision.get("fragment_id", "")).startswith("pool-50-")
    )
    assert unbound_decision["action"] == "leave_unbound"
    assert unbound_decision["evidence_classes"] == ["appearance"]
    assert unbound_report["output_counts"]["dropped_real_detections"] == 0
    assert any(
        [frame["frame_idx"] for frame in observation["frames"]] == [100]
        for observation in unbound_selected["unbound_observations"]
    )
    schema = json.loads(
        (ROOT / "docs/racketsport/player_selection_report_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["$defs"]["dropTriggerReason"]["enum"] == list(
        DROP_TRIGGER_REASON_VOCABULARY
    )
    assert_matches_json_schema(dropped_report, schema)
    dishonest_report = json.loads(json.dumps(dropped_report))
    dishonest_drop = next(
        decision
        for decision in dishonest_report["decisions"]
        if decision["action"] == "drop"
    )
    dishonest_drop["reasons"] = ["identity_reject", "fusion_at_or_above_0_5"]
    with pytest.raises(AssertionError):
        assert_matches_json_schema(dishonest_report, schema)


def test_appearance_reject_low_court_with_adequate_persistence_motion_stays_unbound() -> (
    None
):
    positions = {
        1: (-1.5, -3.0),
        2: (1.5, -3.0),
        3: (-1.5, 3.0),
        4: (1.5, 3.0),
    }
    embeddings = {
        1: (1.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0, 0.0),
    }
    records = [
        {
            "frame": frame_idx,
            "source": source,
            "xy": positions[source],
            "embedding": embeddings[source],
        }
        for frame_idx in range(30)
        for source in sorted(positions)
    ]
    records.append(
        {
            "frame": 100,
            "source": 50,
            "xy": (6.0, -3.0),
            "embedding": (0.0, 0.0, 0.0, 0.0, 1.0),
        }
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records, feature_dim=5)

    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )

    decision = next(
        row
        for row in report["decisions"]
        if str(row.get("fragment_id", "")).startswith("pool-50-")
    )
    assert decision["open_set"] == OpenSetDecision.REJECT.value
    assert decision["court_presence"] < 0.5
    assert decision["persistence"] >= PlayerSelectionConfig().selection_score_min
    assert decision["motion_envelope_consistent"] is True
    assert decision["action"] == "leave_unbound"
    assert decision["evidence_classes"] == ["appearance"]
    assert report["output_counts"]["dropped_real_detections"] == 0
    assert any(
        [frame["frame_idx"] for frame in observation["frames"]] == [100]
        for observation in selected["unbound_observations"]
    )


def test_identity_accept_off_court_motion_inconsistent_fragment_remains_unbound() -> (
    None
):
    records, _positions, embeddings = _four_slot_records()
    records.append(
        {
            "frame": 31,
            "source": 50,
            "xy": (6.0, -3.0),
            "embedding": embeddings[1],
        }
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    decision = next(
        row
        for row in report["decisions"]
        if str(row.get("fragment_id", "")).startswith("pool-50-")
    )
    assert decision["open_set"] == "accept"
    assert decision["court_presence"] < 0.5
    assert decision["motion_envelope_consistent"] is False
    assert decision["action"] == "leave_unbound"
    assert decision["evidence_classes"] == ["geometry"]
    assert report["output_counts"]["dropped_real_detections"] == 0
    assert any(
        [frame["frame_idx"] for frame in observation["frames"]] == [31]
        for observation in selected["unbound_observations"]
    )


def test_fragment_drop_requires_reject_for_every_real_detection() -> None:
    records, _positions, _embeddings = _four_slot_records()
    basis = {
        1: (1.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0, 0.0),
    }
    for record in records:
        record["embedding"] = basis[int(record["source"])]
    records.extend(
        [
            {
                "frame": 31,
                "source": 50,
                "xy": (6.0, -3.0),
                "embedding": (0.0, 0.0, 0.0, 0.0, 1.0),
            },
            {
                "frame": 32,
                "source": 50,
                "xy": (6.0, -3.0),
                "embedding": None,
            },
        ]
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records, feature_dim=5)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    decision = next(
        row
        for row in report["decisions"]
        if str(row.get("fragment_id", "")).startswith("pool-50-")
    )
    assert decision["open_set"] == "reject"
    assert decision["action"] == "leave_unbound"
    assert decision["evidence_classes"] == ["geometry"]
    assert report["output_counts"]["dropped_real_detections"] == 0
    assert any(
        [frame["frame_idx"] for frame in observation["frames"]] == [31, 32]
        for observation in selected["unbound_observations"]
    )


def test_fragment_centroid_reject_cannot_destroy_one_accepted_detection() -> None:
    records, _positions, _embeddings = _four_slot_records()
    basis = {
        1: (1.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0, 0.0),
    }
    for record in records:
        record["embedding"] = basis[int(record["source"])]
    accepted_embedding = (0.66, 0.0, 0.0, 0.0, math.sqrt(1.0 - 0.66**2))
    rejected_embedding = (0.50, 0.0, 0.0, 0.0, math.sqrt(1.0 - 0.50**2))
    records.extend(
        {
            "frame": frame_idx,
            "source": 51,
            "xy": (6.0, -3.0),
            "embedding": accepted_embedding if frame_idx == 40 else rejected_embedding,
        }
        for frame_idx in range(40, 50)
    )
    raw_pool, embedding_payload = _raw_selection_inputs(records, feature_dim=5)
    selected, report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embedding_payload,
        calibration=_identity_calibration(),
        enabled=True,
    )
    decision = next(
        row
        for row in report["decisions"]
        if str(row.get("fragment_id", "")).startswith("pool-51-")
    )
    assert decision["open_set"] == "reject"
    assert decision["embedding_distance"] > 0.42
    assert decision["action"] == "leave_unbound"
    assert decision["evidence_classes"] == ["geometry"]
    assert report["output_counts"]["dropped_real_detections"] == 0
    assert any(
        [frame["frame_idx"] for frame in observation["frames"]]
        == list(range(40, 50))
        for observation in selected["unbound_observations"]
    )


def test_reports_match_full_schema_for_every_status_family() -> None:
    schema = json.loads(
        (ROOT / "docs/racketsport/player_selection_report_schema.json").read_text(
            encoding="utf-8"
        )
    )
    disabled_payload = _empty_tracks()
    disabled_payload["players"] = [
        {"id": 1, "side": "near", "role": "left", "frames": []}
    ]
    _, disabled_report = select_players_payload(disabled_payload, enabled=False)
    assert disabled_report["input_counts"]["players"] == 1
    assert disabled_report["output_counts"]["players"] == 1
    assert_matches_json_schema(disabled_report, schema)
    raw_pool, embeddings = _raw_selection_inputs(
        [{"frame": 0, "source": 1, "xy": (0.0, -2.0), "embedding": (1.0, 0.0)}]
    )
    _, partial_report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embeddings,
        calibration=_identity_calibration(),
        enabled=True,
    )
    assert_matches_json_schema(partial_report, schema)
    records, _positions, _embeddings = _four_slot_records()
    raw_pool, embeddings = _raw_selection_inputs(records)
    _, complete_report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embeddings,
        calibration=_identity_calibration(),
        enabled=True,
    )
    assert complete_report["status"] == "preview_selection_complete"
    assert_matches_json_schema(complete_report, schema)
    records.append(
        {
            "frame": 40,
            "source": 50,
            "xy": (0.0, -3.0),
            "embedding": None,
        }
    )
    raw_pool, embeddings = _raw_selection_inputs(records)
    _, abstention_report = select_players_payload(
        _empty_tracks(),
        raw_pool_payload=raw_pool,
        embedding_payload=embeddings,
        calibration=_identity_calibration(),
        enabled=True,
    )
    assert abstention_report["status"] == "preview_selection_partial_with_abstentions"
    assert_matches_json_schema(abstention_report, schema)


@pytest.mark.parametrize(
    ("report_family", "path", "replacement"),
    [
        ("disabled", ("source_only",), True),
        ("disabled", ("selection_mode",), "enrolled_four_slot"),
        (
            "disabled",
            ("source_attestations", "status"),
            "explicitly_attested",
        ),
        ("disabled", ("enrollment", "status"), "enrolled"),
        ("disabled", ("output_counts", "unbound_real_detections"), 9),
        ("enabled", ("source_only",), False),
        ("enabled", ("selection_mode",), "disabled"),
        (
            "enabled",
            ("source_attestations", "status"),
            "not_applicable_disabled",
        ),
        ("enabled", ("source_attestations", "uses_cvat_labels"), True),
        ("enabled", ("source_attestations", "promote_trk"), True),
        ("enabled", ("enrollment", "slot_count"), 0),
        ("enabled", ("output_counts", "unbound_real_detections"), 1),
        ("no_enrollment", ("enrollment", "status"), "enrolled"),
        ("no_enrollment", ("output_counts", "interpolated_frames"), 1),
        ("abstention", ("output_counts", "unbound_real_detections"), 0),
    ],
)
def test_report_schema_rejects_status_attestation_mismatches(
    report_family: str,
    path: tuple[str, ...],
    replacement: object,
) -> None:
    schema = json.loads(
        (ROOT / "docs/racketsport/player_selection_report_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if report_family == "disabled":
        _, report = select_players_payload(_empty_tracks(), enabled=False)
    elif report_family == "no_enrollment":
        raw_pool, embeddings = _raw_selection_inputs(
            [
                {
                    "frame": 0,
                    "source": 1,
                    "xy": (0.0, -2.0),
                    "embedding": (1.0, 0.0),
                }
            ]
        )
        _, report = select_players_payload(
            _empty_tracks(),
            raw_pool_payload=raw_pool,
            embedding_payload=embeddings,
            calibration=_identity_calibration(),
            enabled=True,
        )
    else:
        records, _positions, _embeddings = _four_slot_records()
        if report_family == "abstention":
            records.append(
                {
                    "frame": 40,
                    "source": 50,
                    "xy": (0.0, -3.0),
                    "embedding": None,
                }
            )
        raw_pool, embeddings = _raw_selection_inputs(records)
        _, report = select_players_payload(
            _empty_tracks(),
            raw_pool_payload=raw_pool,
            embedding_payload=embeddings,
            calibration=_identity_calibration(),
            enabled=True,
        )
    mutated = json.loads(json.dumps(report))
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(AssertionError):
        assert_matches_json_schema(mutated, schema)


def test_selection_off_cli_golden_preserves_crlf_and_source_sha(tmp_path: Path) -> None:
    source = tmp_path / "tracks.golden.json"
    output = tmp_path / "tracks.output.json"
    report = tmp_path / "selection.report.json"
    golden = (
        b'{\r\n  "rally_spans" : [ ], "players" : [ ],\r\n'
        b'  "fps": 3.0e1, "schema_version" : 1\r\n}\r\n \t\r\n'
    )
    source.write_bytes(golden)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/racketsport/select_players_from_pool.py"),
            "--tracks",
            str(source),
            "--out-tracks",
            str(output),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert source.read_bytes() == golden
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    assert output.read_bytes() == golden
    assert hashlib.sha256(output.read_bytes()).hexdigest() == source_sha
    assert not (tmp_path / "tracks.output.scoring_projection.json").exists()


def test_enabled_cli_validates_tracks_and_publishes_interpolated_true(
    tmp_path: Path,
) -> None:
    records, _positions, _embeddings = _four_slot_records(end_frame=44)
    records = [
        record
        for record in records
        if not (int(record["source"]) == 1 and int(record["frame"]) in {40, 41, 42})
    ]
    records.append(
        {
            "frame": 50,
            "source": 50,
            "xy": (0.0, -2.5),
            "embedding": None,
        }
    )
    raw_pool, embeddings = _raw_selection_inputs(records)
    tracks_path = tmp_path / "tracks.json"
    raw_path = tmp_path / "raw.json"
    embeddings_path = tmp_path / "embeddings.json"
    calibration_path = tmp_path / "calibration.json"
    output_path = tmp_path / "selected.json"
    report_path = tmp_path / "report.json"
    coercible_tracks = _empty_tracks()
    coercible_tracks["fps"] = "30.0"
    coercible_tracks["rally_spans"] = [{"t0": "0.0", "t1": "1.5"}]
    tracks_path.write_text(json.dumps(coercible_tracks), encoding="utf-8")
    raw_path.write_text(json.dumps(raw_pool), encoding="utf-8")
    embeddings_path.write_text(json.dumps(embeddings), encoding="utf-8")
    calibration_path.write_text(
        json.dumps(_identity_calibration().model_dump(mode="json")),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/racketsport/select_players_from_pool.py"),
            "--tracks",
            str(tracks_path),
            "--raw-pool",
            str(raw_path),
            "--embeddings",
            str(embeddings_path),
            "--calibration",
            str(calibration_path),
            "--out-tracks",
            str(output_path),
            "--report",
            str(report_path),
            "--enable-selection",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["players"]) == 4
    assert len(payload["unbound_observations"]) == 1
    assert payload["unbound_observations"][0]["selection_state"] == (
        "unbound_abstention"
    )
    assert payload["unbound_observations"][0]["abstention_reasons"]
    assert payload["unbound_observations"][0]["raw_detection_uids"] == ["raw:50:0"]
    canonical_tracks = dict(payload)
    del canonical_tracks["unbound_observations"]
    validated = Tracks.model_validate(canonical_tracks)
    assert isinstance(payload["fps"], float)
    assert isinstance(payload["rally_spans"][0]["t0"], float)
    assert isinstance(payload["rally_spans"][0]["t1"], float)
    interpolated = [
        frame
        for player in payload["players"]
        for frame in player["frames"]
        if frame.get("interpolated", False)
    ]
    assert [frame["frame_idx"] for frame in interpolated] == [40, 41, 42]
    assert (
        sum(
            frame.interpolated
            for player in validated.players
            for frame in player.frames
        )
        == 3
    )
    projection_path = tmp_path / "selected.scoring_projection.json"
    projection_sha_path = tmp_path / "selected.scoring_projection.json.sha256"
    projection_bytes = projection_path.read_bytes()
    projection = json.loads(projection_bytes)
    assert "unbound_observations" not in projection
    expected_projection = json.loads(json.dumps(payload))
    del expected_projection["unbound_observations"]
    for player in expected_projection["players"]:
        player["frames"] = [
            {key: value for key, value in frame.items() if key != "interpolated"}
            for frame in player["frames"]
            if frame.get("interpolated") is not True
        ]
    assert projection == expected_projection
    assert not any(
        "interpolated" in frame
        for player in projection["players"]
        for frame in player["frames"]
    )
    Tracks.model_validate(projection)
    projection_sha = hashlib.sha256(projection_bytes).hexdigest()
    assert projection_sha_path.read_text(encoding="utf-8") == f"{projection_sha}\n"
    assert f"scoring_projection_sha256={projection_sha}" in completed.stdout


def test_cli_rejects_scoring_projection_collision_before_writing(
    tmp_path: Path,
) -> None:
    tracks = tmp_path / "tracks.json"
    output = tmp_path / "selected.json"
    report = tmp_path / "report.json"
    tracks.write_text(
        '{"schema_version":1,"fps":30,"players":[],"rally_spans":[]}\n',
        encoding="utf-8",
    )
    output.write_bytes(b"existing output\n")
    before_output = output.read_bytes()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/racketsport/select_players_from_pool.py"),
            "--tracks",
            str(tracks),
            "--out-tracks",
            str(output),
            "--report",
            str(report),
            "--scoring-projection",
            str(output),
            "--enable-selection",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "pairwise-distinct" in completed.stderr
    assert output.read_bytes() == before_output
    assert not report.exists()


@pytest.mark.parametrize("collision_target", ["tracks", "out_tracks"])
def test_cli_rejects_report_path_collisions_without_modifying_input_or_output(
    tmp_path: Path,
    collision_target: str,
) -> None:
    tracks = tmp_path / "tracks.json"
    output = tmp_path / "out.json"
    tracks.write_bytes(b'{"schema_version":1,"fps":30,"players":[],"rally_spans":[]}\n')
    output.write_bytes(b"existing output\n")
    before_tracks = tracks.read_bytes()
    before_output = output.read_bytes()
    report = tracks if collision_target == "tracks" else output
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/racketsport/select_players_from_pool.py"),
            "--tracks",
            str(tracks),
            "--out-tracks",
            str(output),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "pairwise-distinct" in completed.stderr
    assert tracks.read_bytes() == before_tracks
    assert output.read_bytes() == before_output


def test_cli_rejects_hardlink_aliases(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.json"
    alias = tmp_path / "tracks.alias.json"
    output = tmp_path / "out.json"
    tracks.write_text(
        '{"schema_version":1,"fps":30,"players":[],"rally_spans":[]}\n',
        encoding="utf-8",
    )
    os.link(tracks, alias)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/racketsport/select_players_from_pool.py"),
            "--tracks",
            str(tracks),
            "--out-tracks",
            str(output),
            "--report",
            str(alias),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "pairwise-distinct" in completed.stderr
    assert not output.exists()


def test_cli_rejects_case_only_destination_aliases_before_writing(
    tmp_path: Path,
) -> None:
    tracks = tmp_path / "tracks.json"
    output = tmp_path / "Output.json"
    report = tmp_path / "output.json"
    tracks.write_text(
        '{"schema_version":1,"fps":30,"players":[],"rally_spans":[]}\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/racketsport/select_players_from_pool.py"),
            "--tracks",
            str(tracks),
            "--out-tracks",
            str(output),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "pairwise-distinct" in completed.stderr
    assert not output.exists()
    assert not report.exists()


def test_enabled_cli_rejects_nonfinite_fps_without_publishing(
    tmp_path: Path,
) -> None:
    tracks = tmp_path / "tracks.json"
    raw_pool = tmp_path / "raw.json"
    embeddings = tmp_path / "embeddings.json"
    calibration = tmp_path / "calibration.json"
    output = tmp_path / "selected.json"
    report = tmp_path / "report.json"
    tracks.write_text(
        '{"schema_version":1,"fps":1e999,"players":[],"rally_spans":[]}\n',
        encoding="utf-8",
    )
    raw_pool.write_text(
        '{"schema_version":1,"fps":30,"frames":[]}\n',
        encoding="utf-8",
    )
    embeddings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_only": True,
                "uses_cvat_labels": False,
                "promote_trk": False,
                "feature_dim": 2,
                "l2_normalized": True,
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    calibration.write_text(
        json.dumps(_identity_calibration().model_dump(mode="json")),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/racketsport/select_players_from_pool.py"),
            "--tracks",
            str(tracks),
            "--raw-pool",
            str(raw_pool),
            "--embeddings",
            str(embeddings),
            "--calibration",
            str(calibration),
            "--out-tracks",
            str(output),
            "--report",
            str(report),
            "--enable-selection",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "finite and positive" in completed.stderr
    assert not output.exists()
    assert not report.exists()


@pytest.mark.parametrize("failure_destination", ["report", "tracks"])
def test_output_pair_failure_rolls_back_both_destinations_and_cleans_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_destination: str,
) -> None:
    tracks = tmp_path / "tracks.json"
    report = tmp_path / "report.json"
    tracks.write_bytes(b"old tracks\n")
    report.write_bytes(b"old report\n")
    staged_tracks = selection_cli._stage_text(tracks, "new tracks\n")
    staged_report = selection_cli._stage_text(report, "new report\n")
    real_replace = os.replace

    def fail_selected_replace(source: object, destination: object) -> None:
        target = Path(destination)  # type: ignore[arg-type]
        should_fail = (
            failure_destination == "report"
            and target == report
            and Path(source) == staged_report  # type: ignore[arg-type]
        ) or (
            failure_destination == "tracks"
            and target == tracks
            and Path(source) == staged_tracks  # type: ignore[arg-type]
        )
        if should_fail:
            raise OSError(f"simulated {failure_destination} replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(selection_cli.os, "replace", fail_selected_replace)
    with pytest.raises(OSError, match=r"simulated (report|tracks) replace failure"):
        selection_cli._publish_output_pair(
            staged_tracks=staged_tracks,
            tracks_destination=tracks,
            staged_report=staged_report,
            report_destination=report,
        )
    assert tracks.read_bytes() == b"old tracks\n"
    assert report.read_bytes() == b"old report\n"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    "failure_destination",
    ["projection", "projection_sha256", "report", "tracks"],
)
def test_enabled_output_bundle_failure_restores_every_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_destination: str,
) -> None:
    destinations = {
        "projection": tmp_path / "projection.json",
        "projection_sha256": tmp_path / "projection.json.sha256",
        "report": tmp_path / "report.json",
        "tracks": tmp_path / "tracks.json",
    }
    for name, destination in destinations.items():
        destination.write_text(f"old {name}\n", encoding="utf-8")
    real_replace = os.replace
    failed = False

    def fail_selected_replace_once(source: object, destination: object) -> None:
        nonlocal failed
        target = Path(destination)  # type: ignore[arg-type]
        if not failed and target == destinations[failure_destination]:
            failed = True
            raise OSError(f"simulated {failure_destination} replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(selection_cli.os, "replace", fail_selected_replace_once)
    with pytest.raises(OSError, match=f"simulated {failure_destination} replace failure"):
        selection_cli._stage_and_publish_enabled_outputs(
            tracks_text="new tracks\n",
            tracks_destination=destinations["tracks"],
            report_text="new report\n",
            report_destination=destinations["report"],
            projection_text="new projection\n",
            projection_destination=destinations["projection"],
            projection_sha256_text="new projection sha256\n",
            projection_sha256_destination=destinations["projection_sha256"],
        )
    for name, destination in destinations.items():
        assert destination.read_text(encoding="utf-8") == f"old {name}\n"
    assert list(tmp_path.glob(".*.tmp")) == []
