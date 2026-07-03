from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.racketsport.apply_confidence_gate import main as apply_confidence_gate_main
from scripts.racketsport.calibrate_confidence_bands import main as calibrate_confidence_bands_main
from threed.racketsport.confidence_gate import (
    ConfidenceGateConfig,
    apply_confidence_gate_to_world,
    apply_correction_sanity_gate,
    apply_hysteresis,
    band_from_sigma,
    classify_low_confidence_spans,
)


def test_gap_classification_distinguishes_short_long_and_no_anchor_spans() -> None:
    confidences = [0.1, 0.2, 0.95, 0.1, 0.2, 0.9, 0.1, 0.2, 0.3, 0.2, 0.95]

    spans = classify_low_confidence_spans(confidences, threshold=0.5, short_gap_max_frames=2)

    assert [(span.kind, span.start, span.end, span.horizon_frames) for span in spans] == [
        ("NO_ANCHOR", 0, 1, 2),
        ("SHORT_GAP", 3, 4, 2),
        ("LONG_GAP", 6, 9, 4),
    ]


def test_hysteresis_requires_consecutive_frames_before_switching_bands() -> None:
    raw = [
        "measured",
        "physics_predicted_warn",
        "measured",
        "physics_predicted_warn",
        "physics_predicted_warn",
        "physics_predicted_warn",
        "measured",
        "measured",
        "measured",
    ]

    smoothed = apply_hysteresis(raw, min_consecutive=3)

    assert smoothed == [
        "measured",
        "measured",
        "measured",
        "measured",
        "measured",
        "physics_predicted_warn",
        "physics_predicted_warn",
        "physics_predicted_warn",
        "measured",
    ]


def test_sigma_maps_to_calibrated_band_thresholds() -> None:
    curves = {
        "ball": {
            "horizon_buckets": {
                "1-3": {"p50_m": 0.20, "p90_m": 0.60},
            }
        }
    }

    assert band_from_sigma(0.19, entity="ball", horizon_frames=2, calibration_curves=curves) == "physics_predicted"
    assert band_from_sigma(0.40, entity="ball", horizon_frames=2, calibration_curves=curves) == "physics_predicted_warn"
    assert band_from_sigma(0.90, entity="ball", horizon_frames=2, calibration_curves=curves) == "physics_predicted_low"


def test_correction_sanity_gate_degrades_when_non_target_joint_moves_too_far() -> None:
    original = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    corrected = [[0.3, 0.0, 0.0], [1.2, 1.0, 0.0]]

    result = apply_correction_sanity_gate(
        original,
        corrected,
        target_joint_indices={0},
        base_band="physics_corrected",
        max_non_target_displacement_m=0.15,
    )

    assert result.band == "physics_corrected_warn"
    assert result.max_non_target_displacement_m == pytest.approx(0.2)


def test_apply_confidence_gate_preserves_existing_world_values_and_adds_provenance() -> None:
    world = {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court": {"sport": "pickleball", "coordinate_frame": "court_Z0"},
        "players": [
            {
                "id": 1,
                "representation": "joints",
                "frames": [
                    {
                        "t": 0.0,
                        "track_conf": 0.9,
                        "joint_count": 2,
                        "mesh_vertex_count": 0,
                        "joints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        "joint_conf": [0.95, 0.94],
                        "trust_band": {"gate_status": "corrected"},
                    }
                ],
            }
        ],
        "ball": {
            "source": "physics_filled",
            "frames": [
                {"t": 0.0, "xy": [10.0, 20.0], "conf": 0.9, "visible": True, "world_xyz": [1.0, 2.0, 0.5]},
                {"t": 1.0 / 30.0, "xy": [11.0, 20.0], "conf": 0.2, "visible": False, "world_xyz": [1.1, 2.0, 0.4]},
            ],
        },
        "paddles": [],
        "summary": {"player_count": 1, "ball_frame_count": 2, "paddle_frame_count": 0},
    }
    filled_ball = {
        "frames": [
            {"t": 0.0, "conf": 0.9, "world_xyz": [1.0, 2.0, 0.5]},
            {
                "t": 1.0 / 30.0,
                "conf": 0.2,
                "world_xyz": [1.1, 2.0, 0.4],
                "source": "physics_interpolated",
                "physics_fill": {
                    "render_only": True,
                    "not_for_detection_metrics": True,
                    "uncertainty_m": 0.32,
                    "gap_distance_frames": 1,
                },
            },
        ]
    }
    before = copy.deepcopy(world)

    gated = apply_confidence_gate_to_world(
        world,
        ball_track_physics_filled=filled_ball,
        physics_footlock=None,
        racket_pose_estimate={"summary": {"estimate_count": 0}},
        contact_windows=None,
        calibration_curves={
            "ball": {"horizon_buckets": {"1-3": {"p50_m": 0.2, "p90_m": 0.6}}},
            "player_joints": {"horizon_buckets": {"1-3": {"p50_m": 0.05, "p90_m": 0.15}}},
        },
        config=ConfidenceGateConfig(confidence_threshold=0.5, hysteresis_frames=1),
    )

    assert world == before
    assert gated["ball"]["frames"][0]["world_xyz"] == [1.0, 2.0, 0.5]
    assert gated["players"][0]["frames"][0]["joints_world"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert gated["ball"]["frames"][0]["confidence_provenance"]["band"] == "measured"
    assert gated["ball"]["frames"][0]["confidence_provenance"]["display_band"] == "measured"
    assert gated["ball"]["frames"][1]["confidence_provenance"]["predictor"] == "BallBallisticAdapter"
    assert gated["ball"]["frames"][1]["render_only"] is True
    assert gated["ball"]["frames"][1]["not_for_detection_metrics"] is True


def test_calibrate_confidence_bands_cli_writes_curves_from_existing_loo(tmp_path) -> None:
    run_dir = tmp_path / "chain"
    run_dir.mkdir()
    (run_dir / "ball_track_physics_filled.json").write_text(
        json.dumps(
            {
                "frames": [],
                "physics_fill": {
                    "validation": {
                        "leave_one_out": {
                            "error_3d_m": {"count": 40, "median": 0.1649, "p90": 0.4893, "p95": 0.5621}
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "virtual_world.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "players": [],
                "ball": {"frames": []},
                "paddles": [],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "calibration_curves.json"

    assert calibrate_confidence_bands_main(["--run-dir", str(run_dir), "--out", str(out)]) == 0

    curves = json.loads(out.read_text(encoding="utf-8"))
    assert curves["ball"]["horizon_buckets"]["1-3"]["p50_m"] == pytest.approx(0.1649)
    assert curves["ball"]["known_loo_comparison"]["median_m"] == pytest.approx(0.1649)


def test_apply_confidence_gate_cli_writes_confidence_gated_world(tmp_path) -> None:
    run_dir = tmp_path / "chain"
    run_dir.mkdir()
    world = {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court": {"sport": "pickleball", "coordinate_frame": "court_Z0"},
        "players": [],
        "ball": {"source": "physics_filled", "frames": [{"t": 0.0, "xy": [1.0, 2.0], "conf": 0.9, "visible": True, "world_xyz": [0.0, 0.0, 1.0]}]},
        "paddles": [],
        "summary": {"player_count": 0, "ball_frame_count": 1, "paddle_frame_count": 0},
    }
    (run_dir / "virtual_world.json").write_text(json.dumps(world), encoding="utf-8")
    (run_dir / "ball_track_physics_filled.json").write_text(json.dumps({"frames": [world["ball"]["frames"][0]]}), encoding="utf-8")
    out_dir = tmp_path / "out"
    curves = tmp_path / "calibration_curves.json"
    curves.write_text(json.dumps({"ball": {"horizon_buckets": {"1-3": {"p50_m": 0.2, "p90_m": 0.6}}}}), encoding="utf-8")

    assert apply_confidence_gate_main(
        ["--run-dir", str(run_dir), "--out", str(out_dir), "--calibration-curves", str(curves)]
    ) == 0

    gated = json.loads((out_dir / "confidence_gated_world.json").read_text(encoding="utf-8"))
    assert gated["ball"]["frames"][0]["world_xyz"] == [0.0, 0.0, 1.0]
    assert gated["ball"]["frames"][0]["confidence_provenance"]["band"] == "measured"


def test_real_wolverine_chain_adapter_mode_preserves_existing_world_values() -> None:
    run_dir = Path("runs/phys_chain_20260702T174041Z/wolverine_v1_chain")
    if not (run_dir / "virtual_world.json").is_file():
        pytest.skip("real Wolverine PHYS chain fixture is not present")
    world = json.loads((run_dir / "virtual_world.json").read_text(encoding="utf-8"))
    gated = apply_confidence_gate_to_world(
        world,
        ball_track_physics_filled=json.loads((run_dir / "ball_track_physics_filled.json").read_text(encoding="utf-8")),
        physics_footlock=json.loads((run_dir / "physics_footlock.json").read_text(encoding="utf-8")),
        racket_pose_estimate=json.loads((run_dir / "racket_pose_estimate.json").read_text(encoding="utf-8")),
        contact_windows=json.loads((run_dir / "contact_windows.json").read_text(encoding="utf-8")),
        calibration_curves={"ball": {"horizon_buckets": {"1-3": {"p50_m": 0.1649, "p90_m": 0.4893}}}},
        config=ConfidenceGateConfig(),
    )

    assert [frame.get("world_xyz") for frame in gated["ball"]["frames"]] == [
        frame.get("world_xyz") for frame in world["ball"]["frames"]
    ]
    for player_index, player in enumerate(world["players"]):
        for frame_index, frame in enumerate(player["frames"]):
            assert gated["players"][player_index]["frames"][frame_index].get("joints_world") == frame.get("joints_world")
    assert gated["confidence_gate"]["counts_by_entity_band"]["paddle"]["hidden_no_prediction"] == 63
