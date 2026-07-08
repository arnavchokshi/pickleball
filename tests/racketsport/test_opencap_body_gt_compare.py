from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.racketsport.opencap_body_gt_compare as opencap_cli
from threed.racketsport.external_gt_aspset510 import SHARED_CORE_JOINT_NAMES
from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES

COMMAND_PATH = "scripts/racketsport/opencap_body_gt_compare.py"


# ---------------------------------------------------------------------------
# Direct-CLI reference tests
# ---------------------------------------------------------------------------


def test_opencap_compare_cli_exposes_direct_help_reference() -> None:
    completed = subprocess.run(
        [sys.executable, COMMAND_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for flag in (
        "--opencap-trc",
        "--opencap-mot",
        "--our-joints",
        "--player-id",
        "--fps",
        "--sync-offset-seconds",
        "--gate-variant",
        "--out",
        "--self-test",
        "--self-test-fixture-dir",
    ):
        assert flag in completed.stdout


def test_opencap_compare_cli_self_test_passes_and_recovers_injected_error(tmp_path: Path) -> None:
    out_path = tmp_path / "self_test_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            COMMAND_PATH,
            "--self-test",
            "--self-test-fixture-dir",
            str(tmp_path / "fixtures"),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["self_test_passed"] is True
    assert all(check["passed"] for check in result["checks"])
    # the harness must exercise every one of the six angle definitions on the fixture
    assert any(check["name"] == "all_six_angle_definitions_scored" for check in result["checks"])


def test_opencap_compare_cli_direct_invocation_on_synthetic_fixture_writes_report(tmp_path: Path) -> None:
    fixture = opencap_cli.build_synthetic_fixture(fixture_dir=tmp_path / "fixtures")
    out_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            COMMAND_PATH,
            "--opencap-trc",
            str(fixture["trc"]),
            "--opencap-mot",
            str(fixture["mot"]),
            "--our-joints",
            str(fixture["our_joints"]),
            "--player-id",
            "1",
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["artifact_type"] == "racketsport_opencap_body_gt_comparison"
    assert set(report["shared_joint_names"]) == set(SHARED_CORE_JOINT_NAMES)
    assert report["position"]["variants_m"]["clip_level_rigid_aligned_mpjpe"] < 0.003
    assert report["angles"]["knee_flexion_r"]["status"] == "scored"
    assert abs(report["angles"]["knee_flexion_r"]["mae_deg"] - fixture["injected"]["angle_bias_deg"]) < 0.5


def test_opencap_compare_cli_rejects_missing_input_files(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            COMMAND_PATH,
            "--opencap-trc",
            str(tmp_path / "missing.trc"),
            "--our-joints",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "out.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "missing --opencap-trc file" in completed.stderr


def test_opencap_compare_cli_requires_trc_and_joints_without_self_test(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, COMMAND_PATH, "--out", str(tmp_path / "out.json")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "required unless --self-test" in completed.stderr


# ---------------------------------------------------------------------------
# Synthetic-fixture recovery test (task 2), called directly (fast, no subprocess)
# ---------------------------------------------------------------------------


def test_run_self_test_recovers_injected_error_directly(tmp_path: Path) -> None:
    result = opencap_cli.run_self_test(fixture_dir=tmp_path / "fixtures")
    assert result["self_test_passed"] is True
    checks_by_name = {check["name"]: check for check in result["checks"]}
    aligned_check = checks_by_name["clip_level_rigid_aligned_mpjpe_recovers_near_zero_after_injected_similarity_transform"]
    assert aligned_check["value_m"] < opencap_cli.SELF_TEST_POSITION_TOLERANCE_M
    raw_check = checks_by_name["raw_mpjpe_reflects_injected_translation_when_unaligned"]
    assert raw_check["value_m"] >= raw_check["expected_at_least_m"]
    for angle_name in ("knee_flexion_r", "knee_flexion_l", "elbow_flexion_r", "elbow_flexion_l", "hip_flexion_r", "hip_flexion_l"):
        check = checks_by_name[f"angle_mae_recovers_injected_bias[{angle_name}]"]
        assert abs(check["mae_deg"] - check["injected_bias_deg"]) <= check["tolerance_deg"]


def test_build_synthetic_fixture_is_reproducible_given_same_seed(tmp_path: Path) -> None:
    fixture_a = opencap_cli.build_synthetic_fixture(fixture_dir=tmp_path / "a", seed=42)
    fixture_b = opencap_cli.build_synthetic_fixture(fixture_dir=tmp_path / "b", seed=42)
    assert fixture_a["trc"].read_text() == fixture_b["trc"].read_text()
    assert fixture_a["mot"].read_text() == fixture_b["mot"].read_text()


# ---------------------------------------------------------------------------
# .trc / .mot parsing round-trip unit tests
# ---------------------------------------------------------------------------


def test_parse_trc_round_trips_marker_names_units_and_positions(tmp_path: Path) -> None:
    marker_names = ["left_hip", "right_hip", "left_knee"]
    times = np.array([0.0, 1 / 30.0, 2 / 30.0])
    positions_m = np.array(
        [
            [[0.1, 0.9, 0.0], [-0.1, 0.9, 0.0], [0.12, 0.48, 0.02]],
            [[0.1, 0.9, 0.0], [-0.1, 0.9, 0.0], [0.12, 0.40, 0.05]],
            [[0.1, 0.9, 0.0], [-0.1, 0.9, 0.0], [0.12, 0.48, 0.02]],
        ]
    )
    path = tmp_path / "sample.trc"
    opencap_cli._write_trc(path, marker_names=marker_names, times=times, positions_m=positions_m, data_rate=30.0)

    trc = opencap_cli.parse_trc(path)
    assert trc.marker_names == marker_names
    assert trc.data_rate == pytest.approx(30.0)
    np.testing.assert_allclose(trc.times, times, atol=1e-6)
    np.testing.assert_allclose(trc.positions_m, positions_m, atol=1e-6)


def test_parse_trc_converts_millimeter_units_to_meters(tmp_path: Path) -> None:
    path = tmp_path / "mm.trc"
    lines = [
        "PathFileType\t4\t(X/Y/Z)\tmm.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        "30.0\t30.0\t1\t1\tmm\t30.0\t1\t1",
        "Frame#\tTime\tLKnee\t\t",
        "\t\tX1\tY1\tZ1",
        "1\t0.000000\t1000.0\t2000.0\t3000.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    trc = opencap_cli.parse_trc(path)
    np.testing.assert_allclose(trc.positions_m[0, 0], [1.0, 2.0, 3.0])


def test_parse_trc_rejects_num_markers_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.trc"
    lines = [
        "PathFileType\t4\t(X/Y/Z)\tbad.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        "30.0\t30.0\t1\t2\tmm\t30.0\t1\t1",
        "Frame#\tTime\tLKnee\t\t",
        "\t\tX1\tY1\tZ1",
        "1\t0.000000\t1000.0\t2000.0\t3000.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(opencap_cli.OpenCapCompareError, match="NumMarkers"):
        opencap_cli.parse_trc(path)


def test_parse_mot_round_trips_columns_and_converts_radians_to_degrees(tmp_path: Path) -> None:
    column_names = ["knee_angle_r", "hip_flexion_r"]
    times = np.array([0.0, 1 / 30.0])
    values_deg = np.array([[10.0, 5.0], [20.0, 6.0]])
    path = tmp_path / "sample.mot"
    opencap_cli._write_mot(path, column_names=column_names, times=times, values_deg=values_deg)

    mot = opencap_cli.parse_mot(path)
    assert mot.column_names == column_names
    np.testing.assert_allclose(mot.values_deg, values_deg, atol=1e-6)

    # radians header must be converted to degrees
    radians_path = tmp_path / "radians.mot"
    radians_lines = [
        "Coordinates",
        "version=1",
        "nRows=1",
        "nColumns=2",
        "inDegrees=no",
        "",
        "endheader",
        "time\tknee_angle_r",
        f"0.000000\t{np.pi / 2:.6f}",
    ]
    radians_path.write_text("\n".join(radians_lines) + "\n", encoding="utf-8")
    mot_radians = opencap_cli.parse_mot(radians_path)
    assert mot_radians.values_deg[0, 0] == pytest.approx(90.0, abs=1e-3)


def test_parse_mot_rejects_missing_endheader(tmp_path: Path) -> None:
    path = tmp_path / "no_endheader.mot"
    path.write_text("Coordinates\nversion=1\ninDegrees=yes\ntime\tknee_angle_r\n0.0\t10.0\n", encoding="utf-8")
    with pytest.raises(opencap_cli.OpenCapCompareError, match="endheader"):
        opencap_cli.parse_mot(path)


# ---------------------------------------------------------------------------
# Predicted-joints loader unit tests (both schema shapes + fallbacks)
# ---------------------------------------------------------------------------


def test_load_predicted_sequence_players_shape_with_explicit_joint_names(tmp_path: Path) -> None:
    payload = {
        "fps": 30.0,
        "joint_names": list(SHARED_CORE_JOINT_NAMES),
        "players": [
            {
                "id": 1,
                "frames": [
                    {"frame_idx": 0, "joints_world": [[0.0, 0.0, 0.0]] * len(SHARED_CORE_JOINT_NAMES)},
                    {"frame_idx": 1, "joints_world": [[0.1, 0.0, 0.0]] * len(SHARED_CORE_JOINT_NAMES)},
                ],
            }
        ],
    }
    path = tmp_path / "world.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    sequence = opencap_cli.load_predicted_sequence(path, player_id=1, fps=None)
    assert sequence.joint_names == list(SHARED_CORE_JOINT_NAMES)
    np.testing.assert_allclose(sequence.times, [0.0, 1 / 30.0])


def test_load_predicted_sequence_falls_back_to_mhr70_when_joint_names_absent_and_width_70(tmp_path: Path) -> None:
    payload = {
        "fps": 60.0,
        "players": [{"id": 1, "frames": [{"frame_idx": 0, "joints_world": [[0.0, 0.0, 0.0]] * 70}]}],
    }
    path = tmp_path / "smpl_motion.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    sequence = opencap_cli.load_predicted_sequence(path, player_id=1, fps=None)
    assert sequence.joint_names == list(MHR70_JOINT_NAMES)


def test_load_predicted_sequence_rejects_ambiguous_joint_count_without_names(tmp_path: Path) -> None:
    payload = {"fps": 30.0, "players": [{"id": 1, "frames": [{"frame_idx": 0, "joints_world": [[0.0, 0.0, 0.0]] * 5}]}]}
    path = tmp_path / "ambiguous.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(opencap_cli.OpenCapCompareError, match="cannot infer joint identity"):
        opencap_cli.load_predicted_sequence(path, player_id=1, fps=None)


def test_load_predicted_sequence_samples_shape_requires_fps_and_honors_accepted_flag(tmp_path: Path) -> None:
    payload = {
        "joint_names": list(SHARED_CORE_JOINT_NAMES),
        "samples": [
            {"frame_index": 0, "joints_world": [[0.0, 0.0, 0.0]] * len(SHARED_CORE_JOINT_NAMES), "accepted": True},
            {"frame_index": 1, "joints_world": [[9.0, 9.0, 9.0]] * len(SHARED_CORE_JOINT_NAMES), "accepted": False},
        ],
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(opencap_cli.OpenCapCompareError, match="--fps"):
        opencap_cli.load_predicted_sequence(path, player_id=None, fps=None)
    sequence = opencap_cli.load_predicted_sequence(path, player_id=None, fps=30.0)
    assert sequence.times.shape[0] == 1  # the accepted=False sample must be dropped


def test_load_predicted_sequence_multiple_players_requires_explicit_player_id(tmp_path: Path) -> None:
    payload = {
        "fps": 30.0,
        "joint_names": list(SHARED_CORE_JOINT_NAMES),
        "players": [
            {"id": 1, "frames": [{"frame_idx": 0, "joints_world": [[0.0, 0.0, 0.0]] * len(SHARED_CORE_JOINT_NAMES)}]},
            {"id": 2, "frames": [{"frame_idx": 0, "joints_world": [[1.0, 1.0, 1.0]] * len(SHARED_CORE_JOINT_NAMES)}]},
        ],
    }
    path = tmp_path / "two_players.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(opencap_cli.OpenCapCompareError, match="multiple players"):
        opencap_cli.load_predicted_sequence(path, player_id=None, fps=None)
    sequence = opencap_cli.load_predicted_sequence(path, player_id=2, fps=None)
    np.testing.assert_allclose(sequence.positions_m[0, 0], [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Marker-name alias resolution + angle math unit tests
# ---------------------------------------------------------------------------


def test_resolve_shared_joints_matches_opencap_style_aliases() -> None:
    trc_marker_names = ["LKnee", "RKnee", "r_hip_study", "LeftAnkle", "SomeUnrelatedMarker"]
    our_joint_names = list(SHARED_CORE_JOINT_NAMES)
    resolved = opencap_cli.resolve_shared_joints(
        canonical_names=SHARED_CORE_JOINT_NAMES, trc_marker_names=trc_marker_names, our_joint_names=our_joint_names
    )
    assert resolved["left_knee"]["trc_marker"] == "LKnee"
    assert resolved["right_knee"]["trc_marker"] == "RKnee"
    assert resolved["right_hip"]["trc_marker"] == "r_hip_study"
    assert resolved["left_ankle"]["trc_marker"] == "LeftAnkle"
    assert "right_ankle" not in resolved  # no matching marker supplied


def test_resolve_shared_joints_matches_exact_canonical_name_without_alias() -> None:
    resolved = opencap_cli.resolve_shared_joints(
        canonical_names=SHARED_CORE_JOINT_NAMES,
        trc_marker_names=list(SHARED_CORE_JOINT_NAMES),
        our_joint_names=list(SHARED_CORE_JOINT_NAMES),
    )
    assert set(resolved) == set(SHARED_CORE_JOINT_NAMES)


def test_three_point_flexion_deg_straight_line_is_zero_and_right_angle_is_ninety() -> None:
    straight = opencap_cli._three_point_flexion_deg(
        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
    )
    assert straight == pytest.approx(0.0, abs=1e-6)

    right_angle = opencap_cli._three_point_flexion_deg(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    assert right_angle == pytest.approx(90.0, abs=1e-6)


def test_compare_clip_raises_when_too_few_shared_joints_resolve(tmp_path: Path) -> None:
    fixture = opencap_cli.build_synthetic_fixture(fixture_dir=tmp_path / "fixtures")
    trc = opencap_cli.parse_trc(fixture["trc"])
    # rename all but 2 markers so fewer than MIN_SHARED_JOINT_COUNT resolve
    trc.marker_names = ["totally_unmatched_marker"] * (len(trc.marker_names) - 2) + trc.marker_names[-2:]
    predicted = opencap_cli.load_predicted_sequence(fixture["our_joints"], player_id=1, fps=None)
    with pytest.raises(opencap_cli.OpenCapCompareError, match="shared joints resolved"):
        opencap_cli.compare_clip(trc=trc, mot=None, predicted=predicted, sync_offset_seconds=0.0)
