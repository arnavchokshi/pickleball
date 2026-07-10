from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.racketsport import attribute_body_decode_residual as attribution
from threed.racketsport import worldhmr


CLI_PATH = "scripts/racketsport/attribute_body_decode_residual.py"
ROOT = Path(__file__).resolve().parents[2]
BANKED_RUN = (
    ROOT
    / "runs/lanes/w7_p22gate_20260709/gpu_instrument_wolverine_mixed_0200_raw_postchain"
    / "wolverine_mixed_0200_mid_steep_corner"
)
BANKED_INDEX = ROOT / "runs/lanes/w7_p22gate_20260709/fast_sam_subprocess_sample/index.json"


def test_banked_fixture_grounding_and_disabled_replay_pass_frozen_limits(tmp_path: Path) -> None:
    out = tmp_path / "attribution.json"
    report = attribution.run(
        Namespace(
            run_dir=BANKED_RUN,
            calibration=BANKED_RUN / "court_calibration.json",
            raw_grounded=BANKED_RUN / "body_raw_grounded_joints.json",
            sam3d_output_index=BANKED_INDEX,
            out=out,
            max_frames_per_player=0,
            checkpoint=Path("missing.ckpt"),
            mhr_asset=Path("missing_mhr.pt"),
            device=None,
        )
    )

    assert report["input_summary"]["selected_raw_record_count"] == 32
    assert report["inputs_coherent"] is True
    assert report["input_coherence"]["coverage_fraction"] == 1.0
    assert report["grounding_determinism"]["overall"]["p95_mm"] <= 1.0
    assert report["grounding_determinism"]["overall"]["max_mm"] <= 2.0
    assert report["grounding_determinism"]["passed_1mm"] is True
    assert report["postchain_attribution"]["all_stages_disabled"] is True
    assert report["postchain_attribution"]["all_stages_disabled_identity_max_m"] <= 1e-6
    assert report["replay_validation"]["chain_reproduced_1mm"] is True
    assert report["fk_vs_head_divergence"]["status"] in {
        "measured",
        "blocked_mhr_runtime_unavailable",
    }
    assert out.is_file()


def test_incoherent_body_mesh_pairing_writes_evidence_and_exits_nonzero(tmp_path: Path) -> None:
    run_dir = tmp_path / "stale_pairing"
    run_dir.mkdir()
    (run_dir / "body_mesh.json").write_text(
        json.dumps(
            {
                "players": [
                    {
                        "id": 999,
                        "frames": [{"frame_idx": 999, "joints_world": [[0.0, 0.0, 0.0]]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "incoherent.json"
    exit_code = attribution.main(
        [
            "--run-dir",
            str(run_dir),
            "--calibration",
            str(BANKED_RUN / "court_calibration.json"),
            "--raw-grounded",
            str(BANKED_RUN / "body_raw_grounded_joints.json"),
            "--sam3d-output-index",
            str(BANKED_INDEX),
            "--out",
            str(out),
        ]
    )

    assert exit_code != 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "incoherent_inputs"
    assert report["inputs_coherent"] is False
    evidence = report["input_coherence"]
    assert evidence["missing_request_id_count"] == 1
    assert evidence["missing_request_ids_first_10"] == ["999:999"]
    assert "grounding_determinism" not in report


def test_allow_incoherent_inputs_stamps_every_computed_result_block(tmp_path: Path) -> None:
    raw_reference = json.loads((BANKED_RUN / "body_raw_grounded_joints.json").read_text(encoding="utf-8"))
    present_ids = set(attribution.load_present_raw_records(BANKED_INDEX))
    players = []
    for player in raw_reference["players"]:
        frames = [
            frame
            for frame in player["frames"]
            if f"{frame['frame_idx']}:{player['id']}" in present_ids
        ]
        if frames:
            players.append({"id": player["id"], "frames": frames})
    players.append(
        {
            "id": 999,
            "frames": [{"frame_idx": 999, "joints_world": [[0.0, 0.0, 0.0]]}],
        }
    )
    run_dir = tmp_path / "forensic_pairing"
    run_dir.mkdir()
    (run_dir / "body_mesh.json").write_text(json.dumps({"players": players}), encoding="utf-8")
    out = tmp_path / "forensic.json"

    exit_code = attribution.main(
        [
            "--run-dir",
            str(run_dir),
            "--calibration",
            str(BANKED_RUN / "court_calibration.json"),
            "--raw-grounded",
            str(BANKED_RUN / "body_raw_grounded_joints.json"),
            "--sam3d-output-index",
            str(BANKED_INDEX),
            "--allow-incoherent-inputs",
            "--checkpoint",
            "missing.ckpt",
            "--mhr-asset",
            "missing_mhr.pt",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["inputs_coherent"] is False
    assert report["input_coherence"]["missing_request_ids_first_10"] == ["999:999"]
    for block_name in (
        "grounding_determinism",
        "postchain_attribution",
        "replay_validation",
        "fk_vs_head_divergence",
    ):
        assert report[block_name]["inputs_coherent"] is False
        assert report[block_name]["input_coherence"] == report["input_coherence"]


def test_direct_provenance_stamp_mismatch_is_incoherent() -> None:
    evidence = attribution._input_coherence_evidence(
        expected_request_ids=["1:2"],
        records={"1:2": {}},
        index_payload={
            "provenance": {
                "batch_id": "old-batch",
                "out_path": "/run_20260709T010101Z/batch_outputs-aaaa.json",
            }
        },
        run_payload={
            "body_stage_phase_timing.json": {
                "batch_id": "new-batch",
                "out_path": "/run_20260709T020202Z/batch_outputs-bbbb.json.timing.json",
            }
        },
        coverage_threshold=1.0,
        coverage_source="body_mesh_player_frames",
    )

    assert evidence["inputs_coherent"] is False
    assert {mismatch["stamp"] for mismatch in evidence["stamp_mismatches"]} == {
        "batch_id",
        "out_path",
        "sam3d_batch_id_from_path",
        "sam3d_execution_timestamp_from_path",
    }


def test_auto_discovery_ambiguity_lists_every_candidate(tmp_path: Path) -> None:
    candidates = [
        tmp_path / "fast_sam_subprocess" / f"batch_outputs-{name}.json.chunks" / "index.json"
        for name in ("one", "two")
    ]
    for candidate in candidates:
        candidate.parent.mkdir(parents=True)
        candidate.write_text("{}", encoding="utf-8")

    with pytest.raises(attribution.AttributionInputError) as exc_info:
        attribution._discover_index(tmp_path)
    message = str(exc_info.value)
    assert all(str(candidate) in message for candidate in candidates)


def test_grounding_without_reference_is_explicitly_not_measured() -> None:
    result = attribution._grounding_determinism({("1", 0): [[0.0, 0.0, 0.0]]}, None)
    assert result == {
        "status": "no_reference",
        "passed_1mm": None,
        "per_player": {},
        "overall": None,
    }


def test_all_stages_disabled_replay_is_identity_to_one_micrometre() -> None:
    grounded = {("7", 3): [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]}
    report = attribution._postchain_attribution(
        grounded,
        {},
        grounded,
        postchain=attribution.BodyPostChainConfig.raw(),
    )
    assert report["all_stages_disabled"] is True
    assert report["all_stages_disabled_identity_max_m"] == 0.0
    assert all(stage["delta_vs_previous"]["overall"]["max_mm"] == 0.0 for stage in report["stages"])


def test_snapshot_capture_wraps_real_stage_functions_and_restores_them() -> None:
    frame = {
        "frame_idx": 1,
        "player_id": 2,
        "t": 1 / 30.0,
        "transl_world": [0.0, 0.0, 0.0],
        "track_world_xy": [0.0, 0.0],
        "joints_world": [[0.0, 0.0, 0.0]],
        "vertices_world": [],
        "confidence": 1.0,
        "foot_lock": {"left": False, "right": False},
    }
    original = worldhmr._bypass_temporal_smoothing
    capture = attribution.WorldHmrStageCapture(worldhmr)
    with capture:
        worldhmr._bypass_temporal_smoothing([frame], stance_aware_grounding=False)
        worldhmr._bypass_footlock_for_player_frames([frame])
        payload = {"joint_names": ["root"], "players": [{"id": 2, "frames": [frame]}]}
        worldhmr._apply_world_joint_visual_smoothing(payload, payload, fps=30.0, enabled=False)
        assert "temporal_smoothing" in capture.snapshots
        assert "foot_lock" in capture.snapshots
        assert "world_joint_visual_smoothing" in capture.snapshots
    assert worldhmr._bypass_temporal_smoothing is original


def test_cli_help_and_scaffold_direct_reference() -> None:
    help_completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--raw-grounded" in help_completed.stdout
    assert "--sam3d-output-index" in help_completed.stdout
    assert "--allow-incoherent-inputs" in help_completed.stdout

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/list_scaffold_tools.py", "--root", "."],
        check=True,
        capture_output=True,
        text=True,
    )
    tools = {tool["command_path"]: tool for tool in json.loads(completed.stdout)["tools"]}
    assert tools[CLI_PATH]["category"] == "decode"
    assert tools[CLI_PATH]["direct_cli_reference_test"] == "tests/racketsport/test_attribute_body_decode_residual.py"
