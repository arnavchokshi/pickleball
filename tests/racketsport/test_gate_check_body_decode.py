from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport import gate_check_body_decode as gate
from threed.racketsport import mhr_decode


CLI_PATH = "scripts/racketsport/gate_check_body_decode.py"


def _numbers(count: int, *, start: float = 0.0, step: float = 0.001) -> list[float]:
    return [start + step * idx for idx in range(count)]


def _fixture_body_mesh(*, include_scale: bool = True) -> dict:
    smplx_params = {
        "global_orient": [0.1, 0.2, 0.3],
        "body_pose": _numbers(mhr_decode.BODY_POSE_EULER_DIM, start=0.01, step=0.002),
        "betas": _numbers(mhr_decode.SHAPE_DIM, start=0.2, step=0.003),
        "left_hand_pose": _numbers(mhr_decode.HAND_COMPS_DIM, start=1.0, step=0.01),
        "right_hand_pose": _numbers(mhr_decode.HAND_COMPS_DIM, start=-1.0, step=-0.01),
        "transl_world": [1.25, -2.5, 0.0],
    }
    if include_scale:
        smplx_params["scale"] = _numbers(mhr_decode.SCALE_DIM, start=0.5, step=0.01)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        "clip": "unit_fixture",
        "players": [
            {
                "id": 42,
                "frames": [
                    {
                        "frame_idx": 7,
                        "t": 7 / 30.0,
                        "smplx_params": smplx_params,
                        "joints_world": [[1.0, 2.0, 3.0]],
                        "mesh_vertices_world": [[4.0, 5.0, 6.0]],
                    }
                ],
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_extract_frames_carries_scale_and_hand_pose_verbatim() -> None:
    frame = gate.extract_frames(_fixture_body_mesh())["42"][0]
    params = _fixture_body_mesh()["players"][0]["frames"][0]["smplx_params"]

    assert frame.scale == params["scale"]
    assert frame.left_hand_pose == params["left_hand_pose"]
    assert frame.right_hand_pose == params["right_hand_pose"]
    assert frame.hand_pose == params["left_hand_pose"] + params["right_hand_pose"]
    assert frame.decode_kwargs(scale_source="field")["scale"] == params["scale"]
    assert frame.decode_kwargs(scale_source="field")["hand_pose"] == params["left_hand_pose"] + params["right_hand_pose"]


def test_requested_field_scale_fails_loudly_when_absent() -> None:
    frame = gate.extract_frames(_fixture_body_mesh(include_scale=False))["42"][0]

    with pytest.raises(gate.HarnessInputError, match="--scale-source field requested"):
        frame.decode_kwargs(scale_source="field")


def test_decoder_provenance_records_paths_hashes_and_module_stamp(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    mhr_asset = tmp_path / "mhr_model.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    mhr_asset.write_bytes(b"mhr asset bytes")

    provenance = gate.build_decoder_provenance(checkpoint_path=checkpoint, mhr_asset_path=mhr_asset)

    assert provenance["checkpoint"]["path"] == str(checkpoint)
    assert provenance["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint bytes").hexdigest()
    assert provenance["mhr_asset"]["path"] == str(mhr_asset)
    assert provenance["mhr_asset"]["sha256"] == hashlib.sha256(b"mhr asset bytes").hexdigest()
    assert provenance["mhr_decode_module"]["path"].endswith("threed/racketsport/mhr_decode.py")
    assert len(provenance["mhr_decode_module"]["sha256"]) == 64
    assert provenance["mhr_runtime_available"] is mhr_decode.MHR_RUNTIME_AVAILABLE


def test_blocked_runtime_report_keeps_metric_key_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.mhr_decode, "MHR_RUNTIME_AVAILABLE", False)
    monkeypatch.setattr(gate.mhr_decode, "MHR_RUNTIME_IMPORT_ERROR", ModuleNotFoundError("No module named 'roma'"))
    body_mesh = _write_json(tmp_path / "body_mesh.json", _fixture_body_mesh())
    out = tmp_path / "report.json"

    parser = gate.build_arg_parser()
    args = parser.parse_args(
        [
            "--body-mesh",
            str(body_mesh),
            "--court-calibration",
            str(tmp_path / "court_calibration.json"),
            "--out",
            str(out),
            "--scale-source",
            "field",
            "--checkpoint",
            str(tmp_path / "missing_model.ckpt"),
            "--mhr-asset",
            str(tmp_path / "missing_mhr_model.pt"),
        ]
    )
    report = gate.run(args)

    assert report["measurement_status"] == "blocked_mhr_runtime_unavailable"
    assert report["total_real_frame_sample_count"] == 1
    assert report["gate_1a"]["gate"] == "gate_1a_euler_cont_euler_idempotence"
    assert report["gate_1b"]["gate"] == "gate_1b_world_round_trip"
    assert "worst_joints_world_max_abs_error_mm" in report["gate_1b"]
    assert "worst_p95_mm_over_sample" in report["mesh_skeleton_divergence"]
    assert report["decoder_provenance"]["checkpoint"]["sha256"] is None
    assert out.is_file()


def test_self_check_uses_stub_decoder_and_records_field_flow(tmp_path: Path) -> None:
    out = tmp_path / "self_check_report.json"
    exit_code = gate.main(["--self-check", "--out", str(out)])

    assert exit_code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    self_check = report["self_check"]
    assert self_check["passed"] is True
    assert self_check["decode_call_count"] == 1
    assert self_check["divergence_call_count"] == 1
    assert self_check["first_decode_call"]["scale"] == report["self_check_fixture"]["scale"]
    assert self_check["first_decode_call"]["hand_pose"] == report["self_check_fixture"]["hand_pose"]


def test_cli_help_references_gate_1b_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--scale-source" in completed.stdout
    assert "--self-check" in completed.stdout
    assert "GATE-1b" in completed.stdout


def test_scaffold_index_places_cli_in_decode_category_with_direct_reference() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/list_scaffold_tools.py",
            "--root",
            ".",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    tools = {tool["command_path"]: tool for tool in payload["tools"]}

    entry = tools[CLI_PATH]
    assert entry["category"] == "decode"
    assert entry["direct_cli_reference_test"] == "tests/racketsport/test_gate_check_body_decode.py"
