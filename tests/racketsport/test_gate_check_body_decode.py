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


class _CameraPointDecoder:
    def __init__(self, *, joints_camera: list[list[float]], vertices_camera: list[list[float]]) -> None:
        self.joints_camera = joints_camera
        self.vertices_camera = vertices_camera

    def decode_euler_frame(self, **_kwargs: object) -> dict:
        return {
            "joints_camera": [self.joints_camera],
            "vertices_camera": [self.vertices_camera],
        }

    def mesh_skeleton_divergence_mm(self, **_kwargs: object) -> dict:
        return {"p95_mm": 1.25}


def _install_identity_ground(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    captured: list[dict] = []

    def fake_ground(**kwargs: object) -> dict:
        captured.append(dict(kwargs))
        return {
            "joints_world": mhr_decode.apply_pred_cam_t_once(
                kwargs["joints_camera"],
                pred_cam_t=kwargs.get("pred_cam_t"),
                already_applied=bool(kwargs.get("pred_cam_t_already_applied")),
            ),
            "vertices_world": mhr_decode.apply_pred_cam_t_once(
                kwargs["vertices_camera"],
                pred_cam_t=kwargs.get("pred_cam_t"),
                already_applied=bool(kwargs.get("pred_cam_t_already_applied")),
            ),
        }

    monkeypatch.setattr(gate.mhr_decode, "ground_decoded_camera_frame", fake_ground)
    return captured


def test_extract_frames_carries_scale_and_hand_pose_verbatim() -> None:
    frame = gate.extract_frames(_fixture_body_mesh())["42"][0]
    params = _fixture_body_mesh()["players"][0]["frames"][0]["smplx_params"]

    assert frame.scale == params["scale"]
    assert frame.left_hand_pose == params["left_hand_pose"]
    assert frame.right_hand_pose == params["right_hand_pose"]
    assert frame.hand_pose == params["left_hand_pose"] + params["right_hand_pose"]
    assert frame.decode_kwargs(scale_source="field")["scale"] == params["scale"]
    assert frame.decode_kwargs(scale_source="field")["hand_pose"] == params["left_hand_pose"] + params["right_hand_pose"]


def test_gate_1b_regrounding_uses_raw_pred_cam_t_once(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_identity_ground(monkeypatch)
    body_mesh = _fixture_body_mesh()
    frame_payload = body_mesh["players"][0]["frames"][0]
    frame_payload["joints_world"] = [[1.25, 1.5, 4.0], [2.25, 1.5, 4.0]]
    frame_payload["mesh_vertices_world"] = [[4.25, 4.5, 7.0]]
    raw_records = {"7:42": {"pred_cam_t": [0.25, -0.5, 1.0]}}
    frames_by_player = gate.extract_frames(body_mesh, raw_records_by_request_id=raw_records)
    decoder = _CameraPointDecoder(
        joints_camera=[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
        vertices_camera=[[4.0, 5.0, 6.0]],
    )

    gate1b, divergence = gate._compute_gate_1b_and_divergence(
        decoder=decoder,
        calibration=object(),
        frames_by_player=frames_by_player,
        scale_source="field",
        max_frames_per_player=1,
    )

    assert captured[0]["pred_cam_t"] == [0.25, -0.5, 1.0]
    assert captured[0]["pred_cam_t_already_applied"] is False
    assert gate1b["worst_joints_world_max_abs_error_mm"] == pytest.approx(0.0, abs=1e-9)
    assert gate1b["worst_vertices_world_max_abs_error_mm"] == pytest.approx(0.0, abs=1e-9)
    assert divergence["worst_p95_mm_over_sample"] == pytest.approx(1.25)


def test_gate_1b_respects_pred_cam_t_already_applied_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_identity_ground(monkeypatch)
    body_mesh = _fixture_body_mesh()
    frame_payload = body_mesh["players"][0]["frames"][0]
    frame_payload["joints_world"] = [[1.25, 1.5, 4.0]]
    frame_payload["mesh_vertices_world"] = [[4.25, 4.5, 7.0]]
    raw_records = {
        "7:42": {
            "pred_cam_t": [0.25, -0.5, 1.0],
            "pred_cam_t_already_applied": True,
        }
    }
    frames_by_player = gate.extract_frames(body_mesh, raw_records_by_request_id=raw_records)
    decoder = _CameraPointDecoder(
        joints_camera=[[1.25, 1.5, 4.0]],
        vertices_camera=[[4.25, 4.5, 7.0]],
    )

    gate1b, _divergence = gate._compute_gate_1b_and_divergence(
        decoder=decoder,
        calibration=object(),
        frames_by_player=frames_by_player,
        scale_source="field",
        max_frames_per_player=1,
    )

    assert captured[0]["pred_cam_t"] == [0.25, -0.5, 1.0]
    assert captured[0]["pred_cam_t_already_applied"] is True
    assert gate1b["worst_joints_world_max_abs_error_mm"] == pytest.approx(0.0, abs=1e-9)
    assert gate1b["worst_vertices_world_max_abs_error_mm"] == pytest.approx(0.0, abs=1e-9)


def test_gate_1b_surfaces_absent_vertices_without_vacuous_measured_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_identity_ground(monkeypatch)
    body_mesh = _fixture_body_mesh()
    frame_payload = body_mesh["players"][0]["frames"][0]
    frame_payload["joints_world"] = [[1.25, 1.5, 4.0]]
    frame_payload["mesh_vertices_world"] = []
    frames_by_player = gate.extract_frames(
        body_mesh,
        raw_records_by_request_id={"7:42": {"pred_cam_t": [0.25, -0.5, 1.0]}},
    )
    decoder = _CameraPointDecoder(joints_camera=[[1.0, 2.0, 3.0]], vertices_camera=[])

    gate1b, _ = gate._compute_gate_1b_and_divergence(
        decoder=decoder,
        calibration=object(),
        frames_by_player=frames_by_player,
        scale_source="field",
        max_frames_per_player=1,
    )

    assert gate1b["passed"] is True
    assert gate1b["vertices_status"] == "absent_not_measured"
    assert gate1b["measured_vertices_frame_count"] == 0
    assert gate1b["worst_vertices_world_max_abs_error_mm"] is None
    assert gate1b["per_player"]["42"]["vertices_world_p95_abs_error_mm"] is None


def test_gate_1a_is_bit_unaffected_by_pred_cam_t_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_gate_1a(global_orient: object, body_pose: object) -> dict:
        assert hasattr(global_orient, "tolist")
        assert hasattr(body_pose, "tolist")
        return {
            "gate": "gate_1a_euler_cont_euler_idempotence",
            "global_orient": global_orient.tolist(),
            "body_pose": body_pose.tolist(),
            "target_max_abs_error_deg": 0.1,
            "max_abs_error_deg": 0.0,
            "passed": True,
        }

    monkeypatch.setattr(gate.mhr_decode, "gate_1a_euler_round_trip", fake_gate_1a)
    baseline = gate._compute_gate_1a(gate.extract_frames(_fixture_body_mesh()))
    with_cam_t = _fixture_body_mesh()
    frame_payload = with_cam_t["players"][0]["frames"][0]
    frame_payload["camera_translation"] = [9.0, 8.0, 7.0]
    frame_payload["pred_cam_t_already_applied"] = True

    after = gate._compute_gate_1a(gate.extract_frames(with_cam_t))

    assert after == baseline


def test_absent_raw_emit_without_camera_translation_blocks_gate_1b_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_mesh_path = _write_json(tmp_path / "body_mesh.json", _fixture_body_mesh())
    court_cal_path = _write_json(tmp_path / "court_calibration.json", {"stub": True})
    out = tmp_path / "report.json"
    _install_identity_ground(monkeypatch)

    monkeypatch.setattr(gate.mhr_decode, "MHR_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(gate.mhr_decode, "MHR_RUNTIME_IMPORT_ERROR", None)
    monkeypatch.setattr(
        gate.mhr_decode,
        "MHRDecoder",
        lambda **_kwargs: _CameraPointDecoder(
            joints_camera=[[1.0, 2.0, 3.0]],
            vertices_camera=[[4.0, 5.0, 6.0]],
        ),
    )
    monkeypatch.setattr(
        gate.mhr_decode,
        "gate_1a_euler_round_trip",
        lambda _global_orient, _body_pose: {
            "gate": "gate_1a_euler_cont_euler_idempotence",
            "max_abs_error_deg": 0.0,
            "passed": True,
        },
    )
    monkeypatch.setattr(gate.CourtCalibration, "model_validate", classmethod(lambda cls, _payload: object()))
    monkeypatch.setattr(gate, "build_decoder_provenance", lambda **_kwargs: {"stubbed_for_unit": True})

    exit_code = gate.main(
        [
            "--body-mesh",
            str(body_mesh_path),
            "--court-calibration",
            str(court_cal_path),
            "--out",
            str(out),
            "--scale-source",
            "field",
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert report["measurement_status"] == "measured_gate_1b_blocked"
    assert report["blocker"] == report["gate_1b"]["blocked_reason"]
    assert report["gate_1a"]["passed"] is True
    assert report["mesh_skeleton_divergence"]["passed"] is True
    assert report["mesh_skeleton_divergence"]["worst_p95_mm_over_sample"] == pytest.approx(1.25)
    assert report["gate_1b"]["status"] == "blocked_missing_pred_cam_t"
    assert report["gate_1b"]["passed"] is None
    assert report["gate_1b"]["worst_joints_world_max_abs_error_mm"] is None
    assert report["gate_1b"]["worst_vertices_world_max_abs_error_mm"] is None
    assert report["gate_1b_world_round_trip"] == report["gate_1b"]
    assert out.is_file()


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


def test_attribution_report_is_embedded_as_additive_residual_decomposition(tmp_path: Path) -> None:
    attribution = _write_json(
        tmp_path / "attribution.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_decode_residual_attribution",
            "grounding_determinism": {"status": "measured", "overall": {"p95_mm": 0.2}},
            "postchain_attribution": {"total_delta": {"p95_mm": 12.0}},
            "fk_vs_head_divergence": {"status": "blocked_mhr_runtime_unavailable"},
        },
    )
    out = tmp_path / "self_check_with_attribution.json"

    exit_code = gate.main(
        ["--self-check", "--out", str(out), "--attribution-report", str(attribution)]
    )
    report = json.loads(out.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["gate_1b_world_round_trip"]["target_max_abs_error_mm"] == 1.0
    assert report["residual_decomposition"] == {
        "source_path": str(attribution),
        "grounding_determinism": {"status": "measured", "overall": {"p95_mm": 0.2}},
        "postchain_totals": {"p95_mm": 12.0},
        "fk_vs_head": {"status": "blocked_mhr_runtime_unavailable"},
    }


def test_cli_help_references_gate_1b_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--scale-source" in completed.stdout
    assert "--self-check" in completed.stdout
    assert "--attribution-report" in completed.stdout
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
