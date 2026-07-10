from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.racketsport import synthetic_body_decode_gate as synthetic


CLI_PATH = "scripts/racketsport/synthetic_body_decode_gate.py"


def test_synthetic_body_decode_gate_cpu_mock_writes_metric_report_and_render(tmp_path: Path) -> None:
    out = tmp_path / "synthetic_report.json"
    render_dir = tmp_path / "renders"

    completed = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--out",
            str(out),
            "--render-dir",
            str(render_dir),
            "--samples",
            "2",
            "--decoder",
            "mock",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(out) in completed.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["artifact_type"] == "racketsport_synthetic_body_decode_gate"
    assert report["measurement_status"] == "measured_mock_decoder"
    assert report["sample_count"] == 2
    assert report["gate_1b_world_round_trip"]["joints_world_p95_abs_error_mm"] <= 1.0
    assert report["mesh_skeleton_divergence"]["p95_mm"] <= 5.0
    assert len(report["renders"]) == 2
    for render in report["renders"]:
        assert (render_dir / render["path"]).is_file()


def test_synthetic_body_decode_gate_cli_help_and_scaffold_reference() -> None:
    help_completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--decoder" in help_completed.stdout
    assert "--samples" in help_completed.stdout
    assert "--checkpoint" in help_completed.stdout
    assert "--mhr-asset" in help_completed.stdout
    assert "--device" in help_completed.stdout

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
    assert entry["direct_cli_reference_test"] == "tests/racketsport/test_synthetic_body_decode_gate.py"


class _FakeMhrDecoder:
    class _Head:
        faces = np.asarray([[0, 0, 0]], dtype=np.int64)

    head = _Head()

    def __init__(self, **_kwargs: object) -> None:
        pass

    def decode_euler_frame(self, **_kwargs: object) -> dict:
        return {
            "joints_camera": np.asarray([[[0.0, 0.0, 0.0]]]),
            "vertices_camera": np.asarray([[[0.0, 0.0, 0.0]]]),
        }


def _sam3d_args(tmp_path: Path):
    return synthetic.build_arg_parser().parse_args(
        [
            "--out",
            str(tmp_path / "report.json"),
            "--render-dir",
            str(tmp_path / "renders"),
            "--samples",
            "1",
            "--decoder",
            "sam3d",
        ]
    )


def test_sam3d_adapter_selection_can_emit_measured_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synthetic.mhr_decode, "MHR_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(synthetic.mhr_decode, "MHRDecoder", _FakeMhrDecoder)
    monkeypatch.setattr(synthetic, "_load_sam3d_estimator", lambda _args: object())
    monkeypatch.setattr(
        synthetic,
        "_write_shaded_mesh_render",
        lambda path, *_args, **_kwargs: path.parent.mkdir(parents=True, exist_ok=True) or path.touch(),
    )
    monkeypatch.setattr(
        synthetic,
        "_decode_sam3d_render",
        lambda *_args, **_kwargs: (
            {"joints_world": [[0.15, -0.05, 3.5]], "vertices_world": [[0.15, -0.05, 3.5]]},
            {"valid_detection": True, "record_count": 1},
        ),
    )

    report = synthetic.run(_sam3d_args(tmp_path))

    assert report["measurement_status"] == "measured"
    assert report["attempts"][0]["valid_detection"] is True
    assert report["gate_1b_world_round_trip"]["passed"] is True


def test_sam3d_adapter_runtime_unavailable_is_explicit_blocked_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synthetic.mhr_decode, "MHR_RUNTIME_AVAILABLE", False)
    monkeypatch.setattr(
        synthetic.mhr_decode,
        "MHR_RUNTIME_IMPORT_ERROR",
        ModuleNotFoundError("No module named 'roma'"),
    )

    report = synthetic.run(_sam3d_args(tmp_path))

    assert report["measurement_status"] == "blocked_sam3d_runtime_unavailable"
    assert "roma" in report["blocker"]
    assert report["attempts"] == []


def test_sam3d_two_attempt_kill_rule_reports_undetectable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synthetic.mhr_decode, "MHR_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(synthetic.mhr_decode, "MHRDecoder", _FakeMhrDecoder)
    monkeypatch.setattr(synthetic, "_load_sam3d_estimator", lambda _args: object())
    monkeypatch.setattr(
        synthetic,
        "_write_shaded_mesh_render",
        lambda path, *_args, **_kwargs: path.parent.mkdir(parents=True, exist_ok=True) or path.touch(),
    )
    monkeypatch.setattr(
        synthetic,
        "_decode_sam3d_render",
        lambda *_args, **_kwargs: (None, {"valid_detection": False, "record_count": 0}),
    )

    report = synthetic.run(_sam3d_args(tmp_path))

    assert report["measurement_status"] == "blocked_synthetic_render_not_detectable"
    assert len(report["attempts"]) == 2
    assert [attempt["attempt"] for attempt in report["attempts"]] == [1, 2]
