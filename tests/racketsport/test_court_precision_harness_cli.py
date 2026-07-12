from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_court_precision_metrics import _calibration, _write_court_video


def test_court_precision_harness_direct_cli_writes_table_and_overlay(tmp_path: Path) -> None:
    """Direct-CLI reference test for scripts/racketsport/court_precision_harness.py."""

    run_dir = tmp_path / "synthetic_internal_clip"
    run_dir.mkdir()
    (run_dir / "court_calibration.json").write_text(json.dumps(_calibration()), encoding="utf-8")
    _write_court_video(run_dir / "source.mp4", _calibration(), frame_count=3)
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/court_precision_harness.py",
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--sample-count",
            "3",
            "--overlay-count",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    table = json.loads((out_dir / "baseline_table_v2.json").read_text(encoding="utf-8"))
    assert table["artifact_type"] == "court_precision_baseline_table"
    assert table["diagnostic_only"] is True
    assert table["promotion_gate"] is False
    assert table["best_stack_delta"] == "none"
    assert table["scorer_version"] == "cpm_v2_frozen_20260712"
    assert table["freeze_contract"]["scorer_version_policy"] == "manager_bump_required"
    assert table["provenance"]["candidate_evidence_api_imported"] is False
    assert table["clip_count"] == 1
    assert set(table["clips"][0]["metrics"]) == {"M1", "M2", "M3", "M4", "M5"}
    assert table["clips"][0]["freeze_contract"]["frozen_frame_indexes"] == [0, 1, 2]
    assert len(table["clips"][0]["freeze_contract"]["input_sha256"]["video"]["sha256"]) == 64
    assert len(table["clips"][0]["metrics"]["M1"]["frozen_visible_sample_set_sha256"]) == 64
    assert table["clips"][0]["metrics"]["M1"]["overflow_count"] >= 0
    assert table["clips"][0]["metrics"]["M2"]["reason"] == "per_frame_calibration_missing"
    assert set(table["clips"][0]["metrics"]["M5"]) == {
        "status",
        "image_to_world_scale_jacobian",
        "observation_perturbation_bootstrap",
    }
    assert len(table["overlays"]) == 1
    overlay = Path(table["overlays"][0]["path"])
    assert overlay.is_file()
    assert overlay.suffix == ".png"
