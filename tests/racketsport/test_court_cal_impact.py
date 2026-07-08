from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.court_cal_impact import build_impact_report


COMMAND_PATH = "scripts/racketsport/court_calibration_impact_harness.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _calibration_payload(*, tx_px: float = 100.0, ty_px: float = 100.0) -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[10.0, 0.0, tx_px], [0.0, 10.0, ty_px], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0, "dist": [], "source": "test"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 0.0],
            "camera_height_m": 1.5,
        },
        "reprojection_error_px": {"median": 1.0, "p95": 2.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[100.0, 100.0], [200.0, 100.0], [100.0, 200.0], [200.0, 200.0]],
        "world_pts": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 10.0, 0.0]],
    }


def _tracks_payload() -> dict:
    frames = []
    for idx, bottom_x in enumerate((110.0, 120.0, 130.0)):
        frames.append(
            {
                "frame_idx": idx,
                "t": idx / 30.0,
                "bbox": [bottom_x - 4.0, 100.0, bottom_x + 4.0, 130.0],
                "conf": 0.9,
            }
        )
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [{"id": 7, "side": "near", "role": "left", "frames": frames}],
    }


def _placement_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_placement",
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "frames": [
                    {"frame_idx": 0, "smoothed_world_xy": [1.0, 3.0]},
                    {"frame_idx": 1, "smoothed_world_xy": [2.0, 3.0]},
                    {"frame_idx": 2, "smoothed_world_xy": [3.0, 3.0]},
                ],
            }
        ],
        "summary": {},
    }


def _fixture_paths(tmp_path: Path, *, candidate_tx_px: float) -> dict[str, Path]:
    paths = {
        "baseline": tmp_path / "baseline_calibration.json",
        "candidate": tmp_path / "candidate_calibration.json",
        "tracks": tmp_path / "tracks.json",
        "placement": tmp_path / "placement.json",
        "grounding": tmp_path / "body_grounding_quality.json",
    }
    _write_json(paths["baseline"], _calibration_payload(tx_px=100.0))
    _write_json(paths["candidate"], _calibration_payload(tx_px=candidate_tx_px))
    _write_json(paths["tracks"], _tracks_payload())
    _write_json(paths["placement"], _placement_payload())
    _write_json(
        paths["grounding"],
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_grounding_quality",
            "status": "pass",
            "grounding_metrics": {
                "max_foot_lock_slide_m": 0.02,
                "foot_lock_slide_p95_m": 0.01,
                "max_candidate_phase_slide_m": 0.02,
            },
        },
    )
    return paths


def _live_metric(report: dict, name: str) -> dict:
    return report["live_metrics"][name]


def test_perturbed_calibration_reports_signed_nonzero_placement_deltas(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path, candidate_tx_px=110.0)

    report = build_impact_report(
        baseline_calibration_path=paths["baseline"],
        candidate_calibration_path=paths["candidate"],
        tracks_path=paths["tracks"],
        placement_path=paths["placement"],
        body_grounding_quality_path=paths["grounding"],
        clip="synthetic_clip",
        generated_at="2026-07-08T00:00:00Z",
    )

    assert _live_metric(report, "placement_mean_world_x_m")["baseline"] == pytest.approx(2.0)
    assert _live_metric(report, "placement_mean_world_x_m")["candidate"] == pytest.approx(1.0)
    assert _live_metric(report, "placement_mean_world_x_m")["delta"] == pytest.approx(-1.0)
    assert _live_metric(report, "placement_residual_to_existing_mean_dx_m")["delta"] == pytest.approx(-1.0)
    assert _live_metric(report, "placement_residual_to_existing_p95_m")["candidate"] == pytest.approx(1.0)
    assert _live_metric(report, "placement_world_delta_p95_m")["candidate"] == pytest.approx(1.0)
    assert report["deferred_requires_pipeline"]["grounding.max_foot_lock_slide_m"]["current_artifact_value"] == pytest.approx(0.02)
    assert report["promotion_recommendation"]["never_auto_promotes"] is True


def test_identical_calibrations_report_zero_live_deltas(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path, candidate_tx_px=100.0)

    report = build_impact_report(
        baseline_calibration_path=paths["baseline"],
        candidate_calibration_path=paths["candidate"],
        tracks_path=paths["tracks"],
        placement_path=paths["placement"],
        clip="synthetic_clip",
        generated_at="2026-07-08T00:00:00Z",
    )

    for metric in report["live_metrics"].values():
        assert metric["delta"] == pytest.approx(0.0)


def test_cli_writes_reference_report(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path, candidate_tx_px=110.0)
    out_path = tmp_path / "impact_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            COMMAND_PATH,
            "--baseline-calibration",
            str(paths["baseline"]),
            "--candidate-calibration",
            str(paths["candidate"]),
            "--tracks",
            str(paths["tracks"]),
            "--placement",
            str(paths["placement"]),
            "--body-grounding-quality",
            str(paths["grounding"]),
            "--clip",
            "synthetic_clip",
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_report = json.loads(completed.stdout)
    file_report = json.loads(out_path.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert stdout_report["artifact_type"] == "racketsport_court_calibration_impact_report"
    assert stdout_report["live_metrics"]["placement_mean_world_x_m"]["delta"] == pytest.approx(-1.0)
    assert "CALV1 IMPACT 2026-07-08" in stdout_report["build_checklist_bullet"]
