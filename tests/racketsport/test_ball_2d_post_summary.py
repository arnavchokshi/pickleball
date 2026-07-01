from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_2d_post_summary import build_ball_2d_postprocess_summary


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fusion_summary(*, radius_px: float = 60.0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_model_fusion",
        "status": "fused_not_gate_verified",
        "primary_ball_track": "runs/tracknet/ball_track.json",
        "stable_ball_track": "runs/local_trajectory/ball_track.json",
        "verifier_ball_tracks": ["runs/wasb/ball_track.json"],
        "outlier_distance_px": radius_px,
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }


def _court_summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_target_court_metric_filter",
        "status": "filtered_not_gate_verified",
        "source_ball_track": "runs/tracknet/ball_track.json",
        "sport": "pickleball",
        "court_margin_m": 0.5,
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }


def test_ball_2d_postprocess_summary_records_evidenced_and_missing_components(tmp_path: Path) -> None:
    fusion = tmp_path / "consensus_60px" / "summary.json"
    court = tmp_path / "tracknet_court_margin05m" / "summary.json"
    speed = tmp_path / "world_speed" / "summary.json"
    ransac = tmp_path / "ransac" / "summary.json"
    kalman = tmp_path / "kalman" / "summary.json"
    _write_json(fusion, _fusion_summary())
    _write_json(court, _court_summary())
    _write_json(
        speed,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_world_speed_gate",
            "status": "filtered_not_gate_verified",
            "max_world_speed_mps": 30.0,
            "coordinate_model": "court_plane_xy",
            "uses_human_clicks": False,
            "not_ground_truth": True,
        },
    )
    _write_json(
        ransac,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_ransac_arc_recovery",
            "status": "filtered_not_gate_verified",
            "max_residual_px": 5.0,
            "uses_human_clicks": False,
            "not_ground_truth": True,
        },
    )
    _write_json(
        kalman,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_kalman_rts_smoother",
            "status": "smoothed_not_gate_verified",
            "max_gap_fill_frames": 6,
            "jitter_px_std": 1.2,
            "uses_human_clicks": False,
            "not_ground_truth": True,
        },
    )

    summary = build_ball_2d_postprocess_summary(
        model_consensus_summary_paths=[fusion],
        court_gating_summary_path=court,
        max_speed_summary_path=speed,
        ransac_summary_path=ransac,
        kalman_rts_summary_path=kalman,
        primary_model="tracknet",
        verifier_model="wasb",
    )

    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert summary["artifact_type"] == "racketsport_ball_2d_postprocess_summary"
    assert summary["model_consensus"]["evidence_present"] is True
    assert summary["model_consensus"]["primary"] == "tracknet"
    assert summary["model_consensus"]["verifier"] == "wasb"
    assert summary["model_consensus"]["radius_px_1080p"] == pytest.approx(60.0)
    assert summary["court_gating"]["evidence_present"] is True
    assert summary["court_gating"]["margin_m"] == pytest.approx(0.5)
    assert summary["max_speed_gate"]["evidence_present"] is True
    assert summary["max_speed_gate"]["max_world_speed_mps"] == pytest.approx(30.0)
    assert summary["ransac"]["evidence_present"] is True
    assert summary["ransac"]["max_residual_px"] == pytest.approx(5.0)
    assert summary["local_search"]["evidence_present"] is False
    assert summary["kalman_rts"]["evidence_present"] is True
    assert summary["kalman_rts"]["max_gap_fill_frames"] == 6
    assert summary["kalman_rts"]["jitter_px_std"] == pytest.approx(1.2)
    assert summary["missing_components"] == ["local_search"]
    assert summary["not_ground_truth"] is True


def test_ball_2d_postprocess_summary_cli_writes_truthful_partial_summary(tmp_path: Path) -> None:
    fusion = tmp_path / "consensus_60px" / "summary.json"
    out = tmp_path / "m2_postprocess_summary.json"
    _write_json(fusion, _fusion_summary())

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_ball_2d_post_summary.py",
            "--model-consensus-summary",
            str(fusion),
            "--primary-model",
            "tracknet",
            "--verifier-model",
            "wasb",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert json.loads(completed.stdout)["model_consensus"]["evidence_present"] is True
    assert written["court_gating"]["evidence_present"] is False
    assert "court_gating" in written["missing_components"]


def test_ball_2d_postprocess_summary_does_not_count_pixel_search_as_heatmap_recovery(tmp_path: Path) -> None:
    local_search = tmp_path / "pixel_local_search" / "summary.json"
    _write_json(
        local_search,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_local_search_filter",
            "status": "TESTED-ON-REAL-DATA",
            "suppress_conf_threshold": 0.25,
            "source_video": "cvat_upload/example.mp4",
            "uses_human_clicks": False,
            "not_ground_truth": True,
        },
    )

    summary = build_ball_2d_postprocess_summary(local_search_summary_path=local_search)

    assert summary["local_search"]["evidence_present"] is False
    assert summary["local_search"]["recovery_heatmap_threshold"] is None
    assert summary["local_search"]["source_artifact_type"] == "racketsport_ball_local_search_filter"
    assert "local_search" in summary["missing_components"]
