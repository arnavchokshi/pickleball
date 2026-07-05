from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, Mapping
from pathlib import Path

from scripts.racketsport import run_ball_chain


def test_run_ball_chain_refuses_heldout_clip_without_authorization(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_ball_chain.py",
            "--clip",
            "outdoor_webcam_iynbd_1500_long_high_baseline",
            "--fused-track",
            str(tmp_path / "fused.json"),
            "--court-calibration",
            str(tmp_path / "court.json"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--heldout-authorized" in completed.stderr


def test_run_ball_chain_degrades_product_view_when_solver_b_killed(tmp_path: Path, monkeypatch: Any) -> None:
    fused_frames = [
        {
            "t": 0.0,
            "xy": [101.0, 201.0],
            "conf": 0.91,
            "visible": True,
            "world_xyz": None,
            "spin_rpm": None,
            "speed_mps": None,
            "approx": False,
        },
        {
            "t": 1.0 / 30.0,
            "xy": [111.0, 211.0],
            "conf": 0.72,
            "visible": True,
            "world_xyz": None,
            "spin_rpm": None,
            "speed_mps": None,
            "approx": False,
        },
    ]
    fused_path = _write_json(
        tmp_path / "fused.json",
        {"schema_version": 1, "fps": 30.0, "source": "fused", "frames": fused_frames, "bounces": []},
    )
    court_path = _write_json(
        tmp_path / "court.json",
        {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": [[100.0, 0.0, 960.0], [0.0, 100.0, 540.0], [0.0, 0.0, 1.0]],
            "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "test"},
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 12.0],
                "camera_height_m": 12.0,
            },
            "image_size": [1920, 1080],
        },
    )
    kill_reasons = ["physical_sanity_violation_fraction 1.000000 exceeds 0.200000"]

    def fake_write_bounces(**kwargs: Any) -> dict[str, Any]:
        out_path = kwargs["out_path"]
        payload = {"schema_version": 1, "summary": {"final_candidate_count": 0}, "candidates": []}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def fake_run_solver(*, config: Any, **_kwargs: Any) -> dict[str, Any]:
        status = "experimental_off" if config.candidate_association_mode == "rescue_only" else "ran"
        return {
            "status": status,
            "kill_reasons": kill_reasons if status != "ran" else [],
            "frames": [
                {"t": 0.0, "band": "anchored_measured", "world_xyz": [0.0, 0.0, 1.0]},
                {"t": 1.0 / 30.0, "band": "arc_interpolated", "world_xyz": [0.2, 0.0, 1.0]},
            ],
        }

    monkeypatch.setattr(run_ball_chain, "write_bounce_candidate_payload", fake_write_bounces)
    monkeypatch.setattr(run_ball_chain, "_run_solver", fake_run_solver)

    manifest = run_ball_chain.run_chain(
        argparse.Namespace(
            clip="unit_solver_killed",
            fused_track=fused_path,
            court_calibration=court_path,
            out_dir=tmp_path / "out",
            ball_candidates=[],
            candidate_extra_track=[],
            extra_anchors_from_arc=None,
            fusion_decisions=None,
            ball_type="no_drag_test",
            max_candidates_per_frame=12,
            candidate_selection_max_iterations=5,
            rescue_tracknet_floor=0.5,
            rescue_wasb_floor=0.0,
            veto_px=1.0,
            veto_weak_support_required=False,
            heldout_authorized=False,
        )
    )

    product_veto_path = Path(manifest["outputs"]["product_veto"]["path"])
    product_veto = json.loads(product_veto_path.read_text(encoding="utf-8"))
    product_report = json.loads((tmp_path / "out" / "product_view" / "arc_measured_fallback_fused_veto_report.json").read_text(encoding="utf-8"))
    assert product_veto["product_view_mode"] == "fused_only_solver_killed"
    assert product_veto["frames"] == fused_frames
    assert product_report["veto"]["enabled"] is False
    assert manifest["product_view_mode"] == "fused_only_solver_killed"
    assert manifest["killed"] is True
    assert manifest["solver_b_killed"] is True
    assert manifest["solver_b_kill_reasons"] == kill_reasons
    assert manifest["summary"]["product_veto_dropped_count"] == 0


def test_run_ball_chain_demotes_flight_sanity_violations_in_solver_and_product_outputs(tmp_path: Path, monkeypatch: Any) -> None:
    fused_frames = [
        {
            "t": frame / 30.0,
            "xy": [100.0 + frame, 200.0 + frame],
            "conf": 0.91,
            "visible": True,
            "world_xyz": None,
            "spin_rpm": None,
            "speed_mps": None,
            "approx": False,
        }
        for frame in range(31)
    ]
    fused_path = _write_json(
        tmp_path / "fused.json",
        {"schema_version": 1, "fps": 30.0, "source": "fused", "frames": fused_frames, "bounces": []},
    )
    court_path = _write_json(
        tmp_path / "court.json",
        {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": [[100.0, 0.0, 960.0], [0.0, 100.0, 540.0], [0.0, 0.0, 1.0]],
            "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "test"},
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 12.0],
                "camera_height_m": 12.0,
            },
            "image_size": [1920, 1080],
        },
    )

    def fake_write_bounces(**kwargs: Any) -> dict[str, Any]:
        out_path = kwargs["out_path"]
        payload = {"schema_version": 1, "summary": {"final_candidate_count": 0}, "candidates": []}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def fake_run_solver(*, clip: str, config: Any, **_kwargs: Any) -> dict[str, Any]:
        frames = []
        for frame in range(31):
            t = frame / 30.0
            frames.append(
                {
                    "t": t,
                    "visible": True,
                    "world_xyz": [5.0 * t, 0.0, 1.5 + 0.3 * __import__("math").sin(4.0 * __import__("math").pi * t)],
                    "band": "anchored_measured",
                }
            )
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_track_arc_solved",
            "clip_id": clip,
            "status": "ran",
            "summary": {"anchored_measured_count": 31, "arc_weak_count": 0},
            "anchors": [
                {"anchor_id": "a0", "kind": "bounce", "t": 0.0, "frame": 0, "world_xyz": frames[0]["world_xyz"], "sigma_m": 0.05, "status": "human_reviewed"},
                {"anchor_id": "a1", "kind": "bounce", "t": 1.0, "frame": 30, "world_xyz": frames[-1]["world_xyz"], "sigma_m": 0.05, "status": "human_reviewed"},
            ],
            "frames": frames,
            "event_selection": {"selected": [], "selected_count": 0},
        }

    monkeypatch.setattr(run_ball_chain, "write_bounce_candidate_payload", fake_write_bounces)
    monkeypatch.setattr(run_ball_chain, "_run_solver", fake_run_solver)

    manifest = run_ball_chain.run_chain(
        argparse.Namespace(
            clip="unit_flight_sanity",
            fused_track=fused_path,
            court_calibration=court_path,
            out_dir=tmp_path / "out",
            ball_candidates=[],
            candidate_extra_track=[],
            extra_anchors_from_arc=None,
            fusion_decisions=None,
            ball_type="no_drag_test",
            max_candidates_per_frame=12,
            candidate_selection_max_iterations=5,
            rescue_tracknet_floor=0.5,
            rescue_wasb_floor=0.0,
            veto_px=1.0,
            veto_weak_support_required=False,
            heldout_authorized=False,
        )
    )

    solver_b_path = Path(manifest["outputs"]["solver_b_ball_track_arc_solved"]["path"])
    solver_b = json.loads(solver_b_path.read_text(encoding="utf-8"))
    product_veto = json.loads(Path(manifest["outputs"]["product_veto"]["path"]).read_text(encoding="utf-8"))

    demoted_solver_frames = [frame for frame in solver_b["frames"] if frame.get("flight_sanity_demoted") is True]
    demoted_product_frames = [frame for frame in product_veto["frames"] if frame.get("flight_sanity_demoted") is True]
    assert demoted_solver_frames
    assert demoted_product_frames
    assert all(frame["band"] == "arc_weak" for frame in demoted_solver_frames)
    assert all(frame["band"] == "arc_weak" for frame in demoted_product_frames)
    assert manifest["summary"]["flight_sanity"]["solver_b"]["failed_segment_count"] == 1
    assert manifest["summary"]["flight_sanity"]["solver_b"]["demoted_frame_count"] == len(demoted_solver_frames)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
