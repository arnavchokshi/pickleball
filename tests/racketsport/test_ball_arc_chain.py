from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from threed.racketsport import ball_arc_chain
from threed.racketsport.ball_arc_chain import BallArcSolverRun


ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROW22_MANIFEST = (
    ROOT
    / "runs/lanes/ball_heldout_chain_run_20260704/chain_runs/"
    / "outdoor_webcam_iynbd_1500_long_high_baseline/ball_chain_manifest.json"
)


def test_default_chain_config_matches_frozen_row22_manifest() -> None:
    frozen_configs = json.loads(FROZEN_ROW22_MANIFEST.read_text(encoding="utf-8"))["configs"]

    assert ball_arc_chain.default_ball_chain_configs() == frozen_configs


def test_default_chain_consumes_candidate_sidecars_and_records_manifest(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _ball_track_payload())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _calibration_payload())
    sidecar_path = _write_json(tmp_path / "wasb_candidates.json", _ball_candidates_payload(source="wasb"))
    captured: dict[str, Any] = {}
    _patch_default_chain_solver(monkeypatch, captured)

    result = ball_arc_chain.run_default_ball_arc_chain(
        clip="unit_clip",
        ball_track_path=ball_track_path,
        court_calibration_path=calibration_path,
        ball_candidate_paths=[sidecar_path],
        out_dir=tmp_path / "out",
        generated_at="2026-07-05T00:00:00Z",
    )

    assert len(captured["ball_candidate_sidecars"]) == 1
    assert captured["ball_candidate_sidecars"][0]["source"] == "wasb"
    assert captured["config"].enable_event_discovery is False
    assert captured["config"].enable_event_subset_selection is False
    assert captured["config"].candidate_association_mode == "free"
    assert captured["config"].max_candidates_per_frame == 12
    assert "chain_config_degraded" not in json.loads(Path(result["outputs"]["ball_track_arc_solved"]).read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out" / "ball_chain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["configs"] == ball_arc_chain.default_ball_chain_configs()
    assert manifest["inputs"]["ball_candidates_0"]["path"] == str(sidecar_path)
    assert "chain_config_degraded" not in manifest


def test_default_chain_records_degraded_marker_without_candidate_sidecars(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _ball_track_payload())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _calibration_payload())
    captured: dict[str, Any] = {}
    _patch_default_chain_solver(monkeypatch, captured)

    result = ball_arc_chain.run_default_ball_arc_chain(
        clip="unit_clip",
        ball_track_path=ball_track_path,
        court_calibration_path=calibration_path,
        out_dir=tmp_path / "out",
        generated_at="2026-07-05T00:00:00Z",
    )

    assert captured["ball_candidate_sidecars"] == []
    artifact = json.loads(Path(result["outputs"]["ball_track_arc_solved"]).read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out" / "ball_chain_manifest.json").read_text(encoding="utf-8"))
    assert artifact["chain_config_degraded"] == "no_candidate_sidecars"
    assert manifest["chain_config_degraded"] == "no_candidate_sidecars"
    assert result["summary"]["chain_config_degraded"] == "no_candidate_sidecars"


def test_default_chain_uses_event_sidecars_only_for_seed_anchor_prepass(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _ball_track_payload())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _calibration_payload())
    sidecar_path = _write_json(tmp_path / "wasb_candidates.json", _ball_candidates_payload(source="wasb"))
    contact_windows_path = _write_json(
        tmp_path / "contact_windows.json",
        {"schema_version": 1, "events": [{"type": "contact", "frame": 3, "t": 0.1, "confidence": 0.8}]},
    )
    skeleton_path = _write_json(tmp_path / "skeleton3d.json", {"schema_version": 1, "players": []})
    calls: list[dict[str, Any]] = []

    def _fake_write_bounces(**kwargs: Any) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_bounce_candidates_track_geometry",
            "summary": {"final_candidate_count": 1},
            "candidates": [],
        }
        _write_json(kwargs["out_path"], payload)
        return payload

    def _fake_solve(**kwargs: Any) -> BallArcSolverRun:
        calls.append(dict(kwargs))
        out_dir = Path(kwargs["out_dir"])
        is_seed = out_dir.name == "ball_arc_seed"
        artifact = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_track_arc_solved",
            "status": "ran",
            "summary": {"coverage_world_xyz_count": 3, "segment_count": 1},
            "frames": [],
            "anchors": [
                {
                    "anchor_id": "seed_contact_3",
                    "kind": "contact",
                    "t": 0.1,
                    "frame": 3,
                    "world_xyz": [0.0, 0.0, 0.5],
                    "sigma_m": 0.35,
                    "status": "contact_candidate" if is_seed else "extra_seed_anchor",
                    "source": "seed_prepass",
                }
            ]
            if is_seed
            else [],
        }
        flight_sanity = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_flight_sanity",
            "summary": {
                "segment_count": 1,
                "passed_segment_count": 1,
                "failed_segment_count": 0,
                "skipped_segment_count": 0,
                "demoted_frame_count": 0,
            },
        }
        artifact_path = _write_json(out_dir / "ball_track_arc_solved.json", artifact)
        flight_path = _write_json(out_dir / "ball_flight_sanity.json", flight_sanity)
        return BallArcSolverRun(
            artifact=artifact,
            flight_sanity=flight_sanity,
            artifact_path=artifact_path,
            flight_sanity_path=flight_path,
            events_selected_path=None,
        )

    monkeypatch.setattr(ball_arc_chain, "write_bounce_candidate_payload", _fake_write_bounces)
    monkeypatch.setattr(ball_arc_chain, "solve_arc_with_flight_sanity", _fake_solve)

    result = ball_arc_chain.run_default_ball_arc_chain(
        clip="unit_clip",
        ball_track_path=ball_track_path,
        court_calibration_path=calibration_path,
        ball_candidate_paths=[sidecar_path],
        contact_windows_path=contact_windows_path,
        skeleton3d_path=skeleton_path,
        out_dir=tmp_path / "out",
        generated_at="2026-07-05T00:00:00Z",
    )

    assert len(calls) == 2
    seed_call, final_call = calls
    assert seed_call["contact_windows"] == {"schema_version": 1, "events": [{"type": "contact", "frame": 3, "t": 0.1, "confidence": 0.8}]}
    assert final_call["contact_windows"] is None
    assert final_call["skeleton3d"] is None
    assert final_call["auto_bounce_candidates"] is None
    assert len(final_call["extra_anchors"]) == 1
    assert final_call["extra_anchors"][0].anchor_id == "seed_contact_3"
    manifest = json.loads((tmp_path / "out" / "ball_chain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy"]["seed_anchor_prepass"] is True
    assert "seed_anchor_ball_track_arc_solved" in manifest["outputs"]
    assert result["summary"]["seed_anchor_count"] == 1


def test_default_chain_main_solve_consumes_valid_net_plane_and_records_provenance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _ball_track_payload())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _calibration_payload())
    net_plane_path = _write_json(tmp_path / "net_plane.json", _net_plane_payload())
    captured: dict[str, Any] = {}
    _patch_default_chain_solver(monkeypatch, captured)

    result = ball_arc_chain.run_default_ball_arc_chain(
        clip="unit_clip",
        ball_track_path=ball_track_path,
        court_calibration_path=calibration_path,
        net_plane_path=net_plane_path,
        out_dir=tmp_path / "out",
        generated_at="2026-07-05T00:00:00Z",
    )

    expected_provenance = {"consumed_net_plane": True, "reason": "consumed"}
    assert captured["net_plane"] == _net_plane_payload()
    artifact = json.loads(Path(result["outputs"]["ball_track_arc_solved"]).read_text(encoding="utf-8"))
    assert artifact["net_plane_provenance"] == expected_provenance
    manifest = json.loads((tmp_path / "out" / "ball_chain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["net_plane_provenance"] == expected_provenance
    assert manifest["summary"]["net_plane_provenance"] == expected_provenance
    assert manifest["inputs"]["net_plane"]["path"] == str(net_plane_path)
    assert result["summary"]["net_plane_provenance"] == expected_provenance


def test_default_chain_main_solve_omits_absent_net_plane_with_recorded_reason(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _ball_track_payload())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _calibration_payload())
    captured: dict[str, Any] = {}
    _patch_default_chain_solver(monkeypatch, captured)

    result = ball_arc_chain.run_default_ball_arc_chain(
        clip="unit_clip",
        ball_track_path=ball_track_path,
        court_calibration_path=calibration_path,
        out_dir=tmp_path / "out",
        generated_at="2026-07-05T00:00:00Z",
    )

    expected_provenance = {"consumed_net_plane": False, "reason": "absent"}
    assert captured["net_plane"] is None
    artifact = json.loads(Path(result["outputs"]["ball_track_arc_solved"]).read_text(encoding="utf-8"))
    assert artifact["net_plane_provenance"] == expected_provenance
    manifest = json.loads((tmp_path / "out" / "ball_chain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["net_plane_provenance"] == expected_provenance
    assert "net_plane" not in manifest["inputs"]
    assert result["summary"]["net_plane_provenance"] == expected_provenance


def test_default_chain_main_solve_ignores_malformed_net_plane_without_crashing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _ball_track_payload())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _calibration_payload())
    net_plane_path = _write_json(
        tmp_path / "net_plane.json",
        {"schema_version": 1, "plane": {"point": [0.0, 0.0, 0.0]}},  # missing normal/endpoints/heights
    )
    captured: dict[str, Any] = {}
    _patch_default_chain_solver(monkeypatch, captured)

    result = ball_arc_chain.run_default_ball_arc_chain(
        clip="unit_clip",
        ball_track_path=ball_track_path,
        court_calibration_path=calibration_path,
        net_plane_path=net_plane_path,
        out_dir=tmp_path / "out",
        generated_at="2026-07-05T00:00:00Z",
    )

    assert captured["net_plane"] is None
    artifact = json.loads(Path(result["outputs"]["ball_track_arc_solved"]).read_text(encoding="utf-8"))
    provenance = artifact["net_plane_provenance"]
    assert provenance["consumed_net_plane"] is False
    assert provenance["reason"].startswith("invalid_schema:")
    manifest = json.loads((tmp_path / "out" / "ball_chain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["net_plane_provenance"] == provenance
    # the rejected file's path/hash is still recorded for audit even though it was not consumed.
    assert manifest["inputs"]["net_plane"]["path"] == str(net_plane_path)
    assert result["summary"]["net_plane_provenance"] == provenance


def test_load_net_plane_for_default_solve_absent_when_no_path() -> None:
    net_plane, consumed, reason = ball_arc_chain.load_net_plane_for_default_solve(None)
    assert net_plane is None
    assert consumed is False
    assert reason == "absent"


def test_load_net_plane_for_default_solve_absent_when_file_missing(tmp_path: Path) -> None:
    net_plane, consumed, reason = ball_arc_chain.load_net_plane_for_default_solve(tmp_path / "missing.json")
    assert net_plane is None
    assert consumed is False
    assert reason == "absent"


def test_load_net_plane_for_default_solve_accepts_valid_schema(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "net_plane.json", _net_plane_payload())

    net_plane, consumed, reason = ball_arc_chain.load_net_plane_for_default_solve(path)

    assert net_plane == _net_plane_payload()
    assert consumed is True
    assert reason == "consumed"


def test_load_net_plane_for_default_solve_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "net_plane.json"
    path.write_text("{not valid json", encoding="utf-8")

    net_plane, consumed, reason = ball_arc_chain.load_net_plane_for_default_solve(path)

    assert net_plane is None
    assert consumed is False
    assert reason.startswith("invalid_json:")


def test_load_net_plane_for_default_solve_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "net_plane.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    net_plane, consumed, reason = ball_arc_chain.load_net_plane_for_default_solve(path)

    assert net_plane is None
    assert consumed is False
    assert reason == "invalid_type:not_object"


def test_load_net_plane_for_default_solve_rejects_schema_violation(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "net_plane.json",
        {"schema_version": 1, "plane": {"point": [0.0, 0.0, 0.0]}},
    )

    net_plane, consumed, reason = ball_arc_chain.load_net_plane_for_default_solve(path)

    assert net_plane is None
    assert consumed is False
    assert reason.startswith("invalid_schema:")


def _patch_default_chain_solver(monkeypatch: Any, captured: dict[str, Any]) -> None:
    def _fake_write_bounces(**kwargs: Any) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_bounce_candidates_track_geometry",
            "summary": {"final_candidate_count": 1},
            "candidates": [],
        }
        _write_json(kwargs["out_path"], payload)
        return payload

    def _fake_solve(**kwargs: Any) -> BallArcSolverRun:
        captured.update(kwargs)
        out_dir = Path(kwargs["out_dir"])
        artifact = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_track_arc_solved",
            "status": "ran",
            "summary": {"coverage_world_xyz_count": 3, "segment_count": 1},
            "frames": [],
        }
        flight_sanity = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_flight_sanity",
            "summary": {
                "segment_count": 1,
                "passed_segment_count": 1,
                "failed_segment_count": 0,
                "skipped_segment_count": 0,
                "demoted_frame_count": 0,
            },
        }
        artifact_path = _write_json(out_dir / "ball_track_arc_solved.json", artifact)
        flight_path = _write_json(out_dir / "ball_flight_sanity.json", flight_sanity)
        return BallArcSolverRun(
            artifact=artifact,
            flight_sanity=flight_sanity,
            artifact_path=artifact_path,
            flight_sanity_path=flight_path,
            events_selected_path=None,
        )

    monkeypatch.setattr(ball_arc_chain, "write_bounce_candidate_payload", _fake_write_bounces)
    monkeypatch.setattr(ball_arc_chain, "solve_arc_with_flight_sanity", _fake_solve)


def _ball_track_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "source": "wasb",
        "frames": [
            {"t": frame / 30.0, "xy": [400.0 + frame, 300.0], "conf": 0.9, "visible": True, "approx": False}
            for frame in range(6)
        ],
        "bounces": [],
    }


def _calibration_payload() -> dict[str, Any]:
    return {
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
    }


def _net_plane_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
        "endpoints": [[-3.3528, 0.0, 0.9144], [3.3528, 0.0, 0.9144]],
        "center_height_in": 34.0,
        "post_height_in": 36.0,
    }


def _ball_candidates_payload(*, source: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_candidates",
        "source": source,
        "source_mode": f"{source}_test",
        "fps": 30.0,
        "primary_output": "ball_track.json",
        "max_candidates_per_frame": 5,
        "not_ground_truth": True,
        "candidate_prediction": True,
        "frames": [
            {
                "frame": 0,
                "candidates": [
                    {"xy": [400.0, 300.0], "score": 0.95, "source_detector": f"{source}_candidate"},
                ],
            }
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
