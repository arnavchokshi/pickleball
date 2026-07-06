from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pytest

from threed.racketsport import ball_arc_chain
from threed.racketsport.ball_arc_chain import BallArcSolverRun
from threed.racketsport.schemas import BallArcRender, validate_artifact_file


ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROW22_MANIFEST = (
    ROOT
    / "runs/lanes/ball_heldout_chain_run_20260704/chain_runs/"
    / "outdoor_webcam_iynbd_1500_long_high_baseline/ball_chain_manifest.json"
)


def test_default_chain_config_matches_frozen_row22_manifest() -> None:
    frozen_configs = json.loads(FROZEN_ROW22_MANIFEST.read_text(encoding="utf-8"))["configs"]
    frozen_configs["solver_a_free"]["enable_event_discovery"] = True
    frozen_configs["solver_a_free"]["enable_event_subset_selection"] = True

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
    assert captured["config"].enable_event_discovery is True
    assert captured["config"].enable_event_subset_selection is True
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


def test_default_chain_writes_events_selected_from_final_arc_solution(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _ball_track_payload())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _calibration_payload())

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
        out_dir = Path(kwargs["out_dir"])
        artifact = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_track_arc_solved",
            "clip_id": "unit_clip",
            "status": "ran",
            "summary": {"coverage_world_xyz_count": 3, "segment_count": 1},
            "event_selection": _events_selected_payload(),
            "frames": [],
        }
        flight_sanity = _flight_sanity_payload(verdicts={0: "pass"})
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
        out_dir=tmp_path / "out",
        generated_at="2026-07-05T00:00:00Z",
    )

    events_path = Path(result["outputs"]["events_selected"])
    assert events_path.name == "events_selected.json"
    assert events_path.is_file()
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_ball_arc_events_selected"
    assert [event["kind"] for event in payload["selected"]] == ["contact", "bounce"]
    assert payload["provenance"] == {
        "derived_from": "ball_track_arc_solved.json",
        "writer": "run_default_ball_arc_chain",
        "solver_status": "ran",
        "clip_id": "unit_clip",
    }
    manifest = json.loads((tmp_path / "out" / "ball_chain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["events_selected"]["path"] == str(events_path)


def test_default_chain_omits_events_selected_when_solver_self_kills(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _ball_track_payload())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _calibration_payload())

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
        out_dir = Path(kwargs["out_dir"])
        artifact = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_track_arc_solved",
            "clip_id": "unit_clip",
            "status": "experimental_off",
            "kill_reasons": ["self_kill_for_test"],
            "summary": {"coverage_world_xyz_count": 0, "segment_count": 0},
            "event_selection": _events_selected_payload(),
            "frames": [],
        }
        flight_sanity = _flight_sanity_payload(verdicts={})
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
        out_dir=tmp_path / "out",
        generated_at="2026-07-05T00:00:00Z",
    )

    assert "events_selected" not in result["outputs"]
    assert not (tmp_path / "out" / "events_selected.json").exists()
    manifest = json.loads((tmp_path / "out" / "ball_chain_manifest.json").read_text(encoding="utf-8"))
    assert "events_selected" not in manifest["outputs"]


def test_ball_arc_render_builds_dense_parametric_samples_and_pbvision_summary() -> None:
    artifact = _arc_solved_render_payload()
    flight_sanity = _flight_sanity_payload(verdicts={0: "pass"})

    render = ball_arc_chain.build_ball_arc_render_artifact(
        artifact,
        flight_sanity=flight_sanity,
        generated_at="2026-07-05T00:00:00Z",
    )

    assert render["artifact_type"] == "racketsport_ball_arc_render"
    assert render["render_only"] is True
    assert render["not_for_detection_metrics"] is True
    assert render["trusted_for_ball_detection_metrics"] is False
    samples = [sample for sample in render["samples"] if sample["segment_id"] == 0 and sample["bridge"] is False]
    assert len(samples) >= 13
    assert samples[0]["t"] == pytest.approx(0.0)
    assert samples[1]["t"] - samples[0]["t"] == pytest.approx(0.025)
    assert samples[-1]["t"] == pytest.approx(0.3)
    assert all(sample["render_only"] is True for sample in samples)
    assert all(sample["not_for_detection_metrics"] is True for sample in samples)
    assert samples[4]["world_xyz"] == pytest.approx([0.2, 0.0, 0.850967], abs=1e-6)

    segment = render["segments"][0]
    assert segment["anchor_types"] == ["contact", "bounce"]
    assert segment["anchor_frames"] == [0, 3]
    assert segment["flight_sanity_verdict"] == "pass"
    assert 0.5 < segment["confidence"] <= 1.0
    shot = segment["shot"]
    assert shot["start"]["world_xyz"] == pytest.approx([0.0, 0.0, 0.5], abs=1e-6)
    assert shot["start"]["court_xy"] == pytest.approx([0.0, 0.0])
    assert shot["end"]["court_xy"] == pytest.approx([0.6, 0.0])
    assert shot["peak"]["world_xyz"][2] > shot["start"]["world_xyz"][2]
    assert shot["speed_mps"] == pytest.approx(math.sqrt(2.0**2 + 4.0**2), abs=1e-6)
    assert shot["speed_mph"] == pytest.approx(shot["speed_mps"] * 2.2369362920544, abs=1e-6)
    assert shot["height_over_net_m"] == pytest.approx(0.18)
    assert shot["distance_m"] == pytest.approx(0.6)


def test_ball_arc_render_bridges_rally_span_gaps_with_low_confidence_samples() -> None:
    artifact = _arc_solved_render_payload(two_segments=True)
    flight_sanity = _flight_sanity_payload(verdicts={0: "pass", 1: "pass"})

    render = ball_arc_chain.build_ball_arc_render_artifact(
        artifact,
        flight_sanity=flight_sanity,
        rally_spans={"schema_version": 1, "spans": [{"t0": 0.0, "t1": 0.8}]},
        generated_at="2026-07-05T00:00:00Z",
    )

    bridge_samples = [sample for sample in render["samples"] if sample["bridge"] is True]
    assert bridge_samples
    assert any(sample["t"] == pytest.approx(0.4) for sample in bridge_samples)
    assert all(sample["confidence"] <= 0.25 for sample in bridge_samples)
    assert all(sample["band"] == "arc_weak" for sample in bridge_samples)
    assert all(sample["world_xyz"] is not None for sample in bridge_samples)
    assert render["summary"]["bridge_sample_count"] == len(bridge_samples)
    assert render["summary"]["rally_span_count"] == 1


def test_ball_arc_render_uses_bvp_fallback_segment_and_dense_samples_stay_in_bounds() -> None:
    artifact = _arc_solved_render_payload()
    segment = artifact["segments"][0]
    segment["status"] = "fit_bvp_fallback"
    segment["initial_position_m"] = [0.0, -7.0, 0.0371]
    segment["initial_velocity_mps"] = [0.4, 2.0, 1.5]
    segment["endpoint_error_m"] = 0.001
    segment["diagnostics"] = {
        "fit_validity_gate": {
            "reason": "zero_inliers",
            "original_status": "fit",
            "original_endpoint_error_m": 18.46,
        }
    }
    flight_sanity = _flight_sanity_payload(verdicts={0: "pass"})

    render = ball_arc_chain.build_ball_arc_render_artifact(
        artifact,
        flight_sanity=flight_sanity,
        generated_at="2026-07-05T00:00:00Z",
    )

    samples = [sample for sample in render["samples"] if sample["segment_id"] == 0]
    assert samples
    assert samples[0]["t"] > 0.0
    assert render["segments"][0]["fit_status"] == "fit_bvp_fallback"
    assert render["segments"][0]["confidence"] < 0.38
    assert all(sample["band"] == "arc_weak" for sample in samples)
    assert all(-3.048 <= sample["world_xyz"][0] <= 3.048 for sample in samples)
    assert all(-6.7056 <= sample["world_xyz"][1] <= 6.7056 for sample in samples)
    assert all(sample["world_xyz"][2] >= -0.15 for sample in samples)


def test_default_chain_writes_ball_arc_render_artifact_and_registers_schema(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ball_track_path = _write_json(tmp_path / "ball_track.json", _ball_track_payload())
    calibration_path = _write_json(tmp_path / "court_calibration.json", _calibration_payload())

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
        out_dir = Path(kwargs["out_dir"])
        artifact = _arc_solved_render_payload()
        flight_sanity = _flight_sanity_payload(verdicts={0: "pass"})
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
        out_dir=tmp_path / "out",
        generated_at="2026-07-05T00:00:00Z",
    )

    render_path = Path(result["outputs"]["ball_arc_render"])
    assert render_path.name == "ball_arc_render.json"
    assert render_path.is_file()
    render = json.loads(render_path.read_text(encoding="utf-8"))
    assert render["artifact_type"] == "racketsport_ball_arc_render"
    assert isinstance(BallArcRender.model_validate(render), BallArcRender)
    assert isinstance(validate_artifact_file("ball_arc_render", render_path), BallArcRender)
    manifest = json.loads((tmp_path / "out" / "ball_chain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["ball_arc_render"]["path"] == str(render_path)
    assert result["summary"]["ball_arc_render_sample_count"] == render["summary"]["sample_count"]


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
    assert seed_call["config"] == final_call["config"]
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


def _arc_solved_render_payload(*, two_segments: bool = False) -> dict[str, Any]:
    frames = [
        {"t": index / 10.0, "visible": True, "conf": 0.9, "world_xyz": None, "band": "hidden"}
        for index in range(9 if two_segments else 4)
    ]
    segments = [
        {
            "segment_id": 0,
            "status": "fit",
            "t0": 0.0,
            "t1": 0.3,
            "frame_start": 0,
            "frame_end": 3,
            "start_anchor": "contact_0",
            "end_anchor": "bounce_3",
            "anchors_used": [
                {
                    "anchor_id": "contact_0",
                    "kind": "contact",
                    "t": 0.0,
                    "frame": 0,
                    "world_xyz": [0.0, 0.0, 0.5],
                    "sigma_m": 0.1,
                    "status": "contact_prior",
                    "immovable": True,
                },
                {
                    "anchor_id": "bounce_3",
                    "kind": "bounce",
                    "t": 0.3,
                    "frame": 3,
                    "world_xyz": [0.6, 0.0, 1.259706],
                    "sigma_m": 0.1,
                    "status": "auto_bounce_candidate",
                    "immovable": True,
                },
            ],
            "initial_position_m": [0.0, 0.0, 0.5],
            "initial_velocity_mps": [2.0, 0.0, 4.0],
            "initial_speed_mps": math.sqrt(20.0),
            "inlier_count": 4,
            "outlier_count": 0,
            "inlier_frames": [0, 1, 2, 3],
            "outlier_frames": [],
            "reprojection_rmse_px": 2.0,
            "max_reprojection_error_px": 3.0,
            "endpoint_error_m": 0.02,
            "net_clearance_m": 0.18,
            "net_clearance_ok": True,
            "physical_sanity": {"initial_speed_mps": math.sqrt(20.0), "apex_height_m": 1.31558, "violation": False, "violations": []},
            "size_residuals_m": {"count": 0},
        }
    ]
    anchors = list(segments[0]["anchors_used"])
    if two_segments:
        second = {
            **segments[0],
            "segment_id": 1,
            "t0": 0.6,
            "t1": 0.8,
            "frame_start": 6,
            "frame_end": 8,
            "start_anchor": "contact_6",
            "end_anchor": "bounce_8",
            "anchors_used": [
                {
                    "anchor_id": "contact_6",
                    "kind": "contact",
                    "t": 0.6,
                    "frame": 6,
                    "world_xyz": [0.9, 0.0, 0.4],
                    "sigma_m": 0.1,
                    "status": "contact_prior",
                    "immovable": True,
                },
                {
                    "anchor_id": "bounce_8",
                    "kind": "bounce",
                    "t": 0.8,
                    "frame": 8,
                    "world_xyz": [1.3, 0.0, 0.603734],
                    "sigma_m": 0.1,
                    "status": "auto_bounce_candidate",
                    "immovable": True,
                },
            ],
            "initial_position_m": [0.9, 0.0, 0.4],
            "initial_velocity_mps": [2.0, 0.0, 3.0],
            "initial_speed_mps": math.sqrt(13.0),
            "inlier_count": 3,
            "inlier_frames": [6, 7, 8],
            "physical_sanity": {"initial_speed_mps": math.sqrt(13.0), "apex_height_m": 0.8589, "violation": False, "violations": []},
        }
        segments.append(second)
        anchors.extend(second["anchors_used"])
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_arc_solved",
        "clip_id": "unit_clip",
        "status": "ran",
        "kill_reasons": [],
        "source": "event_anchored_drag_arc_solver",
        "render_only": True,
        "not_for_detection_metrics": True,
        "trusted_for_ball_detection_metrics": False,
        "physics_parameters": {
            "ball_type": "no_drag_test",
            "gravity_mps2": 9.80665,
            "mass_kg": 0.0255,
            "diameter_m": 0.0742,
            "radius_m": 0.0371,
            "rho_air_kg_m3": 1.2,
            "drag_cd": 0.0,
            "drag_k_per_m": 0.0,
        },
        "config": {"integrator_max_step_s": 1 / 240.0},
        "anchors": anchors,
        "segments": segments,
        "frames": frames,
        "summary": {"coverage_world_xyz_count": 4, "segment_count": len(segments)},
    }


def _flight_sanity_payload(*, verdicts: Mapping[int, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_flight_sanity",
        "summary": {
            "segment_count": len(verdicts),
            "passed_segment_count": sum(1 for verdict in verdicts.values() if verdict == "pass"),
            "failed_segment_count": sum(1 for verdict in verdicts.values() if verdict == "fail"),
            "skipped_segment_count": sum(1 for verdict in verdicts.values() if verdict == "not_evaluated"),
            "demoted_frame_count": 0,
        },
        "segments": [
            {"segment_id": segment_id, "verdict": verdict, "reasons": []}
            for segment_id, verdict in sorted(verdicts.items())
        ],
        "frames": [],
    }


def _events_selected_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_arc_events_selected",
        "candidate_prediction": True,
        "not_ground_truth": True,
        "selection_policy": {"mode": "unit_test"},
        "selected_count": 2,
        "selected_optional_count": 1,
        "rejected_count": 0,
        "rejected_optional_count": 0,
        "selected": [
            {
                "anchor_id": "contact_0",
                "kind": "contact",
                "t": 0.0,
                "frame": 0,
                "player_id": 7,
                "world_xyz": [0.0, 0.0, 0.5],
                "selected": True,
                "selection": "selected_optional",
                "status": "candidate_prediction",
                "selection_reason": "unit_test_contact",
                "candidate_confidence": 0.8,
            },
            {
                "anchor_id": "bounce_3",
                "kind": "bounce",
                "t": 0.3,
                "frame": 3,
                "world_xyz": [0.6, 0.0, 0.04],
                "selected": True,
                "selection": "selected_passthrough",
                "status": "auto_bounce_candidate",
                "selection_reason": "unit_test_bounce",
                "candidate_confidence": 0.7,
            },
        ],
        "rejected": [],
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
