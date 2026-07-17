from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from threed.racketsport.one_world_v1 import (
    BALL_RADIUS_M,
    OneWorldV1,
    build_one_world,
    canonical_json,
    interpolate_wrist,
    lift_camera_pose_to_world,
    resolve_two_hypothesis_sequence,
    soft_surface_refinement,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _joint_set(wrist: list[float], confidence: float = 0.95) -> tuple[list[list[float]], list[float]]:
    joints = [[0.0, 0.0, 1.0] for _ in range(17)]
    joints[9] = list(wrist)
    joints[10] = [wrist[0] + 0.02, wrist[1], wrist[2]]
    return joints, [confidence] * 17


def make_run(
    root: Path,
    *,
    arc_generation: str = "solved",
    ball_world: list[float] | None = None,
    player_wrists: dict[int, list[float]] | None = None,
    contact: bool = True,
    audio_onsets: list[dict] | None = None,
    bounce_xy: list[float] | None = None,
    approx: bool = False,
    observed_xy: list[float] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ball_world = ball_world or [0.0, 0.0, 1.0]
    player_wrists = player_wrists or {1: [0.0, 0.0, 1.0], 2: [2.0, 0.0, 1.0]}
    observed_xy = observed_xy or [100.0, 100.0]
    calibration = {
        "schema_version": 1,
        "coordinate_frame": "court_netcenter_z_up_m",
        "homography": [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 100.0, "cy": 100.0, "dist": [], "source": "synthetic"},
        "extrinsics": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.0, 0.0, 10.0], "camera_height_m": 2.0},
        "metric_confidence": "low",
    }
    _write(root / "court_calibration.json", calibration)
    _write(root / "trust_bands.json", {"court": {"stage": "CAL", "gate_id": "synthetic", "gate_status": "unverified", "badge": "preview", "reason": "synthetic", "evidence_path": None}})
    track_players = []
    placement_players = []
    body_players = []
    for player_id, wrist in sorted(player_wrists.items()):
        joints, confs = _joint_set(wrist)
        track_frame = {"frame_idx": 0, "t": 0.0, "world_xy": [float(player_id), 0.0], "conf": 0.95, "bbox": [90.0, 90.0, 110.0, 120.0]}
        track_players.append({"id": player_id, "frames": [track_frame]})
        placement_players.append({"id": player_id, "frames": [{"frame_idx": 0, "t": 0.0, "smoothed_world_xy": [float(player_id), 0.0], "fused_world_xy": [float(player_id), 0.0], "covariance_m2": [[0.04, 0.0], [0.0, 0.04]]}]})
        body_players.append({"id": player_id, "frames": [{"frame_idx": 0, "t": 0.0, "transl_world": [float(player_id) + 0.2, 0.0, 0.0], "joints_world": joints, "joint_conf": confs}]})
    _write(root / "tracks.json", {"schema_version": 1, "fps": 30.0, "players": track_players})
    _write(root / "placement.json", {"schema_version": 1, "fps": 30.0, "players": placement_players})
    _write(root / "smpl_motion.json", {"schema_version": 1, "fps": 30.0, "model": "sam3dbody_world_joints", "world_frame": "court_Z0", "players": body_players})
    bounce = []
    if bounce_xy is not None:
        bounce = [{"frame": 0, "t": 0.0, "world_xy": bounce_xy, "p_bounce": 0.9, "confidence": 0.9, "uncertainty_m": 0.12, "source": "synthetic", "render_only": True, "not_for_detection_metrics": True}]
    _write(root / "ball_track.json", {"schema_version": 1, "fps": 30.0, "source": "synthetic", "frames": [{"t": 0.0, "xy": observed_xy, "conf": 0.95, "visible": True, "world_xyz": None, "approx": approx}], "bounces": bounce})
    arc_frame = {"t": 0.0, "frame": 0, "world_xyz": ball_world, "conf": 0.95, "sigma_m": 0.20, "band": "preview", "approx": approx, "source": "synthetic", "render_only": True, "not_for_detection_metrics": True}
    if arc_generation == "render":
        _write(root / "ball_arc_render.json", {"schema_version": 1, "solver_status": "ran", "samples": [{**arc_frame, "frame_float": 0.0, "confidence": arc_frame["conf"]}], "segments": []})
    else:
        _write(root / "ball_track_arc_solved.json", {"schema_version": 1, "status": "ran", "render_only": True, "not_for_detection_metrics": True, "frames": [arc_frame], "anchors": [], "segments": []})
    events = []
    if contact:
        events.append({"type": "contact", "t": 0.0, "frame": 0, "player_id": 1, "confidence": 0.95, "sources": {"audio": 0.9 if audio_onsets else None, "wrist_vel": 0.9, "ball_inflection": 0.9, "human_review": None}, "window": {"t0": 0.0, "t1": 0.03, "importance": 0.95}, "trust_band_note": "preview"})
    _write(root / "contact_windows.json", {"schema_version": 1, "events": events})
    _write(root / "audio_onsets_v2.json", {"schema_version": 1, "frame_rate": 30.0, "status": "ran" if audio_onsets else "blocked", "not_gate_verified": True, "trusted_for_contact": False, "onsets": audio_onsets or []})
    _write(root / "court_zones.json", {"schema_version": 1, "zones": {"court": [[-3.0, -6.0], [3.0, -6.0], [3.0, 6.0], [-3.0, 6.0]]}})
    _write(root / "net_plane.json", {"schema_version": 1, "plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]}, "endpoints": [[-3.0, 0.0, 0.9], [3.0, 0.0, 0.9]], "center_height_in": 34.0, "post_height_in": 36.0})
    _write(root / "rally_spans.json", {"schema_version": 1, "spans": [{"t0": 0.0, "t1": 1.0 / 30.0, "sources": ["synthetic"]}], "not_ground_truth": True})
    _write(root / "virtual_world.json", {"schema_version": 1, "artifact_type": "racketsport_virtual_world", "world_frame": "court_Z0", "fps": 30.0, "players": [], "ball": {"frames": []}, "paddles": [], "summary": {}})
    return root


def test_both_ball_generations_and_determinism(tmp_path: Path) -> None:
    solved = build_one_world(make_run(tmp_path / "solved", arc_generation="solved"))
    render_dir = make_run(tmp_path / "render", arc_generation="render")
    render_a = build_one_world(render_dir)
    render_b = build_one_world(render_dir)
    assert solved.frames[0].ball is not None
    assert solved.frames[0].ball.source_generation == "ball_track_arc_solved"
    assert render_a.frames[0].ball is not None
    assert render_a.frames[0].ball.source_generation == "ball_arc_render"
    assert canonical_json(render_a) == canonical_json(render_b)
    assert OneWorldV1.model_validate_json(canonical_json(render_a)) == render_a


def test_same_run_refined_identity_fails_closed_to_placement(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run")
    wrong = copy.deepcopy(json.loads((run / "smpl_motion.json").read_text()))
    wrong["artifact_type"] = "placement_trajectory_refined"
    wrong["coordinate_space"] = "world_court_netcenter_z_up_m"
    wrong["preview_band"] = True
    wrong["VERIFIED"] = 0
    wrong["placement_trajectory_refinement"] = {"provenance": {"inputs": {"tracks": {"sha256": "bad"}}}}
    _write(run / "placement_trajectory_refined.json", wrong)
    output = build_one_world(run)
    assert set(output.summary.placement_tier_counts) == {"placement_fused"}
    assert "placement_trajectory_refined_identity_mismatch_fallback" in output.summary.warnings


def test_covariance_weighting_and_marker_discounts(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", approx=True, audio_onsets=[{"time_s": 0.0, "features": {"pop_band_ratio": 1.0}}])
    _write(run / "repair_summary.json", {"input_paths": {"tracks_path": str(run / "tracks.json"), "tracks_sha256": hashlib.sha256((run / "tracks.json").read_bytes()).hexdigest()}, "summary": {"confidence_repairs": [{"player_id": 1, "frame_index": 0, "conf_source": "interpolated_endpoint_min_capped_0_35", "repaired": True}]}})
    output = build_one_world(run)
    state = output.frames[0].players[0]
    assert 1.0 <= state.root_world[0] <= 1.15
    assert state.covariance_m2[0][0] < 0.04
    contact = output.contacts[0]
    discounts = contact.contact_evidence_vector.hypotheses[0].discounts
    assert "approx=0.25" in discounts
    assert "audio_review_only_not_gate_verified=0.20" in discounts
    assert contact.contact_evidence_vector.audio_bounded_multiplier <= 1.2215
    assert "player_repaired=0.25" in state.provenance.discounts


def test_no_snap_and_out_of_court_flag(tmp_path: Path) -> None:
    refined, before, after, _, _ = soft_surface_refinement([0.0, 0.0, 0.22], plane_point=[0.0, 0.0, 0.0], plane_normal=[0.0, 0.0, 1.0], ball_confidence=0.9, bounce_confidence=0.9, sigma_ball_m=0.1, sigma_cal_m=0.12, sigma_event_m=0.12, calibration_multiplier=0.3)
    assert before != 0.0 and after != 0.0
    assert refined[2] != BALL_RADIUS_M
    output = build_one_world(make_run(tmp_path / "run", ball_world=[9.0, 9.0, 1.0], bounce_xy=[9.0, 9.0]))
    assert output.bounces[0].out_of_court_bounds is True
    assert output.bounces[0].refined_ball_world is not None
    assert output.bounces[0].refined_ball_world[2] != BALL_RADIUS_M


def test_huge_outlier_and_colocation_discount_leave_raw_event_immutable(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", ball_world=[20.0, 20.0, 5.0])
    before = hashlib.sha256((run / "contact_windows.json").read_bytes()).hexdigest()
    output = build_one_world(run)
    after = hashlib.sha256((run / "contact_windows.json").read_bytes()).hexdigest()
    contact = output.contacts[0]
    assert before == after
    assert contact.hitter_band == "unsupported"
    assert contact.refined_ball_world is None
    assert "no_player_wrist_within_1.2m" in contact.provenance.degraded_reasons


def test_wrist_interpolation_guards() -> None:
    joints0, conf0 = _joint_set([0.0, 0.0, 1.0])
    joints2, conf2 = _joint_set([0.1, 0.0, 1.0])
    frames = {0: {"joints_world": joints0, "joint_conf": conf0}, 2: {"joints_world": joints2, "joint_conf": conf2}}
    result = interpolate_wrist(frames, 1, 9, 30.0)
    assert result is not None and "interpolated_latent_wrist" in result[3]
    low = copy.deepcopy(frames)
    low[2]["joint_conf"][9] = 0.49
    assert interpolate_wrist(low, 1, 9, 30.0) is None
    fast = copy.deepcopy(frames)
    fast[2]["joints_world"][9] = [10.0, 0.0, 1.0]
    assert interpolate_wrist(fast, 1, 9, 30.0) is None


def test_hitter_tie_is_too_close_to_call(tmp_path: Path) -> None:
    output = build_one_world(make_run(tmp_path / "run", player_wrists={1: [0.0, 0.0, 1.0], 2: [0.0, 0.0, 1.0]}))
    assert output.contacts[0].hitter_band == "too_close_to_call"
    assert output.contacts[0].hitter_id is None


def test_viterbi_resolution_tie_and_reprojection_never_chooses() -> None:
    frames = []
    for _ in range(3):
        frames.append({"hypotheses": [{"id": "primary", "wrist": 0.0, "contact": 0.0, "momentum": 0.0, "reprojection_error_px": 9999.0}, {"id": "alt", "wrist": 1.0, "contact": 1.0, "momentum": 1.0, "reprojection_error_px": 0.0}], "transition": {}})
    resolved = resolve_two_hypothesis_sequence(frames)
    assert resolved["status"] == "resolved"
    assert resolved["path"] == ["primary"] * 3
    tied = copy.deepcopy(frames)
    for frame in tied:
        frame["hypotheses"][1].update({"wrist": 0.0, "contact": 0.0, "momentum": 0.0})
    assert resolve_two_hypothesis_sequence(tied)["reason"] == "energy_tie"


def test_racket_generation_two_is_lifted_and_retained_without_reprojection_choice(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run")
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pose = {"pose_se3": {"R": identity, "t": [0.0, 0.0, 1100.0]}, "confidence": 0.9, "frame_conf": 0.9, "reprojection_error_px": 999.0, "source": "primary"}
    alt = {"pose_se3": {"R": identity, "t": [10.0, 0.0, 1100.0]}, "confidence": 0.8, "frame_conf": 0.8, "reprojection_error_px": 0.1, "source": "alt"}
    _write(run / "racket_pose.json", {"schema_version": 1, "fps": 30.0, "world_frame": "camera", "translation_unit": "cm", "players": []})
    _write(run / "racket_pose_hypotheses.json", {"schema_version": 1, "artifact_type": "racketsport_racket_pose_hypotheses", "fps": 30.0, "world_frame": "camera", "translation_unit": "cm", "players": [{"id": 1, "paddle_dims_in": {"length": 16.0, "width": 8.0}, "frames": [{"t": 0.0, "primary_pose": pose, "alt_pose": alt, "candidate_reprojection_errors_px": [999.0, 0.1], "ambiguity_margin_px": 0.01, "ambiguous": True}]}]})
    output = build_one_world(run)
    gen2 = next(state for state in output.frames[0].paddles if state.player_id == 1)
    assert gen2.status == "unresolved"
    assert gen2.pose_world is None
    assert gen2.display_pose_world is not None
    assert gen2.display_tier == "unresolved_best_evidence"
    assert len(gen2.retained_hypotheses) == 2
    assert "reprojection_carried_not_scored" in gen2.provenance.discounts
    assert gen2.trust_band.badge != "verified"


def test_legacy_paddle_proxy_is_carried_for_display(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run")
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    _write(run / "racket_pose_estimate.json", {"schema_version": 1, "fps": 30.0, "world_frame": "court_Z0", "translation_unit": "m", "render_only": True, "not_for_detection_metrics": True, "trust": "estimated_from_wrist", "players": [{"id": 1, "frames": [{"frame": 0, "t": 0.0, "pose_se3": {"R": identity, "t": [0.0, 0.0, 1.0]}, "conf": 0.5, "source": "wrist_proxy"}]}]})
    output = build_one_world(run)
    paddle = output.frames[0].paddles[0]
    assert paddle.status == paddle.display_tier == "unresolved_legacy_wrist_proxy"
    assert paddle.pose_world is None and paddle.display_pose_world is not None
    assert paddle.trust_band.badge != "verified"


def test_all_four_viewer_event_types_and_net_cross_no_pull(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", bounce_xy=[0.0, 0.0])
    contacts = json.loads((run / "contact_windows.json").read_text())
    contacts["events"].extend([
        {"type": "into_net", "t": 0.0, "frame": 0, "confidence": 0.8, "sources": {"ball_inflection": 0.8}},
        {"type": "net_cross", "t": 0.0, "frame": 0, "confidence": 0.7, "sources": {"ball_inflection": 0.7}},
    ])
    _write(run / "contact_windows.json", contacts)
    output = build_one_world(run)
    assert {event.type for event in output.events} == {"paddle_contact", "floor_bounce", "net_contact", "net_cross"}
    cross = next(event for event in output.events if event.type == "net_cross")
    assert cross.world_location_refined == cross.world_location_raw
    assert cross.trust_band.badge != "verified"


def test_ball_continuity_tiers_are_display_only_and_gap_bounded(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run")
    ball = json.loads((run / "ball_track.json").read_text())
    ball["frames"] = [{"t": i / 30.0, "xy": [100.0 + i, 100.0], "conf": 0.9, "visible": i != 19, "world_xyz": None, "approx": False} for i in range(21)]
    _write(run / "ball_track.json", ball)
    arc = json.loads((run / "ball_track_arc_solved.json").read_text())
    template = arc["frames"][0]
    arc["frames"] = [{**template, "frame": i, "t": i / 30.0, "world_xyz": [i / 10.0, 0.0, 1.0]} for i in (0, 10, 20)]
    _write(run / "ball_track_arc_solved.json", arc)
    output = build_one_world(run)
    by_frame = {frame.frame_idx: frame for frame in output.frames}
    assert by_frame[5].ball.estimate_tier == "physics_predicted"
    assert by_frame[5].ball.approx is True
    assert by_frame[5].ball.confidence_provenance.predictor == "bounded_ballistic_bridge"
    assert by_frame[5].ball.confidence_provenance.horizon_frames == 5
    assert by_frame[5].ball.trust_band.badge == "low_confidence"
    # A support gap larger than 0.5s refuses physics bridging and falls back to
    # the explicitly low-confidence ray tier when a 2D detection exists.
    arc["frames"] = [{**template, "frame": i, "t": i / 30.0, "world_xyz": [i / 10.0, 0.0, 1.0]} for i in (0, 20)]
    _write(run / "ball_track_arc_solved.json", arc)
    output_gap = build_one_world(run)
    gap_map = {frame.frame_idx: frame for frame in output_gap.frames}
    assert gap_map[10].ball.estimate_tier == "ray_court_projection"
    assert gap_map[10].ball.altitude_unknown is True
    assert "not_metric_eligible" in gap_map[10].ball.provenance.discounts
    assert gap_map[19].ball is None
    assert "ball:world_xyz" in gap_map[19].missing


def test_camera_frame_cm_lift_uses_typed_api() -> None:
    pose = {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [100.0, 200.0, 300.0]}
    cal = {"extrinsics": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.0, 0.0, 10.0]}}
    world = lift_camera_pose_to_world(pose, cal, input_unit="cm")
    assert world["t"] == pytest.approx([1.0, 2.0, -7.0])


def test_audio_only_cannot_confirm_and_neighbor_bleed_creates_nothing(tmp_path: Path) -> None:
    onset = [{"time_s": 0.0, "features": {"pop_band_ratio": 1.0}}]
    audio_only = build_one_world(make_run(tmp_path / "audio_only", ball_world=[10.0, 10.0, 3.0], audio_onsets=onset))
    contact = audio_only.contacts[0]
    assert contact.hitter_band == "unsupported"
    assert contact.refined_ball_world is None
    assert contact.confidence == 0.0
    no_event = build_one_world(make_run(tmp_path / "bleed", contact=False, audio_onsets=onset))
    assert no_event.contacts == []


def test_reprojection_regression_suppresses_contact_refinement(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", ball_world=[0.3, 0.0, 1.0], player_wrists={1: [0.0, 0.0, 1.0]}, observed_xy=[127.272727, 100.0])
    calibration = json.loads((run / "court_calibration.json").read_text())
    calibration["intrinsics"]["fx"] = 1000.0
    _write(run / "court_calibration.json", calibration)
    output = build_one_world(run)
    contact = output.contacts[0]
    assert contact.refined_ball_world is None
    assert contact.hitter_band == "unsupported"
    assert "reprojection_regression" in contact.provenance.degraded_reasons
    assert output.summary.regression_kills


def test_every_consumed_raw_input_hash_is_immutable(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in run.glob("*.json")}
    output = build_one_world(run)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in run.glob("*.json")}
    assert before == after
    assert {ref.path: ref.sha256 for ref in output.inputs if ref.missing_reason is None}.items() <= before.items()
