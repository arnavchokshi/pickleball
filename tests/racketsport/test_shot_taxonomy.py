from __future__ import annotations

from threed.racketsport.shot_taxonomy import classify_shots_from_payloads


def test_pb_vision_geometry_labels_and_rally_indexing() -> None:
    payload = classify_shots_from_payloads(
        clip_id="synthetic_clip",
        ball_arc_payload=_arc_payload(
            [
                _shot_segment(
                    "serve_drive",
                    index=0,
                    start=[0.0, -5.8, 1.1],
                    landing=[0.2, 5.6, 0.0371],
                    velocity=[0.0, 14.0, 1.0],
                    apex=1.45,
                    speed=14.1,
                ),
                _shot_segment(
                    "return_drive",
                    index=1,
                    start=[0.2, 5.3, 1.0],
                    landing=[-0.4, -5.2, 0.0371],
                    velocity=[-0.4, -13.0, 0.8],
                    apex=1.35,
                    speed=13.1,
                ),
                _shot_segment(
                    "third_drop",
                    index=2,
                    start=[-0.2, -5.9, 1.0],
                    landing=[0.1, 1.25, 0.0371],
                    velocity=[0.1, 7.0, 2.5],
                    apex=1.55,
                    speed=7.4,
                ),
                _shot_segment(
                    "kitchen_dink",
                    index=3,
                    start=[0.2, 1.45, 0.75],
                    landing=[-0.1, -1.35, 0.0371],
                    velocity=[-0.1, -4.5, 1.0],
                    apex=1.05,
                    speed=4.7,
                ),
                _shot_segment(
                    "deep_lob",
                    index=4,
                    start=[-0.4, -3.5, 0.85],
                    landing=[0.3, 6.0, 0.0371],
                    velocity=[0.2, 6.5, 5.8],
                    apex=3.4,
                    speed=8.8,
                ),
                _shot_segment(
                    "overhead_smash",
                    index=5,
                    start=[0.3, 1.0, 1.7],
                    landing=[-0.2, -5.8, 0.0371],
                    velocity=[-0.2, -18.0, -4.5],
                    apex=1.78,
                    speed=18.6,
                ),
                _shot_segment(
                    "around_post",
                    index=6,
                    start=[3.42, -1.1, 0.75],
                    landing=[2.2, 5.2, 0.0371],
                    velocity=[-1.2, 6.0, 0.7],
                    apex=1.15,
                    speed=6.2,
                    net_clearance=None,
                ),
                _shot_segment(
                    "erne_launch",
                    index=7,
                    start=[3.38, 1.3, 1.15],
                    landing=[-1.5, -4.2, 0.0371],
                    velocity=[-7.0, -8.0, -0.5],
                    apex=1.25,
                    speed=10.7,
                ),
                _shot_segment(
                    "tweener_low",
                    index=8,
                    start=[0.0, -6.95, 0.42],
                    landing=[0.2, 3.8, 0.0371],
                    velocity=[0.1, 10.5, 0.6],
                    apex=0.62,
                    speed=10.6,
                ),
            ]
        ),
        events_selected_payload=_events_payload(
            [
                _contact("serve_drive", 1, 0, [0.0, -5.8, 1.1]),
                _contact("return_drive", 2, 10, [0.2, 5.3, 1.0]),
                _contact("third_drop", 1, 20, [-0.2, -5.9, 1.0]),
                _contact("kitchen_dink", 3, 30, [0.2, 1.45, 0.75]),
                _contact("deep_lob", 4, 40, [-0.4, -3.5, 0.85]),
                _contact("overhead_smash", 2, 50, [0.3, 1.0, 1.7]),
                _contact("around_post", 3, 60, [3.42, -1.1, 0.75]),
                _contact("erne_launch", 4, 70, [3.38, 1.3, 1.15]),
                _contact("tweener_low", 1, 80, [0.0, -6.95, 0.42]),
            ]
        ),
        court_zones_payload=_court_zones(),
        net_plane_payload=_net_plane(),
        tracks_payload=_tracks_payload(),
    )

    by_anchor = {shot["event_anchor_id"]: shot for shot in payload["shots"]}

    assert by_anchor["serve_drive"]["shot_type"] == "drive"
    assert by_anchor["serve_drive"]["rally_index"] == {"contact_index": 1, "label": "serve"}
    assert by_anchor["return_drive"]["shot_type"] == "drive"
    assert by_anchor["return_drive"]["rally_index"] == {"contact_index": 2, "label": "return"}
    assert by_anchor["third_drop"]["shot_type"] == "drop"
    assert by_anchor["third_drop"]["rally_index"] == {
        "contact_index": 3,
        "label": "third",
        "third_shot": "drop",
    }
    assert by_anchor["kitchen_dink"]["shot_type"] == "dink"
    assert by_anchor["deep_lob"]["shot_type"] == "lob"
    assert by_anchor["overhead_smash"]["shot_type"] == "smash"
    assert by_anchor["around_post"]["shot_type"] == "atp"
    assert by_anchor["erne_launch"]["shot_type"] == "erne"
    assert by_anchor["tweener_low"]["shot_type"] == "tweener"
    assert by_anchor["kitchen_dink"]["launch_zone"] == "far_nvz"
    assert by_anchor["kitchen_dink"]["landing"]["zone"] == "near_nvz"
    assert 31.0 <= by_anchor["serve_drive"]["speed_mph"] <= 32.0
    assert payload["summary"]["shot_type_counts"] == {
        "atp": 1,
        "dink": 1,
        "drive": 2,
        "drop": 1,
        "erne": 1,
        "lob": 1,
        "smash": 1,
        "tweener": 1,
    }


def test_fault_out_net_let_excess_bounce_and_uncertainty() -> None:
    payload = classify_shots_from_payloads(
        clip_id="synthetic_faults",
        ball_arc_payload=_arc_payload(
            [
                _shot_segment(
                    "net_fault",
                    index=0,
                    start=[0.0, -3.0, 0.9],
                    landing=[0.0, -0.05, 0.70],
                    velocity=[0.0, 6.0, 0.1],
                    apex=0.92,
                    speed=6.1,
                    net_clearance=-0.06,
                    net_clearance_ok=False,
                ),
                _shot_segment(
                    "let_cord",
                    index=1,
                    start=[0.0, -3.0, 0.9],
                    landing=[0.0, 4.5, 0.0371],
                    velocity=[0.0, 7.5, 1.2],
                    apex=1.2,
                    speed=7.6,
                    net_clearance=0.025,
                    net_clearance_ok=True,
                ),
                _shot_segment(
                    "wide_out",
                    index=2,
                    start=[0.0, -2.8, 1.0],
                    landing=[3.8, 4.2, 0.0371],
                    velocity=[5.0, 7.0, 1.0],
                    apex=1.2,
                    speed=8.7,
                ),
                _shot_segment(
                    "double_bounce",
                    index=3,
                    start=[0.0, -2.8, 1.0],
                    landing=[0.5, 4.0, 0.0371],
                    velocity=[0.5, 6.5, 1.0],
                    apex=1.1,
                    speed=6.6,
                    end_anchor="double_bounce_bounce1",
                ),
                _segment_between_bounces("double_bounce_bounce1", "double_bounce_bounce2", index=4),
            ]
        ),
        events_selected_payload=_events_payload(
            [
                _contact("net_fault", 1, 0, [0.0, -3.0, 0.9]),
                _contact("let_cord", 2, 10, [0.0, -3.0, 0.9]),
                _contact("wide_out", 3, 20, [0.0, -2.8, 1.0]),
                _contact("double_bounce", 4, 30, [0.0, -2.8, 1.0]),
                _bounce("double_bounce_bounce1", 35, [0.5, 4.0, 0.0371]),
                _bounce("double_bounce_bounce2", 45, [0.8, 5.0, 0.0371]),
            ]
        ),
        court_zones_payload=_court_zones(),
        net_plane_payload=_net_plane(),
        tracks_payload=_tracks_payload(),
    )

    by_anchor = {shot["event_anchor_id"]: shot for shot in payload["shots"]}

    assert by_anchor["net_fault"]["outcome"] == {
        "call": "net_hit",
        "faults": ["net_hit"],
        "net_clearance_m": -0.06,
        "let_candidate": False,
    }
    assert by_anchor["let_cord"]["outcome"]["call"] == "in"
    assert by_anchor["let_cord"]["outcome"]["let_candidate"] is True
    assert by_anchor["let_cord"]["outcome"]["faults"] == ["let_candidate"]
    assert by_anchor["wide_out"]["outcome"]["call"] == "out"
    assert by_anchor["wide_out"]["outcome"]["out"] == {
        "direction": "wide",
        "side": "right",
        "landed": True,
    }
    assert by_anchor["double_bounce"]["outcome"]["call"] == "excess_bounce"
    assert by_anchor["double_bounce"]["outcome"]["faults"] == ["excess_bounce"]
    ellipse = by_anchor["wide_out"]["landing"]["uncertainty_ellipse"]
    assert ellipse["semi_major_m"] > ellipse["semi_minor_m"] > 0.0
    assert ellipse["source"] == "segment_sigma_endpoint_reprojection_v1"


def test_low_confidence_abstains_from_shot_type_but_keeps_evidence() -> None:
    payload = classify_shots_from_payloads(
        clip_id="synthetic_weak",
        ball_arc_payload=_arc_payload(
            [
                _shot_segment(
                    "weak_contact",
                    index=0,
                    start=[0.0, -5.8, 1.0],
                    landing=[0.2, 5.0, 0.0371],
                    velocity=[0.1, 7.0, 2.0],
                    apex=1.4,
                    speed=7.3,
                    endpoint_error=24.0,
                    rmse=90.0,
                    inliers=0,
                    outliers=20,
                )
            ]
        ),
        events_selected_payload=_events_payload([_contact("weak_contact", 1, 0, [0.0, -5.8, 1.0], confidence=0.24)]),
        court_zones_payload=_court_zones(),
        net_plane_payload=_net_plane(),
        tracks_payload=_tracks_payload(),
    )

    shot = payload["shots"][0]
    assert "shot_type" not in shot
    assert shot["shot_type_abstained"] is True
    assert shot["confidence"] < 0.45
    assert shot["outcome"]["call"] == "in"
    assert "low_segment_confidence" in shot["warnings"]
    assert payload["summary"]["abstained_count"] == 1


def test_serve_and_return_indices_do_not_emit_trick_shot_types() -> None:
    payload = classify_shots_from_payloads(
        clip_id="synthetic_serve_return_tricks",
        ball_arc_payload=_arc_payload(
            [
                _shot_segment(
                    "serve_wide_path",
                    index=0,
                    start=[3.5, -1.0, 1.0],
                    landing=[3.3, 5.2, 0.0371],
                    velocity=[-0.2, 9.0, 0.8],
                    apex=1.2,
                    speed=9.1,
                    net_clearance=None,
                ),
                _shot_segment(
                    "return_wide_path",
                    index=1,
                    start=[3.5, 1.0, 1.0],
                    landing=[3.3, -5.2, 0.0371],
                    velocity=[-0.2, -9.0, 0.8],
                    apex=1.2,
                    speed=9.1,
                    net_clearance=None,
                ),
                _shot_segment(
                    "third_wide_path",
                    index=2,
                    start=[3.5, -1.0, 1.0],
                    landing=[3.3, 5.2, 0.0371],
                    velocity=[-0.2, 9.0, 0.8],
                    apex=1.2,
                    speed=9.1,
                    net_clearance=None,
                ),
            ]
        ),
        events_selected_payload=_events_payload(
            [
                _contact("serve_wide_path", 1, 0, [3.5, -1.0, 1.0]),
                _contact("return_wide_path", 2, 10, [3.5, 1.0, 1.0]),
                _contact("third_wide_path", 1, 20, [3.5, -1.0, 1.0]),
            ]
        ),
        court_zones_payload=_court_zones(),
        net_plane_payload=_net_plane(),
        tracks_payload=_tracks_payload(),
    )

    by_anchor = {shot["event_anchor_id"]: shot for shot in payload["shots"]}

    assert by_anchor["serve_wide_path"]["rally_index"]["label"] == "serve"
    assert by_anchor["serve_wide_path"]["shot_type"] == "drive"
    assert by_anchor["return_wide_path"]["rally_index"]["label"] == "return"
    assert by_anchor["return_wide_path"]["shot_type"] == "drive"
    assert by_anchor["third_wide_path"]["rally_index"]["label"] == "third"
    assert by_anchor["third_wide_path"]["shot_type"] == "atp"


def _events_payload(selected: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_arc_events_selected",
        "candidate_prediction": True,
        "not_ground_truth": True,
        "selected": selected,
        "selected_count": len(selected),
        "rejected": [],
        "rejected_count": 0,
    }


def _arc_payload(segments: list[dict[str, object]]) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for segment in segments:
        landing = segment["anchors_used"][1]["world_xyz"]  # type: ignore[index]
        frames.append(
            {
                "t": segment["t1"],
                "frame": segment["frame_end"],
                "world_xyz": landing,
                "sigma_m": 0.12,
                "band": "arc_interpolated",
                "conf": 0.8,
                "arc_solver": {"segment_id": segment["segment_id"]},
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_arc_solved",
        "clip_id": "synthetic_clip",
        "render_only": True,
        "not_for_detection_metrics": True,
        "segments": segments,
        "frames": frames,
        "inputs": {},
        "summary": {"segment_count": len(segments)},
    }


def _contact(
    anchor_id: str,
    player_id: int,
    frame: int,
    world_xyz: list[float],
    *,
    confidence: float = 0.86,
) -> dict[str, object]:
    return {
        "anchor_id": anchor_id,
        "kind": "contact",
        "frame": frame,
        "t": frame / 10.0,
        "player_id": player_id,
        "candidate_confidence": confidence,
        "sigma_m": 0.12,
        "selected": True,
        "world_xyz": world_xyz,
    }


def _bounce(anchor_id: str, frame: int, world_xyz: list[float]) -> dict[str, object]:
    return {
        "anchor_id": anchor_id,
        "kind": "bounce",
        "frame": frame,
        "t": frame / 10.0,
        "candidate_confidence": 0.9,
        "sigma_m": 0.08,
        "selected": True,
        "world_xyz": world_xyz,
    }


def _shot_segment(
    anchor_id: str,
    *,
    index: int,
    start: list[float],
    landing: list[float],
    velocity: list[float],
    apex: float,
    speed: float,
    net_clearance: float | None = 0.35,
    net_clearance_ok: bool | None = True,
    endpoint_error: float = 0.30,
    rmse: float = 4.0,
    inliers: int = 10,
    outliers: int = 0,
    end_anchor: str | None = None,
) -> dict[str, object]:
    return {
        "segment_id": index,
        "status": "fit",
        "start_anchor": anchor_id,
        "end_anchor": end_anchor or f"{anchor_id}_landing",
        "anchors_used": [
            {
                "anchor_id": anchor_id,
                "kind": "contact",
                "frame": index * 10,
                "t": float(index),
                "world_xyz": start,
                "sigma_m": 0.12,
            },
            {
                "anchor_id": end_anchor or f"{anchor_id}_landing",
                "kind": "bounce",
                "frame": index * 10 + 8,
                "t": float(index) + 0.8,
                "world_xyz": landing,
                "sigma_m": 0.10,
            },
        ],
        "frame_start": index * 10,
        "frame_end": index * 10 + 8,
        "t0": float(index),
        "t1": float(index) + 0.8,
        "initial_position_m": start,
        "initial_velocity_mps": velocity,
        "initial_speed_mps": speed,
        "physical_sanity": {"apex_height_m": apex, "initial_speed_mps": speed, "violation": False, "violations": []},
        "net_clearance_m": net_clearance,
        "net_clearance_ok": net_clearance_ok,
        "endpoint_error_m": endpoint_error,
        "reprojection_rmse_px": rmse,
        "max_reprojection_error_px": rmse,
        "inlier_count": inliers,
        "outlier_count": outliers,
    }


def _segment_between_bounces(start_anchor: str, end_anchor: str, *, index: int) -> dict[str, object]:
    return {
        "segment_id": index,
        "status": "fit",
        "start_anchor": start_anchor,
        "end_anchor": end_anchor,
        "anchors_used": [
            {"anchor_id": start_anchor, "kind": "bounce", "frame": 35, "t": 3.5, "world_xyz": [0.5, 4.0, 0.0371], "sigma_m": 0.08},
            {"anchor_id": end_anchor, "kind": "bounce", "frame": 45, "t": 4.5, "world_xyz": [0.8, 5.0, 0.0371], "sigma_m": 0.08},
        ],
        "frame_start": 35,
        "frame_end": 45,
        "t0": 3.5,
        "t1": 4.5,
        "initial_position_m": [0.5, 4.0, 0.0371],
        "initial_velocity_mps": [0.3, 1.0, 1.5],
        "initial_speed_mps": 1.8,
        "physical_sanity": {"apex_height_m": 0.5, "initial_speed_mps": 1.8, "violation": False, "violations": []},
        "net_clearance_m": None,
        "net_clearance_ok": None,
        "endpoint_error_m": 0.2,
        "reprojection_rmse_px": 3.0,
        "max_reprojection_error_px": 3.0,
        "inlier_count": 5,
        "outlier_count": 0,
    }


def _court_zones() -> dict[str, object]:
    return {
        "schema_version": 1,
        "zones": {
            "court": [[-3.048, -6.7056], [3.048, -6.7056], [3.048, 6.7056], [-3.048, 6.7056]],
            "near_nvz": [[-3.048, -2.1336], [3.048, -2.1336], [3.048, 0.0], [-3.048, 0.0]],
            "far_nvz": [[-3.048, 0.0], [3.048, 0.0], [3.048, 2.1336], [-3.048, 2.1336]],
        },
    }


def _net_plane() -> dict[str, object]:
    return {
        "schema_version": 1,
        "center_height_in": 34.0,
        "post_height_in": 36.0,
        "plane": {"normal": [0.0, 1.0, 0.0], "point": [0.0, 0.0, 0.0]},
    }


def _tracks_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 10.0,
        "players": [
            {"id": 1, "frames": [{"t": 0.0, "world_xy": [0.0, -5.8], "conf": 0.9}]},
            {"id": 2, "frames": [{"t": 1.0, "world_xy": [0.2, 5.3], "conf": 0.9}]},
            {"id": 3, "frames": [{"t": 3.0, "world_xy": [0.2, 1.45], "conf": 0.9}]},
            {"id": 4, "frames": [{"t": 4.0, "world_xy": [-0.4, -3.5], "conf": 0.9}]},
        ],
    }
