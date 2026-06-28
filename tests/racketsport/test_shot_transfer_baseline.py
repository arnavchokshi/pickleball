import json
import subprocess
import sys

from threed.racketsport.shot_transfer_baseline import classify_shots_from_payloads


def _contact_payload():
    return {
        "schema_version": 1,
        "events": [
            {
                "t": 1.0,
                "frame": 60,
                "player_id": 2,
                "confidence": 0.9,
                "sources": {"human_review": 1.0},
                "type": "contact",
                "window": {"t0": 0.92, "t1": 1.08, "importance": 0.9},
            }
        ],
    }


def test_baseline_classifies_slow_nvz_contact_as_dink():
    payload = classify_shots_from_payloads(
        clip_id="clip_dink",
        contact_windows_payload=_contact_payload(),
        ball_inflections_payload={
            "schema_version": 1,
            "candidates": [
                {
                    "time_s": 1.02,
                    "frame": 61,
                    "ball_world_xyz": [0.2, 1.1, 0.0],
                    "speed_before_mps": 2.8,
                    "speed_after_mps": 3.1,
                    "turn_angle_deg": 126.0,
                    "confidence": 0.8,
                }
            ],
        },
    )

    assert payload["artifact_type"] == "racketsport_shot_classification"
    assert payload["classifier"]["name"] == "shot_transfer_baseline_v1"
    assert payload["classifier"]["not_gate_verified"] is True
    assert payload["shots"][0]["type"] == "dink"
    assert payload["shots"][0]["player_id"] == 2
    assert payload["shots"][0]["evidence"]["matched_ball_inflection"]["dt_s"] == 0.02


def test_baseline_fails_closed_when_contact_has_no_near_ball_evidence():
    payload = classify_shots_from_payloads(
        clip_id="clip_unknown",
        contact_windows_payload=_contact_payload(),
        ball_inflections_payload={
            "schema_version": 1,
            "candidates": [
                {
                    "time_s": 2.5,
                    "frame": 150,
                    "ball_world_xyz": [0.2, 1.1, 0.0],
                    "speed_before_mps": 2.8,
                    "speed_after_mps": 3.1,
                    "turn_angle_deg": 126.0,
                    "confidence": 0.8,
                }
            ],
        },
        max_ball_dt_s=0.15,
    )

    assert payload["shots"][0]["type"] == "unknown"
    assert payload["shots"][0]["type_conf"] == 0.0
    assert payload["shots"][0]["gate_reasons"] == ["no ball inflection within 0.150s"]
    assert payload["summary"]["unknown_count"] == 1


def test_baseline_uses_semantic_wrist_joints_when_ball_inflection_is_missing():
    payload = classify_shots_from_payloads(
        clip_id="clip_pose_fallback",
        contact_windows_payload=_contact_payload(),
        ball_inflections_payload={"schema_version": 1, "candidates": []},
        skeleton3d_payload={
            "schema_version": 1,
            "joint_names": ["left_shoulder", "right_shoulder", "left_wrist", "right_wrist"],
            "players": [
                {
                    "id": 2,
                    "frames": [
                        {
                            "t": 1.0,
                            "joints_world": [
                                [-0.3, -2.0, 1.4],
                                [0.3, -2.0, 1.4],
                                [-0.2, -2.0, 1.0],
                                [1.2, -2.0, 1.0],
                            ],
                            "joint_conf": [0.9, 0.9, 0.85, 0.9],
                        }
                    ],
                }
            ],
        },
    )

    shot = payload["shots"][0]
    assert shot["type"] == "fh_shot"
    assert shot["specific_type_candidate"] == "fh_drive"
    assert shot["type_conf"] > 0.4
    assert shot["gated"] is False
    assert shot["evidence"]["pose_track_fallback"]["source"] == "semantic_wrist_extension"
    assert payload["summary"]["unknown_count"] == 0


def test_baseline_uses_sam3d_mhr70_semantic_adapter_when_joint_names_are_generic():
    joints = [[0.0, 0.0, 1.0] for _index in range(70)]
    joints[5] = [-0.3, -2.0, 1.4]
    joints[6] = [0.3, -2.0, 1.4]
    joints[41] = [1.2, -2.0, 1.0]
    joints[62] = [-0.2, -2.0, 1.0]
    payload = classify_shots_from_payloads(
        clip_id="clip_sam3d_semantic_adapter",
        contact_windows_payload=_contact_payload(),
        ball_inflections_payload={"schema_version": 1, "candidates": []},
        skeleton3d_payload={
            "schema_version": 1,
            "joint_names": [f"sam3dbody_joint_{index:03d}" for index in range(70)],
            "players": [
                {
                    "id": 2,
                    "frames": [
                        {
                            "t": 1.0,
                            "joints_world": joints,
                            "joint_conf": [0.9] * 70,
                        }
                    ],
                }
            ],
        },
    )

    shot = payload["shots"][0]
    assert shot["type"] == "fh_shot"
    assert shot["specific_type_candidate"] == "fh_drive"
    assert shot["evidence"]["pose_track_fallback"]["source"] == "semantic_wrist_extension"
    assert shot["evidence"]["pose_track_fallback"]["semantic_joint_source"] == "sam3d_body_mhr70_v1"


def test_baseline_keeps_wrong_count_generic_joints_fail_closed_without_track_fallback():
    payload = classify_shots_from_payloads(
        clip_id="clip_wrong_count_generic",
        contact_windows_payload=_contact_payload(),
        ball_inflections_payload={"schema_version": 1, "candidates": []},
        skeleton3d_payload={
            "schema_version": 1,
            "joint_names": [f"sam3dbody_joint_{index:03d}" for index in range(69)],
            "players": [{"id": 2, "frames": [{"t": 1.0, "joints_world": [[0.0, 0.0, 1.0]] * 69}]}],
        },
    )

    assert payload["shots"][0]["type"] == "unknown"
    assert payload["shots"][0]["gated"] is True
    assert payload["shots"][0]["evidence"].get("pose_track_fallback") is None


def test_baseline_uses_ball_track_and_player_bbox_when_pose_joints_are_not_semantic():
    payload = classify_shots_from_payloads(
        clip_id="clip_track_fallback",
        contact_windows_payload=_contact_payload(),
        ball_inflections_payload={"schema_version": 1, "candidates": []},
        tracks_payload={
            "schema_version": 1,
            "fps": 60.0,
            "players": [
                {
                    "id": 2,
                    "side": "near",
                    "role": "right",
                    "frames": [{"t": 1.0, "bbox": [100.0, 100.0, 180.0, 260.0], "world_xy": [0.0, -3.0], "conf": 0.8}],
                }
            ],
            "rally_spans": [],
        },
        ball_track_payload={
            "schema_version": 1,
            "fps": 60.0,
            "source": "tracknet",
            "frames": [{"t": 1.01, "xy": [230.0, 150.0], "conf": 1.0, "visible": True, "approx": False}],
            "bounces": [],
        },
    )

    shot = payload["shots"][0]
    assert shot["type"] == "fh_shot"
    assert shot["specific_type_candidate"] == "fh_drive"
    assert shot["type_conf"] < 0.7
    assert shot["evidence"]["pose_track_fallback"]["source"] == "ball_track_bbox_side"


def test_baseline_infers_hitter_from_tracks_when_contact_has_no_player_id():
    contact_payload = _contact_payload()
    contact_payload["events"][0]["player_id"] = None
    payload = classify_shots_from_payloads(
        clip_id="clip_track_hitter_fallback",
        contact_windows_payload=contact_payload,
        ball_inflections_payload={"schema_version": 1, "candidates": []},
        tracks_payload={
            "schema_version": 1,
            "fps": 60.0,
            "players": [
                {
                    "id": 1,
                    "frames": [{"t": 1.0, "bbox": [10.0, 100.0, 90.0, 260.0], "conf": 0.8}],
                },
                {
                    "id": 2,
                    "frames": [{"t": 1.0, "bbox": [210.0, 100.0, 290.0, 260.0], "conf": 0.8}],
                },
            ],
        },
        ball_track_payload={
            "schema_version": 1,
            "fps": 60.0,
            "frames": [{"t": 1.01, "xy": [310.0, 150.0], "conf": 1.0, "visible": True}],
        },
    )

    shot = payload["shots"][0]
    assert shot["type"] == "fh_shot"
    assert shot["specific_type_candidate"] == "fh_drive"
    assert shot["player_id"] == 2
    assert shot["gated"] is False
    assert shot["evidence"]["pose_track_fallback"]["source"] == "ball_track_bbox_side"
    assert payload["summary"]["unknown_count"] == 0


def test_baseline_uses_low_confidence_ball_image_side_when_player_track_is_missing():
    contact_payload = _contact_payload()
    contact_payload["events"][0]["player_id"] = None
    payload = classify_shots_from_payloads(
        clip_id="clip_ball_side_fallback",
        contact_windows_payload=contact_payload,
        ball_inflections_payload={"schema_version": 1, "candidates": []},
        tracks_payload={"schema_version": 1, "players": []},
        ball_track_payload={
            "schema_version": 1,
            "frame_width": 640,
            "frames": [
                {"t": 0.5, "xy": [180.0, 150.0], "conf": 0.8, "visible": True},
                {"t": 1.42, "xy": [500.0, 150.0], "conf": 0.8, "visible": True},
            ],
        },
        max_ball_dt_s=0.3,
    )

    shot = payload["shots"][0]
    assert shot["type"] == "fh_shot"
    assert shot["specific_type_candidate"] == "fh_drive"
    assert shot["player_id"] is None
    assert 0.0 < shot["type_conf"] < 0.4
    assert shot["gated"] is False
    assert shot["evidence"]["pose_track_fallback"]["source"] == "ball_track_image_side"
    assert shot["evidence"]["pose_track_fallback"]["ball_track_dt_s"] == 0.42
    assert payload["summary"]["unknown_count"] == 0


def test_cli_writes_classification_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "contact_windows.json").write_text(json.dumps(_contact_payload()), encoding="utf-8")
    (run_dir / "ball_inflections.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {
                        "time_s": 1.0,
                        "frame": 60,
                        "ball_world_xyz": [0.0, -5.8, 0.0],
                        "speed_before_mps": 9.0,
                        "speed_after_mps": 11.0,
                        "turn_angle_deg": 108.0,
                        "confidence": 0.7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "shots.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/generate_shot_classifications.py",
            "--run-dir",
            str(run_dir),
            "--clip-id",
            "clip_drive",
            "--out-json",
            str(out),
        ],
        check=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["clip_id"] == "clip_drive"
    assert payload["shots"][0]["type"] == "fh_drive"
    assert payload["summary"]["shot_count"] == 1
