from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_classify_shots_cli_exposes_direct_help_reference() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/classify_shots.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--ball-track-arc-solved" in completed.stdout
    assert "--events-selected" in completed.stdout
    assert "--out-json" in completed.stdout


def test_classify_shots_cli_writes_shots_json(tmp_path: Path) -> None:
    arc_path = tmp_path / "ball_track_arc_solved.json"
    events_path = tmp_path / "events_selected.json"
    zones_path = tmp_path / "court_zones.json"
    net_path = tmp_path / "net_plane.json"
    tracks_path = tmp_path / "tracks.json"
    out_json = tmp_path / "shots.json"

    arc_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_track_arc_solved",
                "clip_id": "cli_clip",
                "render_only": True,
                "not_for_detection_metrics": True,
                "inputs": {},
                "segments": [
                    {
                        "segment_id": 0,
                        "status": "fit",
                        "start_anchor": "contact_drive",
                        "end_anchor": "bounce_drive",
                        "anchors_used": [
                            {
                                "anchor_id": "contact_drive",
                                "kind": "contact",
                                "frame": 0,
                                "t": 0.0,
                                "world_xyz": [0.0, -5.6, 1.0],
                                "sigma_m": 0.12,
                            },
                            {
                                "anchor_id": "bounce_drive",
                                "kind": "bounce",
                                "frame": 10,
                                "t": 1.0,
                                "world_xyz": [0.2, 5.4, 0.0371],
                                "sigma_m": 0.10,
                            },
                        ],
                        "frame_start": 0,
                        "frame_end": 10,
                        "t0": 0.0,
                        "t1": 1.0,
                        "initial_position_m": [0.0, -5.6, 1.0],
                        "initial_velocity_mps": [0.2, 13.0, 0.8],
                        "initial_speed_mps": 13.1,
                        "physical_sanity": {"apex_height_m": 1.35, "initial_speed_mps": 13.1, "violation": False},
                        "net_clearance_m": 0.3,
                        "net_clearance_ok": True,
                        "endpoint_error_m": 0.3,
                        "reprojection_rmse_px": 4.0,
                        "max_reprojection_error_px": 4.0,
                        "inlier_count": 8,
                        "outlier_count": 0,
                    }
                ],
                "frames": [],
                "summary": {"segment_count": 1},
            }
        ),
        encoding="utf-8",
    )
    events_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_arc_events_selected",
                "candidate_prediction": True,
                "not_ground_truth": True,
                "selected": [
                    {
                        "anchor_id": "contact_drive",
                        "kind": "contact",
                        "frame": 0,
                        "t": 0.0,
                        "player_id": 1,
                        "candidate_confidence": 0.9,
                        "sigma_m": 0.12,
                        "selected": True,
                        "world_xyz": [0.0, -5.6, 1.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    zones_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "zones": {
                    "court": [[-3.048, -6.7056], [3.048, -6.7056], [3.048, 6.7056], [-3.048, 6.7056]],
                    "near_nvz": [[-3.048, -2.1336], [3.048, -2.1336], [3.048, 0.0], [-3.048, 0.0]],
                    "far_nvz": [[-3.048, 0.0], [3.048, 0.0], [3.048, 2.1336], [-3.048, 2.1336]],
                },
            }
        ),
        encoding="utf-8",
    )
    net_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "center_height_in": 34.0,
                "post_height_in": 36.0,
                "plane": {"normal": [0.0, 1.0, 0.0], "point": [0.0, 0.0, 0.0]},
            }
        ),
        encoding="utf-8",
    )
    tracks_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 10.0,
                "players": [{"id": 1, "frames": [{"t": 0.0, "world_xy": [0.0, -5.6], "conf": 0.9}]}],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/classify_shots.py",
            "--clip-id",
            "cli_clip",
            "--ball-track-arc-solved",
            str(arc_path),
            "--events-selected",
            str(events_path),
            "--court-zones",
            str(zones_path),
            "--net-plane",
            str(net_path),
            "--tracks",
            str(tracks_path),
            "--out-json",
            str(out_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["out_json"] == str(out_json)
    assert stdout["shot_count"] == 1
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_shots"
    assert payload["shots"][0]["shot_type"] == "drive"
    assert payload["shots"][0]["outcome"]["call"] == "in"
