from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.virtual_world_review import build_virtual_world_review_html


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _virtual_world_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 60.0,
        "court": {
            "sport": "pickleball",
            "coordinate_frame": "origin_net_center_x_width_y_length_z_up_m",
            "length_m": 13.4112,
            "width_m": 6.096,
            "line_segments": {
                "near_baseline": [[-3.048, -6.7056, 0.0], [3.048, -6.7056, 0.0]],
                "net": [[-3.3528, 0.0, 0.0], [3.3528, 0.0, 0.0]],
            },
            "net": {
                "endpoints": [[-3.3528, 0.0, 0.0], [3.3528, 0.0, 0.0]],
                "center_height_m": 0.8636,
                "post_height_m": 0.9144,
            },
        },
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "representation": "mesh",
                "frames": [
                    {
                        "t": 0.0,
                        "track_world_xy": [0.2, -2.0],
                        "track_conf": 0.9,
                        "bbox": [100.0, 100.0, 200.0, 300.0],
                        "transl_world": [0.2, -2.0, 0.0],
                        "joints_world": [[0.2, -2.0, 0.0], [0.2, -2.0, 1.5]],
                        "joint_conf": [0.9, 0.8],
                        "mesh_vertices_world": [[0.1, -2.1, 0.0], [0.3, -1.9, 1.6]],
                        "joint_count": 2,
                        "mesh_vertex_count": 2,
                    }
                ],
            }
        ],
        "ball": {
            "source": "tracknet",
            "frames": [
                {
                    "t": 0.0,
                    "xy": [100.0, 200.0],
                    "conf": 0.8,
                    "visible": True,
                    "world_xyz": [0.0, -1.0, 0.0],
                    "approx": True,
                }
            ],
        },
        "paddles": [
            {
                "player_id": 7,
                "paddle_dims_in": {"length": 16.0, "width": 8.0},
                "frames": [
                    {
                        "t": 0.0,
                        "pose_se3": {
                            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "t": [0.3, -1.8, 0.8],
                        },
                        "mesh_vertices_world": [
                            [0.2, -1.7, 0.8],
                            [0.4, -1.7, 0.8],
                            [0.4, -1.9, 0.8],
                            [0.2, -1.9, 0.8],
                        ],
                        "mesh_faces": [[0, 1, 2], [0, 2, 3]],
                        "conf": 0.4,
                        "world_frame": "court_Z0",
                        "translation_unit": "m",
                        "source": "draft:pnp_ippe_preview:court_Z0",
                        "reprojection_error_px": 0.2,
                        "ambiguous": True,
                    }
                ],
            }
        ],
        "summary": {
            "player_count": 1,
            "mesh_player_count": 1,
            "mesh_player_frame_count": 1,
            "joint_player_frame_count": 1,
            "track_only_player_frame_count": 0,
            "ball_frame_count": 1,
            "approx_ball_frame_count": 1,
            "paddle_player_count": 1,
            "paddle_frame_count": 1,
            "ambiguous_paddle_frame_count": 1,
            "warnings": ["ambiguous_paddle_pose"],
        },
    }


def test_build_virtual_world_review_html_embeds_three_viewer_and_world_payload() -> None:
    html = build_virtual_world_review_html(_virtual_world_payload(), title="clip_001 World")

    assert "import * as THREE" in html
    assert 'id="virtual-world-data"' in html
    assert "ambiguous_paddle_pose" in html
    assert "mesh_vertices_world" in html
    assert "Approx ball frames" in html
    assert "Ambiguous paddle frames" in html
    assert "clip_001 World" in html


def test_build_virtual_world_review_cli_writes_html_and_packet_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_001"
    world = _write_json(run_dir / "virtual_world_paddle_preview.json", _virtual_world_payload())
    html = run_dir / "virtual_world_paddle_preview.html"
    index = run_dir / "virtual_world_review_index.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_virtual_world_review.py",
            "--virtual-world",
            str(world),
            "--out-html",
            str(html),
            "--index-out",
            str(index),
            "--title",
            "clip_001 Paddle Preview",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "clip_001 Paddle Preview" in html.read_text(encoding="utf-8")
    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_virtual_world_review"
    assert payload["status"] == "rendered"
    assert payload["clip"] == "clip_001"
    assert payload["review_html"] == str(html)
    assert "Approx ball frames: 1" in payload["details"]
    assert "Ambiguous paddle frames: 1" in payload["details"]
    assert payload["warnings"] == ["ambiguous_paddle_pose"]
    assert json.loads(completed.stdout)["review_html"] == str(html)
