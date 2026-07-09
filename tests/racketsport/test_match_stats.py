from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.list_scaffold_tools import build_scaffold_tool_index

from threed.racketsport.match_stats import compute_match_stats_for_run_dir


def test_compute_match_stats_reports_body_court_only_player_movement(tmp_path: Path) -> None:
    run_dir = _write_stats_fixture(tmp_path)

    payload = compute_match_stats_for_run_dir(run_dir)

    assert payload["artifact_type"] == "racketsport_match_stats"
    assert payload["policy"] == {
        "body_court_only": True,
        "ball_paddle_stats_excluded": True,
        "post_hoc_consumer_only": True,
    }
    assert payload["excluded_stats"] == [
        "shot_counts",
        "rally_stats",
        "contact_stats",
        "ball_speed",
        "paddle_contact",
    ]
    player = payload["players"][0]
    assert player["player_id"] == 1
    assert player["source_frames_used"] == 3
    assert player["source_frames_total"] == 4
    assert player["coverage_fraction"] == 0.75

    distance = player["stats"]["distance_covered_m"]
    assert distance["value"] == round(2.0 * math.sqrt(5.0), 6)
    assert distance["coverage_fraction"] == 0.75
    assert distance["trust_bands"]["position"]["badge"] == "low_confidence"
    assert distance["trust_bands"]["court"]["badge"] == "preview"

    speed = player["stats"]["movement_speed_distribution_mps"]
    assert speed["value"]["p50"] == round(math.sqrt(5.0), 6)
    assert speed["value"]["p95"] == round(math.sqrt(5.0), 6)
    assert speed["frames_used"] == 2
    assert speed["frames_total"] == 3

    zones = player["stats"]["time_in_zone_s"]
    assert zones["value"] == {
        "baseline": 1.0,
        "kitchen": 1.0,
        "out_of_court": 0.0,
        "transition": 1.0,
    }
    assert zones["fractions"] == {
        "baseline": round(1 / 3, 6),
        "kitchen": round(1 / 3, 6),
        "out_of_court": 0.0,
        "transition": round(1 / 3, 6),
    }
    assert player["sanity"]["zone_fraction_sum"] == 1.0

    balance = player["stats"]["left_right_court_balance"]
    assert balance["value"] == {
        "left_fraction": round(1 / 3, 6),
        "right_fraction": round(2 / 3, 6),
    }

    heatmap = player["stats"]["court_coverage_heatmap"]
    assert heatmap["value"]["total_count"] == 3
    assert sum(sum(row) for row in heatmap["value"]["counts"]) == 3
    assert heatmap["coverage_fraction"] == 0.75


def test_compute_match_stats_flags_world_jumps_without_counting_them(tmp_path: Path) -> None:
    run_dir = _write_stats_fixture(
        tmp_path,
        frames=[
            {"frame_idx": 0, "t": 0.0, "smoothed_world_xy": [0.0, -5.0]},
            {"frame_idx": 1, "t": 1.0, "smoothed_world_xy": [20.0, -5.0]},
            {"frame_idx": 2, "t": 2.0, "smoothed_world_xy": [20.5, -5.0]},
        ],
    )

    payload = compute_match_stats_for_run_dir(run_dir)

    player = payload["players"][0]
    assert player["sanity"]["world_jump_count"] == 1
    assert player["sanity"]["world_jumps"][0]["from_frame_idx"] == 0
    assert player["sanity"]["world_jumps"][0]["to_frame_idx"] == 1
    assert player["stats"]["distance_covered_m"]["value"] == 0.5
    assert player["stats"]["movement_speed_distribution_mps"]["value"] == {"p50": 0.5, "p95": 0.5}


def test_compute_match_stats_cli_writes_json_and_help_mentions_run_dir(tmp_path: Path) -> None:
    run_dir = _write_stats_fixture(tmp_path)
    out_json = tmp_path / "match_stats.json"

    help_completed = subprocess.run(
        [sys.executable, "scripts/racketsport/compute_match_stats.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_completed.returncode == 0
    assert "--run-dir" in help_completed.stdout
    assert "--out-json" in help_completed.stdout

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/compute_match_stats.py",
            "--run-dir",
            str(run_dir),
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
    assert stdout["player_count"] == 1
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_match_stats"
    assert payload["players"][0]["stats"]["distance_covered_m"]["value"] == round(2.0 * math.sqrt(5.0), 6)


def test_scaffold_index_registers_compute_match_stats_cli() -> None:
    index = build_scaffold_tool_index(Path("."))
    by_path = {tool["command_path"]: tool for tool in index["tools"]}

    tool = by_path["scripts/racketsport/compute_match_stats.py"]
    assert tool["category"] == "report"
    assert tool["workstream"] == "RPT"
    assert tool["related_test"] == "tests/racketsport/test_match_stats.py"
    assert tool["direct_cli_reference_test"] == "tests/racketsport/test_match_stats.py"


def _write_stats_fixture(
    tmp_path: Path,
    *,
    frames: list[dict[str, object]] | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    frames = frames or [
        {"frame_idx": 0, "t": 0.0, "smoothed_world_xy": [-1.0, -5.0]},
        {"frame_idx": 1, "t": 1.0, "smoothed_world_xy": [0.0, -3.0]},
        {"frame_idx": 2, "t": 2.0, "smoothed_world_xy": [1.0, -1.0]},
    ]
    placement_frames = []
    skeleton_frames = []
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        t = float(frame["t"])
        xy = [float(value) for value in frame["smoothed_world_xy"]]  # type: ignore[index]
        placement_frames.append(
            {
                "frame_idx": frame_idx,
                "t": t,
                "smoothed_world_xy": xy,
                "fused_world_xy": xy,
                "original_world_xy": xy,
                "covariance_m2": [[0.01, 0.0], [0.0, 0.01]],
                "output_source": "synthetic",
                "signals": [],
                "source_counts": {"synthetic": 1},
                "stance": False,
            }
        )
        skeleton_frames.append(
            {
                "frame_idx": frame_idx,
                "t": t,
                "transl_world": [xy[0], xy[1], 0.0],
                "joints_world": [[xy[0], xy[1], 0.0]],
                "joint_conf": [0.9],
            }
        )

    _write_json(
        run_dir / "placement.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_placement",
            "fps": 1.0,
            "players": [{"id": 1, "frames": placement_frames}],
            "provenance": {"source": "synthetic"},
        },
    )
    _write_json(
        run_dir / "skeleton3d.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_skeleton3d",
            "fps": 1.0,
            "world_frame": "court_Z0",
            "source_model": "synthetic_body",
            "joint_names": ["root"],
            "preview_only": True,
            "players": [{"id": 1, "frames": skeleton_frames}],
            "provenance": {"source": "synthetic"},
        },
    )
    _write_json(
        run_dir / "court_zones.json",
        {
            "schema_version": 1,
            "zones": {
                "court": [[-3.0, -6.0], [3.0, -6.0], [3.0, 6.0], [-3.0, 6.0]],
                "near_nvz": [[-3.0, -2.0], [3.0, -2.0], [3.0, 0.0], [-3.0, 0.0]],
                "far_nvz": [[-3.0, 0.0], [3.0, 0.0], [3.0, 2.0], [-3.0, 2.0]],
            },
        },
    )
    _write_json(
        run_dir / "trust_bands.json",
        {
            "body": _trust_band("BODY", "low_confidence"),
            "court": _trust_band("CAL", "preview"),
            "track": _trust_band("TRK", "low_confidence"),
        },
    )
    _write_json(
        run_dir / "frame_times.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_frame_times",
            "fps": 1.0,
            "frame_count": 4,
            "duration_s": 4.0,
            "frames": [{"frame": idx, "pts_s": float(idx)} for idx in range(4)],
        },
    )
    _write_json(run_dir / "ball_track.json", {"artifact_type": "must_not_be_consumed", "frames": [{"bad": True}]})
    _write_json(run_dir / "racket_pose.json", {"artifact_type": "must_not_be_consumed", "frames": [{"bad": True}]})
    return run_dir


def _trust_band(stage: str, badge: str) -> dict[str, object]:
    return {
        "stage": stage,
        "gate_id": f"{stage.lower()}_fixture_gate",
        "gate_status": "fixture",
        "badge": badge,
        "reason": "synthetic fixture trust band",
        "evidence_path": None,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
