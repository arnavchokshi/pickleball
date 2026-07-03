from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_overlay import load_ball_track
from threed.racketsport.ball_player_proximity_prior import (
    BallPlayerProximityPriorConfig,
    apply_ball_player_proximity_prior,
    apply_ball_player_proximity_prior_from_files,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ball_track_payload(frames: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "fps": 10.0,
        "source": "wasb",
        "frames": frames,
        "bounces": [],
    }


def _tracks_payload(*, fps: float, player_boxes_by_frame: dict[int, list[tuple[float, float, float, float]]]) -> dict:
    max_players = max((len(boxes) for boxes in player_boxes_by_frame.values()), default=0)
    players = []
    for player_idx in range(max_players):
        frames = []
        for frame_idx, boxes in sorted(player_boxes_by_frame.items()):
            if player_idx >= len(boxes):
                continue
            x1, y1, x2, y2 = boxes[player_idx]
            frames.append({"t": frame_idx / fps, "bbox": [x1, y1, x2, y2], "world_xy": [0.0, 0.0], "conf": 0.9})
        players.append({"id": player_idx + 1, "side": "near", "role": "left", "frames": frames})
    return {"schema_version": 1, "fps": fps, "players": players, "rally_spans": []}


def test_soft_prior_downweights_confidence_smoothly_without_deleting_candidates(tmp_path: Path) -> None:
    fps = 10.0
    track_path = tmp_path / "ball_track.json"
    _write_json(
        track_path,
        _ball_track_payload(
            [
                {"t": 0 / fps, "xy": [5.0, 5.0], "conf": 0.8, "visible": True},
                {"t": 1 / fps, "xy": [24.142, 5.0], "conf": 0.8, "visible": True},
                {"t": 2 / fps, "xy": [100.0, 100.0], "conf": 0.8, "visible": True},
                {"t": 3 / fps, "xy": [5.0, 5.0], "conf": 0.8, "visible": False},
                {"t": 4 / fps, "xy": [5.0, 5.0], "conf": 0.8, "visible": True},
            ]
        ),
    )
    tracks_path = tmp_path / "tracks.json"
    player_box = (0.0, 0.0, 10.0, 10.0)
    _write_json(
        tracks_path,
        _tracks_payload(
            fps=fps,
            player_boxes_by_frame={0: [player_box], 1: [player_box], 2: [player_box], 3: [player_box]},
        ),
    )

    config = BallPlayerProximityPriorConfig(strength=0.5, influence_diag_fraction=2.0)
    result = apply_ball_player_proximity_prior(ball_track_path=track_path, tracks_path=tracks_path, config=config)

    output = result["ball_track"]
    assert len(output["frames"]) == 5
    assert [frame["visible"] for frame in output["frames"]] == [True, True, True, False, True]
    assert [frame["xy"] for frame in output["frames"]] == [
        [5.0, 5.0],
        [24.142, 5.0],
        [100.0, 100.0],
        [5.0, 5.0],
        [5.0, 5.0],
    ]

    frame_reports = {row["frame"]: row for row in result["report"]["frames"]}
    assert frame_reports[0]["factor"] == pytest.approx(0.5, abs=1e-6)
    assert 0.5 < frame_reports[1]["factor"] < 1.0
    assert frame_reports[2]["factor"] == pytest.approx(1.0)
    assert frame_reports[4]["factor"] == pytest.approx(1.0)
    assert output["frames"][0]["conf"] == pytest.approx(0.4)
    assert output["frames"][1]["conf"] == pytest.approx(0.8 * frame_reports[1]["factor"])
    assert output["frames"][2]["conf"] == pytest.approx(0.8)
    assert output["frames"][4]["conf"] == pytest.approx(0.8)
    assert result["report"]["additive_safe"]["frame_count_preserved"] is True
    assert result["report"]["additive_safe"]["visible_flags_preserved"] is True
    assert result["report"]["additive_safe"]["only_confidence_changed"] is True
    assert result["report"]["downstream_thresholding_required"] is True


def test_soft_prior_uses_nearest_player_box_for_factor(tmp_path: Path) -> None:
    fps = 10.0
    track_path = tmp_path / "ball_track.json"
    _write_json(
        track_path,
        _ball_track_payload([{"t": 0.0, "xy": [105.0, 105.0], "conf": 0.9, "visible": True}]),
    )
    tracks_path = tmp_path / "tracks.json"
    _write_json(
        tracks_path,
        _tracks_payload(
            fps=fps,
            player_boxes_by_frame={
                0: [
                    (0.0, 0.0, 10.0, 10.0),
                    (100.0, 100.0, 110.0, 110.0),
                ]
            },
        ),
    )

    result = apply_ball_player_proximity_prior(
        ball_track_path=track_path,
        tracks_path=tracks_path,
        config=BallPlayerProximityPriorConfig(strength=0.4, influence_diag_fraction=1.0),
    )

    row = result["report"]["frames"][0]
    assert row["distance_to_nearest_player_box_px"] == pytest.approx(0.0)
    assert row["factor"] == pytest.approx(0.6)
    assert result["ball_track"]["frames"][0]["conf"] == pytest.approx(0.54)


def test_soft_prior_from_files_writes_schema_valid_ball_track_and_report(tmp_path: Path) -> None:
    fps = 10.0
    track_path = tmp_path / "ball_track.json"
    _write_json(
        track_path,
        _ball_track_payload([{"t": 0.0, "xy": [5.0, 5.0], "conf": 0.8, "visible": True}]),
    )
    tracks_path = tmp_path / "tracks.json"
    _write_json(tracks_path, _tracks_payload(fps=fps, player_boxes_by_frame={0: [(0.0, 0.0, 10.0, 10.0)]}))
    out_track = tmp_path / "out" / "ball_track.json"
    out_report = tmp_path / "out" / "player_proximity_prior_report.json"

    report = apply_ball_player_proximity_prior_from_files(
        ball_track_path=track_path,
        tracks_path=tracks_path,
        out_ball_track_path=out_track,
        out_report_path=out_report,
        config=BallPlayerProximityPriorConfig(strength=0.25, influence_diag_fraction=1.5),
    )

    loaded = load_ball_track(out_track)
    assert loaded.frames[0].conf == pytest.approx(0.6)
    saved_report = json.loads(out_report.read_text(encoding="utf-8"))
    assert saved_report == report
    assert saved_report["artifact_type"] == "racketsport_ball_player_proximity_prior"
    assert saved_report["source_ball_track_path"] == str(track_path)
    assert saved_report["source_tracks_path"] == str(tracks_path)


def test_soft_prior_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="strength"):
        BallPlayerProximityPriorConfig(strength=1.0)
    with pytest.raises(ValueError, match="influence_diag_fraction"):
        BallPlayerProximityPriorConfig(influence_diag_fraction=0.0)


def test_apply_ball_player_proximity_prior_cli_writes_artifacts(tmp_path: Path) -> None:
    fps = 10.0
    track_path = tmp_path / "ball_track.json"
    _write_json(
        track_path,
        _ball_track_payload([{"t": 0.0, "xy": [5.0, 5.0], "conf": 0.8, "visible": True}]),
    )
    tracks_path = tmp_path / "tracks.json"
    _write_json(tracks_path, _tracks_payload(fps=fps, player_boxes_by_frame={0: [(0.0, 0.0, 10.0, 10.0)]}))
    out_track = tmp_path / "out" / "ball_track.json"
    out_report = tmp_path / "out" / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/apply_ball_player_proximity_prior.py",
            "--ball-track",
            str(track_path),
            "--tracks",
            str(tracks_path),
            "--out-ball-track",
            str(out_track),
            "--out-report",
            str(out_report),
            "--strength",
            "0.25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert load_ball_track(out_track).frames[0].conf == pytest.approx(0.6)
    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["adjusted_frame_count"] == 1
    assert '"adjusted_frame_count": 1' in completed.stdout


def test_benchmark_prior_script_help_and_strict_holdout_guard(tmp_path: Path) -> None:
    help_completed = subprocess.run(
        [sys.executable, "scripts/racketsport/benchmark_ball_player_proximity_prior_internal.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_completed.returncode == 0
    assert "Burlington/Wolverine" in help_completed.stdout

    out_root = tmp_path / "bench"
    guard_completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/benchmark_ball_player_proximity_prior_internal.py",
            "--clip",
            "outdoor_webcam_iynbd_1500_long_high_baseline",
            "--out-root",
            str(out_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert guard_completed.returncode == 2
    assert "strict held-out" in guard_completed.stderr
