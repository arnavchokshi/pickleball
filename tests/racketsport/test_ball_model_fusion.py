from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_model_fusion import fuse_ball_tracks_with_verifiers
from threed.racketsport.schemas import BallTrack, validate_artifact_file


def _write_track(path: Path, frames: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "fps": 30.0, "source": "tracknet", "frames": frames, "bounces": []}),
        encoding="utf-8",
    )


def _frames(points: list[tuple[float, float] | None]) -> list[dict[str, object]]:
    frames = []
    for index, point in enumerate(points):
        visible = point is not None
        frames.append(
            {
                "t": index / 30.0,
                "xy": [point[0], point[1]] if point is not None else [0.0, 0.0],
                "conf": 1.0 if visible else 0.0,
                "visible": visible,
            }
        )
    return frames


def test_fusion_keeps_stable_track_adds_verifier_consensus_and_suppresses_unconfirmed_primary(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    stable = tmp_path / "stable.json"
    verifier_a = tmp_path / "verifier_a.json"
    verifier_b = tmp_path / "verifier_b.json"
    _write_track(primary, _frames([(10, 10), (20, 10), (200, 200), None, (40, 10)]))
    _write_track(stable, _frames([(10, 10), None, None, None, None]))
    _write_track(verifier_a, _frames([None, (21, 10), None, (30, 10), None]))
    _write_track(verifier_b, _frames([None, None, None, (32, 12), None]))

    payload, summary = fuse_ball_tracks_with_verifiers(
        primary_ball_track_path=primary,
        stable_ball_track_path=stable,
        verifier_ball_track_paths=[verifier_a, verifier_b],
        outlier_distance_px=8.0,
    )

    fused = BallTrack.model_validate(payload)
    assert fused.frames[0].visible is True
    assert fused.frames[0].xy == [10.0, 10.0]
    assert fused.frames[1].visible is True
    assert fused.frames[1].xy == [20.0, 10.0]
    assert fused.frames[2].visible is False
    assert fused.frames[3].visible is True
    assert fused.frames[3].xy == pytest.approx([31.0, 11.0])
    assert fused.frames[3].approx is True
    assert fused.frames[4].visible is False
    assert summary["uses_human_clicks"] is False
    assert summary["kept_stable_count"] == 1
    assert summary["kept_primary_consensus_count"] == 1
    assert summary["added_verifier_consensus_count"] == 1
    assert summary["suppressed_primary_count"] == 2


def test_fusion_can_veto_unverified_stable_backbone_frame(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    stable = tmp_path / "stable.json"
    verifier = tmp_path / "verifier.json"
    _write_track(primary, _frames([None, (20, 10), (40, 10)]))
    _write_track(stable, _frames([(300, 300), (20, 10), (400, 400)]))
    _write_track(verifier, _frames([None, (22, 11), None]))

    payload, summary = fuse_ball_tracks_with_verifiers(
        primary_ball_track_path=primary,
        stable_ball_track_path=stable,
        verifier_ball_track_paths=[verifier],
        outlier_distance_px=8.0,
        require_stable_verifier_support=True,
    )

    fused = BallTrack.model_validate(payload)
    assert fused.frames[0].visible is False
    assert fused.frames[1].visible is True
    assert fused.frames[1].xy == [20.0, 10.0]
    assert fused.frames[2].visible is False
    assert summary["vetoed_stable_count"] == 2
    assert summary["kept_stable_count"] == 1


def test_fusion_cli_writes_schema_valid_output(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    stable = tmp_path / "stable.json"
    verifier_a = tmp_path / "verifier_a.json"
    verifier_b = tmp_path / "verifier_b.json"
    out = tmp_path / "fused.json"
    summary_out = tmp_path / "summary.json"
    _write_track(primary, _frames([(10, 10), (20, 10), (200, 200)]))
    _write_track(stable, _frames([(10, 10), None, None]))
    _write_track(verifier_a, _frames([None, (21, 10), None]))
    _write_track(verifier_b, _frames([None, (20, 11), None]))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/fuse_ball_tracks.py",
            "--primary-ball-track",
            str(primary),
            "--stable-ball-track",
            str(stable),
            "--verifier-ball-track",
            str(verifier_a),
            "--verifier-ball-track",
            str(verifier_b),
            "--outlier-distance-px",
            "8",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["uses_human_clicks"] is False
    assert isinstance(validate_artifact_file("ball_track", out), BallTrack)
