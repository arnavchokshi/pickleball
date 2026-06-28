from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.schemas import BallTrack
from threed.racketsport.totnet_adapter import (
    totnet_predictions_to_ball_track,
    write_ball_track_from_totnet_predictions,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_totnet_predictions",
        "fps": 30.0,
        "image_size": [1920, 1080],
        "input_size": [512, 288],
        "model": {"id": "totnet_tennis_5f_288x512"},
        "frames": [
            {"frame_index": 0, "xy": None, "confidence": 0.0, "visible": False},
            {"frame_index": 4, "xy": [120.5, 240.25], "confidence": 0.71, "visible": True},
        ],
    }


def test_totnet_predictions_to_ball_track_writes_schema_valid_source() -> None:
    ball_track = totnet_predictions_to_ball_track(_payload(), confidence_threshold=0.5)

    parsed = BallTrack.model_validate(ball_track)
    assert parsed.source == "totnet"
    assert len(parsed.frames) == 2
    assert parsed.frames[0].visible is False
    assert parsed.frames[1].t == pytest.approx(4.0 / 30.0)
    assert parsed.frames[1].xy == [120.5, 240.25]
    assert parsed.frames[1].conf == pytest.approx(0.71)


def test_write_totnet_predictions_applies_confidence_threshold(tmp_path: Path) -> None:
    payload = _payload()
    out = tmp_path / "ball_track.json"
    meta = tmp_path / "totnet_run.json"

    summary = write_ball_track_from_totnet_predictions(
        payload,
        out=out,
        metadata_out=meta,
        confidence_threshold=0.8,
    )

    assert summary["visible_frame_count"] == 0
    assert summary["model"] == {"id": "totnet_tennis_5f_288x512"}
    assert json.loads(out.read_text(encoding="utf-8"))["frames"][1]["visible"] is False
    assert json.loads(meta.read_text(encoding="utf-8"))["confidence_threshold"] == pytest.approx(0.8)
