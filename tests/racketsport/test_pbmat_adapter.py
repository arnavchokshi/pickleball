from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.pbmat_adapter import (
    decode_pbmat_heatmap_candidates,
    pbmat_predictions_to_ball_track,
    remap_crop_refined_xy,
    write_ball_track_from_pbmat_predictions,
)
from threed.racketsport.schemas import BallTrack, validate_artifact_file


def _prediction_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_pbmat_predictions",
        "source_mode": "pbmat_json",
        "fps": 60.0,
        "image_size": [1920, 1080],
        "output_stride": 4,
        "model": {"id": "pbmat_hybrid", "checkpoint_sha256": "abc123"},
        "frames": [
            {
                "frame_index": 12,
                "t": 0.2,
                "visibility_score": 0.91,
                "blur_score": 0.3,
                "selected_candidate": 0,
                "candidates": [
                    {
                        "xy": [101.0, 202.0],
                        "confidence": 0.72,
                        "source": "coarse_heatmap",
                        "refined_xy": [103.25, 204.5],
                        "refined_confidence": 0.86,
                    }
                ],
            }
        ],
    }


def test_pbmat_predictions_to_ball_track_uses_refined_candidate_and_metadata(tmp_path: Path) -> None:
    out = tmp_path / "ball_track.json"
    meta = tmp_path / "pbmat_run.json"

    metadata = write_ball_track_from_pbmat_predictions(_prediction_payload(), out=out, metadata_out=meta)

    assert metadata["source_mode"] == "pbmat_json"
    assert metadata["frame_count"] == 1
    assert metadata["visible_frame_count"] == 1
    assert metadata["not_ground_truth"] is True
    assert metadata["verified"] is False
    assert metadata["model"] == {"id": "pbmat_hybrid", "checkpoint_sha256": "abc123"}
    ball_track = BallTrack.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert ball_track.source == "pbmat"
    assert ball_track.frames[0].t == pytest.approx(0.2)
    assert ball_track.frames[0].xy == [103.25, 204.5]
    assert ball_track.frames[0].conf == pytest.approx(0.86)
    assert ball_track.frames[0].visible is True
    assert isinstance(validate_artifact_file("ball_track", out), BallTrack)


def test_pbmat_predictions_to_ball_track_hides_low_visibility_frames() -> None:
    payload = _prediction_payload()
    frame = payload["frames"][0]  # type: ignore[index]
    frame["visibility_score"] = 0.2

    ball_track = pbmat_predictions_to_ball_track(payload, visibility_threshold=0.5)

    assert ball_track["source"] == "pbmat"
    assert ball_track["frames"][0]["visible"] is False
    assert ball_track["frames"][0]["conf"] == pytest.approx(0.0)
    assert ball_track["frames"][0]["approx"] is False


def test_decode_heatmap_topk_applies_stride_offsets_and_nms() -> None:
    heatmap = [[0.0, 0.1, 0.0], [0.2, 0.95, 0.93], [0.0, 0.1, 0.0]]
    offset_x = [[0.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.0]]
    offset_y = [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]]

    candidates = decode_pbmat_heatmap_candidates(
        heatmap=heatmap,
        offset_x=offset_x,
        offset_y=offset_y,
        output_stride=4,
        top_k=2,
        nms_radius=1,
    )

    assert len(candidates) == 1
    assert candidates[0].xy == pytest.approx([5.0, 2.0])
    assert candidates[0].confidence == pytest.approx(0.95)
    assert candidates[0].source == "coarse_heatmap"


def test_remap_crop_refinement_clips_to_source_frame() -> None:
    xy = remap_crop_refined_xy(
        crop_origin_xy=[1800.0, 980.0],
        crop_size=[256, 256],
        refined_crop_xy=[300.0, -20.0],
        image_size=[1920, 1080],
    )

    assert xy == [1919.0, 980.0]


def test_run_pbmat_ball_cli_converts_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "pbmat_predictions.json"
    out = tmp_path / "ball_track.json"
    meta = tmp_path / "pbmat_run.json"
    predictions.write_text(json.dumps(_prediction_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_pbmat_ball.py",
            "--predictions-json",
            str(predictions),
            "--out",
            str(out),
            "--metadata-out",
            str(meta),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["frame_count"] == 1
    assert BallTrack.model_validate(json.loads(out.read_text(encoding="utf-8"))).source == "pbmat"
