from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from scripts.racketsport import build_court_proposals
from threed.racketsport.court_model_infer import CourtModelCheckpointResolution


def test_build_court_proposals_cli_writes_fail_closed_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    video = "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_court_proposals.py",
            "--video",
            video,
            "--clip",
            "wolverine_mixed_0200_mid_steep_corner",
            "--out-dir",
            str(out_dir),
            "--max-frames",
            "2",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "court_proposals.json").read_text())
    assert payload["artifact_type"] == "racketsport_court_proposals"
    assert payload["verified"] is False
    assert payload["not_cal3_verified"] is True
    assert payload["ranking"]["abstain"] is True
    assert payload["proposals"]
    assert all(proposal["verified"] is False for proposal in payload["proposals"])
    assert all(proposal["not_cal3_verified"] is True for proposal in payload["proposals"])


def test_structured_static_result_ranks_first_but_stays_review_only(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((120, 200, 3), dtype=np.uint8))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test")
    monkeypatch.setattr(
        build_court_proposals,
        "resolve_court_model_checkpoint_path",
        lambda _path: CourtModelCheckpointResolution(checkpoint, "test", "a" * 64),
    )
    monkeypatch.setattr(
        build_court_proposals,
        "infer_static_court_model",
        lambda *_args, **_kwargs: {
            "best_court": {
                "keypoints_xy": {
                    "near_left_corner": [10.0, 100.0],
                    "near_right_corner": [190.0, 100.0],
                    "far_right_corner": [150.0, 20.0],
                    "far_left_corner": [50.0, 20.0],
                },
                "court_confidence": 0.7,
                "hypothesis_margin": 0.3,
                "homography_image_from_court": np.eye(3).tolist(),
                "residual_stats_px": {"median": 2.0, "p95": 4.0},
                "score_components": {"line_alignment": 0.8, "surface_overlap": 0.7},
                "inlier_ratio": 1.0,
                "inlier_observations": [],
                "ignored_observations": [],
                "source": "observation_hypothesis",
                "supported_view": True,
                "measurement_valid": False,
                "authority_state": "review_only",
            },
            "static_motion": {"status": "static"},
            "court_lock": {"schema_version": 1},
            "selected_frame_indices": [0],
        },
    )
    report = build_court_proposals.build_court_proposal_report(
        video=str(image_path),
        clip="fixture",
        max_frames=8,
    ).to_json_dict()
    selected = report["proposals"][0]
    assert selected["source"] == "confidence_aware_structured_court_v31"
    assert selected["verified"] is False
    assert selected["gate"]["auto_usable"] is False
    assert report["ranking"]["selected_proposal_id"] == "proposal_structured_v31_0001"
