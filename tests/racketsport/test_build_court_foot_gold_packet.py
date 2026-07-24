from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from threed.racketsport.court_foot_gold import (
    GoldClipSpec,
    build_gold_packet,
    score_gold_review,
)


SCRIPT = Path("scripts/racketsport/build_court_foot_gold_packet.py")


def test_packet_builds_prelabels_and_exact_foot_error_budget(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (320, 240),
    )
    for frame_index in range(10):
        frame = np.full((240, 320, 3), 20 + frame_index, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    homography = [[20.0, 0.0, 160.0], [0.0, -10.0, 120.0], [0.0, 0.0, 1.0]]
    (tmp_path / "court_lock.json").write_text(
        json.dumps({"homography_image_from_court": homography})
    )
    frames = [
        {"frame_idx": index, "bbox": [100.0, 80.0, 140.0, 180.0]}
        for index in range(10)
    ]
    (artifacts / "tracks.json").write_text(
        json.dumps({"players": [{"id": 7, "role": "left", "side": "near", "frames": frames}]})
    )
    body_frames = [
        {
            "frame_idx": index,
            "keypoints": [
                {"name": "left_ankle", "xy_px": [112.0, 168.0]},
                {"name": "left_heel", "xy_px": [111.0, 178.0]},
                {"name": "left_toe", "xy_px": [116.0, 180.0]},
                {"name": "right_ankle", "xy_px": [130.0, 168.0]},
                {"name": "right_heel", "xy_px": [129.0, 176.0]},
                {"name": "right_toe", "xy_px": [134.0, 177.0]},
            ],
        }
        for index in range(10)
    ]
    (artifacts / "sam3d_keypoints_2d.json").write_text(
        json.dumps({"players": [{"id": 7, "frames": body_frames}]})
    )

    packet = build_gold_packet(
        [
            GoldClipSpec(
                clip_id="fixture",
                video_path=video_path,
                court_lock_path=tmp_path / "court_lock.json",
                artifacts_dir=artifacts,
            )
        ],
        tmp_path / "packet",
        frames_per_clip=3,
    )

    assert packet["frame_count"] == 3
    assert (tmp_path / "packet/START_HERE.html").is_file()
    player = packet["clips"][0]["frames"][0]["players"][0]
    assert player["points"]["left_contact"] == [116.0, 180.0]
    assert player["support_foot"] == "left"

    review_frames = {}
    for row in packet["clips"][0]["frames"]:
        review_frames[row["frame_id"]] = {
            "status": "accepted",
            "court_points": {},
            "players": {
                "7": {
                    "support_foot": "left",
                    "contact_state": "planted",
                    "points": {"left_contact": {"xy": [118.0, 180.0], "occluded": False}},
                }
            },
        }
    report = score_gold_review(packet, {"frames": review_frames})

    assert report["matching_policy"] == "exact_player_id_exact_support_foot_exact_semantic_name"
    assert report["sample_count"] == 3
    assert report["manual_correction_sample_count"] == 3
    assert report["accepted_prelabel_sample_count"] == 0
    assert report["review_coverage"]["accepted_frame_count"] == 3
    assert report["review_coverage"]["manual_foot_point_count"] == 3
    assert report["summary_manual_corrections"]["sample_count"] == 3
    assert {sample["reference_source"] for sample in report["samples"]} == {"manual_correction"}
    assert report["summary"]["calibration_error_m"]["max"] < 1.0e-6
    assert report["summary"]["foot_localization_error_m"]["median"] > 0.0


def test_gold_packet_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "compact court/foot human-reference packet" in completed.stdout
