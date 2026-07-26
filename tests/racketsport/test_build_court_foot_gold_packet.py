from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from threed.racketsport.court_foot_gold import (
    GoldClipSpec,
    build_gold_packet,
    build_stabilization_review_packet,
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
                {"name": "left_big_toe_tip", "index": 15, "xy_px": [116.0, 180.0]},
                {"name": "left_small_toe_tip", "index": 16, "xy_px": [117.0, 179.0]},
                {"name": "right_ankle", "xy_px": [130.0, 168.0]},
                {"name": "right_heel", "xy_px": [129.0, 176.0]},
                {"name": "right_big_toe_tip", "index": 18, "xy_px": [134.0, 177.0]},
                {"name": "right_small_toe_tip", "index": 19, "xy_px": [135.0, 176.0]},
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
    assert player["points"]["left_contact"] == [116.5, 179.5]
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


def test_stabilization_packet_locks_eight_moments_per_category(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    frame_dir = source_dir / "frames"
    frame_dir.mkdir(parents=True)
    frames = []
    category_y = [3.0] * 8 + [2.0] * 8 + [2.3] * 8
    for frame_index, y_coord in enumerate(category_y):
        image_name = f"frame_{frame_index:03d}.jpg"
        assert cv2.imwrite(
            str(frame_dir / image_name),
            np.full((24, 32, 3), frame_index, dtype=np.uint8),
        )
        frames.append(
            {
                "frame_id": f"fixture:{frame_index}",
                "frame_index": frame_index,
                "image": f"frames/{image_name}",
                "players": [
                    {
                        "player_id": 7,
                        "support_foot": "left",
                        "contact_state": "planted",
                        "prelabel_source": "sam3d_body_foot_keypoints",
                        "points": {"left_contact": [0.0, y_coord]},
                    }
                ],
            }
        )
    source_packet = {
        "artifact_type": "racketsport_court_foot_human_reference_packet",
        "schema_version": 1,
        "verified": False,
        "clips": [
            {
                "clip_id": "fixture",
                "automatic_homography_image_from_court": np.eye(3).tolist(),
                "frames": frames,
            }
        ],
        "frame_count": len(frames),
    }
    source_path = source_dir / "review_packet.json"
    source_path.write_text(json.dumps(source_packet), encoding="utf-8")
    template = tmp_path / "template.html"
    template.write_text("<script>__PACKET_JSON__</script>", encoding="utf-8")

    packet = build_stabilization_review_packet(
        source_path,
        tmp_path / "stabilization",
        template_path=template,
    )

    assert packet["frame_count"] == 24
    assert packet["stabilization_review"]["locked_for_final_selection"] is True
    assert packet["stabilization_review"]["candidate_outputs_used_for_selection"] is False
    assert packet["stabilization_review"]["category_counts"] == {
        "clear_outside": 8,
        "line_or_inside": 8,
        "ambiguous_or_dynamic": 8,
    }
    selected = [frame for clip in packet["clips"] for frame in clip["frames"]]
    assert all(len(frame["players"]) == 1 for frame in selected)
    assert len(list((tmp_path / "stabilization/frames").glob("*.jpg"))) == 24


def test_legacy_index_16_right_toe_is_recovered_as_left_small_toe() -> None:
    from threed.racketsport import court_foot_gold

    payload = {
        "players": [
            {
                "id": 4,
                "frames": [
                    {
                        "frame_idx": 9,
                        "keypoints": [
                            {"name": "left_toe", "index": 15, "xy_px": [10.0, 20.0]},
                            {"name": "right_toe", "index": 16, "xy_px": [14.0, 22.0]},
                            {"name": "right_heel", "index": 20, "xy_px": [40.0, 21.0]},
                        ],
                    }
                ],
            }
        ]
    }

    points = court_foot_gold._foot_index(payload)[9][4]

    assert points["left_big_toe_tip"] == [10.0, 20.0]
    assert points["left_small_toe_tip"] == [14.0, 22.0]
    assert points["left_toe"] == [12.0, 21.0]
    assert "right_toe" not in points


def test_legacy_stabilization_packet_cannot_be_scored() -> None:
    with pytest.raises(ValueError, match="authoritative MHR70 foot semantics"):
        score_gold_review(
            {"artifact_type": "racketsport_foot_anchor_stabilization_review_packet"},
            {"frames": {}},
        )
