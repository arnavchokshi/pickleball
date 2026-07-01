from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.court_keypoint_net import keypoint_labels_from_court_corners


def test_run_court_line_keypoint_video_writes_prediction_overlay_and_metric(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from scripts.racketsport.run_court_line_keypoints import run_court_line_keypoint_video

    clip_dir = tmp_path / "synthetic_clip"
    labels_dir = clip_dir / "labels"
    labels_dir.mkdir(parents=True)
    video_path = clip_dir / "source.mp4"
    corners = {
        "near_left": [120.0, 360.0],
        "near_right": [620.0, 300.0],
        "far_right": [380.0, 90.0],
        "far_left": [80.0, 130.0],
    }
    labels = keypoint_labels_from_court_corners(corners)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (720, 420))
    assert writer.isOpened()
    for _ in range(5):
        frame = np.zeros((420, 720, 3), dtype=np.uint8)
        frame[:, :] = (36, 42, 45)
        for start, end in (
            ("near_left_corner", "near_right_corner"),
            ("far_left_corner", "far_right_corner"),
            ("near_left_corner", "far_left_corner"),
            ("near_right_corner", "far_right_corner"),
            ("near_nvz_left", "near_nvz_right"),
            ("far_nvz_left", "far_nvz_right"),
            ("net_left_sideline", "net_right_sideline"),
            ("near_baseline_center", "far_baseline_center"),
        ):
            cv2.line(frame, _point(labels[start]), _point(labels[end]), (245, 245, 245), 8, cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    label_path = labels_dir / "court_corners.json"
    label_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "annotation": {
                    "items": [
                        {
                            "frame": "frame_000001.jpg",
                            "court_corners": corners,
                            "review_id": "synthetic",
                            "source": "test",
                            "status": "corrected_unverified",
                        }
                    ],
                    "notes": "",
                    "review_imports": [],
                    "target_file": "",
                    "teacher_payload": {},
                },
                "frames": {
                    "frame_dir": str(clip_dir / "frames"),
                    "frame_count": 5,
                    "frames": [],
                    "manifest_path": "",
                    "sample_every_frames": 1,
                    "source_duration_s": 5 / 30,
                    "source_fps": 30,
                    "source_resolution": [720, 420],
                },
            }
        ),
        encoding="utf-8",
    )
    prediction_path = tmp_path / "court_keypoints.json"
    overlay_path = tmp_path / "court_keypoints_overlay.mp4"
    summary_path = tmp_path / "court_keypoints_summary.json"

    summary = run_court_line_keypoint_video(
        video_path=video_path,
        out_path=prediction_path,
        overlay_out_path=overlay_path,
        summary_out_path=summary_path,
        label_corners_path=label_path,
    )

    assert prediction_path.is_file()
    assert overlay_path.is_file()
    assert summary_path.is_file()
    assert summary["gate"]["passed"] is True
    assert summary["median_keypoint_reprojection_px"] < 5.0
    assert summary["overlay_frame_count"] == 5
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert len(payload["frames"]) == 5
    assert len(payload["frames"][1]["keypoints"]) == 15


def _point(xy: list[float]) -> tuple[int, int]:
    return (int(round(xy[0])), int(round(xy[1])))
