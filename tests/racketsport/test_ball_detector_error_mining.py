from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.ball_detector_error_mining import (
    DetectorTrackInput,
    mine_ball_detector_errors,
)


def _reviewed_boxes_payload(
    *,
    clip_id: str,
    frame_count: int,
    ball_frames: dict[int, tuple[float, float, float, float]],
) -> dict[str, object]:
    frames = []
    for frame_index in range(frame_count):
        boxes = []
        bbox = ball_frames.get(frame_index)
        if bbox is not None:
            x, y, width, height = bbox
            boxes.append(
                {
                    "track_id": 7,
                    "label": "ball",
                    "frame_index": frame_index,
                    "bbox_xyxy": [x, y, x + width, y + height],
                    "bbox_xywh": [x, y, width, height],
                    "keyframe": True,
                    "occluded": False,
                    "source": "manual",
                }
            )
        frames.append({"frame_index": frame_index, "boxes": boxes})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": clip_id,
        "source_format": "cvat_video_1_1",
        "source_path": f"{clip_id}.zip",
        "task": {
            "task_id": 42,
            "name": clip_id,
            "size": frame_count,
            "mode": "interpolation",
            "start_frame": 0,
            "stop_frame": frame_count - 1,
            "original_size": [1920, 1080],
            "source": f"{clip_id}.mp4",
        },
        "frames": frames,
        "tracks": [
            {
                "track_id": 7,
                "label": "ball",
                "visible_box_count": len(ball_frames),
                "outside_box_count": frame_count - len(ball_frames),
                "keyframe_count": len(ball_frames),
                "first_visible_frame": min(ball_frames) if ball_frames else None,
                "last_visible_frame": max(ball_frames) if ball_frames else None,
            }
        ],
        "summary": {
            "frame_count": frame_count,
            "visible_box_count": len(ball_frames),
            "outside_box_count": frame_count - len(ball_frames),
            "labels": ["ball"],
            "track_count_by_label": {"ball": 1},
            "visible_box_count_by_label": {"ball": len(ball_frames)},
        },
    }


def _write_reviewed_boxes(
    root: Path,
    *,
    clip_id: str,
    frame_count: int,
    ball_frames: dict[int, tuple[float, float, float, float]],
) -> None:
    clip_dir = root / clip_id
    clip_dir.mkdir(parents=True)
    clip_dir.joinpath("reviewed_boxes.json").write_text(
        json.dumps(_reviewed_boxes_payload(clip_id=clip_id, frame_count=frame_count, ball_frames=ball_frames)),
        encoding="utf-8",
    )


def _write_ball_track(path: Path, *, fps: float, visible_frames: dict[int, tuple[float, float, float]]) -> None:
    frame_count = max(visible_frames, default=0) + 1
    frames = []
    for frame_index in range(frame_count):
        if frame_index in visible_frames:
            x, y, conf = visible_frames[frame_index]
            frames.append(
                {
                    "t": frame_index / fps,
                    "xy": [x, y],
                    "conf": conf,
                    "visible": True,
                    "approx": False,
                }
            )
        else:
            frames.append(
                {
                    "t": frame_index / fps,
                    "xy": [0.0, 0.0],
                    "conf": 0.0,
                    "visible": False,
                    "approx": False,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "fps": fps, "source": "tracknet", "frames": frames, "bounces": []}),
        encoding="utf-8",
    )


def test_mine_ball_detector_errors_writes_compatible_plan_and_keeps_validation_held_out(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(
        cvat_root,
        clip_id="clip_train",
        frame_count=7,
        ball_frames={
            1: (95.0, 195.0, 10.0, 10.0),
            2: (115.0, 205.0, 10.0, 10.0),
            5: (295.0, 395.0, 10.0, 10.0),
        },
    )
    _write_reviewed_boxes(cvat_root, clip_id="clip_val", frame_count=4, ball_frames={1: (45.0, 45.0, 10.0, 10.0)})
    train_track = tmp_path / "tracks" / "train" / "ball_track.json"
    val_track = tmp_path / "tracks" / "val" / "ball_track.json"
    _write_ball_track(
        train_track,
        fps=60.0,
        visible_frames={
            0: (20.0, 20.0, 0.81),
            1: (100.0, 200.0, 0.91),
            3: (25.0, 25.0, 0.82),
            4: (26.0, 26.0, 0.83),
            5: (500.0, 500.0, 0.92),
        },
    )
    _write_ball_track(val_track, fps=60.0, visible_frames={0: (5.0, 5.0, 0.7), 1: (50.0, 50.0, 0.9)})

    out_json = tmp_path / "plan.json"
    plan = mine_ball_detector_errors(
        cvat_root=cvat_root,
        tracks=[
            DetectorTrackInput(clip="clip_train", candidate="tracknet", path=train_track, split="train"),
            DetectorTrackInput(clip="clip_val", candidate="tracknet", path=val_track, split="val"),
        ],
        out_json=out_json,
        radius_px=20.0,
    )

    assert out_json.is_file()
    assert plan["artifact_type"] == "racketsport_ball_hard_negative_iteration_plan"
    assert plan["status"] == "TESTED-ON-REAL-DATA"
    assert plan["ball_verified"] is False
    assert plan["promotion_claimed"] is False
    assert plan["train_clips"] == ["clip_train"]
    assert plan["validation_clips"] == ["clip_val"]
    train = plan["clips"]["clip_train"]
    assert train["split_role"] == "train_hard_negative_candidate"
    assert train["hard_negative_hidden_fp_ranges"] == [
        {"start": 0, "end": 0, "count": 1, "max_conf": 0.81},
        {"start": 3, "end": 4, "count": 2, "max_conf": 0.83},
    ]
    assert train["visible_miss_ranges"] == [{"start": 2, "end": 2, "count": 1}]
    assert train["visible_mislocalized_ranges"] == [{"start": 5, "end": 5, "count": 1, "max_distance_px": 223.607}]
    assert train["metrics"]["false_positive_count"] == 3
    assert train["metrics"]["false_negative_count_at_radius"] == 2
    assert train["metrics"]["true_positive_count_at_radius"] == 1
    val = plan["clips"]["clip_val"]
    assert val["split_role"] == "validation_only_do_not_train"
    assert val["hard_negative_hidden_fp_ranges"] == [{"start": 0, "end": 0, "count": 1, "max_conf": 0.7}]


def test_mine_ball_detector_errors_cli_writes_json_summary(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(cvat_root, clip_id="clip_train", frame_count=2, ball_frames={1: (10.0, 10.0, 4.0, 4.0)})
    track = tmp_path / "ball_track.json"
    _write_ball_track(track, fps=60.0, visible_frames={0: (30.0, 30.0, 0.75)})
    out_json = tmp_path / "plan.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/mine_ball_detector_errors.py",
            "--cvat-root",
            str(cvat_root),
            "--track",
            f"clip_train:train:tracknet={track}",
            "--out-json",
            str(out_json),
            "--radius-px",
            "20",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)
    assert output["out_json"] == str(out_json)
    assert output["summary"]["total_false_positive_count"] == 1
    assert json.loads(out_json.read_text(encoding="utf-8"))["clips"]["clip_train"]["visible_miss_ranges"] == [
        {"start": 1, "end": 1, "count": 1}
    ]
