from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

import threed.racketsport.ball_tracknet_cvat_dataset as dataset_module
from threed.racketsport.ball_tracknet_cvat_dataset import (
    build_ball_tracknet_cvat_dataset,
    dense_tracknet_labels_from_cvat,
)


def _reviewed_boxes_payload(*, clip_id: str, frame_count: int, ball_frames: dict[int, tuple[float, float, float, float]]) -> dict[str, object]:
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


def _write_reviewed_boxes(root: Path, *, clip_id: str, frame_count: int, ball_frames: dict[int, tuple[float, float, float, float]]) -> None:
    clip_dir = root / clip_id
    clip_dir.mkdir(parents=True)
    payload = _reviewed_boxes_payload(clip_id=clip_id, frame_count=frame_count, ball_frames=ball_frames)
    (clip_dir / "reviewed_boxes.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_yolo_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_cvat_detection_yolo_dataset",
                "split_mode": "by_clip",
                "val_clips": ["clip_val"],
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def _write_hard_negative_plan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_ball_hard_negative_iteration_plan",
                "ball_verified": False,
                "promotion_claimed": False,
                "train_clips": ["clip_train"],
                "validation_clips": ["clip_val"],
                "clips": {
                    "clip_train": {
                        "split_role": "train_hard_negative_candidate",
                        "hard_negative_hidden_fp_ranges": [{"start": 2, "end": 3, "count": 2}],
                    },
                    "clip_val": {
                        "split_role": "validation_only_do_not_train",
                        "hard_negative_hidden_fp_ranges": [{"start": 0, "end": 1, "count": 2}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_dense_tracknet_labels_from_cvat_preserves_hidden_negatives(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed_boxes.json"
    reviewed.write_text(
        json.dumps(
            _reviewed_boxes_payload(
                clip_id="clip_train",
                frame_count=4,
                ball_frames={1: (10.0, 20.0, 8.0, 10.0), 3: (30.0, 40.0, 6.0, 4.0)},
            )
        ),
        encoding="utf-8",
    )

    labels = dense_tracknet_labels_from_cvat(reviewed)

    assert [(row.frame, row.visibility, row.x, row.y, row.source) for row in labels] == [
        (0, 0, 0.0, 0.0, "reviewed_hidden"),
        (1, 1, 14.0, 25.0, "reviewed_cvat_ball_box"),
        (2, 0, 0.0, 0.0, "reviewed_hidden"),
        (3, 1, 33.0, 42.0, "reviewed_cvat_ball_box"),
    ]


def test_build_ball_tracknet_cvat_dataset_writes_disjoint_manifest_csv_and_markdown(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(
        cvat_root,
        clip_id="clip_train",
        frame_count=3,
        ball_frames={0: (100.0, 200.0, 10.0, 8.0), 2: (120.0, 210.0, 6.0, 6.0)},
    )
    _write_reviewed_boxes(
        cvat_root,
        clip_id="clip_val",
        frame_count=2,
        ball_frames={1: (300.0, 400.0, 8.0, 8.0)},
    )
    yolo_manifest = tmp_path / "yolo" / "manifest.json"
    _write_yolo_manifest(
        yolo_manifest,
        [
            {"clip_id": "clip_train", "frame_index": 0, "split": "train"},
            {"clip_id": "clip_train", "frame_index": 2, "split": "train"},
            {"clip_id": "clip_val", "frame_index": 1, "split": "val"},
        ],
    )

    manifest = build_ball_tracknet_cvat_dataset(
        cvat_root=cvat_root,
        yolo_manifest=yolo_manifest,
        out_dir=tmp_path / "out",
        fps=60.0,
    )

    assert manifest["artifact_type"] == "racketsport_ball_tracknet_cvat_dataset"
    assert manifest["status"] == "labels_prepared_frames_not_materialized"
    assert manifest["label_counts"] == {
        "clip_count": 2,
        "frame_count": 5,
        "reviewed_hidden_frame_count": 2,
        "reviewed_visible_ball_frame_count": 3,
    }
    assert manifest["leakage_checks"] == {
        "clips_with_multiple_splits": [],
        "disjoint_clip_splits": True,
        "split_clip_counts": {"train": 1, "val": 1},
    }
    assert [row["clip"] for row in manifest["splits"]["train"]] == ["clip_train"]
    assert [row["clip"] for row in manifest["splits"]["val"]] == ["clip_val"]
    train_csv = Path(manifest["splits"]["train"][0]["csv"])
    with train_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["Frame", "Visibility", "X", "Y"],
        ["0", "1", "105.000", "204.000"],
        ["1", "0", "0.000", "0.000"],
        ["2", "1", "123.000", "213.000"],
    ]
    assert (tmp_path / "out" / "ball_tracknet_cvat_dataset_manifest.json").is_file()
    markdown = (tmp_path / "out" / "ball_tracknet_cvat_dataset_manifest.md").read_text(encoding="utf-8")
    assert "BALL is not verified by this artifact." in markdown
    assert "clip_train" in markdown
    assert "clip_val" in markdown


def test_build_ball_tracknet_cvat_dataset_rejects_split_leakage(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(cvat_root, clip_id="clip_leak", frame_count=1, ball_frames={0: (1.0, 2.0, 3.0, 4.0)})
    yolo_manifest = tmp_path / "yolo" / "manifest.json"
    _write_yolo_manifest(
        yolo_manifest,
        [
            {"clip_id": "clip_leak", "frame_index": 0, "split": "train"},
            {"clip_id": "clip_leak", "frame_index": 0, "split": "val"},
        ],
    )

    with pytest.raises(ValueError, match="split leakage"):
        build_ball_tracknet_cvat_dataset(
            cvat_root=cvat_root,
            yolo_manifest=yolo_manifest,
            out_dir=tmp_path / "out",
        )


def test_build_ball_tracknet_cvat_dataset_can_materialize_frame_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(cvat_root, clip_id="clip_train", frame_count=1, ball_frames={0: (1.0, 2.0, 3.0, 4.0)})
    yolo_manifest = tmp_path / "yolo" / "manifest.json"
    _write_yolo_manifest(yolo_manifest, [{"clip_id": "clip_train", "frame_index": 0, "split": "train"}])
    video = tmp_path / "clip_train.mp4"
    video.write_bytes(b"fake video")
    calls: list[tuple[Path, Path, int]] = []

    def fake_extract_frames(video_path: Path, frame_dir: Path, frame_count: int) -> None:
        calls.append((video_path, frame_dir, frame_count))
        frame_dir.mkdir(parents=True)
        (frame_dir / "0.png").write_bytes(b"fake png")

    def fake_write_median(frame_dir: Path, frame_count: int) -> None:
        assert frame_count == 1
        (frame_dir / "median.npz").write_bytes(b"fake median")

    monkeypatch.setattr(dataset_module, "_extract_frames", fake_extract_frames)
    monkeypatch.setattr(dataset_module, "_write_median", fake_write_median)

    manifest = build_ball_tracknet_cvat_dataset(
        cvat_root=cvat_root,
        yolo_manifest=yolo_manifest,
        out_dir=tmp_path / "out",
        materialize_frames=True,
        video_paths={"clip_train": video},
    )

    row = manifest["splits"]["train"][0]
    assert manifest["status"] == "tracknet_dataset_materialized"
    assert row["frames_materialized"] is True
    assert row["source_video_path"] == str(video)
    assert calls == [(video, Path(row["frame_dir"]), 1)]
    assert (Path(row["frame_dir"]) / "median.npz").is_file()


def test_extract_frames_disables_ffmpeg_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        frame_pattern = Path(command[-1])
        frame_pattern.parent.mkdir(parents=True, exist_ok=True)
        (frame_pattern.parent / "0.png").write_bytes(b"fake png")
        (frame_pattern.parent / "2.png").write_bytes(b"fake png")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dataset_module.subprocess, "run", fake_run)

    dataset_module._extract_frames(tmp_path / "clip.mp4", tmp_path / "frames", 3)

    assert calls
    assert "-nostdin" in calls[0]
    assert calls[0].index("-nostdin") < calls[0].index("-i")


def test_extract_frame_window_disables_ffmpeg_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        frame_pattern = Path(command[-1])
        frame_pattern.parent.mkdir(parents=True, exist_ok=True)
        (frame_pattern.parent / "0.png").write_bytes(b"fake png")
        (frame_pattern.parent / "2.png").write_bytes(b"fake png")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dataset_module.subprocess, "run", fake_run)

    dataset_module._extract_frame_window(tmp_path / "clip.mp4", tmp_path / "frames", 5, 3)

    assert calls
    assert "-nostdin" in calls[0]
    assert calls[0].index("-nostdin") < calls[0].index("-i")


def test_build_ball_tracknet_cvat_dataset_consumes_hard_negative_plan_with_context_and_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(
        cvat_root,
        clip_id="clip_train",
        frame_count=6,
        ball_frames={0: (100.0, 200.0, 10.0, 8.0), 5: (120.0, 210.0, 6.0, 6.0)},
    )
    _write_reviewed_boxes(cvat_root, clip_id="clip_val", frame_count=4, ball_frames={2: (300.0, 400.0, 8.0, 8.0)})
    yolo_manifest = tmp_path / "yolo" / "manifest.json"
    _write_yolo_manifest(
        yolo_manifest,
        [
            {"clip_id": "clip_train", "frame_index": 0, "split": "train"},
            {"clip_id": "clip_train", "frame_index": 5, "split": "train"},
            {"clip_id": "clip_val", "frame_index": 2, "split": "val"},
        ],
    )
    hard_negative_plan = tmp_path / "hard_negative_plan.json"
    _write_hard_negative_plan(hard_negative_plan)
    train_video = tmp_path / "clip_train.mp4"
    val_video = tmp_path / "clip_val.mp4"
    train_video.write_bytes(b"fake train video")
    val_video.write_bytes(b"fake val video")
    full_extract_calls: list[tuple[Path, Path, int]] = []
    window_extract_calls: list[tuple[Path, Path, int, int]] = []

    def fake_extract_frames(video_path: Path, frame_dir: Path, frame_count: int) -> None:
        full_extract_calls.append((video_path, frame_dir, frame_count))
        frame_dir.mkdir(parents=True)
        (frame_dir / "0.png").write_bytes(b"fake png")
        (frame_dir / f"{frame_count - 1}.png").write_bytes(b"fake png")

    def fake_extract_frame_window(video_path: Path, frame_dir: Path, start_frame: int, frame_count: int) -> None:
        window_extract_calls.append((video_path, frame_dir, start_frame, frame_count))
        frame_dir.mkdir(parents=True)
        (frame_dir / "0.png").write_bytes(b"fake png")
        (frame_dir / f"{frame_count - 1}.png").write_bytes(b"fake png")

    def fake_write_median(frame_dir: Path, frame_count: int) -> None:
        (frame_dir / "median.npz").write_bytes(b"fake median")

    monkeypatch.setattr(dataset_module, "_extract_frames", fake_extract_frames)
    monkeypatch.setattr(dataset_module, "_extract_frame_window", fake_extract_frame_window)
    monkeypatch.setattr(dataset_module, "_write_median", fake_write_median)

    manifest = build_ball_tracknet_cvat_dataset(
        cvat_root=cvat_root,
        yolo_manifest=yolo_manifest,
        out_dir=tmp_path / "out",
        materialize_frames=True,
        video_paths={"clip_train": train_video, "clip_val": val_video},
        hard_negative_plan=hard_negative_plan,
        hard_negative_context_frames=1,
        hard_negative_repeat=2,
    )

    assert manifest["status"] == "tracknet_dataset_materialized_hard_negative_augmented"
    assert manifest["hard_negative_plan"]["source_plan"] == str(hard_negative_plan)
    assert manifest["hard_negative_plan"]["context_frames"] == 1
    assert manifest["hard_negative_plan"]["repeat"] == 2
    assert manifest["hard_negative_plan"]["validation_clips_held_out"] == ["clip_val"]
    assert manifest["hard_negative_plan"]["generated_window_count"] == 2
    assert [row["training_sample_type"] for row in manifest["splits"]["train"]] == [
        "dense_clip",
        "hard_negative_oversample",
        "hard_negative_oversample",
    ]
    assert [row["training_sample_type"] for row in manifest["splits"]["val"]] == ["dense_clip"]
    assert window_extract_calls == [
        (train_video, Path(manifest["splits"]["train"][1]["frame_dir"]), 1, 4),
        (train_video, Path(manifest["splits"]["train"][2]["frame_dir"]), 1, 4),
    ]
    assert all(call[0] != val_video for call in window_extract_calls)
    assert len(full_extract_calls) == 2
    next_command = manifest["next_gpu_commands"][0]
    assert "--out-dir <new_empty_out_dir_like_out>" in next_command
    assert next_command.count("--video clip_train=") == 1
    assert next_command.count("--video clip_val=") == 1
    hard_negative_csv = Path(manifest["splits"]["train"][1]["csv"])
    with hard_negative_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["Frame", "Visibility", "X", "Y"],
        ["0", "0", "0.000", "0.000"],
        ["1", "0", "0.000", "0.000"],
        ["2", "0", "0.000", "0.000"],
        ["3", "0", "0.000", "0.000"],
    ]
    markdown = Path(manifest["manifest_md"]).read_text(encoding="utf-8")
    assert "Frame directories were materialized." in markdown


def test_build_ball_tracknet_cvat_dataset_consumes_visible_detector_error_ranges(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(
        cvat_root,
        clip_id="clip_train",
        frame_count=6,
        ball_frames={
            3: (100.0, 200.0, 10.0, 8.0),
            5: (120.0, 210.0, 6.0, 6.0),
        },
    )
    yolo_manifest = tmp_path / "yolo" / "manifest.json"
    _write_yolo_manifest(yolo_manifest, [{"clip_id": "clip_train", "frame_index": 3, "split": "train"}])
    hard_negative_plan = tmp_path / "detector_error_plan.json"
    hard_negative_plan.write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_ball_hard_negative_iteration_plan",
                "ball_verified": False,
                "promotion_claimed": False,
                "train_clips": ["clip_train"],
                "validation_clips": [],
                "clips": {
                    "clip_train": {
                        "split_role": "train_hard_negative_candidate",
                        "hard_negative_hidden_fp_ranges": [{"start": 1, "end": 1, "count": 1}],
                        "visible_miss_ranges": [{"start": 3, "end": 3, "count": 1}],
                        "visible_mislocalized_ranges": [{"start": 5, "end": 5, "count": 1}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_ball_tracknet_cvat_dataset(
        cvat_root=cvat_root,
        yolo_manifest=yolo_manifest,
        out_dir=tmp_path / "out",
        hard_negative_plan=hard_negative_plan,
    )

    assert manifest["hard_negative_plan"]["generated_window_count"] == 3
    assert manifest["hard_negative_plan"]["hard_negative_window_count"] == 1
    assert manifest["hard_negative_plan"]["visible_error_window_count"] == 2
    assert [row["training_sample_type"] for row in manifest["splits"]["train"]] == [
        "dense_clip",
        "hard_negative_oversample",
        "detector_error_oversample",
        "detector_error_oversample",
    ]
    assert [row.get("detector_error_kind") for row in manifest["splits"]["train"][1:]] == [
        "hidden_false_positive",
        "visible_miss",
        "visible_mislocalized",
    ]
    miss_csv = Path(manifest["splits"]["train"][2]["csv"])
    with miss_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [["Frame", "Visibility", "X", "Y"], ["0", "1", "105.000", "204.000"]]


def test_build_ball_tracknet_cvat_dataset_rejects_existing_out_dir_for_hard_negative_plan(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(cvat_root, clip_id="clip_train", frame_count=1, ball_frames={})
    yolo_manifest = tmp_path / "yolo" / "manifest.json"
    _write_yolo_manifest(yolo_manifest, [{"clip_id": "clip_train", "frame_index": 0, "split": "train"}])
    hard_negative_plan = tmp_path / "hard_negative_plan.json"
    _write_hard_negative_plan(hard_negative_plan)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stale.npz").write_bytes(b"old cache")

    with pytest.raises(ValueError, match="new empty out_dir"):
        build_ball_tracknet_cvat_dataset(
            cvat_root=cvat_root,
            yolo_manifest=yolo_manifest,
            out_dir=out_dir,
            hard_negative_plan=hard_negative_plan,
        )


def test_build_ball_tracknet_cvat_dataset_cli_writes_json_and_md(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(cvat_root, clip_id="clip_train", frame_count=1, ball_frames={0: (1.0, 2.0, 3.0, 4.0)})
    yolo_manifest = tmp_path / "yolo" / "manifest.json"
    _write_yolo_manifest(yolo_manifest, [{"clip_id": "clip_train", "frame_index": 0, "split": "train"}])
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_ball_tracknet_cvat_dataset.py",
            "--cvat-root",
            str(cvat_root),
            "--yolo-manifest",
            str(yolo_manifest),
            "--out-dir",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)
    assert output["manifest_json"] == str(out_dir / "ball_tracknet_cvat_dataset_manifest.json")
    assert output["manifest_md"] == str(out_dir / "ball_tracknet_cvat_dataset_manifest.md")
    assert (out_dir / "ball_tracknet_cvat_dataset_manifest.json").is_file()
    assert (out_dir / "ball_tracknet_cvat_dataset_manifest.md").is_file()


def test_build_ball_tracknet_cvat_dataset_cli_accepts_hard_negative_plan(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    _write_reviewed_boxes(cvat_root, clip_id="clip_train", frame_count=4, ball_frames={0: (1.0, 2.0, 3.0, 4.0)})
    _write_reviewed_boxes(cvat_root, clip_id="clip_val", frame_count=2, ball_frames={1: (5.0, 6.0, 7.0, 8.0)})
    yolo_manifest = tmp_path / "yolo" / "manifest.json"
    _write_yolo_manifest(
        yolo_manifest,
        [
            {"clip_id": "clip_train", "frame_index": 0, "split": "train"},
            {"clip_id": "clip_val", "frame_index": 1, "split": "val"},
        ],
    )
    hard_negative_plan = tmp_path / "hard_negative_plan.json"
    _write_hard_negative_plan(hard_negative_plan)
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_ball_tracknet_cvat_dataset.py",
            "--cvat-root",
            str(cvat_root),
            "--yolo-manifest",
            str(yolo_manifest),
            "--out-dir",
            str(out_dir),
            "--hard-negative-plan",
            str(hard_negative_plan),
            "--hard-negative-context-frames",
            "1",
            "--hard-negative-repeat",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)
    manifest = json.loads((out_dir / "ball_tracknet_cvat_dataset_manifest.json").read_text(encoding="utf-8"))
    assert output["status"] == "labels_prepared_hard_negative_augmented_frames_not_materialized"
    assert manifest["hard_negative_plan"]["context_frames"] == 1
    assert manifest["hard_negative_plan"]["repeat"] == 2
    assert manifest["hard_negative_plan"]["generated_window_count"] == 2
