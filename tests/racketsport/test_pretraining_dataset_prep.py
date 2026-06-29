from __future__ import annotations

from pathlib import Path

import pytest

from scripts.racketsport.prepare_tracknetv3_finetune_dataset import (
    TrackNetLabel,
    build_tracknetv3_dataset,
    interpolated_tracknet_labels,
    write_tracknet_csv,
)
from scripts.racketsport.train_court_keypoint_heatmap import court_corner_keypoint_labels


def test_interpolated_tracknet_labels_preserve_clicks_and_fill_short_visible_gaps() -> None:
    items = [
        {"frame_index": 0, "visible": True, "xy_px": [10.0, 20.0]},
        {"frame_index": 4, "visible": True, "xy_px": [18.0, 28.0]},
        {"frame_index": 7, "visible": False, "xy_px": [99.0, 99.0]},
        {"frame_index": 9, "visible": True, "xy_px": [40.0, 50.0]},
    ]

    labels = interpolated_tracknet_labels(items, frame_count=10, max_gap_frames=4)

    assert labels[0] == TrackNetLabel(frame=0, visibility=1, x=10.0, y=20.0, source="human_click")
    assert labels[2] == TrackNetLabel(frame=2, visibility=1, x=14.0, y=24.0, source="interpolated")
    assert labels[4] == TrackNetLabel(frame=4, visibility=1, x=18.0, y=28.0, source="human_click")
    assert labels[7] == TrackNetLabel(frame=7, visibility=0, x=0.0, y=0.0, source="human_hidden")
    assert labels[8].visibility == 0
    assert labels[9] == TrackNetLabel(frame=9, visibility=1, x=40.0, y=50.0, source="human_click")


def test_write_tracknet_csv_uses_official_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "1_01_00_ball.csv"
    write_tracknet_csv(
        csv_path,
        [
            TrackNetLabel(frame=0, visibility=1, x=10.0, y=20.0, source="human_click"),
            TrackNetLabel(frame=1, visibility=0, x=0.0, y=0.0, source="hidden"),
        ],
    )

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "Frame,Visibility,X,Y",
        "0,1,10.000,20.000",
        "1,0,0.000,0.000",
    ]


def test_tracknetv3_overwrite_rejects_broad_paths_before_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_rmtree(path: Path) -> None:
        raise AssertionError(f"rmtree must not run for unsafe path: {path}")

    monkeypatch.setattr("scripts.racketsport.prepare_tracknetv3_finetune_dataset.shutil.rmtree", fail_rmtree)

    with pytest.raises(ValueError, match="refusing to overwrite broad output directory"):
        build_tracknetv3_dataset(
            run_root=Path("/tmp/run-root"),
            review_root=Path("/tmp/review-root"),
            out=Path("/"),
            splits={"train": ()},
            overwrite=True,
        )


def test_tracknetv3_overwrite_rejects_source_tree_paths_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = Path("scripts/racketsport/tracknetv3_dataset_tmp")
    unsafe.mkdir(exist_ok=True)

    def fail_rmtree(path: Path) -> None:
        raise AssertionError(f"rmtree must not run for unsafe path: {path}")

    monkeypatch.setattr("scripts.racketsport.prepare_tracknetv3_finetune_dataset.shutil.rmtree", fail_rmtree)

    try:
        with pytest.raises(ValueError, match="source-controlled project directory"):
            build_tracknetv3_dataset(
                run_root=tmp_path / "run-root",
                review_root=tmp_path / "review-root",
                out=unsafe,
                splits={"train": ()},
                overwrite=True,
            )
    finally:
        unsafe.rmdir()


def test_court_corner_keypoint_labels_map_manual_seed_to_taxonomy(tmp_path: Path) -> None:
    frame = tmp_path / "frame_000001.jpg"
    frame.write_bytes(b"fake")
    payload = {
        "annotation": {
            "items": [
                {
                    "frame": "frame_000001.jpg",
                    "court_corners": {
                        "near_left": [10, 100],
                        "near_right": [90, 100],
                        "far_right": [80, 20],
                        "far_left": [20, 20],
                    },
                }
            ]
        },
        "frames": {"frame_dir": str(tmp_path)},
    }

    labels = court_corner_keypoint_labels(payload)

    assert labels["image_path"] == str(frame)
    assert labels["keypoints"]["near_left_corner"] == pytest.approx([10.0, 100.0])
    assert labels["keypoints"]["near_right_corner"] == pytest.approx([90.0, 100.0])
    assert labels["keypoints"]["far_right_corner"] == pytest.approx([80.0, 20.0])
    assert labels["keypoints"]["far_left_corner"] == pytest.approx([20.0, 20.0])
    assert set(labels["keypoints"]) == {
        "near_left_corner",
        "near_right_corner",
        "far_right_corner",
        "far_left_corner",
    }


def test_court_corner_keypoint_labels_reject_missing_items() -> None:
    with pytest.raises(ValueError, match="court corner item"):
        court_corner_keypoint_labels({"annotation": {"items": []}, "frames": {"frame_dir": "."}})
