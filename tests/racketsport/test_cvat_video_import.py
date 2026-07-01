from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from threed.racketsport.cvat_video import (
    import_cvat_video_zip,
    write_cvat_video_annotations,
    write_person_ground_truth_from_cvat_video,
)
from threed.racketsport.schemas import CvatVideoAnnotations, PersonGroundTruth, validate_artifact_file


def _write_cvat_video_zip(path: Path) -> None:
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>42</id>
      <name>clip task</name>
      <size>3</size>
      <mode>interpolation</mode>
      <start_frame>0</start_frame>
      <stop_frame>2</stop_frame>
      <labels>
        <label><name>player</name></label>
        <label><name>paddle</name></label>
        <label><name>ball</name></label>
      </labels>
      <original_size><width>1920</width><height>1080</height></original_size>
      <source>clip.mp4</source>
    </task>
    <dumped>2026-06-30 00:31:24+00:00</dumped>
  </meta>
  <track id="0" label="player" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="10" ytl="20" xbr="110" ybr="220" z_order="0" />
    <box frame="1" keyframe="0" outside="0" occluded="1" xtl="12" ytl="22" xbr="112" ybr="222" z_order="0" />
  </track>
  <track id="5" label="paddle" source="manual">
    <box frame="1" keyframe="1" outside="0" occluded="0" xtl="300" ytl="400" xbr="330" ybr="450" z_order="0" />
    <box frame="2" keyframe="1" outside="1" occluded="0" xtl="300" ytl="400" xbr="330" ybr="450" z_order="0" />
  </track>
  <track id="8" label="ball" source="manual">
    <box frame="2" keyframe="1" outside="0" occluded="0" xtl="700" ytl="100" xbr="708" ybr="108" z_order="0" />
  </track>
</annotations>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("annotations.xml", xml)


def _write_cvat_video_zip_with_ball_ellipse(path: Path) -> None:
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>43</id>
      <name>ellipse task</name>
      <size>2</size>
      <mode>interpolation</mode>
      <start_frame>0</start_frame>
      <stop_frame>1</stop_frame>
      <labels>
        <label><name>player</name></label>
        <label><name>ball</name></label>
      </labels>
      <original_size><width>640</width><height>360</height></original_size>
      <source>ellipse_clip.mp4</source>
    </task>
  </meta>
  <track id="0" label="player" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="10" ytl="20" xbr="110" ybr="220" z_order="0" />
  </track>
  <track id="4" label="ball" source="manual">
    <ellipse frame="0" keyframe="1" outside="0" occluded="0" cx="300" cy="150" rx="6" ry="4" z_order="0" />
    <ellipse frame="1" keyframe="1" outside="1" occluded="0" cx="300" cy="150" rx="6" ry="4" z_order="0" />
  </track>
</annotations>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("annotations.xml", xml)


def test_import_cvat_video_zip_preserves_player_paddle_and_ball_visible_boxes(tmp_path: Path) -> None:
    zip_path = tmp_path / "annotations_cvat_video.zip"
    _write_cvat_video_zip(zip_path)

    annotations, person_gt = import_cvat_video_zip(zip_path, clip_id="clip_a", fps=59.94)

    assert annotations.artifact_type == "racketsport_cvat_video_annotations"
    assert annotations.source_format == "cvat_video_1_1"
    assert annotations.clip_id == "clip_a"
    assert annotations.task.source == "clip.mp4"
    assert annotations.task.original_size == (1920, 1080)
    assert annotations.summary.track_count_by_label == {"ball": 1, "paddle": 1, "player": 1}
    assert annotations.summary.visible_box_count_by_label == {"ball": 1, "paddle": 1, "player": 2}
    assert annotations.summary.outside_box_count == 1
    assert annotations.frames[1].boxes[0].label == "player"
    assert annotations.frames[1].boxes[1].label == "paddle"
    assert annotations.frames[1].boxes[1].bbox_xywh == pytest.approx((300.0, 400.0, 30.0, 50.0))
    assert [box.label for box in annotations.frames[2].boxes] == ["ball"]

    assert person_gt.artifact_type == "racketsport_person_ground_truth"
    assert person_gt.source_format == "cvat_video_1_1"
    assert person_gt.summary.valid_label_count == 2
    assert person_gt.summary.track_ids == [1]
    assert person_gt.frames[0].frame_index == 0
    assert person_gt.frames[0].source_frame_id == 1
    assert person_gt.frames[0].labels[0].track_id == 1
    assert person_gt.frames[0].labels[0].bbox_xywh == pytest.approx((10.0, 20.0, 100.0, 200.0))


def test_import_cvat_video_zip_converts_ball_ellipses_to_boxes(tmp_path: Path) -> None:
    zip_path = tmp_path / "annotations_cvat_video_ellipse.zip"
    _write_cvat_video_zip_with_ball_ellipse(zip_path)

    annotations, person_gt = import_cvat_video_zip(zip_path, clip_id="clip_ellipse", fps=30)

    assert annotations.summary.track_count_by_label == {"ball": 1, "player": 1}
    assert annotations.summary.visible_box_count_by_label == {"ball": 1, "player": 1}
    assert annotations.summary.outside_box_count == 1
    ball_boxes = [box for box in annotations.frames[0].boxes if box.label == "ball"]
    assert len(ball_boxes) == 1
    assert ball_boxes[0].bbox_xyxy == pytest.approx((294.0, 146.0, 306.0, 154.0))
    assert ball_boxes[0].bbox_xywh == pytest.approx((294.0, 146.0, 12.0, 8.0))
    assert person_gt.summary.valid_label_count == 1


def test_import_cvat_video_zip_respects_max_frame_index(tmp_path: Path) -> None:
    zip_path = tmp_path / "annotations_cvat_video.zip"
    _write_cvat_video_zip(zip_path)

    annotations, person_gt = import_cvat_video_zip(zip_path, clip_id="clip_capped", fps=30, max_frame_index=1)

    assert annotations.task.size == 2
    assert annotations.task.stop_frame == 1
    assert annotations.summary.frame_count == 2
    assert annotations.summary.track_count_by_label == {"ball": 1, "paddle": 1, "player": 1}
    assert annotations.summary.visible_box_count_by_label == {"paddle": 1, "player": 2}
    assert annotations.summary.outside_box_count == 0
    assert len(annotations.frames) == 2
    assert [box.label for box in annotations.frames[1].boxes] == ["player", "paddle"]
    assert person_gt.summary.frame_count == 2
    assert person_gt.summary.valid_label_count == 2


def test_write_cvat_video_annotations_registers_schema(tmp_path: Path) -> None:
    zip_path = tmp_path / "annotations_cvat_video.zip"
    out_path = tmp_path / "reviewed_boxes.json"
    _write_cvat_video_zip(zip_path)
    annotations, _ = import_cvat_video_zip(zip_path, clip_id="clip_a")

    write_cvat_video_annotations(out_path, annotations)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_cvat_video_annotations"
    parsed = validate_artifact_file("cvat_video_annotations", out_path)
    assert isinstance(parsed, CvatVideoAnnotations)


def test_import_cvat_video_cli_writes_reviewed_boxes_and_person_ground_truth(tmp_path: Path) -> None:
    zip_path = tmp_path / "annotations_cvat_video.zip"
    out_dir = tmp_path / "imported"
    _write_cvat_video_zip(zip_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/import_cvat_video_annotations.py",
            "--cvat-zip",
            str(zip_path),
            "--clip-id",
            "clip_cli",
            "--fps",
            "30",
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "reviewed_boxes.json" in result.stdout
    reviewed = validate_artifact_file("cvat_video_annotations", out_dir / "reviewed_boxes.json")
    person = validate_artifact_file("person_ground_truth", out_dir / "person_ground_truth.json")
    assert isinstance(reviewed, CvatVideoAnnotations)
    assert isinstance(person, PersonGroundTruth)
    assert reviewed.clip_id == "clip_cli"
    assert person.clip_id == "clip_cli"


def test_write_person_ground_truth_from_cvat_video_registers_existing_person_schema(tmp_path: Path) -> None:
    zip_path = tmp_path / "annotations_cvat_video.zip"
    out_path = tmp_path / "person_ground_truth.json"
    _write_cvat_video_zip(zip_path)
    _, person_gt = import_cvat_video_zip(zip_path, clip_id="clip_a")

    write_person_ground_truth_from_cvat_video(out_path, person_gt)

    parsed = validate_artifact_file("person_ground_truth", out_path)
    assert isinstance(parsed, PersonGroundTruth)
    assert parsed.source_format == "cvat_video_1_1"
