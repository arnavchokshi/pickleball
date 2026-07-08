from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

CLI_PATH = "scripts/racketsport/ingest_owner_ball_labels.py"


def _reviewed_boxes_payload(
    *,
    clip_id: str,
    frame_count: int,
    reviewed_frame_indices: list[int],
    ball_frames: dict[int, tuple[float, float, float, float]],
    visibility_levels: dict[int, str] | None = None,
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
                    "visibility_level": (visibility_levels or {}).get(frame_index, "clear"),
                }
            )
        frame_payload: dict[str, object] = {"frame_index": frame_index, "boxes": boxes}
        if visibility_levels and frame_index in visibility_levels and bbox is None:
            frame_payload["visibility_levels_by_label"] = {"ball": visibility_levels[frame_index]}
        frames.append(frame_payload)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": clip_id,
        "source_format": "cvat_video_1_1",
        "source_path": f"{clip_id}.zip",
        "reviewed_frame_indices": reviewed_frame_indices,
        "reviewed_frame_indices_source": "explicit",
        "task": {
            "task_id": 42,
            "name": clip_id,
            "size": frame_count,
            "mode": "annotation",
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
                "outside_box_count": 0,
                "keyframe_count": len(ball_frames),
                "first_visible_frame": min(ball_frames) if ball_frames else None,
                "last_visible_frame": max(ball_frames) if ball_frames else None,
            }
        ],
        "summary": {
            "frame_count": frame_count,
            "visible_box_count": len(ball_frames),
            "outside_box_count": 0,
            "labels": ["ball"],
            "track_count_by_label": {"ball": 1},
            "visible_box_count_by_label": {"ball": len(ball_frames)},
        },
    }


def _write_base_clip(root: Path) -> None:
    clip_dir = root / "73VurrTKCZ8_rally_0002"
    clip_dir.mkdir(parents=True)
    payload = _reviewed_boxes_payload(
        clip_id="73VurrTKCZ8_rally_0002",
        frame_count=300,
        reviewed_frame_indices=[136, 272],
        ball_frames={136: (1809.0, 679.0, 20.0, 20.0)},
    )
    (clip_dir / "reviewed_boxes.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_labelpack_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_type": "w5_labelpack_20260708_package_manifest",
                "ball_sessions": [
                    {
                        "session_id": "ball_session_01",
                        "frame_count": 5,
                        "clip_counts": {
                            "73VurrTKCZ8_rally_0002": 4,
                            "Ezz6HDNHlnk_rally_0004": 1,
                        },
                        "source_ids": ["73VurrTKCZ8", "Ezz6HDNHlnk"],
                        "source_classes": {
                            "73VurrTKCZ8": "outdoor_day_multicam",
                            "Ezz6HDNHlnk": "outdoor_night_fenced",
                        },
                        "disagreement_type_counts": {
                            "large-offset": 3,
                            "student-only": 1,
                            "teacher-only": 1,
                        },
                        "task_name": "w5_ball_sst_ball_session_01_20260708",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_cvat_images_zip(path: Path, *, protected: bool = False) -> None:
    protected_name = "0006__pwxNwFfYQlQ__pwxNwFfYQlQ_rally_0001__f000010__large_offset.jpg"
    images = [
        ('0', "0401__73VurrTKCZ8__73VurrTKCZ8_rally_0002__f000272__student_only.jpg", ""),
        ('1', "0495__73VurrTKCZ8__73VurrTKCZ8_rally_0002__f000136__large_offset.jpg", ""),
        (
            '2',
            "0496__73VurrTKCZ8__73VurrTKCZ8_rally_0002__f000100__large_offset.jpg",
            """
    <box label="ball" source="file" occluded="0" xtl="10" ytl="20" xbr="30" ybr="40" z_order="0">
      <attribute name="visibility">true</attribute>
      <attribute name="center_convention">review_to_blur_streak_center</attribute>
      <attribute name="blur_label_quality"></attribute>
      <attribute name="visibility_level">clear</attribute>
      <attribute name="blur_angle_deg">0</attribute>
      <attribute name="blur_length_px">0</attribute>
      <attribute name="blur_width_px">0</attribute>
    </box>""",
        ),
        ('3', "0497__Ezz6HDNHlnk__Ezz6HDNHlnk_rally_0004__f000010__teacher_only.jpg", ""),
        (
            '4',
            "0498__Ezz6HDNHlnk__Ezz6HDNHlnk_rally_0004__f000020__large_offset.jpg",
            """
    <box label="ball" source="file" occluded="0" xtl="10" ytl="20" xbr="30" ybr="40" z_order="0">
      <attribute name="visibility_level">clear</attribute>
    </box>
    <box label="ball" source="file" occluded="0" xtl="11" ytl="21" xbr="31" ybr="41" z_order="0">
      <attribute name="visibility_level">clear</attribute>
    </box>""",
        ),
    ]
    if protected:
        images.append(('5', protected_name, ""))
    image_xml = "\n".join(
        f'  <image id="{image_id}" name="{name}" width="1920" height="1080">{boxes}\n  </image>'
        for image_id, name, boxes in images
    )
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <job>
      <id>14</id>
      <size>{len(images)}</size>
      <mode>annotation</mode>
      <start_frame>0</start_frame>
      <stop_frame>{len(images) - 1}</stop_frame>
      <labels>
        <label>
          <name>ball</name>
          <type>rectangle</type>
          <attributes>
            <attribute><name>visibility_level</name><values>clear\npartial\nfull\nout_of_frame</values></attribute>
            <attribute><name>center_convention</name><values></values></attribute>
            <attribute><name>blur_angle_deg</name><values>0\n360\n1</values></attribute>
            <attribute><name>blur_length_px</name><values>0\n5000\n1</values></attribute>
            <attribute><name>blur_width_px</name><values>0\n5000\n1</values></attribute>
            <attribute><name>blur_label_quality</name><values></values></attribute>
          </attributes>
        </label>
      </labels>
    </job>
    <dumped>2026-07-08 20:50:20.897740+00:00</dumped>
  </meta>
{image_xml}
</annotations>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("annotations.xml", xml)


def test_ingest_owner_ball_labels_merges_sparse_rows_and_accounts_skips(tmp_path: Path) -> None:
    from scripts.racketsport.ingest_owner_ball_labels import build_reviewed_corpus
    from scripts.racketsport.train_ball_stage2 import sparse_tracknet_labels_from_cvat

    base_root = tmp_path / "base"
    _write_base_clip(base_root)
    manifest = tmp_path / "package_manifest.json"
    _write_labelpack_manifest(manifest)
    export_zip = tmp_path / "w5_session_01_ball_annotations.zip"
    _write_cvat_images_zip(export_zip)

    report = build_reviewed_corpus(
        base_cvat_export_root=base_root,
        export_zips=[export_zip],
        labelpack_manifest=manifest,
        out_root=tmp_path / "out",
    )

    assert report["summary"]["base_reviewed_row_count"] == 2
    assert report["summary"]["new_reviewed_row_count"] == 2
    assert report["summary"]["total_reviewed_row_count"] == 4
    assert report["summary"]["new_positive_row_count"] == 1
    assert report["summary"]["new_negative_row_count"] == 1
    assert report["zip_reports"][0]["skip_reason_counts"] == {
        "duplicate_existing_reviewed_same_absent": 1,
        "duplicate_existing_reviewed_conflict": 1,
        "multiple_ball_boxes": 1,
    }
    assert report["protected_scan"]["status"] == "NO_MATCH"

    clip_path = tmp_path / "out" / "reviewed_corpus" / "73VurrTKCZ8_rally_0002" / "reviewed_boxes.json"
    labels = sparse_tracknet_labels_from_cvat(clip_path)
    by_frame = {row.frame: row for row in labels}
    assert sorted(by_frame) == [100, 136, 272]
    assert by_frame[100].visibility == 1
    assert by_frame[100].center_convention == "review_to_blur_streak_center"
    assert by_frame[136].visibility == 1
    assert by_frame[272].visibility == 0

    loso_manifest = json.loads((tmp_path / "out" / "loso_fold_manifest.json").read_text(encoding="utf-8"))
    outdoor = [fold for fold in loso_manifest["folds"] if fold["source_id"] == "73VurrTKCZ8"][0]
    assert outdoor["source_class"] == "outdoor_day_multicam"
    assert outdoor["is_outdoor_fold"] is True
    for fold in loso_manifest["folds"]:
        assert set(fold["train_row_keys"]).isdisjoint(fold["val_row_keys"])


def test_ingest_owner_ball_labels_rejects_protected_patterns(tmp_path: Path) -> None:
    from scripts.racketsport.ingest_owner_ball_labels import ProtectedPatternError, build_reviewed_corpus

    base_root = tmp_path / "base"
    _write_base_clip(base_root)
    manifest = tmp_path / "package_manifest.json"
    _write_labelpack_manifest(manifest)
    export_zip = tmp_path / "w5_session_01_ball_annotations.zip"
    _write_cvat_images_zip(export_zip, protected=True)

    with pytest.raises(ProtectedPatternError, match="pwxNwFfYQlQ"):
        build_reviewed_corpus(
            base_cvat_export_root=base_root,
            export_zips=[export_zip],
            labelpack_manifest=manifest,
            out_root=tmp_path / "out",
        )


def test_ingest_owner_ball_labels_manifest_is_deterministic(tmp_path: Path) -> None:
    from scripts.racketsport.ingest_owner_ball_labels import build_reviewed_corpus

    base_root = tmp_path / "base"
    _write_base_clip(base_root)
    manifest = tmp_path / "package_manifest.json"
    _write_labelpack_manifest(manifest)
    export_zip = tmp_path / "w5_session_01_ball_annotations.zip"
    _write_cvat_images_zip(export_zip)

    first = build_reviewed_corpus(
        base_cvat_export_root=base_root,
        export_zips=[export_zip],
        labelpack_manifest=manifest,
        out_root=tmp_path / "out_a",
    )
    second = build_reviewed_corpus(
        base_cvat_export_root=base_root,
        export_zips=[export_zip],
        labelpack_manifest=manifest,
        out_root=tmp_path / "out_b",
    )

    assert first["manifest_md5"] == second["manifest_md5"]
    assert (tmp_path / "out_a" / "corpus_md5_manifest.json").read_bytes() == (
        tmp_path / "out_b" / "corpus_md5_manifest.json"
    ).read_bytes()


def test_ingest_owner_ball_labels_cli_help_is_indexed() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--base-cvat-export-root" in completed.stdout
    assert "--export-zip" in completed.stdout
    assert "--labelpack-manifest" in completed.stdout
    assert "--out-root" in completed.stdout
