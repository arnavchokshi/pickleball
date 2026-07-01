from __future__ import annotations

import json
import sys
from pathlib import Path

from threed.racketsport.cvat_annotation_sanity import build_annotation_sanity_report
from threed.racketsport.schemas import (
    CvatVideoAnnotationSummary,
    CvatVideoAnnotations,
    CvatVideoBox,
    CvatVideoFrame,
    CvatVideoTask,
    CvatVideoTrackSummary,
)


def _annotations() -> CvatVideoAnnotations:
    frames = [
        CvatVideoFrame(
            frame_index=0,
            boxes=[
                CvatVideoBox(
                    track_id=1,
                    label="player",
                    frame_index=0,
                    bbox_xyxy=(10.0, 10.0, 50.0, 110.0),
                    bbox_xywh=(10.0, 10.0, 40.0, 100.0),
                    keyframe=True,
                    occluded=False,
                ),
                CvatVideoBox(
                    track_id=9,
                    label="ball",
                    frame_index=0,
                    bbox_xyxy=(90.0, 40.0, 100.0, 50.0),
                    bbox_xywh=(90.0, 40.0, 10.0, 10.0),
                    keyframe=True,
                    occluded=False,
                ),
            ],
        ),
        CvatVideoFrame(frame_index=1, boxes=[]),
        CvatVideoFrame(
            frame_index=2,
            boxes=[
                CvatVideoBox(
                    track_id=1,
                    label="player",
                    frame_index=2,
                    bbox_xyxy=(500.0, 10.0, 540.0, 110.0),
                    bbox_xywh=(500.0, 10.0, 40.0, 100.0),
                    keyframe=True,
                    occluded=False,
                )
            ],
        ),
    ]
    return CvatVideoAnnotations(
        schema_version=1,
        artifact_type="racketsport_cvat_video_annotations",
        clip_id="clip",
        source_format="cvat_video_1_1",
        source_path="annotations.zip",
        task=CvatVideoTask(size=3, start_frame=0, stop_frame=2, original_size=(200, 120)),
        frames=frames,
        tracks=[
            CvatVideoTrackSummary(track_id=1, label="player", visible_box_count=2, outside_box_count=0, keyframe_count=2),
            CvatVideoTrackSummary(track_id=9, label="ball", visible_box_count=1, outside_box_count=0, keyframe_count=1),
        ],
        summary=CvatVideoAnnotationSummary(
            frame_count=3,
            visible_box_count=3,
            outside_box_count=0,
            labels=["player", "ball"],
            track_count_by_label={"player": 1, "ball": 1},
            visible_box_count_by_label={"player": 2, "ball": 1},
        ),
    )


def test_build_annotation_sanity_report_flags_missing_players_and_bounds() -> None:
    report = build_annotation_sanity_report(_annotations(), expected_players=2, long_gap_frames=1, jump_factor=2.0)

    assert report["clip_id"] == "clip"
    assert report["frame_count"] == 3
    assert report["player_frame_count_histogram"] == {"0": 1, "1": 2}
    assert report["frames_with_expected_players"] == 0
    assert report["labels"]["player"]["visible_box_count"] == 2
    assert report["labels"]["player"]["out_of_bounds_box_count"] == 1
    assert report["labels"]["player"]["tracks"]["1"]["long_gaps"][0]["gap_frames"] == 1
    assert report["labels"]["player"]["tracks"]["1"]["weird_jumps"][0]["from_frame"] == 0
    assert report["warnings"]


def test_build_annotation_sanity_report_preserves_imported_outside_box_count() -> None:
    annotations = _annotations()
    annotations = annotations.model_copy(
        update={"summary": annotations.summary.model_copy(update={"outside_box_count": 7})}
    )

    report = build_annotation_sanity_report(annotations)

    assert report["outside_box_count"] == 7


def test_check_cvat_video_annotations_cli_reports_errors_without_traceback(monkeypatch, tmp_path: Path, capsys) -> None:
    import scripts.racketsport.check_cvat_video_annotations as check_cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/check_cvat_video_annotations.py",
            "--clip",
            f"clip-a={tmp_path / 'missing.json'}",
            "--out",
            str(tmp_path / "report.json"),
        ],
    )

    assert check_cli.main() == 1
    captured = capsys.readouterr()
    assert "CVAT annotation sanity check failed:" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "report.json").exists()


def test_check_cvat_video_annotations_cli_writes_report(monkeypatch, tmp_path: Path, capsys) -> None:
    import scripts.racketsport.check_cvat_video_annotations as check_cli

    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(_annotations().model_dump_json(), encoding="utf-8")
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/check_cvat_video_annotations.py",
            "--clip",
            f"clip-a={reviewed}",
            "--out",
            str(out),
            "--expected-players",
            "2",
        ],
    )

    assert check_cli.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_cvat_annotation_sanity_report"
    assert payload["clips"][0]["clip_id"] == "clip"
    assert str(out) in capsys.readouterr().out
