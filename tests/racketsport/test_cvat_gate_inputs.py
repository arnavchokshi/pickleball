from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.cvat_gate_inputs import (
    CvatGateClipSpec,
    Data1CvatClipSpec,
    build_cvat_gate_input_payloads,
    canonical_data1_cvat_clip_specs,
    write_cvat_gate_input_package,
    write_data1_substitute_package,
)
from threed.racketsport.eval.label_checks import score_ball_labels, score_player_bbox_labels
from threed.racketsport.cvat_video import write_cvat_video_annotations
from threed.racketsport.schemas import (
    BallTrack,
    CvatVideoAnnotationSummary,
    CvatVideoAnnotations,
    CvatVideoBox,
    CvatVideoFrame,
    CvatVideoTask,
    CvatVideoTrackSummary,
    Tracks,
)
from threed.racketsport.testclips import REQUIRED_LABEL_FILES


def _annotations() -> CvatVideoAnnotations:
    return CvatVideoAnnotations(
        schema_version=1,
        artifact_type="racketsport_cvat_video_annotations",
        clip_id="clip_001",
        source_format="cvat_video_1_1",
        source_path="cvat_upload/exports/clip_001.zip",
        task=CvatVideoTask(
            size=2,
            start_frame=0,
            stop_frame=1,
            original_size=(640, 360),
            source="clip_001.mp4",
        ),
        frames=[
            CvatVideoFrame(
                frame_index=0,
                boxes=[
                    CvatVideoBox(
                        track_id=0,
                        label="player",
                        frame_index=0,
                        bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
                        bbox_xywh=(10.0, 20.0, 100.0, 200.0),
                        keyframe=True,
                        occluded=False,
                        source="manual",
                    ),
                    CvatVideoBox(
                        track_id=4,
                        label="ball",
                        frame_index=0,
                        bbox_xyxy=(300.0, 150.0, 308.0, 158.0),
                        bbox_xywh=(300.0, 150.0, 8.0, 8.0),
                        keyframe=True,
                        occluded=False,
                        source="manual",
                    ),
                    CvatVideoBox(
                        track_id=8,
                        label="paddle",
                        frame_index=0,
                        bbox_xyxy=(400.0, 100.0, 430.0, 150.0),
                        bbox_xywh=(400.0, 100.0, 30.0, 50.0),
                        keyframe=True,
                        occluded=False,
                        source="manual",
                    ),
                ],
            ),
            CvatVideoFrame(frame_index=1, boxes=[]),
        ],
        tracks=[
            CvatVideoTrackSummary(
                track_id=0,
                label="player",
                visible_box_count=1,
                outside_box_count=0,
                keyframe_count=1,
                first_visible_frame=0,
                last_visible_frame=0,
            ),
            CvatVideoTrackSummary(
                track_id=4,
                label="ball",
                visible_box_count=1,
                outside_box_count=0,
                keyframe_count=1,
                first_visible_frame=0,
                last_visible_frame=0,
            ),
            CvatVideoTrackSummary(
                track_id=8,
                label="paddle",
                visible_box_count=1,
                outside_box_count=0,
                keyframe_count=1,
                first_visible_frame=0,
                last_visible_frame=0,
            ),
        ],
        summary=CvatVideoAnnotationSummary(
            frame_count=2,
            visible_box_count=3,
            outside_box_count=0,
            labels=["player", "ball", "paddle"],
            track_count_by_label={"player": 1, "ball": 1, "paddle": 1},
            visible_box_count_by_label={"player": 1, "ball": 1, "paddle": 1},
        ),
    )


def test_build_cvat_gate_input_payloads_are_consumed_by_label_checks(tmp_path: Path) -> None:
    payloads = build_cvat_gate_input_payloads(
        _annotations(),
        reviewed_boxes_path=tmp_path / "reviewed_boxes.json",
    )

    assert sorted(payloads) == ["ball", "combined", "paddle", "player"]
    player = payloads["player"]
    ball = payloads["ball"]
    paddle = payloads["paddle"]
    combined = payloads["combined"]
    assert player["status"] == "human_reviewed"
    assert player["not_ground_truth"] is False
    assert player["annotation"]["target_file"] == "players.json"
    assert player["annotation"]["items"][0]["bbox_xyxy"] == [10.0, 20.0, 110.0, 220.0]
    assert player["annotation"]["items"][0]["track_id"] == 1
    assert ball["annotation"]["target_file"] == "ball.json"
    assert ball["annotation"]["items"][0]["xy_px"] == [304.0, 154.0]
    assert ball["annotation"]["items"][0]["visible"] is True
    assert paddle["annotation"]["target_file"] == "paddle_boxes.json"
    assert paddle["limitations"] == [
        "paddle boxes are detector labels only; they are not true paddle corners or 6DoF racket_pose labels"
    ]
    assert combined["summary"]["label_counts_by_name"] == {"ball": 1, "paddle": 1, "player": 1}


def test_build_cvat_gate_input_payloads_pass_through_ball_visibility_level(tmp_path: Path) -> None:
    annotations = _annotations()
    partial_ball = annotations.frames[0].boxes[1].model_copy(update={"visibility_level": "partial"})
    annotations = annotations.model_copy(
        update={
            "frames": [
                annotations.frames[0].model_copy(
                    update={"boxes": [annotations.frames[0].boxes[0], partial_ball, annotations.frames[0].boxes[2]]}
                ),
                annotations.frames[1],
            ]
        }
    )

    payloads = build_cvat_gate_input_payloads(
        annotations,
        reviewed_boxes_path=tmp_path / "reviewed_boxes.json",
    )

    ball_item = payloads["ball"]["annotation"]["items"][0]
    assert ball_item["visible"] is True
    assert ball_item["visibility"] == "partial"
    assert ball_item["visibility_level"] == "partial"
    assert ball_item["wbce_weight"] == 2


def test_cvat_gate_player_and_ball_payloads_drive_existing_gates(tmp_path: Path) -> None:
    payloads = build_cvat_gate_input_payloads(
        _annotations(),
        reviewed_boxes_path=tmp_path / "reviewed_boxes.json",
    )
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for dataset_name, filename in (("player", "players.json"), ("ball", "ball.json")):
        (labels_dir / filename).write_text(json.dumps(payloads[dataset_name]), encoding="utf-8")

    tracks = Tracks.model_validate(
        {
            "schema_version": 1,
            "fps": 30.0,
            "players": [
                {
                    "id": 1,
                    "side": "near",
                    "role": "left",
                    "frames": [
                        {
                            "t": 0.0,
                            "bbox": [10.0, 20.0, 110.0, 220.0],
                            "world_xy": [0.0, 0.0],
                            "conf": 0.99,
                        }
                    ],
                }
            ],
            "rally_spans": [],
        }
    )
    ball_track = BallTrack.model_validate(
        {
            "schema_version": 1,
            "fps": 30.0,
            "source": "tracknet",
            "frames": [
                {
                    "t": 0.0,
                    "xy": [304.0, 154.0],
                    "conf": 0.99,
                    "visible": True,
                    "world_xyz": [0.0, 0.0, 1.0],
                }
            ],
            "bounces": [],
        }
    )

    player_metrics, player_notes = score_player_bbox_labels(labels_dir=labels_dir, tracks=tracks)
    ball_metrics, ball_notes = score_ball_labels(labels_dir=labels_dir, ball_track=ball_track)

    assert player_notes == []
    assert player_metrics["player_bbox_recall_iou50"].passed is True
    assert player_metrics["player_bbox_precision_iou50"].passed is True
    assert ball_notes == []
    assert ball_metrics["ball_f1_at_10px"].passed is True


def test_write_cvat_gate_input_package_writes_clip_payloads_and_manifest(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed_boxes.json"
    write_cvat_video_annotations(reviewed, _annotations())

    manifest = write_cvat_gate_input_package(
        clips=[CvatGateClipSpec(clip_id="clip_001", reviewed_boxes_path=reviewed)],
        out_dir=tmp_path / "gate_inputs",
    )

    assert manifest["artifact_type"] == "racketsport_cvat_gate_input_manifest"
    assert manifest["clip_count"] == 1
    assert manifest["datasets"]["player"]["item_count"] == 1
    assert manifest["datasets"]["paddle"]["item_count"] == 1
    assert manifest["datasets"]["ball"]["item_count"] == 1
    assert manifest["datasets"]["combined"]["item_count"] == 3
    for filename in ("players.json", "paddle_boxes.json", "ball.json", "combined_detector_labels.json"):
        assert (tmp_path / "gate_inputs" / "clip_001" / "labels" / filename).is_file()
    saved = json.loads((tmp_path / "gate_inputs" / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest


def test_build_cvat_gate_inputs_cli_writes_compact_summary(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed_boxes.json"
    out_dir = tmp_path / "gate_inputs"
    write_cvat_video_annotations(reviewed, _annotations())

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_cvat_gate_inputs.py",
            "--clip",
            f"clip_001={reviewed}",
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["artifact_type"] == "racketsport_cvat_gate_input_manifest"
    assert payload["clip_count"] == 1
    assert payload["datasets"]["combined"]["item_count"] == 3
    assert (out_dir / "clip_001" / "labels" / "players.json").is_file()


def test_write_data1_substitute_package_writes_registration_plan_skeletons_and_missing_inputs(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat_upload"
    imports_root = tmp_path / "runs" / "cvat_imports" / "2026_06_30"
    data_root = tmp_path / "data" / "testclips"
    out_dir = imports_root / "data1_substitute"
    clip_a_video = cvat_root / "01_clip_a.mp4"
    clip_a_export = cvat_root / "exports" / "01_clip_a_cvat_for_video_1.1.zip"
    clip_a_reviewed = imports_root / "clip_a" / "reviewed_boxes.json"
    clip_b_video = cvat_root / "04_clip_b.mp4"
    clip_b_export = cvat_root / "exports" / "04_clip_b_cvat_for_video_1.1.zip"
    clip_b_reviewed = imports_root / "clip_b" / "reviewed_boxes.json"
    clip_a_video.parent.mkdir(parents=True)
    clip_a_export.parent.mkdir(parents=True)
    clip_a_reviewed.parent.mkdir(parents=True)
    clip_a_video.write_bytes(b"video")
    clip_a_export.write_bytes(b"zip")
    clip_a_reviewed.write_text("{}", encoding="utf-8")
    clip_b_video.write_bytes(b"video")
    detector_players = imports_root / "gate_inputs" / "clip_a" / "labels" / "players.json"
    detector_players.parent.mkdir(parents=True)
    detector_players.write_text("{}", encoding="utf-8")

    manifest = write_data1_substitute_package(
        clips=[
            Data1CvatClipSpec(
                clip_id="clip_a",
                source_video_path=clip_a_video,
                cvat_export_path=clip_a_export,
                reviewed_boxes_path=clip_a_reviewed,
                metadata={
                    "camera_height": "low",
                    "camera_angle": "steep_corner",
                    "play_type": "doubles",
                    "environment": "outdoor",
                    "frame_rate_fps": 60,
                    "duration_s": 10.0,
                    "racket_gt": False,
                },
            ),
            Data1CvatClipSpec(
                clip_id="clip_b",
                source_video_path=clip_b_video,
                cvat_export_path=clip_b_export,
                reviewed_boxes_path=clip_b_reviewed,
                metadata={
                    "camera_height": "mid",
                    "camera_angle": "shallow_baseline",
                    "play_type": "doubles",
                    "environment": "indoor",
                    "frame_rate_fps": 30,
                    "duration_s": 30.0,
                    "racket_gt": False,
                },
            ),
        ],
        out_dir=out_dir,
        data_testclips_root=data_root,
        detector_gate_inputs_root=imports_root / "gate_inputs",
    )

    assert manifest["artifact_type"] == "racketsport_data1_cvat_substitute_manifest"
    assert manifest["status"] == "blocked_missing_data1_inputs"
    assert manifest["data1_ready"] is False
    assert manifest["detector_package"]["separate_from_data1"] is True
    assert manifest["summary"]["source_video_count"] == 2
    assert manifest["summary"]["cvat_export_count"] == 1
    assert manifest["summary"]["reviewed_boxes_count"] == 1
    assert manifest["summary"]["missing_input_count"] > 0
    assert manifest["clips"][0]["detector_gate_inputs"]["players.json"]["present"] is True
    assert manifest["clips"][1]["cvat_export_exists"] is False

    registration = json.loads((out_dir / "canonical_testclips_registration_manifest.json").read_text(encoding="utf-8"))
    assert registration["clips"][0]["source"] == str(clip_a_video)
    assert registration["clips"][0]["name"] == "clip_a"
    assert registration["clips"][0]["symlink"] is True
    assert registration["clips"][1]["environment"] == "indoor"

    skeleton = json.loads((out_dir / "label_skeletons" / "clip_a" / "labels" / "players.json").read_text(encoding="utf-8"))
    assert skeleton["artifact_type"] == "racketsport_data1_label_skeleton"
    assert skeleton["not_ground_truth"] is True
    assert skeleton["status"] == "missing_human_review"
    assert skeleton["detector_label_substitute"]["present"] is True
    for label_file in REQUIRED_LABEL_FILES:
        assert (out_dir / "label_skeletons" / "clip_b" / "labels" / label_file).is_file()

    missing = json.loads((out_dir / "missing_inputs.json").read_text(encoding="utf-8"))
    missing_kinds = {item["kind"] for item in missing["missing_inputs"]}
    assert "cvat_video_export" in missing_kinds
    assert "data1_label_file" in missing_kinds
    coverage = json.loads((out_dir / "coverage_report.json").read_text(encoding="utf-8"))
    assert coverage["data1_ready"] is False
    sanity = json.loads((out_dir / "sanity_checks.json").read_text(encoding="utf-8"))
    assert sanity["checks"]["detector_package_not_promoted"] is True
    assert sanity["checks"]["skeletons_marked_not_ground_truth"] is True
    assert "No DATA-1 promotion is claimed" in (out_dir / "DATA1_SUBSTITUTE_report.md").read_text(encoding="utf-8")


def test_build_cvat_gate_inputs_cli_writes_data1_substitute_package(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat_upload"
    imports_root = tmp_path / "runs" / "cvat_imports" / "2026_06_30"
    out_dir = imports_root / "data1_substitute"
    for filename in (
        "01_burlington_gold_0300_low_steep_corner_10s.mp4",
        "02_wolverine_mixed_0200_mid_steep_corner_10s.mp4",
        "03_outdoor_webcam_iynbd_1500_long_high_baseline_frames_0000_1150.mp4",
        "04_indoor_doubles_fwuks_0500_long_mid_baseline_30s.mp4",
    ):
        path = cvat_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
    export_path = cvat_root / "exports" / "01_burlington_gold_0300_low_steep_corner_cvat_for_video_1.1.zip"
    export_path.parent.mkdir(parents=True)
    export_path.write_bytes(b"zip")
    reviewed = imports_root / "burlington_gold_0300_low_steep_corner" / "reviewed_boxes.json"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_cvat_gate_inputs.py",
            "--data1-substitute-out-dir",
            str(out_dir),
            "--cvat-upload-root",
            str(cvat_root),
            "--imports-root",
            str(imports_root),
            "--data-testclips-root",
            str(tmp_path / "data" / "testclips"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["artifact_type"] == "racketsport_data1_cvat_substitute_manifest"
    assert payload["canonical_clip_count"] == 4
    assert payload["data1_ready"] is False
    assert (out_dir / "missing_inputs.json").is_file()
    assert (out_dir / "canonical_testclips_registration_manifest.json").is_file()


def test_data1_substitute_package_records_current_strict_holdout_exports(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat_upload"
    imports_root = tmp_path / "runs" / "cvat_imports" / "2026_06_30"
    out_dir = imports_root / "data1_substitute"
    for clip in canonical_data1_cvat_clip_specs(cvat_upload_root=cvat_root, imports_root=imports_root):
        clip.source_video_path.parent.mkdir(parents=True, exist_ok=True)
        clip.source_video_path.write_bytes(b"video")
        clip.cvat_export_path.parent.mkdir(parents=True, exist_ok=True)
        clip.cvat_export_path.write_bytes(b"zip")
        clip.reviewed_boxes_path.parent.mkdir(parents=True, exist_ok=True)
        clip.reviewed_boxes_path.write_text("{}", encoding="utf-8")

    manifest = write_data1_substitute_package(
        clips=canonical_data1_cvat_clip_specs(cvat_upload_root=cvat_root, imports_root=imports_root),
        out_dir=out_dir,
        data_testclips_root=tmp_path / "data" / "testclips",
        detector_gate_inputs_root=imports_root / "gate_inputs",
    )

    assert manifest["summary"]["cvat_export_count"] == 4
    assert manifest["summary"]["reviewed_boxes_count"] == 4
    clips_by_id = {clip["clip_id"]: clip for clip in manifest["clips"]}
    indoor = clips_by_id["indoor_doubles_fwuks_0500_long_mid_baseline"]
    outdoor = clips_by_id["outdoor_webcam_iynbd_1500_long_high_baseline"]
    assert indoor["cvat_export_exists"] is True
    assert indoor["eval_policy"]["role"] == "strict_holdout"
    assert outdoor["eval_policy"]["role"] == "strict_holdout"
    assert outdoor["source_video_path"].endswith("_frames_0000_1150.mp4")
    assert outdoor["metadata"]["duration_s"] == 19.183333

    sanity = json.loads((out_dir / "sanity_checks.json").read_text(encoding="utf-8"))
    assert sanity["checks"]["indoor_cvat_export_status_recorded"] is True
    assert sanity["checks"]["strict_holdouts_not_promoted"] is True
    report = (out_dir / "DATA1_SUBSTITUTE_report.md").read_text(encoding="utf-8")
    assert "Import the missing Indoor" not in report
    assert "Outdoor and Indoor remain strict held-out eval clips" in report


def test_data1_substitute_registration_manifest_uses_absolute_sources_for_relative_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_video = Path("cvat_upload/clip_a.mp4")
    cvat_export = Path("cvat_upload/exports/clip_a_cvat_for_video_1.1.zip")
    reviewed = Path("runs/cvat_imports/2026_06_30/clip_a/reviewed_boxes.json")
    source_video.parent.mkdir(parents=True)
    cvat_export.parent.mkdir(parents=True)
    reviewed.parent.mkdir(parents=True)
    source_video.write_bytes(b"video")
    cvat_export.write_bytes(b"zip")
    reviewed.write_text("{}", encoding="utf-8")

    manifest = write_data1_substitute_package(
        clips=[
            Data1CvatClipSpec(
                clip_id="clip_a",
                source_video_path=source_video,
                cvat_export_path=cvat_export,
                reviewed_boxes_path=reviewed,
                metadata={
                    "camera_height": "low",
                    "camera_angle": "steep_corner",
                    "play_type": "doubles",
                    "environment": "outdoor",
                    "frame_rate_fps": 60,
                    "duration_s": 10.0,
                    "racket_gt": False,
                },
            ),
        ],
        out_dir=Path("runs/cvat_imports/2026_06_30/data1_substitute"),
        data_testclips_root=Path("data/testclips"),
    )

    registration = json.loads(Path(manifest["registration_manifest"]).read_text(encoding="utf-8"))
    source = Path(registration["clips"][0]["source"])
    assert source.is_absolute()
    assert source == source_video.resolve()
