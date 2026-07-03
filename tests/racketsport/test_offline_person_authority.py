from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from threed.racketsport.offline_person_authority import OfflineAuthorityConfig, run_offline_authority_candidate
from threed.racketsport.schemas import PersonGroundTruth, PlayerTrack, TrackFrame, Tracks, validate_artifact_file


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (40, 24))
    assert writer.isOpened()
    try:
        for frame_idx in range(2):
            frame = np.zeros((24, 40, 3), dtype=np.uint8)
            frame[:, :20] = (0, 0, 255 - frame_idx)
            frame[:, 20:] = (255 - frame_idx, 0, 0)
            writer.write(frame)
    finally:
        writer.release()


def _tracks() -> Tracks:
    return Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[
                    TrackFrame(t=0.0, bbox=(2.0, 2.0, 12.0, 18.0), world_xy=(0.0, -1.0), conf=0.9),
                    TrackFrame(t=0.1, bbox=(3.0, 2.0, 13.0, 18.0), world_xy=(0.2, -1.0), conf=0.8),
                ],
            )
        ],
        rally_spans=[],
    )


def _detections_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [{"bbox": [2.0, 2.0, 12.0, 18.0], "class": "person", "conf": 0.9, "track_id": 7}],
            },
            {
                "frame": 1,
                "detections": [{"bbox": [3.0, 2.0, 13.0, 18.0], "class": "person", "conf": 0.8, "track_id": 7}],
            },
        ],
    }


def _ground_truth() -> PersonGroundTruth:
    return PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_a",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic",
            "fps": 10.0,
            "frames": [
                {
                    "frame_index": 0,
                    "source_frame_id": 1,
                    "labels": [
                        {
                            "track_id": 1,
                            "bbox_xywh": [2.0, 2.0, 10.0, 16.0],
                            "ignored": False,
                            "visibility": 1.0,
                            "confidence": 1.0,
                            "class_id": None,
                            "class_name": "player",
                            "person_class": True,
                        }
                    ],
                },
                {
                    "frame_index": 1,
                    "source_frame_id": 2,
                    "labels": [
                        {
                            "track_id": 1,
                            "bbox_xywh": [3.0, 2.0, 10.0, 16.0],
                            "ignored": False,
                            "visibility": 1.0,
                            "confidence": 1.0,
                            "class_id": None,
                            "class_name": "player",
                            "person_class": True,
                        }
                    ],
                },
            ],
            "summary": {
                "frame_count": 2,
                "valid_label_count": 2,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )


def test_offline_authority_candidate_writes_osnet_embeddings_global_tracks_and_score(tmp_path: Path) -> None:
    source_run = tmp_path / "source" / "clip_a" / "candidate_a"
    source_run.mkdir(parents=True)
    tracks_path = source_run / "tracks.json"
    detections_path = source_run / "tracked_detections.json"
    tracks_path.write_text(json.dumps(_tracks().model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    detections_path.write_text(json.dumps(_detections_payload(), indent=2) + "\n", encoding="utf-8")
    video_path = tmp_path / "source.mp4"
    gt_path = tmp_path / "person_ground_truth.json"
    model_path = tmp_path / "osnet.pth"
    _write_video(video_path)
    gt_path.write_text(json.dumps(_ground_truth().model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    model_path.write_bytes(b"fake osnet checkpoint")

    report = run_offline_authority_candidate(
        clip_id="clip_a",
        candidate="candidate_a",
        video_path=video_path,
        source_run_dir=source_run,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        ground_truth_path=gt_path,
        expected_players=1,
        feature_extractor=lambda crops: [[1.0, 0.0] for _crop in crops],
    )

    assert report["status"] == "ok"
    assert report["source_only"] is True
    assert report["uses_cvat_labels_for_candidate"] is False
    assert report["global_association"]["output_player_count"] == 1
    assert report["score"]["idf1"] == 1.0
    assert Path(report["tracks_path"]).is_file()
    assert Path(report["embedding_export_path"]).is_file()
    assert Path(report["score_path"]).is_file()
    parsed_tracks = validate_artifact_file("tracks", Path(report["tracks_path"]))
    assert isinstance(parsed_tracks, Tracks)
    embeddings = json.loads(Path(report["embedding_export_path"]).read_text(encoding="utf-8"))
    assert embeddings["feature_type"] == "osnet_reid_embedding"


def test_offline_authority_post_association_court_margin_trims_off_court_output_frame(tmp_path: Path) -> None:
    """Wiring the court-polygon filter into the champion offline-authority
    path must be able to remove an off-court output frame (e.g. a
    spectator/bystander briefly picked up while a real player is missing)
    without needing a candidate-construction apron tight enough to also
    reject real boundary-line play."""

    source_run = tmp_path / "source" / "clip_a" / "candidate_a"
    source_run.mkdir(parents=True)
    off_court_tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[
                    # Pickleball court half-width is 3.048m: frame 0 sits just
                    # inside the sideline, frame 1 just outside it (0.45m past
                    # the strict boundary). The two stay within the default
                    # fragment-continuity distance budget (~2.45m for a 0.1s
                    # gap) so they remain one fragment/track -- the strict
                    # (margin=0.0) post-association filter is what has to
                    # separate them, not fragment splitting.
                    TrackFrame(t=0.0, bbox=(2.0, 2.0, 12.0, 18.0), world_xy=(2.0, -1.0), conf=0.9),
                    TrackFrame(t=0.1, bbox=(3.0, 2.0, 13.0, 18.0), world_xy=(3.5, -1.0), conf=0.8),
                ],
            )
        ],
        rally_spans=[],
    )
    tracks_path = source_run / "tracks.json"
    detections_path = source_run / "tracked_detections.json"
    tracks_path.write_text(json.dumps(off_court_tracks.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    detections_path.write_text(json.dumps(_detections_payload(), indent=2) + "\n", encoding="utf-8")
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "osnet.pth"
    _write_video(video_path)
    model_path.write_bytes(b"fake osnet checkpoint")

    report = run_offline_authority_candidate(
        clip_id="clip_a",
        candidate="candidate_a",
        video_path=video_path,
        source_run_dir=source_run,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        expected_players=1,
        feature_extractor=lambda crops: [[1.0, 0.0] for _crop in crops],
        config=OfflineAuthorityConfig(
            expected_players=1,
            drop_outside_court=True,
            court_margin_m=100.0,  # generous apron: both frames survive candidate construction
            post_association_court_margin_m=0.0,  # strict: trims the off-court output frame
        ),
    )

    assert report["global_association"]["court_rejected_detection_count"] == 0
    assert report["global_association"]["post_association_court_rejected_frame_count"] == 1
    assert report["global_association"]["court_filter_skipped_reason"] == ""
    parsed_tracks = validate_artifact_file("tracks", Path(report["tracks_path"]))
    assert isinstance(parsed_tracks, Tracks)
    assert len(parsed_tracks.players) == 1
    assert [round(float(frame.t), 3) for frame in parsed_tracks.players[0].frames] == [0.0]


def test_offline_authority_can_reuse_existing_source_only_embedding_export(tmp_path: Path) -> None:
    source_run = tmp_path / "source" / "clip_a" / "candidate_a"
    source_run.mkdir(parents=True)
    (source_run / "tracks.json").write_text(json.dumps(_tracks().model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    (source_run / "tracked_detections.json").write_text(json.dumps(_detections_payload(), indent=2) + "\n", encoding="utf-8")
    video_path = tmp_path / "source.mp4"
    gt_path = tmp_path / "person_ground_truth.json"
    model_path = tmp_path / "osnet.pth"
    embedding_path = tmp_path / "reid_embeddings.json"
    _write_video(video_path)
    gt_path.write_text(json.dumps(_ground_truth().model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    model_path.write_bytes(b"fake osnet checkpoint")
    embedding_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_person_reid_embedding_export",
                "source_only": True,
                "uses_cvat_labels": False,
                "promote_trk": False,
                "feature_type": "osnet_reid_embedding",
                "feature_extractor": "test_osnet",
                "feature_dim": 2,
                "l2_normalized": True,
                "detection_count": 2,
                "detections": [
                    {"frame": 0, "source_track_id": 1, "detection_index": 0, "bbox": [2.0, 2.0, 12.0, 18.0], "embedding": [1.0, 0.0]},
                    {"frame": 1, "source_track_id": 1, "detection_index": 0, "bbox": [3.0, 2.0, 13.0, 18.0], "embedding": [1.0, 0.0]},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_offline_authority_candidate(
        clip_id="clip_a",
        candidate="candidate_a",
        video_path=video_path,
        source_run_dir=source_run,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        embedding_export_path=embedding_path,
        ground_truth_path=gt_path,
        expected_players=1,
        feature_extractor=lambda _crops: (_ for _ in ()).throw(AssertionError("should not export embeddings")),
    )

    assert report["status"] == "ok"
    assert report["reused_embedding_export"] is True
    assert report["embedding_export_path"] == str(embedding_path)
    assert report["embedding_export"]["feature_extractor"] == "test_osnet"
    assert report["score"]["idf1"] == 1.0


def test_offline_authority_prefers_raw_detections_and_infers_bbox_scale(tmp_path: Path) -> None:
    source_run = tmp_path / "source" / "clip_a" / "candidate_scaled"
    source_run.mkdir(parents=True)
    tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(1.0, 1.0, 6.0, 9.0), world_xy=(0.0, -1.0), conf=0.9)],
            )
        ],
        rally_spans=[],
    )
    raw_detections = {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [{"bbox": [2.0, 2.0, 12.0, 18.0], "class": "person", "conf": 0.9, "track_id": 7}],
            }
        ],
    }
    scaled_detections = {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [{"bbox": [1.0, 1.0, 6.0, 9.0], "class": "person", "conf": 0.9, "track_id": 7}],
            }
        ],
    }
    (source_run / "tracks.json").write_text(json.dumps(tracks.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    (source_run / "raw_tracked_detections.json").write_text(json.dumps(raw_detections, indent=2) + "\n", encoding="utf-8")
    (source_run / "tracked_detections.json").write_text(json.dumps(scaled_detections, indent=2) + "\n", encoding="utf-8")
    (source_run / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "counts": {
                    "source_width": 1920,
                    "source_height": 1080,
                    "calibration_width": 960,
                    "calibration_height": 540,
                    "bbox_scale_x": 0.5,
                    "bbox_scale_y": 0.5,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "osnet.pth"
    _write_video(video_path)
    model_path.write_bytes(b"fake osnet checkpoint")

    gt_path = tmp_path / "person_ground_truth.json"
    gt = PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_a",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic",
            "fps": 10.0,
            "frames": [
                {
                    "frame_index": 0,
                    "source_frame_id": 1,
                    "labels": [
                        {
                            "track_id": 1,
                            "bbox_xywh": [2.0, 2.0, 10.0, 16.0],
                            "ignored": False,
                            "visibility": 1.0,
                            "confidence": 1.0,
                            "class_id": None,
                            "class_name": "player",
                            "person_class": True,
                        }
                    ],
                }
            ],
            "summary": {
                "frame_count": 1,
                "valid_label_count": 1,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )
    gt_path.write_text(json.dumps(gt.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")

    report = run_offline_authority_candidate(
        clip_id="clip_a",
        candidate="candidate_scaled",
        video_path=video_path,
        source_run_dir=source_run,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        ground_truth_path=gt_path,
        expected_players=1,
        feature_extractor=lambda crops: [[1.0, 0.0] for _crop in crops],
    )

    embeddings = json.loads(Path(report["embedding_export_path"]).read_text(encoding="utf-8"))

    assert report["source_detections_path"].endswith("raw_tracked_detections.json")
    assert report["embedding_bbox_scale"] == 2.0
    assert report["score_bbox_scale_x"] == 2.0
    assert report["score_bbox_scale_y"] == 2.0
    assert report["score"]["idf1"] == 1.0
    assert embeddings["detections"][0]["bbox"] == [2.0, 2.0, 12.0, 18.0]
    assert report["global_association"]["output_player_count"] == 1


def test_offline_authority_can_use_separate_detection_run_dir_for_repaired_tracks(tmp_path: Path) -> None:
    tracks_run = tmp_path / "source" / "clip_a" / "candidate_repair"
    detection_run = tmp_path / "source" / "clip_a" / "candidate_detector"
    tracks_run.mkdir(parents=True)
    detection_run.mkdir(parents=True)
    tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(1.0, 1.0, 6.0, 9.0), world_xy=(0.0, -1.0), conf=0.9)],
            )
        ],
        rally_spans=[],
    )
    raw_detections = {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [{"bbox": [2.0, 2.0, 12.0, 18.0], "class": "person", "conf": 0.9, "track_id": 7}],
            }
        ],
    }
    (tracks_run / "tracks.json").write_text(json.dumps(tracks.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    (detection_run / "raw_tracked_detections.json").write_text(json.dumps(raw_detections, indent=2) + "\n", encoding="utf-8")
    (detection_run / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "counts": {
                    "source_width": 1920,
                    "source_height": 1080,
                    "calibration_width": 960,
                    "calibration_height": 540,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    video_path = tmp_path / "source.mp4"
    gt_path = tmp_path / "person_ground_truth.json"
    model_path = tmp_path / "osnet.pth"
    _write_video(video_path)
    model_path.write_bytes(b"fake osnet checkpoint")
    gt = PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_a",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic",
            "fps": 10.0,
            "frames": [
                {
                    "frame_index": 0,
                    "source_frame_id": 1,
                    "labels": [
                        {
                            "track_id": 1,
                            "bbox_xywh": [2.0, 2.0, 10.0, 16.0],
                            "ignored": False,
                            "visibility": 1.0,
                            "confidence": 1.0,
                            "class_id": None,
                            "class_name": "player",
                            "person_class": True,
                        }
                    ],
                }
            ],
            "summary": {
                "frame_count": 1,
                "valid_label_count": 1,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )
    gt_path.write_text(json.dumps(gt.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")

    report = run_offline_authority_candidate(
        clip_id="clip_a",
        candidate="candidate_repair",
        video_path=video_path,
        source_run_dir=tracks_run,
        detections_run_dir=detection_run,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        ground_truth_path=gt_path,
        expected_players=1,
        feature_extractor=lambda crops: [[1.0, 0.0] for _crop in crops],
    )

    assert report["source_run_dir"].endswith("candidate_repair")
    assert report["detections_run_dir"].endswith("candidate_detector")
    assert report["source_detections_path"].endswith("candidate_detector/raw_tracked_detections.json")
    assert report["embedding_bbox_scale"] == 2.0
    assert report["score_bbox_scale_x"] == 2.0
    assert report["score"]["idf1"] == 1.0


def test_offline_authority_threads_reid_device_and_batch_size_into_embedding_export(tmp_path: Path) -> None:
    """``OfflineAuthorityConfig.reid_device``/``reid_batch_size`` (the device
    plumbing this lane's device-selection fix relies on) must reach the
    persisted embedding export config unchanged, so a caller requesting
    ``reid_device="mps"``/a larger batch size actually gets it -- this is
    exercised with an injected ``feature_extractor`` so it needs no real
    torch/torchreid/MPS runtime, only the config plumbing."""

    source_run = tmp_path / "source" / "clip_a" / "candidate_a"
    source_run.mkdir(parents=True)
    tracks_path = source_run / "tracks.json"
    detections_path = source_run / "tracked_detections.json"
    tracks_path.write_text(json.dumps(_tracks().model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    detections_path.write_text(json.dumps(_detections_payload(), indent=2) + "\n", encoding="utf-8")
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "osnet.pth"
    _write_video(video_path)
    model_path.write_bytes(b"fake osnet checkpoint")

    report = run_offline_authority_candidate(
        clip_id="clip_a",
        candidate="candidate_a",
        video_path=video_path,
        source_run_dir=source_run,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        expected_players=1,
        config=OfflineAuthorityConfig(expected_players=1, reid_device="mps", reid_batch_size=64),
        feature_extractor=lambda crops: [[1.0, 0.0] for _crop in crops],
    )

    assert report["config"]["reid_device"] == "mps"
    assert report["config"]["reid_batch_size"] == 64
    embeddings = json.loads(Path(report["embedding_export_path"]).read_text(encoding="utf-8"))
    assert embeddings["config"]["device"] == "mps"
    assert embeddings["config"]["batch_size"] == 64


def test_offline_authority_config_rejects_non_positive_reid_batch_size() -> None:
    with pytest.raises(ValueError, match="reid_batch_size must be positive"):
        OfflineAuthorityConfig(reid_batch_size=0)


def test_run_offline_person_authority_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/run_offline_person_authority.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Run offline authoritative person ReID/global association" in completed.stdout
    assert "--reid-backend" in completed.stdout
    assert "--ground-truth" in completed.stdout
    assert "--detections-run-dir" in completed.stdout
