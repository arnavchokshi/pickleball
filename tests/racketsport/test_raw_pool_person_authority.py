from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from threed.racketsport.raw_pool_person_authority import (
    RawPoolAuthorityConfig,
    raw_pool_four_player_ceiling,
    run_raw_pool_authority_candidate,
    source_only_cov4_proxy,
    summarize_implausible_motion_spans,
)
from threed.racketsport.schemas import CourtCalibration, PersonGroundTruth, PlayerTrack, TrackFrame, Tracks, validate_artifact_file


def _calibration_payload() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 20.0, "cy": 12.0, "dist": [], "source": "test"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 5.0],
            "camera_height_m": 5.0,
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "world_pts": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    }


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


def _raw_pool_payload() -> dict:
    # Frame 0/1 both carry the real player (foot near world (2, 4), well
    # inside the +-3.048 x +-6.706m pickleball template) plus a spectator far
    # off the court so the raw pool is not already capped to the expected
    # cardinality.
    return {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [1.0, 2.0, 3.0, 4.0], "class": "person", "conf": 0.9, "track_id": 7},
                    {"bbox": [500.0, 500.0, 510.0, 520.0], "class": "person", "conf": 0.6, "track_id": 99},
                ],
            },
            {
                "frame": 1,
                "detections": [
                    {"bbox": [1.5, 2.0, 3.5, 4.0], "class": "person", "conf": 0.8, "track_id": 7},
                    {"bbox": [501.0, 500.0, 511.0, 520.0], "class": "person", "conf": 0.55, "track_id": 99},
                ],
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
                            "bbox_xywh": [1.0, 2.0, 2.0, 2.0],
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
                            "bbox_xywh": [1.5, 2.0, 2.0, 2.0],
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


def test_raw_pool_authority_rejects_off_court_spectator_from_full_pool(tmp_path: Path) -> None:
    pool_dir = tmp_path / "raw_pool" / "clip_a" / "botsort_reid_raw"
    pool_dir.mkdir(parents=True)
    (pool_dir / "tracked_detections.json").write_text(json.dumps(_raw_pool_payload(), indent=2) + "\n", encoding="utf-8")

    calibration_path = tmp_path / "court_calibration.json"
    calibration_path.write_text(json.dumps(_calibration_payload(), indent=2) + "\n", encoding="utf-8")

    video_path = tmp_path / "source.mp4"
    gt_path = tmp_path / "person_ground_truth.json"
    model_path = tmp_path / "osnet.pth"
    _write_video(video_path)
    gt_path.write_text(json.dumps(_ground_truth().model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    model_path.write_bytes(b"fake osnet checkpoint")

    report = run_raw_pool_authority_candidate(
        clip_id="clip_a",
        candidate="botsort_reid_raw",
        video_path=video_path,
        raw_pool_dir=pool_dir,
        calibration_path=calibration_path,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        ground_truth_path=gt_path,
        expected_players=1,
        config=RawPoolAuthorityConfig(expected_players=1, drop_outside_court=True, court_margin_m=0.5),
        feature_extractor=lambda crops: [[1.0, 0.0] for _crop in crops],
    )

    assert report["status"] == "ok"
    assert report["source_only"] is True
    assert report["uses_cvat_labels_for_candidate"] is False
    assert report["global_association"]["output_player_count"] == 1
    # The spectator at world (500, 500) must be rejected by the court-polygon
    # prior, not merely deprioritized -- only the real player track survives.
    assert report["global_association"]["court_rejected_detection_count"] == 2
    assert report["score"]["idf1"] == 1.0
    parsed_tracks = validate_artifact_file("tracks", Path(report["tracks_path"]))
    assert isinstance(parsed_tracks, Tracks)
    assert len(parsed_tracks.players) == 1
    assert Path(report["embedding_export_path"]).is_file()
    ceiling = report["detection_limited_ceiling"]
    assert ceiling["source_only"] is True
    assert ceiling["pool_frame_count"] == 2
    assert ceiling["frames_with_sufficient_on_court_detections"] == 2
    assert report["cov4_proxy"]["source_only"] is True
    assert report["cov4_proxy"]["uses_cvat_labels"] is False
    assert report["cov4_proxy"]["not_gt"] is True
    assert report["cov4_proxy"]["value"] == 1.0
    assert report["four_player_detection_ceiling"]["source_only"] is True
    assert report["four_player_detection_ceiling"]["uses_cvat_labels"] is False
    assert report["four_player_detection_ceiling"]["not_gt"] is True
    assert report["four_player_detection_ceiling"]["value"] == 1.0
    assert report["implausible_motion_spans"] == []
    assert report["implausible_motion_span_count"] == 0
    assert report["implausible_motion_step_count"] == 0
    assert report["implausible_motion_worst_offender"] is None


def test_raw_pool_authority_post_association_court_margin_trims_off_court_output_frame(tmp_path: Path) -> None:
    """A generous candidate-construction apron (needed so a real player
    stepping just past the line is not rejected before association) can
    still leave a frame outside the strict court polygon in the final
    output track. ``post_association_court_margin_m`` trims that frame from
    the already-selected track without re-running fragment/ID selection."""

    pool_dir = tmp_path / "raw_pool" / "clip_a" / "botsort_reid_raw"
    pool_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            # Foot point (2, 4): well inside the +-3.048 x +-6.706m template.
            {"frame": 0, "detections": [{"bbox": [1.0, 2.0, 3.0, 4.0], "class": "person", "conf": 0.9, "track_id": 7}]},
            # Foot point (2, 7.2): 0.494m past the +-6.706m baseline -- just
            # outside the strict (margin=0.0) court polygon, but comfortably
            # inside a 5m candidate-construction apron.
            {"frame": 1, "detections": [{"bbox": [1.0, 5.2, 3.0, 7.2], "class": "person", "conf": 0.8, "track_id": 7}]},
        ],
    }
    (pool_dir / "tracked_detections.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    calibration_path = tmp_path / "court_calibration.json"
    calibration_path.write_text(json.dumps(_calibration_payload(), indent=2) + "\n", encoding="utf-8")

    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "osnet.pth"
    _write_video(video_path)
    model_path.write_bytes(b"fake osnet checkpoint")

    report = run_raw_pool_authority_candidate(
        clip_id="clip_a",
        candidate="botsort_reid_raw",
        video_path=video_path,
        raw_pool_dir=pool_dir,
        calibration_path=calibration_path,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        expected_players=1,
        config=RawPoolAuthorityConfig(
            expected_players=1,
            drop_outside_court=True,
            court_margin_m=5.0,
            post_association_court_margin_m=0.0,
            max_fragment_speed_m_s=50.0,
        ),
        feature_extractor=lambda crops: [[1.0, 0.0] for _crop in crops],
    )

    assert report["global_association"]["court_rejected_detection_count"] == 0
    assert report["global_association"]["post_association_court_rejected_frame_count"] == 1
    assert report["global_association"]["court_filter_skipped_reason"] == ""
    parsed_tracks = validate_artifact_file("tracks", Path(report["tracks_path"]))
    assert isinstance(parsed_tracks, Tracks)
    assert len(parsed_tracks.players) == 1
    assert [round(float(frame.t), 3) for frame in parsed_tracks.players[0].frames] == [0.0]


def test_raw_pool_authority_defaults_to_best_measured_outdoor_reid_profile() -> None:
    config = RawPoolAuthorityConfig()

    assert config.court_margin_m == 2.0
    assert config.max_gap_fill_frames == 48
    assert config.max_merge_cost == 2.0
    assert config.cardinality_backfill is False


def test_raw_pool_authority_can_reuse_existing_embedding_export(tmp_path: Path) -> None:
    pool_dir = tmp_path / "raw_pool" / "clip_a" / "botsort_reid_raw"
    pool_dir.mkdir(parents=True)
    (pool_dir / "tracked_detections.json").write_text(json.dumps(_raw_pool_payload(), indent=2) + "\n", encoding="utf-8")

    calibration_path = tmp_path / "court_calibration.json"
    calibration_path.write_text(json.dumps(_calibration_payload(), indent=2) + "\n", encoding="utf-8")

    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "osnet.pth"
    embedding_path = tmp_path / "reid_embeddings.json"
    _write_video(video_path)
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
                    {"frame": 0, "source_track_id": 7, "bbox": [2.0, 2.0, 12.0, 18.0], "embedding": [1.0, 0.0]},
                    {"frame": 1, "source_track_id": 7, "bbox": [3.0, 2.0, 13.0, 18.0], "embedding": [1.0, 0.0]},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_raw_pool_authority_candidate(
        clip_id="clip_a",
        candidate="botsort_reid_raw",
        video_path=video_path,
        raw_pool_dir=pool_dir,
        calibration_path=calibration_path,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        embedding_export_path=embedding_path,
        expected_players=1,
        config=RawPoolAuthorityConfig(expected_players=1, drop_outside_court=True, court_margin_m=0.5),
        feature_extractor=lambda _crops: (_ for _ in ()).throw(AssertionError("should not export embeddings")),
    )

    assert report["reused_embedding_export"] is True
    assert report["embedding_export_path"] == str(embedding_path)
    assert report["embedding_export"]["feature_extractor"] == "test_osnet"


def test_raw_pool_four_player_ceiling_counts_on_court_non_overlapping_boxes() -> None:
    calibration = validate_artifact_file("court_calibration", _write_temp_calibration())
    assert isinstance(calibration, CourtCalibration)

    payload = {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [0.0, 0.0, 1.0, 1.0], "class": "person", "conf": 0.9, "track_id": 1},
                    {"bbox": [1.5, 0.0, 2.5, 1.0], "class": "person", "conf": 0.9, "track_id": 2},
                    {"bbox": [500.0, 500.0, 510.0, 520.0], "class": "person", "conf": 0.9, "track_id": 3},
                ],
            }
        ],
    }

    ceiling = raw_pool_four_player_ceiling(payload, calibration=calibration, expected_players=2, court_margin_m=3.0)

    assert ceiling["pool_frame_count"] == 1
    assert ceiling["frames_with_sufficient_on_court_detections"] == 1
    assert ceiling["four_player_detection_ceiling"] == 1.0
    assert ceiling["source_only"] is True


def test_source_only_cov4_proxy_counts_exact_player_frames_without_gt() -> None:
    tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(0, 0, 1, 1), world_xy=[0, 0], conf=1.0)],
            ),
            PlayerTrack(
                id=2,
                side="near",
                role="right",
                frames=[TrackFrame(t=0.0, bbox=(1, 0, 2, 1), world_xy=[1, 0], conf=1.0)],
            ),
            PlayerTrack(
                id=3,
                side="far",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(2, 0, 3, 1), world_xy=[2, 0], conf=1.0)],
            ),
            PlayerTrack(
                id=4,
                side="far",
                role="right",
                frames=[
                    TrackFrame(t=0.0, bbox=(3, 0, 4, 1), world_xy=[3, 0], conf=1.0),
                    TrackFrame(t=0.1, bbox=(3, 0, 4, 1), world_xy=[3, 0], conf=1.0),
                ],
            ),
        ],
        rally_spans=[],
    )

    proxy = source_only_cov4_proxy(tracks, expected_players=4, denominator_frame_count=2)

    assert proxy["source_only"] is True
    assert proxy["uses_cvat_labels"] is False
    assert proxy["not_gt"] is True
    assert proxy["expected_players"] == 4
    assert proxy["exact_expected_player_frames"] == 1
    assert proxy["denominator_frame_count"] == 2
    assert proxy["value"] == 0.5


def test_implausible_motion_gate_groups_sustained_world_xy_speed_spans() -> None:
    tracks = Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(
                id=3,
                side="near",
                role="left",
                frames=[
                    TrackFrame(t=0.0, bbox=(0, 0, 1, 1), world_xy=[0.0, 0.0], conf=1.0),
                    TrackFrame(t=0.1, bbox=(0, 0, 1, 1), world_xy=[2.0, 0.0], conf=1.0),
                    TrackFrame(t=0.2, bbox=(0, 0, 1, 1), world_xy=[4.2, 0.0], conf=1.0),
                    TrackFrame(t=0.3, bbox=(0, 0, 1, 1), world_xy=[4.25, 0.0], conf=1.0),
                ],
            ),
            PlayerTrack(
                id=4,
                side="far",
                role="right",
                frames=[
                    TrackFrame(t=0.0, bbox=(0, 0, 1, 1), world_xy=[0.0, 0.0], conf=1.0),
                    TrackFrame(t=0.1, bbox=(0, 0, 1, 1), world_xy=[0.2, 0.0], conf=1.0),
                ],
            ),
        ],
        rally_spans=[],
    )

    diagnostic = summarize_implausible_motion_spans(tracks, speed_threshold_m_s=10.0)

    assert diagnostic["source_only"] is True
    assert diagnostic["uses_cvat_labels"] is False
    assert diagnostic["speed_threshold_m_s"] == 10.0
    assert diagnostic["implausible_motion_span_count"] == 1
    assert diagnostic["implausible_motion_step_count"] == 2
    assert diagnostic["implausible_motion_worst_offender"]["player_id"] == 3
    assert diagnostic["implausible_motion_worst_offender"]["max_speed_m_s"] == pytest.approx(22.0)
    assert diagnostic["implausible_motion_spans"] == [
        {
            "player_id": 3,
            "start_frame": 0,
            "end_frame": 2,
            "start_t": 0.0,
            "end_t": 0.2,
            "step_count": 2,
            "max_speed_m_s": pytest.approx(22.0),
            "max_step": {
                "from_frame": 1,
                "to_frame": 2,
                "from_t": 0.1,
                "to_t": 0.2,
                "dt_s": pytest.approx(0.1),
                "distance_m": pytest.approx(2.2),
                "speed_m_s": pytest.approx(22.0),
                "from_world_xy": [2.0, 0.0],
                "to_world_xy": [4.2, 0.0],
            },
        }
    ]


def _write_temp_calibration() -> Path:
    import tempfile

    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(_calibration_payload(), handle)
    handle.close()
    return Path(handle.name)


def test_raw_pool_authority_threads_reid_device_and_batch_size_into_embedding_export(tmp_path: Path) -> None:
    """``RawPoolAuthorityConfig.reid_device``/``reid_batch_size`` must reach
    the persisted embedding export config unchanged. Exercised with an
    injected ``feature_extractor`` so it needs no real torch/torchreid/MPS
    runtime, only the config plumbing this lane's device-selection fix
    depends on."""

    pool_dir = tmp_path / "raw_pool" / "clip_a" / "botsort_reid_raw"
    pool_dir.mkdir(parents=True)
    (pool_dir / "tracked_detections.json").write_text(json.dumps(_raw_pool_payload(), indent=2) + "\n", encoding="utf-8")

    calibration_path = tmp_path / "court_calibration.json"
    calibration_path.write_text(json.dumps(_calibration_payload(), indent=2) + "\n", encoding="utf-8")

    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "osnet.pth"
    _write_video(video_path)
    model_path.write_bytes(b"fake osnet checkpoint")

    report = run_raw_pool_authority_candidate(
        clip_id="clip_a",
        candidate="botsort_reid_raw",
        video_path=video_path,
        raw_pool_dir=pool_dir,
        calibration_path=calibration_path,
        out_dir=tmp_path / "authority",
        reid_model_path=model_path,
        expected_players=1,
        config=RawPoolAuthorityConfig(expected_players=1, reid_device="mps", reid_batch_size=64),
        feature_extractor=lambda crops: [[1.0, 0.0] for _crop in crops],
    )

    assert report["config"]["reid_device"] == "mps"
    assert report["config"]["reid_batch_size"] == 64
    embeddings = json.loads(Path(report["embedding_export_path"]).read_text(encoding="utf-8"))
    assert embeddings["config"]["device"] == "mps"
    assert embeddings["config"]["batch_size"] == 64


def test_raw_pool_authority_config_rejects_non_positive_reid_batch_size() -> None:
    with pytest.raises(ValueError, match="reid_batch_size must be positive"):
        RawPoolAuthorityConfig(reid_batch_size=0)


def test_run_raw_pool_person_authority_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/run_raw_pool_person_authority.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "pre-role-lock" in completed.stdout
    assert "--raw-pool-dir" in completed.stdout
    assert "--calibration" in completed.stdout
    assert "--local-switch-split-distance-m" in completed.stdout
    assert "--local-switch-split-embedding-distance" in completed.stdout
