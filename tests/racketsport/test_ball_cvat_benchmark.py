from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_cvat_benchmark import (
    CvatBallCandidate,
    benchmark_cvat_ball_track_candidate,
    evaluate_ball_contact_cue_coverage,
    write_cvat_ball_tracker_benchmark,
)


def _write_track(path: Path) -> None:
    frames = [
        {"t": 0 / 30.0, "xy": [11.0, 10.0], "conf": 0.9, "visible": True},
        {"t": 1 / 30.0, "xy": [200.0, 10.0], "conf": 0.9, "visible": True},
        {"t": 2 / 30.0, "xy": [65.0, 10.0], "conf": 0.9, "visible": True},
        {"t": 3 / 30.0, "xy": [0.0, 0.0], "conf": 0.0, "visible": False},
        {"t": 4 / 30.0, "xy": [0.0, 0.0], "conf": 0.0, "visible": False},
    ]
    path.write_text(
        json.dumps({"schema_version": 1, "fps": 30.0, "source": "tracknet", "frames": frames, "bounces": []}),
        encoding="utf-8",
    )


def _ball_box(frame_index: int, x: float, y: float) -> dict:
    return {
        "track_id": 8,
        "label": "ball",
        "frame_index": frame_index,
        "bbox_xyxy": [x - 5.0, y - 5.0, x + 5.0, y + 5.0],
        "bbox_xywh": [x - 5.0, y - 5.0, 10.0, 10.0],
        "keyframe": True,
        "occluded": False,
        "source": "manual",
    }


def _write_cvat(path: Path) -> None:
    frames = [
        {"frame_index": 0, "boxes": [_ball_box(0, 10.0, 10.0)]},
        {"frame_index": 1, "boxes": []},
        {"frame_index": 2, "boxes": [_ball_box(2, 40.0, 10.0)]},
        {"frame_index": 3, "boxes": [_ball_box(3, 100.0, 100.0)]},
        {"frame_index": 4, "boxes": []},
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_cvat_video_annotations",
                "clip_id": "clip_a",
                "source_format": "cvat_video_1_1",
                "source_path": "annotations.zip",
                "task": {
                    "task_id": 1,
                    "name": "clip_a",
                    "size": 5,
                    "mode": "interpolation",
                    "start_frame": 0,
                    "stop_frame": 4,
                    "original_size": [640, 360],
                    "source": "clip_a.mp4",
                    "dumped": None,
                },
                "frames": frames,
                "tracks": [
                    {
                        "track_id": 8,
                        "label": "ball",
                        "visible_box_count": 3,
                        "outside_box_count": 0,
                        "keyframe_count": 3,
                        "first_visible_frame": 0,
                        "last_visible_frame": 3,
                    }
                ],
                "summary": {
                    "frame_count": 5,
                    "visible_box_count": 3,
                    "outside_box_count": 0,
                    "labels": ["ball"],
                    "track_count_by_label": {"ball": 1},
                    "visible_box_count_by_label": {"ball": 3},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_sparse_cvat(path: Path) -> None:
    frames = [
        {"frame_index": 0, "boxes": [_ball_box(0, 10.0, 10.0)], "visibility_levels_by_label": {"ball": "clear"}},
        {"frame_index": 1, "boxes": [], "visibility_levels_by_label": {}},
        {
            "frame_index": 2,
            "boxes": [{**_ball_box(2, 65.0, 10.0), "visibility_level": "partial"}],
            "visibility_levels_by_label": {"ball": "partial"},
        },
        {"frame_index": 3, "boxes": [], "visibility_levels_by_label": {}},
        {"frame_index": 4, "boxes": [], "visibility_levels_by_label": {}},
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_cvat_video_annotations",
                "clip_id": "clip_sparse",
                "source_format": "cvat_video_1_1",
                "source_path": "annotations.zip",
                "reviewed_frame_indices": [0, 2, 4],
                "reviewed_frame_indices_source": "cvat_frame_filter",
                "task": {
                    "task_id": 1,
                    "name": "clip_sparse",
                    "size": 5,
                    "mode": "interpolation",
                    "start_frame": 0,
                    "stop_frame": 4,
                    "frame_filter": "step=2",
                    "original_size": [640, 360],
                    "source": "clip_sparse.mp4",
                    "dumped": None,
                },
                "frames": frames,
                "tracks": [
                    {
                        "track_id": 8,
                        "label": "ball",
                        "visible_box_count": 2,
                        "outside_box_count": 0,
                        "keyframe_count": 2,
                        "first_visible_frame": 0,
                        "last_visible_frame": 2,
                    }
                ],
                "summary": {
                    "frame_count": 5,
                    "visible_box_count": 2,
                    "outside_box_count": 0,
                    "labels": ["ball"],
                    "track_count_by_label": {"ball": 1},
                    "visible_box_count_by_label": {"ball": 2},
                },
            }
        ),
        encoding="utf-8",
    )


def test_cvat_candidate_scores_ball_f1_hidden_false_positives_error_and_teleports(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    cvat_path = tmp_path / "reviewed_boxes.json"
    _write_track(track_path)
    _write_cvat(cvat_path)

    report = benchmark_cvat_ball_track_candidate(
        ball_track_path=track_path,
        cvat_labels_path=cvat_path,
        candidate_name="strict_localtraj",
        hit_radius_px=36.0,
        teleport_px_per_frame=100.0,
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["clip"] == "clip_a"
    assert report["label_metrics"]["visible_label_count"] == 3
    assert report["label_metrics"]["visible_prediction_count"] == 2
    assert report["label_metrics"]["visible_hit_count"] == 2
    assert report["label_metrics"]["visible_recall_at_20px"] == pytest.approx(1.0 / 3.0)
    assert report["label_metrics"]["hidden_label_count"] == 2
    assert report["label_metrics"]["hidden_false_positive_count"] == 1
    assert report["label_metrics"]["hidden_false_positive_rate"] == pytest.approx(0.5)
    assert report["label_metrics"]["precision_at_20px"] == pytest.approx(1.0 / 3.0)
    assert report["label_metrics"]["label_f1_at_20px"] == pytest.approx(1.0 / 3.0)
    assert report["label_metrics"]["precision_at_10px"] == pytest.approx(1.0 / 3.0)
    assert report["label_metrics"]["label_f1_at_10px"] == pytest.approx(1.0 / 3.0)
    assert report["label_metrics"]["f1_true_positive_count_at_10px"] == 1
    assert report["label_metrics"]["f1_false_positive_count_at_10px"] == 2
    assert report["label_metrics"]["f1_false_negative_count_at_10px"] == 2
    assert report["label_metrics"]["median_error_px"] == pytest.approx(13.0)
    assert report["label_metrics"]["p90_error_px"] == pytest.approx(22.6)
    assert report["label_metrics"]["p95_error_px"] == pytest.approx(23.8)
    assert report["jitter_metrics"]["teleport_count"] == 2
    assert report["quality_score"] < 0.5


def test_cvat_candidate_scores_sparse_reviewed_frames_only(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    cvat_path = tmp_path / "reviewed_boxes.json"
    _write_track(track_path)
    _write_sparse_cvat(cvat_path)
    payload = json.loads(track_path.read_text(encoding="utf-8"))
    payload["frames"][4]["visible"] = True
    payload["frames"][4]["xy"] = [80.0, 80.0]
    payload["frames"][4]["conf"] = 0.7
    track_path.write_text(json.dumps(payload), encoding="utf-8")

    report = benchmark_cvat_ball_track_candidate(
        ball_track_path=track_path,
        cvat_labels_path=cvat_path,
        candidate_name="sparse_candidate",
    )

    assert report["reviewed_frame_indices_source"] == "cvat_frame_filter"
    assert report["reviewed_frame_count"] == 3
    assert report["evaluated_reviewed_frame_count"] == 3
    assert report["label_metrics"]["visible_label_count"] == 2
    assert report["label_metrics"]["hidden_label_count"] == 1
    assert report["label_metrics"]["hidden_false_positive_count"] == 1
    assert report["label_metrics"]["f1_true_positive_count"] == 2
    assert report["label_metrics"]["precision_at_20px"] == pytest.approx(2.0 / 3.0)
    assert report["label_metrics"]["visible_recall_at_20px"] == pytest.approx(1.0)
    assert report["label_metrics"]["label_f1_at_20px"] == pytest.approx(0.8)


def test_cvat_candidate_can_exclude_approx_candidate_points(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    cvat_path = tmp_path / "reviewed_boxes.json"
    _write_track(track_path)
    _write_sparse_cvat(cvat_path)
    payload = json.loads(track_path.read_text(encoding="utf-8"))
    payload["frames"][0]["approx"] = True
    payload["frames"][4]["visible"] = True
    payload["frames"][4]["xy"] = [80.0, 80.0]
    payload["frames"][4]["conf"] = 0.7
    payload["frames"][4]["approx"] = True
    track_path.write_text(json.dumps(payload), encoding="utf-8")

    with_approx = benchmark_cvat_ball_track_candidate(
        ball_track_path=track_path,
        cvat_labels_path=cvat_path,
        candidate_name="with_approx",
        include_approx=True,
    )
    without_approx = benchmark_cvat_ball_track_candidate(
        ball_track_path=track_path,
        cvat_labels_path=cvat_path,
        candidate_name="without_approx",
        include_approx=False,
    )

    assert with_approx["label_metrics"]["f1_true_positive_count"] == 2
    assert with_approx["label_metrics"]["hidden_false_positive_count"] == 1
    assert without_approx["excluded_candidate_approx_frame_count"] == 2
    assert without_approx["label_metrics"]["f1_true_positive_count"] == 1
    assert without_approx["label_metrics"]["hidden_false_positive_count"] == 0
    assert without_approx["label_metrics"]["precision_at_20px"] == pytest.approx(1.0)
    assert without_approx["label_metrics"]["visible_recall_at_20px"] == pytest.approx(0.5)


def test_cvat_candidate_reports_excluded_cvat_label_counts_and_frame_ranges(tmp_path: Path) -> None:
    track_path = tmp_path / "ball_track.json"
    cvat_path = tmp_path / "reviewed_boxes.json"
    _write_track(track_path)
    _write_cvat(cvat_path)

    payload = json.loads(track_path.read_text(encoding="utf-8"))
    payload["frames"] = payload["frames"][:3]
    track_path.write_text(json.dumps(payload), encoding="utf-8")

    report = benchmark_cvat_ball_track_candidate(
        ball_track_path=track_path,
        cvat_labels_path=cvat_path,
        candidate_name="short_track",
    )

    assert report["track_frame_range"] == [0, 2]
    assert report["evaluated_cvat_frame_range"] == [0, 2]
    assert report["excluded_cvat_frame_range"] == [3, 4]
    assert report["cvat_visible_label_count"] == 3
    assert report["evaluated_cvat_visible_label_count"] == 2
    assert report["excluded_cvat_visible_label_count"] == 1
    assert report["excluded_cvat_hidden_frame_count"] == 1


def test_contact_cue_coverage_matches_ball_inflections_to_reviewed_contacts(tmp_path: Path) -> None:
    review_input = tmp_path / "review.json"
    review_input.write_text(
        json.dumps(
            {
                "clips": {
                    "clip_a": {
                        "contacts": [
                            {"player": "P1", "time_s": 1.0, "note": ""},
                            {"player": "P1", "time_s": 2.0, "note": ""},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cue_root = tmp_path / "run"
    clip_dir = cue_root / "clip_a"
    clip_dir.mkdir(parents=True)
    (clip_dir / "ball_inflections.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_ball_inflections",
                "not_gate_verified": True,
                "candidates": [
                    {"time_s": 1.05, "frame": 21, "confidence": 0.9},
                    {"time_s": 3.0, "frame": 60, "confidence": 0.8},
                ],
                "summary": {"candidate_count": 2},
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_ball_contact_cue_coverage(
        review_input_path=review_input,
        cue_root=cue_root,
        clips=["clip_a"],
        fps=20.0,
        max_match_delta_frames=2.0,
    )

    assert report["verification_scope"] == "ball_inflection_cue_vs_reviewed_contacts"
    assert report["ball_verified"] is False
    assert report["summary"]["reviewed_contact_count"] == 2
    assert report["summary"]["matched_contact_count"] == 1
    assert report["summary"]["cue_coverage_rate"] == pytest.approx(0.5)
    assert report["summary"]["extra_cue_count"] == 1
    assert report["summary"]["p90_abs_delta_frames"] == pytest.approx(1.0)
    assert report["clips"][0]["missing_reviewed_contact_count"] == 1


def test_write_cvat_benchmark_aggregates_and_renders_markdown(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    cvat_root = tmp_path / "cvat"
    clip_dir = run_root / "clip_a" / "tracks"
    cvat_dir = cvat_root / "clip_a"
    clip_dir.mkdir(parents=True)
    cvat_dir.mkdir(parents=True)
    _write_track(clip_dir / "ball_track.json")
    _write_cvat(cvat_dir / "reviewed_boxes.json")

    summary = write_cvat_ball_tracker_benchmark(
        candidates=[CvatBallCandidate(clip="clip_a", name="strict", path=clip_dir / "ball_track.json")],
        cvat_root=cvat_root,
        out_json=tmp_path / "summary.json",
        out_markdown=tmp_path / "summary.md",
    )

    assert summary["artifact_type"] == "racketsport_cvat_ball_tracker_benchmark"
    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert summary["results"][0]["status"] == "TESTED-ON-REAL-DATA"
    assert summary["aggregate"]["strict"]["clip_count"] == 1
    assert summary["aggregate"]["strict"]["total_visible_label_count"] == 3
    assert summary["aggregate"]["strict"]["total_hidden_false_positive_count"] == 1
    assert summary["aggregate"]["strict"]["micro_precision_at_10px"] == pytest.approx(1.0 / 3.0)
    assert summary["aggregate"]["strict"]["micro_recall_at_10px"] == pytest.approx(1.0 / 3.0)
    assert summary["aggregate"]["strict"]["micro_label_f1_at_10px"] == pytest.approx(1.0 / 3.0)
    rendered = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "# CVAT Ball Tracker Benchmark" in rendered
    assert "BALL is not verified" in rendered
    assert "## Verification Blockers" in rendered
    assert "## Candidate Paths" in rendered
    assert str(clip_dir / "ball_track.json") in rendered


def test_write_cvat_benchmark_reports_full_horizon_blocker_and_next_recommendation(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    cvat_root = tmp_path / "cvat"
    clip_dir = run_root / "clip_a" / "tracks"
    cvat_dir = cvat_root / "clip_a"
    clip_dir.mkdir(parents=True)
    cvat_dir.mkdir(parents=True)
    _write_track(clip_dir / "ball_track.json")
    _write_cvat(cvat_dir / "reviewed_boxes.json")

    payload = json.loads((clip_dir / "ball_track.json").read_text(encoding="utf-8"))
    payload["frames"] = payload["frames"][:3]
    (clip_dir / "ball_track.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = write_cvat_ball_tracker_benchmark(
        candidates=[CvatBallCandidate(clip="clip_a", name="short", path=clip_dir / "ball_track.json")],
        cvat_root=cvat_root,
        out_json=tmp_path / "summary.json",
        out_markdown=tmp_path / "summary.md",
    )

    assert summary["full_horizon"]["all_cvat_visible_labels_evaluated"] is False
    assert summary["full_horizon"]["total_cvat_visible_label_count"] == 3
    assert summary["full_horizon"]["total_evaluated_cvat_visible_label_count"] == 2
    assert summary["full_horizon"]["total_excluded_cvat_visible_label_count"] == 1
    assert summary["full_horizon"]["blockers"] == [
        "clip_a has 1 reviewed visible ball labels beyond scored track spans over frames 3-4"
    ]
    assert "generate full-span candidate tracks before training or promotion" in summary["next_training_eval_recommendation"]

    rendered = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Full-Horizon Coverage" in rendered
    assert "| 3 | 2 | 1 | no |" in rendered
    assert "## Next Training/Eval Recommendation" in rendered
    assert "generate full-span candidate tracks before training or promotion" in rendered


def test_cvat_benchmark_cli_expands_candidate_specs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    cvat_root = tmp_path / "cvat"
    clip_dir = run_root / "clip_a" / "tracks"
    cvat_dir = cvat_root / "clip_a"
    clip_dir.mkdir(parents=True)
    cvat_dir.mkdir(parents=True)
    _write_track(clip_dir / "ball_track.json")
    _write_cvat(cvat_dir / "reviewed_boxes.json")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/benchmark_ball_tracks_against_cvat.py",
            "--run-root",
            str(run_root),
            "--cvat-root",
            str(cvat_root),
            "--clip",
            "clip_a",
            "--candidate",
            "strict:no_click=tracks/ball_track.json",
            "--out-json",
            str(tmp_path / "summary.json"),
            "--out-md",
            str(tmp_path / "summary.md"),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 0, completed.stderr
    assert "strict" in completed.stdout
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["candidate_count"] == 1
