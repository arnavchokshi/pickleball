from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


CLI = str(Path("scripts/racketsport/ball_loso_validation.py"))


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


def _write_cvat_source(
    path: Path,
    *,
    clip_id: str,
    visible_frame_count: int,
    hidden_frame_count: int,
    ball_xy: tuple[float, float] = (100.0, 100.0),
) -> None:
    total = visible_frame_count + hidden_frame_count
    frames = [
        {"frame_index": i, "boxes": [_ball_box(i, ball_xy[0], ball_xy[1])]} for i in range(visible_frame_count)
    ]
    frames.extend({"frame_index": visible_frame_count + j, "boxes": []} for j in range(hidden_frame_count))
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": clip_id,
        "source_format": "cvat_video_1_1",
        "source_path": "annotations.zip",
        "task": {
            "task_id": 1,
            "name": clip_id,
            "size": total,
            "mode": "interpolation",
            "start_frame": 0,
            "stop_frame": total - 1,
            "original_size": [640, 360],
            "source": f"{clip_id}.mp4",
            "dumped": None,
        },
        "frames": frames,
        "tracks": [
            {
                "track_id": 8,
                "label": "ball",
                "visible_box_count": visible_frame_count,
                "outside_box_count": 0,
                "keyframe_count": visible_frame_count,
                "first_visible_frame": 0,
                "last_visible_frame": max(0, visible_frame_count - 1),
            }
        ],
        "summary": {
            "frame_count": total,
            "visible_box_count": visible_frame_count,
            "outside_box_count": 0,
            "labels": ["ball"],
            "track_count_by_label": {"ball": 1},
            "visible_box_count_by_label": {"ball": visible_frame_count},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ball_track(
    path: Path,
    *,
    total_frames: int,
    ball_xy: tuple[float, float],
    hit_frames: set[int],
    hidden_false_positive_frames: set[int] = frozenset(),
) -> None:
    frames = []
    for i in range(total_frames):
        if i in hit_frames:
            frames.append({"t": i / 30.0, "xy": [ball_xy[0], ball_xy[1]], "conf": 0.9, "visible": True})
        elif i in hidden_false_positive_frames:
            frames.append({"t": i / 30.0, "xy": [5.0, 5.0], "conf": 0.9, "visible": True})
        else:
            frames.append({"t": i / 30.0, "xy": [0.0, 0.0], "conf": 0.0, "visible": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "fps": 30.0, "source": "tracknet", "frames": frames, "bounces": []}),
        encoding="utf-8",
    )


def _build_two_source_two_candidate_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Builds a fixture where the POOLED metric and the LoSO-mean metric disagree.

    clip_big (20 visible + 1 hidden) dominates frame count over clip_small (2 visible +
    1 hidden). ``chain_like`` is near-perfect on clip_big but collapses on clip_small
    (misses both visible labels and commits a hidden false positive); ``wasb_like`` is
    good-but-imperfect on clip_big and perfect on clip_small. Hand-verified expected
    numbers (see design derivation in the lane's DESIGN_NOTES.md / lane report):

      clip_big: wasb F1@20=0.857142857 (15/20 hit), chain F1@20=1.0 (20/20 hit)
      clip_small: wasb F1@20=1.0 (2/2 hit, 0 hidden FP), chain F1@20=0.0 (0/2 hit, 1 hidden FP)

      pooled (micro, both clips): wasb=0.871795, chain=0.930233 -> POOLED WRONGLY FAVORS CHAIN
      loso_mean (unweighted mean of the two per-clip F1s): wasb=0.928571, chain=0.5
        -> LoSO-MEAN CORRECTLY FAVORS WASB

    This reproduces, in miniature and with hand-checkable arithmetic, the real BALL
    inversion pattern this harness targets (ledger rows 4/22/23): a candidate that wins
    the pooled/mixed internal-val metric because it dominates the frame-heavy source,
    while quietly collapsing on a source-specific weakness that an unweighted
    leave-one-source-out mean surfaces instead of hiding.
    """

    cvat_root = tmp_path / "cvat_root"
    _write_cvat_source(cvat_root / "clip_big" / "reviewed_boxes.json", clip_id="clip_big", visible_frame_count=20, hidden_frame_count=1)
    _write_cvat_source(cvat_root / "clip_small" / "reviewed_boxes.json", clip_id="clip_small", visible_frame_count=2, hidden_frame_count=1)

    tracks_dir = tmp_path / "tracks"
    _write_ball_track(
        tracks_dir / "wasb_like" / "clip_big" / "ball_track.json",
        total_frames=21,
        ball_xy=(100.0, 100.0),
        hit_frames=set(range(15)),
    )
    _write_ball_track(
        tracks_dir / "wasb_like" / "clip_small" / "ball_track.json",
        total_frames=3,
        ball_xy=(100.0, 100.0),
        hit_frames={0, 1},
    )
    _write_ball_track(
        tracks_dir / "chain_like" / "clip_big" / "ball_track.json",
        total_frames=21,
        ball_xy=(100.0, 100.0),
        hit_frames=set(range(20)),
    )
    _write_ball_track(
        tracks_dir / "chain_like" / "clip_small" / "ball_track.json",
        total_frames=3,
        ball_xy=(100.0, 100.0),
        hit_frames=set(),
        hidden_false_positive_frames={2},
    )
    return cvat_root, tracks_dir


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, CLI, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_loso_validation_flags_pooled_optimism_and_predicts_heldout_winner(tmp_path: Path) -> None:
    cvat_root, tracks_dir = _build_two_source_two_candidate_fixture(tmp_path)
    out_dir = tmp_path / "out"

    completed = _run_cli(
        [
            "--cvat-root",
            str(cvat_root),
            "--out-dir",
            str(out_dir),
            "--candidate-track",
            f"wasb_like=clip_big={tracks_dir / 'wasb_like' / 'clip_big' / 'ball_track.json'}",
            "--candidate-track",
            f"wasb_like=clip_small={tracks_dir / 'wasb_like' / 'clip_small' / 'ball_track.json'}",
            "--candidate-track",
            f"chain_like=clip_big={tracks_dir / 'chain_like' / 'clip_big' / 'ball_track.json'}",
            "--candidate-track",
            f"chain_like=clip_small={tracks_dir / 'chain_like' / 'clip_small' / 'ball_track.json'}",
            "--heldout-metric",
            "wasb_like=outdoor_webcam_iynbd_1500_long_high_baseline=label_f1_at_20px=0.90",
            "--heldout-metric",
            "chain_like=outdoor_webcam_iynbd_1500_long_high_baseline=label_f1_at_20px=0.60",
        ]
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["objective_result"] == "PASS"

    report = json.loads((out_dir / "loso_report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["artifact_type"] == "racketsport_ball_loso_validation"
    assert report["ball_verified"] is False
    assert report["strict_holdout_clip_ids_never_scored"] == [
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "indoor_doubles_fwuks_0500_long_mid_baseline",
    ]

    wasb = report["candidates"]["wasb_like"]
    chain = report["candidates"]["chain_like"]
    assert wasb["fold_count"] == 2 and wasb["sufficient_for_loso"] is True
    assert chain["fold_count"] == 2 and chain["sufficient_for_loso"] is True

    assert wasb["per_source_metrics"]["clip_big"]["label_f1_at_20px"] == pytest.approx(0.857142857, abs=1e-6)
    assert wasb["per_source_metrics"]["clip_small"]["label_f1_at_20px"] == pytest.approx(1.0, abs=1e-6)
    assert chain["per_source_metrics"]["clip_big"]["label_f1_at_20px"] == pytest.approx(1.0, abs=1e-6)
    assert chain["per_source_metrics"]["clip_small"]["label_f1_at_20px"] == pytest.approx(0.0, abs=1e-6)

    # Pooled/mixed metric: chain wrongly wins (dominant clip_big frame count masks its
    # clip_small collapse).
    pooled_wasb = wasb["pooled_mixed_metrics"]["micro_label_f1_at_20px"]
    pooled_chain = chain["pooled_mixed_metrics"]["micro_label_f1_at_20px"]
    assert pooled_wasb == pytest.approx(0.871795, abs=1e-5)
    assert pooled_chain == pytest.approx(0.930233, abs=1e-5)
    assert pooled_chain > pooled_wasb

    # LoSO-mean: wasb correctly wins (unweighted mean surfaces chain's clip_small collapse).
    loso_mean_wasb = wasb["loso_mean_metrics"]["label_f1_at_20px"]
    loso_mean_chain = chain["loso_mean_metrics"]["label_f1_at_20px"]
    assert loso_mean_wasb == pytest.approx(0.928571, abs=1e-5)
    assert loso_mean_chain == pytest.approx(0.5, abs=1e-6)
    assert loso_mean_wasb > loso_mean_chain

    # Generalization gap = pooled - loso_mean; chain's gap is large and positive (pooled
    # is far more optimistic than any single fold would have been), wasb's is small.
    gap_wasb = wasb["generalization_gap_pooled_minus_losomean"]["label_f1_at_20px"]
    gap_chain = chain["generalization_gap_pooled_minus_losomean"]["label_f1_at_20px"]
    assert gap_wasb == pytest.approx(pooled_wasb - loso_mean_wasb, abs=1e-9)
    assert gap_chain == pytest.approx(pooled_chain - loso_mean_chain, abs=1e-9)
    assert gap_chain > gap_wasb

    # Held-out comparison: the pooled/mixed metric predicts the WRONG winner; LoSO-mean
    # predicts the CORRECT winner (matches the supplied held-out literals).
    assert len(report["heldout_comparisons"]) == 1
    comparison = report["heldout_comparisons"][0]
    assert comparison["clip"] == "outdoor_webcam_iynbd_1500_long_high_baseline"
    assert comparison["metric"] == "label_f1_at_20px"
    assert comparison["heldout_winner"] == "wasb_like"
    assert comparison["pooled_mixed_predicted_winner"] == "chain_like"
    assert comparison["loso_mean_predicted_winner"] == "wasb_like"
    assert comparison["pooled_mixed_correctly_predicted_winner"] is False
    assert comparison["loso_mean_correctly_predicted_winner"] is True

    markdown = (out_dir / "loso_report.md").read_text(encoding="utf-8")
    assert "BALL LoSO" in markdown
    assert "wasb_like" in markdown and "chain_like" in markdown


def test_loso_validation_refuses_strict_holdout_clip_and_writes_nothing(tmp_path: Path) -> None:
    cvat_root, tracks_dir = _build_two_source_two_candidate_fixture(tmp_path)
    out_dir = tmp_path / "out"

    completed = _run_cli(
        [
            "--cvat-root",
            str(cvat_root),
            "--out-dir",
            str(out_dir),
            "--candidate-track",
            f"wasb_like=outdoor_webcam_iynbd_1500_long_high_baseline={tracks_dir / 'wasb_like' / 'clip_big' / 'ball_track.json'}",
        ]
    )

    assert completed.returncode == 2
    assert "strict held-out clip" in completed.stderr
    assert "outdoor_webcam_iynbd_1500_long_high_baseline" in completed.stderr
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_loso_validation_rejects_malformed_candidate_track_spec(tmp_path: Path) -> None:
    completed = _run_cli(
        [
            "--cvat-root",
            str(tmp_path / "cvat_root"),
            "--out-dir",
            str(tmp_path / "out"),
            "--candidate-track",
            "not-a-valid-spec",
        ]
    )
    assert completed.returncode == 2
    assert "CANDIDATE=CLIP=PATH" in completed.stderr


def test_loso_validation_rejects_unknown_heldout_metric_name(tmp_path: Path) -> None:
    cvat_root, tracks_dir = _build_two_source_two_candidate_fixture(tmp_path)
    completed = _run_cli(
        [
            "--cvat-root",
            str(cvat_root),
            "--out-dir",
            str(tmp_path / "out"),
            "--candidate-track",
            f"wasb_like=clip_big={tracks_dir / 'wasb_like' / 'clip_big' / 'ball_track.json'}",
            "--heldout-metric",
            "wasb_like=outdoor_webcam_iynbd_1500_long_high_baseline=not_a_real_metric=0.5",
        ]
    )
    assert completed.returncode == 2
    assert "metric must be one of" in completed.stderr


def test_loso_validation_reports_insufficient_folds_for_single_source_candidate(tmp_path: Path) -> None:
    cvat_root, tracks_dir = _build_two_source_two_candidate_fixture(tmp_path)
    out_dir = tmp_path / "out"

    completed = _run_cli(
        [
            "--cvat-root",
            str(cvat_root),
            "--out-dir",
            str(out_dir),
            "--candidate-track",
            f"solo_candidate=clip_big={tracks_dir / 'wasb_like' / 'clip_big' / 'ball_track.json'}",
        ]
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((out_dir / "loso_report.json").read_text(encoding="utf-8"))
    solo = report["candidates"]["solo_candidate"]
    assert solo["fold_count"] == 1
    assert solo["sufficient_for_loso"] is False
    assert solo["loso_mean_metrics"] == {}
    assert solo["generalization_gap_pooled_minus_losomean"] == {}
    # Objective result reflects that no candidate had >=2 folds.
    assert report["objective_result"] == "PARTIAL"
