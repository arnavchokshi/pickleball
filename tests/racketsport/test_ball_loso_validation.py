from __future__ import annotations

import hashlib
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


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _b0_row(
    *,
    clip: str,
    parent: str,
    frame_index: int,
    ordinal: int,
    ball_present: bool,
    source_video: Path,
    image_zip: Path,
) -> dict:
    image_name = f"{parent}__{clip}__abs_{frame_index:06d}.png"
    return {
        "clip_id": clip,
        "evaluation_eligible": True,
        "final_label": {
            "ball_present": ball_present,
            "bbox_xyxy": [95.0, 95.0, 105.0, 105.0] if ball_present else None,
            "visibility_level": "clear" if ball_present else "none",
        },
        "frame_index": frame_index,
        "ground_truth": True,
        "image_height": 360,
        "image_md5": hashlib.md5(image_name.encode("utf-8")).hexdigest(),
        "image_name": image_name,
        "image_width": 640,
        "image_zip": str(image_zip),
        "image_zip_member": image_name,
        "lineage_class": "scratch",
        "lineage_origin": "scratch_no_prelabel_package",
        "original_prelabel": None,
        "parent_source_id": parent,
        "row_key": f"{clip}:{frame_index:06d}",
        "sample_ordinal": ordinal,
        "source_class": "fixture_source",
        "source_id": parent,
        "split": "validation",
        "teacher_derived": False,
        "training_weight": 1.0,
        "_fixture_source_video": str(source_video),
    }


def _build_b0_parent_source_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path], list[dict]]:
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    image_zip = tmp_path / "scratch_images.zip"
    image_zip.write_bytes(b"fixture-image-zip")
    parent = "original_game_a"
    clip_specs = {
        f"{parent}_rally_0001": {"frame_count": 21, "rows": [(0, True), (20, False)]},
        f"{parent}_rally_0002": {"frame_count": 3, "rows": [(0, True), (2, False)]},
    }
    tracks: dict[str, Path] = {}
    validation_rows: list[dict] = []
    lineage_rows: list[dict] = []
    sampling_frames: list[dict] = []
    universe_videos: list[dict] = []
    ordinal = 0
    for clip, spec in clip_specs.items():
        video = tmp_path / "media" / "rallies" / parent / f"{clip}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes((f"canonical-video:{clip}\n" * 3).encode("utf-8"))
        video_sha = _digest(video)
        universe_videos.append(
            {
                "rally_id": clip,
                "source_id": parent,
                "video_path": str(video),
                "source_video_sha256": video_sha,
                "frame_count": spec["frame_count"],
                "width": 640,
                "height": 360,
            }
        )
        for frame_index, ball_present in spec["rows"]:
            row = _b0_row(
                clip=clip,
                parent=parent,
                frame_index=frame_index,
                ordinal=ordinal,
                ball_present=ball_present,
                source_video=video,
                image_zip=image_zip,
            )
            row.pop("_fixture_source_video")
            validation_rows.append(row)
            lineage_rows.append(
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key
                        in {
                            "clip_id",
                            "evaluation_eligible",
                            "final_label",
                            "frame_index",
                            "ground_truth",
                            "lineage_class",
                            "lineage_origin",
                            "original_prelabel",
                            "parent_source_id",
                            "row_key",
                            "source_class",
                            "source_id",
                            "split",
                            "teacher_derived",
                            "training_weight",
                        }
                    },
                    "video_path": str(video),
                    "source_video_sha256": video_sha,
                }
            )
            sampling_frames.append(
                {
                    "rally_id": clip,
                    "frame_index": frame_index,
                    "row_key": row["row_key"],
                    "source_id": parent,
                    "source_class": row["source_class"],
                    "sample_ordinal": ordinal,
                    "image_name": row["image_name"],
                    "image_zip_member": row["image_zip_member"],
                    "image_md5": row["image_md5"],
                    "image_width": 640,
                    "image_height": 360,
                    "video_path": str(video),
                    "source_video_sha256": video_sha,
                }
            )
            ordinal += 1

        track_path = tmp_path / "predictions" / clip / "ball_track.json"
        _write_ball_track(
            track_path,
            total_frames=spec["frame_count"],
            ball_xy=(100.0, 100.0),
            hit_frames={0},
        )
        metadata = {
            "schema_version": 1,
            "artifact_type": "racketsport_wasb_ball_run",
            "frame_count": spec["frame_count"],
            "out": str(track_path),
            "runtime": {
                "video": str(video),
                "source_video_sha256": video_sha,
                "source_video_frame_count": spec["frame_count"],
                "source_video_size": [640, 360],
            },
        }
        track_path.with_name("ball_track_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        tracks[clip] = track_path

    validation_path = split_dir / "validation.jsonl"
    validation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in validation_rows), encoding="utf-8"
    )
    lineage_path = split_dir / "lineage_rows.jsonl"
    lineage_path.write_text(
        "".join(json.dumps(row) + "\n" for row in lineage_rows), encoding="utf-8"
    )
    train_path = split_dir / "train.jsonl"
    train_path.write_text("", encoding="utf-8")
    sampling_path = tmp_path / "sampling_manifest.json"
    sampling_path.write_text(
        json.dumps({"frames": sampling_frames, "universe_videos": universe_videos}),
        encoding="utf-8",
    )
    report_path = split_dir / "report.json"
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_regroup_parent_source_split",
        "verdict": "BALL_CLEAN_JUDGE",
        "split_semantics": "parent_source",
        "split_counts": {"validation": len(validation_rows)},
        "validation_sources": [parent],
        "checks": {
            "evaluation_lineage": {"verdict": "PASS"},
            "protected_collision_count": {"verdict": "PASS"},
            "scratch_package_reconciled": {"verdict": "PASS"},
            "train_validation_source_intersection": {"verdict": "PASS"},
        },
        "artifacts": {
            "report": str(report_path),
            "train": str(train_path),
            "validation": str(validation_path),
            "lineage_rows": str(lineage_path),
        },
        "input_contract": {
            "scratch_sampling_manifest": str(sampling_path),
            "scratch_sampling_manifest_md5": _digest(sampling_path, "md5"),
            "scratch_image_zip": str(image_zip),
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return split_dir, tracks, validation_rows


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


def test_loso_validation_groups_multiple_clips_into_one_true_source(tmp_path: Path) -> None:
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
            "--source-group",
            "clip_big=recording_a",
            "--source-group",
            "clip_small=recording_a",
        ]
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((out_dir / "loso_report.json").read_text(encoding="utf-8"))
    candidate = report["candidates"]["wasb_like"]
    assert candidate["source_grouping_applied"] is True
    assert candidate["fold_count"] == 1
    assert candidate["clips_by_source_group"] == {"recording_a": ["clip_big", "clip_small"]}
    assert candidate["per_source_metrics"]["recording_a"]["clip_count"] == 2
    assert candidate["per_source_metrics"]["recording_a"]["label_f1_at_20px"] == pytest.approx(
        candidate["pooled_mixed_metrics"]["micro_label_f1_at_20px"]
    )
    assert "p99_error_px_worst_clip" in candidate["per_source_metrics"]["recording_a"]


def test_loso_validation_consumes_b0_parent_source_split_jsonl(tmp_path: Path) -> None:
    split_dir, tracks, _rows = _build_b0_parent_source_fixture(tmp_path)
    out_dir = tmp_path / "out"
    clips = sorted(tracks)

    completed = _run_cli(
        [
            "--out-dir",
            str(out_dir),
            "--parent-source-split",
            str(split_dir),
            "--candidate-track",
            f"wasb_like={clips[0]}={tracks[clips[0]]}",
            "--candidate-track",
            f"wasb_like={clips[1]}={tracks[clips[1]]}",
        ]
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((out_dir / "loso_report.json").read_text(encoding="utf-8"))
    assert report["source_grouping_mode"] == "b0_parent_source_split"
    assert report["label_source"] == "b0_validation_final_label"
    assert report["parent_source_split"]["row_count"] == 4
    assert report["parent_source_split"]["label_source"] == "validation_jsonl.final_label"
    assert report["parent_source_split"]["identity_mode"] == "structurally_bound_fixture"
    assert report["parent_source_split"]["parent_sources"] == ["original_game_a"]
    candidate = report["candidates"]["wasb_like"]
    assert candidate["sources_scored"] == ["original_game_a"]
    assert candidate["clips_by_source_group"] == {
        "original_game_a": clips
    }
    assert candidate["pooled_parent_source_metrics"] == {
        "hidden_false_positive_rate": 0.0,
        "label_f1_at_20px": 1.0,
        "precision_at_20px": 1.0,
        "visible_recall_at_20px": 1.0,
    }
    assert set(report["prediction_artifacts"]["wasb_like"]) == set(clips)
    for clip in clips:
        binding = report["prediction_artifacts"]["wasb_like"][clip]
        assert len(binding["prediction_sha256"]) == 64
        assert len(binding["metadata_sha256"]) == 64
        assert len(binding["canonical_source_video_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "error_text"),
    [
        ("swapped_parent", "caller-swapped parent/source"),
        ("negative_frame", "nonnegative integer frame_index"),
        ("bogus_row_key", "row_key identity mismatch"),
        ("bogus_image", "image identity mismatch"),
    ],
)
def test_b0_parent_source_refuses_caller_controlled_row_identity(
    tmp_path: Path,
    mutation: str,
    error_text: str,
) -> None:
    split_dir, tracks, rows = _build_b0_parent_source_fixture(tmp_path)
    if mutation == "swapped_parent":
        rows[0]["parent_source_id"] = "other_game"
        rows[0]["source_id"] = "other_game"
    elif mutation == "negative_frame":
        rows[0]["frame_index"] = -1
    elif mutation == "bogus_row_key":
        rows[0]["row_key"] = "bogus:999999"
    else:
        rows[0]["image_name"] = "caller_swapped.png"
    (split_dir / "validation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    args = ["--out-dir", str(tmp_path / "out"), "--parent-source-split", str(split_dir)]
    for clip, track in sorted(tracks.items()):
        args.extend(["--candidate-track", f"wasb={clip}={track}"])

    completed = _run_cli(args)

    assert completed.returncode == 2
    assert error_text in completed.stderr
    assert not (tmp_path / "out" / "loso_report.json").exists()


def test_b0_parent_source_refuses_conflicting_prediction_video_aliases(tmp_path: Path) -> None:
    split_dir, tracks, _rows = _build_b0_parent_source_fixture(tmp_path)
    clip = sorted(tracks)[0]
    metadata_path = tracks[clip].with_name("ball_track_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # Exact audit regression: the bogus alias retains the valid
    # rallies/<parent>/<clip>.mp4 suffix. Suffix identity alone must not make two
    # different canonical paths agree.
    metadata["runtime"]["source_video"] = (
        f"/definitely/not/the/canonical/prefix/rallies/original_game_a/{clip}.mp4"
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    args = ["--out-dir", str(tmp_path / "out"), "--parent-source-split", str(split_dir)]
    for candidate_clip, track in sorted(tracks.items()):
        args.extend(["--candidate-track", f"wasb={candidate_clip}={track}"])

    completed = _run_cli(args)

    assert completed.returncode == 2
    assert "conflicting video/source_video aliases" in completed.stderr
    assert "aliases disagree on canonical path identity" in completed.stderr


def test_b0_parent_source_refuses_source_video_sha_mismatch(tmp_path: Path) -> None:
    split_dir, tracks, _rows = _build_b0_parent_source_fixture(tmp_path)
    clip = sorted(tracks)[0]
    metadata = json.loads(
        tracks[clip].with_name("ball_track_metadata.json").read_text(encoding="utf-8")
    )
    Path(metadata["runtime"]["video"]).write_bytes(b"caller-swapped-video")
    args = ["--out-dir", str(tmp_path / "out"), "--parent-source-split", str(split_dir)]
    for candidate_clip, track in sorted(tracks.items()):
        args.extend(["--candidate-track", f"wasb={candidate_clip}={track}"])

    completed = _run_cli(args)

    assert completed.returncode == 2
    assert "source-video SHA-256 mismatch" in completed.stderr


def test_b0_parent_source_refuses_symlinked_prediction(tmp_path: Path) -> None:
    split_dir, tracks, _rows = _build_b0_parent_source_fixture(tmp_path)
    clip = sorted(tracks)[0]
    target = tracks[clip]
    symlink_root = tmp_path / "symlink_predictions" / clip
    symlink_root.mkdir(parents=True)
    symlink_track = symlink_root / "ball_track.json"
    symlink_track.symlink_to(target)
    tracks[clip] = symlink_track
    args = ["--out-dir", str(tmp_path / "out"), "--parent-source-split", str(split_dir)]
    for candidate_clip, track in sorted(tracks.items()):
        args.extend(["--candidate-track", f"wasb={candidate_clip}={track}"])

    completed = _run_cli(args)

    assert completed.returncode == 2
    assert "may not use symlinks" in completed.stderr


def test_live_b0_split_scores_final_labels_and_binds_prediction_media(tmp_path: Path) -> None:
    split_dir = Path("runs/lanes/ball_b0_split_20260721/split")
    validation_rows = [
        json.loads(line)
        for line in (split_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clips = sorted({row["clip_id"] for row in validation_rows})
    assert len(validation_rows) == 167
    assert len(clips) == 8
    args = ["--out-dir", str(tmp_path / "live_out"), "--parent-source-split", str(split_dir)]
    for clip in clips:
        track = Path("data/online_harvest_20260706/prelabels") / clip / "ball_track.json"
        args.extend(["--candidate-track", f"wasb={clip}={track}"])

    completed = _run_cli(args)

    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "live_out" / "loso_report.json").read_text(encoding="utf-8"))
    assert report["objective_result"] == "PASS"
    assert report["label_source"] == "b0_validation_final_label"
    assert report["parent_source_split"]["identity_mode"] == "frozen_b0_20260721"
    assert report["parent_source_split"]["validation_sha256"] == (
        "39a07ed6d5211cbdc2ccc8a3f1f73b298a1ed262a6cae1f8a6190e5aa1533429"
    )
    assert report["reviewed_row_filter"] == {"clip_count": 8, "row_count": 167}
    assert report["candidates"]["wasb"]["sources_scored"] == ["Ezz6HDNHlnk", "HyUqT7zFiwk"]
    assert set(report["prediction_artifacts"]["wasb"]) == set(clips)
    assert report["candidates"]["wasb"]["pooled_parent_source_metrics"] == {
        "hidden_false_positive_rate": pytest.approx(0.4931506849315068),
        "label_f1_at_20px": pytest.approx(0.5670103092783506),
        "precision_at_20px": pytest.approx(0.55),
        "visible_recall_at_20px": pytest.approx(0.5851063829787234),
    }


def test_live_b0_split_refuses_same_hyu_prediction_reused_for_ezz(tmp_path: Path) -> None:
    split_dir = Path("runs/lanes/ball_b0_split_20260721/split")
    clips = sorted(
        {
            json.loads(line)["clip_id"]
            for line in (split_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    hyu_clip = "HyUqT7zFiwk_rally_0001"
    hyu_track = Path("data/online_harvest_20260706/prelabels") / hyu_clip / "ball_track.json"
    args = ["--out-dir", str(tmp_path / "out"), "--parent-source-split", str(split_dir)]
    for index, clip in enumerate(clips):
        track = (
            hyu_track
            if index == 0
            else Path("data/online_harvest_20260706/prelabels") / clip / "ball_track.json"
        )
        args.extend(["--candidate-track", f"wasb={clip}={track}"])

    completed = _run_cli(args)

    assert completed.returncode == 2
    assert "canonical path is bound to 'HyUqT7zFiwk_rally_0001'" in completed.stderr


def test_loso_validation_rejects_conflicting_source_group_mapping(tmp_path: Path) -> None:
    cvat_root, tracks_dir = _build_two_source_two_candidate_fixture(tmp_path)
    completed = _run_cli(
        [
            "--cvat-root",
            str(cvat_root),
            "--out-dir",
            str(tmp_path / "out"),
            "--candidate-track",
            f"wasb_like=clip_big={tracks_dir / 'wasb_like' / 'clip_big' / 'ball_track.json'}",
            "--source-group",
            "clip_big=recording_a",
            "--source-group",
            "clip_big=recording_b",
        ]
    )
    assert completed.returncode == 2
    assert "conflicting --source-group" in completed.stderr


def test_loso_validation_scores_explicit_reviewed_row_stratum(tmp_path: Path) -> None:
    cvat_root, tracks_dir = _build_two_source_two_candidate_fixture(tmp_path)
    out_dir = tmp_path / "out"
    row_list = tmp_path / "random_rows.json"
    row_list.write_text(
        json.dumps(
            {
                "rows": [
                    {"row_key": "clip_big:000000"},
                    {"clip_id": "clip_small", "frame_index": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    completed = _run_cli(
        [
            "--cvat-root",
            str(cvat_root),
            "--out-dir",
            str(out_dir),
            "--reviewed-row-list",
            str(row_list),
            "--candidate-track",
            f"wasb_like=clip_big={tracks_dir / 'wasb_like' / 'clip_big' / 'ball_track.json'}",
            "--candidate-track",
            f"wasb_like=clip_small={tracks_dir / 'wasb_like' / 'clip_small' / 'ball_track.json'}",
        ]
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((out_dir / "loso_report.json").read_text(encoding="utf-8"))
    assert report["reviewed_row_filter"] == {"clip_count": 2, "row_count": 2}
    candidate = report["candidates"]["wasb_like"]
    assert candidate["pooled_mixed_metrics"]["total_reviewed_frame_count"] == 2
    assert candidate["pooled_mixed_metrics"]["micro_label_f1_at_20px"] == 1.0
