from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _write_yolo_row(root: Path, *, name: str, label_lines: list[str]) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "images" / f"{name}.jpg").write_bytes(b"fake-jpg")
    (root / "labels" / f"{name}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def _mini_rkt_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "rkt"
    det = root / "yolo_paddle_detector"
    seg = root / "yolo_paddle_seg"
    det_rows = [
        {
            "output_image": "images/racket_ai_00000.jpg",
            "output_label": "labels/racket_ai_00000.txt",
            "dataset": "racket_ai",
            "source_filename": "Championship-Ben-Johns-vs-Federico-Staksrud-at-the-Carvana-Mesa-Arizona-Open_007_mp4-0001_jpg.rf.aaa.jpg",
            "n_paddle_boxes": 1,
        },
        {
            "output_image": "images/pickleball_seg_00000.jpg",
            "output_label": "labels/pickleball_seg_00000.txt",
            "dataset": "pickleball_seg",
            "source_filename": "output_frame_file4_128_jpg.rf.bbb.jpg",
            "n_paddle_boxes": 1,
        },
        {
            "output_image": "images/racket_ai_00001.jpg",
            "output_label": "labels/racket_ai_00001.txt",
            "dataset": "racket_ai",
            "source_filename": "independent_source_a_mp4-0001_jpg.rf.ccc.jpg",
            "n_paddle_boxes": 1,
        },
        {
            "output_image": "images/racket_ai_00002.jpg",
            "output_label": "labels/racket_ai_00002.txt",
            "dataset": "racket_ai",
            "source_filename": "independent_source_b_mp4-0001_jpg.rf.ddd.jpg",
            "n_paddle_boxes": 1,
        },
    ]
    for row in det_rows:
        _write_yolo_row(det, name=Path(row["output_image"]).stem, label_lines=["0 0.5 0.5 0.2 0.2"])
    (det / "per_image_manifest.json").write_text(json.dumps(det_rows, indent=2), encoding="utf-8")
    (det / "data.yaml").write_text("names:\n  0: paddle\nnc: 1\n", encoding="utf-8")

    seg_rows = [
        {
            "output_image": "images/00000.jpg",
            "output_label": "labels/00000.txt",
            "source_filename": "output_frame_file4_129_jpg.rf.eee.jpg",
            "source_stem": "output_frame_file4_129",
            "n_paddle_polygons": 1,
        },
        {
            "output_image": "images/00001.jpg",
            "output_label": "labels/00001.txt",
            "source_filename": "output_frame_200_jpg.rf.fff.jpg",
            "source_stem": "output_frame_200",
            "n_paddle_polygons": 1,
        },
    ]
    for row in seg_rows:
        _write_yolo_row(seg, name=Path(row["output_image"]).stem, label_lines=["0 0.1 0.1 0.2 0.1 0.2 0.2"])
    (seg / "per_image_manifest.json").write_text(json.dumps(seg_rows, indent=2), encoding="utf-8")
    (seg / "data.yaml").write_text("names:\n  0: paddle\nnc: 1\n", encoding="utf-8")

    manifest = {
        "corpus": "RKT",
        "policy": "External Roboflow Universe data only. No eval clip files are part of this mini fixture.",
        "yolo_paddle_detector_corpus": {"output_dir": str(det)},
        "yolo_paddle_seg_corpus": {"output_dir": str(seg)},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def test_build_training_configs_writes_external_only_splits_and_commands(tmp_path: Path) -> None:
    from scripts.racketsport.rkt_prep_build_training_configs import build_training_configs

    manifest_path = _mini_rkt_corpus(tmp_path)
    out_dir = tmp_path / "runs" / "rkt_prep_20260702T000000Z"

    summary = build_training_configs(
        manifest_path=manifest_path,
        out_dir=out_dir,
        repo_root=tmp_path,
        remote_repo="/home/arnavchokshi/pickleball_git",
        val_fraction=0.5,
    )

    assert summary["artifact_type"] == "racketsport_rkt_prep_training_configs"
    assert summary["policy"]["eval_guard"]["det_train"]["status"] == "clean"
    assert summary["det"]["totals"]["image_count"] == 4
    assert summary["seg"]["totals"]["image_count"] == 2
    assert summary["det"]["known_shared_camera_source_group"] == "carvana_mesa_johns_staksrud_shared_broadcast"

    det_manifest = json.loads((out_dir / "det" / "split_manifest.json").read_text(encoding="utf-8"))
    shared_rows = [
        row
        for row in det_manifest["rows"]
        if row["source_group"] == "carvana_mesa_johns_staksrud_shared_broadcast"
    ]
    assert len(shared_rows) == 2
    assert len({row["split"] for row in shared_rows}) == 1

    det_yaml = (out_dir / "det" / "data.yaml").read_text(encoding="utf-8")
    assert "train: /home/arnavchokshi/pickleball_git/" in det_yaml
    assert "val: /home/arnavchokshi/pickleball_git/" in det_yaml
    assert "paddle" in det_yaml
    assert "outdoor_webcam_iynbd_1500_long_high_baseline" not in det_yaml

    commands = out_dir / "COMMANDS.sh"
    assert os.access(commands, os.X_OK)
    completed = subprocess.run(["bash", "-n", str(commands)], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_build_training_configs_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/rkt_prep_build_training_configs.py", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--out-dir" in completed.stdout


def test_build_training_configs_cli_fails_closed_on_missing_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "runs" / "rkt_prep_missing_manifest"
    missing_manifest = tmp_path / "does_not_exist_manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/rkt_prep_build_training_configs.py",
            "--manifest",
            str(missing_manifest),
            "--out-dir",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not out_dir.exists()


def test_external_val_evaluator_scores_yolo_prediction_directory(tmp_path: Path) -> None:
    from scripts.racketsport.rkt_prep_eval_paddle_boxes import evaluate_external_split

    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    labels.mkdir()
    predictions.mkdir()
    (labels / "img_a.txt").write_text("0 0.20 0.20 0.20 0.20\n0 0.70 0.70 0.20 0.20\n", encoding="utf-8")
    (labels / "img_b.txt").write_text("0 0.50 0.50 0.20 0.20\n", encoding="utf-8")
    (predictions / "img_a.txt").write_text(
        "0 0.20 0.20 0.20 0.20 0.95\n"
        "0 0.70 0.70 0.20 0.20 0.90\n"
        "0 0.05 0.05 0.10 0.10 0.10\n",
        encoding="utf-8",
    )
    (predictions / "img_b.txt").write_text("0 0.90 0.90 0.05 0.05 0.80\n", encoding="utf-8")
    split_manifest = {
        "rows": [
            {"split": "val", "image_path": "images/img_a.jpg", "label_path": str(labels / "img_a.txt")},
            {"split": "val", "image_path": "images/img_b.jpg", "label_path": str(labels / "img_b.txt")},
            {"split": "train", "image_path": "images/train_only.jpg", "label_path": str(labels / "img_b.txt")},
        ]
    }
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest), encoding="utf-8")

    report = evaluate_external_split(
        split_manifest_path=manifest_path,
        predictions_path=predictions,
        split="val",
        iou_threshold=0.5,
    )

    assert report["artifact_type"] == "racketsport_rkt_prep_paddle_box_eval"
    assert report["dataset"]["source"] == "external_corpus_val_split"
    assert report["dataset"]["review_only"] is False
    assert report["metrics"]["ground_truth_count"] == 3
    assert report["metrics"]["prediction_count"] == 4
    assert report["metrics"]["true_positive_count"] == 2
    assert report["metrics"]["false_positive_count"] == 2
    assert report["metrics"]["false_negative_count"] == 1
    assert report["metrics"]["precision50"] == pytest.approx(0.5)
    assert report["metrics"]["recall50"] == pytest.approx(2 / 3)
    assert 0.0 <= report["metrics"]["map50"] <= 1.0


def test_cvat_evaluator_is_review_only_and_scores_records_json(tmp_path: Path) -> None:
    from scripts.racketsport.rkt_prep_eval_paddle_boxes import evaluate_cvat_review_boxes

    label_dir = tmp_path / "cvat" / "gate_inputs" / "clip_a" / "labels"
    label_dir.mkdir(parents=True)
    paddle_boxes = {
        "annotation": {
            "items": [
                {
                    "clip_id": "clip_a",
                    "frame_index": 0,
                    "label": "paddle",
                    "bbox_xyxy": [10.0, 10.0, 30.0, 30.0],
                },
                {
                    "clip_id": "clip_a",
                    "frame_index": 0,
                    "label": "paddle",
                    "bbox_xyxy": [60.0, 60.0, 90.0, 90.0],
                },
            ]
        }
    }
    (label_dir / "paddle_boxes.json").write_text(json.dumps(paddle_boxes), encoding="utf-8")
    manifest = {
        "clips": [
            {
                "clip_id": "clip_a",
                "datasets": {
                    "paddle": {
                        "path": str(label_dir / "paddle_boxes.json"),
                        "item_count": 2,
                    }
                },
            }
        ]
    }
    manifest_path = tmp_path / "cvat" / "gate_inputs" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "clip_id": "clip_a",
                        "frame_index": 0,
                        "detections": [
                            {"bbox_xyxy": [11.0, 11.0, 29.0, 29.0], "score": 0.9},
                            {"bbox_xyxy": [100.0, 100.0, 130.0, 130.0], "score": 0.8},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_cvat_review_boxes(
        cvat_manifest_path=manifest_path,
        predictions_path=predictions,
        iou_threshold=0.5,
    )

    assert report["dataset"]["source"] == "cvat_paddle_rectangles"
    assert report["dataset"]["review_only"] is True
    assert report["dataset"]["trusted_for_training"] is False
    assert report["metrics"]["ground_truth_count"] == 2
    assert report["metrics"]["prediction_count"] == 2
    assert report["metrics"]["true_positive_count"] == 1
    assert report["metrics"]["precision50"] == pytest.approx(0.5)
    assert report["metrics"]["recall50"] == pytest.approx(0.5)
    assert "allowed for scoring only" in " ".join(report["notes"])


def test_eval_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/rkt_prep_eval_paddle_boxes.py", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--predictions" in completed.stdout
