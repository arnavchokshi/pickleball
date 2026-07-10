from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


CLI = Path("scripts/racketsport/build_ball_eval_reset.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reviewed_payload(clip_id: str) -> dict:
    frames = []
    for frame_index in range(4):
        boxes = []
        if frame_index == 1:
            boxes = [
                {
                    "label": "ball",
                    "track_id": 1,
                    "frame_index": frame_index,
                    "bbox_xyxy": [10, 10, 14, 14],
                    "bbox_xywh": [10, 10, 4, 4],
                    "visibility_level": "partial",
                }
            ]
        frames.append({"frame_index": frame_index, "boxes": boxes})
    return {
        "clip_id": clip_id,
        "frames": frames,
        "reviewed_frame_indices": [0, 1],
    }


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    reviewed_root = tmp_path / "reviewed"
    provenance_root = tmp_path / "provenance"
    source_metadata: dict[str, dict] = {}
    clips = ["source_a_rally_0001", "source_b_rally_0001", "source_c_rally_0001"]
    for index, clip_id in enumerate(clips):
        source_id = clip_id.split("_rally_", 1)[0]
        _write_json(reviewed_root / clip_id / "reviewed_boxes.json", _reviewed_payload(clip_id))
        _write_json(
            provenance_root / source_id / f"{clip_id}.provenance.json",
            {
                "source_sha256": hashlib.sha256(source_id.encode()).hexdigest(),
                "source": {"source_id": source_id, "title": source_id, "channel": "fixture"},
            },
        )
        source_metadata[source_id] = {
            "source_id": source_id,
            "source_class": f"class_{index}",
            "title": source_id,
            "channel": "fixture",
        }
    metadata_path = tmp_path / "source_metadata.json"
    _write_json(metadata_path, {"source_metadata": source_metadata})
    corpus_manifest = tmp_path / "corpus_md5_manifest.json"
    _write_json(corpus_manifest, {"artifact_type": "fixture"})
    w5_selection = tmp_path / "w5_selection.json"
    _write_json(
        w5_selection,
        {
            "sessions": [
                {
                    "frames": [
                        {
                            "clip_id": clips[0],
                            "frame_index": 1,
                            "disagreement_type": "large-offset",
                        }
                    ]
                }
            ]
        },
    )
    w6_selection = tmp_path / "w6_selection.json"
    _write_json(
        w6_selection,
        {
            "sessions": [
                {
                    "frames": [
                        {
                            "clip_id": clips[1],
                            "frame_index": 0,
                            "disagreement_type": "teacher-only",
                        }
                    ]
                }
            ]
        },
    )
    legacy_review_root = tmp_path / "legacy_review"
    legacy_review_root.mkdir()
    pre_w7_reviewed_root = tmp_path / "pre_w7_reviewed"
    pre_w7_reviewed_root.mkdir()
    return {
        "reviewed_root": reviewed_root,
        "provenance_root": provenance_root,
        "metadata_path": metadata_path,
        "corpus_manifest": corpus_manifest,
        "w5_selection": w5_selection,
        "w6_selection": w6_selection,
        "legacy_review_root": legacy_review_root,
        "pre_w7_reviewed_root": pre_w7_reviewed_root,
    }


def _run_builder(paths: dict[str, Path], out_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--reviewed-root",
            str(paths["reviewed_root"]),
            "--corpus-md5-manifest",
            str(paths["corpus_manifest"]),
            "--source-metadata-manifest",
            str(paths["metadata_path"]),
            "--provenance-root",
            str(paths["provenance_root"]),
            "--w5-selection-manifest",
            str(paths["w5_selection"]),
            "--w6-selection-manifest",
            str(paths["w6_selection"]),
            "--legacy-review-root",
            str(paths["legacy_review_root"]),
            "--pre-w7-reviewed-root",
            str(paths["pre_w7_reviewed_root"]),
            "--out-root",
            str(out_root),
            "--expected-row-count",
            "6",
            "--reviewed-sample-size",
            "3",
            "--unlabeled-sample-size",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_ball_eval_reset_direct_cli_is_deterministic_and_source_disjoint(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    out_root = tmp_path / "out"

    completed = _run_builder(paths, out_root)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["row_count"] == 6
    assert summary["source_group_count"] == 3
    assert summary["unknown_source_row_count"] == 0
    assert summary["leakage_test"] == "PASS"

    with (out_root / "source_group_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert len({row["row_key"] for row in rows}) == 6
    assert len({row["source_group_id"] for row in rows}) == 3

    grouped = json.loads((out_root / "source_grouped_fold_manifest.json").read_text(encoding="utf-8"))
    assert grouped["fold_count"] == 3
    for fold in grouped["folds"]:
        train = set(fold["train_source_group_ids"])
        selection = set(fold["selection_source_group_ids"])
        test = set(fold["test_source_group_ids"])
        assert not (train & selection or train & test or selection & test)

    before = (out_root / "artifact_manifest.json").read_bytes()
    rerun = _run_builder(paths, out_root)
    assert rerun.returncode == 0, rerun.stderr
    assert (out_root / "artifact_manifest.json").read_bytes() == before

    verified = subprocess.run(
        [sys.executable, str(CLI), "--out-root", str(out_root), "--verify-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
    verification = json.loads(verified.stdout)
    assert verification["all_disjoint"] is True
    assert verification["source_table_unique_row_count"] == 6


def test_build_ball_eval_reset_refuses_protected_path(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    protected_root = tmp_path / "outdoor_webcam_iynbd_labels"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--reviewed-root",
            str(protected_root),
            "--out-root",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "refusing protected source token" in completed.stderr
