from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.validate_shot_dataset import validate_manifest
from threed.racketsport.shot_dataset_builder import build_shot_dataset


def _contact_windows() -> dict[str, object]:
    return {
        "schema_version": 1,
        "events": [
            {
                "id": "contact_001",
                "type": "contact",
                "t": 1.0,
                "frame": 60,
                "player_id": 2,
                "confidence": 0.92,
                "sources": {"human_review": 1.0},
                "window": {"t0": 0.55, "t1": 1.45, "importance": 0.9},
            }
        ],
    }


def _reviewed_shot_labels(label: str = "fh_shot", t: float = 1.04) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "human_reviewed",
        "not_ground_truth": False,
        "annotation": {
            "target_file": "events.json",
            "items": [
                {
                    "id": "truth_001",
                    "status": "accepted",
                    "t": t,
                    "frame_index": int(t * 60),
                    "player_id": 2,
                    "shot_label": label,
                }
            ],
        },
    }


def test_build_shot_dataset_writes_validator_compatible_manifest_and_feature_window(tmp_path: Path) -> None:
    manifest = build_shot_dataset(
        dataset_id="tiny_pb_shots",
        clip_id="clip_001",
        truth_events_payload=_reviewed_shot_labels(),
        contact_windows_payload=_contact_windows(),
        out_dir=tmp_path,
        split="train",
        fps=60.0,
        window_ms=900.0,
    )

    manifest_path = tmp_path / "shot_dataset_manifest.json"
    summary = validate_manifest(manifest_path)
    feature_path = tmp_path / manifest["entries"][0]["path"]
    feature = json.loads(feature_path.read_text(encoding="utf-8"))

    assert manifest_path.is_file()
    assert summary["valid"] is True
    assert summary["entry_count"] == 1
    assert manifest["entries"][0] == {
        "id": "clip_001_truth_001",
        "path": "features/clip_001_truth_001.json",
        "split": "train",
        "shot_label": "fh_shot",
        "source_type": "manual_review",
        "fps": 60.0,
        "contact_time_ms": 1040.0,
        "window_ms": 900.0,
        "player_id": "2",
        "notes": "matched_contact_dt_s=0.040",
    }
    assert feature["truth"]["shot_label"] == "fh_shot"
    assert feature["contact"]["id"] == "contact_001"
    assert feature["window"]["start_t"] == pytest.approx(0.59)
    assert feature["window"]["end_t"] == pytest.approx(1.49)


def test_build_shot_dataset_rejects_prediction_output_as_truth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prediction output cannot be used as DATA-5 truth"):
        build_shot_dataset(
            dataset_id="bad",
            clip_id="clip_001",
            truth_events_payload={"artifact_type": "racketsport_shot_classification", "shots": []},
            contact_windows_payload=_contact_windows(),
            out_dir=tmp_path,
            split="train",
            fps=60.0,
        )


def test_build_shot_dataset_rejects_unmatched_reviewed_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no contact within 0.300s for truth_001"):
        build_shot_dataset(
            dataset_id="unmatched",
            clip_id="clip_001",
            truth_events_payload=_reviewed_shot_labels(t=4.0),
            contact_windows_payload=_contact_windows(),
            out_dir=tmp_path,
            split="train",
            fps=60.0,
        )


def test_build_shot_dataset_cli_smoke(tmp_path: Path) -> None:
    truth_path = tmp_path / "events.json"
    contact_path = tmp_path / "contact_windows.json"
    out_dir = tmp_path / "shots"
    truth_path.write_text(json.dumps(_reviewed_shot_labels(label="bh_shot")), encoding="utf-8")
    contact_path.write_text(json.dumps(_contact_windows()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_shot_dataset.py",
            "--truth-events",
            str(truth_path),
            "--contact-windows",
            str(contact_path),
            "--out-dir",
            str(out_dir),
            "--dataset-id",
            "cli_pb_shots",
            "--clip-id",
            "clip_001",
            "--split",
            "val",
            "--fps",
            "60",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["entries"][0]["shot_label"] == "bh_shot"
    assert validate_manifest(out_dir / "shot_dataset_manifest.json")["valid"] is True
