from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from threed.racketsport.shot_trainable_baseline import (
    ShotFeatureSchema,
    abstract_prediction,
    abstract_shot_label,
    phase_label_for_shot,
    train_shot_window_baseline,
)


def _window_payload(sample_id: str, label: str, *, x: float, y: float, speed: float) -> dict[str, object]:
    t = 1.0 + (abs(hash(sample_id)) % 100) / 1000.0
    return {
        "schema_version": 1,
        "dataset_id": "tiny_pb_trainable",
        "clip_id": f"clip_{sample_id}",
        "truth": {"id": sample_id, "t": t, "frame_index": round(t * 60), "player_id": "2", "shot_label": label},
        "contact": {
            "id": f"contact_{sample_id}",
            "t": t,
            "frame": round(t * 60),
            "player_id": "2",
            "confidence": 0.92,
            "window": {"t0": t - 0.05, "t1": t + 0.05, "importance": 0.87},
        },
        "window": {
            "center_t": t,
            "start_t": max(0.0, t - 0.45),
            "end_t": t + 0.45,
            "center_frame": round(t * 60),
            "start_frame": max(0, round((t - 0.45) * 60)),
            "end_frame": round((t + 0.45) * 60),
            "fps": 60.0,
            "window_ms": 900.0,
        },
        "features": {
            "body": {
                "right_wrist_x": x,
                "left_wrist_x": -x,
                "wrist_ball_dx": x,
                "shoulder_rotation": y,
            },
            "ball": {
                "speed_after_mps": speed,
                "court_y_m": y,
            },
        },
    }


def _write_manifest(root: Path) -> Path:
    features_dir = root / "features"
    features_dir.mkdir(parents=True)
    samples = [
        ("train_fh", "fh_drive", "train", 2.0, -4.0, 12.0),
        ("train_bh", "bh_drive", "train", -2.0, -4.0, 12.0),
        ("train_serve", "serve", "train", 0.0, -7.0, 15.0),
        ("val_fh", "fh_drive", "val", 2.1, -4.1, 11.8),
        ("val_bh", "bh_drive", "val", -2.1, -4.1, 11.8),
        ("test_fh", "fh_drive", "test", 2.05, -4.0, 12.2),
        ("test_bh", "bh_drive", "test", -2.05, -4.0, 12.2),
        ("test_serve", "serve", "test", 0.05, -7.1, 15.2),
    ]
    entries: list[dict[str, object]] = []
    for sample_id, label, split, x, y, speed in samples:
        feature_path = features_dir / f"{sample_id}.json"
        feature_path.write_text(
            json.dumps(_window_payload(sample_id, label, x=x, y=y, speed=speed), indent=2),
            encoding="utf-8",
        )
        entries.append(
            {
                "id": sample_id,
                "path": f"features/{sample_id}.json",
                "split": split,
                "shot_label": label,
                "source_type": "manual_review",
                "fps": 60.0,
                "contact_time_ms": 1000.0,
                "window_ms": 900.0,
                "player_id": "2",
            }
        )
    manifest_path = root / "shot_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "tiny_pb_trainable",
                "description": "Tiny trainable shot-window fixture.",
                "entries": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_feature_schema_vectorizes_multimodal_windows_with_masks_and_no_label_leakage() -> None:
    payload = _window_payload("sample", "fh_drive", x=1.5, y=-3.0, speed=10.0)
    del payload["features"]["body"]["left_wrist_x"]  # type: ignore[index]

    schema = ShotFeatureSchema.from_windows([payload])
    vector = schema.vectorize(payload)

    assert "truth.shot_label" not in schema.feature_names
    assert "contact.t" not in schema.feature_names
    assert "window.center_frame" not in schema.feature_names
    assert "features.body.right_wrist_x" in schema.feature_names
    assert schema.vector_size == len(schema.feature_names) * 2
    assert vector.shape == (schema.vector_size,)
    assert np.isfinite(vector).all()
    present_index = schema.feature_names.index("features.body.right_wrist_x")
    assert vector[present_index] == pytest.approx(1.5)
    assert vector[schema.presence_offset + present_index] == 1.0


def test_phase_and_abstract_prediction_policy_backs_off_from_specific_to_side_family() -> None:
    assert phase_label_for_shot("serve") == "serve"
    assert phase_label_for_shot("overhead") == "overhead_candidate"
    assert phase_label_for_shot("fh_drive") == "normal_hit"
    assert abstract_shot_label("bh_drive") == "bh_shot"

    exact = abstract_prediction([("bh_drive", 0.81), ("fh_drive", 0.11)], exact_min_confidence=0.65)
    backed_off = abstract_prediction(
        [("fh_drive", 0.52), ("fh_shot", 0.31), ("bh_drive", 0.17)],
        exact_min_confidence=0.65,
        family_min_confidence=0.75,
    )

    assert exact["type"] == "bh_drive"
    assert exact["abstraction_level"] == "specific"
    assert backed_off["type"] == "fh_shot"
    assert backed_off["specific_type_candidate"] == "fh_drive"
    assert backed_off["abstraction_level"] == "family"
    assert backed_off["type_conf"] == pytest.approx(0.83)


def test_train_shot_window_baseline_scores_holdout_and_keeps_top2_predictions(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)

    payload = train_shot_window_baseline(manifest_path=manifest_path)

    assert payload["status"] == "trainable_baseline_not_poseconv3d_or_bst"
    assert payload["model"]["name"] == "shot_window_centroid_baseline"
    assert payload["splits"]["test"]["accuracy"] == 1.0
    assert payload["splits"]["test"]["macro_f1"] == 1.0
    assert payload["splits"]["test"]["sample_count"] == 3
    assert payload["splits"]["test"]["predictions"][0]["top2"][0]["type"] in {"fh_drive", "bh_drive", "serve"}
    assert payload["splits"]["test"]["predictions"][0]["phase"]["type"] in {"normal_hit", "serve"}


def test_train_shots_cli_smoke_writes_metrics_json(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "dataset")
    out_path = tmp_path / "metrics.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/train_shots.py",
            "--manifest",
            str(manifest_path),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["splits"]["test"]["accuracy"] == 1.0
    assert payload["model"]["feature_schema"]["vector_size"] > 0
