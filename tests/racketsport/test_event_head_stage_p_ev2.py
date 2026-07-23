from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import torch

from scripts.racketsport.train_event_head import (
    CORPUS_PBVISION_AGREEMENT,
    STAGE_P_THRESHOLD_LOCK_FILENAME,
    _threshold_selection_key,
    _validate_full_training_manifest,
    _validation_threshold_sweep,
)
from threed.racketsport.event_head.model import EventHead, checkpoint_payload


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/racketsport/train_event_head.py"
ACTUAL_STAGE_P_MANIFEST = (
    ROOT
    / "runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/arm_b_manifest.json"
)


def _agreement_row(
    *, source_video: str, source_start_frame: int, class_name: str,
    sample_weight: float,
) -> dict[str, object]:
    event_id = f"{source_video}-{class_name.lower()}"
    return {
        "source": "pbvision_teacher_predictions",
        "source_video": source_video,
        "video": event_id,
        "video_path": str(ROOT / "tests/racketsport/fixtures/event_head/tiny.avi"),
        "media_present": True,
        "split": "train",
        "fps": 10.0,
        "source_start_frame": source_start_frame,
        "num_frames": 3,
        "events": [{
            "frame": 1,
            "class": class_name,
            "event_id": event_id,
            "subframe_offset_frames": 0.25,
            "independent_agreements": [{"family": "ball_velocity_kink"}],
        }],
        "loss_validity_mask": [True, True, True],
        "unknown_frame_mask": [False, False, False],
        "license_posture": "pbvision_signed_full_usage",
        "training_eligible": True,
        "sample_weight": sample_weight,
    }


def _agreement_manifest(path: Path) -> Path:
    rows = []
    for source_index, source_video in enumerate(("source_a", "source_b")):
        rows.extend([
            _agreement_row(
                source_video=source_video,
                source_start_frame=source_index * 3,
                class_name="HIT",
                sample_weight=0.25,
            ),
            _agreement_row(
                source_video=source_video,
                source_start_frame=source_index * 3,
                class_name="BOUNCE",
                sample_weight=0.5,
            ),
        ])
    manifest = {
        "schema_version": 2,
        "artifact_type": "event_head_pbvision_arm_b_dataset_manifest",
        "verified": False,
        "teacher_derived": True,
        "ground_truth": False,
        "arm": "B",
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "permanent_compare_only_denylist": ["compare_only_source"],
        "rows": rows,
    }
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return path


class _CountingValidationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        logits = torch.tensor([[
            [8.0, -8.0, -8.0],
            [-8.0, 8.0, -8.0],
            [-8.0, -8.0, 12.0],
        ]])
        self.register_buffer("fixture_logits", logits)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.fixture_logits.expand(frames.shape[0], -1, -1)


def test_internal_threshold_sweep_runs_inference_once_masks_unknown_and_ties_low() -> None:
    model = _CountingValidationModel()
    loader = [{
        "frames": torch.zeros((1, 3, 3, 4, 4)),
        "targets": torch.tensor([[0, 1, 0]]),
        "validity_mask": torch.ones((1, 3), dtype=torch.bool),
        # The strongest BOUNCE logit is UNKNOWN and must not become a false positive.
        "frame_loss_mask": torch.tensor([[True, True, False]]),
    }]

    selected = _validation_threshold_sweep(
        model,
        loader,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        thresholds=(0.2, 0.5, 0.7),
        nms_radius=2,
    )

    assert model.calls == 1
    assert selected["macro_f1_at_2"] == 0.5
    assert selected["fp"] == 0
    assert selected["fn"] == 0
    assert selected["threshold"] == 0.2


def test_stage_p_threshold_key_freezes_fp_then_fn_then_lower_threshold() -> None:
    baseline = {"macro_f1_at_2": 0.4, "fp": 2, "fn": 3, "threshold": 0.2}
    lower_fp = {**baseline, "fp": 1, "fn": 99, "threshold": 0.7}
    lower_fn = {**baseline, "fn": 2, "threshold": 0.7}
    lower_threshold = {**baseline, "threshold": 0.15}

    assert _threshold_selection_key(lower_fp) > _threshold_selection_key(baseline)
    assert _threshold_selection_key(lower_fn) > _threshold_selection_key(baseline)
    assert _threshold_selection_key(lower_threshold) > _threshold_selection_key(baseline)


def test_actual_stage_p_manifest_and_weight_tiers_are_accepted() -> None:
    manifest = json.loads(ACTUAL_STAGE_P_MANIFEST.read_text())

    _validate_full_training_manifest(
        manifest, corpus_kind=CORPUS_PBVISION_AGREEMENT,
    )

    assert len(manifest["rows"]) == 1189
    assert Counter(row["sample_weight"] for row in manifest["rows"]) == {
        0.25: 803,
        0.5: 386,
    }


def test_stage_p_cli_wires_recipe_weights_reset_and_decode_lock(tmp_path: Path) -> None:
    manifest_path = _agreement_manifest(tmp_path / "agreement.json")
    source_model = EventHead(weights="none")
    init_checkpoint = tmp_path / "t20_init.pt"
    torch.save(
        checkpoint_payload(
            source_model,
            completed_steps=9000,
            best_val_f1=0.7,
            best_val_max_positive_class_probability=0.9,
        ),
        init_checkpoint,
    )
    out = tmp_path / "stage_p"
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(CLI),
            "--full",
            "--manifest", str(manifest_path),
            "--device", "cpu",
            "--out", str(out),
            "--steps", "1",
            "--image-size", "32",
            "--window-frames", "3",
            "--batch-size", "2",
            "--lr", "0.001",
            "--val-every", "1",
            "--seed", "20260722",
            "--num-workers", "0",
            "--corpus-kind", "pbvision-agreement",
            "--internal-val-source-count", "1",
            "--sqrt-frequency-class-weights",
            "--label-dilation-frames", "1",
            "--label-dilation-soft-weight", "0.5",
            "--label-assignment", "hungarian",
            "--assignment-max-shift-frames", "2",
            "--assignment-class-cost-weight", "1.0",
            "--assignment-temporal-cost-weight", "0.25",
            "--offset-regression-head",
            "--offset-loss-weight", "0.2",
            "--validation-thresholds", "0.2", "0.25", "0.3",
            "--validation-nms-radius", "2",
            "--init-checkpoint-model-only", str(init_checkpoint),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    report = json.loads((out / "train_manifest.json").read_text())
    config = report["config"]
    assert report["status"] == "complete"
    assert report["verified"] is False
    assert report["start_step"] == 0
    assert report["completed_steps"] == 1
    assert report["init_checkpoint_completed_steps"] == 9000
    assert report["resume_mode"] == "model_only"
    assert "pbvision_signed_full_usage" in report["license_reason"]
    assert "public broadcast pixels" not in report["license_reason"]
    assert config["training_objective"] == "ev2_dense_assignment"
    assert config["class_weighting"] == "sqrt_frequency"
    assert config["class_weights"][0] == 1.0
    assert all(value > 0 for value in config["class_weights"])
    assert config["train_window_sample_weight_counts"] == {"0.25": 1, "0.5": 1}
    assert config["train_window_sample_weight_total"] == 0.75
    assert len(report["recipe_loss_stats"]) == 1
    assert report["recipe_loss_stats"][0]["batch_sample_weight_sum"] == 0.75
    assert report["assignment_totals"]["event_count"] == 2
    assert report["best_val_metric"] == "macro_f1_at_2_internal"

    lock_path = out / STAGE_P_THRESHOLD_LOCK_FILENAME
    assert report["decode_threshold_lock"] == str(lock_path)
    assert lock_path.is_file()
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    assert report["decode_threshold_lock_sha256"] == hashlib.sha256(lock_bytes).hexdigest()
    assert lock["owner_val_used"] is False
    assert lock["threshold"] in {0.2, 0.25, 0.3}
    assert lock["threshold"] == report["locked_decode_threshold"]
    assert lock["nms_radius_frames"] == 2
    assert lock["threshold_tie_break"] == [
        "macro_f1_at_2_desc", "fp_asc", "fn_asc", "threshold_asc",
        "checkpoint_step_asc_strict_tie",
    ]
    assert len(lock["internal_validation_source_videos"]) == 1
    assert lock["checkpoint"] == report["best_checkpoint"]
    assert lock["checkpoint_step"] == report["best_validation_step"]

    best_payload = torch.load(
        report["best_checkpoint"], map_location="cpu", weights_only=False
    )
    assert best_payload["model_config"]["offset_regression_head"] is True
    assert best_payload["completed_steps"] == 1
    assert best_payload["best_validation_threshold"] == lock["threshold"]
