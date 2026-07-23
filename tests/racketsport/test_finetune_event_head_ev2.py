from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.racketsport.finetune_event_head as module
from scripts.racketsport.finetune_event_head import (
    _rank_hard_negative_score_rows,
    _full_video_train_source_rate,
    _weighted_window,
    FineTuneInputError,
    assignment_recipe_loss,
    derive_audio_only_hard_negative_pool,
    run_finetune,
    run_internal_stage_f_guards,
)
from threed.racketsport.event_head.model import EventHead, checkpoint_payload


ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "tests/racketsport/fixtures/event_head/tiny.avi"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _row(
    *,
    source_video: str,
    split: str,
    start: int,
    event_class: str | None,
) -> dict[str, object]:
    return {
        "source": "synthetic_ev2",
        "source_video": source_video,
        "video_path": str(VIDEO),
        "media_present": True,
        "split": split,
        "fps": 10.0,
        "source_start_frame": start,
        "num_frames": 3,
        "events": (
            [{"frame": 1, "class": event_class}]
            if event_class is not None else []
        ),
        "loss_validity_mask": [True, True, True],
        "license_posture": "TEST_ONLY",
    }


def _owner_manifest_with_poisoned_val() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "event_head_owner_reviewed_dataset_manifest",
        "teacher_derived": False,
        "ground_truth": True,
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "config": {
            "window_frames": 3,
            "split_unit": "original_source_video_id",
            "train_source_groups": ["owner_train_a", "owner_train_b"],
            "validation_source_groups": ["owner_val_unopened"],
        },
        "protected_seed_check": {
            "status": "pass",
            "overlap_rows": 0,
            "checked_training_windows": 2,
        },
        "rows": [
            _row(
                source_video="owner_train_a",
                split="train",
                start=0,
                event_class="HIT",
            ),
            _row(
                source_video="owner_train_b",
                split="train",
                start=3,
                event_class="BOUNCE",
            ),
            {
                "split": "val",
                "events": {"poison": "must never be validated or decoded"},
                "video_path": 17,
                "num_frames": "not-an-integer",
            },
        ],
    }


def _hard_negative_manifests(tmp_path: Path) -> tuple[Path, Path]:
    old_row = _row(
        source_video="audio_teacher_train",
        split="train",
        start=6,
        event_class="HIT",
    )
    old_row.update({
        "focal_event_id": "audio-only-0001",
        "agreement_count": 1,
        "sample_weight": 0.25,
    })
    old_row["events"][0]["independent_agreements"] = [  # type: ignore[index]
        {"family": "audio_onset", "absolute_delta_s": 0.01}
    ]
    common = {
        "schema_version": 1,
        "artifact_type": "event_head_pbvision_arm_b_dataset_manifest",
        "teacher_derived": True,
        "ground_truth": False,
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
    }
    invalid = {**common, "config": {"window_frames": 3}, "rows": [old_row]}
    repaired = {
        **common,
        "config": {
            "window_frames": 3,
            "arm_b_required_agreement_family": "ball_velocity_kink",
            "audio_only_rejection_reason": "audio_only_no_physical_cue",
        },
        "rows": [],
    }
    return (
        _write_json(tmp_path / "invalid_b.json", invalid),
        _write_json(tmp_path / "repaired_b.json", repaired),
    )


def _registered_hard_negative_manifests(tmp_path: Path) -> tuple[Path, Path]:
    invalid_path, repaired_path = _hard_negative_manifests(tmp_path)
    invalid = json.loads(invalid_path.read_text())
    template = invalid["rows"][0]
    rows = []
    for index in range(292):
        row = json.loads(json.dumps(template))
        focal_event_id = f"audio-only-{index:04d}"
        row["focal_event_id"] = focal_event_id
        row["events"][0]["event_id"] = focal_event_id
        row["source_video"] = (
            "st0epgnab7dr" if index >= 262 else "audio_teacher_train"
        )
        rows.append(row)
    invalid["rows"] = rows
    _write_json(invalid_path, invalid)
    return invalid_path, repaired_path


def _checkpoint(path: Path, *, offset_head: bool = True) -> Path:
    model = EventHead(
        weights="none",
        feature_dim=8,
        hidden_dim=8,
        offset_regression_head=offset_head,
    )
    torch.save(
        checkpoint_payload(
            model,
            image_size=32,
            window_frames=3,
            completed_steps=100,
            best_validation_threshold=0.4,
            license_posture="TEST_ONLY",
            license_reason="synthetic Stage-P fixture",
        ),
        path,
    )
    return path


def _stage_p_bundle(
    tmp_path: Path, *, offset_head: bool = True
) -> tuple[Path, Path, Path]:
    checkpoint = _checkpoint(tmp_path / "stage_p.pt", offset_head=offset_head)
    data_manifest = _write_json(
        tmp_path / "stage_p_data.json", {"artifact_type": "synthetic_stage_p_data"}
    )
    lock = {
        "schema_version": 1,
        "artifact_type": "event_head_stage_p_decode_threshold_lock",
        "verified": False,
        "status": "locked_from_stage_p_internal_validation",
        "owner_val_used": False,
        "data_manifest": str(data_manifest),
        "data_manifest_sha256": _sha256(data_manifest),
        "internal_validation_policy": "sha256_seeded_source_video_holdout",
        "internal_validation_source_videos": ["st0epgnab7dr"],
        "seed": 20260722,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": 100,
        "threshold": 0.4,
        "threshold_grid": [round(0.20 + 0.05 * index, 2) for index in range(11)],
        "threshold_tie_break": [
            "macro_f1_at_2_desc", "fp_asc", "fn_asc", "threshold_asc",
            "checkpoint_step_asc_strict_tie",
        ],
        "nms_radius_frames": 2,
        "match_tolerance_frames": 2,
    }
    lock_path = _write_json(tmp_path / "stage_p_decode_threshold_lock.json", lock)
    train_manifest = _write_json(tmp_path / "train_manifest.json", {
        "decode_threshold_lock": str(lock_path),
        "decode_threshold_lock_sha256": _sha256(lock_path),
        "best_validation_threshold": 0.4,
        "locked_decode_threshold": 0.4,
        "best_validation_step": 100,
        "best_checkpoint": str(checkpoint),
        "data_manifest": str(data_manifest),
        "data_manifest_sha256": _sha256(data_manifest),
    })
    return checkpoint, lock_path, train_manifest


def _registered_media_root(tmp_path: Path) -> Path:
    root = tmp_path / "owner_media"
    for source in (
        "73VurrTKCZ8", "Ezz6HDNHlnk", "_L0HVmAlCQI", "wBu8bC4OfUY",
        "HyUqT7zFiwk", "zwCtH_i1_S4",
    ):
        (root / source).mkdir(parents=True)
    return root


def test_audio_only_delta_is_deterministic_all_background_training_truth(
    tmp_path: Path,
) -> None:
    invalid, repaired = _hard_negative_manifests(tmp_path)
    kwargs = {
        "invalid_manifest_sha256": _sha256(invalid),
        "repaired_manifest_sha256": _sha256(repaired),
        "expected_candidates": 1,
        "window_frames": 3,
    }

    first, first_report = derive_audio_only_hard_negative_pool(
        invalid, repaired, **kwargs
    )
    second, second_report = derive_audio_only_hard_negative_pool(
        invalid, repaired, **kwargs
    )

    assert first == second
    assert first_report == second_report
    assert len(first) == 1
    candidate = first[0]
    assert candidate.focal_event_id == "audio-only-0001"
    assert candidate.excluded_event_frame == 1
    assert candidate.excluded_event_class == 1
    assert candidate.window.spec.events == ()
    assert candidate.window.spec.validity_mask == (True, True, True)
    assert candidate.window.is_hard_negative is True
    assert candidate.window.is_pseudo is False
    assert first_report["candidate_agreement_family_counts"] == {"audio_onset": 1}


def test_hard_negative_delta_rejects_non_audio_agreement_family(
    tmp_path: Path,
) -> None:
    invalid, repaired = _hard_negative_manifests(tmp_path)
    payload = json.loads(invalid.read_text())
    payload["rows"][0]["events"][0]["independent_agreements"][0]["family"] = (
        "ball_velocity_kink"
    )
    _write_json(invalid, payload)

    with pytest.raises(FineTuneInputError, match="is not audio-only"):
        derive_audio_only_hard_negative_pool(
            invalid,
            repaired,
            invalid_manifest_sha256=_sha256(invalid),
            repaired_manifest_sha256=_sha256(repaired),
            expected_candidates=1,
            window_frames=3,
        )


def test_weighted_window_propagates_subframe_offset_and_source_video() -> None:
    row = _row(
        source_video="owner_train_a", split="train", start=0, event_class="HIT"
    )
    row["events"][0]["subframe_offset_frames"] = 0.25  # type: ignore[index]

    window = _weighted_window(
        row,  # type: ignore[arg-type]
        local_start=0,
        window_frames=3,
        sample_weight=1.0,
        is_pseudo=False,
        is_hard_negative=False,
        row_index=7,
    )

    assert window.spec.event_subframe_offsets == (0.25,)
    assert window.spec.source_video == "owner_train_a"
    assert window.spec.row_index == 7


def test_hard_negative_score_ranking_is_stable_at_ties() -> None:
    rows = [
        {
            "focal_event_id": "z",
            "max_positive_probability_at_excluded_frame": 0.8,
        },
        {
            "focal_event_id": "a",
            "max_positive_probability_at_excluded_frame": 0.8,
        },
        {
            "focal_event_id": "m",
            "max_positive_probability_at_excluded_frame": 0.7,
        },
    ]

    ranked = _rank_hard_negative_score_rows(rows, top_k=2)

    assert [row["focal_event_id"] for row in ranked] == ["a", "z"]


def test_assignment_loss_caps_hard_negative_contribution_and_backpropagates() -> None:
    logits = torch.tensor(
        [
            [[3.0, -1.0, -1.0]],
            [[-3.0, 3.0, 3.0]],
        ],
        requires_grad=True,
    )
    loss, stats = assignment_recipe_loss(
        logits,
        torch.zeros((2, 1), dtype=torch.long),
        torch.ones((2, 3), dtype=torch.bool),
        torch.ones((2, 1), dtype=torch.bool),
        torch.zeros((2, 1)),
        predicted_offsets=None,
        class_weights=torch.ones(3),
        sample_weights=torch.ones(2),
        is_pseudo=torch.tensor([False, False]),
        is_hard_negative=torch.tensor([False, True]),
        pseudo_weight_cap=1.0,
        hard_negative_loss_cap=0.5,
        assignment_mode="fixed",
        assignment_max_shift_frames=0,
        assignment_class_cost_weight=1.0,
        assignment_temporal_cost_weight=1.0,
        label_dilation_frames=1,
        neighbor_positive_weight=0.5,
        offset_loss_weight=0.0,
        offset_smooth_l1_beta=1.0,
    )

    assert stats["hard_negative_capped"] is True
    assert stats["effective_hard_negative_loss"] <= (
        0.5 * stats["reference_human_loss_for_hard_negative"] + 1e-7
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_final_step_probe_never_calls_legacy_val_or_protected_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _write_json(
        tmp_path / "owner.json", _owner_manifest_with_poisoned_val()
    )
    invalid, repaired = _registered_hard_negative_manifests(tmp_path)
    checkpoint, threshold_lock, stage_p_manifest = _stage_p_bundle(tmp_path)
    media_root = _registered_media_root(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("sealed/owner-val path was touched")

    monkeypatch.setattr(module, "validate_manifests", forbidden)
    monkeypatch.setattr(
        module, "_validate_registered_final_step_arguments", lambda _arguments: None
    )
    monkeypatch.setattr(module, "_validation_windows", forbidden)
    monkeypatch.setattr(module, "_protected_frames", forbidden)
    monkeypatch.setattr(
        module,
        "mine_hard_negatives",
        lambda _model, candidates, **_kwargs: (
            [candidate.window for candidate in candidates[:96]],
            {"selected": [
                {"focal_event_id": candidate.focal_event_id}
                for candidate in candidates[:96]
            ]},
        ),
    )

    report = run_finetune(
        owner_manifest_path=owner,
        pseudo_manifest_path=None,
        init_checkpoint_model_only=checkpoint,
        out=tmp_path / "out",
        device_name="cpu",
        steps=1,
        image_size=32,
        window_frames=3,
        batch_size=8,
        lr=1e-3,
        val_every=1,
        seed=20260722,
        stride_frames=3,
        num_workers=0,
        class_weights=(1.0, 5.0, 5.0),
        pseudo_weight_cap=1.0,
        checkpoint_selection="final-step",
        owner_manifest_sha256=_sha256(owner),
        init_checkpoint_sha256=_sha256(checkpoint),
        hard_negative_invalid_manifest_path=invalid,
        hard_negative_repaired_manifest_path=repaired,
        hard_negative_invalid_manifest_sha256=_sha256(invalid),
        hard_negative_repaired_manifest_sha256=_sha256(repaired),
        hard_negative_expected_candidates=262,
        hard_negative_top_k=96,
        hard_negative_batch_size=4,
        hard_negative_excluded_source_video_ids=("st0epgnab7dr",),
        hard_negative_loss_cap=0.5,
        class_weighting="sqrt-frequency",
        assignment_mode="fixed",
        assignment_temporal_cost_weight=0.25,
        label_dilation_frames=1,
        label_neighbor_positive_weight=0.5,
        offset_loss_weight=0.2,
        internal_decode_threshold=0.4,
        stage_p_threshold_lock_path=threshold_lock,
        stage_p_train_manifest_path=stage_p_manifest,
        owner_media_root=media_root,
        rate_media_inventory_path=tmp_path / "unused_rate_media_lock.json",
        rate_media_inventory_sha256="0" * 64,
        owner_train_source_video_ids=(
            "73VurrTKCZ8", "Ezz6HDNHlnk", "_L0HVmAlCQI", "wBu8bC4OfUY",
        ),
        owner_validation_source_video_ids=("HyUqT7zFiwk", "zwCtH_i1_S4"),
        probe_only=True,
        expected_owner_train_rows=2,
        expected_owner_val_rows=1,
    )

    assert report["status"] == "complete_probe_only"
    assert report["probe_only"] is True
    assert report["owner_score_eligible"] is False
    assert report["validation_windows"] == 0
    assert report["validations"] == []
    assert report["best_val_macro_f1_at_2"] is None
    assert report["owner_validation_rows_uninspected"] == 1
    assert report["provenance"]["protected_inventory_opened"] is False
    assert report["hard_negative_candidate_rows"] == 262
    assert report["hard_negative_train_windows"] == 96
    assert report["hard_negative_mining"]["selected"][0][
        "focal_event_id"
    ] == "audio-only-0000"
    assert report["config"]["class_weighting"] == "sqrt-frequency"
    assert report["config"]["label_dilation_frames"] == 1
    assert report["config"]["offset_loss_weight"] == 0.2
    assert report["internal_guards"]["status"] == "not_run_probe_only"
    assert report["elapsed_total_s"] >= report["elapsed_training_s"]
    saved = torch.load(
        report["best_checkpoint"], map_location="cpu", weights_only=False
    )
    assert saved["checkpoint_role"] == (
        "terminal_step_probe_only_not_owner_score_eligible"
    )
    assert saved["model_config"]["offset_regression_head"] is True
    assert saved["config"]["owner_score_eligible"] is False


@pytest.mark.parametrize("guard_pass", [True, False])
def test_terminal_internal_guard_controls_owner_score_eligibility_without_val(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guard_pass: bool,
) -> None:
    owner = _write_json(
        tmp_path / "owner.json", _owner_manifest_with_poisoned_val()
    )
    invalid, repaired = _registered_hard_negative_manifests(tmp_path)
    checkpoint, threshold_lock, stage_p_manifest = _stage_p_bundle(tmp_path)
    media_root = _registered_media_root(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("owner validation was touched")

    monkeypatch.setattr(module, "validate_manifests", forbidden)
    monkeypatch.setattr(
        module, "_validate_registered_final_step_arguments", lambda _arguments: None
    )
    monkeypatch.setattr(module, "_validation_windows", forbidden)
    monkeypatch.setattr(module, "_protected_frames", forbidden)
    monkeypatch.setattr(
        module,
        "mine_hard_negatives",
        lambda _model, candidates, **_kwargs: (
            [candidate.window for candidate in candidates[:96]],
            {"selected": [
                {"focal_event_id": candidate.focal_event_id}
                for candidate in candidates[:96]
            ]},
        ),
    )
    monkeypatch.setattr(
        module,
        "run_internal_stage_f_guards",
        lambda *_args, **_kwargs: {
            "policy": "synthetic_train_side_only_guard",
            "pass": guard_pass,
            "owner_validation_constructed": False,
            "owner_validation_scored": False,
            "protected_inventory_opened": False,
        },
    )

    report = run_finetune(
        owner_manifest_path=owner,
        pseudo_manifest_path=None,
        init_checkpoint_model_only=checkpoint,
        out=tmp_path / "out",
        device_name="cpu",
        steps=1,
        image_size=32,
        window_frames=3,
        batch_size=8,
        lr=1e-3,
        val_every=1,
        seed=20260722,
        stride_frames=3,
        num_workers=0,
        class_weights=(1.0, 5.0, 5.0),
        pseudo_weight_cap=1.0,
        checkpoint_selection="final-step",
        owner_manifest_sha256=_sha256(owner),
        init_checkpoint_sha256=_sha256(checkpoint),
        hard_negative_invalid_manifest_path=invalid,
        hard_negative_repaired_manifest_path=repaired,
        hard_negative_invalid_manifest_sha256=_sha256(invalid),
        hard_negative_repaired_manifest_sha256=_sha256(repaired),
        hard_negative_expected_candidates=262,
        hard_negative_top_k=96,
        hard_negative_batch_size=4,
        hard_negative_excluded_source_video_ids=("st0epgnab7dr",),
        hard_negative_loss_cap=0.5,
        class_weighting="sqrt-frequency",
        assignment_mode="fixed",
        assignment_temporal_cost_weight=0.25,
        label_dilation_frames=1,
        label_neighbor_positive_weight=0.5,
        offset_loss_weight=0.2,
        internal_decode_threshold=0.4,
        stage_p_threshold_lock_path=threshold_lock,
        stage_p_train_manifest_path=stage_p_manifest,
        owner_media_root=media_root,
        rate_media_inventory_path=tmp_path / "unused_rate_media_lock.json",
        rate_media_inventory_sha256="0" * 64,
        owner_train_source_video_ids=(
            "73VurrTKCZ8", "Ezz6HDNHlnk", "_L0HVmAlCQI", "wBu8bC4OfUY",
        ),
        owner_validation_source_video_ids=("HyUqT7zFiwk", "zwCtH_i1_S4"),
        expected_owner_train_rows=2,
        expected_owner_val_rows=1,
    )

    assert report["owner_score_eligible"] is guard_pass
    assert report["validations"] == []
    assert report["validation_windows"] == 0
    assert report["status"] == (
        "complete" if guard_pass else "complete_internal_guard_fail"
    )
    saved = torch.load(
        report["best_checkpoint"], map_location="cpu", weights_only=False
    )
    assert saved["config"]["owner_score_eligible"] is guard_pass
    assert saved["checkpoint_role"] == (
        "terminal_step_internal_guards_pass"
        if guard_pass else
        "terminal_step_internal_guards_fail_not_owner_score_eligible"
    )


def test_final_step_legacy_defaults_rejected_through_direct_cli(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "must-not-be-opened"
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "scripts/racketsport/finetune_event_head.py",
            "--owner-manifest", str(missing),
            "--owner-manifest-sha256", "0" * 64,
            "--init-checkpoint-model-only", str(missing),
            "--init-checkpoint-sha256", "0" * 64,
            "--hard-negative-invalid-manifest", str(missing),
            "--hard-negative-invalid-manifest-sha256", "0" * 64,
            "--hard-negative-repaired-manifest", str(missing),
            "--hard-negative-repaired-manifest-sha256", "0" * 64,
            "--hard-negative-expected-candidates", "262",
            "--hard-negative-top-k", "96",
            "--hard-negative-batch-size", "4",
            "--hard-negative-excluded-source-video", "st0epgnab7dr",
            "--hard-negative-loss-cap", "0.5",
            "--checkpoint-selection", "final-step",
            "--probe-only",
            "--internal-decode-threshold", "0.4",
            "--stage-p-threshold-lock", str(missing),
            "--stage-p-train-manifest", str(missing),
            "--owner-media-root", str(missing),
            "--rate-media-inventory", str(missing),
            "--rate-media-inventory-sha256", "0" * 64,
            "--owner-train-source-video", "73VurrTKCZ8",
            "--owner-train-source-video", "Ezz6HDNHlnk",
            "--owner-train-source-video", "_L0HVmAlCQI",
            "--owner-train-source-video", "wBu8bC4OfUY",
            "--owner-validation-source-video", "HyUqT7zFiwk",
            "--owner-validation-source-video", "zwCtH_i1_S4",
            "--expected-owner-train-rows", "61",
            "--expected-owner-val-rows", "41",
            "--steps", "100",
            "--image-size", "224",
            "--window-frames", "64",
            "--batch-size", "8",
            "--lr", "0.001",
            "--val-every", "100",
            "--stride-frames", "32",
            "--num-workers", "4",
            "--seed", "20260722",
            "--max-wall-minutes", "180",
            "--out", str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 20
    assert "before input read" in completed.stderr
    assert "owner manifest is absent" not in completed.stderr


@pytest.mark.parametrize(
    ("owner_fp", "audio_fired", "rate", "expected_pass"),
    [
        (2, 26, 0.3, True),
        (3, 26, 0.3, False),
        (2, 27, 0.3, False),
        (2, 26, 0.299, False),
        (2, 26, 1.001, False),
    ],
)
def test_internal_stage_f_guard_aggregates_all_three_registered_checks(
    monkeypatch: pytest.MonkeyPatch,
    owner_fp: int,
    audio_fired: int,
    rate: float,
    expected_pass: bool,
) -> None:
    monkeypatch.setattr(
        module,
        "_owner_train_negative_windows",
        lambda *_args, **_kwargs: [object()] * 21,
    )
    proxy_results = iter([
        {
            "rows": 21,
            "predicted_events": owner_fp,
            "rows_with_predictions": min(owner_fp, 21),
            "per_row": [],
        },
        {
            "rows": 262,
            "predicted_events": audio_fired,
            "rows_with_predictions": audio_fired,
            "per_row": [],
        },
    ])
    monkeypatch.setattr(
        module,
        "_prediction_proxy_for_windows",
        lambda *_args, **_kwargs: next(proxy_results),
    )
    monkeypatch.setattr(
        module,
        "_full_video_train_source_rate",
        lambda *_args, **_kwargs: {
            "events_per_second": rate,
            "distinct_source_video_count": 4,
            "unique_media_path_count": 38,
        },
    )
    candidates = [SimpleNamespace(window=object()) for _ in range(262)]

    report = run_internal_stage_f_guards(
        torch.nn.Identity(),
        {"rows": []},
        candidates,  # type: ignore[arg-type]
        image_size=32,
        window_frames=3,
        batch_size=2,
        device=torch.device("cpu"),
        num_workers=0,
        seed=20260722,
        threshold=0.4,
        owner_media_root=Path("unused-by-mock"),
        train_source_video_ids=(
            "73VurrTKCZ8", "Ezz6HDNHlnk", "_L0HVmAlCQI", "wBu8bC4OfUY",
        ),
        validation_source_video_ids=("HyUqT7zFiwk", "zwCtH_i1_S4"),
        expected_owner_negative_rows=21,
        owner_negative_max_fp=2,
        audio_only_max_fired_rows=26,
        rate_min_per_s=0.3,
        rate_max_per_s=1.0,
        expected_train_media_paths=38,
        expected_train_source_videos=4,
        rate_media_inventory_path=Path("unused-by-mock"),
        rate_media_inventory_sha256="0" * 64,
    )

    assert report["checks"]["owner_train_negative_fp"]["value"] == owner_fp
    assert report["checks"]["audio_only_rows_with_predictions"]["value"] == (
        audio_fired
    )
    assert report["checks"]["full_video_rate_per_s"]["value"] == rate
    assert report["pass"] is expected_pass
    assert report["owner_validation_constructed"] is False
    assert report["protected_inventory_opened"] is False
