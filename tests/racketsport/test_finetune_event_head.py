from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.racketsport.finetune_event_head import (
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_WINDOW_FRAMES,
    EXPECTED_OWNER_TRAIN_ROWS,
    EXPECTED_OWNER_VAL_ROWS,
    FineTuneInputError,
    WeightedEventWindowDataset,
    _assert_checkpoint_context,
    _training_windows,
    _validation_metrics_from_batches,
    build_parser,
    run_finetune,
    validate_manifests,
    weighted_masked_cross_entropy,
)
from threed.racketsport.event_head.model import EventHead, checkpoint_payload


ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "tests/racketsport/fixtures/event_head/tiny.avi"


def _row(
    *,
    source_video: str,
    split: str,
    start: int,
    event_class: str | None,
    pseudo_weight: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "synthetic_pseudo" if pseudo_weight is not None else "owner_reviewed",
        "video": source_video,
        "source_video": source_video,
        "video_path": str(VIDEO),
        "media_present": True,
        "split": split,
        "fps": 10.0,
        "source_start_frame": start,
        "num_frames": 3,
        "event_counts": {
            "HIT": int(event_class == "HIT"),
            "BOUNCE": int(event_class == "BOUNCE"),
            "background": int(event_class is None),
        },
        "inventory_event_count": int(event_class is not None),
        "events": (
            [{"frame": 1, "class": event_class}] if event_class is not None else []
        ),
        "loss_validity_mask": [True, True, True],
        "license_id": "synthetic_fixture",
        "license_posture": "TEST_ONLY",
    }
    if pseudo_weight is not None:
        row["sample_weight"] = pseudo_weight
        row["agreement_count"] = 2 if pseudo_weight == 0.5 else 1
    return row


def _manifest(rows: list[dict[str, object]], *, pseudo: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": (
            "event_head_synthetic_teacher_dataset_manifest"
            if pseudo
            else "event_head_owner_reviewed_dataset_manifest"
        ),
        "verified": False,
        "teacher_derived": True if pseudo else False,
        "ground_truth": False if pseudo else True,
        "config": {"window_frames": 3, "split_unit": "source_video"},
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "license_posture": "TEST_ONLY",
        "rows": rows,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _checkpoint(path: Path, *, window_frames: int = 3) -> Path:
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8)
    torch.save(
        checkpoint_payload(
            model,
            license_posture="TEST_ONLY",
            license_reason="synthetic fixture",
            image_size=32,
            window_frames=window_frames,
            completed_steps=9000,
            optimizer_state_dict={"state": {"must_not_restore": True}},
        ),
        path,
    )
    return path


def test_defaults_pin_64_frame_context_and_weighted_ce() -> None:
    args = build_parser().parse_args(["--out", "unused"])
    assert args.window_frames == DEFAULT_WINDOW_FRAMES == 64
    assert tuple(args.class_weights) == DEFAULT_CLASS_WEIGHTS == (1.0, 5.0, 5.0)
    assert args.pseudo_weight_cap == 1.0


def test_current_schema_default_contract_accepts_exact_owner_61_41_split(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            source_video=f"owner_train_{index}",
            split="train",
            start=0,
            event_class="HIT" if index % 2 else None,
        )
        for index in range(EXPECTED_OWNER_TRAIN_ROWS)
    ] + [
        _row(
            source_video=f"owner_val_{index}",
            split="val",
            start=0,
            event_class="BOUNCE" if index % 2 else None,
        )
        for index in range(EXPECTED_OWNER_VAL_ROWS)
    ]
    owner_path = _write_json(tmp_path / "owner_102.json", _manifest(rows))
    seed_path = _write_json(tmp_path / "empty_protected_seed.json", {"labels": []})

    owner, _, pseudo, _ = validate_manifests(
        owner_path,
        None,
        window_frames=3,
        seed_path=seed_path,
    )

    assert sum(row["split"] == "train" for row in owner["rows"]) == 61
    assert sum(row["split"] == "val" for row in owner["rows"]) == 41
    assert pseudo is None


def test_tiny_current_schema_run_reports_owner_val_and_capped_pseudo_weight(
    tmp_path: Path,
) -> None:
    owner_path = _write_json(
        tmp_path / "owner.json",
        _manifest([
            _row(source_video="owner_train", split="train", start=0, event_class="HIT"),
            _row(source_video="owner_val", split="val", start=3, event_class="BOUNCE"),
        ]),
    )
    pseudo_path = _write_json(
        tmp_path / "pseudo.json",
        _manifest([
            _row(
                source_video="pseudo_train",
                split="train",
                start=6,
                event_class="HIT",
                pseudo_weight=0.5,
            )
        ], pseudo=True),
    )
    report = run_finetune(
        owner_manifest_path=owner_path,
        pseudo_manifest_path=pseudo_path,
        init_checkpoint_model_only=_checkpoint(tmp_path / "pretrain.pt"),
        out=tmp_path / "out",
        device_name="cpu",
        steps=1,
        image_size=32,
        window_frames=3,
        batch_size=1,
        lr=1e-3,
        val_every=1,
        seed=20260720,
        stride_frames=3,
        num_workers=0,
        class_weights=DEFAULT_CLASS_WEIGHTS,
        pseudo_weight_cap=0.25,
        expected_owner_train_rows=1,
        expected_owner_val_rows=1,
    )

    assert report["status"] == "complete"
    assert report["owner_train_rows"] == 1
    assert report["owner_validation_rows"] == report["validation_windows"] == 1
    assert report["validation_protocol"] == (
        "fixed_owner_val_only_macro_F1_at_plus_minus_2_frames"
    )
    assert report["validations"][-1]["validation_rows"] == 1
    assert "macro_f1_at_2" in report["validations"][-1]
    assert report["config"]["class_weights"] == [1.0, 5.0, 5.0]
    assert report["batch_weighting"]["cap_basis"] == (
        "post_class_and_frame_weighted_aggregate_loss"
    )
    assert report["batch_weighting"]["effective_pseudo_loss_fraction_max"] <= (
        0.25 / 1.25 + 1e-6
    )
    assert report["resume_mode"] == "model_only"
    assert report["optimizer_state_restored"] is False
    assert report["provenance"]["init_checkpoint_completed_steps"] == 9000
    saved = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
    assert saved["window_frames"] == 3
    assert saved["config"]["class_weights"] == [1.0, 5.0, 5.0]


def test_wall_exit_fails_arm_and_removes_standard_checkpoints(tmp_path: Path) -> None:
    owner_path = _write_json(
        tmp_path / "owner.json",
        _manifest([
            _row(source_video="owner_train", split="train", start=0, event_class="HIT"),
            _row(source_video="owner_val", split="val", start=3, event_class="BOUNCE"),
        ]),
    )
    out = tmp_path / "wall_stopped"
    out.mkdir()
    (out / "finetune_manifest.json").write_text(
        json.dumps({"status": "stale_complete"}) + "\n"
    )

    with pytest.raises(FineTuneInputError) as caught:
        run_finetune(
            owner_manifest_path=owner_path,
            pseudo_manifest_path=None,
            init_checkpoint_model_only=_checkpoint(tmp_path / "pretrain.pt"),
            out=out,
            device_name="cpu",
            steps=2,
            image_size=32,
            window_frames=3,
            batch_size=1,
            lr=1e-3,
            val_every=1,
            seed=20260720,
            stride_frames=3,
            num_workers=0,
            class_weights=DEFAULT_CLASS_WEIGHTS,
            pseudo_weight_cap=1.0,
            expected_owner_train_rows=1,
            expected_owner_val_rows=1,
            max_wall_minutes=1e-12,
        )

    assert caught.value.exit_code == 31
    assert "INCOMPLETE_ARM_STEPS" in str(caught.value)
    failure = json.loads((out / "arm_failure.json").read_text())
    assert failure["completed_steps"] < failure["target_steps"] == 2
    assert failure["equal_step_eligible"] is False
    assert not (out / "finetune_manifest.json").exists()
    assert not (out / ".finetune_manifest.json.tmp").exists()
    assert not (out / "best_event_head_finetuned.pt").exists()
    assert not (out / "event_head_finetuned.pt").exists()


def test_macro_f1_at_2_is_per_class_macro_not_micro() -> None:
    logits = torch.full((1, 7, 3), -10.0)
    logits[..., 0] = 10.0
    logits[0, 2] = torch.tensor([-10.0, 10.0, -10.0])
    logits[0, 4] = torch.tensor([-10.0, -10.0, 10.0])
    targets = torch.tensor([[0, 0, 1, 0, 2, 0, 0]])
    masks = torch.ones((1, 3), dtype=torch.bool)

    metrics = _validation_metrics_from_batches([(logits, targets, masks)])

    assert metrics["macro_f1_at_2"] == 1.0
    assert metrics["per_class"]["HIT"]["f1"] == 1.0
    assert metrics["per_class"]["BOUNCE"]["f1"] == 1.0
    assert metrics["tolerance_frames"] == 2


def test_adversarial_post_weighted_pseudo_influence_is_capped_at_half() -> None:
    logits = torch.zeros((2, 1, 3), dtype=torch.float32)
    targets = torch.tensor([[0], [1]])
    validity_mask = torch.ones((2, 3), dtype=torch.bool)

    _, stats = weighted_masked_cross_entropy(
        logits,
        targets,
        validity_mask,
        class_weights=torch.tensor([1.0, 5.0, 5.0]),
        sample_weights=torch.tensor([1.0, 1.0]),
        is_pseudo=torch.tensor([False, True]),
        pseudo_weight_cap=1.0,
    )

    raw_influence = stats["raw_pseudo_loss"] / (
        stats["raw_human_loss"] + stats["raw_pseudo_loss"]
    )
    assert raw_influence == pytest.approx(5 / 6)
    assert stats["effective_pseudo_loss"] <= stats["raw_human_loss"] + 1e-7
    assert stats["effective_pseudo_loss_fraction"] <= 0.5 + 1e-7
    assert stats["capped"] is True


def test_schema_v2_unknown_frames_contribute_zero_loss_and_gradient() -> None:
    row = _row(
        source_video="owner_train_unknown",
        split="train",
        start=0,
        event_class="HIT",
    )
    row["unknown_frame_mask"] = [False, True, False]
    manifest = _manifest([row])
    manifest["schema_version"] = 2
    windows = _training_windows(
        manifest,
        role="owner",
        window_frames=3,
        stride_frames=3,
    )
    batch = next(iter(torch.utils.data.DataLoader(
        WeightedEventWindowDataset(windows, image_size=32),
        batch_size=1,
    )))
    assert batch["frame_loss_mask"].tolist() == [[True, False, True]]

    logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [8.0, -4.0, 1.0], [0.1, 0.2, 0.3]]],
        requires_grad=True,
    )
    class_weights = torch.tensor(DEFAULT_CLASS_WEIGHTS)
    loss, stats = weighted_masked_cross_entropy(
        logits,
        batch["targets"],
        batch["validity_mask"],
        frame_loss_mask=batch["frame_loss_mask"],
        class_weights=class_weights,
        sample_weights=batch["sample_weight"],
        is_pseudo=batch["is_pseudo"],
        pseudo_weight_cap=1.0,
    )
    reference_loss, reference_stats = weighted_masked_cross_entropy(
        logits.detach()[:, [0, 2]],
        batch["targets"][:, [0, 2]],
        batch["validity_mask"],
        class_weights=class_weights,
        sample_weights=batch["sample_weight"],
        is_pseudo=batch["is_pseudo"],
        pseudo_weight_cap=1.0,
    )

    assert torch.equal(loss.detach(), reference_loss)
    assert stats == reference_stats
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 1]) == 0


def test_all_valid_frame_mask_preserves_loss_and_pseudo_cap_byte_exactly() -> None:
    logits = torch.tensor(
        [
            [[0.2, -0.1, 0.7], [1.1, 0.3, -0.5]],
            [[-0.2, 0.9, 0.1], [0.4, -0.7, 1.2]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 2], [1, 2]])
    validity_mask = torch.ones((2, 3), dtype=torch.bool)
    kwargs = {
        "class_weights": torch.tensor(DEFAULT_CLASS_WEIGHTS),
        "sample_weights": torch.tensor([1.0, 0.5]),
        "is_pseudo": torch.tensor([False, True]),
        "pseudo_weight_cap": 0.01,
    }

    mask_free_loss, mask_free_stats = weighted_masked_cross_entropy(
        logits, targets, validity_mask, **kwargs,
    )
    all_valid_loss, all_valid_stats = weighted_masked_cross_entropy(
        logits,
        targets,
        validity_mask,
        frame_loss_mask=torch.ones_like(targets, dtype=torch.bool),
        **kwargs,
    )

    assert torch.equal(mask_free_loss, all_valid_loss)
    assert mask_free_stats == all_valid_stats
    assert mask_free_stats["capped"] is True


def test_checkpoint_context_mismatch_and_missing_context_fail_loudly(
    tmp_path: Path,
) -> None:
    with pytest.raises(FineTuneInputError, match="window_frames=9"):
        _assert_checkpoint_context(
            _checkpoint(tmp_path / "wrong.pt", window_frames=9),
            window_frames=64,
            image_size=32,
        )
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8)
    missing = tmp_path / "missing.pt"
    torch.save(checkpoint_payload(model, image_size=32), missing)
    with pytest.raises(FineTuneInputError, match="no explicit window_frames"):
        _assert_checkpoint_context(missing, window_frames=64, image_size=32)


def test_checkpoint_conflicting_top_level_and_config_contexts_fail_loudly(
    tmp_path: Path,
) -> None:
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8)
    conflicting = tmp_path / "conflicting.pt"
    torch.save(
        checkpoint_payload(
            model,
            image_size=32,
            window_frames=64,
            config={"image_size": 32, "window_frames": 9},
        ),
        conflicting,
    )

    with pytest.raises(FineTuneInputError, match="conflicting window_frames contexts"):
        _assert_checkpoint_context(conflicting, window_frames=64, image_size=32)


def test_current_manifest_protected_seed_overlap_hard_fails(
    tmp_path: Path,
) -> None:
    protected_video = tmp_path / "protected.mp4"
    owner = _manifest([
        {
            **_row(
                source_video="protected_train",
                split="train",
                start=0,
                event_class="HIT",
            ),
            "video_path": str(protected_video),
            "num_frames": 10,
        },
        _row(source_video="clean_val", split="val", start=3, event_class="BOUNCE"),
    ])
    owner_path = _write_json(tmp_path / "owner.json", owner)
    seed_path = _write_json(
        tmp_path / "protected_seed.json",
        {
            "labels": [{
                "source": {"video_path": str(protected_video)},
                "anchor": {"frame": 5, "pts_s": 0.5},
            }]
        },
    )
    with pytest.raises(FineTuneInputError) as caught:
        validate_manifests(
            owner_path,
            None,
            window_frames=3,
            expected_owner_train_rows=1,
            expected_owner_val_rows=1,
            seed_path=seed_path,
        )
    assert caught.value.exit_code == 22
    assert "PROTECTED_SEED_WINDOW_OVERLAP" in str(caught.value)
