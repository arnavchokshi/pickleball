from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import scripts.racketsport.finetune_event_head as finetune_module

from scripts.racketsport.finetune_event_head import (
    DeterministicWrappedBatchSampler,
    FineTuneInputError,
    derive_audio_only_hard_negative_pool,
    run_finetune,
    validate_stage_f_owner_manifest,
    validate_stage_p_threshold_lock,
    validate_registered_rate_media_inventory,
    _stage_f_wall_has_expired,
    _enforce_stage_f_post_optimizer_wall,
)
from threed.racketsport.event_head.model import EventHead, checkpoint_payload


ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / "runs/lanes/trackD_ev2_design_20260722"
PLAN = LANE / "VM_RUN_PLAN.md"
REGISTRATION = LANE / "REGISTRATION.md"
FINETUNE = ROOT / "scripts/racketsport/finetune_event_head.py"
RATE_MEDIA_LOCK = LANE / "RATE_MEDIA_LOCK.json"
TINY_VIDEO = ROOT / "tests/racketsport/fixtures/event_head/tiny.avi"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_ev2_r1_timing_p90_is_registered_as_subsumed_not_a_gate() -> None:
    registration = REGISTRATION.read_text()
    plan = PLAN.read_text()
    assert "| owner-41 matched timing error p90 |" not in registration
    assert "'timing_p90':" not in plan
    assert "macro-F1@+/-2 greater than\nzero mathematically implies" in registration
    assert "window_frames=64" in registration
    assert "E1-A/C's `64f` values were the\nzero-match sentinel" in registration


def test_ev2_r2_pinned_pool_removes_all_30_held_out_source_rows() -> None:
    candidates, report = derive_audio_only_hard_negative_pool(
        ROOT / "runs/lanes/abc_experiment_20260721/vm_pull/abc_out/arm_b_manifest.json",
        ROOT / "runs/lanes/abc_experiment_20260721/vm_pull_v2/abc_out_v2/arm_b_manifest.json",
        invalid_manifest_sha256="9d3d31aa12bb97369d934c30ebda4ee41663ca65a0527717e1482681180022f5",
        repaired_manifest_sha256="f5c1e3d89d072c4a770ef776378596921ae2e2fa7a91395ca2315df27b53a2a7",
        expected_candidates=262,
        window_frames=64,
        excluded_source_video_ids=("st0epgnab7dr",),
        expected_raw_candidates=292,
        expected_excluded_source_rows=30,
    )
    assert report["raw_audio_only_delta_rows"] == 292
    assert report["excluded_held_out_source_rows"] == 30
    assert report["candidate_rows"] == 262
    assert all(row.source_video_id != "st0epgnab7dr" for row in candidates)


def _thousand_wrapped_batches(seed: int) -> list[list[int]]:
    sampler = DeterministicWrappedBatchSampler(61, 8, seed)
    output: list[list[int]] = []
    while len(output) < 1000:
        output.extend(iter(sampler))
    return output[:1000]


def test_ev2_r3_pinned_61_owner_sampler_is_1000_exact_seeded_batches() -> None:
    first = _thousand_wrapped_batches(20260722)
    second = _thousand_wrapped_batches(20260722)
    assert first == second
    assert len(first) == 1000
    assert {len(batch) for batch in first} == {8}
    assert set(index for batch in first[:8] for index in batch) == set(range(61))


def _registered_final_step_kwargs(tmp_path: Path) -> dict[str, object]:
    dummy = tmp_path / "must-not-be-opened"
    return {
        "owner_manifest_path": dummy,
        "pseudo_manifest_path": None,
        "init_checkpoint_model_only": dummy,
        "out": tmp_path / "out",
        "device_name": "cpu",
        "steps": 100,
        "image_size": 224,
        "window_frames": 64,
        "batch_size": 8,
        "lr": 0.001,
        "val_every": 100,
        "seed": 20260722,
        "stride_frames": 32,
        "num_workers": 4,
        "class_weights": (1.0, 5.0, 5.0),
        "pseudo_weight_cap": 1.0,
        "checkpoint_selection": "final-step",
        "owner_manifest_sha256": "0" * 64,
        "init_checkpoint_sha256": "0" * 64,
        "hard_negative_invalid_manifest_path": dummy,
        "hard_negative_repaired_manifest_path": dummy,
        "hard_negative_invalid_manifest_sha256": "0" * 64,
        "hard_negative_repaired_manifest_sha256": "0" * 64,
        "hard_negative_expected_candidates": 262,
        "hard_negative_top_k": 96,
        "hard_negative_batch_size": 4,
        "hard_negative_excluded_source_video_ids": ("st0epgnab7dr",),
        "hard_negative_loss_cap": 0.5,
        "class_weighting": "sqrt-frequency",
        "assignment_mode": "fixed",
        "assignment_max_shift_frames": 0,
        "assignment_class_cost_weight": 1.0,
        "assignment_temporal_cost_weight": 0.25,
        "label_dilation_frames": 1,
        "label_neighbor_positive_weight": 0.5,
        "offset_loss_weight": 0.2,
        "offset_smooth_l1_beta": 1.0,
        "internal_decode_threshold": 0.4,
        "stage_p_threshold_lock_path": dummy,
        "stage_p_train_manifest_path": dummy,
        "owner_media_root": dummy,
        "rate_media_inventory_path": dummy,
        "rate_media_inventory_sha256": "0" * 64,
        "owner_train_source_video_ids": (
            "73VurrTKCZ8", "Ezz6HDNHlnk", "_L0HVmAlCQI", "wBu8bC4OfUY",
        ),
        "owner_validation_source_video_ids": ("HyUqT7zFiwk", "zwCtH_i1_S4"),
        "expected_owner_train_negative_rows": 21,
        "internal_owner_negative_max_fp": 2,
        "internal_audio_only_max_fired_rows": 26,
        "internal_rate_min_per_s": 0.3,
        "internal_rate_max_per_s": 1.0,
        "expected_owner_train_media_paths": 38,
        "expected_owner_train_source_videos": 4,
        "probe_only": True,
        "expected_owner_train_rows": 61,
        "expected_owner_val_rows": 41,
        "max_wall_minutes": 180.0,
    }


def _instrument_input_boundaries(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def forbidden(name: str):
        def _raise(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"input-open boundary reached: {name}")
        return _raise

    for name in (
        "_load_finetune_manifests",
        "_assert_checkpoint_context",
        "validate_stage_p_threshold_lock",
        "derive_audio_only_hard_negative_pool",
        "validate_registered_rate_media_inventory",
    ):
        monkeypatch.setattr(finetune_module, name, forbidden(name))
    monkeypatch.setattr(Path, "read_bytes", forbidden("Path.read_bytes"))
    monkeypatch.setattr(Path, "open", forbidden("Path.open"))
    monkeypatch.setattr(torch, "load", forbidden("torch.load"))
    return calls


@pytest.mark.parametrize(
    ("argument", "divergent"),
    [
        ("class_weighting", "fixed"),
        ("label_dilation_frames", 0),
        ("assignment_temporal_cost_weight", 1.0),
        ("offset_loss_weight", 0.0),
        ("steps", 99),
        ("image_size", 223),
        ("window_frames", 63),
        ("lr", 0.002),
        ("val_every", 50),
        ("seed", 20260723),
        ("stride_frames", 31),
        ("num_workers", 3),
        ("expected_owner_train_negative_rows", 20),
        ("internal_owner_negative_max_fp", 3),
        ("internal_rate_min_per_s", 0.2),
        ("internal_rate_max_per_s", 1.1),
    ],
)
def test_ev2_r4_each_divergent_recipe_value_fails_with_zero_input_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    divergent: object,
) -> None:
    arguments = _registered_final_step_kwargs(tmp_path)
    arguments[argument] = divergent
    calls = _instrument_input_boundaries(monkeypatch)
    with pytest.raises(FineTuneInputError, match="before input read"):
        run_finetune(**arguments)  # type: ignore[arg-type]
    assert calls == []


@pytest.mark.parametrize(
    "argument",
    [
        "class_weighting",
        "label_dilation_frames",
        "assignment_temporal_cost_weight",
        "offset_loss_weight",
    ],
)
def test_ev2_r4_each_absent_recipe_value_fails_with_zero_input_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
) -> None:
    arguments = _registered_final_step_kwargs(tmp_path)
    arguments.pop(argument)
    calls = _instrument_input_boundaries(monkeypatch)
    with pytest.raises(FineTuneInputError, match="before input read"):
        run_finetune(**arguments)  # type: ignore[arg-type]
    assert calls == []


def test_ev2_r4_legacy_defaults_wholesale_fail_with_zero_input_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _registered_final_step_kwargs(tmp_path)
    for argument in (
        "class_weighting", "label_dilation_frames",
        "assignment_temporal_cost_weight", "offset_loss_weight",
    ):
        arguments.pop(argument)
    calls = _instrument_input_boundaries(monkeypatch)
    with pytest.raises(FineTuneInputError, match="before input read"):
        run_finetune(**arguments)  # type: ignore[arg-type]
    assert calls == []


def test_ev2_r4_exact_registered_arguments_reach_first_input_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _instrument_input_boundaries(monkeypatch)
    with pytest.raises(AssertionError, match="_load_finetune_manifests"):
        run_finetune(**_registered_final_step_kwargs(tmp_path))  # type: ignore[arg-type]
    assert calls == ["_load_finetune_manifests"]


def test_ev2_r4_main_recipe_lock_precedes_accepted_gate_and_zero_input_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = finetune_module.build_parser()
    parsed = parser.parse_args(["--out", str(tmp_path / "out")])
    renamed = {
        "owner_manifest_path": "owner_manifest",
        "pseudo_manifest_path": "pseudo_manifest",
        "device_name": "device",
        "hard_negative_invalid_manifest_path": "hard_negative_invalid_manifest",
        "hard_negative_repaired_manifest_path": "hard_negative_repaired_manifest",
        "hard_negative_excluded_source_video_ids": (
            "hard_negative_excluded_source_video"
        ),
        "stage_p_threshold_lock_path": "stage_p_threshold_lock",
        "stage_p_train_manifest_path": "stage_p_train_manifest",
        "rate_media_inventory_path": "rate_media_inventory",
        "owner_train_source_video_ids": "owner_train_source_video",
        "owner_validation_source_video_ids": "owner_validation_source_video",
    }
    for name, value in _registered_final_step_kwargs(tmp_path).items():
        setattr(parsed, renamed.get(name, name), value)
    parsed.gate_proof = tmp_path / "accepted-gate-proof.json"
    parsed.steps = 99

    gate_calls: list[Path] = []

    def accept_gate_proof(path: Path, **_kwargs: object) -> None:
        gate_calls.append(path)

    input_opens = _instrument_input_boundaries(monkeypatch)
    monkeypatch.setattr(parser, "parse_args", lambda: parsed)
    monkeypatch.setattr(finetune_module, "build_parser", lambda: parser)
    monkeypatch.setattr(finetune_module, "assert_gate_proof", accept_gate_proof)
    monkeypatch.setattr(sys, "argv", [str(FINETUNE)])

    with pytest.raises(SystemExit) as raised:
        finetune_module.main()

    assert raised.value.code == 20
    assert "before input read" in capsys.readouterr().err
    assert gate_calls == []
    assert input_opens == []


@pytest.mark.parametrize("omitted_flag", ["--steps", "--seed", "--num-workers"])
def test_ev2_r4_omitted_cli_exposure_value_rejects_before_input_read(
    tmp_path: Path, omitted_flag: str,
) -> None:
    dummy = tmp_path / "must-not-be-opened"
    command = [
        str(ROOT / ".venv/bin/python"),
        "scripts/racketsport/finetune_event_head.py",
        "--owner-manifest", str(dummy),
        "--owner-manifest-sha256", "0" * 64,
        "--init-checkpoint-model-only", str(dummy),
        "--init-checkpoint-sha256", "0" * 64,
        "--out", str(tmp_path / "out"),
        "--steps", "100",
        "--image-size", "224",
        "--window-frames", "64",
        "--batch-size", "8",
        "--lr", "0.001",
        "--val-every", "100",
        "--seed", "20260722",
        "--stride-frames", "32",
        "--num-workers", "4",
        "--checkpoint-selection", "final-step",
        "--probe-only",
        "--class-weighting", "sqrt-frequency",
        "--assignment-mode", "fixed",
        "--assignment-max-shift-frames", "0",
        "--assignment-class-cost-weight", "1.0",
        "--assignment-temporal-cost-weight", "0.25",
        "--label-dilation-frames", "1",
        "--label-neighbor-positive-weight", "0.5",
        "--offset-loss-weight", "0.2",
        "--offset-smooth-l1-beta", "1.0",
        "--hard-negative-invalid-manifest", str(dummy),
        "--hard-negative-invalid-manifest-sha256", "0" * 64,
        "--hard-negative-repaired-manifest", str(dummy),
        "--hard-negative-repaired-manifest-sha256", "0" * 64,
        "--hard-negative-expected-candidates", "262",
        "--hard-negative-top-k", "96",
        "--hard-negative-batch-size", "4",
        "--hard-negative-excluded-source-video", "st0epgnab7dr",
        "--hard-negative-loss-cap", "0.5",
        "--internal-decode-threshold", "0.4",
        "--stage-p-threshold-lock", str(dummy),
        "--stage-p-train-manifest", str(dummy),
        "--owner-media-root", str(dummy),
        "--rate-media-inventory", str(dummy),
        "--rate-media-inventory-sha256", "0" * 64,
        "--owner-train-source-video", "73VurrTKCZ8",
        "--owner-train-source-video", "Ezz6HDNHlnk",
        "--owner-train-source-video", "_L0HVmAlCQI",
        "--owner-train-source-video", "wBu8bC4OfUY",
        "--owner-validation-source-video", "HyUqT7zFiwk",
        "--owner-validation-source-video", "zwCtH_i1_S4",
        "--expected-owner-train-negative-rows", "21",
        "--internal-owner-negative-max-fp", "2",
        "--internal-audio-only-max-fired-rows", "26",
        "--internal-rate-min-per-s", "0.3",
        "--internal-rate-max-per-s", "1.0",
        "--expected-owner-train-media-paths", "38",
        "--expected-owner-train-source-videos", "4",
        "--expected-owner-train-rows", "61",
        "--expected-owner-val-rows", "41",
        "--max-wall-minutes", "180",
    ]
    flag_index = command.index(omitted_flag)
    del command[flag_index:flag_index + 2]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 20
    assert "before input read" in completed.stderr
    assert "owner manifest is absent" not in completed.stderr


def test_ev2_f1_validation_poison_is_stripped_before_content_scan(
    tmp_path: Path,
) -> None:
    train = {
        "source": "synthetic",
        "source_video": "train-a",
        "video_path": str(TINY_VIDEO),
        "media_present": True,
        "split": "train",
        "fps": 10.0,
        "source_start_frame": 0,
        "num_frames": 3,
        "events": [{"frame": 1, "class": "HIT"}],
        "loss_validity_mask": [True, True, True],
        "license_posture": "TEST_ONLY",
    }
    owner = _write_json(tmp_path / "owner.json", {
        "schema_version": 1,
        "artifact_type": "event_head_owner_reviewed_dataset_manifest",
        "teacher_derived": False,
        "ground_truth": True,
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "config": {
            "window_frames": 3,
            "split_unit": "original_source_video_id",
            "train_source_groups": ["train-a"],
            "validation_source_groups": ["val-a"],
        },
        "protected_seed_check": {
            "status": "pass", "overlap_rows": 0, "checked_training_windows": 1,
        },
        "rows": [
            train,
            {
                "split": "val",
                "events": {"poison": "spot_check_tier_a_50 must be invisible"},
                "video_path": {"poison": "owner_spot_check_results"},
            },
        ],
    })
    sanitized, raw = validate_stage_f_owner_manifest(
        owner,
        owner_manifest_sha256=_sha(owner),
        window_frames=3,
        expected_owner_train_rows=1,
        expected_owner_val_rows=1,
    )
    assert hashlib.sha256(raw).hexdigest() == _sha(owner)
    assert sanitized["rows"] == [train]
    assert sanitized["stage_f_split_envelope"]["validation_row_fields_accessed"] == ["split"]


def _threshold_bundle(tmp_path: Path) -> tuple[Path, dict[str, object], Path, Path]:
    data_manifest = _write_json(tmp_path / "data.json", {"fixture": True})
    checkpoint = tmp_path / "stage_p.pt"
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8, offset_regression_head=True)
    payload = checkpoint_payload(
        model,
        image_size=32,
        window_frames=3,
        completed_steps=100,
        best_validation_threshold=0.4,
    )
    torch.save(payload, checkpoint)
    lock = {
        "schema_version": 1,
        "artifact_type": "event_head_stage_p_decode_threshold_lock",
        "status": "locked_from_stage_p_internal_validation",
        "owner_val_used": False,
        "data_manifest_sha256": _sha(data_manifest),
        "internal_validation_policy": "sha256_seeded_source_video_holdout",
        "internal_validation_source_videos": ["st0epgnab7dr"],
        "checkpoint_sha256": _sha(checkpoint),
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
    lock_path = _write_json(tmp_path / "lock.json", lock)
    train_path = _write_json(tmp_path / "train_manifest.json", {
        "decode_threshold_lock": str(lock_path),
        "decode_threshold_lock_sha256": _sha(lock_path),
        "best_validation_threshold": 0.4,
        "locked_decode_threshold": 0.4,
        "best_validation_step": 100,
        "best_checkpoint": str(checkpoint),
        "data_manifest": str(data_manifest),
        "data_manifest_sha256": _sha(data_manifest),
    })
    return checkpoint, payload, lock_path, train_path


def test_ev2_f2_threshold_lock_is_complete_and_tamper_fails(
    tmp_path: Path,
) -> None:
    checkpoint, payload, lock_path, train_path = _threshold_bundle(tmp_path)
    validated = validate_stage_p_threshold_lock(
        lock_path, train_path, checkpoint, payload, internal_decode_threshold=0.4
    )
    assert validated["checkpoint_step"] == 100
    lock = json.loads(lock_path.read_text())
    lock["threshold_grid"] = [0.4]
    _write_json(lock_path, lock)
    train = json.loads(train_path.read_text())
    train["decode_threshold_lock_sha256"] = _sha(lock_path)
    _write_json(train_path, train)
    with pytest.raises(FineTuneInputError, match="grid/NMS/tie-break"):
        validate_stage_p_threshold_lock(
            lock_path, train_path, checkpoint, payload, internal_decode_threshold=0.4
        )


def test_ev2_f3_inventory_builder_executes_exact_registered_rule_independent_of_owner_manifest(
    tmp_path: Path,
) -> None:
    inventory = json.loads(RATE_MEDIA_LOCK.read_text())
    media_root = tmp_path / "media"
    for index, entry in enumerate(inventory["entries"]):
        media = media_root / entry["relative_path"]
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"synthetic-media-{index}".encode())
        entry["sha256"] = _sha(media)
    inventory_path = _write_json(tmp_path / "rate_inventory.json", inventory)

    # This deliberately manifest-conditioned 23-row subset is never passed to
    # the inventory builder; changing it cannot change the locked 38+2 set.
    _write_json(tmp_path / "synthetic_owner_manifest.json", {
        "rows": [
            {"source_video": "73VurrTKCZ8", "video_path": "subset-only"}
            for _ in range(23)
        ]
    })
    result = validate_registered_rate_media_inventory(
        media_root, inventory_path, _sha(inventory_path)
    )
    assert len(result["train"]) == 38
    assert len(result["validation"]) == 2
    assert result["train_total_frames"] == 57025
    assert result["train_total_duration_s"] == 2063.1827083333333
    assert {
        source: sum(row["source_video_id"] == source for row in result["train"])
        for source in ("73VurrTKCZ8", "Ezz6HDNHlnk", "_L0HVmAlCQI", "wBu8bC4OfUY")
    } == {"73VurrTKCZ8": 8, "Ezz6HDNHlnk": 8, "_L0HVmAlCQI": 19, "wBu8bC4OfUY": 3}

    adversarial = json.loads(inventory_path.read_text())
    adversarial["train_per_source_counts"] = {
        "73VurrTKCZ8": 35, "Ezz6HDNHlnk": 1, "_L0HVmAlCQI": 1, "wBu8bC4OfUY": 1,
    }
    adversarial_path = _write_json(tmp_path / "adversarial_35_1_1_1.json", adversarial)
    with pytest.raises(FineTuneInputError, match="metadata diverges"):
        validate_registered_rate_media_inventory(
            media_root, adversarial_path, _sha(adversarial_path)
        )


def test_ev2_f4_post_optimizer_wall_boundary_executes_typed_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finetune_module.time, "monotonic", lambda: 59.999)
    assert _stage_f_wall_has_expired(0.0, 1.0) is False
    _enforce_stage_f_post_optimizer_wall(0.0, 1.0)

    monkeypatch.setattr(finetune_module.time, "monotonic", lambda: 60.0)
    assert _stage_f_wall_has_expired(0.0, 1.0) is True
    with pytest.raises(
        FineTuneInputError,
        match="STAGE_F_OPTIMIZER_WALL_EXPIRED.*optimizer.step.*guards are forbidden",
    ) as raised:
        _enforce_stage_f_post_optimizer_wall(0.0, 1.0)
    assert raised.value.exit_code == 31


def test_ev2_f5_commit_preflight_executes_against_commit_blobs_not_worktree(
    tmp_path: Path,
) -> None:
    plan = PLAN.read_text()
    block = _between(
        plan,
        "# BEGIN EV2_F5_FROZEN_COMMIT_CODE_PREFLIGHT\n",
        "# END EV2_F5_FROZEN_COMMIT_CODE_PREFLIGHT",
    )
    function_source = block.split("# Prove every reviewed", 1)[0]
    repository = tmp_path / "repository"
    repository.mkdir()

    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args, cwd=repository, text=True, capture_output=True, check=check
        )

    run("git", "init", "-q")
    run("git", "config", "user.email", "ev2@example.invalid")
    run("git", "config", "user.name", "EV2 Test")
    code = repository / "runtime.py"
    code.write_text("reviewed = 1\n")
    digest = _sha(code)
    sums = repository / "CODE_SHA256SUMS"
    sums.write_text(f"{digest}  runtime.py\n")
    run("git", "add", "runtime.py", "CODE_SHA256SUMS")
    run("git", "commit", "-qm", "reviewed")
    reviewed_commit = run("git", "rev-parse", "HEAD").stdout.strip()

    # A divergent working tree cannot influence the proof; it still reads the
    # reviewed commit's digest list and reviewed runtime blob.
    code.write_text("unreviewed_worktree = 2\n")
    accepted = subprocess.run(
        ["bash", "-c", function_source + "\nverify_frozen_commit_code_bytes \"$1\" CODE_SHA256SUMS", "_", reviewed_commit],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr

    # Commit changed runtime bytes while retaining the old sums: the same
    # executable boundary must reject it.
    run("git", "add", "runtime.py")
    run("git", "commit", "-qm", "tampered")
    tampered_commit = run("git", "rev-parse", "HEAD").stdout.strip()
    rejected = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + function_source + "\nverify_frozen_commit_code_bytes \"$1\" CODE_SHA256SUMS", "_", tampered_commit],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    for artifact in (
        "CROSS_TRACK_ASSUMPTIONS.md", "REPAIR_BRIEF_R1.md", "REPAIR_BRIEF_R2.md",
        "report_repair1.json", "report_repair2.json", "RATE_MEDIA_LOCK.json",
    ):
        assert artifact in plan


def test_ev2_f6_setup_failure_executes_delete_and_confirm_without_captured_id(
    tmp_path: Path,
) -> None:
    plan = PLAN.read_text()
    block = _between(
        plan,
        "# BEGIN EV2_F6_CREATE_AND_SETUP_TEARDOWN\n",
        "# END EV2_F6_CREATE_AND_SETUP_TEARDOWN",
    )
    fake_state = tmp_path / "provider"
    fake_state.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text("""#!/usr/bin/env bash
set -euo pipefail
kind="$2"
action="$3"
if test "$action" = describe; then
  marker="$FAKE_PROVIDER_STATE/${kind%?}"
  test -f "$marker" || exit 1
  if printf '%s\n' "$*" | grep -q 'value(id)'; then cat "$marker"; fi
  exit 0
fi
if test "$action" = delete; then
  if test "$kind" = instances; then
    rm -f "$FAKE_PROVIDER_STATE/instance" "$FAKE_PROVIDER_STATE/disk"
  else
    rm -f "$FAKE_PROVIDER_STATE/disk"
  fi
  exit 0
fi
if test "$kind" = scp; then exit 1; fi
exit 0
""")
    gcloud.chmod(0o755)
    state_dir = tmp_path / "controller"
    script = (
        "set -euo pipefail\n"
        "INSTANCE=pickleball-gpu-ev2\nZONE=us-central1-a\nPROJECT=test\n"
        + block
        + "\nINSTANCE_CREATE_ATTEMPTED=1\n"
        + "printf 'provider-id-1\\n' > \"$FAKE_PROVIDER_STATE/instance\"\n"
        + "printf 'disk\\n' > \"$FAKE_PROVIDER_STATE/disk\"\n"
        + "controller_delete_on_failure\n"
        + "test ! -e \"$FAKE_PROVIDER_STATE/instance\"\n"
        + "test ! -e \"$FAKE_PROVIDER_STATE/disk\"\n"
        + "test -s \"$TEARDOWN_CONFIRMED_FILE\"\n"
        + "CONTROLLER_DELETE_ON_EXIT=0\n"
    )
    environment = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "FAKE_PROVIDER_STATE": str(fake_state),
        "EV2_STATE_DIR": str(state_dir),
    }
    completed = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, env=environment,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (state_dir / "instance_id").exists()
    assert (state_dir / "teardown_confirmed").is_file()

    stale = subprocess.run(
        ["bash", "-c", "INSTANCE=x; ZONE=x; PROJECT=x\n" + block],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=False,
    )
    assert stale.returncode == 64
    assert "stale E-v2 controller state" in stale.stderr


def test_ev2_f6_post_create_finalizer_executes_identity_bound_delete_and_confirm(
    tmp_path: Path,
) -> None:
    block = _between(
        PLAN.read_text(),
        "# BEGIN EV2_F6_TOLERANT_FINALIZER\n",
        "# END EV2_F6_TOLERANT_FINALIZER",
    )
    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "instance").write_text("provider-id-1\n")
    (provider / "disk").write_text("disk\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text("""#!/usr/bin/env bash
set -euo pipefail
kind="$2"; action="$3"
if test "$action" = describe; then
  marker="$FAKE_PROVIDER_STATE/${kind%?}"
  test -f "$marker" || exit 1
  if printf '%s\n' "$*" | grep -q 'value(id)'; then cat "$marker"; fi
  exit 0
fi
if test "$action" = delete; then
  if test "$kind" = instances; then rm -f "$FAKE_PROVIDER_STATE/instance" "$FAKE_PROVIDER_STATE/disk"; else rm -f "$FAKE_PROVIDER_STATE/disk"; fi
  exit 0
fi
exit 0
""")
    gcloud.chmod(0o755)
    controller = tmp_path / "controller"
    controller.mkdir()
    (controller / "instance_id").write_text("provider-id-1\n")
    script = (
        "set -euo pipefail\n"
        "INSTANCE=pickleball-gpu-ev2\nZONE=us-central1-a\nPROJECT=test\n"
        f"EXPECTED_INSTANCE_ID=provider-id-1\nTEARDOWN_DONE=0\n"
        f"TEARDOWN_CONFIRMED_FILE={controller / 'teardown_confirmed'}\n"
        + block
        + "\ncleanup_disposable_vm\n"
        + "test \"$TEARDOWN_DONE\" = 1\n"
        + "test -s \"$TEARDOWN_CONFIRMED_FILE\"\n"
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "FAKE_PROVIDER_STATE": str(provider),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (provider / "instance").exists()
    assert not (provider / "disk").exists()


def test_ev2_f7_price_proof_is_mechanical_fresh_complete_and_authoritative() -> None:
    plan = PLAN.read_text()
    for sku in (
        "3178-715E-CFB6", "65A3-16DB-D57A", "39D4-516A-0317",
        "6AE1-525F-8B80", "4AF8-7C1F-39C4",
    ):
        assert sku in plan
    assert "Cloud Billing Catalog API v1 latest public USD pricing" in plan
    assert "0 <= (now - retrieved).total_seconds() <= 900" in plan
    assert "age_s <= 13 * 3600" not in plan
    assert "IFS= read -r ALL_IN_INSTANCE_RATE_USD_PER_HOUR" not in plan
    assert "rate = sum(component_hourly.values())" in plan


def test_ev2_f8_every_workload_timeout_has_kill_escalation() -> None:
    plan = PLAN.read_text()
    timeout_lines = [line for line in plan.splitlines() if "timeout --signal=TERM" in line]
    assert len(timeout_lines) == 6
    assert all("--kill-after=30s" in line for line in timeout_lines)


def test_ev2_f9_exact_plan_heredoc_atomically_emits_pass_only_handoff(
    tmp_path: Path,
) -> None:
    plan = PLAN.read_text()
    section = plan.split("## 9. PASS-only", 1)[1]
    python_source = _between(
        section,
        ".venv/bin/python - \"$PULLED\" <<'PY'\n",
        "\nPY",
    )
    script = tmp_path / "handoff.py"
    script.write_text(python_source)
    pulled = tmp_path / "pass"
    (pulled / "stage_f_full").mkdir(parents=True)
    (pulled / "stage_p_full").mkdir(parents=True)
    _write_json(pulled / "VERDICT.json", {
        "verdict": "EVENT_EV2_RECIPE_REPAIR_PASS",
        "owner41_score_calls": 1,
        "protected50_score_calls": 0,
    })
    (pulled / "stage_f_full/best_event_head_finetuned.pt").write_bytes(b"checkpoint")
    _write_json(pulled / "stage_p_full/stage_p_decode_threshold_lock.json", {"threshold": 0.4})
    _write_json(pulled / "ev2_owner41_once.json", {
        "macro_f1_at_2": 0.2,
        "negative_false_positives": 1,
        "full_video_events_per_second": 0.5,
        "timing_error_p90_frames": 2.0,
    })
    production = ROOT / "configs/racketsport/best_stack.json"
    production_before = _sha(production)
    emitted = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(script), str(pulled)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    handoff = json.loads((pulled / "BEST_STACK_PENDING.json").read_text())
    assert handoff["status"] == "PENDING"
    assert handoff["production_files_mutated"] == []
    assert not (pulled / ".BEST_STACK_PENDING.json.tmp").exists()
    assert _sha(production) == production_before

    failed = tmp_path / "fail"
    failed.mkdir()
    _write_json(failed / "VERDICT.json", {
        "verdict": "EVENT_EV2_RECIPE_REPAIR_NO_LIFT",
        "owner41_score_calls": 1,
        "protected50_score_calls": 0,
    })
    rejected = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(script), str(failed)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert rejected.returncode != 0
    assert not (failed / "BEST_STACK_PENDING.json").exists()
    assert not (failed / ".BEST_STACK_PENDING.json.tmp").exists()
