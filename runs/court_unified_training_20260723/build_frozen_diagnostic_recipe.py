#!/usr/bin/env python3
"""Freeze the exact COURT diagnostic split and launch recipe without training.

This builder is intentionally confined to this run directory.  It uses the normal shared loader
to observe the exact rows the production-relevant v2 trainer will see.  The freshly materialized
external roots must pass the landed positive-owner-act gate at the pinned main revision; this
builder never uses the Python-only diagnostic bypass.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / "runs/court_unified_training_20260723"

TASK88_HOLDOUT = ROOT / "runs/court_diversity_owner_ingest_20260723/holdout"
PBVISION_EVAL = RUN / "pbvision_materialized/eval"
TASK88_TRAIN = ROOT / "runs/court_diversity_owner_ingest_20260723/train"
PRIOR_CVAT_TRAIN = RUN / "prior_cvat_corpus"
ROBOFLOW_TRAIN = RUN / "final_external_corpus/roboflow_train"
PBVISION_TRAIN = RUN / "final_external_corpus/pbvision_train"
PROTECTED_EVAL = ROOT / "eval_clips/ball"
PROTECTED_EVAL_VIEW = RUN / "protected_eval_loadable_32"

CURRENT_BASELINE = ROOT / "models/checkpoints/court_unet_v2/court_model_v2.pt"
IMAGENET_ENCODER = ROOT / "models/checkpoints/court_external/torchvision/resnet34-b627a593.pth"
EXTERNAL_ACT = RUN / "external_training_owner_act_final.json"
EXTERNAL_MANIFEST = RUN / "final_external_corpus/manifest.json"

EXPECTED = {
    "task88_holdout": 6,
    "pbvision_eval": 2,
    "task88_train": 15,
    "prior_cvat_train": 5,
    "roboflow_train": 2833,
    "pbvision_train": 6,
    "protected_eval_images": 43,
    "protected_eval_loadable_rows": 32,
}

EXPECTED_SHA256 = {
    "current_baseline": "cdf0555d49335a946e518b177d85e2ab5be02100ba46eb3e634785c84f337c22",
    "imagenet_encoder": "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f",
    "external_owner_act": "e0f8935c5d42a531d144f74f0c527fc51b0cdd7c18e6c59ed5c5faca26893f29",
    "external_manifest": "8693e56d39b776f725704ee0abcd5f32dfe55908fcbde061af583ab8cf3a977a",
}
TRAINING_REVISION = "12b555824af6804da00330d64d7b3ae6d7891172"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _relative(path: Path | str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve().relative_to(ROOT.resolve()).as_posix()


def _row_entry(row: dict[str, Any], *, component: str, split: str) -> dict[str, Any]:
    image_text = row.get("image_path")
    if not isinstance(image_text, str) or not image_text:
        raise ValueError(f"{component}: row has no materialized image: {row}")
    image_path = Path(image_text)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    keypoints = row.get("keypoints")
    if not isinstance(keypoints, dict) or not keypoints:
        raise ValueError(f"{component}: row has no keypoints")
    image_sha = _sha256(image_path)
    clip = str(row["clip"])
    frame_index = int(row["frame_index"])
    return {
        "row_id": f"{clip}::frame_{frame_index:06d}::{image_sha[:16]}",
        "split": split,
        "component": component,
        "clip": clip,
        "frame_index": frame_index,
        "image": _relative(image_path),
        "image_sha256": image_sha,
        "keypoints_seen_by_trainer_sha256": _canonical_sha256(keypoints),
        "labeled_channel_count": len(keypoints),
        "net_labeled_channel_count": sum(1 for name in keypoints if name.startswith("net_")),
        "label_status": str(row.get("label_status") or "reviewed"),
    }


def _load_component(
    root: Path,
    *,
    component: str,
    split: str,
    allow_pending_diagnostic_only: bool = False,
) -> list[dict[str, Any]]:
    from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels

    rows = load_real_court_keypoint_labels(
        root,
        allow_pending_diagnostic_only=allow_pending_diagnostic_only,
    )
    if len(rows) != EXPECTED[component]:
        raise AssertionError(f"{component}: expected {EXPECTED[component]} rows, got {len(rows)}")
    return [_row_entry(row, component=component, split=split) for row in rows]


def _protected_image_inventory() -> list[dict[str, Any]]:
    paths = sorted(PROTECTED_EVAL.glob("*/labels/court_keypoint_frames/*"))
    paths += sorted(PROTECTED_EVAL.glob("*/labels/court_keypoint_partial_frames/*"))
    paths = sorted(path for path in paths if path.is_file())
    if len(paths) != EXPECTED["protected_eval_images"]:
        raise AssertionError(
            f"protected image inventory: expected {EXPECTED['protected_eval_images']}, got {len(paths)}"
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        clip = path.parents[2].name
        frame_index = int(path.stem.rsplit("_", 1)[1])
        image_sha = _sha256(path)
        rows.append(
            {
                "row_id": f"{clip}::frame_{frame_index:06d}::{image_sha[:16]}",
                "split": "protected_eval_never_train",
                "component": "protected_eval_images",
                "clip": clip,
                "frame_index": frame_index,
                "image": _relative(path),
                "image_sha256": image_sha,
                "label_contract": (
                    "loadable_human_gate_row"
                    if path.parent.name == "court_keypoint_frames" and (path.parents[1] / "court_keypoints.json").is_file()
                    else "protected_image_not_in_current_v2_gate_loader"
                ),
            }
        )
    return rows


def _ensure_protected_eval_view() -> None:
    """Expose only the four current v2-loadable 8-row protected clips to the evaluator."""

    clips = (
        "burlington_gold_0300_low_steep_corner",
        "indoor_doubles_fwuks_0500_long_mid_baseline",
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "wolverine_mixed_0200_mid_steep_corner",
    )
    PROTECTED_EVAL_VIEW.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        link = PROTECTED_EVAL_VIEW / clip
        target = PROTECTED_EVAL / clip
        relative_target = Path(os.path.relpath(target, start=link.parent))
        if link.is_symlink():
            if Path(os.readlink(link)) != relative_target:
                raise AssertionError(f"protected view symlink drift: {link}")
        elif link.exists():
            raise FileExistsError(f"protected view entry is not a symlink: {link}")
        else:
            link.symlink_to(relative_target, target_is_directory=True)


def _assert_sha(path: Path, expected: str, *, name: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise AssertionError(f"{name} SHA-256 mismatch: expected {expected}, got {observed}")


def _shell_command(parts: list[str]) -> str:
    return (" " + "\\" + "\n  ").join(parts)


def main() -> int:
    observed_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    observed_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    observed_origin_main = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True
    ).strip()
    if observed_revision != TRAINING_REVISION:
        raise AssertionError(
            f"freeze must be built at landed training revision {TRAINING_REVISION}, got {observed_revision}"
        )
    if observed_branch != "main" or observed_origin_main != TRAINING_REVISION:
        raise AssertionError(
            f"pinned revision must be pushed main (branch={observed_branch}, origin/main={observed_origin_main})"
        )
    _assert_sha(CURRENT_BASELINE, EXPECTED_SHA256["current_baseline"], name="current baseline")
    _assert_sha(IMAGENET_ENCODER, EXPECTED_SHA256["imagenet_encoder"], name="ImageNet encoder")
    _assert_sha(EXTERNAL_ACT, EXPECTED_SHA256["external_owner_act"], name="external owner act")
    _assert_sha(EXTERNAL_MANIFEST, EXPECTED_SHA256["external_manifest"], name="external manifest")

    # Ordering is part of the v2 split contract: the trainer holds out the first N loaded rows.
    validation = (
        _load_component(TASK88_HOLDOUT, component="task88_holdout", split="validation_internal")
        + _load_component(PBVISION_EVAL, component="pbvision_eval", split="validation_internal")
    )
    train = (
        _load_component(TASK88_TRAIN, component="task88_train", split="train")
        + _load_component(PRIOR_CVAT_TRAIN, component="prior_cvat_train", split="train")
        + _load_component(
            ROBOFLOW_TRAIN,
            component="roboflow_train",
            split="train",
        )
        + _load_component(
            PBVISION_TRAIN,
            component="pbvision_train",
            split="train",
        )
    )
    _ensure_protected_eval_view()
    protected_loadable_rows = _load_component(
        PROTECTED_EVAL_VIEW,
        component="protected_eval_loadable_rows",
        split="protected_eval_never_train",
    )
    protected = _protected_image_inventory()

    if len(validation) != 8 or len(train) != 2859:
        raise AssertionError(f"unexpected split sizes: train={len(train)}, validation={len(validation)}")
    if any(row["net_labeled_channel_count"] for row in train if row["component"] in {"roboflow_train", "pbvision_train"}):
        raise AssertionError("sanitized external training rows still supervise net channels")

    train_hashes = {row["image_sha256"] for row in train}
    val_hashes = {row["image_sha256"] for row in validation}
    protected_hashes = {row["image_sha256"] for row in protected}
    if train_hashes & val_hashes:
        raise AssertionError("exact image collision between train and validation")
    if train_hashes & protected_hashes:
        raise AssertionError("protected-eval image collision in train split")
    if val_hashes & protected_hashes:
        raise AssertionError("protected-eval image collision in internal validation split")

    all_ids = [row["row_id"] for row in validation + train + protected]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("duplicate row_id across frozen memberships")

    root_order = [
        _relative(TASK88_HOLDOUT),
        _relative(PBVISION_EVAL),
        _relative(TASK88_TRAIN),
        _relative(PRIOR_CVAT_TRAIN),
        _relative(ROBOFLOW_TRAIN),
        _relative(PBVISION_TRAIN),
    ]
    split_payload = {
        "validation_internal": validation,
        "train": train,
        "protected_eval_never_train": protected,
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_frozen_diagnostic_split",
        "status": "FROZEN_READY_AT_PINNED_REVISION",
        "created_at": "2026-07-23",
        "trainer": "scripts/racketsport/train_court_model_v2.py",
        "training_revision": {
            "branch": "main",
            "commit": TRAINING_REVISION,
            "origin_main_verified_equal": True,
            "positive_owner_act_gate": "PASS_NORMAL_LOADER_NO_BYPASS",
        },
        "split_mechanism": {
            "type": "ordered_real_roots_then_first_n_rows",
            "real_root_order": root_order,
            "real_val_samples": 8,
            "expected_first_eight_row_ids": [row["row_id"] for row in validation],
            "warning": "Changing root order changes membership. Do not use --real-split-proposal for this run.",
        },
        "counts": {
            "train_rows": len(train),
            "validation_internal_rows": len(validation),
            "validation_independent_human_rows": sum(
                1 for row in validation if row["label_status"] == "reviewed"
            ),
            "validation_external_teacher_rows": sum(
                1 for row in validation if row["label_status"] == "reviewed_external_dataset"
            ),
            "protected_eval_images_never_train": len(protected),
            "protected_eval_loadable_labeled_rows": len(protected_loadable_rows),
        },
        "component_counts": EXPECTED,
        "immutable_inputs": {
            "current_baseline": {
                "path": _relative(CURRENT_BASELINE),
                "sha256": EXPECTED_SHA256["current_baseline"],
            },
            "imagenet_encoder": {
                "path": _relative(IMAGENET_ENCODER),
                "sha256": EXPECTED_SHA256["imagenet_encoder"],
            },
            "external_owner_act": {
                "path": _relative(EXTERNAL_ACT),
                "sha256": EXPECTED_SHA256["external_owner_act"],
            },
            "external_manifest": {
                "path": _relative(EXTERNAL_MANIFEST),
                "sha256": EXPECTED_SHA256["external_manifest"],
            },
        },
        "integrity": {
            "train_validation_exact_image_overlap": 0,
            "train_protected_exact_image_overlap": 0,
            "validation_protected_exact_image_overlap": 0,
            "external_train_rows_with_non_null_net_channels": 0,
            "canonical_split_payload_sha256": _canonical_sha256(split_payload),
        },
        "memberships": split_payload,
        "posture": {
            "training_kind": "diagnostic_candidate_only",
            "promotion_allowed_from_this_run": False,
            "verified": False,
            "reason": (
                "Task 88 contributes only 6 independent validation rows across 4 clips/3 source "
                "families, below the frozen diversity gate; pb.vision rows are external teacher "
                "labels, not independent human ground truth; the 43 protected image assets "
                "(32 currently loadable human-gate rows plus 11 additional protected images) "
                "remain evaluation-only and are not a fresh promotion set."
            ),
        },
    }

    manifest_path = RUN / "frozen_diagnostic_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_file_sha = _sha256(manifest_path)
    (RUN / "frozen_diagnostic_split_manifest.sha256").write_text(
        f"{manifest_file_sha}  {manifest_path.name}\n",
        encoding="utf-8",
    )

    candidate_out = "runs/court_unified_training_20260723/diagnostic_train/court_unet_v2_seed13"
    train_command = _shell_command(
        [
            ".venv/bin/python scripts/racketsport/train_court_model_v2.py",
            f"--out {candidate_out}",
            "--epochs 18",
            "--steps-per-epoch 100",
            "--batch-size 32",
            "--image-width 640",
            "--image-height 360",
            "--encoder-weights-path models/checkpoints/court_external/torchvision/resnet34-b627a593.pth",
            "--lr 0.001",
            "--weight-decay 0.0001",
            "--amp",
            "--seed 13",
            "--val-seed 999983",
            "--val-samples 16",
            "--heatmap-sigma-px 1.5",
            "--seg-loss-weight 1.0",
            "--vis-loss-weight 0.2",
            "--geometric-loss-weight 0.05",
            "--geometric-colinearity-weight 1.0",
            "--geometric-homography-weight 1.0",
            "--synthetic-workers 0",
            "--real-root runs/court_diversity_owner_ingest_20260723/holdout",
            "--real-root runs/court_unified_training_20260723/pbvision_materialized/eval",
            "--real-root runs/court_diversity_owner_ingest_20260723/train",
            "--real-root runs/court_unified_training_20260723/prior_cvat_corpus",
            "--real-root runs/court_unified_training_20260723/final_external_corpus/roboflow_train",
            "--real-root runs/court_unified_training_20260723/final_external_corpus/pbvision_train",
            "--real-weight 0.65",
            "--synthetic-weight 0.35",
            "--real-batch-size 32",
            "--real-photometric-aug",
            "--real-val-samples 8",
            "--eval-every 2",
            "--checkpoint-every-eval",
            "--keep-last-checkpoints 3",
            "--device cuda",
        ]
    )

    recipe = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_frozen_diagnostic_training_recipe",
        "status": "READY_TO_LAUNCH_AFTER_EXPLICIT_PRECONDITIONS",
        "split_manifest": {
            "path": _relative(manifest_path),
            "sha256": manifest_file_sha,
            "canonical_split_payload_sha256": manifest["integrity"]["canonical_split_payload_sha256"],
        },
        "baseline_first": {
            "checkpoint": manifest["immutable_inputs"]["current_baseline"],
            "commands": [
                (
                    ".venv/bin/python scripts/racketsport/evaluate_court_model_v2.py "
                    "--checkpoint models/checkpoints/court_unet_v2/court_model_v2.pt "
                    "--real-root runs/court_diversity_owner_ingest_20260723/holdout "
                    "--out runs/court_unified_training_20260723/diagnostic_eval/baseline_task88/court_keypoint_metrics.json "
                    "--device cuda"
                ),
                (
                    ".venv/bin/python scripts/racketsport/evaluate_court_model_v2.py "
                    "--checkpoint models/checkpoints/court_unet_v2/court_model_v2.pt "
                    "--real-root runs/court_unified_training_20260723/protected_eval_loadable_32 "
                    "--out runs/court_unified_training_20260723/diagnostic_eval/baseline_historical_protected/court_keypoint_metrics.json "
                    "--device cuda"
                ),
                (
                    ".venv/bin/python scripts/racketsport/evaluate_court_model_v2.py "
                    "--checkpoint models/checkpoints/court_unet_v2/court_model_v2.pt "
                    "--real-root runs/court_unified_training_20260723/pbvision_materialized/eval "
                    "--out runs/court_unified_training_20260723/diagnostic_eval/baseline_pbvision/court_keypoint_metrics.json "
                    "--device cuda"
                ),
            ],
        },
        "candidate": {
            "architecture": "court_unet_v2",
            "input_size": [640, 360],
            "heatmap_stride": 4,
            "initialization": "torchvision_resnet34_imagenet",
            "encoder_checkpoint": manifest["immutable_inputs"]["imagenet_encoder"],
            "epochs": 18,
            "steps_per_epoch": 100,
            "total_steps": 1800,
            "batch_size": 32,
            "seed": 13,
            "output": candidate_out,
            "launch_command": train_command,
            "rationale": (
                "Matches the prior comparable real-transfer diagnostic budget and mixture: "
                "1800 steps, 0.65 real / 0.35 synthetic, ImageNet ResNet34 initialization, "
                "640x360 court_unet_v2. synthetic-workers=0 avoids the known torch 2.5.1 "
                "DataLoader in_order incompatibility on the pinned fleet environment."
            ),
        },
        "candidate_evaluation": {
            "same_internal_holdouts": [
                (
                    f".venv/bin/python scripts/racketsport/evaluate_court_model_v2.py --checkpoint {candidate_out}/court_model_v2.pt "
                    "--real-root runs/court_diversity_owner_ingest_20260723/holdout "
                    "--out runs/court_unified_training_20260723/diagnostic_eval/candidate_task88/court_keypoint_metrics.json --device cuda"
                ),
                (
                    f".venv/bin/python scripts/racketsport/evaluate_court_model_v2.py --checkpoint {candidate_out}/court_model_v2.pt "
                    "--real-root runs/court_unified_training_20260723/pbvision_materialized/eval "
                    "--out runs/court_unified_training_20260723/diagnostic_eval/candidate_pbvision/court_keypoint_metrics.json --device cuda"
                ),
            ],
            "historical_protected_eval_after_candidate_is_frozen": (
                f".venv/bin/python scripts/racketsport/evaluate_court_model_v2.py --checkpoint {candidate_out}/court_model_v2.pt "
                "--real-root runs/court_unified_training_20260723/protected_eval_loadable_32 "
                "--out runs/court_unified_training_20260723/diagnostic_eval/candidate_historical_protected/court_keypoint_metrics.json --device cuda"
            ),
            "metric": "PCK@5px and source-pixel median/p95, with per-viewpoint reporting",
            "interpretation": "diagnostic comparison only; no promotion from these rows",
        },
        "required_preconditions": [
            f"The training host checks out exact fetchable main commit {TRAINING_REVISION} and records it before reading data.",
            "The normal loader, without allow_pending_diagnostic_only, loads all six ordered roots as exactly 2867 rows.",
            "The normal loader's first 8 rows exactly match split_mechanism.expected_first_eight_row_ids.",
            "All immutable input and split-manifest SHA-256 values match.",
                "The training host has zero overlap between the 2859 train image hashes and all 43 protected image hashes.",
        ],
        "stop_conditions": [
            "Any hash mismatch or row-count/membership drift: do not train.",
            "Any protected-eval row in trainer inputs: do not train.",
            "Any non-null external net channel: do not train.",
            "Any normal-loader failure requiring the diagnostic bypass: do not train.",
        ],
        "posture": {
            "status_after_successful_execution": "trained_not_phase_verified",
            "verified": False,
            "promotion_allowed": False,
            "reason": manifest["posture"]["reason"],
            "future_promotion_requirement": (
                "A fresh, source-disjoint independent human holdout meeting the frozen diversity "
                "gate and PCK@5px >= 0.95 per viewpoint."
            ),
        },
    }
    recipe_path = RUN / "frozen_diagnostic_training_recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RUN / "frozen_diagnostic_training_recipe.sha256").write_text(
        f"{_sha256(recipe_path)}  {recipe_path.name}\n",
        encoding="utf-8",
    )

    # Final readiness proof follows the v2 trainer's own combined-root load path.  This catches a
    # subtle but critical class of mistakes where each component is valid but root ordering causes
    # the trainer's first-N holdout rule to reserve unintended training rows.
    from scripts.racketsport.train_court_model_v2 import load_real_training_rows

    combined_rows = load_real_training_rows([ROOT / path for path in root_order])
    combined_first_eight = [
        _row_entry(row, component="combined_order_probe", split="validation_internal")["row_id"]
        for row in combined_rows[:8]
    ]
    expected_first_eight = manifest["split_mechanism"]["expected_first_eight_row_ids"]
    launch_command = recipe["candidate"]["launch_command"]
    help_probe = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/racketsport/train_court_model_v2.py"), "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    proof = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_diagnostic_training_readiness_proof",
        "status": "PASS_READY_TO_START_DIAGNOSTIC_TRAINING",
        "training_revision": TRAINING_REVISION,
        "branch": observed_branch,
        "origin_main": observed_origin_main,
        "checks": {
            "normal_loader_no_diagnostic_bypass": True,
            "combined_root_count": len(root_order),
            "combined_row_count": len(combined_rows),
            "reserved_holdout_row_count": len(combined_rows[:8]),
            "gradient_eligible_row_count": len(combined_rows[8:]),
            "first_eight_exact": combined_first_eight == expected_first_eight,
            "additional_training_rows_reserved": 0,
            "real_val_samples_flag_exactly_once": launch_command.count("--real-val-samples 8") == 1,
            "real_root_flag_count": launch_command.count("--real-root "),
            "real_split_proposal_absent": "--real-split-proposal" not in launch_command,
            "protected_training_root_absent": (
                "--real-root eval_clips/ball" not in launch_command
                and "--real-root runs/court_unified_training_20260723/protected_eval_loadable_32"
                not in launch_command
            ),
            "external_train_non_null_net_channel_count": 0,
            "train_protected_exact_image_overlap": 0,
            "trainer_help_exit_code": help_probe.returncode,
            "launch_command_continuation_prefixes_valid": all(
                not line.lstrip().startswith("+") for line in launch_command.splitlines()
            ),
            "training_started": False,
        },
        "expected_first_eight_row_ids": expected_first_eight,
        "observed_first_eight_row_ids": combined_first_eight,
        "split_manifest_sha256": manifest_file_sha,
        "training_recipe_sha256": _sha256(recipe_path),
        "posture": "diagnostic_only_trained_not_phase_verified_after_execution",
    }
    expected_proof_checks = {
        "normal_loader_no_diagnostic_bypass": True,
        "combined_root_count": 6,
        "combined_row_count": 2867,
        "reserved_holdout_row_count": 8,
        "gradient_eligible_row_count": 2859,
        "first_eight_exact": True,
        "additional_training_rows_reserved": 0,
        "real_val_samples_flag_exactly_once": True,
        "real_root_flag_count": 6,
        "real_split_proposal_absent": True,
        "protected_training_root_absent": True,
        "external_train_non_null_net_channel_count": 0,
        "train_protected_exact_image_overlap": 0,
        "trainer_help_exit_code": 0,
        "launch_command_continuation_prefixes_valid": True,
        "training_started": False,
    }
    if proof["checks"] != expected_proof_checks:
        raise AssertionError(f"readiness proof failed: {proof['checks']}")
    proof_path = RUN / "frozen_diagnostic_readiness_proof.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    proof_sha = _sha256(proof_path)
    (RUN / "frozen_diagnostic_readiness_proof.sha256").write_text(
        f"{proof_sha}  {proof_path.name}\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "manifest": _relative(manifest_path),
                "manifest_sha256": manifest_file_sha,
                "recipe": _relative(recipe_path),
                "recipe_sha256": _sha256(recipe_path),
                "readiness_proof": _relative(proof_path),
                "readiness_proof_sha256": proof_sha,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "protected_rows": len(protected),
                "training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
