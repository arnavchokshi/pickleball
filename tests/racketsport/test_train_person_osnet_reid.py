from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.racketsport.train_person_osnet_reid as train_cli
from threed.racketsport.eval_guard import EvalClipLeakError

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "racketsport" / "train_person_osnet_reid.py"


def _torchreid_probe() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        from torchreid import data, models, optim  # noqa: F401
        from torchreid.data.datasets.dataset import ImageDataset  # noqa: F401
        from torchreid.engine import ImageSoftmaxEngine, ImageTripletEngine  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


_TORCHREID_OK, _TORCHREID_REASON = _torchreid_probe()


# ---------------------------------------------------------------------------
# Dependency-free static checks: these must pass in every environment, even
# one where torch/torchreid are missing or broken (see the skip-gated smoke
# test below for the environment note on why that happens on this repo's
# anaconda base interpreter).
# ---------------------------------------------------------------------------


def test_script_imports_correct_torchreid_submodule_paths() -> None:
    """AST-level check that the fixed torchreid.* import paths are used.

    torchreid 1.4.0 (the pinned/installed version) exposes ImageDataset under
    `torchreid.data.datasets.dataset` and the engines under `torchreid.engine`.
    A prior version of this script imported the nonexistent `torchreid.reid.*`
    paths, which crashed on every real run. This check requires no heavy
    dependency: it only parses the script's source.
    """

    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    import_from_modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]

    assert "torchreid.data.datasets.dataset" in import_from_modules
    assert "torchreid.engine" in import_from_modules
    assert not any(module.startswith("torchreid.reid") for module in import_from_modules), (
        "script must not import the nonexistent torchreid.reid.* paths"
    )


def test_script_selects_sampler_by_loss_type_in_source() -> None:
    """Dependency-free guard against re-hardcoding RandomSampler for triplet loss."""

    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'train_sampler = "RandomIdentitySampler" if args.loss == "triplet" else "RandomSampler"' in source


def test_script_does_not_swallow_non_import_errors_as_missing_deps() -> None:
    """The torchreid.data/.engine submodule imports must sit outside the
    ImportError-catching try/except that reports "torch and torchreid are
    required" -- otherwise a real internal API break (e.g. a reintroduced bad
    import path) gets silently misreported as a missing dependency.
    """

    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train_osnet_reid"
    )
    try_node = next(node for node in function.body if isinstance(node, ast.Try))
    guarded_modules: set[str] = set()
    for stmt in try_node.body:
        if isinstance(stmt, ast.Import):
            guarded_modules.update(alias.name for alias in stmt.names)
        elif isinstance(stmt, ast.ImportFrom) and stmt.module:
            guarded_modules.add(stmt.module)

    assert not any(
        module.startswith("torchreid.data") or module.startswith("torchreid.engine") for module in guarded_modules
    )


def test_eval_guard_refuses_manifest_with_outdoor_train_clip() -> None:
    manifest = {
        "clip_counts": {
            "outdoor_webcam_iynbd_1500_long_high_baseline": {"train": 4, "query": 1, "gallery": 1},
        }
    }
    with pytest.raises(EvalClipLeakError, match="outdoor_webcam_iynbd_1500_long_high_baseline"):
        train_cli._assert_manifest_clips_are_not_protected(manifest)


def test_eval_guard_refuses_manifest_with_burlington_train_clip() -> None:
    manifest = {
        "clip_counts": {
            "burlington_gold_0300_low_steep_corner": {"train": 4, "query": 1, "gallery": 1},
        }
    }
    with pytest.raises(EvalClipLeakError, match="burlington_gold_0300_low_steep_corner"):
        train_cli._assert_manifest_clips_are_not_protected(manifest)


def test_eval_guard_allows_burlington_as_query_gallery_only_clip() -> None:
    manifest = {
        "clip_counts": {
            "burlington_gold_0300_low_steep_corner": {"train": 0, "query": 3, "gallery": 3},
            "roboflow_train_pool": {"train": 12, "query": 0, "gallery": 0},
        }
    }
    summary = train_cli._assert_manifest_clips_are_not_protected(manifest)
    assert summary["status"] == "internal_val_used"
    assert summary["internal_val_uses"][0]["clip_id"] == "burlington_gold_0300_low_steep_corner"


def test_eval_guard_refuses_outdoor_even_as_query_gallery_only_clip() -> None:
    manifest = {
        "clip_counts": {
            "outdoor_webcam_iynbd_1500_long_high_baseline": {"train": 0, "query": 3, "gallery": 3},
        }
    }
    with pytest.raises(EvalClipLeakError, match="outdoor_webcam_iynbd_1500_long_high_baseline"):
        train_cli._assert_manifest_clips_are_not_protected(manifest)


def test_eval_guard_passes_clean_manifest_without_clip_counts() -> None:
    summary = train_cli._assert_manifest_clips_are_not_protected({})
    assert summary == {"status": "no_clip_counts_in_manifest"}


# ---------------------------------------------------------------------------
# Full 1-epoch CPU smoke test against a tiny fabricated crop dataset. This
# exercises the real torchreid import + sampler-selection + engine.run() API
# surface end to end, so a future path/API regression is caught by CI instead
# of only by the (mockable) unit tests above. It is skipped when torch/
# torchreid are unavailable *or* unusable in this interpreter -- on this
# repo's default anaconda base python3, `import torchreid` itself currently
# raises (a pinned protobuf/tensorboard incompatibility unrelated to this
# script), so this test is expected to skip there; the AST checks above still
# enforce the import-path fix without needing the heavy dependency.
# ---------------------------------------------------------------------------


def _write_fabricated_crop_dataset(dataset_dir: Path) -> None:
    from PIL import Image

    # 2 identities x 2 crops per split. Query crops use camid=0 and gallery
    # crops use camid=1 so no query/gallery pair is excluded as same-camera
    # "junk" by torchreid's market1501-style eval, and the gallery has enough
    # total images (4) for the engine's hardcoded ranks=[1, 2, 4] CMC report.
    rows: list[dict[str, Any]] = []
    for pid in range(2):
        for i in range(2):
            rel = f"images/train/pid{pid}_train{i}.jpg"
            path = dataset_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 64), color=(10 * pid, 20 * i, 30)).save(path)
            rows.append({"split": "train", "relative_image_path": rel, "pid": pid, "camid": 0})
        for split, camid in (("query", 0), ("gallery", 1)):
            for i in range(2):
                rel = f"images/{split}/pid{pid}_{split}{i}.jpg"
                path = dataset_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 64), color=(10 * pid, 40 + i, 60)).save(path)
                rows.append({"split": split, "relative_image_path": rel, "pid": pid, "camid": camid})

    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_reid_crop_dataset",
        "uses_cvat_labels": True,
        "split_counts": {"train": 4, "query": 4, "gallery": 4},
        "clip_counts": {"clean_training_clip": {"train": 4, "query": 4, "gallery": 4}},
        "rows": rows,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.skipif(
    not _TORCHREID_OK,
    reason=f"torch/torchreid unavailable or unusable in this interpreter: {_TORCHREID_REASON}",
)
@pytest.mark.parametrize("loss", ["softmax", "triplet"])
def test_train_osnet_reid_one_epoch_cpu_smoke(tmp_path: Path, loss: str) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_fabricated_crop_dataset(dataset_dir)
    save_dir = tmp_path / "out"

    args = argparse.Namespace(
        dataset_dir=dataset_dir,
        save_dir=save_dir,
        model_name="osnet_x1_0",
        weights=None,
        loss=loss,
        max_epoch=1,
        batch_size=4,
        batch_size_test=2,
        num_instances=2,
        workers=0,
        height=64,
        width=32,
        lr=0.0003,
        weight_decay=0.0005,
        optim="adam",
        lr_scheduler="single_step",
        stepsize=10,
        gamma=0.1,
        print_freq=1,
        eval_freq=-1,
        test_only=False,
        cpu=True,
    )

    summary = train_cli.train_osnet_reid(args)

    assert summary["status"] == "completed"
    assert summary["train_identity_count"] == 2
    assert summary["loss"] == loss
    assert summary["eval_guard"]["status"] == "clean"
    assert (save_dir / "training_summary.json").is_file()


@pytest.mark.skipif(
    not _TORCHREID_OK,
    reason=f"torch/torchreid unavailable or unusable in this interpreter: {_TORCHREID_REASON}",
)
def test_train_osnet_reid_refuses_protected_eval_clip_before_touching_torchreid(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_fabricated_crop_dataset(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["clip_counts"] = {"outdoor_webcam_iynbd_1500_long_high_baseline": {"train": 4, "query": 2, "gallery": 2}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = argparse.Namespace(
        dataset_dir=dataset_dir,
        save_dir=tmp_path / "out",
        model_name="osnet_x1_0",
        weights=None,
        loss="softmax",
        max_epoch=1,
        batch_size=4,
        batch_size_test=2,
        num_instances=2,
        workers=0,
        height=64,
        width=32,
        lr=0.0003,
        weight_decay=0.0005,
        optim="adam",
        lr_scheduler="single_step",
        stepsize=10,
        gamma=0.1,
        print_freq=1,
        eval_freq=-1,
        test_only=False,
        cpu=True,
    )

    with pytest.raises(EvalClipLeakError, match="outdoor_webcam_iynbd_1500_long_high_baseline"):
        train_cli.train_osnet_reid(args)
