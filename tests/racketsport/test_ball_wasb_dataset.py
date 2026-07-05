from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from threed.racketsport.ball_wasb_dataset import (
    BLURBALL_CSV_FILENAME,
    MANIFEST_JSON,
    WASB_CSV_DIRNAME,
    WASB_FRAME_DIRNAME,
    build_ball_wasb_dataset,
)
from threed.racketsport.eval_guard import EvalClipLeakError


def test_build_ball_wasb_dataset_preserves_tracknet_splits_pngs_and_hidden_rows(tmp_path: Path) -> None:
    tracknet_root = _write_tracknet_layout(
        tmp_path / "tracknet",
        [
            _SampleSpec(
                split="train",
                match="match1",
                rally_id="1_01_00",
                clip="owner_train_clean",
                rows=[(0, 1, 12.5, 20.25), (1, 0, 0.0, 0.0), (2, 1, 14.0, 21.0), (3, 0, 0.0, 0.0)],
            ),
            _SampleSpec(
                split="val",
                match="match2",
                rally_id="2_01_00",
                clip="owner_val_clean",
                rows=[(0, 0, 0.0, 0.0), (1, 1, 30.0, 40.0), (2, 1, 31.0, 41.0)],
            ),
        ],
    )

    manifest = build_ball_wasb_dataset(tracknet_root=tracknet_root, out_dir=tmp_path / "wasb")

    assert manifest["artifact_type"] == "racketsport_ball_wasb_dataset"
    assert manifest["image_extension"] == ".png"
    assert manifest["frame_copy_policy"] == "preserve_png_no_jpeg_conversion"
    assert manifest["label_counts"] == {
        "sample_count": 2,
        "frame_count": 7,
        "visible_frame_count": 4,
        "hidden_frame_count": 3,
    }
    assert manifest["dataset_config"]["train"]["matches"] == ["train_match1"]
    assert manifest["dataset_config"]["val"]["matches"] == ["val_match2"]
    assert manifest["dataset_config"]["test"]["matches"] == []

    train_csv = tmp_path / "wasb" / WASB_CSV_DIRNAME / "train_match1" / "1_01_00.txt"
    assert train_csv.read_text(encoding="utf-8").splitlines() == [
        "12.500 20.250",
        "0.000 0.000",
        "14.000 21.000",
        "0.000 0.000",
    ]
    assert (tmp_path / "wasb" / WASB_FRAME_DIRNAME / "train_match1" / "1_01_00" / "0.png").is_file()
    assert not list((tmp_path / "wasb" / WASB_FRAME_DIRNAME).rglob("*.jpg"))
    blurball_label = tmp_path / "wasb" / "train_match1" / "1_01_00" / BLURBALL_CSV_FILENAME
    assert blurball_label.read_text(encoding="utf-8").splitlines() == [
        "file name,visibility,x-coordinate,y-coordinate",
        "00000.png,1,12.500,20.250",
        "00001.png,0,0.000,0.000",
        "00002.png,1,14.000,21.000",
        "00003.png,0,0.000,0.000",
    ]
    assert (tmp_path / "wasb" / "train_match1" / "1_01_00" / "00000.png").is_file()
    assert manifest["blurball_layout"]["csv_filename"] == BLURBALL_CSV_FILENAME
    assert manifest["blurball_layout"]["matches_by_split"]["train"] == ["train_match1"]
    assert (tmp_path / "wasb" / MANIFEST_JSON).is_file()
    assert "visibility-0 TrackNet rows are written as 0.000 0.000" in (
        tmp_path / "wasb" / "ball_wasb_dataset_manifest.md"
    ).read_text(encoding="utf-8")


def test_build_ball_wasb_dataset_rejects_protected_clip_before_writing(tmp_path: Path) -> None:
    tracknet_root = _write_tracknet_layout(
        tmp_path / "tracknet",
        [
            _SampleSpec(
                split="train",
                match="match1",
                rally_id="1_01_00",
                clip="outdoor_webcam_iynbd_1500_long_high_baseline",
                rows=[(0, 1, 12.0, 20.0), (1, 0, 0.0, 0.0), (2, 1, 13.0, 21.0)],
            ),
        ],
    )

    out_dir = tmp_path / "wasb"
    with pytest.raises(EvalClipLeakError, match="strict held-out eval clip"):
        build_ball_wasb_dataset(tracknet_root=tracknet_root, out_dir=out_dir)

    assert not out_dir.exists()


def test_pickleball_vendored_dataset_reads_hidden_rows_and_one_loader_batch(tmp_path: Path) -> None:
    tracknet_root = _write_tracknet_layout(
        tmp_path / "tracknet",
        [
            _SampleSpec(
                split="train",
                match="match1",
                rally_id="1_01_00",
                clip="owner_train_clean",
                rows=[(0, 1, 12.0, 20.0), (1, 0, 0.0, 0.0), (2, 1, 13.0, 21.0)],
            ),
        ],
        image_size=(64, 48),
    )
    out_dir = tmp_path / "wasb"
    build_ball_wasb_dataset(tracknet_root=tracknet_root, out_dir=out_dir)

    wasb_src = Path("third_party/WASB-SBDT/src").resolve()
    pickleball_module = _load_module(
        "wasb_pickleball_dataset_for_test",
        wasb_src / "datasets" / "pickleball.py",
        prepend_sys_path=wasb_src,
    )
    cfg = _wasb_loader_cfg(out_dir)
    dataset = pickleball_module.Pickleball(cfg)

    assert len(dataset.train) == 1
    assert [anno["center"].is_visible for anno in dataset.train[0]["annos"]] == [True, False, True]

    loader_batch = _one_wasb_dataset_batch(dataset.train)
    assert len(loader_batch) == 1
    assert [Path(path).name for path in loader_batch[0]["frames"]] == ["0.png", "1.png", "2.png"]
    assert [anno["center"].is_visible for anno in loader_batch[0]["annos"]] == [True, False, True]


def test_wasb_factory_build_dataloader_reads_pickleball_batch(tmp_path: Path) -> None:
    tracknet_root = _write_tracknet_layout(
        tmp_path / "tracknet",
        [
            _SampleSpec(
                split="train",
                match="match1",
                rally_id="1_01_00",
                clip="owner_train_clean",
                rows=[(0, 1, 12.0, 20.0), (1, 0, 0.0, 0.0), (2, 1, 13.0, 21.0)],
            ),
        ],
        image_size=(64, 48),
    )
    out_dir = tmp_path / "wasb"
    build_ball_wasb_dataset(tracknet_root=tracknet_root, out_dir=out_dir)

    batch = _factory_loader_batch("third_party/WASB-SBDT/src", _wasb_loader_cfg(out_dir))

    assert tuple(batch[0].shape) == (1, 9, 32, 32)
    assert tuple(batch[1][0].shape) == (1, 3, 32, 32)


def test_blurball_factory_build_dataloader_reads_recovered_label_layout(tmp_path: Path) -> None:
    tracknet_root = _write_tracknet_layout(
        tmp_path / "tracknet",
        [
            _SampleSpec(
                split="train",
                match="match1",
                rally_id="1_01_00",
                clip="owner_train_clean",
                rows=[(0, 1, 12.0, 20.0), (1, 0, 0.0, 0.0), (2, 1, 13.0, 21.0)],
            ),
        ],
        image_size=(64, 48),
    )
    out_dir = tmp_path / "wasb"
    build_ball_wasb_dataset(tracknet_root=tracknet_root, out_dir=out_dir)

    label_rows = (out_dir / "train_match1" / "1_01_00" / BLURBALL_CSV_FILENAME).read_text(
        encoding="utf-8"
    ).splitlines()
    assert label_rows == [
        "file name,visibility,x-coordinate,y-coordinate",
        "00000.png,1,12.000,20.000",
        "00001.png,0,0.000,0.000",
        "00002.png,1,13.000,21.000",
    ]

    batch = _factory_loader_batch("third_party/blurball/src", _blurball_loader_cfg(out_dir))

    assert tuple(batch[0].shape) == (1, 9, 32, 32)
    assert tuple(batch[1][0].shape) == (1, 3, 32, 32)


def test_additive_vendored_pickleball_files_are_marked_not_upstream() -> None:
    paths = [
        Path("third_party/WASB-SBDT/src/configs/dataset/pickleball.yaml"),
        Path("third_party/WASB-SBDT/src/datasets/pickleball.py"),
        Path("third_party/blurball/src/configs/dataset/pickleball.yaml"),
        Path("third_party/blurball/src/datasets/pickleball.py"),
    ]
    for path in paths:
        assert path.is_file()
        assert path.read_text(encoding="utf-8").splitlines()[0] == "# pickleball addition, not upstream"


def test_build_ball_wasb_dataset_cli_smoke_and_scaffold_reference(tmp_path: Path) -> None:
    tracknet_root = _write_tracknet_layout(
        tmp_path / "tracknet",
        [
            _SampleSpec(
                split="train",
                match="match1",
                rally_id="1_01_00",
                clip="owner_train_clean",
                rows=[(0, 1, 12.0, 20.0), (1, 0, 0.0, 0.0), (2, 1, 13.0, 21.0)],
            ),
        ],
    )
    cli = "scripts/racketsport/build_ball_wasb_dataset.py"
    completed = subprocess.run(
        [
            sys.executable,
            cli,
            "--tracknet-root",
            str(tracknet_root),
            "--out-dir",
            str(tmp_path / "wasb"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["manifest_json"].endswith(MANIFEST_JSON)
    assert payload["label_counts"]["hidden_frame_count"] == 1


class _SampleSpec:
    def __init__(
        self,
        *,
        split: str,
        match: str,
        rally_id: str,
        clip: str,
        rows: list[tuple[int, int, float, float]],
    ) -> None:
        self.split = split
        self.match = match
        self.rally_id = rally_id
        self.clip = clip
        self.rows = rows


def _write_tracknet_layout(root: Path, samples: list[_SampleSpec], *, image_size: tuple[int, int] = (32, 32)) -> Path:
    manifest_splits: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        match_dir = root / sample.split / sample.match
        csv_dir = match_dir / ("corrected_csv" if sample.split == "test" else "csv")
        frame_dir = match_dir / "frame" / sample.rally_id
        csv_path = csv_dir / f"{sample.rally_id}_ball.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame_dir.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Frame", "Visibility", "X", "Y"])
            writer.writerows(sample.rows)
        for frame, visibility, x, y in sample.rows:
            color = (int(frame * 40) % 255, int(visibility * 200), int((x + y) % 255))
            Image.new("RGB", image_size, color=color).save(frame_dir / f"{frame}.png")
        np.savez(frame_dir / "median.npz", median=np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8))
        manifest_splits.setdefault(sample.split, []).append(
            {
                "clip": sample.clip,
                "split": sample.split,
                "match": sample.match,
                "rally_id": sample.rally_id,
                "csv": str(csv_path),
                "frame_dir": str(frame_dir),
                "frame_count": len(sample.rows),
                "visible_label_frames": sum(1 for _, visibility, _, _ in sample.rows if visibility == 1),
                "hidden_label_frames": sum(1 for _, visibility, _, _ in sample.rows if visibility == 0),
                "source_video_path": f"/tmp/source/{sample.clip}.mp4",
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_tracknet_cvat_dataset",
        "status": "tracknet_dataset_materialized",
        "splits": manifest_splits,
        "tracknet_columns": ["Frame", "Visibility", "X", "Y"],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "ball_tracknet_cvat_dataset_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


def _wasb_loader_cfg(root_dir: Path) -> dict[str, object]:
    return {
        "dataset": {
            "name": "pickleball",
            "root_dir": str(root_dir),
            "frame_dirname": WASB_FRAME_DIRNAME,
            "csv_dirname": WASB_CSV_DIRNAME,
            "ext": ".png",
            "train": {"matches": ["train_match1"], "num_clip_ratio": 1.0},
            "test": {"matches": [], "num_clip_ratio": 1.0},
        },
        "model": {
            "name": "hrnet",
            "frames_in": 3,
            "frames_out": 3,
            "inp_width": 32,
            "inp_height": 32,
            "out_width": 32,
            "out_height": 32,
            "rgb_diff": False,
            "out_scales": [0],
        },
        "detector": {"step": 1},
        "dataloader": {
            "train": True,
            "test": False,
            "train_clip": False,
            "test_clip": False,
            "sampler": {
                "name": "random",
                "train_batch_size": 1,
                "train_shuffle_batch": False,
                "train_drop_last": True,
                "test_batch_size": 1,
                "test_shuffle_batch": False,
                "test_drop_last": True,
                "inference_video_batch_size": 1,
                "inference_video_shuffle_batch": False,
                "inference_video_drop_last": True,
            },
            "train_num_workers": 0,
            "test_num_workers": 0,
            "inference_video_num_workers": 0,
            "heatmap": {"name": "binary_fixed_size", "sigmas": [2.5], "mags": [1.0], "min_value": 0.6},
        },
        "transform": {
            "train": {
                "color_jitter": {"p": 0.0, "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0},
                "horizontal_flip": {"p": 0.0},
                "crop": {"p": 0.0, "max_rescale": 0.125},
            },
            "test": {
                "color_jitter": {"p": 0.0, "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0},
                "horizontal_flip": {"p": 0.0},
                "crop": {"p": 0.0, "max_rescale": 0.125},
            },
        },
        "runner": {"fp1_filename": None},
        "output_dir": str(root_dir),
    }


def _blurball_loader_cfg(root_dir: Path) -> dict[str, object]:
    cfg = _wasb_loader_cfg(root_dir)
    cfg["dataset"] = {
        "name": "pickleball",
        "root_dir": str(root_dir),
        "csv_filename": BLURBALL_CSV_FILENAME,
        "ext": ".png",
        "visible_flags": [1],
        "train": {"matches": ["train_match1"], "num_clip_ratio": 1.0},
        "test": {"matches": [], "num_clip_ratio": 1.0},
    }
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["name"] = "blurball"
    cfg["dataloader"] = dict(cfg["dataloader"])
    cfg["dataloader"]["heatmap"] = {
        "name": "binary_line_fixed_size",
        "sigmas": [2.5],
        "mags": [1.0],
        "min_value": 0.7,
    }
    return cfg


def _factory_loader_batch(src_dir: str, cfg: dict[str, object]):
    src_path = Path(src_dir).resolve()
    old_cwd = Path.cwd()
    sys.path.insert(0, str(src_path))
    _purge_vendor_modules()
    _install_omegaconf_stub()
    _install_pandas_stub()
    _install_skimage_stub()
    try:
        os.chdir(src_path)
        from dataloaders import build_dataloader

        train_loader, _, _, _ = build_dataloader(cfg)
        return next(iter(train_loader))
    finally:
        os.chdir(old_cwd)
        _purge_vendor_modules()
        sys.path.remove(str(src_path))


def _purge_vendor_modules() -> None:
    prefixes = (
        "datasets",
        "dataloaders",
        "utils",
    )
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            del sys.modules[name]


def _install_omegaconf_stub() -> None:
    if "omegaconf" in sys.modules:
        return
    module = types.ModuleType("omegaconf")
    module.DictConfig = dict
    sys.modules["omegaconf"] = module


def _install_pandas_stub() -> None:
    if "pandas" in sys.modules:
        return
    module = types.ModuleType("pandas")
    module.__spec__ = ModuleSpec("pandas", loader=None)

    def _read_csv(*_args, **_kwargs):
        raise RuntimeError("pandas stub is only present for unused vendored imports")

    module.read_csv = _read_csv
    sys.modules["pandas"] = module


def _install_skimage_stub() -> None:
    if "skimage.metrics" in sys.modules:
        return
    skimage = types.ModuleType("skimage")
    metrics = types.ModuleType("skimage.metrics")
    skimage.__spec__ = ModuleSpec("skimage", loader=None, is_package=True)
    metrics.__spec__ = ModuleSpec("skimage.metrics", loader=None)

    def _structural_similarity(*_args, **_kwargs):
        raise RuntimeError("skimage stub is only present for unused vendored imports")

    metrics.structural_similarity = _structural_similarity
    skimage.metrics = metrics
    sys.modules["skimage"] = skimage
    sys.modules["skimage.metrics"] = metrics


def _one_wasb_dataset_batch(train_sequences: list[dict[str, object]]):
    from torch.utils.data import DataLoader

    loader = DataLoader(train_sequences, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda batch: batch)
    return next(iter(loader))


def _load_module(name: str, path: Path, *, prepend_sys_path: Path):
    sys.path.insert(0, str(prepend_sys_path))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(prepend_sys_path))
