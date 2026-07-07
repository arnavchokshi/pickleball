from __future__ import annotations

import gc
import importlib.util
import json
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("torch")

from threed.racketsport.roboflow_corpus import file_sha256, image_dhash


CLI_PATH = "scripts/racketsport/train_ball_pretrain.py"


def test_train_ball_pretrain_cli_help_is_indexed() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--corpus-index" in completed.stdout
    assert "--core-to-aux-ratio" in completed.stdout
    assert "--zero-shot-baseline" in completed.stdout
    assert "--image-root-rewrite" in completed.stdout
    assert "Roboflow" in completed.stdout


def test_scaffold_index_covers_train_ball_pretrain_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/list_scaffold_tools.py",
            "--root",
            ".",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    by_path = {tool["command_path"]: tool for tool in payload["tools"]}

    assert by_path[CLI_PATH]["category"] == "ball"
    assert by_path[CLI_PATH]["workstream"] == "BALL"
    assert by_path[CLI_PATH]["direct_cli_reference_test"] == "tests/racketsport/test_ball_pretrain_harness.py"


def test_train_ball_pretrain_cpu_smoke_decreases_loss_and_round_trips_checkpoint(tmp_path: Path) -> None:
    corpus_index = _write_smoke_corpus(tmp_path)
    out_dir = tmp_path / "smoke_out"

    completed = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--corpus-index",
            str(corpus_index),
            "--out-dir",
            str(out_dir),
            "--mode",
            "smoke",
            "--model-family",
            "tiny_wasb",
            "--device",
            "cpu",
            "--image-size",
            "32x32",
            "--frames-in",
            "3",
            "--steps",
            "4",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.05",
            "--max-train-samples",
            "4",
            "--max-val-samples",
            "4",
            "--protected-eval-hash",
            "smoke=ffffffffffffffff",
            "--expected-protected-eval-hash-count",
            "1",
            "--seed",
            "7",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["status"] == "smoke_passed"
    assert summary["loss"]["strictly_decreased"] is True
    assert summary["checkpoint"]["round_trip_state_sha256_match"] is True
    assert Path(summary["checkpoint"]["latest_checkpoint"]).is_file()
    assert summary["internal_val"]["metrics"]["sample_count"] == 4
    assert "f1_at_20px" in summary["internal_val"]["metrics"]
    assert (out_dir / "skip_list.json").is_file()


def test_train_loader_cycle_does_not_cache_first_epoch_batches() -> None:
    harness = _load_train_harness_module()

    class Batch:
        pass

    class CountingLoader:
        def __init__(self, size: int) -> None:
            self.size = size
            self.iter_calls = 0
            self.created = 0
            self.alive: weakref.WeakSet[Batch] = weakref.WeakSet()

        def __iter__(self):
            self.iter_calls += 1
            for _ in range(self.size):
                batch = Batch()
                self.created += 1
                self.alive.add(batch)
                yield batch

    loader = CountingLoader(size=8)
    batches = harness._no_cache_cycle(loader)
    last = None
    for _ in range(19):
        last = next(batches)
        gc.collect()

    assert loader.iter_calls == 3
    assert loader.created == 19
    assert last is not None
    gc.collect()
    assert len(loader.alive) <= 2


def test_default_config_loads_three_channel_checkpoint_shape(tmp_path: Path) -> None:
    harness = _load_train_harness_module()
    torch = pytest.importorskip("torch")

    config = json.loads(Path("configs/racketsport/ball_pretrain_roboflow_wasb.json").read_text(encoding="utf-8"))
    output_channels = int(config["defaults"]["output_channels"])
    assert output_channels == 3

    model = harness.build_model(
        model_family="tiny_wasb",
        frames_in=3,
        output_channels=output_channels,
        image_size=(32, 32),
        wasb_repo=Path("third_party/WASB-SBDT"),
    )
    checkpoint = tmp_path / "wasb_three_channel_shape.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)

    reloaded = harness.build_model(
        model_family="tiny_wasb",
        frames_in=3,
        output_channels=output_channels,
        image_size=(32, 32),
        wasb_repo=Path("third_party/WASB-SBDT"),
    )
    summary = harness.load_model_weights(
        checkpoint,
        model=reloaded,
        device=torch.device("cpu"),
        strict=True,
    )

    assert summary["missing_keys"] == []
    assert summary["unexpected_keys"] == []
    assert tuple(reloaded.state_dict()["net.4.weight"].shape) == (3, 16, 1, 1)


def _write_smoke_corpus(tmp_path: Path) -> Path:
    samples = []
    for split, count in (("train", 4), ("valid", 4)):
        for index in range(count):
            image_path = tmp_path / "images" / split / f"frame_{index:06d}.jpg"
            _write_ball_image(image_path, x=8 + index * 2, y=10 + index)
            sample_id = f"core:{split}:{index}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "source_slug": "core/smoke",
                    "bucket": "core_pickleball",
                    "split": split,
                    "image_path": str(image_path),
                    "width": 32,
                    "height": 32,
                    "labels": {
                        "ball": [
                            {
                                "t": 0.0,
                                "xy": [float(8 + index * 2), float(10 + index)],
                                "conf": 1.0,
                                "visible": True,
                                "source_bbox_xywh": [float(7 + index * 2), float(9 + index), 2.0, 2.0],
                                "original_category": "ball",
                                "annotation_id": index,
                            }
                        ],
                        "court": [],
                        "other": [],
                        "paddle": [],
                        "person": [],
                    },
                    "label_kinds": ["ball"],
                    "temporal": {
                        "kind": "isolated_still",
                        "sequence_id": None,
                        "sequence_order": None,
                        "sequence_group_size": 1,
                    },
                    "hashes": {
                        "dhash": f"{image_dhash(image_path):016x}",
                        "sha256": file_sha256(image_path),
                    },
                }
            )
    corpus_index = tmp_path / "aggregated" / "corpus_index.json"
    corpus_index.parent.mkdir(parents=True)
    corpus_index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_roboflow_public_pretrain_corpus_index",
                "index_policy": "index_based_original_image_paths_no_image_copies",
                "hash": {"collision_hamming_threshold": 3},
                "sample_count": len(samples),
                "samples": samples,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return corpus_index


def _write_ball_image(path: Path, *, x: int, y: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), color=(8, 12, 18))
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            px = min(31, max(0, x + dx))
            py = min(31, max(0, y + dy))
            image.putpixel((px, py), (230, 230, 30))
    image.save(path)


def _load_train_harness_module():
    spec = importlib.util.spec_from_file_location("train_ball_pretrain_under_test", CLI_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
