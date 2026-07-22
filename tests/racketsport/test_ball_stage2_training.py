from __future__ import annotations

import json
import hashlib
import copy
import csv
import shutil
import subprocess
import sys
from itertools import islice
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CLI_PATH = "scripts/racketsport/train_ball_stage2.py"
PARITY_BASELINE_REV = "86465272f3b267a1ab5a7c3dc5be4ca824c70d43"
PARITY_BASELINE_SHA256 = "8118ff0e8fbf1d573f61e1ce09de140cb2c9e9e62bf1d57b030560a55a157f47"


def test_sparse_review_semantics_only_emit_reviewed_rows(tmp_path: Path) -> None:
    from scripts.racketsport.train_ball_stage2 import sparse_tracknet_labels_from_cvat

    reviewed = tmp_path / "reviewed_boxes.json"
    reviewed.write_text(
        json.dumps(
                _cvat_payload(
                    frame_count=5,
                    reviewed_frame_indices=[0, 2, 4],
                    ball_frames={0: (10.0, 12.0, 4.0, 6.0)},
                    ball_visibility_levels={0: "clear"},
                    frame_visibility_levels={4: "full"},
                )
            ),
        encoding="utf-8",
    )

    labels = sparse_tracknet_labels_from_cvat(reviewed)

    assert [row.frame for row in labels] == [0, 2, 4]
    assert [(row.frame, row.visibility, row.visibility_level, row.wbce_weight) for row in labels] == [
        (0, 1, "clear", 1),
        (2, 0, None, 1),
        (4, 0, "full", 3),
    ]


def test_stage2_cvat_batch_carries_wbce_weights_into_loss(tmp_path: Path) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    cv2 = pytest.importorskip("cv2")
    cvat_root = tmp_path / "cvat"
    clip_dir = cvat_root / "clip_train"
    clip_dir.mkdir(parents=True)
    (clip_dir / "reviewed_boxes.json").write_text(
        json.dumps(
            _cvat_payload(
                frame_count=3,
                reviewed_frame_indices=[0, 1, 2],
                ball_frames={0: (10.0, 12.0, 4.0, 6.0)},
                ball_visibility_levels={0: "partial"},
                frame_visibility_levels={1: "full", 2: "out_of_frame"},
            )
        ),
        encoding="utf-8",
    )
    video = tmp_path / "clip_train.mp4"
    _write_tiny_video(video, frame_count=3, cv2=cv2)

    dataset = stage2.CvatBallStage2Dataset.from_export_root(
        cvat_root,
        video_paths={"clip_train": video},
        image_size=(32, 32),
        frames_in=3,
        heatmap_radius_px=2.0,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=stage2._collate_batch)
    batch = next(iter(loader))

    assert batch["wbce_weight"].tolist() == pytest.approx([2.0, 3.0, 3.0])

    class ConstantLogit(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs):
            return self.bias.expand(inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1])

    model = ConstantLogit()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss = stage2.train_one_stage2_batch(
        model,
        batch,
        optimizer=optimizer,
        device=torch.device("cpu"),
        torch=torch,
        occluded_prob=0.0,
        occlusion_generator=None,
    )

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros((3, 1, 32, 32)),
        batch["target"].repeat(1, 1, 1, 1),
        reduction="none",
    ).flatten(1).mean(dim=1)
    expected = (expected * batch["wbce_weight"]).mean().item()
    assert loss == pytest.approx(expected, rel=1e-6)


def test_sst_loss_is_capped_after_pseudo_sample_weighting() -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    class ConstantLogit(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs):
            return self.bias.expand(inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1])

    def batch(weight: float) -> dict[str, object]:
        return {
            "input": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
            "target": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            "wbce_weight": torch.tensor([weight], dtype=torch.float32),
        }

    model = ConstantLogit()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    result = stage2.train_one_stage2_human_sst_batch(
        model,
        batch(0.10),
        batch(1.0),
        optimizer=optimizer,
        device=torch.device("cpu"),
        torch=torch,
        occluded_prob=0.0,
        occlusion_generator=None,
        sst_loss_cap=0.25,
    )

    assert result["sst_loss_post_weighting"] > result["human_loss"]
    assert result["sst_loss_applied"] == pytest.approx(0.25 * result["human_loss"], rel=1e-6)
    assert result["sst_loss_applied"] <= 0.25 * result["human_loss"] + 1e-8
    assert result["total_loss"] == pytest.approx(
        result["human_loss"] + result["sst_loss_applied"], rel=1e-6
    )


def test_zero_weight_sst_is_model_state_identical_and_nonzero_sst_cannot_mutate_bn_twice() -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    class BnHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)
            self.head = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)

        def forward(self, inputs):
            return self.head(self.bn(inputs))

    def batch(*, value: float, weight: float) -> dict[str, object]:
        return {
            "input": torch.full((8, 3, 4, 4), value, dtype=torch.float32),
            "target": torch.zeros((8, 1, 4, 4), dtype=torch.float32),
            "wbce_weight": torch.full((8,), weight, dtype=torch.float32),
        }

    torch.manual_seed(44)
    base = BnHead()
    human_only = BnHead()
    human_only.load_state_dict(base.state_dict())
    zero_sst = BnHead()
    zero_sst.load_state_dict(base.state_dict())
    human_optimizer = torch.optim.SGD(human_only.parameters(), lr=0.05)
    zero_optimizer = torch.optim.SGD(zero_sst.parameters(), lr=0.05)
    human_batch = batch(value=1.0, weight=1.0)
    stage2.train_one_stage2_batch(
        human_only,
        human_batch,
        optimizer=human_optimizer,
        device=torch.device("cpu"),
        torch=torch,
        occluded_prob=0.0,
        occlusion_generator=None,
    )
    zero_result = stage2.train_one_stage2_human_sst_batch(
        zero_sst,
        human_batch,
        batch(value=100.0, weight=0.0),
        optimizer=zero_optimizer,
        device=torch.device("cpu"),
        torch=torch,
        occluded_prob=0.0,
        occlusion_generator=None,
        sst_loss_cap=0.25,
    )
    assert zero_result["sst_loss_applied"] == 0.0
    assert stage2.state_dict_sha256(zero_sst.state_dict()) == stage2.state_dict_sha256(human_only.state_dict())

    human_buffers = BnHead()
    human_buffers.load_state_dict(base.state_dict())
    dual_buffers = BnHead()
    dual_buffers.load_state_dict(base.state_dict())
    stage2.train_one_stage2_batch(
        human_buffers,
        human_batch,
        optimizer=torch.optim.SGD(human_buffers.parameters(), lr=0.0),
        device=torch.device("cpu"),
        torch=torch,
        occluded_prob=0.0,
        occlusion_generator=None,
    )
    stage2.train_one_stage2_human_sst_batch(
        dual_buffers,
        human_batch,
        batch(value=100.0, weight=0.25),
        optimizer=torch.optim.SGD(dual_buffers.parameters(), lr=0.0),
        device=torch.device("cpu"),
        torch=torch,
        occluded_prob=0.0,
        occlusion_generator=None,
        sst_loss_cap=0.25,
    )
    assert torch.equal(dual_buffers.bn.running_mean, human_buffers.bn.running_mean)
    assert torch.equal(dual_buffers.bn.running_var, human_buffers.bn.running_var)


def test_stage2_dataset_tensor_and_label_geometry_use_wasb_official_affine(tmp_path: Path) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2
    from threed.racketsport import wasb_adapter

    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    cvat_root = tmp_path / "cvat"
    clip_dir = cvat_root / "clip_train"
    clip_dir.mkdir(parents=True)
    payload = _cvat_payload(
        frame_count=3,
        reviewed_frame_indices=[1],
        ball_frames={1: (14.0, 10.0, 4.0, 4.0)},
        ball_visibility_levels={1: "clear"},
    )
    payload["task"]["original_size"] = [64, 48]  # type: ignore[index]
    (clip_dir / "reviewed_boxes.json").write_text(json.dumps(payload), encoding="utf-8")
    video = tmp_path / "clip_train.mp4"
    _write_gradient_video(video, frame_count=3, width=64, height=48, cv2=cv2, np=np)

    dataset = stage2.CvatBallStage2Dataset.from_export_root(
        cvat_root,
        video_paths={"clip_train": video},
        frames_in=3,
        heatmap_radius_px=2.0,
    )
    item = dataset[0]
    frames = _read_video_rgb_frames(video, [0, 1, 2], cv2=cv2)
    trans_input = wasb_adapter._wasb_official_input_affine(64, 48, cv2=cv2, np=np)
    expected = wasb_adapter._preprocess_wasb_window(
        frames,
        trans_input,
        cv2=cv2,
        np=np,
        torch=torch,
        input_preprocessing="official",
    )
    expected_xy = torch.tensor(
        wasb_adapter._wasb_affine_transform_xy([16.0, 12.0], trans_input, np=np),
        dtype=torch.float32,
    )
    peak_index = int(item["target"][0].flatten().argmax().item())
    peak_xy = torch.tensor([peak_index % 512, peak_index // 512], dtype=torch.float32)

    assert torch.allclose(item["input"], expected, atol=1e-6)
    assert torch.allclose(item["target_xy_px"], expected_xy, atol=1e-6)
    assert torch.allclose(peak_xy, expected_xy.round(), atol=1.0)


def test_occlusion_augmentation_is_seeded_and_requires_wbce() -> None:
    from scripts.racketsport.train_ball_stage2 import apply_occlusion_augmentation

    batch = {
        "input": torch.ones((2, 9, 16, 16), dtype=torch.float32),
        "target_xy_px": torch.tensor([[8.0, 8.0], [3.0, 3.0]], dtype=torch.float32),
        "ball_present": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "wbce_weight": torch.tensor([2.0, 3.0], dtype=torch.float32),
    }

    a = apply_occlusion_augmentation(
        batch,
        occluded_prob=1.0,
        generator=torch.Generator().manual_seed(123),
        torch=torch,
    )
    b = apply_occlusion_augmentation(
        batch,
        occluded_prob=1.0,
        generator=torch.Generator().manual_seed(123),
        torch=torch,
    )

    assert torch.equal(a["input"], b["input"])
    assert torch.count_nonzero(a["input"] == 0).item() > 0
    assert torch.equal(a["wbce_weight"], batch["wbce_weight"])

    unweighted = dict(batch)
    unweighted.pop("wbce_weight")
    with pytest.raises(ValueError, match="visibility-weighted WBCE"):
        apply_occlusion_augmentation(
            unweighted,
            occluded_prob=1.0,
            generator=torch.Generator().manual_seed(123),
            torch=torch,
        )


def test_init_checkpoint_key_diff_aborts(tmp_path: Path) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    model = stage2.build_model(
        model_family="tiny_wasb",
        frames_in=3,
        output_channels=1,
        image_size=(16, 16),
        wasb_repo=Path("third_party/WASB-SBDT"),
    )
    state = dict(model.state_dict())
    state["extra.weight"] = torch.zeros(1)
    checkpoint = tmp_path / "mismatched.pt"
    torch.save({"model_state_dict": state, "frames_in": 3}, checkpoint)

    with pytest.raises(RuntimeError, match="unexpected_keys"):
        stage2.load_required_init_checkpoint(
            checkpoint,
            model=model,
            device=torch.device("cpu"),
            frames_in=3,
        )


def test_train_ball_stage2_cli_help_is_indexed() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--cvat-export-root" in completed.stdout
    assert "--b0-split-root" in completed.stdout
    assert "--baseline-rev" in completed.stdout
    assert "--baseline-sha256" in completed.stdout
    assert "--sst-manifest" in completed.stdout
    assert "--sst-batch-size" in completed.stdout
    assert "--sst-loss-cap" in completed.stdout
    assert "--occluded-prob" in completed.stdout
    assert "--init-checkpoint" in completed.stdout
    assert "--resume-checkpoint" in completed.stdout


def test_train_ball_stage2_resume_checkpoint_continues_loss_and_step(tmp_path: Path) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    cv2 = pytest.importorskip("cv2")
    clip_id = "source_rally_clip_train"
    cvat_root = tmp_path / "cvat"
    clip_dir = cvat_root / clip_id
    clip_dir.mkdir(parents=True)
    payload = _cvat_payload(
        frame_count=3,
        reviewed_frame_indices=[1],
        ball_frames={1: (10.0, 12.0, 4.0, 6.0)},
        ball_visibility_levels={1: "clear"},
    )
    payload["clip_id"] = clip_id
    (clip_dir / "reviewed_boxes.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    video = tmp_path / "source" / f"{clip_id}.mp4"
    _write_tiny_video(video, frame_count=3, cv2=cv2)

    continuous = _run_stage2_train(
        stage2,
        cvat_root=cvat_root,
        rally_root=tmp_path,
        out_dir=tmp_path / "continuous",
        steps=4,
        seed=77,
    )
    first_leg = _run_stage2_train(
        stage2,
        cvat_root=cvat_root,
        rally_root=tmp_path,
        out_dir=tmp_path / "first_leg",
        steps=2,
        seed=77,
    )
    resumed = _run_stage2_train(
        stage2,
        cvat_root=cvat_root,
        rally_root=tmp_path,
        out_dir=tmp_path / "resumed",
        steps=2,
        seed=77,
        resume_checkpoint=Path(str(first_leg["checkpoint"]["latest_checkpoint"])),
    )

    assert first_leg["checkpoint"]["step"] == 2
    assert resumed["checkpoint"]["step"] == 4
    assert resumed["model"]["resume_summary"]["step"] == 2
    assert resumed["loss"]["values"] == pytest.approx(continuous["loss"]["values"][2:], rel=0.0, abs=1e-8)
    assert resumed["checkpoint"]["state_sha256"] == continuous["checkpoint"]["state_sha256"]
    checkpoint_payload = torch.load(
        first_leg["checkpoint"]["latest_checkpoint"], map_location="cpu", weights_only=False
    )
    assert checkpoint_payload["dataset_provenance"]["identity_mode"] == (
        "content_sha256_plus_canonical_clip_parent_and_exact_sample_set"
    )
    assert resumed["model"]["resume_summary"]["dataset_identity_set_sha256"] == (
        checkpoint_payload["dataset_provenance"]["dataset_identity_set_sha256"]
    )


def test_train_ball_stage2_resume_refuses_swapped_dataset_identity_set(tmp_path: Path) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    cv2 = pytest.importorskip("cv2")

    def write_dataset(source: str, x: float) -> tuple[Path, Path]:
        clip_id = f"{source}_rally_0001"
        cvat_root = tmp_path / f"cvat_{source}"
        clip_dir = cvat_root / clip_id
        clip_dir.mkdir(parents=True)
        payload = _cvat_payload(
            frame_count=3,
            reviewed_frame_indices=[1],
            ball_frames={1: (x, 12.0, 4.0, 6.0)},
            ball_visibility_levels={1: "clear"},
        )
        payload["clip_id"] = clip_id
        (clip_dir / "reviewed_boxes.json").write_text(json.dumps(payload), encoding="utf-8")
        video = tmp_path / source / f"{clip_id}.mp4"
        _write_tiny_video(video, frame_count=3, cv2=cv2)
        return cvat_root, tmp_path

    first_cvat, first_rally = write_dataset("source_a", 10.0)
    swapped_cvat, swapped_rally = write_dataset("source_b", 16.0)
    first_leg = _run_stage2_train(
        stage2,
        cvat_root=first_cvat,
        rally_root=first_rally,
        out_dir=tmp_path / "first_leg_swapped",
        steps=1,
        seed=91,
    )

    with pytest.raises(RuntimeError, match="resume checkpoint dataset provenance mismatch"):
        _run_stage2_train(
            stage2,
            cvat_root=swapped_cvat,
            rally_root=swapped_rally,
            out_dir=tmp_path / "resume_swapped",
            steps=1,
            seed=91,
            resume_checkpoint=Path(str(first_leg["checkpoint"]["latest_checkpoint"])),
        )
    assert not (tmp_path / "resume_swapped" / "summary.json").exists()


def test_revision_explicit_harness_loads_pinned_git_source_and_records_exact_compute() -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    batches = []
    expected_order = []
    for step in range(7):
        ids = [f"sample-{step}-{index}" for index in range(8)]
        expected_order.append(ids)
        batches.append(
            {
                "input": torch.full((8, 3, 4, 4), step / 10.0, dtype=torch.float32),
                "target": torch.full((8, 1, 4, 4), (step % 2), dtype=torch.float32),
                "wbce_weight": torch.linspace(0.25, 1.0, 8),
                "sample_id": ids,
            }
        )

    def model_factory():
        torch.manual_seed(20260721)
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 1, kernel_size=1),
        )

    fixture_config = {
        "model_family": "tiny_cpu_fixture",
        "steps": 7,
        "batch_size": 8,
        "device": "cpu",
    }
    result = stage2.run_revision_explicit_compute_parity(
        batches,
        baseline_revision=PARITY_BASELINE_REV,
        baseline_sha256=PARITY_BASELINE_SHA256,
        model_factory=model_factory,
        optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.0),
        steps=7,
        device=torch.device("cpu"),
        torch=torch,
        comparison_config=fixture_config,
        production=False,
    )

    baseline_source = subprocess.run(
        ["git", "show", f"{PARITY_BASELINE_REV}:scripts/racketsport/train_ball_stage2.py"],
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(baseline_source).hexdigest() == PARITY_BASELINE_SHA256
    assert result["baseline_trainer"] == {
        "requested_revision": PARITY_BASELINE_REV,
        "commit": PARITY_BASELINE_REV,
        "source_sha256": PARITY_BASELINE_SHA256,
        "expected_source_sha256": PARITY_BASELINE_SHA256,
        "source_path": "scripts/racketsport/train_ball_stage2.py",
        "load_method": (
            "git show <resolved-commit>:scripts/racketsport/train_ball_stage2.py "
            "then isolated exec"
        ),
    }
    assert result["candidate_trainer"]["source_sha256"] == stage2._sha256_file(
        Path(stage2.__file__)
    )
    assert result["candidate_trainer"]["source_sha256"] != PARITY_BASELINE_SHA256
    assert result["trainer_sources_distinct"] is True
    assert result["artifact_type"] == "racketsport_ball_stage2_head_compute_parity_fixture"
    assert result["production_configuration_executed"] is False
    assert result["comparison_config"] == fixture_config
    assert result["exact_sample_order"] == {
        "baseline": expected_order,
        "candidate": expected_order,
    }
    assert result["sample_order_identical"] is True
    assert result["exact_losses"]["baseline"] == result["exact_losses"]["candidate"]
    assert result["model_state_sha256"]["baseline"] == result["model_state_sha256"]["candidate"]
    checkpoint_comparison = result["checkpoint_format_comparison"]
    assert checkpoint_comparison["full_checkpoint_bytes_compared"] is True
    assert checkpoint_comparison["status"] == "actual_payloads_materialized_and_loaded"
    assert checkpoint_comparison["checkpoint_schema"]["baseline"]["schema_version"] == 1
    assert checkpoint_comparison["checkpoint_schema"]["candidate"]["schema_version"] == 1
    assert checkpoint_comparison["checkpoint_args"]["baseline"]["steps"] == 7
    assert checkpoint_comparison["checkpoint_args"]["candidate"]["steps"] == 7
    assert checkpoint_comparison["train_dataset_summary"]["baseline"] == checkpoint_comparison[
        "train_dataset_summary"
    ]["candidate"]
    assert checkpoint_comparison["loaded_model_state_identical"] is True
    assert checkpoint_comparison["loaded_model_state_sha256"] == result["model_state_sha256"]
    assert "b0_split_root" in checkpoint_comparison["added_args_fields"]
    assert result["verdict"] == "PASS"

    with pytest.raises(ValueError, match="generic compute parity cannot claim production"):
        stage2.run_revision_explicit_compute_parity(
            batches,
            baseline_revision=PARITY_BASELINE_REV,
            baseline_sha256=PARITY_BASELINE_SHA256,
            model_factory=model_factory,
            optimizer_factory=lambda parameters: torch.optim.AdamW(
                parameters, lr=0.01, weight_decay=0.0
            ),
            steps=1,
            device=torch.device("cpu"),
            torch=torch,
            comparison_config=fixture_config,
            production=True,
        )


def test_revision_explicit_parity_rejects_hash_mismatch_and_candidate_versus_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    with pytest.raises(ValueError, match="git blob SHA-256 mismatch"):
        stage2.load_trainer_module_from_git_revision(
            PARITY_BASELINE_REV,
            expected_sha256="0" * 64,
        )

    missing_selector_args = stage2._build_parser().parse_args(
        ["--mode", "verify-head-parity", "--out-dir", str(tmp_path / "missing")]
    )
    with pytest.raises(ValueError, match="explicit --baseline-rev"):
        stage2.run_revision_explicit_production_parity(missing_selector_args)

    head_source = subprocess.run(
        ["git", "show", "HEAD:scripts/racketsport/train_ball_stage2.py"],
        check=True,
        capture_output=True,
    ).stdout
    head_sha256 = hashlib.sha256(head_source).hexdigest()
    simulated_running_file = tmp_path / "train_ball_stage2.py"
    simulated_running_file.write_bytes(head_source)
    monkeypatch.setattr(stage2, "__file__", str(simulated_running_file))
    return_code = stage2.main(
        [
            "--mode",
            "verify-head-parity",
            "--out-dir",
            str(tmp_path / "self_parity"),
            "--baseline-rev",
            "HEAD",
            "--baseline-sha256",
            head_sha256,
        ]
    )
    captured = capsys.readouterr()
    assert return_code == 2
    assert "candidate-versus-itself comparisons are invalid" in captured.err
    assert not (tmp_path / "self_parity").exists()


def test_production_parity_refuses_when_requested_cuda_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    class UnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class TorchWithoutCuda:
        cuda = UnavailableCuda()

    monkeypatch.setattr(stage2, "_torch", lambda: TorchWithoutCuda())
    args = stage2._build_parser().parse_args(
        [
            "--mode",
            "verify-head-parity",
            "--out-dir",
            str(tmp_path / "parity"),
            "--baseline-rev",
            PARITY_BASELINE_REV,
            "--baseline-sha256",
            PARITY_BASELINE_SHA256,
            "--b0-split-root",
            str(stage2.DEFAULT_B0_SPLIT_ROOT),
            "--init-checkpoint",
            "models/checkpoints/wasb/wasb_tennis_best.pth.tar",
            "--steps",
            "2372",
            "--seed",
            "20260721",
            "--occluded-prob",
            "0",
        ]
    )
    with pytest.raises(RuntimeError, match="requires requested CUDA"):
        stage2.run_revision_explicit_production_parity(args)
    assert not (tmp_path / "parity").exists()


def test_frozen_b0_split_and_full_batch_cycle_exclude_every_judge_parent() -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    rows, summary = stage2.load_and_validate_b0_split(stage2.DEFAULT_B0_SPLIT_ROOT)
    assert len(rows) == 2249
    assert summary["b0_lineage_counts"] == {
        "confirmed_prelabel": 1546,
        "corrected_prelabel": 520,
        "scratch": 183,
    }
    assert summary["b0_excluded_judge_reviewed_row_count"] == 960
    assert {row["parent_source_id"] for row in rows} == stage2.B0_TRAIN_SOURCE_IDS
    assert not ({row["parent_source_id"] for row in rows} & stage2.B0_JUDGE_PARENT_IDS)
    assert {row["training_weight"] for row in rows if row["lineage_class"] == "confirmed_prelabel"} == {0.25}

    sampler = stage2.DeterministicFullBatchSampler(len(rows), 8, seed=20260721, torch=torch)
    batches = list(islice(iter(sampler), 2372))
    assert all(len(batch) == 8 for batch in batches)
    flattened = [index for batch in batches for index in batch]
    assert set(flattened[:2249]) == set(range(2249))
    assert len(set(flattened[:2249])) == 2249
    # Batch 282 crosses the 2,249-row boundary and still has exactly eight humans.
    assert len(batches[281]) == 8
    assert all(rows[index]["parent_source_id"] not in stage2.B0_JUDGE_PARENT_IDS for index in flattened)
    repeat = list(islice(iter(stage2.DeterministicFullBatchSampler(2249, 8, seed=20260721, torch=torch)), 2372))
    assert repeat == batches


def test_sst_distinct_sampler_repairs_reviewer_reproduction_and_boundary_splices() -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    reviewer_sampler = stage2.DeterministicDistinctFullBatchSampler(
        1001,
        8,
        seed=20260722,
        torch=torch,
    )
    reviewer_batches = list(islice(iter(reviewer_sampler), 2372))
    assert reviewer_batches[1751] == [256, 432, 546, 743, 401, 497, 120, 947]
    assert reviewer_batches[1751] != [256, 432, 546, 743, 401, 497, 432, 947]
    assert all(len(batch) == len(set(batch)) == 8 for batch in reviewer_batches)

    boundary_sampler = stage2.DeterministicDistinctFullBatchSampler(
        11,
        8,
        seed=20260722,
        torch=torch,
    )
    boundary_batches = list(islice(iter(boundary_sampler), 4))
    assert all(len(batch) == len(set(batch)) == 8 for batch in boundary_batches)
    boundary_repeat = list(
        islice(
            iter(
                stage2.DeterministicDistinctFullBatchSampler(
                    11,
                    8,
                    seed=20260722,
                    torch=torch,
                )
            ),
            4,
        )
    )
    assert boundary_repeat == boundary_batches
    resumed = stage2.DeterministicDistinctFullBatchSampler(
        11,
        8,
        seed=20260722,
        torch=torch,
    )
    resumed.set_start_batch(1)
    assert next(iter(resumed)) == boundary_batches[1]

    with pytest.raises(ValueError, match="sample_count >= batch_size"):
        stage2.DeterministicDistinctFullBatchSampler(7, 8, seed=20260722, torch=torch)


def test_sst_runtime_assertion_rejects_duplicate_sample_ids() -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    reviewer_ids = [256, 432, 546, 743, 401, 497, 432, 947]
    duplicate_batch = {
        "input": torch.zeros((8, 3, 4, 4)),
        "sample_id": [f"sst-{index}" for index in reviewer_ids],
        "parent_source_id": [sorted(stage2.SST_TRAIN_SOURCE_IDS)[0]] * 8,
    }
    with pytest.raises(RuntimeError, match="duplicate sample IDs.*sst-432"):
        stage2._assert_sst_training_batch(duplicate_batch, exact_count=8)


def test_frozen_b0_media_inventory_is_exact_and_symlink_roots_are_refused(tmp_path: Path) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    rows, _ = stage2.load_and_validate_b0_split(stage2.DEFAULT_B0_SPLIT_ROOT)
    observed = stage2._validate_b0_source_media(
        rows,
        rally_root=stage2.DEFAULT_RALLY_ROOT,
    )
    assert observed == stage2.B0_SOURCE_MEDIA_SHA256
    assert len(observed) == 31

    split_alias = tmp_path / "b0_split_alias"
    split_alias.symlink_to(Path(stage2.DEFAULT_B0_SPLIT_ROOT).resolve(), target_is_directory=True)
    with pytest.raises(ValueError, match="direct canonical path|symlink path component"):
        stage2.load_and_validate_b0_split(split_alias)

    rally_alias = tmp_path / "rally_root_alias"
    rally_alias.symlink_to(Path(stage2.DEFAULT_RALLY_ROOT).resolve(), target_is_directory=True)
    with pytest.raises(ValueError, match="direct canonical path|symlink path component"):
        stage2._validate_b0_source_media(rows, rally_root=rally_alias)


def test_judge_parent_can_never_enter_human_training_batch() -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    batch = {
        "input": torch.zeros((8, 3, 4, 4)),
        "parent_source_id": ["73VurrTKCZ8"] * 7 + ["HyUqT7zFiwk"],
    }
    with pytest.raises(RuntimeError, match="judge-parent rows can never enter"):
        stage2._assert_human_training_batch(batch, exact_count=8)


def test_generic_cvat_loader_refuses_renamed_ezz_judge_row_by_media_content_sha(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    alias_clip = "allowedalias_rally_0001"
    cvat_root = tmp_path / "cvat"
    clip_dir = cvat_root / alias_clip
    clip_dir.mkdir(parents=True)
    payload = _cvat_payload(
        frame_count=335,
        reviewed_frame_indices=[334],
        ball_frames={334: (10.0, 12.0, 4.0, 6.0)},
        ball_visibility_levels={334: "clear"},
    )
    payload["clip_id"] = alias_clip
    (clip_dir / "reviewed_boxes.json").write_text(json.dumps(payload), encoding="utf-8")
    ezz_judge_media = Path(
        "data/online_harvest_20260706/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0001.mp4"
    )

    with pytest.raises(ValueError, match="resolves by content SHA to frozen B0 judge media"):
        stage2.CvatBallStage2Dataset.from_export_root(
            cvat_root,
            video_paths={alias_clip: ezz_judge_media},
            image_size=(32, 32),
            frames_in=3,
            heatmap_radius_px=2.0,
        )


def test_production_training_and_parity_refuse_image_path_rewrites_before_execution(
    tmp_path: Path,
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    training_args = stage2._build_parser().parse_args(
        [
            "--b0-split-root",
            str(stage2.DEFAULT_B0_SPLIT_ROOT),
            "--out-dir",
            str(tmp_path / "training"),
            "--image-root-rewrite",
            "/canonical=/substitute",
        ]
    )
    with pytest.raises(ValueError, match="refuses --image-root-rewrite"):
        stage2.run(training_args)

    parity_args = stage2._build_parser().parse_args(
        [
            "--mode",
            "verify-head-parity",
            "--out-dir",
            str(tmp_path / "parity"),
            "--baseline-rev",
            PARITY_BASELINE_REV,
            "--baseline-sha256",
            PARITY_BASELINE_SHA256,
            "--b0-split-root",
            str(stage2.DEFAULT_B0_SPLIT_ROOT),
            "--init-checkpoint",
            "models/checkpoints/wasb/wasb_tennis_best.pth.tar",
            "--steps",
            "2372",
            "--seed",
            "20260721",
            "--occluded-prob",
            "0",
            "--image-root-rewrite",
            "/canonical=/substitute",
        ]
    )
    with pytest.raises(ValueError, match="frozen production config"):
        stage2.run_revision_explicit_production_parity(parity_args)


def test_dual_loaders_keep_exact_eight_humans_and_add_eight_sst_without_changing_human_order() -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    class TinyDataset:
        def __init__(self, count: int, parents: list[str], prefix: str) -> None:
            self.count = count
            self.parents = parents
            self.prefix = prefix

        def __len__(self) -> int:
            return self.count

        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "input": torch.zeros((3, 4, 4)),
                "target": torch.zeros((1, 4, 4)),
                "target_xy_px": torch.zeros(2),
                "wbce_weight": torch.tensor(1.0),
                "ball_present": torch.tensor(1.0),
                "sample_id": f"{self.prefix}-{index}",
                "source_slug": self.prefix,
                "parent_source_id": self.parents[index % len(self.parents)],
                "bucket": self.prefix,
                "source_split": "train",
                "image_path": f"{self.prefix}.mp4",
                "window_sample_ids": [f"{self.prefix}-{index}"],
                "temporal_sample_kind": "video_window",
                "visibility_level": None,
            }

    human = TinyDataset(13, sorted(stage2.B0_TRAIN_SOURCE_IDS), "human")
    sst = TinyDataset(11, sorted(stage2.SST_TRAIN_SOURCE_IDS), "sst")
    human_a, _ = stage2._make_training_loader(
        human, batch_size=8, seed=20260721, num_workers=0, torch=torch, require_full_batches=True
    )
    human_b, _ = stage2._make_training_loader(
        human, batch_size=8, seed=20260721, num_workers=0, torch=torch, require_full_batches=True
    )
    sst_b, _ = stage2._make_sst_training_loader(
        sst, batch_size=8, seed=20260722, num_workers=0, torch=torch
    )
    batches_a = list(islice(iter(human_a), 25))
    batches_b = list(islice(iter(human_b), 25))
    pseudo_batches = list(islice(iter(sst_b), 25))
    assert [batch["sample_id"] for batch in batches_a] == [batch["sample_id"] for batch in batches_b]
    for human_batch, pseudo_batch in zip(batches_b, pseudo_batches):
        stage2._assert_human_training_batch(human_batch, exact_count=8)
        stage2._assert_sst_training_batch(pseudo_batch, exact_count=8)
        assert int(human_batch["input"].shape[0]) + int(pseudo_batch["input"].shape[0]) == 16


def test_failed_gate_sst_fixture_and_sst_only_invocation_are_refused(tmp_path: Path) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    failed_manifest = Path("runs/lanes/ball_b1b2_prep_20260721/schema_valid_sample_manifest.json")
    with pytest.raises(ValueError, match="gate verdict must be PASS"):
        stage2.load_and_validate_production_sst_manifest(failed_manifest)

    args = stage2._build_parser().parse_args(
        [
            "--sst-manifest",
            str(failed_manifest),
            "--out-dir",
            str(tmp_path / "must_not_train"),
            "--device",
            "cpu",
        ]
    )
    with pytest.raises(ValueError, match="requires the frozen --b0-split-root"):
        stage2.run(args)
    assert not (tmp_path / "must_not_train").exists()


def test_sst_trust_boundary_recounts_and_refuses_forged_row_authority_hashes_and_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    manifest_path, manifest = _write_production_sst_fixture(tmp_path, stage2=stage2, monkeypatch=monkeypatch)
    validated, rows = stage2.load_and_validate_production_sst_manifest(manifest_path)
    assert len(rows) == 1000
    assert validated["trainer_validation"]["accepted_windows_recounted"] == 1000
    assert validated["trainer_validation"]["accepted_sources_recounted"] == 7

    adversarial = copy.deepcopy(manifest)
    adversarial["clips"][0]["samples"][0]["ground_truth"] = True
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="authority mismatch"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    adversarial["clips"][0]["samples"][0]["dependency_hashes"]["pbvision_cv_export_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="row/clip dependency mismatch"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    adversarial["clips"][0]["samples"][0]["teacher_xy"] = [999.0, 80.0]
    adversarial["clips"][0]["samples"][0]["agreement"]["teacher_xy"] = [999.0, 80.0]
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher evidence does not match hashed cv_export|out of image bounds"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    adversarial.pop("artifact_verification")
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_verification"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    adversarial["requested_parameters"]["teacher_confidence_min"] = 0.1
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="requested parameter"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    adversarial["preregistration"]["teacher_input_authority"][
        "expected_sha256_by_source"
    ][adversarial["clips"][0]["clip_id"]]["cv_export.json"] = "f" * 64
    adversarial["preregistered_parameters"] = dict(adversarial["preregistration"])
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher-input authority SHA map mismatch"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    first_clip = adversarial["clips"][0]
    source_id = first_clip["clip_id"]
    alternate_media = tmp_path / "alternate_media" / source_id / "max.mp4"
    alternate_media.parent.mkdir(parents=True)
    shutil.copyfile(first_clip["rally_video"], alternate_media)
    first_clip["samples"][0]["frame_ref"]["video"] = str(alternate_media)
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical bound file"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    adversarial["clips"][0]["samples"][0]["frame_ref"].pop("source_video_sha256")
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="frame_ref source media SHA mismatch"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    adversarial["clips"][0]["dependencies"]["models_manifest_sha256"] = "f" * 64
    for row in adversarial["clips"][0]["samples"]:
        row["dependency_hashes"]["models_manifest_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="global dependency mismatch"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    row = adversarial["clips"][0]["samples"][0]
    row["agreement"]["wasb_xy"] = [row["teacher_xy"][0] + 3.0, row["teacher_xy"][1]]
    row["agreement"]["distance_px"] = 3.0
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="hashed WASB track"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    adversarial = copy.deepcopy(manifest)
    rows_for_clip = adversarial["clips"][0]["samples"]
    prior, current, following = rows_for_clip[0], rows_for_clip[1], rows_for_clip[2]
    current["agreement_reason"] = stage2.SST_TEMPORAL_REASON
    current["agreement"] = {
        "policy_id": stage2.SST_TEMPORAL_REASON,
        "independent_verifier": "pinned_frozen_wasb",
        "current_source_frame_index": current["frame_index"],
        "current_teacher_xy": current["teacher_xy"],
        "current_teacher_confidence": current["score"],
        "current_wasb": {"status": "absent", "present": False},
        "prior_anchor": {
            "source_frame_index": prior["frame_index"],
            "teacher_xy": prior["teacher_xy"],
            "teacher_confidence": prior["score"],
            "wasb_xy": prior["agreement"]["wasb_xy"],
            "wasb_confidence": prior["agreement"]["wasb_confidence"],
            "distance_px": prior["agreement"]["distance_px"],
        },
        "following_anchor": {
            "source_frame_index": following["frame_index"],
            "teacher_xy": following["teacher_xy"],
            "teacher_confidence": following["score"],
            "wasb_xy": following["agreement"]["wasb_xy"],
            "wasb_confidence": following["agreement"]["wasb_confidence"],
            "distance_px": following["agreement"]["distance_px"],
        },
        "interpolated_wasb_xy": [current["teacher_xy"][0] + 2.0, current["teacher_xy"][1]],
        "interpolation_residual_px": 2.0,
        "max_gap_source_frames": 2,
        "anchor_agreement_radius_px": 20.0,
        "interpolation_residual_max_px": 20.0,
        "image_width": 640,
        "image_height": 480,
        "all_points_in_bounds": True,
    }
    manifest_path.write_text(json.dumps(adversarial), encoding="utf-8")
    with pytest.raises(ValueError, match="not a teacher-only WASB gap"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wasb_metadata_path = Path(manifest["clips"][0]["dependencies"]["wasb_metadata_path"])
    wasb_metadata_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="WASB run metadata schema is not builder-bound"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)


def test_trainer_revalidates_total_temporal_gap_and_contradictory_interior_wasb() -> None:
    from scripts.racketsport import build_pbvision_ball_sst as builder
    from scripts.racketsport import train_ball_stage2 as stage2

    def fixture(following_anchor: int, *, builder_max_gap: int):
        teachers = {
            frame: builder.TeacherObservation(
                teacher_frame_index=frame,
                teacher_time_s=frame / 30.0,
                xy_px=(float(frame * 10), 80.0),
                confidence=0.95,
            )
            for frame in range(8, following_anchor + 1)
        }
        wasb = {
            frame: builder.WasbObservation(frame, teachers[frame].xy_px, 0.0, False)
            for frame in teachers
        }
        wasb[8] = builder.WasbObservation(8, teachers[8].xy_px, 0.99, True)
        wasb[following_anchor] = builder.WasbObservation(
            following_anchor, teachers[following_anchor].xy_px, 0.99, True
        )
        accepted, reason, evidence = builder.eligibility_decision(
            teachers[10],
            teacher_observations=teachers,
            wasb=wasb[10],
            width=640,
            height=360,
            source_frame_index=10,
            teacher_by_source_frame=teachers,
            wasb_observations=wasb,
            temporal_max_gap_source_frames=builder_max_gap,
        )
        assert accepted is True
        assert reason == builder.TEMPORAL_GEOMETRY_POLICY["policy_id"]
        wasb_frames = [
            {"xy": [0.0, 0.0], "conf": 0.0, "visible": False}
            for _ in range(following_anchor + 1)
        ]
        for frame, observation in wasb.items():
            wasb_frames[frame] = {
                "xy": list(observation.xy_px),
                "conf": observation.confidence,
                "visible": observation.visible,
            }
        teacher_path = {
            frame: {"xy": observation.xy_px, "confidence": observation.confidence}
            for frame, observation in teachers.items()
        }
        sample = {"sample_id": "temporal-reviewer-8-10-anchor", "score": teachers[10].confidence}
        return sample, evidence, teacher_path, wasb_frames

    sample, gap_two, teacher_path, wasb_frames = fixture(11, builder_max_gap=2)
    stage2._validate_sst_temporal_evidence(
        sample,
        gap_two,
        teacher_xy=(100.0, 80.0),
        frame_index=10,
        width=640,
        height=360,
        wasb_frames=wasb_frames,
        teacher_by_source_frame=teacher_path,
    )

    sample, gap_three, teacher_path, wasb_frames = fixture(12, builder_max_gap=3)
    gap_three["max_gap_source_frames"] = 2
    with pytest.raises(ValueError, match="exceeds frozen total teacher-only gap"):
        stage2._validate_sst_temporal_evidence(
            sample,
            gap_three,
            teacher_xy=(100.0, 80.0),
            frame_index=10,
            width=640,
            height=360,
            wasb_frames=wasb_frames,
            teacher_by_source_frame=teacher_path,
        )

    sample, gap_two, teacher_path, wasb_frames = fixture(11, builder_max_gap=2)
    wasb_frames[9] = {"xy": [400.0, 250.0], "conf": 0.99, "visible": True}
    with pytest.raises(ValueError, match="contradictory high-confidence WASB"):
        stage2._validate_sst_temporal_evidence(
            sample,
            gap_two,
            teacher_xy=(100.0, 80.0),
            frame_index=10,
            width=640,
            height=360,
            wasb_frames=wasb_frames,
            teacher_by_source_frame=teacher_path,
        )


def test_sst_rejects_semantic_metadata_forgery_after_all_dependent_hashes_are_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    manifest_path, manifest = _write_production_sst_fixture(
        tmp_path,
        stage2=stage2,
        monkeypatch=monkeypatch,
    )
    clip = manifest["clips"][0]
    metadata_path = Path(clip["dependencies"]["wasb_metadata_path"])
    forged_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    forged_metadata["source_mode"] = "wasb_csv"
    metadata_path.write_text(json.dumps(forged_metadata), encoding="utf-8")
    forged_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    clip["dependencies"]["wasb_metadata_sha256"] = forged_sha
    clip["dependencies"]["wasb_runtime"] = forged_metadata
    for row in clip["samples"]:
        row["dependency_hashes"]["wasb_metadata_sha256"] = forged_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source_mode is not production-authentic"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)


def test_sst_rejects_invalid_prediction_csv_with_self_consistent_hashes_and_replay_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    manifest_path, manifest = _write_production_sst_fixture(
        tmp_path,
        stage2=stage2,
        monkeypatch=monkeypatch,
    )
    clip = manifest["clips"][0]
    csv_path = Path(clip["dependencies"]["wasb_predictions_csv_path"])
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    rows[1][1] = "2"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    metadata_path = Path(clip["dependencies"]["wasb_metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["builder_bindings"]["wasb_predictions_csv_sha256"] = csv_sha
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    clip["dependencies"]["wasb_predictions_csv_sha256"] = csv_sha
    clip["dependencies"]["wasb_metadata_sha256"] = metadata_sha
    clip["dependencies"]["wasb_runtime"] = metadata
    for row in clip["samples"]:
        row["dependency_hashes"]["wasb_predictions_csv_sha256"] = csv_sha
        row["dependency_hashes"]["wasb_metadata_sha256"] = metadata_sha
    manifest["artifact_verification"]["replayed_prediction_sha256_by_clip"][
        clip["clip_id"]
    ] = csv_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="visibility must be exactly 0 or 1"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)


def test_sst_rejects_replay_mismatch_adapter_forgery_and_symlink_dependency_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    manifest_path, manifest = _write_production_sst_fixture(
        tmp_path,
        stage2=stage2,
        monkeypatch=monkeypatch,
    )
    forged = copy.deepcopy(manifest)
    forged["preregistration"]["wasb_adapter_code_sha256"] = "f" * 64
    forged["preregistered_parameters"] = dict(forged["preregistration"])
    manifest_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="adapter code identity"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(stage2, "_run_official_wasb_replay", lambda **_kwargs: b"different")
    with pytest.raises(ValueError, match="official WASB inference replay differs"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)

    monkeypatch.setattr(
        stage2,
        "_run_official_wasb_replay",
        lambda **kwargs: (
            Path(kwargs["frame_times_path"]).parent / "wasb_predictions.csv"
        ).read_bytes(),
    )
    dependency_root = tmp_path / "sst_manifest_dependencies"
    dependency_backing = tmp_path / "sst_manifest_dependencies_backing"
    dependency_root.rename(dependency_backing)
    dependency_root.symlink_to(dependency_backing, target_is_directory=True)
    with pytest.raises(ValueError, match="direct canonical path|symlink path component"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)


def test_sst_rejects_out_of_range_pbvision_probabilities_even_when_cv_export_is_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    manifest_path, manifest = _write_production_sst_fixture(
        tmp_path,
        stage2=stage2,
        monkeypatch=monkeypatch,
    )
    clip = manifest["clips"][0]
    cv_export_path = Path(manifest["gallery_root"]) / clip["clip_id"] / "cv_export.json"
    cv_export = json.loads(cv_export_path.read_text(encoding="utf-8"))
    cv_export["sessions"][0]["rallies"][0]["frames"][0]["actions"]["ball"]["u"] = 1.01
    cv_export_path.write_text(json.dumps(cv_export), encoding="utf-8")
    cv_export_sha = hashlib.sha256(cv_export_path.read_bytes()).hexdigest()
    clip["dependencies"]["pbvision_cv_export_sha256"] = cv_export_sha
    for row in clip["samples"]:
        row["dependency_hashes"]["pbvision_cv_export_sha256"] = cv_export_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen PBVision gallery bundle SHA mismatch"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)


def test_sst_rejects_valid_rehashed_teacher_edit_against_independent_gallery_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    manifest_path, manifest = _write_production_sst_fixture(
        tmp_path,
        stage2=stage2,
        monkeypatch=monkeypatch,
    )
    clip = manifest["clips"][0]
    sample = clip["samples"][0]
    source_id = clip["clip_id"]
    forged_x = float(sample["teacher_xy"][0]) + 1.0
    cv_export_path = Path(manifest["gallery_root"]) / source_id / "cv_export.json"
    cv_export = json.loads(cv_export_path.read_text(encoding="utf-8"))
    cv_export["sessions"][0]["rallies"][0]["frames"][0]["actions"]["ball"]["u"] = (
        forged_x / 640.0
    )
    cv_export_path.write_text(json.dumps(cv_export), encoding="utf-8")
    forged_sha = hashlib.sha256(cv_export_path.read_bytes()).hexdigest()
    clip["dependencies"]["pbvision_cv_export_sha256"] = forged_sha
    for row in clip["samples"]:
        row["dependency_hashes"]["pbvision_cv_export_sha256"] = forged_sha
    sample["teacher_xy"] = [forged_x, 80.0]
    sample["agreement"]["teacher_xy"] = [forged_x, 80.0]
    sample["agreement"]["distance_px"] = abs(
        float(sample["agreement"]["wasb_xy"][0]) - forged_x
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen PBVision gallery bundle SHA mismatch"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)


def test_sst_rejects_monotonic_rehashed_pts_that_differ_from_encoded_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.racketsport import train_ball_stage2 as stage2

    manifest_path, manifest = _write_production_sst_fixture(
        tmp_path,
        stage2=stage2,
        monkeypatch=monkeypatch,
    )
    clip = manifest["clips"][0]
    dependencies = clip["dependencies"]
    frame_times_path = Path(dependencies["frame_times_path"])
    frame_times = json.loads(frame_times_path.read_text(encoding="utf-8"))
    for row in frame_times["frames"]:
        row["pts_s"] = float(row["pts_s"]) + 1e-6
    frame_times_path.write_text(json.dumps(frame_times), encoding="utf-8")
    frame_times_sha = hashlib.sha256(frame_times_path.read_bytes()).hexdigest()

    wasb_track_path = Path(dependencies["wasb_ball_track"])
    regenerated_track = stage2.wasb_csv_to_ball_track(
        Path(dependencies["wasb_predictions_csv_path"]),
        fps=30.0,
        frame_times=frame_times_path,
        visible_threshold=0.9,
        input_preprocessing="official",
    )
    wasb_track_path.write_text(json.dumps(regenerated_track), encoding="utf-8")
    wasb_track_sha = hashlib.sha256(wasb_track_path.read_bytes()).hexdigest()

    wasb_metadata_path = Path(dependencies["wasb_metadata_path"])
    wasb_metadata = json.loads(wasb_metadata_path.read_text(encoding="utf-8"))
    wasb_metadata["builder_bindings"]["frame_times_sha256"] = frame_times_sha
    wasb_metadata["builder_bindings"]["wasb_ball_track_sha256"] = wasb_track_sha
    wasb_metadata_path.write_text(json.dumps(wasb_metadata), encoding="utf-8")
    wasb_metadata_sha = hashlib.sha256(wasb_metadata_path.read_bytes()).hexdigest()

    dependencies["frame_times_sha256"] = frame_times_sha
    dependencies["wasb_ball_track_sha256"] = wasb_track_sha
    dependencies["wasb_metadata_sha256"] = wasb_metadata_sha
    dependencies["wasb_runtime"] = wasb_metadata
    for sample in clip["samples"]:
        sample["t"] = float(sample["t"]) + 1e-6
        sample["frame_ref"]["t"] = float(sample["frame_ref"]["t"]) + 1e-6
        sample["dependency_hashes"]["frame_times_sha256"] = frame_times_sha
        sample["dependency_hashes"]["wasb_ball_track_sha256"] = wasb_track_sha
        sample["dependency_hashes"]["wasb_metadata_sha256"] = wasb_metadata_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frame PTS differ from canonical encoded media"):
        stage2.load_and_validate_production_sst_manifest(manifest_path)


def test_scaffold_index_covers_train_ball_stage2_cli() -> None:
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
    assert by_path[CLI_PATH]["direct_cli_reference_test"] == "tests/racketsport/test_ball_stage2_training.py"


def _cvat_payload(
    *,
    frame_count: int,
    reviewed_frame_indices: list[int] | None = None,
    ball_frames: dict[int, tuple[float, float, float, float]] | None = None,
    ball_visibility_levels: dict[int, str] | None = None,
    frame_visibility_levels: dict[int, str] | None = None,
) -> dict[str, object]:
    ball_frames = ball_frames or {}
    frames = []
    for frame_index in range(frame_count):
        boxes = []
        bbox = ball_frames.get(frame_index)
        if bbox is not None:
            x, y, width, height = bbox
            box: dict[str, object] = {
                "track_id": 7,
                "label": "ball",
                "frame_index": frame_index,
                "bbox_xyxy": [x, y, x + width, y + height],
                "bbox_xywh": [x, y, width, height],
                "keyframe": True,
                "occluded": False,
                "source": "manual",
            }
            if ball_visibility_levels and frame_index in ball_visibility_levels:
                box["visibility_level"] = ball_visibility_levels[frame_index]
            boxes.append(box)
        frame_payload: dict[str, object] = {"frame_index": frame_index, "boxes": boxes}
        if frame_visibility_levels and frame_index in frame_visibility_levels:
            frame_payload["visibility_levels_by_label"] = {"ball": frame_visibility_levels[frame_index]}
        frames.append(frame_payload)
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": "clip_train",
        "source_format": "cvat_video_1_1",
        "source_path": "clip_train.zip",
        "task": {
            "task_id": 42,
            "name": "clip_train",
            "size": frame_count,
            "mode": "interpolation",
            "start_frame": 0,
            "stop_frame": frame_count - 1,
            "original_size": [32, 32],
            "source": "clip_train.mp4",
        },
        "frames": frames,
        "tracks": [
            {
                "track_id": 7,
                "label": "ball",
                "visible_box_count": len(ball_frames),
                "outside_box_count": 0,
                "keyframe_count": len(ball_frames),
                "first_visible_frame": min(ball_frames) if ball_frames else None,
                "last_visible_frame": max(ball_frames) if ball_frames else None,
            }
        ],
        "summary": {
            "frame_count": frame_count,
            "visible_box_count": len(ball_frames),
            "outside_box_count": 0,
            "labels": ["ball"],
            "track_count_by_label": {"ball": 1},
            "visible_box_count_by_label": {"ball": len(ball_frames)},
        },
    }
    if reviewed_frame_indices is not None:
        payload["reviewed_frame_indices"] = reviewed_frame_indices
        payload["reviewed_frame_indices_source"] = "explicit"
    return payload


def _run_stage2_train(
    stage2,
    *,
    cvat_root: Path,
    rally_root: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    resume_checkpoint: Path | None = None,
    legacy_namespace: bool = False,
) -> dict[str, object]:
    argv = [
        "--cvat-export-root",
        str(cvat_root),
        "--out-dir",
        str(out_dir),
        "--model-family",
        "tiny_wasb",
        "--device",
        "cpu",
        "--image-size",
        "32x32",
        "--frames-in",
        "3",
        "--output-channels",
        "1",
        "--steps",
        str(steps),
        "--batch-size",
        "1",
        "--learning-rate",
        "0.02",
        "--weight-decay",
        "0.0",
        "--checkpoint-every",
        "1",
        "--num-workers",
        "0",
        "--seed",
        str(seed),
        "--occluded-prob",
        "0",
        "--rally-root",
        str(rally_root),
    ]
    if resume_checkpoint is not None:
        argv.extend(["--resume-checkpoint", str(resume_checkpoint)])
    args = stage2._build_parser().parse_args(argv)
    if legacy_namespace:
        delattr(args, "sst_batch_size")
        delattr(args, "sst_loss_cap")
    summary = stage2.run(args)
    return summary


def _write_production_sst_fixture(tmp_path: Path, *, stage2, monkeypatch: pytest.MonkeyPatch):
    builder_sha = stage2._sha256_file(Path("scripts/racketsport/build_pbvision_ball_sst.py").resolve())
    adapter_path = Path(stage2.SST_WASB_ADAPTER_PATH).resolve()
    adapter_sha = stage2._sha256_file(adapter_path)
    wasb = stage2._expected_wasb_identity()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    checkpoint_path = Path(wasb["checkpoint_path"]).resolve()
    repo_path = Path(stage2.SST_WASB_REPO_ROOT).resolve()
    gallery_root = (tmp_path / "gallery").resolve()
    gallery_root.mkdir(parents=True)
    split_manifest = (tmp_path / "frozen_split_manifest.json").resolve()
    split_manifest.write_text(json.dumps({"schema_version": 1, "rows": []}), encoding="utf-8")
    split_sha = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    media_root = (tmp_path / "media").resolve()
    media_hashes = {}
    media_paths = {}
    for source_id in sorted(stage2.SST_TRAIN_SOURCE_IDS):
        media_path = media_root / source_id / "max.mp4"
        media_path.parent.mkdir(parents=True)
        media_path.write_bytes(f"fixture-media-{source_id}".encode())
        media_paths[source_id] = media_path
        media_hashes[source_id] = hashlib.sha256(media_path.read_bytes()).hexdigest()
    monkeypatch.setattr(stage2, "SST_EXPECTED_SOURCE_VIDEO_SHA256", media_hashes)
    monkeypatch.setattr(stage2, "SST_FROZEN_GALLERY_ROOT", gallery_root)
    monkeypatch.setattr(stage2, "SST_FROZEN_SPLIT_MANIFEST", split_manifest)
    monkeypatch.setattr(stage2, "SST_FROZEN_SPLIT_SHA256", split_sha)
    monkeypatch.setattr(stage2, "_video_size", lambda _path: (640, 480))
    monkeypatch.setattr(
        stage2,
        "_run_official_wasb_replay",
        lambda **kwargs: (
            Path(kwargs["frame_times_path"]).parent / "wasb_predictions.csv"
        ).read_bytes(),
    )

    top_dependencies = {
        "split_manifest_sha256": split_sha,
        "models_manifest_sha256": wasb["models_manifest_sha256"],
        "builder_code_sha256": builder_sha,
        "wasb_adapter_code_sha256": adapter_sha,
        "wasb_checkpoint_sha256": wasb["checkpoint_sha256"],
        "wasb_repo_commit": wasb["repo_commit"],
    }
    clips = []
    encoded_timings = {}
    source_ids = sorted(stage2.SST_TRAIN_SOURCE_IDS)
    manifest_path = tmp_path / "sst_manifest.json"
    dependency_root = tmp_path / "sst_manifest_dependencies"
    remaining = 1000
    for clip_index, source_id in enumerate(source_ids):
        count = remaining // (len(source_ids) - clip_index)
        remaining -= count
        encoded_timings[source_id] = {
            "fps": 30.0,
            "duration_s": count / 30.0,
            "width": 640,
            "height": 480,
            "frame_count": count,
            "pts_s": [frame_index / 30.0 for frame_index in range(count)],
        }
        source_gallery = gallery_root / source_id
        source_gallery.mkdir(parents=True)
        teacher_frames = []
        pts_rows = []
        for frame_index in range(count):
            teacher_xy = [100.0 + (frame_index % 10), 80.0]
            teacher_frames.append(
                {
                    "actions": {
                        "ball": {
                            "confidence": 0.95,
                            "u": teacher_xy[0] / 640.0,
                            "v": teacher_xy[1] / 480.0,
                        }
                    }
                }
            )
            pts_rows.append({"frame": frame_index, "pts_s": frame_index / 30.0})
        cv_export_path = source_gallery / "cv_export.json"
        cv_export_path.write_text(
            json.dumps(
                {
                    "camera": {"fps": 30.0},
                    "sessions": [{"rallies": [{"frame_index": 0, "frames": teacher_frames}]}],
                }
            ),
            encoding="utf-8",
        )
        metadata_path = source_gallery / "api_get_metadata.json"
        metadata_path.write_text(
            json.dumps({"metadata": {"width": 640, "height": 480, "fps": 30.0}}),
            encoding="utf-8",
        )
        provenance_path = source_gallery / "video_provenance.json"
        provenance_path.write_text(
            json.dumps(
                {
                    "video_id": source_id,
                    "source_video_url": f"https://storage.googleapis.com/pbv-pro/{source_id}/max.mp4",
                }
            ),
            encoding="utf-8",
        )
        source_dependencies = dependency_root / source_id
        source_dependencies.mkdir(parents=True)
        frame_times_path = source_dependencies / "frame_times.json"
        frame_times_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "racketsport_frame_times",
                    "source_video_sha256": media_hashes[source_id],
                    "fps": 30.0,
                    "duration_s": count / 30.0,
                    "width": 640,
                    "height": 480,
                    "frame_count": count,
                    "frames": pts_rows,
                }
            ),
            encoding="utf-8",
        )
        wasb_predictions_csv_path = source_dependencies / "wasb_predictions.csv"
        with wasb_predictions_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Frame", "Visibility", "X", "Y", "Confidence"])
            for frame_index in range(count):
                teacher_x = 100.0 + (frame_index % 10)
                writer.writerow([frame_index, 1, teacher_x + 2.0, 80.0, 0.97])
        wasb_track_path = source_dependencies / "wasb_ball_track.json"
        wasb_track_path.write_text(
            json.dumps(
                stage2.wasb_csv_to_ball_track(
                    wasb_predictions_csv_path,
                    fps=30.0,
                    frame_times=frame_times_path,
                    visible_threshold=0.9,
                    input_preprocessing="official",
                )
            ),
            encoding="utf-8",
        )
        wasb_bindings = {
            "source_video_sha256": media_hashes[source_id],
            "frame_times_sha256": hashlib.sha256(frame_times_path.read_bytes()).hexdigest(),
            "wasb_predictions_csv_sha256": hashlib.sha256(
                wasb_predictions_csv_path.read_bytes()
            ).hexdigest(),
            "wasb_ball_track_sha256": hashlib.sha256(wasb_track_path.read_bytes()).hexdigest(),
            "wasb_checkpoint_sha256": wasb["checkpoint_sha256"],
            "wasb_repo_commit": wasb["repo_commit"],
            "wasb_adapter_code_sha256": adapter_sha,
        }
        wall_seconds = count / 30.0
        runtime = {
            "wasb_repo": str(repo_path),
            "wasb_repo_commit": wasb["repo_commit"],
            "wasb_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": wasb["checkpoint_sha256"],
            },
            "video": str(media_paths[source_id]),
            "source_video_fps": 30.0,
            "source_video_frame_count": count,
            "source_video_size": [640, 480],
            "processed_frame_count": count,
            "processed_window_count": count - 2,
            "read_frame_count": count,
            "video_range_seconds": None,
            "max_frames": None,
            "batch_size": 8,
            "device": "cpu",
            "input_preprocessing": "official",
            "non_promotable_measurement_mode": False,
            "wall_seconds": wall_seconds,
            "effective_fps": count / wall_seconds,
            "realtime_factor": (count / wall_seconds) / 30.0,
        }
        wasb_metadata = {
            "schema_version": 1,
            "artifact_type": "racketsport_wasb_ball_run",
            "status": stage2.STATUS_TESTED,
            "source_mode": "wasb_predict",
            "predictions_csv": str(wasb_predictions_csv_path),
            "out": str(wasb_track_path),
            "fps": 30.0,
            "frame_count": count,
            "visible_frame_count": count,
            "confidence_semantics": stage2.WASB_CONFIDENCE_SEMANTICS,
            "visible_threshold": 0.9,
            "input_preprocessing": "official",
            "non_promotable_measurement_mode": False,
            "not_ground_truth": True,
            "official_repo_url": stage2.WASB_REPO_URL,
            "official_model_zoo_url": stage2.WASB_MODEL_ZOO_URL,
            "runtime": runtime,
            "builder_bindings": wasb_bindings,
        }
        wasb_metadata_path = source_dependencies / "wasb_ball_track_metadata.json"
        wasb_metadata_path.write_text(json.dumps(wasb_metadata), encoding="utf-8")
        dependency_hashes = {
            **top_dependencies,
            "source_video_sha256": media_hashes[source_id],
            "frame_times_sha256": hashlib.sha256(frame_times_path.read_bytes()).hexdigest(),
            "pbvision_cv_export_sha256": hashlib.sha256(cv_export_path.read_bytes()).hexdigest(),
            "pbvision_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "pbvision_provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            "wasb_ball_track_sha256": hashlib.sha256(wasb_track_path.read_bytes()).hexdigest(),
            "wasb_metadata_sha256": hashlib.sha256(wasb_metadata_path.read_bytes()).hexdigest(),
            "wasb_predictions_csv_sha256": hashlib.sha256(
                wasb_predictions_csv_path.read_bytes()
            ).hexdigest(),
        }
        clip_dependencies = {
            **dependency_hashes,
            "frame_times_path": str(frame_times_path.resolve()),
            "wasb_ball_track": str(wasb_track_path.resolve()),
            "wasb_metadata_path": str(wasb_metadata_path.resolve()),
            "wasb_predictions_csv_path": str(wasb_predictions_csv_path.resolve()),
            "wasb_runtime": wasb_metadata,
        }
        samples = []
        for frame_index in range(count):
            teacher_xy = [100.0 + (frame_index % 10), 80.0]
            wasb_xy = [teacher_xy[0] + 2.0, teacher_xy[1]]
            samples.append(
                {
                    "sample_id": f"{source_id}:{frame_index}",
                    "clip_id": source_id,
                    "canonical_source_id": source_id,
                    "frame_index": frame_index,
                    "teacher_frame_index": frame_index,
                    "t": frame_index / 30.0,
                    "frame_ref": {
                        "video": str(media_paths[source_id]),
                        "frame_index": frame_index,
                        "t": frame_index / 30.0,
                        "source_video_sha256": media_hashes[source_id],
                    },
                    "source_video_sha256": media_hashes[source_id],
                    "teacher_xy": teacher_xy,
                    "score": 0.95,
                    "weight": 0.25,
                    "teacher_source": "pbvision_actions_ball",
                    "teacher_derived": True,
                    "ground_truth": False,
                    "ball_present": True,
                    "agreement_reason": "frozen_wasb_spatial",
                    "agreement": {
                        "policy_id": "frozen_wasb_spatial_v2",
                        "source_frame_index": frame_index,
                        "teacher_xy": teacher_xy,
                        "wasb_xy": wasb_xy,
                        "teacher_confidence": 0.95,
                        "wasb_confidence": 0.97,
                        "distance_px": 2.0,
                        "agreement_radius_px": 20.0,
                        "image_width": 640,
                        "image_height": 480,
                        "all_points_in_bounds": True,
                    },
                    "dependency_hashes": dict(dependency_hashes),
                }
            )
        clips.append(
            {
                "clip_id": source_id,
                "canonical_source_id": source_id,
                "split": "train",
                "teacher_derived": True,
                "ground_truth": False,
                "rally_video": str(media_paths[source_id]),
                "source_video_sha256": media_hashes[source_id],
                "source_width": 640,
                "source_height": 480,
                "fps": 30.0,
                "sample_count": len(samples),
                "dependencies": dict(clip_dependencies),
                "samples": samples,
            }
        )
    preregistration = {
        "policy_id": "pbv_ball_sst_production_v2",
        "teacher_confidence_min": 0.9,
        "agreement_radius_px": 20.0,
        "pseudo_weight": 0.25,
        "temporal_max_gap_source_frames": 2,
        "temporal_geometry": dict(stage2.SST_TEMPORAL_GEOMETRY_POLICY),
        "canonical_media_relative_path": "<video_id>/max.mp4",
        "expected_source_video_sha256": dict(media_hashes),
        "builder_code_sha256": builder_sha,
        "builder_git_commit": git_commit,
        "wasb_adapter_code_sha256": adapter_sha,
        "wasb_adapter_git_commit": git_commit,
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_sst_manifest",
        "production_eligible": True,
        "production_policy_selected": True,
        "policy_override_fields": [],
        "declared_policy_override_fields": [],
        "policy_override_declaration_matches": True,
        "teacher_derived": True,
        "ground_truth": False,
        "protected_eval_clips_touched": False,
        "source_policy": {
            "train_ids": source_ids,
            "teacher_absence_policy": "ignored_never_negative",
            "positive_rows_only": True,
        },
        "gallery_root": str(gallery_root),
        "media_root": str(media_root),
        "split_manifest": str(split_manifest),
        "wasb_checkpoint": str(checkpoint_path),
        "preregistration": preregistration,
        "preregistered_parameters": dict(preregistration),
        "requested_parameters": {
            "teacher_confidence_min": 0.9,
            "agreement_radius_px": 20.0,
            "pseudo_weight": 0.25,
        },
        "builder_identity": {
            "builder_path": "scripts/racketsport/build_pbvision_ball_sst.py",
            "builder_code_sha256": builder_sha,
            "builder_git_commit": git_commit,
            "wasb_adapter_path": str(stage2.SST_WASB_ADAPTER_PATH),
            "wasb_adapter_code_sha256": adapter_sha,
            "wasb_adapter_git_commit": git_commit,
        },
        "dependency_hashes": top_dependencies,
        "wasb_identity": {
            "manifest_model_id": "wasb_tennis_bmvc2023",
            "models_manifest_path": str(Path("models/MANIFEST.json").resolve()),
            "models_manifest_sha256": wasb["models_manifest_sha256"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": wasb["checkpoint_sha256"],
            "expected_checkpoint_sha256": wasb["checkpoint_sha256"],
            "repo_path": str(repo_path),
            "repo_commit": wasb["repo_commit"],
            "expected_repo_commit": wasb["repo_commit"],
            "repo_clean": True,
            "production_identity_verified": True,
        },
        "accepted_windows": 1000,
        "accepted_sources": 7,
        "holdout_rows_present": 0,
        "decode_status": "completed",
        "decode_failures": 0,
        "artifact_verification": {
            "verified": True,
            "status": "passed",
            "reason": "fixture dependencies rehashed",
            "verified_clip_count": 7,
            "verified_sample_count": 1000,
            "split_manifest_sha256": split_sha,
            "builder_code_sha256": builder_sha,
            "wasb_adapter_code_sha256": adapter_sha,
            "wasb_checkpoint_sha256": wasb["checkpoint_sha256"],
            "wasb_repo_commit": wasb["repo_commit"],
            "official_wasb_replay_verified": True,
            "official_wasb_replay_clip_count": 7,
            "replayed_prediction_sha256_by_clip": {
                clip["clip_id"]: clip["dependencies"]["wasb_predictions_csv_sha256"]
                for clip in clips
            },
        },
        "gate": {
            "verdict": "PASS",
            "production_eligible": {"after": True, "target": True},
            "artifacts_verified": {"after": True, "target": True},
            "accepted_windows": {"after": 1000, "target": 1000},
            "accepted_sources": {"after": 7, "target": 5},
            "holdout_rows_present": {"after": 0, "target": 0},
            "decode_failures": {"after": 0, "target": 0},
        },
        "clips": clips,
    }
    gallery_identity = stage2._compute_pbvision_gallery_bundle(gallery_root)
    teacher_input_authority = {
        "authority_id": stage2.SST_PBVISION_GALLERY_AUTHORITY_ID,
        "canonical_gallery_relative_path": stage2.SST_FROZEN_GALLERY_ROOT.as_posix(),
        "artifact_filenames": list(stage2.SST_PBVISION_GALLERY_FILENAMES),
        "expected_sha256_by_source": gallery_identity["sha256_by_source"],
    }
    manifest["preregistration"]["teacher_input_authority"] = teacher_input_authority
    manifest["preregistered_parameters"] = dict(manifest["preregistration"])
    manifest["artifact_verification"]["pbvision_gallery_authority_id"] = (
        stage2.SST_PBVISION_GALLERY_AUTHORITY_ID
    )
    manifest["artifact_verification"]["verified_pbvision_gallery_sha256_by_source"] = (
        gallery_identity["sha256_by_source"]
    )
    monkeypatch.setattr(
        stage2,
        "SST_FROZEN_GALLERY_BUNDLE_SHA256",
        gallery_identity["bundle_sha256"],
    )
    monkeypatch.setattr(
        stage2,
        "_probe_canonical_media_timing",
        lambda _source_video, *, clip_id: copy.deepcopy(encoded_timings[clip_id]),
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def _write_tiny_video(path: Path, *, frame_count: int, cv2) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("cv2 VideoWriter mp4v is unavailable in this environment")
    for index in range(frame_count):
        frame = np.full((32, 32, 3), 20 + index * 20, dtype=np.uint8)
        frame[10:14, 10:14] = (0, 255, 255)
        writer.write(frame)
    writer.release()


def _write_gradient_video(path: Path, *, frame_count: int, width: int, height: int, cv2, np) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    if not writer.isOpened():
        pytest.skip("cv2 VideoWriter mp4v is unavailable in this environment")
    yy, xx = np.mgrid[0:height, 0:width]
    for index in range(frame_count):
        rgb = np.stack(
            [
                (xx * 5 + yy * 3 + index * 11) % 256,
                (xx * 7 + 13 + index * 17) % 256,
                (yy * 9 + 19 + index * 23) % 256,
            ],
            axis=2,
        ).astype(np.uint8)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def _read_video_rgb_frames(path: Path, indices: list[int], cv2) -> list[object]:
    frames = []
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise ValueError(f"could not read frame {frame_index} from {path}")
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames
