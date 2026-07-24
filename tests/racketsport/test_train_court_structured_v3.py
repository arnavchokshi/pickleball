from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = "scripts/racketsport/train_court_structured_v3.py"

from scripts.racketsport.train_court_structured_v3 import (  # noqa: E402
    ALL_KEYPOINT_NAMES,
    CANONICAL_FLOOR_NAMES,
    StructuredLossWeights,
    build_parser,
    load_unsupported_view_rows,
    real_row_to_structured_arrays,
    structured_v3_losses,
    synthetic_sample_to_structured_arrays,
    unsupported_row_to_structured_arrays,
)
from threed.racketsport.court_structured_model import (  # noqa: E402
    STRUCTURED_DISTANCE_CLASS_NAMES,
    STRUCTURED_FLOOR_KEYPOINT_COUNT,
    STRUCTURED_FLOOR_KEYPOINT_NAMES,
    STRUCTURED_FLOOR_KEYPOINTS,
    covariance_matrices_from_params,
)
from threed.racketsport.court_structured_training import (  # noqa: E402
    project_homography,
    semantic_segment_distance_targets,
    structured_floor_world_xy,
    weighted_masked_mean,
)


def _write_image(path: Path, *, width: int = 256, height: int = 192) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (38, 92, 73)).save(path)


def _reviewed_row(image_path: Path) -> dict[str, object]:
    world = {point.name: point.world_xyz_m[:2] for point in STRUCTURED_FLOOR_KEYPOINTS}

    def image_xy(name: str) -> list[float]:
        x, y = world[name]
        return [float(100.0 + 8.0 * x), float(96.0 + 8.0 * y)]

    anchors = ("near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner")
    return {
        "image_path": str(image_path),
        "video_path": None,
        "frame_index": 1,
        "source_video_size": [256, 192],
        "keypoints": {name: image_xy(name) for name in anchors},
        "label_status": "reviewed",
        "clip": "human_clip",
    }


def test_distance_taxonomy_is_eight_semantic_painted_segments() -> None:
    assert STRUCTURED_DISTANCE_CLASS_NAMES == (
        "near_baseline",
        "far_baseline",
        "left_sideline",
        "right_sideline",
        "near_nvz",
        "far_nvz",
        "near_centerline",
        "far_centerline",
    )
    points = structured_floor_world_xy().unsqueeze(0)
    target = semantic_segment_distance_targets(points, height=12, width=16, max_distance_px=8.0)
    assert target.shape == (1, 8, 12, 16)
    assert torch.all((0.0 <= target) & (target <= 1.0))


def test_real_adapter_infers_only_auxiliary_targets_from_human_floor_anchors(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    _write_image(image)
    row = real_row_to_structured_arrays(
        _reviewed_row(image),
        model_width=256,
        model_height=192,
        sigma_px=1.5,
        sample_weight=0.25,
        source_kind="external_reviewed",
    )

    assert row["image"].shape == (3, 192, 256)
    assert row["target_xy_heatmap"].shape == (30, 2)
    assert row["heatmap_target"].shape == (30, 48, 64)
    assert row["distance_target"].shape == (8, 48, 64)
    assert row["anchor_report"]["available"] is True
    assert row["sample_weight"].item() == pytest.approx(0.25)
    assert row["keypoint_mask"][STRUCTURED_FLOOR_KEYPOINT_NAMES.index("near_baseline_center")] == 0.0
    assert row["visibility_mask"].sum() == 4.0
    assert row["keypoint_mask"][len(CANONICAL_FLOOR_NAMES) :].sum() == 18.0
    assert row["distance_mask"].sum() == 8.0
    assert not any(name.startswith("net_") for name in STRUCTURED_FLOOR_KEYPOINT_NAMES)


def test_real_adapter_masks_auxiliary_and_dense_targets_without_four_anchors(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    _write_image(image)
    raw = _reviewed_row(image)
    raw["keypoints"] = {
        "near_left_corner": raw["keypoints"]["near_left_corner"],
        "near_right_corner": raw["keypoints"]["near_right_corner"],
        "far_right_corner": raw["keypoints"]["far_right_corner"],
    }
    row = real_row_to_structured_arrays(
        raw,
        model_width=256,
        model_height=192,
        sigma_px=1.5,
        sample_weight=1.0,
        source_kind="human_reviewed",
    )
    assert row["keypoint_mask"].sum() == 3.0
    assert row["keypoint_mask"][len(CANONICAL_FLOOR_NAMES) :].sum() == 0.0
    assert row["distance_mask"].sum() == 0.0
    assert row["anchor_report"]["reason"] == "fewer_than_four_reviewed_floor_anchors"


def test_real_adapter_rejects_four_collinear_anchors_for_auxiliary_targets(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.jpg"
    _write_image(image)
    raw = _reviewed_row(image)
    world = {point.name: point.world_xyz_m[:2] for point in STRUCTURED_FLOOR_KEYPOINTS}
    names = ("near_left_corner", "near_nvz_left", "far_nvz_left", "far_left_corner")
    raw["keypoints"] = {
        name: [float(100.0 + 8.0 * world[name][0]), float(96.0 + 8.0 * world[name][1])]
        for name in names
    }
    row = real_row_to_structured_arrays(
        raw,
        model_width=256,
        model_height=192,
        sigma_px=1.5,
        sample_weight=1.0,
        source_kind="human_reviewed",
    )
    assert row["keypoint_mask"].sum() == 4.0
    assert row["keypoint_mask"][len(CANONICAL_FLOOR_NAMES) :].sum() == 0.0
    assert row["distance_mask"].sum() == 0.0
    assert row["anchor_report"]["reason"] == "degenerate_reviewed_floor_anchors"


def test_real_adapter_requires_median_3px_and_p95_5px_for_derived_targets(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.jpg"
    _write_image(image)
    raw = _reviewed_row(image)
    world = {point.name: point.world_xyz_m[:2] for point in STRUCTURED_FLOOR_KEYPOINTS}
    raw["keypoints"] = {
        name: [float(100.0 + 8.0 * xy[0]), float(96.0 + 8.0 * xy[1])]
        for name, xy in world.items()
        if name in CANONICAL_FLOOR_NAMES
    }
    raw["keypoints"]["near_baseline_center"][0] += 20.0
    row = real_row_to_structured_arrays(
        raw,
        model_width=256,
        model_height=192,
        sigma_px=1.5,
        sample_weight=1.0,
        source_kind="human_reviewed",
    )
    assert row["anchor_report"]["available"] is False
    assert row["anchor_report"]["reason"] == "reviewed_anchor_reprojection_above_limit"
    assert row["anchor_report"]["max_median_reprojection_px"] == pytest.approx(3.0)
    assert row["anchor_report"]["max_p95_reprojection_px"] == pytest.approx(5.0)
    assert row["keypoint_mask"][len(CANONICAL_FLOOR_NAMES) :].sum() == 0.0
    assert row["distance_mask"].sum() == 0.0


def test_unsupported_owner_views_supervise_only_supported_view_head(tmp_path: Path) -> None:
    image = tmp_path / "unsupported" / "venue" / "frames" / "bad.jpg"
    _write_image(image)
    manifest = tmp_path / "unsupported_view_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_court_unsupported_view_labels",
                "authority": "owner_reviewed",
                "items": [
                    {
                        "image": "unsupported/venue/frames/bad.jpg",
                        "supported_view": False,
                        "reasons": ["bad_angle"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = load_unsupported_view_rows([manifest])
    adapted = unsupported_row_to_structured_arrays(rows[0], model_width=128, model_height=96)
    assert adapted["supported_view_target"] == 0.0
    assert adapted["supported_view_mask"] == 1.0
    assert adapted["keypoint_mask"].sum() == 0.0
    assert adapted["visibility_mask"].sum() == 0.0
    assert adapted["distance_mask"].sum() == 0.0
    assert adapted["segmentation_mask"] == 0.0


def test_synthetic_adapter_removes_net_and_masks_unsupported_portrait() -> None:
    height, width = 96, 160
    xy = np.stack(
        (np.linspace(10.0, 140.0, len(ALL_KEYPOINT_NAMES)), np.linspace(10.0, 80.0, len(ALL_KEYPOINT_NAMES))),
        axis=-1,
    ).astype(np.float32)
    base = {
        "image_bgr": np.zeros((height, width, 3), dtype=np.uint8),
        "keypoints_xy": xy,
        "keypoints_vis": np.full((len(ALL_KEYPOINT_NAMES),), 2, dtype=np.int64),
        "keypoint_heatmap_mask": np.ones((len(ALL_KEYPOINT_NAMES), height // 4, width // 4), dtype=np.float32),
        "line_family_mask": np.zeros((height, width), dtype=np.uint8),
        "surface_mask": np.full((height, width), 2, dtype=np.uint8),
        "meta": {"scenario": "dedicated_indoor"},
    }
    supported = synthetic_sample_to_structured_arrays(
        base, model_width=width, model_height=height, sigma_px=1.5
    )
    assert supported["keypoint_mask"].sum() == STRUCTURED_FLOOR_KEYPOINT_COUNT
    assert supported["distance_mask"].sum() == 8.0
    assert supported["supported_view_target"] == 1.0

    portrait = {**base, "meta": {"scenario": "portrait_phone"}}
    unsupported = synthetic_sample_to_structured_arrays(
        portrait, model_width=width, model_height=height, sigma_px=1.5
    )
    assert unsupported["keypoint_mask"].sum() == 0.0
    assert unsupported["distance_mask"].sum() == 0.0
    assert unsupported["segmentation_mask"] == 0.0
    assert unsupported["supported_view_target"] == 0.0


def test_external_sample_weight_is_applied_before_masked_reduction() -> None:
    value = weighted_masked_mean(
        torch.tensor([1.0, 9.0]),
        torch.ones(2),
        torch.tensor([1.0, 0.25]),
    )
    assert value.item() == pytest.approx((1.0 + 0.25 * 9.0) / 1.25)


def test_all_v3_heads_backpropagate_with_missing_row_masked() -> None:
    batch_size, height, width = 2, 8, 12
    heatmaps = torch.randn(
        (batch_size, STRUCTURED_FLOOR_KEYPOINT_COUNT, height, width), requires_grad=True
    )
    vis_logits = torch.zeros((batch_size, STRUCTURED_FLOOR_KEYPOINT_COUNT), requires_grad=True)
    covariance_params = torch.zeros(
        (batch_size, STRUCTURED_FLOOR_KEYPOINT_COUNT, 3), requires_grad=True
    )
    line_logits = torch.randn((batch_size, 5, height, width), requires_grad=True)
    distance = torch.rand((batch_size, 8, height, width), requires_grad=True)
    supported_logit = torch.zeros((batch_size,), requires_grad=True)

    world = structured_floor_world_xy()
    truth_h = torch.tensor(
        [[[1.0, 0.0, 6.0], [0.0, -0.7, 4.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
    )
    target_xy = torch.zeros((batch_size, STRUCTURED_FLOOR_KEYPOINT_COUNT, 2))
    target_xy[0] = project_homography(truth_h, world)[0]
    keypoint_mask = torch.zeros((batch_size, STRUCTURED_FLOOR_KEYPOINT_COUNT))
    keypoint_mask[0] = 1.0
    target_heatmaps = torch.zeros_like(heatmaps)
    for index, (x, y) in enumerate(target_xy[0]):
        target_heatmaps[0, index, int(round(float(y))) % height, int(round(float(x))) % width] = 1.0
    batch = {
        "target_xy_heatmap": target_xy,
        "keypoint_mask": keypoint_mask,
        "heatmap_target": target_heatmaps,
        "visibility_target": torch.zeros_like(vis_logits),
        "visibility_mask": keypoint_mask.clone(),
        "segmentation_target": torch.zeros((batch_size, height, width), dtype=torch.long),
        "segmentation_mask": torch.tensor([1.0, 0.0]),
        "distance_target": torch.zeros_like(distance),
        "distance_mask": torch.tensor([[1.0] * 8, [0.0] * 8]),
        "supported_view_target": torch.tensor([1.0, 0.0]),
        "supported_view_mask": torch.ones(2),
        "sample_weight": torch.tensor([1.0, 0.25]),
    }
    losses = structured_v3_losses(
        {
            "keypoint_heatmaps": heatmaps,
            "keypoint_vis_logits": vis_logits,
            "keypoint_covariance": covariance_matrices_from_params(covariance_params),
            "line_family_logits": line_logits,
            "line_distance_maps": distance,
            "supported_view_logit": supported_logit,
        },
        batch,
        weights=StructuredLossWeights(),
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    for tensor in (heatmaps, vis_logits, covariance_params, line_logits, distance, supported_logit):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_cli_exposes_external_weight_and_v2_warm_start() -> None:
    args = build_parser().parse_args(
        [
            "--out",
            "candidate",
            "--external-real-root",
            "external",
            "--external-data-weight",
            "0.25",
            "--unsupported-view-manifest",
            "unsupported.json",
        ]
    )
    assert args.init_v2_checkpoint == Path("models/checkpoints/court_unet_v2/court_model_v2.pt")
    assert args.external_data_weight == pytest.approx(0.25)
    assert args.max_grad_norm == pytest.approx(5.0)
    assert args.external_real_root == [Path("external")]
    assert args.unsupported_view_manifest == [Path("unsupported.json")]


def test_cli_help_is_directly_invocable() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--external-data-weight" in completed.stdout
    assert "--init-v2-checkpoint" in completed.stdout
