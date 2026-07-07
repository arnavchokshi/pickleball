from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CLI_PATH = "scripts/racketsport/train_ball_stage2.py"


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
    assert "--sst-manifest" in completed.stdout
    assert "--occluded-prob" in completed.stdout
    assert "--init-checkpoint" in completed.stdout


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
