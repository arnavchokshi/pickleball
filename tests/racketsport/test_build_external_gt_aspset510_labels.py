from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "racketsport" / "build_external_gt_aspset510_labels.py"
SPEC = importlib.util.spec_from_file_location("build_external_gt_aspset510_labels", MODULE_PATH)
build_external_gt_aspset510_labels = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_external_gt_aspset510_labels
SPEC.loader.exec_module(build_external_gt_aspset510_labels)  # type: ignore[union-attr]

from threed.racketsport.external_gt_aspset510 import ASPSET17J_JOINT_NAMES  # noqa: E402


def test_select_sample_frame_indices_covers_first_and_last_frame() -> None:
    indices = build_external_gt_aspset510_labels.select_sample_frame_indices(291, stride=10)
    assert indices[0] == 0
    assert indices[-1] == 290
    assert indices == sorted(set(indices))


def test_select_sample_frame_indices_stride_one_returns_every_frame() -> None:
    indices = build_external_gt_aspset510_labels.select_sample_frame_indices(5, stride=1)
    assert indices == [0, 1, 2, 3, 4]


def test_select_sample_frame_indices_rejects_bad_stride() -> None:
    with pytest.raises(ValueError):
        build_external_gt_aspset510_labels.select_sample_frame_indices(10, stride=0)


def test_select_sample_frame_indices_empty_for_zero_frames() -> None:
    assert build_external_gt_aspset510_labels.select_sample_frame_indices(0, stride=5) == []


def test_build_payload_matches_gate_readable_schema() -> None:
    frames_mm = [
        {name: (10.0 * idx, 20.0 * idx, 30.0 * idx) for idx, name in enumerate(ASPSET17J_JOINT_NAMES)}
        for _ in range(3)
    ]
    payload = build_external_gt_aspset510_labels.build_payload(
        frames_joint_positions_mm=frames_mm,
        frame_indices=[0, 1, 2],
        player_id=1,
        clip_id="0001",
        subject_id="1e28",
        camera_id="left",
    )
    assert payload["artifact_type"] == "racketsport_body_world_joints_labels"
    assert payload["not_ground_truth"] is False
    assert payload["trusted_for_world_mpjpe"] is True
    assert "draft" not in payload["status"]
    assert "unverified" not in payload["status"]
    assert "teacher" not in payload["status"]
    assert len(payload["samples"]) == 3
    assert payload["samples"][0]["label_source"] == "external_ground_truth"
    assert payload["provenance"]["subject_id"] == "1e28"
    assert payload["provenance"]["source_frame_count"] == 3
    assert payload["provenance"]["sampled_frame_count"] == 3


def test_build_payload_empty_frame_indices_yields_no_samples() -> None:
    payload = build_external_gt_aspset510_labels.build_payload(
        frames_joint_positions_mm=[],
        frame_indices=[],
        player_id=1,
        clip_id="0001",
        subject_id="1e28",
        camera_id="left",
    )
    assert payload["samples"] == []
    assert payload["joint_names"] == []
