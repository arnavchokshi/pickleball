from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts" / "racketsport"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_external_gt_aspset510_body_results as scoring  # noqa: E402
from threed.racketsport.external_gt_aspset510 import SHARED_CORE_JOINT_NAMES  # noqa: E402
from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES  # noqa: E402


def _fake_mhr70_frame(*, base_xyz: tuple[float, float, float]) -> list[list[float]]:
    # every joint offset deterministically by its index so we can verify the exact
    # selection-by-name logic picks the right rows out of the 70-length array.
    return [[base_xyz[0] + index, base_xyz[1], base_xyz[2]] for index in range(len(MHR70_JOINT_NAMES))]


def test_select_shared_core_from_mhr70_picks_correct_indices_not_positions() -> None:
    raw = _fake_mhr70_frame(base_xyz=(0.0, 0.0, 0.0))
    selected = scoring._select_shared_core_from_mhr70(raw)
    assert len(selected) == len(SHARED_CORE_JOINT_NAMES)
    for output_index, name in enumerate(SHARED_CORE_JOINT_NAMES):
        expected_mhr_index = MHR70_JOINT_NAMES.index(name)
        assert selected[output_index][0] == pytest.approx(float(expected_mhr_index))
    # regression guard: wrist selection must NOT come from positional indices 4/5
    # (SHARED_CORE_JOINT_NAMES' own wrist slots), it must come from MHR70's real
    # wrist indices (41 right, 62 left).
    left_wrist_out_index = list(SHARED_CORE_JOINT_NAMES).index("left_wrist")
    right_wrist_out_index = list(SHARED_CORE_JOINT_NAMES).index("right_wrist")
    assert selected[right_wrist_out_index][0] == pytest.approx(41.0)
    assert selected[left_wrist_out_index][0] == pytest.approx(62.0)


def test_predicted_frame_index_reads_player_id_one() -> None:
    smpl_motion = {
        "players": [
            {
                "id": 1,
                "frames": [
                    {"frame_idx": 0, "joints_world": _fake_mhr70_frame(base_xyz=(0.0, 0.0, 0.0))},
                    {"frame_idx": 10, "joints_world": _fake_mhr70_frame(base_xyz=(1.0, 0.0, 0.0))},
                ],
            }
        ]
    }
    index = scoring._predicted_frame_index(smpl_motion)
    assert set(index.keys()) == {0, 10}
    assert len(index[0]) == 70


def test_predicted_frame_index_rejects_wrong_joint_count() -> None:
    smpl_motion = {"players": [{"id": 1, "frames": [{"frame_idx": 0, "joints_world": [[0.0, 0.0, 0.0]]}]}]}
    with pytest.raises(scoring.ScoringError):
        scoring._predicted_frame_index(smpl_motion)


def test_score_one_clip_end_to_end_with_synthetic_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    inference_root = tmp_path / "inference"
    clip = "aspset510_fake_0001_left"
    labels_dir = run_dir / "labels" / clip
    labels_dir.mkdir(parents=True)
    inf_dir = inference_root / clip
    inf_dir.mkdir(parents=True)

    frame_indices = [0, 10, 20]
    rng = np.random.default_rng(0)
    gt_frames = []
    smpl_frames = []
    for frame_index in frame_indices:
        gt_joints = rng.normal(scale=0.3, size=(len(SHARED_CORE_JOINT_NAMES), 3))
        gt_frames.append(
            {
                "accepted": True,
                "frame_index": frame_index,
                "player_id": 1,
                "joint_names": list(SHARED_CORE_JOINT_NAMES),
                "joints_world": gt_joints.tolist(),
            }
        )
        raw_mhr = np.zeros((70, 3))
        for name, gt_row in zip(SHARED_CORE_JOINT_NAMES, gt_joints):
            raw_mhr[MHR70_JOINT_NAMES.index(name)] = gt_row + 0.01  # small, known offset
        smpl_frames.append({"frame_idx": frame_index, "joints_world": raw_mhr.tolist()})

    (labels_dir / "body_world_joints.json").write_text(
        json.dumps(
            {
                "joint_names": list(SHARED_CORE_JOINT_NAMES),
                "provenance": {"subject_id": "fake"},
                "samples": gt_frames,
            }
        ),
        encoding="utf-8",
    )
    (inf_dir / "smpl_motion.json").write_text(
        json.dumps({"players": [{"id": 1, "frames": smpl_frames}]}), encoding="utf-8"
    )

    result = scoring.score_one_clip(clip=clip, run_dir=run_dir, inference_root=inference_root)
    assert result["matched_frame_count"] == 3
    assert result["unmatched_gt_frame_indices"] == []
    assert result["scored"]["variants"]["mpjpe"]["value_m"] == pytest.approx(0.01 * (3**0.5), abs=1e-6)
    assert result["gate_threshold_m"] == pytest.approx(0.05)
    assert set(result["per_joint_breakdown_m"].keys()) == set(SHARED_CORE_JOINT_NAMES)


def test_pooled_variants_is_frame_count_weighted() -> None:
    clip_results = [
        {"matched_frame_count": 2, "scored": {"variants": {"mpjpe": {"value_m": 0.10}}}},
        {"matched_frame_count": 8, "scored": {"variants": {"mpjpe": {"value_m": 0.02}}}},
    ]
    pooled = scoring.pooled_variants(
        [
            {**r, "scored": {"variants": {name: r["scored"]["variants"].get(name, {"value_m": 0.0}) for name in scoring.VARIANT_NAMES}}}
            for r in clip_results
        ]
    )
    expected = (0.10 * 2 + 0.02 * 8) / 10
    assert pooled["mpjpe"] == pytest.approx(expected)
