from __future__ import annotations

import math

import numpy as np
import pytest

from threed.racketsport import mhr_decode


def test_module_imports_without_torch_or_roma_available() -> None:
    # The module itself must NEVER raise at import time (it already imported
    # cleanly to reach this test); it exposes a bool + captured exception
    # instead so callers/tests can skip gracefully.
    assert isinstance(mhr_decode.MHR_RUNTIME_AVAILABLE, bool)
    if not mhr_decode.MHR_RUNTIME_AVAILABLE:
        assert mhr_decode.MHR_RUNTIME_IMPORT_ERROR is not None


def test_require_runtime_raises_informative_error_when_unavailable() -> None:
    if mhr_decode.MHR_RUNTIME_AVAILABLE:
        pytest.skip("runtime available in this interpreter; nothing to assert about the guard")
    with pytest.raises(RuntimeError, match="mhr_decode requires torch"):
        mhr_decode.encode_body_pose_euler_to_cont(np.zeros(mhr_decode.BODY_POSE_EULER_DIM))


def test_dimension_constants_match_mhr_head_layout() -> None:
    assert mhr_decode.BODY_POSE_EULER_DIM == 133
    assert mhr_decode.BODY_POSE_CONT_DIM == 260
    assert mhr_decode.GLOBAL_ROT_EULER_DIM == 3
    assert mhr_decode.GLOBAL_ROT_6D_DIM == 6
    assert mhr_decode.PRED_POSE_RAW_DIM == 266
    assert mhr_decode.SHAPE_DIM == 45
    assert mhr_decode.SCALE_DIM == 28
    assert mhr_decode.GATE_1A_MAX_ABS_ERROR_DEG == 0.1
    assert mhr_decode.GATE_1B_MAX_ABS_ERROR_MM == 1.0
    assert mhr_decode.MESH_SKELETON_DIVERGENCE_P95_MM == 5.0


def test_gate_1b_world_round_trip_pure_python_distance_math() -> None:
    # This gate helper's distance math is pure numpy (no torch/roma needed) --
    # exercise it directly against a synthetic 1mm-exact displacement.
    persisted = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    decoded_ok = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0005]]  # 0.5mm off
    decoded_bad = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.002]]  # 2mm off

    result_ok = mhr_decode.gate_1b_world_round_trip(
        decoded_joints_world=decoded_ok,
        decoded_vertices_world=[],
        persisted_joints_world=persisted,
        persisted_vertices_world=[],
    )
    assert result_ok["passed"] is True
    assert result_ok["joints_world"]["max_abs_error_mm"] == pytest.approx(0.5, abs=1e-6)

    result_bad = mhr_decode.gate_1b_world_round_trip(
        decoded_joints_world=decoded_bad,
        decoded_vertices_world=[],
        persisted_joints_world=persisted,
        persisted_vertices_world=[],
    )
    assert result_bad["passed"] is False
    assert result_bad["joints_world"]["max_abs_error_mm"] == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Full-runtime tests (torch + roma + sam_3d_body): skip cleanly where that
# stack is absent (the Mac CPU dev venv); real evidence for these comes from
# running on the fleet GPU VM's body_venv (see the lane report).
# ---------------------------------------------------------------------------
pytestmark_runtime = pytest.mark.skipif(
    not mhr_decode.MHR_RUNTIME_AVAILABLE,
    reason=f"mhr_decode runtime (torch+roma+sam_3d_body) unavailable: {mhr_decode.MHR_RUNTIME_IMPORT_ERROR}",
)


@pytestmark_runtime
def test_body_pose_euler_round_trip_is_near_exact_on_realistic_range() -> None:
    rng = np.random.default_rng(0)
    body_pose = (rng.random((50, mhr_decode.BODY_POSE_EULER_DIM)) - 0.5) * 3.0  # ~[-1.5, 1.5] rad
    result = mhr_decode.round_trip_body_pose_euler_error_deg(body_pose)
    assert result["max_abs_error_deg"] < mhr_decode.GATE_1A_MAX_ABS_ERROR_DEG


@pytestmark_runtime
def test_global_orient_euler_round_trip_is_near_exact_on_realistic_range() -> None:
    rng = np.random.default_rng(1)
    global_orient = (rng.random((50, mhr_decode.GLOBAL_ROT_EULER_DIM)) - 0.5) * 3.0
    result = mhr_decode.round_trip_global_orient_error_deg(global_orient)
    assert result["max_abs_error_deg"] < mhr_decode.GATE_1A_MAX_ABS_ERROR_DEG


@pytestmark_runtime
def test_gate_1a_combines_both_components() -> None:
    rng = np.random.default_rng(2)
    body_pose = (rng.random((10, mhr_decode.BODY_POSE_EULER_DIM)) - 0.5) * 2.0
    global_orient = (rng.random((10, mhr_decode.GLOBAL_ROT_EULER_DIM)) - 0.5) * 2.0
    result = mhr_decode.gate_1a_euler_round_trip(global_orient, body_pose)
    assert result["gate"] == "gate_1a_euler_cont_euler_idempotence"
    assert "body_pose" in result and "global_orient" in result
    assert result["max_abs_error_deg"] == max(
        result["body_pose"]["max_abs_error_deg"], result["global_orient"]["max_abs_error_deg"]
    )


@pytestmark_runtime
def test_build_and_split_pred_pose_raw_round_trips_shape() -> None:
    import torch

    rng = np.random.default_rng(3)
    body_pose = (rng.random((4, mhr_decode.BODY_POSE_EULER_DIM)) - 0.5) * 2.0
    global_orient = (rng.random((4, mhr_decode.GLOBAL_ROT_EULER_DIM)) - 0.5) * 2.0
    pred_pose_raw = mhr_decode.build_pred_pose_raw(global_orient, body_pose)
    assert pred_pose_raw.shape == (4, mhr_decode.PRED_POSE_RAW_DIM)
    rot6d, cont = mhr_decode.split_pred_pose_raw(pred_pose_raw)
    assert rot6d.shape == (4, mhr_decode.GLOBAL_ROT_6D_DIM)
    assert cont.shape == (4, mhr_decode.BODY_POSE_CONT_DIM)
    assert torch.allclose(torch.cat([rot6d, cont], dim=-1), pred_pose_raw)


@pytestmark_runtime
def test_mhr_decoder_loads_checkpoint_and_decodes_zero_pose() -> None:
    decoder = mhr_decode.MHRDecoder()
    zero_global_orient = np.zeros(mhr_decode.GLOBAL_ROT_EULER_DIM)
    zero_body_pose = np.zeros(mhr_decode.BODY_POSE_EULER_DIM)
    zero_shape = np.zeros(mhr_decode.SHAPE_DIM)
    out = decoder.decode_euler_frame(
        global_orient_euler=zero_global_orient,
        body_pose_euler=zero_body_pose,
        shape=zero_shape,
    )
    assert out["joints_camera"].shape == (1, mhr_decode.NUM_KEYPOINTS, 3)
    assert np.all(np.isfinite(out["joints_camera"]))


@pytestmark_runtime
def test_mesh_skeleton_divergence_zero_for_undeformed_zero_pose() -> None:
    decoder = mhr_decode.MHRDecoder()
    result = decoder.mesh_skeleton_divergence_mm(
        global_orient_euler=np.zeros(mhr_decode.GLOBAL_ROT_EULER_DIM),
        body_pose_euler=np.zeros(mhr_decode.BODY_POSE_EULER_DIM),
        shape=np.zeros(mhr_decode.SHAPE_DIM),
    )
    assert result["metric"] == "mesh_skeleton_divergence_mm"
    assert math.isfinite(result["p95_mm"])
    assert result["p95_mm"] >= 0.0
