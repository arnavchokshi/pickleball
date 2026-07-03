import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ane_960_diagnostics import (
    postprocess_tensor_shapes,
    project_partial_loop_ms,
    torch_trace_kwargs,
    yolo_grid_points,
)


def test_yolo_grid_points_match_detector_scales():
    assert yolo_grid_points(640) == 8400
    assert yolo_grid_points(960) == 18900


def test_postprocess_shapes_capture_960_topk_risk():
    shapes = postprocess_tensor_shapes(960)

    assert shapes["raw_predictions"] == [1, 84, 18900]
    assert shapes["first_topk_input"] == [1, 18900]
    assert shapes["second_topk_input"] == [1, 24000]
    assert shapes["final_output"] == [1, 300, 6]


def test_project_partial_loop_ms_supports_tiled_640_comparison():
    projected = project_partial_loop_ms(detector_ms=3.185, ball_ms=1.407, detector_runs=4)

    assert projected["mean_ms"] == 14.147
    assert projected["fps"] == 70.69


def test_torch_trace_disables_redundant_check_for_ultralytics_head_cache():
    assert torch_trace_kwargs() == {"strict": False, "check_trace": False}
