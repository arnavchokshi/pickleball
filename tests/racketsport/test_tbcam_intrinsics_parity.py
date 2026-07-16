from __future__ import annotations

from pathlib import Path

from threed.racketsport.court_auto_evidence import calibration_for_image_size
from threed.racketsport.court_calibration import calibration_image_size
from threed.racketsport.sam3d_body_input_prep import static_camera_intrinsics_k
from threed.racketsport.schemas import CourtCalibration


ROOT = Path(__file__).resolve().parents[2]
WOLVERINE_CALIBRATION = (
    ROOT
    / "eval_clips"
    / "ball"
    / "wolverine_mixed_0200_mid_steep_corner"
    / "labels"
    / "court_calibration_metric15pt.json"
)


def _wolverine_calibration() -> CourtCalibration:
    return CourtCalibration.model_validate_json(WOLVERINE_CALIBRATION.read_text(encoding="utf-8"))


def _clean_float(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if abs(rounded) < 1e-9 else rounded


def test_static_body_intrinsics_typed_scaler_is_byte_value_identical_to_legacy_formula() -> None:
    calibration = _wolverine_calibration()
    width, height = 960.0, 540.0
    base_width, base_height = calibration_image_size(
        calibration,
        fallback_target=(width, height),
    )
    scale_x = width / base_width
    scale_y = height / base_height
    intrinsics = calibration.intrinsics
    legacy = [
        [_clean_float(float(intrinsics.fx) * scale_x), 0.0, _clean_float(float(intrinsics.cx) * scale_x)],
        [0.0, _clean_float(float(intrinsics.fy) * scale_y), _clean_float(float(intrinsics.cy) * scale_y)],
        [0.0, 0.0, 1.0],
    ]

    assert static_camera_intrinsics_k(calibration, image_size_px=(960, 540)) == legacy


def test_court_evidence_typed_scaler_is_model_dump_identical_to_legacy_formula() -> None:
    calibration = _wolverine_calibration()
    width, height = 1280, 720
    base_width, base_height = calibration_image_size(
        calibration,
        fallback_target=(float(width), float(height)),
    )
    scale_x = float(width) / base_width
    scale_y = float(height) / base_height
    homography = [
        [float(value) * scale_x for value in calibration.homography[0]],
        [float(value) * scale_y for value in calibration.homography[1]],
        [float(value) for value in calibration.homography[2]],
    ]
    intrinsics = calibration.intrinsics.model_copy(
        update={
            "fx": float(calibration.intrinsics.fx) * scale_x,
            "fy": float(calibration.intrinsics.fy) * scale_y,
            "cx": float(calibration.intrinsics.cx) * scale_x,
            "cy": float(calibration.intrinsics.cy) * scale_y,
        }
    )
    reprojection_scale = (abs(scale_x) + abs(scale_y)) / 2.0
    reprojection_error = calibration.reprojection_error_px.model_copy(
        update={
            "median": float(calibration.reprojection_error_px.median) * reprojection_scale,
            "p95": float(calibration.reprojection_error_px.p95) * reprojection_scale,
        }
    )
    legacy = calibration.model_copy(
        deep=True,
        update={
            "homography": homography,
            "intrinsics": intrinsics,
            "reprojection_error_px": reprojection_error,
            "image_size": (width, height),
            "image_pts": [
                [float(point[0]) * scale_x, float(point[1]) * scale_y]
                for point in calibration.image_pts
            ],
        },
    )

    typed = calibration_for_image_size(calibration, width=width, height=height)

    assert typed.model_dump(mode="json") == legacy.model_dump(mode="json")
