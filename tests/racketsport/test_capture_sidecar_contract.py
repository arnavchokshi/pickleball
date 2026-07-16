from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from threed.racketsport.schemas import CaptureSidecar, validate_artifact_file


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "capture_sidecar"
FIXTURE_PATHS = tuple(sorted(FIXTURE_DIR.glob("*.json")))


@pytest.mark.parametrize("fixture_path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_swift_capture_sidecar_golden_fixture_validates(fixture_path: Path) -> None:
    sidecar = validate_artifact_file("capture_sidecar", fixture_path)

    assert isinstance(sidecar, CaptureSidecar)


def test_contract_fixture_set_covers_live_missing_and_camera_roll_paths() -> None:
    assert {path.name for path in FIXTURE_PATHS} == {
        "camera_roll_import.json",
        "full_sensors.json",
        "missing_sensors.json",
    }

    camera_roll = validate_artifact_file("capture_sidecar", FIXTURE_DIR / "camera_roll_import.json")
    assert isinstance(camera_roll, CaptureSidecar)
    assert camera_roll.provenance == "camera_roll_import"
    assert camera_roll.locked is None
    assert camera_roll.intrinsics is None
    assert camera_roll.gravity is None


def test_capture_sidecar_still_forbids_unknown_top_level_keys(tmp_path: Path) -> None:
    payload = json.loads((FIXTURE_DIR / "full_sensors.json").read_text(encoding="utf-8"))
    payload["unexpected_contract_key"] = True
    invalid_path = tmp_path / "unknown-key-sidecar.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="unexpected_contract_key"):
        validate_artifact_file("capture_sidecar", invalid_path)


def test_historical_python_sidecar_without_provenance_remains_valid(tmp_path: Path) -> None:
    payload = json.loads((FIXTURE_DIR / "full_sensors.json").read_text(encoding="utf-8"))
    payload.pop("provenance")
    historical_path = tmp_path / "historical-sidecar.json"
    historical_path.write_text(json.dumps(payload), encoding="utf-8")

    sidecar = validate_artifact_file("capture_sidecar", historical_path)

    assert isinstance(sidecar, CaptureSidecar)
    assert sidecar.provenance is None


def test_new_crop_and_rolling_shutter_fields_are_optional_with_typed_absence() -> None:
    for fixture_path in FIXTURE_PATHS:
        sidecar = validate_artifact_file("capture_sidecar", fixture_path)
        assert isinstance(sidecar, CaptureSidecar)
        assert sidecar.reference_crop is None
        assert sidecar.rolling_shutter is None


def test_crop_and_rolling_shutter_declarations_are_strict_and_typed() -> None:
    payload = json.loads((FIXTURE_DIR / "missing_sensors.json").read_text(encoding="utf-8"))
    payload["reference_crop"] = {
        "x_px": 64.0,
        "y_px": 24.0,
        "width_px": 1792.0,
        "height_px": 1032.0,
    }
    payload["rolling_shutter"] = {
        "frame_readout_s": 0.008,
        "direction": "top_to_bottom",
    }

    sidecar = CaptureSidecar.model_validate(payload)

    assert sidecar.reference_crop is not None
    assert sidecar.reference_crop.width_px == 1792.0
    assert sidecar.rolling_shutter is not None
    assert sidecar.rolling_shutter.frame_readout_s == 0.008

    payload["rolling_shutter"]["unknown"] = True
    with pytest.raises(ValidationError, match="unknown"):
        CaptureSidecar.model_validate(payload)


@pytest.mark.parametrize("bad_readout", [0.0, -0.001, True, float("nan")])
def test_rolling_shutter_readout_rejects_nonpositive_or_nonfinite_values(bad_readout: object) -> None:
    payload = json.loads((FIXTURE_DIR / "missing_sensors.json").read_text(encoding="utf-8"))
    payload["rolling_shutter"] = {
        "frame_readout_s": bad_readout,
        "direction": "bottom_to_top",
    }
    with pytest.raises(ValidationError, match="frame_readout_s"):
        CaptureSidecar.model_validate(payload)
