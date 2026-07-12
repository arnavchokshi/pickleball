from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import json
import math
from pathlib import Path
import random

import pytest

from tests.racketsport.json_schema_assertions import assert_matches_json_schema
from threed.racketsport import timebase
from threed.racketsport.timebase import (
    AcousticPropagationModel,
    AlignmentMethod,
    AlignmentMissingReason,
    AlignmentStatus,
    AudioTime,
    ClockDomain,
    CorrectedPTS,
    CorrectionProvenance,
    FrameAbsenceReason,
    FrameAvailability,
    FrameAvailabilityStatus,
    FrameTime,
    NumericSensorSample,
    RawEncodedPTS,
    RollingShutterDirection,
    RollingShutterModel,
    SensorClockMapping,
    TimeBasis,
    TimebaseContract,
    TimebaseValidationError,
    align_interpolated_sensor_sample,
    align_nearest_sensor_sample,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs" / "racketsport" / "timebase_schema.json"


def _correction() -> CorrectionProvenance:
    return CorrectionProvenance(
        method="av_mux_offset_and_drift",
        source_clock=ClockDomain.MEDIA,
        target_clock=ClockDomain.MEDIA,
        model_id="mux-fit-1",
        offset_s=0.002,
        drift_ppm=1.25,
        uncertainty_s=0.0005,
    )


def _present(frame_index: int, ticks: int, *, corrected_time_s: float | None = None) -> FrameTime:
    return FrameTime(
        frame_index=frame_index,
        availability=FrameAvailability(FrameAvailabilityStatus.PRESENT),
        raw_encoded_pts=RawEncodedPTS(
            frame_index=frame_index,
            pts_ticks=ticks,
            timescale=90_000,
            duration_ticks=None,
        ),
        corrected_pts=(
            None
            if corrected_time_s is None
            else CorrectedPTS(corrected_time_s=corrected_time_s, provenance=_correction())
        ),
    )


def _clock_mapping() -> SensorClockMapping:
    return SensorClockMapping(
        mapping_id="motion-monotonic-to-media",
        source_clock=ClockDomain.MONOTONIC,
        target_clock=ClockDomain.MEDIA,
        reference_source_time_s=1_000.0,
        offset_s=-999.99,
        drift_ppm=12.5,
        declared_tolerance_s=1e-8,
    )


def _contract() -> TimebaseContract:
    acoustic = AcousticPropagationModel(
        model_id="court-to-phone-1",
        source_to_microphone_distance_m=10.29,
        speed_of_sound_mps=343.0,
        air_temperature_c=20.0,
        relative_humidity_fraction=0.45,
        air_pressure_kpa=101.325,
        distance_uncertainty_m=0.20,
    )
    return TimebaseContract(
        capture_id="vfr-drop-fixture",
        frames=(
            _present(0, 0, corrected_time_s=0.002),
            FrameTime(
                frame_index=1,
                availability=FrameAvailability(
                    FrameAvailabilityStatus.DROPPED,
                    FrameAbsenceReason.LATE_FRAME_DISCARDED,
                ),
                raw_encoded_pts=None,
            ),
            _present(2, 3_690, corrected_time_s=0.043),
            FrameTime(
                frame_index=3,
                availability=FrameAvailability(
                    FrameAvailabilityStatus.MISSING,
                    FrameAbsenceReason.SIDECAR_GAP,
                ),
                raw_encoded_pts=None,
            ),
            _present(4, 9_810, corrected_time_s=0.111),
        ),
        audio_times=(
            acoustic.correct_audio_time("pop-1", 2.03, raw_clock=ClockDomain.AUDIO),
            AudioTime("uncorrected-review-cue", 3.5, ClockDomain.AUDIO),
        ),
        acoustic_models=(acoustic,),
        sensor_clock_mappings=(_clock_mapping(),),
        rolling_shutter_model=RollingShutterModel(
            model_id="iphone-readout-profile",
            frame_readout_s=0.008,
            direction=RollingShutterDirection.TOP_TO_BOTTOM,
        ),
    )


def test_monotonic_vfr_fixture_preserves_exact_pts_and_explicit_absence() -> None:
    contract = _contract()
    observed = [frame.raw_encoded_pts.seconds for frame in contract.frames if frame.raw_encoded_pts]

    assert observed == [0.0, 0.041, 0.109]
    assert [round(b - a, 6) for a, b in zip(observed, observed[1:])] == [0.041, 0.068]
    assert contract.vfr_policy == "encoded_pts_required"
    assert contract.frames[1].availability == FrameAvailability(
        FrameAvailabilityStatus.DROPPED,
        FrameAbsenceReason.LATE_FRAME_DISCARDED,
    )
    assert contract.frames[3].availability.reason is FrameAbsenceReason.SIDECAR_GAP
    assert contract.frames[1].raw_encoded_pts is None
    assert contract.frames[3].corrected_pts is None


def test_contract_rejects_non_monotonic_or_duplicate_raw_pts() -> None:
    with pytest.raises(TimebaseValidationError, match="strictly monotonic"):
        TimebaseContract(
            capture_id="bad-order",
            frames=(_present(0, 100), _present(1, 100)),
            audio_times=(),
            acoustic_models=(),
            sensor_clock_mappings=(),
            rolling_shutter_model=None,
        )


def test_monotonic_check_uses_exact_rationals_not_float_rounding() -> None:
    large = 2**60
    contract = TimebaseContract(
        capture_id="exact-rational-order",
        frames=(
            FrameTime(
                0,
                FrameAvailability(FrameAvailabilityStatus.PRESENT),
                RawEncodedPTS(0, large, 90_000),
            ),
            FrameTime(
                1,
                FrameAvailability(FrameAvailabilityStatus.PRESENT),
                RawEncodedPTS(1, large + 1, 90_000),
            ),
        ),
        audio_times=(),
        acoustic_models=(),
        sensor_clock_mappings=(),
        rolling_shutter_model=None,
    )
    assert contract.frames[1].raw_encoded_pts is not None
    assert contract.frames[1].raw_encoded_pts.pts_ticks == large + 1


def test_missing_and_dropped_frames_require_reasons_and_cannot_invent_pts() -> None:
    with pytest.raises(TimebaseValidationError, match="explicit reason"):
        FrameAvailability(FrameAvailabilityStatus.DROPPED)
    with pytest.raises(TimebaseValidationError, match="cannot contain invented timestamps"):
        FrameTime(
            frame_index=7,
            availability=FrameAvailability(
                FrameAvailabilityStatus.MISSING,
                FrameAbsenceReason.CAPTURE_NOT_EMITTED,
            ),
            raw_encoded_pts=RawEncodedPTS(7, 7_000, 1_000),
        )


def test_raw_observations_are_frozen_and_corrections_are_separate() -> None:
    frame = _contract().frames[0]
    assert frame.raw_encoded_pts is not None
    assert frame.raw_encoded_pts.pts_ticks == 0
    assert frame.corrected_pts is not None
    assert frame.corrected_pts.corrected_time_s == pytest.approx(0.002)

    with pytest.raises(FrozenInstanceError):
        frame.raw_encoded_pts.pts_ticks = 123  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        frame.corrected_pts.corrected_time_s = 9.0  # type: ignore[misc]


def test_round_trip_json_is_schema_validated_and_byte_stable() -> None:
    contract = _contract()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    encoded = contract.to_json_bytes()
    payload = json.loads(encoded)

    assert_matches_json_schema(payload, schema)
    restored = TimebaseContract.from_json_bytes(encoded)
    assert restored == contract
    assert restored.to_json_bytes() == encoded
    assert encoded.endswith(b"\n")


def test_schema_rejects_destructive_or_implicit_cfr_shapes() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = _contract().to_dict()
    payload["fps"] = 30.0
    with pytest.raises(AssertionError, match="unexpected key 'fps'"):
        assert_matches_json_schema(payload, schema)

    payload = _contract().to_dict()
    del payload["frames"][0]["raw_encoded_pts"]
    with pytest.raises(AssertionError, match="raw_encoded_pts"):
        assert_matches_json_schema(payload, schema)


def test_clock_mapping_property_invertible_within_declared_tolerance() -> None:
    rng = random.Random(20260712)
    for _ in range(1_000):
        mapping = SensorClockMapping(
            mapping_id="property-case",
            source_clock=ClockDomain.MONOTONIC,
            target_clock=ClockDomain.MEDIA,
            reference_source_time_s=rng.uniform(-10_000.0, 10_000.0),
            offset_s=rng.uniform(-1_000.0, 1_000.0),
            drift_ppm=rng.uniform(-2_000.0, 2_000.0),
            declared_tolerance_s=1e-8,
        )
        source = rng.uniform(-100_000.0, 100_000.0)
        restored = mapping.to_source(mapping.to_target(source))
        assert abs(restored - source) <= mapping.declared_tolerance_s


@pytest.mark.parametrize(
    ("distance_m", "speed_mps", "expected_delay_s"),
    [(0.0, 343.0, 0.0), (10.29, 343.0, 0.03), (34.0, 340.0, 0.1)],
)
def test_acoustic_delay_is_distance_dependent_and_preserves_raw_audio(
    distance_m: float,
    speed_mps: float,
    expected_delay_s: float,
) -> None:
    model = AcousticPropagationModel(
        model_id="declared-air",
        source_to_microphone_distance_m=distance_m,
        speed_of_sound_mps=speed_mps,
        air_temperature_c=18.5,
        relative_humidity_fraction=0.40,
        air_pressure_kpa=100.8,
        distance_uncertainty_m=0.1,
    )
    corrected = model.correct_audio_time("contact", 1.25, raw_clock=ClockDomain.AUDIO)

    assert model.delay_s == pytest.approx(expected_delay_s)
    assert corrected.raw_time_s == 1.25
    assert corrected.corrected_time_s == pytest.approx(1.25 - expected_delay_s)
    assert corrected.correction is not None
    assert corrected.correction.method == "acoustic_distance_over_declared_speed"
    assert corrected.correction.offset_s == pytest.approx(-expected_delay_s)


def test_rolling_shutter_row_time_respects_direction_and_declared_readout() -> None:
    top_down = RollingShutterModel("top-down", 0.008, RollingShutterDirection.TOP_TO_BOTTOM)
    bottom_up = RollingShutterModel("bottom-up", 0.008, RollingShutterDirection.BOTTOM_TO_TOP)

    assert top_down.row_time(1.0, row_index=0, frame_height=5).row_time_s == 1.0
    assert top_down.row_time(1.0, row_index=4, frame_height=5).row_time_s == pytest.approx(1.008)
    assert bottom_up.row_time(1.0, row_index=0, frame_height=5).row_time_s == pytest.approx(1.008)
    assert bottom_up.row_time(1.0, row_index=4, frame_height=5).row_time_s == 1.0


def test_nearest_alignment_demands_tolerance_and_returns_provenance() -> None:
    frame = _present(0, 900, corrected_time_s=0.012)
    mapping = SensorClockMapping(
        "sensor-media",
        ClockDomain.MONOTONIC,
        ClockDomain.MEDIA,
        reference_source_time_s=10.0,
        offset_s=-10.0,
        drift_ppm=0.0,
        declared_tolerance_s=0.001,
    )
    samples = (
        NumericSensorSample("early", ClockDomain.MONOTONIC, 10.008, (1.0, 2.0)),
        NumericSensorSample("near", ClockDomain.MONOTONIC, 10.011, (3.0, 4.0)),
    )

    with pytest.raises(TypeError, match="tolerance_s"):
        align_nearest_sensor_sample(frame, samples, mapping)  # type: ignore[call-arg]
    result = align_nearest_sensor_sample(
        frame,
        samples,
        mapping,
        tolerance_s=0.002,
        time_basis=TimeBasis.CORRECTED_PTS,
    )
    assert result.status is AlignmentStatus.ALIGNED
    assert result.values == (3.0, 4.0)
    assert result.provenance is not None
    assert result.provenance.method is AlignmentMethod.NEAREST
    assert result.provenance.tolerance_s == 0.002
    assert result.provenance.source_sample_ids == ("near",)
    assert result.provenance.time_basis is TimeBasis.CORRECTED_PTS


def test_nearest_alignment_returns_missing_outside_tolerance() -> None:
    result = align_nearest_sensor_sample(
        _present(0, 900),
        (NumericSensorSample("far", ClockDomain.MONOTONIC, 10.2, (1.0,)),),
        SensorClockMapping(
            "sensor-media",
            ClockDomain.MONOTONIC,
            ClockDomain.MEDIA,
            10.0,
            -10.0,
            0.0,
            0.001,
        ),
        tolerance_s=0.01,
        time_basis=TimeBasis.RAW_ENCODED_PTS,
    )
    assert result == timebase.SensorAlignment(
        AlignmentStatus.MISSING,
        None,
        None,
        AlignmentMissingReason.OUTSIDE_TOLERANCE,
    )


def test_interpolation_demands_bracketing_samples_tolerance_and_provenance() -> None:
    frame = _present(0, 900)
    mapping = SensorClockMapping(
        "sensor-media",
        ClockDomain.MONOTONIC,
        ClockDomain.MEDIA,
        10.0,
        -10.0,
        0.0,
        0.001,
    )
    samples = (
        NumericSensorSample("left", ClockDomain.MONOTONIC, 10.0, (0.0, 2.0)),
        NumericSensorSample("right", ClockDomain.MONOTONIC, 10.02, (2.0, 4.0)),
    )

    result = align_interpolated_sensor_sample(
        frame,
        samples,
        mapping,
        tolerance_s=0.011,
        time_basis=TimeBasis.RAW_ENCODED_PTS,
    )
    assert result.status is AlignmentStatus.ALIGNED
    assert result.values == pytest.approx((1.0, 3.0))
    assert result.provenance is not None
    assert result.provenance.method is AlignmentMethod.LINEAR_INTERPOLATION
    assert result.provenance.source_sample_ids == ("left", "right")


def test_alignment_never_invents_a_sample_for_missing_or_dropped_frame() -> None:
    mapping = _clock_mapping()
    sample = NumericSensorSample("available", ClockDomain.MONOTONIC, 1_000.0, (99.0,))
    for status, reason in (
        (FrameAvailabilityStatus.MISSING, FrameAbsenceReason.SIDECAR_GAP),
        (FrameAvailabilityStatus.DROPPED, FrameAbsenceReason.ENCODER_REJECTED),
    ):
        frame = FrameTime(8, FrameAvailability(status, reason), None)
        result = align_nearest_sensor_sample(
            frame,
            (sample,),
            mapping,
            tolerance_s=1.0,
            time_basis=TimeBasis.RAW_ENCODED_PTS,
        )
        assert result.status is AlignmentStatus.MISSING
        assert result.values is None
        assert result.provenance is None
        assert result.missing_reason is AlignmentMissingReason.FRAME_NOT_PRESENT


def test_public_alignment_api_has_no_latest_sample_or_implicit_tolerance() -> None:
    public_callables = {
        name: member
        for name, member in inspect.getmembers(timebase, callable)
        if not name.startswith("_") and getattr(member, "__module__", None) == timebase.__name__
    }
    assert all("latest" not in name.lower() for name in public_callables)
    for helper in (align_nearest_sensor_sample, align_interpolated_sensor_sample):
        signature = inspect.signature(helper)
        assert signature.parameters["tolerance_s"].default is inspect.Parameter.empty
        assert signature.parameters["time_basis"].default is inspect.Parameter.empty


def test_constructor_rejects_non_finite_time_values() -> None:
    with pytest.raises(TimebaseValidationError, match="finite"):
        AudioTime("bad", math.nan, ClockDomain.AUDIO)
    with pytest.raises(TimebaseValidationError, match="finite"):
        CorrectedPTS(math.inf, _correction())
