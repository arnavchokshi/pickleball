"""Immutable, VFR-first timing contracts for racket-sport capture evidence.

This module is deliberately pure: it does not decode media, read capture files,
or choose a pipeline clock.  Encoded presentation timestamps remain exact raw
observations.  Corrections and alignments are separate values with provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_timebase_contract"


class TimebaseValidationError(ValueError):
    """Raised when a typed timebase value violates the contract."""


class ClockDomain(str, Enum):
    MONOTONIC = "monotonic"
    WALL = "wall"
    MEDIA = "media"
    AUDIO = "audio"


class FrameAvailabilityStatus(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    DROPPED = "dropped"


class FrameAbsenceReason(str, Enum):
    CAPTURE_NOT_EMITTED = "capture_not_emitted"
    ENCODER_REJECTED = "encoder_rejected"
    LATE_FRAME_DISCARDED = "late_frame_discarded"
    DECODE_FAILED = "decode_failed"
    PTS_UNAVAILABLE = "pts_unavailable"
    SIDECAR_GAP = "sidecar_gap"
    EXPLICITLY_UNAVAILABLE = "explicitly_unavailable"


class TimeBasis(str, Enum):
    RAW_ENCODED_PTS = "raw_encoded_pts"
    CORRECTED_PTS = "corrected_pts"


class AlignmentMethod(str, Enum):
    NEAREST = "nearest"
    LINEAR_INTERPOLATION = "linear_interpolation"


class AlignmentStatus(str, Enum):
    ALIGNED = "aligned"
    MISSING = "missing"


class AlignmentMissingReason(str, Enum):
    FRAME_NOT_PRESENT = "frame_not_present"
    NO_SENSOR_SAMPLES = "no_sensor_samples"
    OUTSIDE_TOLERANCE = "outside_tolerance"
    NO_BRACKETING_SAMPLES = "no_bracketing_samples"


class SensorClockMappingStatus(str, Enum):
    MAPPED = "mapped"
    MISSING = "missing"


class SensorClockMappingMissingReason(str, Enum):
    SIDECAR_UNAVAILABLE = "sidecar_unavailable"
    SIDECAR_DECLARES_UNAVAILABLE = "sidecar_declares_unavailable"
    NO_DUAL_CLOCK_PAIRS = "no_dual_clock_pairs"
    INSUFFICIENT_DUAL_CLOCK_PAIRS = "insufficient_dual_clock_pairs"
    DEGENERATE_SOURCE_CLOCK = "degenerate_source_clock"


class RollingShutterDirection(str, Enum):
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTTOM_TO_TOP = "bottom_to_top"


def _finite(value: float, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise TimebaseValidationError(f"{field} must be finite")
    return parsed


def _non_negative(value: float, field: str) -> float:
    parsed = _finite(value, field)
    if parsed < 0.0:
        raise TimebaseValidationError(f"{field} must be non-negative")
    return parsed


def _positive(value: float, field: str) -> float:
    parsed = _finite(value, field)
    if parsed <= 0.0:
        raise TimebaseValidationError(f"{field} must be positive")
    return parsed


def _non_empty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TimebaseValidationError(f"{field} must be a non-empty string")
    return value


def _enum(enum_type: type[Enum], value: Any, field: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise TimebaseValidationError(f"invalid {field}: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class RawEncodedPTS:
    """An exact encoded PTS observation expressed as integer ticks."""

    frame_index: int
    pts_ticks: int
    timescale: int
    duration_ticks: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise TimebaseValidationError("frame_index must be a non-negative integer")
        if isinstance(self.pts_ticks, bool) or not isinstance(self.pts_ticks, int):
            raise TimebaseValidationError("pts_ticks must be an integer")
        if isinstance(self.timescale, bool) or not isinstance(self.timescale, int) or self.timescale <= 0:
            raise TimebaseValidationError("timescale must be a positive integer")
        if self.duration_ticks is not None and (
            isinstance(self.duration_ticks, bool)
            or not isinstance(self.duration_ticks, int)
            or self.duration_ticks <= 0
        ):
            raise TimebaseValidationError("duration_ticks must be a positive integer when present")

    @property
    def seconds(self) -> float:
        return self.pts_ticks / self.timescale


@dataclass(frozen=True, slots=True)
class CorrectionProvenance:
    """Declared method and uncertainty for a derived media-clock time."""

    method: str
    source_clock: ClockDomain
    target_clock: ClockDomain
    model_id: str
    offset_s: float
    drift_ppm: float
    uncertainty_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _non_empty(self.method, "method"))
        object.__setattr__(self, "source_clock", _enum(ClockDomain, self.source_clock, "source_clock"))
        object.__setattr__(self, "target_clock", _enum(ClockDomain, self.target_clock, "target_clock"))
        object.__setattr__(self, "model_id", _non_empty(self.model_id, "model_id"))
        object.__setattr__(self, "offset_s", _finite(self.offset_s, "offset_s"))
        object.__setattr__(self, "drift_ppm", _finite(self.drift_ppm, "drift_ppm"))
        object.__setattr__(self, "uncertainty_s", _non_negative(self.uncertainty_s, "uncertainty_s"))


@dataclass(frozen=True, slots=True)
class CorrectedPTS:
    """A derived PTS; the authoritative raw observation lives beside it."""

    corrected_time_s: float
    provenance: CorrectionProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "corrected_time_s", _finite(self.corrected_time_s, "corrected_time_s"))
        if not isinstance(self.provenance, CorrectionProvenance):
            raise TimebaseValidationError("provenance must be CorrectionProvenance")


@dataclass(frozen=True, slots=True)
class FrameAvailability:
    status: FrameAvailabilityStatus
    reason: FrameAbsenceReason | None = None

    def __post_init__(self) -> None:
        status = _enum(FrameAvailabilityStatus, self.status, "frame availability status")
        reason = None if self.reason is None else _enum(FrameAbsenceReason, self.reason, "frame absence reason")
        if status is FrameAvailabilityStatus.PRESENT and reason is not None:
            raise TimebaseValidationError("present frames cannot have an absence reason")
        if status is not FrameAvailabilityStatus.PRESENT and reason is None:
            raise TimebaseValidationError("missing and dropped frames require an explicit reason")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class FrameTime:
    """Per-frame availability plus raw and optional corrected PTS."""

    frame_index: int
    availability: FrameAvailability
    raw_encoded_pts: RawEncodedPTS | None
    corrected_pts: CorrectedPTS | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise TimebaseValidationError("frame_index must be a non-negative integer")
        if not isinstance(self.availability, FrameAvailability):
            raise TimebaseValidationError("availability must be FrameAvailability")
        present = self.availability.status is FrameAvailabilityStatus.PRESENT
        if present and self.raw_encoded_pts is None:
            raise TimebaseValidationError("present frames require raw_encoded_pts")
        if not present and (self.raw_encoded_pts is not None or self.corrected_pts is not None):
            raise TimebaseValidationError("missing/dropped frames cannot contain invented timestamps")
        if self.raw_encoded_pts is not None and self.raw_encoded_pts.frame_index != self.frame_index:
            raise TimebaseValidationError("raw_encoded_pts.frame_index must match frame_index")

    def time_s(self, basis: TimeBasis) -> float:
        basis = _enum(TimeBasis, basis, "time basis")
        if self.availability.status is not FrameAvailabilityStatus.PRESENT:
            raise TimebaseValidationError("missing/dropped frames have no alignment time")
        if basis is TimeBasis.RAW_ENCODED_PTS:
            assert self.raw_encoded_pts is not None
            return self.raw_encoded_pts.seconds
        if self.corrected_pts is None:
            raise TimebaseValidationError("corrected_pts requested but unavailable")
        return self.corrected_pts.corrected_time_s


@dataclass(frozen=True, slots=True)
class AudioTime:
    """Raw audio-clock observation with an optional non-destructive correction."""

    event_id: str
    raw_time_s: float
    raw_clock: ClockDomain
    corrected_time_s: float | None = None
    correction: CorrectionProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _non_empty(self.event_id, "event_id"))
        object.__setattr__(self, "raw_time_s", _finite(self.raw_time_s, "raw_time_s"))
        object.__setattr__(self, "raw_clock", _enum(ClockDomain, self.raw_clock, "raw_clock"))
        if (self.corrected_time_s is None) != (self.correction is None):
            raise TimebaseValidationError("corrected_time_s and correction must be present together")
        if self.corrected_time_s is not None:
            object.__setattr__(self, "corrected_time_s", _finite(self.corrected_time_s, "corrected_time_s"))
            if self.correction is None or self.correction.source_clock is not self.raw_clock:
                raise TimebaseValidationError("audio correction source_clock must match raw_clock")


@dataclass(frozen=True, slots=True)
class AcousticPropagationModel:
    """Distance-dependent sound delay with explicit atmospheric assumptions."""

    model_id: str
    source_to_microphone_distance_m: float
    speed_of_sound_mps: float
    air_temperature_c: float
    relative_humidity_fraction: float
    air_pressure_kpa: float
    distance_uncertainty_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _non_empty(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "source_to_microphone_distance_m",
            _non_negative(self.source_to_microphone_distance_m, "source_to_microphone_distance_m"),
        )
        object.__setattr__(self, "speed_of_sound_mps", _positive(self.speed_of_sound_mps, "speed_of_sound_mps"))
        object.__setattr__(self, "air_temperature_c", _finite(self.air_temperature_c, "air_temperature_c"))
        humidity = _finite(self.relative_humidity_fraction, "relative_humidity_fraction")
        if not 0.0 <= humidity <= 1.0:
            raise TimebaseValidationError("relative_humidity_fraction must be between 0 and 1")
        object.__setattr__(self, "relative_humidity_fraction", humidity)
        object.__setattr__(self, "air_pressure_kpa", _positive(self.air_pressure_kpa, "air_pressure_kpa"))
        object.__setattr__(
            self, "distance_uncertainty_m", _non_negative(self.distance_uncertainty_m, "distance_uncertainty_m")
        )

    @property
    def delay_s(self) -> float:
        return self.source_to_microphone_distance_m / self.speed_of_sound_mps

    def correct_audio_time(self, event_id: str, raw_time_s: float, *, raw_clock: ClockDomain) -> AudioTime:
        raw_time_s = _finite(raw_time_s, "raw_time_s")
        uncertainty_s = self.distance_uncertainty_m / self.speed_of_sound_mps
        provenance = CorrectionProvenance(
            method="acoustic_distance_over_declared_speed",
            source_clock=raw_clock,
            target_clock=ClockDomain.MEDIA,
            model_id=self.model_id,
            offset_s=-self.delay_s,
            drift_ppm=0.0,
            uncertainty_s=uncertainty_s,
        )
        return AudioTime(
            event_id=event_id,
            raw_time_s=raw_time_s,
            raw_clock=raw_clock,
            corrected_time_s=raw_time_s - self.delay_s,
            correction=provenance,
        )


@dataclass(frozen=True, slots=True)
class SensorClockMapping:
    """Affine source-to-target clock mapping with explicit offset and drift."""

    mapping_id: str
    source_clock: ClockDomain
    target_clock: ClockDomain
    reference_source_time_s: float
    offset_s: float
    drift_ppm: float
    declared_tolerance_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping_id", _non_empty(self.mapping_id, "mapping_id"))
        object.__setattr__(self, "source_clock", _enum(ClockDomain, self.source_clock, "source_clock"))
        object.__setattr__(self, "target_clock", _enum(ClockDomain, self.target_clock, "target_clock"))
        if self.source_clock is self.target_clock:
            raise TimebaseValidationError("source_clock and target_clock must differ")
        object.__setattr__(
            self, "reference_source_time_s", _finite(self.reference_source_time_s, "reference_source_time_s")
        )
        object.__setattr__(self, "offset_s", _finite(self.offset_s, "offset_s"))
        object.__setattr__(self, "drift_ppm", _finite(self.drift_ppm, "drift_ppm"))
        if 1.0 + self.drift_ppm * 1e-6 <= 0.0:
            raise TimebaseValidationError("drift_ppm makes the clock mapping non-invertible")
        object.__setattr__(
            self, "declared_tolerance_s", _positive(self.declared_tolerance_s, "declared_tolerance_s")
        )

    @property
    def scale(self) -> float:
        return 1.0 + self.drift_ppm * 1e-6

    def to_target(self, source_time_s: float) -> float:
        source_time_s = _finite(source_time_s, "source_time_s")
        return self.reference_source_time_s + self.offset_s + (
            source_time_s - self.reference_source_time_s
        ) * self.scale

    def to_source(self, target_time_s: float) -> float:
        target_time_s = _finite(target_time_s, "target_time_s")
        return self.reference_source_time_s + (
            target_time_s - self.reference_source_time_s - self.offset_s
        ) / self.scale


@dataclass(frozen=True, slots=True)
class SensorClockMappingOutcome:
    """Typed result of adapting capture-sidecar clocks into the core contract."""

    status: SensorClockMappingStatus
    mapping: SensorClockMapping | None
    missing_reason: SensorClockMappingMissingReason | None
    dual_clock_pair_count: int
    max_abs_residual_s: float | None
    sidecar_unavailable_reasons: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        status = _enum(SensorClockMappingStatus, self.status, "sensor clock mapping status")
        reason = None if self.missing_reason is None else _enum(
            SensorClockMappingMissingReason,
            self.missing_reason,
            "sensor clock mapping missing reason",
        )
        if isinstance(self.dual_clock_pair_count, bool) or not isinstance(self.dual_clock_pair_count, int):
            raise TimebaseValidationError("dual_clock_pair_count must be an integer")
        if self.dual_clock_pair_count < 0:
            raise TimebaseValidationError("dual_clock_pair_count must be non-negative")
        if status is SensorClockMappingStatus.MAPPED:
            if self.mapping is None or reason is not None or self.max_abs_residual_s is None:
                raise TimebaseValidationError("mapped clock outcome requires mapping/residual and no missing reason")
        elif self.mapping is not None or reason is None or self.max_abs_residual_s is not None:
            raise TimebaseValidationError("missing clock outcome requires only an explicit missing reason")
        unavailable = tuple(
            (
                _non_empty(name, "unavailable sensor name"),
                _non_empty(value, "unavailable sensor reason"),
            )
            for name, value in self.sidecar_unavailable_reasons
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "missing_reason", reason)
        object.__setattr__(self, "sidecar_unavailable_reasons", unavailable)
        if self.max_abs_residual_s is not None:
            object.__setattr__(
                self,
                "max_abs_residual_s",
                _non_negative(self.max_abs_residual_s, "max_abs_residual_s"),
            )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def build_sensor_clock_mapping_from_sidecar(
    sidecar: Mapping[str, Any] | None,
    *,
    mapping_id: str = "arkit_monotonic_to_video_media_v1",
) -> SensorClockMappingOutcome:
    """Fit ARKit monotonic time to encoded video PTS without inventing samples."""

    if sidecar is None:
        return SensorClockMappingOutcome(
            SensorClockMappingStatus.MISSING,
            None,
            SensorClockMappingMissingReason.SIDECAR_UNAVAILABLE,
            0,
            None,
        )
    if not isinstance(sidecar, Mapping):
        raise TimebaseValidationError("capture sidecar must be an object")

    raw_unavailable = sidecar.get("unavailable_sensor_reasons", {})
    if raw_unavailable is None:
        raw_unavailable = {}
    if not isinstance(raw_unavailable, Mapping):
        raise TimebaseValidationError("unavailable_sensor_reasons must be an object")
    unavailable = tuple(sorted((str(name), str(reason)) for name, reason in raw_unavailable.items()))

    raw_samples = sidecar.get("arkit_frame_samples", [])
    if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, (str, bytes)):
        raise TimebaseValidationError("arkit_frame_samples must be an array")
    pairs: list[tuple[float, float]] = []
    for sample in raw_samples:
        if not isinstance(sample, Mapping):
            raise TimebaseValidationError("arkit_frame_samples entries must be objects")
        source = sample.get("arkit_timestamp_s")
        target = sample.get("video_pts_s")
        if source is None or target is None:
            continue
        pairs.append((_finite(source, "arkit_timestamp_s"), _finite(target, "video_pts_s")))

    if len(pairs) < 2:
        if unavailable or any(
            isinstance(sample, Mapping) and sample.get("unavailable_reason")
            for sample in raw_samples
        ):
            reason = SensorClockMappingMissingReason.SIDECAR_DECLARES_UNAVAILABLE
        elif not pairs:
            reason = SensorClockMappingMissingReason.NO_DUAL_CLOCK_PAIRS
        else:
            reason = SensorClockMappingMissingReason.INSUFFICIENT_DUAL_CLOCK_PAIRS
        return SensorClockMappingOutcome(
            SensorClockMappingStatus.MISSING,
            None,
            reason,
            len(pairs),
            None,
            unavailable,
        )

    reference = pairs[0][0]
    mean_source = sum(source for source, _target in pairs) / len(pairs)
    mean_target = sum(target for _source, target in pairs) / len(pairs)
    source_variance = sum((source - mean_source) ** 2 for source, _target in pairs)
    if source_variance <= 0.0:
        return SensorClockMappingOutcome(
            SensorClockMappingStatus.MISSING,
            None,
            SensorClockMappingMissingReason.DEGENERATE_SOURCE_CLOCK,
            len(pairs),
            None,
            unavailable,
        )
    scale = sum(
        (source - mean_source) * (target - mean_target)
        for source, target in pairs
    ) / source_variance
    intercept = mean_target - scale * mean_source
    offset = intercept + scale * reference - reference
    residuals = [target - (intercept + scale * source) for source, target in pairs]
    max_abs_residual = max(abs(residual) for residual in residuals)
    mapping = SensorClockMapping(
        mapping_id=mapping_id,
        source_clock=ClockDomain.MONOTONIC,
        target_clock=ClockDomain.MEDIA,
        reference_source_time_s=reference,
        offset_s=offset,
        drift_ppm=(scale - 1.0) * 1_000_000.0,
        declared_tolerance_s=max(1e-9, max_abs_residual),
    )
    return SensorClockMappingOutcome(
        SensorClockMappingStatus.MAPPED,
        mapping,
        None,
        len(pairs),
        max_abs_residual,
        unavailable,
    )


@dataclass(frozen=True, slots=True)
class RollingShutterRowTime:
    row_time_s: float
    row_fraction: float
    offset_from_frame_time_s: float
    model_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_time_s", _finite(self.row_time_s, "row_time_s"))
        fraction = _finite(self.row_fraction, "row_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise TimebaseValidationError("row_fraction must be between 0 and 1")
        object.__setattr__(self, "row_fraction", fraction)
        object.__setattr__(
            self,
            "offset_from_frame_time_s",
            _non_negative(self.offset_from_frame_time_s, "offset_from_frame_time_s"),
        )
        object.__setattr__(self, "model_id", _non_empty(self.model_id, "model_id"))


@dataclass(frozen=True, slots=True)
class RollingShutterModel:
    """A linear sensor-row readout model relative to the first-read row."""

    model_id: str
    frame_readout_s: float
    direction: RollingShutterDirection
    frame_time_reference: str = "first_read_row"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _non_empty(self.model_id, "model_id"))
        object.__setattr__(self, "frame_readout_s", _non_negative(self.frame_readout_s, "frame_readout_s"))
        object.__setattr__(self, "direction", _enum(RollingShutterDirection, self.direction, "direction"))
        if self.frame_time_reference != "first_read_row":
            raise TimebaseValidationError("frame_time_reference must be 'first_read_row'")

    def row_time(self, frame_time_s: float, *, row_index: int, frame_height: int) -> RollingShutterRowTime:
        frame_time_s = _finite(frame_time_s, "frame_time_s")
        if isinstance(frame_height, bool) or not isinstance(frame_height, int) or frame_height <= 0:
            raise TimebaseValidationError("frame_height must be a positive integer")
        if isinstance(row_index, bool) or not isinstance(row_index, int) or not 0 <= row_index < frame_height:
            raise TimebaseValidationError("row_index must be within the frame")
        fraction = 0.0 if frame_height == 1 else row_index / (frame_height - 1)
        if self.direction is RollingShutterDirection.BOTTOM_TO_TOP:
            fraction = 1.0 - fraction
        offset = fraction * self.frame_readout_s
        return RollingShutterRowTime(
            row_time_s=frame_time_s + offset,
            row_fraction=fraction,
            offset_from_frame_time_s=offset,
            model_id=self.model_id,
        )


@dataclass(frozen=True, slots=True)
class NumericSensorSample:
    sample_id: str
    clock: ClockDomain
    time_s: float
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _non_empty(self.sample_id, "sample_id"))
        object.__setattr__(self, "clock", _enum(ClockDomain, self.clock, "clock"))
        object.__setattr__(self, "time_s", _finite(self.time_s, "time_s"))
        values = tuple(_finite(value, "sensor value") for value in self.values)
        if not values:
            raise TimebaseValidationError("values must not be empty")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class AlignmentProvenance:
    method: AlignmentMethod
    tolerance_s: float
    time_basis: TimeBasis
    target_time_s: float
    source_sample_ids: tuple[str, ...]
    source_times_in_target_clock_s: tuple[float, ...]
    max_abs_delta_s: float
    clock_mapping_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _enum(AlignmentMethod, self.method, "alignment method"))
        object.__setattr__(self, "tolerance_s", _positive(self.tolerance_s, "tolerance_s"))
        object.__setattr__(self, "time_basis", _enum(TimeBasis, self.time_basis, "time_basis"))
        object.__setattr__(self, "target_time_s", _finite(self.target_time_s, "target_time_s"))
        sample_ids = tuple(_non_empty(item, "source_sample_id") for item in self.source_sample_ids)
        source_times = tuple(
            _finite(item, "source_times_in_target_clock_s")
            for item in self.source_times_in_target_clock_s
        )
        if not sample_ids or len(sample_ids) != len(source_times):
            raise TimebaseValidationError("alignment provenance requires matching source ids and times")
        object.__setattr__(self, "source_sample_ids", sample_ids)
        object.__setattr__(self, "source_times_in_target_clock_s", source_times)
        object.__setattr__(self, "max_abs_delta_s", _non_negative(self.max_abs_delta_s, "max_abs_delta_s"))
        object.__setattr__(self, "clock_mapping_id", _non_empty(self.clock_mapping_id, "clock_mapping_id"))


@dataclass(frozen=True, slots=True)
class SensorAlignment:
    status: AlignmentStatus
    values: tuple[float, ...] | None
    provenance: AlignmentProvenance | None
    missing_reason: AlignmentMissingReason | None

    def __post_init__(self) -> None:
        status = _enum(AlignmentStatus, self.status, "alignment status")
        reason = None if self.missing_reason is None else _enum(
            AlignmentMissingReason, self.missing_reason, "alignment missing reason"
        )
        if status is AlignmentStatus.ALIGNED:
            if self.values is None or self.provenance is None or reason is not None:
                raise TimebaseValidationError("aligned result requires values/provenance and no missing_reason")
            object.__setattr__(self, "values", tuple(_finite(value, "aligned value") for value in self.values))
        elif self.values is not None or self.provenance is not None or reason is None:
            raise TimebaseValidationError("missing result requires only missing_reason")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "missing_reason", reason)


def _alignment_inputs(
    frame: FrameTime,
    samples: Sequence[NumericSensorSample],
    mapping: SensorClockMapping,
    *,
    tolerance_s: float,
    time_basis: TimeBasis,
) -> tuple[float, list[tuple[float, NumericSensorSample]]] | SensorAlignment:
    tolerance_s = _positive(tolerance_s, "tolerance_s")
    time_basis = _enum(TimeBasis, time_basis, "time_basis")
    if frame.availability.status is not FrameAvailabilityStatus.PRESENT:
        return SensorAlignment(
            status=AlignmentStatus.MISSING,
            values=None,
            provenance=None,
            missing_reason=AlignmentMissingReason.FRAME_NOT_PRESENT,
        )
    if not samples:
        return SensorAlignment(
            status=AlignmentStatus.MISSING,
            values=None,
            provenance=None,
            missing_reason=AlignmentMissingReason.NO_SENSOR_SAMPLES,
        )
    target_time = frame.time_s(time_basis)
    converted: list[tuple[float, NumericSensorSample]] = []
    width: int | None = None
    for sample in samples:
        if sample.clock is not mapping.source_clock:
            raise TimebaseValidationError("sensor sample clock must match mapping.source_clock")
        if width is None:
            width = len(sample.values)
        elif len(sample.values) != width:
            raise TimebaseValidationError("all sensor samples must have equal value dimensions")
        converted.append((mapping.to_target(sample.time_s), sample))
    converted.sort(key=lambda item: (item[0], item[1].sample_id))
    return target_time, converted


def align_nearest_sensor_sample(
    frame: FrameTime,
    samples: Sequence[NumericSensorSample],
    mapping: SensorClockMapping,
    *,
    tolerance_s: float,
    time_basis: TimeBasis,
) -> SensorAlignment:
    """Align the nearest sample only when it is within the explicit tolerance."""

    tolerance_s = _positive(tolerance_s, "tolerance_s")
    prepared = _alignment_inputs(
        frame, samples, mapping, tolerance_s=tolerance_s, time_basis=time_basis
    )
    if isinstance(prepared, SensorAlignment):
        return prepared
    target_time, converted = prepared
    sample_time, sample = min(converted, key=lambda item: (abs(item[0] - target_time), item[0], item[1].sample_id))
    delta = abs(sample_time - target_time)
    if delta > tolerance_s:
        return SensorAlignment(AlignmentStatus.MISSING, None, None, AlignmentMissingReason.OUTSIDE_TOLERANCE)
    provenance = AlignmentProvenance(
        method=AlignmentMethod.NEAREST,
        tolerance_s=tolerance_s,
        time_basis=_enum(TimeBasis, time_basis, "time_basis"),
        target_time_s=target_time,
        source_sample_ids=(sample.sample_id,),
        source_times_in_target_clock_s=(sample_time,),
        max_abs_delta_s=delta,
        clock_mapping_id=mapping.mapping_id,
    )
    return SensorAlignment(AlignmentStatus.ALIGNED, sample.values, provenance, None)


def align_interpolated_sensor_sample(
    frame: FrameTime,
    samples: Sequence[NumericSensorSample],
    mapping: SensorClockMapping,
    *,
    tolerance_s: float,
    time_basis: TimeBasis,
) -> SensorAlignment:
    """Linearly interpolate bracketing samples within an explicit tolerance."""

    tolerance_s = _positive(tolerance_s, "tolerance_s")
    prepared = _alignment_inputs(
        frame, samples, mapping, tolerance_s=tolerance_s, time_basis=time_basis
    )
    if isinstance(prepared, SensorAlignment):
        return prepared
    target_time, converted = prepared
    exact = [item for item in converted if item[0] == target_time]
    if exact:
        sample_time, sample = exact[0]
        provenance = AlignmentProvenance(
            AlignmentMethod.LINEAR_INTERPOLATION,
            tolerance_s,
            _enum(TimeBasis, time_basis, "time_basis"),
            target_time,
            (sample.sample_id,),
            (sample_time,),
            0.0,
            mapping.mapping_id,
        )
        return SensorAlignment(AlignmentStatus.ALIGNED, sample.values, provenance, None)
    before = [item for item in converted if item[0] < target_time]
    after = [item for item in converted if item[0] > target_time]
    if not before or not after:
        return SensorAlignment(AlignmentStatus.MISSING, None, None, AlignmentMissingReason.NO_BRACKETING_SAMPLES)
    left_time, left = before[-1]
    right_time, right = after[0]
    max_delta = max(target_time - left_time, right_time - target_time)
    if max_delta > tolerance_s:
        return SensorAlignment(AlignmentStatus.MISSING, None, None, AlignmentMissingReason.OUTSIDE_TOLERANCE)
    fraction = (target_time - left_time) / (right_time - left_time)
    values = tuple(a + (b - a) * fraction for a, b in zip(left.values, right.values, strict=True))
    provenance = AlignmentProvenance(
        AlignmentMethod.LINEAR_INTERPOLATION,
        tolerance_s,
        _enum(TimeBasis, time_basis, "time_basis"),
        target_time,
        (left.sample_id, right.sample_id),
        (left_time, right_time),
        max_delta,
        mapping.mapping_id,
    )
    return SensorAlignment(AlignmentStatus.ALIGNED, values, provenance, None)


@dataclass(frozen=True, slots=True)
class TimebaseContract:
    capture_id: str
    frames: tuple[FrameTime, ...]
    audio_times: tuple[AudioTime, ...]
    acoustic_models: tuple[AcousticPropagationModel, ...]
    sensor_clock_mappings: tuple[SensorClockMapping, ...]
    rolling_shutter_model: RollingShutterModel | None
    schema_version: int = SCHEMA_VERSION
    artifact_type: str = ARTIFACT_TYPE
    vfr_policy: str = "encoded_pts_required"

    def __post_init__(self) -> None:
        object.__setattr__(self, "capture_id", _non_empty(self.capture_id, "capture_id"))
        if self.schema_version != SCHEMA_VERSION:
            raise TimebaseValidationError(f"schema_version must be {SCHEMA_VERSION}")
        if self.artifact_type != ARTIFACT_TYPE:
            raise TimebaseValidationError(f"artifact_type must be {ARTIFACT_TYPE!r}")
        if self.vfr_policy != "encoded_pts_required":
            raise TimebaseValidationError("vfr_policy must be 'encoded_pts_required'")
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "audio_times", tuple(self.audio_times))
        object.__setattr__(self, "acoustic_models", tuple(self.acoustic_models))
        object.__setattr__(self, "sensor_clock_mappings", tuple(self.sensor_clock_mappings))
        indices = [frame.frame_index for frame in self.frames]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise TimebaseValidationError("frames must have unique ascending frame_index values")
        raw_times = [
            frame.raw_encoded_pts
            for frame in self.frames
            if frame.raw_encoded_pts is not None
        ]
        if any(
            current.pts_ticks * previous.timescale <= previous.pts_ticks * current.timescale
            for previous, current in zip(raw_times, raw_times[1:])
        ):
            raise TimebaseValidationError("encoded raw PTS must be strictly monotonic")
        if len({item.event_id for item in self.audio_times}) != len(self.audio_times):
            raise TimebaseValidationError("audio event_id values must be unique")
        if len({item.model_id for item in self.acoustic_models}) != len(self.acoustic_models):
            raise TimebaseValidationError("acoustic model_id values must be unique")
        if len({item.mapping_id for item in self.sensor_clock_mappings}) != len(self.sensor_clock_mappings):
            raise TimebaseValidationError("sensor mapping_id values must be unique")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    def to_json_bytes(self) -> bytes:
        """Return the canonical, byte-stable JSON representation."""

        return (
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TimebaseContract":
        if not isinstance(payload, Mapping):
            raise TimebaseValidationError("timebase payload must be an object")
        expected = {
            "schema_version",
            "artifact_type",
            "vfr_policy",
            "capture_id",
            "frames",
            "audio_times",
            "acoustic_models",
            "sensor_clock_mappings",
            "rolling_shutter_model",
        }
        extra = set(payload) - expected
        missing = expected - set(payload)
        if extra or missing:
            raise TimebaseValidationError(f"invalid top-level keys; missing={sorted(missing)}, extra={sorted(extra)}")
        return cls(
            schema_version=payload["schema_version"],
            artifact_type=payload["artifact_type"],
            vfr_policy=payload["vfr_policy"],
            capture_id=payload["capture_id"],
            frames=tuple(_frame_from_dict(item) for item in payload["frames"]),
            audio_times=tuple(_audio_from_dict(item) for item in payload["audio_times"]),
            acoustic_models=tuple(AcousticPropagationModel(**item) for item in payload["acoustic_models"]),
            sensor_clock_mappings=tuple(SensorClockMapping(**item) for item in payload["sensor_clock_mappings"]),
            rolling_shutter_model=(
                None
                if payload["rolling_shutter_model"] is None
                else RollingShutterModel(**payload["rolling_shutter_model"])
            ),
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes | bytearray | str) -> "TimebaseContract":
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise TimebaseValidationError("invalid timebase JSON") from exc
        return cls.from_dict(decoded)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _correction_from_dict(payload: Mapping[str, Any]) -> CorrectionProvenance:
    return CorrectionProvenance(**payload)


def _frame_from_dict(payload: Mapping[str, Any]) -> FrameTime:
    availability = FrameAvailability(**payload["availability"])
    raw = None if payload["raw_encoded_pts"] is None else RawEncodedPTS(**payload["raw_encoded_pts"])
    corrected_payload = payload["corrected_pts"]
    corrected = None
    if corrected_payload is not None:
        corrected = CorrectedPTS(
            corrected_time_s=corrected_payload["corrected_time_s"],
            provenance=_correction_from_dict(corrected_payload["provenance"]),
        )
    return FrameTime(
        frame_index=payload["frame_index"],
        availability=availability,
        raw_encoded_pts=raw,
        corrected_pts=corrected,
    )


def _audio_from_dict(payload: Mapping[str, Any]) -> AudioTime:
    correction = payload["correction"]
    return AudioTime(
        event_id=payload["event_id"],
        raw_time_s=payload["raw_time_s"],
        raw_clock=payload["raw_clock"],
        corrected_time_s=payload["corrected_time_s"],
        correction=None if correction is None else _correction_from_dict(correction),
    )


__all__ = [
    "ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "AcousticPropagationModel",
    "AlignmentMethod",
    "AlignmentMissingReason",
    "AlignmentProvenance",
    "AlignmentStatus",
    "AudioTime",
    "ClockDomain",
    "CorrectedPTS",
    "CorrectionProvenance",
    "FrameAbsenceReason",
    "FrameAvailability",
    "FrameAvailabilityStatus",
    "FrameTime",
    "NumericSensorSample",
    "RawEncodedPTS",
    "RollingShutterDirection",
    "RollingShutterModel",
    "RollingShutterRowTime",
    "SensorAlignment",
    "SensorClockMapping",
    "SensorClockMappingMissingReason",
    "SensorClockMappingOutcome",
    "SensorClockMappingStatus",
    "TimeBasis",
    "TimebaseContract",
    "TimebaseValidationError",
    "align_interpolated_sensor_sample",
    "align_nearest_sensor_sample",
    "build_sensor_clock_mapping_from_sidecar",
]
