"""CPU-only drill repetition verification primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


DEFAULT_ENTER_SPEED_MPS = 2.0
DEFAULT_EXIT_SPEED_MPS = 0.75
DEFAULT_MAX_CONTACT_DELAY_S = 0.35
DEFAULT_EVENT_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class WristVelocitySample:
    t: float
    speed_mps: float


@dataclass(frozen=True)
class DrillEventFlag:
    t: float
    flag: str
    fault: bool = True


@dataclass(frozen=True)
class DetectedRep:
    t: float
    start_t: float
    peak_speed_mps: float


@dataclass(frozen=True)
class DrillRepVerdict:
    quality: Literal["clean", "fault"]
    reasons: list[str]


def detect_reps(
    *,
    contact_timestamps: Sequence[float],
    wrist_velocity_samples: Sequence[WristVelocitySample],
    enter_speed_mps: float = DEFAULT_ENTER_SPEED_MPS,
    exit_speed_mps: float = DEFAULT_EXIT_SPEED_MPS,
    max_contact_delay_s: float = DEFAULT_MAX_CONTACT_DELAY_S,
) -> list[DetectedRep]:
    """Detect one drill rep per wrist swing that reaches a contact event."""

    _validate_detection_config(
        enter_speed_mps=enter_speed_mps,
        exit_speed_mps=exit_speed_mps,
        max_contact_delay_s=max_contact_delay_s,
    )

    events: list[tuple[float, int, WristVelocitySample | float]] = []
    events.extend((sample.t, 0, sample) for sample in wrist_velocity_samples)
    events.extend((float(contact_t), 1, float(contact_t)) for contact_t in contact_timestamps)
    events.sort(key=lambda event: (event[0], event[1]))

    reps: list[DetectedRep] = []
    swing_start_t: float | None = None
    swing_peak_speed_mps = 0.0
    counted_current_swing = False

    for _event_t, event_kind, payload in events:
        if event_kind == 0:
            sample = payload
            if not isinstance(sample, WristVelocitySample):
                raise TypeError("velocity event payload must be WristVelocitySample")

            if swing_start_t is None:
                if sample.speed_mps >= enter_speed_mps:
                    swing_start_t = sample.t
                    swing_peak_speed_mps = sample.speed_mps
                    counted_current_swing = False
                continue

            swing_peak_speed_mps = max(swing_peak_speed_mps, sample.speed_mps)
            if sample.speed_mps <= exit_speed_mps:
                swing_start_t = None
                swing_peak_speed_mps = 0.0
                counted_current_swing = False
            continue

        contact_t = float(payload)
        if swing_start_t is None or counted_current_swing:
            continue
        if contact_t - swing_start_t > max_contact_delay_s:
            continue

        reps.append(
            DetectedRep(
                t=contact_t,
                start_t=swing_start_t,
                peak_speed_mps=swing_peak_speed_mps,
            )
        )
        counted_current_swing = True

    return reps


def classify_rep_quality(
    rep: DetectedRep,
    *,
    event_flags: Sequence[DrillEventFlag] = (),
    tolerance_s: float = DEFAULT_EVENT_TOLERANCE_S,
) -> DrillRepVerdict:
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")

    reasons = sorted(
        {
            flag.flag
            for flag in event_flags
            if flag.fault and abs(flag.t - rep.t) <= tolerance_s
        }
    )
    return DrillRepVerdict(
        quality="fault" if reasons else "clean",
        reasons=reasons,
    )


def build_drill_report(
    *,
    drill: str,
    contact_timestamps: Sequence[float],
    wrist_velocity_samples: Sequence[WristVelocitySample],
    event_flags: Sequence[DrillEventFlag] = (),
    enter_speed_mps: float = DEFAULT_ENTER_SPEED_MPS,
    exit_speed_mps: float = DEFAULT_EXIT_SPEED_MPS,
    max_contact_delay_s: float = DEFAULT_MAX_CONTACT_DELAY_S,
    event_tolerance_s: float = DEFAULT_EVENT_TOLERANCE_S,
) -> dict[str, object]:
    reps = detect_reps(
        contact_timestamps=contact_timestamps,
        wrist_velocity_samples=wrist_velocity_samples,
        enter_speed_mps=enter_speed_mps,
        exit_speed_mps=exit_speed_mps,
        max_contact_delay_s=max_contact_delay_s,
    )

    per_rep = []
    clean_reps = 0
    for rep in reps:
        verdict = classify_rep_quality(
            rep,
            event_flags=event_flags,
            tolerance_s=event_tolerance_s,
        )
        if verdict.quality == "clean":
            clean_reps += 1
        per_rep.append({"t": rep.t, "quality": verdict.quality, "reasons": verdict.reasons})

    return {
        "schema_version": 1,
        "drill": drill,
        "reps": len(reps),
        "clean_reps": clean_reps,
        "per_rep": per_rep,
    }


def _validate_detection_config(
    *,
    enter_speed_mps: float,
    exit_speed_mps: float,
    max_contact_delay_s: float,
) -> None:
    if enter_speed_mps < 0:
        raise ValueError("enter_speed_mps must be non-negative")
    if exit_speed_mps < 0:
        raise ValueError("exit_speed_mps must be non-negative")
    if exit_speed_mps > enter_speed_mps:
        raise ValueError("exit_speed_mps must be less than or equal to enter_speed_mps")
    if max_contact_delay_s < 0:
        raise ValueError("max_contact_delay_s must be non-negative")
