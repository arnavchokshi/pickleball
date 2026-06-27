"""Foot-contact and foot-skate elimination primitives.

This module intentionally stops at deterministic CPU helpers. It does not run a
body-model IK solve or rewrite upstream SMPL pose parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


SCAFFOLD_NOTE = "cpu_foot_lock_primitives_no_smpl_ik"


@dataclass(frozen=True)
class ContactHysteresis:
    enter_height_m: float = 0.025
    exit_height_m: float = 0.050
    enter_speed_mps: float = 0.20
    exit_speed_mps: float = 0.35
    min_confidence: float = 0.50


@dataclass(frozen=True)
class FootContactObservation:
    height_m: float
    speed_mps: float
    confidence: float


@dataclass(frozen=True)
class FootKinematics:
    position_xyz: list[float]
    velocity_xyz: list[float]
    contact: bool


@dataclass(frozen=True)
class FootLockMetrics:
    max_slide_m: float
    max_penetration_m: float
    contact_frames: int
    scaffold: str = SCAFFOLD_NOTE


def classify_contact(
    observation: FootContactObservation,
    *,
    previous_contact: bool,
    hysteresis: ContactHysteresis = ContactHysteresis(),
) -> bool:
    _validate_hysteresis(hysteresis)
    if observation.confidence < hysteresis.min_confidence:
        return False

    if previous_contact:
        return observation.height_m <= hysteresis.exit_height_m and observation.speed_mps <= hysteresis.exit_speed_mps
    return observation.height_m <= hysteresis.enter_height_m and observation.speed_mps <= hysteresis.enter_speed_mps


def classify_contact_sequence(
    observations: Sequence[FootContactObservation],
    *,
    initial_contact: bool = False,
    hysteresis: ContactHysteresis = ContactHysteresis(),
) -> list[bool]:
    contacts: list[bool] = []
    previous_contact = initial_contact
    for observation in observations:
        previous_contact = classify_contact(
            observation,
            previous_contact=previous_contact,
            hysteresis=hysteresis,
        )
        contacts.append(previous_contact)
    return contacts


def snap_stance_foot(foot: FootKinematics, *, court_z_m: float = 0.0) -> FootKinematics:
    _validate_vector3(foot.position_xyz, name="position_xyz")
    _validate_vector3(foot.velocity_xyz, name="velocity_xyz")
    if not foot.contact:
        return FootKinematics(
            position_xyz=list(foot.position_xyz),
            velocity_xyz=list(foot.velocity_xyz),
            contact=False,
        )

    return FootKinematics(
        position_xyz=[foot.position_xyz[0], foot.position_xyz[1], court_z_m],
        velocity_xyz=[0.0, 0.0, 0.0],
        contact=True,
    )


def foot_lock_metrics(samples: Sequence[FootKinematics], *, court_z_m: float = 0.0) -> FootLockMetrics:
    max_slide_m = 0.0
    max_penetration_m = 0.0
    contact_frames = 0
    previous_contact_sample: FootKinematics | None = None

    for sample in samples:
        _validate_vector3(sample.position_xyz, name="position_xyz")
        _validate_vector3(sample.velocity_xyz, name="velocity_xyz")

        penetration_m = max(0.0, court_z_m - sample.position_xyz[2])
        max_penetration_m = max(max_penetration_m, penetration_m)

        if sample.contact:
            contact_frames += 1
            if previous_contact_sample is not None:
                slide_m = math.hypot(
                    sample.position_xyz[0] - previous_contact_sample.position_xyz[0],
                    sample.position_xyz[1] - previous_contact_sample.position_xyz[1],
                )
                max_slide_m = max(max_slide_m, slide_m)
            previous_contact_sample = sample
        else:
            previous_contact_sample = None

    return FootLockMetrics(
        max_slide_m=max_slide_m,
        max_penetration_m=max_penetration_m,
        contact_frames=contact_frames,
    )


def _validate_hysteresis(hysteresis: ContactHysteresis) -> None:
    if hysteresis.enter_height_m < 0 or hysteresis.exit_height_m < 0:
        raise ValueError("contact height thresholds must be non-negative")
    if hysteresis.enter_speed_mps < 0 or hysteresis.exit_speed_mps < 0:
        raise ValueError("contact speed thresholds must be non-negative")
    if hysteresis.min_confidence < 0 or hysteresis.min_confidence > 1:
        raise ValueError("min_confidence must be between 0 and 1")
    if hysteresis.exit_height_m < hysteresis.enter_height_m:
        raise ValueError("exit_height_m must be greater than or equal to enter_height_m")
    if hysteresis.exit_speed_mps < hysteresis.enter_speed_mps:
        raise ValueError("exit_speed_mps must be greater than or equal to enter_speed_mps")


def _validate_vector3(values: Sequence[float], *, name: str) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector")
