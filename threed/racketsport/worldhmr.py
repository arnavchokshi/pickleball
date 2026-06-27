"""World-grounded SMPL reconstruction helpers.

This module intentionally stops at deterministic CPU primitives. It does not
run Fast SAM-3D-Body or infer SMPL parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Sequence


SCAFFOLD_NOTE = "cpu_worldhmr_primitives_no_sam3dbody_integration"


@dataclass(frozen=True)
class WorldTranslationSample:
    """Single per-player root translation and optional mesh vertices."""

    frame_idx: int
    player_id: int
    root_xyz: list[float]
    mesh_vertices_xyz: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_vector3(self.root_xyz, name="root_xyz")
        for idx, vertex in enumerate(self.mesh_vertices_xyz):
            _validate_vector3(vertex, name=f"mesh_vertices_xyz/{idx}")

        object.__setattr__(self, "root_xyz", [float(value) for value in self.root_xyz])
        object.__setattr__(
            self,
            "mesh_vertices_xyz",
            [[float(value) for value in vertex] for vertex in self.mesh_vertices_xyz],
        )


@dataclass(frozen=True)
class WorldGroundingMetrics:
    """Residual summary for adjusted world root translations."""

    sample_count: int
    rms_root_residual_m: float
    max_root_residual_m: float
    rms_ground_z_error_m: float
    max_ground_z_error_m: float
    scaffold: str = SCAFFOLD_NOTE


def snap_player_translation_to_court(
    sample: WorldTranslationSample,
    *,
    court_z_m: float = 0.0,
) -> WorldTranslationSample:
    """Move a player's root translation to the court plane.

    The same vertical delta is applied to mesh vertices so root-relative mesh
    height is preserved.
    """

    z_delta = court_z_m - sample.root_xyz[2]
    return WorldTranslationSample(
        frame_idx=sample.frame_idx,
        player_id=sample.player_id,
        root_xyz=[sample.root_xyz[0], sample.root_xyz[1], court_z_m],
        mesh_vertices_xyz=[
            [vertex[0], vertex[1], vertex[2] + z_delta]
            for vertex in sample.mesh_vertices_xyz
        ],
    )


def smooth_world_translations(
    samples: Sequence[WorldTranslationSample],
    *,
    alpha: float = 0.5,
) -> list[WorldTranslationSample]:
    """Apply deterministic per-player EMA smoothing to root translations."""

    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must be greater than 0 and less than or equal to 1")

    previous_by_player: dict[int, list[float]] = {}
    smoothed: list[WorldTranslationSample] = []
    for sample in samples:
        previous = previous_by_player.get(sample.player_id)
        if previous is None:
            root_xyz = list(sample.root_xyz)
        else:
            root_xyz = [
                alpha * sample.root_xyz[idx] + (1.0 - alpha) * previous[idx]
                for idx in range(3)
            ]

        previous_by_player[sample.player_id] = root_xyz
        smoothed.append(
            WorldTranslationSample(
                frame_idx=sample.frame_idx,
                player_id=sample.player_id,
                root_xyz=root_xyz,
                mesh_vertices_xyz=sample.mesh_vertices_xyz,
            )
        )

    return smoothed


def residual_metrics(
    observed: Sequence[WorldTranslationSample],
    adjusted: Sequence[WorldTranslationSample],
    *,
    court_z_m: float = 0.0,
) -> WorldGroundingMetrics:
    """Compute root residuals and adjusted root z error against the court."""

    if len(observed) != len(adjusted):
        raise ValueError("observed and adjusted must have the same length")

    root_residuals: list[float] = []
    ground_z_errors: list[float] = []
    for observed_sample, adjusted_sample in zip(observed, adjusted):
        if observed_sample.frame_idx != adjusted_sample.frame_idx:
            raise ValueError("observed and adjusted frame_idx values must match")
        if observed_sample.player_id != adjusted_sample.player_id:
            raise ValueError("observed and adjusted player_id values must match")

        root_residuals.append(_distance3(observed_sample.root_xyz, adjusted_sample.root_xyz))
        ground_z_errors.append(abs(adjusted_sample.root_xyz[2] - court_z_m))

    return WorldGroundingMetrics(
        sample_count=len(root_residuals),
        rms_root_residual_m=_rms(root_residuals),
        max_root_residual_m=max(root_residuals, default=0.0),
        rms_ground_z_error_m=_rms(ground_z_errors),
        max_ground_z_error_m=max(ground_z_errors, default=0.0),
    )


def _distance3(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((left[idx] - right[idx]) ** 2 for idx in range(3)))


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sqrt(sum(value * value for value in values) / len(values))


def _validate_vector3(values: Sequence[float], *, name: str) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector")
