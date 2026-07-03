"""Adapter predictors for confidence-gated physics provenance.

These predictors intentionally do not rewrite upstream PHYS artifacts.  Adapters
return states that already exist in wrapped artifacts; the kinematic predictor is
only a bounded short-gap fallback for joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Mapping, Sequence


Vector = Sequence[float]


@dataclass(frozen=True)
class PredictionResult:
    available: bool
    state: Any | None
    sigma_m: float | None
    predictor: str
    horizon_frames: int
    reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


class PaddleNullPredictor:
    """Honest paddle adapter for the current 0/63-estimate input limit."""

    name = "PaddleNullPredictor"

    def predict(self, history: Sequence[Mapping[str, Any]], horizon_frames: int) -> PredictionResult:
        return PredictionResult(
            available=False,
            state=None,
            sigma_m=None,
            predictor=self.name,
            horizon_frames=horizon_frames,
            reason="no_paddle_prediction_possible",
            provenance={"history_count": len(history)},
        )


class JointKinematicPredictor:
    """Bounded constant-velocity predictor for short joint gaps."""

    name = "JointKinematicPredictor"

    def __init__(
        self,
        *,
        fps: float,
        max_speed_mps: float = 6.0,
        base_sigma_m: float = 0.03,
        sigma_per_frame_m: float = 0.015,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        self.fps = float(fps)
        self.max_speed_mps = float(max_speed_mps)
        self.base_sigma_m = float(base_sigma_m)
        self.sigma_per_frame_m = float(sigma_per_frame_m)

    def predict(self, history: Sequence[Mapping[str, Any]], horizon_frames: int) -> PredictionResult:
        if horizon_frames < 0:
            raise ValueError("horizon_frames must be nonnegative")
        if len(history) < 2:
            return self._unavailable(horizon_frames, "insufficient_joint_history")

        prev, last = history[-2], history[-1]
        prev_state = _state_vector(prev)
        last_state = _state_vector(last)
        if prev_state is None or last_state is None:
            return self._unavailable(horizon_frames, "missing_joint_state")
        if len(prev_state) != len(last_state):
            return self._unavailable(horizon_frames, "state_dimension_mismatch")

        prev_index = int(prev.get("frame_index", len(history) - 2))
        last_index = int(last.get("frame_index", len(history) - 1))
        frame_delta = max(1, last_index - prev_index)
        dt_s = frame_delta / self.fps
        velocity = [(b - a) / dt_s for a, b in zip(prev_state, last_state)]
        velocity = _clamp_vector(velocity, self.max_speed_mps)
        horizon_s = horizon_frames / self.fps
        predicted = [value + velocity_i * horizon_s for value, velocity_i in zip(last_state, velocity)]
        return PredictionResult(
            available=True,
            state=predicted,
            sigma_m=self.base_sigma_m + self.sigma_per_frame_m * horizon_frames,
            predictor=self.name,
            horizon_frames=horizon_frames,
            provenance={
                "max_speed_mps": self.max_speed_mps,
                "frame_delta": frame_delta,
            },
        )

    def _unavailable(self, horizon_frames: int, reason: str) -> PredictionResult:
        return PredictionResult(
            available=False,
            state=None,
            sigma_m=None,
            predictor=self.name,
            horizon_frames=horizon_frames,
            reason=reason,
        )


class BallBallisticAdapter:
    """Read-only adapter over `ball_track_physics_filled.json` frames."""

    name = "BallBallisticAdapter"

    def __init__(self, ball_track_physics_filled: Mapping[str, Any] | None) -> None:
        payload = ball_track_physics_filled or {}
        frames = payload.get("frames", [])
        self.frames = list(frames) if isinstance(frames, list) else []

    def predict(self, history: Sequence[Mapping[str, Any]], horizon_frames: int) -> PredictionResult:
        if not history:
            return self._unavailable(horizon_frames, "missing_ball_history")
        last_index = int(history[-1].get("frame_index", -1))
        target_index = last_index + horizon_frames
        if target_index < 0 or target_index >= len(self.frames):
            return self._unavailable(horizon_frames, "target_frame_out_of_range")
        frame = self.frames[target_index]
        if not isinstance(frame, Mapping):
            return self._unavailable(horizon_frames, "target_frame_invalid")
        world_xyz = frame.get("world_xyz")
        if world_xyz is None:
            return self._unavailable(horizon_frames, "target_frame_missing_world_xyz")
        physics_fill = frame.get("physics_fill") if isinstance(frame.get("physics_fill"), Mapping) else {}
        sigma = _finite_float(physics_fill.get("uncertainty_m")) if physics_fill else None
        if sigma is None:
            sigma = 0.05 + 0.04 * max(0, horizon_frames)
        return PredictionResult(
            available=True,
            state=list(world_xyz),
            sigma_m=sigma,
            predictor=self.name,
            horizon_frames=horizon_frames,
            provenance={
                "frame_index": target_index,
                "source": frame.get("source"),
                "render_only": bool(physics_fill.get("render_only", frame.get("render_only", False))),
                "not_for_detection_metrics": bool(
                    physics_fill.get("not_for_detection_metrics", frame.get("not_for_detection_metrics", False))
                ),
            },
        )

    def _unavailable(self, horizon_frames: int, reason: str) -> PredictionResult:
        return PredictionResult(
            available=False,
            state=None,
            sigma_m=None,
            predictor=self.name,
            horizon_frames=horizon_frames,
            reason=reason,
        )


class FootContactLockAdapter:
    """Read-only adapter over `physics_footlock.json` corrected joint frames."""

    name = "FootContactLockAdapter"

    def __init__(self, physics_footlock: Mapping[str, Any] | None) -> None:
        self.by_player: dict[int, dict[int, Mapping[str, Any]]] = {}
        payload = physics_footlock or {}
        players = payload.get("players", [])
        if not isinstance(players, list):
            return
        for player in players:
            if not isinstance(player, Mapping):
                continue
            player_id = player.get("id")
            if player_id is None:
                continue
            frames_by_index: dict[int, Mapping[str, Any]] = {}
            for offset, frame in enumerate(player.get("frames", []) or []):
                if not isinstance(frame, Mapping):
                    continue
                frame_index = int(frame.get("frame_index", offset))
                frames_by_index[frame_index] = frame
            self.by_player[int(player_id)] = frames_by_index

    def predict(self, history: Sequence[Mapping[str, Any]], horizon_frames: int) -> PredictionResult:
        if not history:
            return self._unavailable(horizon_frames, "missing_footlock_history")
        last = history[-1]
        player_id = last.get("player_id")
        if player_id is None:
            return self._unavailable(horizon_frames, "missing_player_id")
        target_index = int(last.get("frame_index", -1)) + horizon_frames
        frame = self.by_player.get(int(player_id), {}).get(target_index)
        if not frame:
            return self._unavailable(horizon_frames, "target_frame_not_in_footlock")
        joints_world = frame.get("joints_world")
        if joints_world is None:
            return self._unavailable(horizon_frames, "target_frame_missing_joints")
        return PredictionResult(
            available=True,
            state=joints_world,
            sigma_m=0.02 + 0.01 * max(0, horizon_frames),
            predictor=self.name,
            horizon_frames=horizon_frames,
            provenance={"player_id": int(player_id), "frame_index": target_index},
        )

    def _unavailable(self, horizon_frames: int, reason: str) -> PredictionResult:
        return PredictionResult(
            available=False,
            state=None,
            sigma_m=None,
            predictor=self.name,
            horizon_frames=horizon_frames,
            reason=reason,
        )


def _state_vector(item: Mapping[str, Any]) -> list[float] | None:
    state = item.get("state", item.get("world_xyz"))
    if not isinstance(state, Sequence) or isinstance(state, (str, bytes)):
        return None
    values = [_finite_float(value) for value in state]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values if value is not None]


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _clamp_vector(vector: Sequence[float], max_norm: float) -> list[float]:
    norm = sqrt(sum(component * component for component in vector))
    if norm <= max_norm or norm == 0.0:
        return [float(component) for component in vector]
    scale = max_norm / norm
    return [float(component) * scale for component in vector]


__all__ = [
    "BallBallisticAdapter",
    "FootContactLockAdapter",
    "JointKinematicPredictor",
    "PaddleNullPredictor",
    "PredictionResult",
]
