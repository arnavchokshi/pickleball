"""Pure-numpy pickleball flight simulator for synthetic BALL lift pairs.

The simulator intentionally reuses ``ball_arc_solver.PhysicsParameters`` and
``ball_arc_solver._rk4_step`` for the drag+gravity core. Spin, bounce, detector
noise, and corpus formatting live here so the production solver remains
unchanged until its own P1-4 Magnus task.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .ball_arc_solver import (
    BALL_RADIUS_M,
    AnchorEvent,
    BallArcSolverConfig,
    BallObservation,
    PhysicsParameters,
    _rk4_step,
    fit_flight_segment,
)
from .ball_flight_sanity import evaluate_ball_flight_sanity
from .court_calibration import project_world_points
from .schemas import CourtCalibration


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_synthetic_flight_pair"
CORPUS_ARTIFACT_TYPE = "racketsport_flight_corpus_report"
STEYN_CL_PER_SPIN = 0.195
DEFAULT_CALIBRATION_PATH = Path(
    "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json"
)

SHOT_FAMILY_ASSUMPTIONS: dict[str, dict[str, Any]] = {
    "serve": {
        "speed_mps": [8.0, 17.0],
        "launch_angle_deg": [3.0, 14.0],
        "start_y_m": [-6.35, -5.7],
        "target_y_m": [1.5, 5.8],
        "height_m": [0.65, 1.15],
        "source": "engineering prior from pickleball court geometry; unmeasured pending owner/H13 speed capture",
    },
    "drive": {
        "speed_mps": [10.0, 20.0],
        "launch_angle_deg": [-1.0, 11.0],
        "start_y_m": [-5.6, -2.2],
        "target_y_m": [1.0, 6.2],
        "height_m": [0.45, 1.25],
        "source": "engineering prior for flat attacking shots; unmeasured pending owner/H13 speed capture",
    },
    "drop": {
        "speed_mps": [4.5, 10.0],
        "launch_angle_deg": [10.0, 28.0],
        "start_y_m": [-6.0, -3.0],
        "target_y_m": [0.4, 2.3],
        "height_m": [0.5, 1.2],
        "source": "engineering prior for soft transition shots; unmeasured pending owner/H13 speed capture",
    },
    "lob": {
        "speed_mps": [7.0, 14.0],
        "launch_angle_deg": [28.0, 50.0],
        "start_y_m": [-5.5, -2.0],
        "target_y_m": [3.5, 6.6],
        "height_m": [0.55, 1.2],
        "source": "engineering prior constrained by <=8m sanity apex; unmeasured pending owner/H13 speed capture",
    },
    "dink": {
        "speed_mps": [2.5, 6.5],
        "launch_angle_deg": [5.0, 24.0],
        "start_y_m": [-2.2, -0.35],
        "target_y_m": [0.35, 2.2],
        "height_m": [0.3, 0.8],
        "source": "engineering prior for kitchen exchanges; unmeasured pending owner/H13 speed capture",
    },
}


@dataclass(frozen=True)
class BounceParameters:
    """Simple court bounce model.

    Defaults are deliberately labeled unmeasured because H13 surface restitution
    and friction measurements do not exist yet.
    """

    restitution: float = 0.58
    friction: float = 0.16
    status: str = "unmeasured_default_pending_H13"

    def to_json(self) -> dict[str, Any]:
        return {
            "restitution": float(self.restitution),
            "friction": float(self.friction),
            "status": self.status,
            "source": "simulator default, not measured; replace after H13 court-surface measurement",
        }


@dataclass(frozen=True)
class DetectorNoiseProfile:
    p95_jitter_px: float = 34.0
    recall: float = 0.578
    hidden_fp_rate: float = 0.021

    def to_json(self) -> dict[str, float]:
        return {
            "p95_jitter_px": float(self.p95_jitter_px),
            "recall": float(self.recall),
            "hidden_fp_rate": float(self.hidden_fp_rate),
        }


@dataclass(frozen=True)
class FlightSimulationConfig:
    fps: float = 240.0
    dt_s: float = 1.0 / 240.0
    max_time_s: float = 0.95
    spin_scalar: float = 0.0
    ball_type: str = "outdoor"
    bounce: BounceParameters = BounceParameters()
    max_bounces: int = 1
    court_exit_margin_m: float = 4.0

    def to_json(self) -> dict[str, Any]:
        return {
            "fps": float(self.fps),
            "dt_s": float(self.dt_s),
            "max_time_s": float(self.max_time_s),
            "spin_scalar": float(self.spin_scalar),
            "ball_type": self.ball_type,
            "bounce": self.bounce.to_json(),
            "max_bounces": int(self.max_bounces),
            "court_exit_margin_m": float(self.court_exit_margin_m),
        }


@dataclass(frozen=True)
class ShotInitialState:
    family: str
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    speed_mps: float
    speed_mps_min: float
    speed_mps_max: float
    launch_angle_deg: float
    launch_angle_deg_min: float
    launch_angle_deg_max: float
    target_xy_m: tuple[float, float]
    assumptions: Mapping[str, Any]

    def with_velocity(self, velocity_mps: Sequence[float]) -> ShotInitialState:
        velocity = _vec3(velocity_mps)
        horizontal = math.hypot(velocity[0], velocity[1])
        speed = _norm3(velocity)
        angle = math.degrees(math.atan2(velocity[2], horizontal)) if speed > 0.0 else 0.0
        return replace(self, velocity_mps=velocity, speed_mps=speed, launch_angle_deg=angle)

    def with_position(self, position_m: Sequence[float]) -> ShotInitialState:
        return replace(self, position_m=_vec3(position_m))

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "position_m": _list3(self.position_m),
            "velocity_mps": _list3(self.velocity_mps),
            "speed_mps": _round(self.speed_mps, 6),
            "speed_mps_range": [_round(self.speed_mps_min, 6), _round(self.speed_mps_max, 6)],
            "launch_angle_deg": _round(self.launch_angle_deg, 6),
            "launch_angle_deg_range": [
                _round(self.launch_angle_deg_min, 6),
                _round(self.launch_angle_deg_max, 6),
            ],
            "target_xy_m": [_round(self.target_xy_m[0], 6), _round(self.target_xy_m[1], 6)],
            "assumptions": dict(self.assumptions),
        }


@dataclass(frozen=True)
class SimulatedSample:
    frame: int
    t: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    segment_id: int

    def to_json(self) -> dict[str, Any]:
        return {
            "frame": int(self.frame),
            "t": _round(self.t, 9),
            "world_xyz_m": _list3(self.position_m),
            "velocity_mps": _list3(self.velocity_mps),
            "segment_id": int(self.segment_id),
        }


@dataclass(frozen=True)
class SimulatedTrajectory:
    shot: ShotInitialState
    samples: tuple[SimulatedSample, ...]
    bounces: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "shot": self.shot.to_json(),
            "samples": [sample.to_json() for sample in self.samples],
            "bounces": [dict(item) for item in self.bounces],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LoadedCourtCalibration:
    payload: Mapping[str, Any]
    model: CourtCalibration
    path: Path | None = None


def load_court_calibration(path: str | Path) -> LoadedCourtCalibration:
    path_obj = Path(path)
    import json

    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    model = CourtCalibration.model_validate(payload)
    return LoadedCourtCalibration(payload=payload, model=model, path=path_obj)


def sample_shot_family(
    family: str,
    rng: np.random.Generator,
    *,
    start_side: str = "near",
) -> ShotInitialState:
    if family not in SHOT_FAMILY_ASSUMPTIONS:
        raise ValueError(f"unsupported shot family: {family}")
    if start_side not in {"near", "far"}:
        raise ValueError("start_side must be 'near' or 'far'")
    spec = SHOT_FAMILY_ASSUMPTIONS[family]
    speed_min, speed_max = [float(value) for value in spec["speed_mps"]]
    angle_min, angle_max = [float(value) for value in spec["launch_angle_deg"]]
    start_y_min, start_y_max = [float(value) for value in spec["start_y_m"]]
    target_y_min, target_y_max = [float(value) for value in spec["target_y_m"]]
    height_min, height_max = [float(value) for value in spec["height_m"]]

    side_sign = 1.0 if start_side == "near" else -1.0
    x0 = float(rng.uniform(-1.4, 1.4))
    y0 = side_sign * float(rng.uniform(start_y_min, start_y_max))
    z0 = float(rng.uniform(height_min, height_max))
    target_x = float(rng.uniform(-2.4, 2.4))
    target_y = -side_sign * float(rng.uniform(target_y_min, target_y_max))
    speed = float(rng.uniform(speed_min, speed_max))
    angle = float(rng.uniform(angle_min, angle_max))
    launch_rad = math.radians(angle)
    horizontal_speed = speed * math.cos(launch_rad)
    vz = speed * math.sin(launch_rad)
    dx = target_x - x0
    dy = target_y - y0
    horizontal_norm = math.hypot(dx, dy)
    if horizontal_norm <= 1e-9:
        direction = (0.0, -side_sign)
    else:
        direction = (dx / horizontal_norm, dy / horizontal_norm)
    velocity = (horizontal_speed * direction[0], horizontal_speed * direction[1], vz)
    assumptions = {
        "status": "plausible_unmeasured_prior",
        "source": spec["source"],
        "constants": "Cd from Lindsey/Steyn roadmap note; Cl formula from Steyn arXiv:2501.00163",
    }
    return ShotInitialState(
        family=family,
        position_m=(x0, y0, z0),
        velocity_mps=velocity,
        speed_mps=speed,
        speed_mps_min=speed_min,
        speed_mps_max=speed_max,
        launch_angle_deg=angle,
        launch_angle_deg_min=angle_min,
        launch_angle_deg_max=angle_max,
        target_xy_m=(target_x, target_y),
        assumptions=assumptions,
    )


def simulate_flight(
    shot: ShotInitialState,
    *,
    physics: PhysicsParameters | None = None,
    config: FlightSimulationConfig | None = None,
) -> SimulatedTrajectory:
    cfg = config or FlightSimulationConfig()
    phys = physics or PhysicsParameters.for_ball_type(cfg.ball_type)
    if cfg.dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if cfg.max_time_s <= 0.0:
        raise ValueError("max_time_s must be positive")
    spin_axis = _spin_axis_for_velocity(shot.velocity_mps)
    state = (*shot.position_m, *shot.velocity_mps)
    samples: list[SimulatedSample] = [
        SimulatedSample(frame=0, t=0.0, position_m=shot.position_m, velocity_mps=shot.velocity_mps, segment_id=0)
    ]
    bounces: list[dict[str, Any]] = []
    t = 0.0
    frame = 0
    segment_id = 0
    max_steps = int(math.ceil(cfg.max_time_s / cfg.dt_s)) + 2
    for _ in range(max_steps):
        if t >= cfg.max_time_s - 1e-12:
            break
        dt = min(cfg.dt_s, cfg.max_time_s - t)
        previous = state
        if abs(cfg.spin_scalar) <= 1e-12:
            state = _rk4_step(state, dt, phys)
        else:
            state = _rk4_step_with_magnus(state, dt, phys, spin_axis, cfg.spin_scalar)
        next_t = t + dt
        if (
            len(bounces) < cfg.max_bounces
            and previous[2] >= BALL_RADIUS_M
            and state[2] <= BALL_RADIUS_M
            and state[5] < 0.0
        ):
            impact_state, alpha = _interpolate_impact(previous, state, BALL_RADIUS_M)
            pre_velocity = (impact_state[3], impact_state[4], impact_state[5])
            post_velocity = _bounce_velocity(pre_velocity, cfg.bounce)
            impact_t = t + alpha * dt
            event = {
                "bounce_index": len(bounces),
                "frame": frame + 1,
                "t": _round(impact_t, 9),
                "world_xyz_m": _list3((impact_state[0], impact_state[1], BALL_RADIUS_M)),
                "pre_velocity_mps": _list3(pre_velocity),
                "post_velocity_mps": _list3(post_velocity),
                "restitution": float(cfg.bounce.restitution),
                "friction": float(cfg.bounce.friction),
                "model_status": cfg.bounce.status,
            }
            bounces.append(event)
            segment_id += 1
            state = (impact_state[0], impact_state[1], BALL_RADIUS_M, *post_velocity)
        elif len(bounces) >= cfg.max_bounces and state[2] < BALL_RADIUS_M:
            break
        if _outside_loose_court(state, cfg.court_exit_margin_m):
            break
        t = next_t
        frame += 1
        samples.append(
            SimulatedSample(
                frame=frame,
                t=t,
                position_m=(state[0], state[1], state[2]),
                velocity_mps=(state[3], state[4], state[5]),
                segment_id=segment_id,
            )
        )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "physics_core": "ball_arc_solver._rk4_step",
        "physics_parameters": phys.summary(),
        "magnus": {
            "enabled": abs(cfg.spin_scalar) > 1e-12,
            "spin_scalar": float(cfg.spin_scalar),
            "fixed_axis": _list3(spin_axis),
            "cl": _round(STEYN_CL_PER_SPIN * cfg.spin_scalar, 9),
            "source": "Steyn arXiv:2501.00163 Cl=0.195*S",
            "solver_modified": False,
        },
        "config": cfg.to_json(),
    }
    return SimulatedTrajectory(shot=shot, samples=tuple(samples), bounces=tuple(bounces), metadata=metadata)


def generate_trajectory_pair(
    *,
    trajectory_id: str,
    rng: np.random.Generator,
    calibration: LoadedCourtCalibration,
    family: str | None = None,
    shot: ShotInitialState | None = None,
    config: FlightSimulationConfig | None = None,
    physics: PhysicsParameters | None = None,
    noise_profile: DetectorNoiseProfile | None = None,
    clean_only: bool = False,
) -> dict[str, Any]:
    if shot is None:
        families = tuple(SHOT_FAMILY_ASSUMPTIONS)
        selected_family = family or str(families[int(rng.integers(0, len(families)))])
        shot = sample_shot_family(selected_family, rng, start_side="near")
    cfg = config or FlightSimulationConfig(spin_scalar=float(rng.uniform(-0.8, 0.8)))
    phys = physics or PhysicsParameters.for_ball_type(cfg.ball_type)
    trajectory = simulate_flight(shot, physics=phys, config=cfg)
    clean_track = _project_clean_track(trajectory, calibration)
    noisy = (
        {"detections": [], "dropped_frames": [], "spurious_detections": [], "profile": (noise_profile or DetectorNoiseProfile()).to_json()}
        if clean_only
        else apply_detector_noise(
            clean_track,
            rng=rng,
            image_size=_image_size(calibration.model),
            profile=noise_profile or DetectorNoiseProfile(),
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "trajectory_id": trajectory_id,
        "truth_3d": trajectory.to_json(),
        "clean_2d_track": clean_track,
        "noisy_2d_detections": noisy,
        "projection": {
            "schema": "CourtCalibration",
            "calibration_path": str(calibration.path or ""),
            "image_size": list(_image_size(calibration.model)),
            "projector": "court_calibration.project_world_points",
        },
    }


def apply_detector_noise(
    clean_track: Sequence[Mapping[str, Any]],
    *,
    rng: np.random.Generator,
    image_size: tuple[int, int],
    profile: DetectorNoiseProfile | None = None,
) -> dict[str, Any]:
    noise = profile or DetectorNoiseProfile()
    visible_indices = [idx for idx, frame in enumerate(clean_track) if bool(frame.get("visible"))]
    target_tp = int(round(float(noise.recall) * len(visible_indices)))
    target_tp = max(0, min(len(visible_indices), target_tp))
    visible_scores = rng.random(len(visible_indices)) if visible_indices else np.asarray([])
    selected_visible = {
        visible_indices[int(idx)]
        for idx in np.argsort(visible_scores)[:target_tp]
    }

    raw_jitter = rng.normal(0.0, 1.0, size=(target_tp, 2)) if target_tp else np.zeros((0, 2))
    radial = np.linalg.norm(raw_jitter, axis=1) if target_tp else np.asarray([])
    p95 = float(np.percentile(radial, 95)) if radial.size else 1.0
    scale = float(noise.p95_jitter_px) / p95 if p95 > 1e-12 else 0.0
    jitter = raw_jitter * scale
    jitter_index = 0
    detections: list[dict[str, Any]] = []
    dropped: list[int] = []
    for index, frame in enumerate(clean_track):
        if not bool(frame.get("visible")):
            continue
        if index not in selected_visible:
            dropped.append(int(frame["frame"]))
            continue
        xy = [float(value) for value in frame["xy_px"]]
        offset = jitter[jitter_index]
        jitter_index += 1
        detections.append(
            {
                "frame": int(frame["frame"]),
                "t": float(frame["t"]),
                "xy_px": [_round(xy[0] + float(offset[0]), 6), _round(xy[1] + float(offset[1]), 6)],
                "confidence": _round(float(rng.uniform(0.35, 0.99)), 6),
                "kind": "true_positive",
                "matched_clean_frame": int(frame["frame"]),
            }
        )

    fp_count = int(round(float(noise.hidden_fp_rate) * len(clean_track)))
    fp_count = max(0, fp_count)
    fp_frames = rng.choice(len(clean_track), size=fp_count, replace=False) if fp_count and clean_track else []
    spurious: list[dict[str, Any]] = []
    width, height = image_size
    for raw_index in sorted(int(value) for value in fp_frames):
        frame = clean_track[raw_index]
        detection = {
            "frame": int(frame["frame"]),
            "t": float(frame["t"]),
            "xy_px": [_round(float(rng.uniform(0.0, width)), 6), _round(float(rng.uniform(0.0, height)), 6)],
            "confidence": _round(float(rng.uniform(0.20, 0.80)), 6),
            "kind": "hidden_false_positive",
            "matched_clean_frame": None,
        }
        spurious.append(detection)
        detections.append(detection)
    detections.sort(key=lambda item: (int(item["frame"]), str(item["kind"])))
    return {
        "profile": noise.to_json(),
        "detections": detections,
        "dropped_frames": dropped,
        "spurious_detections": spurious,
    }


def detector_noise_stats(
    clean_track: Sequence[Mapping[str, Any]],
    noisy: Mapping[str, Any],
) -> dict[str, Any]:
    clean_by_frame = {int(frame["frame"]): frame for frame in clean_track if bool(frame.get("visible"))}
    detections = list(noisy.get("detections") or [])
    tp = [det for det in detections if det.get("kind") == "true_positive"]
    fp = [det for det in detections if det.get("kind") == "hidden_false_positive"]
    errors: list[float] = []
    for det in tp:
        frame = clean_by_frame.get(int(det["matched_clean_frame"]))
        if frame is None:
            continue
        clean_xy = [float(value) for value in frame["xy_px"]]
        det_xy = [float(value) for value in det["xy_px"]]
        errors.append(math.hypot(det_xy[0] - clean_xy[0], det_xy[1] - clean_xy[1]))
    profile = DetectorNoiseProfile(**dict(noisy.get("profile") or DetectorNoiseProfile().to_json()))
    stats = {
        "visible_clean_frames": len(clean_by_frame),
        "true_positive_count": len(tp),
        "hidden_fp_count": len(fp),
        "jitter_p95_px": _round(float(np.percentile(errors, 95)) if errors else 0.0, 6),
        "recall": _round(len(tp) / len(clean_by_frame) if clean_by_frame else 0.0, 9),
        "hidden_fp_rate": _round(len(fp) / len(clean_track) if clean_track else 0.0, 9),
        "target_profile": profile.to_json(),
    }
    stats["within_20_percent"] = _noise_within_tolerance(stats, profile, rel=0.20)
    return stats


def build_flight_sanity_artifact(record: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(record["truth_3d"]["samples"])
    bounces = list(record["truth_3d"].get("bounces") or [])
    anchors = [_anchor_payload("start", "contact", samples[0])]
    for bounce in bounces:
        anchors.append(
            {
                "anchor_id": f"bounce_{int(bounce['bounce_index']):02d}",
                "kind": "bounce",
                "t": float(bounce["t"]),
                "frame": int(bounce["frame"]),
                "world_xyz": [float(value) for value in bounce["world_xyz_m"]],
                "sigma_m": 0.05,
                "status": "simulated_bounce",
            }
        )
    anchors.append(_anchor_payload("end", "endpoint", samples[-1]))
    anchors = sorted(anchors, key=lambda item: (float(item["t"]), int(item["frame"])))
    frames = [
        {
            "t": float(sample["t"]),
            "visible": True,
            "world_xyz": [float(value) for value in sample["world_xyz_m"]],
            "band": "anchored_measured",
            "arc_solver": {"segment_id": int(sample.get("segment_id", 0))},
        }
        for sample in samples
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_arc_solved",
        "clip_id": str(record.get("trajectory_id") or "synthetic"),
        "fps": float(record["truth_3d"]["metadata"]["config"]["fps"]),
        "status": "ran",
        "anchors": anchors,
        "frames": frames,
        "config": {"court_sport": "pickleball", "court_margin_m": 4.0, "court_z_min_m": -0.15},
        "summary": {
            "anchored_measured_count": len(frames),
            "arc_interpolated_count": 0,
            "arc_extrapolated_count": 0,
            "arc_weak_count": 0,
            "hidden_count": 0,
        },
    }


def evaluate_simulated_flight_sanity(record: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_ball_flight_sanity(build_flight_sanity_artifact(record))


def round_trip_fit_report(
    record: Mapping[str, Any],
    calibration: LoadedCourtCalibration,
    *,
    physics: PhysicsParameters | None = None,
    solver_config: BallArcSolverConfig | None = None,
) -> dict[str, Any]:
    phys = physics or PhysicsParameters.for_ball_type(str(record["truth_3d"]["metadata"]["config"]["ball_type"]))
    cfg = solver_config or BallArcSolverConfig(max_reprojection_inlier_px=18.0, robust_pixel_sigma=6.0)
    samples = list(record["truth_3d"]["samples"])
    first_segment_id = int(samples[0].get("segment_id", 0))
    segment_samples = [sample for sample in samples if int(sample.get("segment_id", 0)) == first_segment_id]
    if len(segment_samples) < 3:
        return {"status": "blocked", "reason": "insufficient_segment_samples", "sample_count": len(segment_samples)}
    clean_by_frame = {int(frame["frame"]): frame for frame in record["clean_2d_track"] if bool(frame.get("visible"))}
    observations = [
        BallObservation(
            frame=int(sample["frame"]),
            t=float(sample["t"]),
            xy=tuple(float(value) for value in clean_by_frame[int(sample["frame"])]["xy_px"]),  # type: ignore[arg-type]
            confidence=1.0,
            visible=True,
        )
        for sample in segment_samples
        if int(sample["frame"]) in clean_by_frame
    ]
    if len(observations) < 3:
        return {"status": "blocked", "reason": "insufficient_visible_observations", "sample_count": len(observations)}
    start = _anchor_event("sim_start", "contact", segment_samples[0])
    end_kind = "bounce" if _segment_ends_at_bounce(record, segment_samples[-1]) else "endpoint"
    end = _anchor_event("sim_end", end_kind, segment_samples[-1])
    fit = fit_flight_segment(
        segment_id=0,
        start_anchor=start,
        end_anchor=end,
        observations=observations,
        calibration=calibration.payload,
        physics=phys,
        config=cfg,
    )
    errors: list[float] = []
    truth_by_t = {float(sample["t"]): tuple(float(value) for value in sample["world_xyz_m"]) for sample in segment_samples}
    for obs in observations:
        predicted = fit.predict(obs.t, phys, cfg)
        truth = truth_by_t.get(float(obs.t))
        if truth is None:
            continue
        errors.append(_distance3(predicted, truth))
    return {
        "status": fit.status,
        "sample_count": len(errors),
        "position_error_m": _error_summary(errors),
        "fit": {
            "status": fit.status,
            "initial_velocity_mps": _list3(fit.initial_velocity_mps),
            "reprojection_rmse_px": fit.reprojection_rmse_px,
            "endpoint_error_m": fit.endpoint_error_m,
            "inlier_count": fit.inlier_count,
            "outlier_count": fit.outlier_count,
        },
    }


def generate_corpus(
    *,
    count: int,
    seed: int,
    calibration: LoadedCourtCalibration,
    roundtrip_samples: int = 10,
    noise_profile: DetectorNoiseProfile | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    records = [
        generate_trajectory_pair(
            trajectory_id=f"sim_{idx:06d}",
            rng=rng,
            calibration=calibration,
            noise_profile=noise_profile or DetectorNoiseProfile(),
        )
        for idx in range(count)
    ]
    sanity_reports = [evaluate_simulated_flight_sanity(record) for record in records]
    failed_segments = sum(int(report["summary"]["failed_segment_count"]) for report in sanity_reports)
    demoted_frames = sum(int(report["summary"]["demoted_frame_count"]) for report in sanity_reports)
    noise_stats = _aggregate_noise_stats(records, noise_profile or DetectorNoiseProfile())
    roundtrip: list[dict[str, Any]] = []
    for record in records[: max(0, min(roundtrip_samples, len(records)))]:
        roundtrip.append(round_trip_fit_report(record, calibration))
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": CORPUS_ARTIFACT_TYPE,
        "deterministic_seed": int(seed),
        "trajectory_count": int(count),
        "calibration_path": str(calibration.path or ""),
        "acceptance": {
            "flight_sanity": {
                "failed_segments": failed_segments,
                "demoted_frames": demoted_frames,
                "passed": failed_segments == 0 and demoted_frames == 0,
            },
            "noise_profile": noise_stats,
        },
        "round_trip": {
            "samples_evaluated": len(roundtrip),
            "reports": roundtrip,
            "position_error_m": _combine_roundtrip_errors(roundtrip),
        },
        "performance": {
            "trajectory_count": int(count),
            "wall_seconds": _round(elapsed, 6),
            "trajectories_per_second": _round(count / elapsed if elapsed > 0.0 else 0.0, 6),
            "projected_1000_trajectory_seconds": _round((elapsed / count) * 1000.0 if count else 0.0, 6),
            "under_60s_for_1000_projected": bool(count and (elapsed / count) * 1000.0 < 60.0),
        },
        "assumptions": {
            "shot_families": SHOT_FAMILY_ASSUMPTIONS,
            "bounce": BounceParameters().to_json(),
            "protected_eval_policy": "uses eval calibration JSON as projection fixture only; does not touch labels",
            "mujoco": "not used in Phase 1",
        },
    }
    return records, report


def _project_clean_track(
    trajectory: SimulatedTrajectory,
    calibration: LoadedCourtCalibration,
) -> list[dict[str, Any]]:
    width, height = _image_size(calibration.model)
    positions = [sample.position_m for sample in trajectory.samples]
    projected = project_world_points(calibration.model.extrinsics, calibration.model.intrinsics, positions)
    frames: list[dict[str, Any]] = []
    for sample, xy in zip(trajectory.samples, projected, strict=True):
        x, y = float(xy[0]), float(xy[1])
        visible = bool(math.isfinite(x) and math.isfinite(y) and 0.0 <= x < width and 0.0 <= y < height)
        frames.append(
            {
                "frame": int(sample.frame),
                "t": _round(sample.t, 9),
                "xy_px": [_round(x, 6), _round(y, 6)],
                "visible": visible,
                "world_xyz_m": _list3(sample.position_m),
                "segment_id": int(sample.segment_id),
            }
        )
    return frames


def _rk4_step_with_magnus(
    state: tuple[float, float, float, float, float, float],
    dt: float,
    physics: PhysicsParameters,
    spin_axis: tuple[float, float, float],
    spin_scalar: float,
) -> tuple[float, float, float, float, float, float]:
    def deriv(s: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
        vx, vy, vz = s[3], s[4], s[5]
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        drag = physics.drag_k_per_m * speed
        ax = -drag * vx
        ay = -drag * vy
        az = -physics.gravity_mps2 - drag * vz
        if speed > 1e-9:
            lift_dir = _unit(_cross(spin_axis, (vx / speed, vy / speed, vz / speed)))
            if lift_dir is not None:
                area = math.pi * physics.radius_m * physics.radius_m
                lift_k = 0.5 * physics.rho_air_kg_m3 * area / physics.mass_kg
                lift_acc = lift_k * (speed * speed) * (STEYN_CL_PER_SPIN * spin_scalar)
                ax += lift_acc * lift_dir[0]
                ay += lift_acc * lift_dir[1]
                az += lift_acc * lift_dir[2]
        return (vx, vy, vz, ax, ay, az)

    k1 = deriv(state)
    k2 = deriv(_add_state(state, _scale_state(k1, dt / 2.0)))
    k3 = deriv(_add_state(state, _scale_state(k2, dt / 2.0)))
    k4 = deriv(_add_state(state, _scale_state(k3, dt)))
    return tuple(state[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0 for i in range(6))  # type: ignore[return-value]


def _aggregate_noise_stats(records: Sequence[Mapping[str, Any]], profile: DetectorNoiseProfile) -> dict[str, Any]:
    errors: list[float] = []
    visible = 0
    tp_count = 0
    fp_count = 0
    frame_count = 0
    for record in records:
        clean = list(record["clean_2d_track"])
        noisy = record["noisy_2d_detections"]
        frame_count += len(clean)
        clean_by_frame = {int(frame["frame"]): frame for frame in clean if bool(frame.get("visible"))}
        visible += len(clean_by_frame)
        for detection in noisy.get("detections") or []:
            if detection.get("kind") == "hidden_false_positive":
                fp_count += 1
                continue
            if detection.get("kind") != "true_positive":
                continue
            matched = clean_by_frame.get(int(detection["matched_clean_frame"]))
            if matched is None:
                continue
            tp_count += 1
            clean_xy = [float(value) for value in matched["xy_px"]]
            det_xy = [float(value) for value in detection["xy_px"]]
            errors.append(math.hypot(det_xy[0] - clean_xy[0], det_xy[1] - clean_xy[1]))
    stats = {
        "visible_clean_frames": visible,
        "true_positive_count": tp_count,
        "hidden_fp_count": fp_count,
        "jitter_p95_px": _round(float(np.percentile(errors, 95)) if errors else 0.0, 6),
        "recall": _round(tp_count / visible if visible else 0.0, 9),
        "hidden_fp_rate": _round(fp_count / frame_count if frame_count else 0.0, 9),
        "target_profile": profile.to_json(),
    }
    stats["within_20_percent"] = _noise_within_tolerance(stats, profile, rel=0.20)
    return stats


def _noise_within_tolerance(stats: Mapping[str, Any], profile: DetectorNoiseProfile, *, rel: float) -> bool:
    checks = [
        _relative_close(float(stats["jitter_p95_px"]), profile.p95_jitter_px, rel),
        _relative_close(float(stats["recall"]), profile.recall, rel),
        _relative_close(float(stats["hidden_fp_rate"]), profile.hidden_fp_rate, rel),
    ]
    return all(checks)


def _relative_close(value: float, target: float, rel: float) -> bool:
    if target == 0.0:
        return abs(value) <= rel
    return abs(value - target) <= abs(target) * rel


def _anchor_payload(anchor_id: str, kind: str, sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_id": anchor_id,
        "kind": kind,
        "t": float(sample["t"]),
        "frame": int(sample["frame"]),
        "world_xyz": [float(value) for value in sample["world_xyz_m"]],
        "sigma_m": 0.05,
        "status": "simulated",
    }


def _anchor_event(anchor_id: str, kind: str, sample: Mapping[str, Any]) -> AnchorEvent:
    return AnchorEvent(
        anchor_id=anchor_id,
        kind=kind,
        t=float(sample["t"]),
        frame=int(sample["frame"]),
        world_xyz=tuple(float(value) for value in sample["world_xyz_m"]),  # type: ignore[arg-type]
        sigma_m=0.02,
        status="simulated",
        immovable=True,
        source="flight_simulator_truth",
    )


def _segment_ends_at_bounce(record: Mapping[str, Any], sample: Mapping[str, Any]) -> bool:
    frame = int(sample["frame"])
    return any(abs(int(bounce["frame"]) - frame) <= 1 for bounce in record["truth_3d"].get("bounces") or [])


def _combine_roundtrip_errors(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    p95s = [
        float(report["position_error_m"]["p95"])
        for report in reports
        if isinstance(report.get("position_error_m"), Mapping)
    ]
    return _error_summary(p95s)


def _error_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "mean": _round(float(np.mean(finite)), 6),
        "p50": _round(float(np.percentile(finite, 50)), 6),
        "p95": _round(float(np.percentile(finite, 95)), 6),
        "max": _round(max(finite), 6),
    }


def _image_size(calibration: CourtCalibration) -> tuple[int, int]:
    if calibration.image_size is not None:
        return int(calibration.image_size[0]), int(calibration.image_size[1])
    return int(round(calibration.intrinsics.cx * 2.0)), int(round(calibration.intrinsics.cy * 2.0))


def _interpolate_impact(
    previous: tuple[float, float, float, float, float, float],
    current: tuple[float, float, float, float, float, float],
    z: float,
) -> tuple[tuple[float, float, float, float, float, float], float]:
    denom = previous[2] - current[2]
    alpha = 0.0 if abs(denom) <= 1e-12 else max(0.0, min(1.0, (previous[2] - z) / denom))
    state = tuple(previous[idx] + alpha * (current[idx] - previous[idx]) for idx in range(6))
    return (state[0], state[1], z, state[3], state[4], state[5]), alpha  # type: ignore[return-value]


def _bounce_velocity(velocity: tuple[float, float, float], bounce: BounceParameters) -> tuple[float, float, float]:
    horizontal_scale = max(0.0, 1.0 - float(bounce.friction))
    return (
        velocity[0] * horizontal_scale,
        velocity[1] * horizontal_scale,
        -float(bounce.restitution) * velocity[2],
    )


def _outside_loose_court(
    state: tuple[float, float, float, float, float, float],
    margin_m: float,
) -> bool:
    half_width = 3.048 + float(margin_m)
    half_length = 6.7056 + float(margin_m)
    return abs(state[0]) > half_width or abs(state[1]) > half_length or state[2] > 8.0


def _spin_axis_for_velocity(velocity: tuple[float, float, float]) -> tuple[float, float, float]:
    vx, vy = velocity[0], velocity[1]
    norm = math.hypot(vx, vy)
    if norm <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (vy / norm, -vx / norm, 0.0)


def _add_state(
    state: tuple[float, float, float, float, float, float],
    delta: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    return tuple(state[i] + delta[i] for i in range(6))  # type: ignore[return-value]


def _scale_state(
    state: tuple[float, float, float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float, float, float]:
    return tuple(value * scale for value in state)  # type: ignore[return-value]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = _norm3(v)
    if norm <= 1e-12:
        return None
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _norm3(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]) + float(v[2]) * float(v[2]))


def _distance3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[idx]) - float(b[idx])) ** 2 for idx in range(3)))


def _vec3(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("expected 3 values")
    return (float(value[0]), float(value[1]), float(value[2]))


def _list3(value: Sequence[float]) -> list[float]:
    return [_round(float(value[0]), 9), _round(float(value[1]), 9), _round(float(value[2]), 9)]


def _round(value: float, digits: int = 6) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    rounded = round(float(value), digits)
    return 0.0 if abs(rounded) < 10 ** (-(digits + 1)) else rounded
