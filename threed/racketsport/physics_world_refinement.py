"""Bridge court_Z0 virtual-world artifacts into physics-refinement scaffolds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .physics_refine import (
    ContactWindow,
    FootContactSample,
    MotionRefinementRequest,
    PlayerRootSample,
    choose_execution_plan,
    package_refinement_artifact,
    select_refinement_windows,
    summarize_constraints,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_physics_refinement"
FOOT_OFFSET_M = 0.08
DEFAULT_PLAYER_RADIUS_M = 0.35


def build_physics_refinement_from_virtual_world(
    *,
    clip_id: str,
    virtual_world: Mapping[str, Any],
    requested_mode: str = "auto",
    mjx_available: bool = False,
    pad_frames: int = 3,
) -> dict[str, Any]:
    """Build a runnable physics-refinement artifact from a virtual world.

    This uses the existing deterministic CPU scaffold. It does not claim that
    MuJoCo, MJX, PhysPT, PHC/PULSE, or MultiPhys simulation ran.
    """

    players = _players(virtual_world)
    fps = _fps(virtual_world)
    player_ids = tuple(str(player["id"]) for player in players)
    total_frames = _total_frames(virtual_world, fps=fps)
    request = MotionRefinementRequest(
        clip_id=clip_id,
        frame_rate_hz=fps,
        total_frames=total_frames,
        player_ids=player_ids,
        requested_mode=requested_mode,
    )
    foot_samples = _foot_samples(players, fps=fps)
    root_samples = _root_samples(players, fps=fps)
    raw_windows = _contact_windows(foot_samples)
    windows = select_refinement_windows(raw_windows, total_frames=total_frames, pad_frames=pad_frames)
    summary = summarize_constraints(foot_samples, root_samples, floor_z_m=0.0)
    plan = choose_execution_plan(request, mjx_available=mjx_available)
    artifact = package_refinement_artifact(request, windows, summary, plan)
    artifact.update(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "source": "virtual_world_floor_constraints",
            "source_summary": {
                "floor_placed_player_frame_count": int(
                    _summary(virtual_world).get("floor_placed_player_frame_count", len(root_samples))
                ),
                "mesh_player_frame_count": int(_summary(virtual_world).get("mesh_player_frame_count", 0)),
                "foot_sample_count": len(foot_samples),
                "root_sample_count": len(root_samples),
            },
        }
    )
    return artifact


def build_physics_refinement_from_file(
    *,
    clip_id: str,
    virtual_world_path: str | Path,
    requested_mode: str = "auto",
    mjx_available: bool = False,
    pad_frames: int = 3,
) -> dict[str, Any]:
    payload = json.loads(Path(virtual_world_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("virtual world payload must be a JSON object")
    return build_physics_refinement_from_virtual_world(
        clip_id=clip_id,
        virtual_world=payload,
        requested_mode=requested_mode,
        mjx_available=mjx_available,
        pad_frames=pad_frames,
    )


def write_physics_refinement(path: str | Path, artifact: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _players(virtual_world: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    players = virtual_world.get("players")
    if not isinstance(players, list) or not players:
        raise ValueError("virtual_world.players must contain at least one player")
    return [player for player in players if isinstance(player, Mapping)]


def _summary(virtual_world: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = virtual_world.get("summary")
    return summary if isinstance(summary, Mapping) else {}


def _fps(virtual_world: Mapping[str, Any]) -> float:
    fps = float(virtual_world.get("fps", 30.0))
    if fps <= 0.0:
        raise ValueError("virtual_world.fps must be positive")
    return fps


def _total_frames(virtual_world: Mapping[str, Any], *, fps: float) -> int:
    indices = [
        _frame_index(frame, fps=fps)
        for player in _players(virtual_world)
        for frame in _frames(player)
    ]
    ball = virtual_world.get("ball")
    if isinstance(ball, Mapping):
        ball_frames = ball.get("frames")
        if isinstance(ball_frames, list):
            indices.extend(_frame_index(frame, fps=fps) for frame in ball_frames if isinstance(frame, Mapping))
    return max(indices, default=0) + 1


def _frames(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = player.get("frames")
    if not isinstance(frames, list):
        return []
    return [frame for frame in frames if isinstance(frame, Mapping)]


def _foot_samples(players: list[Mapping[str, Any]], *, fps: float) -> tuple[FootContactSample, ...]:
    samples: list[FootContactSample] = []
    for player in players:
        player_id = str(player["id"])
        for frame in _frames(player):
            floor = _vec3(frame.get("floor_world_xyz"))
            contact = frame.get("foot_contact")
            if floor is None or not isinstance(contact, Mapping):
                continue
            z = _sample_floor_z(frame, floor_z=floor[2])
            frame_index = _frame_index(frame, fps=fps)
            for foot, x_offset in (("left", -FOOT_OFFSET_M), ("right", FOOT_OFFSET_M)):
                samples.append(
                    FootContactSample(
                        frame_index=frame_index,
                        player_id=player_id,
                        foot=foot,
                        position_xyz=(floor[0] + x_offset, floor[1], z),
                        contact=bool(contact.get(foot, False)),
                    )
                )
    return tuple(samples)


def _root_samples(players: list[Mapping[str, Any]], *, fps: float) -> tuple[PlayerRootSample, ...]:
    roots: list[PlayerRootSample] = []
    for player in players:
        player_id = str(player["id"])
        for frame in _frames(player):
            root = _vec3(frame.get("transl_world")) or _vec3(frame.get("floor_world_xyz"))
            if root is None:
                continue
            roots.append(
                PlayerRootSample(
                    frame_index=_frame_index(frame, fps=fps),
                    player_id=player_id,
                    center_xyz=(root[0], root[1], root[2]),
                    radius_m=DEFAULT_PLAYER_RADIUS_M,
                )
            )
    return tuple(roots)


def _contact_windows(samples: tuple[FootContactSample, ...]) -> tuple[ContactWindow, ...]:
    windows: list[ContactWindow] = []
    active: dict[tuple[str, str], int] = {}
    ordered = sorted(samples, key=lambda sample: (sample.player_id, sample.foot, sample.frame_index))
    previous: dict[tuple[str, str], int] = {}

    for sample in ordered:
        key = (sample.player_id, sample.foot)
        last = previous.get(key)
        if sample.contact and key not in active:
            active[key] = sample.frame_index
        elif sample.contact and last is not None and sample.frame_index > last + 1:
            windows.append(_window_from_active(key, active.pop(key), last))
            active[key] = sample.frame_index
        elif not sample.contact and key in active:
            windows.append(_window_from_active(key, active.pop(key), last if last is not None else sample.frame_index))
        previous[key] = sample.frame_index

    for key, start in active.items():
        windows.append(_window_from_active(key, start, previous[key]))
    return tuple(windows)


def _window_from_active(key: tuple[str, str], start: int, end: int) -> ContactWindow:
    player_id, foot = key
    return ContactWindow(start_frame=start, end_frame=end, player_id=player_id, reason=f"{foot}_foot_contact")


def _frame_index(frame: Mapping[str, Any], *, fps: float) -> int:
    return max(0, round(float(frame.get("t", 0.0)) * fps))


def _sample_floor_z(frame: Mapping[str, Any], *, floor_z: float) -> float:
    min_mesh_z = frame.get("min_mesh_z_m")
    return float(min_mesh_z) if isinstance(min_mesh_z, (int, float)) else floor_z


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


__all__ = [
    "build_physics_refinement_from_file",
    "build_physics_refinement_from_virtual_world",
    "write_physics_refinement",
]
