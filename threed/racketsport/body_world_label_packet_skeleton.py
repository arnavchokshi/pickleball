"""Convert a BODY world-label review packet into a `Skeleton3D`-shaped mapping.

`body_world_label_packet.json` (see `threed.racketsport.body_world_label_packet`)
is a compact, review-focused packet of *predicted* per-frame world joints. It
is explicitly `not_ground_truth` and is not meant to be renamed into a
`skeleton3d.json`/`smpl_motion.json` artifact.

For the W3-SCRUBBER-V0 3D-world preview, the compact packet is often the only
locally-available source of BODY world joints for a run (the full
`smpl_motion.json`/mesh vertices may only exist on the compute VM). This
module reshapes the packet's `predicted_joints_world` samples into the
`Skeleton3D` artifact schema so the existing `virtual_world` builder can
render them -- it never changes their trust level, never claims they are
reviewed, and always marks the result `preview_only`.
"""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_skeleton3d"
DEFAULT_SOURCE_MODEL = "body_world_label_packet_preview"


def skeleton3d_from_body_world_label_packet(
    packet: Mapping[str, Any],
    *,
    fps: float,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> dict[str, Any]:
    """Build a `Skeleton3D`-shaped mapping from a `body_world_label_packet.json` payload.

    Raises ``ValueError`` for a non-positive fps or a packet missing a
    ``samples`` list, so a caller cannot silently render an empty/garbled
    skeleton.
    """

    if fps <= 0.0:
        raise ValueError("fps must be positive")
    samples = packet.get("samples")
    if not isinstance(samples, list):
        raise ValueError("packet.samples must be a list")

    players_by_id: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        player_id = _maybe_int(sample.get("player_id"))
        frame_index = _maybe_int(sample.get("frame_index"))
        t = _maybe_float(sample.get("t"))
        joints = sample.get("predicted_joints_world")
        if player_id is None or frame_index is None or t is None or not isinstance(joints, list) or not joints:
            continue
        joints_world = [[float(component) for component in joint] for joint in joints]
        conf = sample.get("joint_conf")
        if not isinstance(conf, list) or len(conf) != len(joints_world):
            joint_conf = [0.0] * len(joints_world)
        else:
            joint_conf = [float(value) for value in conf]
        frame = {
            "frame_idx": frame_index,
            "t": t,
            "joints_world": joints_world,
            "joint_conf": joint_conf,
        }
        track_world_xy = sample.get("track_world_xy")
        if isinstance(track_world_xy, list) and len(track_world_xy) >= 2:
            frame["transl_world"] = [float(track_world_xy[0]), float(track_world_xy[1]), 0.0]
        players_by_id.setdefault(player_id, []).append(
            frame
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "fps": float(fps),
        "world_frame": "court_Z0",
        "source_model": source_model,
        "joint_names": [str(name) for name in (packet.get("joint_names") or [])],
        "preview_only": True,
        "players": [
            {"id": player_id, "frames": sorted(frames, key=lambda frame: frame["t"])}
            for player_id, frames in sorted(players_by_id.items())
        ],
        "provenance": {
            "source_artifact": str(packet.get("artifact_type") or "racketsport_body_world_label_packet"),
            "not_ground_truth": bool(packet.get("not_ground_truth", True)),
            "trusted_for_world_mpjpe": bool(packet.get("trusted_for_world_mpjpe", False)),
            "clip": str(packet.get("clip") or ""),
        },
    }


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["skeleton3d_from_body_world_label_packet"]
