"""W3-REPLAY-NATIVE phase 1: bake a `racketsport_body_mesh` artifact (see
`mesh_export.py`) into a `.usdz` package via `usd-core` (the `pxr` Python
bindings for Pixar's USD -- installable from PyPI, no Xcode `usdzconvert`
required; see `scripts/racketsport/build_replay_animated_usdz.py`).

Unlike the glTF bake (`replay_glb_bake.py`), which has to fabricate morph
targets + a weights animation because glTF has no native "keyframe the raw
vertex buffer" mechanism, USD's `UsdGeom.Mesh.points` attribute natively
supports arbitrary time samples on a fixed-topology mesh -- exactly our
scheduled-frame-vertex data. So this bake sets one `points` time sample per
scheduled frame directly (constant `faceVertexCounts`/`faceVertexIndices`
from `mesh_faces`), no delta encoding needed. For a preview-size USDZ tier,
callers may provide a `max_mesh_frames` budget: the bake preserves each
scheduled window's endpoints and keeps evenly spaced interior mesh poses.

Honesty note: USD linearly interpolates an attribute's value between its
two nearest time samples (and holds the boundary value outside the sampled
range) by default -- unlike the glTF bake's deliberate STEP interpolation.
Left uncorrected, that would silently interpolate a player's mesh across
gaps between contact windows that were never computed (152 of 2288 frames
have mesh data; see `body_mesh_readiness.json`). To avoid implying motion we
never computed, this module also keyframes each mesh's `visibility`
attribute: `inherited` for the frame range actually covered by each
contiguous scheduled window, `invisible` one frame before/after -- so the
solid mesh only appears during the windows it was actually baked for, same
as the web viewer's `solidBodyMeshFramesForTime` contact-window gating.
"""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Mapping, Sequence

from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, Vt


class BodyMeshUsdzBakeError(ValueError):
    """Raised when `body_mesh.json` cannot be baked into a valid USDZ."""


def build_animated_body_usdz(
    body_mesh: Mapping[str, Any],
    *,
    clip: str,
    out_path: str | Path,
    max_mesh_frames: int | None = None,
    round_decimals: int | None = None,
) -> dict[str, Any]:
    """Write `out_path` as a `.usdz` package baking every player in
    `body_mesh` (the `racketsport_body_mesh` artifact) as a time-sampled,
    fixed-topology mesh. Returns a small summary dict. Raises
    `BodyMeshUsdzBakeError` on invalid input.

    `max_mesh_frames` is a lossy preview control. It budgets the total number
    of authored mesh point samples across all players and windows while keeping
    every window's first/last sample, so visibility gating still spans the
    originally computed windows. `round_decimals` rounds authored point values
    before converting them to USD float32 arrays; it is precision quantization,
    not a guaranteed USDZ byte-size reduction.
    """

    mesh_faces = _mesh_faces(body_mesh)
    if not mesh_faces:
        raise BodyMeshUsdzBakeError("body_mesh.mesh_faces is empty; nothing to bake")
    players = [p for p in body_mesh.get("players", []) if isinstance(p, Mapping) and p.get("frames")]
    if not players:
        raise BodyMeshUsdzBakeError("body_mesh.players has no frames to bake")
    if max_mesh_frames is not None and max_mesh_frames < 1:
        raise BodyMeshUsdzBakeError("max_mesh_frames must be >= 1 when provided")
    if round_decimals is not None and round_decimals < 0:
        raise BodyMeshUsdzBakeError("round_decimals must be >= 0 when provided")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `.usdc` (binary Crate format) rather than `.usda` (ASCII text): the same
    # per-vertex-float-as-JSON-text size blowup that made the raw smpl_motion.json
    # pull impractical (see build_body_mesh_export.py) applies equally to USD's
    # ASCII layer format -- Crate stores floats as binary and compresses, and is
    # what real USDZ tooling ships (ASCII .usda is a human-authoring/debug format).
    usd_path = out_path.with_suffix(".usdc")

    fps = float(body_mesh.get("fps", 30.0)) or 30.0
    face_vertex_counts = Vt.IntArray([3] * len(mesh_faces))
    face_vertex_indices = Vt.IntArray([index for face in mesh_faces for index in face])

    stage = Usd.Stage.CreateNew(str(usd_path))
    stage.SetTimeCodesPerSecond(fps)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)  # world_frame is court_Z0 (Z-up)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)  # court_Z0 positions are already meters
    root = UsdGeom.Xform.Define(stage, f"/{_sanitize(clip)}")
    stage.SetDefaultPrim(root.GetPrim())

    player_summaries: list[dict[str, Any]] = []
    min_frame_idx: int | None = None
    max_frame_idx: int | None = None

    prepared_players: list[tuple[Mapping[str, Any], list[list[Mapping[str, Any]]]]] = []
    source_mesh_frame_count = 0
    for player in players:
        frames = sorted(
            (f for f in player.get("frames", []) if isinstance(f, Mapping) and f.get("mesh_vertices_world")),
            key=lambda f: int(f.get("frame_idx", 0)),
        )
        if not frames:
            continue
        source_mesh_frame_count += len(frames)
        prepared_players.append((player, _contiguous_windows(frames)))
    if not prepared_players:
        raise BodyMeshUsdzBakeError("body_mesh.players has no frames with mesh_vertices_world to bake")
    selected_players = _budget_player_windows(prepared_players, max_mesh_frames)

    baked_mesh_frame_count = 0
    for player, selected_windows in selected_players:
        player_id = int(player.get("id", 0))
        frames = [frame for window in selected_windows for frame in window]
        vertex_count = len(_vertices(frames[0], round_decimals=round_decimals))
        mesh = UsdGeom.Mesh.Define(stage, f"/{_sanitize(clip)}/player{player_id}_body")
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
        points_attr = mesh.CreatePointsAttr()
        visibility_attr = UsdGeom.Imageable(mesh).CreateVisibilityAttr()
        visibility_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode.Default())
        # USD holds an attribute's value constant *outside* its sampled time range
        # (using whichever sample is nearest). Without an explicit "invisible" sample
        # strictly before the first window, USD would hold the *first* window's
        # "inherited" sample backwards across the entire preceding timeline, making
        # the mesh appear visible (and frozen in its first baked pose) from the start
        # of the clip -- not honest given we never computed those frames.
        visibility_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(int(frames[0].get("frame_idx", 0)) - 1))

        for window_frames in selected_windows:
            for frame in window_frames:
                frame_idx = int(frame.get("frame_idx", 0))
                vertices = _vertices(frame, round_decimals=round_decimals)
                if len(vertices) != vertex_count:
                    raise BodyMeshUsdzBakeError(
                        f"player {player_id} frame_idx={frame_idx} has {len(vertices)} vertices, "
                        f"expected {vertex_count} (from frame_idx={frames[0].get('frame_idx')})"
                    )
                points_attr.Set(
                    Vt.Vec3fArray([Gf.Vec3f(*vertex) for vertex in vertices]),
                    Usd.TimeCode(frame_idx),
                )
                min_frame_idx = frame_idx if min_frame_idx is None else min(min_frame_idx, frame_idx)
                max_frame_idx = frame_idx if max_frame_idx is None else max(max_frame_idx, frame_idx)
                baked_mesh_frame_count += 1

            window_start = int(window_frames[0].get("frame_idx", 0))
            window_end = int(window_frames[-1].get("frame_idx", 0))
            visibility_attr.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(window_start))
            visibility_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(window_end + 1))

        player_summaries.append(
            {
                "id": player_id,
                "frame_count": len(frames),
                "window_count": len(selected_windows),
                "vertex_count": vertex_count,
            }
        )

    if min_frame_idx is not None and max_frame_idx is not None:
        stage.SetStartTimeCode(min_frame_idx)
        stage.SetEndTimeCode(max_frame_idx)
    stage.GetRootLayer().Save()

    ok = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(usd_path)), str(out_path))
    if not ok:
        raise BodyMeshUsdzBakeError(f"UsdUtils.CreateNewUsdzPackage failed for {usd_path} -> {out_path}")

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh_usdz_bake",
        "clip": clip,
        "out": str(out_path),
        "out_bytes": out_path.stat().st_size,
        "fps": fps,
        "mesh_faces_count": len(mesh_faces),
        "source_mesh_frame_count": source_mesh_frame_count,
        "baked_mesh_frame_count": baked_mesh_frame_count,
        "compression_profile": {
            "max_mesh_frames": max_mesh_frames,
            "round_decimals": round_decimals,
            "window_endpoints_preserved": True,
            "selection": "proportional_even_per_window" if max_mesh_frames is not None else "all_frames",
        },
        "players": player_summaries,
        "frame_range": [min_frame_idx, max_frame_idx],
    }


def _budget_player_windows(
    player_windows: Sequence[tuple[Mapping[str, Any], list[list[Mapping[str, Any]]]]],
    max_mesh_frames: int | None,
) -> list[tuple[Mapping[str, Any], list[list[Mapping[str, Any]]]]]:
    if max_mesh_frames is None:
        return [(player, [list(window) for window in windows]) for player, windows in player_windows]

    flat_windows: list[list[Mapping[str, Any]]] = [window for _, windows in player_windows for window in windows]
    source_frame_count = sum(len(window) for window in flat_windows)
    if max_mesh_frames >= source_frame_count:
        return [(player, [list(window) for window in windows]) for player, windows in player_windows]

    min_frame_count = sum(min(2, len(window)) for window in flat_windows)
    if max_mesh_frames < min_frame_count:
        raise BodyMeshUsdzBakeError(
            f"max_mesh_frames={max_mesh_frames} is too small to preserve endpoints for "
            f"{len(flat_windows)} scheduled windows; minimum is {min_frame_count}"
        )

    allocations = _allocate_window_frame_budget(flat_windows, max_mesh_frames)
    selected: list[tuple[Mapping[str, Any], list[list[Mapping[str, Any]]]]] = []
    allocation_index = 0
    for player, windows in player_windows:
        selected_windows: list[list[Mapping[str, Any]]] = []
        for window in windows:
            keep_count = allocations[allocation_index]
            allocation_index += 1
            indices = _evenly_spaced_indices(len(window), keep_count)
            selected_windows.append([window[index] for index in indices])
        selected.append((player, selected_windows))
    return selected


def _allocate_window_frame_budget(windows: Sequence[Sequence[Mapping[str, Any]]], max_mesh_frames: int) -> list[int]:
    minimums = [min(2, len(window)) for window in windows]
    capacities = [len(window) - minimum for window, minimum in zip(windows, minimums)]
    remaining = max_mesh_frames - sum(minimums)
    if remaining <= 0 or sum(capacities) == 0:
        return minimums

    capacity_sum = sum(capacities)
    fractional_extras = [remaining * capacity / capacity_sum for capacity in capacities]
    extras = [math.floor(extra) for extra in fractional_extras]
    unallocated = remaining - sum(extras)
    order = sorted(
        range(len(windows)),
        key=lambda idx: (fractional_extras[idx] - extras[idx], capacities[idx]),
        reverse=True,
    )
    for idx in order:
        if unallocated <= 0:
            break
        if extras[idx] < capacities[idx]:
            extras[idx] += 1
            unallocated -= 1

    idx = 0
    while unallocated > 0:
        if extras[idx] < capacities[idx]:
            extras[idx] += 1
            unallocated -= 1
        idx = (idx + 1) % len(windows)
    return [minimum + extra for minimum, extra in zip(minimums, extras)]


def _evenly_spaced_indices(frame_count: int, keep_count: int) -> list[int]:
    if keep_count >= frame_count:
        return list(range(frame_count))
    if keep_count <= 1:
        return [0]
    indices = [round(i * (frame_count - 1) / (keep_count - 1)) for i in range(keep_count)]
    indices[0] = 0
    indices[-1] = frame_count - 1
    deduped: list[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    missing = [index for index in range(frame_count) if index not in deduped]
    while len(deduped) < keep_count and missing:
        best = max(missing, key=lambda index: min(abs(index - kept) for kept in deduped))
        deduped.append(best)
        missing.remove(best)
    return sorted(deduped)


def _contiguous_windows(frames: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Group already frame_idx-sorted frames into contiguous
    source_window_index runs (falls back to grouping by contiguous
    frame_idx when source_window_index is missing)."""
    windows: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_window_index: object = object()
    previous_frame_idx: int | None = None
    for frame in frames:
        window_index = frame.get("source_window_index")
        frame_idx = int(frame.get("frame_idx", 0))
        starts_new = not current or window_index != current_window_index or (
            window_index is None and previous_frame_idx is not None and frame_idx != previous_frame_idx + 1
        )
        if starts_new:
            if current:
                windows.append(current)
            current = []
            current_window_index = window_index
        current.append(frame)
        previous_frame_idx = frame_idx
    if current:
        windows.append(current)
    return windows


def _vertices(frame: Mapping[str, Any], *, round_decimals: int | None = None) -> list[tuple[float, float, float]]:
    vertices = frame.get("mesh_vertices_world", [])
    parsed = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    if round_decimals is None:
        return parsed
    return [tuple(round(coord, round_decimals) for coord in vertex) for vertex in parsed]


def _mesh_faces(body_mesh: Mapping[str, Any]) -> list[list[int]]:
    faces = body_mesh.get("mesh_faces", [])
    parsed: list[list[int]] = []
    for face in faces:
        if not isinstance(face, (list, tuple)) or len(face) != 3:
            return []
        parsed.append([int(index) for index in face])
    return parsed


def _sanitize(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name) or "clip"


__all__ = ["build_animated_body_usdz", "BodyMeshUsdzBakeError"]
