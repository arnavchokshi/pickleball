"""CPU-only helpers for ReplayScene-compatible replay export manifests."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.schemas import ReplayScene


SCHEMA_VERSION = 1
WORLD_FRAME = "court_Z0"
GLB_MAGIC = b"glTF"
GLB_VERSION = 2
REVIEW_GLB_GENERATOR = "pickleball_review_replay_export"
REVIEW_STATIC_GLB_BLOCKER = "review_static_glb_export"
MODE_POINTS = 0
MODE_LINES = 1
MODE_LINE_STRIP = 3
MODE_TRIANGLES = 4


def build_replay_export_manifest(
    *,
    export_root: str | Path,
    court_glb: str | Path,
    point_glbs: Sequence[Mapping[str, Any]],
    players: Sequence[int],
    fps: float,
) -> ReplayScene:
    """Build a validated ReplayScene manifest from already-written GLB files."""

    root = Path(export_root)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("fps must be a positive number")

    court_path = _validate_glb_ref(root, court_glb, field="court_glb")
    points: list[dict[str, Any]] = []
    for index, point in enumerate(point_glbs):
        glb_ref = _point_glb_ref(point)
        glb_path = _validate_glb_ref(root, glb_ref, field=f"points/{index}/glb_url")
        points.append(
            {
                "id": _point_int(point, "id"),
                "t0": _point_float(point, "t0"),
                "t1": _point_float(point, "t1"),
                "glb_url": glb_path.as_posix(),
                "size_mb": _size_mb(root / glb_path),
            }
        )

    return validate_replay_export_manifest(
        root,
        {
            "schema_version": SCHEMA_VERSION,
            "world_frame": WORLD_FRAME,
            "fps": float(fps),
            "court_glb": court_path.as_posix(),
            "players": sorted(_player_ids(players)),
            "points": points,
        },
    )


def build_replay_review_export_from_virtual_world(
    virtual_world: Mapping[str, Any],
    *,
    export_root: str | Path,
    scene_root: str | Path | None = None,
    point_id: int = 1,
) -> ReplayScene:
    """Write simple review GLBs from a virtual world and return a ReplayScene.

    This is a CPU-only review export. It turns the inspectable JSON world into
    static GLB traces for loading checks, not a compressed/animated production
    replay.
    """

    root = Path(export_root)
    manifest_root = Path(scene_root) if scene_root is not None else root
    root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    fps = _positive_float(virtual_world.get("fps"), field="virtual_world.fps")
    court_glb = Path("court_review.glb")
    point_glb = Path("points") / f"point_{point_id:03d}_review.glb"
    _write_review_glb(root / court_glb, _court_primitives(virtual_world))
    _write_review_glb(root / point_glb, _clip_primitives(virtual_world))
    court_manifest_ref = _relative_to_root(root / court_glb, root=manifest_root, field="court_glb")
    point_manifest_ref = _relative_to_root(root / point_glb, root=manifest_root, field="points/0/glb_url")
    t0, t1 = _time_range(virtual_world, fps=fps)
    return build_replay_export_manifest(
        export_root=manifest_root,
        court_glb=court_manifest_ref,
        point_glbs=[{"id": point_id, "t0": t0, "t1": t1, "glb_path": point_manifest_ref}],
        players=_virtual_world_player_ids(virtual_world),
        fps=fps,
    )


def write_replay_scene(path: str | Path, scene: ReplayScene | Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = scene.model_dump(mode="json") if isinstance(scene, ReplayScene) else dict(scene)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect_glb_file(path: str | Path) -> dict[str, Any]:
    """Return structural metadata for a GLB 2.0 file or raise ValueError."""

    glb_path = Path(path)
    data = glb_path.read_bytes()
    if len(data) < 4 or data[:4] != GLB_MAGIC:
        raise ValueError(f"invalid GLB magic for {glb_path}: {data[:4]!r}")
    if len(data) < 20:
        raise ValueError(f"GLB file is too small: {glb_path}")

    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if version != GLB_VERSION:
        raise ValueError(f"unsupported GLB version for {glb_path}: {version}")
    if declared_length != len(data):
        raise ValueError(
            f"GLB declared length {declared_length} does not match file size {len(data)} for {glb_path}"
        )

    offset = 12
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError(f"truncated GLB chunk header at byte {offset} for {glb_path}")
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > len(data):
            raise ValueError(f"truncated GLB chunk payload at byte {offset} for {glb_path}")
        chunks.append((chunk_type, data[offset:chunk_end]))
        offset = chunk_end

    if not chunks or chunks[0][0] != b"JSON":
        raise ValueError(f"GLB first chunk must be JSON for {glb_path}")

    json_bytes = chunks[0][1]
    try:
        gltf = json.loads(json_bytes.rstrip(b" \t\r\n\x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid GLB JSON chunk for {glb_path}: {exc}") from exc
    if not isinstance(gltf, Mapping):
        raise ValueError(f"GLB JSON chunk must contain an object for {glb_path}")

    asset = gltf.get("asset")
    if not isinstance(asset, Mapping) or str(asset.get("version", "")) != "2.0":
        raise ValueError(f"GLB asset.version must be 2.0 for {glb_path}")

    meshes = gltf.get("meshes", [])
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    materials = gltf.get("materials", [])
    buffers = gltf.get("buffers", [])
    accessors = gltf.get("accessors", [])
    animations = gltf.get("animations", [])
    skins = gltf.get("skins", [])
    extensions_used = gltf.get("extensionsUsed", [])
    primitive_count = sum(
        len(mesh.get("primitives", []))
        for mesh in meshes
        if isinstance(mesh, Mapping) and isinstance(mesh.get("primitives", []), list)
    )
    bin_chunk_bytes = sum(len(chunk_data) for chunk_type, chunk_data in chunks[1:] if chunk_type == b"BIN\x00")
    return {
        "valid": True,
        "path": str(glb_path),
        "byte_length": len(data),
        "version": version,
        "chunk_count": len(chunks),
        "json_chunk_bytes": len(json_bytes),
        "bin_chunk_bytes": bin_chunk_bytes,
        "asset_version": str(asset.get("version")),
        "generator": str(asset.get("generator", "")),
        "scene_count": len(scenes) if isinstance(scenes, list) else 0,
        "node_count": len(nodes) if isinstance(nodes, list) else 0,
        "mesh_count": len(meshes) if isinstance(meshes, list) else 0,
        "primitive_count": primitive_count,
        "material_count": len(materials) if isinstance(materials, list) else 0,
        "buffer_count": len(buffers) if isinstance(buffers, list) else 0,
        "accessor_count": len(accessors) if isinstance(accessors, list) else 0,
        "animation_count": len(animations) if isinstance(animations, list) else 0,
        "skin_count": len(skins) if isinstance(skins, list) else 0,
        "extensions_used": [str(extension) for extension in extensions_used] if isinstance(extensions_used, list) else [],
    }


def audit_replay_export_manifest(export_root: str | Path, manifest: Mapping[str, Any] | ReplayScene) -> dict[str, Any]:
    """Classify replay_scene GLBs so review exports cannot masquerade as production."""

    root = Path(export_root)
    scene = validate_replay_export_manifest(root, manifest)
    refs = [("court", scene.court_glb), *((f"point_{point.id}", point.glb_url) for point in scene.points)]
    glbs: list[dict[str, Any]] = []
    blockers: list[str] = []
    for role, ref in refs:
        info = inspect_glb_file(root / ref)
        glbs.append({"role": role, "ref": ref, **info})

    review_static = any(
        info.get("generator") == REVIEW_GLB_GENERATOR
        or Path(str(info.get("ref", ""))).name.endswith("_review.glb")
        or "replay_review" in Path(str(info.get("ref", ""))).parts
        for info in glbs
    )
    if review_static:
        blockers.append(REVIEW_STATIC_GLB_BLOCKER)

    point_glbs = [info for info in glbs if str(info.get("role", "")).startswith("point_")]
    if any(int(info.get("animation_count") or 0) <= 0 for info in point_glbs):
        blockers.append("missing_skeletal_animation")
    if any(int(info.get("skin_count") or 0) <= 0 for info in point_glbs):
        blockers.append("missing_skinning")
    if any(not _has_production_compression(info) for info in point_glbs):
        blockers.append("missing_production_compression")

    blockers = sorted(set(blockers))
    return {
        "artifact_class": "review_static_glb" if review_static else ("production_candidate" if not blockers else "non_production_glb"),
        "production_replay_ready": not blockers,
        "blockers": blockers,
        "glbs": glbs,
        "production_requirements": {
            "point_glbs": [
                "true skeletal animation channels",
                "skinned player meshes",
                "MeshOpt/Draco/KTX2 production compression metadata",
            ],
            "not_allowed": ["static review GLBs generated from virtual_world_paddle_preview.json"],
        },
    }


def validate_replay_export_manifest(
    export_root: str | Path,
    manifest: Mapping[str, Any] | ReplayScene,
    *,
    require_production: bool = False,
) -> ReplayScene:
    """Validate schema, safe relative paths, referenced files, and recorded GLB sizes."""

    root = Path(export_root)
    scene = manifest if isinstance(manifest, ReplayScene) else ReplayScene.model_validate(manifest)
    _validate_glb_ref(root, scene.court_glb, field="court_glb")
    for index, point in enumerate(scene.points):
        glb_path = _validate_glb_ref(root, point.glb_url, field=f"points/{index}/glb_url")
        actual_size_mb = _size_mb(root / glb_path)
        if point.size_mb != actual_size_mb:
            raise ValueError(
                f"points/{index}/size_mb must equal measured GLB size {actual_size_mb} MB for {point.glb_url}"
            )
    if require_production:
        audit = audit_replay_export_manifest(root, scene)
        if not audit["production_replay_ready"]:
            blockers = ", ".join(audit["blockers"])
            raise ValueError(f"production replay manifest is not ready: {blockers}")
    return scene


def resolve_replay_glb_path(export_root: str | Path, value: str | Path, *, field: str) -> Path:
    """Resolve a replay GLB ref after validating it is a safe relative path."""

    root = Path(export_root)
    return root / _validate_glb_ref(root, value, field=field)


def _court_primitives(virtual_world: Mapping[str, Any]) -> list[dict[str, Any]]:
    court = virtual_world.get("court")
    if not isinstance(court, Mapping):
        raise ValueError("virtual_world.court must be an object")
    court_lines: list[tuple[float, float, float]] = []
    line_segments = court.get("line_segments", {})
    if isinstance(line_segments, Mapping):
        for points in line_segments.values():
            court_lines.extend(_line_pairs(points))
    net_lines = _line_pairs((court.get("net") or {}).get("endpoints", []) if isinstance(court.get("net"), Mapping) else [])
    primitives = []
    if court_lines:
        primitives.append({"name": "court_lines", "mode": MODE_LINES, "material": "court", "positions": court_lines})
    if net_lines:
        primitives.append({"name": "net", "mode": MODE_LINES, "material": "net", "positions": net_lines})
    return primitives


def _clip_primitives(virtual_world: Mapping[str, Any]) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    players = virtual_world.get("players", [])
    if isinstance(players, list):
        for player in players:
            if not isinstance(player, Mapping):
                continue
            player_id = int(player.get("id", 0) or 0)
            frames = player.get("frames", [])
            track_path = _player_track_positions(frames)
            if len(track_path) >= 2:
                primitives.append(
                    {
                        "name": f"player_{player_id}_track_path",
                        "mode": MODE_LINE_STRIP,
                        "material": "player_path",
                        "positions": track_path,
                    }
                )
            joints = _player_joint_positions(frames)
            if joints:
                primitives.append(
                    {
                        "name": f"player_{player_id}_joints",
                        "mode": MODE_POINTS,
                        "material": "body_joints",
                        "positions": joints,
                    }
                )
            mesh_points = _player_mesh_positions(frames)
            if mesh_points:
                primitives.append(
                    {
                        "name": f"player_{player_id}_mesh_points",
                        "mode": MODE_POINTS,
                        "material": "body_mesh",
                        "positions": mesh_points,
                    }
                )

    ball_path = _ball_positions(virtual_world)
    if len(ball_path) >= 2:
        primitives.append({"name": "ball_path", "mode": MODE_LINE_STRIP, "material": "ball", "positions": ball_path})
    elif ball_path:
        primitives.append({"name": "ball_points", "mode": MODE_POINTS, "material": "ball", "positions": ball_path})

    paddle_triangles = _paddle_triangle_positions(virtual_world)
    if paddle_triangles:
        primitives.append({"name": "paddle_preview_meshes", "mode": MODE_TRIANGLES, "material": "paddle", "positions": paddle_triangles})
    if not primitives:
        primitives.append({"name": "empty_review_marker", "mode": MODE_POINTS, "material": "marker", "positions": [(0.0, 0.0, 0.0)]})
    return primitives


def _write_review_glb(path: Path, primitives: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_bytes, bin_bytes = _review_glb_chunks(primitives)
    total_length = 12 + 8 + len(json_bytes) + (8 + len(bin_bytes) if bin_bytes else 0)
    with path.open("wb") as handle:
        handle.write(struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, total_length))
        handle.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        handle.write(json_bytes)
        if bin_bytes:
            handle.write(struct.pack("<I4s", len(bin_bytes), b"BIN\x00"))
            handle.write(bin_bytes)


def _review_glb_chunks(primitives: list[dict[str, Any]]) -> tuple[bytes, bytes]:
    bin_blob = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    materials = _materials()
    material_indexes = {str(material["name"]): index for index, material in enumerate(materials)}

    for primitive in primitives:
        positions = [_vec3(position) for position in primitive.get("positions", [])]
        positions = [position for position in positions if position is not None]
        if not positions:
            continue
        _pad_bin(bin_blob)
        byte_offset = len(bin_blob)
        for x, y, z in positions:
            bin_blob.extend(struct.pack("<fff", x, y, z))
        byte_length = len(positions) * 12
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": byte_length, "target": 34962})
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": [min(position[index] for position in positions) for index in range(3)],
                "max": [max(position[index] for position in positions) for index in range(3)],
            }
        )
        material_name = str(primitive.get("material", "marker"))
        meshes.append(
            {
                "name": str(primitive.get("name", "review_primitive")),
                "primitives": [
                    {
                        "attributes": {"POSITION": len(accessors) - 1},
                        "mode": int(primitive.get("mode", MODE_POINTS)),
                        "material": material_indexes.get(material_name, material_indexes["marker"]),
                    }
                ],
            }
        )
        nodes.append({"name": str(primitive.get("name", "review_primitive")), "mesh": len(meshes) - 1})

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "pickleball_review_replay_export"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
    }
    if bin_blob:
        _pad_bin(bin_blob)
        gltf["buffers"] = [{"byteLength": len(bin_blob)}]
        gltf["bufferViews"] = buffer_views
        gltf["accessors"] = accessors
    json_bytes = json.dumps(gltf, separators=(",", ":"), sort_keys=True).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    return json_bytes, bytes(bin_blob)


def _materials() -> list[dict[str, Any]]:
    return [
        _material("court", (0.1, 0.65, 0.35, 1.0)),
        _material("net", (0.12, 0.12, 0.12, 1.0)),
        _material("player_path", (0.12, 0.35, 0.95, 1.0)),
        _material("body_joints", (0.95, 0.55, 0.15, 1.0)),
        _material("body_mesh", (0.55, 0.65, 0.95, 0.55)),
        _material("ball", (0.9, 0.9, 0.05, 1.0)),
        _material("paddle", (0.95, 0.15, 0.2, 0.55)),
        _material("marker", (0.8, 0.8, 0.8, 1.0)),
    ]


def _material(name: str, color: tuple[float, float, float, float]) -> dict[str, Any]:
    return {
        "name": name,
        "doubleSided": True,
        "pbrMetallicRoughness": {
            "baseColorFactor": list(color),
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
    }


def _line_pairs(value: Any) -> list[tuple[float, float, float]]:
    points = [_vec3(point) for point in value] if isinstance(value, list) else []
    points = [point for point in points if point is not None]
    pairs: list[tuple[float, float, float]] = []
    for index in range(max(0, len(points) - 1)):
        pairs.extend([points[index], points[index + 1]])
    return pairs


def _player_track_positions(frames: Any) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    if not isinstance(frames, list):
        return positions
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        world_xy = frame.get("track_world_xy")
        if isinstance(world_xy, list) and len(world_xy) >= 2:
            positions.append((float(world_xy[0]), float(world_xy[1]), 0.03))
    return positions


def _player_joint_positions(frames: Any) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    if not isinstance(frames, list):
        return positions
    for frame in frames:
        if isinstance(frame, Mapping):
            positions.extend(_vec3(point) for point in frame.get("joints_world", []) if _vec3(point) is not None)
    return positions


def _player_mesh_positions(frames: Any) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    if not isinstance(frames, list):
        return positions
    for frame in frames:
        if isinstance(frame, Mapping):
            positions.extend(_vec3(point) for point in frame.get("mesh_vertices_world", []) if _vec3(point) is not None)
    return positions


def _ball_positions(virtual_world: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    ball = virtual_world.get("ball")
    frames = ball.get("frames", []) if isinstance(ball, Mapping) else []
    positions: list[tuple[float, float, float]] = []
    if not isinstance(frames, list):
        return positions
    for frame in frames:
        if not isinstance(frame, Mapping) or frame.get("visible") is False:
            continue
        position = _vec3(frame.get("world_xyz"))
        if position is not None:
            positions.append(position)
    return positions


def _paddle_triangle_positions(virtual_world: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    paddles = virtual_world.get("paddles", [])
    positions: list[tuple[float, float, float]] = []
    if not isinstance(paddles, list):
        return positions
    for paddle in paddles:
        frames = paddle.get("frames", []) if isinstance(paddle, Mapping) else []
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            vertices = [_vec3(vertex) for vertex in frame.get("mesh_vertices_world", [])]
            faces = frame.get("mesh_faces", [])
            if not isinstance(faces, list):
                continue
            for face in faces:
                if not isinstance(face, list) or len(face) != 3:
                    continue
                try:
                    triangle = [vertices[int(index)] for index in face]
                except (IndexError, TypeError, ValueError):
                    continue
                if all(vertex is not None for vertex in triangle):
                    positions.extend(vertex for vertex in triangle if vertex is not None)
    return positions


def _virtual_world_player_ids(virtual_world: Mapping[str, Any]) -> list[int]:
    players = virtual_world.get("players", [])
    ids: list[int] = []
    if isinstance(players, list):
        for player in players:
            if isinstance(player, Mapping) and isinstance(player.get("id"), int):
                ids.append(int(player["id"]))
    return sorted(set(ids))


def _time_range(virtual_world: Mapping[str, Any], *, fps: float) -> tuple[float, float]:
    times: list[float] = []
    players = virtual_world.get("players", [])
    if isinstance(players, list):
        for player in players:
            frames = player.get("frames", []) if isinstance(player, Mapping) else []
            times.extend(_frame_times(frames))
    ball = virtual_world.get("ball")
    if isinstance(ball, Mapping):
        times.extend(_frame_times(ball.get("frames", [])))
    paddles = virtual_world.get("paddles", [])
    if isinstance(paddles, list):
        for paddle in paddles:
            frames = paddle.get("frames", []) if isinstance(paddle, Mapping) else []
            times.extend(_frame_times(frames))
    if not times:
        return (0.0, 1.0 / fps)
    return (min(times), max(times) if max(times) > min(times) else min(times) + 1.0 / fps)


def _frame_times(frames: Any) -> list[float]:
    if not isinstance(frames, list):
        return []
    times = []
    for frame in frames:
        if isinstance(frame, Mapping) and isinstance(frame.get("t"), (int, float)) and not isinstance(frame.get("t"), bool):
            times.append(float(frame["t"]))
    return times


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field} must be a positive number")
    return float(value)


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _pad_bin(bin_blob: bytearray) -> None:
    padding = (4 - len(bin_blob) % 4) % 4
    if padding:
        bin_blob.extend(b"\x00" * padding)


def _has_production_compression(glb_info: Mapping[str, Any]) -> bool:
    extensions = set(glb_info.get("extensions_used", [])) if isinstance(glb_info.get("extensions_used"), list) else set()
    return bool({"EXT_meshopt_compression", "KHR_draco_mesh_compression", "KHR_texture_basisu"} & extensions)


def _point_glb_ref(point: Mapping[str, Any]) -> str | Path:
    if "glb_path" in point:
        return point["glb_path"]
    if "glb_url" in point:
        return point["glb_url"]
    raise ValueError("point_glbs entries require glb_path")


def _point_int(point: Mapping[str, Any], field: str) -> int:
    value = point.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"point_glbs entries require integer {field}")
    return value


def _point_float(point: Mapping[str, Any], field: str) -> float:
    value = point.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"point_glbs entries require numeric {field}")
    return float(value)


def _player_ids(players: Sequence[int]) -> list[int]:
    player_ids: list[int] = []
    for player in players:
        if isinstance(player, bool) or not isinstance(player, int):
            raise ValueError("players must contain integer player ids")
        player_ids.append(player)
    return player_ids


def _validate_glb_ref(root: Path, value: str | Path, *, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field} must be a relative GLB path")

    relative_path = Path(value)
    if (
        str(value) == ""
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.suffix.lower() != ".glb"
    ):
        raise ValueError(f"{field} must be relative and stay within export_root")

    root_resolved = root.resolve()
    target = root / relative_path
    try:
        target.resolve(strict=False).relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{field} must be relative and stay within export_root") from exc

    if not target.is_file():
        raise FileNotFoundError(f"{field} file does not exist: {relative_path.as_posix()}")
    return Path(*relative_path.parts)


def _relative_to_root(path: Path, *, root: Path, field: str) -> Path:
    try:
        relative = path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"{field} must be inside scene_root") from exc
    return Path(*relative.parts)


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1_000_000, 6)
