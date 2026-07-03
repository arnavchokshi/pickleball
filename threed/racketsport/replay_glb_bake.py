"""W3-REPLAY-NATIVE phase 1: bake a `racketsport_body_mesh` artifact (see
`mesh_export.py`) into an animated, morph-target-driven glTF/GLB.

Design (documented here because there is no prior animated-mesh bake in this
repo to follow -- `replay_export.py`'s GLBs are static review snapshots, see
its `audit_replay_export_manifest`, which explicitly flags "missing skeletal
animation" / "missing skinning" as known gaps of that pipeline):

- Each player's *first* scheduled mesh frame (by `frame_idx`) becomes the
  mesh's base/rest pose (`POSITION` accessor + shared triangle `mesh_faces`
  indices).
- Every *other* scheduled frame becomes a glTF morph target: a `POSITION`
  accessor holding the per-vertex delta from the base pose. glTF has no
  native "keyframe the raw position buffer" mechanism outside skinning
  (which needs LBS skin weights we do not have available locally -- SMPL-X
  skin weights live with the model asset, not in `smpl_motion.json`) or
  morph targets, so morph targets are the correct, standards-compliant tool
  for baking arbitrary per-frame vertex deformation without a skeleton.
- One glTF `animation` per player drives the mesh's `weights` channel with
  STEP interpolation: at each scheduled frame's real timestamp, that frame's
  morph target snaps to weight 1.0 (all others 0.0), and holds until the
  next scheduled frame. This is a deliberately honest "flipbook" -- it never
  fabricates continuous motion between temporally-distant contact windows
  that were never computed (mesh vertices exist for 152 of 2288 frames in
  the source clip; see `body_mesh_readiness.json`'s
  `requested_world_mesh_frame_count`).
- A second glTF `animation` per player drives that same node's `scale`
  channel with STEP interpolation, zeroing the mesh out (`scale=(0,0,0)`)
  everywhere outside the contiguous windows the mesh was actually baked
  for, and restoring it to `(1,1,1)` for each window's duration. Core glTF
  2.0 has no native node-visibility channel, so scale-to-zero is the
  portable, spec-compliant stand-in -- the same honesty gating the USDZ
  bake (`replay_usdz_bake.py`) does natively via its `visibility` attribute.
  Without this, both glTF's "hold the nearest sample" extrapolation before
  the first weights keyframe and STEP interpolation's "hold until the next
  keyframe" behavior *between* keyframes would otherwise leave the last
  sampled mesh pose visible and frozen across any gap between windows
  (e.g. contact windows at frames 10-11 and 40 leaving frames 12-39 showing
  a frozen full body that was never computed for those frames).
- Position and delta accessors are written as plain float32 (no
  KHR_mesh_quantization at this stage); a separate compression pass
  (`gltf-transform optimize --compress meshopt`, see
  `scripts/racketsport/build_replay_animated_glb.py`) quantizes and
  meshopt-compresses the raw GLB this module produces, per the
  W3-REPLAY-NATIVE gate's "meshopt+quantization" language. Keeping this
  module's output uncompressed-but-correct keeps the numerically-tricky
  part (delta encoding, accessor bounds, animation timing) easy to test in
  pure Python without depending on the npm compressor being importable from
  pytest.
"""

from __future__ import annotations

import struct
from typing import Any, Mapping, Sequence

import pygltflib as gltf


ARTIFACT_TYPE = "racketsport_body_mesh_glb_bake"
COPYRIGHT = "racketsport replay -- preview bake, not gate-verified"


class BodyMeshBakeError(ValueError):
    """Raised when `body_mesh.json` cannot be baked into a valid animated GLB."""


def build_animated_body_glb(body_mesh: Mapping[str, Any], *, clip: str) -> bytes:
    """Return raw `.glb` bytes baking every player in `body_mesh` (the
    `racketsport_body_mesh` artifact produced by `mesh_export.py`) into an
    animated, morph-target mesh. Raises `BodyMeshBakeError` if there is no
    bakeable player (every player needs >=1 frame with vertices+faces)."""

    mesh_faces = _mesh_faces(body_mesh)
    if not mesh_faces:
        raise BodyMeshBakeError("body_mesh.mesh_faces is empty; nothing to bake")
    players = [p for p in body_mesh.get("players", []) if isinstance(p, Mapping) and p.get("frames")]
    if not players:
        raise BodyMeshBakeError("body_mesh.players has no frames to bake")

    document = gltf.GLTF2()
    document.asset = gltf.Asset(version="2.0", generator="racketsport replay_glb_bake (preview)", copyright=COPYRIGHT)
    document.scenes = [gltf.Scene(nodes=[])]
    document.scene = 0

    builder = _BufferBuilder()
    index_accessor = _write_index_accessor(builder, mesh_faces)
    fps = float(body_mesh.get("fps", 30.0)) or 30.0

    for player in players:
        player_id = int(player.get("id", 0))
        frames = sorted(
            (f for f in player.get("frames", []) if isinstance(f, Mapping) and f.get("mesh_vertices_world")),
            key=lambda f: int(f.get("frame_idx", 0)),
        )
        if not frames:
            continue
        _bake_player(
            document, builder, frames=frames, index_accessor=index_accessor, player_id=player_id, clip=clip, fps=fps
        )

    document.buffers = [gltf.Buffer(byteLength=len(builder.blob))]
    document.bufferViews = builder.buffer_views
    document.accessors = builder.accessors
    document.set_binary_blob(bytes(builder.blob))
    return _to_glb_bytes(document)


def _to_glb_bytes(document: "gltf.GLTF2") -> bytes:
    # pygltflib's save_to_bytes signature has varied across versions (returns
    # either `bytes` or a list of chunk bytes); normalize defensively so this
    # module keeps working across the pinned version and nearby ones.
    result = document.save_to_bytes()
    if isinstance(result, (bytes, bytearray)):
        return bytes(result)
    if isinstance(result, (list, tuple)):
        return b"".join(bytes(chunk) for chunk in result)
    raise BodyMeshBakeError(f"unexpected save_to_bytes() return type: {type(result)!r}")


def _bake_player(
    document: "gltf.GLTF2",
    builder: "_BufferBuilder",
    *,
    frames: Sequence[Mapping[str, Any]],
    index_accessor: int,
    player_id: int,
    clip: str,
    fps: float,
) -> None:
    base_frame = frames[0]
    base_vertices = _vertices(base_frame)
    vertex_count = len(base_vertices)
    base_position_accessor = _write_position_accessor(builder, base_vertices, name=f"player{player_id}_base_position")

    targets: list[dict[str, int]] = []
    for frame in frames[1:]:
        vertices = _vertices(frame)
        if len(vertices) != vertex_count:
            raise BodyMeshBakeError(
                f"player {player_id} frame_idx={frame.get('frame_idx')} has {len(vertices)} vertices, "
                f"expected {vertex_count} (base frame_idx={base_frame.get('frame_idx')})"
            )
        delta = [
            (v[0] - b[0], v[1] - b[1], v[2] - b[2])
            for v, b in zip(vertices, base_vertices)
        ]
        delta_accessor = _write_position_accessor(
            builder, delta, name=f"player{player_id}_delta_f{frame.get('frame_idx')}", is_delta=True
        )
        targets.append({"POSITION": delta_accessor})

    primitive = gltf.Primitive(
        attributes=gltf.Attributes(POSITION=base_position_accessor),
        indices=index_accessor,
        targets=targets if targets else None,
        mode=gltf.TRIANGLES,
    )
    mesh_index = len(document.meshes) if document.meshes else 0
    document.meshes = (document.meshes or []) + [
        gltf.Mesh(
            name=f"{clip}_player{player_id}_body_mesh",
            primitives=[primitive],
            weights=[0.0] * len(targets) if targets else None,
        )
    ]

    node_index = len(document.nodes) if document.nodes else 0
    document.nodes = (document.nodes or []) + [gltf.Node(name=f"player{player_id}_body", mesh=mesh_index)]
    document.scenes[0].nodes.append(node_index)

    if targets:
        keyframe_times = [float(frame.get("t", 0.0)) for frame in frames]
        time_accessor = _write_scalar_accessor(builder, keyframe_times, name=f"player{player_id}_keyframe_times")

        weight_count = len(targets)
        weight_values: list[float] = [0.0] * weight_count  # base frame keyframe: all-zero weights
        for target_index in range(weight_count):
            row = [0.0] * weight_count
            row[target_index] = 1.0
            weight_values.extend(row)
        weights_accessor = _write_scalar_accessor(
            builder, weight_values, name=f"player{player_id}_weights", set_min_max=False
        )

        sampler = gltf.AnimationSampler(input=time_accessor, output=weights_accessor, interpolation=gltf.ANIM_STEP)
        channel = gltf.AnimationChannel(
            sampler=0,
            target=gltf.AnimationChannelTarget(node=node_index, path="weights"),
        )
        document.animations = (document.animations or []) + [
            gltf.Animation(name=f"player{player_id}_body_mesh_flipbook", samplers=[sampler], channels=[channel])
        ]
        # A single-frame player still needs visibility gating below (a mesh baked for
        # exactly one frame must not render as a frozen static body for the whole clip).

    visibility_times, visibility_scales = _visibility_scale_keyframes(frames, fps=fps)
    if visibility_times:
        visibility_time_accessor = _write_scalar_accessor(
            builder, visibility_times, name=f"player{player_id}_visibility_times"
        )
        visibility_scale_accessor = _write_position_accessor(
            builder, visibility_scales, name=f"player{player_id}_visibility_scale", is_vertex_attribute=False
        )
        visibility_sampler = gltf.AnimationSampler(
            input=visibility_time_accessor, output=visibility_scale_accessor, interpolation=gltf.ANIM_STEP
        )
        visibility_channel = gltf.AnimationChannel(
            sampler=0,
            target=gltf.AnimationChannelTarget(node=node_index, path="scale"),
        )
        document.animations = (document.animations or []) + [
            gltf.Animation(
                name=f"player{player_id}_body_mesh_visibility",
                samplers=[visibility_sampler],
                channels=[visibility_channel],
            )
        ]


def _contiguous_windows(frames: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Group already frame_idx-sorted frames into contiguous `source_window_index`
    runs (falls back to grouping by contiguous `frame_idx` when `source_window_index`
    is missing). Mirrors `replay_usdz_bake.py`'s `_contiguous_windows` exactly, kept
    as a separate copy so this module never has to import `pxr` (USD) just to bake a
    GLB."""

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


def _visibility_scale_keyframes(
    frames: Sequence[Mapping[str, Any]], *, fps: float
) -> tuple[list[float], list[tuple[float, float, float]]]:
    """STEP keyframes for a node's `scale` channel that zero the mesh out
    (``scale=(0,0,0)``) everywhere outside the contiguous scheduled windows in
    ``frames``, and restore it to ``(1,1,1)`` for each window's duration -- the
    glTF-side equivalent of the USDZ bake's ``visibility`` attribute gating (see
    module docstring). Returns parallel ``(times, scales)`` lists, empty if
    ``frames`` is empty.
    """

    windows = _contiguous_windows(frames)
    if not windows:
        return [], []

    epsilon = (0.5 / fps) if fps > 0 else 1e-3
    times: list[float] = []
    scales: list[tuple[float, float, float]] = []

    def _add(time: float, *, visible: bool) -> None:
        # glTF sampler input values must be strictly increasing; nudge forward on
        # any collision (e.g. a single-frame window whose start == end) rather than
        # emit a malformed accessor.
        if times and time <= times[-1]:
            time = times[-1] + epsilon
        times.append(time)
        scales.append((1.0, 1.0, 1.0) if visible else (0.0, 0.0, 0.0))

    first_window_start_t = float(windows[0][0].get("t", 0.0))
    _add(first_window_start_t - epsilon, visible=False)
    for window in windows:
        _add(float(window[0].get("t", 0.0)), visible=True)
        _add(float(window[-1].get("t", 0.0)) + epsilon, visible=False)
    return times, scales


class _BufferBuilder:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.buffer_views: list[gltf.BufferView] = []
        self.accessors: list[gltf.Accessor] = []

    def add_bytes(self, data: bytes, *, target: int | None = None) -> int:
        while len(self.blob) % 4 != 0:
            self.blob.append(0)
        offset = len(self.blob)
        self.blob.extend(data)
        self.buffer_views.append(gltf.BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        return len(self.buffer_views) - 1

    def add_accessor(self, accessor: gltf.Accessor) -> int:
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def _write_position_accessor(
    builder: _BufferBuilder,
    vectors: Sequence[tuple[float, float, float]],
    *,
    name: str,
    is_delta: bool = False,
    is_vertex_attribute: bool = True,
) -> int:
    packed = b"".join(struct.pack("<3f", *v) for v in vectors)
    # Only real per-vertex mesh attributes (POSITION / morph-target POSITION deltas)
    # belong to a `gltf.ARRAY_BUFFER` bufferView. An animation sampler's `output`
    # accessor (e.g. the visibility scale keyframes below) is not a vertex buffer;
    # tagging it ARRAY_BUFFER anyway trips glTF-Validator's BUFFER_VIEW_TARGET_OVERRIDE
    # error once `gltf-transform optimize` restructures bufferViews by actual usage.
    view = builder.add_bytes(packed, target=gltf.ARRAY_BUFFER if is_vertex_attribute else None)
    mins = [min(v[axis] for v in vectors) for axis in range(3)]
    maxs = [max(v[axis] for v in vectors) for axis in range(3)]
    accessor = gltf.Accessor(
        bufferView=view,
        componentType=gltf.FLOAT,
        count=len(vectors),
        type=gltf.VEC3,
        min=mins,
        max=maxs,
        name=name,
    )
    return builder.add_accessor(accessor)


def _write_index_accessor(builder: _BufferBuilder, faces: Sequence[Sequence[int]]) -> int:
    flat_indices = [index for face in faces for index in face]
    max_index = max(flat_indices) if flat_indices else 0
    if max_index < 65536:
        packed = b"".join(struct.pack("<H", index) for index in flat_indices)
        component_type = gltf.UNSIGNED_SHORT
    else:
        packed = b"".join(struct.pack("<I", index) for index in flat_indices)
        component_type = gltf.UNSIGNED_INT
    view = builder.add_bytes(packed, target=gltf.ELEMENT_ARRAY_BUFFER)
    accessor = gltf.Accessor(
        bufferView=view,
        componentType=component_type,
        count=len(flat_indices),
        type=gltf.SCALAR,
        min=[0],
        max=[max_index],
        name="mesh_faces_indices",
    )
    return builder.add_accessor(accessor)


def _write_scalar_accessor(
    builder: _BufferBuilder,
    values: Sequence[float],
    *,
    name: str,
    set_min_max: bool = True,
) -> int:
    packed = b"".join(struct.pack("<f", value) for value in values)
    view = builder.add_bytes(packed)
    accessor = gltf.Accessor(
        bufferView=view,
        componentType=gltf.FLOAT,
        count=len(values),
        type=gltf.SCALAR,
        min=[min(values)] if set_min_max and values else None,
        max=[max(values)] if set_min_max and values else None,
        name=name,
    )
    return builder.add_accessor(accessor)


def _vertices(frame: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    vertices = frame.get("mesh_vertices_world", [])
    return [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]


def _mesh_faces(body_mesh: Mapping[str, Any]) -> list[list[int]]:
    faces = body_mesh.get("mesh_faces", [])
    parsed: list[list[int]] = []
    for face in faces:
        if not isinstance(face, (list, tuple)) or len(face) != 3:
            return []
        parsed.append([int(index) for index in face])
    return parsed


__all__ = ["build_animated_body_glb", "BodyMeshBakeError", "ARTIFACT_TYPE"]
