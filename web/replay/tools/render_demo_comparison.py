#!/usr/bin/env python3
"""Render a polished source-video + virtual-baseline comparison MP4.

The renderer consumes the final immutable ``virtual_world.json`` placement and,
when available, the matching BODY mesh index.  Joint-only runs are displayed as
translucent articulated avatars built directly from their final grounded joints;
this is a presentation surface, not fabricated measurement geometry.  Missing
world samples remain missing.  Optional between-frame interpolation is
display-only and is used only when both adjacent measured samples exist.

The visual language mirrors the replay viewer: warm white canvas, regulation
pickleball court, restrained ink lines, translucent per-player surfaces, and a
fixed baseline camera that makes comparisons across clips straightforward.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# BGR values matching web/replay's cream + player evidence palette.
CREAM = (232, 241, 244)  # #f4f1e8
COURT_WHITE = (238, 245, 247)  # #f7f5ee
COURT_KITCHEN = (222, 239, 230)
COURT_APRON = (218, 229, 224)
COURT_INK = (104, 111, 101)  # #656f68
NET_TAPE = (82, 87, 77)
NET_MESH = (131, 138, 127)
TEXT_INK = (28, 33, 29)
TEXT_MUTED = (106, 112, 107)
WHITE = (255, 255, 255)

PLAYER_COLORS = {
    1: (61, 255, 223),   # lime #dfff3d
    2: (255, 219, 64),   # cyan #40dbff
    3: (112, 143, 255),  # coral #ff8f70
    4: (148, 255, 176),  # mint #b0ff94
}

COURT_HALF_WIDTH_M = 3.05
COURT_HALF_LENGTH_M = 6.705
NVZ_DISTANCE_FROM_NET_M = 2.13
NET_POST_HEIGHT_M = 0.9144
NET_CENTER_HEIGHT_M = 0.8636
DISPLAY_INTERPOLATION_MAX_GAP_S = 0.05


@dataclass(frozen=True)
class MeshFrame:
    vertices_mm: np.ndarray
    joints_mm: np.ndarray
    joint_conf: np.ndarray
    blend_weight: float
    source_window_index: int


@dataclass(frozen=True)
class WorldFrame:
    joints_m: np.ndarray
    joint_conf: np.ndarray
    mesh_player_id: int


def load_mesh_frames(
    index_path: Path,
) -> tuple[dict, np.ndarray, dict[int, dict[int, MeshFrame]]]:
    """Decode every mesh window into frame-keyed, millimetre int16 arrays."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    faces_path = index_path.parent / index["faces_url"]
    faces = np.asarray(
        json.loads(faces_path.read_text(encoding="utf-8"))["mesh_faces"],
        dtype=np.int32,
    )
    decoded: dict[int, dict[int, MeshFrame]] = {}
    for window in index["windows"]:
        chunk_path = index_path.parent / window["url"]
        with gzip.open(chunk_path, "rb") as handle:
            raw = handle.read()
        view = memoryview(raw)
        scale = float(window["quantization"]["scale"])
        if scale <= 0:
            raise ValueError(f"invalid mesh quantization scale: {scale}")
        offset = 0
        for player in window["players"]:
            player_id = int(player["id"])
            frames = decoded.setdefault(player_id, {})
            previous_vertices: np.ndarray | None = None
            previous_joints: np.ndarray | None = None
            for meta in player["frames"]:
                vertex_count = int(meta["vertex_count"])
                joint_count = int(meta["joint_count"])
                vertex_values = np.frombuffer(
                    view[offset : offset + vertex_count * 6], dtype="<i2"
                ).reshape(vertex_count, 3).astype(np.int32)
                offset += vertex_count * 6
                joint_values = np.frombuffer(
                    view[offset : offset + joint_count * 6], dtype="<i2"
                ).reshape(joint_count, 3).astype(np.int32)
                offset += joint_count * 6
                if meta.get("delta_from_previous"):
                    if previous_vertices is None or previous_joints is None:
                        raise ValueError("delta mesh frame has no preceding absolute frame")
                    vertex_values += previous_vertices
                    joint_values += previous_joints
                previous_vertices = vertex_values
                previous_joints = joint_values
                frame_idx = int(meta["frame_idx"])
                # The on-disk integer is quantized units/m. Store millimetres to
                # preserve the compact original renderer memory footprint.
                vertices_mm = np.rint(vertex_values / scale * 1000.0).astype(np.int16)
                joints_mm = np.rint(joint_values / scale * 1000.0).astype(np.int16)
                frames.setdefault(
                    frame_idx,
                    MeshFrame(
                        vertices_mm=vertices_mm,
                        joints_mm=joints_mm,
                        joint_conf=np.asarray(meta.get("joint_conf", [1.0] * joint_count), dtype=np.float32),
                        blend_weight=float(meta.get("blend_weight", 1.0)),
                        source_window_index=int(meta.get("source_window_index", window.get("source_window_index", 0))),
                    ),
                )
        if offset != len(raw):
            raise ValueError(
                f"mesh chunk decode mismatch for {chunk_path}: "
                f"consumed {offset} bytes, found {len(raw)}"
            )
    return index, faces, decoded


def load_world_frames(
    world_path: Path,
    mesh_fps: float,
) -> dict[int, dict[int, WorldFrame]]:
    """Load only measured/final world skeleton frames; empty gaps stay empty."""
    world = json.loads(world_path.read_text(encoding="utf-8"))
    joint_names = [str(name).lower() for name in world.get("joint_names", [])]
    if len(joint_names) < 11 or joint_names[9] != "left_hip" or joint_names[10] != "right_hip":
        raise ValueError(
            "virtual_world joint_names must expose MHR70 left_hip/right_hip at indices 9/10; "
            "refusing an unverified mesh-root alignment"
        )
    world_frames: dict[int, dict[int, WorldFrame]] = {}
    for player in world.get("players", []):
        player_id = int(player["id"])
        frames: dict[int, WorldFrame] = {}
        for frame in player.get("frames", []):
            if frame.get("display_interpolated") is True:
                continue
            joints = np.asarray(frame.get("joints_world", []), dtype=np.float32)
            if joints.ndim != 2 or joints.shape[0] < 11 or joints.shape[1] != 3 or not np.all(np.isfinite(joints)):
                continue
            conf = np.asarray(frame.get("joint_conf", [1.0] * len(joints)), dtype=np.float32)
            if len(conf) != len(joints):
                continue
            t = float(frame.get("t", -1.0))
            if not math.isfinite(t) or t < 0:
                continue
            frame_idx = int(round(t * mesh_fps))
            mesh_ref = frame.get("mesh_ref")
            mesh_player_id = player_id
            if isinstance(mesh_ref, dict) and mesh_ref.get("player_id") is not None:
                mesh_player_id = int(mesh_ref["player_id"])
            frames[frame_idx] = WorldFrame(
                joints_m=joints,
                joint_conf=conf,
                mesh_player_id=mesh_player_id,
            )
        world_frames[player_id] = frames
    if not world_frames:
        raise ValueError(f"virtual_world contains no players: {world_path}")
    return world_frames


def camera_basis(
    eye: np.ndarray,
    target: np.ndarray,
    up_hint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, forward


def baseline_camera(width: int, height: int, orbit_degrees: float = 0.0) -> dict:
    angle = math.radians(orbit_degrees)
    radius = 15.7
    eye = np.asarray(
        [radius * math.sin(angle), -radius * math.cos(angle), 6.8],
        dtype=np.float32,
    )
    target = np.asarray([0.0, 0.35, 0.72], dtype=np.float32)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return {
        "eye": eye,
        "basis": camera_basis(eye, target, up),
        # Fit the complete regulation rectangle at 16:9. The earlier demo
        # renderer cropped the near baseline, which made spatial review harder.
        "focal": float(height) * 1.40,
    }


def project(
    points: np.ndarray,
    camera: dict,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    eye = camera["eye"]
    right, up, forward = camera["basis"]
    relative = points - eye
    x = relative @ right
    y = relative @ up
    depth = relative @ forward
    focal = float(camera["focal"])
    safe_depth = np.maximum(depth, 0.05)
    screen = np.column_stack(
        (
            width * 0.5 + focal * x / safe_depth,
            height * 0.535 - focal * y / safe_depth,
        )
    )
    return np.rint(screen).astype(np.int32), depth


def _fill_world_polygon(
    panel: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    camera: dict,
) -> None:
    pixels, depth = project(polygon, camera, panel.shape[1], panel.shape[0])
    if np.all(depth > 0.05):
        cv2.fillConvexPoly(panel, pixels, color, lineType=cv2.LINE_AA)


def _draw_world_polyline(
    panel: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    width: int,
    camera: dict,
    closed: bool = False,
    opacity: float = 1.0,
) -> None:
    pixels, depth = project(points, camera, panel.shape[1], panel.shape[0])
    if not np.all(depth > 0.05):
        return
    if opacity >= 0.999:
        cv2.polylines(panel, [pixels], closed, color, width, cv2.LINE_AA)
        return
    overlay = panel.copy()
    cv2.polylines(overlay, [pixels], closed, color, width, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, panel, 1.0 - opacity, 0, panel)


def draw_court(panel: np.ndarray, camera: dict) -> None:
    """Draw a metric regulation court and a sagging regulation net."""
    half_w = COURT_HALF_WIDTH_M
    half_l = COURT_HALF_LENGTH_M
    apron = 1.15

    # Soft apron shadow gives the nearly-white court separation without a dark
    # synthetic environment.
    apron_poly = np.asarray(
        [
            [-half_w - apron, -half_l - apron, -0.035],
            [half_w + apron, -half_l - apron, -0.035],
            [half_w + apron, half_l + apron, -0.035],
            [-half_w - apron, half_l + apron, -0.035],
        ],
        dtype=np.float32,
    )
    _fill_world_polygon(panel, apron_poly, COURT_APRON, camera)

    court_poly = np.asarray(
        [
            [-half_w, -half_l, 0],
            [half_w, -half_l, 0],
            [half_w, half_l, 0],
            [-half_w, half_l, 0],
        ],
        dtype=np.float32,
    )
    _fill_world_polygon(panel, court_poly, COURT_WHITE, camera)

    kitchen_poly = np.asarray(
        [
            [-half_w, -NVZ_DISTANCE_FROM_NET_M, 0.004],
            [half_w, -NVZ_DISTANCE_FROM_NET_M, 0.004],
            [half_w, NVZ_DISTANCE_FROM_NET_M, 0.004],
            [-half_w, NVZ_DISTANCE_FROM_NET_M, 0.004],
        ],
        dtype=np.float32,
    )
    _fill_world_polygon(panel, kitchen_poly, COURT_KITCHEN, camera)

    line_z = 0.018
    lines = [
        np.asarray(
            [
                [-half_w, -half_l, line_z],
                [half_w, -half_l, line_z],
                [half_w, half_l, line_z],
                [-half_w, half_l, line_z],
                [-half_w, -half_l, line_z],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [[-half_w, -NVZ_DISTANCE_FROM_NET_M, line_z], [half_w, -NVZ_DISTANCE_FROM_NET_M, line_z]],
            dtype=np.float32,
        ),
        np.asarray(
            [[-half_w, NVZ_DISTANCE_FROM_NET_M, line_z], [half_w, NVZ_DISTANCE_FROM_NET_M, line_z]],
            dtype=np.float32,
        ),
        np.asarray([[0, -half_l, line_z], [0, -NVZ_DISTANCE_FROM_NET_M, line_z]], dtype=np.float32),
        np.asarray([[0, NVZ_DISTANCE_FROM_NET_M, line_z], [0, half_l, line_z]], dtype=np.float32),
    ]
    for line in lines:
        _draw_world_polyline(panel, line, COURT_INK, 3, camera)

    # Net posts.
    for x in (-half_w, half_w):
        _draw_world_polyline(
            panel,
            np.asarray([[x, 0, 0], [x, 0, NET_POST_HEIGHT_M + 0.06]], dtype=np.float32),
            NET_TAPE,
            6,
            camera,
        )

    # Net mesh follows the regulation 36in posts / 34in center sag. The mesh is
    # deliberately subtle so translucent people remain legible through it.
    xs = np.linspace(-half_w, half_w, 27, dtype=np.float32)
    top_z = NET_CENTER_HEIGHT_M + (NET_POST_HEIGHT_M - NET_CENTER_HEIGHT_M) * np.abs(xs) / half_w
    for x, z_top in zip(xs, top_z, strict=True):
        _draw_world_polyline(
            panel,
            np.asarray([[x, 0, 0.04], [x, 0, z_top]], dtype=np.float32),
            NET_MESH,
            1,
            camera,
            opacity=0.38,
        )
    for fraction in np.linspace(0.13, 0.88, 6):
        zs = np.maximum(0.04, top_z * fraction)
        net_row = np.column_stack((xs, np.zeros_like(xs), zs)).astype(np.float32)
        _draw_world_polyline(panel, net_row, NET_MESH, 1, camera, opacity=0.34)
    net_top = np.column_stack((xs, np.zeros_like(xs), top_z)).astype(np.float32)
    _draw_world_polyline(panel, net_top, NET_TAPE, 4, camera)


def mesh_sample_at(
    frames: dict[int, MeshFrame],
    frame_position: float,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    bracket = _display_interpolation_bracket(frames, frame_position, fps)
    if bracket is None:
        return None
    left_idx, right_idx, alpha = bracket
    current = frames[left_idx]
    if left_idx == right_idx:
        return (
            current.vertices_mm.astype(np.float32) / 1000.0,
            current.joints_mm.astype(np.float32) / 1000.0,
            current.joint_conf,
        )
    following = frames[right_idx]
    if (
        following.source_window_index == current.source_window_index
        and following.vertices_mm.shape == current.vertices_mm.shape
        and following.joints_mm.shape == current.joints_mm.shape
    ):
        vertices = (
            (1.0 - alpha) * current.vertices_mm.astype(np.float32)
            + alpha * following.vertices_mm.astype(np.float32)
        ) / 1000.0
        joints = (
            (1.0 - alpha) * current.joints_mm.astype(np.float32)
            + alpha * following.joints_mm.astype(np.float32)
        ) / 1000.0
        confidence = (1.0 - alpha) * current.joint_conf + alpha * following.joint_conf
        return vertices, joints, confidence
    return None


def world_sample_at(
    frames: dict[int, WorldFrame],
    frame_position: float,
    fps: float,
) -> WorldFrame | None:
    bracket = _display_interpolation_bracket(frames, frame_position, fps)
    if bracket is None:
        return None
    left_idx, right_idx, alpha = bracket
    current = frames[left_idx]
    if left_idx == right_idx:
        return current
    following = frames[right_idx]
    if (
        following is None
        or following.mesh_player_id != current.mesh_player_id
        or following.joints_m.shape != current.joints_m.shape
    ):
        # Display interpolation is allowed only between two uninterrupted,
        # same-identity final world samples. Never hold across a missing gap.
        return None
    return WorldFrame(
        joints_m=(1.0 - alpha) * current.joints_m + alpha * following.joints_m,
        joint_conf=(1.0 - alpha) * current.joint_conf + alpha * following.joint_conf,
        mesh_player_id=current.mesh_player_id,
    )


def _display_interpolation_bracket(
    frames: dict[int, object],
    frame_position: float,
    fps: float,
    max_gap_s: float = DISPLAY_INTERPOLATION_MAX_GAP_S,
) -> tuple[int, int, float] | None:
    """Find exact or <=50ms bracketing measured ticks without holding gaps."""
    if not frames or not math.isfinite(frame_position) or fps <= 0:
        return None
    exact_idx = int(round(frame_position))
    if abs(frame_position - exact_idx) <= 1e-7 and exact_idx in frames:
        return exact_idx, exact_idx, 0.0
    search_radius = max(1, int(math.ceil(max_gap_s * fps)))
    lower: int | None = None
    upper: int | None = None
    lower_start = int(math.floor(frame_position))
    upper_start = int(math.ceil(frame_position))
    for step in range(search_radius + 1):
        candidate = lower_start - step
        if candidate in frames:
            lower = candidate
            break
    for step in range(search_radius + 1):
        candidate = upper_start + step
        if candidate in frames:
            upper = candidate
            break
    if lower is None or upper is None or lower >= upper:
        return None
    gap_s = (upper - lower) / fps
    if gap_s > max_gap_s + 1e-9:
        return None
    alpha = (frame_position - lower) / (upper - lower)
    if not 0.0 < alpha < 1.0:
        return None
    return lower, upper, float(alpha)


def resolve_fps_multiplier(
    mesh_fps: float,
    source_fps: float,
    requested: int | None,
) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--fps-multiplier must be >= 1")
        return requested
    if mesh_fps <= 0 or source_fps <= 0:
        return 1
    ratio = source_fps / mesh_fps
    nearest = int(round(ratio))
    if 1 <= nearest <= 4 and abs(ratio - nearest) <= 1e-3:
        return nearest
    return 1


def _hip_root(joints: np.ndarray, confidence: np.ndarray) -> np.ndarray | None:
    if len(joints) <= 10 or len(confidence) <= 10:
        return None
    if confidence[9] < 0.05 or confidence[10] < 0.05:
        return None
    root = (joints[9] + joints[10]) * 0.5
    return root if np.all(np.isfinite(root)) else None


def align_mesh_to_final_world(
    mesh_sample: tuple[np.ndarray, np.ndarray, np.ndarray],
    world_frame: WorldFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Mirror replay viewer root alignment and its -8cm floor guard."""
    vertices, mesh_joints, mesh_conf = mesh_sample
    mesh_root = _hip_root(mesh_joints, mesh_conf)
    world_root = _hip_root(world_frame.joints_m, world_frame.joint_conf)
    if mesh_root is None or world_root is None:
        return None
    delta = world_root - mesh_root
    translated_lowest = float(np.min(vertices[:, 2] + delta[2]))
    floor_lift = -translated_lowest if translated_lowest < -0.08 else 0.0
    if floor_lift > 0:
        delta = delta.copy()
        delta[2] += floor_lift
    return vertices + delta, world_frame.joints_m, delta


def _scaled_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, round(channel * factor)))) for channel in color)


def draw_player_shadow(panel: np.ndarray, vertices: np.ndarray, camera: dict) -> None:
    min_z = float(np.percentile(vertices[:, 2], 1.5))
    feet = vertices[vertices[:, 2] <= min_z + 0.10]
    if not len(feet):
        return
    center = np.median(feet, axis=0)
    center[2] = 0.026
    ellipse_world = np.asarray(
        [
            center + [math.cos(theta) * 0.34, math.sin(theta) * 0.18, 0]
            for theta in np.linspace(0, 2 * math.pi, 40)
        ],
        dtype=np.float32,
    )
    pixels, depth = project(ellipse_world, camera, panel.shape[1], panel.shape[0])
    if np.all(depth > 0.05):
        shadow = panel.copy()
        cv2.fillConvexPoly(shadow, pixels, (105, 111, 105), cv2.LINE_AA)
        cv2.addWeighted(shadow, 0.12, panel, 0.88, 0, panel)


def draw_translucent_joint_avatar(
    panel: np.ndarray,
    joints: np.ndarray,
    confidence: np.ndarray,
    color: tuple[int, int, int],
    camera: dict,
) -> None:
    """Draw a soft volumetric avatar without inventing additional pose data.

    Every primitive is centered on an observed final-world joint or bone.  The
    wider translucent strokes improve demo readability while the thin joint
    skeleton drawn afterward remains the exact pose reference.
    """
    if len(joints) < 70 or len(confidence) < 70:
        return
    pixels, depth = project(joints, camera, panel.shape[1], panel.shape[0])
    overlay = panel.copy()

    # Torso: shoulders and hips are the only anchors.  This is deliberately a
    # soft silhouette rather than a claimed SMPL surface.
    torso_indices = (5, 6, 10, 9)
    if all(confidence[index] >= 0.05 and depth[index] > 0.05 for index in torso_indices):
        torso = np.asarray([pixels[index] for index in torso_indices], dtype=np.int32)
        cv2.fillConvexPoly(overlay, torso, _scaled_color(color, 0.94), cv2.LINE_AA)

    limb_specs = (
        (69, 5, 15), (69, 6, 15),
        (5, 7, 13), (7, 62, 10),
        (6, 8, 13), (8, 41, 10),
        (5, 9, 15), (6, 10, 15), (9, 10, 15),
        (9, 11, 17), (11, 13, 14),
        (10, 12, 17), (12, 14, 14),
        (13, 15, 9), (13, 16, 8), (13, 17, 9),
        (14, 18, 9), (14, 19, 8), (14, 20, 9),
    )
    for left, right, width in limb_specs:
        if (
            confidence[left] < 0.05
            or confidence[right] < 0.05
            or depth[left] <= 0.05
            or depth[right] <= 0.05
        ):
            continue
        start = tuple(int(value) for value in pixels[left])
        end = tuple(int(value) for value in pixels[right])
        cv2.line(overlay, start, end, _scaled_color(color, 0.98), width, cv2.LINE_AA)
        radius = max(3, width // 2)
        cv2.circle(overlay, start, radius, _scaled_color(color, 0.98), -1, cv2.LINE_AA)
        cv2.circle(overlay, end, radius, _scaled_color(color, 0.98), -1, cv2.LINE_AA)

    # Head volume uses only nose/neck separation for scale.
    if confidence[0] >= 0.05 and confidence[69] >= 0.05 and depth[0] > 0.05:
        head_radius = int(np.clip(np.linalg.norm(pixels[0] - pixels[69]) * 0.72, 7, 18))
        cv2.circle(
            overlay,
            tuple(int(value) for value in pixels[0]),
            head_radius,
            _scaled_color(color, 1.03),
            -1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.30, panel, 0.70, 0, panel)


def draw_translucent_mesh(
    panel: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    camera: dict,
) -> None:
    pixels, depth = project(vertices, camera, panel.shape[1], panel.shape[0])
    valid_faces = faces[np.all(depth[faces] > 0.05, axis=1)]
    if not len(valid_faces):
        return

    triangles_world = vertices[valid_faces]
    normals = np.cross(
        triangles_world[:, 1] - triangles_world[:, 0],
        triangles_world[:, 2] - triangles_world[:, 0],
    )
    centers = triangles_world.mean(axis=1)
    toward_camera = camera["eye"] - centers
    facing = np.einsum("ij,ij->i", normals, toward_camera) > 0
    if np.count_nonzero(facing) < len(valid_faces) * 0.1:
        facing = ~facing
    valid_faces = valid_faces[facing]
    normals = normals[facing]
    centers = centers[facing]
    if not len(valid_faces):
        return

    norm_length = np.linalg.norm(normals, axis=1)
    normalized = normals / np.maximum(norm_length[:, None], 1e-6)
    light_direction = np.asarray([-0.32, -0.48, 0.82], dtype=np.float32)
    light_direction /= np.linalg.norm(light_direction)
    lighting = np.clip(np.abs(normalized @ light_direction), 0.0, 1.0)
    face_depth = np.linalg.norm(centers - camera["eye"], axis=1)
    depth_edges = np.quantile(face_depth, [0.0, 0.25, 0.5, 0.75, 1.0])
    shade_edges = np.asarray([0.0, 0.34, 0.68, 1.01], dtype=np.float32)

    triangles_px = np.clip(pixels[valid_faces], (-4096, -4096), (4096, 4096)).astype(np.int32)
    # Painter-style depth bands are a fast compromise between the old flat
    # fill and a full software z-buffer. Far surfaces render first.
    for depth_bin in range(3, -1, -1):
        depth_mask = (face_depth >= depth_edges[depth_bin]) & (
            face_depth <= depth_edges[depth_bin + 1] + 1e-6
        )
        for shade_bin in range(3):
            mask = depth_mask & (lighting >= shade_edges[shade_bin]) & (lighting < shade_edges[shade_bin + 1])
            if not np.any(mask):
                continue
            shade = (0.78, 0.94, 1.08)[shade_bin]
            overlay = panel.copy()
            cv2.fillPoly(overlay, triangles_px[mask], _scaled_color(color, shade), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.24, panel, 0.76, 0, panel)

    # Sparse luminous topology makes the surface read as a true articulated
    # mesh while remaining lighter than the replay UI's default fill.
    wire = triangles_px[:: max(1, len(triangles_px) // 320)]
    if len(wire):
        wire_overlay = panel.copy()
        cv2.polylines(wire_overlay, wire, True, _scaled_color(color, 0.76), 1, cv2.LINE_AA)
        cv2.addWeighted(wire_overlay, 0.42, panel, 0.58, 0, panel)


CORE_MHR70_BONES = (
    (0, 69),   # nose -> neck
    (69, 5), (69, 6), (5, 6),
    (5, 7), (7, 62),
    (6, 8), (8, 41),
    (5, 9), (6, 10), (9, 10),
    (9, 11), (11, 13),
    (10, 12), (12, 14),
    (13, 15), (13, 16), (13, 17),
    (14, 18), (14, 19), (14, 20),
)


def draw_skeleton(
    panel: np.ndarray,
    joints: np.ndarray,
    color: tuple[int, int, int],
    camera: dict,
) -> None:
    if len(joints) < 70:
        return
    pixels, depth = project(joints, camera, panel.shape[1], panel.shape[0])
    overlay = panel.copy()
    for left, right in CORE_MHR70_BONES:
        if depth[left] <= 0.05 or depth[right] <= 0.05:
            continue
        cv2.line(overlay, tuple(pixels[left]), tuple(pixels[right]), WHITE, 3, cv2.LINE_AA)
        cv2.line(overlay, tuple(pixels[left]), tuple(pixels[right]), _scaled_color(color, 0.72), 1, cv2.LINE_AA)
    for joint_idx in sorted({index for bone in CORE_MHR70_BONES for index in bone}):
        if depth[joint_idx] <= 0.05:
            continue
        cv2.circle(overlay, tuple(pixels[joint_idx]), 3, WHITE, -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(pixels[joint_idx]), 2, _scaled_color(color, 0.76), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, panel, 0.28, 0, panel)


def draw_player_label(
    panel: np.ndarray,
    player_id: int,
    vertices: np.ndarray,
    color: tuple[int, int, int],
    camera: dict,
    occupied: list[tuple[int, int, int, int]],
) -> None:
    head = vertices[int(np.argmax(vertices[:, 2]))].copy()
    head[2] += 0.18
    point, depth = project(head[None, :], camera, panel.shape[1], panel.shape[0])
    if depth[0] <= 0.05:
        return
    x, y = int(point[0, 0]), int(point[0, 1])
    label = f"P{player_id}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    left, top = x - text_w // 2 - 8, y - text_h - 9
    right, bottom = x + text_w // 2 + 8, y + 4
    anchor_y = bottom
    while any(
        left < other_right + 5
        and right > other_left - 5
        and top < other_bottom + 5
        and bottom > other_top - 5
        for other_left, other_top, other_right, other_bottom in occupied
    ):
        top -= text_h + 14
        bottom -= text_h + 14
    occupied.append((left, top, right, bottom))
    if bottom < anchor_y - 5:
        cv2.line(panel, (x, anchor_y), (x, bottom), _scaled_color(color, 0.72), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (left, top), (right, bottom), WHITE, -1, cv2.LINE_AA)
    cv2.rectangle(panel, (left, top), (right, bottom), _scaled_color(color, 0.72), 1, cv2.LINE_AA)
    cv2.putText(panel, label, (x - text_w // 2, y - 4), cv2.FONT_HERSHEY_DUPLEX, 0.55, TEXT_INK, 1, cv2.LINE_AA)


def draw_virtual_panel(
    width: int,
    height: int,
    frame_position: float,
    mesh_frames: dict[int, dict[int, MeshFrame]],
    world_frames: dict[int, dict[int, WorldFrame]],
    faces: np.ndarray,
    fps: float,
    orbit_degrees: float = 0.0,
    alignment_stats: dict[str, int] | None = None,
) -> np.ndarray:
    # Very subtle vertical warmth replaces the old black void.
    top = np.asarray(CREAM, dtype=np.float32)
    bottom = np.asarray((219, 231, 226), dtype=np.float32)
    gradient = np.linspace(top, bottom, height, dtype=np.float32)[:, None, :]
    panel = np.repeat(gradient, width, axis=1).astype(np.uint8)
    camera = baseline_camera(width, height, orbit_degrees)
    draw_court(panel, camera)

    samples: list[tuple[int, np.ndarray | None, np.ndarray, np.ndarray]] = []
    for player_id in sorted(world_frames):
        world_frame = world_sample_at(world_frames[player_id], frame_position, fps)
        if world_frame is None:
            if alignment_stats is not None:
                alignment_stats["missing_world_player_frames"] += 1
            continue
        player_mesh_frames = mesh_frames.get(world_frame.mesh_player_id)
        mesh_sample = mesh_sample_at(player_mesh_frames, frame_position, fps) if player_mesh_frames is not None else None
        if mesh_sample is None:
            samples.append((player_id, None, world_frame.joints_m, world_frame.joint_conf))
            if alignment_stats is not None:
                alignment_stats["skeleton_avatar_frames"] += 1
            continue
        aligned = align_mesh_to_final_world(mesh_sample, world_frame)
        if aligned is None:
            if alignment_stats is not None:
                alignment_stats["unalignable_player_frames"] += 1
            continue
        samples.append((player_id, aligned[0], aligned[1], world_frame.joint_conf))
        if alignment_stats is not None:
            alignment_stats["aligned_player_frames"] += 1
    for _, vertices, joints, _ in samples:
        draw_player_shadow(panel, vertices if vertices is not None else joints, camera)
    # Far players render first for stable inter-player occlusion.
    samples.sort(
        key=lambda item: float(
            np.mean(
                np.linalg.norm(
                    (item[1] if item[1] is not None else item[2]) - camera["eye"],
                    axis=1,
                )
            )
        ),
        reverse=True,
    )
    occupied_labels: list[tuple[int, int, int, int]] = []
    for player_id, vertices, joints, confidence in samples:
        color = PLAYER_COLORS.get(player_id, (190, 190, 190))
        if vertices is not None and len(faces):
            draw_translucent_mesh(panel, vertices, faces, color, camera)
        else:
            draw_translucent_joint_avatar(panel, joints, confidence, color, camera)
        draw_skeleton(panel, joints, color, camera)
        draw_player_label(
            panel,
            player_id,
            vertices if vertices is not None else joints,
            color,
            camera,
            occupied_labels,
        )
    return panel


def fit_source_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), CREAM, dtype=np.uint8)
    if frame.size == 0:
        return panel
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    target_w = max(1, int(round(source_w * scale)))
    target_h = max(1, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
    x0 = (width - target_w) // 2
    y0 = (height - target_h) // 2
    panel[y0 : y0 + target_h, x0 : x0 + target_w] = resized
    return panel


def draw_panel_chrome(panel: np.ndarray, label: str, detail: str) -> None:
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.63, 1)
    detail_size, _ = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    chip_width = max(label_size[0], detail_size[0]) + 28
    overlay = panel.copy()
    cv2.rectangle(overlay, (18, 18), (18 + chip_width, 70), WHITE, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.91, panel, 0.09, 0, panel)
    cv2.rectangle(panel, (18, 18), (18 + chip_width, 70), (201, 207, 202), 1, cv2.LINE_AA)
    cv2.putText(panel, label, (31, 42), cv2.FONT_HERSHEY_DUPLEX, 0.63, TEXT_INK, 1, cv2.LINE_AA)
    cv2.putText(panel, detail, (31, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)


def read_source_frame_at(
    capture: cv2.VideoCapture,
    source_frame_idx: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    cached = cache.get(source_frame_idx)
    if cached is not None:
        return cached
    current_position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    if current_position != source_frame_idx:
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
    ok, frame = capture.read()
    if not ok:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    cache.clear()
    cache[source_frame_idx] = frame
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="optional matching BODY mesh index; omit for joint-avatar rendering",
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument(
        "--world",
        type=Path,
        default=None,
        help="final virtual_world.json; defaults to the body_mesh_index run directory",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--fps-multiplier",
        type=int,
        default=None,
        help="display multiplier; defaults to the integer native-source/mesh FPS ratio",
    )
    parser.add_argument("--panel-width", type=int, default=1280)
    parser.add_argument("--panel-height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--encoder-preset", default="medium")
    parser.add_argument(
        "--camera-motion",
        choices=("static", "subtle_orbit"),
        default="static",
        help="static is the comparison/acceptance view; subtle_orbit is presentation-only",
    )
    args = parser.parse_args()

    if args.panel_width < 320 or args.panel_height < 240:
        raise ValueError("panel dimensions must be at least 320x240")
    if args.index is not None:
        index, faces, mesh_frames = load_mesh_frames(args.index)
        fps = float(index.get("fps", 30.0))
    else:
        index, faces, mesh_frames = {}, np.empty((0, 3), dtype=np.int32), {}
        fps = 0.0
    world_path = args.world or (
        args.index.parent.parent / "virtual_world.json"
        if args.index is not None
        else None
    )
    if world_path is None:
        raise ValueError("--world is required when --index is omitted")
    if not world_path.is_file():
        raise ValueError(
            f"final virtual_world.json is required for refined mesh placement: {world_path}"
        )
    capture = cv2.VideoCapture(str(args.source_video))
    if not capture.isOpened():
        raise ValueError(f"could not open source video: {args.source_video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps or 30.0)
    if fps <= 0:
        world_payload = json.loads(world_path.read_text(encoding="utf-8"))
        fps = float(world_payload.get("fps") or source_fps)
    world_frames = load_world_frames(world_path, fps)
    frame_start = (
        min(int(window.get("frame_start", 0)) for window in index["windows"])
        if index.get("windows")
        else 0
    )
    if index.get("windows"):
        frame_end = max(int(window.get("frame_end", 0)) for window in index["windows"])
        total_frames = frame_end - frame_start + 1
    else:
        source_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        total_frames = max(1, int(math.ceil(source_frame_count * fps / source_fps)))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)
    output_width = args.panel_width * 2
    output_height = args.panel_height
    fps_multiplier = resolve_fps_multiplier(fps, source_fps, args.fps_multiplier)
    output_fps = fps * fps_multiplier
    output_frames = total_frames * fps_multiplier

    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{output_width}x{output_height}", "-r", f"{output_fps:g}", "-i", "-",
        "-i", str(args.source_video),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", args.encoder_preset, "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
        str(args.out),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    source_cache: dict[int, np.ndarray] = {}
    alignment_stats = {
        "aligned_player_frames": 0,
        "missing_world_player_frames": 0,
        "missing_mesh_player_frames": 0,
        "unalignable_player_frames": 0,
        "skeleton_avatar_frames": 0,
    }
    try:
        for local_frame_idx in range(total_frames):
            mesh_frame_idx = frame_start + local_frame_idx
            for subframe in range(fps_multiplier):
                alpha = subframe / fps_multiplier
                frame_position = mesh_frame_idx + alpha
                time_seconds = frame_position / fps
                source_position = time_seconds * source_fps
                source_idx = int(math.floor(source_position + 1e-7))
                source_alpha = float(source_position - source_idx)
                source_current = read_source_frame_at(capture, source_idx, source_cache)
                source = source_current
                if source_alpha > 1e-6:
                    source_following = read_source_frame_at(capture, source_idx + 1, source_cache)
                    if source_following.shape == source_current.shape:
                        source = cv2.addWeighted(source_current, 1.0 - source_alpha, source_following, source_alpha, 0)
                source_panel = fit_source_panel(source, args.panel_width, args.panel_height)

                if args.camera_motion == "subtle_orbit":
                    fraction = (local_frame_idx + alpha) / max(1, total_frames - 1)
                    orbit = 2.4 * math.sin(fraction * math.pi * 2)
                else:
                    orbit = 0.0
                virtual_panel = draw_virtual_panel(
                    args.panel_width,
                    args.panel_height,
                    frame_position,
                    mesh_frames,
                    world_frames,
                    faces,
                    fps,
                    orbit_degrees=orbit,
                    alignment_stats=alignment_stats,
                )
                draw_panel_chrome(source_panel, "SOURCE", "input video")
                draw_panel_chrome(virtual_panel, "VIRTUAL BASELINE", "final world placement")
                canvas = np.hstack((source_panel, virtual_panel))
                cv2.line(
                    canvas,
                    (args.panel_width, 0),
                    (args.panel_width, output_height),
                    WHITE,
                    5,
                    cv2.LINE_AA,
                )
                time_label = f"{time_seconds:05.2f}s"
                (time_w, _), _ = cv2.getTextSize(time_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.putText(
                    canvas,
                    time_label,
                    (output_width - time_w - 22, output_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    TEXT_MUTED,
                    1,
                    cv2.LINE_AA,
                )
                assert encoder.stdin is not None
                encoder.stdin.write(canvas.tobytes())
            if local_frame_idx % 30 == 0:
                print(
                    json.dumps(
                        {
                            "source_frame": local_frame_idx,
                            "source_total": total_frames,
                            "output_frame": (local_frame_idx + 1) * fps_multiplier,
                            "output_total": output_frames,
                        }
                    ),
                    flush=True,
                )
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with {return_code}")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "source_frames": total_frames,
                "frames": output_frames,
                "fps": output_fps,
                "duration_s": output_frames / output_fps,
                "fps_multiplier": fps_multiplier,
                "layout": "source_plus_virtual_baseline",
                "camera_motion": args.camera_motion,
                "renderer_authority": "presentation_only",
                "mesh_alignment": "virtual_world_skeleton_root_plus_floor_guard",
                "virtual_representation": (
                    "body_mesh_plus_exact_skeleton"
                    if args.index is not None
                    else "translucent_joint_avatar_plus_exact_skeleton"
                ),
                "virtual_world": str(world_path),
                **alignment_stats,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
