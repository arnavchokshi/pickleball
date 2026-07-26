#!/usr/bin/env python3
"""Render a polished source-video + virtual-baseline comparison MP4.

The renderer consumes the final immutable ``virtual_world.json`` placement and,
when available, the matching BODY mesh index. Native-mesh closeups render the
actual 18,439-vertex BODY surface and may be configured to fail instead of
falling back. Joint-only runs are displayed as translucent articulated avatars
built directly from their final grounded joints; this is a presentation surface,
not fabricated measurement geometry. Missing world samples remain missing.
Optional between-frame interpolation is display-only and is used only when both
adjacent measured samples exist. Native closeups can denoise corresponding mesh
vertices in body-local space and interpolate the immutable 30 Hz evidence onto
a 60 FPS presentation timeline; neither operation creates measurements.

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
PRESENTATION_ANCHOR_TAU_S = 0.15
PRESENTATION_OFFSET_TAU_S = 0.30
PRESENTATION_ROOT_TAU_S = 0.20
PRESENTATION_STABILIZATION_MEDIAN_RADIUS = 3
PRESENTATION_MAX_SPEED_MPS = 7.5
PRESENTATION_CROP_TAU_S = 0.18
PRESENTATION_CROP_MEDIAN_RADIUS = 3
PRESENTATION_MESH_TAU_S = 0.065
PRESENTATION_MESH_MEDIAN_RADIUS = 1
PLAYER_CLOSEUP_MAX_BODY_HZ = 30.0


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
    floor_xy_m: np.ndarray | None = None
    bbox_xyxy: np.ndarray | None = None
    raw_bbox_xyxy: np.ndarray | None = None
    translation_world: np.ndarray | None = None


def stabilize_mesh_frames_for_presentation(
    frames: dict[int, MeshFrame],
    fps: float,
    *,
    median_radius: int = PRESENTATION_MESH_MEDIAN_RADIUS,
    tau_s: float = PRESENTATION_MESH_TAU_S,
) -> tuple[dict[int, MeshFrame], dict[str, float | int | str]]:
    """Denoise native surface motion in body-local space for display only.

    BODY produces independent 30 Hz mesh observations.  Filtering their raw
    court translation does not help a body-local close-up, so this operates on
    every corresponding native vertex and joint after removing hip-root XY.
    It preserves frame keys, topology, identity/window boundaries, and missing
    gaps.  The filtered coordinates are never written back to BODY artifacts or
    exposed as measurements.
    """
    if fps <= 0:
        raise ValueError("mesh presentation stabilization requires positive fps")
    if median_radius < 0 or tau_s <= 0:
        raise ValueError("mesh presentation stabilization parameters are invalid")

    output = dict(frames)
    segments: list[list[int]] = []
    segment: list[int] = []
    for frame_idx in sorted(frames):
        frame = frames[frame_idx]
        valid = bool(
            frame.vertices_mm.ndim == 2
            and frame.vertices_mm.shape[1] == 3
            and frame.joints_mm.ndim == 2
            and frame.joints_mm.shape[1] == 3
            and len(frame.joints_mm) > 10
            and len(frame.joint_conf) == len(frame.joints_mm)
        )
        if not valid:
            if segment:
                segments.append(segment)
                segment = []
            continue
        if segment:
            previous_idx = segment[-1]
            previous = frames[previous_idx]
            if (
                frame_idx != previous_idx + 1
                or frame.source_window_index != previous.source_window_index
                or frame.vertices_mm.shape != previous.vertices_mm.shape
                or frame.joints_mm.shape != previous.joints_mm.shape
            ):
                segments.append(segment)
                segment = []
        segment.append(frame_idx)
    if segment:
        segments.append(segment)

    raw_steps: list[float] = []
    filtered_steps: list[float] = []
    corrections: list[float] = []
    filtered_frames = 0
    filtered_segments = 0
    for indices in segments:
        # A symmetric filter needs enough context to avoid turning a short
        # appearance burst into an implied smooth track.
        if len(indices) < 5:
            continue
        vertices = np.asarray(
            [frames[index].vertices_mm for index in indices],
            dtype=np.float32,
        ) / 1000.0
        joints = np.asarray(
            [frames[index].joints_mm for index in indices],
            dtype=np.float32,
        ) / 1000.0
        roots_xy = 0.5 * (joints[:, 9, :2] + joints[:, 10, :2])
        local_vertices = vertices.copy()
        local_joints = joints.copy()
        local_vertices[:, :, :2] -= roots_xy[:, None, :]
        local_joints[:, :, :2] -= roots_xy[:, None, :]
        times = np.asarray(indices, dtype=np.float64) / fps

        filtered_vertices = _symmetric_ema(
            _coordinate_median_filter(local_vertices, median_radius),
            times,
            tau_s,
        )
        filtered_joints = _symmetric_ema(
            _coordinate_median_filter(local_joints, median_radius),
            times,
            tau_s,
        )
        confidence = np.asarray(
            [frames[index].joint_conf for index in indices],
            dtype=np.float32,
        )
        filtered_confidence = np.clip(
            _symmetric_ema(confidence, times, tau_s),
            0.0,
            1.0,
        )

        raw_delta = np.diff(local_vertices, axis=0)
        filtered_delta = np.diff(filtered_vertices, axis=0)
        raw_steps.extend(
            np.sqrt(np.mean(np.sum(raw_delta * raw_delta, axis=2), axis=1)).tolist()
        )
        filtered_steps.extend(
            np.sqrt(
                np.mean(np.sum(filtered_delta * filtered_delta, axis=2), axis=1)
            ).tolist()
        )
        correction = filtered_vertices - local_vertices
        corrections.extend(
            np.sqrt(np.mean(np.sum(correction * correction, axis=2), axis=1)).tolist()
        )

        # Restore the original root translation so the frame remains a valid
        # native BODY coordinate sample.  The studio path removes it again.
        filtered_vertices[:, :, :2] += roots_xy[:, None, :]
        filtered_joints[:, :, :2] += roots_xy[:, None, :]
        for offset, frame_idx in enumerate(indices):
            original = frames[frame_idx]
            output[frame_idx] = MeshFrame(
                vertices_mm=np.rint(filtered_vertices[offset] * 1000.0).astype(np.int16),
                joints_mm=np.rint(filtered_joints[offset] * 1000.0).astype(np.int16),
                joint_conf=filtered_confidence[offset].astype(np.float32),
                blend_weight=original.blend_weight,
                source_window_index=original.source_window_index,
            )
            filtered_frames += 1
        filtered_segments += 1

    def _percentile(values: list[float], percentile: float) -> float:
        return float(np.percentile(values, percentile)) if values else 0.0

    return output, {
        "mode": "body_local_native_vertex_joint_robust_symmetric",
        "authority": "presentation_only",
        "segments": filtered_segments,
        "frames": filtered_frames,
        "median_radius_frames": median_radius,
        "symmetric_ema_tau_s": tau_s,
        "raw_surface_step_p95_m": _percentile(raw_steps, 95),
        "filtered_surface_step_p95_m": _percentile(filtered_steps, 95),
        "surface_correction_median_m": _percentile(corrections, 50),
        "surface_correction_p95_m": _percentile(corrections, 95),
        "surface_correction_max_m": max(corrections, default=0.0),
    }


def _hip_root_xy(frame: WorldFrame) -> np.ndarray | None:
    """Return the final-world hip root used only for rigid display translation."""
    root = _hip_root(frame.joints_m, frame.joint_conf)
    if root is None:
        return None
    return root[:2].astype(np.float64)


def _coordinate_median_filter(values: np.ndarray, radius: int) -> np.ndarray:
    filtered = values.copy()
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        filtered[index] = np.median(values[start:end], axis=0)
    return filtered


def _symmetric_ema(values: np.ndarray, times: np.ndarray, tau_s: float) -> np.ndarray:
    """Zero-lag-ish offline smoother that retains real low-frequency motion."""

    def _forward(samples: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
        output = samples.copy()
        for index in range(1, len(samples)):
            dt = max(1e-4, float(sample_times[index] - sample_times[index - 1]))
            alpha = dt / (tau_s + dt)
            output[index] = output[index - 1] + alpha * (samples[index] - output[index - 1])
        return output

    forward = _forward(values, times)
    backward = _forward(values[::-1], -times[::-1])[::-1]
    return 0.5 * (forward + backward)


def _bounded_speed_projection(
    values: np.ndarray,
    times: np.ndarray,
    max_speed_mps: float,
) -> np.ndarray:
    """Symmetrically limit only physically implausible display-root snaps."""

    def _forward(samples: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
        output = samples.copy()
        for index in range(1, len(samples)):
            dt = max(1e-4, float(sample_times[index] - sample_times[index - 1]))
            delta = samples[index] - output[index - 1]
            distance = float(np.linalg.norm(delta))
            maximum = max_speed_mps * dt
            if distance > maximum and distance > 1e-9:
                delta *= maximum / distance
            output[index] = output[index - 1] + delta
        return output

    forward = _forward(values, times)
    backward = _forward(values[::-1], -times[::-1])[::-1]
    return 0.5 * (forward + backward)


def stabilize_world_frames_for_presentation(
    world_frames: dict[int, dict[int, WorldFrame]],
    fps: float,
    *,
    max_gap_s: float = DISPLAY_INTERPOLATION_MAX_GAP_S,
    median_radius: int = PRESENTATION_STABILIZATION_MEDIAN_RADIUS,
    root_tau_s: float = PRESENTATION_ROOT_TAU_S,
    anchor_tau_s: float = PRESENTATION_ANCHOR_TAU_S,
    offset_tau_s: float = PRESENTATION_OFFSET_TAU_S,
    max_speed_mps: float = PRESENTATION_MAX_SPEED_MPS,
) -> tuple[dict[int, dict[int, WorldFrame]], dict[str, float | int | str]]:
    """Smooth root XY per uninterrupted identity without changing pose or data.

    This transform is intentionally confined to exported presentation video.
    It never creates a frame, bridges a missing gap, changes player identity, or
    mutates ``virtual_world.json``.  The same rigid XY delta is applied to all
    joints, so joint angles and bone lengths remain exactly as persisted.
    """
    if fps <= 0:
        raise ValueError("presentation stabilization requires positive fps")
    stabilized: dict[int, dict[int, WorldFrame]] = {}
    corrections: list[float] = []
    raw_steps: list[float] = []
    stabilized_steps: list[float] = []
    segment_count = 0
    stabilized_frame_count = 0
    anchor_segment_count = 0
    root_fallback_segment_count = 0

    for player_id, player_frames in world_frames.items():
        output = dict(player_frames)
        ordered = sorted(player_frames)
        segments: list[list[int]] = []
        segment: list[int] = []
        for frame_idx in ordered:
            current = player_frames[frame_idx]
            root = _hip_root_xy(current)
            if root is None:
                if segment:
                    segments.append(segment)
                    segment = []
                continue
            if segment:
                previous_idx = segment[-1]
                previous = player_frames[previous_idx]
                gap_s = (frame_idx - previous_idx) / fps
                if (
                    gap_s > max_gap_s + 1e-9
                    or current.mesh_player_id != previous.mesh_player_id
                ):
                    segments.append(segment)
                    segment = []
            segment.append(frame_idx)
        if segment:
            segments.append(segment)

        for indices in segments:
            # Very short bursts are left untouched; smoothing them can turn an
            # isolated detection into a visually implied track.
            if len(indices) < 5:
                continue
            roots = np.asarray(
                [_hip_root_xy(player_frames[frame_idx]) for frame_idx in indices],
                dtype=np.float64,
            )
            times = np.asarray(indices, dtype=np.float64) / fps
            anchors = [player_frames[frame_idx].floor_xy_m for frame_idx in indices]
            if all(anchor is not None and np.all(np.isfinite(anchor)) for anchor in anchors):
                anchor_values = np.asarray(anchors, dtype=np.float64)
                offsets = roots - anchor_values
                smooth_anchor = _symmetric_ema(
                    _coordinate_median_filter(anchor_values, median_radius),
                    times,
                    anchor_tau_s,
                )
                smooth_offset = _symmetric_ema(
                    _coordinate_median_filter(offsets, median_radius),
                    times,
                    offset_tau_s,
                )
                smooth_roots = smooth_anchor + smooth_offset
                anchor_segment_count += 1
            else:
                robust_roots = _coordinate_median_filter(roots, median_radius)
                smooth_roots = _symmetric_ema(robust_roots, times, root_tau_s)
                root_fallback_segment_count += 1
            smooth_roots = _bounded_speed_projection(
                smooth_roots,
                times,
                max_speed_mps,
            )
            raw_steps.extend(np.linalg.norm(np.diff(roots, axis=0), axis=1).tolist())
            stabilized_steps.extend(
                np.linalg.norm(np.diff(smooth_roots, axis=0), axis=1).tolist()
            )
            segment_count += 1
            for frame_idx, raw_root, smooth_root in zip(
                indices, roots, smooth_roots, strict=True
            ):
                frame = player_frames[frame_idx]
                delta_xy = (smooth_root - raw_root).astype(np.float32)
                joints = frame.joints_m.copy()
                joints[:, :2] += delta_xy
                output[frame_idx] = WorldFrame(
                    joints_m=joints,
                    joint_conf=frame.joint_conf,
                    mesh_player_id=frame.mesh_player_id,
                    floor_xy_m=frame.floor_xy_m,
                    bbox_xyxy=frame.bbox_xyxy,
                    raw_bbox_xyxy=frame.raw_bbox_xyxy,
                    translation_world=frame.translation_world,
                )
                corrections.append(float(np.linalg.norm(delta_xy)))
                stabilized_frame_count += 1
        stabilized[player_id] = output

    def _percentile(values: list[float], percentile: float) -> float:
        return float(np.percentile(values, percentile)) if values else 0.0

    return stabilized, {
        "mode": "robust_gap_preserving_root_xy",
        "authority": "presentation_only",
        "segments": segment_count,
        "anchor_segments": anchor_segment_count,
        "root_fallback_segments": root_fallback_segment_count,
        "frames": stabilized_frame_count,
        "maximum_display_speed_mps": max_speed_mps,
        "correction_median_m": _percentile(corrections, 50),
        "correction_p95_m": _percentile(corrections, 95),
        "correction_max_m": max(corrections, default=0.0),
        "raw_root_step_p95_m": _percentile(raw_steps, 95),
        "stabilized_root_step_p95_m": _percentile(stabilized_steps, 95),
        "raw_root_step_max_m": max(raw_steps, default=0.0),
        "stabilized_root_step_max_m": max(stabilized_steps, default=0.0),
    }


def _valid_bbox_xyxy(value: np.ndarray | None) -> bool:
    return bool(
        value is not None
        and value.shape == (4,)
        and np.all(np.isfinite(value))
        and value[2] > value[0]
        and value[3] > value[1]
    )


def stabilize_bboxes_for_presentation(
    player_frames: dict[int, WorldFrame],
    fps: float,
    *,
    max_gap_s: float = DISPLAY_INTERPOLATION_MAX_GAP_S,
    median_radius: int = PRESENTATION_CROP_MEDIAN_RADIUS,
    tau_s: float = PRESENTATION_CROP_TAU_S,
) -> tuple[dict[int, WorldFrame], dict[str, float | int | str]]:
    """Smooth tracked crop motion without inventing boxes across missing gaps."""
    if fps <= 0:
        raise ValueError("crop stabilization requires positive fps")
    output = dict(player_frames)
    segments: list[list[int]] = []
    segment: list[int] = []
    for frame_idx in sorted(player_frames):
        frame = player_frames[frame_idx]
        if not _valid_bbox_xyxy(frame.bbox_xyxy):
            if segment:
                segments.append(segment)
                segment = []
            continue
        if segment:
            previous = player_frames[segment[-1]]
            time_gap = (frame_idx - segment[-1]) / fps
            if (
                time_gap > max_gap_s + 1e-9
                or frame.mesh_player_id != previous.mesh_player_id
            ):
                segments.append(segment)
                segment = []
        segment.append(frame_idx)
    if segment:
        segments.append(segment)

    raw_center_steps: list[float] = []
    smooth_center_steps: list[float] = []
    smoothed_frames = 0
    smoothed_segments = 0
    for indices in segments:
        if len(indices) < 5:
            continue
        boxes = np.asarray(
            [player_frames[frame_idx].bbox_xyxy for frame_idx in indices],
            dtype=np.float64,
        )
        centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 1.0)
        parameters = np.column_stack((centers, np.log(sizes)))
        times = np.asarray(indices, dtype=np.float64) / fps
        smooth = _symmetric_ema(
            _coordinate_median_filter(parameters, median_radius),
            times,
            tau_s,
        )
        smooth_centers = smooth[:, :2]
        smooth_sizes = np.exp(smooth[:, 2:])
        raw_center_steps.extend(
            np.linalg.norm(np.diff(centers, axis=0), axis=1).tolist()
        )
        smooth_center_steps.extend(
            np.linalg.norm(np.diff(smooth_centers, axis=0), axis=1).tolist()
        )
        for frame_idx, center, size in zip(
            indices, smooth_centers, smooth_sizes, strict=True
        ):
            frame = player_frames[frame_idx]
            half = 0.5 * size
            output[frame_idx] = WorldFrame(
                joints_m=frame.joints_m,
                joint_conf=frame.joint_conf,
                mesh_player_id=frame.mesh_player_id,
                floor_xy_m=frame.floor_xy_m,
                bbox_xyxy=np.concatenate((center - half, center + half)).astype(np.float32),
                raw_bbox_xyxy=frame.raw_bbox_xyxy,
                translation_world=frame.translation_world,
            )
            smoothed_frames += 1
        smoothed_segments += 1

    def _percentile(values: list[float], percentile: float) -> float:
        return float(np.percentile(values, percentile)) if values else 0.0

    return output, {
        "mode": "gap_preserving_tracked_crop",
        "authority": "presentation_only",
        "segments": smoothed_segments,
        "frames": smoothed_frames,
        "raw_center_step_p95_px": _percentile(raw_center_steps, 95),
        "smoothed_center_step_p95_px": _percentile(smooth_center_steps, 95),
    }


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
            floor_value = frame.get("floor_world_xyz")
            if not isinstance(floor_value, (list, tuple)) or len(floor_value) < 2:
                floor_value = frame.get("track_world_xy")
            floor_xy_m: np.ndarray | None = None
            if isinstance(floor_value, (list, tuple)) and len(floor_value) >= 2:
                candidate_floor = np.asarray(floor_value[:2], dtype=np.float32)
                if np.all(np.isfinite(candidate_floor)):
                    floor_xy_m = candidate_floor
            bbox_value = frame.get("bbox")
            bbox_xyxy: np.ndarray | None = None
            if isinstance(bbox_value, (list, tuple)) and len(bbox_value) == 4:
                candidate_bbox = np.asarray(bbox_value, dtype=np.float32)
                if _valid_bbox_xyxy(candidate_bbox):
                    bbox_xyxy = candidate_bbox
            translation_value = frame.get("transl_world")
            translation_world: np.ndarray | None = None
            if isinstance(translation_value, (list, tuple)) and len(translation_value) == 3:
                candidate_translation = np.asarray(translation_value, dtype=np.float32)
                if np.all(np.isfinite(candidate_translation)):
                    translation_world = candidate_translation
            frames[frame_idx] = WorldFrame(
                joints_m=joints,
                joint_conf=conf,
                mesh_player_id=mesh_player_id,
                floor_xy_m=floor_xy_m,
                bbox_xyxy=bbox_xyxy,
                raw_bbox_xyxy=bbox_xyxy.copy() if bbox_xyxy is not None else None,
                translation_world=translation_world,
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
        # Leave a generous apron around the full court so players legitimately
        # standing behind a baseline remain visible instead of clipping against
        # the panel edge.
        "focal": float(height) * 1.25,
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
    *,
    terminal_hold_max_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    bracket = _display_interpolation_bracket(
        frames,
        frame_position,
        fps,
        terminal_hold_max_s=terminal_hold_max_s,
    )
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
    *,
    max_gap_frames: int | None = None,
    terminal_hold_max_s: float = 0.0,
) -> WorldFrame | None:
    bracket = _display_interpolation_bracket(
        frames,
        frame_position,
        fps,
        max_gap_frames=max_gap_frames,
        terminal_hold_max_s=terminal_hold_max_s,
    )
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
        floor_xy_m=(
            (1.0 - alpha) * current.floor_xy_m + alpha * following.floor_xy_m
            if current.floor_xy_m is not None and following.floor_xy_m is not None
            else None
        ),
        bbox_xyxy=(
            (1.0 - alpha) * current.bbox_xyxy + alpha * following.bbox_xyxy
            if _valid_bbox_xyxy(current.bbox_xyxy)
            and _valid_bbox_xyxy(following.bbox_xyxy)
            else None
        ),
        raw_bbox_xyxy=(
            (1.0 - alpha) * current.raw_bbox_xyxy + alpha * following.raw_bbox_xyxy
            if _valid_bbox_xyxy(current.raw_bbox_xyxy)
            and _valid_bbox_xyxy(following.raw_bbox_xyxy)
            else None
        ),
        translation_world=(
            (1.0 - alpha) * current.translation_world + alpha * following.translation_world
            if current.translation_world is not None
            and following.translation_world is not None
            else None
        ),
    )


def _display_interpolation_bracket(
    frames: dict[int, object],
    frame_position: float,
    fps: float,
    max_gap_s: float = DISPLAY_INTERPOLATION_MAX_GAP_S,
    *,
    max_gap_frames: int | None = None,
    terminal_hold_max_s: float = 0.0,
) -> tuple[int, int, float] | None:
    """Find exact or <=50ms bracketing ticks without bridging missing gaps.

    A caller may permit one sub-frame hold after the final native observation.
    This exists solely to encode a complete 60 FPS final second from 30 FPS
    input; it cannot fill an internal gap or extend a missing identity.
    """
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
    if (
        upper is None
        and lower is not None
        and lower == max(frames)
        and terminal_hold_max_s > 0
        and (frame_position - lower) / fps <= terminal_hold_max_s + 1e-9
    ):
        return lower, lower, 0.0
    if lower is None or upper is None or lower >= upper:
        return None
    if max_gap_frames is not None and upper - lower > max_gap_frames:
        return None
    gap_s = (upper - lower) / fps
    if gap_s > max_gap_s + 1e-9:
        return None
    alpha = (frame_position - lower) / (upper - lower)
    if not 0.0 < alpha < 1.0:
        return None
    return lower, upper, float(alpha)


def display_sample_kind(
    frames: dict[int, object],
    frame_position: float,
    fps: float,
    *,
    max_gap_frames: int | None = None,
    terminal_hold_max_s: float = 0.0,
) -> str:
    """Return honest timing provenance for a display sample."""
    bracket = _display_interpolation_bracket(
        frames,
        frame_position,
        fps,
        max_gap_frames=max_gap_frames,
        terminal_hold_max_s=terminal_hold_max_s,
    )
    if bracket is None:
        return "missing"
    left_idx, right_idx, _ = bracket
    exact_idx = int(round(frame_position))
    if (
        left_idx == right_idx
        and abs(frame_position - exact_idx) <= 1e-7
        and exact_idx in frames
    ):
        return "measured_tick"
    if left_idx == right_idx:
        return "terminal_display_hold"
    return "display_interpolated"


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
    *,
    surface_opacity: float = 0.24,
    wire_opacity: float = 0.42,
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
            cv2.addWeighted(
                overlay,
                surface_opacity,
                panel,
                1.0 - surface_opacity,
                0,
                panel,
            )

    # Sparse luminous topology makes the surface read as a true articulated
    # mesh while remaining lighter than the replay UI's default fill.
    wire = triangles_px[:: max(1, len(triangles_px) // 320)]
    if len(wire):
        wire_overlay = panel.copy()
        cv2.polylines(wire_overlay, wire, True, _scaled_color(color, 0.76), 1, cv2.LINE_AA)
        cv2.addWeighted(
            wire_overlay,
            wire_opacity,
            panel,
            1.0 - wire_opacity,
            0,
            panel,
        )


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

DETAIL_MHR70_BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 67), (6, 68), (7, 63), (7, 65), (8, 64), (8, 66),
    (41, 24), (24, 23), (23, 22), (22, 21),
    (41, 28), (28, 27), (27, 26), (26, 25),
    (41, 32), (32, 31), (31, 30), (30, 29),
    (41, 36), (36, 35), (35, 34), (34, 33),
    (41, 40), (40, 39), (39, 38), (38, 37),
    (62, 45), (45, 44), (44, 43), (43, 42),
    (62, 49), (49, 48), (48, 47), (47, 46),
    (62, 53), (53, 52), (52, 51), (51, 50),
    (62, 57), (57, 56), (56, 55), (55, 54),
    (62, 61), (61, 60), (60, 59), (59, 58),
)
DETAILED_MHR70_BONES = CORE_MHR70_BONES + DETAIL_MHR70_BONES


def draw_skeleton(
    panel: np.ndarray,
    joints: np.ndarray,
    color: tuple[int, int, int],
    camera: dict,
    bones: tuple[tuple[int, int], ...] = CORE_MHR70_BONES,
    confidence: np.ndarray | None = None,
    min_confidence: float = 0.05,
    underlay_width: int = 3,
    line_width: int = 1,
    joint_radius: int = 3,
) -> None:
    if len(joints) < 70:
        return
    pixels, depth = project(joints, camera, panel.shape[1], panel.shape[0])
    overlay = panel.copy()
    for left, right in bones:
        if (
            confidence is not None
            and (confidence[left] < min_confidence or confidence[right] < min_confidence)
        ):
            continue
        if depth[left] <= 0.05 or depth[right] <= 0.05:
            continue
        cv2.line(
            overlay,
            tuple(pixels[left]),
            tuple(pixels[right]),
            WHITE,
            underlay_width,
            cv2.LINE_AA,
        )
        cv2.line(
            overlay,
            tuple(pixels[left]),
            tuple(pixels[right]),
            _scaled_color(color, 0.72),
            line_width,
            cv2.LINE_AA,
        )
    for joint_idx in sorted({index for bone in bones for index in bone}):
        if confidence is not None and confidence[joint_idx] < min_confidence:
            continue
        if depth[joint_idx] <= 0.05:
            continue
        cv2.circle(overlay, tuple(pixels[joint_idx]), joint_radius, WHITE, -1, cv2.LINE_AA)
        cv2.circle(
            overlay,
            tuple(pixels[joint_idx]),
            max(1, joint_radius - 2),
            _scaled_color(color, 0.76),
            -1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.72, panel, 0.28, 0, panel)


def person_studio_camera(
    width: int,
    height: int,
    orbit_degrees: float,
) -> dict:
    angle = math.radians(orbit_degrees)
    radius = 3.55
    eye = np.asarray(
        [radius * math.sin(angle), -radius * math.cos(angle), 1.72],
        dtype=np.float32,
    )
    target = np.asarray([0.0, 0.0, 0.92], dtype=np.float32)
    return {
        "eye": eye,
        "basis": camera_basis(eye, target, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)),
        "focal": float(height) * 1.72,
    }


def center_joints_for_studio(frame: WorldFrame) -> np.ndarray | None:
    """Remove court-plane travel while preserving pose and vertical motion."""
    root = _hip_root(frame.joints_m, frame.joint_conf)
    if root is None:
        return None
    joints = frame.joints_m.copy()
    joints[:, 0] -= root[0]
    joints[:, 1] -= root[1]
    return joints


def center_mesh_for_studio(
    mesh_sample: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Remove only mesh court-plane travel; preserve its exact surface pose."""
    vertices, joints, confidence = mesh_sample
    root = _hip_root(joints, confidence)
    if root is None or vertices.ndim != 2 or vertices.shape[1] != 3:
        return None
    local_vertices = vertices.copy()
    local_joints = joints.copy()
    local_vertices[:, :2] -= root[:2]
    local_joints[:, :2] -= root[:2]
    return local_vertices, local_joints, confidence


def draw_person_studio_floor(panel: np.ndarray, camera: dict) -> None:
    # A soft, unscaled studio pedestal provides visual grounding without
    # implying court placement or metric floor authority.
    disc = np.asarray(
        [[1.05 * math.cos(theta), 1.05 * math.sin(theta), -0.012]
         for theta in np.linspace(0, 2 * math.pi, 72)],
        dtype=np.float32,
    )
    _fill_world_polygon(panel, disc, (235, 239, 235), camera)
    ring = np.asarray(
        [[0.72 * math.cos(theta), 0.72 * math.sin(theta), 0.005]
         for theta in np.linspace(0, 2 * math.pi, 72)],
        dtype=np.float32,
    )
    _draw_world_polyline(panel, ring, (192, 204, 197), 2, camera, closed=True, opacity=0.42)


def draw_person_studio_panel(
    width: int,
    height: int,
    frame: WorldFrame | None,
    player_id: int,
    orbit_degrees: float,
    *,
    mesh_sample: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    faces: np.ndarray | None = None,
    require_native_mesh: bool = False,
) -> np.ndarray:
    top = np.asarray((242, 246, 242), dtype=np.float32)
    bottom = np.asarray((220, 231, 225), dtype=np.float32)
    gradient = np.linspace(top, bottom, height, dtype=np.float32)[:, None, :]
    panel = np.repeat(gradient, width, axis=1).astype(np.uint8)
    camera = person_studio_camera(width, height, orbit_degrees)
    draw_person_studio_floor(panel, camera)
    if frame is None:
        cv2.putText(
            panel,
            "NO MEASURED BODY SAMPLE",
            (width // 2 - 155, height // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
        return panel
    color = PLAYER_COLORS.get(player_id, (190, 190, 190))
    if mesh_sample is not None and faces is not None and len(faces):
        centered_mesh = center_mesh_for_studio(mesh_sample)
        if centered_mesh is not None:
            vertices, joints, confidence = centered_mesh
            draw_translucent_mesh(
                panel,
                vertices,
                faces,
                color,
                camera,
                surface_opacity=0.46,
                wire_opacity=0.26,
            )
            # A restrained core overlay keeps limb motion readable while the
            # native 36k-face surface remains the dominant representation.
            draw_skeleton(
                panel,
                joints,
                color,
                camera,
                bones=CORE_MHR70_BONES,
                confidence=confidence,
                min_confidence=0.05,
                underlay_width=2,
                line_width=1,
                joint_radius=2,
            )
            draw_player_label(panel, player_id, vertices, color, camera, [])
            return panel
        if require_native_mesh:
            raise ValueError("native BODY mesh has invalid hip/surface geometry")
    if require_native_mesh:
        raise ValueError("native BODY mesh sample/topology is unavailable")
    joints = center_joints_for_studio(frame)
    if joints is None:
        return panel
    draw_translucent_joint_avatar(panel, joints, frame.joint_conf, color, camera)
    draw_skeleton(
        panel,
        joints,
        color,
        camera,
        bones=CORE_MHR70_BONES,
        confidence=frame.joint_conf,
        min_confidence=0.05,
        underlay_width=4,
        line_width=2,
        joint_radius=4,
    )
    # Fine face, foot, and finger topology is additive detail. Confidence dips
    # may hide this pass, but never erase the readable core body skeleton.
    draw_skeleton(
        panel,
        joints,
        color,
        camera,
        bones=DETAIL_MHR70_BONES,
        confidence=frame.joint_conf,
        min_confidence=0.5,
        underlay_width=2,
        line_width=1,
        joint_radius=2,
    )
    draw_player_label(panel, player_id, joints, color, camera, [])
    return panel


def studio_orbit_angle(progress: float) -> float:
    """Hold four readable body-relative studio angles with eased transitions."""
    value = float(np.clip(progress, 0.0, 1.0))

    def _smoothstep(x: float) -> float:
        clipped = float(np.clip(x, 0.0, 1.0))
        return clipped * clipped * (3.0 - 2.0 * clipped)

    if value < 0.22:
        return 0.0
    if value < 0.32:
        return 45.0 * _smoothstep((value - 0.22) / 0.10)
    if value < 0.48:
        return 45.0
    if value < 0.58:
        return 45.0 + 45.0 * _smoothstep((value - 0.48) / 0.10)
    if value < 0.74:
        return 90.0
    if value < 0.84:
        return 90.0 + 90.0 * _smoothstep((value - 0.74) / 0.10)
    return 180.0


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


def tracked_person_crop_geometry(
    smoothed_bbox: np.ndarray,
    raw_bbox: np.ndarray | None,
    width: int,
    height: int,
    source_width: int,
    source_height: int,
    *,
    body_padding: float = 1.35,
    containment_margin: float = 0.10,
) -> tuple[np.ndarray, float, float]:
    """Return a real-pixel crop with raw-box containment and best-effort guard."""
    if not _valid_bbox_xyxy(smoothed_bbox):
        raise ValueError("tracked crop requires a valid smoothed bbox")
    aspect = width / height
    center = 0.5 * (smoothed_bbox[:2] + smoothed_bbox[2:])
    size = smoothed_bbox[2:] - smoothed_bbox[:2]
    target_h = max(float(size[1]) * body_padding, float(size[0]) * body_padding / aspect)
    target_w = target_h * aspect

    if _valid_bbox_xyxy(raw_bbox):
        raw_size = raw_bbox[2:] - raw_bbox[:2]
        guard = raw_size * containment_margin
        required_half_w = max(
            float(center[0] - raw_bbox[0] + guard[0]),
            float(raw_bbox[2] - center[0] + guard[0]),
        )
        required_half_h = max(
            float(center[1] - raw_bbox[1] + guard[1]),
            float(raw_bbox[3] - center[1] + guard[1]),
        )
        target_w = max(target_w, 2.0 * required_half_w)
        target_h = max(target_h, 2.0 * required_half_h)
        if target_w / target_h < aspect:
            target_w = target_h * aspect
        else:
            target_h = target_w / aspect

    # Never synthesize a mirrored copy of the tracked player at image edges.
    # Fit the crop inside the real source image, then shift its center only as
    # much as necessary. The player can become slightly off-center near a true
    # frame boundary, which is preferable to showing duplicated source pixels.
    fit_scale = min(
        1.0,
        float(source_width) / max(target_w, 1.0),
        float(source_height) / max(target_h, 1.0),
    )
    target_w = max(8.0, target_w * fit_scale)
    target_h = max(8.0, target_h * fit_scale)
    center[0] = np.clip(center[0], target_w * 0.5, source_width - target_w * 0.5)
    center[1] = np.clip(center[1], target_h * 0.5, source_height - target_h * 0.5)
    return center.astype(np.float32), target_w, target_h


def source_seek_seconds(
    base_frame_start: int,
    world_fps: float,
    start_seconds: float,
) -> float:
    """Map a renderer-local window offset to the source video's timebase."""
    if base_frame_start < 0 or world_fps <= 0 or start_seconds < 0:
        raise ValueError("invalid source seek inputs")
    return base_frame_start / world_fps + start_seconds


def fit_tracked_person_panel(
    frame: np.ndarray,
    smoothed_bbox: np.ndarray,
    raw_bbox: np.ndarray | None,
    width: int,
    height: int,
    *,
    body_padding: float = 1.35,
    containment_margin: float = 0.10,
    player_id: int | None = None,
) -> np.ndarray:
    """Create an aspect-fixed close crop that still contains the raw detection."""
    if frame.size == 0 or not _valid_bbox_xyxy(smoothed_bbox):
        return fit_source_panel(frame, width, height)
    source_h, source_w = frame.shape[:2]
    center, target_w, target_h = tracked_person_crop_geometry(
        smoothed_bbox,
        raw_bbox,
        width,
        height,
        source_w,
        source_h,
        body_padding=body_padding,
        containment_margin=containment_margin,
    )
    scale_x = target_w / width
    scale_y = target_h / height
    transform = np.asarray(
        [
            [scale_x, 0.0, float(center[0]) - scale_x * width * 0.5],
            [0.0, scale_y, float(center[1]) - scale_y * height * 0.5],
        ],
        dtype=np.float32,
    )
    panel = cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    marker_bbox = raw_bbox if _valid_bbox_xyxy(raw_bbox) else smoothed_bbox
    if player_id is None or not _valid_bbox_xyxy(marker_bbox):
        return panel

    origin = center - np.asarray([target_w, target_h], dtype=np.float32) * 0.5
    mapped = np.asarray(
        [
            (marker_bbox[0] - origin[0]) / scale_x,
            (marker_bbox[1] - origin[1]) / scale_y,
            (marker_bbox[2] - origin[0]) / scale_x,
            (marker_bbox[3] - origin[1]) / scale_y,
        ],
        dtype=np.float32,
    )
    x1 = int(np.clip(round(float(mapped[0])), 2, width - 4))
    y1 = int(np.clip(round(float(mapped[1])), 2, height - 4))
    x2 = int(np.clip(round(float(mapped[2])), x1 + 1, width - 2))
    y2 = int(np.clip(round(float(mapped[3])), y1 + 1, height - 2))
    color = PLAYER_COLORS.get(player_id, (190, 190, 190))
    cv2.rectangle(panel, (x1, y1), (x2, y2), WHITE, 4, cv2.LINE_AA)
    cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"P{player_id}"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
    tab_w = label_w + 16
    tab_h = label_h + 12
    tab_x = int(np.clip(x1, 2, max(2, width - tab_w - 2)))
    tab_y = y1 - tab_h if y1 >= tab_h + 4 else min(height - tab_h - 2, y1 + 3)
    cv2.rectangle(panel, (tab_x, tab_y), (tab_x + tab_w, tab_y + tab_h), WHITE, -1, cv2.LINE_AA)
    cv2.rectangle(panel, (tab_x, tab_y), (tab_x + tab_w, tab_y + tab_h), color, 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        label,
        (tab_x + 8, tab_y + tab_h - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        TEXT_INK,
        1,
        cv2.LINE_AA,
    )
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
    parser.add_argument(
        "--layout",
        choices=("source_virtual_baseline", "player_closeup"),
        default="source_virtual_baseline",
    )
    parser.add_argument(
        "--player-id",
        type=int,
        default=None,
        help="required for player_closeup; selected player identity to follow",
    )
    parser.add_argument(
        "--require-native-mesh",
        action="store_true",
        help=(
            "player_closeup only: require a native BODY surface at every output "
            "tick (measured or interpolated between adjacent measured surfaces); "
            "never substitute a joint avatar"
        ),
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="zero renders the remaining source interval",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--fps-multiplier",
        type=int,
        default=None,
        help="display multiplier; defaults to the integer native-source/mesh FPS ratio",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=0.0,
        help="explicit presentation FPS; zero retains the native/default rate",
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
    parser.add_argument(
        "--presentation-root-stabilization",
        choices=("none", "robust"),
        default="robust",
        help=(
            "gap-preserving display-only rigid root stabilization; never mutates "
            "virtual_world.json or fills missing player samples"
        ),
    )
    parser.add_argument(
        "--mesh-presentation-stabilization",
        choices=("none", "robust"),
        default="robust",
        help=(
            "player_closeup only: display-only body-local temporal filtering of "
            "native vertices/joints; preserves topology, gaps, and source artifacts"
        ),
    )
    args = parser.parse_args()

    if args.panel_width < 320 or args.panel_height < 240:
        raise ValueError("panel dimensions must be at least 320x240")
    if args.start_seconds < 0 or args.duration_seconds < 0:
        raise ValueError("start/duration seconds must be nonnegative")
    if args.output_fps < 0:
        raise ValueError("--output-fps must be nonnegative")
    if args.layout == "player_closeup" and args.player_id is None:
        raise ValueError("--player-id is required for player_closeup")
    if args.require_native_mesh and args.layout != "player_closeup":
        raise ValueError("--require-native-mesh is valid only for player_closeup")
    if args.require_native_mesh and args.index is None:
        raise ValueError("--require-native-mesh requires --index")
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
    stabilization_stats: dict[str, float | int | str]
    crop_stabilization_stats: dict[str, float | int | str] = {
        "mode": "not_applicable",
        "authority": "presentation_only",
        "segments": 0,
        "frames": 0,
    }
    mesh_stabilization_stats: dict[str, object] = {
        "mode": "none",
        "authority": "presentation_only",
        "players": {},
    }
    if args.layout == "player_closeup":
        assert args.player_id is not None
        if args.player_id not in world_frames:
            raise ValueError(
                f"player {args.player_id} is not present in {world_path}; "
                f"available={sorted(world_frames)}"
            )
        smoothed_boxes, crop_stabilization_stats = stabilize_bboxes_for_presentation(
            world_frames[args.player_id],
            fps,
        )
        world_frames[args.player_id] = smoothed_boxes
        if args.index is not None and args.mesh_presentation_stabilization == "robust":
            referenced_mesh_ids = sorted(
                {
                    frame.mesh_player_id
                    for frame in world_frames[args.player_id].values()
                }
            )
            per_mesh_player: dict[str, dict[str, float | int | str]] = {}
            for mesh_player_id in referenced_mesh_ids:
                player_mesh_frames = mesh_frames.get(mesh_player_id)
                if player_mesh_frames is None:
                    continue
                stabilized_mesh_frames, player_stats = (
                    stabilize_mesh_frames_for_presentation(player_mesh_frames, fps)
                )
                mesh_frames[mesh_player_id] = stabilized_mesh_frames
                per_mesh_player[str(mesh_player_id)] = player_stats
            mesh_stabilization_stats = {
                "mode": "body_local_native_vertex_joint_robust_symmetric",
                "authority": "presentation_only",
                "players": per_mesh_player,
            }
        stabilization_stats = {
            "mode": "not_applied_body_local_layout",
            "authority": "presentation_only",
            "segments": 0,
            "frames": 0,
        }
    elif args.presentation_root_stabilization == "robust":
        world_frames, stabilization_stats = stabilize_world_frames_for_presentation(
            world_frames,
            fps,
        )
    else:
        stabilization_stats = {
            "mode": "none",
            "authority": "presentation_only",
            "segments": 0,
            "frames": 0,
        }
    base_frame_start = (
        min(int(window.get("frame_start", 0)) for window in index["windows"])
        if index.get("windows")
        else 0
    )
    if index.get("windows"):
        frame_end = max(int(window.get("frame_end", 0)) for window in index["windows"])
        available_frames = frame_end - base_frame_start + 1
    else:
        source_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        available_frames = max(1, int(math.ceil(source_frame_count * fps / source_fps)))
    start_offset_frames = int(round(args.start_seconds * fps))
    if start_offset_frames >= available_frames:
        raise ValueError("--start-seconds is beyond the available video interval")
    frame_start = base_frame_start + start_offset_frames
    total_frames = available_frames - start_offset_frames
    if args.duration_seconds > 0:
        total_frames = min(total_frames, max(1, int(round(args.duration_seconds * fps))))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)
    output_width = args.panel_width * 2
    output_height = args.panel_height
    default_fps_multiplier = resolve_fps_multiplier(fps, source_fps, args.fps_multiplier)
    default_output_fps = fps * default_fps_multiplier
    if args.layout == "player_closeup" and args.require_native_mesh:
        # The immutable BODY observations remain at their native cadence.  A
        # strict native-mesh closeup defaults to a 60 FPS display timeline and
        # fills only half ticks between adjacent native surfaces.
        default_output_fps = 60.0
    output_fps = args.output_fps or default_output_fps
    output_frames = max(1, int(round(total_frames / fps * output_fps)))
    fps_multiplier = output_fps / fps
    terminal_display_hold_s = 1.0 / output_fps + 1e-9
    audio_seek_seconds = source_seek_seconds(base_frame_start, fps, args.start_seconds)

    if args.require_native_mesh:
        assert args.player_id is not None
        missing_native_frames: list[int] = []
        for output_frame_idx in range(output_frames):
            frame_position = frame_start + output_frame_idx / output_fps * fps
            focus_frame = world_sample_at(
                world_frames[args.player_id],
                frame_position,
                fps,
                max_gap_frames=max(
                    1,
                    int(math.ceil(fps / PLAYER_CLOSEUP_MAX_BODY_HZ - 1e-9)),
                ),
                terminal_hold_max_s=terminal_display_hold_s,
            )
            player_mesh_frames = (
                mesh_frames.get(focus_frame.mesh_player_id)
                if focus_frame is not None
                else None
            )
            native_sample = (
                mesh_sample_at(
                    player_mesh_frames,
                    frame_position,
                    fps,
                    terminal_hold_max_s=terminal_display_hold_s,
                )
                if player_mesh_frames is not None
                else None
            )
            if focus_frame is None or native_sample is None:
                missing_native_frames.append(int(round(frame_position)))
        if missing_native_frames:
            raise ValueError(
                "--require-native-mesh found missing/non-interpolatable native surfaces: "
                f"count={len(missing_native_frames)} frames={missing_native_frames[:12]}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{output_width}x{output_height}", "-r", f"{output_fps:g}", "-i", "-",
        "-ss", f"{audio_seek_seconds:g}",
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
        "focused_player_frames": 0,
        "focused_missing_frames": 0,
        "tracked_crop_frames": 0,
        "native_mesh_frames": 0,
        "native_mesh_exact_frames": 0,
        "native_mesh_display_interpolated_frames": 0,
        "native_mesh_terminal_display_hold_frames": 0,
        "joint_avatar_frames": 0,
    }
    try:
        for output_frame_idx in range(output_frames):
            local_time_seconds = output_frame_idx / output_fps
            frame_position = frame_start + local_time_seconds * fps
            time_seconds = frame_position / fps
            source_position = time_seconds * source_fps
            source_idx = int(math.floor(source_position + 1e-7))
            source_alpha = float(source_position - source_idx)
            source_current = read_source_frame_at(capture, source_idx, source_cache)
            source = source_current
            if source_alpha > 1e-6:
                source_following = read_source_frame_at(capture, source_idx + 1, source_cache)
                if source_following.shape == source_current.shape:
                    source = cv2.addWeighted(
                        source_current,
                        1.0 - source_alpha,
                        source_following,
                        source_alpha,
                        0,
                    )
            fraction = output_frame_idx / max(1, output_frames - 1)
            if args.layout == "player_closeup":
                assert args.player_id is not None
                focus_frame = world_sample_at(
                    world_frames[args.player_id],
                    frame_position,
                    fps,
                    max_gap_frames=max(
                        1,
                        int(math.ceil(fps / PLAYER_CLOSEUP_MAX_BODY_HZ - 1e-9)),
                    ),
                    terminal_hold_max_s=terminal_display_hold_s,
                )
                has_measured_crop = bool(
                    focus_frame is not None and _valid_bbox_xyxy(focus_frame.bbox_xyxy)
                )
                focused_mesh_sample = None
                focused_mesh_kind = "missing"
                if focus_frame is not None:
                    player_mesh_frames = mesh_frames.get(focus_frame.mesh_player_id)
                    if player_mesh_frames is not None:
                        focused_mesh_kind = display_sample_kind(
                            player_mesh_frames,
                            frame_position,
                            fps,
                            terminal_hold_max_s=terminal_display_hold_s,
                        )
                        focused_mesh_sample = mesh_sample_at(
                            player_mesh_frames,
                            frame_position,
                            fps,
                            terminal_hold_max_s=terminal_display_hold_s,
                        )
                if has_measured_crop:
                    assert focus_frame is not None
                    source_panel = fit_tracked_person_panel(
                        source,
                        focus_frame.bbox_xyxy,
                        focus_frame.raw_bbox_xyxy,
                        args.panel_width,
                        args.panel_height,
                        player_id=args.player_id,
                    )
                    alignment_stats["tracked_crop_frames"] += 1
                else:
                    source_panel = fit_source_panel(
                        source,
                        args.panel_width,
                        args.panel_height,
                    )
                orbit = studio_orbit_angle(fraction)
                virtual_panel = draw_person_studio_panel(
                    args.panel_width,
                    args.panel_height,
                    focus_frame,
                    args.player_id,
                    orbit,
                    mesh_sample=focused_mesh_sample,
                    faces=faces,
                    require_native_mesh=args.require_native_mesh,
                )
                if focus_frame is None:
                    alignment_stats["focused_missing_frames"] += 1
                else:
                    alignment_stats["focused_player_frames"] += 1
                if focused_mesh_sample is not None:
                    alignment_stats["native_mesh_frames"] += 1
                    if focused_mesh_kind == "measured_tick":
                        alignment_stats["native_mesh_exact_frames"] += 1
                    elif focused_mesh_kind == "display_interpolated":
                        alignment_stats["native_mesh_display_interpolated_frames"] += 1
                    elif focused_mesh_kind == "terminal_display_hold":
                        alignment_stats["native_mesh_terminal_display_hold_frames"] += 1
                elif focus_frame is not None and not args.require_native_mesh:
                    alignment_stats["joint_avatar_frames"] += 1
                draw_panel_chrome(
                    source_panel,
                    "TRACKED CLOSE-UP" if has_measured_crop else "SOURCE CONTEXT",
                    (
                        f"P{args.player_id} | tracked source crop"
                        if has_measured_crop
                        else f"P{args.player_id} | no measured crop"
                    ),
                )
                draw_panel_chrome(
                    virtual_panel,
                    (
                        "DISPLAY-SMOOTHED NATIVE MESH"
                        if focused_mesh_sample is not None
                        else (
                            "MESH ABSENT - JOINT FALLBACK"
                            if args.index is not None
                            else "BODY-LOCAL 3D"
                        )
                    ),
                    (
                        f"18,439 vertices | {output_fps:g} FPS | view {orbit:03.0f} deg | display only"
                        if focused_mesh_sample is not None
                        else f"joint avatar | studio view {orbit:03.0f} deg | presentation only"
                    ),
                )
            else:
                source_panel = fit_source_panel(source, args.panel_width, args.panel_height)
                if args.camera_motion == "subtle_orbit":
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
                draw_panel_chrome(
                    virtual_panel,
                    "VIRTUAL BASELINE",
                    (
                        "display-stabilized final placement"
                        if args.presentation_root_stabilization == "robust"
                        else "final world placement"
                    ),
                )
            canvas = np.hstack((source_panel, virtual_panel))
            cv2.line(
                canvas,
                (args.panel_width, 0),
                (args.panel_width, output_height),
                WHITE,
                5,
                cv2.LINE_AA,
            )
            time_label = f"{local_time_seconds:05.2f}s"
            (time_w, _), _ = cv2.getTextSize(
                time_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                1,
            )
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
            if output_frame_idx % max(1, int(round(output_fps))) == 0:
                print(
                    json.dumps(
                        {
                            "source_frame": int(round(frame_position)),
                            "source_total": total_frames,
                            "output_frame": output_frame_idx + 1,
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
                "layout": args.layout,
                "player_id": args.player_id,
                "start_seconds": args.start_seconds,
                "camera_motion": (
                    "staged_0_45_90_180_degree_views"
                    if args.layout == "player_closeup"
                    else args.camera_motion
                ),
                "renderer_authority": "presentation_only",
                "presentation_stabilization": stabilization_stats,
                "crop_stabilization": crop_stabilization_stats,
                "mesh_presentation_stabilization": mesh_stabilization_stats,
                "native_mesh_observation_fps": fps if args.index is not None else None,
                "display_interpolation": (
                    "adjacent_native_surfaces_only_plus_terminal_subframe_hold"
                    if args.layout == "player_closeup" and args.require_native_mesh
                    else "default"
                ),
                "body_local_centering": (
                    "hip_center_xy_preserve_world_height"
                    if args.layout == "player_closeup"
                    else "not_applicable"
                ),
                "mesh_alignment": (
                    "body_local_hip_centered_no_court_placement"
                    if args.layout == "player_closeup"
                    else "virtual_world_skeleton_root_plus_floor_guard"
                ),
                "virtual_representation": (
                    (
                        "native_body_mesh_18439_vertices_36874_faces_plus_core_skeleton"
                        if args.index is not None
                        else "translucent_detailed_joint_avatar_plus_exact_skeleton"
                    )
                    if args.layout == "player_closeup"
                    else (
                        "body_mesh_plus_exact_skeleton"
                        if args.index is not None
                        else "translucent_joint_avatar_plus_exact_skeleton"
                    )
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
