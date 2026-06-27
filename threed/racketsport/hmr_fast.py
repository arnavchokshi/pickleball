"""CPU-only fast-tier HMR preview scaffold primitives.

This module intentionally contains deterministic validation and packaging
helpers only. It does not run SAT-HMR, Multi-HMR 2, Core ML conversion, GPU
inference, checkpoint selection, or SMPL fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SCAFFOLD_NOTE = "cpu_hmr_fast_primitives_no_sat_hmr_or_multihmr_integration"
CAMERA_SPACE = "camera_space"
TASK_ID = "BODY-3"


@dataclass(frozen=True)
class PreviewMesh:
    """Validated camera-space mesh for a single player preview frame."""

    vertices_camera: Sequence[Sequence[float]]
    faces: Sequence[Sequence[int]]
    bbox_xyxy: Sequence[float]

    def __post_init__(self) -> None:
        vertices = _validate_vertices(self.vertices_camera)
        faces = _validate_faces(self.faces, vertex_count=len(vertices))
        bbox = _validate_bbox_xyxy(self.bbox_xyxy)

        object.__setattr__(self, "vertices_camera", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "bbox_xyxy", bbox)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices_camera)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def to_payload(self) -> dict[str, Any]:
        return {
            "coordinate_frame": CAMERA_SPACE,
            "vertices_camera": [list(vertex) for vertex in self.vertices_camera],
            "faces": [list(face) for face in self.faces],
            "bbox_xyxy": list(self.bbox_xyxy),
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
        }


@dataclass(frozen=True)
class FastTierMetadata:
    """Latency-tier and model-family metadata for scaffold preview payloads."""

    model_family: str
    elapsed_ms: float | None = None
    target_latency_ms: float = 10_000.0
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.model_family:
            raise ValueError("model_family must be non-empty")
        target_latency_ms = _require_finite_float(self.target_latency_ms, "target_latency_ms")
        if target_latency_ms <= 0.0:
            raise ValueError("target_latency_ms must be positive")

        elapsed_ms = None
        if self.elapsed_ms is not None:
            elapsed_ms = _require_finite_float(self.elapsed_ms, "elapsed_ms")
            if elapsed_ms < 0.0:
                raise ValueError("elapsed_ms must be non-negative")

        object.__setattr__(self, "model_family", str(self.model_family))
        object.__setattr__(self, "target_latency_ms", target_latency_ms)
        object.__setattr__(self, "elapsed_ms", elapsed_ms)

    @property
    def latency_tier(self) -> str:
        return "fast"

    @property
    def coordinate_frame(self) -> str:
        return CAMERA_SPACE

    @property
    def preview_only(self) -> bool:
        return True

    @property
    def scaffold_only(self) -> bool:
        return True

    @property
    def real_inference(self) -> bool:
        return False

    def to_payload(self) -> dict[str, Any]:
        return {
            "latency_tier": self.latency_tier,
            "model_family": self.model_family,
            "coordinate_frame": self.coordinate_frame,
            "target_latency_ms": self.target_latency_ms,
            "elapsed_ms": self.elapsed_ms,
            "scaffold_only": self.scaffold_only,
            "real_inference": self.real_inference,
            "checkpoint": self.checkpoint,
        }


def package_frame_preview(
    *,
    frame_idx: int,
    t: float,
    metadata: FastTierMetadata,
    players: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic fast-tier camera-space preview payload."""

    return _base_payload(
        frame_idx=frame_idx,
        t=t,
        metadata=metadata,
        status="preview",
        fallback={"active": False, "reason": None},
        players=[_player_payload(player) for player in players],
    )


def build_fail_closed_payload(
    *,
    frame_idx: int,
    t: float,
    metadata: FastTierMetadata,
    reason: str,
) -> dict[str, Any]:
    """Build a fail-closed preview payload without mesh data."""

    if not reason:
        raise ValueError("reason must be non-empty")
    return _base_payload(
        frame_idx=frame_idx,
        t=t,
        metadata=metadata,
        status="fail_closed",
        fallback={"active": True, "reason": str(reason)},
        players=[],
    )


def _base_payload(
    *,
    frame_idx: int,
    t: float,
    metadata: FastTierMetadata,
    status: str,
    fallback: dict[str, Any],
    players: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": TASK_ID,
        "status": status,
        "preview_only": True,
        "frame_idx": _require_non_negative_int(frame_idx, "frame_idx"),
        "t": _require_finite_float(t, "t"),
        "metadata": metadata.to_payload(),
        "fallback": fallback,
        "players": players,
    }


def _player_payload(player: Mapping[str, Any]) -> dict[str, Any]:
    try:
        mesh = player["mesh"]
        player_id = player["player_id"]
        track_id = player["track_id"]
        confidence = player["confidence"]
    except KeyError as exc:
        raise ValueError(f"player is missing {exc.args[0]}") from exc

    if not isinstance(mesh, PreviewMesh):
        raise ValueError("player mesh must be a PreviewMesh")
    confidence_float = _require_finite_float(confidence, "confidence")
    if not 0.0 <= confidence_float <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    return {
        "player_id": _require_non_negative_int(player_id, "player_id"),
        "track_id": _require_non_negative_int(track_id, "track_id"),
        "confidence": confidence_float,
        "mesh": mesh.to_payload(),
    }


def _validate_vertices(vertices_camera: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(vertices_camera, (str, bytes)):
        raise ValueError("vertices_camera must be a sequence of 3-vectors")
    try:
        vertices = list(vertices_camera)
    except TypeError as exc:
        raise ValueError("vertices_camera must be a sequence of 3-vectors") from exc
    if not vertices:
        raise ValueError("vertices_camera must contain at least one vertex")
    return [
        list(_validate_float_vector(vertex, f"vertices_camera/{index}", length=3))
        for index, vertex in enumerate(vertices)
    ]


def _validate_faces(faces: Sequence[Sequence[int]], *, vertex_count: int) -> list[tuple[int, int, int]]:
    if isinstance(faces, (str, bytes)):
        raise ValueError("faces must be a sequence of triangle index triples")
    try:
        face_rows = list(faces)
    except TypeError as exc:
        raise ValueError("faces must be a sequence of triangle index triples") from exc

    validated: list[tuple[int, int, int]] = []
    for face_index, face in enumerate(face_rows):
        if isinstance(face, (str, bytes)):
            raise ValueError(f"faces/{face_index} must be a triangle index triple")
        try:
            indices = tuple(face)
        except TypeError as exc:
            raise ValueError(f"faces/{face_index} must be a triangle index triple") from exc
        if len(indices) != 3:
            raise ValueError(f"faces/{face_index} must be a triangle index triple")

        int_indices: list[int] = []
        for raw_index in indices:
            index = _require_non_negative_int(raw_index, f"faces/{face_index} index")
            if index >= vertex_count:
                raise ValueError(f"faces/{face_index} index {index} is outside vertices_camera")
            int_indices.append(index)
        validated.append((int_indices[0], int_indices[1], int_indices[2]))
    return validated


def _validate_bbox_xyxy(values: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _validate_float_vector(values, "bbox_xyxy", length=4)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must be ordered as x1, y1, x2, y2")
    return (x1, y1, x2, y2)


def _validate_float_vector(values: Sequence[float], name: str, *, length: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a {length}-vector")
    try:
        vector = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a {length}-vector") from exc
    if len(vector) != length:
        raise ValueError(f"{name} must be a {length}-vector")
    return tuple(_require_finite_float(value, f"{name}/{index}") for index, value in enumerate(vector))


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


__all__ = [
    "CAMERA_SPACE",
    "SCAFFOLD_NOTE",
    "TASK_ID",
    "FastTierMetadata",
    "PreviewMesh",
    "build_fail_closed_payload",
    "package_frame_preview",
]
