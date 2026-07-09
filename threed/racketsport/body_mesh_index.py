"""Build windowed body-mesh index artifacts for the replay viewer."""

from __future__ import annotations

import gzip
import json
import math
import mmap
import platform
import resource
import shutil
import sys
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .trust_band import TRUST_BADGES


INDEX_ARTIFACT_TYPE = "racketsport_body_mesh_index"
FACES_ARTIFACT_TYPE = "racketsport_body_mesh_faces"
BUILD_ARTIFACT_TYPE = "racketsport_body_mesh_index_build"
DEFAULT_QUANTIZATION_SCALE = 1000
DEFAULT_GZIP_COMPRESSLEVEL = 9


@dataclass
class _WindowAccumulator:
    source_window_index: int
    raw_bytes: bytearray = field(default_factory=bytearray)
    players: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    player_ids: set[int] = field(default_factory=set)
    frame_indices: set[int] = field(default_factory=set)
    reason_counts: dict[str, int] = field(default_factory=dict)
    player_frame_count: int = 0

    def add_frame(self, *, player_id: int, frame: Mapping[str, Any], quantization_scale: int) -> None:
        frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * 30.0)))
        vertex_bytes, vertex_count = _quantized_vec3_bytes_from_raw(
            frame.get("mesh_vertices_world", []),
            name=f"window[{self.source_window_index}].vertices",
            scale=quantization_scale,
        )
        joint_bytes, joint_count = _quantized_vec3_bytes_from_raw(
            frame.get("joints_world", []),
            name=f"window[{self.source_window_index}].joints",
            scale=quantization_scale,
        )
        if vertex_count == 0:
            return
        self.raw_bytes.extend(vertex_bytes)
        self.raw_bytes.extend(joint_bytes)
        reasons = [str(reason) for reason in frame.get("reasons", []) or []]
        for reason in reasons:
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        frame_payload = {
            "frame_idx": frame_idx,
            "t": float(frame.get("t", frame_idx / 30.0)),
            "source_window_index": self.source_window_index,
            "blend_weight": float(frame.get("blend_weight", 1.0)),
            "vertex_count": vertex_count,
            "joint_count": joint_count,
            "joint_conf": _float_list(frame.get("joint_conf", [])),
            "reasons": reasons,
        }
        trust_badge = _optional_trust_badge(frame.get("trust_badge"))
        if trust_badge is not None:
            frame_payload["trust_badge"] = trust_badge
        self.players.setdefault(int(player_id), []).append(frame_payload)
        self.player_ids.add(int(player_id))
        self.frame_indices.add(frame_idx)
        self.player_frame_count += 1


def build_body_mesh_index(
    clip_dir_or_body_mesh: str | Path,
    *,
    out_dir: str | Path,
    quantization_scale: int = DEFAULT_QUANTIZATION_SCALE,
    compresslevel: int = DEFAULT_GZIP_COMPRESSLEVEL,
) -> dict[str, Any]:
    """Split ``body_mesh.json`` into gzip-compressed per-window mesh chunks.

    The implementation uses an mmap-backed incremental reader: top-level
    metadata and one player object at a time are decoded, while the monolithic
    ``players`` array is never loaded as a single Python object.
    """

    started = time.monotonic()
    body_mesh_path = _resolve_body_mesh_path(clip_dir_or_body_mesh)
    quantization_scale, compresslevel = _validate_build_options(
        quantization_scale=quantization_scale,
        compresslevel=compresslevel,
    )

    with body_mesh_path.open("rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            metadata = _read_top_level_metadata(mm)
            return _build_body_mesh_index_core(
                metadata=metadata,
                players=_iter_player_payloads(mm, players_range=metadata["players_range"]),
                out_dir=Path(out_dir),
                quantization_scale=quantization_scale,
                compresslevel=compresslevel,
                started=started,
                body_mesh_path=body_mesh_path,
                input_bytes=body_mesh_path.stat().st_size,
                used_streaming_parser=True,
            )
        finally:
            mm.close()


def build_body_mesh_index_from_payload(
    payload: Mapping[str, Any],
    *,
    out_dir: str | Path,
    quantization_scale: int = DEFAULT_QUANTIZATION_SCALE,
    compresslevel: int = DEFAULT_GZIP_COMPRESSLEVEL,
) -> dict[str, Any]:
    """Build the viewer mesh index directly from an in-memory ``body_mesh`` payload.

    This path shares the same accumulator, quantization, window finalization, and
    gzip chunk writer as the file-backed builder. It iterates the existing
    payload's player/frame objects and does not serialize or deep-copy the mesh
    monolith before chunking it.
    """

    started = time.monotonic()
    quantization_scale, compresslevel = _validate_build_options(
        quantization_scale=quantization_scale,
        compresslevel=compresslevel,
    )
    metadata, players = _metadata_and_players_from_payload(payload)
    return _build_body_mesh_index_core(
        metadata=metadata,
        players=players,
        out_dir=Path(out_dir),
        quantization_scale=quantization_scale,
        compresslevel=compresslevel,
        started=started,
        body_mesh_path="<in_memory_body_mesh_payload>",
        input_bytes=0,
        used_streaming_parser=False,
    )


def build_body_mesh_index_from_arrays(
    *,
    metadata: Mapping[str, Any],
    players: Iterable[Mapping[str, Any]],
    out_dir: str | Path,
    quantization_scale: int = DEFAULT_QUANTIZATION_SCALE,
    compresslevel: int = DEFAULT_GZIP_COMPRESSLEVEL,
) -> dict[str, Any]:
    """Build the viewer mesh index from array-backed body mesh records."""

    started = time.monotonic()
    quantization_scale, compresslevel = _validate_build_options(
        quantization_scale=quantization_scale,
        compresslevel=compresslevel,
    )
    return _build_body_mesh_index_core(
        metadata=_metadata_from_arrays(metadata),
        players=players,
        out_dir=Path(out_dir),
        quantization_scale=quantization_scale,
        compresslevel=compresslevel,
        started=started,
        body_mesh_path="<array_body_mesh_source>",
        input_bytes=0,
        used_streaming_parser=False,
    )


def _validate_build_options(*, quantization_scale: int, compresslevel: int) -> tuple[int, int]:
    if quantization_scale <= 0:
        raise ValueError("quantization_scale must be positive")
    if int(compresslevel) < 1 or int(compresslevel) > 9:
        raise ValueError("compresslevel must be between 1 and 9")
    return int(quantization_scale), int(compresslevel)


def _build_body_mesh_index_core(
    *,
    metadata: Mapping[str, Any],
    players: Iterable[Mapping[str, Any]],
    out_dir: Path,
    quantization_scale: int,
    compresslevel: int,
    started: float,
    body_mesh_path: str | Path,
    input_bytes: int,
    used_streaming_parser: bool,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunk_dir = out / "body_mesh_chunks"
    _reset_chunk_dir(chunk_dir)
    accumulators: dict[int, _WindowAccumulator] = {}
    for player in players:
        if isinstance(player, Mapping):
            _process_player_payload(
                player,
                accumulators=accumulators,
                chunk_dir=chunk_dir,
                quantization_scale=quantization_scale,
            )

    faces_payload = {
        "schema_version": 1,
        "artifact_type": FACES_ARTIFACT_TYPE,
        "faces_ref": metadata["faces_ref"],
        "mesh_faces": metadata["mesh_faces"],
    }
    faces_path = out / "body_mesh_faces.json"
    _write_json(faces_path, faces_payload)

    windows = _finalize_windows(
        accumulators=accumulators,
        windows_metadata=metadata["windows"],
        chunk_dir=out / "body_mesh_chunks",
        quantization_scale=int(quantization_scale),
        fps=float(metadata["fps"]),
        compresslevel=int(compresslevel),
    )
    player_ids = sorted({player_id for window in windows for player_id in window["player_ids"]})
    index_payload = {
        "schema_version": 1,
        "artifact_type": INDEX_ARTIFACT_TYPE,
        "clip": metadata["clip"],
        "model": metadata["model"],
        "fps": float(metadata["fps"]),
        "world_frame": metadata["world_frame"],
        "faces_ref": metadata["faces_ref"],
        "faces_url": "body_mesh_faces.json",
        "windows": windows,
        "summary": {
            "window_count": len(windows),
            "mesh_frame_count": sum(int(window["player_frame_count"]) for window in windows),
            "player_count": len(player_ids),
            "faces_count": len(metadata["mesh_faces"]),
        },
    }
    index_path = out / "body_mesh_index.json"
    _write_json(index_path, index_payload)

    return {
        "artifact_type": BUILD_ARTIFACT_TYPE,
        "body_mesh_path": body_mesh_path,
        "index_path": index_path,
        "faces_path": faces_path,
        "out_dir": out,
        "used_streaming_parser": bool(used_streaming_parser),
        "gzip_compresslevel": compresslevel,
        "input_bytes": int(input_bytes),
        "wall_seconds": time.monotonic() - started,
        "peak_rss_mb": _peak_rss_mb(),
        "summary": dict(index_payload["summary"]),
    }


def build_body_mesh_index_cli_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": BUILD_ARTIFACT_TYPE,
        "body_mesh_path": str(result["body_mesh_path"]),
        "index_path": str(result["index_path"]),
        "faces_path": str(result["faces_path"]),
        "out_dir": str(result["out_dir"]),
        "used_streaming_parser": bool(result["used_streaming_parser"]),
        "gzip_compresslevel": int(result.get("gzip_compresslevel", DEFAULT_GZIP_COMPRESSLEVEL)),
        "input_bytes": int(result["input_bytes"]),
        "wall_seconds": float(result["wall_seconds"]),
        "peak_rss_mb": float(result["peak_rss_mb"]),
        "summary": dict(result["summary"]),
    }


def _resolve_body_mesh_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "body_mesh.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"body_mesh.json not found: {candidate}")
    return candidate


def _read_top_level_metadata(mm: mmap.mmap) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "mesh_faces": [],
        "windows": [],
    }
    for key in ("clip", "model", "fps", "world_frame", "faces_ref", "mesh_faces", "windows"):
        value_start = _find_top_level_value_start(mm, key)
        if value_start is not None:
            metadata[key] = _loads_slice(mm, value_start, _json_value_end(mm, value_start))
    players_start = _find_top_level_value_start(mm, "players")
    if players_start is None:
        raise ValueError("body_mesh.players is required")
    for key in ("clip", "model", "fps", "world_frame", "faces_ref"):
        if key not in metadata:
            raise ValueError(f"body_mesh.{key} is required")
    metadata["players_range"] = (players_start, len(mm))
    return metadata


def _metadata_and_players_from_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], Iterable[Mapping[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise ValueError("body_mesh payload must be an object")
    metadata: dict[str, Any] = {
        "mesh_faces": payload.get("mesh_faces", []),
        "windows": payload.get("windows", []),
    }
    for key in ("clip", "model", "fps", "world_frame", "faces_ref"):
        if key not in payload:
            raise ValueError(f"body_mesh.{key} is required")
        metadata[key] = payload[key]
    players = payload.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes, bytearray)):
        raise ValueError("body_mesh.players must be an array")
    return metadata, (player for player in players if isinstance(player, Mapping))


def _metadata_from_arrays(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("body mesh array metadata must be an object")
    parsed: dict[str, Any] = {
        "mesh_faces": metadata.get("mesh_faces", []),
        "windows": metadata.get("windows", []),
    }
    for key in ("clip", "model", "fps", "world_frame", "faces_ref"):
        if key not in metadata:
            raise ValueError(f"body mesh array metadata.{key} is required")
        parsed[key] = metadata[key]
    return parsed


def _find_top_level_value_start(mm: mmap.mmap, key: str) -> int | None:
    needle = json.dumps(key).encode("utf-8")
    pos = mm.find(needle)
    while pos != -1:
        before = _skip_ws_reverse(mm, pos - 1)
        if before >= 0 and mm[before] not in (ord("{"), ord(",")):
            pos = mm.find(needle, pos + 1)
            continue
        after = _skip_ws(mm, pos + len(needle))
        if after < len(mm) and mm[after] == ord(":"):
            return _skip_ws(mm, after + 1)
        pos = mm.find(needle, pos + 1)
    return None


def _iter_player_payloads(mm: mmap.mmap, *, players_range: tuple[int, int]) -> Iterator[Mapping[str, Any]]:
    start, end = players_range
    pos = _skip_ws(mm, start)
    if pos >= end or mm[pos] != ord("["):
        raise ValueError("body_mesh.players must be an array")
    pos += 1
    while True:
        pos = _skip_ws(mm, pos)
        if pos >= end:
            raise ValueError("unterminated body_mesh.players array")
        if mm[pos] == ord("]"):
            return
        player_start = pos
        player_end = _json_value_end(mm, player_start)
        player = _loads_slice(mm, player_start, player_end)
        if isinstance(player, Mapping):
            yield player
        pos = _skip_ws(mm, player_end)
        if pos < end and mm[pos] == ord(","):
            pos += 1


def _process_players(
    mm: mmap.mmap,
    *,
    players_range: tuple[int, int],
    accumulators: dict[int, _WindowAccumulator],
    chunk_dir: Path,
    quantization_scale: int,
) -> None:
    start, end = players_range
    pos = _skip_ws(mm, start)
    if pos >= end or mm[pos] != ord("["):
        raise ValueError("body_mesh.players must be an array")
    pos += 1
    while True:
        pos = _skip_ws(mm, pos)
        if pos >= end:
            raise ValueError("unterminated body_mesh.players array")
        if mm[pos] == ord("]"):
            return
        player_start = pos
        player_end = _json_value_end(mm, player_start)
        player = _loads_slice(mm, player_start, player_end)
        if isinstance(player, Mapping):
            _process_player_payload(
                player,
                accumulators=accumulators,
                chunk_dir=chunk_dir,
                quantization_scale=quantization_scale,
            )
        pos = _skip_ws(mm, player_end)
        if pos < end and mm[pos] == ord(","):
            pos += 1


def _process_player_payload(
    player: Mapping[str, Any],
    *,
    accumulators: dict[int, _WindowAccumulator],
    chunk_dir: Path,
    quantization_scale: int,
) -> None:
    if "id" not in player:
        return
    player_id = int(player["id"])
    fallback_window_index = 0
    for frame in player.get("frames", []) or []:
        if not isinstance(frame, Mapping):
            continue
        raw_window_index = frame.get("source_window_index")
        window_index = int(raw_window_index) if raw_window_index is not None else fallback_window_index
        accumulator = accumulators.setdefault(
            window_index,
            _WindowAccumulator(
                source_window_index=window_index,
            ),
        )
        accumulator.add_frame(player_id=player_id, frame=frame, quantization_scale=quantization_scale)


def _process_player(
    mm: mmap.mmap,
    start: int,
    end: int,
    *,
    accumulators: dict[int, _WindowAccumulator],
    chunk_dir: Path,
    quantization_scale: int,
) -> None:
    player_id: int | None = None
    frames_range: tuple[int, int] | None = None
    for key, value_start, value_end in _iter_object_items(mm, start, end):
        if key == "id":
            player_id = int(_loads_slice(mm, value_start, value_end))
        elif key == "frames":
            frames_range = (value_start, value_end)
    if player_id is None or frames_range is None:
        return
    _process_frames(
        mm,
        frames_range=frames_range,
        player_id=player_id,
        accumulators=accumulators,
        chunk_dir=chunk_dir,
        quantization_scale=quantization_scale,
    )


def _process_frames(
    mm: mmap.mmap,
    *,
    frames_range: tuple[int, int],
    player_id: int,
    accumulators: dict[int, _WindowAccumulator],
    chunk_dir: Path,
    quantization_scale: int,
) -> None:
    start, end = frames_range
    pos = _skip_ws(mm, start)
    if pos >= end or mm[pos] != ord("["):
        raise ValueError("body_mesh.players[].frames must be an array")
    pos += 1
    fallback_window_index = 0
    while True:
        pos = _skip_ws(mm, pos)
        if pos >= end:
            raise ValueError("unterminated body_mesh.players[].frames array")
        if mm[pos] == ord("]"):
            return
        frame_end = _json_value_end(mm, pos)
        frame = _loads_slice(mm, pos, frame_end)
        if isinstance(frame, Mapping):
            raw_window_index = frame.get("source_window_index")
            window_index = int(raw_window_index) if raw_window_index is not None else fallback_window_index
            accumulator = accumulators.setdefault(
                window_index,
                _WindowAccumulator(
                    source_window_index=window_index,
                ),
            )
            accumulator.add_frame(player_id=player_id, frame=frame, quantization_scale=quantization_scale)
        pos = _skip_ws(mm, frame_end)
        if pos < end and mm[pos] == ord(","):
            pos += 1


def _finalize_windows(
    *,
    accumulators: Mapping[int, _WindowAccumulator],
    windows_metadata: Sequence[Mapping[str, Any]],
    chunk_dir: Path,
    quantization_scale: int,
    fps: float,
    compresslevel: int,
) -> list[dict[str, Any]]:
    metadata_by_index = {
        int(window.get("source_window_index", index)): dict(window)
        for index, window in enumerate(windows_metadata)
        if isinstance(window, Mapping)
    }
    windows: list[dict[str, Any]] = []
    for source_window_index, accumulator in sorted(accumulators.items()):
        if accumulator.player_frame_count == 0:
            continue
        compressed_path = chunk_dir / f"window_{source_window_index:03d}.bin.gz"
        _gzip_bytes(accumulator.raw_bytes, compressed_path, compresslevel=compresslevel)
        meta = metadata_by_index.get(source_window_index, {})
        frame_start = int(meta.get("frame_start", min(accumulator.frame_indices)))
        frame_end = int(meta.get("frame_end", max(accumulator.frame_indices)))
        frame_count = int(meta.get("frame_count", len(accumulator.frame_indices)))
        window = {
            "source_window_index": int(source_window_index),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "t0": float(meta.get("t0", frame_start / fps)),
            "t1": float(meta.get("t1", (frame_end + 1) / fps)),
            "frame_count": frame_count,
            "player_frame_count": int(accumulator.player_frame_count),
            "target_player_ids": [int(player_id) for player_id in meta.get("target_player_ids", sorted(accumulator.player_ids))],
            "player_ids": sorted(accumulator.player_ids),
            "target_representation": str(meta.get("target_representation", "world_mesh")),
            "fallback_representation": str(meta.get("fallback_representation", "lane_a_skeleton")),
            "reason_counts": _int_mapping(meta.get("reason_counts", accumulator.reason_counts)),
            "max_score": float(meta.get("max_score", 0.0) or 0.0),
            "url": f"body_mesh_chunks/window_{source_window_index:03d}.bin.gz",
            "byte_size": compressed_path.stat().st_size,
            "encoding": "gzip_int16_world_vertices_v1",
            "quantization": {"scale": int(quantization_scale), "unit": "m"},
            "players": [
                {"id": int(player_id), "frames": frames}
                for player_id, frames in sorted(accumulator.players.items(), key=lambda item: item[0])
            ],
        }
        windows.append(window)
    return windows


def _iter_object_items(mm: mmap.mmap, start: int, end: int) -> Iterator[tuple[str, int, int]]:
    pos = _skip_ws(mm, start)
    if pos >= end or mm[pos] != ord("{"):
        raise ValueError("expected JSON object")
    pos += 1
    while True:
        pos = _skip_ws(mm, pos)
        if pos >= end:
            raise ValueError("unterminated JSON object")
        if mm[pos] == ord("}"):
            return
        key, pos = _read_string(mm, pos)
        pos = _skip_ws(mm, pos)
        if pos >= end or mm[pos] != ord(":"):
            raise ValueError("expected ':' after object key")
        value_start = _skip_ws(mm, pos + 1)
        value_end = _json_value_end(mm, value_start)
        yield key, value_start, value_end
        pos = _skip_ws(mm, value_end)
        if pos < end and mm[pos] == ord(","):
            pos += 1


def _read_string(mm: mmap.mmap, pos: int) -> tuple[str, int]:
    if mm[pos] != ord('"'):
        raise ValueError("expected JSON string")
    end = pos + 1
    escaped = False
    while end < len(mm):
        char = mm[end]
        if escaped:
            escaped = False
        elif char == ord("\\"):
            escaped = True
        elif char == ord('"'):
            return json.loads(mm[pos : end + 1].decode("utf-8")), end + 1
        end += 1
    raise ValueError("unterminated JSON string")


def _json_value_end(mm: mmap.mmap, pos: int) -> int:
    pos = _skip_ws(mm, pos)
    first = mm[pos]
    if first in (ord("{"), ord("[")):
        depth = 0
        in_string = False
        escaped = False
        index = pos
        while index < len(mm):
            char = mm[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == ord("\\"):
                    escaped = True
                elif char == ord('"'):
                    in_string = False
            else:
                if char == ord('"'):
                    in_string = True
                elif char in (ord("{"), ord("[")):
                    depth += 1
                elif char in (ord("}"), ord("]")):
                    depth -= 1
                    if depth == 0:
                        return index + 1
            index += 1
        raise ValueError("unterminated JSON value")
    if first == ord('"'):
        _, end = _read_string(mm, pos)
        return end
    index = pos
    while index < len(mm) and mm[index] not in (ord(","), ord("}"), ord("]"), 9, 10, 13, 32):
        index += 1
    return index


def _skip_ws(mm: mmap.mmap, pos: int) -> int:
    while pos < len(mm) and mm[pos] in (9, 10, 13, 32):
        pos += 1
    return pos


def _skip_ws_reverse(mm: mmap.mmap, pos: int) -> int:
    while pos >= 0 and mm[pos] in (9, 10, 13, 32):
        pos -= 1
    return pos


def _loads_slice(mm: mmap.mmap, start: int, end: int) -> Any:
    return json.loads(mm[start:end].decode("utf-8"))


def _quantized_vec3_bytes_from_raw(values: Any, *, name: str, scale: int) -> tuple[bytes, int]:
    if values is None:
        return b"", 0
    values = _to_python_container(values)
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise ValueError(f"{name} must be a sequence of 3-vectors")
    out = array("h")
    count = 0
    scale_float = float(scale)
    for index, point in enumerate(values):
        if not isinstance(point, Sequence) or isinstance(point, str | bytes) or len(point) != 3:
            raise ValueError(f"{name}[{index}] must be a 3-vector")
        for component in point:
            value = float(component)
            if not math.isfinite(value):
                raise ValueError(f"{name}[{index}] must contain finite values")
            quantized = int(round(value * scale_float))
            if quantized < -32768 or quantized > 32767:
                raise ValueError(f"quantized mesh coordinate {quantized} exceeds int16 range at scale={scale}")
            out.append(quantized)
        count += 1
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes(), count


def _float_list(values: Any) -> list[float]:
    values = _to_python_container(values)
    if values is None:
        return []
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        return []
    return [round(float(value), 6) for value in values]


def _optional_trust_badge(value: Any) -> str | None:
    if value is None:
        return None
    badge = str(value)
    if badge not in TRUST_BADGES:
        raise ValueError(f"body_mesh_index frame trust_badge must be one of {TRUST_BADGES}, got {badge!r}")
    return badge


def _to_python_container(value: Any) -> Any:
    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            try:
                item = method()
            except Exception:
                return value
    tolist = getattr(item, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            return value
    return item


def _gzip_bytes(raw_bytes: bytes | bytearray, compressed_path: Path, *, compresslevel: int) -> None:
    compressed_path.parent.mkdir(parents=True, exist_ok=True)
    with compressed_path.open("wb") as dst:
        with gzip.GzipFile(filename="", mode="wb", fileobj=dst, mtime=0, compresslevel=int(compresslevel)) as gz:
            gz.write(raw_bytes)


def _reset_chunk_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): int(count) for key, count in value.items()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


__all__ = [
    "BUILD_ARTIFACT_TYPE",
    "DEFAULT_GZIP_COMPRESSLEVEL",
    "DEFAULT_QUANTIZATION_SCALE",
    "FACES_ARTIFACT_TYPE",
    "INDEX_ARTIFACT_TYPE",
    "build_body_mesh_index",
    "build_body_mesh_index_cli_summary",
    "build_body_mesh_index_from_arrays",
    "build_body_mesh_index_from_payload",
]
