"""CPU-only helpers for ReplayScene-compatible replay export manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.schemas import ReplayScene


SCHEMA_VERSION = 1
WORLD_FRAME = "court_Z0"


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


def validate_replay_export_manifest(export_root: str | Path, manifest: Mapping[str, Any] | ReplayScene) -> ReplayScene:
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
    return scene


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


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1_000_000, 6)
