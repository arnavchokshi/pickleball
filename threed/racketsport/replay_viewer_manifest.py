"""Build browser-friendly replay viewer manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from threed.racketsport.schemas import PersonGroundTruth, ReplayViewerManifest, validate_artifact_file


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_replay_viewer_manifest"


def build_replay_viewer_manifest(
    *,
    clip: str,
    video_path: str | Path,
    virtual_world_path: str | Path,
    player_labels_path: str | Path | None = None,
    replay_scene_path: str | Path | None = None,
    physics_refinement_path: str | Path | None = None,
    contact_windows_path: str | Path | None = None,
    annotation_sources: Iterable[str | Path] = (),
    vite_allow_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a manifest the local Vite replay viewer can load via /@fs URLs."""

    if not clip:
        raise ValueError("clip is required")
    allow_root = _vite_allow_root(vite_allow_root)
    video = _existing_file(video_path, "video", allow_root=allow_root)
    virtual_world = _existing_file(virtual_world_path, "virtual_world", allow_root=allow_root)
    replay_scene = _optional_existing_file(replay_scene_path, "replay_scene", allow_root=allow_root)
    physics_refinement = _optional_existing_file(
        physics_refinement_path,
        "physics_refinement",
        allow_root=allow_root,
    )
    contact_windows = _optional_existing_file(contact_windows_path, "contact_windows", allow_root=allow_root)
    player_labels = _optional_existing_file(player_labels_path, "player_labels", allow_root=allow_root)

    label_overlays = []
    if player_labels is not None:
        label_overlays.append(_player_label_overlay(player_labels))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "video_url": _vite_file_url(video),
        "virtual_world_url": _vite_file_url(virtual_world),
        "replay_scene_url": _vite_file_url(replay_scene) if replay_scene is not None else None,
        "physics_refinement_url": _vite_file_url(physics_refinement) if physics_refinement is not None else None,
        "contact_windows_url": _vite_file_url(contact_windows) if contact_windows is not None else None,
        "label_overlays": label_overlays,
        "annotation_sources": [_annotation_source(path, allow_root=allow_root) for path in annotation_sources],
        "notes": [
            "Viewer evidence is review-only and does not promote BODY, PHYSICS, RKT, or E2E gates.",
            "Vite /@fs URLs are intended for local review server use.",
            f"Vite allow root: {allow_root.as_posix()}",
        ],
    }
    return ReplayViewerManifest.model_validate(payload).model_dump(mode="json")


def write_replay_viewer_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = ReplayViewerManifest.model_validate(manifest).model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _player_label_overlay(path: Path) -> dict[str, Any]:
    payload = _read_mapping_json(path)
    if "not_ground_truth" not in payload or not isinstance(payload["not_ground_truth"], bool):
        raise ValueError(f"player_labels must explicitly declare boolean not_ground_truth: {path}")
    not_ground_truth = payload["not_ground_truth"]
    reviewed = payload.get("status") == "human_reviewed"
    trusted_for_metrics = reviewed and not not_ground_truth
    return {
        "kind": "player_boxes",
        "label": "reviewed player boxes" if trusted_for_metrics else "prototype player boxes",
        "url": _vite_file_url(path),
        "trusted_for_metrics": trusted_for_metrics,
        "not_ground_truth": not_ground_truth,
    }


def _annotation_source(path: str | Path, *, allow_root: Path) -> dict[str, Any]:
    source = _existing_file(path, "annotation_source", allow_root=allow_root)
    payload = _read_mapping_json(source)
    artifact_type = str(payload.get("artifact_type", ""))
    if artifact_type == "racketsport_person_ground_truth":
        parsed = validate_artifact_file("person_ground_truth", source)
        if not isinstance(parsed, PersonGroundTruth):
            raise ValueError(f"annotation source did not parse as PersonGroundTruth: {source}")
        return {
            "kind": "person_ground_truth",
            "clip_id": parsed.clip_id,
            "url": _vite_file_url(source),
            "trusted_for_metrics": True,
        }
    return {
        "kind": "annotation",
        "clip_id": str(payload.get("clip_id", source.stem)),
        "url": _vite_file_url(source),
        "trusted_for_metrics": False,
    }


def _read_mapping_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _existing_file(path: str | Path, field: str, *, allow_root: Path | None = None) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"{field} file not found: {file_path}")
    if allow_root is not None:
        _require_within_vite_allow_root(file_path, field=field, allow_root=allow_root)
    return file_path


def _optional_existing_file(path: str | Path | None, field: str, *, allow_root: Path | None = None) -> Path | None:
    if path is None:
        return None
    return _existing_file(path, field, allow_root=allow_root)


def _vite_allow_root(path: str | Path | None) -> Path:
    if path is None:
        return Path(__file__).resolve().parents[2]
    return Path(path).expanduser().resolve()


def _require_within_vite_allow_root(path: Path, *, field: str, allow_root: Path) -> None:
    try:
        path.relative_to(allow_root)
    except ValueError as exc:
        raise ValueError(f"{field} file is outside Vite allow root {allow_root}: {path}") from exc


def _vite_file_url(path: Path) -> str:
    return f"/@fs/{path.as_posix()}"


__all__ = ["build_replay_viewer_manifest", "write_replay_viewer_manifest"]
