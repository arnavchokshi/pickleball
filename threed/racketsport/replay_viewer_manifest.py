"""Build browser-friendly replay viewer manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


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
) -> dict[str, Any]:
    """Return a manifest the local Vite replay viewer can load via /@fs URLs."""

    if not clip:
        raise ValueError("clip is required")
    video = _existing_file(video_path, "video")
    virtual_world = _existing_file(virtual_world_path, "virtual_world")
    replay_scene = _optional_existing_file(replay_scene_path, "replay_scene")
    physics_refinement = _optional_existing_file(physics_refinement_path, "physics_refinement")
    contact_windows = _optional_existing_file(contact_windows_path, "contact_windows")
    player_labels = _optional_existing_file(player_labels_path, "player_labels")

    label_overlays = []
    if player_labels is not None:
        payload = _read_mapping_json(player_labels)
        not_ground_truth = bool(payload.get("not_ground_truth", False))
        label_overlays.append(
            {
                "kind": "player_boxes",
                "label": "prototype player boxes",
                "url": _vite_file_url(player_labels),
                "trusted_for_metrics": not not_ground_truth,
                "not_ground_truth": not_ground_truth,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "video_url": _vite_file_url(video),
        "virtual_world_url": _vite_file_url(virtual_world),
        "replay_scene_url": _vite_file_url(replay_scene) if replay_scene is not None else None,
        "physics_refinement_url": _vite_file_url(physics_refinement) if physics_refinement is not None else None,
        "contact_windows_url": _vite_file_url(contact_windows) if contact_windows is not None else None,
        "label_overlays": label_overlays,
        "annotation_sources": [_annotation_source(path) for path in annotation_sources],
        "notes": [
            "Viewer evidence is review-only and does not promote BODY, PHYSICS, RKT, or E2E gates.",
            "Vite /@fs URLs are intended for local review server use.",
        ],
    }


def write_replay_viewer_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _annotation_source(path: str | Path) -> dict[str, Any]:
    source = _existing_file(path, "annotation_source")
    payload = _read_mapping_json(source)
    artifact_type = str(payload.get("artifact_type", ""))
    kind = "person_ground_truth" if artifact_type == "racketsport_person_ground_truth" else "annotation"
    return {
        "kind": kind,
        "clip_id": str(payload.get("clip_id", source.stem)),
        "url": _vite_file_url(source),
        "trusted_for_metrics": kind == "person_ground_truth",
    }


def _read_mapping_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _existing_file(path: str | Path, field: str) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"{field} file not found: {file_path}")
    return file_path


def _optional_existing_file(path: str | Path | None, field: str) -> Path | None:
    if path is None:
        return None
    return _existing_file(path, field)


def _vite_file_url(path: Path) -> str:
    return f"/@fs/{path.as_posix()}"


__all__ = ["build_replay_viewer_manifest", "write_replay_viewer_manifest"]
