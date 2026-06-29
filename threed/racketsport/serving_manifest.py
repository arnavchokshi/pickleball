from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from threed.racketsport.model_manifest import ModelEntry, load_model_manifest


READY_CHECKPOINT_STATUS = "available_on_h100"
READY_RUNTIME_STATUS = "available_runtime_on_h100"

SAFE_CHECKPOINT_PREFIXES = (
    PurePosixPath("/workspace/checkpoints"),
    PurePosixPath("/workspace/sam4dbody/weights"),
)
SAFE_RUNTIME_PREFIXES = (
    PurePosixPath("/opt/conda/envs"),
    PurePosixPath("/workspace/envs"),
)


ComponentKind = Literal["checkpoint", "runtime"]


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    stage: str
    kind: ComponentKind
    serving_backend: str
    role: str


TIER_SPECS: dict[str, tuple[ComponentSpec, ...]] = {
    "offline_deep": (
        ComponentSpec("person_detection", "person_detect", "checkpoint", "tensorrt_or_python_backend", "detect <=4 court players"),
        ComponentSpec("whole_body_pose", "2d_pose", "checkpoint", "tensorrt_or_python_backend", "whole-body pose candidates"),
        ComponentSpec("per_crop_pose", "2d_pose_per_crop", "checkpoint", "tensorrt_or_python_backend", "per-crop body pose candidates"),
        ComponentSpec("body_backbone", "3d_body_backbone", "checkpoint", "python_backend", "deep-tier source-of-truth body mesh"),
        ComponentSpec("camera_fov_depth_prior", "camera_fov_depth_prior", "checkpoint", "python_backend", "camera FOV/depth prior for body mesh"),
        ComponentSpec("ball_tracking", "ball_tracking", "checkpoint", "python_backend", "ball trajectory and inpainting"),
        ComponentSpec("racket_segmentation", "racket_segmentation", "checkpoint", "python_backend", "racket/paddle silhouette candidates"),
        ComponentSpec("physics_refinement", "physics", "runtime", "python_backend", "MuJoCo/MJX physics runtime"),
    ),
    "live_light": (
        ComponentSpec("person_detection", "person_detect", "checkpoint", "tensorrt_or_python_backend", "server fallback detector for light tier"),
        ComponentSpec("whole_body_pose", "2d_pose", "checkpoint", "tensorrt_or_python_backend", "server fallback pose for light tier"),
        ComponentSpec("per_crop_pose", "2d_pose_per_crop", "checkpoint", "tensorrt_or_python_backend", "server fallback per-crop pose for light tier"),
        ComponentSpec("fast_mesh_preview", "fast_mesh_preview", "checkpoint", "python_backend", "camera-space mesh preview candidates"),
        ComponentSpec("ball_tracking", "ball_tracking", "checkpoint", "python_backend", "server fallback ball tracking for light tier"),
    ),
}


def build_serving_manifest(manifest_path: str | Path = "models/MANIFEST.json") -> dict[str, Any]:
    source_path = Path(manifest_path)
    manifest = load_model_manifest(source_path)
    entries_by_stage = _entries_by_stage(manifest.models)

    tiers = {
        tier: _build_tier(tier, specs, entries_by_stage=entries_by_stage)
        for tier, specs in TIER_SPECS.items()
    }
    source_bytes = source_path.read_bytes()

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_serving_manifest",
        "source_manifest": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "schema_version": manifest.schema_version,
        },
        "execution": {
            "cpu_only": True,
            "starts_triton": False,
            "downloads_models": False,
            "uses_gpu": False,
            "mutates_model_manifest": False,
            "claims_env_or_eval_completion": False,
        },
        "tiers": tiers,
        "summary": _summarize(manifest.models, tiers),
        "notes": [
            "This report is a CPU-only serving inventory scaffold and does not start Triton.",
            "EVAL-0 approval is not inferred here; model variants remain subject to benchmark and human approval gates.",
            "Local checkpoint paths are validated syntactically for serving safety only; file existence is not probed on this host.",
        ],
    }


def _build_tier(
    tier: str,
    specs: tuple[ComponentSpec, ...],
    *,
    entries_by_stage: dict[str, list[ModelEntry]],
) -> dict[str, Any]:
    components = [_build_component(spec, entries_by_stage.get(spec.stage, [])) for spec in specs]
    return {
        "tier": tier,
        "checkpoint_runtime_inventory_ready": all(component["inventory_ready"] for component in components),
        "serving_ready": False,
        "eval0_approval": "not_evaluated_by_cpu_manifest",
        "triton_ensemble_status": "scaffold_only_not_started",
        "components": components,
    }


def _build_component(spec: ComponentSpec, entries: list[ModelEntry]) -> dict[str, Any]:
    ready_status = READY_RUNTIME_STATUS if spec.kind == "runtime" else READY_CHECKPOINT_STATUS
    safe_prefixes = SAFE_RUNTIME_PREFIXES if spec.kind == "runtime" else SAFE_CHECKPOINT_PREFIXES
    sorted_entries = sorted(entries, key=lambda entry: entry.id)
    entry_reports = [_entry_report(entry, safe_prefixes=safe_prefixes) for entry in sorted_entries]
    available_entries = [report for report in entry_reports if report["status"] == ready_status]
    inventory_ready_entries = [report for report in available_entries if report["path_safety"]["safe"]]
    missing_or_pending = _component_missing_or_pending(spec, entry_reports, ready_status=ready_status)
    inventory_ready = bool(inventory_ready_entries)

    return {
        "component_id": spec.component_id,
        "stage": spec.stage,
        "kind": spec.kind,
        "role": spec.role,
        "serving_backend": spec.serving_backend,
        "required_status": ready_status,
        "checkpoint_available": bool(available_entries) if spec.kind == "checkpoint" else None,
        "runtime_available": bool(available_entries) if spec.kind == "runtime" else None,
        "safe_paths": all(report["path_safety"]["safe"] for report in entry_reports if report["local_path"] is not None),
        "inventory_ready": inventory_ready,
        "serving_ready": False,
        "serving_blockers": ["triton_not_started", "eval0_not_approved"],
        "entries": entry_reports,
        "missing_or_pending": missing_or_pending,
    }


def _entry_report(entry: ModelEntry, *, safe_prefixes: tuple[PurePosixPath, ...]) -> dict[str, Any]:
    path_safety = validate_serving_path(entry.local_path, safe_prefixes=safe_prefixes)
    return {
        "id": entry.id,
        "stage": entry.stage,
        "status": entry.status,
        "local_path": entry.local_path,
        "license": entry.license,
        "commercial_posture": entry.commercial_posture,
        "sha256_present": bool(entry.sha256),
        "fallbacks": list(entry.fallbacks),
        "path_safety": path_safety,
    }


def validate_serving_path(
    local_path: str | None,
    *,
    safe_prefixes: tuple[PurePosixPath, ...] = SAFE_CHECKPOINT_PREFIXES,
) -> dict[str, Any]:
    if local_path is None:
        return {"safe": True, "reason": "not_applicable"}
    if "\x00" in local_path:
        return {"safe": False, "reason": "contains_nul"}

    path = PurePosixPath(local_path)
    if not path.is_absolute():
        return {"safe": False, "reason": "not_absolute"}
    if any(part in {"", ".", ".."} for part in path.parts):
        return {"safe": False, "reason": "contains_relative_segment"}
    if not any(_is_relative_to(path, prefix) for prefix in safe_prefixes):
        return {
            "safe": False,
            "reason": "outside_allowed_prefixes",
            "allowed_prefixes": [prefix.as_posix() for prefix in safe_prefixes],
        }
    return {"safe": True, "reason": "ok"}


def _component_missing_or_pending(
    spec: ComponentSpec,
    entry_reports: list[dict[str, Any]],
    *,
    ready_status: str,
) -> list[dict[str, Any]]:
    if not entry_reports:
        return [
            {
                "id": None,
                "stage": spec.stage,
                "reason": "missing_manifest_entry",
                "required_status": ready_status,
            }
        ]

    items: list[dict[str, Any]] = []
    for report in entry_reports:
        if report["status"] != ready_status:
            items.append(
                {
                    "id": report["id"],
                    "stage": report["stage"],
                    "reason": report["status"],
                    "required_status": ready_status,
                }
            )
        elif not report["path_safety"]["safe"]:
            items.append(
                {
                    "id": report["id"],
                    "stage": report["stage"],
                    "reason": "unsafe_model_path",
                    "path_reason": report["path_safety"]["reason"],
                }
            )
    return items


def _summarize(models: list[ModelEntry], tiers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    components = [component for tier in tiers.values() for component in tier["components"]]
    unsafe_ids = sorted(
        {
            entry["id"]
            for component in components
            for entry in component["entries"]
            if not entry["path_safety"]["safe"]
        }
    )
    pending_ids = sorted(
        {
            item["id"]
            for component in components
            for item in component["missing_or_pending"]
            if item["id"] is not None and item["reason"] != "unsafe_model_path"
        }
    )
    missing_components = sorted(
        {
            component["component_id"]
            for component in components
            for item in component["missing_or_pending"]
            if item["reason"] == "missing_manifest_entry"
        }
    )
    return {
        "source_model_count": len(models),
        "checkpoint_available_count": sum(1 for model in models if model.status == READY_CHECKPOINT_STATUS),
        "runtime_available_count": sum(1 for model in models if model.status == READY_RUNTIME_STATUS),
        "tier_count": len(tiers),
        "component_count": len(components),
        "inventory_ready_component_count": sum(1 for component in components if component["inventory_ready"]),
        "serving_ready_component_count": sum(1 for component in components if component["serving_ready"]),
        "unsafe_model_path_count": len(unsafe_ids),
        "unsafe_model_path_ids": unsafe_ids,
        "pending_item_count": len(pending_ids) + len(missing_components),
        "pending_item_ids": pending_ids,
        "missing_component_ids": missing_components,
    }


def _entries_by_stage(entries: list[ModelEntry]) -> dict[str, list[ModelEntry]]:
    by_stage: dict[str, list[ModelEntry]] = {}
    for entry in entries:
        by_stage.setdefault(entry.stage, []).append(entry)
    return by_stage


def _is_relative_to(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return path == prefix or prefix in path.parents
