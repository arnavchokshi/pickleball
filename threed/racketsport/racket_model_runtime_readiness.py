"""CPU-only readiness report for paddle detector, mask, and 6DoF pose runtimes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .model_manifest import ModelEntry, load_model_manifest
from .serving_manifest import SAFE_CHECKPOINT_PREFIXES, validate_serving_path


ARTIFACT_TYPE = "racketsport_racket_model_runtime_readiness"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RacketRuntimeComponentSpec:
    component_id: str
    role: str
    expected_manifest_ids: tuple[str, ...]


COMPONENT_SPECS: tuple[RacketRuntimeComponentSpec, ...] = (
    RacketRuntimeComponentSpec(
        "sam3_concept_tracker",
        "paddle detection/segmentation/tracking",
        ("sam3_concept_tracker",),
    ),
    RacketRuntimeComponentSpec("dinox_detector", "open-vocabulary paddle detection", ("dinox_detector",)),
    RacketRuntimeComponentSpec(
        "grounded_sam2_video_masks",
        "grounded paddle masks and video propagation",
        ("grounded_sam2_video_masks",),
    ),
    RacketRuntimeComponentSpec(
        "foundationpose_pose",
        "CAD/reference-image 6DoF paddle pose",
        ("foundationpose_pose",),
    ),
    RacketRuntimeComponentSpec("gigapose_pose", "CAD-template RGB 6DoF paddle pose", ("gigapose_pose",)),
    RacketRuntimeComponentSpec("foundpose_pose", "foundation-feature 6DoF paddle pose", ("foundpose_pose",)),
)


DEFAULT_ASSET_PATHS: Mapping[str, str] = {
    "paddle_cad": "assets/racketsport/paddle_cad",
    "reference_images": "assets/racketsport/paddle_reference_images",
    "aruco_or_apriltag_gt": "data/racketsport/paddle_reference_pose_gt",
    "face_corner_labels": "data/racketsport/paddle_face_corner_labels",
}


def build_racket_model_runtime_readiness(
    manifest_path: str | Path = "models/MANIFEST.json",
    *,
    check_files: bool = False,
    allowed_checkpoint_prefixes: tuple[str, ...] | None = None,
    asset_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a non-inference readiness report for the premium paddle stack."""

    source_path = Path(manifest_path)
    manifest = load_model_manifest(source_path)
    manifest_bytes = source_path.read_bytes()
    entries_by_id = {entry.id: entry for entry in manifest.models}
    safe_prefixes = _safe_prefixes(allowed_checkpoint_prefixes)
    components = [
        _component_report(spec, entries_by_id, check_files=check_files, safe_prefixes=safe_prefixes)
        for spec in COMPONENT_SPECS
    ]
    assets = _asset_readiness(asset_paths or DEFAULT_ASSET_PATHS)
    asset_ready = all(item["present"] for item in assets.values())
    runtime_ready_count = sum(1 for component in components if component["runtime_ready"])
    component_count = len(components)
    may_run_gpu_smoke = runtime_ready_count == component_count and asset_ready
    blockers = _aggregate_blockers(components, assets)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "stage": "racket_6dof",
        "status": "ready_for_gpu_smoke" if may_run_gpu_smoke else "blocked",
        "source_manifest": {
            "path": str(source_path),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "schema_version": manifest.schema_version,
        },
        "execution": {
            "cpu_only": True,
            "uses_gpu": False,
            "downloads_models": False,
            "imports_model_runtimes": False,
            "runs_inference": False,
            "claims_model_has_run": False,
            "mutates_model_manifest": False,
        },
        "components": components,
        "asset_readiness": assets,
        "summary": {
            "component_count": component_count,
            "runtime_ready_count": runtime_ready_count,
            "asset_ready": asset_ready,
            "may_run_gpu_smoke": may_run_gpu_smoke,
            "may_promote_rkt": False,
        },
        "blockers": blockers,
        "recommended_next_actions": _recommended_next_actions(blockers),
    }


def write_racket_model_runtime_readiness(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _component_report(
    spec: RacketRuntimeComponentSpec,
    entries_by_id: Mapping[str, ModelEntry],
    *,
    check_files: bool,
    safe_prefixes: tuple[PurePosixPath, ...],
) -> dict[str, Any]:
    entries = [entries_by_id[model_id] for model_id in spec.expected_manifest_ids if model_id in entries_by_id]
    entry_reports = [_entry_report(entry, check_files=check_files, safe_prefixes=safe_prefixes) for entry in entries]
    blockers = _component_blockers(spec, entry_reports)
    return {
        "component_id": spec.component_id,
        "role": spec.role,
        "expected_manifest_ids": list(spec.expected_manifest_ids),
        "manifest_status": _manifest_status(spec, entry_reports),
        "license_review_status": _license_review_status(entry_reports),
        "repo_status": _repo_status(entry_reports),
        "checkpoint_status": _checkpoint_status(entry_reports),
        "path_safety": _combined_path_safety(entry_reports),
        "runtime_ready": not blockers,
        "blockers": blockers,
        "entries": entry_reports,
    }


def _entry_report(
    entry: ModelEntry,
    *,
    check_files: bool,
    safe_prefixes: tuple[PurePosixPath, ...],
) -> dict[str, Any]:
    path_safety = validate_serving_path(entry.local_path, safe_prefixes=safe_prefixes)
    file_check = _file_check(entry) if check_files and path_safety["safe"] and entry.local_path else {"status": "not_checked"}
    return {
        "id": entry.id,
        "stage": entry.stage,
        "status": entry.status,
        "local_path": entry.local_path,
        "sha256_present": bool(entry.sha256),
        "license": entry.license,
        "commercial_posture": entry.commercial_posture,
        "repo_commit": entry.repo_commit,
        "source": entry.source,
        "path_safety": path_safety,
        "file_check": file_check,
    }


def _file_check(entry: ModelEntry) -> dict[str, Any]:
    if not entry.local_path:
        return {"status": "missing_local_path"}
    path = Path(entry.local_path)
    if not path.is_file():
        return {"status": "missing_file"}
    if not entry.sha256:
        return {"status": "missing_sha256"}
    digest = _sha256_file(path)
    return {"status": "verified" if digest.lower() == entry.sha256.lower() else "sha256_mismatch", "sha256": digest}


def _component_blockers(spec: RacketRuntimeComponentSpec, entry_reports: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    if len(entry_reports) < len(spec.expected_manifest_ids):
        blockers.append("missing_manifest_entry")
        blockers.append("missing_license_review")
        blockers.append("missing_runtime_probe")
        return blockers
    for report in entry_reports:
        if report["status"] != "available_on_h100":
            _append_once(blockers, "pending_manifest_entry")
        if report["commercial_posture"] != "ok":
            _append_once(blockers, "license_review_required")
        if report["path_safety"]["safe"] is not True:
            _append_once(blockers, "unsafe_local_path")
        file_status = report["file_check"]["status"]
        if file_status in {"missing_file", "missing_local_path", "missing_sha256", "sha256_mismatch"}:
            _append_once(blockers, file_status)
    if "pending_manifest_entry" in blockers:
        _append_once(blockers, "missing_runtime_probe")
    return blockers


def _manifest_status(spec: RacketRuntimeComponentSpec, entry_reports: list[dict[str, Any]]) -> str:
    if len(entry_reports) < len(spec.expected_manifest_ids):
        return "missing"
    statuses = sorted({str(report["status"]) for report in entry_reports})
    return statuses[0] if len(statuses) == 1 else "mixed:" + ",".join(statuses)


def _license_review_status(entry_reports: list[dict[str, Any]]) -> str:
    if not entry_reports:
        return "missing"
    if all(report["commercial_posture"] == "ok" for report in entry_reports):
        return "declared_ok"
    return "needs_review"


def _repo_status(entry_reports: list[dict[str, Any]]) -> str:
    if not entry_reports:
        return "not_declared"
    if any(report.get("repo_commit") for report in entry_reports):
        return "commit_declared"
    if any(report.get("source") for report in entry_reports):
        return "source_declared"
    return "not_declared"


def _checkpoint_status(entry_reports: list[dict[str, Any]]) -> str:
    if not entry_reports:
        return "not_declared"
    if any(report["path_safety"]["safe"] is not True for report in entry_reports):
        return "unsafe_path"
    file_statuses = {str(report["file_check"]["status"]) for report in entry_reports}
    if "sha256_mismatch" in file_statuses:
        return "sha256_mismatch"
    if "missing_file" in file_statuses:
        return "missing_file"
    if "missing_sha256" in file_statuses or "missing_local_path" in file_statuses:
        return "incomplete"
    if file_statuses == {"verified"}:
        return "verified"
    if all(report["status"] == "available_on_h100" for report in entry_reports):
        return "declared_not_verified"
    return "pending"


def _combined_path_safety(entry_reports: list[dict[str, Any]]) -> dict[str, Any]:
    unsafe = [report for report in entry_reports if report["path_safety"]["safe"] is not True]
    if unsafe:
        first = unsafe[0]["path_safety"]
        return {"safe": False, "reason": first.get("reason", "unsafe_path")}
    return {"safe": True, "reason": "ok" if entry_reports else "not_applicable"}


def _asset_readiness(asset_paths: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    readiness: dict[str, dict[str, Any]] = {}
    blocker_by_asset = {
        "paddle_cad": "missing_paddle_cad_asset",
        "reference_images": "missing_reference_images",
        "aruco_or_apriltag_gt": "missing_reference_pose_gt",
        "face_corner_labels": "missing_true_face_corners",
    }
    for key, raw_path in asset_paths.items():
        path = Path(raw_path)
        present = path.exists() and (path.is_file() or any(child.is_file() for child in path.rglob("*")))
        readiness[key] = {
            "path": str(path),
            "present": present,
            "blockers": [] if present else [blocker_by_asset.get(key, f"missing_asset:{key}")],
        }
    return readiness


def _aggregate_blockers(components: list[dict[str, Any]], assets: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for component in components:
        component_id = str(component["component_id"])
        for blocker in component["blockers"]:
            blockers.append(f"{component_id}:{blocker}")
    for asset_id, asset in assets.items():
        for blocker in asset.get("blockers", []):
            blockers.append(f"{asset_id}:{blocker}")
    return sorted(blockers)


def _recommended_next_actions(blockers: list[str]) -> list[str]:
    actions: list[str] = []
    if any(blocker.endswith(":missing_manifest_entry") for blocker in blockers):
        actions.append("add explicit manifest entries for SAM 3, DINO-X/Grounded-SAM2, FoundationPose, GigaPose, and FoundPose")
    if any("license" in blocker for blocker in blockers):
        actions.append("record license/commercial posture review before runtime setup")
    if any(blocker.endswith(":unsafe_local_path") for blocker in blockers):
        actions.append("move declared checkpoints under approved H100 checkpoint prefixes")
    if any(blocker.endswith(":missing_reference_pose_gt") for blocker in blockers):
        actions.append("capture or declare ArUco/AprilTag reference-pose evaluation assets")
    if any(blocker.endswith(":missing_paddle_cad_asset") for blocker in blockers):
        actions.append("add a measured paddle CAD/reference geometry asset for 6DoF pose methods")
    return actions


def _safe_prefixes(allowed_checkpoint_prefixes: tuple[str, ...] | None) -> tuple[PurePosixPath, ...]:
    if allowed_checkpoint_prefixes is None:
        return SAFE_CHECKPOINT_PREFIXES
    return tuple(PurePosixPath(prefix) for prefix in allowed_checkpoint_prefixes)


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ARTIFACT_TYPE",
    "COMPONENT_SPECS",
    "build_racket_model_runtime_readiness",
    "write_racket_model_runtime_readiness",
]
