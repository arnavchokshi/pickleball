"""Fail-closed seams for premium paddle detector and pose-model adapters.

The actual SAM 3, DINO-X/Grounded-SAM2, FoundationPose, GigaPose, and FoundPose
runtimes are intentionally not imported here. This module converts the
CPU-only runtime-readiness report into an explicit adapter plan so future model
runners have one place to check before touching GPU/runtime code.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .racket_model_runtime_readiness import ARTIFACT_TYPE as READINESS_ARTIFACT_TYPE
from .racket_model_runtime_readiness import COMPONENT_SPECS


ARTIFACT_TYPE = "racketsport_racket_model_adapter_plan"
SCHEMA_VERSION = 1
READY_STATUS = "ready_for_gpu_smoke"


class RacketModelAdapterBlocked(RuntimeError):
    """Raised when a paddle model adapter is requested before readiness gates pass."""


def build_racket_model_adapter_plan(
    readiness_report: Mapping[str, Any],
    *,
    requested_components: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a non-inference adapter plan from racket runtime readiness."""

    if readiness_report.get("artifact_type") != READINESS_ARTIFACT_TYPE:
        raise ValueError("readiness_report must be a racket model runtime readiness artifact")

    known_component_ids = [spec.component_id for spec in COMPONENT_SPECS]
    requested = list(requested_components) if requested_components is not None else known_component_ids
    unknown = sorted(set(requested) - set(known_component_ids))
    if unknown:
        raise ValueError(f"unknown racket model adapter: {unknown[0]}")

    readiness_components = {
        str(component.get("component_id")): component
        for component in readiness_report.get("components", [])
        if isinstance(component, Mapping) and component.get("component_id")
    }
    summary = readiness_report.get("summary")
    may_run_gpu_smoke = bool(summary.get("may_run_gpu_smoke")) if isinstance(summary, Mapping) else False
    assets_ready = bool(summary.get("asset_ready")) if isinstance(summary, Mapping) else False
    asset_blockers = _asset_blockers(readiness_report.get("asset_readiness"))

    components = [
        _component_plan(
            component_id,
            readiness_components.get(component_id),
            may_run_gpu_smoke=may_run_gpu_smoke,
            assets_ready=assets_ready,
            asset_blockers=asset_blockers,
        )
        for component_id in requested
    ]
    adapter_ready_count = sum(1 for component in components if component["adapter_status"] == READY_STATUS)
    blocked = [component for component in components if component["adapter_status"] != READY_STATUS]
    blockers = _aggregate_blockers(readiness_report, components, asset_blockers)
    status = READY_STATUS if not blocked and may_run_gpu_smoke else "blocked"

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "stage": "racket_6dof",
        "status": status,
        "source_readiness_status": str(readiness_report.get("status", "unknown")),
        "execution": {
            "cpu_only": True,
            "uses_gpu": False,
            "downloads_models": False,
            "imports_model_runtimes": False,
            "runs_inference": False,
            "claims_pose_output": False,
        },
        "components": components,
        "summary": {
            "component_count": len(components),
            "adapter_ready_count": adapter_ready_count,
            "may_run_gpu_smoke": status == READY_STATUS,
            "may_promote_rkt": False,
        },
        "blockers": blockers,
        "recommended_next_actions": _recommended_next_actions(blockers),
    }


def assert_adapter_may_run_gpu_smoke(plan: Mapping[str, Any], component_id: str) -> Mapping[str, Any]:
    """Return the component plan or raise before a model adapter touches runtime code."""

    components = {
        str(component.get("component_id")): component
        for component in plan.get("components", [])
        if isinstance(component, Mapping) and component.get("component_id")
    }
    if component_id not in {spec.component_id for spec in COMPONENT_SPECS}:
        raise ValueError(f"unknown racket model adapter: {component_id}")
    component = components.get(component_id)
    if not component or plan.get("status") != READY_STATUS or component.get("adapter_status") != READY_STATUS:
        blockers = component.get("blockers", []) if isinstance(component, Mapping) else ["missing_adapter_plan"]
        raise RacketModelAdapterBlocked(f"{component_id} adapter blocked: {', '.join(str(item) for item in blockers)}")
    return component


def _component_plan(
    component_id: str,
    readiness_component: Mapping[str, Any] | None,
    *,
    may_run_gpu_smoke: bool,
    assets_ready: bool,
    asset_blockers: list[str],
) -> dict[str, Any]:
    if readiness_component is None:
        blockers = ["missing_readiness_component"]
        runtime_ready = False
        role = "unknown"
    else:
        blockers = [str(blocker) for blocker in readiness_component.get("blockers", [])]
        runtime_ready = readiness_component.get("runtime_ready") is True
        role = str(readiness_component.get("role", "unknown"))
    if not assets_ready:
        blockers.extend(asset_blockers)
    if not may_run_gpu_smoke and "readiness_report_not_ready_for_gpu_smoke" not in blockers:
        blockers.append("readiness_report_not_ready_for_gpu_smoke")
    blockers = sorted(set(blockers))
    return {
        "component_id": component_id,
        "role": role,
        "runtime_ready": runtime_ready,
        "adapter_status": READY_STATUS if runtime_ready and assets_ready and may_run_gpu_smoke and not blockers else "blocked",
        "blockers": blockers,
    }


def _asset_blockers(asset_readiness: Any) -> list[str]:
    if not isinstance(asset_readiness, Mapping):
        return ["missing_asset_readiness"]
    blockers: list[str] = []
    for asset_id, asset in asset_readiness.items():
        if not isinstance(asset, Mapping):
            blockers.append(f"{asset_id}:invalid_asset_readiness")
            continue
        for blocker in asset.get("blockers", []):
            blockers.append(f"{asset_id}:{blocker}")
    return sorted(set(blockers))


def _aggregate_blockers(
    readiness_report: Mapping[str, Any],
    components: list[Mapping[str, Any]],
    asset_blockers: list[str],
) -> list[str]:
    blockers = [str(blocker) for blocker in readiness_report.get("blockers", [])]
    blockers.extend(asset_blockers)
    for component in components:
        component_id = str(component.get("component_id", "unknown"))
        for blocker in component.get("blockers", []):
            blocker_text = str(blocker)
            if ":" in blocker_text:
                blockers.append(blocker_text)
            else:
                blockers.append(f"{component_id}:{blocker_text}")
    return sorted(set(blockers))


def _recommended_next_actions(blockers: list[str]) -> list[str]:
    actions: list[str] = []
    if any(blocker.endswith(":missing_manifest_entry") for blocker in blockers):
        actions.append("declare and verify paddle detector/pose model manifest entries")
    if any(blocker.endswith(":missing_runtime_probe") for blocker in blockers):
        actions.append("run explicit runtime probes before enabling adapter inference")
    if any("missing_paddle_cad_asset" in blocker or "missing_reference_images" in blocker for blocker in blockers):
        actions.append("add measured paddle CAD or reference images before CAD/reference pose adapters")
    if any("missing_reference_pose_gt" in blocker for blocker in blockers):
        actions.append("add ArUco/AprilTag reference-pose evaluation assets before RKT promotion")
    return actions


__all__ = [
    "ARTIFACT_TYPE",
    "RacketModelAdapterBlocked",
    "assert_adapter_may_run_gpu_smoke",
    "build_racket_model_adapter_plan",
]
