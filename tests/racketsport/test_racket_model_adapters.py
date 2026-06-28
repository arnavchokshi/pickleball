from __future__ import annotations

import pytest

from threed.racketsport.racket_model_adapters import (
    RacketModelAdapterBlocked,
    assert_adapter_may_run_gpu_smoke,
    build_racket_model_adapter_plan,
)


def _readiness_report(*, ready: bool = False) -> dict:
    component_ids = [
        "sam3_concept_tracker",
        "dinox_detector",
        "grounded_sam2_video_masks",
        "foundationpose_pose",
        "gigapose_pose",
        "foundpose_pose",
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_model_runtime_readiness",
        "stage": "racket_6dof",
        "status": "ready_for_gpu_smoke" if ready else "blocked",
        "execution": {
            "cpu_only": True,
            "uses_gpu": False,
            "downloads_models": False,
            "imports_model_runtimes": False,
            "runs_inference": False,
            "claims_model_has_run": False,
        },
        "components": [
            {
                "component_id": component_id,
                "role": "test component",
                "runtime_ready": ready,
                "blockers": [] if ready else ["missing_manifest_entry", "missing_runtime_probe"],
            }
            for component_id in component_ids
        ],
        "asset_readiness": {
            "paddle_cad": {"present": ready, "blockers": [] if ready else ["missing_paddle_cad_asset"]},
            "reference_images": {"present": ready, "blockers": [] if ready else ["missing_reference_images"]},
            "aruco_or_apriltag_gt": {"present": ready, "blockers": [] if ready else ["missing_reference_pose_gt"]},
            "face_corner_labels": {"present": ready, "blockers": [] if ready else ["missing_true_face_corners"]},
        },
        "summary": {
            "component_count": len(component_ids),
            "runtime_ready_count": len(component_ids) if ready else 0,
            "asset_ready": ready,
            "may_run_gpu_smoke": ready,
            "may_promote_rkt": False,
        },
        "blockers": [] if ready else ["sam3_concept_tracker:missing_manifest_entry"],
    }


def test_adapter_plan_blocks_missing_runtime_without_claiming_inference() -> None:
    plan = build_racket_model_adapter_plan(_readiness_report(ready=False))

    assert plan["artifact_type"] == "racketsport_racket_model_adapter_plan"
    assert plan["status"] == "blocked"
    assert plan["execution"] == {
        "cpu_only": True,
        "uses_gpu": False,
        "downloads_models": False,
        "imports_model_runtimes": False,
        "runs_inference": False,
        "claims_pose_output": False,
    }
    assert plan["summary"]["adapter_ready_count"] == 0
    assert plan["summary"]["may_run_gpu_smoke"] is False
    assert plan["summary"]["may_promote_rkt"] is False
    assert "sam3_concept_tracker:missing_manifest_entry" in plan["blockers"]
    assert {component["adapter_status"] for component in plan["components"]} == {"blocked"}


def test_adapter_plan_allows_only_gpu_smoke_when_readiness_is_complete() -> None:
    plan = build_racket_model_adapter_plan(_readiness_report(ready=True))

    assert plan["status"] == "ready_for_gpu_smoke"
    assert plan["execution"]["runs_inference"] is False
    assert plan["execution"]["claims_pose_output"] is False
    assert plan["summary"]["adapter_ready_count"] == 6
    assert plan["summary"]["may_run_gpu_smoke"] is True
    assert plan["summary"]["may_promote_rkt"] is False
    assert {component["adapter_status"] for component in plan["components"]} == {"ready_for_gpu_smoke"}


def test_assert_adapter_may_run_gpu_smoke_raises_for_blocked_component() -> None:
    plan = build_racket_model_adapter_plan(_readiness_report(ready=False))

    with pytest.raises(RacketModelAdapterBlocked, match="sam3_concept_tracker"):
        assert_adapter_may_run_gpu_smoke(plan, "sam3_concept_tracker")


def test_assert_adapter_may_run_gpu_smoke_rejects_unknown_component() -> None:
    plan = build_racket_model_adapter_plan(_readiness_report(ready=True))

    with pytest.raises(ValueError, match="unknown racket model adapter"):
        assert_adapter_may_run_gpu_smoke(plan, "not_a_component")
