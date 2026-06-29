from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.serving_manifest import build_serving_manifest


def _write_manifest(path: Path, models: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "models": models}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _available_model(model_id: str, stage: str, local_path: str) -> dict[str, object]:
    return {
        "id": model_id,
        "stage": stage,
        "use": f"{stage} test use",
        "source": "https://example.com/model",
        "license": "MIT",
        "commercial_posture": "ok",
        "status": "available_on_h100",
        "local_path": local_path,
        "sha256": hashlib.sha256(model_id.encode("utf-8")).hexdigest(),
        "fallbacks": [],
    }


def _available_runtime(model_id: str, stage: str, local_path: str) -> dict[str, object]:
    return {
        "id": model_id,
        "stage": stage,
        "use": f"{stage} runtime",
        "source": "https://example.com/runtime",
        "license": "Apache-2.0",
        "commercial_posture": "ok",
        "status": "available_runtime_on_h100",
        "local_path": local_path,
        "fallbacks": [],
    }


def _pending_model(model_id: str, stage: str) -> dict[str, object]:
    return {
        "id": model_id,
        "stage": stage,
        "use": f"{stage} pending use",
        "source": "https://example.com/pending",
        "license": "MIT",
        "commercial_posture": "ok",
        "status": "pending_download",
        "fallbacks": [],
    }


def test_serving_manifest_reports_pending_and_unsafe_entries_without_touching_source(tmp_path: Path) -> None:
    manifest_path = tmp_path / "models" / "MANIFEST.json"
    _write_manifest(
        manifest_path,
        [
            _available_model("fast_sam_3d_body_dinov3", "3d_body_backbone", "/workspace/checkpoints/body/model.ckpt"),
            _available_model("bad_person_detector", "person_detect", "../escaped.pt"),
            _pending_model("tracknetv3", "ball_tracking"),
            _available_runtime("mujoco_mjx", "physics", "/opt/conda/envs/racketsport_mjx"),
        ],
    )
    before = manifest_path.read_bytes()

    report = build_serving_manifest(manifest_path)

    assert manifest_path.read_bytes() == before
    assert report["schema_version"] == 1
    assert report["artifact_type"] == "racketsport_serving_manifest"
    assert report["execution"]["cpu_only"] is True
    assert report["execution"]["starts_triton"] is False
    assert report["execution"]["mutates_model_manifest"] is False

    summary = report["summary"]
    assert summary["unsafe_model_path_count"] == 1
    assert summary["pending_item_count"] >= 1
    assert "bad_person_detector" in summary["unsafe_model_path_ids"]
    assert "tracknetv3" in summary["pending_item_ids"]

    offline_by_id = {component["component_id"]: component for component in report["tiers"]["offline_deep"]["components"]}
    assert offline_by_id["body_backbone"]["checkpoint_available"] is True
    assert offline_by_id["body_backbone"]["safe_paths"] is True
    assert offline_by_id["body_backbone"]["inventory_ready"] is True
    assert offline_by_id["body_backbone"]["serving_ready"] is False
    assert "triton_not_started" in offline_by_id["body_backbone"]["serving_blockers"]
    assert offline_by_id["person_detection"]["checkpoint_available"] is True
    assert offline_by_id["person_detection"]["safe_paths"] is False
    assert offline_by_id["person_detection"]["inventory_ready"] is False
    assert offline_by_id["person_detection"]["serving_ready"] is False
    assert offline_by_id["ball_tracking"]["checkpoint_available"] is False
    assert offline_by_id["ball_tracking"]["missing_or_pending"][0]["reason"] == "pending_download"
    assert offline_by_id["physics_refinement"]["runtime_available"] is True


def test_build_serving_manifest_cli_writes_json_and_preserves_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "models" / "MANIFEST.json"
    out_path = tmp_path / "serving_manifest.json"
    _write_manifest(
        manifest_path,
        [
            _available_model("yolo26m", "person_detect", "/workspace/checkpoints/body4d/yolo26/yolo26m.pt"),
            _available_model("rtmw_l_384", "2d_pose", "/workspace/checkpoints/body4d/rtmw/rtmw-l_384x288.pth"),
        ],
    )
    before = manifest_path.read_text(encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_serving_manifest.py",
            "--manifest",
            str(manifest_path),
            "--out",
            str(out_path),
        ],
        check=True,
    )

    assert manifest_path.read_text(encoding="utf-8") == before
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source_manifest"]["path"] == str(manifest_path)
    assert payload["tiers"]["live_light"]["components"][0]["component_id"] == "person_detection"
    assert payload["tiers"]["live_light"]["eval0_approval"] == "not_evaluated_by_cpu_manifest"


def test_repo_serving_manifest_covers_current_manifest_runtime_and_tiers() -> None:
    report = build_serving_manifest(Path("models/MANIFEST.json"))

    assert report["summary"]["source_model_count"] >= 10
    assert report["summary"]["runtime_available_count"] >= 1
    assert report["tiers"]["offline_deep"]["tier"] == "offline_deep"
    assert report["tiers"]["live_light"]["tier"] == "live_light"
    assert report["tiers"]["offline_deep"]["eval0_approval"] == "not_evaluated_by_cpu_manifest"
    assert report["tiers"]["live_light"]["eval0_approval"] == "not_evaluated_by_cpu_manifest"
    assert report["tiers"]["offline_deep"]["serving_ready"] is False
    assert report["summary"]["serving_ready_component_count"] == 0

    offline_components = {component["component_id"] for component in report["tiers"]["offline_deep"]["components"]}
    assert {"body_backbone", "physics_refinement", "racket_segmentation", "ball_tracking"}.issubset(offline_components)
