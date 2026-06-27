from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from threed.racketsport.model_manifest import ModelManifest, load_model_manifest


def test_repo_manifest_records_h100_body_checkpoint_inventory():
    manifest = load_model_manifest(Path("models/MANIFEST.json"))

    by_id = {entry.id: entry for entry in manifest.models}

    sam = by_id["fast_sam_3d_body_dinov3"]
    assert sam.status == "available_on_h100"
    assert sam.sha256 == "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"
    assert sam.local_path == "/workspace/checkpoints/body4d/sam-3d-body-dinov3/model.ckpt"
    assert sam.fallbacks == ["sat_hmr", "nlf", "hmr2"]

    assert by_id["sam2_hiera_base_plus"].status == "available_on_h100"
    assert by_id["moge_2_vitl_normal"].status == "available_on_h100"


def test_model_manifest_requires_sha_for_available_entries():
    payload = {
        "schema_version": 1,
        "models": [
            {
                "id": "bad",
                "stage": "test",
                "use": "test",
                "source": "https://example.com/model",
                "license": "MIT",
                "commercial_posture": "ok",
                "status": "available_on_h100",
                "fallbacks": [],
            }
        ],
    }

    with pytest.raises(ValidationError):
        ModelManifest.model_validate(payload)
