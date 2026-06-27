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

    rtmpose_m = by_id["rtmpose_m_body26_384"]
    assert rtmpose_m.status == "available_on_h100"
    assert rtmpose_m.local_path == "/workspace/checkpoints/body4d/mmpose/rtmpose-m_body26_384x288.pth"
    assert rtmpose_m.sha256 == "89e6428b38901b5003d0f753619e94d269999e1f5a922349f9068430387863f2"

    rtmw_x = by_id["rtmw_x_384"]
    assert rtmw_x.status == "available_on_h100"
    assert rtmw_x.sha256 == "f840f2044fe46cb3821b7cea86be83e1f6cba406ccd28f5475ac010412dcda95"

    multihmr2 = by_id["multihmr2_b"]
    assert multihmr2.status == "available_on_h100"
    assert multihmr2.local_path == "/workspace/checkpoints/body4d/multihmr2/multihmr2.pt"
    assert multihmr2.sha256 == "fd2d9ab7010a5f590a3db2480f5f476fd3ec6afbbde80babbec0d821ba9763d6"


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
