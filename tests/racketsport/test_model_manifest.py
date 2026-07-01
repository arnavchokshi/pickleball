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
    assert (
        sam.local_path
        == "/home/arnavchokshi/body_runtime/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt"
    )
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


def test_repo_manifest_records_official_ball_checkpoint_inventory_without_finetune_promotion():
    manifest = load_model_manifest(Path("models/MANIFEST.json"))

    by_id = {entry.id: entry for entry in manifest.models}

    tracknet = by_id["tracknetv3"]
    assert tracknet.stage == "ball_tracking"
    assert tracknet.source == "https://github.com/qaz812345/TrackNetV3"
    assert tracknet.license == "MIT"
    assert tracknet.status == "available_on_h100"
    assert tracknet.local_path == "models/checkpoints/tracknetv3/TrackNet_best.pt"
    assert tracknet.sha256 == "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"
    assert tracknet.repo_commit == "77c123ad4dd449b7d275f16cc43f316ba5b54042"
    assert tracknet.fine_tuned_on_pickleball is False
    assert tracknet.training_data == ["official_tracknetv3_badminton_pretrained"]

    inpaintnet = by_id["tracknetv3_inpaintnet"]
    assert inpaintnet.stage == "ball_tracking"
    assert inpaintnet.source == "https://github.com/qaz812345/TrackNetV3"
    assert inpaintnet.license == "MIT"
    assert inpaintnet.status == "available_on_h100"
    assert inpaintnet.local_path == "models/checkpoints/tracknetv3/InpaintNet_best.pt"
    assert inpaintnet.sha256 == "5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c"
    assert inpaintnet.repo_commit == "77c123ad4dd449b7d275f16cc43f316ba5b54042"
    assert inpaintnet.fine_tuned_on_pickleball is False
    assert inpaintnet.training_data == ["official_tracknetv3_inpaintnet_pretrained"]

    wasb = by_id["wasb_tennis_bmvc2023"]
    assert "https://github.com/nttcom/WASB-SBDT" in wasb.source
    assert wasb.license == "MIT"
    assert wasb.status == "available_on_h100"
    assert wasb.local_path == "models/checkpoints/wasb/wasb_tennis_best.pth.tar"
    assert wasb.sha256 == "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
    assert wasb.repo_commit == "923462cacdeb3353b84ddebdedb3f4b7a8553b0f"
    assert wasb.fine_tuned_on_pickleball is False
    assert wasb.training_data == ["official_wasb_tennis_pretrained"]


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
