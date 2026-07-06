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

    multihmr2 = by_id["multihmr2_b"]
    assert multihmr2.status == "available_on_h100"
    assert multihmr2.local_path == "/workspace/checkpoints/body4d/multihmr2/multihmr2.pt"
    assert multihmr2.sha256 == "fd2d9ab7010a5f590a3db2480f5f476fd3ec6afbbde80babbec0d821ba9763d6"


def test_repo_manifest_marks_rtmw_family_retired_in_favor_of_fast_sam3d_body():
    manifest = load_model_manifest(Path("models/MANIFEST.json"))
    by_id = {entry.id: entry for entry in manifest.models}

    retired_ids = {
        "rtmw_l_384",
        "rtmw_x_384",
        "rtmw3d_x",
        "rtmpose_m_body26_384",
        "rtmpose_l_body26_384",
        "rtmpose_x_body26_384",
        "rtmpose_m_wholebody_256",
        "rtmpose_l_wholebody_384",
        "rtmpose_x_wholebody_384",
    }

    for model_id in retired_ids:
        entry = by_id[model_id]
        assert entry.status == "retired", model_id
        assert "retired" in entry.use.lower(), model_id
        assert entry.fallbacks == [], model_id
        joined_notes = " ".join(entry.notes)
        assert "Fast SAM-3D-Body" in joined_notes, model_id
        assert "speed" in joined_notes.lower(), model_id
        assert "accuracy" in joined_notes.lower(), model_id


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


def test_repo_manifest_records_local_racket_detector_mask_gate_weights():
    manifest = load_model_manifest(Path("models/MANIFEST.json"))

    by_id = {entry.id: entry for entry in manifest.models}

    detector = by_id["grounding_dino_tiny_paddle_detector"]
    assert detector.stage == "racket_detection"
    assert detector.status == "available_local"
    assert detector.local_path == "models/checkpoints/racket/grounding-dino-tiny/model.safetensors"
    assert detector.sha256 == "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
    assert detector.fine_tuned_on_pickleball is False

    base_probe = by_id["grounding_dino_base_paddle_detector_failed_probe"]
    assert base_probe.stage == "racket_detection"
    assert base_probe.status == "available_on_h100"
    assert (
        base_probe.local_path
        == "/home/arnavchokshi/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb/model.safetensors"
    )
    assert base_probe.sha256 == "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"
    assert base_probe.fine_tuned_on_pickleball is False
    assert any("failed the detector+mask gate" in note for note in base_probe.notes)

    sam2 = by_id["sam2_hiera_tiny_racket_masks"]
    assert sam2.stage == "racket_segmentation"
    assert sam2.status == "available_local"
    assert sam2.local_path == "models/checkpoints/racket/sam2-hiera-tiny/sam2_hiera_tiny.pt"
    assert sam2.sha256 == "65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125"
    assert sam2.fine_tuned_on_pickleball is False

    yolo_probe = by_id["yolo11n_paddle_cpu320_e2_failed_probe"]
    assert yolo_probe.stage == "racket_detection"
    assert yolo_probe.status == "available_local"
    assert (
        yolo_probe.local_path
        == "runs/detect/runs/cvat_imports/2026_06_30/racket_yolo_train/yolo11n_paddle_cpu320_e5/weights/best.pt"
    )
    assert yolo_probe.sha256 == "76b9e47ce128d1300ace493d3356bd305150d0256c61c50f778b3274dac8c5cc"
    assert yolo_probe.fine_tuned_on_pickleball is True

    yolo11n_a100 = by_id["yolo11n_paddle_a100_img960_e50_failed_probe"]
    assert yolo11n_a100.stage == "racket_detection"
    assert yolo11n_a100.status == "available_on_h100"
    assert (
        yolo11n_a100.local_path
        == "/home/arnavchokshi/pickleball_git/runs/cvat_imports/2026_06_30/racket_yolo_train_gpu/yolo11n_paddle_img960_e50/weights/best.pt"
    )
    assert yolo11n_a100.sha256 == "7134203e168dbcaef802f26e092d20b14a45e91a757240887becd99741b4bdbe"
    assert yolo11n_a100.fine_tuned_on_pickleball is True
    assert any("failed the RKT detector gate" in note for note in yolo11n_a100.notes)

    yolo26s_a100 = by_id["yolo26s_paddle_a100_img1280_e80_failed_probe"]
    assert yolo26s_a100.stage == "racket_detection"
    assert yolo26s_a100.status == "available_on_h100"
    assert (
        yolo26s_a100.local_path
        == "/home/arnavchokshi/pickleball_git/runs/detect/runs/cvat_imports/2026_06_30/racket_yolo_train_gpu/yolo26s_paddle_img1280_e80/weights/best.pt"
    )
    assert yolo26s_a100.sha256 == "3b6b97a41e4ec14bc61e5468ffc2a5065bef8ff65533cf5d6c82e64a7393f4d8"
    assert yolo26s_a100.fine_tuned_on_pickleball is True
    assert any("must not be promoted" in note for note in yolo26s_a100.notes)


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
