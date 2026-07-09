from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from threed.racketsport.court_detector_v2 import detect_court_v2_from_frame, detect_court_v2_from_frames
from threed.racketsport.court_detector_v2_hypotheses import generate_neural_seed_hypotheses
from threed.racketsport.court_model_infer import resolve_court_model_checkpoint_path

from tests.racketsport.test_court_e4_fusion import (
    _hypothesis,
    _mock_inference,
    _patch_frame_evidence,
    _projected_court,
    _wrong_geometric_keypoints,
)


def test_default_checkpoint_resolution_precedence_and_sha_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    explicit = tmp_path / "explicit.pt"
    env = tmp_path / "env.pt"
    explicit.write_bytes(b"explicit")
    env.write_bytes(b"env")

    monkeypatch.setenv("PICKLEBALL_COURT_UNET_CKPT", str(env))
    caplog.set_level(logging.INFO, logger="threed.racketsport.court_model_infer")

    resolved_explicit = resolve_court_model_checkpoint_path(explicit)
    assert resolved_explicit is not None
    assert resolved_explicit.path == explicit
    assert resolved_explicit.source == "explicit_arg"

    resolved_env = resolve_court_model_checkpoint_path()
    assert resolved_env is not None
    assert resolved_env.path == env
    assert resolved_env.source == "PICKLEBALL_COURT_UNET_CKPT"

    monkeypatch.delenv("PICKLEBALL_COURT_UNET_CKPT", raising=False)
    resolved_default = resolve_court_model_checkpoint_path()
    assert resolved_default is not None
    assert resolved_default.path == Path("models/checkpoints/court_unet_v2/court_model_v2.pt")
    assert resolved_default.source == "promoted_default"
    assert resolved_default.sha256.startswith("cdf0555d49")
    assert "cdf0555d49" in caplog.text


def test_missing_env_checkpoint_fails_safe_instead_of_falling_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing = tmp_path / "missing.pt"
    monkeypatch.setenv("PICKLEBALL_COURT_UNET_CKPT", str(missing))
    caplog.set_level(logging.WARNING, logger="threed.racketsport.court_model_infer")

    assert resolve_court_model_checkpoint_path() is None
    assert "court_unet_v2 checkpoint unavailable" in caplog.text
    assert str(missing) in caplog.text


def test_default_fusion_provider_is_used_by_multiframe_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threed.racketsport.court_detector_v2 as detector
    import threed.racketsport.court_model_infer as infer

    _patch_frame_evidence(monkeypatch)
    gt = _projected_court()

    def fake_generate(frame_bgr: Any, **kwargs: Any) -> list[dict[str, Any]]:
        frame_index = int(frame_bgr[0, 0, 0])
        geometric = _hypothesis(
            f"geometric_wrong_{frame_index}",
            _wrong_geometric_keypoints(frame_index),
            score=1.0,
            evidence_score=0.88,
            source_tag="geometric" if kwargs.get("neural_inference") is not None else None,
        )
        neural = []
        if kwargs.get("neural_inference") is not None:
            neural = generate_neural_seed_hypotheses(
                kwargs["neural_inference"],
                image_size=(int(frame_bgr.shape[1]), int(frame_bgr.shape[0])),
            )
            for item in neural:
                item["score"] = 2.0
                item["evidence_score"] = 0.78
                item["score_components"]["evidence_score"] = 0.78
        return [geometric, *neural]

    def fake_provider(*, checkpoint_path=None, infer_callable=None, device="cpu"):
        assert checkpoint_path is None
        assert infer_callable is None
        assert device == "cpu"
        return lambda _frame: _mock_inference(gt, confidence=0.94)

    monkeypatch.setattr(detector, "generate_homography_hypotheses", fake_generate)
    monkeypatch.setattr(infer, "make_court_model_infer_provider", fake_provider)

    frames = [np.full((540, 720, 3), fill_value=index, dtype=np.uint8) for index in range(3)]
    single = detect_court_v2_from_frame(frames[0], clip_id="synthetic_default")
    fused = detect_court_v2_from_frames(frames, clip_id="synthetic_default")

    assert any(candidate.get("source_tag") == "neural_seeded" for candidate in single["hypotheses"])
    assert fused["provider_enabled"] is True
    assert fused["selected_hypothesis"]["source_tag"] == "neural_seeded"
    assert fused["selected_hypothesis"]["model_confidence"] == pytest.approx(0.94, abs=1e-6)


def test_corrupt_default_provider_keeps_geometric_output_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import threed.racketsport.court_detector_v2 as detector

    _patch_frame_evidence(monkeypatch)
    caplog.set_level(logging.WARNING, logger="threed.racketsport.court_model_infer")

    def fake_generate(frame_bgr: Any, **kwargs: Any) -> list[dict[str, Any]]:
        frame_index = int(frame_bgr[0, 0, 0])
        return [
            _hypothesis(
                f"geometric_{frame_index}",
                _wrong_geometric_keypoints(frame_index),
                score=1.0,
                evidence_score=0.88,
                source_tag=None,
            )
        ]

    monkeypatch.setattr(detector, "generate_homography_hypotheses", fake_generate)

    frames = [np.full((540, 720, 3), fill_value=index, dtype=np.uint8) for index in range(2)]
    missing = tmp_path / "missing.pt"
    monkeypatch.setenv("PICKLEBALL_COURT_UNET_CKPT", str(missing))
    geometric = detect_court_v2_from_frames(frames, clip_id="synthetic_default")

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_text("not a torch checkpoint", encoding="utf-8")
    monkeypatch.setenv("PICKLEBALL_COURT_UNET_CKPT", str(corrupt))
    corrupt_result = detect_court_v2_from_frames(frames, clip_id="synthetic_default")

    assert corrupt_result == geometric
    assert "court_unet_v2 provider disabled" in caplog.text
    assert str(corrupt) in caplog.text


def test_explicit_checkpoint_provider_is_built_once_for_two_frame_fusion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import threed.racketsport.court_detector_v2 as detector
    import threed.racketsport.court_model_infer as infer

    _patch_frame_evidence(monkeypatch)
    gt = _projected_court()
    checkpoint = tmp_path / "court_model_v2.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    build_calls: list[Path | None] = []
    provider_calls = 0

    def fake_generate(frame_bgr: Any, **kwargs: Any) -> list[dict[str, Any]]:
        geometric = _hypothesis(
            "geometric_wrong",
            _wrong_geometric_keypoints(int(frame_bgr[0, 0, 0])),
            score=1.0,
            evidence_score=0.88,
            source_tag="geometric" if kwargs.get("neural_inference") is not None else None,
        )
        neural = []
        if kwargs.get("neural_inference") is not None:
            neural = generate_neural_seed_hypotheses(
                kwargs["neural_inference"],
                image_size=(int(frame_bgr.shape[1]), int(frame_bgr.shape[0])),
            )
        return [geometric, *neural]

    def fake_provider(*, checkpoint_path=None, infer_callable=None, device="cpu"):
        nonlocal provider_calls
        build_calls.append(Path(checkpoint_path) if checkpoint_path is not None else None)
        time.sleep(0.05)

        def _provider(_frame: Any) -> dict[str, Any]:
            nonlocal provider_calls
            provider_calls += 1
            return _mock_inference(gt, confidence=0.93)

        return _provider

    monkeypatch.setattr(detector, "generate_homography_hypotheses", fake_generate)
    monkeypatch.setattr(infer, "make_court_model_infer_provider", fake_provider)

    frames = [np.full((540, 720, 3), fill_value=index, dtype=np.uint8) for index in range(2)]
    started = time.perf_counter()
    fused = detect_court_v2_from_frames(frames, clip_id="synthetic_default", neural_checkpoint_path=checkpoint)
    elapsed = time.perf_counter() - started

    assert build_calls == [checkpoint]
    assert provider_calls == 2
    assert elapsed < 0.15
    assert fused["provider_enabled"] is True
