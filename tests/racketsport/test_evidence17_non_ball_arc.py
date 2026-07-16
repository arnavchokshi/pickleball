from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.racketsport.calibration_fixtures import (
    minimal_calibration_image_pts,
    minimal_calibration_world_pts,
)
from threed.racketsport import event_fusion, racket6dof
from threed.racketsport.racket6dof import (
    estimate_planar_paddle_pose_with_diagnostics,
    paddle_face_corners_object_cm,
)
from threed.racketsport.racket_stage_runner import RacketStageRunner
from threed.racketsport.schemas import RacketPose, validate_artifact_file


AUDIO_NO_FEATURES_BASELINE_SHA256 = "cf411f114c0a763c5097c950c9eb048286f5b9b09f72486aec7359ed46f7a41b"
RACKET_PRIMARY_BASELINE_SHA256 = "379e156f8baa2a601c53315512fbd954f1ec67a03135f714987a3e9f7daea955"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fusion_payload(features: dict[str, float] | None) -> dict[str, object]:
    onset = {
        "time_s": 1.0,
        "raw_time_s": 1.0,
        "corrected_time_s": 1.01,
        "score": 0.9,
    }
    if features is not None:
        onset["features"] = features
    return event_fusion.fuse_contact_windows_from_cue_payloads(
        fps=100.0,
        audio_onsets_payload={"status": "review_only", "onsets": [onset]},
        wrist_velocity_peaks_payload={
            "peaks": [
                {
                    "time_s": 1.012,
                    "player_id": 7,
                    "wrist_world_xyz": [1.0, 0.0, 0.9],
                    "speed_mps": 8.0,
                    "confidence": 0.8,
                }
            ]
        },
        ball_inflections_payload={
            "candidates": [
                {
                    "time_s": 1.008,
                    "ball_world_xyz": [1.0, 0.0, 0.88],
                    "confidence": 0.7,
                }
            ]
        },
        require_audio=False,
        max_time_delta_s=0.005,
    )


def test_audio_pop_soft_evidence_is_bounded_advisory_and_featureless_path_is_byte_parity() -> None:
    featureless = _fusion_payload(None)
    high_pop = _fusion_payload(
        {
            "spectral_flux": 2.5,
            "high_frequency_content": 3.5,
            "band_energy_delta": 1.2,
            "pop_band_ratio": 0.95,
        }
    )
    low_pop = _fusion_payload(
        {
            "spectral_flux": 2.5,
            "high_frequency_content": 3.5,
            "band_energy_delta": 1.2,
            "pop_band_ratio": 0.05,
        }
    )

    assert _canonical_digest(featureless) == AUDIO_NO_FEATURES_BASELINE_SHA256
    high_event = high_pop["events"][0]
    low_event = low_pop["events"][0]
    assert high_event["confidence"] > featureless["events"][0]["confidence"] > low_event["confidence"]
    high_width = high_event["window"]["t1"] - high_event["window"]["t0"]
    low_width = low_event["window"]["t1"] - low_event["window"]["t0"]
    assert high_width < low_width
    assert high_event["sources"] == low_event["sources"] == {
        "audio": 0.9,
        "wrist_vel": 0.8,
        "ball_inflection": 0.7,
    }


def test_audio_onset_coercion_preserves_raw_feature_values_and_review_only_payload() -> None:
    features = {"pop_band_ratio": 1.25, "future_classifier_logit": -3.0}
    payload = {
        "status": "review_only",
        "warnings": ["pop_transient_heuristic_not_classifier", "cue_not_gate_verified"],
        "onsets": [{"time_s": 0.5, "score": 0.7, "features": features}],
    }

    candidate = event_fusion._coerce_audio_onset(payload["onsets"][0], fps=30.0, frame_times=None)

    assert candidate.features == features
    assert payload["status"] == "review_only"
    assert payload["warnings"] == ["pop_transient_heuristic_not_classifier", "cue_not_gate_verified"]
    assert payload["onsets"][0]["features"] is features


def _projected_paddle_fixture() -> tuple[list[list[float]], list[list[float]], dict[str, float]]:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    camera_matrix = [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]]
    dims = {"length": 16.0, "width": 8.0}
    object_points = np.asarray(paddle_face_corners_object_cm(dims), dtype=np.float64)
    rvec = np.asarray([[0.18], [-0.10], [0.07]], dtype=np.float64)
    tvec = np.asarray([[3.0], [-1.5], [95.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, np.asarray(camera_matrix), None)
    return projected.reshape(-1, 2).tolist(), camera_matrix, dims


def test_second_ippe_pose_is_retained_without_changing_primary_digest() -> None:
    image_points, camera_matrix, dims = _projected_paddle_fixture()

    estimate = estimate_planar_paddle_pose_with_diagnostics(image_points, camera_matrix, dims)
    primary = {
        "R": estimate.pose.R,
        "t": estimate.pose.t,
        "confidence": estimate.pose.confidence,
        "source": estimate.pose.source,
        "reprojection_error_px": estimate.reprojection_error_px,
        "candidate_reprojection_errors_px": estimate.candidate_reprojection_errors_px,
        "ambiguity_margin_px": estimate.ambiguity_margin_px,
        "ambiguous": estimate.ambiguous,
    }

    assert _canonical_digest(primary) == RACKET_PRIMARY_BASELINE_SHA256
    assert estimate.alt_pose is not None
    assert estimate.alt_pose.t != estimate.pose.t
    assert estimate.candidate_count >= 2


def test_second_ippe_pose_fails_closed_on_non_positive_camera_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    import numpy as np

    image_points = np.asarray(
        [[100.0, 100.0], [200.0, 100.0], [200.0, 300.0], [100.0, 300.0]],
        dtype=np.float64,
    )

    class FakeCv2:
        SOLVEPNP_IPPE = 1
        SOLVEPNP_ITERATIVE = 2

        @staticmethod
        def solvePnPGeneric(*_args, **_kwargs):
            zero = np.zeros((3, 1), dtype=np.float64)
            return True, [zero, zero], [np.asarray([[0.0], [0.0], [100.0]]), np.asarray([[0.0], [0.0], [-100.0]])], None

        @staticmethod
        def projectPoints(_object_points, _rvec, tvec, _camera, _dist):
            projected = image_points.reshape(-1, 1, 2).copy()
            if float(np.asarray(tvec).reshape(3)[2]) < 0.0:
                projected += 1.0
            return projected, None

        @staticmethod
        def Rodrigues(rvec):
            return cv2.Rodrigues(rvec)

    monkeypatch.setattr(racket6dof, "_cv2_np", lambda: (FakeCv2, np))
    with pytest.raises(ValueError, match="second IPPE paddle pose is degenerate"):
        estimate_planar_paddle_pose_with_diagnostics(
            image_points.tolist(),
            [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
            {"length": 16.0, "width": 8.0},
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_racket_stage_default_carries_ambiguous_frame_and_both_hypotheses(tmp_path: Path) -> None:
    image_points, _camera_matrix, dims = _projected_paddle_fixture()
    inputs = tmp_path / "inputs" / "clip_001"
    run_dir = tmp_path / "runs" / "clip_001"
    _write_json(
        run_dir / "court_calibration.json",
        {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
            "intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 320.0, "cy": 240.0, "dist": [], "source": "synthetic"},
            "extrinsics": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.0, 0.0, 12.0], "camera_height_m": 12.0},
            "reprojection_error_px": {"median": 0.0, "p95": 0.0},
            "capture_quality": {"grade": "good", "reasons": []},
            "image_pts": minimal_calibration_image_pts(),
            "world_pts": minimal_calibration_world_pts(),
        },
    )
    _write_json(
        inputs / "racket_candidates.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_candidates",
            "fps": 60.0,
            "players": [
                {
                    "id": 7,
                    "paddle_dims_in": dims,
                    "frames": [{"t": 0.0, "corners_px": image_points, "conf": 0.92, "source": "synthetic_corners"}],
                }
            ],
        },
    )
    context = SimpleNamespace(inputs_dir=inputs, run_dir=run_dir, clip="clip_001")

    result = RacketStageRunner(ambiguity_margin_threshold_px=10.0).run(context)

    assert result.metrics["reject_ambiguous"] is False
    assert result.metrics["carried_ambiguous_count"] == 1
    assert result.metrics["rejected_ambiguous_count"] == 0
    primary = validate_artifact_file("racket_pose", run_dir / "racket_pose.json")
    assert isinstance(primary, RacketPose)
    assert primary.players[0].frames[0].ambiguous is True
    hypotheses = json.loads((run_dir / "racket_pose_hypotheses.json").read_text(encoding="utf-8"))
    frame = hypotheses["players"][0]["frames"][0]
    assert frame["ambiguous"] is True
    assert frame["primary_pose"] is not None
    assert frame["alt_pose"] is not None
    assert frame["ambiguity_margin_px"] <= 10.0
