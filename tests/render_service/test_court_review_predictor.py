import json
import subprocess
from pathlib import Path

import pytest

from server import court_review
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


def _fake_ffprobe_run(width: int = 1000, height: int = 600, ok: bool = True):
    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        if not ok:
            return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="ffprobe failed")
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "width": width,
                            "height": height,
                            "avg_frame_rate": "60/1",
                            "duration": "10.0",
                            "nb_frames": "600",
                        }
                    ]
                }
            ),
            stderr="",
        )

    return fake_run


def _fake_proposal_report(*, image_size=(1000, 600), frame_indices=(4,)) -> dict:
    keypoints = {
        "near_left_corner": [180.0, 520.0],
        "near_baseline_center": [500.0, 520.0],
        "near_right_corner": [820.0, 520.0],
        "far_right_corner": [780.0, 180.0],
        "far_baseline_center": [500.0, 180.0],
        "far_left_corner": [220.0, 180.0],
        "near_nvz_left": [220.0, 400.0],
        "near_nvz_center": [500.0, 400.0],
        "near_nvz_right": [780.0, 400.0],
        "net_left_sideline": [230.0, 330.0],
        "net_center": [500.0, 330.0],
        "net_right_sideline": [770.0, 330.0],
        "far_nvz_left": [240.0, 260.0],
        "far_nvz_center": [500.0, 260.0],
        "far_nvz_right": [760.0, 260.0],
    }
    runner_up_keypoints = dict(keypoints)
    runner_up_keypoints["near_left_corner"] = [140.0, 520.0]
    return {
        "artifact_type": "racketsport_court_proposals",
        "schema_version": 1,
        "clip": "clip_1",
        "status": "ranked_not_verified",
        "verified": False,
        "not_cal3_verified": True,
        "input": {
            "video": "clip.mp4",
            "frame_indices": list(frame_indices),
            "image_size": list(image_size),
            "motion_mode": "static",
        },
        "assist": {"mode": "one_inside_tap", "tap_points": [[500.0, 300.0]], "line_label": None},
        "ranking": {
            "selected_proposal_id": "proposal_0001",
            "selection_reason": "best_score_but_review_required",
            "abstain": True,
            "abstain_reasons": ["not_cal3_verified"],
        },
        "proposals": [
            {
                "proposal_id": "proposal_0001",
                "source": "surface_lines",
                "verified": False,
                "not_cal3_verified": True,
                "court_keypoints": keypoints,
                "homography_image_from_court": None,
                "scores": {"overall": 0.82},
                "gate": {"auto_usable": False, "review_usable": True, "failed": ["not_verified"], "warnings": []},
                "evidence": {},
            },
            {
                "proposal_id": "proposal_0002",
                "source": "net_anchor",
                "verified": False,
                "not_cal3_verified": True,
                "court_keypoints": runner_up_keypoints,
                "homography_image_from_court": None,
                "scores": {"overall": 0.55},
                "gate": {"auto_usable": False, "review_usable": True, "failed": ["not_verified"], "warnings": []},
                "evidence": {},
            },
        ],
    }


def test_public_court_predictor_defaults_to_template_seed(monkeypatch) -> None:
    monkeypatch.delenv("PICKLEBALL_COURT_PREDICTOR_MODE", raising=False)
    monkeypatch.setattr(court_review.subprocess, "run", _fake_ffprobe_run())

    prediction = court_review.predict_court_layout_from_video(video_path=Path("/tmp/clip.mp4"), clip="clip_1")

    assert prediction["prediction_source"] == "template_projection_seed:ffprobe_metadata"
    assert prediction["verified"] is False
    assert prediction["not_cal3_verified"] is True
    assert prediction["image_size"] == [1000, 600]
    assert len(prediction["points"]) == 15
    assert "template_seed_not_automatic_detection" in prediction["promotion_blockers"]
    # Default mode is "proposals"; with no propose_court_from_video implementation present,
    # this must fall back to the template seed with an explicit warning, not crash.
    assert "proposal_pipeline_unavailable" in prediction["warnings"]


def test_detector_mode_falls_back_to_template_seed_when_detector_fails(monkeypatch) -> None:
    def fake_detector(**kwargs):
        raise RuntimeError("detector unavailable")

    monkeypatch.setenv("PICKLEBALL_COURT_PREDICTOR_MODE", "detector")
    monkeypatch.setattr(court_review.subprocess, "run", _fake_ffprobe_run(ok=False))
    monkeypatch.setattr(court_review, "_predict_with_detector", fake_detector)

    prediction = court_review.predict_court_layout_from_video(video_path=Path("/tmp/clip.mp4"), clip="clip_1")

    assert prediction["prediction_source"] == "template_projection_seed:ffprobe_metadata"
    assert prediction["image_size"] == [1280, 720]
    assert "court_detector_v2_failed_fell_back_to_template:RuntimeError" in prediction["warnings"]


def test_proposals_mode_falls_back_to_template_with_explicit_warning_when_pipeline_unavailable(monkeypatch) -> None:
    """propose_court_from_video does not exist yet (CAL-GEO lane in flight): any failure
    (ImportError/AttributeError/etc) must fail closed to the template seed, never crash."""

    monkeypatch.setenv("PICKLEBALL_COURT_PREDICTOR_MODE", "proposals")
    monkeypatch.setattr(court_review.subprocess, "run", _fake_ffprobe_run())

    prediction = court_review.predict_court_layout_from_video(video_path=Path("/tmp/clip.mp4"), clip="clip_1")

    assert prediction["prediction_source"] == "template_projection_seed:ffprobe_metadata"
    assert prediction["verified"] is False
    assert prediction["not_cal3_verified"] is True
    assert "proposal_pipeline_unavailable" in prediction["warnings"]
    assert any(warning.startswith("court_proposals_failed:") for warning in prediction["warnings"])
    # All 15 points still flow through the shared needs_user_input/assist finalization.
    assert set(prediction["needs_user_input"]) == set(PICKLEBALL_COURT_KEYPOINT_NAMES)
    assert prediction["assist"] == {"mode": "none", "tap_points": [], "line_label": None}


def test_proposals_mode_strict_env_reraises_instead_of_masking_failures(monkeypatch) -> None:
    monkeypatch.setenv("PICKLEBALL_COURT_PREDICTOR_MODE", "proposals")
    monkeypatch.setenv("PICKLEBALL_COURT_PREDICTOR_STRICT", "1")

    with pytest.raises(ImportError):
        court_review.predict_court_layout_from_video(video_path=Path("/tmp/clip.mp4"), clip="clip_1")


def test_proposals_mode_uses_selected_proposal_when_pipeline_succeeds(monkeypatch) -> None:
    report = _fake_proposal_report()

    def fake_propose_court_from_video(video_path, *, max_frames=24, top_k=8, tracks_path=None):
        assert max_frames == 24
        assert top_k == 8
        return report

    monkeypatch.setenv("PICKLEBALL_COURT_PREDICTOR_MODE", "proposals")
    monkeypatch.setattr(court_review.subprocess, "run", _fake_ffprobe_run())

    import threed.racketsport.court_proposals as court_proposals_module

    monkeypatch.setattr(court_proposals_module, "propose_court_from_video", fake_propose_court_from_video, raising=False)

    prediction = court_review.predict_court_layout_from_video(video_path=Path("/tmp/clip.mp4"), clip="clip_1")

    assert prediction["prediction_source"] == "court_proposals:selected_proposal=proposal_0001"
    assert prediction["verified"] is False
    assert prediction["not_cal3_verified"] is True
    assert prediction["image_size"] == [1000, 600]
    assert prediction["frame_index"] == 4
    assert prediction["points"]["near_left_corner"]["xy"] == [180.0, 520.0]
    assert prediction["points"]["near_left_corner"]["confidence"] == pytest.approx(0.82)
    assert prediction["needs_user_input"] == []
    assert prediction["assist"] == {"mode": "one_inside_tap", "tap_points": [[500.0, 300.0]], "line_label": None}
    assert prediction["selected_proposal_id"] == "proposal_0001"
    assert [proposal["proposal_id"] for proposal in prediction["proposals"]] == ["proposal_0001", "proposal_0002"]
    assert prediction["proposal_report"]["ranking"]["selected_proposal_id"] == "proposal_0001"


def test_proposals_mode_falls_back_when_selected_proposal_missing_required_keypoints(monkeypatch) -> None:
    report = _fake_proposal_report()
    del report["proposals"][0]["court_keypoints"]["near_left_corner"]

    def fake_propose_court_from_video(video_path, *, max_frames=24, top_k=8, tracks_path=None):
        return report

    monkeypatch.setenv("PICKLEBALL_COURT_PREDICTOR_MODE", "proposals")
    monkeypatch.setattr(court_review.subprocess, "run", _fake_ffprobe_run())

    import threed.racketsport.court_proposals as court_proposals_module

    monkeypatch.setattr(court_proposals_module, "propose_court_from_video", fake_propose_court_from_video, raising=False)

    prediction = court_review.predict_court_layout_from_video(video_path=Path("/tmp/clip.mp4"), clip="clip_1")

    assert prediction["prediction_source"] == "template_projection_seed:ffprobe_metadata"
    assert "proposal_pipeline_unavailable" in prediction["warnings"]
