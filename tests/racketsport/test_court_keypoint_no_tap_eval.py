from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.racketsport.calibration_fixtures import minimal_ready_court_line_evidence
from threed.racketsport.court_keypoint_eval import (
    CourtKeypointNoTapEvalReport,
    build_court_keypoint_prediction_validation,
    build_court_keypoint_no_tap_eval_report,
    select_active_clip_inputs,
)
from threed.racketsport.court_calibration import project_planar_points
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_clip(
    run_root: Path,
    clip: str,
    *,
    ready_evidence: bool = True,
    top_net_dx: float = 0.0,
) -> None:
    clip_dir = run_root / clip
    clip_dir.mkdir(parents=True)
    (clip_dir / "court_calibration.json").write_text("{}", encoding="utf-8")
    (clip_dir / "court_zones.json").write_text("{}", encoding="utf-8")
    (clip_dir / "net_plane.json").write_text("{}", encoding="utf-8")
    video = clip_dir / "tracknet_smoke_0000_0010" / "input_0000_0010.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not a real video; dry-run only")

    evidence = minimal_ready_court_line_evidence()
    if not ready_evidence:
        evidence["aggregate"]["auto_calibration_ready"] = False
        evidence["aggregate"]["missing_required_line_ids"] = ["near_nvz"]
        evidence["aggregate"]["reasons"] = ["missing_near_nvz"]
    evidence["net_observations"][0]["image_points"] = [
        [100.0 + top_net_dx, 240.0],
        [500.0 + top_net_dx, 238.0],
        [900.0 + top_net_dx, 240.0],
    ]
    _write_json(clip_dir / "court_line_evidence.json", evidence)
    _write_json(
        clip_dir / "top_net_review_points.json",
        {
            "schema_version": 1,
            "artifact_type": "pickleball_human_top_net_review_points",
            "clip": clip,
            "click_coordinate_space": {"width": 960, "height": 540},
            "click_points": {
                "left": {"x": 100.0, "y": 240.0, "time_s": 0.0, "video_width": 960, "video_height": 540},
                "right": {"x": 900.0, "y": 240.0, "time_s": 0.0, "video_width": 960, "video_height": 540},
            },
            "evidence_scale": {"x": 2.0, "y": 2.0},
            "evidence_points": {
                "left": [200.0, 480.0],
                "center": [1000.0, 476.0],
                "right": [1800.0, 480.0],
            },
            "source_review_input": "runs/review_inputs/pickleball_cv_review_latest.json",
            "notes": "",
        },
    )


def _checkpoint_inputs(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "court_keypoint_heatmap.pt"
    checkpoint.write_bytes(b"checkpoint bytes are not loaded in dry-run")
    metrics = tmp_path / "court_keypoint_metrics.json"
    _write_json(
        metrics,
        {
            "schema_version": 1,
            "artifact_type": "court_keypoint_pretraining_run",
            "status": "trained_not_phase_verified",
            "checkpoint": str(checkpoint),
            "before": {},
            "after": {"real_corner_mean_px": 72.688, "synthetic_mean_px": 27.531},
            "history": [],
            "real_train_count": 3,
            "real_holdout_count": 1,
            "note": "not a verified CAL-3 no-tap solver",
        },
    )
    return checkpoint, metrics


def _calibration_payload() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[18.0, 1.5, 120.0], [0.5, 20.0, 220.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "dist": [], "source": "test"},
        "image_size": [640, 480],
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 2.0],
            "camera_height_m": 2.0,
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[65.0, 70.0], [575.0, 70.0], [575.0, 410.0], [65.0, 410.0]],
        "world_pts": [[-3.048, -6.7056, 0.0], [3.048, -6.7056, 0.0], [3.048, 6.7056, 0.0], [-3.048, 6.7056, 0.0]],
    }


def _prediction_payload(
    calibration: dict,
    *,
    confidence: float,
    border_biased: bool = False,
    frame_count: int = 3,
) -> dict:
    expected = {
        point.name: project_planar_points(calibration["homography"], [point.world_xyz_m[:2]])[0]
        for point in PICKLEBALL_KEYPOINTS
    }
    frames = []
    for frame_index in range(frame_count):
        keypoints = {}
        for point_index, point in enumerate(PICKLEBALL_KEYPOINTS):
            if border_biased:
                xy = [635.0, 5.0 + float(point_index % 3)]
            else:
                base_x, base_y = expected[point.name]
                xy = [base_x + (0.5 * frame_index), base_y - (0.25 * frame_index)]
            keypoints[point.name] = {"xy": xy, "confidence": confidence, "heatmap_score": confidence}
        frames.append(
            {
                "frame_index": frame_index,
                "keypoints": keypoints,
                "confident_keypoint_count": 0,
                "solvepnp_correspondence_count": 0,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "court_keypoint_no_tap_predictions",
        "clip": "wolverine_mixed_0200_mid_steep_corner",
        "video": "input.mp4",
        "coordinate_space": "source_video_pixels",
        "model_input_size": [160, 90],
        "source_size": [640, 480],
        "min_confidence": 0.5,
        "frames": frames,
        "verified": False,
        "not_cal3_verified": True,
    }


def test_load_trusted_court_keypoint_checkpoint_allows_repo_checkpoint_metadata(tmp_path: Path) -> None:
    from threed.racketsport import court_keypoint_eval

    loader = getattr(court_keypoint_eval, "_load_trusted_court_keypoint_checkpoint", None)
    assert loader is not None
    checkpoint = tmp_path / "court_keypoint_heatmap.pt"
    checkpoint.write_bytes(b"repo-owned checkpoint")
    calls = []

    class FakeTorch:
        def load(self, checkpoint_path: Path, **kwargs):
            calls.append((checkpoint_path, kwargs))
            if kwargs.get("weights_only") is not False:
                raise RuntimeError("weights-only checkpoint loading rejects metadata paths")
            return {
                "args": {"checkpoint": tmp_path},
                "image_size": [160, 90],
                "keypoint_names": ["near_left_corner"],
                "model": {"0.weight": object()},
            }

    payload = loader(FakeTorch(), checkpoint, device="cpu")

    assert payload["image_size"] == [160, 90]
    assert calls == [(checkpoint, {"map_location": "cpu", "weights_only": False})]


def test_select_active_clip_inputs_excludes_retired_court_clip_and_checks_reviewed_top_net(tmp_path: Path) -> None:
    run_root = tmp_path / "prototype_gate_h100_v2"
    _write_clip(run_root, "burlington_gold_0300_low_steep_corner")
    _write_clip(run_root, "wolverine_mixed_0200_mid_steep_corner", top_net_dx=4.0)
    _write_clip(run_root, "outdoor_webcam_iynbd_1500_long_high_baseline", ready_evidence=False)

    selected = select_active_clip_inputs(run_root)

    assert [clip.clip for clip in selected.active] == ["wolverine_mixed_0200_mid_steep_corner"]
    assert selected.skipped["burlington_gold_0300_low_steep_corner"].status == "skipped_retired_for_court"
    assert selected.skipped["outdoor_webcam_iynbd_1500_long_high_baseline"].status == "blocked"
    assert selected.active[0].top_net_review_match.passed is True
    assert selected.active[0].top_net_review_match.best_coordinate_space == "click_points"
    assert selected.active[0].top_net_review_match.max_endpoint_delta_px == pytest.approx(4.0)


def test_build_dry_run_report_is_schema_valid_and_never_claims_cal3_verified(tmp_path: Path) -> None:
    run_root = tmp_path / "prototype_gate_h100_v2"
    _write_clip(run_root, "wolverine_mixed_0200_mid_steep_corner")
    checkpoint, metrics = _checkpoint_inputs(tmp_path)

    report = build_court_keypoint_no_tap_eval_report(
        run_root=run_root,
        checkpoint=checkpoint,
        metrics=metrics,
        out=tmp_path / "dry_run.json",
        dry_run=True,
        device="cpu",
        frames_per_clip=3,
    )
    parsed = CourtKeypointNoTapEvalReport.model_validate(report.model_dump(mode="json"))

    assert parsed.status == "ready_for_h100"
    assert parsed.claim_scope == "dry_run_plumbing"
    assert parsed.verified is False
    assert parsed.not_cal3_verified is True
    assert parsed.min_confidence == pytest.approx(0.5)
    assert parsed.summary.active_clip_count == 1
    device_index = parsed.summary.h100_gate_command.index("--device")
    assert parsed.summary.h100_gate_command[device_index + 1] == "cuda"
    assert "evaluate_court_keypoint_no_tap.py" in parsed.summary.h100_gate_command[1]
    assert parsed.clips[0].status == "ready_for_h100"


def test_build_report_records_diagnostic_min_confidence_threshold(tmp_path: Path) -> None:
    run_root = tmp_path / "prototype_gate_h100_v2"
    _write_clip(run_root, "wolverine_mixed_0200_mid_steep_corner")
    checkpoint, metrics = _checkpoint_inputs(tmp_path)

    report = build_court_keypoint_no_tap_eval_report(
        run_root=run_root,
        checkpoint=checkpoint,
        metrics=metrics,
        out=tmp_path / "dry_run_min005.json",
        dry_run=True,
        device="cpu",
        frames_per_clip=3,
        min_confidence=0.05,
    )

    assert report.min_confidence == pytest.approx(0.05)
    assert "--min-confidence" in report.summary.h100_gate_command
    threshold_index = report.summary.h100_gate_command.index("--min-confidence")
    assert report.summary.h100_gate_command[threshold_index + 1] == "0.05"


def test_build_report_includes_img1605_partial_visible_gate(tmp_path: Path) -> None:
    eval_root = Path("eval_clips/ball")
    partial_path = eval_root / "owner_IMG_1605_8a193402780b" / "labels" / "court_keypoints_partial.json"
    proposal_path = Path("runs/img1605_court_detector_hardened/court_corner_proposals.json")
    if not partial_path.is_file() or not proposal_path.is_file():
        pytest.skip("IMG_1605 partial labels and hardened proposal are unavailable")
    checkpoint, metrics = _checkpoint_inputs(tmp_path)

    report = build_court_keypoint_no_tap_eval_report(
        run_root=tmp_path / "empty_full_clip_root",
        checkpoint=checkpoint,
        metrics=metrics,
        out=tmp_path / "img1605_partial_eval.json",
        dry_run=False,
        device="cpu",
        accepted_clips=(),
        eval_root=eval_root,
        include_partial=["owner_IMG_1605_8a193402780b"],
    )

    assert report.status == "ran_not_verified"
    assert report.verified is False
    assert report.not_cal3_verified is True
    assert report.summary.partial_clip_count == 1
    assert report.summary.partial_gate_blocked_clip_count == 1
    partial_clip = report.clips[0]
    assert partial_clip.clip == "owner_IMG_1605_8a193402780b"
    assert partial_clip.partial_label_path == str(partial_path)
    assert partial_clip.partial_visible_gate is not None
    gate = partial_clip.partial_visible_gate
    assert gate.visible_keypoint_count == 14
    assert gate.missing_keypoints == ["near_left_corner"]
    assert gate.gate_passed is False
    assert gate.floor_visible_error_px is not None
    assert gate.floor_visible_error_px.median > 200.0
    assert "visible_floor_median_gt_15" in gate.blockers
    assert "self_verification_not_promotable" in gate.blockers


def test_prediction_validation_sweeps_thresholds_and_blocks_border_biased_low_threshold_points() -> None:
    calibration = _calibration_payload()
    prediction = _prediction_payload(calibration, confidence=0.06, border_biased=True)

    validation = build_court_keypoint_prediction_validation(
        prediction_payload=prediction,
        calibration_payload=calibration,
        thresholds=[0.5, 0.05],
    )

    high, low = validation.thresholds
    assert high.threshold == pytest.approx(0.5)
    assert high.total_confident_keypoints == 0
    assert high.frames_with_solvepnp_min4 == 0
    assert "no_frames_with_solvepnp_min4" in high.blockers

    assert low.threshold == pytest.approx(0.05)
    assert low.frames_with_solvepnp_min4 == 3
    assert low.diagnostic_gate_passed is False
    assert low.calibration_reprojection_error_px is not None
    assert low.calibration_reprojection_error_px.p95 > 300.0
    assert low.border_sanity.near_border_keypoint_ratio == pytest.approx(1.0)
    assert "border_keypoint_ratio_too_high" in low.blockers
    assert validation.best_threshold == pytest.approx(0.05)
    assert validation.best_threshold_gate_passed is False


def test_prediction_validation_passes_stable_calibration_consistent_points() -> None:
    calibration = _calibration_payload()
    prediction = _prediction_payload(calibration, confidence=0.9)

    validation = build_court_keypoint_prediction_validation(
        prediction_payload=prediction,
        calibration_payload=calibration,
        thresholds=[0.5],
    )

    threshold = validation.thresholds[0]
    assert threshold.total_confident_keypoints == 45
    assert threshold.frames_with_solvepnp_min4 == 3
    assert threshold.diagnostic_gate_passed is True
    assert threshold.calibration_reprojection_error_px is not None
    assert threshold.calibration_reprojection_error_px.p95 < 2.0
    assert threshold.temporal_jitter_px is not None
    assert threshold.temporal_jitter_px.p95 < 2.0
    assert threshold.line_corner_consistency.triplet_pass_ratio == pytest.approx(1.0)
    assert threshold.court_zone_sanity.outside_court_ratio == pytest.approx(0.0)
    assert validation.best_threshold_gate_passed is True


def test_dry_run_cli_writes_schema_valid_report(tmp_path: Path) -> None:
    run_root = tmp_path / "prototype_gate_h100_v2"
    _write_clip(run_root, "wolverine_mixed_0200_mid_steep_corner")
    checkpoint, metrics = _checkpoint_inputs(tmp_path)
    out = tmp_path / "court_keypoint_no_tap_eval_dry_run.json"
    markdown_out = tmp_path / "court_keypoint_no_tap_eval_dry_run.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_keypoint_no_tap.py",
            "--run-root",
            str(run_root),
            "--checkpoint",
            str(checkpoint),
            "--metrics",
            str(metrics),
            "--out",
            str(out),
            "--dry-run",
            "--device",
            "cpu",
            "--frames-per-clip",
            "2",
            "--thresholds",
            "0.5,0.05",
            "--markdown-out",
            str(markdown_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    parsed = CourtKeypointNoTapEvalReport.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert parsed.status == "ready_for_h100"
    assert parsed.summary.active_clip_count == 1
    assert parsed.thresholds == [0.5, 0.05]
    markdown = markdown_out.read_text(encoding="utf-8")
    assert "not CAL-3 verified" in markdown
    assert "| threshold |" in markdown


def test_cli_accepts_eval_root_include_partial_for_img1605(tmp_path: Path) -> None:
    partial_path = Path("eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_keypoints_partial.json")
    proposal_path = Path("runs/img1605_court_detector_hardened/court_corner_proposals.json")
    if not partial_path.is_file() or not proposal_path.is_file():
        pytest.skip("IMG_1605 partial labels and hardened proposal are unavailable")
    checkpoint, metrics = _checkpoint_inputs(tmp_path)
    out = tmp_path / "img1605_no_tap_eval.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_keypoint_no_tap.py",
            "--eval-root",
            "eval_clips/ball",
            "--include-partial",
            "owner_IMG_1605_8a193402780b",
            "--checkpoint",
            str(checkpoint),
            "--metrics",
            str(metrics),
            "--out",
            str(out),
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    parsed = CourtKeypointNoTapEvalReport.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert parsed.verified is False
    assert parsed.not_cal3_verified is True
    assert parsed.summary.partial_clip_count == 1
    assert parsed.clips[0].partial_visible_gate is not None
    assert parsed.clips[0].partial_visible_gate.gate_passed is False


def test_cli_accepts_detector_v2_proposal_root_for_img1605(tmp_path: Path) -> None:
    partial_path = Path("eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_keypoints_partial.json")
    detector_v2_root = tmp_path / "detector_v2"
    proposal_dir = detector_v2_root / "owner_IMG_1605_8a193402780b"
    proposal_dir.mkdir(parents=True)
    _write_json(
        proposal_dir / "court_detector_v2_proposals.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_court_detector_v2_proposals",
            "clip": "owner_IMG_1605_8a193402780b",
            "source_frame": "frame_000151.jpg",
            "image_size": [1080, 1920],
            "promoted": False,
            "verified": False,
            "not_cal3_verified": True,
            "promotion_status": "needs_user_input",
            "promotion_blockers": ["self_verification_not_promotable"],
            "selected_hypothesis_id": "hypothesis_0001",
            "hypotheses": [],
            "net_evidence": {},
            "surface_evidence": {},
            "verification": {},
            "needs_user_input": ["near_left_corner"],
        },
    )
    if not partial_path.is_file():
        pytest.skip("IMG_1605 partial labels are unavailable")
    checkpoint, metrics = _checkpoint_inputs(tmp_path)
    out = tmp_path / "img1605_no_tap_eval_detector_v2.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_keypoint_no_tap.py",
            "--eval-root",
            "eval_clips/ball",
            "--include-partial",
            "owner_IMG_1605_8a193402780b",
            "--detector-v2-proposal-root",
            str(detector_v2_root),
            "--checkpoint",
            str(checkpoint),
            "--metrics",
            str(metrics),
            "--out",
            str(out),
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    parsed = CourtKeypointNoTapEvalReport.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert parsed.verified is False
    assert parsed.not_cal3_verified is True
    assert "--detector-v2-proposal-root" in parsed.summary.h100_gate_command


def test_report_schema_rejects_cal3_verified_claim() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "court_keypoint_no_tap_eval_report",
        "status": "ready_for_h100",
        "claim_scope": "dry_run_plumbing",
        "run_root": "runs/eval0/prototype_gate_h100_v2",
        "checkpoint": "runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_heatmap.pt",
        "metrics": None,
        "dry_run": True,
        "device": "cpu",
        "verified": True,
        "not_cal3_verified": False,
        "summary": {
            "active_clip_count": 0,
            "skipped_clip_count": 0,
            "blocked_clip_count": 0,
            "ready_clip_count": 0,
            "ran_clip_count": 0,
            "h100_gate_command": ["python", "scripts/racketsport/evaluate_court_keypoint_no_tap.py"],
        },
        "clips": [],
        "notes": [],
    }

    with pytest.raises(ValueError, match="must not claim CAL-3 verification"):
        CourtKeypointNoTapEvalReport.model_validate(payload)
