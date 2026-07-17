from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from threed.racketsport.orchestrator import (
    ExternalCalibrationPolicyError,
    ExternalCalibrationRunner,
    StageContext,
)
from threed.racketsport.schemas import CourtCalibration, validate_artifact_file
from threed.racketsport.trust_band import derive_court_trust_band


REPO_ROOT = Path(__file__).resolve().parents[2]
BANKED_SOLVED_CALIBRATION = (
    REPO_ROOT
    / "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_calibration_solved.json"
)
BANKED_CODE_IDENTITY = "ac0b14ab0d3a5c00418671f84a725affc54a8213"
PREVIEW_SOURCE = ExternalCalibrationRunner.LINE_EVIDENCE_SOLVED_PREVIEW_SOURCE


def _preview_payload_from_banked_fixture() -> tuple[dict, str, bytes]:
    original_bytes = BANKED_SOLVED_CALIBRATION.read_bytes()
    payload = json.loads(original_bytes)
    solver_method = payload["source"]
    payload["source"] = PREVIEW_SOURCE
    payload["intrinsics"]["source"] = PREVIEW_SOURCE
    payload["provenance"] = {
        "method": solver_method,
        "inputs": [
            "owner_cal_seed/owner_court_corners_verbatim.json",
            "owner_cal_seed/court_corners_seed.json",
            "pbvision_11min_20260713/source_video.mp4@sha256:272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383",
        ],
        "code_identity": BANKED_CODE_IDENTITY,
    }
    return payload, hashlib.sha256(original_bytes).hexdigest(), original_bytes


def _write_payload(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _context(tmp_path: Path, clip: str) -> StageContext:
    inputs_dir = tmp_path / "inputs" / clip
    inputs_dir.mkdir(parents=True)
    return StageContext(
        clip=clip,
        inputs_dir=inputs_dir,
        run_dir=tmp_path / "runs" / clip,
        sport="pickleball",
    )


def _run_preview(tmp_path: Path, payload: dict, *, clip: str = "pbv11_preview"):
    source_path = _write_payload(tmp_path / "external" / f"{clip}.json", payload)
    return ExternalCalibrationRunner(source_path=source_path).run(_context(tmp_path, clip))


def test_real_banked_pbv11_solve_ingests_read_only_and_is_permanently_preview_banded(tmp_path: Path) -> None:
    payload, original_sha256, original_bytes = _preview_payload_from_banked_fixture()

    stage = _run_preview(tmp_path, payload)

    assert stage.status == "ran"
    assert stage.source_mode == "external_line_evidence_solved_preview"
    assert stage.metrics["calibration_source_class"] == PREVIEW_SOURCE
    assert stage.metrics["trust_band"] == "preview"
    assert any("permanently preview-banded" in note for note in stage.notes)

    emitted = validate_artifact_file("court_calibration", tmp_path / "runs/pbv11_preview/court_calibration.json")
    assert isinstance(emitted, CourtCalibration)
    assert emitted.source == PREVIEW_SOURCE
    assert emitted.intrinsics.source == PREVIEW_SOURCE
    assert emitted.trust_band == "preview"
    assert emitted.provenance is not None
    assert emitted.provenance.method == "line_evidence_intersections_15pt_single_view_planar"
    assert emitted.provenance.code_identity == BANKED_CODE_IDENTITY

    assert BANKED_SOLVED_CALIBRATION.read_bytes() == original_bytes
    assert hashlib.sha256(BANKED_SOLVED_CALIBRATION.read_bytes()).hexdigest() == original_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("coordinate_contract"), "requires coordinate_contract declarations"),
        (
            lambda payload: (payload.pop("per_keypoint_residual_px"), payload.pop("reprojection_error_px")),
            "failed strict court-calibration schema validation",
        ),
        (lambda payload: payload.pop("provenance"), "requires full provenance"),
    ],
    ids=("missing-space-and-distortion-state", "missing-residual-diagnostics", "missing-provenance"),
)
def test_preview_source_missing_required_disclosure_refuses_with_typed_error(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload, _, _ = _preview_payload_from_banked_fixture()
    mutation(payload)
    source_path = _write_payload(tmp_path / "external" / "invalid_preview.json", payload)

    with pytest.raises(ExternalCalibrationPolicyError, match=message):
        ExternalCalibrationRunner(source_path=source_path).run(_context(tmp_path, "invalid_preview"))

    assert not (tmp_path / "runs/invalid_preview/court_calibration.json").exists()


@pytest.mark.parametrize(
    "tamper",
    [
        lambda payload: payload["capture_quality"]["reasons"].append("reviewed_15pt_correspondences"),
        lambda payload: payload["intrinsics"].__setitem__("source", "metric_15pt_reviewed"),
    ],
    ids=("reviewed-reason-marker", "mixed-reviewed-intrinsics-source"),
)
def test_preview_source_cannot_sneak_through_reviewed_calibration_checks(tmp_path: Path, tamper) -> None:
    payload, _, _ = _preview_payload_from_banked_fixture()
    tamper(payload)
    source_path = _write_payload(tmp_path / "external" / "sneak.json", payload)

    with pytest.raises(ExternalCalibrationPolicyError):
        ExternalCalibrationRunner(
            source_path=source_path,
            trusted_intrinsics_sources=frozenset({"metric_15pt_reviewed", PREVIEW_SOURCE}),
        ).run(_context(tmp_path, "sneak"))


def test_preview_source_string_does_not_match_reviewed_trust_band_consumer() -> None:
    payload, _, _ = _preview_payload_from_banked_fixture()

    band = derive_court_trust_band(payload, evidence_path="fixture://pbv11-preview")

    assert PREVIEW_SOURCE not in ExternalCalibrationRunner.TRUSTED_INTRINSICS_SOURCES
    assert ExternalCalibrationRunner.PREVIEW_INTRINSICS_SOURCES.isdisjoint(
        ExternalCalibrationRunner.TRUSTED_INTRINSICS_SOURCES
    )
    assert band["gate_status"] != "metric15_unverified"
    assert "metric-15pt reviewed calibration" not in band["reason"]


def test_metric_15pt_reviewed_ingestion_preserves_emitted_bytes_and_stage_metadata(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json"
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    expected_payload = CourtCalibration.model_validate(source_payload).model_dump(mode="json")
    expected_bytes = (json.dumps(expected_payload, indent=2, sort_keys=True) + "\n").encode()

    stage = ExternalCalibrationRunner(source_path=source_path).run(_context(tmp_path, "metric_parity"))

    emitted_path = tmp_path / "runs/metric_parity/court_calibration.json"
    assert emitted_path.read_bytes() == expected_bytes
    assert stage.source_mode == "external_metric_calibration"
    assert stage.metrics == {
        "reprojection_median_px": source_payload["reprojection_error_px"]["median"],
        "reprojection_p95_px": source_payload["reprojection_error_px"]["p95"],
        "intrinsics_source": "metric_15pt_reviewed",
        "intrinsics_dist_nonzero": any(abs(value) > 1e-9 for value in source_payload["intrinsics"]["dist"]),
    }
