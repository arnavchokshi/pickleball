"""A `calibration`-stage `StageRunner` that accepts a pre-built, trusted
`court_calibration.json` directly, for footage that cannot go through
`threed.racketsport.orchestrator.ManualCalibrationRunner` (the pipeline's only built-in
calibration path, which unconditionally requires either a `capture_sidecar.json`
manual-taps seed or `court_keypoints.json` -- neither of which exist, or make sense, for
external-ground-truth footage with its own already-trusted real camera calibration; see
`threed.racketsport.external_gt_aspset510_body_inputs`).

This is intentionally narrow: it only accepts a calibration file that is *already*
present in `context.inputs_dir` (produced by
`scripts/racketsport/build_external_gt_aspset510_body_inputs.py` from ASPset-510's own
real, measured per-camera calibration) -- it never invents or estimates calibration
itself. `court_zones.json`/`net_plane.json` are synthesized the same way
`ManualCalibrationRunner` does (`build_court_zones`/`build_net_plane`, pure functions of
`sport` with no real-court dependency; this project's own pickleball calibration path
also treats them as schema-compatibility formalities, not per-clip measurements).
`court_line_evidence.json` is written via the pipeline's own existing "no trusted seed"
fail-closed path (`aggregate_court_line_evidence` with zero observations) so it stays
schema-valid while being *honestly* marked as not a real line-detection result --
identical semantics to what happens today when a real pickleball clip has no calibration
seed at all, just reused for a different (real-calibration, no-video-court) reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .court_line_evidence import aggregate_court_line_evidence, required_court_line_ids, required_court_net_ids
from .court_templates import Sport
from .court_zones import build_court_zones
from .net_plane import build_net_plane
from .orchestrator import StageContext, StageRun
from .schemas import CourtCalibration, validate_artifact_file


def _write_json_artifact(path: Path, artifact: Any) -> None:
    payload = artifact.model_dump(mode="json") if hasattr(artifact, "model_dump") else artifact
    path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fail_closed_court_line_evidence(sport: Sport, *, source: str, reason: str) -> Any:
    evidence = aggregate_court_line_evidence(
        sport=sport,
        line_observations=[],
        net_observations=[],
        required_line_ids=required_court_line_ids(sport),
        required_net_ids=required_court_net_ids(sport),
    )
    evidence.source = source
    if reason not in evidence.aggregate.reasons:
        evidence.aggregate.reasons.append(reason)
    return evidence


class PrecomputedCalibrationRunner:
    """Copies a pre-built, trusted `court_calibration.json` from `inputs_dir`.

    ``source_note`` is recorded in the produced `StageRun.notes` so anyone reading
    `pipeline_run.json` sees exactly why this run did not go through the normal
    manual-taps/ARKit calibration paths.
    """

    stage = "calibration"
    real_model = False
    source_mode = "precomputed_external_gt_calibration"

    def __init__(self, *, source_note: str) -> None:
        self.source_note = source_note

    def run(self, context: StageContext) -> StageRun:
        calibration_path = context.inputs_dir / "court_calibration.json"
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"PrecomputedCalibrationRunner requires a pre-built court_calibration.json at "
                f"{calibration_path}; none found"
            )
        calibration = validate_artifact_file("court_calibration", calibration_path)
        if not isinstance(calibration, CourtCalibration):
            raise ValueError("court_calibration.json did not validate as CourtCalibration")

        net_plane = build_net_plane(context.sport)
        line_evidence = _fail_closed_court_line_evidence(
            context.sport,
            source="external_gt_precomputed_calibration_no_video_court",
            reason="external_ground_truth_footage_has_no_pickleball_court_to_detect",
        )
        artifacts = {
            "court_calibration.json": calibration,
            "court_zones.json": build_court_zones(context.sport),
            "net_plane.json": net_plane,
            "court_line_evidence.json": line_evidence,
        }
        for filename, artifact in artifacts.items():
            _write_json_artifact(context.run_dir / filename, artifact)

        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=tuple(artifacts),
            notes=(
                self.source_note,
                "court_zones.json/net_plane.json are schema-compatibility formalities, not real "
                "measurements of this non-pickleball scene",
                "court_line_evidence.json is deliberately fail-closed (zero observations): there is "
                "no real pickleball court to detect lines/net on in this footage",
            ),
            metrics={
                "reprojection_median_px": calibration.reprojection_error_px.median,
                "reprojection_p95_px": calibration.reprojection_error_px.p95,
            },
        )


__all__ = ["PrecomputedCalibrationRunner"]
