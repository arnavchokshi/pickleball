"""Shared BALL arc-chain helpers for process_video and internal runners."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.ball_arc_solver import (
    AnchorEvent,
    BallArcSolverConfig,
    PhysicsParameters,
    _court_volume_bounds,
    _integrate_positions,
    solve_ball_arc_track,
)
from threed.racketsport.ball_joint_anchor_search import (
    AnchorSearchRefusal,
    BoundedPlane,
    CameraModel,
    ImageObservation,
    JointAnchorSearchConfig,
    NetConstraint,
    search_joint_anchor_candidates,
)
from threed.racketsport.ball_bounce_candidates import BounceCandidateConfig, write_bounce_candidate_payload
from threed.racketsport.ball_flight_sanity import apply_flight_sanity_demotions, evaluate_ball_flight_sanity
from threed.racketsport.ball_ransac_arc_gate import write_ransac_arc_filtered_ball_track
from threed.racketsport.ball_ukf_fallback import (
    RecoveryPolicyV2Config,
    build_recovery_policy_v2,
    build_ukf_fallback,
)
from threed.racketsport.virtual_world import (
    BALL_ARC_FAIL_CLOSED_POLICY,
    ball_arc_segment_fail_closed_verdicts,
)
from threed.racketsport.court_templates import get_court_template
from threed.racketsport.io_decode import frame_time_lookup
from threed.racketsport.schemas import NetPlane, load_ball_candidates_file
from pydantic import ValidationError


@dataclass(frozen=True)
class BallArcSolverRun:
    artifact: dict[str, Any]
    flight_sanity: dict[str, Any]
    artifact_path: Path
    flight_sanity_path: Path
    events_selected_path: Path | None


BALL_ARC_RENDER_ARTIFACT_TYPE = "racketsport_ball_arc_render"
BALL_ARC_RENDER_SOURCE = "parametric_ball_arc_render_v1"
MPS_TO_MPH = 2.2369362920544
BRIDGE_CONFIDENCE = 0.2


FROZEN_ROW22_CHAIN_CONFIGS: dict[str, dict[str, Any]] = {
    "product_veto": {
        "veto_px": 40.0,
        "weak_support_required": True,
    },
    "solver_a_free": {
        "candidate_association_mode": "free",
        "candidate_score_floors": {},
        "candidate_selection_max_iterations": 5,
        "enable_event_discovery": True,
        "enable_event_subset_selection": True,
        "max_candidates_per_frame": 12,
    },
    "solver_b_rescue_tn05": {
        "candidate_association_mode": "rescue_only",
        "candidate_score_floors": {
            "tracknet": 0.5,
            "wasb": 0.0,
        },
        "candidate_selection_max_iterations": 5,
        "enable_event_discovery": False,
        "enable_event_subset_selection": False,
        "max_candidates_per_frame": 12,
    },
}


def default_ball_chain_configs() -> dict[str, dict[str, Any]]:
    """Return the frozen row-22 BALL chain config block used by defaults."""

    return json.loads(json.dumps(FROZEN_ROW22_CHAIN_CONFIGS))


def default_ball_arc_solver_config() -> BallArcSolverConfig:
    """Default in-pipeline solver-A config from the frozen row-22 chain."""

    solver_a = FROZEN_ROW22_CHAIN_CONFIGS["solver_a_free"]
    return BallArcSolverConfig(
        enable_event_subset_selection=bool(solver_a["enable_event_subset_selection"]),
        enable_event_discovery=bool(solver_a["enable_event_discovery"]),
        candidate_selection_max_iterations=int(solver_a["candidate_selection_max_iterations"]),
        max_candidates_per_frame=int(solver_a["max_candidates_per_frame"]),
        candidate_association_mode=str(solver_a["candidate_association_mode"]),
        candidate_score_floors=dict(solver_a["candidate_score_floors"]),
    )


def run_default_ball_arc_chain(
    *,
    clip: str,
    ball_track_path: Path,
    court_calibration_path: Path,
    out_dir: Path,
    ball_candidate_paths: Sequence[Path] = (),
    contact_windows_path: Path | None = None,
    skeleton3d_path: Path | None = None,
    net_plane_path: Path | None = None,
    rally_spans_path: Path | None = None,
    frame_times_path: Path | None = None,
    ball_type: str = "outdoor",
    enable_ransac_arc_gate: bool = False,
    ransac_max_residual_px: float = 5.0,
    ransac_min_fit_points: int = 5,
    ransac_max_gap_frames: int = 6,
    enable_ukf_fallback: bool = False,
    enable_recovery_policy_v2: bool = False,
    recovery_v2_enable_two_sided_bridge: bool = True,
    recovery_v2_enable_covariance_one_sided: bool = True,
    recovery_v2_enable_low_confidence_2d_updates: bool = False,
    enable_joint_anchor_search: bool = False,
    enable_joint_anchor_pinning: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run the default single-primary-track arc chain into ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or utc_stamp()
    if enable_joint_anchor_pinning and not enable_joint_anchor_search:
        raise ValueError("joint-anchor pinning requires enable_joint_anchor_search=true")
    if enable_joint_anchor_search and enable_ransac_arc_gate:
        raise ValueError("joint-anchor search cannot be combined with the rejected RANSAC arm")
    if enable_ukf_fallback and enable_recovery_policy_v2:
        raise ValueError("conservative UKF v1 and recovery policy v2 are separate candidate arms")
    effective_ball_track_path = ball_track_path
    ransac_summary: dict[str, Any] | None = None
    ransac_track_path: Path | None = None
    ransac_summary_path: Path | None = None
    if enable_ransac_arc_gate:
        ransac_track_path = out_dir / "ball_track_ransac_candidate.json"
        ransac_summary_path = out_dir / "ball_ransac_arc_gate_summary.json"
        ransac_summary = write_ransac_arc_filtered_ball_track(
            ball_track_path=ball_track_path,
            out_path=ransac_track_path,
            summary_path=ransac_summary_path,
            max_residual_px=ransac_max_residual_px,
            min_fit_points=ransac_min_fit_points,
            max_gap_frames=ransac_max_gap_frames,
        )
        effective_ball_track_path = ransac_track_path
    bounce_candidates_path = out_dir / "ball_bounce_candidates.json"
    frame_times = read_optional_json_object(frame_times_path)
    auto_bounces = write_bounce_candidate_payload(
        ball_track_path=effective_ball_track_path,
        calibration_path=court_calibration_path,
        out_path=bounce_candidates_path,
        clip_id=clip,
        config=BounceCandidateConfig(),
        frame_times=frame_times,
    )
    ball_track = read_json_object(effective_ball_track_path, "ball_track")
    calibration = read_json_object(court_calibration_path, "court_calibration")
    candidate_paths = [Path(path) for path in ball_candidate_paths]
    ball_candidate_sidecars = [load_ball_candidates_file(path).model_dump() for path in candidate_paths]
    chain_config_degraded = None if ball_candidate_sidecars else "no_candidate_sidecars"
    contact_windows = read_optional_json_object(contact_windows_path)
    net_plane_for_solve, net_plane_consumed, net_plane_reason = load_net_plane_for_default_solve(net_plane_path)
    rally_spans = read_optional_json_object(rally_spans_path)
    seed_run: BallArcSolverRun | None = None
    seed_extra_anchors: list[AnchorEvent] = []
    if contact_windows is not None:
        seed_run = solve_arc_with_flight_sanity(
            clip=clip,
            ball_track=ball_track,
            calibration=calibration,
            auto_bounce_candidates=auto_bounces,
            contact_windows=contact_windows,
            skeleton3d=read_optional_json_object(skeleton3d_path),
            net_plane=read_optional_json_object(net_plane_path),
            rally_spans=rally_spans,
            frame_times=frame_times,
            physics=PhysicsParameters.for_ball_type(ball_type),
            config=default_ball_arc_solver_config(),
            out_dir=out_dir / "ball_arc_seed",
            generated_at=generated_at,
            write_events_selected=False,
        )
        seed_extra_anchors = _anchors_from_arc_artifact(seed_run.artifact)
    solver_auto_bounces = None if seed_extra_anchors else auto_bounces
    solver_config = replace(
        default_ball_arc_solver_config(),
        enable_both_ends_pinning_inlier_pass=bool(enable_joint_anchor_pinning),
    )
    joint_anchor_path: Path | None = None
    joint_anchor_baseline_run: BallArcSolverRun | None = None
    joint_anchor_payload: dict[str, Any] | None = None
    joint_anchor_extra_anchors: list[AnchorEvent] = []
    if enable_joint_anchor_search:
        joint_anchor_baseline_run = solve_arc_with_flight_sanity(
            clip=clip,
            ball_track=ball_track,
            calibration=calibration,
            auto_bounce_candidates=solver_auto_bounces,
            ball_candidate_sidecars=ball_candidate_sidecars,
            contact_windows=None,
            skeleton3d=None,
            net_plane=net_plane_for_solve,
            rally_spans=None,
            frame_times=frame_times,
            extra_anchors=seed_extra_anchors,
            physics=PhysicsParameters.for_ball_type(ball_type),
            config=default_ball_arc_solver_config(),
            out_dir=out_dir / "tt3d_baseline_seed",
            generated_at=generated_at,
            write_events_selected=False,
        )
        joint_anchor_payload, joint_anchor_extra_anchors = _build_joint_anchor_candidate_sidecar(
            clip=clip,
            ball_track=ball_track,
            calibration=calibration,
            frame_times=frame_times,
            net_plane=net_plane_for_solve,
            baseline_artifact=joint_anchor_baseline_run.artifact,
            source_ball_track_path=ball_track_path,
            solver_config=solver_config,
            generated_at=generated_at,
        )
        joint_anchor_path = out_dir / "ball_joint_anchor_candidates.json"

    run = solve_arc_with_flight_sanity(
        clip=clip,
        ball_track=ball_track,
        calibration=calibration,
        auto_bounce_candidates=solver_auto_bounces,
        ball_candidate_sidecars=ball_candidate_sidecars,
        contact_windows=None,
        skeleton3d=None,
        net_plane=net_plane_for_solve,
        rally_spans=None,
        frame_times=frame_times,
        extra_anchors=[*seed_extra_anchors, *joint_anchor_extra_anchors],
        physics=PhysicsParameters.for_ball_type(ball_type),
        config=solver_config,
        out_dir=out_dir,
        generated_at=generated_at,
        write_events_selected=False,
    )
    run.artifact["configs"] = default_ball_chain_configs()
    run.artifact["inputs"] = {
        **dict(run.artifact.get("inputs") if isinstance(run.artifact.get("inputs"), Mapping) else {}),
        "ball_track": str(effective_ball_track_path),
        "court_calibration": str(court_calibration_path),
        "ball_candidates": [str(path) for path in candidate_paths],
        "frame_times": str(frame_times_path or ""),
    }
    if ransac_track_path is not None:
        run.artifact["inputs"]["source_ball_track"] = str(ball_track_path)
    if chain_config_degraded is not None:
        run.artifact["chain_config_degraded"] = chain_config_degraded
    net_plane_provenance = {"consumed_net_plane": net_plane_consumed, "reason": net_plane_reason}
    run.artifact["net_plane_provenance"] = net_plane_provenance
    if joint_anchor_payload is not None and joint_anchor_path is not None:
        _finalize_joint_anchor_candidate_sidecar(joint_anchor_payload, run.artifact)
        write_json(joint_anchor_path, joint_anchor_payload)
        run.artifact["joint_anchor_candidate"] = {
            "enabled": True,
            "pinning_enabled": bool(enable_joint_anchor_pinning),
            "sidecar": joint_anchor_path.name,
            "chosen_anchor_count": int(joint_anchor_payload["summary"]["chosen_anchor_count"]),
            "candidate_only": True,
            "marks_measured": False,
        }
    write_json(run.artifact_path, run.artifact)
    ukf_fallback_path: Path | None = None
    ukf_fallback: dict[str, Any] | None = None
    if enable_ukf_fallback:
        ukf_fallback_path = out_dir / "ball_ukf_fallback.json"
        ukf_fallback = build_ukf_fallback(
            run.artifact,
            contact_proposals=contact_windows,
            net_plane=net_plane_for_solve,
            generated_at=generated_at,
        )
        write_json(ukf_fallback_path, ukf_fallback)
    recovery_v2_path: Path | None = None
    recovery_v2: dict[str, Any] | None = None
    if enable_recovery_policy_v2:
        recovery_v2_path = out_dir / "ball_recovery_policy_v2.json"
        recovery_v2 = build_recovery_policy_v2(
            run.artifact,
            calibration=calibration,
            contact_proposals=contact_windows,
            net_plane=net_plane_for_solve,
            config=RecoveryPolicyV2Config(
                enable_two_sided_bridge=recovery_v2_enable_two_sided_bridge,
                enable_covariance_one_sided=recovery_v2_enable_covariance_one_sided,
                enable_low_confidence_2d_updates=recovery_v2_enable_low_confidence_2d_updates,
            ),
            generated_at=generated_at,
        )
        write_json(recovery_v2_path, recovery_v2)
    ball_arc_render = build_ball_arc_render_artifact(
        run.artifact,
        flight_sanity=run.flight_sanity,
        rally_spans=rally_spans,
        generated_at=generated_at,
        source_artifact=run.artifact_path.name,
    )
    ball_arc_render_path = out_dir / "ball_arc_render.json"
    write_json(ball_arc_render_path, ball_arc_render)
    if joint_anchor_payload is not None and joint_anchor_path is not None:
        _attach_joint_anchor_fail_closed_audit(joint_anchor_payload, run.artifact, ball_arc_render)
        write_json(joint_anchor_path, joint_anchor_payload)
    summary = run.artifact.get("summary") if isinstance(run.artifact.get("summary"), Mapping) else {}
    render_summary = ball_arc_render.get("summary") if isinstance(ball_arc_render.get("summary"), Mapping) else {}
    bounce_summary = auto_bounces.get("summary") if isinstance(auto_bounces.get("summary"), Mapping) else {}
    flight_summary = flight_sanity_manifest_summary(run.flight_sanity)
    manifest_path = out_dir / "ball_chain_manifest.json"
    events_selected_path = _write_events_selected_from_arc_solution(
        run.artifact,
        out_dir=out_dir,
        generated_at=generated_at,
        writer="run_default_ball_arc_chain",
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_chain_run_manifest",
        "generated_at": generated_at,
        "clip": clip,
        "heldout_authorized": False,
        "configs": default_ball_chain_configs(),
        "inputs": _hash_entries(
            {
                "ball_track": ball_track_path,
                "court_calibration": court_calibration_path,
                **({"net_plane": net_plane_path} if net_plane_path is not None else {}),
                **({"frame_times": frame_times_path} if frame_times_path is not None else {}),
                **{f"ball_candidates_{idx}": path for idx, path in enumerate(candidate_paths)},
            }
        ),
        "outputs": _hash_entries(
            {
                "auto_bounce_candidates": bounce_candidates_path,
                "seed_anchor_ball_track_arc_solved": seed_run.artifact_path if seed_run is not None else None,
                "seed_anchor_flight_sanity": seed_run.flight_sanity_path if seed_run is not None else None,
                "ball_track_arc_solved": run.artifact_path,
                "ball_track_ransac_candidate": ransac_track_path,
                "ball_ransac_arc_gate_summary": ransac_summary_path,
                "ball_ukf_fallback": ukf_fallback_path,
                "ball_recovery_policy_v2": recovery_v2_path,
                "joint_anchor_candidates": joint_anchor_path,
                "joint_anchor_baseline_ball_track_arc_solved": (
                    joint_anchor_baseline_run.artifact_path if joint_anchor_baseline_run is not None else None
                ),
                "joint_anchor_baseline_flight_sanity": (
                    joint_anchor_baseline_run.flight_sanity_path if joint_anchor_baseline_run is not None else None
                ),
                "events_selected": events_selected_path,
                "ball_arc_render": ball_arc_render_path,
                "ball_flight_sanity": run.flight_sanity_path,
            }
        ),
        "summary": {
            "auto_bounce_candidate_count": int(bounce_summary.get("final_candidate_count") or 0),
            "solver_status": str(run.artifact.get("status") or "unknown"),
            "coverage_world_xyz_count": int(summary.get("coverage_world_xyz_count") or 0),
            "segment_count": int(summary.get("segment_count") or 0),
            "seed_anchor_count": len(seed_extra_anchors),
            "net_plane_provenance": net_plane_provenance,
            "ball_arc_render_sample_count": int(render_summary.get("sample_count") or 0),
            "ball_arc_render_bridge_sample_count": int(render_summary.get("bridge_sample_count") or 0),
        },
        "policy": {
            "outdoor_indoor_labels_read": False,
            "render_only_3d": True,
            "ball_arc_render_only": True,
            "candidate_sidecars_consumed_when_present": True,
            "seed_anchor_prepass": bool(seed_extra_anchors),
            "net_plane_consumed_when_valid": True,
        },
    }
    if enable_ransac_arc_gate or enable_ukf_fallback or enable_recovery_policy_v2 or enable_joint_anchor_search:
        manifest["candidate_flags"] = {
            "ransac_arc_gate": bool(enable_ransac_arc_gate),
            "ukf_fallback": bool(enable_ukf_fallback),
        }
        if enable_recovery_policy_v2:
            manifest["candidate_flags"]["recovery_policy_v2"] = True
        if enable_joint_anchor_search:
            manifest["candidate_flags"].update(
                {
                    "joint_anchor_search": True,
                    "joint_anchor_pinning": bool(enable_joint_anchor_pinning),
                }
            )
        manifest["policy"]["candidate_flags_default_off"] = True
    if enable_ransac_arc_gate:
        manifest["summary"]["ransac_rejected_outlier_count"] = int(
            (ransac_summary or {}).get("rejected_ransac_outlier_count") or 0
        )
        manifest["policy"]["ransac_inlier_gate_separate_from_segmentation"] = True
    if enable_ukf_fallback:
        manifest["summary"]["ukf_recovered_sample_count"] = int(
            ((ukf_fallback or {}).get("summary") or {}).get("recovered_sample_count") or 0
        )
        manifest["policy"]["ukf_predictions_never_measured"] = True
    if enable_recovery_policy_v2:
        manifest["summary"]["recovery_v2_recovered_sample_count"] = int(
            ((recovery_v2 or {}).get("summary") or {}).get("recovered_sample_count") or 0
        )
        manifest["policy"].update(
            {
                "recovery_v2_predictions_never_measured": True,
                "recovery_v2_two_sided_bridge": bool(recovery_v2_enable_two_sided_bridge),
                "recovery_v2_covariance_one_sided": bool(recovery_v2_enable_covariance_one_sided),
                "recovery_v2_low_confidence_2d_updates": bool(
                    recovery_v2_enable_low_confidence_2d_updates
                ),
            }
        )
    if joint_anchor_payload is not None:
        manifest["summary"]["tt3d_baseline_fallback_segment_count"] = int(
            joint_anchor_payload["summary"]["baseline_fallback_segment_count"]
        )
        manifest["summary"]["tt3d_candidate_fallback_segment_count"] = _fallback_segment_count(run.artifact)
        manifest["summary"]["tt3d_chosen_anchor_count"] = int(
            joint_anchor_payload["summary"]["chosen_anchor_count"]
        )
        manifest["policy"].update(
            {
                "joint_anchor_candidates_default_off": True,
                "joint_anchor_hypotheses_never_measured": True,
                "joint_anchor_raw_observations_immutable": True,
                "joint_anchor_samples_use_existing_fail_closed_gates": True,
                "joint_anchor_pinning_separately_killable": True,
            }
        )
    segment_budget_exceeded_count = int(summary.get("segment_budget_exceeded_count") or 0)
    segment_budget_exceeded_ids: list[int] = []
    if segment_budget_exceeded_count:
        segment_budget_exceeded_ids = [
            int(segment_id)
            for segment_id in summary.get("segment_budget_exceeded_ids", [])
        ]
        manifest["summary"].update(
            {
                "degraded_segment_count": segment_budget_exceeded_count,
                "missing_segment_count": segment_budget_exceeded_count,
                "segment_budget_exceeded_count": segment_budget_exceeded_count,
                "segment_budget_exceeded_ids": segment_budget_exceeded_ids,
                "missing_segment_reasons": {"segment_budget_exceeded": segment_budget_exceeded_count},
            }
        )
        manifest["degraded_reasons"] = [
            {
                "reason": "segment_budget_exceeded",
                "evidence_provenance": "missing",
                "segment_ids": segment_budget_exceeded_ids,
            }
        ]
    manifest["net_plane_provenance"] = net_plane_provenance
    if chain_config_degraded is not None:
        manifest["chain_config_degraded"] = chain_config_degraded
        manifest["summary"]["chain_config_degraded"] = chain_config_degraded
    write_json(manifest_path, manifest)
    result_summary: dict[str, Any] = {
        "auto_bounce_candidate_count": int(bounce_summary.get("final_candidate_count") or 0),
        "coverage_world_xyz_count": int(summary.get("coverage_world_xyz_count") or 0),
        "segment_count": int(summary.get("segment_count") or 0),
        "flight_sanity_demoted_frame_count": flight_summary["demoted_frame_count"],
        "flight_sanity_failed_segment_count": flight_summary["failed_segment_count"],
        "seed_anchor_count": len(seed_extra_anchors),
        "net_plane_provenance": net_plane_provenance,
        "ball_arc_render_sample_count": int(render_summary.get("sample_count") or 0),
        "ball_arc_render_bridge_sample_count": int(render_summary.get("bridge_sample_count") or 0),
    }
    if enable_ransac_arc_gate:
        result_summary["ransac_rejected_outlier_count"] = int(
            (ransac_summary or {}).get("rejected_ransac_outlier_count") or 0
        )
    if enable_ukf_fallback:
        result_summary["ukf_recovered_sample_count"] = int(
            ((ukf_fallback or {}).get("summary") or {}).get("recovered_sample_count") or 0
        )
    if enable_recovery_policy_v2:
        result_summary["recovery_v2_recovered_sample_count"] = int(
            ((recovery_v2 or {}).get("summary") or {}).get("recovered_sample_count") or 0
        )
    if joint_anchor_payload is not None:
        result_summary["tt3d_baseline_fallback_segment_count"] = int(
            joint_anchor_payload["summary"]["baseline_fallback_segment_count"]
        )
        result_summary["tt3d_candidate_fallback_segment_count"] = _fallback_segment_count(run.artifact)
        result_summary["tt3d_chosen_anchor_count"] = int(joint_anchor_payload["summary"]["chosen_anchor_count"])
    if segment_budget_exceeded_count:
        result_summary.update(
            {
                "degraded_segment_count": segment_budget_exceeded_count,
                "missing_segment_count": segment_budget_exceeded_count,
                "segment_budget_exceeded_count": segment_budget_exceeded_count,
                "segment_budget_exceeded_ids": segment_budget_exceeded_ids,
                "missing_segment_reasons": {"segment_budget_exceeded": segment_budget_exceeded_count},
            }
        )
    if chain_config_degraded is not None:
        result_summary["chain_config_degraded"] = chain_config_degraded
    output_paths = {
        "ball_bounce_candidates": str(bounce_candidates_path),
        "ball_track_arc_solved": str(run.artifact_path),
        "ball_arc_render": str(ball_arc_render_path),
        "ball_flight_sanity": str(run.flight_sanity_path),
        "ball_chain_manifest": str(manifest_path),
    }
    if ransac_track_path is not None:
        output_paths["ball_track_ransac_candidate"] = str(ransac_track_path)
    if ransac_summary_path is not None:
        output_paths["ball_ransac_arc_gate_summary"] = str(ransac_summary_path)
    if ukf_fallback_path is not None:
        output_paths["ball_ukf_fallback"] = str(ukf_fallback_path)
    if recovery_v2_path is not None:
        output_paths["ball_recovery_policy_v2"] = str(recovery_v2_path)
    if joint_anchor_path is not None:
        output_paths["ball_joint_anchor_candidates"] = str(joint_anchor_path)
    events_selected = manifest["outputs"].get("events_selected")
    if isinstance(events_selected, Mapping) and isinstance(events_selected.get("path"), str):
        output_paths["events_selected"] = str(events_selected["path"])
    return {
        "status": str(run.artifact.get("status") or "unknown"),
        "summary": result_summary,
        "outputs": output_paths,
    }


def _build_joint_anchor_candidate_sidecar(
    *,
    clip: str,
    ball_track: Mapping[str, Any],
    calibration: Mapping[str, Any],
    frame_times: Mapping[str, Any] | None,
    net_plane: Mapping[str, Any] | None,
    baseline_artifact: Mapping[str, Any],
    source_ball_track_path: Path,
    solver_config: BallArcSolverConfig,
    generated_at: str,
) -> tuple[dict[str, Any], list[AnchorEvent]]:
    """Search each baseline fallback span and adapt only rank-1 bounces.

    Every ranked hypothesis remains in the sidecar. Lower-ranked hypotheses are
    explicitly rejected before solver evaluation; rank-1 hypotheses still go
    through the production optional-event selector and unchanged fail-closed
    chain. The source ``ball_track`` mapping and file are never modified.
    """

    source_digest_before = _sha256(source_ball_track_path)
    camera = CameraModel.from_calibration(calibration)
    x_min, x_max, y_min, y_max, _ = _court_volume_bounds(solver_config)
    court_plane = BoundedPlane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        # For z-up the core's deterministic basis is u=-y, v=x. Court bounds
        # are symmetric today, but the reversal is kept explicit.
        u_bounds_m=(-y_max, -y_min),
        v_bounds_m=(x_min, x_max),
    )
    try:
        net = NetConstraint.from_solver_mapping(net_plane) if isinstance(net_plane, Mapping) else None
    except (TypeError, ValueError):
        net = None
    search_config = JointAnchorSearchConfig(
        max_time_seeds=15,
        starts_per_seed=1,
        max_ranked_candidates=5,
        max_nfev=800,
    )
    time_map = frame_time_lookup(frame_times if frame_times is not None else ball_track.get("frame_times"))
    frames = ball_track.get("frames")
    raw_frames = list(frames) if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)) else []
    fps = _artifact_fps(ball_track)
    fallback_segments = [
        segment
        for segment in baseline_artifact.get("segments", [])
        if isinstance(segment, Mapping) and str(segment.get("status") or "") == "fit_bvp_fallback"
    ]
    windows: list[dict[str, Any]] = []
    proposed_anchors: list[AnchorEvent] = []
    for ordinal, segment in enumerate(fallback_segments):
        segment_id = int(segment.get("segment_id") or ordinal)
        frame_start = int(segment.get("frame_start") or 0)
        frame_end = int(segment.get("frame_end") or frame_start)
        window_id = f"tt3d_window_{segment_id:03d}_{frame_start:06d}_{frame_end:06d}"
        observations: list[ImageObservation] = []
        for frame_index in range(max(0, frame_start), min(len(raw_frames) - 1, frame_end) + 1):
            frame = raw_frames[frame_index]
            if not isinstance(frame, Mapping) or frame.get("visible") is not True:
                continue
            xy = frame.get("xy")
            if not isinstance(xy, Sequence) or isinstance(xy, (str, bytes)) or len(xy) < 2:
                continue
            try:
                u, v = float(xy[0]), float(xy[1])
                confidence = float(frame.get("conf", 1.0))
                pts_s = float(time_map.get(frame_index, frame.get("t", frame_index / max(fps, 1e-9))))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (u, v, confidence, pts_s)) or confidence <= 0.0:
                continue
            observations.append(
                ImageObservation(
                    frame=frame_index,
                    pts_s=pts_s,
                    u=u,
                    v=v,
                    confidence=min(1.0, confidence),
                )
            )
        window: dict[str, Any] = {
            "window_id": window_id,
            "baseline_segment_id": segment_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "baseline_status": "fit_bvp_fallback",
            "observation_count": len(observations),
            "hypotheses": [],
        }
        try:
            result = search_joint_anchor_candidates(
                observations,
                camera,
                court_plane,
                net=net,
                config=search_config,
                seed=segment_id,
            )
        except AnchorSearchRefusal as exc:
            window["status"] = "refused"
            window["refusal"] = exc.to_json()
            windows.append(window)
            continue
        window["status"] = "candidates_only"
        window["search"] = {
            key: value
            for key, value in result.to_json().items()
            if key != "candidates"
        }
        for candidate in result.candidates:
            candidate_payload = candidate.to_json()
            unique_candidate_id = f"{window_id}_{candidate.candidate_id}"
            candidate_payload["candidate_id"] = unique_candidate_id
            for anchor_payload in candidate_payload["anchor_events"]:
                anchor_payload["anchor_id"] = f"{window_id}_{anchor_payload['anchor_id']}"
                details = dict(anchor_payload.get("details") or {})
                details.update(
                    {
                        "window_id": window_id,
                        "candidate_id": unique_candidate_id,
                        "candidate_rank": candidate.rank,
                        "measured": False,
                    }
                )
                anchor_payload["details"] = details
            integration = {
                "submitted_to_solver": candidate.rank == 1,
                "disposition": "proposed_rank_1" if candidate.rank == 1 else "rejected_lower_rank",
                "reason": "lowest_joint_search_cost" if candidate.rank == 1 else "not_lowest_joint_search_cost",
            }
            window["hypotheses"].append({"candidate": candidate_payload, "integration": integration})
            if candidate.rank == 1:
                bounce_payload = candidate_payload["anchor_events"][1]
                proposed_anchors.append(_anchor_event_from_candidate_payload(bounce_payload))
        windows.append(window)

    source_digest_after = _sha256(source_ball_track_path)
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_joint_anchor_integration_candidates",
        "generated_at": generated_at,
        "clip_id": clip,
        "status": "candidates_only",
        "candidate_flag_default": False,
        "hypothesis_only": True,
        "search_method": "tt3d_pattern_piecewise_ballistic_huber_v1",
        "inputs": {
            "ball_track": str(source_ball_track_path),
            "raw_observation_sha256_before": source_digest_before,
            "raw_observation_sha256_after": source_digest_after,
            "raw_observations_byte_identical": source_digest_before == source_digest_after,
        },
        "config": {
            "window_policy": "each_existing_fit_bvp_fallback_segment",
            "candidate_policy": "rank_1_bounce_to_existing_optional_event_selector",
            "search": next(
                (
                    dict(item["search"]["config"])
                    for item in windows
                    if isinstance(item.get("search"), Mapping) and isinstance(item["search"].get("config"), Mapping)
                ),
                {},
            ),
            "both_ends_pinning_inlier_pass": bool(solver_config.enable_both_ends_pinning_inlier_pass),
        },
        "windows": windows,
        "summary": {
            "baseline_fallback_segment_count": len(fallback_segments),
            "searched_window_count": sum(1 for item in windows if item["status"] == "candidates_only"),
            "refused_window_count": sum(1 for item in windows if item["status"] == "refused"),
            "hypothesis_count": sum(len(item["hypotheses"]) for item in windows),
            "submitted_anchor_count": len(proposed_anchors),
            "chosen_anchor_count": 0,
            "solver_rejected_anchor_count": 0,
        },
        "policy": {
            "candidate_only": True,
            "marks_measured": False,
            "alters_defaults": False,
            "raw_observations_immutable": True,
            "existing_event_selector_required": True,
            "existing_flight_sanity_required": True,
            "existing_rev11_fail_closed_required": True,
            "protected_labels_read": False,
        },
    }
    return payload, proposed_anchors


def _anchor_event_from_candidate_payload(payload: Mapping[str, Any]) -> AnchorEvent:
    details = dict(payload.get("details") or {})
    if details.get("measured") is not False or payload.get("status") != "candidate_hypothesis":
        raise ValueError("joint-anchor hypotheses must be explicitly unmeasured candidate_hypothesis events")
    xyz = _required_vec3(payload.get("world_xyz"), "candidate.world_xyz")
    return AnchorEvent(
        anchor_id=str(payload["anchor_id"]),
        kind=str(payload["kind"]),
        t=float(payload["t"]),
        frame=int(payload["frame"]),
        world_xyz=xyz,
        sigma_m=float(payload["sigma_m"]),
        status="candidate_hypothesis",
        immovable=False,
        source=str(payload.get("source") or "tt3d_pattern_piecewise_ballistic_huber_v1"),
        details=details,
    )


def _finalize_joint_anchor_candidate_sidecar(
    payload: dict[str, Any],
    artifact: Mapping[str, Any],
) -> None:
    event_selection = artifact.get("event_selection")
    selected_rows = event_selection.get("selected", []) if isinstance(event_selection, Mapping) else []
    rejected_rows = event_selection.get("rejected", []) if isinstance(event_selection, Mapping) else []
    selected = {
        str(item.get("anchor_id")): item
        for item in selected_rows
        if isinstance(item, Mapping) and item.get("anchor_id")
    }
    rejected = {
        str(item.get("anchor_id")): item
        for item in rejected_rows
        if isinstance(item, Mapping) and item.get("anchor_id")
    }
    chosen_count = 0
    rejected_count = 0
    for window in payload["windows"]:
        for hypothesis in window["hypotheses"]:
            integration = hypothesis["integration"]
            if not integration["submitted_to_solver"]:
                continue
            bounce = hypothesis["candidate"]["anchor_events"][1]
            anchor_id = str(bounce["anchor_id"])
            if anchor_id in selected:
                row = selected[anchor_id]
                integration.update(
                    {
                        "disposition": "chosen_by_existing_solver",
                        "reason": str(row.get("selection_reason") or "selected"),
                        "solver_selection": dict(row),
                    }
                )
                chosen_count += 1
            elif anchor_id in rejected:
                row = rejected[anchor_id]
                integration.update(
                    {
                        "disposition": "rejected_by_existing_solver",
                        "reason": str(row.get("selection_reason") or "rejected"),
                        "solver_selection": dict(row),
                    }
                )
                rejected_count += 1
            else:
                integration.update(
                    {
                        "disposition": "rejected_not_returned_by_existing_solver",
                        "reason": "anchor_absent_after_existing_ordering_and_selection",
                    }
                )
                rejected_count += 1
    payload["summary"]["chosen_anchor_count"] = chosen_count
    payload["summary"]["solver_rejected_anchor_count"] = rejected_count
    payload["summary"]["candidate_fallback_segment_count"] = _fallback_segment_count(artifact)


def _attach_joint_anchor_fail_closed_audit(
    payload: dict[str, Any],
    artifact: Mapping[str, Any],
    render_artifact: Mapping[str, Any],
) -> None:
    verdicts = ball_arc_segment_fail_closed_verdicts(
        [segment for segment in artifact.get("segments", []) if isinstance(segment, Mapping)]
    )
    suppressed_ids = set(render_artifact.get("summary", {}).get("fail_closed_suppressed_segment_ids") or [])
    untrusted_ids = {segment_id for segment_id, verdict in verdicts.items() if not bool(verdict.get("trusted"))}
    fail_open_ids = sorted(untrusted_ids - suppressed_ids)
    payload["fail_closed_audit"] = {
        "policy": BALL_ARC_FAIL_CLOSED_POLICY,
        "segment_count": len(verdicts),
        "untrusted_segment_ids": sorted(untrusted_ids),
        "suppressed_segment_ids": sorted(suppressed_ids),
        "fail_open_segment_ids": fail_open_ids,
        "fail_open_sample_count": 0 if not fail_open_ids else None,
        "verdicts": {str(key): value for key, value in sorted(verdicts.items())},
    }


def _fallback_segment_count(artifact: Mapping[str, Any]) -> int:
    return sum(
        1
        for segment in artifact.get("segments", [])
        if isinstance(segment, Mapping) and str(segment.get("status") or "") == "fit_bvp_fallback"
    )


def build_ball_arc_render_artifact(
    arc_solved: Mapping[str, Any],
    *,
    flight_sanity: Mapping[str, Any] | None = None,
    rally_spans: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
    source_artifact: str = "ball_track_arc_solved.json",
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Build a render-only dense parametric arc artifact for the replay viewer.

    This is a presentation contract only. It evaluates the already-solved
    segment equations; it does not create detection metrics or alter solver
    acceptance.
    """

    solver_status = str(arc_solved.get("status") or "ran")
    trusted = solver_status == "ran"
    clip_id = str(arc_solved.get("clip_id") or "")
    fps = _artifact_fps(arc_solved)
    frame_times = _artifact_frame_times(arc_solved, fps=fps)
    segment_reports = _flight_sanity_reports(flight_sanity)
    physics = _physics_from_arc_artifact(arc_solved)
    config = _config_from_arc_artifact(arc_solved)
    raw_segments = _renderable_segments(arc_solved) if trusted else []
    # Fail-closed dense-trail emission: the viewer draws these samples as the
    # ball trail independently of world frames, so a segment the world overlay
    # suppresses must not survive here either. Same-artifact segment_id join
    # (never against ball_flight_sanity.json, whose ids are shifted).
    emission_verdicts = (
        ball_arc_segment_fail_closed_verdicts(arc_solved.get("segments")) if fail_closed else {}
    )
    suppressed_segment_ids = sorted(
        segment_id
        for segment_id, verdict in emission_verdicts.items()
        if not verdict["trusted"]
    )
    if suppressed_segment_ids:
        suppressed_set = set(suppressed_segment_ids)
        raw_segments = [
            raw
            for raw in raw_segments
            if not (
                isinstance(raw.get("segment_id"), int)
                and not isinstance(raw.get("segment_id"), bool)
                and raw.get("segment_id") in suppressed_set
            )
        ]

    segments: list[dict[str, Any]] = []
    base_samples: list[dict[str, Any]] = []
    for raw in raw_segments:
        sample_times = _dense_segment_times(raw, frame_times=frame_times, fps=fps)
        evaluated = _evaluate_segment_samples(
            raw,
            sample_times,
            physics=physics,
            config=config,
            confidence=_segment_confidence(raw, segment_reports.get(_segment_id_key(raw))),
            bridge=False,
        )
        if not evaluated:
            continue
        base_samples.extend(evaluated)
        segments.append(
            _render_segment_summary(
                raw,
                samples=evaluated,
                flight_report=segment_reports.get(_segment_id_key(raw)),
            )
        )

    bridge_samples, bridges = _bridge_rally_span_gaps(
        base_samples,
        raw_segments=raw_segments,
        rally_spans=rally_spans,
        frame_times=frame_times,
        fps=fps,
    )
    samples = _dedupe_samples([*base_samples, *bridge_samples])

    return {
        "schema_version": 1,
        "artifact_type": BALL_ARC_RENDER_ARTIFACT_TYPE,
        "clip_id": clip_id,
        "generated_at": generated_at or utc_stamp(),
        "source": BALL_ARC_RENDER_SOURCE,
        "source_artifact": source_artifact,
        "solver_status": solver_status,
        "solver_trusted_for_render": trusted,
        "render_only": True,
        "not_for_detection_metrics": True,
        "trusted_for_ball_detection_metrics": False,
        "policy": {
            "render_only": True,
            "not_for_detection_metrics": True,
            "does_not_feed_detection_metrics": True,
            "does_not_change_solver_status": True,
            "dense_samples_from_parametric_solver": True,
            "bridge_samples_low_confidence": True,
        },
        "segments": segments,
        "bridges": bridges,
        "samples": samples,
        "summary": {
            "segment_count": len(segments),
            "sample_count": len(samples),
            "base_sample_count": len(base_samples),
            "bridge_sample_count": len(bridge_samples),
            "rally_span_count": len(_rally_span_ranges(rally_spans)),
            "supersample_rate": 4,
            "fps": _round_float(fps, 6),
            "solver_trusted_for_render": trusted,
            "fail_closed_enabled": bool(fail_closed),
            "fail_closed_policy": BALL_ARC_FAIL_CLOSED_POLICY,
            "fail_closed_suppressed_segment_ids": suppressed_segment_ids,
        },
    }


def solve_arc_with_flight_sanity(
    *,
    clip: str,
    ball_track: Mapping[str, Any],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    out_dir: Path,
    generated_at: str | None = None,
    auto_bounce_candidates: Mapping[str, Any] | None = None,
    contact_windows: Mapping[str, Any] | None = None,
    skeleton3d: Mapping[str, Any] | None = None,
    net_plane: Mapping[str, Any] | None = None,
    rally_spans: Mapping[str, Any] | None = None,
    frame_times: Mapping[str, Any] | None = None,
    reviewed_bounces: Mapping[str, Any] | None = None,
    ball_sizes: Mapping[str, Any] | None = None,
    ball_candidate_sidecars: Sequence[Mapping[str, Any]] = (),
    candidate_extra_tracks: Mapping[str, Mapping[str, Any]] | None = None,
    extra_anchors: Sequence[AnchorEvent] = (),
    write_events_selected: bool = True,
) -> BallArcSolverRun:
    """Solve arcs and immediately apply the render flight-sanity demotion gate."""

    artifact = solve_ball_arc_track(
        ball_track=ball_track,
        calibration=calibration,
        ball_sizes=ball_sizes,
        ball_candidate_sidecars=ball_candidate_sidecars,
        candidate_extra_tracks=candidate_extra_tracks or {},
        contact_windows=contact_windows,
        skeleton3d=skeleton3d,
        reviewed_bounces=reviewed_bounces,
        auto_bounce_candidates=auto_bounce_candidates,
        rally_spans=rally_spans,
        net_plane=net_plane,
        extra_anchors=extra_anchors,
        frame_times=frame_times,
        physics=physics,
        config=config,
        clip_id=clip,
    )
    artifact["generated_at"] = generated_at or utc_stamp()
    flight_sanity = evaluate_ball_flight_sanity(artifact)
    gated_artifact = apply_flight_sanity_demotions(artifact, flight_sanity)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "ball_track_arc_solved.json"
    flight_sanity_path = out_dir / "ball_flight_sanity.json"
    write_json(artifact_path, gated_artifact)
    write_json(flight_sanity_path, flight_sanity)
    events_selected_path = None
    if write_events_selected:
        events_selected_path = _write_events_selected_from_arc_solution(
            gated_artifact,
            out_dir=out_dir,
            generated_at=generated_at,
            writer="solve_arc_with_flight_sanity",
        )
    return BallArcSolverRun(
        artifact=gated_artifact,
        flight_sanity=dict(flight_sanity),
        artifact_path=artifact_path,
        flight_sanity_path=flight_sanity_path,
        events_selected_path=events_selected_path,
    )


def _write_events_selected_from_arc_solution(
    artifact: Mapping[str, Any],
    *,
    out_dir: Path,
    generated_at: str | None,
    writer: str,
) -> Path | None:
    if str(artifact.get("status") or "") != "ran":
        return None
    event_selection = artifact.get("event_selection")
    if not isinstance(event_selection, Mapping):
        return None
    payload = json.loads(json.dumps(event_selection))
    payload["generated_at"] = generated_at or utc_stamp()
    payload["clip_id"] = str(artifact.get("clip_id") or "")
    payload["source"] = "default_ball_arc_chain_event_selection"
    payload["source_artifact"] = "ball_track_arc_solved.json"
    payload["provenance"] = {
        "derived_from": "ball_track_arc_solved.json",
        "writer": writer,
        "solver_status": str(artifact.get("status") or ""),
        "clip_id": str(artifact.get("clip_id") or ""),
    }
    events_selected_path = out_dir / "events_selected.json"
    write_json(events_selected_path, payload)
    return events_selected_path


def flight_sanity_manifest_summary(report: Mapping[str, Any]) -> dict[str, int]:
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return {
            "segment_count": 0,
            "passed_segment_count": 0,
            "failed_segment_count": 0,
            "skipped_segment_count": 0,
            "demoted_frame_count": 0,
        }
    return {
        "segment_count": int(summary.get("segment_count") or 0),
        "passed_segment_count": int(summary.get("passed_segment_count") or 0),
        "failed_segment_count": int(summary.get("failed_segment_count") or 0),
        "skipped_segment_count": int(summary.get("skipped_segment_count") or 0),
        "demoted_frame_count": int(summary.get("demoted_frame_count") or 0),
    }


def _artifact_fps(artifact: Mapping[str, Any]) -> float:
    explicit = _float_or_none(artifact.get("fps"))
    if explicit is not None and explicit > 0:
        return explicit
    frames = artifact.get("frames")
    if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
        times = [_float_or_none(frame.get("t")) for frame in frames if isinstance(frame, Mapping)]
        valid = [time for time in times if time is not None]
        deltas = [right - left for left, right in zip(valid, valid[1:], strict=False) if right > left]
        if deltas:
            return 1.0 / (sum(deltas) / len(deltas))
    return 30.0


def _artifact_frame_times(artifact: Mapping[str, Any], *, fps: float) -> list[tuple[int, float]]:
    frames = artifact.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return []
    output: list[tuple[int, float]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        t = _float_or_none(frame.get("t"))
        output.append((index, t if t is not None else index / max(fps, 1e-9)))
    return output


def _renderable_segments(artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = artifact.get("segments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [segment for segment in raw if isinstance(segment, Mapping) and str(segment.get("status") or "").startswith("fit")]


def _flight_sanity_reports(report: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    raw = report.get("segments") if isinstance(report, Mapping) else None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return {}
    output: dict[str, Mapping[str, Any]] = {}
    for segment in raw:
        if not isinstance(segment, Mapping):
            continue
        output[_segment_id_key(segment)] = segment
    return output


def _dense_segment_times(
    segment: Mapping[str, Any],
    *,
    frame_times: Sequence[tuple[int, float]],
    fps: float,
) -> list[float]:
    t0 = _required_float(segment.get("t0"), "segment.t0")
    t1 = _required_float(segment.get("t1"), "segment.t1")
    if t1 < t0:
        t0, t1 = t1, t0
    step = 1.0 / max(fps * 4.0, 1e-9)
    values = [t0, t1]
    values.extend(time for _, time in frame_times if t0 - 1e-9 <= time <= t1 + 1e-9)
    count = max(0, int(math.floor((t1 - t0) / step)))
    values.extend(t0 + idx * step for idx in range(count + 1))
    return _sorted_unique_times(values, t0=t0, t1=t1)


def _evaluate_segment_samples(
    segment: Mapping[str, Any],
    times: Sequence[float],
    *,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    confidence: float,
    bridge: bool,
) -> list[dict[str, Any]]:
    p0 = _required_vec3(segment.get("initial_position_m"), "segment.initial_position_m")
    v0 = _required_vec3(segment.get("initial_velocity_mps"), "segment.initial_velocity_mps")
    t0 = _required_float(segment.get("t0"), "segment.t0")
    segment_id = segment.get("segment_id")
    evaluated = _integrate_positions(p0, v0, times, t0=t0, physics=physics, config=config)
    out: list[dict[str, Any]] = []
    frame_start = _float_or_none(segment.get("frame_start"))
    fps = _segment_fps_from_times(segment, times)
    status = str(segment.get("status") or "")
    bounds = _court_volume_bounds(config)
    fallback_bounds = _strict_court_xy_bounds(config) if status == "fit_bvp_fallback" else None
    for t, xyz in zip(times, evaluated, strict=True):
        if not _point_inside_court_volume(xyz, bounds):
            continue
        if fallback_bounds is not None and not _point_inside_strict_court_xy(xyz, fallback_bounds):
            continue
        band = "arc_weak" if bridge or confidence < 0.45 or status in {"fit_weak", "fit_bvp_fallback"} else "arc_interpolated"
        frame_float = frame_start + (t - t0) * fps if frame_start is not None else None
        out.append(
            {
                "t": _round_float(t, 9),
                "frame_float": _round_float(frame_float, 6) if frame_float is not None else None,
                "segment_id": segment_id,
                "world_xyz": _vec_json_local(xyz),
                "court_xy": [_round_float(xyz[0], 6), _round_float(xyz[1], 6)],
                "confidence": _round_float(confidence, 6),
                "band": band,
                "bridge": bool(bridge),
                "render_only": True,
                "not_for_detection_metrics": True,
            }
        )
    return out


def _point_inside_court_volume(
    xyz: tuple[float, float, float],
    bounds: tuple[float, float, float, float, float],
) -> bool:
    x_min, x_max, y_min, y_max, z_min = bounds
    x, y, z = xyz
    return x_min <= x <= x_max and y_min <= y <= y_max and z >= z_min


def _strict_court_xy_bounds(config: BallArcSolverConfig) -> tuple[float, float, float, float]:
    template = get_court_template(config.court_sport)
    return (
        -template.width_m / 2.0,
        template.width_m / 2.0,
        -template.length_m / 2.0,
        template.length_m / 2.0,
    )


def _point_inside_strict_court_xy(
    xyz: tuple[float, float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    x_min, x_max, y_min, y_max = bounds
    x, y = xyz[0], xyz[1]
    return x_min <= x <= x_max and y_min <= y <= y_max


def _render_segment_summary(
    segment: Mapping[str, Any],
    *,
    samples: Sequence[Mapping[str, Any]],
    flight_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("render segment summary requires samples")
    start_sample = samples[0]
    end_sample = samples[-1]
    peak_sample = max(samples, key=lambda sample: float(sample["world_xyz"][2]))  # type: ignore[index]
    anchors = segment.get("anchors_used")
    anchor_items = [anchor for anchor in anchors if isinstance(anchor, Mapping)] if isinstance(anchors, Sequence) and not isinstance(anchors, (str, bytes)) else []
    confidence = _segment_confidence(segment, flight_report)
    return {
        "segment_id": segment.get("segment_id"),
        "t0": _round_float(_required_float(segment.get("t0"), "segment.t0"), 9),
        "t1": _round_float(_required_float(segment.get("t1"), "segment.t1"), 9),
        "frame_start": int(segment.get("frame_start") or 0),
        "frame_end": int(segment.get("frame_end") or segment.get("frame_start") or 0),
        "anchor_types": [str(anchor.get("kind") or "unknown") for anchor in anchor_items],
        "anchor_frames": [int(anchor.get("frame") or 0) for anchor in anchor_items],
        "confidence": _round_float(confidence, 6),
        "flight_sanity_verdict": str(flight_report.get("verdict") if isinstance(flight_report, Mapping) else "not_evaluated"),
        "flight_sanity_reasons": [str(reason) for reason in flight_report.get("reasons", [])] if isinstance(flight_report, Mapping) and isinstance(flight_report.get("reasons"), list) else [],
        "fit_status": str(segment.get("status") or "unknown"),
        "reprojection_rmse_px": _optional_round_float(segment.get("reprojection_rmse_px"), 6),
        "max_reprojection_error_px": _optional_round_float(segment.get("max_reprojection_error_px"), 6),
        "endpoint_error_m": _optional_round_float(segment.get("endpoint_error_m"), 6),
        "net_clearance_m": _optional_round_float(segment.get("net_clearance_m"), 6),
        "net_clearance_ok": segment.get("net_clearance_ok") if isinstance(segment.get("net_clearance_ok"), bool) else None,
        "bridge": False,
        "render_only": True,
        "not_for_detection_metrics": True,
        "shot": _shot_summary(segment, start_sample=start_sample, peak_sample=peak_sample, end_sample=end_sample, samples=samples),
    }


def _shot_summary(
    segment: Mapping[str, Any],
    *,
    start_sample: Mapping[str, Any],
    peak_sample: Mapping[str, Any],
    end_sample: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    speed_mps = _float_or_none(segment.get("initial_speed_mps"))
    if speed_mps is None:
        velocity = _required_vec3(segment.get("initial_velocity_mps"), "segment.initial_velocity_mps")
        speed_mps = math.sqrt(sum(component * component for component in velocity))
    distance_m = _court_distance(start_sample["court_xy"], end_sample["court_xy"])  # type: ignore[arg-type,index]
    path_distance_m = _path_distance(samples)
    net_clearance_m = _optional_round_float(segment.get("net_clearance_m"), 6)
    return {
        "start": _shot_point(start_sample),
        "peak": _shot_point(peak_sample),
        "end": _shot_point(end_sample),
        "speed_mps": _round_float(speed_mps, 6),
        "speed_mph": _round_float(speed_mps * MPS_TO_MPH, 6),
        "height_over_net_m": net_clearance_m,
        "height_over_net_definition": "ball_bottom_clearance_over_net_top",
        "distance_m": _round_float(distance_m, 6),
        "path_distance_m": _round_float(path_distance_m, 6),
        "render_only": True,
        "not_for_detection_metrics": True,
    }


def _bridge_rally_span_gaps(
    base_samples: Sequence[Mapping[str, Any]],
    *,
    raw_segments: Sequence[Mapping[str, Any]],
    rally_spans: Mapping[str, Any] | None,
    frame_times: Sequence[tuple[int, float]],
    fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spans = _rally_span_ranges(rally_spans)
    if not spans or not base_samples:
        return [], []
    base_by_time = {round(float(sample["t"]), 9): sample for sample in base_samples}
    sorted_base = sorted(base_samples, key=lambda sample: float(sample["t"]))
    step = 1.0 / max(fps * 4.0, 1e-9)
    bridge_samples: list[dict[str, Any]] = []
    bridges: list[dict[str, Any]] = []
    segment_ranges = [(_required_float(segment.get("t0"), "segment.t0"), _required_float(segment.get("t1"), "segment.t1")) for segment in raw_segments]

    for span_index, (span_t0, span_t1) in enumerate(spans):
        span_times = set(_sorted_unique_times([span_t0, span_t1], t0=span_t0, t1=span_t1))
        span_times.update(time for _, time in frame_times if span_t0 - 1e-9 <= time <= span_t1 + 1e-9)
        count = max(0, int(math.floor((span_t1 - span_t0) / step)))
        span_times.update(_round_float(span_t0 + idx * step, 9) for idx in range(count + 1))
        active_bridge_id: str | None = None
        active_bridge_times: list[float] = []
        for t in sorted(span_times):
            if any(start - 1e-9 <= t <= end + 1e-9 for start, end in segment_ranges):
                if active_bridge_id is not None and active_bridge_times:
                    bridges.append(_bridge_record(active_bridge_id, active_bridge_times))
                active_bridge_id = None
                active_bridge_times = []
                continue
            if t in base_by_time:
                continue
            left, right = _bracketing_samples(sorted_base, t)
            if left is None and right is None:
                continue
            bridge_id = active_bridge_id or f"bridge_{span_index}_{len(bridges):03d}"
            active_bridge_id = bridge_id
            active_bridge_times.append(t)
            bridge_samples.append(_bridge_sample(bridge_id, t, left=left, right=right, fps=fps))
        if active_bridge_id is not None and active_bridge_times:
            bridges.append(_bridge_record(active_bridge_id, active_bridge_times))
    return bridge_samples, bridges


def _bridge_sample(
    bridge_id: str,
    t: float,
    *,
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    fps: float,
) -> dict[str, Any]:
    if left is not None and right is not None and float(right["t"]) > float(left["t"]):
        alpha = (t - float(left["t"])) / (float(right["t"]) - float(left["t"]))
        left_xyz = _required_vec3(left["world_xyz"], "bridge.left.world_xyz")
        right_xyz = _required_vec3(right["world_xyz"], "bridge.right.world_xyz")
        xyz = tuple(left_xyz[idx] + (right_xyz[idx] - left_xyz[idx]) * alpha for idx in range(3))
        arch = min(0.35, max(0.0, 0.12 * (float(right["t"]) - float(left["t"])))) * 4.0 * alpha * (1.0 - alpha)
        xyz = (xyz[0], xyz[1], max(0.0, xyz[2] + arch))
    else:
        anchor = left if left is not None else right
        xyz = _required_vec3(anchor["world_xyz"], "bridge.anchor.world_xyz") if anchor is not None else (0.0, 0.0, 0.0)
    return {
        "t": _round_float(t, 9),
        "frame_float": _round_float(t * fps, 6),
        "segment_id": bridge_id,
        "world_xyz": _vec_json_local(xyz),
        "court_xy": [_round_float(xyz[0], 6), _round_float(xyz[1], 6)],
        "confidence": BRIDGE_CONFIDENCE,
        "band": "arc_weak",
        "bridge": True,
        "bridge_id": bridge_id,
        "render_only": True,
        "not_for_detection_metrics": True,
    }


def _bridge_record(bridge_id: str, times: Sequence[float]) -> dict[str, Any]:
    return {
        "bridge_id": bridge_id,
        "t0": _round_float(min(times), 9),
        "t1": _round_float(max(times), 9),
        "reason": "rally_span_gap",
        "confidence": BRIDGE_CONFIDENCE,
        "render_only": True,
        "not_for_detection_metrics": True,
    }


def _dedupe_samples(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[float, str], Mapping[str, Any]] = {}
    for sample in samples:
        key = (_round_float(float(sample["t"]), 9), str(sample.get("segment_id")))
        existing = by_key.get(key)
        if existing is None or (existing.get("bridge") is True and sample.get("bridge") is False):
            by_key[key] = sample
    return [dict(sample) for _, sample in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1]))]


def _segment_confidence(segment: Mapping[str, Any], flight_report: Mapping[str, Any] | None) -> float:
    if str(segment.get("status") or "") == "fit_bvp_fallback":
        base = 0.30
    elif str(segment.get("status") or "") == "fit_weak":
        base = 0.38
    else:
        base = 0.92
    rmse = _float_or_none(segment.get("reprojection_rmse_px"))
    if rmse is not None:
        base *= max(0.35, min(1.0, 1.0 - (rmse / 48.0)))
    outliers = _float_or_none(segment.get("outlier_count")) or 0.0
    inliers = _float_or_none(segment.get("inlier_count")) or 0.0
    if inliers + outliers > 0:
        base *= max(0.45, inliers / (inliers + outliers))
    verdict = str(flight_report.get("verdict") if isinstance(flight_report, Mapping) else "not_evaluated")
    if verdict == "fail":
        base *= 0.25
    elif verdict == "not_evaluated":
        base *= 0.70
    physical = segment.get("physical_sanity")
    if isinstance(physical, Mapping) and physical.get("violation") is True:
        base *= 0.50
    return max(0.05, min(0.98, base))


def _physics_from_arc_artifact(artifact: Mapping[str, Any]) -> PhysicsParameters:
    payload = artifact.get("physics_parameters")
    if not isinstance(payload, Mapping):
        return PhysicsParameters()
    return PhysicsParameters(
        ball_type=str(payload.get("ball_type") or "outdoor"),
        gravity_mps2=float(payload.get("gravity_mps2") or PhysicsParameters.gravity_mps2),
        mass_kg=float(payload.get("mass_kg") or PhysicsParameters.mass_kg),
        diameter_m=float(payload.get("diameter_m") or PhysicsParameters.diameter_m),
        rho_air_kg_m3=float(payload.get("rho_air_kg_m3") or PhysicsParameters.rho_air_kg_m3),
        drag_cd=float(payload.get("drag_cd") or 0.0),
    )


def _config_from_arc_artifact(artifact: Mapping[str, Any]) -> BallArcSolverConfig:
    payload = artifact.get("config")
    if not isinstance(payload, Mapping):
        return BallArcSolverConfig()
    allowed = set(BallArcSolverConfig.__dataclass_fields__)
    kwargs = {key: value for key, value in payload.items() if key in allowed}
    return BallArcSolverConfig(**kwargs)


def _rally_span_ranges(rally_spans: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    spans = rally_spans.get("spans") if isinstance(rally_spans, Mapping) else None
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        return []
    parsed: list[tuple[float, float]] = []
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        t0 = _float_or_none(span.get("t0"))
        t1 = _float_or_none(span.get("t1"))
        if t0 is not None and t1 is not None and t1 >= t0:
            parsed.append((t0, t1))
    return parsed


def _bracketing_samples(samples: Sequence[Mapping[str, Any]], t: float) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    left = None
    right = None
    for sample in samples:
        sample_t = float(sample["t"])
        if sample_t <= t:
            left = sample
        if sample_t >= t:
            right = sample
            break
    return left, right


def _shot_point(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "world_xyz": list(sample["world_xyz"]),
        "court_xy": list(sample["court_xy"]),
    }


def _court_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def _path_distance(samples: Sequence[Mapping[str, Any]]) -> float:
    total = 0.0
    for left, right in zip(samples, samples[1:], strict=False):
        left_xyz = _required_vec3(left["world_xyz"], "path.left.world_xyz")
        right_xyz = _required_vec3(right["world_xyz"], "path.right.world_xyz")
        total += math.sqrt(sum((right_xyz[idx] - left_xyz[idx]) ** 2 for idx in range(3)))
    return total


def _segment_fps_from_times(segment: Mapping[str, Any], times: Sequence[float]) -> float:
    frame_start = _float_or_none(segment.get("frame_start"))
    frame_end = _float_or_none(segment.get("frame_end"))
    t0 = _float_or_none(segment.get("t0"))
    t1 = _float_or_none(segment.get("t1"))
    if frame_start is not None and frame_end is not None and t0 is not None and t1 is not None and t1 > t0:
        return (frame_end - frame_start) / (t1 - t0)
    if len(times) >= 2:
        delta = times[1] - times[0]
        if delta > 0:
            return 1.0 / delta
    return 30.0


def _sorted_unique_times(values: Sequence[float], *, t0: float, t1: float) -> list[float]:
    rounded = {
        _round_float(min(max(float(value), t0), t1), 9)
        for value in values
        if math.isfinite(float(value)) and t0 - 1e-9 <= float(value) <= t1 + 1e-9
    }
    return sorted(rounded)


def _segment_id_key(segment: Mapping[str, Any]) -> str:
    return str(segment.get("segment_id"))


def _required_float(value: Any, field: str) -> float:
    parsed = _float_or_none(value)
    if parsed is None:
        raise ValueError(f"{field} must be finite")
    return parsed


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_round_float(value: Any, digits: int) -> float | None:
    parsed = _float_or_none(value)
    return None if parsed is None else _round_float(parsed, digits)


def _round_float(value: float, digits: int) -> float:
    return round(float(value), digits)


def _required_vec3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{field} must be a 3-vector")
    parsed = (_required_float(value[0], field), _required_float(value[1], field), _required_float(value[2], field))
    return parsed


def _vec_json_local(value: Sequence[float]) -> list[float]:
    return [_round_float(float(value[0]), 6), _round_float(float(value[1]), 6), _round_float(float(value[2]), 6)]


def read_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def read_optional_json_object(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return read_json_object(path, path.name)


def load_net_plane_for_default_solve(
    net_plane_path: Path | None,
) -> tuple[dict[str, Any] | None, bool, str]:
    """Load ``net_plane.json`` for the default (main) arc solve, fail-closed.

    Returns ``(net_plane, consumed, reason)``: ``net_plane`` is the raw JSON
    object to pass into :func:`solve_ball_arc_track` when -- and only when --
    it is present and structurally valid per the ``NetPlane`` schema; else
    ``None``. Absence or any structural defect never raises here: it is
    recorded as ``consumed=False`` with a ``reason`` and the caller proceeds
    exactly as it does when no net plane is supplied at all (byte-identical
    default-solve behavior to before net_plane was wired in).
    """

    if net_plane_path is None or not net_plane_path.is_file():
        return None, False, "absent"
    try:
        raw = json.loads(net_plane_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, False, f"invalid_json:{type(exc).__name__}"
    if not isinstance(raw, dict):
        return None, False, "invalid_type:not_object"
    try:
        NetPlane.model_validate(raw)
    except ValidationError as exc:
        locs = sorted(
            {".".join(str(part) for part in error["loc"]) or "root" for error in exc.errors()}
        )
        return None, False, "invalid_schema:" + ",".join(locs)
    return raw, True, "consumed"


def _anchors_from_arc_artifact(payload: Mapping[str, Any]) -> list[AnchorEvent]:
    anchors = payload.get("anchors")
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
        return []
    output: list[AnchorEvent] = []
    for item in anchors:
        if not isinstance(item, Mapping):
            continue
        output.append(
            AnchorEvent(
                anchor_id=str(item["anchor_id"]),
                kind=str(item["kind"]),
                t=float(item["t"]),
                frame=int(item["frame"]),
                world_xyz=tuple(float(value) for value in item["world_xyz"]),  # type: ignore[arg-type]
                sigma_m=float(item["sigma_m"]),
                status=str(item["status"]),
                player_id=item.get("player_id"),
                immovable=bool(item.get("immovable", False)),
                source=item.get("source"),
                details=item.get("details") if isinstance(item.get("details"), Mapping) else None,
            )
        )
    return output


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_entries(paths: Mapping[str, Path | None]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for name, path in sorted(paths.items()):
        if path is None:
            continue
        entries[name] = {"path": str(path), "sha256": _sha256(path) if path.is_file() else None}
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
