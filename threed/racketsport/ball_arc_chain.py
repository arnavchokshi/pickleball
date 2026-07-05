"""Shared BALL arc-chain helpers for process_video and internal runners."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.ball_arc_solver import (
    AnchorEvent,
    BallArcSolverConfig,
    PhysicsParameters,
    solve_ball_arc_track,
)
from threed.racketsport.ball_bounce_candidates import BounceCandidateConfig, write_bounce_candidate_payload
from threed.racketsport.ball_flight_sanity import apply_flight_sanity_demotions, evaluate_ball_flight_sanity
from threed.racketsport.schemas import NetPlane, load_ball_candidates_file
from pydantic import ValidationError


@dataclass(frozen=True)
class BallArcSolverRun:
    artifact: dict[str, Any]
    flight_sanity: dict[str, Any]
    artifact_path: Path
    flight_sanity_path: Path
    events_selected_path: Path | None


FROZEN_ROW22_CHAIN_CONFIGS: dict[str, dict[str, Any]] = {
    "product_veto": {
        "veto_px": 40.0,
        "weak_support_required": True,
    },
    "solver_a_free": {
        "candidate_association_mode": "free",
        "candidate_score_floors": {},
        "candidate_selection_max_iterations": 5,
        "enable_event_discovery": False,
        "enable_event_subset_selection": False,
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
    ball_type: str = "outdoor",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run the default single-primary-track arc chain into ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or utc_stamp()
    bounce_candidates_path = out_dir / "ball_bounce_candidates.json"
    auto_bounces = write_bounce_candidate_payload(
        ball_track_path=ball_track_path,
        calibration_path=court_calibration_path,
        out_path=bounce_candidates_path,
        clip_id=clip,
        config=BounceCandidateConfig(),
    )
    ball_track = read_json_object(ball_track_path, "ball_track")
    calibration = read_json_object(court_calibration_path, "court_calibration")
    candidate_paths = [Path(path) for path in ball_candidate_paths]
    ball_candidate_sidecars = [load_ball_candidates_file(path).model_dump() for path in candidate_paths]
    chain_config_degraded = None if ball_candidate_sidecars else "no_candidate_sidecars"
    contact_windows = read_optional_json_object(contact_windows_path)
    net_plane_for_solve, net_plane_consumed, net_plane_reason = load_net_plane_for_default_solve(net_plane_path)
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
            rally_spans=read_optional_json_object(rally_spans_path),
            physics=PhysicsParameters.for_ball_type(ball_type),
            config=BallArcSolverConfig(candidate_association_mode="free"),
            out_dir=out_dir / "ball_arc_seed",
            generated_at=generated_at,
            write_events_selected=False,
        )
        seed_extra_anchors = _anchors_from_arc_artifact(seed_run.artifact)
    solver_auto_bounces = None if seed_extra_anchors else auto_bounces
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
        extra_anchors=seed_extra_anchors,
        physics=PhysicsParameters.for_ball_type(ball_type),
        config=default_ball_arc_solver_config(),
        out_dir=out_dir,
        generated_at=generated_at,
        write_events_selected=False,
    )
    run.artifact["configs"] = default_ball_chain_configs()
    run.artifact["inputs"] = {
        **dict(run.artifact.get("inputs") if isinstance(run.artifact.get("inputs"), Mapping) else {}),
        "ball_track": str(ball_track_path),
        "court_calibration": str(court_calibration_path),
        "ball_candidates": [str(path) for path in candidate_paths],
    }
    if chain_config_degraded is not None:
        run.artifact["chain_config_degraded"] = chain_config_degraded
    net_plane_provenance = {"consumed_net_plane": net_plane_consumed, "reason": net_plane_reason}
    run.artifact["net_plane_provenance"] = net_plane_provenance
    write_json(run.artifact_path, run.artifact)
    summary = run.artifact.get("summary") if isinstance(run.artifact.get("summary"), Mapping) else {}
    bounce_summary = auto_bounces.get("summary") if isinstance(auto_bounces.get("summary"), Mapping) else {}
    flight_summary = flight_sanity_manifest_summary(run.flight_sanity)
    manifest_path = out_dir / "ball_chain_manifest.json"
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
                **{f"ball_candidates_{idx}": path for idx, path in enumerate(candidate_paths)},
            }
        ),
        "outputs": _hash_entries(
            {
                "auto_bounce_candidates": bounce_candidates_path,
                "seed_anchor_ball_track_arc_solved": seed_run.artifact_path if seed_run is not None else None,
                "seed_anchor_flight_sanity": seed_run.flight_sanity_path if seed_run is not None else None,
                "ball_track_arc_solved": run.artifact_path,
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
        },
        "policy": {
            "outdoor_indoor_labels_read": False,
            "render_only_3d": True,
            "candidate_sidecars_consumed_when_present": True,
            "seed_anchor_prepass": bool(seed_extra_anchors),
            "net_plane_consumed_when_valid": True,
        },
    }
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
    }
    if chain_config_degraded is not None:
        result_summary["chain_config_degraded"] = chain_config_degraded
    return {
        "status": str(run.artifact.get("status") or "unknown"),
        "summary": result_summary,
        "outputs": {
            "ball_bounce_candidates": str(bounce_candidates_path),
            "ball_track_arc_solved": str(run.artifact_path),
            "ball_flight_sanity": str(run.flight_sanity_path),
            "ball_chain_manifest": str(manifest_path),
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
    if write_events_selected and isinstance(gated_artifact.get("event_selection"), Mapping):
        events_selected_path = out_dir / "events_selected.json"
        write_json(events_selected_path, gated_artifact["event_selection"])
    return BallArcSolverRun(
        artifact=gated_artifact,
        flight_sanity=dict(flight_sanity),
        artifact_path=artifact_path,
        flight_sanity_path=flight_sanity_path,
        events_selected_path=events_selected_path,
    )


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
