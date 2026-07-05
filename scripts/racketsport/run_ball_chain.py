#!/usr/bin/env python3
"""Run the BALL W3.2 dual-artifact chain for one clip.

The runner is internal-val/preregister plumbing. It consumes existing detector
artifacts, writes a render-only 3D arc artifact, writes the 2D product-view
artifact, and records a manifest with input/output hashes. It does not read
reviewed Outdoor/Indoor labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport import solve_ball_arcs as solve_cli  # noqa: E402
from threed.racketsport.ball_arc_solver import (  # noqa: E402
    AnchorEvent,
    BallArcSolverConfig,
    PhysicsParameters,
)
from threed.racketsport.ball_bounce_candidates import BounceCandidateConfig, write_bounce_candidate_payload  # noqa: E402
from threed.racketsport.ball_arc_chain import (  # noqa: E402
    flight_sanity_manifest_summary,
    solve_arc_with_flight_sanity,
)
from threed.racketsport.ball_flight_sanity import (  # noqa: E402
    apply_flight_sanity_demotions,
    apply_product_view_flight_sanity_demotions,
    evaluate_ball_flight_sanity,
)


HELDOUT_CLIP_IDS = {
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
}


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.clip in HELDOUT_CLIP_IDS and not args.heldout_authorized:
        parser.exit(2, f"{parser.prog}: error: {args.clip} is held out; pass --heldout-authorized only for manager-approved held-out runs\n")

    try:
        manifest = run_chain(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps({"manifest": str(manifest["manifest_path"]), "summary": manifest["summary"]}, sort_keys=True))
    return 0


def run_chain(args: argparse.Namespace) -> dict[str, Any]:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = _utc_stamp()
    commands_log = out_dir / "commands.log"
    _append_command(commands_log, " ".join(sys.argv))

    fused_track = _read_json(args.fused_track, "fused_track")
    calibration = _read_json(args.court_calibration, "court_calibration")
    fusion_decisions = _read_optional_json(args.fusion_decisions)
    candidate_sidecars = [solve_cli._read_ball_candidates(path) for path in args.ball_candidates]
    candidate_extra_tracks = {
        name: _read_json(path, f"candidate_extra_track:{name}")
        for name, path in solve_cli._candidate_extra_track_specs(args.candidate_extra_track).items()
    }
    extra_anchors = _anchors_from_arc_artifact(_read_optional_json(args.extra_anchors_from_arc))

    fused_copy = out_dir / "fused" / "ball_track.json"
    court_copy = out_dir / "inputs" / "court_calibration.json"
    _write_json(fused_copy, fused_track)
    _write_json(court_copy, calibration)

    auto_bounces_path = out_dir / "auto_bounce_candidates.json"
    auto_bounces = write_bounce_candidate_payload(
        ball_track_path=fused_copy,
        calibration_path=court_copy,
        out_path=auto_bounces_path,
        clip_id=args.clip,
        config=BounceCandidateConfig(),
    )
    solver_auto_bounces = None if extra_anchors else auto_bounces

    physics = PhysicsParameters.for_ball_type(args.ball_type)
    free_config = BallArcSolverConfig(
        enable_event_subset_selection=False,
        enable_event_discovery=False,
        candidate_selection_max_iterations=args.candidate_selection_max_iterations,
        max_candidates_per_frame=args.max_candidates_per_frame,
        candidate_association_mode="free",
    )
    rescue_config = BallArcSolverConfig(
        enable_event_subset_selection=False,
        enable_event_discovery=False,
        candidate_selection_max_iterations=args.candidate_selection_max_iterations,
        max_candidates_per_frame=args.max_candidates_per_frame,
        candidate_association_mode="rescue_only",
        candidate_score_floors={
            "tracknet": float(args.rescue_tracknet_floor),
            "wasb": float(args.rescue_wasb_floor),
        },
    )

    free_root = out_dir / "solver_a_free"
    rescue_root = out_dir / "solver_b_rescue_tn05"
    free_artifact, free_flight_sanity, free_flight_sanity_path = _coerce_solver_result(
        _run_solver(
            clip=args.clip,
            out_root=free_root,
            generated_at=generated_at,
            ball_track=fused_track,
            calibration=calibration,
            auto_bounces=solver_auto_bounces,
            extra_anchors=extra_anchors,
            candidate_sidecars=candidate_sidecars,
            candidate_extra_tracks=candidate_extra_tracks,
            physics=physics,
            config=free_config,
        ),
        out_root=free_root,
        clip=args.clip,
    )
    rescue_artifact, rescue_flight_sanity, rescue_flight_sanity_path = _coerce_solver_result(
        _run_solver(
            clip=args.clip,
            out_root=rescue_root,
            generated_at=generated_at,
            ball_track=fused_track,
            calibration=calibration,
            auto_bounces=solver_auto_bounces,
            extra_anchors=extra_anchors,
            candidate_sidecars=candidate_sidecars,
            candidate_extra_tracks=candidate_extra_tracks,
            physics=physics,
            config=rescue_config,
        ),
        out_root=rescue_root,
        clip=args.clip,
    )

    base_view, base_report = solve_cli.build_product_ball_track_view(
        arc_solved=rescue_artifact,
        fused_track=fused_track,
        calibration=calibration,
        measured_bands={"anchored_measured"},
        veto_px=None,
        weak_support_required=args.veto_weak_support_required,
        fusion_decisions=fusion_decisions,
    )
    veto_view, veto_report = solve_cli.build_product_ball_track_view(
        arc_solved=rescue_artifact,
        fused_track=fused_track,
        calibration=calibration,
        measured_bands={"anchored_measured"},
        veto_px=args.veto_px,
        weak_support_required=args.veto_weak_support_required,
        fusion_decisions=fusion_decisions,
    )
    base_view = apply_product_view_flight_sanity_demotions(base_view, rescue_flight_sanity)
    veto_view = apply_product_view_flight_sanity_demotions(veto_view, rescue_flight_sanity)
    base_report["flight_sanity"] = flight_sanity_manifest_summary(rescue_flight_sanity)
    veto_report["flight_sanity"] = flight_sanity_manifest_summary(rescue_flight_sanity)
    product_dir = out_dir / "product_view"
    base_view_path = product_dir / "arc_measured_fallback_fused_no_veto.json"
    veto_view_path = product_dir / f"arc_measured_fallback_fused_veto_v{_value_token(args.veto_px)}_{'weak' if args.veto_weak_support_required else 'all'}.json"
    _write_json(base_view_path, base_view)
    _write_json(product_dir / "arc_measured_fallback_fused_no_veto_report.json", base_report)
    _write_json(veto_view_path, veto_view)
    _write_json(product_dir / "arc_measured_fallback_fused_veto_report.json", veto_report)

    solver_a_killed = _solver_killed(free_artifact)
    solver_b_killed = _solver_killed(rescue_artifact)
    solver_a_kill_reasons = _artifact_kill_reasons(free_artifact)
    solver_b_kill_reasons = _artifact_kill_reasons(rescue_artifact)
    product_view_mode = str(veto_report.get("product_view_mode") or "arc_composed")
    manifest_path = out_dir / "ball_chain_manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_chain_run_manifest",
        "generated_at": generated_at,
        "clip": args.clip,
        "heldout_authorized": bool(args.heldout_authorized),
        "killed": bool(solver_a_killed or solver_b_killed),
        "solver_a_killed": solver_a_killed,
        "solver_a_kill_reasons": solver_a_kill_reasons,
        "solver_b_killed": solver_b_killed,
        "solver_b_kill_reasons": solver_b_kill_reasons,
        "product_view_mode": product_view_mode,
        "artifact_consumers": {
            "solver_a_3d": {
                "render_only": True,
                "killed": solver_a_killed,
                "kill_reasons": solver_a_kill_reasons,
            },
            "solver_b_product_view": {
                "killed": solver_b_killed,
                "kill_reasons": solver_b_kill_reasons,
                "mode": product_view_mode,
            },
        },
        "policy": {
            "outdoor_indoor_labels_read": False,
            "render_only_3d": True,
            "product_view_veto_layer_only": True,
            "product_view_degrades_to_fused_only_when_solver_b_killed": True,
            "extra_anchor_artifact_overrides_fresh_auto_bounces": bool(extra_anchors),
        },
        "configs": {
            "solver_a_free": _config_payload(free_config),
            "solver_b_rescue_tn05": _config_payload(rescue_config),
            "product_veto": {
                "veto_px": float(args.veto_px),
                "weak_support_required": bool(args.veto_weak_support_required),
            },
        },
        "inputs": _hash_entries(
            {
                "fused_track": args.fused_track,
                "court_calibration": args.court_calibration,
                "fusion_decisions": args.fusion_decisions,
                "extra_anchors_from_arc": args.extra_anchors_from_arc,
                **{f"ball_candidates_{idx}": path for idx, path in enumerate(args.ball_candidates)},
            }
        ),
        "outputs": _hash_entries(
            {
                "fused_track_copy": fused_copy,
                "auto_bounce_candidates": auto_bounces_path,
                "solver_a_ball_track_arc_solved": out_dir / "solver_a_free" / args.clip / "ball_track_arc_solved.json",
                "solver_a_flight_sanity": free_flight_sanity_path,
                "solver_b_ball_track_arc_solved": out_dir / "solver_b_rescue_tn05" / args.clip / "ball_track_arc_solved.json",
                "solver_b_flight_sanity": rescue_flight_sanity_path,
                "product_base": base_view_path,
                "product_veto": veto_view_path,
            }
        ),
        "summary": {
            "auto_bounce_candidate_count": auto_bounces["summary"]["final_candidate_count"],
            "solver_a_status": free_artifact["status"],
            "solver_b_status": rescue_artifact["status"],
            "solver_a_killed": solver_a_killed,
            "solver_a_kill_reasons": solver_a_kill_reasons,
            "solver_b_killed": solver_b_killed,
            "solver_b_kill_reasons": solver_b_kill_reasons,
            "product_view_mode": product_view_mode,
            "product_base_visible_count": base_report["visible_count"],
            "product_veto_visible_count": veto_report["visible_count"],
            "product_veto_dropped_count": veto_report["veto"]["dropped_count"],
            "flight_sanity": {
                "solver_a": flight_sanity_manifest_summary(free_flight_sanity),
                "solver_b": flight_sanity_manifest_summary(rescue_flight_sanity),
            },
        },
    }
    manifest["manifest_path"] = str(manifest_path)
    _write_json(manifest_path, manifest)
    return manifest


def _run_solver(
    *,
    clip: str,
    out_root: Path,
    generated_at: str,
    ball_track: Mapping[str, Any],
    calibration: Mapping[str, Any],
    auto_bounces: Mapping[str, Any] | None,
    extra_anchors: Sequence[AnchorEvent],
    candidate_sidecars: Sequence[Mapping[str, Any]],
    candidate_extra_tracks: Mapping[str, Mapping[str, Any]],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    clip_dir = out_root / clip
    run = solve_arc_with_flight_sanity(
        clip=clip,
        ball_track=ball_track,
        calibration=calibration,
        ball_candidate_sidecars=candidate_sidecars,
        candidate_extra_tracks=candidate_extra_tracks,
        auto_bounce_candidates=auto_bounces,
        extra_anchors=extra_anchors,
        physics=physics,
        config=config,
        out_dir=clip_dir,
        generated_at=generated_at,
        write_events_selected=True,
    )
    events_path = clip_dir / "events_selected.json"
    report_path = clip_dir / "ball_arc_solver_report.json"
    _write_json(report_path, solve_cli._report_payload(clip, run.artifact, run.artifact_path, events_path))
    _write_text(clip_dir / "REPORT.md", solve_cli._markdown_report(clip, run.artifact, run.artifact_path, report_path))
    return run.artifact, run.flight_sanity, run.flight_sanity_path


def _coerce_solver_result(
    result: tuple[dict[str, Any], dict[str, Any], Path] | Mapping[str, Any],
    *,
    out_root: Path,
    clip: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if isinstance(result, tuple) and len(result) == 3:
        return result
    if not isinstance(result, Mapping):
        raise TypeError(f"_run_solver returned unsupported result type: {type(result).__name__}")
    report = evaluate_ball_flight_sanity(result)
    gated_artifact = apply_flight_sanity_demotions(result, report)
    clip_dir = out_root / clip
    artifact_path = clip_dir / "ball_track_arc_solved.json"
    report_path = clip_dir / "ball_flight_sanity.json"
    _write_json(artifact_path, gated_artifact)
    _write_json(report_path, report)
    return gated_artifact, dict(report), report_path


def _anchors_from_arc_artifact(payload: Mapping[str, Any] | None) -> list[AnchorEvent]:
    if not isinstance(payload, Mapping):
        return []
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--fused-track", type=Path, required=True)
    parser.add_argument("--court-calibration", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ball-candidates", type=Path, action="append", default=[])
    parser.add_argument("--candidate-extra-track", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--extra-anchors-from-arc", type=Path, default=None)
    parser.add_argument("--fusion-decisions", type=Path, default=None)
    parser.add_argument("--ball-type", choices=("outdoor", "indoor", "no_drag_test"), default="outdoor")
    parser.add_argument("--max-candidates-per-frame", type=int, default=12)
    parser.add_argument("--candidate-selection-max-iterations", type=int, default=5)
    parser.add_argument("--rescue-tracknet-floor", type=float, default=0.5)
    parser.add_argument("--rescue-wasb-floor", type=float, default=0.0)
    parser.add_argument("--veto-px", type=float, default=60.0)
    parser.add_argument("--veto-weak-support-required", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heldout-authorized", action="store_true")
    return parser


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _read_json(path, path.name)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_command(path: Path, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(command.rstrip("\n") + "\n")


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


def _config_payload(config: BallArcSolverConfig) -> dict[str, Any]:
    return {
        "candidate_association_mode": config.candidate_association_mode,
        "candidate_score_floors": dict(config.candidate_score_floors or {}),
        "candidate_selection_max_iterations": config.candidate_selection_max_iterations,
        "max_candidates_per_frame": config.max_candidates_per_frame,
        "enable_event_subset_selection": config.enable_event_subset_selection,
        "enable_event_discovery": config.enable_event_discovery,
    }


def _solver_killed(artifact: Mapping[str, Any]) -> bool:
    return str(artifact.get("status") or "") != "ran"


def _artifact_kill_reasons(artifact: Mapping[str, Any]) -> list[str]:
    raw = artifact.get("kill_reasons")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [str(item) for item in raw]


def _value_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
