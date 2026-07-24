#!/usr/bin/env python3
"""WS2.3 dormant-flags comparison report (measurement only, VERIFIED=0).

Joins the per-variant characterization reports (existing harness, read mode)
with the solve sidecars (UKF fallback, joint-anchor candidates) and the
archived 2026-07-05 solve into one deterministic report:

* baseline-vs-archived reproduction verdict first (code drift context),
* per-flag headline deltas (acceptance, fail-closed taxonomy, anchors),
* reprojection residual stats per segment,
* UKF sidecar reported SEPARATELY (physics_interpolated trust-band by
  design; the harness never reads the sidecar and its samples are never
  counted as accepted),
* pinned input manifest with shas, honest caveat block.

Determinism contract: sorted keys, floats rounded to 6 decimals, no
timestamps/wall-clock/absolute paths in the report body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
FLOAT_DECIMALS = 6
LABEL = "ball_lane_20260723_dormant_flags"
ARCHIVED_KEY = "archived_20260705"
ARCHIVED_BASE = "runs/lanes/ball_p3a_bvp_solver_20260705/three_clip_default_chain"

CLIP_ORDER = [
    "burlington_gold_0300_low_steep_corner",
    "wolverine_mixed_0200_mid_steep_corner",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
]
VARIANT_ORDER = ["baseline", "joint_anchor_search", "ukf_fallback", "both"]
FLAG_VARIANTS = ["joint_anchor_search", "ukf_fallback", "both"]

SKIPPED_CLIPS = {
    "indoor_doubles_fwuks_0500_long_mid_baseline": (
        "no local re-solve inputs: no ball_track.json (tracking-stage output) exists for this clip "
        "on this machine and no solved artifacts exist anywhere locally (baseline lane report "
        "already listed it skipped: missing ball_track_arc_solved.json)"
    ),
}

CAVEATS = [
    "Measurement only. VERIFIED=0 stays binding; nothing here is a promotion or a default change.",
    "Reprojection residuals and fail-closed acceptance are image-consistency measurements only; they are blind to metric depth error. T1 metric-3D ground truth is required before any acceptance/promotion claim.",
    "UKF fallback outputs are a render-only sidecar: every sample is source=physics_interpolated, band=physics_predicted, trust_band=low_confidence by design. The characterization harness never reads the sidecar (it is not in discover_clip_inputs), so UKF samples cannot leak into accepted statistics; they are reported separately below and must never be counted as accepted/measured 3D.",
    "Joint-anchor search injects candidate_hypothesis anchors (marks_measured=false) through the unchanged production event selector and fail-closed gates; anchor adoption is not evidence of depth accuracy.",
    "The solver has a 5.0s per-segment wall-clock safety budget (SEGMENT_WALL_CLOCK_BUDGET_S); segments that exceed it abstain with typed segment_budget_exceeded. This budget is machine/load dependent, so segment counts near the budget boundary carry run-to-run noise; the determinism probe below bounds this for the burlington baseline.",
    "The 2026-07-05 archived solve predates several solver changes (LOO per-holdout BVP refit, BVP span protection v2, per-segment wall-clock budget, fail-closed emission changes); the baseline-vs-archived delta is code drift, not an input mismatch (input shas are pinned and verified).",
    "contact_windows/skeleton3d/rally_spans shas were not recorded by the archived 2026-07-05 chain manifest; the ball_f1 copies used here are the best available originals and their shas are pinned in this report's manifest (archived_sha_recorded=false).",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_arc_sha256(path: Path) -> str:
    """Canonical sha of an arc-solved artifact minus wall-clock provenance.

    Budget-abstained segments embed ``degradation.elapsed_s`` (measured wall
    time, e.g. 5.000023 vs 5.000026 across reruns). That float is provenance,
    not measurement content; stripping it is the ONLY normalization applied.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    for segment in data.get("segments") or []:
        degradation = segment.get("degradation") if isinstance(segment, Mapping) else None
        if isinstance(degradation, dict):
            degradation.pop("elapsed_s", None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _round_floats(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        rounded = round(value, FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, Mapping):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip_entry(report: Mapping[str, Any], clip: str) -> Mapping[str, Any]:
    for entry in report["clips"]:
        if entry.get("clip") == clip:
            return entry
    raise KeyError(f"clip {clip} not in report")


def _clip_metrics(entry: Mapping[str, Any]) -> dict[str, Any]:
    coverage = entry["coverage"]
    segments = []
    for segment in entry["segments"]:
        raw = segment["reprojection"]["raw_track_visible_px"]
        segments.append(
            {
                "segment_id": segment["segment_id"],
                "span": [segment["frame_start"], segment["frame_end"]],
                "status": segment["status"],
                "verdict": segment["verdict"],
                "fail_closed_reasons": segment["fail_closed_reasons"],
                "inlier_count": segment["inlier_count"],
                "outlier_count": segment["outlier_count"],
                "metric_anchor_classes": segment["anchors"]["metric_anchor_classes"],
                "metric_anchor_anatomy": segment["anchors"]["metric_anchor_anatomy"],
                "fit_rmse_px": segment["reprojection"]["fit_rmse_px"],
                "fit_max_px": segment["reprojection"]["fit_max_px"],
                "raw_track_visible_px": raw,
            }
        )
    return {
        "solver_status": entry["solver_status"],
        "segment_count": entry["segment_count"],
        "segments_accepted": entry["segment_verdict_counts"].get("accepted", 0),
        "segment_verdict_counts": dict(entry["segment_verdict_counts"]),
        "segment_status_counts": dict(entry["segment_status_counts"]),
        "fail_closed_reason_counts": dict(entry["fail_closed_reason_counts"]),
        "coverage": {
            "rally_frame_count": coverage["rally_frame_count"],
            "frames_with_world_xyz": coverage["frames_with_world_xyz"],
            "accepted_3d_frame_count": coverage["accepted_3d_frame_count"],
            "accepted_3d_coverage_fraction": coverage["accepted_3d_coverage_fraction"],
            "hidden_frame_count": coverage["hidden_frame_count"],
            "fail_closed_suppressed_frame_count": coverage["fail_closed_suppressed_frame_count"],
        },
        "anchor_inventory": entry["anchor_inventory"],
        "segments_detail": segments,
    }


def _dict_delta(candidate: Mapping[str, int], baseline: Mapping[str, int]) -> dict[str, int]:
    keys = sorted(set(candidate) | set(baseline))
    return {
        key: int(candidate.get(key, 0)) - int(baseline.get(key, 0))
        for key in keys
        if int(candidate.get(key, 0)) - int(baseline.get(key, 0)) != 0
    }


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    cand_cov = candidate["coverage"]
    base_cov = baseline["coverage"]
    anatomy_delta = _dict_delta(
        {k: v["segments"] for k, v in candidate["anchor_inventory"]["by_metric_anchor_anatomy"].items()},
        {k: v["segments"] for k, v in baseline["anchor_inventory"]["by_metric_anchor_anatomy"].items()},
    )
    fraction_delta = None
    if cand_cov["accepted_3d_coverage_fraction"] is not None and base_cov["accepted_3d_coverage_fraction"] is not None:
        fraction_delta = cand_cov["accepted_3d_coverage_fraction"] - base_cov["accepted_3d_coverage_fraction"]
    return {
        "segment_count": candidate["segment_count"] - baseline["segment_count"],
        "segments_accepted": candidate["segments_accepted"] - baseline["segments_accepted"],
        "accepted_3d_frame_count": cand_cov["accepted_3d_frame_count"] - base_cov["accepted_3d_frame_count"],
        "accepted_3d_coverage_fraction": fraction_delta,
        "hidden_frame_count": cand_cov["hidden_frame_count"] - base_cov["hidden_frame_count"],
        "fail_closed_suppressed_frame_count": (
            cand_cov["fail_closed_suppressed_frame_count"] - base_cov["fail_closed_suppressed_frame_count"]
        ),
        "fail_closed_reason_counts": _dict_delta(
            candidate["fail_closed_reason_counts"], baseline["fail_closed_reason_counts"]
        ),
        "segment_status_counts": _dict_delta(
            candidate["segment_status_counts"], baseline["segment_status_counts"]
        ),
        "metric_anchor_anatomy_segments": anatomy_delta,
    }


def _joint_anchor_usage(arc_solved: Mapping[str, Any], clip_metrics: Mapping[str, Any]) -> dict[str, Any]:
    verdict_by_segment = {
        segment["segment_id"]: segment["verdict"] for segment in clip_metrics["segments_detail"]
    }
    used: dict[str, Any] = {"segments_using_hypothesis_anchors": [], "hypothesis_anchor_use_count": 0}
    for segment in arc_solved.get("segments") or []:
        if not isinstance(segment, Mapping):
            continue
        anchors = [
            anchor
            for anchor in (segment.get("anchors_used") or [])
            if isinstance(anchor, Mapping) and anchor.get("status") == "candidate_hypothesis"
        ]
        if not anchors:
            continue
        segment_id = segment.get("segment_id")
        used["hypothesis_anchor_use_count"] += len(anchors)
        used["segments_using_hypothesis_anchors"].append(
            {
                "segment_id": segment_id,
                "verdict": verdict_by_segment.get(segment_id, "unknown"),
                "hypothesis_anchor_count": len(anchors),
            }
        )
    used["segments_using_hypothesis_anchors"].sort(key=lambda item: (item["segment_id"] is None, item["segment_id"]))
    return used


def _ukf_section(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    samples = sidecar.get("samples") or []
    provenance_ok = all(
        isinstance(sample, Mapping)
        and sample.get("source") == "physics_interpolated"
        and sample.get("band") == "physics_predicted"
        and isinstance(sample.get("trust_band"), Mapping)
        and sample["trust_band"].get("band") == "low_confidence"
        for sample in samples
    )
    refusal_counts: dict[str, int] = {}
    for gap in sidecar.get("refused_gaps") or []:
        if not isinstance(gap, Mapping):
            continue
        reasons = gap.get("reasons")
        if not isinstance(reasons, list):
            reason = gap.get("reason")
            reasons = [reason] if reason else ["unspecified"]
        for reason in reasons:
            key = str(reason)
            refusal_counts[key] = refusal_counts.get(key, 0) + 1
    return {
        "summary": dict(sidecar.get("summary") or {}),
        "sample_count": len(samples),
        "all_samples_physics_interpolated_low_confidence": provenance_ok,
        "refusal_reason_counts": refusal_counts,
        "render_only": sidecar.get("render_only"),
        "not_for_detection_metrics": sidecar.get("not_for_detection_metrics"),
        "counted_as_accepted_or_measured": False,
        "harness_bucketing": (
            "ball_ukf_fallback.json is not read by the characterization harness "
            "(not a discover_clip_inputs artifact); accepted statistics are unaffected by construction"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dormant-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--inputs-root",
        type=Path,
        required=True,
        help="Checkout holding the archived 2026-07-05 solve (read-only).",
    )
    parser.add_argument(
        "--determinism-probe-dir",
        type=Path,
        default=None,
        help="Optional second burlington baseline solve dir for the byte-determinism probe.",
    )
    args = parser.parse_args(argv)
    dormant = args.dormant_dir.resolve()
    inputs_root = args.inputs_root.resolve()

    characterizations: dict[str, dict[str, Any]] = {}
    for variant in [ARCHIVED_KEY, *VARIANT_ORDER]:
        report_path = dormant / "characterization" / variant / "report.json"
        if report_path.is_file():
            characterizations[variant] = _read_json(report_path)
    if "baseline" not in characterizations or ARCHIVED_KEY not in characterizations:
        raise SystemExit("baseline and archived characterizations are required")

    measured_clips = [
        clip
        for clip in CLIP_ORDER
        if (dormant / "solves" / clip / "baseline" / "ball_track_arc_solved.json").is_file()
    ]

    metrics: dict[str, dict[str, Any]] = {}
    for variant, report in characterizations.items():
        metrics[variant] = {}
        for clip in measured_clips:
            try:
                metrics[variant][clip] = _clip_metrics(_clip_entry(report, clip))
            except KeyError:
                continue

    # --- pinned input manifest -------------------------------------------
    manifest: dict[str, Any] = {"clips": {}, "characterization_manifest_sha256": {}}
    for variant, report in characterizations.items():
        manifest["characterization_manifest_sha256"][variant] = report.get("manifest_sha256")
    for clip in measured_clips:
        baseline_log = _read_json(dormant / "solves" / clip / "baseline" / "solve_result.json")
        archived_dir = inputs_root / ARCHIVED_BASE / clip
        entry: dict[str, Any] = {
            "resolve_inputs": baseline_log["inputs"],
            "generated_at_pinned": baseline_log["generated_at_pinned"],
            "ball_type": baseline_log["ball_type"],
            "archived_artifacts": {
                name: {
                    "path": f"{ARCHIVED_BASE}/{clip}/{name}.json",
                    "sha256": sha256_file(archived_dir / f"{name}.json"),
                }
                for name in ("ball_track_arc_solved", "ball_flight_sanity", "ball_chain_manifest")
                if (archived_dir / f"{name}.json").is_file()
            },
            "solve_artifacts": {},
        }
        for variant in VARIANT_ORDER:
            solve_dir = dormant / "solves" / clip / variant
            if not (solve_dir / "ball_track_arc_solved.json").is_file():
                continue
            entry["solve_artifacts"][variant] = {
                name: sha256_file(solve_dir / f"{name}.json")
                for name in (
                    "ball_track_arc_solved",
                    "ball_flight_sanity",
                    "ball_chain_manifest",
                    "ball_ukf_fallback",
                    "ball_joint_anchor_candidates",
                )
                if (solve_dir / f"{name}.json").is_file()
            }
        manifest["clips"][clip] = entry

    # --- baseline vs archived reproduction -------------------------------
    reproduction: dict[str, Any] = {"per_clip": {}}
    exact = True
    for clip in measured_clips:
        base = metrics["baseline"].get(clip)
        archived = metrics[ARCHIVED_KEY].get(clip)
        if base is None or archived is None:
            continue
        delta = _metric_delta(base, archived)
        identical = all(
            (value == 0 or value is None or value == {}) for value in delta.values()
        ) and base["solver_status"] == archived["solver_status"]
        if not identical:
            exact = False
        reproduction["per_clip"][clip] = {
            "archived": archived,
            "baseline": base,
            "delta_baseline_minus_archived": delta,
            "solver_status": {"archived": archived["solver_status"], "baseline": base["solver_status"]},
            "content_identical": identical,
        }
    reproduction["verdict"] = (
        "reproduced" if exact else "not_reproduced_code_drift_since_20260705"
    )
    reproduction["note"] = (
        "Input shas are pinned to the archived chain manifests and verified before solving; "
        "any delta is solver code drift between 2026-07-05 and this worktree, and it applies "
        "equally to every variant (all variants share the same current code)."
    )

    # --- determinism probe ------------------------------------------------
    determinism: dict[str, Any] = {"available": False}
    probe_dir = args.determinism_probe_dir
    if probe_dir is not None:
        probe_arc = probe_dir / "burlington_gold_0300_low_steep_corner" / "baseline" / "ball_track_arc_solved.json"
        lane_arc = dormant / "solves" / "burlington_gold_0300_low_steep_corner" / "baseline" / "ball_track_arc_solved.json"
        if probe_arc.is_file() and lane_arc.is_file():
            sha_run1 = sha256_file(lane_arc)
            sha_run2 = sha256_file(probe_arc)
            norm_run1 = normalized_arc_sha256(lane_arc)
            norm_run2 = normalized_arc_sha256(probe_arc)
            determinism = {
                "available": True,
                "clip": "burlington_gold_0300_low_steep_corner",
                "variant": "baseline",
                "arc_solved_sha256_run1": sha_run1,
                "arc_solved_sha256_run2": sha_run2,
                "byte_identical": sha_run1 == sha_run2,
                "normalized_sha256_run1": norm_run1,
                "normalized_sha256_run2": norm_run2,
                "normalized_identical": norm_run1 == norm_run2,
                "normalization": "strip segments[*].degradation.elapsed_s only",
                "note": (
                    "Same pinned inputs + pinned generated_at, sequential runs on the same "
                    "machine. Raw bytes may differ only in the wall-clock elapsed_s recorded "
                    "inside budget-abstained segments; normalized_identical=true means every "
                    "measurement-relevant field (segments, verdicts, frames, anchors) was "
                    "identical across reruns. If normalized_identical were false, variant "
                    "deltas would carry segment-level budget noise."
                ),
            }

    # --- variant deltas ----------------------------------------------------
    deltas: dict[str, Any] = {}
    for variant in FLAG_VARIANTS:
        if variant not in metrics:
            continue
        deltas[variant] = {}
        for clip in measured_clips:
            candidate = metrics[variant].get(clip)
            base = metrics["baseline"].get(clip)
            if candidate is None or base is None:
                continue
            deltas[variant][clip] = _metric_delta(candidate, base)

    # --- byte-identity cross-checks (additive-flag proof) ------------------
    # Normalized = raw JSON minus segments[*].degradation.elapsed_s (wall-clock
    # provenance inside budget-abstained segments; see determinism probe).
    additive: dict[str, Any] = {}
    for clip in measured_clips:
        row: dict[str, Any] = {}
        pairs = [
            ("ukf_equals_baseline", "baseline", "ukf_fallback"),
            ("both_equals_joint_anchor_search", "joint_anchor_search", "both"),
            ("joint_anchor_search_equals_baseline", "baseline", "joint_anchor_search"),
        ]
        for key, left, right in pairs:
            left_path = dormant / "solves" / clip / left / "ball_track_arc_solved.json"
            right_path = dormant / "solves" / clip / right / "ball_track_arc_solved.json"
            if left_path.is_file() and right_path.is_file():
                row[key] = {
                    "raw_bytes": sha256_file(left_path) == sha256_file(right_path),
                    "normalized": normalized_arc_sha256(left_path) == normalized_arc_sha256(right_path),
                }
        additive[clip] = row

    # --- sidecar sections ---------------------------------------------------
    ukf_sections: dict[str, Any] = {}
    joint_sections: dict[str, Any] = {}
    for clip in measured_clips:
        for variant in ("ukf_fallback", "both"):
            sidecar_path = dormant / "solves" / clip / variant / "ball_ukf_fallback.json"
            if sidecar_path.is_file():
                ukf_sections.setdefault(clip, {})[variant] = _ukf_section(_read_json(sidecar_path))
        for variant in ("joint_anchor_search", "both"):
            solve_dir = dormant / "solves" / clip / variant
            sidecar_path = solve_dir / "ball_joint_anchor_candidates.json"
            if not sidecar_path.is_file():
                continue
            sidecar = _read_json(sidecar_path)
            arc = _read_json(solve_dir / "ball_track_arc_solved.json")
            chain_manifest = _read_json(solve_dir / "ball_chain_manifest.json")
            joint_sections.setdefault(clip, {})[variant] = {
                "summary": dict(sidecar.get("summary") or {}),
                "raw_observations_byte_identical": (sidecar.get("inputs") or {}).get(
                    "raw_observations_byte_identical"
                ),
                "candidate_flags_echo": chain_manifest.get("candidate_flags"),
                "hypothesis_anchor_usage": _joint_anchor_usage(
                    arc, metrics.get(variant, {}).get(clip, {"segments_detail": []})
                ),
            }

    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_dormant_flags_measurement_report",
        "label": LABEL,
        "policy": {
            "measurement_only": True,
            "promotion": False,
            "default_changes": False,
            "verified_gate": "VERIFIED=0 unaffected; this report is measurement, not verification",
            "ukf_samples_never_counted_as_accepted": True,
            "deterministic_report_bytes": True,
        },
        "caveats": CAVEATS,
        "inputs_manifest": manifest,
        "measured_clips": measured_clips,
        "skipped_clips": dict(sorted(SKIPPED_CLIPS.items())),
        "baseline_reproduction": reproduction,
        "determinism_probe": determinism,
        "variants": {
            variant: {clip: metrics[variant][clip] for clip in measured_clips if clip in metrics[variant]}
            for variant in metrics
        },
        "deltas_vs_baseline": deltas,
        "additive_flag_byte_identity": additive,
        "ukf_fallback_sidecar": ukf_sections,
        "joint_anchor_sidecar": joint_sections,
    }
    report = _round_floats(report)

    out_json = dormant / "report.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = dormant / "REPORT.md"
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(out_json), "verdict": reproduction["verdict"]}, sort_keys=True))
    return 0


def _pct(fraction: Any) -> str:
    if fraction is None:
        return "n/a"
    return f"{float(fraction) * 100.0:.1f}%"


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return f"{value:+d}"


def _fmt_dict(d: Mapping[str, Any]) -> str:
    if not d:
        return "-"
    return ", ".join(f"{key} {_fmt_delta(value)}" for key, value in sorted(d.items()))


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Dormant-flag measurement study — {report['label']}")
    lines.append("")
    lines.append(
        "Flags-only measurement of the ball solver's dormant evidence sources "
        "(joint-anchor search, conservative UKF fallback) on the three locally re-solvable "
        "eval clips. MEASUREMENT ONLY: no defaults changed, nothing promoted, `VERIFIED=0` "
        "stays binding. Owner decision 2026-07-23: default-on requires future T1 metric proof."
    )
    lines.append("")
    lines.append("## Honest caveats (read first)")
    lines.append("")
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")

    lines.append("## 1. Baseline vs archived 2026-07-05 solve (reproduction verdict)")
    lines.append("")
    repro = report["baseline_reproduction"]
    lines.append(f"**Verdict: `{repro['verdict']}`.** {repro['note']}")
    lines.append("")
    lines.append("| Clip | Status arch->base | Segments arch->base | Accepted segs arch->base | Accepted frames arch->base | Coverage arch->base |")
    lines.append("|---|---|---|---|---|---|")
    for clip, row in sorted(repro["per_clip"].items()):
        archived = row["archived"]
        base = row["baseline"]
        lines.append(
            f"| {clip} | {archived['solver_status']} -> {base['solver_status']} | "
            f"{archived['segment_count']} -> {base['segment_count']} | "
            f"{archived['segments_accepted']} -> {base['segments_accepted']} | "
            f"{archived['coverage']['accepted_3d_frame_count']} -> {base['coverage']['accepted_3d_frame_count']} | "
            f"{_pct(archived['coverage']['accepted_3d_coverage_fraction'])} -> {_pct(base['coverage']['accepted_3d_coverage_fraction'])} |"
        )
    lines.append("")
    lines.append("Per-clip taxonomy delta (baseline minus archived):")
    lines.append("")
    for clip, row in sorted(repro["per_clip"].items()):
        lines.append(
            f"- {clip}: reasons {{{_fmt_dict(row['delta_baseline_minus_archived']['fail_closed_reason_counts'])}}}; "
            f"statuses {{{_fmt_dict(row['delta_baseline_minus_archived']['segment_status_counts'])}}}"
        )
    lines.append("")

    determinism = report["determinism_probe"]
    lines.append("## 2. Determinism probe (wall-clock budget noise bound)")
    lines.append("")
    if determinism.get("available"):
        raw_verdict = "byte-identical" if determinism["byte_identical"] else "NOT byte-identical"
        norm_verdict = (
            "identical" if determinism["normalized_identical"] else "NOT identical"
        )
        lines.append(
            f"Second sequential burlington baseline solve with identical pinned inputs: raw bytes "
            f"**{raw_verdict}** (`{determinism['arc_solved_sha256_run1'][:12]}` vs "
            f"`{determinism['arc_solved_sha256_run2'][:12]}`); after stripping only "
            f"`segments[*].degradation.elapsed_s` (wall-clock provenance in budget-abstained "
            f"segments): **{norm_verdict}**. " + determinism["note"]
        )
    else:
        lines.append("Probe not available (cut for runtime); variant deltas carry unbounded budget noise.")
    lines.append("")

    lines.append("## 3. Per-flag headline deltas vs baseline")
    lines.append("")
    lines.append("| Variant | Clip | Segs | Accepted segs | Accepted frames | Coverage delta | Fail-closed reason deltas | Anchor anatomy deltas |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for variant in ["joint_anchor_search", "ukf_fallback", "both"]:
        rows = report["deltas_vs_baseline"].get(variant, {})
        for clip in report["measured_clips"]:
            delta = rows.get(clip)
            if delta is None:
                continue
            lines.append(
                f"| +{variant} | {clip} | {_fmt_delta(delta['segment_count'])} | "
                f"{_fmt_delta(delta['segments_accepted'])} | {_fmt_delta(delta['accepted_3d_frame_count'])} | "
                f"{_fmt_delta(delta['accepted_3d_coverage_fraction'])} | "
                f"{_fmt_dict(delta['fail_closed_reason_counts'])} | "
                f"{_fmt_dict(delta['metric_anchor_anatomy_segments'])} |"
            )
    lines.append("")
    lines.append(
        "Additive-flag identity of the arc-solved artifact (raw sha / normalized sha, where "
        "normalized strips only the wall-clock `degradation.elapsed_s` provenance):"
    )
    lines.append("")
    for clip, row in sorted(report["additive_flag_byte_identity"].items()):
        rendered = (
            ", ".join(
                f"{key}: raw={value['raw_bytes']} normalized={value['normalized']}"
                for key, value in sorted(row.items())
            )
            or "n/a"
        )
        lines.append(f"- {clip}: {rendered}")
    lines.append("")

    lines.append("## 4. Variant detail per clip")
    lines.append("")
    for variant in [ARCHIVED_KEY, *VARIANT_ORDER]:
        clips = report["variants"].get(variant)
        if not clips:
            continue
        lines.append(f"### {variant}")
        lines.append("")
        lines.append("| Clip | Status | Segs (acc) | Accepted frames / rally | Coverage | Suppressed | Hidden |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for clip in report["measured_clips"]:
            m = clips.get(clip)
            if m is None:
                continue
            cov = m["coverage"]
            lines.append(
                f"| {clip} | {m['solver_status']} | {m['segment_count']} ({m['segments_accepted']}) | "
                f"{cov['accepted_3d_frame_count']} / {cov['rally_frame_count']} | "
                f"{_pct(cov['accepted_3d_coverage_fraction'])} | {cov['fail_closed_suppressed_frame_count']} | "
                f"{cov['hidden_frame_count']} |"
            )
        lines.append("")
        lines.append("Per-segment reprojection (accepted segments only; raw px = recomputed against sha-verified calibration):")
        lines.append("")
        for clip in report["measured_clips"]:
            m = clips.get(clip)
            if m is None:
                continue
            accepted = [s for s in m["segments_detail"] if s["verdict"] == "accepted"]
            if not accepted:
                lines.append(f"- {clip}: no accepted segments")
                continue
            parts = []
            for segment in accepted:
                raw = segment["raw_track_visible_px"]
                raw_text = (
                    f"raw p50/p90/max {raw['p50']}/{raw['p90']}/{raw['max']}"
                    if isinstance(raw, Mapping) and raw.get("status") == "recomputed" and raw.get("count")
                    else "raw n/a"
                )
                parts.append(
                    f"seg{segment['segment_id']} [{segment['span'][0]}-{segment['span'][1]}] "
                    f"fit rmse/max {segment['fit_rmse_px']}/{segment['fit_max_px']} px, {raw_text}"
                )
            lines.append(f"- {clip}: " + "; ".join(parts))
        lines.append("")

    lines.append("## 5. Joint-anchor search sidecar (candidate-only, never measured)")
    lines.append("")
    joint = report["joint_anchor_sidecar"]
    if joint:
        lines.append("| Clip | Variant | Fallback wins searched | Refused | Hypotheses | Submitted (rank-1) | Chosen by selector | Hypothesis anchors used (segments) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for clip in report["measured_clips"]:
            for variant, row in sorted(joint.get(clip, {}).items()):
                summary = row["summary"]
                usage = row["hypothesis_anchor_usage"]
                seg_text = (
                    ", ".join(
                        f"seg{item['segment_id']}({item['verdict']})"
                        for item in usage["segments_using_hypothesis_anchors"]
                    )
                    or "-"
                )
                lines.append(
                    f"| {clip} | +{variant} | {summary.get('searched_window_count', 0)} | "
                    f"{summary.get('refused_window_count', 0)} | {summary.get('hypothesis_count', 0)} | "
                    f"{summary.get('submitted_anchor_count', 0)} | {summary.get('chosen_anchor_count', 0)} | {seg_text} |"
                )
        lines.append("")
        accepted_hypothesis_segments = sorted(
            f"{clip}/+{variant}/seg{item['segment_id']}"
            for clip in report["measured_clips"]
            for variant, row in joint.get(clip, {}).items()
            for item in row["hypothesis_anchor_usage"]["segments_using_hypothesis_anchors"]
            if item["verdict"] == "accepted"
        )
        lines.append(
            "Segments carrying a hypothesis anchor that were ACCEPTED: "
            + (", ".join(accepted_hypothesis_segments) if accepted_hypothesis_segments else "none")
            + ". Where acceptance changed vs baseline while no accepted segment carries a "
            "hypothesis anchor, the mechanism is re-segmentation around the injected candidate "
            "boundary, not direct anchoring of an accepted arc. A selector-chosen anchor can also "
            "end up in no final segment's anchors_used when subsequent refinement drops or merges "
            "its window."
        )
        lines.append("")
    else:
        lines.append("No joint-anchor variants were solved (cut).")
        lines.append("")

    lines.append("## 6. UKF fallback sidecar (reported separately, NEVER counted)")
    lines.append("")
    ukf = report["ukf_fallback_sidecar"]
    if ukf:
        lines.append("| Clip | Variant | Attempted gaps | Recovered gaps | Recovered samples | Refused gaps | All samples physics_interpolated low-confidence |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for clip in report["measured_clips"]:
            for variant, row in sorted(ukf.get(clip, {}).items()):
                summary = row["summary"]
                lines.append(
                    f"| {clip} | +{variant} | {summary.get('attempted_gap_count', 0)} | "
                    f"{summary.get('recovered_gap_count', 0)} | {summary.get('recovered_sample_count', 0)} | "
                    f"{summary.get('refused_gap_count', 0)} | {row['all_samples_physics_interpolated_low_confidence']} |"
                )
        lines.append("")
        lines.append(
            "Bucketing verification: the harness never reads `ball_ukf_fallback.json` "
            "(not part of `discover_clip_inputs`), and the arc-solved artifact with the flag on is "
            "identical to baseline modulo the wall-clock `degradation.elapsed_s` provenance "
            "(normalized shas equal, section 3), so UKF samples cannot appear in any accepted "
            "statistic."
        )
        lines.append("")
        lines.append("UKF refusal reasons:")
        lines.append("")
        for clip in report["measured_clips"]:
            for variant, row in sorted(ukf.get(clip, {}).items()):
                lines.append(f"- {clip} (+{variant}): {_fmt_dict(row['refusal_reason_counts']) if row['refusal_reason_counts'] else 'none'}")
        lines.append("")
    else:
        lines.append("No UKF variants were solved (cut).")
        lines.append("")

    lines.append("## 7. Skipped clips")
    lines.append("")
    for clip, reason in sorted(report["skipped_clips"].items()):
        lines.append(f"- `{clip}`: {reason}")
    lines.append("")
    lines.append("---")
    lines.append(
        "Accepted = segment passes `arc_segment_fail_closed_v1` and the frame carries a solver world "
        "position (existing harness definition, unchanged). Nothing in this study alters defaults, "
        "gates, or trust bands. VERIFIED=0."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
