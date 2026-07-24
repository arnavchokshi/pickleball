"""Characterization harness for the CURRENT fail-closed ball 3D arc solver.

Measurement-only, default-off. This module builds the "metric report for the
current physics-only solver" milestone named by the ball-3D program reframe
(runs/ball3d_lifting_plan_20260723/PLAN.md, v2): per-segment accepted vs
fail-closed verdicts, anchor inventory anatomy, rally-frame 3D coverage,
zero-return rates, and the far- vs near-court failure split — from existing
``ball_track_arc_solved.json`` artifacts (or a fresh ``run_default_ball_arc_chain``
solve when explicitly requested by the CLI).

It is NOT a promotion instrument: ``VERIFIED=0`` stays binding, nothing here
touches acceptance gates, and the physics-fill artifact (render-only,
``predicted`` frames) is reported separately, never blended into accepted
statistics. Reprojection numbers here are image-consistency measurements only;
they are blind to metric depth error (the plan's gap #2).

Determinism contract: given the same pinned input manifest, the report is
byte-identical across re-runs — sorted keys, floats rounded to
``FLOAT_DECIMALS``, no timestamps/hostnames/absolute paths in the report body
(the manifest hash stands in for provenance).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.ball_arc_chain import (
    default_ball_arc_solver_config,
    default_ball_chain_configs,
)
from threed.racketsport.ball_arc_solver import _project_world_point
from threed.racketsport.virtual_world import (
    BALL_ARC_FAIL_CLOSED_MAX_REPROJECTION_PX,
    BALL_ARC_FAIL_CLOSED_MIN_INLIERS,
    BALL_ARC_FAIL_CLOSED_POLICY,
    ball_arc_segment_fail_closed_verdicts,
)

SCHEMA_VERSION = 1
MANIFEST_ARTIFACT_TYPE = "racketsport_ball_solver_characterization_manifest"
REPORT_ARTIFACT_TYPE = "racketsport_ball_solver_characterization_report"
FLOAT_DECIMALS = 6

ARC_SOLVED_FILENAME = "ball_track_arc_solved.json"
FLIGHT_SANITY_FILENAME = "ball_flight_sanity.json"
PHYSICS_FILLED_FILENAME = "ball_track_physics_filled.json"
CALIBRATION_FILENAME = "court_calibration.json"
RALLY_SPANS_FILENAME = "rally_spans.json"
CHAIN_MANIFEST_FILENAME = "ball_chain_manifest.json"

_ANCHOR_CLASS_RALLY_ENDPOINT = "rally_endpoint_weak"
_METRICLESS_ANCHOR_CLASSES = frozenset({_ANCHOR_CLASS_RALLY_ENDPOINT, "none"})


@dataclass(frozen=True)
class ClipInputs:
    """Resolved on-disk artifact paths for one clip (``None`` when absent)."""

    clip: str
    clip_dir: Path
    arc_solved: Path | None
    flight_sanity: Path | None
    physics_filled: Path | None
    calibration: Path | None
    rally_spans: Path | None
    chain_manifest: Path | None


def discover_clip_inputs(
    clip: str,
    clip_dir: str | Path,
    *,
    calibration_override: str | Path | None = None,
) -> ClipInputs:
    """Locate the known artifact filenames inside ``clip_dir``.

    ``calibration_override`` points at a calibration file outside the clip
    directory (e.g. a sha-matched copy of the calibration the solve consumed).
    """

    directory = Path(clip_dir)

    def _existing(name: str) -> Path | None:
        candidate = directory / name
        return candidate if candidate.is_file() else None

    calibration: Path | None
    if calibration_override is not None:
        calibration = Path(calibration_override)
        if not calibration.is_file():
            raise FileNotFoundError(f"calibration override not found: {calibration}")
    else:
        calibration = _existing(CALIBRATION_FILENAME)
    return ClipInputs(
        clip=clip,
        clip_dir=directory,
        arc_solved=_existing(ARC_SOLVED_FILENAME),
        flight_sanity=_existing(FLIGHT_SANITY_FILENAME),
        physics_filled=_existing(PHYSICS_FILLED_FILENAME),
        calibration=calibration,
        rally_spans=_existing(RALLY_SPANS_FILENAME),
        chain_manifest=_existing(CHAIN_MANIFEST_FILENAME),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_sha_verified(inputs: ClipInputs) -> bool | None:
    """Does the calibration file byte-match the solve's recorded input?

    Returns ``None`` when no calibration is available, ``True`` only when the
    clip's ``ball_chain_manifest.json`` records a ``court_calibration`` input
    sha256 equal to the provided file's hash, ``False`` otherwise (fail
    closed: unverifiable calibration is never used for residual recompute).
    """

    if inputs.calibration is None:
        return None
    if inputs.chain_manifest is None:
        return False
    try:
        manifest = json.loads(inputs.chain_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    recorded = manifest.get("inputs") if isinstance(manifest, Mapping) else None
    entry = recorded.get("court_calibration") if isinstance(recorded, Mapping) else None
    recorded_sha = entry.get("sha256") if isinstance(entry, Mapping) else None
    if not isinstance(recorded_sha, str) or not recorded_sha:
        return False
    return sha256_file(inputs.calibration) == recorded_sha


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_characterization_manifest(
    clip_inputs: Sequence[ClipInputs],
    *,
    root: str | Path,
    label: str,
) -> dict[str, Any]:
    """Pinned, timestamp-free input manifest: path + sha256 per artifact."""

    root_path = Path(root).resolve()
    clips: dict[str, Any] = {}
    for inputs in clip_inputs:
        artifacts: dict[str, Any] = {}
        missing: list[str] = []
        named = {
            "ball_track_arc_solved": (inputs.arc_solved, ARC_SOLVED_FILENAME),
            "ball_flight_sanity": (inputs.flight_sanity, FLIGHT_SANITY_FILENAME),
            "ball_track_physics_filled": (inputs.physics_filled, PHYSICS_FILLED_FILENAME),
            "court_calibration": (inputs.calibration, CALIBRATION_FILENAME),
            "rally_spans": (inputs.rally_spans, RALLY_SPANS_FILENAME),
            "ball_chain_manifest": (inputs.chain_manifest, CHAIN_MANIFEST_FILENAME),
        }
        for kind, (path, filename) in sorted(named.items()):
            if path is None:
                if kind == "ball_track_arc_solved":
                    missing.append(filename)
                continue
            artifacts[kind] = {
                "path": _relative_posix(path, root=root_path),
                "sha256": sha256_file(path),
            }
        entry: dict[str, Any] = {
            "clip_dir": _relative_posix(inputs.clip_dir, root=root_path),
            "artifacts": artifacts,
        }
        if missing:
            entry["missing"] = sorted(missing)
        verified = calibration_sha_verified(inputs)
        if inputs.calibration is not None:
            entry["calibration_sha_matches_solver_input"] = bool(verified)
        clips[inputs.clip] = entry
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "label": label,
        "policy": _policy_block(),
        "solver_config_echo": {
            "fail_closed": {
                "policy": BALL_ARC_FAIL_CLOSED_POLICY,
                "min_inlier_count": BALL_ARC_FAIL_CLOSED_MIN_INLIERS,
                "max_reprojection_error_px": BALL_ARC_FAIL_CLOSED_MAX_REPROJECTION_PX,
            },
            "default_chain_configs": default_ball_chain_configs(),
            "default_solver_config": asdict(default_ball_arc_solver_config()),
        },
        "clips": clips,
    }


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _policy_block() -> dict[str, Any]:
    return {
        "measurement_only": True,
        "promotion": False,
        "verified_gate": "VERIFIED=0 unaffected; this report is measurement, not verification",
        "reprojection_blind_to_depth": True,
        "physics_fill_reported_separately": True,
        "deterministic_report_bytes": True,
    }


def _relative_posix(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


# ---------------------------------------------------------------------------
# Per-clip characterization
# ---------------------------------------------------------------------------


def characterize_clip_payloads(
    *,
    clip: str,
    arc_solved: Mapping[str, Any],
    flight_sanity: Mapping[str, Any] | None = None,
    physics_filled: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    calibration_sha_verified: bool | None = None,
    rally_spans: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Characterize one solved clip from already-parsed artifact payloads.

    Accepted-frame definition (mirrors the production fail-closed overlay in
    ``virtual_world.apply_ball_track_arc_solved_overlay``): a frame counts as
    an accepted 3D sample only when its band is not ``hidden``, it carries a
    ``world_xyz``, and its owning segment's own fit statistics pass
    ``ball_arc_segment_fail_closed_verdicts``. Frames without per-frame
    segment provenance fail closed inside any untrusted segment span.
    """

    frames = _frames(arc_solved)
    segments = arc_solved.get("segments") if isinstance(arc_solved.get("segments"), list) else []
    verdicts = ball_arc_segment_fail_closed_verdicts(arc_solved.get("segments"))
    untrusted_spans = _untrusted_spans(verdicts)
    sanity_by_span = _flight_sanity_by_span(flight_sanity)

    rally_ranges = _rally_span_ranges(rally_spans)
    rally_flags = [_in_rally(frame, index, rally_ranges) for index, frame in enumerate(frames)]

    frame_states = [
        _frame_state(frame, index, verdicts=verdicts, untrusted_spans=untrusted_spans)
        for index, frame in enumerate(frames)
    ]

    residuals_by_segment, residual_status = _residuals_by_segment(
        frames,
        calibration=calibration,
        calibration_sha_verified=calibration_sha_verified,
    )
    stored_residuals = _stored_residuals_by_segment(frames)

    segment_reports: list[dict[str, Any]] = []
    verdict_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        report = _segment_report(
            segment,
            verdicts=verdicts,
            sanity_by_span=sanity_by_span,
            residuals=residuals_by_segment.get(_segment_id(segment)),
            residual_status=residual_status,
            stored_residuals=stored_residuals.get(_segment_id(segment)),
            frame_states=frame_states,
            rally_flags=rally_flags,
            net_plane_consumed=_net_plane_consumed(arc_solved),
        )
        segment_reports.append(report)
        verdict_counts[report["verdict"]] = verdict_counts.get(report["verdict"], 0) + 1
        status_counts[report["status"]] = status_counts.get(report["status"], 0) + 1
        for reason in report["fail_closed_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    segment_reports.sort(key=lambda item: item["segment_id"])

    coverage = _coverage(frame_states, rally_flags, rally_spans_provided=bool(rally_ranges))
    zero_return = _zero_return(segment_reports, coverage)
    court_split = _court_split(
        frame_states,
        segment_reports,
        rally_flags,
        calibration=calibration,
        calibration_sha_verified=calibration_sha_verified,
    )

    return _round_floats({
        "clip": clip,
        "skipped": None,
        "solver_status": str(arc_solved.get("status") or "unknown"),
        "kill_reasons": _string_list(arc_solved.get("kill_reasons")),
        "net_plane_provenance": _mapping_or_none(arc_solved.get("net_plane_provenance")),
        "chain_config_degraded": arc_solved.get("chain_config_degraded"),
        "segment_count": len(segment_reports),
        "segments": segment_reports,
        "segment_verdict_counts": verdict_counts,
        "segment_status_counts": status_counts,
        "fail_closed_reason_counts": reason_counts,
        "anchor_inventory": _anchor_inventory(segment_reports),
        "coverage": coverage,
        "zero_return": zero_return,
        "court_split": court_split,
        "flight_sanity": _flight_sanity_section(flight_sanity),
        "physics_fill": _physics_fill_section(physics_filled),
    })


def _frames(arc_solved: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = arc_solved.get("frames")
    if not isinstance(frames, list):
        return []
    return [frame for frame in frames if isinstance(frame, Mapping)]


def _segment_id(segment: Mapping[str, Any]) -> int | None:
    raw = segment.get("segment_id")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw


def _untrusted_spans(verdicts: Mapping[int, Mapping[str, Any]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for verdict in verdicts.values():
        if verdict.get("trusted"):
            continue
        start = verdict.get("frame_start")
        end = verdict.get("frame_end")
        if isinstance(start, int) and isinstance(end, int):
            spans.append((start, end))
    return spans


@dataclass(frozen=True)
class _FrameState:
    segment_id: int | None
    has_world: bool
    hidden: bool
    accepted: bool
    suppressed: bool
    band: str
    world_y: float | None


def _frame_state(
    frame: Mapping[str, Any],
    index: int,
    *,
    verdicts: Mapping[int, Mapping[str, Any]],
    untrusted_spans: Sequence[tuple[int, int]],
) -> _FrameState:
    band = str(frame.get("band") or "")
    world = frame.get("world_xyz")
    has_world = isinstance(world, Sequence) and not isinstance(world, (str, bytes)) and len(world) == 3
    hidden = band == "hidden" or not has_world
    solver_info = frame.get("arc_solver")
    segment_id: int | None = None
    if isinstance(solver_info, Mapping):
        raw = solver_info.get("segment_id")
        if isinstance(raw, int) and not isinstance(raw, bool):
            segment_id = raw
    if hidden:
        return _FrameState(segment_id, has_world, True, False, False, band, None)
    if segment_id is not None:
        verdict = verdicts.get(segment_id)
        untrusted = verdict is None or not bool(verdict.get("trusted"))
    else:
        # No per-frame provenance: fail closed inside any untrusted span.
        untrusted = any(start <= index <= end for start, end in untrusted_spans)
    world_y = float(world[1]) if has_world else None
    if untrusted:
        return _FrameState(segment_id, has_world, False, False, True, band, world_y)
    return _FrameState(segment_id, has_world, False, True, False, band, world_y)


def _in_rally(frame: Mapping[str, Any], index: int, ranges: Sequence[tuple[float, float]]) -> bool:
    if not ranges:
        return True
    t = _float_or_none(frame.get("t"))
    if t is None:
        return False
    return any(t0 - 1e-9 <= t <= t1 + 1e-9 for t0, t1 in ranges)


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


# ---------------------------------------------------------------------------
# Segment reports
# ---------------------------------------------------------------------------


def _segment_report(
    segment: Mapping[str, Any],
    *,
    verdicts: Mapping[int, Mapping[str, Any]],
    sanity_by_span: Mapping[tuple[int, int], Mapping[str, Any]],
    residuals: Sequence[float] | None,
    residual_status: dict[str, Any],
    stored_residuals: Sequence[float] | None,
    frame_states: Sequence[_FrameState],
    rally_flags: Sequence[bool],
    net_plane_consumed: bool,
) -> dict[str, Any]:
    segment_id = _segment_id(segment)
    verdict = verdicts.get(segment_id) if segment_id is not None else None
    reasons = _string_list(verdict.get("reasons")) if isinstance(verdict, Mapping) else ["missing_fail_closed_verdict"]
    trusted = bool(verdict.get("trusted")) if isinstance(verdict, Mapping) else False

    anchors_used = segment.get("anchors_used") if isinstance(segment.get("anchors_used"), list) else []
    anchor_entries = [anchor for anchor in anchors_used if isinstance(anchor, Mapping)]
    classes = sorted({_anchor_class(anchor) for anchor in anchor_entries}) or ["none"]
    metric_classes = sorted(cls for cls in classes if cls not in _METRICLESS_ANCHOR_CLASSES)

    frame_total = 0
    frame_accepted = 0
    frame_with_world = 0
    for index, state in enumerate(frame_states):
        if state.segment_id != segment_id or segment_id is None:
            continue
        if not rally_flags[index]:
            continue
        frame_total += 1
        if state.has_world:
            frame_with_world += 1
        if state.accepted:
            frame_accepted += 1

    physical_sanity = segment.get("physical_sanity")
    violations = (
        _string_list(physical_sanity.get("violations")) if isinstance(physical_sanity, Mapping) else []
    )
    span = (segment.get("frame_start"), segment.get("frame_end"))
    sanity = sanity_by_span.get(span) if isinstance(span[0], int) and isinstance(span[1], int) else None

    recomputed = _residual_stats(residuals)
    if recomputed is not None:
        raw_track_stats: dict[str, Any] = {"status": "recomputed", **recomputed}
    elif residual_status["status"] == "recomputed":
        raw_track_stats = {"status": "recomputed", "count": 0, "p50": None, "p90": None, "max": None}
    else:
        raw_track_stats = dict(residual_status)

    return {
        "segment_id": segment_id if segment_id is not None else -1,
        "status": str(segment.get("status") or ""),
        "verdict": "accepted" if trusted else "rejected_fail_closed",
        "fail_closed_reasons": sorted(reasons),
        "frame_start": segment.get("frame_start"),
        "frame_end": segment.get("frame_end"),
        "inlier_count": _int_or_none(segment.get("inlier_count")),
        "outlier_count": _int_or_none(segment.get("outlier_count")),
        "initial_speed_mps": _float_or_none(segment.get("initial_speed_mps")),
        "reprojection": {
            "fit_rmse_px": _float_or_none(segment.get("reprojection_rmse_px")),
            "fit_max_px": _float_or_none(segment.get("max_reprojection_error_px")),
            "solver_selected_residuals_px": _residual_stats(stored_residuals),
            "raw_track_visible_px": raw_track_stats,
        },
        "anchors": {
            "start": _anchor_summary(anchor_entries, segment.get("start_anchor")),
            "end": _anchor_summary(anchor_entries, segment.get("end_anchor")),
            "classes": classes,
            "metric_anchor_classes": metric_classes,
            "metric_anchor_anatomy": _metric_anchor_anatomy(anchor_entries),
            "net_constraint_evaluated": bool(
                net_plane_consumed and segment.get("net_clearance_m") is not None
            ),
        },
        "net_clearance_m": _float_or_none(segment.get("net_clearance_m")),
        "net_clearance_ok": segment.get("net_clearance_ok"),
        "physical_sanity_violations": sorted(violations),
        "flight_sanity": {
            "verdict": str(sanity.get("verdict")) if isinstance(sanity, Mapping) and sanity.get("verdict") is not None else None,
            "reasons": _string_list(sanity.get("reasons")) if isinstance(sanity, Mapping) else [],
        },
        "frames": {
            "rally_total": frame_total,
            "with_world_xyz": frame_with_world,
            "accepted": frame_accepted,
        },
    }


def _anchor_class(anchor: Mapping[str, Any]) -> str:
    kind = str(anchor.get("kind") or "")
    status = str(anchor.get("status") or "")
    if kind == "bounce":
        if "human_reviewed" in status or status == "reviewed":
            return "bounce_reviewed"
        if status == "solver_proposed":
            return "bounce_solver_proposed"
        return "bounce_auto"
    if kind == "contact":
        return "contact_wrist_seed"
    if kind == "rally_endpoint":
        return _ANCHOR_CLASS_RALLY_ENDPOINT
    if kind == "net_plane":
        return "net_plane"
    return f"other_{kind}" if kind else "none"


def _metric_anchor_anatomy(anchors: Sequence[Mapping[str, Any]]) -> str:
    metric_flags = [
        _anchor_class(anchor) not in _METRICLESS_ANCHOR_CLASSES for anchor in anchors
    ]
    metric_count = sum(1 for flag in metric_flags if flag)
    if not anchors or metric_count == 0:
        return "no_metric_anchor"
    if metric_count >= 2:
        return "double_metric_anchor"
    return "single_sided_metric_anchor"


def _anchor_summary(anchors: Sequence[Mapping[str, Any]], anchor_id: Any) -> dict[str, Any] | None:
    if not isinstance(anchor_id, str):
        return None
    for anchor in anchors:
        if anchor.get("anchor_id") == anchor_id:
            return {
                "anchor_id": anchor_id,
                "kind": str(anchor.get("kind") or ""),
                "status": str(anchor.get("status") or ""),
                "source": str(anchor.get("source") or ""),
                "class": _anchor_class(anchor),
            }
    return {"anchor_id": anchor_id, "kind": None, "status": None, "source": None, "class": None}


def _anchor_inventory(segment_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_anatomy: dict[str, dict[str, int]] = {}
    by_class_combo: dict[str, dict[str, int]] = {}
    for report in segment_reports:
        anchors = report["anchors"]
        anatomy = str(anchors["metric_anchor_anatomy"])
        combo = "+".join(anchors["metric_anchor_classes"]) or "none"
        accepted = report["verdict"] == "accepted"
        for table, key in ((by_anatomy, anatomy), (by_class_combo, combo)):
            entry = table.setdefault(key, {"segments": 0, "accepted_segments": 0})
            entry["segments"] += 1
            if accepted:
                entry["accepted_segments"] += 1
    return {
        "by_metric_anchor_anatomy": by_anatomy,
        "by_metric_anchor_class_combo": by_class_combo,
    }


def _stored_residuals_by_segment(
    frames: Sequence[Mapping[str, Any]],
) -> dict[int | None, list[float]]:
    """Per-frame ``candidate_residual_px`` values newer solver artifacts store."""

    stored: dict[int | None, list[float]] = {}
    for frame in frames:
        solver_info = frame.get("arc_solver")
        if not isinstance(solver_info, Mapping):
            continue
        residual = _float_or_none(solver_info.get("candidate_residual_px"))
        if residual is None:
            continue
        raw = solver_info.get("segment_id")
        segment_id = raw if isinstance(raw, int) and not isinstance(raw, bool) else None
        stored.setdefault(segment_id, []).append(residual)
    return stored


def _residual_stats(values: Sequence[float] | None) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": _percentile(ordered, 50.0),
        "p90": _percentile(ordered, 90.0),
        "max": ordered[-1],
    }


# ---------------------------------------------------------------------------
# Reprojection residual recompute
# ---------------------------------------------------------------------------


def _residuals_by_segment(
    frames: Sequence[Mapping[str, Any]],
    *,
    calibration: Mapping[str, Any] | None,
    calibration_sha_verified: bool | None,
) -> tuple[dict[int | None, list[float]], dict[str, Any]]:
    """Per-segment residuals of solver world positions against raw track 2D.

    Only runs with a sha-verified calibration (the exact bytes the solve
    consumed): recomputing against any other calibration would fabricate
    residuals. Visible raw frames with a solver world position only, matching
    the w7_ball3ddiag methodology.
    """

    if calibration is None:
        return {}, {"status": "skipped", "reason": "missing_calibration"}
    if calibration_sha_verified is not True:
        return {}, {"status": "skipped", "reason": "calibration_not_sha_verified"}
    residuals: dict[int | None, list[float]] = {}
    for frame in frames:
        if frame.get("visible") is not True:
            continue
        world = frame.get("world_xyz")
        xy = frame.get("xy")
        if not _is_vec(world, 3) or not _is_vec(xy, 2):
            continue
        solver_info = frame.get("arc_solver")
        segment_id: int | None = None
        if isinstance(solver_info, Mapping):
            raw = solver_info.get("segment_id")
            if isinstance(raw, int) and not isinstance(raw, bool):
                segment_id = raw
        projected = _project_world_point(
            calibration, (float(world[0]), float(world[1]), float(world[2]))
        )
        residual = math.dist(projected, (float(xy[0]), float(xy[1])))
        residuals.setdefault(segment_id, []).append(residual)
    return residuals, {"status": "recomputed"}


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile over an already-sorted sequence."""

    if not ordered:
        raise ValueError("percentile of empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight


# ---------------------------------------------------------------------------
# Coverage / zero-return / court split
# ---------------------------------------------------------------------------


def _coverage(
    frame_states: Sequence[_FrameState],
    rally_flags: Sequence[bool],
    *,
    rally_spans_provided: bool,
) -> dict[str, Any]:
    rally_states = [state for state, in_rally in zip(frame_states, rally_flags) if in_rally]
    band_counts: dict[str, int] = {}
    for state in rally_states:
        band_counts[state.band or "missing"] = band_counts.get(state.band or "missing", 0) + 1
    accepted = sum(1 for state in rally_states if state.accepted)
    denominator_kind = "rally_spans" if rally_spans_provided else "all_input_frames"
    total = len(rally_states)
    return {
        "rally_frame_count": total,
        "rally_frame_denominator": denominator_kind,
        "frames_with_world_xyz": sum(1 for state in rally_states if state.has_world),
        "accepted_3d_frame_count": accepted,
        "accepted_3d_coverage_fraction": (accepted / total) if total else None,
        "hidden_frame_count": sum(1 for state in rally_states if state.hidden),
        "fail_closed_suppressed_frame_count": sum(1 for state in rally_states if state.suppressed),
        "band_counts": band_counts,
    }


def _zero_return(
    segment_reports: Sequence[Mapping[str, Any]], coverage: Mapping[str, Any]
) -> dict[str, Any]:
    total_segments = len(segment_reports)
    zero_segments = sum(1 for report in segment_reports if report["frames"]["accepted"] == 0)
    fraction = coverage.get("accepted_3d_coverage_fraction")
    return {
        "frame_zero_return_rate": (1.0 - float(fraction)) if fraction is not None else None,
        "segments_with_zero_accepted_frames": zero_segments,
        "segment_zero_return_rate": (zero_segments / total_segments) if total_segments else None,
    }


def _court_split(
    frame_states: Sequence[_FrameState],
    segment_reports: Sequence[Mapping[str, Any]],
    rally_flags: Sequence[bool],
    *,
    calibration: Mapping[str, Any] | None,
    calibration_sha_verified: bool | None,
) -> dict[str, Any]:
    """Failure split by court half (world y sign, net plane at y=0).

    Hidden frames carry no world position and cannot be attributed to a half;
    the split therefore covers only frames with a solver world position.
    """

    camera = _camera_center(calibration)
    camera_side: str | None = None
    camera_side_source: str | None = None
    if camera is not None:
        camera_side = "negative_y" if camera[1] < 0.0 else "positive_y"
        camera_side_source = (
            "sha_verified_calibration" if calibration_sha_verified is True else "unverified_calibration"
        )

    halves = {
        "y_negative": {"frames_with_world_xyz": 0, "accepted_frame_count": 0, "suppressed_frame_count": 0},
        "y_positive": {"frames_with_world_xyz": 0, "accepted_frame_count": 0, "suppressed_frame_count": 0},
    }
    for state, in_rally in zip(frame_states, rally_flags):
        if not in_rally or state.world_y is None:
            continue
        key = "y_negative" if state.world_y < 0.0 else "y_positive"
        halves[key]["frames_with_world_xyz"] += 1
        if state.accepted:
            halves[key]["accepted_frame_count"] += 1
        elif state.suppressed:
            halves[key]["suppressed_frame_count"] += 1
    for entry in halves.values():
        total = entry["frames_with_world_xyz"]
        entry["accepted_fraction"] = (entry["accepted_frame_count"] / total) if total else None

    segment_halves: dict[str, dict[str, int]] = {
        "y_negative": {"segments": 0, "accepted_segments": 0},
        "y_positive": {"segments": 0, "accepted_segments": 0},
    }
    mean_y_by_segment = _segment_mean_y(frame_states, rally_flags)
    for report in segment_reports:
        mean_y = mean_y_by_segment.get(report["segment_id"])
        if mean_y is None:
            continue
        key = "y_negative" if mean_y < 0.0 else "y_positive"
        segment_halves[key]["segments"] += 1
        if report["verdict"] == "accepted":
            segment_halves[key]["accepted_segments"] += 1

    near_half = far_half = None
    if camera_side == "negative_y":
        near_half, far_half = "y_negative", "y_positive"
    elif camera_side == "positive_y":
        near_half, far_half = "y_positive", "y_negative"
    return {
        "camera_side": camera_side,
        "camera_side_source": camera_side_source,
        "camera_center_world": list(camera) if camera is not None else None,
        "near_half": near_half,
        "far_half": far_half,
        "halves": halves,
        "segment_halves": segment_halves,
        "unattributable_hidden_frames": sum(
            1 for state, in_rally in zip(frame_states, rally_flags) if in_rally and state.hidden
        ),
    }


def _segment_mean_y(
    frame_states: Sequence[_FrameState], rally_flags: Sequence[bool]
) -> dict[int, float]:
    sums: dict[int, list[float]] = {}
    for state, in_rally in zip(frame_states, rally_flags):
        if not in_rally or state.world_y is None or state.segment_id is None:
            continue
        sums.setdefault(state.segment_id, []).append(state.world_y)
    return {segment_id: sum(values) / len(values) for segment_id, values in sums.items()}


def _camera_center(calibration: Mapping[str, Any] | None) -> tuple[float, float, float] | None:
    if not isinstance(calibration, Mapping):
        return None
    extrinsics = calibration.get("extrinsics")
    if not isinstance(extrinsics, Mapping):
        return None
    rotation = extrinsics.get("R")
    translation = extrinsics.get("t")
    if not isinstance(rotation, Sequence) or not isinstance(translation, Sequence):
        return None
    try:
        rows = [[float(value) for value in row] for row in rotation]
        t = [float(value) for value in translation]
    except (TypeError, ValueError):
        return None
    if len(rows) != 3 or any(len(row) != 3 for row in rows) or len(t) != 3:
        return None
    # Camera center C = -R^T t for the x_cam = R X + t convention.
    return (
        -(rows[0][0] * t[0] + rows[1][0] * t[1] + rows[2][0] * t[2]),
        -(rows[0][1] * t[0] + rows[1][1] * t[1] + rows[2][1] * t[2]),
        -(rows[0][2] * t[0] + rows[1][2] * t[1] + rows[2][2] * t[2]),
    )


# ---------------------------------------------------------------------------
# Sidecar sections
# ---------------------------------------------------------------------------


def _flight_sanity_by_span(
    flight_sanity: Mapping[str, Any] | None,
) -> dict[tuple[int, int], Mapping[str, Any]]:
    # w7_ball3ddiag documented a segment-id join mismatch between the sanity
    # and solver artifacts; the (frame_start, frame_end) span is the stable key.
    if not isinstance(flight_sanity, Mapping):
        return {}
    segments = flight_sanity.get("segments")
    if not isinstance(segments, list):
        return {}
    by_span: dict[tuple[int, int], Mapping[str, Any]] = {}
    for entry in segments:
        if not isinstance(entry, Mapping):
            continue
        start = entry.get("frame_start")
        end = entry.get("frame_end")
        if isinstance(start, int) and isinstance(end, int):
            by_span[(start, end)] = entry
    return by_span


def _flight_sanity_section(flight_sanity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(flight_sanity, Mapping):
        return {"artifact_present": False, "summary": None}
    summary = flight_sanity.get("summary")
    return {
        "artifact_present": True,
        "summary": dict(summary) if isinstance(summary, Mapping) else None,
    }


def _physics_fill_section(physics_filled: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(physics_filled, Mapping):
        return {
            "artifact_present": False,
            "policy": "render_only_not_blended_into_accepted_stats",
        }
    frames = physics_filled.get("frames")
    frame_list = [frame for frame in frames if isinstance(frame, Mapping)] if isinstance(frames, list) else []
    interpolated = sum(1 for frame in frame_list if frame.get("source") == "physics_interpolated")
    approx = sum(1 for frame in frame_list if frame.get("approx") is True)
    return {
        "artifact_present": True,
        "frame_count": len(frame_list),
        "physics_interpolated_frame_count": interpolated,
        "approx_frame_count": approx,
        "policy": "render_only_not_blended_into_accepted_stats",
    }


def _net_plane_consumed(arc_solved: Mapping[str, Any]) -> bool:
    provenance = arc_solved.get("net_plane_provenance")
    if isinstance(provenance, Mapping):
        return provenance.get("consumed_net_plane") is True
    # Older artifacts lack the provenance block; per-segment net_clearance_m
    # is then the only evidence a net constraint was evaluated.
    return True


# ---------------------------------------------------------------------------
# Pooled report
# ---------------------------------------------------------------------------


def build_characterization_report(
    clip_results: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the deterministic pooled report from per-clip results."""

    measured = [result for result in clip_results if not result.get("skipped")]
    skipped = [result for result in clip_results if result.get("skipped")]

    pooled_rally = sum(result["coverage"]["rally_frame_count"] for result in measured)
    pooled_accepted = sum(result["coverage"]["accepted_3d_frame_count"] for result in measured)
    pooled_hidden = sum(result["coverage"]["hidden_frame_count"] for result in measured)
    pooled_suppressed = sum(
        result["coverage"]["fail_closed_suppressed_frame_count"] for result in measured
    )
    verdict_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    anatomy: dict[str, dict[str, int]] = {}
    combos: dict[str, dict[str, int]] = {}
    halves = {
        "y_negative": {"frames_with_world_xyz": 0, "accepted_frame_count": 0, "suppressed_frame_count": 0},
        "y_positive": {"frames_with_world_xyz": 0, "accepted_frame_count": 0, "suppressed_frame_count": 0},
    }
    zero_segments = 0
    total_segments = 0
    for result in measured:
        for key, value in result["segment_verdict_counts"].items():
            verdict_counts[key] = verdict_counts.get(key, 0) + value
        for key, value in result["segment_status_counts"].items():
            status_counts[key] = status_counts.get(key, 0) + value
        for key, value in result["fail_closed_reason_counts"].items():
            reason_counts[key] = reason_counts.get(key, 0) + value
        for table, source_key in ((anatomy, "by_metric_anchor_anatomy"), (combos, "by_metric_anchor_class_combo")):
            for key, entry in result["anchor_inventory"][source_key].items():
                pooled_entry = table.setdefault(key, {"segments": 0, "accepted_segments": 0})
                pooled_entry["segments"] += entry["segments"]
                pooled_entry["accepted_segments"] += entry["accepted_segments"]
        for key in halves:
            for field in halves[key]:
                halves[key][field] += result["court_split"]["halves"][key][field]
        zero_segments += result["zero_return"]["segments_with_zero_accepted_frames"]
        total_segments += result["segment_count"]
    for entry in halves.values():
        total = entry["frames_with_world_xyz"]
        entry["accepted_fraction"] = (entry["accepted_frame_count"] / total) if total else None

    pooled = {
        "clip_count": len(measured),
        "skipped_clip_count": len(skipped),
        "coverage": {
            "rally_frame_count": pooled_rally,
            "accepted_3d_frame_count": pooled_accepted,
            "accepted_3d_coverage_fraction": (pooled_accepted / pooled_rally) if pooled_rally else None,
            "hidden_frame_count": pooled_hidden,
            "fail_closed_suppressed_frame_count": pooled_suppressed,
        },
        "zero_return": {
            "frame_zero_return_rate": (1.0 - pooled_accepted / pooled_rally) if pooled_rally else None,
            "segments_with_zero_accepted_frames": zero_segments,
            "segment_zero_return_rate": (zero_segments / total_segments) if total_segments else None,
        },
        "segment_count": total_segments,
        "segment_verdict_counts": verdict_counts,
        "segment_status_counts": status_counts,
        "fail_closed_reason_counts": reason_counts,
        "anchor_inventory": {
            "by_metric_anchor_anatomy": anatomy,
            "by_metric_anchor_class_combo": combos,
        },
        "court_split_halves": halves,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "label": manifest.get("label"),
        "manifest_sha256": manifest_sha256(manifest),
        "policy": _policy_block(),
        "fail_closed_policy": {
            "policy": BALL_ARC_FAIL_CLOSED_POLICY,
            "min_inlier_count": BALL_ARC_FAIL_CLOSED_MIN_INLIERS,
            "max_reprojection_error_px": BALL_ARC_FAIL_CLOSED_MAX_REPROJECTION_PX,
        },
        "pooled": pooled,
        "clips": [dict(result) for result in clip_results],
    }
    return _round_floats(report)


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


# ---------------------------------------------------------------------------
# Markdown + output writing
# ---------------------------------------------------------------------------


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Deterministic human summary (no timestamps, no absolute paths)."""

    pooled = report["pooled"]
    lines: list[str] = []
    lines.append(f"# Ball 3D solver characterization — {report.get('label')}")
    lines.append("")
    lines.append(
        "Measurement-only report for the CURRENT fail-closed physics-only ball arc solver. "
        "`VERIFIED=0` stays binding; nothing here is a promotion. Reprojection statistics are "
        "image-consistency only and are blind to metric depth error (no 3D ground truth exists yet)."
    )
    lines.append("")
    lines.append(f"Input manifest sha256: `{report['manifest_sha256']}` (see `manifest.json`).")
    lines.append("")
    lines.append("## Pooled headline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Clips measured / skipped | {pooled['clip_count']} / {pooled['skipped_clip_count']} |")
    lines.append(f"| Rally frames | {pooled['coverage']['rally_frame_count']} |")
    lines.append(
        f"| Accepted 3D frames | {pooled['coverage']['accepted_3d_frame_count']} "
        f"({_pct(pooled['coverage']['accepted_3d_coverage_fraction'])}) |"
    )
    lines.append(f"| Frame zero-return rate | {_pct(pooled['zero_return']['frame_zero_return_rate'])} |")
    lines.append(
        f"| Segments accepted / total | {pooled['segment_verdict_counts'].get('accepted', 0)} / "
        f"{pooled['segment_count']} |"
    )
    lines.append(
        f"| Segments returning zero accepted frames | "
        f"{pooled['zero_return']['segments_with_zero_accepted_frames']} |"
    )
    lines.append("")
    lines.append("### Fail-closed reason taxonomy (pooled)")
    lines.append("")
    reasons = pooled["fail_closed_reason_counts"]
    if reasons:
        lines.append("| Reason | Segments |")
        lines.append("|---|---:|")
        for key in sorted(reasons):
            lines.append(f"| {key} | {reasons[key]} |")
    else:
        lines.append("No fail-closed rejections.")
    lines.append("")
    lines.append("### Anchor anatomy vs acceptance (pooled)")
    lines.append("")
    lines.append("| Metric-anchor combo | Segments | Accepted |")
    lines.append("|---|---:|---:|")
    combos = pooled["anchor_inventory"]["by_metric_anchor_class_combo"]
    for key in sorted(combos):
        entry = combos[key]
        lines.append(f"| {key} | {entry['segments']} | {entry['accepted_segments']} |")
    lines.append("")
    lines.append("### Court-half split (frames with a solver world position)")
    lines.append("")
    lines.append("| Half | Frames | Accepted | Accepted % |")
    lines.append("|---|---:|---:|---:|")
    for key in sorted(pooled["court_split_halves"]):
        entry = pooled["court_split_halves"][key]
        lines.append(
            f"| {key} | {entry['frames_with_world_xyz']} | {entry['accepted_frame_count']} | "
            f"{_pct(entry['accepted_fraction'])} |"
        )
    lines.append("")
    lines.append("## Per-clip results")
    lines.append("")
    for clip in report["clips"]:
        lines.append(f"### {clip['clip']}")
        lines.append("")
        if clip.get("skipped"):
            missing = ", ".join(clip.get("missing", []))
            lines.append(f"Skipped: `{clip['skipped']}` (missing: {missing}).")
            lines.append("")
            continue
        coverage = clip["coverage"]
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Solver status | {clip['solver_status']} |")
        lines.append(
            f"| Accepted 3D coverage | {coverage['accepted_3d_frame_count']} / "
            f"{coverage['rally_frame_count']} ({_pct(coverage['accepted_3d_coverage_fraction'])}) |"
        )
        lines.append(f"| Hidden frames | {coverage['hidden_frame_count']} |")
        lines.append(f"| Fail-closed suppressed frames | {coverage['fail_closed_suppressed_frame_count']} |")
        lines.append(
            f"| Segments accepted / total | "
            f"{clip['segment_verdict_counts'].get('accepted', 0)} / {clip['segment_count']} |"
        )
        split = clip["court_split"]
        lines.append(f"| Camera side | {split['camera_side']} ({split['camera_side_source']}) |")
        lines.append("")
        lines.append("| Seg | Span | Status | Verdict | Inliers/Outliers | Fit RMSE/max px | Raw p50/p90/max px | Metric anchors | Sanity |")
        lines.append("|---:|---|---|---|---|---|---|---|---|")
        for segment in clip["segments"]:
            raw = segment["reprojection"]["raw_track_visible_px"]
            if raw.get("status") == "recomputed" and raw.get("count"):
                raw_text = f"{raw['p50']}/{raw['p90']}/{raw['max']}"
            else:
                raw_text = raw.get("reason", "n/a")
            anchors = "+".join(segment["anchors"]["metric_anchor_classes"]) or "none"
            sanity = segment["flight_sanity"]["verdict"] or "n/a"
            reasons = ",".join(segment["fail_closed_reasons"]) or "-"
            lines.append(
                f"| {segment['segment_id']} | {segment['frame_start']}-{segment['frame_end']} | "
                f"{segment['status']} | {segment['verdict']} ({reasons}) | "
                f"{segment['inlier_count']}/{segment['outlier_count']} | "
                f"{segment['reprojection']['fit_rmse_px']}/{segment['reprojection']['fit_max_px']} | "
                f"{raw_text} | {anchors} | {sanity} |"
            )
        lines.append("")
    lines.append("---")
    lines.append(
        "Accepted = segment passes `arc_segment_fail_closed_v1` "
        f"(min inliers {BALL_ARC_FAIL_CLOSED_MIN_INLIERS}, max reprojection "
        f"{BALL_ARC_FAIL_CLOSED_MAX_REPROJECTION_PX} px) and the frame carries a solver world "
        "position. Physics-fill artifacts are render-only and reported separately, never counted."
    )
    lines.append("")
    return "\n".join(lines)


def _pct(fraction: Any) -> str:
    if fraction is None:
        return "n/a"
    return f"{float(fraction) * 100.0:.1f}%"


def write_characterization_outputs(
    *,
    out_dir: str | Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    report_path = directory / "report.json"
    markdown_path = directory / "REPORT.md"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_report_markdown(report), encoding="utf-8")
    return {"manifest": manifest_path, "report": report_path, "markdown": markdown_path}


# ---------------------------------------------------------------------------
# High-level entry (read mode)
# ---------------------------------------------------------------------------


def run_characterization(
    clip_inputs: Sequence[ClipInputs],
    *,
    root: str | Path,
    label: str,
) -> dict[str, Any]:
    """Read-mode harness: manifest + report from existing solved artifacts."""

    manifest = build_characterization_manifest(clip_inputs, root=root, label=label)
    clip_results: list[dict[str, Any]] = []
    for inputs in clip_inputs:
        if inputs.arc_solved is None:
            clip_results.append(
                {
                    "clip": inputs.clip,
                    "skipped": "missing_artifacts",
                    "missing": [ARC_SOLVED_FILENAME],
                }
            )
            continue
        verified = calibration_sha_verified(inputs)
        clip_results.append(
            characterize_clip_payloads(
                clip=inputs.clip,
                arc_solved=_read_json(inputs.arc_solved),
                flight_sanity=_read_optional_json(inputs.flight_sanity),
                physics_filled=_read_optional_json(inputs.physics_filled),
                calibration=_read_optional_json(inputs.calibration),
                calibration_sha_verified=verified,
                rally_spans=_read_optional_json(inputs.rally_spans),
            )
        )
    report = build_characterization_report(clip_results, manifest=manifest)
    return {"manifest": manifest, "report": report}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return payload


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _read_json(path)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _is_vec(value: Any, length: int) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == length
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )
