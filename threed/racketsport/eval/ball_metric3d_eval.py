"""A-4 frozen metric-3D evaluation harness for ball trajectories.

FROZEN_JUDGE_PROTOCOL
=====================
This module is the JUDGE for the ball-3D program (Phase A-4 of
``runs/ball3d_lifting_plan_20260723/PLAN.md``, §5.A + v2 reframe). The judge
is FROZEN while the candidate changes: once a GT set + this harness score a
candidate system, later candidates must be scored by the same harness on the
same immutable GT/split — never by a harness edited to fit the candidate.
Changes to metric definitions require a new schema_version and invalidate
cross-version comparisons.

Rules baked in:

- Metric 3D error only. Reprojection error is out of scope here (it is blind
  to depth; see PLAN §0.3).
- ``acceptance_rate`` is ALWAYS reported next to accepted-frame accuracy.
  Rejection is never hidden: every block carries frame counts, the accepted
  fraction, and the error of rejected-but-solved frames when they exist.
- MVP thresholds are echoed from PLAN §5.D3 as PROVISIONAL constants
  (``MVP_ACCEPTED_MEDIAN_3D_M``, ``MVP_BOUNCE_MEDIAN_M``). They are internal
  go/no-go markers, NOT promotion gates: ``VERIFIED=0`` stays binding and a
  threshold comparison in a report is a measurement, never a promotion.
- Deterministic output: sorted keys, floats rounded to ``FLOAT_DECIMALS``,
  no timestamps/hostnames/absolute paths.

Pure computation: no I/O beyond the explicit ``write_report_json`` helper and
the A-3 contract loaders it composes with.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.ball_metric3d_contract import (
    GroundTruthObservationSet,
)

SCHEMA_VERSION = 1
REPORT_ARTIFACT_TYPE = "racketsport_ball_metric3d_eval_report"
VARIANTS_ARTIFACT_TYPE = "racketsport_ball_metric3d_eval_variants_report"
FLOAT_DECIMALS = 6

# PLAN §5.D3 MVP markers, echoed verbatim. PROVISIONAL: internal go/no-go
# only, set before any real GT exists; not research promises, not promotion
# gates, and subject to replacement by A1's own error curves.
MVP_ACCEPTED_MEDIAN_3D_M = 0.25
MVP_BOUNCE_MEDIAN_M = 0.20
MVP_THRESHOLD_STATUS = "provisional_plan_5d3_internal_marker_not_a_promotion_gate"

# Canonical variant names for the oracle hooks (PLAN v2 A-4: evaluate the
# same candidate under oracle-events / oracle-anchors inputs when provided).
VARIANT_PREDICTED = "predicted"
VARIANT_ORACLE_EVENTS = "oracle_events"
VARIANT_ORACLE_ANCHORS = "oracle_anchors"

EVENT_KIND_BOUNCE = "bounce"
EVENT_KIND_APEX = "apex"
KNOWN_EVENT_KINDS = frozenset({EVENT_KIND_BOUNCE, EVENT_KIND_APEX, "hit", "net_crossing"})

COURT_HALVES = ("y_negative", "y_positive")
_TIMESTAMP_MATCH_TOLERANCE_S = 1e-6
_DEFAULT_EVENT_MATCH_MAX_DT_S = 0.25


class Metric3DEvalError(ValueError):
    """Raised when evaluation inputs are malformed or misaligned."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSample:
    """Candidate world position at one GT timestamp (``None`` = no output)."""

    timestamp_s: float
    xyz_world_m: tuple[float, float, float] | None


@dataclass(frozen=True)
class EventEstimate:
    """One continuous-time event (GT or candidate) for the event metrics."""

    kind: str
    timestamp_s: float
    xyz_world_m: tuple[float, float, float] | None = None
    height_m: float | None = None

    def validate(self, *, path: str = "event") -> None:
        if self.kind not in KNOWN_EVENT_KINDS:
            raise Metric3DEvalError(
                f"{path}.kind: unknown event kind {self.kind!r}; "
                f"known: {sorted(KNOWN_EVENT_KINDS)}"
            )
        if not isinstance(self.timestamp_s, (int, float)) or isinstance(self.timestamp_s, bool):
            raise Metric3DEvalError(f"{path}.timestamp_s: expected a number")
        if not math.isfinite(float(self.timestamp_s)):
            raise Metric3DEvalError(f"{path}.timestamp_s: expected finite")

    @property
    def event_height_m(self) -> float | None:
        if self.height_m is not None:
            return float(self.height_m)
        if self.xyz_world_m is not None:
            return float(self.xyz_world_m[2])
        return None


@dataclass(frozen=True)
class CandidateRun:
    """One system's output aligned to the GT timeline.

    ``samples[i]`` and ``accepted[i]`` correspond to GT observation ``i``.
    ``accepted`` is the system's own acceptance mask (fail-closed verdicts);
    a frame may carry an xyz yet be rejected — that error is still reported.
    """

    samples: tuple[CandidateSample, ...]
    accepted: tuple[bool, ...]
    events: tuple[EventEstimate, ...] = ()


@dataclass(frozen=True)
class _PairedFrame:
    index: int
    timestamp_s: float
    gt_xyz: tuple[float, float, float]
    gt_quality_flags: tuple[str, ...]
    candidate_xyz: tuple[float, float, float] | None
    accepted: bool
    observed: bool | None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate(
    ground_truth: GroundTruthObservationSet,
    run: CandidateRun,
    *,
    observed_mask: Sequence[bool] | None = None,
    near_half: str | None = None,
    gt_events: Sequence[EventEstimate] = (),
    event_match_max_dt_s: float = _DEFAULT_EVENT_MATCH_MAX_DT_S,
) -> dict[str, Any]:
    """Score one candidate run against A-3 ground truth.

    ``observed_mask[i]`` marks whether the production detector observed the
    ball at GT frame ``i`` (drives the observed-vs-missing slice; the slice
    is reported ``not_measured`` when absent). ``near_half`` names which
    court half (``y_negative``/``y_positive``) is nearest the production
    camera; the court-half slice is always emitted by y-sign and annotated
    with near/far only when ``near_half`` is provided (never guessed).
    """

    ground_truth.validate()
    frames = _pair_frames(ground_truth, run, observed_mask=observed_mask)
    if near_half is not None and near_half not in COURT_HALVES:
        raise Metric3DEvalError(
            f"near_half: expected one of {sorted(COURT_HALVES)} or None, got {near_half!r}"
        )
    for index, event in enumerate(gt_events):
        event.validate(path=f"gt_events[{index}]")
    for index, event in enumerate(run.events):
        event.validate(path=f"candidate_events[{index}]")

    overall = _metrics_block(frames)
    slices = {
        "court_half": _court_half_slice(frames, near_half=near_half),
        "detection": _detection_slice(frames, observed_mask_provided=observed_mask is not None),
        "bounce_phase": _bounce_phase_slice(frames, gt_events=gt_events),
        "acceptance": _acceptance_slice(frames),
    }
    events = {
        "bounce": _bounce_event_metrics(
            gt_events, run.events, max_dt_s=event_match_max_dt_s
        ),
        "apex": _apex_event_metrics(gt_events, run.events, max_dt_s=event_match_max_dt_s),
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "clip": ground_truth.clip,
        "world_frame": ground_truth.world_frame,
        "frozen_judge": _frozen_judge_block(),
        "frame_count": len(frames),
        "overall": overall,
        "slices": slices,
        "events": events,
        "mvp_threshold_check": _mvp_threshold_check(overall, events),
    }
    return _round_floats(report)


def evaluate_variants(
    ground_truth: GroundTruthObservationSet,
    runs: Mapping[str, CandidateRun],
    *,
    observed_mask: Sequence[bool] | None = None,
    near_half: str | None = None,
    gt_events: Sequence[EventEstimate] = (),
    event_match_max_dt_s: float = _DEFAULT_EVENT_MATCH_MAX_DT_S,
) -> dict[str, Any]:
    """Oracle-variant hook: score named variants of the same candidate.

    Callers pass e.g. ``{VARIANT_PREDICTED: run, VARIANT_ORACLE_EVENTS:
    run_with_gt_event_times, VARIANT_ORACLE_ANCHORS: run_with_gt_anchors}``.
    Each variant gets the identical frozen judge, so differences localize
    whether failure is event timing, anchor localization, or depth itself
    (PLAN v2 A-4). Only variants actually provided are scored; nothing is
    synthesized.
    """

    if not runs:
        raise Metric3DEvalError("evaluate_variants: at least one variant run is required")
    systems = {
        name: evaluate_candidate(
            ground_truth,
            run,
            observed_mask=observed_mask,
            near_half=near_half,
            gt_events=gt_events,
            event_match_max_dt_s=event_match_max_dt_s,
        )
        for name, run in sorted(runs.items())
    }
    return _round_floats(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": VARIANTS_ARTIFACT_TYPE,
            "clip": ground_truth.clip,
            "frozen_judge": _frozen_judge_block(),
            "variant_names": sorted(systems),
            "systems": systems,
        }
    )


def _frozen_judge_block() -> dict[str, Any]:
    return {
        "protocol": "judge_frozen_while_candidate_changes",
        "verified_gate": "VERIFIED=0 unaffected; this report is measurement, not verification",
        "reprojection_out_of_scope": True,
        "acceptance_rate_always_reported": True,
        "mvp_thresholds": {
            "accepted_median_3d_m": MVP_ACCEPTED_MEDIAN_3D_M,
            "bounce_median_m": MVP_BOUNCE_MEDIAN_M,
            "status": MVP_THRESHOLD_STATUS,
        },
    }


def _mvp_threshold_check(
    overall: Mapping[str, Any], events: Mapping[str, Any]
) -> dict[str, Any]:
    accepted_error = overall.get("accepted_error")
    accepted_median = (
        accepted_error.get("err_3d_median_m") if isinstance(accepted_error, Mapping) else None
    )
    bounce_position = events["bounce"].get("position_error_m")
    bounce_median = (
        bounce_position.get("median") if isinstance(bounce_position, Mapping) else None
    )
    return {
        "status": MVP_THRESHOLD_STATUS,
        "accepted_median_3d_m": {
            "threshold_m": MVP_ACCEPTED_MEDIAN_3D_M,
            "measured_m": accepted_median,
            "under_threshold": (
                None if accepted_median is None else bool(accepted_median < MVP_ACCEPTED_MEDIAN_3D_M)
            ),
        },
        "bounce_median_m": {
            "threshold_m": MVP_BOUNCE_MEDIAN_M,
            "measured_m": bounce_median,
            "under_threshold": (
                None if bounce_median is None else bool(bounce_median < MVP_BOUNCE_MEDIAN_M)
            ),
        },
    }


# ---------------------------------------------------------------------------
# Frame pairing
# ---------------------------------------------------------------------------


def _pair_frames(
    ground_truth: GroundTruthObservationSet,
    run: CandidateRun,
    *,
    observed_mask: Sequence[bool] | None,
) -> list[_PairedFrame]:
    observations = ground_truth.observations
    if len(run.samples) != len(observations):
        raise Metric3DEvalError(
            f"candidate sample count {len(run.samples)} != GT observation count "
            f"{len(observations)}"
        )
    if len(run.accepted) != len(observations):
        raise Metric3DEvalError(
            f"acceptance mask length {len(run.accepted)} != GT observation count "
            f"{len(observations)}"
        )
    if observed_mask is not None and len(observed_mask) != len(observations):
        raise Metric3DEvalError(
            f"observed mask length {len(observed_mask)} != GT observation count "
            f"{len(observations)}"
        )
    frames: list[_PairedFrame] = []
    for index, (observation, sample) in enumerate(zip(observations, run.samples)):
        if abs(float(sample.timestamp_s) - float(observation.timestamp_s)) > _TIMESTAMP_MATCH_TOLERANCE_S:
            raise Metric3DEvalError(
                f"frame {index}: candidate timestamp {sample.timestamp_s!r} does not match "
                f"GT timestamp {observation.timestamp_s!r}"
            )
        candidate_xyz = sample.xyz_world_m
        if candidate_xyz is not None:
            candidate_xyz = (
                float(candidate_xyz[0]),
                float(candidate_xyz[1]),
                float(candidate_xyz[2]),
            )
        accepted = bool(run.accepted[index])
        if accepted and candidate_xyz is None:
            raise Metric3DEvalError(
                f"frame {index}: accepted=True but candidate has no world position "
                "(acceptance of nothing is contradictory)"
            )
        frames.append(
            _PairedFrame(
                index=index,
                timestamp_s=float(observation.timestamp_s),
                gt_xyz=observation.xyz_world_m,
                gt_quality_flags=observation.quality_flags,
                candidate_xyz=candidate_xyz,
                accepted=accepted,
                observed=None if observed_mask is None else bool(observed_mask[index]),
            )
        )
    return frames


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------


def _metrics_block(frames: Sequence[_PairedFrame]) -> dict[str, Any]:
    """Frame counts + acceptance rate + accepted/rejected error, one shape.

    ``acceptance_rate`` is accepted frames over ALL GT frames in the block
    (a system that outputs nothing scores 0, not n/a) — rejection is never
    hidden behind a shrunken denominator.
    """

    total = len(frames)
    with_candidate = [frame for frame in frames if frame.candidate_xyz is not None]
    accepted = [frame for frame in with_candidate if frame.accepted]
    rejected_with_xyz = [frame for frame in with_candidate if not frame.accepted]
    return {
        "frame_count": total,
        "frames_with_candidate_xyz": len(with_candidate),
        "accepted_frame_count": len(accepted),
        "acceptance_rate": (len(accepted) / total) if total else None,
        "accepted_error": _error_stats(accepted),
        "rejected_with_xyz_error": _error_stats(rejected_with_xyz),
    }


def _error_stats(frames: Sequence[_PairedFrame]) -> dict[str, Any] | None:
    if not frames:
        return None
    abs_dx: list[float] = []
    abs_dy: list[float] = []
    abs_dz: list[float] = []
    err_3d: list[float] = []
    for frame in frames:
        assert frame.candidate_xyz is not None  # guarded by callers
        dx = frame.candidate_xyz[0] - frame.gt_xyz[0]
        dy = frame.candidate_xyz[1] - frame.gt_xyz[1]
        dz = frame.candidate_xyz[2] - frame.gt_xyz[2]
        abs_dx.append(abs(dx))
        abs_dy.append(abs(dy))
        abs_dz.append(abs(dz))
        err_3d.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    ordered_3d = sorted(err_3d)
    return {
        "count": len(frames),
        "mae_x_m": sum(abs_dx) / len(abs_dx),
        "mae_y_m": sum(abs_dy) / len(abs_dy),
        "mae_z_m": sum(abs_dz) / len(abs_dz),
        "rmse_3d_m": math.sqrt(sum(err * err for err in err_3d) / len(err_3d)),
        "err_3d_median_m": _percentile(ordered_3d, 50.0),
        "err_3d_p90_m": _percentile(ordered_3d, 90.0),
        "err_3d_p95_m": _percentile(ordered_3d, 95.0),
        "err_3d_max_m": ordered_3d[-1],
    }


# ---------------------------------------------------------------------------
# Slices (partitions of the GT frames; parts always sum to the total)
# ---------------------------------------------------------------------------


def _court_half_slice(
    frames: Sequence[_PairedFrame], *, near_half: str | None
) -> dict[str, Any]:
    negative = [frame for frame in frames if frame.gt_xyz[1] < 0.0]
    positive = [frame for frame in frames if frame.gt_xyz[1] >= 0.0]
    far_half: str | None = None
    if near_half == "y_negative":
        far_half = "y_positive"
    elif near_half == "y_positive":
        far_half = "y_negative"
    return {
        "status": "measured",
        "near_half": near_half,
        "far_half": far_half,
        "partitions": {
            "y_negative": _metrics_block(negative),
            "y_positive": _metrics_block(positive),
        },
    }


def _detection_slice(
    frames: Sequence[_PairedFrame], *, observed_mask_provided: bool
) -> dict[str, Any]:
    if not observed_mask_provided:
        return {"status": "not_measured", "reason": "missing_observed_mask", "partitions": None}
    observed = [frame for frame in frames if frame.observed is True]
    missing = [frame for frame in frames if frame.observed is not True]
    return {
        "status": "measured",
        "partitions": {
            "observed": _metrics_block(observed),
            "missing": _metrics_block(missing),
        },
    }


def _bounce_phase_slice(
    frames: Sequence[_PairedFrame], *, gt_events: Sequence[EventEstimate]
) -> dict[str, Any]:
    bounce_times = sorted(
        float(event.timestamp_s) for event in gt_events if event.kind == EVENT_KIND_BOUNCE
    )
    if not bounce_times:
        return {"status": "not_measured", "reason": "no_gt_bounce_events", "partitions": None}
    first_bounce = bounce_times[0]
    pre = [frame for frame in frames if frame.timestamp_s < first_bounce]
    post = [frame for frame in frames if frame.timestamp_s >= first_bounce]
    return {
        "status": "measured",
        "first_gt_bounce_time_s": first_bounce,
        "partitions": {
            "pre_bounce": _metrics_block(pre),
            "post_bounce": _metrics_block(post),
        },
    }


def _acceptance_slice(frames: Sequence[_PairedFrame]) -> dict[str, Any]:
    accepted = [frame for frame in frames if frame.accepted]
    rejected = [frame for frame in frames if not frame.accepted]
    return {
        "status": "measured",
        "partitions": {
            "accepted": _metrics_block(accepted),
            "rejected": _metrics_block(rejected),
        },
    }


# ---------------------------------------------------------------------------
# Event metrics scaffold (continuous time)
# ---------------------------------------------------------------------------


def _bounce_event_metrics(
    gt_events: Sequence[EventEstimate],
    candidate_events: Sequence[EventEstimate],
    *,
    max_dt_s: float,
) -> dict[str, Any]:
    gt_bounces = [event for event in gt_events if event.kind == EVENT_KIND_BOUNCE]
    candidate_bounces = [
        event for event in candidate_events if event.kind == EVENT_KIND_BOUNCE
    ]
    pairs = _match_events(gt_bounces, candidate_bounces, max_dt_s=max_dt_s)
    time_errors = sorted(
        abs(float(candidate.timestamp_s) - float(gt.timestamp_s)) for gt, candidate in pairs
    )
    position_errors = sorted(
        math.dist(gt.xyz_world_m, candidate.xyz_world_m)
        for gt, candidate in pairs
        if gt.xyz_world_m is not None and candidate.xyz_world_m is not None
    )
    return {
        "gt_count": len(gt_bounces),
        "candidate_count": len(candidate_bounces),
        "matched_count": len(pairs),
        "unmatched_gt_count": len(gt_bounces) - len(pairs),
        "unmatched_candidate_count": len(candidate_bounces) - len(pairs),
        "match_max_dt_s": float(max_dt_s),
        "time_error_s": _ordered_stats(time_errors),
        "position_error_m": _ordered_stats(position_errors),
    }


def _apex_event_metrics(
    gt_events: Sequence[EventEstimate],
    candidate_events: Sequence[EventEstimate],
    *,
    max_dt_s: float,
) -> dict[str, Any]:
    gt_apexes = [event for event in gt_events if event.kind == EVENT_KIND_APEX]
    candidate_apexes = [event for event in candidate_events if event.kind == EVENT_KIND_APEX]
    pairs = _match_events(gt_apexes, candidate_apexes, max_dt_s=max_dt_s)
    height_errors = sorted(
        abs(candidate.event_height_m - gt.event_height_m)
        for gt, candidate in pairs
        if gt.event_height_m is not None and candidate.event_height_m is not None
    )
    return {
        "gt_count": len(gt_apexes),
        "candidate_count": len(candidate_apexes),
        "matched_count": len(pairs),
        "unmatched_gt_count": len(gt_apexes) - len(pairs),
        "unmatched_candidate_count": len(candidate_apexes) - len(pairs),
        "match_max_dt_s": float(max_dt_s),
        "height_error_m": _ordered_stats(height_errors),
    }


def _match_events(
    gt_events: Sequence[EventEstimate],
    candidate_events: Sequence[EventEstimate],
    *,
    max_dt_s: float,
) -> list[tuple[EventEstimate, EventEstimate]]:
    """Deterministic one-to-one greedy matching by ascending |dt|.

    Ties break on (gt index, candidate index) so the pairing is stable across
    runs regardless of input dict/set ordering upstream.
    """

    scored: list[tuple[float, int, int]] = []
    for gt_index, gt_event in enumerate(gt_events):
        for cand_index, cand_event in enumerate(candidate_events):
            dt = abs(float(cand_event.timestamp_s) - float(gt_event.timestamp_s))
            if dt <= float(max_dt_s):
                scored.append((dt, gt_index, cand_index))
    scored.sort()
    used_gt: set[int] = set()
    used_candidate: set[int] = set()
    pairs: list[tuple[EventEstimate, EventEstimate]] = []
    for _, gt_index, cand_index in scored:
        if gt_index in used_gt or cand_index in used_candidate:
            continue
        used_gt.add(gt_index)
        used_candidate.add(cand_index)
        pairs.append((gt_events[gt_index], candidate_events[cand_index]))
    pairs.sort(key=lambda pair: float(pair[0].timestamp_s))
    return pairs


def _ordered_stats(ordered: Sequence[float]) -> dict[str, Any] | None:
    if not ordered:
        return None
    return {
        "count": len(ordered),
        "median": _percentile(ordered, 50.0),
        "p90": _percentile(ordered, 90.0),
        "p95": _percentile(ordered, 95.0),
        "max": float(ordered[-1]),
    }


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
# Deterministic serialization
# ---------------------------------------------------------------------------


def dumps_report_json(report: Mapping[str, Any]) -> str:
    """Deterministic bytes: sorted keys, fixed separators, trailing newline."""

    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def write_report_json(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_report_json(report), encoding="utf-8")
    return target


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
