"""Score existing player tracks against reviewed person ground truth.

Gate v1 vs gate v2
------------------

**v1 (as originally documented, still computed and reported unchanged for
backward compatibility)** treats every unmatched prediction that doesn't
overlap an ``ignored`` GT box as one undifferentiated
``spectator_or_background_false_positives`` count, gated at ``== 0``.

The 2026-07-01 raw-pool offline-authority run
(`runs/phase2/trk_offline_authority_rawpool_20260701T222255Z/REPORT.md`)
showed this conflates two very different failure modes once exactly-4
global association with real appearance embeddings is in play:

1. A real detection box on a real player that simply has poor localization
   (IoU 0.2-0.5 against the matching GT box) -- a **detector-accuracy**
   problem that the association layer cannot fix and that the planned
   detector retrain (`W1-TRK-DET`) directly targets.
2. A box that has *zero* overlap with every real player in the frame --
   either a genuine spectator/background detection (safety-critical: never
   show a spectator as a player), or the exactly-N selector emitting a
   phantom Nth box during a frame where GT itself has fewer than
   ``expected_players`` players on court (a between-rally cardinality
   artifact, not a spectator-selection failure).

**v2 decomposes every unmatched, non-ignored-overlapping prediction into
exactly one of four mutually exclusive buckets** (in priority order):

- ``near_miss_false_positives``: best IoU against any real (non-ignored) GT
  player box in the frame is strictly greater than 0 (in practice this means
  ``(0, iou_threshold)`` -- a leftover prediction from the optimal bipartite
  matcher can only exceed ``iou_threshold`` against an already-claimed GT box
  in a measure-zero multi-match-tie edge case, and is folded into this bucket
  too since it is, by definition, close to a real player, not a spectator).
  Reported with localization diagnostics (median/p90 IoU, and a rate against
  total scored predictions).
- ``no_gt_frame_false_positives``: zero IoU against every real GT box in the
  frame, **and** the frame's own non-ignored GT player count is below
  ``expected_players`` (a "no full GT" frame -- e.g. a between-rally frame
  with 0-3 players on court). This is a cardinality/presence-awareness
  problem (recommendation 3 in the raw-pool REPORT.md), not a false
  detection of a bystander.
- ``true_spectator_or_background_false_positives``: zero IoU against every
  real GT box in the frame, the frame *does* have a full GT complement
  (``>= expected_players``), and the box's center falls inside the known
  image bounds when they are available (``image_width``/``image_height``).
  This is the narrow, safety-critical axis: a genuine bystander/background
  detection selected instead of (or in addition to) the real players.
- ``outside_image_false_positives``: same zero-IoU/full-GT-frame case as
  above but the box center falls outside the known image bounds -- a
  coordinate/projection artifact rather than a real spectator detection.
  Always ``0`` when ``image_width``/``image_height`` are not supplied (the
  bounds check is skipped and every zero-IoU/full-GT-frame box is treated as
  "inside image", matching the pre-v2 behavior).

``near_miss_false_positives + no_gt_frame_false_positives +
true_spectator_or_background_false_positives + outside_image_false_positives
== spectator_or_background_false_positives`` (the v1 aggregate) always holds.

**Gate v2 promotion rule** (``build_source_promotion_decision_v2``): same as
v1 -- IDF1 >= 0.85 per required clip, 0 ID switches, 0 off-court FP frames,
four-player coverage >= 0.95 -- except the strict ``== 0`` FP requirement is
narrowed to ``true_spectator_or_background_false_positives == 0`` (the v1
aggregate is no longer gated directly), **plus** a new localization target:
the near-miss false-positive rate (``near_miss_false_positives`` divided by
total scored predictions) must not exceed
``DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2`` (0.10). A rate
threshold was chosen over a raw median/p90 IoU threshold because it is
monotonic in "how much of the prediction stream is near-miss noise" and
directly comparable across clips with very different prediction volumes;
median and p90 near-miss IoU are still computed and reported
(``near_miss_localization``) as non-gating diagnostics for the detector-retrain
lane. 10% is deliberately "less punitive" than the true-spectator axis's
``== 0`` per `NORTH_STAR_ROADMAP.md`'s TRK gate v2 refinement, reflecting that this
axis is expected to close via detector data, not tracking logic.

Gate v2.1: apron-margin off-court refinement (2026-07-02)
-----------------------------------------------------------

Two independent pieces of evidence, produced by different lanes on the same
clip/config, both showed the v1/v2 off-court axis
(``off_court_false_positive_frames``: any world point strictly outside the
court template rectangle, 0.0m margin) penalizing legitimate boundary-line
play rather than catching spectators:

- ``runs/synergy_wirings_20260702T043529Z/w4_court_polygon_filter/w4_gate_v2_before_after.json``
  (the "w4" court-polygon-filter synergy wiring task): Burlington's 30
  baseline off-court FP frames (candidate margin=2.0m, iter5 pre-registered
  config) are one real player, track id 2, stepping 0.004-0.82m past the
  sideline for a single continuous 1.2s / 73-frame excursion
  (t=5.906-7.107s) that also contains 43 correctly-matched (true positive)
  frames of the same excursion. Both geometric fixes tried in that task
  that zero the axis (a strict post-association 0.0m filter, and a
  tightened 0.3m candidate-construction margin) cost four-player coverage
  (cov4): -0.122 and -0.085 respectively, because a geometric filter at
  candidate-construction time cannot distinguish the 30 FP frames from the
  43 TP frames riding in the same excursion -- removing one removes the
  other.
- ``runs/trk_r2_tiled_20260702T045400Z/REPORT.md`` ("Off-court axis note
  for the gate-v2 owner" section): reaches the same conclusion
  independently from the tiled-inference negative-result lane, noting the
  true-spectator axis is already 0 on Burlington/Outdoor, so the off-court
  axis is "double-charging legitimate boundary play rather than catching
  spectators", and recommending "an apron-margin refinement to the
  off-court axis (e.g. flag only frames >1.0m outside, or
  excursion-duration-aware)".

**Formulation chosen: a pure distance-based apron, no duration/sustained-time
term.** Every prediction (a matched real player OR an unmatched false
positive) whose world point falls beyond the court lines is split into:

- ``apron_off_court_excursion_*`` fields -- world point beyond the lines by
  no more than ``DEFAULT_OFF_COURT_APRON_MARGIN_M`` (1.0m, chosen because it
  comfortably covers the evidence excursion's observed 0.004-0.82m range
  with headroom, while still being tight enough that a genuine spectator
  standing well off the court -- not one player's boundary-line excursion --
  would not be absorbed by it). Computed over *all* predictions, matched or
  unmatched, so the diagnostic shows the full shape of the excursion, not
  just the FP-labeled slice of it. Never gate-blocking.
- ``far_off_court_false_positive_frames`` -- *unmatched* predictions whose
  world point is more than the apron margin beyond the court lines. This is
  the v2.1 gate-blocking axis (``== 0``), replacing
  ``off_court_false_positive_frames`` as the promotion-relevant off-court
  check for v2.1 only (the v1/v2 field and its own ``== 0`` gate are
  unchanged and still computed and reported). A real spectator/background
  detection with no matching GT player is overwhelmingly likely to be many
  meters off court, not centimeters.

A sustained-time criterion (e.g. "an excursion under 2 seconds never blocks
even if it strays past the apron") was considered, per the
``trk_r2_tiled`` REPORT.md's own "excursion-duration-aware" suggestion, and
deliberately *not* added, for three reasons: (1) the evidence excursion is
fully explained by distance alone -- its worst point (0.82m) sits well
inside the 1.0m apron regardless of its 1.2-second duration, so a duration
term would not change this case's outcome; (2) a duration term requires
stitching per-track excursion continuity across frames (gap-tolerant runs,
not just per-frame counts), which is materially more code and more edge
cases (what counts as "the same" excursion across a brief occlusion or a
track-id switch?) to add for a case with zero supporting evidence in either
evidence run; (3) YAGNI -- if a future run surfaces a *brief* *far*-off-court
FP (e.g. a single-frame calibration glitch many meters off court) that a
duration exemption would legitimately fix, that is a principled,
evidence-driven reason to extend ``far_off_court_false_positive_frames``
with a duration term then, not a reason to speculatively build it now. Pure
distance is the simplest formulation the current evidence defends.

**Gate v2.1 promotion rule** (``build_source_promotion_decision_v2_1``):
identical to gate v2 except the off-court check is
``far_off_court_false_positive_frames == 0`` instead of
``off_court_false_positive_frames == 0``. All v1/v2 fields and decisions
remain unchanged and are still computed and reported alongside v2.1 --
``build_scoring_report`` emits ``decision``, ``decision_v2``, and
``decision_v2_1`` side by side for every source; nothing is silently
replaced.

**PROSPECTIVE ONLY.** Gate v2.1 was defined 2026-07-02 and applies to future
scoring runs only. It must never be used to retroactively recompute or
overwrite the verdict of a row already recorded in
``runs/manager/heldout_eval_ledger.md``. See that ledger's TRK-13 entry
(added alongside this change) for the explicit no-motive proof:
re-reporting TRK-11's already-scored, pre-registered artifacts under v2.1
clears the off-court blocking axis on Burlington, but Burlington still fails
gate v2.1 overall on four-player coverage (cov4) -- exactly as it already
fails v1/v2 on cov4 today. This refinement changes zero recorded promotion
verdicts; it was not, and could not have been, motivated by wanting to flip
one.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .court_templates import Sport, get_court_template
from .mobile_person_eval import _bbox_iou, _match_frame, _overlaps_ignored, score_mobile_person_tracks
from .schemas import (
    OnDevicePersonFrame,
    OnDevicePersonTracks,
    OnDevicePersonTracksSummary,
    PersonGroundTruth,
    Tracks,
)


DEFAULT_IDF1_THRESHOLD = 0.85
DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD = 0.95
DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2 = 0.10
DEFAULT_OFF_COURT_APRON_MARGIN_M = 1.0
FAILURE_MODE_ORDER = (
    "missing_gt_detections",
    "spectator_or_background_false_positives",
    "off_court_false_positives",
    "four_player_coverage_gap",
    "id_switches",
)
FAILURE_MODE_DESCRIPTIONS = {
    "missing_gt_detections": "reviewed player boxes with no matching prediction",
    "spectator_or_background_false_positives": "predictions that did not match reviewed or ignored GT",
    "off_court_false_positives": "unmatched predictions whose world point is outside the court template",
    "four_player_coverage_gap": "GT frames without exactly the expected player count predicted",
    "id_switches": "GT identities matched to a different predicted track id",
}
FAILURE_MODE_ORDER_V2 = (
    "missing_gt_detections",
    "true_spectator_or_background_false_positives",
    "near_miss_localization",
    "off_court_false_positives",
    "four_player_coverage_gap",
    "id_switches",
    "no_gt_frame_false_positives",
)
FAILURE_MODE_DESCRIPTIONS_V2 = {
    "missing_gt_detections": "reviewed player boxes with no matching prediction",
    "true_spectator_or_background_false_positives": "zero-IoU predictions on a full-GT-complement frame (real bystander/background)",
    "near_miss_localization": "predictions with 0 < IoU < iou_threshold against a real player (detector localization noise)",
    "off_court_false_positives": "unmatched predictions whose world point is outside the court template",
    "four_player_coverage_gap": "GT frames without exactly the expected player count predicted",
    "id_switches": "GT identities matched to a different predicted track id",
    "no_gt_frame_false_positives": "zero-IoU predictions during frames where GT itself has fewer than expected_players players (cardinality artifact, not gated)",
}
FAILURE_MODE_ORDER_V2_1 = (
    "missing_gt_detections",
    "true_spectator_or_background_false_positives",
    "near_miss_localization",
    "far_off_court_false_positives",
    "four_player_coverage_gap",
    "id_switches",
    "no_gt_frame_false_positives",
    "apron_off_court_excursions",
)
FAILURE_MODE_DESCRIPTIONS_V2_1 = {
    "missing_gt_detections": "reviewed player boxes with no matching prediction",
    "true_spectator_or_background_false_positives": "zero-IoU predictions on a full-GT-complement frame (real bystander/background)",
    "near_miss_localization": "predictions with 0 < IoU < iou_threshold against a real player (detector localization noise)",
    "far_off_court_false_positives": (
        f"unmatched predictions whose world point is more than {DEFAULT_OFF_COURT_APRON_MARGIN_M:.1f}m beyond the "
        "court template lines (gate v2.1's off-court axis; apron excursions within the margin are excluded)"
    ),
    "four_player_coverage_gap": "GT frames without exactly the expected player count predicted",
    "id_switches": "GT identities matched to a different predicted track id",
    "no_gt_frame_false_positives": "zero-IoU predictions during frames where GT itself has fewer than expected_players players (cardinality artifact, not gated)",
    "apron_off_court_excursions": (
        f"predictions (matched or unmatched) whose world point is beyond the court lines but within "
        f"{DEFAULT_OFF_COURT_APRON_MARGIN_M:.1f}m of them -- boundary-line play, diagnostic only, never "
        "gate-blocking under v2.1"
    ),
}


def derive_track_source_id(path: str | Path, *, clip_ids: list[str]) -> str:
    parts = list(Path(path).parts)
    if parts and parts[-1] == "tracks.json":
        parts = parts[:-1]

    if len(parts) >= 3 and parts[0] == "runs" and parts[1] == "eval0":
        rest = parts[2:]
        if len(rest) >= 2 and rest[1] in clip_ids:
            if len(rest) == 2:
                return f"eval0/{rest[0]}/canonical_tracks"
            return "eval0/" + "/".join([rest[0], *rest[2:]])
        return "eval0/" + "/".join(_without_clip_ids(rest, clip_ids=clip_ids))

    if len(parts) >= 3 and parts[0] == "runs" and parts[1] == "phase2":
        rest = _without_clip_ids(parts[2:], clip_ids=clip_ids)
        while len(rest) >= 2 and rest[-1] == rest[-2]:
            rest.pop()
        return "phase2/" + "/".join(rest)

    return "/".join(_without_clip_ids(parts, clip_ids=clip_ids))


def score_tracks_against_person_ground_truth(
    *,
    ground_truth: PersonGroundTruth,
    tracks: Tracks,
    candidate: str,
    tracks_path: str | Path,
    iou_threshold: float = 0.5,
    expected_players: int | None = None,
    bbox_scale_x: float = 1.0,
    bbox_scale_y: float = 1.0,
    sport: Sport = "pickleball",
    image_width: float | None = None,
    image_height: float | None = None,
    off_court_apron_margin_m: float = DEFAULT_OFF_COURT_APRON_MARGIN_M,
) -> dict[str, Any]:
    predictions, prediction_world, outside_gt = _tracks_to_predictions(
        ground_truth=ground_truth,
        tracks=tracks,
        candidate=candidate,
        bbox_scale_x=bbox_scale_x,
        bbox_scale_y=bbox_scale_y,
    )
    expected = expected_players if expected_players is not None else ground_truth.summary.max_valid_players_per_frame
    metrics = score_mobile_person_tracks(
        ground_truth,
        predictions,
        iou_threshold=iou_threshold,
        expected_players=expected,
    )
    false_positive_details = _false_positive_details(
        ground_truth=ground_truth,
        predictions=predictions,
        prediction_world=prediction_world,
        iou_threshold=iou_threshold,
        sport=sport,
        expected_players=expected,
        image_width=image_width,
        image_height=image_height,
        off_court_apron_margin_m=off_court_apron_margin_m,
    )
    switch_diagnostics = _identity_switch_diagnostics(
        ground_truth=ground_truth,
        predictions=predictions,
        iou_threshold=iou_threshold,
    )
    hota_metrics = _hota_metrics(
        ground_truth=ground_truth,
        predictions=predictions,
        iou_threshold=iou_threshold,
        true_positives=metrics.matches,
        false_positives=metrics.false_positives,
        false_negatives=metrics.false_negatives,
    )

    near_miss_rate = _safe_rate(false_positive_details["near_miss_false_positives"], metrics.pred_detections)
    false_positive_details = {
        **false_positive_details,
        "near_miss_localization": {
            **false_positive_details["near_miss_localization"],
            "rate": near_miss_rate,
            "rate_denominator": metrics.pred_detections,
            "rate_threshold_v2": DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2,
        },
        "near_miss_false_positive_rate": near_miss_rate,
    }

    track_frame_count = sum(len(player.frames) for player in tracks.players)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_track_gt_score",
        "clip_id": ground_truth.clip_id,
        "candidate": candidate,
        "tracks_path": str(tracks_path),
        "iou_threshold": iou_threshold,
        "bbox_scale_x": bbox_scale_x,
        "bbox_scale_y": bbox_scale_y,
        "gt_frame_count": ground_truth.summary.frame_count,
        "gt_detections": metrics.gt_detections,
        "pred_detections": metrics.pred_detections,
        "matches": metrics.matches,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "spectator_or_background_false_positives": metrics.false_positives,
        "id_switches": metrics.id_switches,
        "idf1": metrics.idf1,
        **hota_metrics,
        "mota": metrics.mota,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "expected_players": metrics.expected_players,
        "four_player_coverage": metrics.expected_player_coverage,
        "expected_four_player_frames": metrics.expected_player_frames,
        "exact_four_player_frames": metrics.exact_expected_player_frames,
        "track_count": len(tracks.players),
        "track_frame_count": track_frame_count,
        "tracks_fps": tracks.fps,
        "outside_gt_prediction_count": outside_gt["prediction_count"],
        "outside_gt_prediction_track_ids": outside_gt["track_ids"],
        "identity_switch_event_count": switch_diagnostics["event_count"],
        "identity_switch_events": switch_diagnostics["events"],
        "identity_switch_transitions": switch_diagnostics["transitions"],
        "temporal_coverage": _temporal_coverage_diagnostics(ground_truth, predictions),
        **false_positive_details,
    }


def build_scoring_report(
    rows: list[dict[str, Any]],
    *,
    required_clip_ids: list[str],
    iou_threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["track_source_id"])].append(row)

    sources = []
    for source_id in sorted(grouped):
        source_rows = [
            _annotate_failure_modes(row)
            for row in sorted(grouped[source_id], key=lambda row: str(row.get("clip_id")))
        ]
        decision = build_source_promotion_decision(source_rows, required_clip_ids=required_clip_ids)
        decision_v2 = build_source_promotion_decision_v2(source_rows, required_clip_ids=required_clip_ids)
        decision_v2_1 = build_source_promotion_decision_v2_1(source_rows, required_clip_ids=required_clip_ids)
        sources.append(
            {
                "track_source_id": source_id,
                "clip_count": len({row.get("clip_id") for row in source_rows}),
                "clips": [row.get("clip_id") for row in source_rows],
                "decision": decision,
                "decision_v2": decision_v2,
                "decision_v2_1": decision_v2_1,
                "aggregate": _aggregate_source_rows(source_rows),
                "failure_analysis": _aggregate_failure_analysis(source_rows),
                "failure_analysis_v2": _aggregate_failure_analysis_v2(source_rows),
                "failure_analysis_v2_1": _aggregate_failure_analysis_v2_1(source_rows),
                "rows": source_rows,
            }
        )

    return {
        "schema_version": 2,
        "artifact_type": "racketsport_person_track_gt_scoring_report",
        "status": "scored_existing_tracks_only",
        "iou_threshold": iou_threshold,
        "required_clip_ids": required_clip_ids,
        "track_source_count": len(sources),
        "track_file_count": len(rows),
        "promotion_policy": {
            "idf1_threshold": DEFAULT_IDF1_THRESHOLD,
            "requires_zero_id_switches": True,
            "requires_zero_spectator_or_background_false_positives": True,
            "requires_zero_off_court_false_positive_frames": True,
            "four_player_coverage_threshold": DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD,
        },
        "promotion_policy_v2": {
            "idf1_threshold": DEFAULT_IDF1_THRESHOLD,
            "requires_zero_id_switches": True,
            "requires_zero_true_spectator_or_background_false_positives": True,
            "requires_zero_off_court_false_positive_frames": True,
            "four_player_coverage_threshold": DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD,
            "near_miss_false_positive_rate_threshold": DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2,
        },
        "promotion_policy_v2_1": {
            "idf1_threshold": DEFAULT_IDF1_THRESHOLD,
            "requires_zero_id_switches": True,
            "requires_zero_true_spectator_or_background_false_positives": True,
            "requires_zero_far_off_court_false_positive_frames": True,
            "off_court_apron_margin_m": DEFAULT_OFF_COURT_APRON_MARGIN_M,
            "four_player_coverage_threshold": DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD,
            "near_miss_false_positive_rate_threshold": DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2,
            "prospective_only": True,
            "prospective_only_note": (
                "Gate v2.1 was defined 2026-07-02; it applies to future scoring runs only and never "
                "retroactively changes a verdict already recorded in runs/manager/heldout_eval_ledger.md."
            ),
        },
        "sources": sources,
    }


def render_scoring_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Person Track GT Scoring",
        "",
        "- status: `scored_existing_tracks_only`",
        f"- IoU threshold: `{report['iou_threshold']}`",
        f"- track sources: `{report['track_source_count']}`",
        f"- track files: `{report['track_file_count']}`",
        "- inference: not run",
        "",
        "Promotion policy: IDF1 >= 0.85 on every required clip, zero ID switches, zero spectator/background false positives, zero off-court false-positive frames, and four-player coverage >= 0.95.",
        "",
        "## Source Decisions",
        "",
        "| Source | Decision | Clips | Mean IDF1 | Worst IDF1 | Mean HOTA | Worst HOTA | Switches | FP | Off-court FP | Mean cov4 | Worst cov4 | FPS | Primary failure | Blockers |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for source in report["sources"]:
        aggregate = source["aggregate"]
        decision = source["decision"]
        fps = aggregate.get("mean_effective_fps")
        lines.append(
            "| `{source}` | `{decision}` | {clips} | {mean_idf1} | {worst_idf1} | {mean_hota} | {worst_hota} | {switches} | {fp} | {offcourt} | {mean_cov} | {worst_cov} | {fps} | {primary_failure} | {blockers} |".format(
                source=source["track_source_id"],
                decision=decision["status"],
                clips=source["clip_count"],
                mean_idf1=_fmt(aggregate.get("mean_idf1")),
                worst_idf1=_fmt(aggregate.get("worst_idf1")),
                mean_hota=_fmt(aggregate.get("mean_hota")),
                worst_hota=_fmt(aggregate.get("worst_hota")),
                switches=aggregate.get("total_id_switches"),
                fp=aggregate.get("total_spectator_or_background_false_positives"),
                offcourt=aggregate.get("total_off_court_false_positive_frames"),
                mean_cov=_fmt(aggregate.get("mean_four_player_coverage")),
                worst_cov=_fmt(aggregate.get("worst_four_player_coverage")),
                fps=_fmt(fps) if fps is not None else "n/a",
                primary_failure=source.get("failure_analysis", {}).get("primary_failure_mode", "none"),
                blockers=", ".join(decision["blockers"]) if decision["blockers"] else "none",
            )
        )

    lines.extend(
        [
            "",
            "## Source Decisions (Gate v2)",
            "",
            (
                f"Gate v2 promotion policy: IDF1 >= {DEFAULT_IDF1_THRESHOLD:.2f} on every required clip, zero ID "
                "switches, zero **true** spectator/background false positives (near-miss localization FPs on real "
                "players no longer count against this axis), zero off-court false-positive frames, four-player "
                f"coverage >= {DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD:.2f}, and a near-miss false-positive rate <= "
                f"{DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2:.2f} (localization-quality target, non-strict)."
            ),
            "",
            "| Source | Decision v2 | True spectator/bg FP | Near-miss FP | Near-miss rate | No-GT-frame FP | Off-court FP | Worst cov4 | Blockers v2 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for source in report["sources"]:
        decision_v2 = source["decision_v2"]
        aggregate = source["aggregate"]
        true_spectator_total = sum(int(row.get("true_spectator_or_background_false_positives", 0)) for row in source["rows"])
        near_miss_total = sum(int(row.get("near_miss_false_positives", 0)) for row in source["rows"])
        no_gt_frame_total = sum(int(row.get("no_gt_frame_false_positives", 0)) for row in source["rows"])
        near_miss_pred_total = sum(int(row.get("pred_detections", 0)) for row in source["rows"])
        near_miss_rate = _safe_rate(near_miss_total, near_miss_pred_total)
        lines.append(
            "| `{source}` | `{decision}` | {true_spectator} | {near_miss} | {rate} | {no_gt_frame} | {offcourt} | {worst_cov} | {blockers} |".format(
                source=source["track_source_id"],
                decision=decision_v2["status"],
                true_spectator=true_spectator_total,
                near_miss=near_miss_total,
                rate=_fmt(near_miss_rate),
                no_gt_frame=no_gt_frame_total,
                offcourt=aggregate.get("total_off_court_false_positive_frames"),
                worst_cov=_fmt(aggregate.get("worst_four_player_coverage")),
                blockers=", ".join(decision_v2["blockers"]) if decision_v2["blockers"] else "none",
            )
        )

    lines.extend(
        [
            "",
            "## Source Decisions (Gate v2.1)",
            "",
            (
                f"Gate v2.1 promotion policy: identical to gate v2 (IDF1 >= {DEFAULT_IDF1_THRESHOLD:.2f}, zero ID "
                "switches, zero true spectator/background false positives, four-player coverage >= "
                f"{DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD:.2f}, near-miss rate <= "
                f"{DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2:.2f}), **except the off-court axis is "
                f"narrowed from any world point outside the court lines to only points more than "
                f"{DEFAULT_OFF_COURT_APRON_MARGIN_M:.1f}m beyond them** "
                "(`far_off_court_false_positive_frames == 0`). Excursions within the "
                f"{DEFAULT_OFF_COURT_APRON_MARGIN_M:.1f}m apron are reported as `apron_off_court_excursion_*` "
                "diagnostics and are never gate-blocking. See the module docstring "
                "(`threed/racketsport/person_track_gt_scoring.py`) for the evidence and rationale. "
                "**PROSPECTIVE ONLY: this does not change the verdict of any row already recorded in "
                "`runs/manager/heldout_eval_ledger.md`.**"
            ),
            "",
            "| Source | Decision v2.1 | True spectator/bg FP | Near-miss rate | Apron excursion frames | Far off-court FP | Worst cov4 | Blockers v2.1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for source in report["sources"]:
        decision_v2_1 = source["decision_v2_1"]
        aggregate = source["aggregate"]
        true_spectator_total = sum(int(row.get("true_spectator_or_background_false_positives", 0)) for row in source["rows"])
        near_miss_total = sum(int(row.get("near_miss_false_positives", 0)) for row in source["rows"])
        near_miss_pred_total = sum(int(row.get("pred_detections", 0)) for row in source["rows"])
        near_miss_rate = _safe_rate(near_miss_total, near_miss_pred_total)
        apron_frame_total = sum(int(row.get("apron_off_court_excursion_frame_count", 0)) for row in source["rows"])
        far_off_court_total = sum(
            int(row.get("far_off_court_false_positive_frames", row.get("off_court_false_positive_frames", 0)))
            for row in source["rows"]
        )
        lines.append(
            "| `{source}` | `{decision}` | {true_spectator} | {rate} | {apron} | {far_offcourt} | {worst_cov} | {blockers} |".format(
                source=source["track_source_id"],
                decision=decision_v2_1["status"],
                true_spectator=true_spectator_total,
                rate=_fmt(near_miss_rate),
                apron=apron_frame_total,
                far_offcourt=far_off_court_total,
                worst_cov=_fmt(aggregate.get("worst_four_player_coverage")),
                blockers=", ".join(decision_v2_1["blockers"]) if decision_v2_1["blockers"] else "none",
            )
        )

    lines.extend(
        [
            "",
            "## Clip Scores",
            "",
            "| Source | Clip | IDF1 | HOTA | DetA | AssA | MOTA | Switches | FP | FN | Off-court FP | cov4 | exact/expected cov4 frames | FPS | Tracks | Primary failure | Path |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for source in report["sources"]:
        for row in source["rows"]:
            fps = _timing_value(row, "effective_fps")
            lines.append(
                "| `{source}` | {clip} | {idf1} | {hota} | {deta} | {assa} | {mota} | {switches} | {fp} | {fn} | {offcourt} | {cov} | {frames} | {fps} | {tracks} | {primary_failure} | `{path}` |".format(
                    source=source["track_source_id"],
                    clip=row["clip_id"],
                    idf1=_fmt(row["idf1"]),
                    hota=_fmt(row.get("hota")),
                    deta=_fmt(row.get("deta")),
                    assa=_fmt(row.get("assa")),
                    mota=_fmt(row["mota"]),
                    switches=row["id_switches"],
                    fp=row["spectator_or_background_false_positives"],
                    fn=row["false_negatives"],
                    offcourt=row["off_court_false_positive_frames"],
                    cov=_fmt(row["four_player_coverage"]),
                    frames=f"{row['exact_four_player_frames']}/{row['expected_four_player_frames']}",
                    fps=_fmt(fps) if fps is not None else "n/a",
                    tracks=row["track_count"],
                    primary_failure=row.get("primary_failure_mode", "none"),
                    path=row["tracks_path"],
                )
            )
    lines.append("")

    lines.extend(
        [
            "## Temporal Coverage Diagnostics",
            "",
            "| Source | Clip | GT range | Prediction range | GT frames after last prediction | GT detections after last prediction | GT frames without predictions |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for source in report["sources"]:
        for row in source["rows"]:
            temporal = row.get("temporal_coverage") if isinstance(row.get("temporal_coverage"), dict) else {}
            gt_range = temporal.get("gt_frame_range") if isinstance(temporal.get("gt_frame_range"), dict) else {}
            pred_range = (
                temporal.get("prediction_frame_range")
                if isinstance(temporal.get("prediction_frame_range"), dict)
                else {}
            )
            lines.append(
                "| `{source}` | {clip} | {gt_range} | {pred_range} | {gt_after} | {det_after} | {without_pred} |".format(
                    source=source["track_source_id"],
                    clip=row["clip_id"],
                    gt_range=_range_text(gt_range),
                    pred_range=_range_text(pred_range),
                    gt_after=temporal.get("gt_frames_after_last_prediction", "n/a"),
                    det_after=temporal.get("gt_detections_after_last_prediction", "n/a"),
                    without_pred=temporal.get("gt_frames_without_predictions", "n/a"),
                )
            )

    lines.extend(
        [
            "",
            "## Identity Switch Events",
            "",
            "Full per-row switch event lists are in the JSON report. Markdown shows the first 10 events per scored clip.",
            "",
            "| Source | Clip | Frame | GT id | Previous pred id | New pred id | Previous match frame | Gap frames | IoU |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source in report["sources"]:
        for row in source["rows"]:
            for event in row.get("identity_switch_events", [])[:10]:
                lines.append(
                    "| `{source}` | {clip} | {frame} | {gt_id} | {prev_pred} | {new_pred} | {prev_frame} | {gap} | {iou} |".format(
                        source=source["track_source_id"],
                        clip=row["clip_id"],
                        frame=event["frame_index"],
                        gt_id=event["gt_track_id"],
                        prev_pred=event["previous_pred_track_id"],
                        new_pred=event["new_pred_track_id"],
                        prev_frame=event["previous_match_frame_index"],
                        gap=event["frames_since_previous_match"],
                        iou=_fmt(event.get("iou")),
                    )
                )
    lines.append("")
    return "\n".join(lines)


def build_source_promotion_decision(
    rows: list[dict[str, Any]],
    *,
    required_clip_ids: list[str],
    idf1_threshold: float = DEFAULT_IDF1_THRESHOLD,
    four_player_coverage_threshold: float = DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    blockers: list[str] = []
    rows_by_clip = {str(row.get("clip_id")): row for row in rows}
    missing = [clip_id for clip_id in required_clip_ids if clip_id not in rows_by_clip]
    if missing:
        blockers.append("missing_required_clips:" + ",".join(missing))

    for clip_id in sorted(rows_by_clip):
        row = rows_by_clip[clip_id]
        if float(row.get("idf1", 0.0)) < idf1_threshold:
            blockers.append(f"{clip_id}:idf1_below_{idf1_threshold:.2f}")
        if int(row.get("id_switches", 0)) > 0:
            blockers.append(f"{clip_id}:id_switches_present")
        if int(row.get("spectator_or_background_false_positives", 0)) > 0:
            blockers.append(f"{clip_id}:spectator_or_background_false_positives_present")
        if int(row.get("off_court_false_positive_frames", 0)) > 0:
            blockers.append(f"{clip_id}:off_court_false_positives_present")
        if float(row.get("four_player_coverage", 0.0)) < four_player_coverage_threshold:
            blockers.append(f"{clip_id}:four_player_coverage_below_{four_player_coverage_threshold:.2f}")

    return {
        "promote": not blockers,
        "status": "promote" if not blockers else "do_not_promote",
        "blockers": blockers,
        "policy": {
            "required_clip_ids": required_clip_ids,
            "idf1_threshold": idf1_threshold,
            "requires_zero_id_switches": True,
            "requires_zero_spectator_or_background_false_positives": True,
            "requires_zero_off_court_false_positive_frames": True,
            "four_player_coverage_threshold": four_player_coverage_threshold,
        },
    }


def build_source_promotion_decision_v2(
    rows: list[dict[str, Any]],
    *,
    required_clip_ids: list[str],
    idf1_threshold: float = DEFAULT_IDF1_THRESHOLD,
    four_player_coverage_threshold: float = DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD,
    near_miss_false_positive_rate_threshold: float = DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2,
) -> dict[str, Any]:
    """Gate v2: narrows the FP axis to true spectator/background boxes only,
    and adds a non-strict near-miss localization rate target. See the module
    docstring for the full rationale. Rows produced before the v2 fields
    existed (no ``true_spectator_or_background_false_positives`` key) are
    treated as if the narrower field equals the v1 aggregate -- i.e. as
    conservatively as possible, never silently passing an unscored row.
    """
    blockers: list[str] = []
    rows_by_clip = {str(row.get("clip_id")): row for row in rows}
    missing = [clip_id for clip_id in required_clip_ids if clip_id not in rows_by_clip]
    if missing:
        blockers.append("missing_required_clips:" + ",".join(missing))

    for clip_id in sorted(rows_by_clip):
        row = rows_by_clip[clip_id]
        if float(row.get("idf1", 0.0)) < idf1_threshold:
            blockers.append(f"{clip_id}:idf1_below_{idf1_threshold:.2f}")
        if int(row.get("id_switches", 0)) > 0:
            blockers.append(f"{clip_id}:id_switches_present")
        true_spectator_fp = row.get(
            "true_spectator_or_background_false_positives",
            row.get("spectator_or_background_false_positives", 0),
        )
        if int(true_spectator_fp) > 0:
            blockers.append(f"{clip_id}:true_spectator_or_background_false_positives_present")
        if int(row.get("off_court_false_positive_frames", 0)) > 0:
            blockers.append(f"{clip_id}:off_court_false_positives_present")
        if float(row.get("four_player_coverage", 0.0)) < four_player_coverage_threshold:
            blockers.append(f"{clip_id}:four_player_coverage_below_{four_player_coverage_threshold:.2f}")
        near_miss_rate = _near_miss_rate(row)
        if near_miss_rate is not None and near_miss_rate > near_miss_false_positive_rate_threshold:
            blockers.append(
                f"{clip_id}:near_miss_false_positive_rate_above_{near_miss_false_positive_rate_threshold:.2f}"
            )

    return {
        "promote": not blockers,
        "status": "promote" if not blockers else "do_not_promote",
        "blockers": blockers,
        "policy": {
            "required_clip_ids": required_clip_ids,
            "idf1_threshold": idf1_threshold,
            "requires_zero_id_switches": True,
            "requires_zero_true_spectator_or_background_false_positives": True,
            "requires_zero_off_court_false_positive_frames": True,
            "four_player_coverage_threshold": four_player_coverage_threshold,
            "near_miss_false_positive_rate_threshold": near_miss_false_positive_rate_threshold,
        },
    }


def build_source_promotion_decision_v2_1(
    rows: list[dict[str, Any]],
    *,
    required_clip_ids: list[str],
    idf1_threshold: float = DEFAULT_IDF1_THRESHOLD,
    four_player_coverage_threshold: float = DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD,
    near_miss_false_positive_rate_threshold: float = DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2,
) -> dict[str, Any]:
    """Gate v2.1: identical to gate v2 except the off-court axis is narrowed
    further, from ``off_court_false_positive_frames == 0`` (any world point
    outside the exact court rectangle) to ``far_off_court_false_positive_frames
    == 0`` (only *unmatched* predictions more than
    ``DEFAULT_OFF_COURT_APRON_MARGIN_M`` beyond the court lines). Boundary-line
    excursions within the apron (``apron_off_court_excursion_*`` fields) are
    reported but never block promotion. See the module docstring's "Gate
    v2.1" section for the evidence and rationale.

    Rows produced before the v2.1 fields existed (no
    ``far_off_court_false_positive_frames`` key) fall back to the legacy
    ``off_court_false_positive_frames`` value -- i.e. as conservatively as
    possible, never silently passing an unscored row.

    PROSPECTIVE ONLY: this function exists for scoring future runs. It must
    not be used to retroactively change the verdict recorded for a past
    ledger row -- see the module docstring's no-motive proof (Burlington's
    off-court axis clears under v2.1 but Burlington still fails on cov4,
    exactly as it does today under v1/v2).
    """
    blockers: list[str] = []
    rows_by_clip = {str(row.get("clip_id")): row for row in rows}
    missing = [clip_id for clip_id in required_clip_ids if clip_id not in rows_by_clip]
    if missing:
        blockers.append("missing_required_clips:" + ",".join(missing))

    for clip_id in sorted(rows_by_clip):
        row = rows_by_clip[clip_id]
        if float(row.get("idf1", 0.0)) < idf1_threshold:
            blockers.append(f"{clip_id}:idf1_below_{idf1_threshold:.2f}")
        if int(row.get("id_switches", 0)) > 0:
            blockers.append(f"{clip_id}:id_switches_present")
        true_spectator_fp = row.get(
            "true_spectator_or_background_false_positives",
            row.get("spectator_or_background_false_positives", 0),
        )
        if int(true_spectator_fp) > 0:
            blockers.append(f"{clip_id}:true_spectator_or_background_false_positives_present")
        far_off_court_fp = row.get(
            "far_off_court_false_positive_frames",
            row.get("off_court_false_positive_frames", 0),
        )
        if int(far_off_court_fp) > 0:
            blockers.append(f"{clip_id}:far_off_court_false_positives_present")
        if float(row.get("four_player_coverage", 0.0)) < four_player_coverage_threshold:
            blockers.append(f"{clip_id}:four_player_coverage_below_{four_player_coverage_threshold:.2f}")
        near_miss_rate = _near_miss_rate(row)
        if near_miss_rate is not None and near_miss_rate > near_miss_false_positive_rate_threshold:
            blockers.append(
                f"{clip_id}:near_miss_false_positive_rate_above_{near_miss_false_positive_rate_threshold:.2f}"
            )

    return {
        "promote": not blockers,
        "status": "promote" if not blockers else "do_not_promote",
        "blockers": blockers,
        "policy": {
            "required_clip_ids": required_clip_ids,
            "idf1_threshold": idf1_threshold,
            "requires_zero_id_switches": True,
            "requires_zero_true_spectator_or_background_false_positives": True,
            "requires_zero_far_off_court_false_positive_frames": True,
            "off_court_apron_margin_m": DEFAULT_OFF_COURT_APRON_MARGIN_M,
            "four_player_coverage_threshold": four_player_coverage_threshold,
            "near_miss_false_positive_rate_threshold": near_miss_false_positive_rate_threshold,
        },
    }


def _near_miss_rate(row: dict[str, Any]) -> float | None:
    localization = row.get("near_miss_localization")
    if isinstance(localization, dict) and localization.get("rate") is not None:
        return float(localization["rate"])
    if "near_miss_false_positive_rate" in row and row["near_miss_false_positive_rate"] is not None:
        return float(row["near_miss_false_positive_rate"])
    if "near_miss_false_positives" in row:
        return _safe_rate(row["near_miss_false_positives"], row.get("pred_detections", 0))
    return None


def summarize_score_failure_modes(row: dict[str, Any]) -> list[dict[str, Any]]:
    modes = [
        _failure_mode_record(mode, count, denominator)
        for mode, (count, denominator) in _failure_mode_counts(row).items()
        if count > 0
    ]
    return sorted(modes, key=_failure_mode_sort_key)


def summarize_score_failure_modes_v2(row: dict[str, Any]) -> list[dict[str, Any]]:
    modes = [
        _failure_mode_record(mode, count, denominator, descriptions=FAILURE_MODE_DESCRIPTIONS_V2, order=FAILURE_MODE_ORDER_V2)
        for mode, (count, denominator) in _failure_mode_counts_v2(row).items()
        if count > 0
    ]
    return sorted(modes, key=lambda mode: _failure_mode_sort_key(mode, order=FAILURE_MODE_ORDER_V2))


def summarize_score_failure_modes_v2_1(row: dict[str, Any]) -> list[dict[str, Any]]:
    modes = [
        _failure_mode_record(
            mode, count, denominator, descriptions=FAILURE_MODE_DESCRIPTIONS_V2_1, order=FAILURE_MODE_ORDER_V2_1
        )
        for mode, (count, denominator) in _failure_mode_counts_v2_1(row).items()
        if count > 0
    ]
    return sorted(modes, key=lambda mode: _failure_mode_sort_key(mode, order=FAILURE_MODE_ORDER_V2_1))


def _annotate_failure_modes(row: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(row)
    modes = summarize_score_failure_modes(annotated)
    annotated["failure_modes"] = modes
    annotated["primary_failure_mode"] = modes[0]["mode"] if modes else "none"
    modes_v2 = summarize_score_failure_modes_v2(annotated)
    annotated["failure_modes_v2"] = modes_v2
    annotated["primary_failure_mode_v2"] = modes_v2[0]["mode"] if modes_v2 else "none"
    modes_v2_1 = summarize_score_failure_modes_v2_1(annotated)
    annotated["failure_modes_v2_1"] = modes_v2_1
    annotated["primary_failure_mode_v2_1"] = modes_v2_1[0]["mode"] if modes_v2_1 else "none"
    return annotated


def _aggregate_failure_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _aggregate_failure_analysis_generic(rows, counts_fn=_failure_mode_counts, order=FAILURE_MODE_ORDER, descriptions=FAILURE_MODE_DESCRIPTIONS)


def _aggregate_failure_analysis_v2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _aggregate_failure_analysis_generic(
        rows, counts_fn=_failure_mode_counts_v2, order=FAILURE_MODE_ORDER_V2, descriptions=FAILURE_MODE_DESCRIPTIONS_V2
    )


def _aggregate_failure_analysis_v2_1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _aggregate_failure_analysis_generic(
        rows, counts_fn=_failure_mode_counts_v2_1, order=FAILURE_MODE_ORDER_V2_1, descriptions=FAILURE_MODE_DESCRIPTIONS_V2_1
    )


def _aggregate_failure_analysis_generic(
    rows: list[dict[str, Any]],
    *,
    counts_fn: Any,
    order: tuple[str, ...],
    descriptions: dict[str, str],
) -> dict[str, Any]:
    totals = {mode: {"count": 0, "denominator": 0} for mode in order}
    for row in rows:
        for mode, (count, denominator) in counts_fn(row).items():
            totals[mode]["count"] += count
            if denominator is not None:
                totals[mode]["denominator"] += denominator

    modes = [
        _failure_mode_record(
            mode,
            values["count"],
            values["denominator"] if values["denominator"] > 0 else None,
            descriptions=descriptions,
            order=order,
        )
        for mode, values in totals.items()
        if values["count"] > 0
    ]
    modes = sorted(modes, key=lambda mode: _failure_mode_sort_key(mode, order=order))
    return {
        "primary_failure_mode": modes[0]["mode"] if modes else "none",
        "modes": modes,
    }


def _failure_mode_counts(row: dict[str, Any]) -> dict[str, tuple[int, int | None]]:
    expected_four_player_frames = _nonnegative_int(row.get("expected_four_player_frames"))
    exact_four_player_frames = _nonnegative_int(row.get("exact_four_player_frames"))
    return {
        "missing_gt_detections": (
            _nonnegative_int(row.get("false_negatives")),
            _positive_int(row.get("gt_detections")),
        ),
        "spectator_or_background_false_positives": (
            _nonnegative_int(row.get("spectator_or_background_false_positives")),
            _positive_int(row.get("pred_detections")),
        ),
        "off_court_false_positives": (
            _nonnegative_int(row.get("off_court_false_positive_frames")),
            _positive_int(row.get("pred_detections")),
        ),
        "four_player_coverage_gap": (
            max(0, expected_four_player_frames - exact_four_player_frames),
            expected_four_player_frames if expected_four_player_frames > 0 else None,
        ),
        "id_switches": (
            _nonnegative_int(row.get("id_switches")),
            _positive_int(row.get("gt_detections")),
        ),
    }


def _failure_mode_counts_v2(row: dict[str, Any]) -> dict[str, tuple[int, int | None]]:
    expected_four_player_frames = _nonnegative_int(row.get("expected_four_player_frames"))
    exact_four_player_frames = _nonnegative_int(row.get("exact_four_player_frames"))
    # Rows scored before the v2 decomposition landed have no
    # `true_spectator_or_background_false_positives` key; fall back to the v1
    # aggregate so an unscored-under-v2 row is never silently reported clean.
    true_spectator_fp = row.get(
        "true_spectator_or_background_false_positives",
        row.get("spectator_or_background_false_positives"),
    )
    return {
        "missing_gt_detections": (
            _nonnegative_int(row.get("false_negatives")),
            _positive_int(row.get("gt_detections")),
        ),
        "true_spectator_or_background_false_positives": (
            _nonnegative_int(true_spectator_fp),
            _positive_int(row.get("pred_detections")),
        ),
        "near_miss_localization": (
            _nonnegative_int(row.get("near_miss_false_positives")),
            _positive_int(row.get("pred_detections")),
        ),
        "off_court_false_positives": (
            _nonnegative_int(row.get("off_court_false_positive_frames")),
            _positive_int(row.get("pred_detections")),
        ),
        "four_player_coverage_gap": (
            max(0, expected_four_player_frames - exact_four_player_frames),
            expected_four_player_frames if expected_four_player_frames > 0 else None,
        ),
        "id_switches": (
            _nonnegative_int(row.get("id_switches")),
            _positive_int(row.get("gt_detections")),
        ),
        "no_gt_frame_false_positives": (
            _nonnegative_int(row.get("no_gt_frame_false_positives")),
            _positive_int(row.get("pred_detections")),
        ),
    }


def _failure_mode_counts_v2_1(row: dict[str, Any]) -> dict[str, tuple[int, int | None]]:
    expected_four_player_frames = _nonnegative_int(row.get("expected_four_player_frames"))
    exact_four_player_frames = _nonnegative_int(row.get("exact_four_player_frames"))
    # Same v1-aggregate fallback discipline as v2 (see _failure_mode_counts_v2).
    true_spectator_fp = row.get(
        "true_spectator_or_background_false_positives",
        row.get("spectator_or_background_false_positives"),
    )
    # Rows scored before the v2.1 apron split landed have no
    # `far_off_court_false_positive_frames` key; fall back to the v1/v2
    # `off_court_false_positive_frames` value so an unscored-under-v2.1 row is
    # never silently reported clean (same discipline as the true-spectator
    # fallback above).
    far_off_court_fp = row.get(
        "far_off_court_false_positive_frames",
        row.get("off_court_false_positive_frames"),
    )
    return {
        "missing_gt_detections": (
            _nonnegative_int(row.get("false_negatives")),
            _positive_int(row.get("gt_detections")),
        ),
        "true_spectator_or_background_false_positives": (
            _nonnegative_int(true_spectator_fp),
            _positive_int(row.get("pred_detections")),
        ),
        "near_miss_localization": (
            _nonnegative_int(row.get("near_miss_false_positives")),
            _positive_int(row.get("pred_detections")),
        ),
        "far_off_court_false_positives": (
            _nonnegative_int(far_off_court_fp),
            _positive_int(row.get("pred_detections")),
        ),
        "four_player_coverage_gap": (
            max(0, expected_four_player_frames - exact_four_player_frames),
            expected_four_player_frames if expected_four_player_frames > 0 else None,
        ),
        "id_switches": (
            _nonnegative_int(row.get("id_switches")),
            _positive_int(row.get("gt_detections")),
        ),
        "no_gt_frame_false_positives": (
            _nonnegative_int(row.get("no_gt_frame_false_positives")),
            _positive_int(row.get("pred_detections")),
        ),
        "apron_off_court_excursions": (
            _nonnegative_int(row.get("apron_off_court_excursion_prediction_count")),
            _positive_int(row.get("pred_detections")),
        ),
    }


def _failure_mode_record(
    mode: str,
    count: int,
    denominator: int | None,
    *,
    descriptions: dict[str, str] = FAILURE_MODE_DESCRIPTIONS,
    order: tuple[str, ...] = FAILURE_MODE_ORDER,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "count": count,
        "rate": (count / denominator) if denominator else None,
        "denominator": denominator,
        "description": descriptions[mode],
    }


def _failure_mode_sort_key(mode: dict[str, Any], *, order: tuple[str, ...] = FAILURE_MODE_ORDER) -> tuple[int, int]:
    index = order.index(str(mode["mode"]))
    return (-int(mode["count"]), index)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _positive_int(value: Any) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed > 0 else None


def _safe_rate(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _aggregate_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    idf1_values = [float(row["idf1"]) for row in rows]
    hota_values = [float(row.get("hota", 0.0)) for row in rows]
    coverage_values = [float(row["four_player_coverage"]) for row in rows]
    fps_values = [
        value
        for row in rows
        for value in [_timing_value(row, "effective_fps")]
        if value is not None
    ]
    return {
        "mean_idf1": sum(idf1_values) / len(idf1_values) if idf1_values else 0.0,
        "worst_idf1": min(idf1_values) if idf1_values else 0.0,
        "mean_hota": sum(hota_values) / len(hota_values) if hota_values else 0.0,
        "worst_hota": min(hota_values) if hota_values else 0.0,
        "mean_four_player_coverage": sum(coverage_values) / len(coverage_values) if coverage_values else 0.0,
        "worst_four_player_coverage": min(coverage_values) if coverage_values else 0.0,
        "total_id_switches": sum(int(row["id_switches"]) for row in rows),
        "total_spectator_or_background_false_positives": sum(
            int(row["spectator_or_background_false_positives"]) for row in rows
        ),
        "total_off_court_false_positive_frames": sum(int(row["off_court_false_positive_frames"]) for row in rows),
        "total_false_negatives": sum(int(row["false_negatives"]) for row in rows),
        "mean_effective_fps": sum(fps_values) / len(fps_values) if fps_values else None,
    }


def _timing_value(row: dict[str, Any], key: str) -> float | None:
    timing = row.get("timing")
    if not isinstance(timing, dict):
        return None
    value = timing.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _without_clip_ids(parts: list[str], *, clip_ids: list[str]) -> list[str]:
    return [part for part in parts if part not in set(clip_ids)]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _range_text(value: dict[str, Any]) -> str:
    first = value.get("first")
    last = value.get("last")
    if first is None or last is None:
        return "n/a"
    return f"{first}-{last}"


def _identity_switch_diagnostics(
    *,
    ground_truth: PersonGroundTruth,
    predictions: OnDevicePersonTracks,
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_frame = {frame.frame_index: [label for label in frame.labels if not label.ignored] for frame in ground_truth.frames}
    pred_by_frame = {frame.frame_index: frame.detections for frame in predictions.frames}
    last_match_for_gt: dict[int, tuple[int, int]] = {}
    events: list[dict[str, Any]] = []
    transition_counts: dict[tuple[int, int, int], dict[str, int]] = {}

    for frame_index in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_labels = gt_by_frame.get(frame_index, [])
        pred_labels = pred_by_frame.get(frame_index, [])
        for gt_index, pred_index, iou in sorted(_match_frame(gt_labels, pred_labels, iou_threshold=iou_threshold)):
            gt_id = int(gt_labels[gt_index].track_id)
            pred_id = int(pred_labels[pred_index].track_id)
            previous = last_match_for_gt.get(gt_id)
            if previous is not None and previous[0] != pred_id:
                previous_pred_id, previous_frame_index = previous
                events.append(
                    {
                        "frame_index": frame_index,
                        "gt_track_id": gt_id,
                        "previous_pred_track_id": previous_pred_id,
                        "new_pred_track_id": pred_id,
                        "previous_match_frame_index": previous_frame_index,
                        "frames_since_previous_match": frame_index - previous_frame_index,
                        "iou": iou,
                    }
                )
                key = (gt_id, previous_pred_id, pred_id)
                transition = transition_counts.setdefault(
                    key,
                    {
                        "gt_track_id": gt_id,
                        "previous_pred_track_id": previous_pred_id,
                        "new_pred_track_id": pred_id,
                        "count": 0,
                        "first_frame_index": frame_index,
                        "last_frame_index": frame_index,
                    },
                )
                transition["count"] += 1
                transition["last_frame_index"] = frame_index
            last_match_for_gt[gt_id] = (pred_id, frame_index)

    transitions = sorted(
        transition_counts.values(),
        key=lambda transition: (
            -transition["count"],
            transition["first_frame_index"],
            transition["gt_track_id"],
            transition["previous_pred_track_id"],
            transition["new_pred_track_id"],
        ),
    )
    return {"event_count": len(events), "events": events, "transitions": transitions}


def _hota_metrics(
    *,
    ground_truth: PersonGroundTruth,
    predictions: OnDevicePersonTracks,
    iou_threshold: float,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, Any]:
    gt_by_frame = {frame.frame_index: [label for label in frame.labels if not label.ignored] for frame in ground_truth.frames}
    pred_by_frame = {frame.frame_index: frame.detections for frame in predictions.frames}
    pair_match_counts: dict[tuple[int, int], int] = defaultdict(int)
    gt_match_counts: dict[int, int] = defaultdict(int)
    pred_match_counts: dict[int, int] = defaultdict(int)

    for frame_index in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_labels = gt_by_frame.get(frame_index, [])
        pred_labels = pred_by_frame.get(frame_index, [])
        for gt_index, pred_index, _iou in _match_frame(gt_labels, pred_labels, iou_threshold=iou_threshold):
            gt_id = int(gt_labels[gt_index].track_id)
            pred_id = int(pred_labels[pred_index].track_id)
            pair_match_counts[(gt_id, pred_id)] += 1
            gt_match_counts[gt_id] += 1
            pred_match_counts[pred_id] += 1

    detection_denominator = true_positives + false_positives + false_negatives
    deta = true_positives / detection_denominator if detection_denominator > 0 else 0.0
    association_scores: list[float] = []
    for (gt_id, pred_id), pair_count in pair_match_counts.items():
        pair_denominator = gt_match_counts[gt_id] + pred_match_counts[pred_id] - pair_count
        pair_score = pair_count / pair_denominator if pair_denominator > 0 else 0.0
        association_scores.extend([pair_score] * pair_count)
    assa = sum(association_scores) / len(association_scores) if association_scores else 0.0
    hota = math.sqrt(deta * assa) if deta > 0.0 and assa > 0.0 else 0.0
    return {
        "hota": hota,
        "deta": deta,
        "assa": assa,
        "hota_iou_threshold": iou_threshold,
        "hota_matched_pair_count": len(pair_match_counts),
        "hota_single_threshold": True,
    }


def _temporal_coverage_diagnostics(
    ground_truth: PersonGroundTruth,
    predictions: OnDevicePersonTracks,
) -> dict[str, Any]:
    gt_by_frame = {frame.frame_index: [label for label in frame.labels if not label.ignored] for frame in ground_truth.frames}
    pred_by_frame = {frame.frame_index: frame.detections for frame in predictions.frames}
    gt_indexes = sorted(gt_by_frame)
    pred_indexes = sorted(pred_by_frame)
    last_prediction = pred_indexes[-1] if pred_indexes else None

    if last_prediction is None:
        gt_after_last_prediction = gt_indexes
    else:
        gt_after_last_prediction = [frame_index for frame_index in gt_indexes if frame_index > last_prediction]
    gt_without_predictions = [frame_index for frame_index in gt_indexes if frame_index not in pred_by_frame]

    return {
        "gt_frame_range": _frame_range(gt_indexes),
        "prediction_frame_range": _frame_range(pred_indexes),
        "gt_frame_count": len(gt_indexes),
        "prediction_frame_count": len(pred_indexes),
        "gt_frames_after_last_prediction": len(gt_after_last_prediction),
        "gt_detections_after_last_prediction": sum(len(gt_by_frame[frame_index]) for frame_index in gt_after_last_prediction),
        "gt_frames_without_predictions": len(gt_without_predictions),
        "gt_detections_without_predictions": sum(len(gt_by_frame[frame_index]) for frame_index in gt_without_predictions),
    }


def _frame_range(frame_indexes: list[int]) -> dict[str, int | None]:
    if not frame_indexes:
        return {"first": None, "last": None}
    return {"first": frame_indexes[0], "last": frame_indexes[-1]}


def _tracks_to_predictions(
    *,
    ground_truth: PersonGroundTruth,
    tracks: Tracks,
    candidate: str,
    bbox_scale_x: float,
    bbox_scale_y: float,
) -> tuple[OnDevicePersonTracks, dict[int, list[tuple[int, list[float]]]], dict[str, Any]]:
    total_frames = ground_truth.summary.frame_count
    detections_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    world_by_frame: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    outside_count = 0
    outside_track_ids: set[int] = set()

    for player in tracks.players:
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(tracks.fps)))
            if frame_index < 0:
                continue
            if frame_index >= total_frames:
                outside_count += 1
                outside_track_ids.add(int(player.id))
                continue
            x1, y1, x2, y2 = [float(value) for value in frame.bbox]
            x1 *= bbox_scale_x
            x2 *= bbox_scale_x
            y1 *= bbox_scale_y
            y2 *= bbox_scale_y
            detection = {
                "track_id": int(player.id),
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                "confidence": float(frame.conf),
                "source": "tracks_json",
                "role": player.role,
            }
            detections_by_frame[frame_index].append(detection)
            world_by_frame[frame_index].append((int(player.id), [float(value) for value in frame.world_xy]))

    frames = [
        OnDevicePersonFrame.model_validate({"frame_index": frame_index, "detections": detections_by_frame[frame_index]})
        for frame_index in sorted(detections_by_frame)
    ]
    track_ids = sorted(
        {
            int(detection["track_id"])
            for detections in detections_by_frame.values()
            for detection in detections
        }
    )
    predictions = OnDevicePersonTracks(
        schema_version=1,
        artifact_type="racketsport_on_device_person_tracks",
        clip_id=ground_truth.clip_id,
        candidate=candidate,
        fps=tracks.fps,
        frames=frames,
        summary=OnDevicePersonTracksSummary(
            frame_count=total_frames,
            detection_count=sum(len(frame.detections) for frame in frames),
            track_ids=track_ids,
        ),
    )
    return predictions, world_by_frame, {"prediction_count": outside_count, "track_ids": sorted(outside_track_ids)}


def _false_positive_details(
    *,
    ground_truth: PersonGroundTruth,
    predictions: OnDevicePersonTracks,
    prediction_world: dict[int, list[tuple[int, list[float]]]],
    iou_threshold: float,
    sport: Sport,
    expected_players: int,
    image_width: float | None = None,
    image_height: float | None = None,
    off_court_apron_margin_m: float = DEFAULT_OFF_COURT_APRON_MARGIN_M,
) -> dict[str, Any]:
    gt_by_frame = {frame.frame_index: [label for label in frame.labels if not label.ignored] for frame in ground_truth.frames}
    ignored_by_frame = {frame.frame_index: [label for label in frame.labels if label.ignored] for frame in ground_truth.frames}
    pred_by_frame = {frame.frame_index: frame.detections for frame in predictions.frames}
    template = get_court_template(sport)
    half_width_m = template.width_m / 2.0
    half_length_m = template.length_m / 2.0

    off_court_count = 0
    off_court_track_ids: set[int] = set()
    false_positive_frames: list[int] = []
    near_miss_count = 0
    near_miss_ious: list[float] = []
    no_gt_frame_count = 0
    true_spectator_count = 0
    outside_image_count = 0

    # Gate v2.1 apron-margin off-court refinement (module docstring has the
    # evidence + rationale). Computed over *every* prediction in the frame
    # (matched real players as well as unmatched false positives) so the
    # apron diagnostic reflects the whole shape of a boundary-line
    # excursion, not just the FP-labeled slice of it. Only *unmatched*
    # predictions beyond the apron count toward the v2.1 gate-blocking axis
    # (`far_off_court_false_positive_frames`) -- a matched prediction is a
    # real player, never a false positive, regardless of court position.
    apron_prediction_count = 0
    apron_frame_indexes: set[int] = set()
    apron_track_ids: set[int] = set()
    apron_matched_prediction_count = 0
    apron_unmatched_prediction_count = 0
    far_off_court_count = 0
    far_off_court_track_ids: set[int] = set()

    for frame_index in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_labels = gt_by_frame.get(frame_index, [])
        ignored_labels = ignored_by_frame.get(frame_index, [])
        pred_labels = pred_by_frame.get(frame_index, [])
        frame_matches = _match_frame(gt_labels, pred_labels, iou_threshold=iou_threshold)
        matched_pred_indexes = {pred_index for _, pred_index, _ in frame_matches}
        world_entries = prediction_world.get(frame_index, [])
        frame_has_full_gt = len(gt_labels) >= expected_players
        for pred_index, pred in enumerate(pred_labels):
            is_matched = pred_index in matched_pred_indexes

            excess_m: float | None = None
            track_id: int | None = None
            if pred_index < len(world_entries):
                track_id, world_xy = world_entries[pred_index]
                excess_m = _off_court_excess_m(world_xy, half_width_m=half_width_m, half_length_m=half_length_m)

            if excess_m is not None and 0.0 < excess_m <= off_court_apron_margin_m:
                apron_prediction_count += 1
                apron_frame_indexes.add(frame_index)
                apron_track_ids.add(track_id)
                if is_matched:
                    apron_matched_prediction_count += 1
                else:
                    apron_unmatched_prediction_count += 1

            if is_matched:
                continue
            if _overlaps_ignored(pred.bbox_xywh, ignored_labels, threshold=iou_threshold):
                continue
            false_positive_frames.append(frame_index)

            best_real_iou = _best_real_player_iou(pred.bbox_xywh, gt_labels)
            if best_real_iou > 0.0:
                near_miss_count += 1
                near_miss_ious.append(best_real_iou)
            elif not frame_has_full_gt:
                no_gt_frame_count += 1
            elif _inside_image_bounds(pred.bbox_xywh, image_width=image_width, image_height=image_height):
                true_spectator_count += 1
            else:
                outside_image_count += 1

            if excess_m is None or excess_m <= 0.0:
                continue
            off_court_count += 1
            off_court_track_ids.add(track_id)
            if excess_m > off_court_apron_margin_m:
                far_off_court_count += 1
                far_off_court_track_ids.add(track_id)

    return {
        "false_positive_frame_count": len(set(false_positive_frames)),
        "off_court_false_positive_frames": off_court_count,
        "off_court_false_positive_track_ids": sorted(off_court_track_ids),
        "false_positive_decomposition_schema_version": 2,
        "near_miss_false_positives": near_miss_count,
        "no_gt_frame_false_positives": no_gt_frame_count,
        "true_spectator_or_background_false_positives": true_spectator_count,
        "outside_image_false_positives": outside_image_count,
        "near_miss_localization": _near_miss_localization_summary(near_miss_ious),
        # Gate v2.1 fields -- additive, never replace the v1/v2 off-court
        # fields above. See module docstring "Gate v2.1" section.
        "off_court_decomposition_schema_version": 1,
        "off_court_apron_margin_m": off_court_apron_margin_m,
        "apron_off_court_excursion_prediction_count": apron_prediction_count,
        "apron_off_court_excursion_frame_count": len(apron_frame_indexes),
        "apron_off_court_excursion_track_ids": sorted(apron_track_ids),
        "apron_off_court_excursion_matched_prediction_count": apron_matched_prediction_count,
        "apron_off_court_excursion_unmatched_prediction_count": apron_unmatched_prediction_count,
        "far_off_court_false_positive_frames": far_off_court_count,
        "far_off_court_false_positive_track_ids": sorted(far_off_court_track_ids),
    }


def _off_court_excess_m(world_xy: list[float], *, half_width_m: float, half_length_m: float) -> float:
    """Euclidean distance a world point sits beyond the court template
    rectangle's lines; ``0.0`` if the point is inside (or on) the rectangle.
    Used by gate v2.1's apron-margin off-court split -- see module
    docstring.
    """
    x, y = float(world_xy[0]), float(world_xy[1])
    dx = max(0.0, abs(x) - half_width_m)
    dy = max(0.0, abs(y) - half_length_m)
    return math.hypot(dx, dy)


def _best_real_player_iou(bbox_xywh: tuple[float, float, float, float], gt_labels: list[Any]) -> float:
    if not gt_labels:
        return 0.0
    return max(_bbox_iou(bbox_xywh, gt.bbox_xywh) for gt in gt_labels)


def _inside_image_bounds(
    bbox_xywh: tuple[float, float, float, float],
    *,
    image_width: float | None,
    image_height: float | None,
) -> bool:
    if image_width is None or image_height is None:
        return True
    x, y, w, h = bbox_xywh
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    return 0.0 <= center_x <= float(image_width) and 0.0 <= center_y <= float(image_height)


def _near_miss_localization_summary(ious: list[float]) -> dict[str, Any]:
    return {
        "count": len(ious),
        "median_iou": _percentile(ious, 0.5),
        "p90_iou": _percentile(ious, 0.9),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    lower_value = ordered[int(lower)] * (upper - position)
    upper_value = ordered[int(upper)] * (position - lower)
    return lower_value + upper_value


__all__ = [
    "DEFAULT_FOUR_PLAYER_COVERAGE_THRESHOLD",
    "DEFAULT_IDF1_THRESHOLD",
    "DEFAULT_NEAR_MISS_FALSE_POSITIVE_RATE_THRESHOLD_V2",
    "DEFAULT_OFF_COURT_APRON_MARGIN_M",
    "build_scoring_report",
    "build_source_promotion_decision",
    "build_source_promotion_decision_v2",
    "build_source_promotion_decision_v2_1",
    "derive_track_source_id",
    "render_scoring_report_markdown",
    "score_tracks_against_person_ground_truth",
    "summarize_score_failure_modes",
    "summarize_score_failure_modes_v2",
    "summarize_score_failure_modes_v2_1",
]
