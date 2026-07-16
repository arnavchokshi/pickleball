# LANE pbv11_calsolve_20260716 — solved court calibration for the pb.vision demo video (Codex, gpt-5.6-sol high)

DISPATCHED 2026-07-16 by the Track A manager. GOAL: turn the owner's banked 4-corner seed into an
explicit SOLVED court calibration whose provenance class legitimately passes
`_court_calibration_needs_correction` in scripts/racketsport/process_video.py — so the metric
stages (tracking court filter, body_world, ball_world, virtual_world_metric) actually RUN on this
video. This gates MOVE-1 attempt #3.

## HARD RULES
- NO edits to scripts/racketsport/process_video.py, threed/racketsport/ball_arc_*.py, or ANY
  pipeline/stage code. You may ONLY add files under your lane dir and under
  runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/ (the banked-seed home).
  Existing repo CLIs/library code are your tools, not your patch surface.
- FORBIDDEN: using pb.vision's own `camera` block from cv_export.json as any input to our
  calibration (competitor output must never seed our pipeline). It may be READ at the very end
  ONLY for a diagnostic delta report (our solve vs theirs), clearly labeled reference-only.
- FORBIDDEN: weakening/bypassing the correction gate, hand-editing grades/reasons, or fabricating
  evidence values. The solve must be REAL: actual correspondences, actual residuals.
- Honest provenance: everything stays `corrected_unverified` / preview-band. This enables metric
  output; it is NOT a CAL accuracy promotion. VERIFIED=0 binding.
- No commits, no branches.

## Facts established by the manager (verify, then build on them)
- Owner 4-corner seed (validated, overlay-proof):
  runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_corners_seed.json
- Preflight with that seed (out dir runs/lanes/pbv11_headtohead_20260713/rerun_20260715/calseed_preflight/):
  court_line_evidence aggregate accepts ALL required court lines (near/far baselines, sidelines,
  nvz lines, centerlines, ground net) at mean residual 2.65px / p95 8.62px over 300 frames, but
  `auto_calibration_ready:false` solely due to `missing_top_net`; top_net observation is refused
  by design when intrinsics.source is 4-corner-estimated (`top_net_untrusted_intrinsics`,
  threed/racketsport/court_auto_evidence.py:477).
- Gate predicate (process_video.py): blocks tracking only when calibration is
  unverified/estimated (token list incl. `estimated_from_declared_court_corners`,
  `process_video_manual_court_corners`) AND line evidence is not ready. Wolverine eval clips pass
  because their explicit `--court-calibration` metric15pt solve carries non-token intrinsics
  provenance — that is the input class to produce here.
- Video: data/pbvision_11min_20260713/source_video.mp4 (1280x720@30, sha 272a2132..., R&D
  reference ONLY — never GT/training).

## Deliverables
1. A solved calibration file `owner_cal_seed/court_calibration_solved.json` for this video built
   from REAL evidence: derive >=10 point correspondences from accepted line-evidence
   intersections (and/or run the repo's existing 15-pt/metric solve tooling — inspect
   scripts/racketsport/validate_metric_calibration_15pt.py, build_calibration_from_review.py,
   solve_net_anchor_court.py, calibrate.py and use whichever fits; sample MORE frames than the
   300-frame preflight if it helps stability). Intrinsics must be genuinely estimated from the
   correspondence set (document conditioning/assumptions honestly — planar-scene limits included),
   with an honest `source` naming this method (do NOT copy a trusted-profile label).
2. PROOF THE GATE OPENS: rerun the manager's exact preflight command but with
   `--court-calibration owner_cal_seed/court_calibration_solved.json` (fresh --out under YOUR lane
   dir): show court_correction_task.json is NOT emitted (or blocked_downstream empty), tracking
   stage status is not `blocked`, with real exit codes. ALSO show trust bands remain
   preview/unverified-class (we are not smuggling authority).
3. Validation: repo validator CLIs green on the produced file (whichever apply); reprojection
   residuals of the solve reported vs the line-evidence residuals; sanity overlay image rendered
   from the solved calibration (court model + net projection drawn on a frame) saved in the lane
   dir for the manager's visual check.
4. Diagnostic-only appendix (clearly reference-only): delta of our solved camera vs pb.vision's
   exported camera (position/orientation delta) — context for the head-to-head, never an input.
5. `report.json` per docs/racketsport/lane_report.schema.json: objective_result vs deliverable 2's
   binary gate-opens proof, honest_issues (esp. planar-intrinsics conditioning), BEST-STACK
   DELTA (c) none.

## Kill rule
If a genuinely-solved calibration cannot open the gate honestly (e.g., intrinsics from planar
evidence remain too ill-conditioned and every honest source label still trips the token list),
STOP and report exactly that with the evidence — do NOT force it. That outcome routes to a design
decision (net-post owner taps for a height anchor, or a gate-policy discussion owned by the spine
team), not to a workaround.
