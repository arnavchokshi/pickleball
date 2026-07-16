# LANE ballarc_anchorfusion_20260716 — STEP 1+2: audio-onset soft anchors → arc recovery (Codex, gpt-5.6-sol high)

DISPATCHED 2026-07-16 by the Track A manager under an OWNER DIRECTIVE: recover arcs on the demo
video by fusing the anchor evidence we already have. Baseline to beat: **1/188 segments fit**
(runs/lanes/ballarc_scale_guard_20260715/full_guard5_r4_metrics.json). Root cause on record: 20
auto-bounce anchors across 697s → anchor-gap segments balloon to 104.7s with 8,381-candidate pools
(same file + REPORT.md). The guard (adopted, af6b8d40f) stays armed throughout.

## HARD RULES
- FILE FENCE: `threed/racketsport/ball_arc_solver.py`, `threed/racketsport/ball_arc_chain.py`,
  their tests under tests/racketsport/, and this lane dir. NO edits to
  `scripts/racketsport/process_video.py` or any other runner/stage file — runner plumbing belongs
  to Track C's refinedstage lane; your chain-level API must be ADDITIVE and DEFAULT-OFF so they can
  consume it later. Document the API surface in your report for their handoff.
- pb.vision demo data = R&D reference ONLY (never GT, never training, never redistributed).
- HONEST PROVENANCE IS THE WHOLE POINT: the audio onsets
  (runs/lanes/ball_evidence_q_20260713/pbvision_11min_audio_onsets_v2.json, 2,309 onsets,
  `not_gate_verified`, review-only quality) enter as a NEW soft anchor class that may ONLY propose
  segment SPLIT boundaries. They must NEVER: pin z=radius or any bounce/contact physics, assert
  event type, or count as bounce evidence inside the flight-sanity gate. Physics plausibility is
  judged by the SAME unmodified flight-sanity gate as before.
- Every segment created via a soft split carries typed provenance (anchor_class=audio_onset_soft,
  onset ids, corrected_time_s, selection-rule id). Segments still exceeding budget keep the loud
  typed timeout (guard unchanged, 5s production default).
- NO THRESHOLD SHOPPING: pre-register 2-3 onset-selection presets BEFORE scoring (document the
  rule: e.g. feature/confidence floor + minimum inter-split spacing >=1.5-2.0s + only within
  rally-active spans), report ALL presets' results. Do not iterate presets against the coverage
  number. ~3.3 onsets/sec raw density means naive splitting is degenerate — say so and handle it.
- VERIFIED=0: arc FIT is not accuracy. No GT exists here. Report fit coverage + physics
  plausibility + provenance only; all outputs preview-band. No promotion claims.
- No branches, no commits (manager commits after ruling). Wide-suite rule as in the guard lane
  spec (8 sandbox socket failures are known-pre-existing at HEAD).

## STEP 1 (primary): full 697s salvaged demo inputs
Inputs: runs/lanes/pbv11_headtohead_20260713/rerun_20260715/vm_pull_partial/pbvision_11min_20260713/
(ball_track.json, ball_candidates.json, bounce candidates) + the onsets artifact above + the
guard lane's driver scaffolding (run_probe.py / metrics jsonl format — reuse it).
Deliverables/metrics (vs the frozen baseline file, same definitions):
1. segments_fit count (baseline 1) and **fit coverage %** — define BOTH: (a) fraction of
   in-rally frames inside fitted segments, (b) fraction of total segments fit. State definitions
   in the metrics artifact.
2. wall time (must stay bounded; target <=45 min CPU full run with guard armed).
3. physics-violation count from the unmodified flight-sanity gate: **MUST be 0** — any violation
   introduced by soft splits = kill that preset.
4. segment-duration distribution before/after (the 104.7s monster must resolve to rally-scale).
5. per-preset table + the pre-registered selection rules.
6. Byte-identity regression: with NO soft anchors supplied, chain outputs byte-identical to the
   adopted-guard behavior (prove on Wolverine and on a demo-run slice).
## STEP 2 (contingent on Step 1 showing recovery): the no-audio boundary
Internal eval card Wolverine ONLY (Outdoor/Indoor protected — do not touch). Cards have NO audio:
prove the mechanism is audio-gated — no soft anchors available → outputs byte-identical, no
regression. State the boundary plainly in the report: this recovery path currently applies only to
audio-bearing captures (all product captures have audio; internal cards do not).

## Kill rules
- Any flight-sanity violation attributable to a soft split → that preset is dead; report it.
- Wall time unbounded or guard bypassed → stop.
- Provenance missing on any split segment → stop and fix before scoring.

## Mandatory structured report (report.json per docs/racketsport/lane_report.schema.json)
objective_result vs "fit coverage materially above 1/188 with 0 physics violations at >=1
pre-registered preset"; the full per-preset metrics table; API surface doc for Track C; honest
issues (incl. the review-only quality of onsets and what that means); BEST-STACK DELTA — expected
(b) PENDING default-off entry for the soft-anchor chain input, or (c) none if pure lane-dir driver.
