# LANE pbv_harness_v2_20260715 — compare_vs_pbvision v2: fix the full-11-min-export crash (Codex lane)

STATUS: DISPATCHED 2026-07-16 per coordinator ruling (second in sequence after
ballarc_scale_guard_20260715; fully file-disjoint from it). This harness gates ALL future MOVE-1
scoring — it must be green on the full 11-minute export before any third GPU attempt is considered.

## HARD RULES
- No branches, no commits. FILE FENCE below is strict. pb.vision export/video = R&D reference ONLY
  (never GT, never training data, never redistributed). VERIFIED=0 binding; the harness is a
  competitor-reference diagnostic, never promotion evidence.
- THE FROZEN ORIGINAL runs/research_pbv_reveng_20260712/** MUST REMAIN BYTE-IDENTICAL. You fix a
  COPY (v2) inside THIS lane dir. Verify at the end: `git status` shows zero modifications under
  runs/research_pbv_reveng_20260712/ and an md5 of compare_vs_pbvision.py unchanged vs HEAD.
- Wide suite not required (no product-source edits — everything lives in this lane dir), but run
  your own v2 test file green and report real exit codes.

## FILE FENCE
Owns ONLY `runs/lanes/pbv_harness_v2_20260715/**` (v2 script copy, its tests, scorecards, report).
Read-only inputs: `data/pbvision_11min_20260713/cv_export.json`,
`runs/lanes/pbv11_headtohead_20260713/rerun_20260715/vm_pull_partial/**`,
`runs/research_pbv_reveng_20260712/**` (frozen), `runs/research_pbv11_20260713/**` (frozen).

## The defect
`compare_vs_pbvision.py` crashes on the full 42-rally 11-min export inside `physics_pillar(pb)` →
`physics_fit` → scipy `least_squares`: "Initial guess is outside of provided bounds" (first-ever
full-scale invocation; single-rally exports never triggered it). Reproduce first:
`.venv/bin/python runs/research_pbv_reveng_20260712/compare_vs_pbvision.py --pb-export
data/pbvision_11min_20260713/cv_export.json --ours
runs/lanes/pbv11_headtohead_20260713/rerun_20260715/vm_pull_partial/pbvision_11min_20260713/ball_track.json
--ours-calibration .../court_calibration.json --frame-offset auto --output /tmp/x.json` (EXIT 1).

## Deliverables
1. `compare_vs_pbvision_v2.py` in this lane dir: minimal, documented fix — diagnose WHICH segment
   group produces the out-of-bounds x0 (likely degenerate/short/extreme segments at full scale) and
   fix it honestly: clamp x0 into bounds AND/OR typed per-segment skip with an explicit
   `physics_fit_skipped` reason recorded in the scorecard (never a silent drop; counts must be
   reported). Also fix the copied test's `parents[3]` root-path bug in YOUR copy of the tests.
2. Regression proof A (frozen-behavior preservation): run v2 on the ORIGINAL single-rally inputs
   that produced the frozen scorecards (runs/research_pbv_reveng_20260712/scorecard_raw_arc.json,
   scorecard_failclosed_world.json, scorecard_pb_only.json — use the pb export path recorded inside
   them; if that original single-rally export is absent on disk, say so honestly and instead prove
   v2==v1 numerically on a constructed single-rally slice of the 11-min export where v1 does not
   crash). v2 output must match the frozen numbers exactly (or byte-identical modulo a version
   field); any difference = FAIL, investigate.
3. Regression proof B (full scale green): v2 runs the FULL 11-min export end-to-end EXIT 0 twice
   with byte-identical output (determinism), both PB-only and with our salvaged 2D ball_track;
   scorecards saved in this lane dir. Cross-check: its 2D coverage numbers must agree with
   runs/lanes/pbv11_headtohead_20260713/rerun_20260715/scorecard_2d_salvage.json (ours 0.7835 /
   pb 0.7564 under that file's stated definitions) or the difference must be explained by
   definition deltas in the report.
4. A test file in this lane dir covering: the crash input class (regression), determinism, and the
   typed-skip accounting. All green with real exit codes in the report.

## Mandatory structured report (report.json via lane schema)
objective_result PASS/FAIL; the diagnosed root cause with the offending segment characteristics;
frozen-original byte-identity proof; regression A + B results with real exit codes; honest_issues.
BEST-STACK DELTA: (c) none — diagnostic tooling only, no stack surface.
