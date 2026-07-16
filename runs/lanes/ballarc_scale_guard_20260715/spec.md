# LANE ballarc_scale_guard_20260715 — DISPATCHED 2026-07-16 (coordinator GO; Codex lane, gpt-5.6-sol high)

STATUS: DISPATCHED per coordinator sequencing ruling 2026-07-16 (Track C's remaining work is a
doc-only window-close; inflight re-checked at dispatch — no code collision with this fence).

## HARD RULES
- No branches, no commits (manager rules on the report and commits). Stay inside the FILE FENCE
  below. pb.vision-derived artifacts are R&D reference ONLY (never GT, never training data).
- Honest reporting; `VERIFIED=0` binding; this is a correctness/scaling fix, not a promotion.
- Run the WIDE suite (`MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q`) before
  claiming PASS. KNOWN PRE-EXISTING failures you may treat as pre-existing ONLY by verifying they
  fail identically on stash/HEAD: 4 ball_physics_fill failures (concurrent tbwire eager
  empty-frame-times fallback) + up to 8 sandbox socket-bind denials (2026-07-15 coordwire close
  evidence in runs/manager/inflight_lanes.md). Any NEW failure = your defect.
- Anti-passive-wait: long local solves must be nohup'd/polled with bounded foreground loops; ending
  your turn to wait = lane death.

## Objective
Fix the production-scale stall in the default ball 3D arc chain that has now killed the MOVE-1
41-rally head-to-head twice (2026-07-13 and 2026-07-15), and make full-game-scale ball_arc runs
bounded BY CONSTRUCTION: a bounded per-segment wall-clock guard whose timeout is a LOUD TYPED
outcome — a timed-out segment must surface as an explicit degraded/missing reason per the trust
contract (Section 1.4 provenance rules), NEVER a silent skip or a silently-absent segment.

Evidence to read first:
- runs/lanes/pbv11_headtohead_20260713/rerun_20260715/pyspy_stall_evidence.md (three concurring
  stack captures: `_select_candidates_for_segment` -> RK4 `predict`, segment_id 7, ~5.16s gap at
  1/240s across a large unassigned pool, GIL-bound pure Python, ~1240 substeps per predict call)
- runs/lanes/pbv11_headtohead_20260713/rerun_20260715/vm_pull_partial/ (full-game ball chain
  artifacts: 20,922-frame ball_track.json, ball_candidates.json, bounce candidates — the exact
  reproduction inputs; WASB chain itself scaled fine at ~37 min)
- runs/HANDOFF_20260714.md §3.4 + the 2026-07-15/16 correction addendum (prior attempt's death
  previously attributed to the Fable spend limit; the stall is the likelier proximate cause)

## Deliverables
1. Reproduce the stall LOCALLY (CPU-only, no GPU needed) from the pulled artifacts: drive
   `run_default_ball_arc_chain` / `solve_ball_arc_track` directly on the 20,922-frame inputs and
   confirm segment 7's pathology (why a ~5.16s gap with a large candidate pool enters
   candidate-association at all; measure per-segment wall time distribution).
2. Root-cause analysis: candidate-pool size × RK4 substep count blow-up; evaluate (a) bounded
   per-segment wall-clock guard with typed `segment_budget_exceeded` degradation (arc stays
   missing/partial for that segment, never a silent hang), (b) step-size/adaptive integration or
   vectorized (numpy) integration for `predict`, (c) pool pre-filtering before association.
3. REGRESSION TEST BUILT FROM THE PULLED REAL ARTIFACTS (mandatory): a checked-in test that
   drives the solver on a trimmed-but-real slice of the salvaged 20,922-frame inputs reproducing
   the segment-7 pathology class, asserting (a) the guard trips within budget, (b) the outcome is
   the loud typed timeout (explicit degraded/missing reason present in the artifact), (c) never a
   silent skip. Trim only as much as needed for CI-speed; document the trim rule.
4. Focused tests: per-segment budget respected on a synthetic pathological pool; byte-identical
   arc outputs on the Wolverine internal-val clip when no budget trips; wide suite per HARD RULES.
   (The compare_vs_pbvision harness crash is explicitly NOT in this lane — separate lane
   pbv_harness_v2_20260715 owns it.)

## Acceptance (coordinator-ruled shape)
- THE DELIVERABLE: the 697s demo video's full ball_arc (from the salvaged real inputs at
  runs/lanes/pbv11_headtohead_20260713/rerun_20260715/vm_pull_partial/pbvision_11min_20260713/)
  completes CPU-side on this Mac within a sane bound (target <= 30 min) OR fails loudly per-segment
  with typed timeout outcomes — proven by actually running it end-to-end locally, NO GPU.
- Segment-7 diagnosis: WHY the association pool explodes (measured pool sizes + per-segment wall
  distribution), documented in the lane REPORT with numbers.
- Unchanged outputs where no budget trips (byte-identical Wolverine proof).
- No promotion claim: correctness/scaling fix; VERIFIED=0 stands.

## Mandatory structured report (report.json via the lane schema)
objective_result PASS/FAIL vs the acceptance above; full_suite counts with pre-existing-failure
proof; measured per-segment timings before/after; honest_issues; artifacts under this lane dir.
BEST-STACK DELTA: expected (c) none; if the guard adds a config knob, add a PENDING best_stack
entry whose default preserves current behavior and say so explicitly.

## File fence (STRICT)
- Owns: `threed/racketsport/ball_arc_solver.py`, `threed/racketsport/ball_arc_chain.py`, its own
  tests under `tests/racketsport/`, and `runs/lanes/ballarc_scale_guard_20260715/**`.
- MUST NOT touch: `scripts/racketsport/process_video.py` (Track C integration owner),
  `threed/racketsport/ball_physics3d.py` and frame-times/timebase surfaces (tbwire),
  coordinate-contract files (coordwire), `runs/research_pbv_reveng_20260712/**` (frozen evidence).
- BEST-STACK DELTA expectation: (c) none, unless the guard adds a config knob — then a PENDING
  best_stack entry with default preserving current behavior.
