# LANE ballarc_scale_guard_20260715 — SPEC ONLY, DO NOT DISPATCH (coordinator sequencing required)

STATUS: DRAFT staged 2026-07-15 by the Track A manager per coordinator order. This lane is NOT
dispatched. Sequencing must be ruled by the coordinating manager first because Track C lanes
(coordwire/tbwire, 2026-07-15) are live on coordinate/timebase/ball-physics surfaces and Track C
owns `scripts/racketsport/process_video.py`.

## Objective
Fix the production-scale stall in the default ball 3D arc chain that has now killed the MOVE-1
41-rally head-to-head twice (2026-07-13 and 2026-07-15), and make full-game-scale ball_arc runs
bounded by construction.

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
3. A separate small fix ticket, NOT in this lane's fence: the frozen
   runs/research_pbv_reveng_20260712/compare_vs_pbvision.py crashes on the full 11-min pb export
   (PB physics pillar, scipy least_squares x0-out-of-bounds). The frozen original must remain
   byte-identical; a v2 copy with the bounded x0 clamp belongs in the fixing lane's own dir with
   a regression test against the single-rally scorecards (byte-identical rerun proof).
4. Focused tests: per-segment budget respected on a synthetic pathological pool; no behavior
   change on the Wolverine internal-val clips (same arc outputs byte-identical when no budget
   trips); wide suite per standing rules.

## Acceptance
- Full 20,922-frame ball_arc completes (or degrades typed-partial) in <= 30 min CPU on the Mac or
  <= 10 min on one H100, with unchanged outputs on clips where no budget trips.
- Stall reproduction + root cause documented with measured per-segment times.
- No promotion claim: this is a correctness/scaling fix; VERIFIED=0 stands.

## File fence (STRICT)
- Owns: `threed/racketsport/ball_arc_solver.py`, `threed/racketsport/ball_arc_chain.py`, its own
  tests under `tests/racketsport/`, and `runs/lanes/ballarc_scale_guard_20260715/**`.
- MUST NOT touch: `scripts/racketsport/process_video.py` (Track C integration owner),
  `threed/racketsport/ball_physics3d.py` and frame-times/timebase surfaces (tbwire),
  coordinate-contract files (coordwire), `runs/research_pbv_reveng_20260712/**` (frozen evidence).
- BEST-STACK DELTA expectation: (c) none, unless the guard adds a config knob — then a PENDING
  best_stack entry with default preserving current behavior.
