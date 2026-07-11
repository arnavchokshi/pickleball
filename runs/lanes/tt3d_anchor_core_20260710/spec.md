# LANE tt3d_anchor_core_20260710 — TT3D-pattern joint anchor search CORE (pure module, no integration)

Ground truth: runs/research_ball3d_20260709/RULINGS.md adopt item #1 + SYNTHESIS.md TT3D section;
runs/lanes/plan_nextmoves_20260710/PLAN.md rank 3; runs/lanes/dr_pipeline_20260710/FINDINGS.md §S4
(segment fates; anchor starvation). Read FIRST. Study the existing solver contract by READING
threed/racketsport/ball_arc_solver.py / ball_arc_chain.py (READ-ONLY — you may not edit them).

## HARD RULES
- No branches/commits/git add.
- FILE OWNERSHIP (new files only): threed/racketsport/ball_joint_anchor_search.py,
  tests/racketsport/test_tt3d_joint_anchor_search.py, runs/lanes/tt3d_anchor_core_20260710/**.
  FORBIDDEN (concurrent owners / integration comes later): ball_arc_chain.py, ball_arc_solver.py,
  ball_physics_fill.py, ball_ransac_arc_gate.py, ball_ukf_*, process_video.py, orchestrator.py.
  Never copy/fork solver code — call nothing, emit data structures matching the read contract.
- The module emits CANDIDATES only (hypotheses + costs + provenance); it cannot mark anything
  measured, touch trust bands, or alter defaults. Deterministic (seedable) behavior.
- Protected labels off-limits. Honest report.

## MISSION
Implement joint bounce/contact anchor-state search: given immutable 2D observations (u,v,conf,PTS),
camera model, and court/net planes, enumerate/optimize candidate anchor events (bounce time + state
via ray-plane constraints on the court plane, net-clearance constraints) as FREE VARIABLES with
robust bounds, producing ranked candidate anchor sets an arc solver could consume — the TT3D pattern
(reprojection-error-minimizing anchor search) adapted to our contract. Include: candidate
enumeration, ray-plane geometry, robust cost (huber on reprojection), refusal on insufficient
observations (typed), full provenance per candidate. Fixture tests: synthetic arcs w/ known bounce
(recovery within tolerance), occlusion gaps, degenerate/insufficient cases (refusal), determinism.
If an honest core is impossible without editing forbidden files, STOP with no-attempt + precise wall.
## REPORT
report.json via schema; API sketch + test names + fixture results; HONEST ISSUES; BEST-STACK DELTA
(c) none (research module, unwired).
