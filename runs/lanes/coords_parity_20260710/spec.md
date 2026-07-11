# LANE coords_parity_20260710 — NS-01.4 typed-coordinate adoption remainder (PARITY-ONLY)

Ground truth: runs/lanes/plan_nextmoves_20260710/PLAN.md rank 5; runs/lanes/ns014_p22residual_20260709/
REPORT.md (coordinates.py slice); runs/lanes/dr_pipeline_20260710/FINDINGS.md §S5 (paddle bypasses
typed API; ad-hoc project_world_points import; undocumented court_Z0 literal). Read FIRST.

## HARD RULES
- No branches/commits/git add.
- FILE OWNERSHIP: threed/racketsport/coordinates.py, threed/racketsport/paddle_pose_fused.py,
  narrowly-required projection adapters in threed/racketsport/court_calibration.py and
  threed/racketsport/person_fast.py, matching tests, runs/lanes/coords_parity_20260710/**.
  FORBIDDEN: ball files (ballcand lane), process_video.py/orchestrator.py (spine lane), track.py,
  court trainer/eval files (court wave), web/, ios/, racket6dof.py.
- PARITY-ONLY: zero numerical behavior change beyond declared float tolerance. Every touched seam
  gets a frozen-fixture parity test (before/after outputs byte-or-tolerance identical on real
  artifact fixtures, e.g. from runs/lanes/w7_critique_20260709/wolv_world). Any unexplained delta =
  STOP that seam, land the rest. Do NOT claim improved paddle placement (hypothesis is unproven,
  needs GT). Raw values always preserved.
- py3.10 compatibility (fleet venvs — StrEnum bit us once; coordinates.py already fixed, keep it so).

## MISSION
1. paddle_pose_fused.py: consume typed coordinate spaces from coordinates.py — typed world-space
   declaration replacing the bare "court_Z0" literal (backwards-compatible artifact encoding:
   write BOTH old and canonical fields or keep old value + add canonical, choose the least-breaking,
   document), route its ad-hoc local project_world_points import through the canonical API.
2. court_calibration.py/person_fast.py: add typed-space adapters at the exact projection seams other
   stages call (declare raw vs undistorted vs reference conventions explicitly) WITHOUT changing math.
3. Wide suite at end (MPLBACKEND=Agg pytest tests/racketsport); failures proven pre-existing.
## REPORT
report.json via schema; per-seam parity evidence (fixture + tolerance + result); HONEST ISSUES;
BEST-STACK DELTA (c) none.
