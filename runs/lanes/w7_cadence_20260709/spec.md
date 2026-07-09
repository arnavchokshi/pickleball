# LANE w7_cadence_20260709 — per-stage cadence knobs + the owner stride-2 BODY default (binding ruling 2026-07-09)

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg; PYTHONDONTWRITEBYTECODE=1. Re-orient from main first (pipepolish just landed — best_stack is rev-7). You own: process_video.py, best_stack.json, the BODY frame-scheduling path (frame_rating/body_compute cadence plumbing), their tests. Do NOT touch: ios/** (owner surface — the 60fps capture half is the app's, NOT yours), web/**, server/**, mhr_decode*/hmr_deep/gate_check files. Artifacts under runs/lanes/w7_cadence_20260709/ only.

## OBJECTIVE (OWNER RULINGS 2026-07-09, binding — speed-cadence doctrine)
"App forces 60fps capture; BODY default stride-2 (30Hz); ball stays full-rate. Cadence = configurable best_stack-class defaults per stage; reducing calculation frequency is a sanctioned lever; every stage cadence is a tunable knob, never hardcoded."
1. Introduce per-stage cadence knobs resolved ONLY through best_stack.json: at minimum body.skeleton_stride (frames), ball full-rate pinned EXPLICITLY (ball.detection_stride=1 as a declared entry so the ruling is visible, not implicit), and a documented pattern for future stage cadences. CLI overrides per knob.
2. BODY DEFAULT stride-2: the skeleton/BODY heavy path processes every 2nd frame by default (30Hz effective at 60fps capture) — implement at the frame-scheduling level (frame_compute_plan/BODY scheduling), NOT by dropping input frames globally (ball must still see every frame). Interpolation/carry behavior for skipped skeleton frames follows the existing house pattern for non-computed frames (reuse; if ambiguous, prefer the existing render-interp machinery and SAY SO).
3. Contact-dense scheduling and mesh byte-budget interplay: stride applies to the BASE cadence; event/contact-dense boosts remain (they already select extra frames) — the stride must not silently disable events-before-frames. Test this interaction explicitly.
4. Honest metrics: PIPELINE_SUMMARY stages[body] gains effective_stride + scheduled-vs-total frame counts so cadence is never silent.
5. BEST-STACK DELTA: new cadence entries WIRED_DEFAULT (stride-2 per the owner ruling — this is a pre-authorized default flip; cite the ruling in the entry notes), revision bump.

## SELF-VERIFICATION
Acceptance through process_video (mocked entry tests). Full blast radius: test_process_video*.py, test_frame_rating.py, test_body_compute*.py (if exists), test_best_stack_*.py + one full tests/racketsport census at the end (no fail-fast; the 6 loopback-bind sandbox failures are known-preexisting). Fix what you introduce.

## REPORT
Self-write runs/lanes/w7_cadence_20260709/report.json (lane_report.schema.json structure): acceptance rows 1-5 w/ exact keys, BEST-STACK DELTA precise, census numbers, honest_issues (esp. any accuracy-relevant caveat the stride introduces — flag, don't hide), next.
