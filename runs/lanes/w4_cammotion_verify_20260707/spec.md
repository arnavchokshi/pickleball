# LANE w4_cammotion_verify_20260707 — ADVERSARIAL VERIFY of the cammotion orientation fix (fresh eyes, try to BREAK it)

## OBJECTIVE
The w4_cammotion_fix lane changed `threed/racketsport/camera_motion.py` (explicit decode
orientation policy + telemetry) and self-reports PASS (img1605 production probe 53.70515 → AUTO
ON; static clips bit-exact at baseline scores; defect proof green). Your job is to REFUTE it. You
are not here to confirm — you are here to find the way it is wrong. Wave-3 precedent: round-1
verification found a fix gating a layer the GPU path never consulted; round-2 found an unfailable
gate. Hunt for THIS fix's version of those. Verdict is CONFIRMED-GOOD only if every attack fails.

## READ FIRST
- `runs/lanes/w4_cammotion_fix_20260707/` (report.json, acceptance_probe_report.json,
  img1605_fixed_probe_pipeline_summary.json) and the diff: `git diff HEAD --
  threed/racketsport/camera_motion.py tests/racketsport/test_camera_motion.py` (the fix is
  UNCOMMITTED in the working tree).
- `runs/lanes/w4_cammotion_diag_20260707/REPORT.md` + `repro/` (root cause + the --assert-fixed proof).

## ATTACK SURFACE (execute each; add your own)
1. REACHABILITY (the r1 failure class): trace the ACTUAL pipeline path from
   `scripts/racketsport/process_video.py`'s camera-motion AUTO decision (read-only on fenced
   files) down to the probe: does every VideoCapture open on that path route through the new
   orientation helper? Any bypass (a different module opening the video for the probe's frames, a
   pre-decoded-frame seam) = DEFECT with file:line proof.
2. SET-FAILURE SEMANTICS: on builds where `cap.set(CAP_PROP_ORIENTATION_AUTO, ...)` returns False
   or is ignored, does the code DETECT the readback mismatch and surface it (fail-closed /
   telemetered), or silently proceed possibly-wrong? Write an executable proof (monkeypatched
   capture simulating a refusing build) showing the actual behavior; silent-wrong = DEFECT.
3. STATIC INVARIANT, INDEPENDENTLY: re-derive (do not trust the lane's JSON) the pre/post decode
   identity on the 3 static eval clips with your own script: first-N-sampled-frame hashes + dims +
   probe scores vs the banked baselines (0.128813 / 0.523815 / 0.566526). Any drift = DEFECT.
4. VACUOUS-TEST CHECK: copy `camera_motion.py` into your lane dir, revert the orientation policy
   in the COPY, and run the new unit tests against the mutated copy (importable via sys.path
   shim): if the new tests still PASS against the reverted copy, they are vacuous = DEFECT.
5. DEFECT-PROOF INTEGRITY: regenerate the img1605 production-probe summary YOURSELF by invoking
   the same production entry the fix lane claims it used; feed YOUR summary to the diagnosis's
   `--assert-fixed` proof. Score mismatch vs 53.70515 beyond float noise, or any sign the lane's
   summary was not produced by the production path = DEFECT.
6. TELEMETRY COMPAT: confirm the additive telemetry didn't break any existing consumer of the
   probe/camera_motion payload keys (grep consumers; run their tests).

## HARD CONSTRAINTS
READ-ONLY on all production/test files (mutations happen only on COPIES inside
`runs/lanes/w4_cammotion_verify_20260707/`). No git operations. `.venv/bin/python`. Never touch
eval labels / ledger / ios/ / runs/manager/.

## STRUCTURED REPORT
objective_result: PASS means YOUR VERIFICATION COMPLETED — state the verdict separately and
prominently in the acceptance table: VERDICT = CONFIRMED-GOOD or DEFECTS-FOUND. For each attack:
what you did, executable artifact path, outcome. Every claimed defect ships a runnable proof
script + exact repro command. honest_issues: attacks you could NOT execute and why (e.g. GPU-build
OpenCV differences unreachable locally — say precisely what the wave-close GPU proof must check).
