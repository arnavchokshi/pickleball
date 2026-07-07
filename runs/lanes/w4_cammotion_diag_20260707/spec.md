# LANE w4_cammotion_diag_20260707 — READ-ONLY diagnosis: camera-motion probe-context discrepancy

## OBJECTIVE
Wave-3 landed the camera-motion motion-conditional AUTO probe (`estimate_camera_motion_probe`,
threshold `CAMERA_MOTION_AUTO_THRESHOLD = 2.5`, `threed/racketsport/camera_motion.py:17` — re-grep,
lines drift). In the offline lane harness the handheld clip img1605 scored **53.7** (AUTO → ON). In
the wave-3 decisive fresh-GPU pipeline run the SAME clip's in-pipeline probe scored **0.329** →
AUTO OFF, so the clip ran uncompensated (it still passed its gates — this is a correctness bug in
the probe context, not a gate failure). Your job: find the ROOT CAUSE of the 0.329-vs-53.7
discrepancy with executable evidence, and design the fix. **You change NO production code.** This is
the general failure class from wave 3: a lane-local harness that does not share the production code
path proves nothing — find exactly where the contexts diverge.

## EVIDENCE TO READ FIRST
- `runs/lanes/w3_cammotion_conditional_20260707/` — the lane that landed the feature (report.json,
  its harness, its measured probe scores: img1605 53.7, wolverine 0.13, burlington 0.52, outdoor 0.57).
- `runs/lanes/p21_cammotion_20260706/` — the original camera-motion lane (module design, proxies).
- The wave-3 decisive-run artifacts: search `runs/` for the freshest per-clip pipeline outputs
  containing camera_motion probe telemetry for img1605 (PIPELINE_SUMMARY.json and/or placement
  artifacts persist the AUTO decision + score; the closeout bullet says the 0.329-vs-53.7 repro is
  banked — locate it; if you cannot find a banked repro artifact, say so honestly and reconstruct
  the in-pipeline number yourself per below).
- Code: `threed/racketsport/camera_motion.py` (`estimate_camera_motion_probe`), its call site in
  `scripts/racketsport/process_video.py` (grep `estimate_camera_motion_probe`), and the offline
  harness the w3 lane used to get 53.7.

## HYPOTHESES TO TEST (confirm/refute each with evidence — do not stop at the first plausible one)
1. Frame-window mismatch: pipeline probes a different frame range (e.g. first-N frames, possibly
   static) than the offline harness (e.g. mid-clip motion window).
2. Input-scale/preprocessing mismatch: downscale, undistort state, color space, or person-mask
   availability differs at probe time in-pipeline vs offline.
3. FPS/PTS handling: frame indexing vs PTS (iPhone VFR) changes which frames the probe compares.
4. Units/normalization: the probe statistic is normalized differently (per-frame vs per-second,
   px vs px/frame) in the two contexts (0.329 vs 53.7 is ~163× — suspiciously scale-like).
5. Different entry point: the pipeline calls a different wrapper that gates/clamps the score.

## DELIVERABLES (all under runs/lanes/w4_cammotion_diag_20260707/ — your only writable area)
1. `REPORT.md` — root cause with file:line evidence; state which hypothesis held and how you
   falsified the others.
2. `repro/` — an executable repro: a script (stdlib+repo imports, `.venv/bin/python`) that produces
   the in-pipeline-context score AND the offline-context score for img1605 from the same inputs,
   demonstrating the divergence deterministically. If GPU-only artifacts are required and absent
   locally, reproduce the CONTEXT divergence on whatever local frames exist (eval_clips) and state
   the limitation honestly.
3. Fix design in REPORT.md: exact files/functions to change, the shape of the fix, expected
   post-fix probe scores on all 4 eval clips, and the risk analysis — the static-clip path is
   bit-exact today (wolverine/burlington/outdoor AUTO OFF): the fix MUST preserve static-clip
   bit-exactness, and you must say how the fix lane will prove that.
4. The executable defect proof a future adversarial verifier would run (a failing assertion that
   passes only when the probe context is fixed).

## HARD CONSTRAINTS
- READ-ONLY on all production/test files. Write ONLY inside your lane dir.
- img1605 clip: locate its eval clip dir by grepping the w3 lane report for the clip id/path — do
  not assume a path. State the path you found as a CHECK result.
- Do not touch anything under `ios/`, `runs/manager/`, eval label files, or the held-out ledger.
- No git operations. No network. `.venv/bin/python` for any execution.

## SELF-VERIFICATION (mandatory)
- Your repro script runs clean twice with identical output.
- `git status --porcelain` shows NO modified tracked files (only your lane-dir additions).

## SELF-ITERATION
Iterate until the root cause is pinned with evidence OR you hit a genuine wall (e.g. the decisive
in-pipeline score cannot be reproduced without GPU artifacts that don't exist locally) — then STOP
and report the wall precisely with what a GPU lane must capture to close it. Do not guess.

## STRUCTURED REPORT
objective_result PASS = root cause pinned + repro runs + fix design complete. PARTIAL = divergence
characterized but root cause not uniquely pinned (say what distinguishes remaining candidates).
Include in `honest_issues` anything that smells wrong beyond this bug. full_suite: not applicable
(read-only) — report 0 failed with the note "read-only diagnosis, no code changed".
