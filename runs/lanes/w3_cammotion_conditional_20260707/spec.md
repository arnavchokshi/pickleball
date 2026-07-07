# LANE w3_cammotion_conditional_20260707 — camera-motion becomes motion-CONDITIONAL by default (wave-3 #6)

## HARD RULES (violating any = lane rejected)
- You are Codex lane `w3_cammotion_conditional_20260707`. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- No git branches/commit/push. Write runs/lanes/w3_cammotion_conditional_20260707/commit_manifest.md for the manager.
- OWNED files (edit only these + new files under tests/racketsport/ and runs/lanes/w3_cammotion_conditional_20260707/):
  `threed/racketsport/camera_motion.py`, `scripts/racketsport/estimate_camera_motion.py`, `scripts/racketsport/process_video.py` (you are its SOLE owner this wave), `tests/racketsport/test_camera_motion.py`, `tests/racketsport/test_process_video.py`.
- FENCED (do not edit; propose diffs if needed): `scripts/racketsport/remote_body_dispatch.py`, `scripts/fleet/*`, `threed/racketsport/roboflow_corpus.py`, `threed/racketsport/ball_tracknet_cvat_dataset.py`, `threed/racketsport/placement.py`, `threed/racketsport/pose_temporal.py`, grounding/refine code.
- Protected data: 4 eval clips EVAL-ONLY (frames/artifacts fine; Outdoor/Indoor labels never). Held-out pwxNwFfYQlQ / vQhtz8l6VqU nowhere.
- Honest reporting; measured numbers only. WIDE suite per standard rule: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q --ignore=tests/racketsport/court_finding_technology_benchmark.py` + focused suites; classify residual failures REAL / PRE-EXISTING (prove at HEAD) / SANDBOX-SUSPECT / CROSS-LANE-SUSPECT.
- importorskip("torch"); new CLI ⇒ direct-CLI reference test same-lane; no new root .md. `.venv/bin/python` always.
- Read first: NORTH_STAR PART 0 + IV; BUILD_CHECKLIST last ~15 (esp. [P2-1 CAMMOTION RULED], [WAVE2-INTEGRATION], [INTEGRATION RULED] — the kill-criterion history).

## CONTEXT (established)
- The hardened camera-motion module LANDED in wave 2 (person-masked LK + MAD flow-track filter + temporal smoothing, 50ms/frame): img1605 handheld beats legacy on ALL 3 proxies (inlier 0.7675→0.8949, jerk 2.621→2.453, court-line stability 14.84→11.75px); wolverine static guard improved (drift p95 0.563→0.427).
- BUT the integration lane's kill criterion fired on default-ON wiring: wolverine placement `jitter_after_p90_mean` 2.2440→2.2670 (~1% regression) → stage was wired FLAG-GATED `--enable-camera-motion`, DEFAULT-OFF. The wave-3 design note booked: motion-CONDITIONAL auto-enable (ON for genuinely moving cameras, OFF for static).
- RAFT-small remains not_enabled_pending_weights. A separate lane may prefetch weights during your run — DO NOT wire RAFT in this lane regardless; scope is the conditional enable only.

## THE DESIGN (manager-ruled shape)
1. **Cheap motion probe**: a subsampled pre-pass (reuse the module's masked-LK internals on every Nth frame / bounded frame budget) that yields a per-clip `motion_score` BEFORE the stage decision. Probe cost target: ≤~2s per eval clip on this Mac (measure and report actual).
2. **Auto decision**: default behavior becomes AUTO — enable the full camera-motion stage iff motion_score > threshold. `--enable-camera-motion` forces ON, new `--disable-camera-motion` forces OFF, AUTO is the new default. The decision record (`camera_motion_auto: {score, threshold, enabled, forced}`) must be persisted in the stage provenance/PIPELINE_SUMMARY so every run is auditable.
3. **Threshold with margin, not overfit**: calibrate on the 4 eval clips. img1605 (handheld) must auto-ON; wolverine (tripod) must auto-OFF. Burlington/outdoor: MEASURE their scores and report them with your recommendation for which side they land — the manager rules on those two; pick the provisional threshold to maximize margin between the known classes. KILL CRITERION: if no threshold separates img1605 from wolverine by ≥2× score ratio, STOP and report scores — do not shave a threshold into the gap.
4. **Static path bit-exactness**: when AUTO decides OFF, the pipeline behavior must be IDENTICAL to today's default-OFF path (same artifacts, same numbers — wolverine placement jitter_after_p90_mean must equal the default-OFF baseline exactly, modulo the probe's presence in provenance). The probe must not perturb any downstream stage input.

## ACCEPTANCE (all must hold, measured on the 4 eval clips' cached frames — CPU only, no GPU needed)
- img1605: auto decision = ON; the module's 3 handheld proxy wins retained within noise (inlier ≈0.8949, jerk ≈2.453, court-line ≈11.75px — report exact).
- wolverine: auto decision = OFF; placement jitter_after_p90_mean == default-OFF baseline 2.2440 (bit-identical stage path; show the equality).
- burlington + outdoor: scores measured + reported with recommendation (no acceptance bar — manager rules).
- Probe overhead measured, target ≤~2s/clip (report actual; a justified miss with a number is reportable, silent slowness is not).
- Decision record present in provenance for all 4 clips; forced ON/OFF flags work; direct-CLI reference test for any new/changed CLI surface.
- WIDE suite green per HARD RULES; existing camera_motion + process_video tests extended, not weakened.

## SELF-ITERATION + BOUNDED FIX AUTHORITY
Iterate to green. You may restructure within your owned files. If the probe needs a hook in a fenced file, deliver flag-complete within owned files + a deferred patch under runs/lanes/w3_cammotion_conditional_20260707/deferred_patches/.

## EVIDENCE
- runs/lanes/wave2_integration_20260706/ (kill-criterion measurement + how default-OFF was wired)
- The wave-2 P2-1 lane artifacts (hardened-module numbers; find under runs/lanes/, referenced from BUILD_CHECKLIST [P2-1 CAMMOTION RULED])
- threed/racketsport/camera_motion.py + tests at HEAD.

## STRUCTURED REPORT
objective_result vs acceptance; acceptance table (metric/baseline/after/target/verdict); per-clip scores incl. burlington+outdoor recommendation; changes file:line; full_suite + classification; HONEST ISSUES; NEXT; commit_manifest path; BUILD_CHECKLIST bullet DRAFT in the report (do not edit the checklist).
