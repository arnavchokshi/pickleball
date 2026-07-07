# LANE w3_slidediag_20260707 — foot-slide MAX gate overshoot: outlier-frame DIAGNOSIS (wave-3 #1; diagnosis ONLY)

## HARD RULES (violating any = lane rejected)
- You are Codex lane `w3_slidediag_20260707`. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- THIS IS A DIAGNOSIS LANE: you own NO repo files. You may NOT edit any file outside runs/lanes/w3_slidediag_20260707/. All analysis scripts you write live in that dir and run from there. No git commit/branch/push.
- Protected data: 4 eval clips EVAL-ONLY; you READ pipeline artifacts/metrics freely (that is not label-touching), but Outdoor/Indoor LABELS are never read or written. Held-out videos pwxNwFfYQlQ / vQhtz8l6VqU appear nowhere.
- GATE THRESHOLDS ARE FROZEN. Any proposal that moves `foot_slide_gate`'s 0.03 bar, guard constants, or review floors to "pass" is auto-rejected. Fix-at-source proposals only.
- Honest reporting: if the banked evidence cannot answer a question offline, say so precisely and specify what instrumentation a fresh GPU run must persist — that is a valid PARTIAL, and vastly better than a guessed mechanism.
- No wide-suite requirement (you change no code) — but every number in your report must carry the artifact path + jq/python snippet that reproduces it.
- Always `.venv/bin/python`. Read first: BUILD_CHECKLIST.md last ~15 bullets (esp. [CLOSING RUN RULED], [MAD A/B RULED], [ROOTJUMP VERIFY RULED]).

## CONTEXT (established facts — do not re-derive)
- The GATED metric is `grounding_metrics.max_foot_lock_slide_m` (MAX, not p95) vs bar 0.03; blocker key `foot_slide_gate_failed`. It is computed INSIDE the GPU BODY stage on camera-frame samples that are NEVER persisted (worldhmr.py:280 area, per the wave-2 verify lane) — so the gated number itself cannot be recomputed offline. Offline per-frame evidence CAN localize which frames/persons drive the max where that evidence was persisted in the wave-2 runs.
- Fresh-worlds closing run (composed wave-2 code): burlington max 46.9→40.6mm (improved, still over 30), outdoor 40.5→56.0mm (WORSE), wolverine 0→18.4mm (passes but shifted), img1605 ~25.6mm (passes). p95 (`foot_lock_slide_p95_m`) is under 30mm EVERYWHERE → the failure is OUTLIER FRAMES, not distribution-wide.
- MAD bone-length smoothing was adversarially CLEARED (A/B: engages 0 frames on both clips, ON-vs-OFF delta ~1e-7m). Do not re-litigate MAD.
- PRIME SUSPECT (manager theory to confirm or refute, not to assume): the wave-2 placement 30fps frame-index fix redistributed placement error — corrected frame indexing moved where placement targets sit; guard counters scale with the shift (outdoor `divergence_snap` 478 + `speed_cap` 718 vs wolverine 22/83). Mechanism sketch: at specific frames the root-to-anchor residual grows → speed_cap/divergence_snap guards fire → the foot lock slips or snaps → MAX slide outliers at exactly those frames.
- Root-jump is WON (outdoor 55→0, burlington 24→1 marginal) — a fix must not regress it.

## OBJECTIVE — answer these, with numbers
1. **Localize the outliers**: for burlington + outdoor (and wolverine's 18.4 shift), identify the exact frames + person tracks driving slide values >30mm in the persisted per-frame evidence of the wave-2 runs. If the exact gated max frame is not offline-recoverable, localize the persisted-slide outliers and state the gap explicitly.
2. **Guard-fire alignment**: do `divergence_snap` / `speed_cap` fire events coincide (frame-level) with the slide outliers? Quantify the overlap (e.g., fraction of outlier frames within ±k frames of a guard fire).
3. **Mechanism at those frames**: what is happening — occlusion, direction change, track handoff, anchor jump, camera edge, placement target jump between consecutive anchors? Use the tracks/placement/anchor artifacts to characterize each outlier cluster.
4. **Suspect verdict**: does the evidence CONFIRM or REFUTE the placement-fix-redistribution theory? If refuted, what does the evidence support instead? (Compare pre-fix p06_freshworlds vs post-fix wave2_freshworlds per-frame where both persisted.)
5. **Cross-clip test**: why does outdoor get WORSE post-fix while burlington improves? Any theory must explain both signs, and wolverine's 0→18.4 shift.
6. **FIX DESIGN (proposal only)**: source-level fix with mechanism, predicted per-clip effect, risk to root-jump (must predict no regression: outdoor 0 jumps, burlington ≤1), and exactly how a fresh GPU run proves/refutes it. Thresholds frozen. Include what extra per-frame instrumentation the fix lane should persist so the wave-end GPU run is decisive (remember: the gated max lives GPU-side; the fresh run is the only gate proof).

## ACCEPTANCE
- Outlier frames+persons named for burlington AND outdoor with values + artifact paths; guard-alignment quantified; a mechanism narrative consistent with ALL THREE clips' deltas; suspect explicitly confirmed/refuted with the comparison that decides it; a one-page fix design; an instrumentation-request list.
- Every claim reproducible from named artifacts. No code changes anywhere.

## EVIDENCE (start here)
- runs/lanes/wave2_freshworlds_20260707/ (post-fix worlds, per-frame evidence, PIPELINE_SUMMARYs, grounding metrics)
- runs/lanes/wave2_mad_ab_20260707/ (true ON/OFF arms; guard counters per clip)
- runs/lanes/wave2_rootjump_verify_20260706/ (metric reconciliation max-vs-p95; counterfactual mechanics; residual-at-cited-frames analysis)
- runs/lanes/p06_freshworlds_20260706/ (PRE-fix baseline worlds: burlington 46.9 / outdoor 40.5)
- Code READ-ONLY for mechanism understanding: threed/racketsport/placement.py, pose_temporal.py, the grounding/footlock path, worldhmr orchestrator (do not edit).

## STRUCTURED REPORT
objective_result PASS (= all 6 questions answered w/ evidence) / PARTIAL (state exactly which question needs GPU-side data + the instrumentation list) / BLOCKED; findings per question; fix-design section; instrumentation-request list; artifact index; HONEST ISSUES; NEXT. No full-suite section needed (no code changes) — state "no code changed" explicitly.
