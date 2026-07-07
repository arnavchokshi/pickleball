# LANE w4_footattr_verify_20260707 — ADVERSARIAL VERIFY of the per-foot phase producer (fresh eyes, try to BREAK it)

## OBJECTIVE
The w4_footattr_fix lane added a BODY-skeleton-direct per-foot contact-phase producer
(`threed/racketsport/foot_contact.py`, UNCOMMITTED in the working tree) + a proposed fenced patch
(`runs/lanes/w4_footattr_fix_20260707/deferred_patches/process_video_body_phase_producer.patch`).
It self-reports: confident-phase counts EXACTLY matching the diagnosis predictions (127/186/33/10),
refine kill=false 4/4, offline slide ≤0.030 on all 4 clips, phaseverify harnesses green, 268/0
tests. Your job is to REFUTE it. Wave-3 precedent on THIS exact gate: r1 found vacuous surrogates +
a fallback leak; r2 found an unfailable gate (threshold-tuning by exclusion). Find this fix's
version. Verdict CONFIRMED-GOOD only if every attack fails.

## READ FIRST
- `git diff HEAD -- threed/racketsport/foot_contact.py tests/racketsport/test_foot_contact.py`
- `runs/lanes/w4_footattr_fix_20260707/` (report.json, producer_summary.json, deferred_patches/)
- `runs/lanes/w4_footattr_diag_20260707/REPORT.md` + `analysis/` (Design A + the measurement code)
- `runs/lanes/w3_phasefix_20260707/` (consumer contract) + `runs/lanes/w3_phaseverify_20260707/`

## ATTACK SURFACE (execute each; add your own)
1. CIRCULARITY (the exact-match suspicion): the producer's confident-phase counts equal the
   diagnosis's predictions to the digit. Establish whether the producer is INDEPENDENT
   production-schema code that the real consumer (`grounding_refine` / the w3_phasefix consumption
   rules) actually ingests — or a re-run of the diagnosis analyzer dressed as a producer (self-
   confirmation). Diff the logic; then prove consumption: feed the producer's artifact through the
   REAL consumer entry (not the predictor) and show refine engages on it.
2. FABRICATED CONFIDENCE: construct adversarial skeleton/gate-stream fixtures where the formula
   (foot in {left,right} AND min_confidence>=0.90 AND body_detector_agreement>=0.90) certifies a
   WRONG foot or certifies during ambiguous crossover frames. If you can make it confidently wrong
   on plausible inputs, that is a DEFECT with an executable proof (fixtures in your lane dir).
3. UNFAILABLE-GATE / EXCLUSION CIRCULARITY (the r2 class): audit every rejection/demotion reason
   path in the new producer for anything correlated with the slide gate or its threshold; construct
   the case where a high-slide candidate phase gets excluded for a reason that tracks the gate.
   Also verify the companion metric (`max_candidate_phase_slide_m`) covers EXCLUDED candidates on
   the four real artifacts (exclusion never silent).
4. FENCED PATCH DRY-RUN: `git apply --check` the deferred patch; then apply it to a COPY of
   `scripts/racketsport/process_video.py` in your lane dir and drive the copied entry as far as a
   CPU replay allows on one clip's banked inputs with the producer artifacts REMOVED/renamed in a
   sandboxed copy of the run dir — prove the stage actually (a) runs before grounding_refine,
   (b) invokes the new producer, (c) writes the phases artifact fresh (not reading prewritten
   ones). Any reachability gap (r1 class) = DEFECT.
5. PROVENANCE MUTATION: the fix lane REGENERATED `foot_contact_phases.json` inside the banked
   `runs/lanes/w3_freshworlds_20260707/` evidence dirs. Determine: were originals preserved
   anywhere? If not, prove the ORIGINAL (weak-placeholder) artifacts are reconstructible from
   HEAD code + the persisted gate stream (HEAD predates the fix — reconstruct one clip's original
   and diff its phase counts against the diagnosis's recorded 22/44/18/6 source phases). Record
   the hygiene finding either way (this informs a standing rule, not necessarily a defect).
6. VACUOUS-TEST CHECK: copy foot_contact.py to your lane dir, revert the producer's confidence
   formula in the COPY (e.g. drop the agreement term), run the new tests against the mutated copy
   via a sys.path shim: still-passing tests = vacuous = DEFECT.
7. INDEPENDENT REPLAY: re-run the unmodified w3 offline replay predictor + both phaseverify attack
   scripts yourself; confirm the reported per-clip slide values and kill=false — from your own
   invocations, not the lane's JSON.

## HARD CONSTRAINTS
READ-ONLY on production/test files (mutations only on COPIES in your lane dir). No git operations
beyond read-only diff/apply --check (and apply to COPIES). `.venv/bin/python`. Never touch eval
labels / ledger / ios/ / runs/manager/. Do not "fix" anything — you report defects, the manager
routes repairs.

## STRUCTURED REPORT
objective_result PASS = your verification COMPLETED. Acceptance table row per attack with VERDICT
= CONFIRMED-GOOD or DEFECTS-FOUND stated prominently; every claimed defect ships a runnable proof
+ exact command. honest_issues: attacks not executable locally and exactly what the wave-close
fresh-GPU proof must therefore check.
