# FIX ROUND for ball_b1b2_prep_20260721 — review REJECT (runs/lanes/ball_b1b2_prep_20260721_review/review.json; full JSON also in log.txt final block)

Same ownership. Read the review first. Decisive requirements:

1. PREREGISTRATION IS STRUCTURAL: freeze conf=0.90 / radius=20px / weight=0.25 as in-code
   production constants. CLI overrides switch the manifest to production_eligible=false and the
   trainer/gate must refuse such manifests. The emitted manifest records the frozen constants +
   builder code identity; recording chosen-after-the-fact CLI values is NOT preregistration.
2. INDEPENDENCE OF THE TEMPORAL PATH: the reviewer proved same-teacher temporal fallback admits
   2,181 rows with WASB entirely absent — self-agreement is not independence (EXACT_PLAN §2.1).
   The temporal/geometry path must require INDEPENDENT WASB evidence (e.g., frozen-WASB agreement
   in neighboring frames bridging a short teacher-only gap, with preregistered gap length). If
   real acceptance then lands under the 1,000-row/5-source gate, that is an honest
   PBV_BALL_INSUFFICIENT_AGREEMENT — do not restore the fallback to pass.
3. Spatial agreement must enforce image bounds; refuse out-of-bounds coordinates.
4. Pin the WASB checkpoint: enforce SHA equality against models/MANIFEST.json and record the
   WASB repo commit; refuse arbitrary checkpoints in production mode.
5. TRAINER TRUST BOUNDARY: train_ball_stage2.py must revalidate any --sst-manifest: require gate
   verdict PASS, minimum counts, canonical source IDs, per-row agreement evidence + dependency
   hashes + media SHA + teacher checkpoint SHA, teacher_derived=true, ground_truth=false; refuse
   otherwise (the shipped failed-gate sample must be refused — currently it loads a row).
6. FROZEN-SPLIT ENFORCEMENT (CRITICAL): the trainer must consume the accepted B0 split artifact
   (runs/lanes/ball_b0_split_20260721/split/) and structurally EXCLUDE every HyUqT7zFiwk/Ezz6HDNHlnk
   row from training (the reviewer showed 960 judge-parent rows would leak). Hard test: judge-parent
   rows can never enter a training batch.
7. Fix the parent-source scorer redirect: score B0 final_label directly (or materialize a
   scratch-specific reviewed root); bind validation identity + prediction artifacts to canonical
   clip/source-video hashes; the live invocation the reviewer ran must succeed.
8. Replace the parity self-test with a pinned-HEAD-vs-working-tree harness on the production
   config (the reviewer's own small-fixture compute parity PASSED — encode that method); record
   exact losses, sample order, model-state SHA.
9. Make the 8-human-rows/step composition exact and tested; SST rows strictly additive.
10. Compare-ID refusal: enforce canonical path identity + expected source-video SHA (not string
    matching); a conflicting video/source_video alias must refuse; missing-media reporting =
    not_attempted (not zero-failures); correct the six-of-seven staging instruction.
Report to report_fix1.json. No NEW wide-suite failures beyond the known environmental set.
