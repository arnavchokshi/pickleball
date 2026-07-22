# FIX ROUND (retry) for ball_b1b2_prep_20260721 — implement the review's required changes

NOTE: a previous run of this fix crashed mid-work; partial edits may exist in the owned files —
resume from the current state. Review with findings: runs/lanes/ball_b1b2_prep_20260721_review/
review.json (full JSON also at the end of that dir's log.txt). Same ownership as spec.md:
build_pbvision_ball_sst.py, train_ball_stage2.py, ball_loso_validation.py + their tests.

Required changes (neutral wording; substance per the review):
1. Production constants: conf=0.90 / radius=20px / weight=0.25 are fixed in code. CLI overrides
   mark the manifest production_eligible=false; the trainer and gate refuse non-production
   manifests. The manifest records the fixed constants + builder code identity.
2. The temporal/geometry acceptance path must rely on INDEPENDENT WASB evidence (frozen-WASB
   agreement in neighboring frames bridging a short, preregistered teacher-only gap). Teacher
   self-consistency alone can never make a row eligible. If real acceptance counts then land under
   the 1,000-row/5-source gate, report PBV_BALL_INSUFFICIENT_AGREEMENT honestly.
3. Spatial agreement validates image bounds; out-of-bounds coordinates are ineligible.
4. The WASB checkpoint SHA must equal the models/MANIFEST.json entry in production mode; record
   the WASB repo commit.
5. train_ball_stage2.py validates any --sst-manifest fully before use: gate verdict PASS, minimum
   counts, canonical source IDs, per-row agreement evidence + dependency hashes + media SHA +
   teacher checkpoint SHA, teacher_derived=true, ground_truth=false; otherwise it refuses (the
   shipped failed-gate sample manifest must be refused).
6. The trainer consumes the accepted B0 split artifact (runs/lanes/ball_b0_split_20260721/split/)
   and filters out every HyUqT7zFiwk/Ezz6HDNHlnk row from training input; add a test proving those
   960 judge-parent rows cannot enter a training batch.
7. ball_loso_validation.py parent-source mode must score the B0 final_label rows directly (or via
   a materialized scratch reviewed root), with validation identity + prediction artifacts bound to
   canonical clip/source-video hashes; the live invocation on the real B0 artifact must succeed.
8. Replace the parity self-test with a pinned-HEAD-vs-working-tree comparison harness on the
   production configuration (record exact losses, sample order, model-state SHA).
9. Exactly 8 human rows per training step, with SST rows strictly additive; tested.
10. Source identity checks use canonical path identity + expected source-video SHA; a row whose
    video and source_video fields disagree is refused; missing media reports not_attempted rather
    than zero failures; correct the six-of-seven staging instruction.
Report to report_fix1.json (schema-valid). No NEW wide-suite failures beyond the known
environmental set.
