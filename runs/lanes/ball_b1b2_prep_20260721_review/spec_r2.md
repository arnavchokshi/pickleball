# ROUND-2 RE-REVIEW of ball_b1b2_prep_20260721 after FIX round (read-only)

Prior review: runs/lanes/ball_b1b2_prep_20260721_review/review.json (REJECT: five decisive
failures). Fix claim: report_fix1.json (spec: FIX_SPEC_B.md). Owned files: build_pbvision_ball_sst.py,
train_ball_stage2.py, ball_loso_validation.py + tests.

Re-check each decisive failure with negative-input validation:
1. Preregistration: are conf/radius/weight truly code-fixed with overrides marking manifests
   non-production, and does the trainer refuse non-production manifests?
2. Independence: can any row become eligible from teacher-only evidence? Examine the new
   bracketing-anchor rule (two direct teacher+WASB anchors around <=2-frame teacher-only gaps) —
   is it genuinely WASB-independent and are its parameters frozen in code? Try the shapes from
   your original findings (WASB absent entirely; caller-lowered thresholds; fabricated manifests
   fed to the trainer; the shipped failed-gate sample).
3. Frozen-split enforcement: can any HyUqT7zFiwk/Ezz6HDNHlnk row reach a training batch through
   any input path?
4. Scorer: verify the live invocation result on the real B0 artifact (pooled F1 0.5670103,
   recall 0.5851064, precision 0.55, hFP 0.4932) matches an independent recount from
   the emitted predictions/rows — this number cross-checks a GPU baseline from another machine,
   so its arithmetic must be right.
5. Parity: is the pinned-HEAD comparator real (git-show HEAD code, not self-comparison)? The
   7-step CPU proof stands in for the 2,372-step CUDA run — is the harness sound so the GPU lane
   can run the full check on-VM?
6. Batch math: exactly 8 human rows/step with SST strictly additive — edge cases.
State residual risks for the GPU lane (what must the VM run verify before B2 arms train).
VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES | REJECT, plus GPU_DISPATCH_DECISION with any
on-VM preconditions.
