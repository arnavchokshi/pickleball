# ORCHESTRATOR RULING — PERSON P1/P2 closeout (2026-07-21T21:05Z)

Status: VERIFIED=0. Refs: review.json (R1 REJECT), review_r2.json (R2 REJECT), review_r3.json
(R3), reports fix1/fix2, EXACT_PLAN §3.4.

1. **P1 disposition: PERSON_RF_POOL_TOO_THIN — ACCEPTED as a verified named negative.**
   Independently confirmed by the R3 reviewer: 8,887 train images / 23,489 boxes / 8 sources /
   7 source families vs the frozen >=5,000-image AND >=8-family bar; final split leak-free by
   fresh pixel-level rerun (0 candidates over 66,359,261 cross-split pairs, production
   three-scale pHash method). The two R2 leaked source families (pickle-es3fs, nigh-workspace)
   are validation-only, moved whole. Root cause: public Roboflow person boxes collapse onto too
   few original-footage families once true video lineage is enforced.
2. **P2: NO_ATTEMPT_PREREQ, permanently closed for this export** (training_ready_gate.json).
   No GPU was spent. Plan §3.4 priced this failure at 50%.
3. **Human quality card: NOT_COMPLETED_PROTOCOL.** The owner's full-gallery pass (zero
   exceptions reported) was reviewed under the amended protocol and ruled POSTHOC_REJECT
   (review_r3.json part_b): binding happened after the pass; the zero-exception report is
   incompatible with the taxonomy (o must be the exact count of off-court person boxes, which
   exist); required binding fields absent. A compliant redo recipe is documented in the review
   (pre-bound manifest + repeat pass with exact i/o/m counts). DECISION: no redo requested —
   P2 is closed either way, the diagnostic's cost now exceeds its value this sprint, and no
   metric from the non-compliant pass is used anywhere. The 100%/100% impression is recorded as
   anecdote only, never as a measurement.
4. **Accepted known issues** (documentation nits, do not change the disposition):
   report_fix2.json lacks a top-level artifacts key; one order-dependent wide-suite flake was
   not proven pre-existing. Both noted; no third fix round for a JSON key on a closed track.

Utilization delta for the ledger: roboflow person core subset -> REJECTED_FOR_TRAINING
(PERSON_RF_POOL_TOO_THIN); audit pack -> staged, protocol-incomplete; protected-collision
tooling (transform-aware, 366M comparisons) -> reusable asset for future packs.
