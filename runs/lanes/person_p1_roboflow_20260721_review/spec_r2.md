# ROUND-2 ULTRA RE-REVIEW of person_p1_roboflow_20260721 after FIX round (read-only)

Prior review: runs/lanes/person_p1_roboflow_20260721_review/review.json (REJECT: P1-SPLIT-LEAK,
P1-BALANCE-NOOP, P1-HUMAN-GATE-BYPASS, P1-COLLISION-WEAK). Fix claim: report_fix1.json. Fix spec:
FIX_SPEC.md. Target: export_roboflow_person_yolo_dataset.py + its test.

Re-run your original probes against the fixed artifacts:
1. SPLIT-LEAK: tsgju+version2+seg one validation family? Re-check the consecutive-frame pair you
   found; scan for any OTHER cross-split temporal/filename-lineage overlap the new detector might
   miss (different prefix conventions, .rf.* derivative pairs across splits).
2. BALANCE: is the 1,308-entry list deterministically derived, honestly documented (racket-ai
   8,067->194/epoch), and actually consumed by the exact P2 yolo command via the gated data.yaml?
3. HUMAN-GATE: is data.yaml truly withheld until review CSV + quarantine + post-quarantine
   retention pass? Any bypass path? Are PENDING rows unmistakably not-PASS?
4. COLLISION: verify the transform-aware detection claims (12/12 on flip/crops/letterbox); probe a
   derivative class the matcher does NOT cover and check it is HONESTLY declared residual.
5. Suite: 30 failed = strict subset of the known environmental set — confirm.
VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES | REJECT.
