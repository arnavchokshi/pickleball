# ROUND-2 ULTRA RE-REVIEW of ball_b0_split_20260721 after FIX round (read-only)

Prior review: runs/lanes/ball_b0_split_20260721_review/review.json (REJECT, 1 primary blocker + 3
HIGH + semantics defect). Fix claim: runs/lanes/ball_b0_split_20260721/report_fix1.json. Fix spec:
runs/lanes/ball_b0_split_20260721/FIX_SPEC.md. Diff target: scripts/racketsport/build_ball_regroup_split.py
+ tests/racketsport/test_build_ball_regroup_split.py.

Verify EACH prior finding is genuinely closed — re-run your original probes:
1. Attestation: does the gate actually bind task_id + export SHA fea4b952...? Probe: wrong-SHA
   attestation, all_frames_inspected=false, missing file — all must yield BALL_NO_CLEAN_JUDGE /
  refusal. Is the REAL filed attestation (cvat_upload/exports/w7_audit_stratum_20260709/
  owner_attestation.json) actually consumed in the real run?
2. Task-87 provenance enforcement (fingerprint/ledger/prelabel_zip=null/manual-source) — can an
   auto-sourced export still pass?
3. Content-binding of all lineage inputs — tamper one byte, must refuse.
4. Canonical protected set — can any override yield the clean verdict?
5. Visibility semantics — out_of_frame + absent model record now corrected_prelabel (1,997/1,029);
   is the new rule sound and tested?
6. NEW-FAILURE ATTRIBUTION: fix1 reports 35 wide failures vs the known ~31 set; the 5 new ones are
   in the P1 exporter's test file, which a CONCURRENT fix lane was rewriting during the suite run.
   Verify those 5 are cross-lane concurrency artifacts, not B0 regressions (the file passed 26/26 in
   isolation per the report — reproduce if cheap).
VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES | REJECT. The judge gates real GPU money.
