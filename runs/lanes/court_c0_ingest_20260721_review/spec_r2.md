# ROUND-2 ULTRA RE-REVIEW of court_c0_ingest_20260721 after FIX round (read-only)

Prior review: runs/lanes/court_c0_ingest_20260721_review/review.json (REJECT: C0-01 fail-open deny,
C0-02 family split absent + PPA collision, C0-03 geometry, C0-04 C1 integration). Fix claim:
report_fix1.json. Fix spec: FIX_SPEC.md. Target: ingest_cvat_court_images.py + its test.

Re-run your original probes:
1. C0-01: is the deny set now hard-required as exactly {IYnbdRs1Jdk} with the 3-row manifest
   assertion? Probe wrong-valid + extra deny configs again.
2. C0-02: is family grouping real (channel/venue connectivity)? Verify 3sC53GlvW_s is
   QUARANTINED_FAMILY_COLLISION, the 8 frozen holdout groups unchanged, and the gate counts
   (66/18) computed over FAMILIES. Probe: any OTHER train-holdout family connection in the real
   manifest the connectivity logic misses?
3. C0-03: probe the geometry gate with your degenerate configurations + a subtle new one.
4. C0-04: does the emitted split satisfy train_court_model_v2.py's parser exactly? Is the corpus
   relocation-safe (no absolute Mac paths)?
5. Suite: 35 failures vs known ~31 — the extra failures should be concurrent-lane test races on
   OTHER lanes' fenced files (exporter/ball tests being rewritten mid-suite). Verify none trace to
   the two owned files.
VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES | REJECT.
