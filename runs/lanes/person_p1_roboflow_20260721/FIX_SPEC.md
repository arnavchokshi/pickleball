# FIX ROUND for person_p1_roboflow_20260721 — address ultra-review REJECT (runs/lanes/person_p1_roboflow_20260721_review/review.json)

Same HARD RULES + FILE OWNERSHIP as spec.md (export_roboflow_person_yolo_dataset.py + its test only).
Read the review JSON first; implement every blocking exact_fix:

- P1-SPLIT-LEAK (CRITICAL): move pickleball-od8al/pickleball-tsgju into ONE family with
  pickleball-od8al/pickleball-version2 + pickleball-seg, assigned to VALIDATION (holdout side wins,
  fail-closed). Build an auditable original-video/game/session/channel family map; FAIL on any
  cross-split temporal/filename-lineage overlap (implement the temporal-prefix check the reviewer
  used); regression test with this exact family. Regenerate ALL counts/audit/collision artifacts —
  no hand-edited numbers.
- P1-BALANCE-NOOP (HIGH): implement deterministic family-balanced sampling that the EXACT P2 yolo
  command actually consumes (e.g., data.yaml train pointing at a generated train list with
  deterministic per-family repetition caps), tested with strongly unequal source sizes; otherwise
  remove the flag and every 'balanced' claim and mark the contract unmet. State which option you took.
- P1-HUMAN-GATE-BYPASS (HIGH): pending-human rows report PENDING (never PASS); pre-quarantine
  retention marked provisional; implement review_template.csv ingestion + below-90% source
  quarantine + post-quarantine retention recompute + an explicit training-ready gate that P2
  requires; tests.
- P1-COLLISION-WEAK (HIGH): make the protected-collision check robust to at least horizontal flips
  and 5-20% crops/letterboxing (compare against flipped protected descriptors + multi-crop tiers;
  the reviewer measured 0/12 flip detection with distances like [26,28,28] flip / [12,12,12] 5%
  crop). Re-run the FULL exhaustive check with the robust matcher; report new collision count and
  the measured detection rate on the reviewer's derivative classes (flip, 10%/20% crop,
  letterbox). Honestly state residual classes not covered.

Acceptance: all fixes + tests; full regenerated artifact set; focused suite green; wide suite no NEW
failures vs the known environmental set. Report to report_fix1.json (schema-valid).
