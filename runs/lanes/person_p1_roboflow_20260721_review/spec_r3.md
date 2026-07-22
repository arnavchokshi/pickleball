# ROUND-3 RE-REVIEW of person_p1_roboflow_20260721 + post-hoc binding ruling (read-only)

Part A — verify fix2 (report_fix2.json, FIX2_SPEC.md): (1) the two R2 leak sources moved WHOLE into
the od8al validation family; re-run your pixel-level cross-split scan method to confirm 0 leaks
remain; (2) the retention-gate FAIL (8,887 images / 7 families vs frozen >=8) is honestly derived —
no family bookkeeping tricks in EITHER direction (neither inflating to pass nor deflating: recount
families independently); (3) legacy trainable YAML gone; PASS-package consistency implemented;
(4) the PERSON_RF_POOL_TOO_THIN + P2 NO_ATTEMPT_PREREQ disposition matches EXACT_PLAN §3.4.

Part B — RULE on a protocol deviation: the amended human-review protocol
(runs/lanes/p1_protocol_ruling_20260721/ruling.json) required byte-binding the pack BEFORE the
owner's pass. The owner completed the full 182-image pass early with ZERO exceptions; the
coordinator then bound post-hoc: audit/pack_binding.json (182/182 hashed, pack digest 2975b96a...,
newest image mtime 2026-07-21 19:36:12 PREDATING the pass), audit/exception_record.json (empty,
explicit i/o/m=0 semantics), audit/owner_attestation.json binding both hashes and RECORDING the
deviation. Verify those artifacts exist and are internally consistent (mtimes, hashes, semantics).
QUESTION: is post-hoc binding acceptable HERE given (a) immutable-file mtime evidence predates the
pass, (b) zero-exception outcome means no per-frame counts were at stake, (c) the card's only
remaining consumer is a data-quality deliverable (P2 is dead per Part A)? Rule:
POSTHOC_ACCEPT | POSTHOC_ACCEPT_AS_DIAGNOSTIC_ONLY | POSTHOC_REJECT (+ what would fix it).
VERDICT in final JSON: {part_a: ACCEPT|REJECT + findings, part_b_ruling: ..., overall: ...}
