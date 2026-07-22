# ROUND-2 FOCUSED CONFIRM of person_mixed_20260722 fix1 (read-only) — DECIDABLE THIS ROUND

Scope: ONLY the three round-1 findings (review.json) as claimed closed in report_fix1.json.
Re-run your exact original cases: (1) percent-encoded/case/Unicode aliases for protected clips,
compare IDs, IYnbdRs1Jdk must refuse; a compare-only media SHA paired with a permitted ID must
refuse on content identity; (2) injected pseudo provenance into validation must REFUSE (not
rewrite); same-content different-ID cross-split must refuse; the final-list validator must be
executable and SHA-pinned in the manifest; (3) the closed-P1 binding hashes are verified at build
time with drift refusal. Spot-check: counts/caps/determinism unchanged (13/13 byte-identical
regeneration claimed). VERDICT in final JSON: ACCEPT | REJECT, plus GPU_DISPATCH_DECISION with
on-VM preconditions for teacher inference + mixed retrain.
