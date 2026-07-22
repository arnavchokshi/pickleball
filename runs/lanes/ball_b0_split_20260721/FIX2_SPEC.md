# FIX ROUND 2 for ball_b0_split_20260721 — close the R2 provenance blocker (runs/lanes/ball_b0_split_20260721_review/review_r2.json finding 2)

Same ownership (build_ball_regroup_split.py + its test). Surgical scope ONLY — do not touch closed findings.

1. BYTE-BIND the image package: provenance contract must REQUIRE image_zip_sha256 and validate it
   against the actual staged zip (computed by the tool, expected
   f1b7ba88084c8664202bf19f73e4704599b46bd42e50e2a6c1a29265cff8b653). FIX THE FAIL-OPEN BUG the
   reviewer found: injected wrong job_id/image_zip_sha256 values must REFUSE (validate always, not
   only-when-present-and-matching-shape).
2. Per-image digests: verify every materialized judge/train frame's bytes against the staged zip
   entry digest map; any mismatch refuses.
3. job_id: bind it if any consumed artifact (export XML metadata, import ledger) carries it; if
   genuinely absent from all artifacts, the provenance contract must record
   job_id_binding: "UNAVAILABLE_IN_ARTIFACTS" explicitly — never silently absent.
4. DECIDABILITY FALLBACK: emit a residual_assumptions list in the judge artifact for whatever
   cannot be cryptographically closed (expected residual: "local CVAT rendered the imported staged
   bytes for task 87; no independent job-id binding exists in the historical import ledger").
   The orchestrator rules on that residual; your job is maximal achievable binding + explicit
   declaration, not silent acceptance or unbounded blocking.
5. Adversarial tests: wrong/absent zip SHA, tampered image byte, injected wrong job_id, residual
   declaration presence.
Re-run the real pipeline; report judge status + the residual_assumptions content.
Report to report_fix2.json (schema-valid). No NEW wide-suite failures beyond the known set.
