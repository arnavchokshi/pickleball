# ROUND-3 FOCUSED RE-REVIEW of ball_b0_split_20260721 (read-only) — DECIDABLE THIS ROUND

Scope: ONLY R2 finding 2 (task-87 provenance) + the fix2 delta (report_fix2.json, FIX2_SPEC.md).
All other findings were CLOSED in R2 — do not relitigate them.
Verify with probes: (1) absent/wrong image_zip_sha256 now REFUSES (your R2 injection was ignored —
retry it); (2) per-image digest binding is real (tamper one staged image byte → refuse);
(3) wrong job_id injection refuses; (4) residual_assumptions is present, accurate, and its content
matches reality (no overclaim of what is bound).
ORCHESTRATOR PRE-RULING for decidability: the declared residual ("local CVAT rendered the imported
staged bytes; no independent job-id in the historical ledger") is ACCEPTED by the orchestrator as a
recorded assumption IF your probes confirm the bindings above. So your verdict is binary:
ACCEPT (bindings real, residual honest) or REJECT (name the exact probe that failed).
Also state: the 2 current-only BALL arc-budget suite failures (passed 2/2 in isolation) — plausibly
concurrency/timing flakes or a real regression from the owned file?
VERDICT in final JSON.
