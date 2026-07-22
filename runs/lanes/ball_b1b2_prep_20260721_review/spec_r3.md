# ROUND-3 FOCUSED CONFIRM of ball_b1b2_prep_20260721 fix2 (read-only) — DECIDABLE THIS ROUND

Scope: ONLY the three round-2 decisive failures (review_r2.json) as claimed closed in
report_fix2.json. Re-run your exact original cases: (1) anchors at 8 and 12 with teacher-only
9-11 must refuse in BOTH builder and trainer; gap exactly 2 must accept; (2) a contradictory
high-confidence WASB observation inside the bridge must refuse (both search directions + trainer
replay); (3) an aliased judge-parent row through the generic CVAT path must refuse on content
SHA; a resume with a swapped dataset must refuse. Spot-check no disturbance: production constants
still enforced; the failed-gate sample still refused; the live B0 scorer invocation still exits 0
with pooled F1 0.5670103.
VERDICT in final JSON: ACCEPT | REJECT, plus GPU_DISPATCH_DECISION with on-VM preconditions
(what the VM must verify before B1 build and before B2 arms train).
