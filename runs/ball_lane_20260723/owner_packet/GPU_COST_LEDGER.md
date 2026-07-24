# GPU cost ledger (invoice-backed) — seeded 2026-07-23

Envelope: **~$65 pre-approved** (controller decision 2026-07-23, relayed in WS4 tasking; this file
is the seed record — the decision is not yet in any other committed repo file). Per-run estimates
below sum to $33–64. `$ est` figures are SPOT-band arithmetic, **not invoice-backed** (same
convention as `runs/manager/gpu_fleet.md`).

| date | lane | run id | instance type | hours | $ est | $ invoiced | invoice ref | gate that authorized it |
|---|---|---|---|---|---|---|---|---|
| — | BALL (Track B) | B1 finish (SST rebuild + gate) | TBD at dispatch (prior grant: A100-40 SPOT us-central1-f) | — | 6–10 | 0 | — | 2026-07-22 manager slot grant (`runs/tracks/trackB_ball_20260722/STATUS.md`) — **needs owner re-confirmation, disks torn down since** (OWNER_ASKS #4) + $65 envelope 2026-07-23 |
| — | BALL (Track B) | B2 pair (A/B vs 0.567 judge) | TBD at dispatch | — | 8–18 | 0 | — | B1 gate PASS + review_r4 superseding `DO_NOT_ARM_B2_YET` (`runs/handoff_20260722/STATE.md` §3) + envelope |
| — | EVENT (Track D) | E-v2 re-fire | TBD at dispatch | — | 3–6 | 0 | — | E-v2 gate + ledger-auth committed; re-fire only after the fleet-cache media-staging root-cause (`OWNER_CHECKIN.md` EVENT row) + envelope |
| — | EVENT (Track D) | multimodal v3 | TBD at dispatch | — | 5–9 | 0 | — | controller envelope 2026-07-23 (run named in tasking; no repo spec file read by this lane — spec required before dispatch) |
| — | BALL (Track B) | outdoor-night ensemble pass | TBD at dispatch | — | 11–21 | 0 | — | Track B contract item 3 — **blocked on B2 winner as co-teacher** (`runs/tracks/trackB_ball_20260722/STATUS.md`) + envelope |
| | | | | **total** | **33–64** | **0** | | ceiling **$65** |

## Procedure
Every GPU dispatch appends its row to this table **before** launch (date, lane, run id, instance
type, authorizing gate filled in; hours/$ blank until teardown). Invoices are reconciled weekly
against GCP billing: fill `$ invoiced` + `invoice ref` per row, and any row older than 7 days with
an empty invoice column is a reconciliation debt to chase, not to estimate away. The envelope
ceiling is $65 total across all rows until the owner raises it in writing; a dispatch whose
worst-case estimate would take the running total past the ceiling does not launch.
