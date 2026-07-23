# Owner check-in

Updated: 2026-07-23 (evening). `VERIFIED=0` binding; nothing promoted past the bar yet.
Single source of owner-facing truth. Full data map: `DATA_INVENTORY.md`. Program: `NORTH_STAR_ROADMAP.md`.

## 👉 Your one high-value to-do: label court pack #3
`cvat_upload/court_labelpack3_20260723/START_HERE.html` — 122 frames across **50 distinct new venues**
(zero overlap with anything you've labeled). Click a point → auto-advances; **C** = copy previous
frame's points; **click a dot + Delete/Backspace** = fix it; Space/X/U = skip point/frame/undo.
Export one JSON, send it back. This is the highest-leverage thing you can do — more venue diversity
is the path from court 0.371 → the 0.95 bar.

## Capability status

| Lane | State | Next |
|---|---|---|
| **COURT** | REAL WIN stands: first real-data retrain, held-out PCK@5 0.079→**0.371**, median err 265px→**~7px**. Adapter just unlocked **293** diverse-venue labels (zero owner time). | Your pack-#3 labels + the 293 adapted → retrain → push toward 0.95. |
| **BALL** | B2 A/B **cut** — 11.5h build was `io_decode`-bottlenecked with the A100 at 0% util; VM+disk deleted, $0. | Fix `io_decode` perf bug → rebuild SST on a **cheap CPU box** → short GPU run for B2 vs 0.567 judge. |
| **EVENT** | E-v2 gate + ledger-auth committed; GPU run **parked** — a fleet-cache-image media-staging bug refused the run 3× (fail-closed, ~$1.5 total, no training). | Proper on-VM post-staging diagnostic to root-cause, then re-fire. |
| **PERSON** | Selection-layer wiring still needs a clean rebuild. **Blocker:** an incomplete change left `best_stack.json` missing `tracking.player_selection_layer`, which breaks the whole `process_video` test suite. | Supply the missing manifest value (unblocks repo-wide tests), then rebuild the spectator filter. |
| **DATA/INFRA** | Data-safety gate COMMITTED. `DATA_INVENTORY.md` live (all 32 datasets, used/unused, generated from the ledger). | — |

## The data picture (see DATA_INVENTORY.md)
Of 32 registered datasets: **7 used, 1 authorized, 12 blocked, 5 held-out (eval), 7 rejected.**
The blocked 12 is the untapped pool — the big one was the Roboflow court keypoints, now being unlocked
via your pack #3 (owner-GT) + the schema adapter (293 auto-mapped labels).

## Cloud / cost
**No GPUs running or billing.** Ball VM + disk torn down. Shared fleet-cache disk/image kept
(`pickleball-cache-data-usc1f`). Your court VM `pickleball-gpu-court23` already terminated. gcloud on
`hello@swayformations.com` / `gifted-electron-498923-h1` (an external `sway-gcp-cutover` switch was
observed + reverted earlier).

## Repo hygiene note
The working tree carries a large pre-existing uncommitted backlog + your active Codex work + the
broken person-track change above. It is NOT safe to wholesale-commit (would land broken state and
clobber in-flight work). A coordinated cleanup pass is a recommended next step.
