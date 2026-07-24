# Owner check-in

Updated: 2026-07-23 (late evening). `VERIFIED=0` binding; nothing promoted past the bar yet.
Single source of owner-facing truth. Full data map: `DATA_INVENTORY.md`. Program: `NORTH_STAR_ROADMAP.md`.

## 👉 Your one high-value to-do: capture real pickleball footage (ground truth)
Nothing can be promoted past `VERIFIED=0` until we have **owner-shot footage with ground truth** — the
one action that unblocks *every* lane (court, ball-3D, person, events). Start small: ~100–300 controlled
flights, baseline iPhone + 2 temporary side/corner cameras for triangulation. Plan:
`runs/ball3d_lifting_plan_20260723/PLAN.md`.

_Court labeling is **handled** — it's Codex's lane, you've already labeled plenty (incl. v3), and Codex
has 293 auto-adapted diverse-venue labels. An optional fresh pack (`court_labelpack3_20260723`, 50 new
venues) exists for cheap extra points, but it is **not needed from you**._

## Capability status

| Lane | State | Next |
|---|---|---|
| **COURT** | REAL WIN stands: first real-data retrain, held-out PCK@5 0.079→**0.371**, median err 265px→**~7px**. Adapter just unlocked **293** diverse-venue labels (zero owner time). | Your pack-#3 labels + the 293 adapted → retrain → push toward 0.95. |
| **BALL** | B2 A/B still **no number**, but re-diagnosed: the `io_decode` bug is **fixed & verified** (seconds, not hours). The real wall is a *different* stage — WASB teacher-inference (~5-6h for the full 7-video SST build; **identical pace on T4 and L4 → not GPU-bound**, likely CPU decode / per-frame Python). VMs+disks torn down, ~$1.30, judge untouched. | Profile the WASB inference path (py-spy) → batch/parallelize it → rebuild SST cheap → short GPU run for B2 vs 0.567 judge. |
| **EVENT** | E-v2 gate + ledger-auth committed; GPU run **parked** — a fleet-cache-image media-staging bug refused the run 3× (fail-closed, ~$1.5 total, no training). | Proper on-VM post-staging diagnostic to root-cause, then re-fire. |
| **PERSON** | Manifest blocker **cleared** — `tracking.player_selection_layer` is present, repo-wide tests green. Selection-layer still needs a clean rebuild to pass review (rejected twice on process-hygiene, not the algorithm). | 3rd from-scratch clean rebuild of the spectator/exactly-4 filter → fixes the position-fabrication issue (P0-I). |
| **DATA/INFRA** | Data-safety gate COMMITTED. `DATA_INVENTORY.md` live (all 32 datasets, used/unused, generated from the ledger). | — |

## The data picture (see DATA_INVENTORY.md)
Of 32 registered datasets: **7 used, 1 authorized, 12 blocked, 5 held-out (eval), 7 rejected.**
The blocked 12 is the untapped pool — the big one was the Roboflow court keypoints, now being unlocked
via your pack #3 (owner-GT) + the schema adapter (293 auto-mapped labels).

## Cloud / cost
**One A100 running: your court VM `pickleball-gpu-court23`** (a2-highgpu-1g spot, us-central1-f) — Codex's
court lane, ~$1-1.5/hr; **tear it down when that lane finishes.** My ball VMs+disks are all torn down. Shared
fleet-cache disk/image kept (`pickleball-cache-data-usc1f`). gcloud on
`hello@swayformations.com` / `gifted-electron-498923-h1` (an external `sway-gcp-cutover` switch was
observed + reverted earlier).

## Repo hygiene note
Completed, test-green, disjoint deliverables were committed to main tonight (`df8bdb0`: person
few-shot pack + court label server + audio-alignment tool, 76 tests). The rest of the working tree
is your **active Codex court work** (static-lock, covariance, skeletons — do not touch) plus a
pre-existing data backlog; it is NOT safe to wholesale-commit. A coordinated cleanup pass, once Codex
confirms which lanes are closed, is the recommended next step.
