# Owner check-in

Updated: 2026-07-23 (overnight). `VERIFIED=0` remains binding; nothing promoted past the bar yet.
Tonight moved the EVENT track from blocked → firing on GPU, kept BALL retrain running, and built
your next court-labeling pack. One real capability gain already stands (COURT, below).

## 👉 WHAT YOU DO NEXT (one thing, ~70 min) — label court pack #2

Open `cvat_upload/court_labelpack2_20260723/START_HERE.html` (double-click it — no server needed).
Click each named court point where you can clearly see it; it **auto-advances to the next point**
after every click (same click-and-go UI as before). Space = skip a point you can't see, X = skip a
bad frame, U = undo. It auto-saves and is resumable. When done, click **Export one results JSON**
and send that one file back.

- 123 new frames across 12 venues you haven't labeled yet (zero overlap with your prior 64).
- This is the single highest-leverage thing you can do: more court labels is the path from today's
  0.371 toward the 0.95 promotion bar. Verify-by-clicking, never a from-scratch draw.

## 🎉 REAL RESULT (stands) — COURT auto-find improved on your labels

First-ever real-data court-keypoint retrain (your 28-venue labels + pb.vision corpus vs a
synthetic-only control). VERIFIED=0 (candidate gain, below 0.95) but REAL:

| Held-out venues | before (median keypoint error) | after your labels |
|---|---|---|
| reserved holdout | PCK@5 0.079, err **265px** | PCK@5 **0.371, err ~7px** |
| protected clips (never seen) | PCK@5 0.117, err **340px** | PCK@5 **0.20, err ~13px** |

Median court-point error ~265px → ~7px on held-out venues — unusable → genuinely good. Evidence:
runs/lanes/court_realtrain_20260723/RESULT_HEADLINE.md. Next: your pack-#2 labels → scale toward 0.95.

## In flight right now (results coming while you're away)

- **EVENT (E-v2) — GPU RUN FIRING (result ~2-3h).** The whole event chain is unblocked and running
  on a fresh A100. Target: beat E1's owner-41 score of **0.1304** (macro-F1@2). SoccerNet-BAS
  curriculum: pretrain on the corrected 1,189-row corpus → finetune on your frozen 61 labels →
  scored ONCE on the frozen owner-41 judge. Self-guarding (hard $19.50 cap, 350-min wall,
  auto-teardown). I'll report the number when it lands.
- **BALL (B2) — A/B retrain RUNNING on pickleball-gpu-ball-f (result this evening).** Number vs the
  0.567 judge. (Its data-build is slow on one video due to a known perf bug — flagged for later
  fix, not blocking the result.)

## This session's landings (committed + pushed to main)

- **DATA-SAFETY GATE — COMMITTED (`f27767f64`).** Fail-closed: every VM proves the gate ran and
  re-derives eligibility from the sha-bound ledger before reading a single training byte. It already
  proved itself by refusing the event run until the data was properly authorized.
- **EVENT ledger authorization — COMMITTED (`c28951baa`), manager-reviewed.** The 4 event inputs are
  queue-authorized; I independently re-verified (gate PASS on the 4 inputs; court-derivatives,
  compare-only, and protected media all still REFUSED). Codex is content-filter-blocked on this
  security code, so you authorized me to review it.
- **EVENT run made dispatch-ready (`bfeb620a0`).** Synced the run's frozen-code manifest to the
  reviewed committed code + landed the run artifacts (no runtime-code change). All preflight gates
  pass → the GPU run above could fire.
- **BALL B1 dependency race — COMMITTED (`f07929bb8`).** Structural fix (race eliminated by
  construction).

## Current A-E state

| Track | State | Next |
|---|---|---|
| **A — COURT** | Real-data retrain WIN (0.371, above). Labels ingested + committed. | **Your pack-#2 labels** → scale toward 0.95, then promote into default pipeline. |
| **B — BALL** | B2 A/B retrain RUNNING on GPU. | Report number vs 0.567 judge (this evening). |
| **C — PERSON** | Selection-layer wiring still REJECT (needs clean rebuild). Note: an incomplete person-track change left `best_stack.json` missing a `tracking.player_selection_layer` entry in the working tree — that lane needs to finish it. | Fresh clean-lane rebuild of the spectator filter. |
| **C — BODY** | Watchdog OOM fix proven. | Memory bench (deferred). |
| **D — EVENTS** | Gate + ledger auth committed; **E-v2 GPU run firing now**. | Report owner-41 number vs 0.1304. |
| **E — DATA/INFRA** | Gate COMMITTED (unblocks all retrains). Fleet cache READY (E-v2 boots from it). | — |

## Cloud / cost

Running: `pickleball-gpu-ball-f` (B2 A/B) + `pickleball-gpu-ev2` (E-v2, firing now, ≤$19.50 cap +
350-min wall + auto-teardown). Owner's court VM already torn down. gcloud note: an external
`sway-gcp-cutover` account switch was observed and reverted; control-plane stays on
`hello@swayformations.com` / `gifted-electron-498923-h1`.
