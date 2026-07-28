# Owner check-in

Updated: 2026-07-28 (morning, after the overnight full-access run). `VERIFIED=0` binding; nothing promoted past the bar yet.
Single source of owner-facing truth. Full data map: `DATA_INVENTORY.md`. Program: `NORTH_STAR_ROADMAP.md`.

## What happened overnight (2026-07-28) — 11+ commits on main, ~$35-45 GPU

1. **Kitchen gating + foot grounding are now ALWAYS-ON defaults** (both presets,
   opt-out flags exist) and were **proven on a fresh six-clip GPU wave** at that
   revision: always-on stages ran on 6/6 clips (one honest typed revert), 4/4
   players everywhere, conservative kitchen calls everywhere, foot-slide 4/6
   under the 0.03 m bar. Evidence: `runs/alwayson_fresh_wave_20260728/REPORT.md`.
   Flags for eyes: wolverine slide 0.0378 (new), indoor-diagonal 0.0546 (known).
2. **Speed, measured**: default-OFF persistent warm BODY worker landed — BODY
   remote command 174–222 s cold → **110 s warm** (load+compile → 0). The
   `--body-local` co-located silent-degrade bug is **fixed** (`757da51`; root
   cause was tonight's NVZ persistence rewriting `court_lock.json` after
   identity fingerprinting) and the **complete co-located pipeline with real
   BODY measured 266.5 s** on wolverine (−24% vs the 352.5 s baseline median;
   all always-on stages ran; foot-slide 0.0158 m PASS). The "everything
   together" integration demo (full preset + ball + ball-aware meshes +
   one_world) ran end-to-end in 987.3 s with five honest typed degrades, all
   pre-existing product behaviors. `runs/lanes/warm_body_worker_20260728/REPORT.md`,
   `runs/lanes/bodylocal_colocated_fix_20260728/REPORT.md`. Known follow-up:
   court23's Exclusive_Process GPU mode blocks co-located BODY (CUDA busy) —
   use Default mode (`nvidia-smi -c 0`) for co-located runs.
   Post-wave: the wolverine foot-slide 0.0378 flag was **diagnosed as NOT a
   regression** (one spurious 4-frame plant phase from sub-cm BODY run-to-run
   noise at an unchanged hysteresis threshold; 42/43 phases at ~1e-15 m in both
   runs — `runs/lanes/wolverine_slide_diag_20260728/REPORT.md`); a plausibility
   fix lane is in flight. A rendered **visual evidence pack is on your Desktop:
   `~/Desktop/visual_evidence_20260728/`** (six overlay videos + gallery +
   scorecard).
3. **EVENT: first above-chance HIT-contact signal in program history.** E-v2's
   Step-0 data gate passed live; a time-boxed resume+fine-tune produced real
   signal (precision 0.5 @ thr 0.1 public sweep); firing-rate gate not yet
   cleared; PARTIAL, no anchors ingested. A **full-scale run is executing now**
   on night2. Bonus catch: corpus builder silently included a ledger-BLOCKED
   asset — quarantined + restarted clean. `runs/lanes/ev2_train_20260728/`.
4. **Models are now durable**: warm-runtime GCP snapshot
   (`pickleball-court23-warm-20260728`, zero-setup VM boots, proven twice) +
   sha-verified S3 store `s3://sway-videos/pickleball-models/20260728/`
   (restore: `scripts/fleet/bootstrap_models_from_s3.sh`).
5. **The exact events + ball-2D→3D plan** you asked for:
   `runs/lanes/next_steps_events_ball3d_20260728/PLAN.md`. It also found queue
   rows 1–4 (sigma, reprojection retirement, k1 fit, sub-frame timing) already
   landed on main — engineering-complete, awaiting scoring vs labels.
6. **A watchable evidence pack is being rendered to
   `~/Desktop/visual_evidence_20260728/`** (per-clip overlay videos + gallery +
   plain-language scorecard).
7. **H100s: this GCP project has ZERO H100 quota in every region.** Owner ask:
   file the a3/H100 quota increase (or approve RunPod H100s). Everything ran on
   3× spot A100s with auto-poweroff rails.

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
