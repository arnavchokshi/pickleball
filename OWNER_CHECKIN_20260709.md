# OWNER CHECK-IN — 2026-07-09 (WAVE 7 BOOT)

⭐ **HEADLINE:** Wave 7 is live on the refreshed North Star. Paddle wiring (P3-1) resumed to finish
its final census; the BALL seed retrain on your 1,121 labeled rows (the 1k-checkpoint gate) is this
wave's critical-path lane. No typed STOPs at boot — PART 0 verified (biometric-consent item stays
non-blocking until we persist a non-owner profile, which wave 7 does not do).

## Blockers
(none — disk RESOLVED 2026-07-09 ~01:10: you ran the cleanup script, 48GB free. Thanks.)

## Overnight log (owner asleep ~01:15-09:30; autonomous mode, nothing owner-gated will run)
- 01:15 overnight start: ball retrain (H100, ~2.6h in), tierprov recovery, watchdog all live;
  dispatching P2-4 masklet GPU eval + P6-2 BODY+COURT stats lane. Held-out shot, mesh display
  ruling, P4-0 re-confirmation all wait for you (queued asks stand).
- 01:5x: P6-2 BODY+COURT stats v0 LANDED (post-hoc, trust-banded). Masklet spike = honest
  NO-ATTEMPT (permission-blocked; ~$1; banked a fresh 307s H100 BODY baseline anchor). Ball
  retrain ~3.2h in; tierprov recovering from the disk crash.
- 02:2x: BALL 1K-CHECKPOINT LANDED (the big one): your labels moved ball F1 0.5329 -> 0.6152
  (+15.5% rel) going 486->1121 rows — the curve is NOT flattening, so labeling sessions keep
  paying. Control reproduced bit-exact; anomaly re-run clean; $4-8 GPU. Adversarial verify next,
  then the completion arms + re-score on your new 1,750-row corpus.
- 05:4x: verify 5/6 CONFIRMED + gap closed w/ real log; aug ablation ANSWERED (aug = the
  hidden-FP mitigation, stays in recipe; A remains lead candidate). Ghost meshes now fully
  landed pipeline+viewer (best_stack rev-6/7); P5-5b input gate live w/ your angle policy;
  match stats wired; your stride-2 ruling being wired now; speed gate + wave close next.

## Numbered asks (easiest first)
1. **Mac disk — 2 min, semi-urgent.** Data volume is 99% full (5.2GB free; boot bar is >25GB; wave-6
   hit ENOSPC mid-errand). /tmp + pip caches are already clean — the real weight is `runs/` (57GB of
   lane artifacts). The script is READY at `runs/manager/disk_cleanup_20260709.sh` (Section A intermediates ~19GB;
   A+B ~35GB; held-out labels / reviewed corpus / regen-inputs / your raw footage all explicitly fenced).
   Review + run:
   `! bash runs/manager/disk_cleanup_20260709.sh`
   Outside the repo, if you want more back: `~/Desktop/CV_pipeline` = 22GB, `~/Library/Caches` = 4.2GB.
2. **Ball labeling continues (critical path).** 1,121 reviewed rows banked; 72-73 tasks / ~45.6k
   frames already imported in CVAT and waiting. Next gate = 3k checkpoint. NEW per refresh: I'll add
   a uniform-random audit stratum task so the corpus isn't 100% disagreement-selected.
3. **Mesh display ruling (1 question).** 300 vs 400 MiB byte budget, and may human_review-tier frames
   show meshes when ghost-styled? (Evidence: outdoor 5.21→21.32 fps at 112.6MiB.) Wave-7 ghost-mesh
   lanes proceed per your 2026-07-09 ghost-render ruling either way; this only sets the byte cap +
   tier eligibility.
4. **One owner game recording with audio** (unlocks M4 coaching + contact labels).
5. **Paddle 4-corner marker GT + paddle photo/orbit** (only path to RKT VERIFIED; P3-1 ships
   ESTIMATED-banded without it).
6. **Two 5-minute phone checks** (LiDAR range; ARKit sidecar pose).
7. **4.0-rated reviewer commitment** for the P6-4 coaching audit (later this wave at the earliest).
8. **GCP invoice glance** (pins the real H100 spot $/hr; wave reports carry a $0.57-4.25 ambiguity).
9. **P2-4 masklet eval needs a one-time grant (NEW, from overnight):** the permission system
   (correctly) blocked running the third-party sam-body4d repo on a fleet VM. If you want P2-4
   attempted: say so + we'll do the narrow grant and verify HuggingFace sam3 access first; the
   cheap shape is a masklets-only A/B, not the full 5-model pipeline. Otherwise it stays queued.
10. **P4-0 vs 3rd auto-find retrain re-confirmation** (CALV1 evidence: 244.3px Burlington / 212.6px
   Wolverine, containment 0/8 and 2/8). Default this wave: P4-0 profiles first, no retrain.

## Fleet-spend-vs-ask (wave-7 planned GPU — all within standing ≤$5/hr × ≤4 envelope; no approval needed)
| lane | SKU / est hours | est $ (honest span) | gating evidence to earn the spend | waits if declined |
|---|---|---|---|---|
| w7_ballretrain (P1-1, 1k-checkpoint) | H100-80GB spot, 2.5-4h | $1.5-17 (mid ~$5-8) | 486-row anomaly re-run + control row + ~100-step measured probe BEFORE full budget | critical path stalls |
| w7_p22gate GPU validation (P2-2) | H100 spot, 0.5-1h | $0.3-5 | synthetic round-trip gate built + audit findings first (local lane) | decode stays NOT-WIRING-READY |
| w7_speedgate (P5-1 clean-room) | H100 spot, 1.5-2.5h | $1-11 | settled tree (paddle + A2 repair landed) + promoted manifest | speed headline stays unscored |
| w7_body4d masklet eval (P2-4) | H100 spot, 1-2h | $0.6-9 | time-boxed; adopt only on measured no-regression | candidate stays queued |
| **wave total** | | **~$4-40 (mid ~$12-20)** | | |

## Owner-time queue (VI.8 refresh — full table lives in NORTH_STAR VI.8; ranks unchanged, status current)
1 labels (1,121 banked; 3k gate next; audit stratum NEW) · 2 owner game w/ audio · 3 paddle GT ·
4 phone checks ×2 · 5 rated reviewer · 6 GCP invoice · 7 mesh ruling (ask #3 above) · 8 P4-0 re-confirm.

## Money/GPU log (live)
- Boot: fleet ZERO VMs list-confirmed (fleet1 TERMINATED disk-intact standing). No spend yet this wave.

## FYI (no action needed unless it surprises you)
- **The 262mm body-decode mystery is ~90% solved, and it was measurement, not model.** The pipeline
  was dropping the camera-translation term (fixed + adversarially verified this wave), and the gate
  HARNESS had the same bug (fix in flight) — measuring with the term applied gives ~23mm p95, not
  262mm. The 1mm gate bar itself is untouched; once the fixed harness re-measures on the next GPU
  run, I'll bring you the recalibration question the refresh anticipated (R1). Your court-kp
  relabel export was auto-detected and its ingest + re-score is running.
- `body4d-waker-ctrl` (e2-micro, us-central1-a) has been RUNNING since Jun 14 — NOT this project's
  (labels cost-center=body4d, role=wake-controller). ~$6-7/mo. Kill it if it's stale.
- Restored `runs/lanes/w6_footpinstub_20260708/report.json` — found gutted (uncommitted) in the
  working tree at boot; committed banked copy was intact.
- No new CVAT exports found (epoch marker current) — label ingest idle until your next session.

## What's running (updated as the wave moves)
- paddlewire_p31 RESUMED (finish final census + report; commits held until verified).
- Disk-triage scout (read-only).
- Next dispatches: P2-2 decode checklist lane, browser-verify dev-bypass, ghost-viewer styling,
  then the ball retrain GPU lane once its spec constants are grep-verified.
