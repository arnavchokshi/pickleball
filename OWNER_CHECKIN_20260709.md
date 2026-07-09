# OWNER CHECK-IN — 2026-07-09 (WAVE 7 BOOT)

⭐ **HEADLINE:** Wave 7 is live on the refreshed North Star. Paddle wiring (P3-1) resumed to finish
its final census; the BALL seed retrain on your 1,121 labeled rows (the 1k-checkpoint gate) is this
wave's critical-path lane. No typed STOPs at boot — PART 0 verified (biometric-consent item stays
non-blocking until we persist a non-owner profile, which wave 7 does not do).

## Blockers
(none)

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
9. **P4-0 vs 3rd auto-find retrain re-confirmation** (CALV1 evidence: 244.3px Burlington / 212.6px
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
