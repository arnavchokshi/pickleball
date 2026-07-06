# OWNER CHECK-IN — 2026-07-06 (wave 2)

## STOP — RESOLVED (2026-07-06): you supplied the Roboflow key. Stored gitignored at data/credentials/roboflow.env (chmod 600, never committed); PART 0 item ticked; P1-0 Universe download lane dispatched (option A). Nothing awaits you here.

⭐ HEADLINE: Wave 2 dispatched — 6 file-fenced Codex lanes + the GVHMR GPU spike; gcloud auth ALIVE and service-account impersonation verified working (no login needed from you; wave-1 queue item 8 is thereby DONE); fleet1 restarting for the spike.

## Blockers
None — Roboflow resolved; all lanes running.

## Wave-2 lanes (your priority order)
1. rootjump_slide_fix — burlington/outdoor blocker root-caused to a per-frame ABAB flip between two self-consistent positions (0.85m / 3.0m alternation) = selection/identity bug; fixing at source, slide → ≤30mm, plus frame_idx-null producer + foot_contact_phases producer ride along.
2. p01b_harvest_ingest — 8 games → rally clips + roles + dedup + corpus card; GPU prelabel run queued after; 2 held-out game PROPOSALS will await your go (ledger stays manager-written).
3. p08_vfr_pts — iPhone VFR/PTS correctness audit + frame-time tables (H27b).
4. p21_cammotion — camera-motion module upgrade (handheld robustness), wiring via deferred patches.
5. p27a_gvhmr_spike — GVHMR vs our stack, 2 tripod clips × 4 players on fleet1; also triangulates the burlington flip.
6. p11_visibility_schema — 4-level ball-visibility taxonomy end-to-end (P1-1 training prereq).
7. dispatch_hardening — wave-1's rsync failure class killed (tar-batch transport + bounded retries).

## Money / GPU log
fleet1 (A100-40GB spot, ~$1.2/hr) RESTARTED for the GVHMR spike; likely reused for the burlington/outdoor verify re-runs after the fix lands; STOP/teardown at wave end per standing rule. No new VMs yet; cap 4 × ≤$5/hr stands.

## Verify when back
- The Roboflow ask above.
- Wave-2 closeout table: burlington/outdoor blockers gone + slide ≤30mm (manager browser-verifies before claiming).
- The 2 proposed held-out harvest games (listed here when the lane reports).
