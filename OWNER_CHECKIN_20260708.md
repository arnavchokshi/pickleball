# OWNER CHECK-IN — 2026-07-08 (wave 6)

⭐ **Headline:** WAVE 6 IS CLOSING COMPLETE — all 8 queue items ruled. Your session-01 labels
went end-to-end: ingested (corpus 486→1,121), LoSO outdoor fold built, and the GPU re-score says
**seed_official (owner-data-trained) is the clear winner** — stage1 actually scores BELOW the
tennis control on hard frames, which vindicates your labeling as THE unlock. Playback critique:
outdoor mesh cadence went 5.2→21.3 fps (4.1x). GATE-1b was finally measured legitimately and
FAILED (262mm vs 1mm target) — so the latent-smoothing path stays honestly unwired; root cause is
queued. Magnus spin: honest kill, plumbing dormant. Next: the deep review + North Star refresh you
ordered (running tonight), then wave 7.

## Blockers (typed STOPs)

RESOLVED 2026-07-08 ~15:0x: owner re-authed; consolidated close errand (re-score + GATE-1b raw + mesh outdoor proof + template re-cut) dispatched on pickleball-h100-w6close.

## Asks (numbered, easiest-first)
1. **Label session 01 + export (CRITICAL PATH — the single highest-value hour you can give):**
   CVAT is UP at http://localhost:8080 with your 5 tasks loaded. Label
   `w5_ball_sst_ball_session_01_20260708` (640f, ~2.7h), then export:
   task page → Actions → Export task dataset → `CVAT for images 1.1` → save under
   `cvat_upload/exports/w5_labelpack_20260708/`. The watchdog sees the zip land and the ingest
   + LoSO-outdoor-fold lane fires automatically. Runbook: cvat_upload/OWNER_SESSION_20260708.md.
2. **More critique items?** Your playback-frame-rate critique is booked and being diagnosed
   (fix menu: render-side interpolation → mesh-cap raise → denser BODY scheduling). If the
   w5_critiqueviewer worlds showed you other worst-moments, reply with them (any format) and
   I'll batch them into a root-cause theory wave per the standing ritual.
3. **Relay to the product-infra session (or approve me to coordinate):** the INFRA-3 sign-in gate
   (commit 109235591) walls `verify_process_video_viewer` headless browser proof. We need a
   dev-bypass/service-login for the verify tool inside web/replay/ (their fence). One line to that
   session: "add a verify-tool bypass per wave-6 queue #6" — or tell me to spec it for them.
4. **Standing (when you can):** 2× 5-min phone tests (LiDAR range → P4-7 build/kill; ARKit sidecar
   pose → P0-10/PF-2/P4-6); recording sessions (held-out WITH-audio pre-registration at capture).

## Verify-when-back
- Wave-6 lane reports under runs/lanes/w6_*/report.json as they land (I rule on each).

## Money / GPU log
- Wave-6 spend so far: $0. Zero fleet VMs up. Planned: ONE H100 spot (~$0.6-4.3/hr) for the
  GATE-1b instrument run once the knob lane lands (+ the label re-score errand if your export
  arrives) — est $3-8, ≤$12 with contingency, inside the ≤$5/hr × ≤4 cap.
- Note: `body4d-waker-ctrl` (e2-micro, us-central1) is RUNNING but labeled cost-center=body4d —
  a different project's VM, not pickleball fleet; I did not touch it.

## Overnight log
- 11:2x dispatched 5 Codex lanes (labelpack / gate1b-knob / magnus / instrudocs / playback-diag);
  armed w6_watchdog (10-min cycles: preemption, cost, stall, quota wall, board regression, auth,
  owner-export-landed wake). Staged w6_labelingest spec hot.
