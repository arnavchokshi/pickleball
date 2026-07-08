# OWNER CHECK-IN — 2026-07-08 (wave 6)

⭐ **Headline:** Wave 6 is OPEN and running. Preflight green (auth OK, zero fleet VMs, ledger clean).
Five Codex lanes dispatched in parallel: Phase-B labelpack packaging (queue #7), the GATE-1b
raw-persist knob (queue #2), Magnus STEP 2 spin fitting (queue #3), instrument/docs debt (queue #5),
and a read-only playback-frame-rate diagnosis for your critique item #1. The critical-path label
ingest (queue #1) is STAGED HOT — it fires the moment your first CVAT export lands. VERIFIED=0 unchanged.

## Blockers (typed STOPs)

### STOP: needs-decision (gcloud reauth)
**One-line ask:** run `gcloud auth login` once (type `! gcloud auth login` in the Claude session for interactive flow).
**Why this needs you:** Google fired a periodic reauth challenge on the hello@ refresh token (known same-day pattern, manual §12); it cannot be answered non-interactively, and no standing rule lets me work around auth.
**Evidence:** `gcloud compute instances list` → "Reauthentication failed. cannot prompt during non-interactive execution" (watchdog class F, 2026-07-08 ~14:3x; re-confirmed manually).
**Options considered:** ssh-key fallback works for existing VMs but there are ZERO VMs up (nothing burning) and ssh cannot CREATE the final errand VM; waiting is safe.
**If you don't answer:** the wave-6 consolidated GPU errand (owner-label re-score + snapshot template re-cut + legitimate GATE-1b raw measurement) stays queued; everything else proceeds to wave close, and the errand items carry to the closeout as auth-gated.
**Everything else keeps running:** labelingest is finishing its wide-suite self-verification; all other wave-6 lanes are landed and ruled; $0 idle GPU spend.

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
