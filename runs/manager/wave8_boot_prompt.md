# WAVE 8 BOOT PROMPT (drafted at wave-7 close 2026-07-09; finalize any [PENDING-CLOSE] before dispatch)

Boot ritual: CLAUDE.md read order + VI.8 owner-time queue + fleet-spend-vs-ask table BEFORE dispatch.
Fleet template: the speed gate did NOT run at w7 close (parked on the reauth STOP) — boot from
`pickleball-fleet-snap-20260708-w6close` + scp the 2 manifest-pinned artifacts FIRST (waveb
calibration_curves.json + court_model_v2.pt, the 2x-confirmed gap); the speed-gate errand cuts the
w7close template when it runs. Watchdog: adapt w7_watchdog.sh (auth-down guard now built in);
class-G epoch marker = .w8_export_epoch (create at boot).

## OWNER RULINGS 2026-07-09 morning (bind wave-8 — full text in BUILD_CHECKLIST [OWNER RULINGS 2026-07-09 ~09:4x])
- MESH FULL-RATE + CONTACT SLO-MO: full 30fps mesh playback (no byte-budget frame exclusion; uncap/raise
  default) + automatic ~2x predicted-mesh slo-mo around contact windows (banded, render-interp first,
  latent-interp stays off). Storage/runtime NOT constraints for now; dedicated efficiency wave later.
  This is wave-8 queue item #2 (lane pair: pipeline mesh scheduling/budget + viewer slo-mo playback).
- Speed bars = INFORMATIONAL until the efficiency wave (measure + report, never gate features on them).
- Broad try-things grant (prose): masklet re-attempt authorized; if the third-party-exec classifier guard
  fires, surface the exact settings.json one-liner instead of retrying.

## Carried items (dispatch-ready, in priority order)
1. **P5-1 SPEED GATE (if not run at w7 close):** spec READY at runs/lanes/w7_speedgate_20260709/spec.md —
   parked on the gcloud reauth STOP of 2026-07-09 ~05:50. Fire it FIRST once auth is live. It carries the
   corrected GATE-1b numeric arm + the w7close template re-bake.
2. **BALL wave-8 lane #1:** train on the 1,750 corpus (md5 0bb2fc592361f2e9246a71701617c3e5) from base
   A_seed_official_aug (best_stack PENDING entry) + occlusion-aug (RULED: stays — it is the hFP mitigation);
   complete ARM B from its banked step-500 ckpt (runs/lanes/w7_ballcomplete_20260709/); score EVERYTHING via
   the prepared 1750 block (GPU_RESCORE_COMMANDS.sh, control-first). BINDING: w6-session rows = box-position-only
   (owner ruling — no visibility weighting/slices); ONE training arm per GPU (co-scheduling collapsed throughput
   3x on w7ballc); re-probe steps/s per fresh VM. LoSO harness micro-fix rides along: add TRUE per-source
   grouped folds alongside the per-clip folds (report both; no retroactive rewrites).
3. **P2-2 residual attribution (small):** the 23.4mm p95 field-A residual is unattributed (euler-decode drift
   vs grounding-replication candidates). One diagnosis lane on the banked w7p22 raw-chunk sample BEFORE any
   workaround/recalibration decision; then the R1 ceiling-rule question goes to the owner WITH the corrected
   GATE-1b numbers from the speed-gate arm. lambda_foot=0 / smoother / latent-interp stay untouched until ruled.
4. **P2-4 masklet:** owner grant question is in the check-in (execution permission + HF sam3 access). If granted:
   masklets-only A/B into mask_prompt_mode=manifest (the cheap shape) — NOT the 5-model full pipeline. If not
   granted: stays queued, no lane.
5. **P6 next steps on honest dependencies:** P6-1 shot classification can now consume trusted BODY+COURT
   (+ ball CANDIDATE arcs banded non-promoted); P6-3 reference library v0; match-stats verified-input upgrade
   when P1 promotes. M4 needs the owner game recording (owner-time queue #2).
6. **Security fixes (pre-launch gates):** the 3 HIGH from P7-4c (PII session-only conflict + 2 authz) become
   fix lanes whenever product work resumes; networked CVE/secret-history scan same lane.
7. **Wave-8 core per NORTH_STAR VI.6:** PF-1 consistency priors + PF-2 contact-coupled optimizer v0 ride once
   ball 3D flight (P1-4) has trusted inputs; P0-10 ARKit build is HARD-SCHEDULED here (owner+device time).

## Standing discipline (unchanged)
Held-out shot ONLY via checkpoint gates + uniform audit stratum + seen/unseen ledger + pre-registered row +
owner go. VERIFIED=0 until a documented gate passes. Best-stack doctrine (manifest rev-9 CONFIRMED, ball PENDING = A_seed_official_aug w/ named gate); BEST-STACK DELTA in every lane. Subagents never on
Fable. <=$5/GPU/hr, <=4 concurrent, DELETE on lane end. One adversarial verify per gate-adjacent repair round;
verifier harness UNMODIFIED. Acceptance through process_video. `;` not `&&` after git adds touching runs/.
Codex lanes each in their OWN run_in_background call. Owner rulings of 2026-07-09 (ghost meshes, 60fps+stride-2,
speed-cadence, court task-13, w6 visibility) are STANDING — read the two committed owner note files.

## Owner-time queue seed (refresh at boot)
1 labels (1,750 banked; 3k checkpoint next; uniform audit stratum task still owed) · 2 owner game w/ audio (M4)
· 3 paddle marker GT · 4 phone checks x2 · 5 4.0 reviewer · 6 GCP invoice · 7 mesh 300-vs-400 + human_review
display ruling (ghost machinery now fully landed — only the cap question remains) · 8 P4-0 re-confirmation ·
9 P2-4 execution grant (NEW).
