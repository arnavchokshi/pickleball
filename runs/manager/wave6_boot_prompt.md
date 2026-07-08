# WAVE 6 BOOT PROMPT — DRAFT (written mid-wave-5 2026-07-08; manager splices [W5-CLOSE] markers at closeout — if any marker survives unfilled, read the [WAVE-5 ...] BUILD_CHECKLIST bullets first)

Marching order = this file, re-derived against the latest `[WAVE-5 ...]` BUILD_CHECKLIST bullets
(closeout wins where it differs). Boot per TECH_BLUEPRINTS PART A §A.5, manual §12-§21 standing
rules. Default fleet boot template: `pickleball-fleet-snap-20260708-ffmpeg` (READY, 200GB,
ffmpeg baked, clean HEAD @ 62d785ce3-adjacent).

## Discipline line (binding, unchanged)
FIRST HELD-OUT SHOT only after: owner in-domain labels materially toward ≥10-20k frames
(wave-5 start: ~486; owner labeling sessions began 2026-07-08 — count = [W5-CLOSE: reviewed-frame
count from labelpack ingests]); candidate selected on internal-val THROUGH the bridge in OFFICIAL
mode AND through LoSO (P1-9) — a mixed/random-split winner does not qualify; pre-registered
heldout_eval_ledger row + owner go. VERIFIED=0 until a gate passes. Public-corpus/thin-seed
students NEVER shoot (4 inversions on record).

## What wave-5 banked (context, verified at close)
- OFFICIAL preprocessing contract in training (c1f707d6f): tensor-identity + label-geometry
  contract tests; all wave-4 harness_v0 ckpts incompatible-by-contract.
- Aligned stage-1 + seed cards, OFFICIAL + LoSO (w5_ballretrain PASS, queue #1 CLOSED): control
  exact 0.7143/0.7826; stage1_official burl 0.8636/wolv 0.7500 (bar 0.6685 both), recall 0.7708 vs
  0.6875, hFP 0.1833 vs 0.65 — w4 degenerate class GONE; LoSO-mean stage1 0.7094/hFP 0.2812 beats
  control. seed_official (486 rows) honest-mixed (wolv 0.60, hFP 0.80, LoSO 0.6404) = SMALL-N
  SUSPECT: re-run seed on the banked stage1_official base when owner labels grow. One spot
  preemption, recovered clean (~19 GPU-min); train_ball_stage2.py lacks --resume-checkpoint (fix
  candidate). Benchmark lacks ball-size-binned recall (booked).
- BVP span protection v2 LANDED 792fa5fc6 (one verify round; axis-4 harness staleness re-ruled;
  fresh floors B 0.7727/W 0.8750). arcdiag ruling: img1605 0-arc = anchor STARVATION quality gap
  (NOT a compensation regression; boot-prompt "p06 had 7" was a transcription error — corrected).
- P2-2 phase 1 (62d785ce3): latent FAITHFUL (GATE 1a 2.7e-5°); scale_params silent-drop schema
  fix; smoother prototype UNWIRED (proxy jitter −47-76%); real-key acceptance table =
  [W5-CLOSE: w5_p22wiring λ-sweep table + wiring recommendation].
- Fast-SAM-3D-Body bench (owner-funded $15): [W5-CLOSE: verdict ADOPT-CANDIDATE/NOT-ADOPT/
  INCONCLUSIVE + speed/MPJPE tables from w5_fastbody_bench].
- Owner labeling LIVE: local CVAT v2.69 at localhost:8080 w/ 5 tasks imported+verified (4x640f
  ball sessions — BOTH OUTDOOR sources — + court-kp 4f); runbook cvat_upload/OWNER_SESSION_20260708.md;
  creds data/credentials/cvat_local.txt. Phase-B 24-clip prediction extension SKIPPED on w5ball
  (videos not on VM) — ride the next GPU errand; queue still 16/40 clips / 2/6 sources until then.
- Transport hardening: [W5-CLOSE: w5_transport lane landing — resumable chunked transport +
  known_hosts-into-pinned-worktrees, or carried with reason].

## Wave-6 queue (priority order — re-derive at boot)
1. **Owner-label ingest + LoSO outdoor fold (CRITICAL PATH):** ingest owner CVAT exports as they
   land (import path proven in cvat_upload/ precedent), rebuild the reviewed corpus, add the
   OUTDOOR fold to LoSO internal-val, re-score the wave-5 aligned candidates through bridge
   OFFICIAL + LoSO-mean with the new fold. This is what makes candidate selection
   inversion-resistant. Throughput watch: owner ~240 frames/hr; budget ≥10-20k frames needs
   many sessions — keep packages flowing (Phase-B predictions extend the queue to 40/40 clips).
2. **P2-2 wiring decision + strict GATE 1b:** [W5-CLOSE: adjust per p22wiring table]. Strict
   apples-to-apples GATE 1b recipe (BANKED, not yet run): instrument ONE clip's BODY dispatch
   with the post-chain (temporal smoothing/foot-lock/foot-pin/contact-splice/wrist-lock) OFF,
   persist raw grounded joints, compare decode(emit) ≤1mm + mesh-skel ≤5mm p95 legitimately.
   Then λ selection on 4-clip coverage → wiring ruling → grounding_refine un-kill attempt rides
   on gate-safe phases (λ_foot stays 0 until then).
3. **Magnus STEP 2 (UNLOCKED by BVP v2):** per TECH_BLUEPRINTS BALL 3D pillar; mandatory order
   honored (span protection landed first). Also carry the img1605 anchor-starvation quality gap:
   any Magnus/arc work reports the img1605 census as a diagnostic row (no bar, no tuning).
4. **Fast-body adoption decision** [W5-CLOSE: only if bench verdict = ADOPT-CANDIDATE]: P5-7
   spike lane on our gates/clips; kill = any internal-val accuracy regression (pre-ruled).
5. **Docs/instrument debt:** v2 BVP verify-harness revision (whole-span policy — w4 harness
   axis-4 check is manager-ruled stale, kept as historical instrument); CAPABILITIES.md
   truth-claims for wave-5 landings; frame-scheduling nondeterminism watch item (541 vs 276
   frames same clip after body_compute_execution.json delete — investigate before any
   reproducibility-sensitive proof).
6. **Owner-gated (fire when owner acts):** 2× 5-min phone tests (LiDAR range → P4-7 build/kill;
   ARKit sidecar pose → P0-10/PF-2/P4-6); recording sessions when owner can (held-out WITH-audio
   pre-registration at capture); court-kp relabel ingest.

## Standing facts current at wave-5 close
- Fleet: [W5-CLOSE: final list-proof — expect zero VMs, fleet1 TERMINATED disk-intact];
  template = pickleball-fleet-snap-20260708-ffmpeg; H100 spot default (BODY 2.37×); a3 needs
  pd-balanced; describe-quota lags create; --remote-host REQUIRED; IPs recycle.
- Costs wave-5: [W5-CLOSE: exact per-VM spans + wave total vs budget].
- Safety system (KEEP for wave-6 — it caught a real preemption in ~15 min): read-only manager
  watchdog script `runs/manager/w5_watchdog.sh` (copy/adapt lane+VM lists per wave; 10-min
  cycles: preemption/cost/stall/quota-wall/board-regression/auth-challenge → first-anomaly
  wake); pre-staged adversarial verify specs for gate-adjacent lanes (BVP closed in ONE round
  vs wave-4's three); mid-lane advisories (corpus-transfer fallback ladder; preemption recovery).
- Concurrent-session reality unchanged: git fetch before push-state assumptions; selective-hunk
  staging; boards manager-single-writer; PR merges can drop board bullets (watchdog checks).

## Per-wave invariants (VI.7 — print into every lane spec, unchanged)
Safe-parallelism · diagnosis-before-fix · exact gated metric keys · one adversarial verify round
per repair round w/ executable defect proofs · acceptance THROUGH the pipeline entry point ·
predictor-gated GPU spend · version-stamp before trusting VM metrics · control rows on every
measurement pipeline · budgets in STEPS w/ measured probe before wall caps · one clean wide-suite
adjudication · fresh-GPU proof + browser verify (right manifest) · docs reconciliation ·
teardown + cost honesty · scorecard + next boot prompt + inflight_lanes + memory.
