# WAVE 5 BOOT PROMPT — owner data in + the first held-out shot (M2 complete; M3 attempted)

DRAFT written 2026-07-07 by the wave-4 manager at close (final numbers from the w4_ballgpu report
spliced at the [BALLGPU] markers — if any marker survives unfilled, read
runs/lanes/w4_ballgpu_20260707/REPORT.md + report.json first). Marching order = NORTH_STAR PART
VI.3 as re-derived by the `[WAVE-4 COMPLETE]` BUILD_CHECKLIST bullet (the closeout queue wins
where it differs). Boot per TECH_BLUEPRINTS PART A §A.5, manual §12-§20 standing rules.

## Discipline line (binding)
The FIRST HELD-OUT SHOT this wave is the P1 promotion attempt — it happens ONLY: (a) after owner
in-domain label volume progresses materially toward the P0-4 budget (≥10-20k frames; wave-4 close
state: ~486 reviewed frames), (b) on a candidate selected on internal-val THROUGH the product
bridge in OFFICIAL preprocessing mode (harness_v0 cards are non-promotable by construction),
(c) with a PRE-REGISTERED heldout_eval_ledger row + owner go. A public-corpus-only or
thin-seed student NEVER takes the shot (4 inversions on record). VERIFIED=0 until a gate passes.

## Wave-5 queue (priority order; re-derive against the closeout bullet at boot)
1. **BALL preprocessing alignment (CRITICAL-PATH, blocks all future ball training):** make the
   training harness use the OFFICIAL WASB inference preprocessing (affine warp + ImageNet
   mean/std — wasb_adapter._preprocess_wasb_window is the contract; harness plain-resize+/255 at
   roboflow_corpus.py:981 is the defect). Then RETRAIN stage-1 (12k steps ≈ 26 min H100, image
   loader) + re-run the seed fine-tune on the aligned base; score everything through the bridge in
   OFFICIAL mode. Evidence: w4 finding (harness ckpts degenerate F1 0.0 under official mode vs
   official ckpt 0.714/0.783 same path; w4_bridgefix landed the measurement-mode stopgap at
   67298e599). [BALLGPU: splice the harness_v0 internal-val card numbers here as the pre-alignment
   reference row.]
2. **img1605 ball-arc empty-census diagnosis (read-only first):** first COMPENSATED decisive run
   produced 0 arc segments / 0 world frames (p06-era code had 7 segments; no w3 reference).
   Suspect: compensated 2D geometry feeding event selection/anchor logic. Evidence:
   runs/lanes/w4_freshproof_20260707/img1605/. Diagnosis → ruling → fix lane.
3. **P2-2 latent smoothing (BODY pillar STEP A)** — THE raw-noise fix, now with wave-4's hard
   motivation: honest per-foot phases are noise-limited (2-12 confident/clip; 120-167
   phase_penetrates_ground rejections/clip; predicted gate breaches 3/4 if wired). λ_foot=0 until
   the phases become gate-safe. The landed-unwired producer (75e438223) + its offline harnesses
   are the acceptance instruments. grounding_refine un-kill rides on this.
4. **BVP span protection (P1-4a completion)** — banked r3 design: frozen-baseline arc params as
   protected-span priors + junction repair BEFORE validity gates; no in-loop BVP. Acceptance =
   the w4_bvp_verify harness axes (fit-tier coverage/residuals/junction sanity/confidence tier),
   runtime ≤1.2× baseline replay. THEN Magnus STEP 2 unlocks (mandatory ordering).
5. **W4-E → W5 owner-data intake** (fires the moment captures land): P0-3 ingest runbook
   (TECH_BLUEPRINTS DATA D2 — roles-at-ingest build gap), ≥2 held-out WITH AUDIO pre-registered,
   P0-4 labeling throughput toward budget. The SST disagreement queue
   [BALLGPU: path + clip/frame counts] front-loads the owner's next ball-labeling session; the
   court-kp relabel ask (HyUqT7zFiwk + zwCtH_i1_S4 net/far-side points + replacement frames)
   extends harvest-cal coverage past 1/6 sources.
6. **Fleet/infra follow-ups:** transport hardening (Mac→GCP bulk upload unreliability — either a
   resumable chunked transport in remote_body_dispatch or standardize VM-local driving for
   decisive runs; w4 evidence in freshproof HONEST ISSUES); known_hosts into pinned worktrees as
   part of the lane recipe; ffmpeg into the snapshot template; consider refreshing the fan
   snapshot template post-wave-5 code churn.
7. **Docs:** fill any surviving [PENDING] markers; register any new artifacts; kill-list additions
   from wave-5 evidence.

## Standing facts current at wave-4 close
- Fleet: all wave-4 VMs deleted (list-proven); fleet1 STOPPED disk-intact; snapshot
  pickleball-fleet1-snap-20260707 = fan template (lacks ffmpeg). SKU policy: H100 spot = default
  for TRAINING and BODY (BODY-validated 2.37×, evidence runs/lanes/w4_h100body_20260707/);
  decisive gate runs stay on proven SKUs; a3 needs pd-balanced boot disk; describe-quota lags
  create. Version stamps hash committed blobs (190dea09f); --remote-host is REQUIRED everywhere.
- Ball assets: official WASB anchor LOCAL + sha-verified (models/checkpoints/wasb/); stage-1 ckpts
  runs/lanes/w3_p11_train_20260707/checkpoints/; wave-4 seed + SST-3k ckpts
  [BALLGPU: pulled paths]; SST manifest recipe proven (40 clips / 58,353 samples / 35-0
  protected-hash); harvest court cal: 1/6 sources manual_bar covering 8/40 clips
  (data/online_harvest_20260706/court_calibrations/).
- Frozen gates stand green on fresh GPU proof @ a93764203-adjacent tree: slide
  20.25/20.04/17.98/23.07mm; img1605 runs COMPENSATED (probe 50.02 AUTO ON) — its +6.4mm slide
  delta is a watch item, not a defect.
- Concurrent-session reality: other Fable sessions + the owner work this repo live. git fetch
  before push-state assumptions; selective-hunk staging for shared files; codex deferred "patches"
  are design notes (re-derive + git apply --check); the anti-passive-wait nudge works (SendMessage
  once, instantly).

## Per-wave invariants (VI.7 — print into every lane spec)
Safe-parallelism per lane · diagnosis-before-fix · exact gated metric keys · one adversarial
verify round per repair round on gate-adjacent claims w/ executable defect proofs · acceptance
THROUGH the pipeline entry point · predictor-gated GPU spend · version-stamp before trusting VM
metrics · one clean wide-suite adjudication · fresh-GPU proof + browser verify (right manifest) ·
docs reconciliation · teardown + cost honesty · scorecard + next boot prompt + inflight_lanes +
memory. Critical-path guard: queue items 1/2/5 are critical-path — satisfied.
