# WAVE 7 BOOT PROMPT — DRAFT (manager fills FINAL at wave-6 close; re-derive against the [WAVE-6 COMPLETE] BUILD_CHECKLIST bullet — closeout wins where it differs)

Marching order = this file, re-derived at boot per TECH_BLUEPRINTS PART A §A.5, manual §12-§22
standing rules. Default fleet boot template: pickleball-fleet-snap-20260708-ffmpeg **UNLESS the
wave-6 auth-gated errand re-cut a new template — check gpu_fleet.md for the newest TEMPLATE POLICY
line; the ffmpeg template booted DIRTY twice (5 truncated core files on the w6 errand; reset --hard
recovers) — hygiene-check EVERY boot regardless.**

## Discipline line (binding, unchanged)
FIRST HELD-OUT SHOT only after: owner in-domain labels materially toward >=10-20k frames (wave-6
close: 1,121 reviewed rows = 5.6-11.2% of bar; owner throughput ~240 f/hr; 72 more tasks imported
and waiting in local CVAT); candidate selected on internal-val THROUGH the bridge in OFFICIAL mode
AND through LoSO (the 6-fold all-disjoint manifest with the OUTDOOR owner fold now EXISTS:
runs/lanes/w6_labelingest_20260708/loso_fold_manifest.json); pre-registered heldout_eval_ledger row
+ owner go. VERIFIED=0 until a gate passes. Public-corpus/thin-seed students NEVER shoot.

## Queue #0 — AUTH-GATED CARRY FROM WAVE 6 (fires FIRST the moment gcloud auth works; all inputs banked)
One consolidated self-tearing H100 errand (est ~2h, $2-8):
(a) LABEL RE-SCORE: run runs/lanes/w6_labelingest_20260708/GPU_RESCORE_COMMANDS.sh
    (official_tennis_control + stage1_official + seed_official over 20 internal-val clips, OFFICIAL
    bridge + LoSO-mean w/ the new outdoor fold; control row mandatory). Manager rules candidate
    ordering; seed_official small-N verdict refresh at 1121 rows.
(b) TEMPLATE RE-CUT: verify tree clean (git status EMPTY after reset), bake data/roboflow corpus +
    harvest rally videos (16-min transfer tax paid twice), cut pickleball-fleet-snap-<date>, update
    gpu_fleet.md TEMPLATE POLICY, verify one boot from it.
(c) LEGITIMATE GATE-1b RAW ARM: post-footpinstub raw dispatch (command:
    runs/lanes/w6_gate1b_knob_20260708/gpu_instrument_command.md) + canonical fixed harness
    scripts/racketsport/gate_check_body_decode.py w/ --scale-source field + provenance (exact
    invocation: w6_gatecheckfix report). THEN the P2-2 wiring ruling: if decode(emit) <=1mm AND
    mesh-skel <=5mm p95 legitimately -> lambda selection on 4-clip decoded coverage -> wiring ruling
    -> grounding_refine un-kill attempt on gate-safe phases; else document NOT-wiring-ready w/ the
    legitimate numbers and keep lambda_foot=0 / smoother UNWIRED. (Wave-5/6 numbers 220-233mm were
    HARNESS-ARTIFACT-suspect: scale+hand_pose dropped — do not cite them as decode truth.)
(d) MESH-CAP OUTDOOR PROOF: process_video outdoor w/ --mesh-coverage-mode ball_aware
    --target-mesh-frame-budget 0 --mesh-byte-budget-mib 400 (predicted ~267f ~13.9 mesh fps vs 5.21
    today); pull index sizes + plan audit fields; feeds the owner-facing playback claim.

## Wave-7 queue (VI.5 core — M4 "it coaches me"; re-derive at boot)
1. **P6-1 rule-based shot classification** (trust-banded, on current P1 outputs; stream-4 design
   landed — see memory + runs/). Spine note: P1-4 FULL lift (learned rescue + STEP 5 view-geometry
   confidence) remains OPEN on the critical path (Magnus STEP 2 was an honest kill — scalar S
   dormant; revisit gated on STEP 5). P6-1 proceeds trust-banded on existing arcs; P1-4 full-flight
   is a wave-7 candidate lane if capacity allows (it unlocks P3-5 reflection + PF-1 ball terms +
   coaching S1 features together).
2. **P6-2 minimal stat set** (unforced errors, third-shot success, dink-rally win rate — each with
   band + "how we measured").
3. **P6-3 reference-range library v0** (trade-benchmark seeds + OWNER/coach review; versioned JSON).
4. **P6-4 grounded coach v0** (3-stage: deterministic features -> rule comparator -> format-locked
   LLM; fabrication audit protocol DEFINED UP FRONT: owner + >=1 4.0-rated reviewer, Talking-Tennis
   rubric, 300-output sample, 0-fabrication bar). LLM sees ONLY comparator verdicts (B.4 rule 12).
5. **P6-5 visual feedback overlays** (>=5 finding types, browser-verified — NEEDS the dev-bypass).
6. **P5-5b pre-flight sanity gate + P5-6 auto-QA** (reliability primitives; P5-6 ships WITHOUT
   Phase-F residuals and says so — B.3).
7. **Labeling flywheel** (standing): ingest owner exports as they land (converter + folds standing;
   watchdog class-G pattern; sessions 02-04 next, then w6 sessions); re-run seed on the banked
   stage1_official base as labels grow.
8. **Browser-verify dev-bypass** (carried queue #6): coordinate w/ product-infra session; wave-6
   closed numeric-only again.
9. **Owner-gated:** 2x phone tests (LiDAR -> P4-7 build/kill; ARKit sidecar) · OWNER GAME RECORDING
   (M4 literally needs one; WITH-audio held-out pre-registration at capture) · paddle marker-GT
   session (RKT promotion; surface at every boot until booked) · 4.0-rated reviewer lined up for the
   P6-4 audit.

## What wave-6 banked (context — verify against the [WAVE-6 COMPLETE] bullet at boot)
- Owner labeling LIVE end-to-end: session-01 ingested (corpus 486->1121 deterministic), LoSO
  outdoor fold all-disjoint, ingest converter + schema landed, 73 tasks / ~46.7k frames imported in
  local CVAT (import script NOT idempotent — clear before re-run).
- BODY post-chain raw knob END-TO-END (CLI->RemoteConfig->VM runner; raw sidecar; loud bypass;
  foot_pin stub validator-complete + fixture-covered) + canonical GATE-1b harness (scale/hand_pose
  plumbing + fail-loud + provenance). GATE-1a fresh PASS 4.098e-05 deg.
- Mesh playback: fixed-100-frame cap -> byte-budget policy (opt-in; no-flag byte-identical); wolverine
  proof 100->167 mesh frames (~1.67x); REAL ceiling on wolverine = human_review-tier exclusion
  (77/244 frames) — whether to render lower-trust meshes BANDED is an OWNER/product ruling, queued.
- Magnus STEP 2: honest kill honored (S plumbing dormant, fit_spin_scalar=False; physics port proven
  S_hat err 0.0006; floors exact both modes). 541/276 closed-as-explained (player-frame units).
- stage2 --resume-checkpoint + BVP whole-span harness v2 + CAPABILITIES residuals (instrudocs).
- Safety: watchdog classes A-G (G=owner-export wake, epoch marker runs/manager/.w6_export_epoch;
  auth-restore-watch variant exists); ignored-path git-add exits 1 even when staging succeeds — use
  `;` not `&&` after adds touching runs/; parent-exit-orphaned codex dispatches survive but lose
  notifications — dispatch each lane as its OWN run_in_background call or arm a report-file watcher.

## Per-wave invariants (VI.7 — print into every lane spec, unchanged)
Safe-parallelism · diagnosis-before-fix · exact gated metric keys · one adversarial verify round per
repair round w/ executable defect proofs · acceptance THROUGH the pipeline entry point ·
predictor-gated GPU spend · version-stamp before trusting VM metrics (committed blobs; commit lane
landings BEFORE their GPU runs) · control rows on every measurement pipeline · budgets in STEPS w/
measured probe · one clean wide-suite adjudication · fresh-GPU proof + browser verify (right
manifest) · docs reconciliation · teardown + cost honesty · scorecard + next boot prompt +
inflight_lanes + memory · OWNER CRITIQUE SESSION on close-proof worlds (standing ritual).
