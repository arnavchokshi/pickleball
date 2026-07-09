# WAVE 7 BOOT PROMPT — FINAL (closed 2026-07-09 at wave-6 close; SUBJECT TO the owner-directed North Star refresh landing later the same session — if NORTH_STAR carries a newer dated PLAN-REFRESH note than this file, THE REFRESH WINS; re-derive at boot against the [WAVE-6 COMPLETE] BUILD_CHECKLIST bullet regardless)

Marching order = this file per TECH_BLUEPRINTS PART A §A.5, manual §12-§22 + best-stack doctrine
(Part IV rule 15 — BEST-STACK DELTA mandatory in every lane spec). Default fleet boot template:
`pickleball-fleet-snap-20260708-w6close` (READY, 200GB/45.6GB, ffmpeg + roboflow corpus + rally
videos BAKED @ clean 37b8ecd3f). Boot hygiene: 2 vendor-submodule git-status lines are BY-DESIGN
(third_party/pickleball_vendor_additions/RESTORE.md); anything else dirty = `git reset --hard`
first (truncated-core-file class observed twice). Mac-side: run a /tmp + pip-cache sweep at boot
(disk hit 100% mid-wave-6; ~17GB free is NOT enough headroom for BODY monolith pulls).

## Discipline line (binding)
FIRST HELD-OUT SHOT only after: owner labels materially toward >=10-20k frames (wave-6 close:
1,121 reviewed rows = 5.6-11.2%; 72 tasks / ~45.6k frames imported and waiting; owner ~240 f/hr);
candidate selected on internal-val THROUGH the bridge OFFICIAL + LoSO (fold manifest:
runs/lanes/w6_labelingest_20260708/loso_fold_manifest.json, 6 sources, all-disjoint); pre-registered
heldout_eval_ledger row + owner go. VERIFIED=0 until a gate passes. Public-corpus/thin-seed students
NEVER shoot. CORPUS CAVEAT (standing): the owner-labeled corpus is disagreement-SELECTED (hard
frames) — absolute F1 on it is NOT comparable to uniform internal-val bars; use it for candidate
ORDERING + training fuel; any bar meant for promotion needs a defined sampling protocol first.

## Wave-7 queue (priority order — re-derive at boot)
1. **BALL seed retrain on the 1121-row corpus (CRITICAL PATH):** seed recipe on the full reviewed
   corpus; BASE CHOICE IS AN EXPLICIT DECISION (wave-6 re-score: seed_official micro-F1 0.5329/hFP
   0.2255 WINNER; stage1_official 0.2971 BELOW control 0.3611 on hard frames — stage1-as-base is
   now questionable; candidate bases: seed_official itself, stage1_official, raw WASB; decide with
   a small ablation, control row mandatory, budgets in STEPS w/ measured probe). Re-score through
   OFFICIAL bridge + LoSO-mean; --resume-checkpoint now exists for stage2. Keep ingesting owner
   exports as they land (watchdog class-G; converter standing; sessions 02-04 then w6 sessions).
2. **P2-2 decode-fidelity ROOT CAUSE (gates all latent work):** GATE-1b legitimately FAILED
   (262.35mm world round-trip vs <=1mm; mesh-skel 53.50mm p95 vs <=5mm; harness now trustworthy —
   scale/hand_pose/provenance verified; p95 IDENTICAL raw-vs-default arm => decode-internal,
   post-chain-invariant). Diagnosis-before-fix: where does the world round-trip lose 260mm
   (world-frame transform? betas? translation/scale application inside mhr_decode? persisted-fields
   gap?). The deep-review research may reshape this — check NORTH_STAR refresh first. Until it
   passes: lambda_foot=0, smoother UNWIRED, latent-interp playback OFF, refine un-kill DOES NOT
   PROCEED (all standing).
3. **P6-1 rule-based shot classification** (trust-banded on current P1 outputs; stream-4 design
   landed). Critical-path note: P1-4 FULL lift (learned rescue + STEP 5 view confidence) remains
   OPEN (Magnus honest-killed, dormant); if capacity allows, P1-4/STEP-5 rides as its own lane —
   it unlocks P3-5 + PF-1 + coaching S1 together (B.3 spine).
4. **P6-2 minimal stat set** (unforced errors, third-shot success, dink-rally win rate; bands +
   "how we measured").
5. **P6-3 reference-range library v0** (owner/coach review; versioned JSON).
6. **P6-4 grounded coach v0** (deterministic features -> rule comparator -> format-locked LLM;
   fabrication audit pre-defined: owner + >=1 4.0-rated reviewer, 300-output sample, 0-fabrication
   bar; LLM sees ONLY comparator verdicts).
7. **P6-5 visual feedback overlays** (>=5 finding types; browser-verified — NEEDS the dev-bypass,
   carried AGAIN: coordinate w/ product-infra session; wave-6 closed numeric-only).
8. **P5-5b pre-flight sanity gate + P5-6 auto-QA** (cheap reliability primitives).
9. **Mesh byte-budget DEFAULT decision (owner):** best_stack PENDING row; evidence 5.21->21.32 fps
   outdoor / 112.6MiB, wolverine 1.67x / 77MiB; promote to result-run default on owner OK; ALSO
   decide human_review-tier mesh display (banded?) — the remaining playback ceiling (742/1151
   outdoor frames tier-excluded).
10. **Micro-debt:** pipeline_summary stages[body].metrics.postchain_bypassed_stages empty (5 other
    sources agree — one-file fix); replay_viewer_manifest degrades from /tmp-worktree orchestration
    (Vite allow-root — tooling note or fix); import_w6_labelpack_tasks.py not idempotent (guard
    before any re-run).
11. **Owner-gated (surface at boot, never mid-wave):** 2x phone tests (LiDAR -> P4-7 build/kill;
    ARKit sidecar) · OWNER GAME RECORDING (M4 needs one; WITH-audio held-out pre-registration at
    capture) · paddle marker-GT session (RKT; every boot until booked) · 4.0-rated reviewer for the
    P6-4 audit · label sessions (the compounding item).

## What wave-6 banked (verify against the [WAVE-6 COMPLETE] bullet)
- Owner labeling flywheel OPERATIONAL end-to-end: label -> export -> auto-ingest (watchdog class-G,
  epoch marker runs/manager/.w6_export_epoch) -> deterministic corpus rebuild -> LoSO re-score
  recipe (GPU_RESCORE_COMMANDS.sh precedent). 73 tasks imported (import script NOT idempotent).
- BODY raw-postchain knob end-to-end (loud bypass, validator-complete stub, fixture-covered) +
  canonical decode harness w/ provenance = the P2-2 measurement infrastructure is TRUSTWORTHY now.
- Mesh byte-budget policy landed (opt-in; no-flag byte-identical); measured playback gains banked.
- Magnus scalar-S dormant (kill honored); 541/276 closed-as-explained (player-frame units);
  BVP whole-span harness v2; stage2 --resume-checkpoint; charuco capability-guard (cv2 5.0.0 has
  aruco WITHOUT detectMarkers — importorskip alone insufficient).
- Adjudication 3245/0/26 fully green post-guards. Costs: wave GPU $2.27-16.96 (mid ~$5-9).
- Ops: parent-exit-orphaned codex dispatches survive but lose notifications (dispatch each lane as
  its OWN run_in_background call or arm a report watcher); ignored-path git-add exits 1 while
  staging (use `;` not `&&`); power-cycle recovery = nohup VM-side + SendMessage transcript-resume
  (proven again); best_stack.json is the defaults manifest (28+ entries; read before wiring).

## Per-wave invariants (VI.7 + rule 15 — print into every lane spec)
Safe-parallelism · diagnosis-before-fix · exact gated metric keys · one adversarial verify round per
repair round w/ executable defect proofs · acceptance THROUGH the pipeline entry point ·
predictor-gated GPU spend · version-stamp @ committed blobs (commit landings BEFORE their GPU runs) ·
control rows everywhere · budgets in STEPS w/ measured probe · one clean wide-suite adjudication ·
fresh-GPU proof + browser verify (right manifest) · docs reconciliation · teardown + cost honesty ·
BEST-STACK DELTA in every lane (promoted/PENDING/DORMANT same-lane; defaults resolve ONLY through
the manifest) · scorecard + next boot prompt + inflight_lanes + memory · OWNER CRITIQUE SESSION on
close-proof worlds (standing ritual).
