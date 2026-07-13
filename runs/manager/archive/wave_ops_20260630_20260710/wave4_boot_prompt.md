# WAVE 4 BOOT PROMPT — the first training wave (M3 opener: "the ball actually learns")

Written 2026-07-07 by the wave-4 manager session at wave open. Marching order = NORTH_STAR PART VI.2
as re-derived by the `[WAVE-3 COMPLETE 2026-07-07]` BUILD_CHECKLIST bullet (its WAVE-4 QUEUE wins).
Manager coordination branch: `worktree-wave4-manager` (boards edited there, merged at close —
wave-3 pattern). Lanes run on the MAIN checkout with file fences.

## Discipline line (binding)
**INTERNAL-VAL ONLY — no held-out shot this wave.** A public-corpus-only student never takes a
held-out shot (4 inversions on record). Held-out attempt = wave 5, after owner in-domain volume.
VERIFIED=0 unchanged. Every gate-adjacent fix gets one adversarial-verify round per repair round
with executable defect proofs (§20). Acceptance for pipeline-integrated features is measured
THROUGH process_video, never a lane-local replica. Exact gated metric keys copied verbatim.

## Concurrent reality at open
- Other session live: wave-2 manager session owns **ios/** (brand-v2 verify leg) + uncommitted
  BUILD_CHECKLIST bullet 710 + working-tree ios changes. NEVER touch ios/**; never `git add -A` on
  the main tree; commit lane landings with explicit pathspecs only. `git fetch` before any
  push-state assumption.
- Fleet at open: fleet1 STOPPED disk-intact (only VM; list-confirmed via impersonated call, auth OK
  2026-07-07). Snapshot `pickleball-fleet1-snap-20260707` READY = fan template. IPs RECYCLE on
  restart: refresh configs/ssh/a100_known_hosts + pass `--remote-host` explicitly, always.
- Owner court-keypoint labels landed on main (task 13: 49 usable points / 5 frames; 3 full sources,
  2 tennis-overlay partial, 1 declared-skip) — feeds queue #4.

## The queue → lane map (fences in [], all file-disjoint)
Phase 1 (dispatched at open, parallel):
1. `w4_cammotion_diag` (Codex, READ-ONLY) — queue #1: why in-pipeline probe scored 0.329 vs 53.7
   offline on img1605. Deliverable: root cause + executable repro + fix design. [writes only its lane dir]
2. `w4_footattr_diag` (Codex, READ-ONLY) — queue #2: upstream per-foot attribution design to un-kill
   grounding_refine (evidence: w3_slidediag §6, w3_groundref_diag §5). [lane dir only]
3. `w4_burlmesh_diag` (Codex micro, READ-ONLY) — queue #5: burlington virtual_world
   missing-mesh-vertices notice root cause. [lane dir only]
4. `w4_fleethosts` (Codex micro) — queue #6: kill DEFAULT_REMOTE_HOST footgun + known_hosts refresh
   protocol. [scripts/racketsport/remote_body_dispatch.py, configs/ssh/, tests/racketsport/test_remote_body_dispatch.py, scripts/fleet/]
5. `w4_bvp` (Codex) — W4-D P1-4a BVP stabilization per BALL3D STEP 1 (a2-first).
   [threed/racketsport/ball_arc_solver.py, tests/racketsport/test_ball_arc_solver.py]
6. `w4_ballcode` (Codex) — BALL2D STEP-3 BUILD GAP: owner-CVAT stage-2 training path + occlusion aug
   + SST pseudo-label feed + disagreement emitter. [scripts/racketsport/train_ball_stage2.py (new),
   train_ball_pretrain.py (minimal shared-helper edits), threed/racketsport/ball_sst_dataset.py (new),
   tests/racketsport/test_ball_stage2_*.py (new)]
7. `w4_court_harvestcal` (Codex) — queue #4: per-source harvest court calibration from owner CVAT
   court-kp labels → calibration artifacts consumable by the ball 3D chain; coverage over the 40
   prelabeled clips. [scripts/racketsport/calibrate_harvest_courts.py (new) + its test; read-only on
   court_calibration*, cvat_upload/court_keypoints_20260707/]
Phase 2 (gated):
8. `w4_ballgpu` (Sonnet GPU, H100 spot, self-provision→run→verify→DELETE) — AFTER w4_ballcode rules
   PASS: STEP-1 WASB prestage (sha 9d391239… per models/MANIFEST.json), seed fine-tune on
   ~274-box owner labels from stage-1 ckpt, SST round 1 (teacher = raw single-WASB = the 40 local
   prelabel sidecars), threshold/recall sweep; ALL scoring through the SCORING BRIDGE
   (run_wasb_ball → fuse → run_ball_tracking_eval_suite) on Burlington+Wolverine internal-val.
9. `w4_cammotion_fix` + `w4_footattr_fix` (Codex) — AFTER manager rules the diagnoses. Each ships
   with an independent adversarial-verify round (gate-adjacent).
10. `w4_h100body` (Sonnet GPU, CONDITIONAL, queue #7) — H100 BODY-compat validation from the
    snapshot; only if wave capacity allows; never a decisive run.
Phase 3 (close): integration micro-lane (if any fenced-file patch was deferred) → ONE clean wide
suite (MPLBACKEND=Agg; tests/racketsport minus test_court_finding_technology_benchmark.py, which
runs standalone) → fresh-GPU proof (snapshot→fan, 4 eval clips, A100 = proven BODY SKU;
version-stamp FIRST; browser-verify replay_viewer_manifest.json) → docs reconciliation (W4-F) →
commit+push → `[WAVE-4 COMPLETE]` bullet + wave-5 boot prompt + inflight_lanes + memory.

## GPU budget (stated up front per VI.2)
H100 spot ball lane ~1.5–3h ≈ $3–12 · wave-end A100 4-clip fan ≈ $2–3 · optional H100 BODY compat
≈ $2–4 · contingency ≈ $5. Expected total ≈ **$12–25**, within ≤$5/GPU/hr × ≤4 concurrent.
>$5/hr or a 5th GPU = needs-purchase-approval STOP. Teardown on completion, always.

## Exit contract (from VI.2, re-derived)
- Seed fine-tune + SST-r1 checkpoints banked WITH internal-val product cards (label_f1_at_20px,
  mean_visible_hit_recall, mean_hidden_false_positive_rate on the 2 DEFAULT_CLIPS) — proxy numbers
  never promote anything.
- Recall levers measured individually (threshold sweep this wave; tiled/motion-channel recipe-ready).
- BVP: the 5 exact baseline intervals keep `fit*` status w/ endpoint_error_m ≤ baseline; D.3(e)
  internal F1 rerun clean.
- Cammotion + foot-attribution: diagnosis rulings booked; fixes landed if diagnoses permit, with
  adversarial verify; else honestly carried with the ruling recorded.
- Harvest court calibration: per-source artifacts + coverage table (which of 40 clips calibrated);
  physics-gated teacher unlock status stated honestly.
- Scorecard states wave-5 held-out-shot preconditions explicitly (owner label volume vs the
  ≥10–20k P0-4 budget; current: ~274 boxes ≈ 480 frames).

## Owner-dependency ladder (surfaced at boot, nothing blocks the wave)
1. **Owner captures** (ETA ~2026-07-09): W4-E fires the moment they land (P0-3 ingest runbook =
   TECH_BLUEPRINTS DATA D2; ≥2 held-out WITH AUDIO reserved before any prelabel).
2. **Next labeling session:** ball corrections toward the 10–20k budget — the SST disagreement
   queue from w4_ballgpu will front-load it (active learning); remaining court-kp sources.
3. Wave-5 prep (not needed yet): paddle 4-marker GT session, ChArUco per-lens sweep, tape-measured
   net heights, lens-preset confirmation.

## Standing invariants (VI.7, printed per contract)
Safe-parallelism per lane · diagnosis-before-fix · exact gated keys · adversarial verify on
gate-adjacent claims · version-stamp before trusting any VM metric · one clean wide-suite
adjudication · fresh-GPU proof + browser verify (right manifest) · docs reconciliation · teardown +
cost honesty · scorecard + next boot prompt + inflight_lanes + memory. Critical-path guard: lanes
5, 6, 7, 8 are critical-path (BALL/flight/DATA) — satisfied.
