# WAVE 7 BOOT PROMPT - REFRESH LANDED (2026-07-09)

North Star refresh status: LANDED. The prior "SUBJECT TO refresh" header is resolved; boot against
`NORTH_STAR_ROADMAP.md` Part VI, `CAPABILITIES.md`, `TECH_BLUEPRINTS.md`, and the wave-6 closeout
bullets. Refresh evidence base: `runs/research_w6refresh_20260709/{RULINGS.md,internal_audit_synthesis.md,sota_ball_synthesis.md,sota_body_synthesis.md}`.

Marching order = refreshed NORTH_STAR + TECH_BLUEPRINTS PART A section A.5 + manual sections 12-22
+ best-stack doctrine. Default fleet boot template: `pickleball-fleet-snap-20260708-w6close`
(READY, 200GB/45.6GB, ffmpeg + Roboflow corpus + rally videos baked at clean 37b8ecd3f). Boot
hygiene: 2 vendor-submodule git-status lines are BY-DESIGN
(`third_party/pickleball_vendor_additions/RESTORE.md`); anything else dirty on a fleet VM =
reset-hard first before running. Mac-side: run a `/tmp` + pip-cache sweep at boot.

## Discipline line (binding)

FIRST HELD-OUT SHOT only after: owner labels materially progress through the gated checkpoint plan
and not a flat 10-20k sprint. Current wave-6 state: 1,121 reviewed rows, disagreement-selected
hard-frame corpus, 72-73 tasks / ~45.6k frames imported and waiting, owner throughput observed around
240 frames/hour. Checkpoint evals at 1k / 3k / 6k / 10k; if the curve flattens, owner time shifts to
venue/lighting diversity or coaching labels. Add a uniform-random audit stratum alongside the
disagreement queue; split ledgers into seen-vs-unseen environments. Pre-registered
heldout_eval_ledger row + owner go still required. VERIFIED=0 until a documented gate passes.

## Required boot tables

1. Fleet-spend-vs-ask table before dispatch: lane, GPU SKU/hours requested, expected dollar range,
   gating evidence to earn the spend, owner decision needed, and what waits if declined.
2. Owner-time queue from NORTH_STAR VI.8, ranked once at boot: labels, owner game recording, paddle
   marker-GT, two phone checks, 4.0 reviewer, GCP invoice, mesh display ruling, and P4-0-vs-retrain
   re-confirmation.

## Wave-7 queue (priority order - re-derive at boot)

1. **BALL seed retrain on the 1,121-row corpus (critical path):** base choice is explicit. Wave-6
   ordering: `seed_official` micro-F1 0.5329 / hidden-FP 0.2255 / LoSO-mean 0.5584 WINNER;
   `stage1_official` 0.2971 below control 0.3611 on hard frames. Re-run the 486-row anomaly before
   the next seed fine-tune. Candidate bases: `seed_official`, raw WASB, `stage1_official`. Include
   control rows, budget in steps with measured probe, and score through OFFICIAL bridge + LoSO.
   R2 gates: 1k/3k/6k/10k checkpoint evals, uniform-random audit stratum, seen-vs-unseen ledger,
   occlusion-augmentation recipe item.
2. **P3-1 paddle wiring IMMEDIATE:** fused 6-DOF estimator is the oldest BUILT-NOT-WIRED orphan
   (4 waves). Wire through `best_stack.json`, emit `racket_pose_estimate.json` by default, fail
   closed when evidence is absent, keep ESTIMATED/preview trust bands, no RKT promotion.
3. **P2-2 decode-fidelity CHECKLIST lane:** GATE-1b legitimately failed (262.35mm world round-trip
   vs <=1mm; mesh-skeleton 53.50mm p95 vs <=5mm). Run R1 checklist, not archaeology:
   audit MHR `conversion.py@4debaacf` L472-516 scale+axis handling; verify `pred_cam_t` added exactly
   once; confirm harness skeleton field (`pred_keypoints_3d` vs `pred_joint_coords`); treat world
   skeleton placement as our extrapolation; build the synthetic render round-trip gate. Apply the
   ceiling rule: if residuals are family-normal around ~50mm p95, stop chasing <=1mm and switch to
   locked identity/scale + latent smoothing workaround (arXiv:2512.21573) with gate recalibration.
   Until resolved: lambda_foot=0, smoother UNWIRED, latent-interp playback OFF, grounding_refine
   un-kill blocked.
4. **Browser-verify dev-bypass small lane:** stop carrying the INFRA-3 sign-in blocker. Restore
   browser verification as a normal closeout tool before coaching/viewer overlays depend on it.
5. **P5-1 clean-room speed gate scoring:** certify or fail the 3.8x speed headline before further
   speed claims. Include Wolverine <=400s, Outdoor <=2x video duration, six-run variance, foot-slide
   bit-identical check, and profile actual BODY/decode share during the R1 lane.
6. **P2-4 SAM-Body4D masklet-conditioning eval candidate:** cheap decode-independent candidate.
   Adopt only on measured batching/runtime or raw-noise win with no accuracy regression.
7. **P6 items stay queued:** P6-1 rule-based shot classification, P6-2 minimal stats, P6-3 reference
   library, P6-4 grounded coach, P6-5 visual overlays. BODY+COURT-only stats can move first; ball/
   paddle-dependent stats wait for trusted P1/P3 outputs.
8. **First-class pre-launch gates added by refresh:** P5-5b input-quality guardrail, P7-4c security/
   PII/secrets review, P7-4d training-data licensing check (GPL PnLCalib, Roboflow ToS vs Stripe
   monetization).
9. **Mesh byte-budget owner decision:** evidence outdoor 5.21->21.32 fps / 112.6MiB, Wolverine 1.67x
   / 77MiB. Owner ask is 300 vs 400 MiB plus whether human_review-tier mesh display is allowed when
   banded.
10. **Micro-debt:** pipeline_summary `stages[body].metrics.postchain_bypassed_stages` empty; Vite
    allow-root degradation from `/tmp` worktrees; `import_w6_labelpack_tasks.py` idempotence guard
    before any re-run.

Court note: P4-0 court-profile library is ruled ahead of any 3rd auto-find retrain pending owner
re-confirmation with the CALV1 evidence (244.3px Burlington / 212.6px Wolverine, pool containment
0/8 and 2/8). CALV1 owns detailed court section/board edits.

## What wave-6 banked

- Owner labeling flywheel operational end to end: label -> export -> auto-ingest -> deterministic
  corpus rebuild -> LoSO re-score recipe. Corpus 486 -> 1,121 rows; 68 labelpack sessions imported.
- BODY raw-postchain knob + canonical decode harness trustworthy; GATE-1b legitimate FAIL, so P2-2
  is NOT-WIRING-READY.
- Magnus scalar-S dormant after real-evidence kill; future spin path is TT3D bounce-kink gated on
  view geometry + bounce-frame availability, not solver retuning.
- Mesh byte-budget policy measured real playback gains; remaining display/default policy is an
  owner decision.
- Adjudication 3245/0/26 green post-guards. Fleet zero VMs list-confirmed. GPU wave cost
  $2.27-16.96, mid roughly $5-9.
- Best-stack doctrine: defaults resolve through `configs/racketsport/best_stack.json`; every lane
  reports PROMOTED/PENDING/DORMANT delta.

## Per-wave invariants

Safe-parallelism; diagnosis-before-fix; exact gated metric keys; one adversarial verify round per
repair; acceptance through `scripts/racketsport/process_video.py`; predictor-gated GPU spend;
version-stamp at committed blobs before trusting VM metrics; control rows everywhere; budget in
steps with measured probe; one clean wide-suite adjudication; fresh-GPU proof + browser verify on
the right manifest; docs reconciliation; teardown + cost honesty; BEST-STACK DELTA in every lane;
scorecard + next boot prompt + inflight_lanes + memory; owner critique session on close-proof worlds.
