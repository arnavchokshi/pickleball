# Benchmark lane spec — NS-03.RKT paddle keypoints/pose first step (SPEC-ONLY, no GPU dispatched)

Status: FINAL (Tier 1 ready to dispatch when the coordinator sequences it; Tier 2 pinned,
owner-GT-gated)
Author: Track F manager, 2026-07-16, per RKT_CROSSCHECK_RULING.md (dual survey + 2-vote
refutation). VERIFIED=0 binding; nothing here is a promotion.

## Objective

Run the first measurable RKT candidate step per the NS-03.RKT ruled sequence (GT capture →
RacketVision 5-keypoint zero-shot → pickleball fine-tune → both-IPPE retention → cross-cue
resolution), scored against the current preview baseline. Honest constraint acknowledged up
front: with ZERO pose/contact GT on disk, NO candidate can score against the real gates
(face-angle p90 ≤5°, contact-point p90 ≤3cm) yet. This lane therefore has two tiers:

- Tier 1 (runnable NOW, review-only + proxy metrics): candidate keypoints vs the existing
  rough-box protocol + both-IPPE pose plausibility diagnostics. Establishes integration,
  runtime, and a floor — NOT accuracy.
- Tier 2 (runnable only after the owner GT capture lands): the real face-angle/contact gates.
  This spec pins the Tier-2 protocol so the capture is collected to fit it.

## Frozen baseline + eval anchors

- Baseline artifact: `paddle.fused_estimator` (best_stack rev 12), render-only
  `estimated_preview`, `racket_pose_estimate.json`; measured rectangle-IoU anchors:
  wolverine 0.235558 / burlington 0.342387 (aggregate 0.224909) at
  `runs/lanes/racket_6dof_20260705/i1_fused_estimator/acceptance_record_v2.json`, per-clip
  scoring at `.../final_v2/<clip>/fused_vs_cvat_paddle_boxes_score_review_only_final_v2.json`.
- Pipeline seams a candidate must feed (do not bypass): `threed/racketsport/racket6dof.py`
  IPPE primitives (BOTH poses retained w/ ambiguity flags since evidence17, commit 8a282d4db),
  `racket_pose_preview.py` (`explicit_four_corner_candidates_pnp_ippe_preview`),
  `racket_stage_runner.py` fail-closed runner.
- Standing rules: rectangle IoU is NEVER 6DoF evidence; one-solution reprojection discard is
  banned; repaired confidence must be marked; no BALL/BODY regression.
- Eval clips for Tier 1: same two labeled clips as TRK (wolverine_mixed_0200, burlington_gold_0300,
  videos at `eval_clips/ball/<clip>/source.mp4`) — they have CVAT rough paddle boxes; plus the
  worst-blur contact windows flagged by events artifacts (diagnostic strata: sharp vs blurred,
  paddle-px bins <40 / 40-80 / >80).

## Tier-1 protocol (candidate vs preview, review-only card)

1. Reproduce baseline IoU numbers from the acceptance record within 0.001 before any candidate
   runs (same scorer script, same clips) or STOP.
2. Candidate arm: RacketVision released 5-keypoint model zero-shot on the same frames
   (adapter to be built in the lane; NO adapter exists in-repo — confirmed 2026-07-16).
   Record per frame: 5 keypoints + confidences, derived 4-corner set, BOTH IPPE poses +
   ambiguity margin through the existing seam, minimal-rect IoU vs CVAT rough boxes
   (comparison-only), keypoint temporal jitter, coverage (frames with usable output),
   runtime ms/frame.
3. Proxy metrics (explicitly labeled NOT-accuracy): box-IoU delta vs baseline; pose temporal
   coherence (frame-to-frame face-normal delta distribution); IPPE ambiguity-margin
   distribution; % frames where the two IPPE poses disagree >30° (ambiguity load for the
   future resolver).
4. Verdict vocabulary: `integration-pass` / `integration-fail` / `no-attempt` + review-only
   observations. NO accuracy verdict is available at Tier 1 by construction.

## Tier-2 protocol (pinned now, runs after owner GT capture)

- GT requirements (to be finalized from the survey lanes' error-budget findings; placeholder
  bars): GT face-angle uncertainty ≤~1.5° p90 and contact-point ≤~1cm so a 5°/3cm gate is
  measurable with margin; markers/multi-view allowed for GT only; sync ≤0.5 frame
  (NS-02.1 protocol); every label carries source/PTS/reviewer/uncertainty (NS-02.2).
- Scoring: face-angle error p50/p90, contact-point error p50/p90 on marker-derived GT frames,
  stratified by blur bin + paddle-px bin + edge-on angle; candidate must also report coverage
  and abstention honesty (abstain ≠ wrong-and-confident).
- Gates: interim milestone face-angle p90 ≤30° (candidate milestone only); promotion gates
  face-angle p90 ≤5° + contact p90 ≤3cm + no BALL/BODY regression (per North Star; not
  reachable this lane).

## Candidate arms (FINAL — per RKT_CROSSCHECK_RULING.md; livechecked 2026-07-16 x2 lanes +
## refutation)

Tier 1 (now):
1. **RacketVision 5-keypoint zero-shot** (ruled first challenger). Weights live on HF
   (`linfeng302/RacketVision-Models`: epoch_300.pth 411,293,859 B; best_PCK_epoch_90.pth
   106,524,759 B — HEAD-confirmed; no model card; pickle-import "unsafe" flags are a
   serialization warning, JFrog-safe/VT 0/74, but load in an isolated venv and pin hashes).
   Adapter must be built in-lane (none exists in-repo). License posture: code+annotations MIT;
   weights/dataset R&D-only pending YouTube-provenance review — fine for this review-only card,
   blocked from the selected product stack until NS-07.3 review.
2. **Both-IPPE diagnostics through the existing seam** (racket6dof.py primitives; evidence17
   both-pose retention): ambiguity-margin distribution, mode-disagreement rate >30°, temporal
   face-normal coherence. These are the design inputs for the temporal-graph build (Gap B).
3. Explicit `no-attempt` arms, recorded with eliminating evidence (per ruling): render-and-
   compare on product frames (seconds/frame + no small/blur evidence; GigaPose small-segment
   failure documented), FoundationPose (RGB-D + non-commercial), category-level (no paddle
   category exists), TT4D-class inverse control as precision source (26.4±4.4° mean).

Tier 2 (after owner GT passes its metrology gate):
4. **RacketVision pickleball fine-tune** vs zero-shot, same frozen split, real gates + strata.
5. **4-way synthetic ablation** (real-small / synth-only / synth→real / mixed) using the
   BlenderProc owner-CAD set (GPL tool, outputs unencumbered; Kubric = actively-maintained
   Apache fallback; both refutation-corrected). Kill scale-up if synth-only misses the 30°
   interim on held-out real.
6. **Offline oracle probes on sharp large crops** (MegaPose Apache code / GigaPose MIT code,
   oracle boxes+masks): information-existence test, never product path.
7. **ShapeFromBlur (MIT) evaluation on blurred paddle crops** — refutation-found prior art for
   the exposure-integrated refinement gap (Gap C); score before building Gap C from scratch.
8. **Temporal hypothesis graph A/B** (single-best IPPE vs two-mode persistence vs graph w/
   wrist/ball/physics soft factors), lag-at-contact measured.
9. **Contact oracle ladder** (GT time/ball/face → swap one input at a time) — isolates the
   limiting sensor; kill geometry-only route if GT-pose+GT-time+predicted-BALL misses 3cm.

## Outputs required from the lane

- `runs/lanes/rkt_kpbench_<date>/report.json`: baseline reproduction proof, per-arm per-clip
  Tier-1 table, runtime, checkpoint sha256s + licenses, integration verdicts, and the exact
  Tier-2-readiness statement (what GT is still missing).
- No best_stack change; trust band stays `estimated_preview` regardless of Tier-1 outcome.

## Provision

Tier 1 is likely CPU/MPS-local or single small GPU; if GPU needed, provision per
`.claude/skills/gpu-fleet-provision/` with ledger + teardown. Estimated wall: 1-3h.
