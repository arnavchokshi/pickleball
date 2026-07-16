# TRK research survey lane A (independent) — person detection/segmentation/ReID/mask-cue for DinkVision

You are one of two INDEPENDENT survey lanes on the same topic. Independence is load-bearing:
do NOT read any sibling directory under `runs/research_trk_rkt_20260716/` other than your own
(`trk_survey_A_20260716/`). Your findings will be cross-checked against the other lane; convergence
is treated as corroboration, so copying would destroy the method.

## Our context (read-only facts; do not re-derive)

- Product: single-camera pickleball reconstruction from a fixed-ish, elevated consumer phone
  (1080p60-class), 4 players on court plus spectators/passers-by near and behind the court.
  Typical failure modes: net-crossing occlusions, players crossing/switching sides, spectators
  picked up as players, far-off-court false positives, partial truncation at frame edges.
- Current TRK baseline (frozen): YOLO26m person detector + BoT-SORT + OSNet x1.0 (market1501)
  ReID + raw-pool global association with court-margin 1.0m (best_stack rev 12, preview band).
  Mean IDF1 ~0.852; worst-clip flip 0.6425→0.8516 IDF1, cov4 0.0433→0.7117, 0 switches.
- Frozen promotion gate (every fresh clip): IDF1 ≥0.85, 0 identity switches, 0 spectator FP,
  0 far-off-court FP, player coverage ≥0.95. Biggest remaining gap: coverage (cov4 0.71-0.76 vs 0.95).
- Ruled sequence (binding): detector/domain leverage FIRST — benchmark RF-DETR det/seg, then ReID,
  then McByte mask cue on worst clips. Association-only threshold sweeps are BANNED without new
  detector/domain/ReID leverage.
- Prior SOTA register (2026-07-09, already settled — do NOT re-survey these basics): RF-DETR M/L
  det/seg = top TRK direction (Apache-2.0); McByte = quick MIT training-free mask-cue test;
  CAMELTrack = later learned-association challenger only; KPR ReID = diagnostic-only (Hippocratic
  license); SAM-Body4D masklets = bounded BODY-input A/B. Your job is what CHANGED since 2026-07-09,
  what that register left OPEN, and the concrete adoption details it never pinned (exact variants,
  checkpoints, fine-tune recipes, license texts).

## Research questions (in priority order for THIS lane)

1. **Person detection/segmentation SOTA for our setting.** RF-DETR today: exact released variants
   (sizes; detection vs segmentation heads), exact checkpoint artifacts, fine-tune recipe maturity,
   COCO/person and crowded-scene (CrowdHuman, MOT20-det class) numbers, license of code AND weights
   (including any Roboflow platform strings attached). Then: anything released or updated through
   July 2026 that beats or plausibly beats RF-DETR for PERSON detection in crowded/occluded sports
   scenes — e.g. D-FINE, DEIM/DEIMv2, RT-DETR successors, LW-DETR lineage, YOLO release lineage
   (incl. what "YOLO26" actually is per Ultralytics vs our internal naming), open-vocabulary
   detectors as spectator filters. Rank by expected gain on OUR failure modes (coverage under
   occlusion, spectator FP), not headline mAP.
2. **Domain adaptation for our setting.** Evidence that fine-tuning a detector on a small
   sports-specific set (hundreds-to-thousands of boxes, elevated fixed camera, 4 players +
   spectators) beats zero-shot; public sports person datasets usable for pretraining/adaptation
   (SportsMOT, SoccerNet-Tracking, DanceTrack, basketball/volleyball sets, any racket-sport MOT
   sets) with license posture each; practical recipes (freeze/LoRA/full fine-tune, image size,
   negative/spectator supervision, court-region conditioning).
3. **Segmentation for mask cues.** Practical per-track mask sources at video scale: RF-DETR seg
   variants, SAM2/2.1 video propagation cost and drift behavior, YOLO-seg class, anything newer
   (2026) that is cheap enough per frame. What resolution/runtime cost per 60fps clip.
4. **ReID.** Current best commercially-clean person ReID vs our OSNet default: SOLIDER,
   TransReID lineage, CLIP-ReID, KPR (license reconfirm), 2025-2026 releases, sports-specific ReID
   (same-jersey/similar-clothing regimes). For each: checkpoint availability, license of code AND
   weights AND training data (Market-1501 & co. are research-only — flag it), expected value in a
   4-known-players + spectators regime.
5. **Mask-cue and multi-cue association status check** (lower priority; detection first per ruling):
   McByte repo status (activity, issues, exact license, reproducibility), alternatives that consume
   masks (MASA, SAM2-based MOT, DEVA/Cutie-style memory), CAMELTrack status. Note only what changed
   since 2026-07-09.

## Method requirements (non-negotiable)

- PRIMARY SOURCES ONLY for load-bearing claims: fetch the actual paper (arXiv abstract page ok),
  the actual repo README/LICENSE file, the actual model card / release asset listing. Blog posts
  and leaderboard aggregators are leads, not evidence.
- LIVE-CHECK every checkpoint/code artifact you rank: HTTP status of the concrete artifact
  (GitHub release asset, HF repo file listing, weights URL). Record: URL, HTTP status or
  fetched-bytes evidence, and fetch date, in `livechecks.md` (one line each).
- License triage per candidate: code license / weights license / training-data terms, and a final
  posture flag: `commercial-clean`, `R&D-only`, or `unknown-needs-review` (NS-07.3 discipline).
  Quote the exact license name, not a guess.
- Published numbers establish that an experiment is worth running; they are NOT pickleball
  accuracy. No promotion language. VERIFIED=0 stands regardless of what you find.
- Distinguish clearly: [FETCHED-PRIMARY] vs [SECONDARY] vs [INFERENCE] on each claim.
- If two sources disagree, record both and say which you trust and why.

## Deliverables (all inside /Users/arnavchokshi/Desktop/pickleball/runs/research_trk_rkt_20260716/trk_survey_A_20260716/)

1. `SURVEY.md` — the report:
   - Ranked candidate table for each research question, per candidate: what it is, key claimed
     numbers (benchmark, value, source URL), code/weights/license posture, adaptation cost
     (data + GPU-hours order of magnitude), expected leverage on OUR gate (coverage/switches/
     spectator-FP), risks.
   - Section "Changed since 2026-07-09 register" and "Open items now answered".
   - Section "Recommended benchmark order" — what to run first against our YOLO26m baseline on
     the worst-clip set and why (spec-only; no GPU dispatch, that is the manager's job).
2. `livechecks.md` — every live-checked URL: status, date.
3. Final message: ≤40-line summary of top findings + any surprises.

## Fences and budget

- Write ONLY inside your lane dir. No pipeline code, no configs, no other runs/ dirs.
- No GPU dispatch, no model downloads beyond what a live-check needs (HEAD/partial fetch preferred;
  do not download multi-GB weights).
- Budget: aim ~2-4 hours. Depth on the top candidates beats breadth of enumeration. When time is
  up, ship SURVEY.md with what is verified rather than continuing to search.
