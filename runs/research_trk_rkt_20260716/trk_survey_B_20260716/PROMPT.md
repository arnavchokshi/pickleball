# TRK research survey lane B (independent) — finding + tracking people better for DinkVision

You are one of two INDEPENDENT survey lanes on the same topic. Independence is load-bearing:
do NOT read any sibling directory under `runs/research_trk_rkt_20260716/` other than your own
(`trk_survey_B_20260716/`). Your findings will be cross-checked against the other lane; convergence
is treated as corroboration, so copying would destroy the method.

## Our context (read-only facts; do not re-derive)

- Task: track exactly 4 pickleball players from one fixed-ish elevated phone (1080p60-class
  consumer video) while REJECTING spectators, passers-by, and far-off-court people. Hard cases:
  net-line occlusion, player crossings, similar clothing, edge truncation.
- Current baseline (frozen identity): YOLO26m person detector + BoT-SORT + OSNet x1.0 market1501
  ReID + raw-pool global association, court-margin 1.0m (rev 12, preview). Mean IDF1 ~0.852;
  worst-clip after the margin+OSNet flip: IDF1 0.8516, cov4 0.7117, 0 switches.
- Frozen full gate per fresh clip: IDF1 ≥0.85, 0 switches, 0 spectator FP, 0 far-off-court FP,
  coverage ≥0.95. The dominant gap is COVERAGE (0.71-0.76 vs 0.95): players are missed
  (detection/occlusion), not mis-associated. Association-only sweeps are BANNED by standing ruling;
  the ruled next steps are detector/domain leverage (RF-DETR det/seg benchmark), then ReID, then
  McByte mask cue on worst clips.
- Already settled on 2026-07-09 (do NOT re-survey the basics): RF-DETR = top detection direction
  (Apache-2.0); McByte = MIT training-free mask cue, quick test; CAMELTrack = later challenger;
  KPR = license-blocked diagnostic. Focus on what is NEW through July 2026, what that register
  left open, and concrete adoption details (exact checkpoints, recipes, license texts).

## Research questions (in priority order for THIS lane — deliberately different from the sibling)

1. **Tracking systems view: what actually wins on sports MOT in 2025-2026.** SportsMOT /
   DanceTrack / SoccerNet-tracking leaderboards and recent papers: which end-to-end or modular
   trackers currently lead, and — decompose — how much of their edge is detector quality vs
   association vs ReID. Identify any tracker purpose-built for few-targets-many-distractors or
   court-sport settings. For each leader: code/weights live-check + license posture.
2. **Occlusion-robust person detection to fix COVERAGE.** What detects partially occluded/truncated
   players best: RF-DETR variants (exact sizes, det vs seg, released checkpoints, fine-tune
   maturity, Roboflow license terms on code AND weights), DETR-lineage 2025-2026 successors
   (D-FINE, DEIM/DEIMv2, RT-DETR successors), crowd-specific heads/NMS-free designs, and
   amodal/visibility-aware detection. Include what "YOLO26" is in the wild (Ultralytics lineage)
   so our baseline naming can be pinned against public reality. Rank by expected coverage gain
   under net/player occlusion, not headline COCO mAP.
3. **ReID under same-uniform/small-gallery regimes.** We need to re-lock 4 known identities after
   occlusion, against spectators. Best 2025-2026 person ReID with usable checkpoints; sports-domain
   ReID results; license triage code/weights/training-data (Market-1501-trained = research-only —
   flag our own OSNet default's posture too); part/visibility-aware ReID status (KPR license
   reconfirm); whether a small per-game enrollment gallery (few-shot identity locking) has
   published support.
4. **Mask cues in association + video-segmentation trackers.** McByte status check (repo, license,
   activity, any follow-up work). SAM2/2.1-based tracking-by-propagation for 4 targets: cost,
   drift, re-init strategies; DEVA/Cutie-class memory trackers; MASA; whether mask-IoU cues have
   published MOT gains in occlusion-heavy sports. Runtime per 60fps clip matters — record it.
5. **Domain adaptation & spectator rejection.** Published evidence for small-data fine-tunes on
   fixed-camera sports (what data volume moved coverage/IDF1); public sports-person datasets with
   licenses (SportsMOT, SoccerNet, DanceTrack, others); explicit spectator/negative supervision
   techniques; court-geometry-conditioned detection or gating (beyond our margin heuristic) in
   the literature.

## Method requirements (non-negotiable)

- PRIMARY SOURCES ONLY for load-bearing claims: fetch the paper (arXiv page ok), the repo
  README/LICENSE, the model card / release asset listing. Blogs and aggregator leaderboards are
  leads, not evidence.
- LIVE-CHECK every ranked checkpoint/code artifact: record URL + HTTP status/bytes + date in
  `livechecks.md` (one line each). HEAD/partial fetch preferred; do not download multi-GB weights.
- License triage per candidate: code / weights / training data, posture flag `commercial-clean`,
  `R&D-only`, or `unknown-needs-review` (NS-07.3). Quote exact license names.
- Published numbers are motivation to benchmark, never pickleball accuracy; no promotion language;
  VERIFIED=0 stands.
- Tag each claim [FETCHED-PRIMARY] / [SECONDARY] / [INFERENCE]. Record disagreements between
  sources explicitly.

## Deliverables (all inside /Users/arnavchokshi/Desktop/pickleball/runs/research_trk_rkt_20260716/trk_survey_B_20260716/)

1. `SURVEY.md`: ranked candidates per question (claimed numbers + source URL, license posture,
   adaptation cost, expected leverage on coverage/switches/spectator-FP, risks); "Changed since
   2026-07-09"; "Open items answered"; "Recommended benchmark order" vs our YOLO26m baseline on
   the worst-clip set (spec-only, no GPU dispatch).
2. `livechecks.md`: every live-checked URL with status + date.
3. Final message: ≤40-line summary of top findings + surprises.

## Fences and budget

- Write ONLY inside your lane dir. No pipeline code, configs, or other runs/ dirs. No GPU work.
- Budget ~2-4 hours; when up, ship what is verified.
