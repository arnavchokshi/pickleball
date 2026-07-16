# RKT research survey lane B (independent) — paddle pose: data-first view for DinkVision

You are one of two INDEPENDENT survey lanes on the same topic. Independence is load-bearing:
do NOT read any sibling directory under `runs/research_trk_rkt_20260716/` other than your own
(`rkt_survey_B_20260716/`). Findings are cross-checked against the other lane; convergence =
corroboration, so copying destroys the method.

## Our context (read-only facts; do not re-derive)

- Task: 6DoF pose (face angle + contact point) of a pickleball paddle from ONE fixed-ish elevated
  consumer phone at 1080p60. Paddle = near-planar composite object, often <40-80px in frame,
  motion-blurred mid-swing, hand/body-occluded, edge-on at contact. Non-owner games use unknown
  paddle models (category variation), but the owner's own paddle can be CAD-modelled exactly.
- Current state: render-only wrist/palm/grip fused preview at rectangle-IoU ~0.224-0.331; ZERO
  pose/contact GT. Both IPPE planar solutions now survive to fusion with ambiguity flags (fixed
  2026-07-16); repaired confidence is marked. Frozen gates: face-angle p90 ≤5°, contact-point p90
  ≤3cm, no BALL/BODY regression; interim milestone face-angle p90 ≤30°. Rectangle IoU never
  promotes as 6DoF.
- Ruled first challenger (settled 2026-07-09; don't re-derive the ruling): RacketVision 5-keypoint
  zero-shot → pickleball fine-tune. GT capture (markers/multi-view) is allowed for validation only;
  product stays single-camera.

## Research questions (in priority order for THIS lane — deliberately data-first, inverse of the sibling)

1. **Synthetic-data as the primary route.** If we CAD-model a paddle: end-to-end synthetic
   pipelines (BlenderProc2, Kubric, NVIDIA Replicator/Isaac, Unreal/Unity tooling, 2025-2026
   releases) — maturity, license, physically-based motion blur + rolling shutter simulation,
   hand-grasp synthesis around an object (grasp datasets/generators), compositing onto real court
   footage. PUBLISHED sim2real outcomes for thin/planar/textureless/small objects: exact numbers
   (what pose error transferred to real), the strongest positive and the strongest negative result
   you can find. Verdict: is synthetic-only or synthetic+small-real proven, plausible, or
   unsupported at our ≤5° face-angle bar?
2. **GT capture design (the validation bottleneck).** Published small-object/racket pose GT
   methodologies: marker boards/ArUco ON the object (mass/aero effect on swing?), IR/mocap markers,
   multi-camera high-FPS triangulation, robot-held references. Error-budget analysis: to certify a
   5° / 3cm p90 gate, what must GT accuracy be (rule-of-thumb ≥3-5x better → ~1-1.5° / ~1cm), and
   what capture rig (camera count, geometry, sync ≤0.5 frame, marker size at distance) delivers
   that with 2-3 consumer phones + ChArUco? Specify a minimal owner half-day capture: shot list,
   marker plan, sync method, expected label yield, what it can and cannot certify.
3. **Dataset landscape.** Any existing datasets with racket/paddle/bat/club POSE (not just boxes):
   RacketVision dataset contents recheck (what supervision it actually ships, its real license and
   video provenance), table-tennis/badminton/tennis analytics datasets, industrial small-object
   pose sets (BOP family) — which regimes they cover vs ours (object pixels, blur, planarity).
   License triage each: `commercial-clean` / `R&D-only` / `unknown-needs-review`.
4. **Method reality-check (secondary for this lane).** For the main method families —
   keypoint+PnP (both-IPPE), render-and-compare (MegaPose/FoundationPose/GigaPose class,
   2025-2026 successors), category-level pose, direct regression, temporal/trajectory-level pose —
   what INPUT quality does each demand (pixels on object, texture, sharpness), and which are
   eliminated OUTRIGHT by our regime? Cite the evidence for elimination, not intuition.
5. **Contact-point specifically.** The 3cm gate is about WHERE on the face the ball hits.
   Published ball-racket impact localization (table tennis/tennis/badminton research, robot
   returns); can contact point be derived from ball track + face plane at the ≤3cm level, and
   what does that require from BALL 3D accuracy? Where is the cheapest reliable signal (audio
   timing + geometry vs visual)?
6. **Gap map (explicit).** Where nothing off-the-shelf exists for our regime; for each gap a
   buildable spec sketch (inputs, supervision source, architecture class, GT needed, risks,
   kill criteria).

## Method requirements (non-negotiable)

- PRIMARY SOURCES ONLY for load-bearing claims: fetch papers, repo README/LICENSE, model cards.
  Blogs/aggregators are leads, not evidence.
- LIVE-CHECK every ranked artifact: URL + HTTP status/bytes + date into `livechecks.md`. No
  multi-GB downloads.
- License triage per candidate/dataset (code/weights/data), exact license names.
- Published numbers ≠ pickleball accuracy; no promotion language; VERIFIED=0 stands.
- Tag claims [FETCHED-PRIMARY] / [SECONDARY] / [INFERENCE]; record disagreements.

## Deliverables (all inside /Users/arnavchokshi/Desktop/pickleball/runs/research_trk_rkt_20260716/rkt_survey_B_20260716/)

1. `SURVEY.md`: findings per question with ranked options; "Synthetic-to-real verdict w/ evidence";
   "Minimal owner GT capture spec"; "Method elimination table"; "Gap map — build-our-own specs";
   "Recommended benchmark order" (spec-only).
2. `livechecks.md`: every live-checked URL with status + date.
3. Final message: ≤40-line summary — synthetic verdict, GT capture answer, biggest gap, surprises.

## Fences and budget

- Write ONLY inside your lane dir. No pipeline code, configs, or other runs/ dirs. No GPU work.
- Budget ~2-4 hours; when up, ship what is verified.
