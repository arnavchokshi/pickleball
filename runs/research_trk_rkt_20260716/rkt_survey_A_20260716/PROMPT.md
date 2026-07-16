# RKT research survey lane A (independent) — paddle detection + 6DoF orientation for DinkVision

You are one of two INDEPENDENT survey lanes on the same topic. Independence is load-bearing:
do NOT read any sibling directory under `runs/research_trk_rkt_20260716/` other than your own
(`rkt_survey_A_20260716/`). Findings are cross-checked against the other lane; convergence =
corroboration, so copying destroys the method.

## Our context (read-only facts; do not re-derive)

- Task: pickleball paddle pose from ONE fixed-ish elevated consumer phone (1080p60-class). The
  paddle is a small (~40cm incl. handle), near-planar, often <40-80px object, frequently
  motion-blurred at swing speed, half-occluded by the hand/body, and edge-on to the camera at the
  worst moments (contact).
- Current state: a render-only wrist/palm/grip fused paddle preview, rectangle-IoU ~0.224-0.331 vs
  rough boxes; ZERO true pose/contact ground truth exists. The pipeline now PRESERVES BOTH IPPE
  planar-pose solutions with ambiguity flags (the collapse was fixed 2026-07-16) and marks repaired
  confidence — so any keypoint+PnP candidate can keep both hypotheses through to fusion.
- Frozen promotion gates: face-angle p90 ≤5° and contact-point p90 ≤3cm, with no BALL/BODY
  regression. Interim candidate milestone only: face-angle p90 ≤30°. Rectangle IoU can NEVER be
  promoted as 6DoF (standing rule).
- Ruled first challenger (2026-07-09 register, do not re-derive): RacketVision — released code +
  weights + MIT-labelled dataset (April 2026), 5-keypoint racket model, zero-shot then pickleball
  fine-tune. Your job: what exists BEYOND/AROUND that ruling, what changed through July 2026, and
  where NO off-the-shelf solution exists at all.
- We CAN CAD-model the owner's paddle exactly (and common commercial paddles approximately), and
  we can print markers/use ChArUco for GT capture sessions. Product inference stays single-camera;
  extra cameras/markers are allowed for GT only.

## Research questions (in priority order for THIS lane)

1. **Small/fast-object 6DoF SOTA.** Instance-level pose WITH a CAD model: render-and-compare and
   template lines (MegaPose, FoundationPose, FoundPose, GigaPose, SAM-6D, and 2025-2026 successors)
   — for each: does it plausibly work at <80px object size with motion blur? published evidence at
   small pixel scale? runtime per frame? code/weights live-check + license triage. Category-level
   pose (paddles vary across players): current SOTA and whether any handles near-planar thin
   objects. Note: BOP benchmark objects are mostly texture-rich and large-in-frame — find any
   evidence about the small/blurred regime specifically.
2. **Keypoint + PnP routes for planar objects.** RacketVision status recheck (repo/weights/dataset
   live-check, license, any follow-ups or citations using it); other racket/bat/club/equipment
   keypoint works in 2025-2026 (tennis/badminton/table-tennis/baseball/golf); planar-pose specifics
   (IPPE ambiguity literature — resolution cues published: temporal, hand-context, physics);
   motion-blur-robust keypoint detection (deblur-first vs blur-aware training).
3. **Video/temporal pose priors.** Anything that solves small-object pose as a TRAJECTORY rather
   than per-frame (temporal render-and-compare, tracking-based pose e.g. BundleTrack/BundleSDF
   lineage, filtering with both-IPPE hypothesis graphs); swing-dynamics priors from sports science
   usable as soft constraints.
4. **Synthetic-data routes.** Pipelines (BlenderProc2, Kubric, Isaac Replicator, 2025-2026
   successors) for rendering a CAD paddle with domain randomization + motion blur + hand occlusion;
   PUBLISHED sim2real evidence for comparable objects (thin/planar/textureless, small in frame):
   what accuracy transferred, what failed; whether synthetic pretraining + small real fine-tune is
   proven for keypoint models of this class. Be precise about what accuracy level synthetic-to-real
   has actually demonstrated — is ~5° face-angle plausible or unsupported?
5. **Minimal owner GT capture spec.** What published racket/small-object pose GT captures contain
   (marker/ChArUco boards on the object, multi-view, robot arms); the ERROR BUDGET question: what
   GT accuracy (marker size, camera count/geometry, sync) is required to MEASURE a 5° face-angle /
   3cm contact-point gate credibly; minimal viable owner half-day capture contents.
6. **Gap map (explicit).** Where NO off-the-shelf solution exists for our regime (e.g. <80px
   blurred near-planar object pose from a single elevated camera at 60fps). For each gap: what we
   would have to build ourselves, sketched as a buildable spec (inputs, supervision, architecture
   class, GT needs, risks).

## Method requirements (non-negotiable)

- PRIMARY SOURCES ONLY for load-bearing claims: fetch the paper, the repo README/LICENSE, the
  model card / release assets. Blogs/aggregators are leads, not evidence.
- LIVE-CHECK every ranked code/checkpoint artifact: URL + HTTP status/bytes + date into
  `livechecks.md` (one line each). No multi-GB downloads.
- License triage per candidate: code / weights / training data; posture `commercial-clean`,
  `R&D-only`, or `unknown-needs-review` (NS-07.3). Exact license names.
- Published numbers ≠ pickleball accuracy; no promotion language; VERIFIED=0 stands.
- Tag claims [FETCHED-PRIMARY] / [SECONDARY] / [INFERENCE]; record source disagreements.

## Deliverables (all inside /Users/arnavchokshi/Desktop/pickleball/runs/research_trk_rkt_20260716/rkt_survey_A_20260716/)

1. `SURVEY.md`: ranked candidates per question (claimed numbers + source, license posture,
   adaptation cost, expected leverage on the ≤5°/≤3cm gates, risks); "Changed since 2026-07-09";
   "Gap map — build-our-own specs"; "Minimal owner GT capture recommendation"; "Recommended
   benchmark order" (spec-only, no GPU dispatch).
2. `livechecks.md`: every live-checked URL with status + date.
3. Final message: ≤40-line summary of top findings, the single biggest gap, and surprises.

## Fences and budget

- Write ONLY inside your lane dir. No pipeline code, configs, or other runs/ dirs. No GPU work.
- Budget ~2-4 hours; when up, ship what is verified.
