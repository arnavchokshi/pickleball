# LANE dr_sota_20260710 — INDEPENDENT Codex research: beat pb.vision on visible quality

You are GPT-5.6 (Codex) with web search enabled, giving an INDEPENDENT research perspective;
a parallel Claude audit runs concurrently. Do NOT read the sibling dr_* lanes. You are one of a
3-lane Codex fan-out: this lane owns EXTERNAL EVIDENCE (competitors + SOTA), grounded on our banked
research so you extend rather than repeat it.

## HARD RULES
- Modify ZERO repo files; write ONLY under runs/lanes/dr_sota_20260710/.
- No branches/commits. Web search allowed and expected; cite URLs + access dates for every claim.
- Honest reporting: mark rumor vs first-party evidence. Our banked research already corroborated
  pb.vision runs physics-over-CV like ours (job posting) — do not re-derive; go DEEPER/NEWER.

## FILE OWNERSHIP
runs/lanes/dr_sota_20260710/** only.

## CONTEXT (owner symptoms on our current demo)
S1 low frame rate; S2 people missing; S3 skeleton/mesh gaps; S4 ball hidden too much — owner wants
visible + predicted ball better than pb.vision; S5 paddle looks bad/mispositioned; S6 unknowns.
Our banked evidence: runs/research_ball3d_20260709/{SYNTHESIS.md,RULINGS.md,pbvision_cv_export/}
(their real "cv" JSON export: per-frame court_position + `interpolated` provenance flag + ball_radius
+ confidences), runs/lanes/w7_pbv_compare_20260709/COMPARISON.md (head-to-head: our 2D coverage 80.6%
vs their 58.7%; they omit 3D on 31% of frames; our fail-closed now hides ~75%; landing pseudo-GT
comparable). Read these FIRST.

## MISSION (per symptom, what do the best products/papers do, and what should WE adopt)
1. pb.vision PRODUCT behavior deep-dive (web + their export schema): how do they RENDER the ball
   during no-3D spans (gap? 2D-only? interpolated marker with distinct styling?), what fps do their
   replays play at, how do they present players (meshes? skeletons? avatars? dots), do they show
   paddles at all? Their blog/changelog/app store notes/videos are fair game. Also: SwingVision
   (tennis, closest analog), and any other pickleball/racket CV products (e.g. PlaySight, Zenniz,
   CourtAI-class). First-party sources preferred.
2. Ball visibility/prediction SOTA (S4): occlusion-robust ball trajectory completion (UKF/smoother
   vs learned lift TT4D/Kienzle-WACV26/Where-Is-The-Ball-class), and specifically PRODUCT-GRADE
   patterns for showing predicted-vs-measured trajectory honestly. What's the strongest 2026 recipe
   compatible with our scipy TRF+huber arc fit + fail-closed policy?
3. Skeleton/mesh fallback UX (S3) + replay fps (S1): how do sports replay products guarantee an
   always-present player representation (skeleton fallback, interpolated mesh, LOD swap)? Web 3D
   perf patterns for mesh-sequence playback (draco/meshopt compression, GPU instancing, keyframe
   interpolation between mesh frames) applicable to our three.js-class viewer.
4. Paddle/racket rendering (S5): practical racket pose presentation in shipped products + SOTA
   racket 6DoF (RacketVision-class 5-keypoint, IPPE two-pose handling); what visual treatment makes
   a low-confidence paddle look GOOD honestly (ghosting, simplified proxy, snap-to-hand)?
5. Rank: top 10 adoptions for US, each with {what, evidence URL, expected symptom impact,
   effort, risk, kill criterion}, explicitly reconciled against our banked adopt-sequence
   (fail-closed [SHIPPED] -> UKF fallback -> TT3D anchor search -> BlurBall/audio -> pinning ->
   DP segmentation) — state where your ranking AGREES or DISAGREES with it and why.

## DELIVERABLES (runs/lanes/dr_sota_20260710/)
- RESEARCH.md: cited findings per mission item; every load-bearing claim marked first-party vs inferred.
- report.json via schema: PASS if missions 1-5 each have >=3 cited sources and the ranked adoption
  list is complete; HONEST ISSUES for anything you could not confirm.

## BEST-STACK DELTA
(c) none — research only.
