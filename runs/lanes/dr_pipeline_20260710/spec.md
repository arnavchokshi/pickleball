# LANE dr_pipeline_20260710 — INDEPENDENT Codex audit: pipeline data production vs owner symptoms

You are GPT-5.6 (Codex) giving an INDEPENDENT perspective; a parallel Claude audit runs concurrently.
Do NOT read runs/lanes/dr_viewer_20260710 or dr_sota_20260710. You are one of a 3-lane Codex
fan-out: this lane owns the PIPELINE/DATA layer (what the artifacts contain, not how they render).

## HARD RULES
- READ-ONLY AUDIT: modify ZERO repo files. Write ONLY under runs/lanes/dr_pipeline_20260710/.
- No branches, no commits. You may RUN read-only python analysis of artifacts (write outputs to your
  lane dir only). Targeted pytest runs allowed read-only if they inform a finding.
- Protected clips (Outdoor/Indoor) labels: never read. Burlington/Wolverine artifacts OK.
- Honest reporting; hypotheses labeled with confirmation paths.
- Read NORTH_STAR_ROADMAP.md §2 + §3 (reuse contract) + RUNBOOK.md stage sections first.

## FILE OWNERSHIP
Owns runs/lanes/dr_pipeline_20260710/** only.

## OWNER SYMPTOMS (from dinkvision_demo_20260710.mp4 of our current stack)
S1 low frame rate; S2 some people not detected/picked up; S3 skeletons missing when meshes absent
(product rule: ALWAYS skeleton or mesh); S4 ball hidden too much — wants shown + predicted better
(pb.vision reference); S5 paddle bad + mispositioned in 3D; S6 unknown additional issues.

## MISSION
1. COVERAGE QUANTIFICATION (S1/S2/S3/S4 data-side): using the real pulled worlds
   (runs/lanes/w7_critique_20260709/wolv_world + world; runs/lanes/demo_beststack_gpu_20260710/
   vm_pull/ zwcth45s R1+R2), compute per-frame coverage tables: tracked players/frame vs expected 4;
   BODY mesh frames per player (vs total frames — stride/cap/byte-budget losses at each step);
   skeleton keypoint frames per player; ball 2D detection coverage vs 3D emitted vs hidden;
   paddle pose frames. Output CSV/JSON + a summary table. Name WHERE each loss happens
   (frames stage cap b437b4118 / stride schedule / byte budget / fail-closed / detector miss).
2. S2 root cause: are "missing people" detector misses (YOLO26m conf), association drops,
   court-membership filtering (spectator exclusion eating real players), or BODY-only losses
   (person tracked but no mesh)? Evidence from tracking artifacts + membership tools.
3. S3 data-side: does the pipeline emit skeleton keypoints for frames where mesh is absent
   (cheap full-rate joints per North Star §3 DAG)? If skeletons exist in artifacts but not in the
   viewer manifest, name the drop point (mesh index? manifest builder? byte budget policy).
4. S4 data-side: per-segment ball fate on wolverine (11 segments: which fit/fallback/suppressed) +
   zwcth45s; what would UKF-seeded fallback (research adopt item #5, runs/research_ball3d_20260709/
   RULINGS.md) + TT3D joint-anchor search (#1) buy in emitted-frame coverage? Check whether ANY
   physics_predicted interpolation is currently emitted between trusted segments (reuse contract
   allows it with provenance). Quantify our 2D coverage that 3D lift wastes (80.6% 2D vs 25% 3D
   emitted on wolverine per w7_pbv_compare).
5. S5 data-side: paddle pose artifact quality — which fields, what confidence, which transform chain
   places it in world space (P0-D raw/undistorted/reference/world inconsistency is a booked blocker;
   threed/racketsport/coordinates.py just landed). Does paddle placement consume the typed API or
   ad-hoc transforms? Cite file:line.
6. S6: regression sweep of the rev-9 -> HEAD window (the demo REPORT names one: cold-clip BODY
   frame-materialization, FIXED this morning by ns016 — verify the fix landed and covers zwcth45s
   signatures; runs/lanes/ns016_bodyframes_20260710/). Also sweep: provenance blocks dropped by
   strict schemas (arc_solved_overlay -> virtual_world.json booked residual), events-before-BODY
   staleness (P0-G), audio ignored on normal path, and any silent-truncation you find in stage code.
7. Every defect: root-cause hypothesis w/ file:line or artifact evidence, severity, fix sketch,
   effort, and which NS task it belongs to (NS-01.x/NS-03.x/NS-04.x per NORTH_STAR).

## EVIDENCE TO READ FIRST
- runs/lanes/demo_beststack_20260710/REPORT.md (3 zwcth45s signatures + booked residuals)
- runs/lanes/ns016_bodyframes_20260710/ (the fix + its report)
- runs/lanes/w7_critique_20260709/ pulled worlds; runs/lanes/demo_beststack_gpu_20260710/report.json
- runs/research_ball3d_20260709/{RULINGS.md,SYNTHESIS.md}; runs/lanes/w7_pbv_compare_20260709/COMPARISON.md
- scripts/racketsport/process_video.py; threed/racketsport/{virtual_world,ball_arc_chain,mhr_decode,coordinates}.py
- configs/racketsport/best_stack.json (rev 11)

## DELIVERABLES (runs/lanes/dr_pipeline_20260710/)
- FINDINGS.md + coverage tables (csv/json) + defect list ranked.
- report.json via schema: PASS if coverage tables computed from real artifacts for >=2 clips AND
  each S1-S5 has an evidenced data-side attribution or explicit "viewer-side, not data" ruling.
  HONEST ISSUES section mandatory.

## BEST-STACK DELTA
(c) none — read-only audit.
