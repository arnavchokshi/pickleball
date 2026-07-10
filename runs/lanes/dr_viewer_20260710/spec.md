# LANE dr_viewer_20260710 — INDEPENDENT Codex audit: replay/viewer rendering vs owner symptoms

You are GPT-5.6 (Codex) giving an INDEPENDENT perspective. A parallel Claude multi-agent audit runs
concurrently; do NOT read runs/lanes/dr_viewer_20260710 siblings (dr_pipeline/dr_sota) or defer to
anyone — form your own conclusions from code + artifacts. You are one of a 3-lane Codex fan-out:
this lane owns the VIEWER/RENDER layer.

## HARD RULES
- READ-ONLY AUDIT: modify ZERO repo files. Write ONLY under runs/lanes/dr_viewer_20260710/.
- No branches, no commits, no `git add`.
- Protected eval clips (Outdoor/Indoor) labels: do not read or use. Burlington/Wolverine artifacts OK.
- Honest reporting; unverified hypotheses must be labeled as hypotheses with what would confirm them.
- Read NORTH_STAR_ROADMAP.md §1.3/§2.2 (REPLAY/STATS row) + RUNBOOK.md viewer sections first.

## FILE OWNERSHIP
Owns runs/lanes/dr_viewer_20260710/** only. Everything else read-only.

## OWNER SYMPTOMS (verbatim intent, from watching dinkvision_demo_20260710.mp4 built from our stack)
- S1: "frame rate still seems low"
- S2: "some people aren't detected or picked up"
- S3: "skeletons aren't showing a lot of the time — there should ALWAYS be either a skeleton or a mesh"
- S4: "the ball is hidden SO much — we need to show it and predict its position better (pb.vision does; we can do better)"
- S5: "the paddle looks really bad in the 3D page and isn't rendered properly positioning-wise"
- S6: "there's more issues out there" — find viewer defects the owner did NOT name.

## MISSION
1. Build a PER-ENTITY RENDER DECISION TABLE for the web replay viewer (web/replay/ source): for each
   of {player mesh, player skeleton, ball trail/marker, paddle, contact markers, trust badges/bands}:
   exactly when is it drawn vs hidden, per frame per player — cite file:line for every branch.
   Specifically: when a frame/player has NO mesh (byte budget, stride, missing BODY), does ANY
   skeleton fallback render? If not, that is S3's proximate cause — prove it.
2. Frame-rate chain (S1): what drives playback cadence (rAF, fixed tick, frame lookup, seek-snap)?
   Where does mesh fps get bound? Quantify with the real worlds in runs/lanes/w7_critique_20260709/
   (wolv_world, world) and web/replay/public/ (demo assets, untracked). Known prior evidence:
   mesh byte-budget lane took 5.21->21.32 mesh fps (runs/lanes/w6_meshcap_20260708/), playback
   decision table at runs/lanes/w6_playbackdiag_20260708/. Distinguish: renderer-bound vs data-bound
   vs demo-video-assembly-bound (the demo mp4 was assembled from headless captures —
   runs/lanes/demo_beststack_render_20260710/).
3. Ball rendering (S4): with fail-closed emission (best_stack rev 11) hiding untrusted segments
   (wolverine: 75/300 frames emitted), what does the viewer show during hidden spans? Nothing? A gap?
   Is there ANY predicted-position rendering with a `physics_predicted` trust band (North Star §1.4
   allows visibly-distinguished predicted geometry)? Cite exactly what a viewer user sees per band.
4. Paddle rendering (S5): what geometry is drawn (flat rect? 3D model?), from which artifact fields,
   through which transforms? Identify why it looks bad + mispositioned: geometry, pose source
   (wrist/palm/grip estimated_preview), coordinate transform, or all three. Cite file:line.
5. S6 sweep: seek-snap sync defect, absolute manifest paths, tap-tier trust dead-end (all booked in
   NORTH_STAR dated notes), plus anything new you find in the viewer code.
6. For every defect: root-cause hypothesis, severity, concrete fix sketch, effort estimate
   (hours/day/multi-day), and whether the fix is viewer-side, data-side, or both.

## EVIDENCE TO READ FIRST
- web/replay/ (viewer source), web/replay/public/ (demo worlds, untracked)
- runs/lanes/w7_critique_20260709/{wolv_world,world,viewer_verify,wolv_viewer_verify}
- runs/lanes/demo_beststack_20260710/REPORT.md + runs/lanes/demo_beststack_render_20260710/
- runs/lanes/w6_playbackdiag_20260708/playback_decision_table.md
- runs/lanes/w6_meshcap_20260708/report.json, runs/lanes/w7_ghostviewer_20260709/report.json
- threed/racketsport/ball_arc_chain.py (ball_arc_render.json producer), virtual_world.py
- scripts/racketsport/verify_process_video_viewer* (headless verify; takes replay_viewer_manifest.json)

## DELIVERABLES (all under runs/lanes/dr_viewer_20260710/)
- FINDINGS.md: render decision table + defects ranked by owner-symptom impact, each with evidence.
- report.json via the output schema: objective_result PASS if the decision table is complete with
  file:line citations AND every S1-S5 symptom has at least one evidenced viewer-layer explanation or
  an explicit "not viewer-layer, data-side" attribution; else FAIL with reasons. List HONEST ISSUES.

## BEST-STACK DELTA
(c) none — read-only audit; no stack change.
