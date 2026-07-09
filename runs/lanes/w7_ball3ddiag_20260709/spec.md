# LANE w7_ball3ddiag_20260709 — READ-ONLY diagnosis: why 2D-good ball becomes 3D-bad (owner critique, live session)

## HARD RULES
STRICTLY READ-ONLY on repo source and all run artifacts — you edit NOTHING outside runs/lanes/w7_ball3ddiag_20260709/. No commits. Diagnosis lane per house doctrine: measure, do not fix.

## OWNER OBSERVATION (verbatim intent, tonight, watching the wolverine world in the viewer)
The 2D ball overlay dot on the video is right ~80-90% of the time, but the 3D-placed ball in the world is "insanely bad" — lagging/jumping/absurd positions. Diagnose WHERE the 2D->3D lift fails.

## EVIDENCE (all local, from a fresh full-stack production run)
runs/lanes/w7_critique_20260709/wolv_world/wolverine_mixed_0200_mid_steep_corner/ : ball_track.json (2D), ball_track_arc_solved.json, ball_track_physics_filled.json, ball_arc_render.json, ball_inflections.json, ball_bounce_candidates.json, contact_windows.json, events_selected.json, court_calibration.json, net_plane.json, confidence_gated_world.json, virtual_world.json, PIPELINE_SUMMARY.json. Also the same set in runs/lanes/w7_critique_20260709/world/owner_critique_zwcth45s/ (a second clip, ball-chain-only) for cross-checking.

## QUESTIONS (each = an acceptance row with NUMBERS + artifact evidence)
1. REPROJECTION vs RAW 2D: reproject the solved/rendered 3D arc positions through the calibration back to image space and measure per-frame pixel divergence from the raw 2D track on frames where 2D exists. Distribution (median/p95/max) overall AND per arc segment. Identify the segments where reprojection is fine but 3D world position is extreme (the depth-ambiguity signature) vs segments where even reprojection diverges (solver ignoring 2D).
2. DEPTH EXCURSIONS: per-frame 3D ball position stats — height/depth ranges, velocity implied between consecutive rendered positions (flag physically absurd jumps >20 m/s), where in the world the ball spends time (behind baselines? underground? above 10m?). Quantify the "insanely bad" as a defect table with frame ranges.
3. SEGMENTATION: how many arc segments were fit; where are their endpoints vs ball_inflections/bounce_candidates/contact_windows; which segments have endpoint anchors that disagree with the 2D track's actual direction changes (mis-split arcs). Count solver kill/fallback statuses and what the renderer shows during them (raw? interpolated? hidden?).
4. SELECTION PATH: from the artifacts, determine which candidate-selection mechanism fed the solver (consensus vs single-detector; the resolved best_stack config is in PIPELINE_SUMMARY) and how many candidate points the solver rejected/accepted per segment. Does the evidence support the standing FUSION-CHANGE theory (2D-consensus artifacts feeding the solver)?
5. RANKED ROOT-CAUSE TABLE: for the top-5 worst visual stretches (from #2), attribute each to: bad segment endpoint / depth ambiguity with weak anchors / candidate junk / render-during-kill / other-with-evidence. State which SINGLE fix would most improve the owner-visible 3D ball on this clip.

## REPORT
Self-write runs/lanes/w7_ball3ddiag_20260709/report.json (lane_report.schema.json structure) + a DIAGNOSIS.md with the defect tables. Numbers + paths only; the manager rules next steps.
