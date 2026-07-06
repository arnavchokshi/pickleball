# Lane P4 Ball Render Fix Report

## Result

PASS. This lane fixed the three final-verify ball-render blockers and reran CPU evidence for the three final-verify clips under:

`runs/lanes/ball_p4_render_fix_20260706/`

## Fixes

1. Schema v2 replay load
   - Before: `web/replay/src/shotTrails.ts` rejected every `ball_track_arc_solved` artifact with `schema_version: 2` and threw `ball_track_arc_solved.schema_version must be 1`.
   - After: the arc-solved consumer accepts schema versions `1 | 2`; the unrelated `shots.schema_version` check is unchanged.
   - Evidence: `web/replay/src/shotTrails.test.ts` loads the final-verify Burlington `schema_version: 2` `ball_track_arc_solved.json` without throwing.
   - Viewer data flow: `App.tsx` loads both `ball_arc_solved_url` through `parseBallArcSolved` and `ball_arc_render_url` through `parseBallArcRender`; both fixture paths are covered by Vitest.

2. Wolverine artifact self-kill
   - Before: final-verify Wolverine was `status: experimental_off`, `kill_reasons: ["physical_sanity_violation_fraction"]`, and rendered 0 ball samples.
   - After: any physical sanity violation on a non-fallback `fit` segment demotes that segment to `fit_bvp_fallback`, preserving original fit details in diagnostics. `_physical_summary` now divides violation fraction by violation-eligible segments rather than all segments, and fallback-contained violations do not poison the whole artifact.
   - Render containment: fallback render samples are filtered to strict court XY bounds, not just the wider solver court volume.
   - Current Wolverine evidence:
     - solver status: `ran`
     - kill reasons: `[]`
     - segment statuses: `{"fit_bvp_fallback": 7}`
     - render samples: `1238`
     - render bands: `{"arc_weak": 1238}`
     - confidence-gate ball bands: `{"hidden_no_prediction": 10, "measured": 207, "physics_predicted": 13, "physics_predicted_low": 70}`
     - strict out-of-court render samples: `0`

3. Court-map empty ball view
   - Before: the court outline could render while ball paths/current ball projected outside the SVG viewbox for centered court coordinates.
   - After: court-map projection uses actual court line-segment bounds when present, clamps SVG coordinates to the visible court viewport, and renders a current-ball marker from `ball_arc_render.samples`.
   - Evidence: `web/replay/src/ballArcRender.test.ts` projects real final-verify Burlington shot paths and current ball samples into the SVG viewport; `CourtMapPanel.test.tsx` asserts shot line, bounce dot, current ball, and player marker markup.

## CPU Rerun Artifact Summary

The full pipeline was run on the three final-verify input sets first. After the fallback strict-bound render change, the ball arc chain artifacts were refreshed with the same CPU inputs so `ball_track_arc_solved.json` and `ball_arc_render.json` below reflect current code.

| Clip | Solver status | Render samples | Render bands | Confidence-gate ball bands | Strict out-of-court render samples |
|---|---:|---:|---|---|---:|
| wolverine | ran | 1238 | `{"arc_weak": 1238}` | `{"hidden_no_prediction": 10, "measured": 207, "physics_predicted": 13, "physics_predicted_low": 70}` | 0 |
| burlington | ran | 2239 | `{"arc_weak": 2135, "arc_interpolated": 104}` | `{"hidden_no_prediction": 12, "measured": 466, "physics_predicted": 13, "physics_predicted_low": 109}` | 278 |
| outdoor | ran | 2156 | `{"arc_weak": 1630, "arc_interpolated": 526}` | `{"hidden_no_anchor": 1, "hidden_no_prediction": 690, "measured": 346, "physics_predicted": 6, "physics_predicted_low": 108}` | 357 |

The explicit zero-out-of-court acceptance target was for Wolverine. Burlington and Outdoor still include strict out-of-court render samples on non-fallback fit segments, but both load and render ball samples with `status: ran`.

Outdoor/Indoor label policy remained untouched for all three reruns:

`{"outdoor_indoor_labels_read": false, "protected_eval_labels_used": false}`

## Verification

Commands run:

- `npm test` in `web/replay`: 14 files passed, 184 tests passed.
- `npm run typecheck` in `web/replay`: passed.
- `npm run build` in `web/replay`: passed with the existing large-chunk warning.
- `.venv/bin/python -m pytest tests/racketsport/test_*ball*.py tests/racketsport/test_run_ball_chain.py tests/racketsport/test_verify_process_video_viewer.py tests/racketsport/test_rebuild_ball_trail_from_arc_solved.py tests/racketsport/test_measure_ball_trail_kinks.py -q`: 380 passed.
- `.venv/bin/python -m pytest tests/racketsport/test_truthful_capabilities.py tests/racketsport/test_scaffold_tool_index.py -q`: 18 passed.
- `git diff --check`: passed.

## Honest limits

- Browser/localhost verification was not run because this lane cannot bind localhost in the Codex sandbox; manager browser verification is still needed.
- Repository capability truth remains `VERIFIED=0`; these are scoped ball/render fixes, not a global promotion.
- Top-level `PIPELINE_SUMMARY.json` files were created by the earlier full pipeline runs; use the refreshed `ball_track_arc_solved.json` and `ball_arc_render.json` artifacts for current-code ball arc/render counts.
