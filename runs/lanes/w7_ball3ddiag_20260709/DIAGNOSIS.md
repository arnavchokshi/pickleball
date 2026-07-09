# W7 BALL 2D-to-3D diagnosis

## Result

| Field | Value | Evidence |
|---|---:|---|
| Lane result | `PARTIAL` | Wolverine fully measured; requested second-clip cross-check unavailable |
| Product status | `VERIFIED=0` | `NORTH_STAR_ROADMAP.md` |
| Owner-visible 3D source | arc-solved overlay, exact equality on 300/300 `world_xyz` frames | `runs/lanes/w7_critique_20260709/wolv_world/wolverine_mixed_0200_mid_steep_corner/ball_track_arc_solved.json`; `.../confidence_gated_world.json` |
| Primary defect | fail-open rendering of rejected/weak BVP fallback arcs | 263/300 fallback frames rendered; 0 hidden; 210 labeled `measured` |
| Single fix | fail closed in world composition for invalid/fallback segments; propagate arc band/demotion/provenance instead of only copying `world_xyz` | artifact comparison above |

All pixel comparisons below use only raw frames with `visible=true`; hidden raw frames contain `[0,0]` placeholders and are excluded. Projection is `K(R X + t)` from the supplied calibration. The solver's frozen visible inlier threshold is 18 px.

## Evidence inventory and cross-check limit

| Clip | Available BALL chain | Status | Evidence |
|---|---:|---|---|
| `wolverine_mixed_0200_mid_steep_corner` | 2D, candidates, arcs, fill, events, CAL, world | full local artifact set | `runs/lanes/w7_critique_20260709/wolv_world/wolverine_mixed_0200_mid_steep_corner/` |
| `owner_critique_zwcth45s` | 0 BALL/arc/world artifacts | failed after 3.662 s at CAL; 15 extra-field validation errors | `runs/lanes/w7_critique_20260709/world/owner_critique_zwcth45s/PIPELINE_SUMMARY.json` |

The second directory has only `source.mp4`, `frame_times.json`, `pipeline_run.json`, and `PIPELINE_SUMMARY.json`. A local `find` found no `ball_track.json`, `ball_track_arc_solved.json`, or `ball_arc_render.json` for that clip.

## Q1. Reprojection versus raw 2D

### Overall

| 3D source | Raw support | n | median px | p95 px | max px | <=18 px | >18 px | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Per-frame solved/world | visible raw frames | 243 | 187.520 | 3869.361 | 3938.759 | 99 | 144 | `ball_track.json`; `ball_track_arc_solved.json`; `court_calibration.json`; `confidence_gated_world.json` |
| Dense render sidecar | both bracketing raw frames visible | 935 | 111.906 | 3876.079 | 3938.759 | — | — | `ball_arc_render.json`; `ball_track.json`; `court_calibration.json` |

Raw visibility is 243/300 = 81.000%. This is coverage, not an accuracy score. Among visible solved frames, 144/243 = 59.259% exceed the solver's 18 px threshold. Of the 99 image-consistent frames, 43/99 = 43.434% still have at least one world defect: outside court, underground, above 10 m, or an implied >20 m/s transition.

### Per solved segment

World counts use the per-frame segment assignment. Endpoint spans are the solver artifact's inclusive anchor declarations.

| Seg | Endpoint span | visible n | reproj median / p95 / max px | <=18 / >18 | selected / unassigned | inlier / outlier | world defects | Classification |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 0 | 0→12 | 10 | 11.952 / 24.811 / 26.639 | 8 / 2 | 8 / 4 | 8 / 4 | behind baseline 9; underground 2; >20 m/s 4 | mixed; mostly image-consistent, bad weak start depth |
| 1 | 12→25 | 14 | 1.609 / 2.381 / 2.519 | 14 / 0 | 14 / 0 | 14 / 0 | behind baseline 6; outside sideline 6 | depth ambiguity signature |
| 2 | 25→157 | 89 | 2200.713 / 3892.688 / 3938.759 | 1 / 88 | 1 / 94 | 1 / 90 | above 10 m 100 total frames; behind baseline 52; >20 m/s 12 | solver ignores 2D; bad/missing endpoints |
| 3 | 157→216 | 51 | 310.662 / 629.914 / 683.136 | 2 / 49 | 0 / 53 | 0 / 52 | underground endpoint; 40.996 m/s entry jump | solver ignores 2D; zero-inlier fallback |
| 4 | 216→223 | 8 | 4.145 / 12.125 / 14.937 | 8 / 0 | 8 / 0 | 7 / 0 | behind baseline 5; underground 3; >20 m/s 8 including entry | depth ambiguity with false/weak endpoint |
| 5 | 223→247 | 22 | 5.443 / 14.761 / 16.221 | 22 / 0 | 23 / 1 | 23 / 1 | behind baseline 12 | depth ambiguity from solver bounce |
| 6 | 247→254 | 8 | 29.294 / 183.163 / 183.879 | 3 / 5 | 3 / 5 | 1 / 5 | underground 2; initial speed 2.657 m/s | solver/2D divergence |
| 7 | 254→266 | 8 | 3.716 / 5.102 / 5.365 | 8 / 0 | 10 / 0 | 10 / 0 | initial speed 2.300 m/s | image-consistent but physically slow |
| 8 | 266→272 | 6 | 3.787 / 5.270 / 5.557 | 6 / 0 | 6 / 0 | 5 / 0 | underground 2 total frames; >20 m/s 6 | strongest depth-ambiguity/candidate-junk signature |
| 9 | 272→289 | 16 | 1.162 / 2.500 / 3.280 | 16 / 0 | 18 / 0 | 16 / 0 | 25+ m/s entry jump; initial speed 2.805 m/s | image-consistent, boundary jump/slow arc |
| 10 | 289→299 | 11 | 1.565 / 2.453 / 2.549 | 11 / 0 | 11 / 0 | 11 / 0 | initial speed 1.790 m/s | image-consistent, physically slow |

Depth-ambiguity segments with good visible reprojection are 1, 4, 5, and 8. Direct solver/2D divergence is dominant in 2 and 3 and present in 6. Segment 0 is mixed. Segments 7, 9, and 10 project cleanly but violate the solver's own 3 m/s minimum initial-speed policy.

Dense render per-segment median/p95/max px: S0 `10.333/23.011/25.250` (n=7), S1 `1.528/2.307/2.551` (68), S2 `2073.678/3895.179/3938.759` (266), S3 `318.719/638.815/683.136` (230), S4 `4.394/10.853/13.437` (11), S5 `6.261/18.231/26.899` (113), S6 `30.611/183.374/183.876` (35), S7 `3.936/10.875/13.245` (44), S8 `3.113/5.567/5.606` (23), S9 `0.991/2.963/3.958` (85), S10 `1.270/2.341/2.549` (53). Evidence: `ball_arc_render.json`.

## Q2. Depth excursions and implied velocity

Court coordinates are `court_netcenter_z_up_m`; regulation bounds are x=±3.048 m and y=±6.7056 m.

| Metric | Value | Frames / ranges | Evidence |
|---|---:|---|---|
| world x | -3.324 to 0.785 m | 300 | `ball_track_arc_solved.json` |
| court-depth y | -0.894 to 11.601 m | 300 | same |
| height z | -0.319 to 23.530 m | 300 | same |
| camera depth | 8.214 to 19.397 m | 300 | same + `court_calibration.json` |
| behind either baseline | 84/300 = 28.000% | 0-8, 20-77, 219-235 | same |
| outside sidelines | 14/300 = 4.667% | 20-33 | same |
| underground z<0 | 10/300 = 3.333% | 10-11, 157, 216-218, 247, 254, 266-267 | same |
| above 10 m | 100/300 = 33.333% | 38-137 | same |
| consecutive speed median | 7.383 m/s | 299 transitions | same |
| consecutive speed p95 | 24.278 m/s | 299 transitions | same |
| consecutive speed max | 40.996 m/s | entry to frame 157 | same |
| speed >20 m/s | 32/299 = 10.702% | 1-4, 27-38, 157, 216-223, 267-273 | same |
| speed >35 m/s | 1/299 = 0.334% | 157 | same |

### Ranked worst owner-visible stretches

| Rank | Frames / time | Defect numbers | Attribution | Evidence |
|---:|---|---|---|---|
| 1 | 26-156 / 0.867-5.200 s | visible reproj p95 3892.688 px; 88/89 >18 px; z>10 m on frames 38-137; max z 23.530 m; 1 inlier / 90 outliers | bad/missing segment endpoints; solver ignores 2D | segments 2; `ball_track_arc_solved.json`; `ball_inflections.json` |
| 2 | 157-215 / 5.233-7.167 s | visible reproj p95 629.914 px; 49/51 >18 px; 0 selected; 0 inliers / 52 outliers; 40.996 m/s entry jump | render-during-fallback | segment 3; `ball_track_arc_solved.json`; `confidence_gated_world.json` |
| 3 | 216-235 / 7.200-7.833 s | 19 visible frames project <=18 px; y reaches 10.219 m; 17 behind-baseline frames; 8 >20 m/s transitions; false bounce at 223 | depth ambiguity with weak/false endpoint | segments 4-5; `events_selected.json`; `ball_track_arc_solved.json` |
| 4 | 0-11 / 0.000-0.367 s | visible median 11.952 px; y reaches 11.601 m; 9 behind-baseline; 4 >20 m/s; 2 underground | bad weak rally endpoint | segment 0; start anchor prior y=13.063 m in `events_selected.json` |
| 5 | 266-273 / 8.867-9.100 s | all 7 visible frames project <=18 px; 7 >20 m/s transitions; 2 underground frames; frame-266 anchor has no raw visible detection | candidate junk from gap anchor plus depth ambiguity | `ball_bounce_candidates.json`; segments 8-9 |

Secondary defect: segment 6, frames 247-254, has visible reprojection `29.294/183.163/183.879` px and only 1 inlier / 5 outliers.

## Q3. Segmentation, endpoints, and render behavior

### Segment/event counts

| Item | Count | Evidence |
|---|---:|---|
| solved segments | 11 | `ball_track_arc_solved.json` |
| normal `fit` | 2 | segments 1, 5 |
| `fit_bvp_fallback` | 9 | segments 0, 2-4, 6-10 |
| top-level solver kills | 0 | `status=ran`, `kill_reasons=[]` |
| auto-bounce anchors | 9 | frames 12, 25, 157, 216, 247, 254, 266, 272, 289 |
| solver-proposed bounce anchors | 1 | frame 223 |
| rally endpoint anchors | 2 | frames 0, 299 |
| contact windows | 0 | `contact_windows.json` |
| selected contact anchors | 0 | `events_selected.json` |
| 2D inflection candidates | 56 from 431 raw candidates; 243 usable frames | `ball_inflections.json` |
| size/depth observations | 0 | `ball_track_arc_solved.json` summary |

### Endpoint alignment to raw 2D direction

Direction angle uses least-squares image velocity over the 3 frames before and after each endpoint. Inflection deltas use `ball_inflections.json`; those candidates are unreviewed.

| Frame | Anchor/method | angle deg | image-y reversal | nearest inflection delta | anchor-vs-raw px | Ruling |
|---:|---|---:|---|---:|---:|---|
| 12 | auto / y cusp | 144.650 | yes | 1 | 0.000 | supported by raw 2D turn |
| 25 | auto / y cusp | 113.707 | yes | 0 | 0.000 | supported by raw 2D turn |
| 157 | auto / y cusp | 158.492 | yes | 1 | 0.000 | supported endpoint; preceding arc remains mis-split |
| 216 | auto / y cusp | 155.487 | yes | 9 | 0.000 | raw turn supports; inflection artifact missed it |
| 223 | solver proposed | 52.037 | no | 2 | 0.000 | mis-split: no bounce-like vertical reversal |
| 247 | auto / y cusp | 114.101 | yes | 1 | 0.000 | supported by raw 2D turn |
| 254 | auto / y cusp | 161.895 | yes | 1 | 0.000 | supported by raw 2D turn |
| 266 | auto / gap ballistic | undefined | no | 5 | 590.087 to hidden `[0,0]` placeholder | no raw 2D exists at anchor frame; candidate-junk risk |
| 272 | auto / y cusp | 170.529 | yes | 0 | 0.000 | supported by raw 2D turn |
| 289 | auto / y cusp | 165.056 | yes | 1 | 0.000 | supported by raw 2D turn |

Missing-split evidence: segment 2 (25→157) contains 29 interior raw-2D inflection candidates yet accepts 1 candidate observation and rejects 90 fit observations. Segment 3 (157→216) contains 6 interior inflections yet accepts 0 candidate observations and has 0 inliers / 52 outliers. With 0 contacts, both long arcs are bounded only by bounce candidates.

### Fallback/kill rendering

| Check | Result | Evidence |
|---|---:|---|
| fallback per-frame positions | 263/300 | `ball_track_arc_solved.json` |
| fallback frames with `flight_sanity_demoted=true` | 263/263 | same |
| fallback frames retaining `world_xyz` | 263/263 | same |
| fallback frames copied into confidence-gated world | 263/263 | `confidence_gated_world.json` |
| fallback frames hidden | 0/263 | same |
| fallback world display bands | 210 measured; 50 physics_predicted_low; 3 physics_predicted | same |
| fallback world frames carrying `render_only=true` | 1/263 | same |
| dense render samples | 1165 across all 11 segments; 0 bridges | `ball_arc_render.json` |
| dense renderer trusted flag | `solver_trusted_for_render=true` | same |
| flight-sanity failed spans | 2/9 evaluated spans | `ball_flight_sanity.json` |
| explicitly demoted sanity frames | 1, frame 157 | same |

The fallback path is neither raw nor hidden. It is solver-generated 3D, marked `arc_weak` upstream, then copied into every world frame. Confidence provenance is inherited from the 2D/fill artifact instead of the arc position.

There is also a segment-ID join mismatch between sanity and dense render artifacts:

- `ball_flight_sanity.json` segment 0 is span 12-25 and passes; `ball_arc_render.json` segment 1 is the same span but is marked failed.
- `ball_flight_sanity.json` segment 2 is span 157-216 and fails; `ball_arc_render.json` segment 3 is the same span but is marked pass.

## Q4. Candidate-selection path

| Field | Value | Evidence |
|---|---|---|
| resolved detector | WASB tennis checkpoint; full-rate stride 1 | `PIPELINE_SUMMARY.json` `best_stack.resolved`; ball stage |
| candidate source mode | `wasb_predict` | `ball_candidates.json` |
| detector provenance | single WASB checkpoint + connected-component postprocessor | `ball_candidates.json` |
| raw candidate points | 813 | 249 nonempty/300 frames; max 5 per frame |
| solver association mode | `free` / `arc_irls_v1` | `ball_track_arc_solved.json` |
| selected candidate-frame observations | 102 | per-segment association totals |
| unassigned candidate-frame opportunities | 157 | per-segment association totals |
| fit inlier / outlier observations | 96 / 152 | per-segment fit totals |
| selected source: primary WASB | 23/102 = 22.549% | arc summary |
| selected source: WASB connected component | 79/102 = 77.451% | arc summary |

### Per-segment selection

| Seg | selected / unassigned | primary / concomp selected | inlier / outlier |
|---:|---:|---:|---:|
| 0 | 8 / 4 | 2 / 6 | 8 / 4 |
| 1 | 14 / 0 | 2 / 12 | 14 / 0 |
| 2 | 1 / 94 | 0 / 1 | 1 / 90 |
| 3 | 0 / 53 | 0 / 0 | 0 / 52 |
| 4 | 8 / 0 | 0 / 8 | 7 / 0 |
| 5 | 23 / 1 | 7 / 16 | 23 / 1 |
| 6 | 3 / 5 | 0 / 3 | 1 / 5 |
| 7 | 10 / 0 | 4 / 6 | 10 / 0 |
| 8 | 6 / 0 | 1 / 5 | 5 / 0 |
| 9 | 18 / 0 | 4 / 14 | 16 / 0 |
| 10 | 11 / 0 | 3 / 8 | 11 / 0 |

FUSION-CHANGE ruling: multi-detector consensus did not feed this solver. `PIPELINE_SUMMARY.json` explicitly says `single-primary-track arc solver`; both primary and alternatives are from WASB. The narrower theory that alternative candidate components altered the primary 2D path is supported: 79/102 selected observations are `wasb:wasb_concomp`. The largest failures, segments 2 and 3, are not explained by consensus fusion; they are endpoint-constrained BVP fallbacks after 2D observation rejection.

## Q5. Root-cause ruling

| Rank | Root cause | Numeric evidence | Affected worst stretch |
|---:|---|---|---|
| 1 | render-during-invalid/fallback plus lost provenance | 263 fallback frames shown; 0 hidden; 210 labeled measured | all; most damaging at 26-215 |
| 2 | missing/weak event anchors and mis-split long arcs | 0 contacts; segment 2 spans 132 anchor frames with 29 interior inflections; segment 3 has 0 inliers | 26-215 |
| 3 | solver rejects/ignores otherwise present 2D | segments 2-3: 137/140 visible frames >18 px; selected 1 and unassigned 147 candidate-frame opportunities | 26-215 |
| 4 | monocular depth ambiguity with weak CAL/anchor constraints | CAL `metric_confidence=low`; `single_view_planar_full_calibration`; size observations=0; 43 image-consistent frames still world-extreme | 0-25, 216-235, 266-273 |
| 5 | single-model candidate junk | 79/102 selected from alternate WASB components; frame-266 gap anchor has no raw visible sample; frame-223 solver bounce lacks vertical reversal | 216-235, 266-273 |

Single highest-impact fix: the world overlay must fail closed per segment. Any segment with fallback status plus failed physical sanity, insufficient inliers, or visible reprojection p95 >18 px must contribute `world_xyz=null`; its arc band/demotion/provenance must reach confidence gating. On this clip, that one boundary removes the absurd 23.530 m apex, the zero-inlier 157-216 arc, and the 210 falsely `measured` fallback frames from owner-visible 3D without changing the 2D detector.

No source or upstream artifact was edited.
