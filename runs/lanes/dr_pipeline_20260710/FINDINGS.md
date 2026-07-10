# dr_pipeline_20260710 — independent pipeline/data audit

## Verdict

The owner symptoms are mostly upstream data failures, not presentation alone.

- **S1 low frame rate:** Wolverine artifacts are authored at 30 fps and BODY covers every frame on which a selected track exists (705/705 tracked player-frames). The W7 data therefore does **not** explain low playback FPS; that portion is viewer/performance-side. The zwcth45s demo does have a data-side cadence failure: no BODY output at all, and the pre-fix BODY execution selected only 200/1,315 tracked frame indices before failing.
- **S2 missing people:** primarily strict target-court filtering plus unstable top-4 identity selection, aggravated by the production OSNet/global-association asset being missing. It is not primarily BODY: people are already absent from `tracks.json`. Raw YOLO/BoT-SORT produced at least four person boxes on 290/300 Wolverine frames and 1,054/1,350 zwcth frames, but zero-margin court projection reduced the source-only four-box ceiling to 77/300 and 73/1,350; selection then reduced true output four-player frames to 39/300 and 0/1,350. Without labels, raw boxes are an upper bound, not proof they are the four players.
- **S3 missing skeletons:** Wolverine has 705 skeleton player-frames and 705 mesh-index player-frames, exactly matching its 705 selected track player-frames; there is no data-to-world skeleton drop there. zwcth R1/R2 has zero skeleton and zero mesh frames because BODY failed before emission. The R2 manifest nevertheless says `mesh_status="skeleton_only"`; that is a packaging/status defect, not evidence that a skeleton exists.
- **S4 hidden ball:** fail-closed policy is working but exposes a weak lift. Wolverine has 243/300 visible 2D detections but only 75/300 fail-closed 3D frames; 172 visible 2D sightings (70.8% of visible 2D) do not survive to 3D. zwcth has 480/1,350 visible 2D but only 58/1,350 3D; 435 visible sightings (90.6%) are unused in 3D. No `source="physics_interpolated"` frame is emitted in either pulled clip.
- **S5 paddle:** Wolverine emits 705 preview paddle frames, but every frame is wrist/palm derived; confidence p50 is 0.506, no frame has a reprojection error, no detector boxes, reflection contacts, or contact locks were used, and RKT remains unscoreable. zwcth emits no paddle because it emits no skeleton. The paddle estimator consumes already-world-space `joints_world` and does not use the new typed coordinate API.

`VERIFIED=0` remains binding. This audit used generated Burlington/Wolverine/zwcth artifacts only and read no Outdoor/Indoor labels.

## Coverage summary

| Clip/run | Frames | raw person boxes >=4 | on-court non-overlap >=4 | selected tracks =4 | BODY JPEG schedule | mesh PF | skeleton PF | paddle PF | ball 2D visible | fail-closed 3D | hidden 3D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wolverine W7 | 300 @30fps | 290 (96.7%) | 77 (25.7%) | 39 (13.0%) | 244/244 tracked indices; 244 materialized | 705 | 705 | 705 | 243 (81.0%) | 75 (25.0%) | 225 (75.0%) |
| zwcth45s R1 | 1,350 @30fps | 1,054 (78.1%) | 73 (5.4%) | 0 (0%) | 1,200/1,315 materialized; BODY requested 200 and missed 18 | 0 | 0 | 0 | 480 (35.6%) | 58 (4.3%) | 1,292 (95.7%) |
| zwcth45s R2 | 1,350 @30fps | 1,054 (78.1%) | 73 (5.4%) | 0 (0%) | same pre-fix mismatch; BODY fails on frame 41 | 0 | 0 | 0 | 480 (35.6%) | 58 (4.3%) | 1,292 (95.7%) |

PF means player-frame. The incomplete W7 `world/owner_critique_zwcth45s` pull contains only `PIPELINE_SUMMARY.json`, `frame_times.json`, `pipeline_run.json`, and a broken VM-local source symlink, so frame coverage is explicitly uncomputable from that directory. The distinct Wolverine and zwcth clips satisfy the two-clip acceptance requirement.

Detailed evidence:

- `per_frame_coverage.csv`: 3,000 rows with raw/on-court/selected TRK, BODY schedule/materialization, mesh, skeleton, paddle, 2D BALL, arc, world, fail-closed, and confidence-band state.
- `per_player_coverage.csv`: tracked/mesh/skeleton/paddle coverage by selected player.
- `coverage_summary.json`: aggregate counts, schedules, artifact presence, source-only ceilings, and counterfactual bounds.
- `ball_segment_fate.{csv,json}`: all 11 Wolverine and 8+8 zwcth R1/R2 segments.
- `paddle_quality.json`: confidence/evidence-channel and field completeness.

## S1/S2 — person and BODY cadence fate

### Where TRK loses people

The detector itself ran at `conf=0.05`, not a high threshold: `RealYOLO26BoTSORTReIDTrackingRunner` declares that default at `threed/racketsport/orchestrator.py:497` and passes it to Ultralytics at `:531-543`. The loss chain is:

1. YOLO/BoT-SORT emits the raw pool (`orchestrator.py:545-559`).
2. `build_tracks` projects each box bottom-center through the court homography and drops it with **zero runoff margin** when outside the regulation footprint (`scripts/racketsport/track.py:156-161`). Metrics record 1,171/2,048 tracked person boxes outside court on Wolverine and 1,617/5,024 on zwcth.
3. It ranks surviving tracker IDs by stability and retains only four (`track.py:184-192`). That drops another 172 player-frames on Wolverine and 603 on zwcth.
4. The intended full raw-pool association never ran because the OSNet file resolved by best-stack was absent; both pipeline summaries say “kept loose-pool tracks.json.” The skip is implemented at `scripts/racketsport/process_video.py:1588-1597`.

Source-only attribution:

| Clip | Raw >=4 | After court projection >=4 | After selected tracks =4 | Court/projection-loss frames (raw >=4, court <4) | Selection/association-loss frames (court >=4, selected <4) |
|---|---:|---:|---:|---:|---:|
| Wolverine | 290 | 77 | 39 | 213 | 38 |
| zwcth45s | 1,054 | 73 | 0 | 981 | 73 |

This rules out “BODY-only loss.” It does **not** prove 213/981 frames are erroneous court rejects: raw pools contain spectators, and protected labels were not used. Confirmation path: score a no-GT source-only candidate with a positive apron margin and the actual OSNet association, then use the frozen reviewed TRK scorer. Do not tune from this audit.

zwcth also has an identity-fragment signature: selected ID 3 covers frames 36–878 and ID 29 covers 880–1,349 with no overlap, while both are assigned far/right. The selected four IDs never coexist. Raw IDs 15 and 45 retain 337 and 266 on-court frames respectively but are not selected. This is consistent with fragmentation/top-4 selection, not a renderer dropping an otherwise complete player.

### Where BODY cadence is lost

- Wolverine: the 300-frame clip has selected tracks on 244 distinct frames. Stride-2 keeps 122 base frames, and the ball-aware mesh plan adds the other 122, so all 244 are scheduled. The 300 MiB byte budget is not binding (`budget_limited=false`, estimated 43.9 MB). BODY, skeleton, mesh-index, and paddle all contain the same 705 selected player-frames. Low visual FPS on this artifact is therefore viewer-side or mesh-download/decode cost, not missing BODY samples.
- zwcth pre-fix: 1,315 tracked indices become a 658-frame stride-2 base, the events plan initially declares all 1,315 eligible, and the frames stage uniformly caps materialization to 1,200. A later BODY execution requests 200 hybrid-plan frames, 18 of which are absent from that independently sampled 1,200. R2 fails on frame 41; local reproduction fails on frame 75. No BODY output is authored.
- Current HEAD fix `7a6fd828e` derives one authoritative BODY request, retains it in the materialization schedule, validates set equality, and hands the same execution to the orchestrator (`process_video_body_frames.py:129-177,192-205`; `process_video.py:2164-2185,2187-2224`; `orchestrator.py:3297-3335`). Focused verification is 33/33 green. The real post-fix proof reports zwcth 1,200 requested/materialized with zero missing and Wolverine byte-identical 244/244.
- Honest residual: the fix still intentionally excludes 115/1,315 tracked zwcth indices under the 1,200 cap. That is surfaced, but it does not satisfy an unconditional “always skeleton or mesh on every tracked frame” product rule. NS-04.2 must either raise/split the cap or treat missing required representation as partial with an explicit per-frame reason.

## S3 — skeleton/mesh contract

The target DAG calls for cheap full-rate joints before deep BODY. The current spine has no independent cheap-joints producer: serial order is BALL → arc → events → fill → frames → BODY (`scripts/racketsport/process_video.py:836-842,871-882`). Events therefore sees no same-run skeleton and writes a blocked wrist artifact (`process_video.py:2747-2758`). BODY later produces both joints and mesh on its selected schedule.

Artifact ruling:

- Wolverine: `skeleton3d.json` has 705 player-frames, `body_mesh_index.json` has 705 player-frames, and the world has 705 joint player-frames. The manifest links the windowed mesh index. There is no skeleton drop in world assembly.
- zwcth: BODY fails; `skeleton3d.json`, `smpl_motion.json`, and the mesh index are absent; world reports `joint_player_frame_count=0`, `mesh_player_frame_count=0`, and 2,803 `track_only` player-frames. R2's manifest still calls this `skeleton_only` because manifest code uses that label whenever neither mesh artifact exists, without checking `skeleton3d.json` (`process_video.py:3819-3835`). The manifest schema has no skeleton URL, so it relies on world-embedded joints that are also absent.

Fix: add a representation status that distinguishes `track_only`/`body_missing` from `skeleton_only`, derive it from validated world counts plus actual skeleton presence, and fail the product bundle partial when neither skeleton nor mesh exists. This belongs to NS-01.5 and NS-04.2.

## S4 — BALL segment fate and recoverable bounds

### Wolverine (11 segments)

| Segment | Status | Frames attributed | Current fail-closed fate | Reason |
|---:|---|---:|---|---|
| 0 | fallback | 12 | suppressed | outside court volume |
| 1 | fit | 14 | emitted | fit always trusted |
| 2 | fallback | 131 | suppressed | 1 inlier / 90 outliers / 3,585 px max reprojection; implausible apex |
| 3 | fallback | 59 | suppressed | 0 inliers / 52 outliers / 620 px max reprojection |
| 4 | fallback | 8 | suppressed | implausible apex |
| 5 | fit | 23 | emitted | fit always trusted; planar-depth residual remains |
| 6 | fallback | 8 | suppressed | 1 inlier / 5 outliers |
| 7 | fallback | 11 | emitted | fit statistics pass; slow-speed warning is not a spatial kill |
| 8 | fallback | 7 | suppressed | below-net slack |
| 9 | fallback | 16 | emitted | fit statistics pass; slow-speed warning is not a spatial kill |
| 10 | fallback | 11 | emitted | fit statistics pass; slow-speed warning is not a spatial kill |

Suppressed IDs are exactly `[0,2,3,4,6,8]`; emitted is 75/300. The checked-in W7 `virtual_world.json` predates the fail-closed re-composition and still has 300 world positions, but recomputation through current HEAD produces the documented 75/300. The zwcth cold artifacts already contain 58/1,350 fail-closed world positions and suppress `[0,2,3,4,5]`; segments 1, 6, and 7 emit 14, 35, and 9 frames.

The world builder does apply the overlay (`threed/racketsport/virtual_world.py:110-132`), and the suppression logic is at `:371-421,475-558`. But `_ball` reconstructs only source/frames and `_ball_frame` copies only per-frame render fields (`:1411-1486`), so the run-level `arc_solved_overlay.fail_closed.segment_verdicts` block is dropped before `virtual_world.json`. That is the booked provenance residual.

### UKF and TT3D bounds

These are bounds, not measured improvements:

- A strict UKF-seeded fallback that requires an **adjacent true `fit`** segment can attempt Wolverine suppressed segments 0, 2, 4, and 6: at most 159 additional frames, lifting 75 to at most 234/300 (78.0%). On zwcth only segment 5 is adjacent to true fit 6: at most 128 additional frames, lifting 58 to at most 186/1,350 (13.8%). Every recovered frame must still pass reprojection/spatial gates; otherwise it remains hidden.
- TT3D joint-anchor search has a Wolverine kill bar of fallback `<5/11`; from 9 fallback segments, at least five must become normal fits. Depending on which five convert, emitted coverage could rise by only 15 frames or by as many as 218 (90–293/300 total). This range is deliberately wide because no candidate was run. Confirmation requires scoring the candidate with the same fail-closed gate and checking the `<5/11` kill criterion.
- There is no current between-segment physics interpolation in the pulled artifacts: `physics_interpolated_frames=0` for all runs. The 13 frames labeled `physics_predicted` by the confidence gate come from `source="physics3d_reconstructed"`, and fail-closed overlay can still null them; they are not provenance-marked bridge interpolation. The code supports `source="physics_interpolated"` (`ball_physics_fill.py:336-385`), but it did not fire here.

## S5 — paddle fields, confidence, and transform chain

Wolverine's `racket_pose_estimate.json` contains per-frame `pose_se3.R`, `pose_se3.t`, `conf`, `world_frame`, translation unit, source, ambiguity, render-only flags, trust band, and optional reprojection error. Measured quality is preview-only:

- 705/705 selected track player-frames emitted; confidence min/p50/p95/max = 0.220/0.506/0.518/0.522.
- 705 `palm_fitted`; 0 `contact_locked`; 0 `grip_extrapolated`; 0 ambiguous.
- 0/705 reprojection errors populated.
- detector boxes disabled, detector handedness disabled, reflection contacts unavailable.
- every frame is `source=wrist_palm_grip_fused`, `estimated_from_wrist`, render-only, and not for RKT promotion.

Transform chain: the estimator reads `skeleton3d.players[].frames[].joints_world` directly (`paddle_pose_fused.py:198-235`), constructs a world hand frame, composes a constant grip rotation/translation as `R_world = R_hand @ R_grip`, `t_world = wrist_world + R_hand @ t_grip` (`:1640-1652`), and writes `world_frame="court_Z0"` (`:80-89,1860-1885`). Optional detector correction imports `court_calibration.project_world_points` locally (`:1606-1637`). It never imports or calls `coordinates.py`.

`coordinates.py` correctly advertises itself as only a small NS-01.4 slice, not full adoption (`coordinates.py:1-5`), and supplies typed spaces plus canonical camera transforms (`:41-125`). `mhr_decode.py` uses it only for apply-translation-once, while paddle still trusts already-world-space joints and ad-hoc projection. Thus P0-D is still booked. Fix sketch: attach a typed coordinate-space value to skeleton/paddle inputs, route any detector pixel evidence through raw/undistorted/reference transforms, replace `court_Z0` with the canonical schema literal, retain both planar pose hypotheses, and populate reprojection/confidence from real paddle corners. NS-01.4 + NS-01.7 + NS-03.RKT/NS-04.5; medium-to-large effort plus GT.

## S6 — rev-9 to HEAD regression sweep

1. **Cold BODY frame materialization — fixed, scoped pass.** Commit `7a6fd828e` is at HEAD ancestry and directly fixes the defect introduced by `b437b4118`. The real zwcth signature goes 18 missing of 200 requested to 0 missing of 1,200; Wolverine stays 244/244. Focused tests: 33 passed. Residual 1,200 cap noted above.
2. **BODY failure silently rewrites upstream planning artifacts — still present, high severity.** R2's stage record says events authored `ball_aware`, 1,315 world-mesh frames, while the pulled final `frame_compute_plan.json` is `hybrid`, 200 frames and `body_compute_execution.json` requests 200. Root cause: after an inner BODY failure, `_write_best_effort_review_artifacts` regenerates `frame_compute_plan.json` unless a successful BODY artifact protects it, then regenerates `body_compute_execution.json` (`orchestrator.py:2756-2764,2767-2838`). This mutates the already-authored upstream plan and makes the final artifact disagree with `PIPELINE_SUMMARY`. Fix: review writers must be side-effect-free or write namespaced review sidecars; never overwrite authoritative stage outputs. NS-01.3/01.6 + NS-04.2; small-to-medium.
3. **Arc overlay provenance — behavior wired, attestation dropped.** Numeric suppression is correct, but world assembly omits the run-level verdict map as described above. NS-01.5/01.7 + NS-04.6; small.
4. **Events-before-BODY staleness — confirmed.** Both pulled clips have `wrist_velocity_peaks.status=blocked(missing_sam3d_skeleton3d)` and `contact_windows.events=[]`; Wolverine later has 705 skeleton frames, but no post-BODY event pass reruns. Audio extraction is attempted on the normal path (`process_video.py:2773-2786,3054-3067`), but both sources report `no_audio_stream`, so audio contributes zero. This is P0-G: same-run BODY/paddle never refines events/arcs/schedule. NS-01.7 + NS-03.EVENTS + NS-04.1/04.3; large spine change.
5. **Camera-motion reference retains parent-video frame numbers — confirmed.** zwcth's calibration has `solved_over_frames=[109050,190860]`; the 45-second child has only 1,350 frames. `_reference_frame_idx` takes `solved_over_frames[0]` literally (`camera_motion.py:1495-1505`) and the stage fails with “reference frame 109050 outside processed frame range 0..1349” (`:88-102`). Fix: ingest must remap parent/source frame identity into child-local PTS or explicitly mark camera motion unavailable; never treat parent indices as local. NS-01.1/01.4; medium.
6. **Unfingerprinted legacy artifacts can still be adopted as reusable — current risk.** The content-addressed wrapper sets `_identity_allows_reuse=True` for `reason="unfingerprinted_stale"` (`process_video.py:1044-1058`), allowing stage-local schema checks to reuse an old artifact and publish it into the first generation. This is deliberately documented compatibility but violates P0-C's “new metadata/models cannot reuse old pixels/results” exit gate. Fix: first generation should rebuild or require an explicit migration/import attestation. NS-01.3; medium.
7. **Manifest representation status lies on total BODY absence — confirmed.** `skeleton_only` is selected from absence of mesh files alone (`process_video.py:3819-3835`). Fix above. NS-01.5/04.2; small.

## Ranked defect list

| Rank | Severity | Defect / owner symptom | Root evidence | Fix sketch | Effort | NS owner |
|---:|---|---|---|---|---|---|
| 1 | Critical | TRK never has four simultaneous zwcth players; W7 only 13% | raw→court→top4 counts; missing OSNet association | run default association asset, score apron/court filter and detector separately, preserve raw-pool provenance | M | NS-03.TRK, NS-01.3/1.4 |
| 2 | Critical | No skeleton/mesh/paddle on zwcth while manifest says skeleton-only | BODY frame mismatch; joint/mesh/paddle=0; manifest code | ns016 fix plus honest `body_missing/track_only`; partial bundle; cap policy ruling | S–M | NS-01.5/1.6, NS-04.2 |
| 3 | Critical | No post-BODY event/contact/arc refinement | blocked wrist before BODY; 0 events; 705 joints arrive later | target two-pass DAG with dependency hashes and refined pass | L | NS-01.7, NS-03.EVENTS, NS-04.1/4.3 |
| 4 | High | 70.8–90.6% of visible 2D BALL is unused by emitted 3D | coverage tables + segment fates | UKF graceful degradation first; TT3D joint-anchor search; score every candidate | M–L | NS-03.BALL, NS-04.3/4.5 |
| 5 | High | Failed BODY rewrites upstream plan/execution | R2 summary/artifact disagreement; orchestrator review writer | namespaced sidecars; immutable authoritative plans | S–M | NS-01.3/1.6, NS-04.2 |
| 6 | High | Paddle placement is unmeasured and bypasses typed coordinate API | 0 reprojection, no boxes/reflections/contact locks; code trace | typed spaces, corner GT, both IPPE poses, refined fusion | M–L + GT | NS-01.4/1.7, NS-03.RKT, NS-04.5 |
| 7 | High | Parent-frame camera-motion reference invalid on clips | `109050` vs `0..1349` | content-addressed source-to-child PTS/frame mapping | M | NS-01.1/1.4 |
| 8 | Medium | Fail-closed verdict provenance absent from world | overlay block exists before `_ball`, absent after | world-adjacent sidecar or schema field | S | NS-01.5/1.7, NS-04.6 |
| 9 | Medium | Legacy `unfingerprinted_stale` adoption weakens content-addressed first run | process wrapper | rebuild or explicit migration attestation | M | NS-01.3 |

## Honest issues

- No protected Outdoor/Indoor labels were read. Consequently, raw-pool attribution is source-only: it separates stages but cannot label every raw box as player vs spectator.
- `world/owner_critique_zwcth45s` is an incomplete pull; no coverage claim is made for it.
- The UKF and TT3D numbers are counterfactual ceilings/ranges, not candidate results. No solver code was changed or run.
- ns016 proves schedule/materialization correctness and BODY input assembly, not GPU BODY accuracy or completion on a new zwcth cold run.
- Wolverine's checked-in W7 world is rev-9/fail-open; the 75/300 figure is a deterministic recomputation through current HEAD and matches the rev-11 report, not a claim that the original file already stores the corrected positions.
- Paddle coverage is high only relative to already-selected tracks. It does not repair missing people, and confidence has no true-corner/pose accuracy meaning.
- S1's remaining playback smoothness cannot be decided from pipeline artifacts alone; this lane rules only that Wolverine authored full selected-track BODY cadence at 30 fps.

