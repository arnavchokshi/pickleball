# dr_viewer_20260710 — independent replay/viewer audit

Date: 2026-07-10  
Scope: read-only audit of `web/replay`, supplied Wolverine/Burlington-safe artifacts, existing headless verification, and the demo assembly. No protected Outdoor/Indoor labels were read. No stack or source change was made; `VERIFIED=0` remains binding.

## Executive ruling

The owner's symptoms are real, but they do not have one cause.

- **S1 (low frame rate):** the watched demo is unambiguously **assembly-bound at 10 unique frames/s**: it was built from screenshots at 0.1 s increments, then encoded/up-converted to a nominal 30 fps MP4. The live viewer is also plausibly renderer-bound under the heavy world: existing headless proof reads 1.2–3.3 3D FPS with four meshes versus 46.2 FPS on the zero-player owner world. Mesh motion remains data-bound by computed/selected frames and hold rules even when the canvas can render faster.
- **S2 (people missing):** primarily **data-side**. The owner-critique world verified with `Players=0`; the Wolverine world has four player identities but per-player renderable skeleton/mesh coverage is highly incomplete. The viewer cannot invent an absent player. Viewer-side chunk loading and implausible-skeleton gating can add transient blanks.
- **S3 (no skeleton or mesh):** the proposed proximate cause is **refuted**. Mesh and skeleton paths are independent. With no mesh, the viewer renders BODY joints when present, or a floor-anchored proxy skeleton when only placement exists. It renders neither when the time-matched world frame has neither usable joints nor a floor anchor, when BODY is absent, or when a skeleton is gated implausible. In the supplied Wolverine world, ideal cached-chunk simulation still yields neither geometry for 19.7–49.7% of source frames depending on player; that is mostly missing BODY/TRK/placement evidence, not a missing viewer fallback.
- **S4 (ball hidden):** mostly the intentional **fail-closed data contract**. The viewer shows no current ball during long suppressed spans; it predicts only across short gaps (<=0.25 s) and labels predicted geometry with dashed/cyan/amber styling plus a `ball: predicted` HUD. It does not display the exact `physics_predicted` provenance token. A new viewer risk is that trail building can connect visible samples across gaps up to 1.75 s without respecting segment IDs.
- **S5 (bad/mispositioned paddle):** both layers contribute. The real artifact is not a flat rectangle: it is a crude 16-vertex/24-triangle extruded face-and-handle proxy plus an oversized normal arrow. Its pose is low-confidence `wrist_palm_grip_fused`, render-only, not true paddle 6DoF. The viewer also holds the nearest paddle frame across arbitrarily large internal gaps; Wolverine contains 1.467 s paddle gaps, so a stale pose can remain visibly attached to the wrong moment. The declared court transform is internally consistent; transform error is an unverified hypothesis because no paddle GT was available.

## 1. Per-entity render decision table

All defaults below are for the normal 3D view, not shot-trails mode. The parent scene switches to a separate shot-trails layer when that camera mode is active, suppressing the ordinary players/paddles/ball/event subtree (`web/replay/src/App.tsx:1229-1274`). Default layer state enables ball, trail, paddles, skeletons, and solid meshes; event/floor markers, implausible skeletons, hands, and debug clouds are off (`web/replay/src/viewState.ts:77-89`).

| Entity | Drawn | Hidden / degraded | Exact viewer branches |
|---|---|---|---|
| **Player solid mesh, per player/time** | The mesh toggle is enabled and at least one active mesh exists; a body mesh chunk/legacy artifact is loaded; the world player maps to a mesh player; `bodyMeshFrameForTime` returns a frame; vertices and faces are non-empty. Exact frames draw; before/after range frames may fade for 0.12 s; gaps <=0.15 s in the same window hold the previous frame. Preview/low-confidence frames are still drawn with estimated material, not hidden. | Whole layer is absent when disabled or active count is zero. Per player: no loaded chunk, no active index window, missing mesh player, time outside tolerance, a gap not covered by the 0.15 s hold or 0.12 s edge fade, or empty vertices/faces. During an async chunk switch the previous chunk generally cannot answer the new time, so mesh count temporarily falls to zero. | Layer count and enable gate: `web/replay/src/viewState.ts:269-315`; parent branch: `web/replay/src/App.tsx:1250-1252`; active mapping/renderability: `web/replay/src/viewerData.ts:1217-1257`, `2689-2695`; exact/hold/fade/undefined rules: `web/replay/src/viewerData.ts:2097-2124`, `2664-2671`; index window and chunk load: `web/replay/src/App.tsx:719-767`, `web/replay/src/viewerData.ts:797-806`; mesh/material draw: `web/replay/src/App.tsx:2217-2244`, `2343-2396`.
| **Player skeleton, per player/time** | The skeleton toggle is enabled and at least one plausible skeleton candidate exists globally. For each player, `frameForTime` selects the nearest frame within the player's overall range. A BODY skeleton draws when >=2 joints survive confidence/named-bone construction and the frame is not implausible (unless the debug toggle permits it). If BODY joints do not produce a skeleton and the frame is not marked implausible, `skeletonForFrame` draws a 13-joint floor-anchored proxy from `floor_world_xyz` or `track_world_xy`. | Hidden if the layer toggle/count gate is false, no in-range frame exists, the frame has neither usable BODY joints nor floor/track anchor, or it is `skeleton_implausible` while the debug toggle is off. The proxy is deliberately suppressed on an implausible frame. **No mesh condition appears anywhere in the skeleton decision.** A bug remains: when all active skeletons are implausible, the plausible-skeleton global count is zero, so `showSkeletons` is false even if the separate implausible toggle is on. | Nearest frame/range: `web/replay/src/viewerData.ts:1095-1100`, `1159-1165`; global count/gate: `web/replay/src/viewState.ts:284-314`, `317-323`; per-player decisions and both draw branches: `web/replay/src/App.tsx:2091-2107`, `2139-2182`; floor proxy geometry: `web/replay/src/App.tsx:3079-3117`; BODY joint/bone and implausible rules: `web/replay/src/App.tsx:3168-3198`, `3216-3235`.
| **Ball trail and current marker** | The ball/trail layer must be enabled and its scene count nonzero. Sample priority is: explicit `ball_arc_render` samples, else trusted arc-solved samples, else virtual-world frames. Current marker draws only when the nearest renderable sample is within 0.12 s. `sampleBallArcRenderAtTime` injects a current-time interpolated render-only sample when bracketing samples are <=0.25 s apart. Measured is solid yellow; interpolated/extrapolated is dashed cyan/blue with a pulsing marker; weak is dashed amber; unknown is dashed gray. Trail draws pairs of renderable samples in the 1.75 s lookback. | `hidden`, `visible=false`, or null world position produces no marker and breaks a world/arc-solved trail. No sample within 0.12 s means no marker. Long fail-closed spans therefore show a fading recent trail, then nothing. An untrusted arc-solver artifact yields no parsed samples; the HUD says solver off, but a nonempty trusted `ball_arc_render` takes visual priority when supplied. | Parent/count: `web/replay/src/viewState.ts:284-300`, `web/replay/src/App.tsx:1262-1272`; sample priority/current draw: `web/replay/src/components/modules/BallTrailLayer.tsx:41-69`; short-gap interpolation/nearest rule: `web/replay/src/ballArcRender.ts:95-127`; band styling including hidden: `web/replay/src/components/modules/ballTrail.ts:161-267`; trail gap/reset/current tolerance: `web/replay/src/components/modules/ballTrail.ts:270-351`, `676-699`; sphere/trail geometry: `web/replay/src/components/modules/BallTrailLayer.tsx:73-151`.
| **Paddle, per player/time** | Paddle layer enabled and at least one paddle has an active frame. For each paddle entity, the viewer picks the nearest frame anywhere between the first/last timestamps (plus a ~1.5-frame edge tolerance). If the frame carries >4 vertices and >2 faces, it draws that indexed mesh; otherwise it constructs a flat rounded fan from `pose_se3` and dimensions. The supplied Wolverine frames use the indexed 16-vertex extruded proxy. Edges and a pose-normal arrow/tip are also drawn. | Hidden only outside the paddle entity's overall time range, if it has no frames, or when the layer is disabled/count zero. **There is no maximum internal-gap rejection**, so missing spans do not hide the paddle—they show a stale nearest pose. | Active-frame gate: `web/replay/src/viewerData.ts:1262-1275`, `2679-2687`; parent: `web/replay/src/App.tsx:1249`; indexed-vs-fallback geometry and transform: `web/replay/src/App.tsx:3248-3306`, `3338-3349`, `3402-3405`; rendered mesh/edges/normal: `web/replay/src/App.tsx:2270-2340`; producer court transform and proxy mesh: `threed/racketsport/virtual_world.py:1578-1635`, `1718-1740`.
| **Contact/bounce/net/landing markers** | Event toggle must be enabled and `eventMarkersForTime` must report an active marker. `ImpactMarkers` then extracts marker candidates. Landing spots persist whenever the parent mounts; bounce rings show from 0.18 s before to 0.75 s after; paddle bursts within +/-0.28 s; net markers within +/-0.6 s. | Default event toggle is off. Parent is suppressed whenever `eventMarkersForTime` count is zero. That parent gate can also hide persistent landing spots even though `LandingSpot` itself has no time gate. Per-kind windows hide bounce/contact/net outside their short windows. | Default off: `web/replay/src/viewState.ts:77-89`; parent count/branch: `web/replay/src/viewState.ts:269-315`, `342-372`, `web/replay/src/App.tsx:1253-1261`; extraction and per-kind dispatch: `web/replay/src/components/modules/ImpactMarkers.tsx:17-69`; visibility windows: `web/replay/src/components/modules/ImpactMarkers.tsx:71-127`.
| **Trust badges/bands** | There are only indirect render encodings: player entity trust colors skeleton/floor/trail; mesh-frame `trust_badge` changes material; ball bands change line/marker style and HUD state; paddle `render_only`/source changes material. Timeline event buttons include badge in class/title. | The required compact per-visible-entity trust badge is not mounted. `TrustBandPanel` and `PlayerTrustBandPanels` are defined but unused. Missing mesh-frame badge defaults to the solid material, so pipeline `human_review`/tap-tier provenance becomes visually solid. Paddle frames carry trust bands but `PaddleProxy` shows no badge/reason. Existing verifier artifacts report `trust_chip_count=0`. | Dead component: `web/replay/src/App.tsx:1786-1805`; player color: `web/replay/src/App.tsx:2144-2149`; mesh material use: `web/replay/src/App.tsx:2366-2374`; ball HUD: `web/replay/src/components/modules/BallHonestyHud.tsx:23-64`; optional mesh-frame badge parser: `web/replay/src/viewerData.ts:1994-2005`; timeline badge only: `web/replay/src/App.tsx:1764-1778`.

### S3 fallback proof

The skeleton path never checks `activeBodyMeshes`, `bodyMesh`, `mesh_ref`, or the solid-mesh layer. `Players` receives world frames and renders skeletons at `App.tsx:2066-2200`; `SolidBodyMeshes` is a separate sibling at `App.tsx:1250-1252`. When BODY joints exist, `bodyJointSkeletonForFrame` draws them. When they do not, `proxySkeleton = ... skeletonForFrame(frame)` creates a floor proxy (`App.tsx:2141-2143`, `3083-3117`). Therefore:

| No-mesh case | Skeleton result |
|---|---|
| Byte budget/mesh stride omitted mesh but world frame has valid BODY joints | Real BODY skeleton draws. |
| BODY joints absent but tracking/placement floor anchor exists | Floor-anchored proxy skeleton draws. |
| BODY and floor/track placement both absent | Nothing draws. This is the common data-side S3 case. |
| Skeleton marked implausible, debug toggle off | Nothing draws unless a mesh independently exists. |
| Chunk is loading/missing but world joints remain | Skeleton still draws; mesh alone blanks. |

Supplied Wolverine quantification (300 source frames, ideal instantaneous chunk availability, reproducing the viewer's range/hold rules):

| Player | skeleton frames | mesh frames | both | neither | neither |
|---:|---:|---:|---:|---:|---:|
| 19 | 158 | 157 | 153 | 138 | 46.0% |
| 20 | 151 | 150 | 150 | 149 | 49.7% |
| 21 | 159 | 158 | 153 | 136 | 45.3% |
| 22 | 238 | 240 | 237 | 59 | 19.7% |

Source artifacts: `runs/lanes/demo_beststack_render_20260710/after_wolv/confidence_gated_world.json` and `body_mesh_index/body_mesh_index.json`. The world itself reports 705 joint/floor player-frames and per-player observed coverage of 50.3–81.3%; the mesh index also has 705 player-frames. Actual browser coverage can be worse during chunk fetches.

## 2. Frame-rate chain (S1)

### Live cadence

1. The HTML video is the clock. A continuous `requestAnimationFrame` loop samples `video.currentTime`; sampling is throttled to `1000 / clamp(world.fps, 24, 60)` and updates React state only when time differs by >4 ms (`web/replay/src/App.tsx:805-821`). `timeupdate`, seeking, seeked, and metadata handlers also update it (`App.tsx:940-956`, `1136-1145`). There is no fixed playback tick.
2. R3F/Three renders on its own browser rAF. `FpsProbe` counts actual `useFrame` callbacks in ~500 ms windows (`App.tsx:1668-1687`). This is renderer FPS, not source-data FPS.
3. Every entity independently looks up the nearest sample. Player lookup has only an overall range test (`viewerData.ts:1095-1100`, `1159-1165`); paddle lookup has no internal gap bound (`viewerData.ts:2679-2687`); default mesh lookup holds <=0.15 s and edge-fades 0.12 s but does not perform continuous interpolation (`viewerData.ts:2097-2124`).
4. The optional **2x FPS** toggle is off by default. When enabled, it inserts midpoint skeleton/world frames and eligible same-window mesh frames; it does not increase measured computation (`viewerData.ts:2398-2431`, `2445-2520`, `2522-2569`).

### Data-bound mesh cadence

- Historical close-proof: all clips had a 100 selected-source-frame cap; effective mesh FPS was Burlington 9.99, Wolverine 10.00, Outdoor 5.21, IMG1605 10.10 (`runs/lanes/w6_playbackdiag_20260708/playback_decision_table.md`).
- The completed Outdoor byte-budget proof retained 409/409 eligible frames at 112.6 MiB and improved effective mesh cadence **5.21 -> 21.32 FPS (4.1x)**; source pointer: `runs/archive/root_docs_20260709/OWNER_CHECKIN_20260709.md:53` and archived checklist `:786`. No protected labels were opened for this audit.
- Current supplied Wolverine W7 artifacts select 244 unique source times over 10 s and contain 705 player-mesh frames in 23 chunks (`wolv_world/.../frame_compute_plan.json`, `body_mesh_index/body_mesh_index.json`). That removes the old 100-frame cap for this run, but per-player BODY/TRK coverage still leaves the large neither-rates above.

### Renderer-bound evidence

- Four-player Wolverine headless captures: existing verifier readouts are 3.1 FPS before fail-closed and 1.2 FPS after in the long screenshot run; smoke snapshots read 3.3 and 2.2 FPS (`runs/lanes/demo_beststack_render_20260710/{frames_before_world,frames_after_world,viewer_smoke,viewer_smoke_after}/headless_verify.json`).
- The owner-critique run with **Players=0** read 46.2 FPS (`runs/lanes/w7_critique_20260709/viewer_verify/headless_verify.json`). This strongly implicates world workload, mesh decode/chunk churn, and/or headless software WebGL—not the video clock.
- Caveat: these are headless Chromium numbers during screenshot automation, not a hardware-browser performance gate. They evidence the demo capture environment and a renderer sensitivity, not current iPhone/Mac live FPS.

### Demo-assembly bound: decisive for what the owner watched

`concat_after.txt` enumerates screenshots at `t0`, `t0.1`, ..., `t10`, each with `duration 0.1`: **102 distinct stills over 10.2 s, i.e. 10 unique frames/s**. `world_after_cleancrop.log` confirms a 10 fps, 102-frame output. The final `seg_world3d.mp4` and `dinkvision_demo_v3.mp4` are tagged 30 fps, but that only duplicates/up-converts those 10 Hz stills. The verifier itself seeks, waits 1 s, and screenshots (`scripts/racketsport/verify_process_video_viewer.py:250-263`, `329-348`).

**S1 attribution:** the specific demo symptom is assembly-bound first; mesh pose changes are data-bound second; existing headless evidence suggests renderer-bound risk as well. A live hardware-browser trace is needed to quantify the latter.

## 3. Ball rendering (S4)

The producer's rev-11 contract is honest: suppressed arc frames get `world_xyz=null` (`threed/racketsport/virtual_world.py:501-535`), and suppressed segments are also removed from dense `ball_arc_render` emission (`threed/racketsport/ball_arc_chain.py:284-347`). The dense render artifact is explicitly render-only and excluded from detection metrics (`ball_arc_chain.py:365-400`).

### What the user sees by band

| Input/effective band | Visible result |
|---|---|
| `anchored_measured` | Solid yellow trail and full yellow sphere; HUD `ball: measured` (`ballTrail.ts:185-200`). |
| `arc_interpolated` | Dashed cyan trail, smaller pulsing cyan sphere; HUD `ball: predicted` (`ballTrail.ts:236-251`). |
| `arc_extrapolated` | Dashed blue, smaller pulsing sphere; HUD predicted (`ballTrail.ts:219-234`). |
| `arc_weak` | Thin dashed amber, smallest pulsing sphere; HUD predicted + low confidence (`ballTrail.ts:202-217`). |
| `physics_predicted` / `physics_predicted_low` from world | Not preserved as an exact UI label. `samplesFromVirtualWorld` treats an unknown provenance band as `arc_interpolated` when render-only/approx/physics-fill flags exist, otherwise as measured (`ballTrail.ts:392-430`). Thus prediction can be visibly distinguished, but provenance is collapsed to generic predicted styling. |
| `hidden`, false visibility, or null world position | No sphere, opacity/width/radius zero, and a trail break on the world/arc-solved path; HUD `ball: not visible` (`ballTrail.ts:168-183`, `323-351`, `676-699`). |
| Untrusted solver status | Parser emits no solver samples and HUD explicitly says `ball: solver off — <reason>` (`ballTrail.ts:373-389`, `355-369`; `BallHonestyHud.tsx:23-29`). |

For the fail-closed demo Wolverine artifact:

- producer report: 75/300 emitted world frames; the final confidence-gated world has 71 world-position frames and 229 null/hidden frames;
- dense `ball_arc_render.json`: 398 render-only samples (124 `arc_interpolated`, 274 `arc_weak`), suppressed segments `[0,2,3,4,6,8]`;
- replaying `sampleBallArcRenderAtTime` plus the 0.12 s current-marker rule on the 300-frame grid yields a current marker on 100 frames and none on 200. Sixty-four of the 100 are client current-time interpolations. The longest render-sample gap is 6.589 s, so the viewer shows a recent fading trail and then nothing through most of that span.

There **is** predicted-position rendering, but only where the producer emitted predicted dense geometry or where two render samples bracket <=0.25 s (`ballArcRender.ts:95-127`). It is not a general tracker through the 225 suppressed frames, and it never advertises the literal `physics_predicted` provenance/band.

New ball defect: `buildBallTrail` connects consecutive renderable samples when their gap is <=1.75 s, without checking segment ID (`ballTrail.ts:270-314`). `ball_arc_render` contains only visible samples, so a suppressed or event boundary is not guaranteed to insert a hidden sentinel. This can draw a straight bridge across a hidden/untrusted interval. The supplied after artifact has three cross-segment adjacent pairs <=1.75 s (max 0.222 s); confirming whether any crosses a *suppressed* span requires a visual/segment-verdict join. Treat that as a high-risk hypothesis, not a confirmed demo artifact.

## 4. Paddle rendering (S5)

### Geometry and transforms

- Producer: `virtual_world.py` converts a non-court pose with `camera_paddle_pose_to_court_world`, or preserves a declared `court_Z0` pose after unit conversion (`virtual_world.py:1586-1604`). It then builds an extruded paddle face plus handle with fixed thickness and faces (`virtual_world.py:36-50`, `1604-1615`, `1718-1740`).
- Actual demo artifact: player 19's first paddle frame has source `wrist_palm_grip_fused:court_Z0`, `render_only=true`, `not_for_detection_metrics=true`, low-confidence `estimated_from_wrist`, 16 world vertices and 24 faces (`after_wolv/virtual_world.json`, `.paddles[0].frames[0]`). All 705 paddle frames come from the same wrist/palm/grip estimated path.
- Viewer: because the artifact has >4 vertices and >2 faces, it uses those world vertices directly; no additional coordinate transform is applied (`App.tsx:3248-3285`). The flat rounded face fan is only the missing-mesh fallback (`App.tsx:3286-3305`). It adds thick boundary cylinders plus a 0.52 m estimated normal arrow and tip rendered as an overlay (`App.tsx:2279-2338`, `3308-3349`).

### Why it looks bad / wrong

1. **Pose source — data-side, confirmed:** every pose is a render-only, low-confidence palm/grip estimate, not a true 6DoF paddle measurement. No marker/corner GT exists, so the observed positioning complaint is expected and cannot be accuracy-verified.
2. **Stale nearest-frame hold — viewer-side, confirmed:** `paddleFrameForTime` selects the nearest frame anywhere inside the full entity span and has no internal-gap threshold (`viewerData.ts:2679-2687`). Wolverine player 19 and 21 each have a 1.467 s maximum internal paddle gap; player 22 has 0.267 s. At the midpoint, the displayed paddle can be ~0.733 s stale.
3. **Geometry/presentation — viewer+producer, confirmed:** the mesh is a generic boxy face/handle with no realistic bevel/texture/appearance, and the estimated normal overlay is longer than the ~0.406 m paddle and 2.6 cm thick. That dominates the visual and reads as debug geometry.
4. **Coordinate transform — hypothesis only:** code declares and consistently consumes `court_Z0`; the viewer does not double-transform it. A transform convention error remains possible because P0-D is open, but would require paddle corner/marker GT plus camera/world reprojection to confirm.

## 5. Ranked defects and fix sketches

| Rank / symptom | Root cause and evidence | Severity | Concrete fix | Effort | Layer |
|---|---|---:|---|---|---|
| 1 — S2/S3 missing people/geometry | Owner run verified `Players=0`; Wolverine per-player neither coverage is 19.7–49.7%. Viewer fallback exists, but has no evidence to draw. | P0 product | Repair TRK/BODY schedule/coverage first; preserve a cheap per-frame track/floor proxy for every stable player ID. Add viewer invariant test: each expected on-court player has mesh, valid skeleton, or explicitly badged proxy/missing state. | multi-day data; 1 day viewer | both |
| 2 — S1 demo stutter | Demo world segments are 10 Hz still sequences upconverted to 30 fps. | High | Record the live canvas/video in real time or render deterministic frames at true 30/60 Hz; never market CFR duplication as visual FPS. Add unique-frame cadence to demo QA. | 4–8 h | demo/viewer tooling |
| 3 — S1 renderer load/chunk churn | Existing headless 4-mesh readouts 1.2–3.3 FPS vs zero-player 46.2; Canvas uses DPR 1.5–2 with antialiasing; 23 chunks load reactively, cache only two, no adjacent prefetch (`App.tsx:719-767`, `1208-1217`). | High | Hardware profile first; prefetch next/previous chunk, avoid geometry recreation, cap DPR adaptively, use on-demand invalidation when paused, and add ≥30 FPS browser gate with four meshes. | 1–3 days | viewer |
| 4 — S4 long hidden ball spans | Rev-11 correctly suppresses untrusted segments; viewer only bridges <=0.25 s and otherwise abstains. | High, honest | Improve BALL/arc anchors data-side. Viewer may show covariance-banded `physics_predicted` geometry only under a bounded horizon and must keep it render-only, dashed, and explicitly labeled. Never re-enable fail-open world positions. | multi-day | both |
| 5 — S5 stale paddle | Nearest paddle is held across unbounded internal gaps; measured max gap 1.467 s. | High | Reject/decay paddle after e.g. 2 source frames unless a bounded predictor supplies `physics_predicted`; interpolate pose with uncertainty only across short same-player gaps. | 4–8 h | viewer |
| 6 — S5 crude/low-confidence paddle | Render-only palm/grip pose plus box proxy and oversized debug normal. | High | Hide normal by default; use a lightweight beveled paddle asset aligned to pose; retain amber translucency and compact preview badge. Data lane needs marker/corner pose GT and true face pose. | 1 day viewer; multi-day data | both |
| 7 — S6 trust dead-end | Required entity badges are unmounted; pipeline artifacts omit mesh-frame `trust_badge`; missing badge defaults solid. Existing headless `trust_chip_count=0`. | High | Emit authority/provenance per frame/entity, fail-safe missing badge to preview/unknown rather than solid, mount compact badges in scene/readout, and add four-entity visual assertions. | 1–2 days | both |
| 8 — S6 seek/snap mismatch | Video keeps exact time while player, mesh, paddle, ball, and labels independently pick nearest samples with different tolerances/holds (`App.tsx:929-956`; `viewerData.ts:1095-1165`, `2097-2124`, `2679-2687`). | High | Use encoded PTS/frame-times as one canonical time map. Interpolate or explicitly hold each entity at that PTS; expose stale age. Event seek should resolve video `seeked` before committing overlay time. | 1–2 days | viewer + timebase |
| 9 — S6 absolute manifest paths | W7 manifest URLs embed `/home/arnavchokshi/...`; demo copies embed `/Users/arnavchokshi/...`. Parser uses strings directly and only body-mesh/replay-scene children get relative resolution (`viewerData.ts:621-694`, `932-935`). | High portability | Package relative URLs and resolve every manifest URL against the manifest URL; verifier must test a moved bundle and every URL. | 1 day | both |
| 10 — S6 optional asset aborts core load | One outer `try` awaits optional artifacts before committing state; any optional 404 rejects the whole load (`App.tsx:575-665`). | High | Commit manifest+video+world first; load optional artifacts independently with per-capability errors and missing badges. | 1 day | viewer |
| 11 — S6 implausible-skeleton toggle dead-end | Global plausible count controls `showSkeletons`; all-implausible time yields false even when debug implausible toggle is enabled (`viewState.ts:284-314`; `App.tsx:1241-1243`). | Medium | Make parent skeleton visibility `(plausibleCount > 0 || enabledImplausibleCount > 0)` and test all-implausible frames. | 1–2 h | viewer |
| 12 — S6 event/landing marker dead-end | Event layer default is off; active-event count gates the entire `ImpactMarkers` parent, so persistent landing spots can be suppressed outside an active contact window. | Medium | Separate persistent landing/bounce history counts from active transient markers; default on with trust badges or expose a clear empty-state reason. | 2–4 h | viewer |
| 13 — S6 trail can bridge boundaries | Trail builder ignores segment IDs and allows 1.75 s visible-sample connections. | High risk, hypothesis for demo | Break on segment-ID change unless an explicit trusted bridge record exists; insert hidden sentinels for suppressed spans; unit-test no cross-verdict line. | 2–4 h | viewer + producer |
| 14 — S6 header/count can disagree with actual ball | Scene `ballDotCount` derives from nearest visible world frame, while actual geometry prefers `ball_arc_render`; world lookup ignores internal hidden gaps (`viewState.ts:284-300`; `viewerData.ts:1103-1156`; `BallTrailLayer.tsx:41-69`). | Medium | Derive layer status/readout from the exact resolved sample source used to draw, with current-time tolerance. | 2–4 h | viewer |

## 6. Honest issues / limits

- No new browser session, live hardware trace, or image-diff run was launched; this audit uses current code, artifacts, existing headless reports, and deterministic decision-rule replay. Headless FPS is not a hardware-browser performance gate.
- The mesh/skeleton grid calculation assumes the relevant mesh chunk is instantly available. Actual async loading can only reduce mesh availability.
- The W7 owner-critique directory no longer contains the manifest/world bundle referenced by its verifier report. `viewer_verify/headless_verify.json` is retained and records `Players=0`, 46.2 FPS, and zero trust chips; the demo report separately records the BODY failure.
- Paddle misposition cannot be decomposed into centimeter/degree error without true paddle marker/corner GT. The stale-hold and proxy geometry findings are proven; coordinate-transform error is only a hypothesis.
- The historical Outdoor 5.21 -> 21.32 result was read from non-label status documents only. No protected Outdoor/Indoor labels were accessed.
- The possible cross-suppressed-span ball trail needs a direct segment-verdict/visual join before calling it present in the demo. The code permits it; the supplied after artifact has qualifying cross-segment pairs.
- No source/default changed. Best-stack delta: **none**.

## Acceptance conclusion

**PASS for the audit objective.** The render decision table covers every requested entity with source-line branches, and S1–S5 each has an evidenced viewer-layer explanation or explicit data-side attribution. This is an audit pass, not a replay capability promotion: REPLAY/STATS remains unverified and still misses the North Star's full-bundle, target-FPS, every-URL, and compact per-entity trust requirements.
