# External evidence review: visible quality beyond PB Vision

Access date for every web source in this report: **2026-07-10**. Research cutoff is the same date.

## Evidence labels

- **[FP-PRODUCT]**: first-party product page, help center, store listing, or vendor documentation.
- **[PRIMARY-RESEARCH]**: paper, proceedings page, official project page, or official technical documentation.
- **[THIRD-PARTY-VISUAL]**: an independent page reproducing or describing a product screenshot; useful for UI corroboration, not algorithm claims.
- **[BANKED-LOCAL]**: supplied repo evidence. It is not independent ground truth.
- **[INFERRED]**: a recommendation or interpretation derived from the cited evidence, not something the source directly claims.

## Executive ruling

PB Vision's visible advantage is not evidence of a continuous, full-fidelity 3D reconstruction. Its public product is centered on source video, Shot Explorer, and a filterable 3D **shot chart** containing colored complete arcs. The owner export is even more revealing: PB emits 3D on 173/252 rally frames, flags only 1/173 emitted frames as interpolated, and emits no 3D on 79/252 frames. Our banked comparison found more 2D detections but much worse emission policy. **[BANKED-LOCAL]** Sources: [RULINGS.md](../../research_ball3d_20260709/RULINGS.md), [COMPARISON.md](../w7_pbv_compare_20260709/COMPARISON.md), accessed 2026-07-10.

The fastest route to a visibly better replay is therefore:

1. never lose a tracked player merely because BODY is absent;
2. animate on a display-time clock instead of stepping at asset cadence;
3. distinguish measured, physics-predicted, 2D-only, and missing ball states at a glance;
4. use a short-horizon covariance-gated smoother for defensible gaps;
5. make BODY assets cheap enough to stay resident; and
6. render low-confidence paddles as attractive proxies, not authoritative face poses.

The current viewer already contains important pieces of this design—`track_only`/joints/mesh representations, floor markers, skeletons, a 2x-FPS option, measured/predicted ball styles, and estimated paddle materials—but several are debug controls, opt-in, nearest-frame, or dependent on incomplete assets. **[BANKED-LOCAL]** Sources: `web/replay/src/App.tsx`, `web/replay/src/viewerData.ts`, `web/replay/src/components/modules/ballTrail.ts`, accessed 2026-07-10. The adoption task is mainly to make the fallback ladder, continuous time interpolation, and compressed animation path the reliable product default.

---

## Mission 1 — Product behavior

### PB Vision: what can actually be confirmed

1. **The public 3D surface is a shot chart, not a documented continuous world replay.** PB calls it an interactive "3D Shot Chart" and says users filter serves, returns, third/fourth/fifth shots, drops, drives, and lobs. Its current site similarly describes "3D shot paths" and Shot Explorer. **[FP-PRODUCT]** Sources: [PB Vision terminology guide](https://help.pb.vision/en/help/articles/5987232-ultimate-terminology-and-definitions-guide), [PB Vision product page](https://pb.vision/), accessed 2026-07-10.

2. **Its public visual treatment is complete, color-coded trajectory arcs on a court.** PB's own help page says shot quality is shown by color and describes a 3D opponent-reach sphere. A first-party help-page image indexed by search shows multiple green/yellow/red third-shot arcs on a clean court. **[FP-PRODUCT]** Sources: [Understanding 3rd Shot Quality](https://help.pb.vision/en/help/articles/8239205-understanding-3rd-shot-quality), [Analyzing Opponent Weaknesses](https://help.pb.vision/en/help/articles/0588944-analyzing-opponent-weaknesses-with-pb-vision), accessed 2026-07-10.

3. **Public screenshots do not establish animated meshes, skeletons, avatars, or continuous per-frame player geometry.** A third-party mirror of PB's App Store screenshots shows video/Shot Explorer, court coverage, coaching cards, and colored 3D trajectory plots. Another independent event page shows video with colored player circles and a green ball line above a separate 3D shot chart. Those visuals corroborate a video-plus-analytics product, not a mesh replay. **[THIRD-PARTY-VISUAL]** Sources: [MWM PB Vision screenshots](https://mwm.ai/apps/pb-vision/6467020610), [Rally Vision event screenshot](https://www.rallyvision.co/events/ipickle-winter-classic), accessed 2026-07-10.

4. **The owner export contains player positions and JPEG avatar crops, but that does not prove those crops are rendered as animated player avatars.** The export's top level includes `avatars`; sessions map `player_avatars`; frames carry four `player_court_positions`. **[BANKED-LOCAL]** Source: `runs/research_ball3d_20260709/pbvision_cv_export/cv_export.json`, accessed 2026-07-10. **[INFERRED]** The safest interpretation is that the JPEGs support identity thumbnails/cards while court positions support dots, heatmaps, or tactical plots.

5. **PB's no-3D policy is mostly omission, not broad interpolation.** It emits 3D on 68.65% of the aligned rally, omits 31.35%, and marks only 0.58% of emitted 3D as `interpolated=true`. **[BANKED-LOCAL]** Source: [COMPARISON.md](../w7_pbv_compare_20260709/COMPARISON.md), accessed 2026-07-10. **Unconfirmed:** no public first-party material shows whether a selected shot chart draws a complete fitted arc across raw no-3D frames, shows a 2D-only marker, or visibly gaps the line during playback.

6. **PB's supported source video is 30 or 60 FPS; the web video player exposes 0.1x, 0.25x, 0.5x, 1x, 2x, and 3x speeds plus frame stepping.** **[FP-PRODUCT]** Sources: [Framing and Court Alignment Guidelines](https://help.pb.vision/articles/1108176-framing-and-court-alignment-guidelines), [Video Controls](https://help.pb.vision/articles/0262044-video-controls-bookmarks-markup-exports), accessed 2026-07-10. **Unconfirmed:** PB does not publish a 3D shot-chart render FPS; public evidence is consistent with a static interactive chart, so assigning it 30 or 60 FPS would be invented.

7. **PB acknowledges some paddle-related product work, but its public docs do not locate that paddle in the 3D viewer or describe its pose fidelity.** The current App Store version note says it "upgraded our paddle." **[FP-PRODUCT]** Source: [PB Vision App Store listing](https://apps.apple.com/us/app/pb-vision/id6467020610), accessed 2026-07-10. This is insufficient to claim a 6DoF paddle or continuous paddle rendering.

8. **PB exposes low-level CV data but warns the schema evolves.** **[FP-PRODUCT]** Sources: [Advanced Insights with Data Exports](https://help.pb.vision/en/help/articles/2010377-advanced-insights-with-data-exports), [CV schema landing page v2.1.0](https://pb-vision.github.io/schemas-cv/), accessed 2026-07-10. The public schema landing page did not expose field-level documentation in this session; the owner export remains the field-level evidence.

### Closest product analogs

| Product | Confirmed shipped/public behavior | What it implies for us |
|---|---|---|
| SwingVision | On-device video analysis tracks ball trajectory and player movement; it removes dead time, filters shots, supports video review, and publishes bounded accuracy at an ideal 60 FPS setup. It does not publicly market player meshes or racket 6DoF. **[FP-PRODUCT]** [Product page](https://swing.vision/), [tournament workflow](https://swing.vision/about/tournaments), accessed 2026-07-10. | Video-first, event-indexed review with a clean overlay is already a successful product pattern. Do not make full BODY the only way to keep players visible. **[INFERRED]** |
| PlaySight | Multi-angle 1080p/60 FPS video, instant replay/VAR, slow motion, highlights, analytics, and player-development tools. **[FP-PRODUCT]** [Tennis SmartCourt](https://playsight.com/our-sports/tennis/), accessed 2026-07-10. | Smooth video and fast seeking are the base layer; analytics enhance it. A low-FPS 3D world should never drag the source video UI down. **[INFERRED]** |
| Zenniz | Four cameras plus 30 microphones, automated line calls, shot maps, analytics, and video replay; audio triangulation supplies landing/strike points and cameras verify decisions/player movement. **[FP-PRODUCT]** [How Zenniz works](https://zenniz.com/smart-corner/how-does-zenniz-work), [product page](https://zenniz.com/), accessed 2026-07-10. | Even a hardware-rich system uses multimodal event timing and video replay, not a body-mesh requirement. Our single-mic audio should be a gated cue, not authority. **[INFERRED]** |

### Product conclusion

**[INFERRED]** To beat PB visibly, do not imitate a presumed continuous reconstruction that the evidence does not show. Beat its actual public product on three axes: a continuous four-player world proxy, an honest animated ball with clear provenance, and instant synchronization with source video. PB's shot-chart aggregation can look polished while omitting weak raw spans; our differentiator can be continuous context without pretending prediction is measurement.

Mission 1 source count: **16 distinct URLs** (target >=3).

---

## Mission 2 — Ball visibility and prediction SOTA

### What changed by 2026

1. **TT3D keeps a physics fit but searches bounce state by minimizing reprojection error.** This is the closest structural match to our SciPy TRF+Huber arc solver. **[PRIMARY-RESEARCH]** Source: [TT3D](https://arxiv.org/abs/2504.10035), accessed 2026-07-10.

2. **TT4D reverses segment-then-lift: it lifts the entire unsegmented 2D track first, then segments in 3D, and reports robustness to unreliable detections and high occlusion.** It is also a large dataset/reconstruction pipeline, not a drop-in filter. **[PRIMARY-RESEARCH]** Source: [TT4D](https://arxiv.org/abs/2605.01234), accessed 2026-07-10.

3. **Kienzle et al. make learned uplifting product-like by separating a real-2D-supervised front end from a synthetic-physics-trained back end.** Exact timestamps are encoded with RoPE; frame rates are randomized in training; missing detections are part of the design. **[PRIMARY-RESEARCH]** Source: [Uplifting Table Tennis, WACV 2026](https://arxiv.org/abs/2511.20250), accessed 2026-07-10.

4. **Where Is The Ball uses a camera-independent canonical representation and synthetic-only training, and evaluates on synthetic plus real datasets.** **[PRIMARY-RESEARCH]** Source: [CVPR 2025 workshop paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Ponglertnapakorn_Where_Is_The_Ball_3D_Ball_Trajectory_Estimation_From_2D_CVPRW_2025_paper.html), accessed 2026-07-10.

5. **More expressive physics is not automatically better under monocular noise.** A 2026 benchmark of seven physics models over nearly 6,000 soccer segments found end-to-end monocular and oracle-3D model rankings reverse; it attributes the limit primarily to observation noise and single-view ambiguity. **[PRIMARY-RESEARCH]** Source: [CVPR 2026 multi-model benchmark](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Grad_Physics-Based_3D_Ball_Trajectory_Reconstruction_from_Monocular_Soccer_Video_A_CVPRW_2026_paper.html), accessed 2026-07-10.

6. **A UKF/EKF is a credible online refinement/fallback, not proof of an entire missing arc.** Prior monocular work compares a real-time UKF with nonlinear maximum likelihood and uses ball image radius as a depth cue; another real-world pipeline uses maximum likelihood followed by EKF refinement. **[PRIMARY-RESEARCH]** Sources: [DFKI UKF/ML study](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/3840), [RoboCup monocular MLE+EKF paper](https://mediawiki.isr.tecnico.ulisboa.pt/images/7/7d/3DParabolaRobocup2011.pdf), accessed 2026-07-10.

7. **Blur is signal, but the paper's claim is detector and prediction improvement—not guaranteed sub-frame contact labels from our existing heatmaps.** BlurBall centers labels on blur streaks and explicitly predicts blur attributes. **[PRIMARY-RESEARCH]** Source: [BlurBall](https://arxiv.org/abs/2509.18387), accessed 2026-07-10.

8. **Racket evidence helps trajectory prediction only when fused selectively.** RacketVision reports naive concatenation degrades trajectory prediction while cross-attention improves over ball-only baselines. **[PRIMARY-RESEARCH]** Source: [AAAI 2026 RacketVision](https://ojs.aaai.org/index.php/AAAI/article/view/37362), accessed 2026-07-10.

### Strongest compatible 2026 recipe

**[INFERRED]** The following preserves the shipped fail-closed policy and the current optimizer:

1. **Primary path:** keep accepted per-segment TRF+Huber reprojection fits; do not replace good arcs with a filter.
2. **Observation model:** feed 2D center, detector confidence, heatmap/blur footprint, and apparent radius into per-frame measurement covariance. Radius is a weak depth cue, never a direct depth measurement.
3. **Anchor search:** TT3D-style joint search over bounce/contact time and state inside the existing robust fit. Retain all hypotheses and their costs.
4. **Short-gap path:** run a forward UKF and backward Rauch–Tung–Striebel smoother inside one physically valid segment, seeded from accepted fit state. Use exact PTS. The smoother may bridge only gaps bracketed by accepted evidence or a tightly capped one-sided horizon.
5. **Hard gates:** reject court/net violations, impossible speed/height, covariance growth above the render threshold, and cross-contact smoothing. Long or unbracketed gaps remain missing.
6. **Output contract per sample:** `measured`, `model_estimated`, `physics_predicted`, or `missing`, plus covariance, supporting observations, solver status, and horizon age.
7. **Visual contract:** measured ball = solid yellow core; physics-predicted = cyan dashed trail plus translucent/pulsing ball and uncertainty halo; 2D-only = court-plane ring and video overlay; missing = no 3D ball. Never draw a measured-looking sphere through a rejected segment.
8. **Promotion:** score every candidate on the frozen ball and 3D gates. Learned lifting becomes a challenger only after TT3D/short-gap work hits its kill criteria.

### Why not jump directly to TT4D/Kienzle

**[INFERRED]** They are the strongest research direction for severe occlusion, but they require a trained back end and an in-domain evaluation campaign. The 2026 multi-model benchmark warns that monocular noise/ambiguity can dominate model expressiveness. Our banked clip already has 80.6% 2D coverage; the immediate loss is turning weak segments into hidden/absurd 3D, not absence of all 2D signal. Therefore learned lift is the correct contingency, not the first visible-quality patch.

Mission 2 source count: **9 distinct URLs** (target >=3).

---

## Mission 3 — Player fallback UX and replay FPS

### Player representation

1. **High-end sports virtual replay is driven from skeletal data and then converted to 3D animation.** Sony/Hawk-Eye's SkeleTRACK tracks 29 key skeletal points, balls, and other objects; HawkVISION/Beyond Sports use that data for virtual recreations and arbitrary viewpoints. **[FP-PRODUCT]** Sources: [Sony SkeleTRACK interview](https://www.sony.com/en/SonyInfo/technology/stories/entries/20240411/hawkeye/), [Sony visualization demo](https://www.sony.com/en/SonyInfo/technology/activities/STEF2022/exhibition_0301/), accessed 2026-07-10.

2. **glTF already supports the right hierarchy: skins deform a reusable mesh from skeleton pose; animations interpolate node transforms between keyframes.** **[PRIMARY-RESEARCH/TECH-DOC]** Sources: [Khronos glTF structure](https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_002_BasicGltfStructure.html), [glTF animation interpolation](https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_007_Animations.html), accessed 2026-07-10.

3. **Three.js provides distance-based LOD and direct loading of Draco, meshopt, GPU instancing, KTX2/Basis, and quantized glTF assets.** **[PRIMARY-RESEARCH/TECH-DOC]** Sources: [Three.js LOD](https://threejs.org/docs/pages/LOD.html), [Three.js GLTFLoader](https://threejs.org/docs/pages/GLTFLoader.html), accessed 2026-07-10.

**[INFERRED] Product fallback ladder:**

- `track_only`: always render an identity-colored court footprint plus a 1.7 m capsule/billboard when TRK is valid;
- `joints`: replace/cross-fade the capsule with an articulated skeleton when plausible joints exist;
- `mesh`: skin one reusable, low-poly neutral avatar from joints; cross-fade to a higher-detail mesh only where BODY supports it;
- `missing`: after a very short hold (one or two source frames), fade the entity and show a timeline gap—never silently retain a player through a long TRK gap.

This gives guaranteed representation while preserving provenance. The stable court footprint carries identity and placement; articulation/detail is an LOD upgrade. It also prevents S2 (missing people) from being caused by S3 (missing BODY).

### Current-repo compatibility

The current viewer already parses `representation: "track_only" | "joints" | "mesh"`, can render floor markers, skeleton graphs, and solid meshes, and has trust-band materials. It currently chooses nearest frames and hides floor markers unless enabled; the 2x-FPS transform is optional and inserts midpoints rather than interpolating at every display tick. **[BANKED-LOCAL]** Sources: `web/replay/src/viewerData.ts:474`, `web/replay/src/App.tsx:2066`, `web/replay/src/viewerData.ts:2398`, accessed 2026-07-10.

**[INFERRED]** Therefore the surgical adoption is:

- make the track footprint/capsule a non-debug product fallback;
- render skeleton and skinned avatar from bracketing PTS at the browser clock time;
- cross-fade LODs without making the entity disappear;
- keep implausible skeletons hidden, but fall back to the capsule rather than nothing.

### Replay FPS and web performance

1. **`requestAnimationFrame` follows display refresh (commonly 60 Hz), while `requestVideoFrameCallback` follows the lower of video FPS and display refresh and exposes frame metadata.** **[PRIMARY-RESEARCH/TECH-DOC]** Sources: [MDN requestAnimationFrame](https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame), [MDN requestVideoFrameCallback](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback), accessed 2026-07-10.

2. **A typical 60 Hz frame budget is 16.66 ms, with about 10 ms available for application work; variable/janky frame pacing is perceptible.** **[PRIMARY-RESEARCH/TECH-DOC]** Source: [web.dev rendering performance](https://web.dev/articles/rendering-performance), accessed 2026-07-10.

3. **Per-object meshes create draw-call overhead; merging/instancing reduces it.** **[PRIMARY-RESEARCH/TECH-DOC]** Source: [Three.js optimize lots of objects](https://threejs.org/manual/en/optimize-lots-of-objects.html), accessed 2026-07-10.

4. **`gltfpack` can quantize, merge meshes, resample/quantize animations, prune nodes, compress with meshopt, and compress textures.** **[PRIMARY-RESEARCH/TECH-DOC]** Source: [meshoptimizer gltfpack documentation](https://github.com/zeux/meshoptimizer/blob/master/gltf/README.md), accessed 2026-07-10.

**[INFERRED] Recommended playback architecture:**

- source video remains the master timeline;
- `requestVideoFrameCallback` updates the exact media PTS/frame identity;
- `requestAnimationFrame` renders the 3D scene at display cadence;
- every render tick samples bracketing BODY/TRK/BALL/RKT states by PTS, linearly interpolates translation, uses quaternion slerp for rotation, and refuses interpolation across entity/trust/contact discontinuities;
- use one skinned avatar asset per identity, not a new full `BufferGeometry` every frame;
- preload a sliding animation window, dispose old GPU buffers, and select capsule/skeleton/avatar LOD by coverage and device budget;
- ship meshopt-compressed GLB plus KTX2/Basis textures and a texture-free low-LOD fallback.

The target should be **display-cadence 60 FPS on a representative iPhone/Safari and laptop**, independent of 30 FPS video or sparse BODY sampling. Source-frame accuracy is preserved because the animation is time-based; interpolated display frames are presentation, not new measurements.

Mission 3 source count: **11 distinct URLs** (target >=3).

---

## Mission 4 — Paddle/racket presentation

### What products and research support

1. **Public product pages for PB Vision, SwingVision, PlaySight, and Zenniz emphasize ball trajectories, player movement/position, video, line calls, and shot maps—not authoritative continuous racket 6DoF.** PB's App Store has a vague "upgraded our paddle" note, but no public pose contract. **[FP-PRODUCT]** Sources: [PB Vision](https://pb.vision/), [PB App Store](https://apps.apple.com/us/app/pb-vision/id6467020610), [SwingVision](https://swing.vision/), [PlaySight Tennis](https://playsight.com/our-sports/tennis/), [Zenniz](https://zenniz.com/), accessed 2026-07-10.

2. **RacketVision defines a five-keypoint racket representation: top, bottom, handle, left, right.** Its multi-sport RTMPose baseline reaches high PCK for top/bottom/handle but much lower side-edge PCK (roughly 65–80%), explicitly attributing the side difficulty to grip occlusion, motion blur, and viewpoint. **[PRIMARY-RESEARCH]** Source: [AAAI 2026 paper PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324), accessed 2026-07-10.

3. **Planar IPPE returns two pose solutions; selecting one silently is mathematically unjustified without extra evidence.** **[PRIMARY-RESEARCH/TECH-DOC]** Sources: [OpenCV PnP documentation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html), [OpenCV calib3d solution counts](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html), accessed 2026-07-10.

4. **Pose ambiguity is image-specific, not just a global object-symmetry issue; evaluating a distribution can change method rankings.** **[PRIMARY-RESEARCH]** Source: [BOP-Distrib, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Meden_BOP-Distrib_Revisiting_6D_Pose_Estimation_Benchmarks_for_Better_Evaluation_under_WACV_2026_paper.html), accessed 2026-07-10.

### Honest treatment that still looks good

**[INFERRED] Recommended render states:**

| Evidence state | Render treatment | Rule |
|---|---|---|
| Wrist/hand only | Short grip stub or small rounded paddle silhouette, snapped to the chosen hand, 25–35% opacity, no face normal | Label `estimated_preview`; this is a render constraint, not a measured paddle center. Hide if wrist confidence or hand identity fails. |
| Partial 2D paddle / ambiguous IPPE | Simplified rounded face proxy at 35–55% opacity; show the alternate plane as a fainter ghost or show an uncertainty fan; suppress the normal arrow | Preserve both IPPE solutions in data. Use temporal, wrist, and ball-contact evidence only to rank, not erase, the alternate. |
| Stable five-keypoint + pose | Solid proxy with a subtle rim; normal appears only in selected paddle-review mode | Require stable face-angle/reprojection/temporal gates over a window, not one frame. |
| Missing | No paddle | Do not freeze a paddle through a swing/contact gap. |

Snap-to-hand should be bounded: translate the grip to the wrist only within an anatomically plausible distance, never stretch or rotate the face to force contact. At a predicted contact, render a translucent contact disc/halo sized by uncertainty rather than an exact spark. Ghosting should encode ambiguity, not add decorative motion blur.

The current viewer already builds an estimated rounded paddle face, supports trust-dependent translucent material, exposes a face-normal review segment, and counts ambiguous paddle frames. **[BANKED-LOCAL]** Sources: `web/replay/src/App.tsx:2247`, `web/replay/src/App.tsx:3248`, `web/replay/src/viewerData.ts:616`, accessed 2026-07-10. The adoption is to make hand-only/ambiguous/full-pose states explicit and to remove the confident normal from normal playback.

Mission 4 source count: **9 distinct URLs** (target >=3).

---

## Mission 5 — Ranked top 10 adoptions

Ranks optimize **visible owner impact first**, then model accuracy. Effort is relative to this repo. Every kill criterion is a stop rule, not a promise.

| Rank | What to adopt | Evidence | Expected impact | Effort | Main risk | Kill criterion |
|---:|---|---|---|---|---|---|
| 1 | Make the player representation ladder product-default: TRK footprint/capsule → plausible skeleton → skinned/solid mesh; cross-fade without disappearance. | [Sony SkeleTRACK](https://www.sony.com/en/SonyInfo/technology/stories/entries/20240411/hawkeye/), [glTF skins](https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_002_BasicGltfStructure.html), accessed 2026-07-10. | **S2 very high; S3 very high.** Four people remain intelligible even when BODY drops. | Low–medium; primitives exist. | A proxy can preserve the wrong ID or stale position. | Stop/rollback if valid-TRK player-visible coverage is <99%, any proxy crosses identities, or >150 ms holds create visible teleport/freeze artifacts. |
| 2 | Use source video PTS as master; render at display cadence with rAF; use rVFC for media-frame sync; interpolate bracketing transforms every tick, not just one midpoint. | [MDN rAF](https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame), [MDN rVFC](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback), [glTF interpolation](https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_007_Animations.html), accessed 2026-07-10. | **S1 very high; S3 high; S5 medium.** Removes stepwise motion without inventing measurements. | Medium. | Cross-contact interpolation smears limbs/paddles; clock drift. | Kill if representative iPhone Safari p95 FPS <45, A/V/3D drift >33 ms, or interpolation crosses a contact/entity gap. |
| 3 | Make compressed animated GLB/skinned-avatar windows the normal BODY path; meshopt/KTX2; reuse geometry; add LOD and deterministic disposal. | [Three GLTFLoader](https://threejs.org/docs/pages/GLTFLoader.html), [meshoptimizer gltfpack](https://github.com/zeux/meshoptimizer/blob/master/gltf/README.md), [Three LOD](https://threejs.org/docs/pages/LOD.html), accessed 2026-07-10. | **S1 high; S3 high.** More BODY can stay resident, with fewer decode/upload stalls. | Medium–high. | Compression/quantization can damage joints or create memory leaks. | Kill if load-to-first-frame and peak GPU memory do not each improve >=30%, or audited joint/mesh error exceeds current tolerances. |
| 4 | Lock the four-state ball visual contract as default: measured solid yellow; predicted cyan dashed/pulsing + halo; 2D-only court ring/video mark; rejected/missing absent. | [PB data exports](https://help.pb.vision/en/help/articles/2010377-advanced-insights-with-data-exports), [uncertainty visualization study](https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.100/1-hci/hci-paper/2023/IMWUT_2023_Uncertainty_Trajectory_final.pdf), accessed 2026-07-10; banked PB omission/interpolation counts. | **S4 very high; S6 high.** Restores visibility without mislabeling prediction. | Low; most primitives exist. | Users may read any visible ball as measured. | Kill a style if blinded users identify provenance correctly <90% or if a rejected 3D sample can render as a solid ball. |
| 5 | Add covariance-gated UKF + backward smoother for short, same-segment gaps, seeded from accepted TRF fits; cap horizon and fail closed beyond it. | [DFKI UKF/ML](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/3840), [RoboCup MLE+EKF](https://mediawiki.isr.tecnico.ulisboa.pt/images/7/7d/3DParabolaRobocup2011.pdf), accessed 2026-07-10. | **S4 high.** Converts defensible short holes into visibly predicted ball while leaving long gaps honest. | Medium. | Overconfident covariance or smoothing through contact. | Kill if frozen-gate landing/contact errors regress, >1% predicted samples violate court/height/speed bounds, or any smoother crosses a contact boundary. |
| 6 | Replace one-size-fits-all paddle with explicit hand-only, ambiguous two-pose, and stable-pose render states; snap only the grip within a tight bound. | [RacketVision PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324), [OpenCV IPPE](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html), [BOP-Distrib](https://openaccess.thecvf.com/content/WACV2026/html/Meden_BOP-Distrib_Revisiting_6D_Pose_Estimation_Benchmarks_for_Better_Evaluation_under_WACV_2026_paper.html), accessed 2026-07-10. | **S5 very high; S3 medium.** A low-confidence paddle looks intentional rather than broken. | Medium. | Attractive proxy may imply accuracy; hand snapping can pick the wrong hand. | Kill any state whose blinded trust interpretation is <90%, whose grip-wrist error exceeds 8 cm, or whose stable-pose GT gate fails. |
| 7 | TT3D-style joint anchor search inside the current TRF+Huber fit. | [TT3D](https://arxiv.org/abs/2504.10035), accessed 2026-07-10. | **S4 high.** Monetizes our higher 2D coverage and attacks anchor starvation. | Medium. | More local minima and compute. | Existing banked kill remains binding: diagnosed fallback must fall from 9/11 to <5/11 with frozen metrics non-regressing. |
| 8 | Run the cheap cue pack: apparent ball radius/heatmap footprint as weak depth covariance; BlurBall-style blur attributes; visual-gated audio contact search. | [BlurBall](https://arxiv.org/abs/2509.18387), [audio-visual tennis hits](https://openresearch.surrey.ac.uk/esploro/outputs/conferencePresentation/Improved-Detection-of-Ball-Hit-Events/99512922402346), [UKF radius cue](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/3840), accessed 2026-07-10. | **S4 medium–high.** Better anchor timing/depth at low architecture cost. | Low–medium. | Compression blur and exposure changes make footprint/radius biased. | Kill each cue independently if preregistered contact/landing/inlier metrics do not improve or any protected split regresses. |
| 9 | Harden the solver with both-end pinning where evidence exists plus a dedicated RANSAC/inlier pass separate from segmentation. | [MonoTrack](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf), accessed 2026-07-10. | **S4 medium.** Reduces drift/local minima and stops segmentation from also owning outlier rejection. | Medium. | False endpoint constraints can lock in a wrong arc. | Kill if accepted-segment reprojection/landing error does not improve or rejection/coverage worsens enough to offset the gain. |
| 10 | Add whole-rally DP segmentation only after joint-anchor kill results; if physics path then stalls, open a separately gated TT4D/Kienzle learned-lift challenger. | [TT3D](https://arxiv.org/abs/2504.10035), [TT4D](https://arxiv.org/abs/2605.01234), [Kienzle WACV 2026](https://arxiv.org/abs/2511.20250), accessed 2026-07-10. | **S4 medium–high, later.** Handles global boundary ambiguity and severe occlusion. | High. | Larger architecture/data surface; synthetic-to-real gap. | DP dies if it does not recover better reviewed boundaries. Learned lift dies if same frozen GT cannot beat the best physics stack without worse hallucination/coverage calibration. |

Mission 5 source count: **21 distinct URLs across the section** (target >=3). Ranked list completeness: **10/10 rows include what, evidence, impact, effort, risk, and kill criterion**.

### Reconciliation with the banked adopt sequence

**AGREES inside BALL:**

- fail-closed is first and remains binding;
- UKF/smoother is the first numerical gap filler;
- TT3D anchor search precedes larger segmentation or learned changes;
- BlurBall/audio are cheap, separately killable cues;
- pinning/inlier separation follows the center-piece search;
- whole-rally DP is deferred until the anchor-search kill read.

**DISAGREES only in global product priority:**

- Player presence and display-time playback rank above new ball solving because they directly address S1–S3 and are mostly existing-viewer integration work.
- The ball visual contract ranks above UKF because it is already largely implemented and immediately fixes honesty/visibility semantics.
- Mesh compression/animation ranks above TT3D because faster persistent BODY changes every frame the owner sees, while TT3D changes only ball spans.
- Apparent ball radius is folded into the cheap cue pack; PB's export and classic monocular filtering both support it as a cue, but it must remain covariance-level evidence.
- Magnus/spin is **not** promoted. The 2026 multi-model benchmark says observation noise and monocular ambiguity can dominate model expressiveness; spin should wait for independent 3D flight/contact GT.
- Learned lift is not a top-ten immediate implementation by itself; it is the explicit contingency behind #10 after the current physics sequence reaches its stop rules.

---

## Honest unknowns and limits

1. PB's exact frame-by-frame visual behavior during its 79/252 no-3D frames could not be confirmed. Public material shows filtered 3D shot charts, not the continuous per-frame render state.
2. PB's 3D chart render FPS is not published. Source support (30/60 FPS) and video speed controls do not answer that question.
3. PB's App Store "upgraded our paddle" note does not establish a 3D paddle, pose method, or confidence treatment.
4. The in-app browser had no available browser instance, so public demos could not be interactively inspected; conclusions use indexed first-party pages/screenshots and the supplied export.
5. Product absence claims are intentionally narrow: public pages do not document PB/SwingVision/PlaySight/Zenniz player meshes or racket 6DoF; that does not prove an unreleased or private feature cannot exist.
6. Table-tennis learned lifting and blur results are not pickleball validation. They justify challengers and design patterns, not promotion.
7. UKF/RTS, TT3D, radius, blur, audio, pinning, and DP must all be scored on the same frozen pickleball gate. Visual smoothness is not accuracy evidence.
8. PB remains a comparator, not ground truth; the shared landing comparison is pseudo-GT and the systems disagree substantially in planar 3D on the one comparable normal-fit segment.

## Bottom line

The strongest visible-quality strategy is a **continuous proxy world around an honest evidence core**: video stays authoritative and smooth; every valid TRK identity remains represented; BODY and paddle detail appear only when supported; measured and predicted ball are unmistakable; short gaps are smoothed with covariance and hard horizons; long gaps remain gaps. That can look better than PB's static/filterable trajectory analytics while making fewer claims than the evidence supports.
