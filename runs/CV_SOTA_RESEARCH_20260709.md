# CV SOTA Research Review and Experiment Register

Date: 2026-07-09  
Repository snapshot: `main` at `89400a0a0`, with a shared dirty worktree  
Decision status: `review-only`; no model is promoted and `VERIFIED=0` remains binding

## Direct answer

Yes. This is a new independent research pass, not a restatement of the existing
roadmap. Three specialist lanes reviewed BALL/RKT/contact, TRK/BODY, and
CAL/mobile/fusion/latency against the live repository and primary sources. A
fourth lane adversarially ranked the results. The earlier deep review at
`runs/CV_PIPELINE_DEEP_REVIEW_20260709.md` had already compared a smaller set of
papers; this pass checked current releases, weights, licenses, resource costs,
repo integration seams, and exact experiments.

The result is deliberately not a model shopping list. A candidate enters the
future plan only when it has:

1. a current repo failure it could plausibly fix;
2. a prerequisite and frozen baseline;
3. an identical independent-data scorer;
4. a cost/runtime measurement;
5. a commercial-license checkpoint; and
6. an explicit kill rule.

Published numbers establish that an experiment is worth running. They do not
establish pickleball accuracy or product readiness.

## Executive decisions

The research changes the emphasis in seven ways:

1. **Repair discarded evidence before training more models.** The main event
   path extracts audio but calls fusion with `require_audio=False`; fresh BODY
   wrists arrive after the event/mesh schedule; valid contact artifacts can be
   reused without upstream dependency identity; the default ball arc chain does
   not consume the implemented apparent-diameter observation; and the paddle
   solver discards the second valid planar pose.
2. **Preserve source pixels for tiny objects.** The current 512x288 BALL path is
   likely resolution-limited. A high-resolution/court-crop detector experiment
   now precedes another broad fusion or architecture sweep.
3. **Make one person-authority artifact feed the whole pipeline.** Stable ID,
   box, true mask, cheap joints, court footpoint, visibility, appearance, and
   confidence should be produced once and reused by TRK, BODY, paddle, camera
   exclusion, event selection, fusion, and replay.
4. **Treat ambiguity as data.** Keep top-K ball observations, both IPPE paddle
   poses, uncertain identities, raw/corrected timing, and covariance until
   cross-lane evidence can resolve them. Do not collapse them early and then
   pretend the chosen answer was measured.
5. **Use temporal models as challengers, not assumed upgrades.** GEM-X is the
   strongest newly released BODY challenger; RF-DETR segmentation plus mask-
   aware/multi-cue association is the strongest TRK direction. Both wait for
   frozen pickleball GT and keep the current defaults until they pass.
6. **Separate metric reconstruction from visual richness.** The MHR/court/ball/
   paddle world remains measurement authority. A MoVieS-style appearance layer
   may later improve viewing, but cannot supply stats, geometry, or unseen body
   detail from one wide camera.
7. **Optimize the integrated product, not isolated FPS.** Decode once,
   persistent workers, stable compilation buckets, batched crops, event-aware
   deep inference, and stage overlap come after correctness, with metric parity.

## Fresh code findings that change the order

| Severity | Finding | Live evidence | Required correction |
|---|---|---|---|
| P0 | Upload is wired but the uploaded run cannot open its own server replay | App sends the exact video/sidecar and refreshes `/api/clips`, but does not resolve `jobId -> ready manifest -> matching replay`; `openReplay` opens the local row | Persist run identity through ready manifest and route that capture to its own server replay |
| P0 | Multipart restart safety is overstated | `upload_state.json` keeps counters/clip ID, but upload target, upload ID, signed part URLs, and ETags live only in the queue actor; the existing relaunch test restarts from a merely queued state | Test process death after a completed part and after video completion/sidecar failure; resume the same server attempt or explicitly abort it without orphaning/duplicating a clip |
| P0 | Capture truth can silently truncate or drift from the movie | CoreMotion/PTS samples cap at 4,800 (about 80 s at 60 fps); the optional BGRA tap may discard late frames without `didDrop`; ARKit setup returns the latest frame rather than a requested PTS; movie intrinsics are FOV-derived | Stream samples, enumerate encoded PTS, record every match/drop reason, deliver native intrinsics with reference crop/orientation, and characterize sensor clocks/rolling shutter |
| P0 | Audio is not a timing input on the normal event path | `process_video.py` passes `require_audio=False`; `event_fusion.py` then matches ball and wrist only | Preserve raw/mux/propagation-corrected audio times and prove that classified audio changes source-disjoint contact error |
| P0 | Fresh BODY cannot improve the schedule that selected BODY frames | EVENTS precedes BODY and emits blocked wrists when no current skeleton exists; no mandatory post-BODY event pass exists | Coarse event windows -> deep BODY/RKT/high-res BALL -> refined events/arcs -> one schedule refinement |
| P1 | Contact reuse is not dependency-safe | A schema-valid `contact_windows.json` can be reused without BALL/BODY/audio/CAL/time hashes | Add content-addressed dependencies and rebuild the dependent closure |
| P1 | Paddle-planar ambiguity is thrown away | `racket6dof.py` evaluates IPPE candidates but returns only the lowest-reprojection pose | Persist both `R,t` hypotheses, ambiguity margin, and provenance; resolve with hand, time, ball side/impulse, and surface factors |
| P1 | A repaired paddle pose keeps observation confidence | Motion-clamped output retains `raw.pose.confidence` | Mark corrected provenance and reduce/recompute confidence from correction magnitude |
| P1 | Known ball size is not reused by the default arc chain | Apparent-diameter/depth support exists, but `ball_arc_chain.py` has no size observation input | Add diameter and uncertainty as a robust optional depth factor; promote only on 3D GT |
| P1 | TRK evaluation/provenance can drift | Different current reports quote materially different clip scores; selected OSNet path is absent locally | Pin scorer commit, label manifest, thresholds, detections, embedding identity, and source groups before the next comparison |
| P1 | Global person association is greedy | Code inference: fragment edges are built once and unioned without recomputing cluster compatibility | Add invariants and compare a globally constrained four-person/multi-cue solution on fixed detections |

These are pipeline correctness issues. They belong before model promotion even
if an external checkpoint looks better in a demo.

## Ordered execution program

### 0. Finish the real product route

Complete NS-01: one capture/sidecar/time/coordinate contract, content-addressed
reuse, restart-safe upload, honest partial status, recursive packaging, and a
physical record/import to own replay trace. The live dirty iOS checkout sends
video+sidecar and refreshes clip status, but ready-job to manifest to the
matching replay is still unwired. Swift package tests are scoped code proof, not
a physical/backend proof.

### 1. Freeze evaluation and collect independent truth

Before inference:

- freeze source-game groups, uniformly random audit strata, scorer/version,
  model/config identity, and a one-shot promotion ledger;
- score all 1,750 prepared BALL rows on one card rather than comparing the
  existing 1,121-row candidate card with the older anchor;
- capture the product phone plus two high-FPS GT phones, surveyed court/net,
  ChArUco, LED/audio synchronization, paddle markers, and scripted occlusions;
- label CAL points, player ID/boxes/masks, 3D joints/soles, ball centers/3D/events,
  paddle corners/normal/contact, reviewer, uncertainty, and raw source identity;
- report random, hard/occluded, seen-source, and unseen-source strata separately.

### 2. Repair multimodal reuse and uncertainty contracts

Do this before a learned event or 3D candidate:

- make audio an actual event measurement, with mux and sound-propagation
  correction kept separate from raw time;
- add a mandatory refined pass after BODY/RKT;
- dependency-hash contacts, arcs, mesh schedules, and downstream outputs;
- pass apparent ball diameter/blur with uncertainty into fusion;
- retain top-K BALL/RKT/identity hypotheses rather than one early answer;
- produce one provenance-rich person authority per frame.

### 3. Run lane experiments in the order below

Every row starts with the current default, uses the same frozen scorer, saves
runtime/VRAM/all-in wall time, and ends `adopt`, `reject`, `partial`, or
`no-attempt`. No row updates `best_stack.json` on setup or visual evidence.

### 4. Integrate only winners through a robust two-pass world model

The refined factor graph should include camera/time offset, player root/pose,
ball states, both paddle hypotheses, contacts, and measurement switches. Inputs
include top-K detector heatmaps, visibility, blur vector, apparent size, court,
wrists/hands, paddle keys/mask, classified audio, and court/net/paddle surfaces.
Raw observations remain immutable; leave-one-modality-out and fixed-anchor
ablations must show where each gain came from.

### 5. Add appearance, coaching, and performance only after metric gates

Generate deterministic facts and metric replay first. An optional appearance
layer is presentation-only. Then profile cold/warm integrated runs and apply one
performance lever at a time with full metric parity.

## Experiment register

### BALL, 3D trajectory, events, and paddle

| Rank | Hypothesis and candidate | Prerequisite and exact test | Gate and kill rule | Posture |
|---:|---|---|---|---|
| 1 | More source resolution fixes the smallest-ball/blur tail better than raw detector voting. Use a cleared high-resolution three-frame heatmap model; Uplifting Table Tennis's SegFormer++ result is the research design. | Frozen 1,750 card. Compare current 512x288, full-frame high-res, court crop, and coarse-to-high-res crop. Add `<4 px`, blur, overlap, hidden, and out-of-frame strata. | Existing BALL gate plus p95/p99/teleports. Kill if F1/tails do not improve or hFP exceeds 0.05. Measure batch-1 wall time. | High priority research. Official Uplifting code is GPL-3.0; do not copy it into a commercial default without review. |
| 2 | TOTNet's five-frame occlusion supervision helps genuine gaps. | Existing repo adapter/official weights. Score no-flow, then flow, then fine-tuned no-flow on the identical card with explicit visible/occluded/out-of-frame labels. | Kill if an occluded subset improves while global F1, hFP, or tails regress. | Runnable challenger; MIT code, dataset terms still need review. |
| 3 | RacketVision released weights/data provide a better BALL/RKT initialization. | Score released BALL weights zero-shot; after paddle GT, score 5-keypoint RKT zero-shot then pickleball fine-tune. | BALL gate unchanged. RKT promotion remains face-angle p90 <=5 degrees and contact-point p90 <=3 cm. | Reclassified from watch to runnable: code, weights, and MIT-labelled dataset were released in April 2026. Source-video provenance still needs review. |
| 4 | Blur and apparent diameter are useful observations even when their detector is not. | Adopt center + blur endpoints/orientation/length + visibility labels. Fit diameter on high-res unblurred crops and compare arc solver with/without robust size factor on 3D GT. | Kill size factor if 3D/landing p90 does not improve. Blur is a weak velocity cue, never detection truth. | Reuse measured evidence; current BlurBall zero-shot hFP prevents detector promotion. |
| 5 | Simpler physics beats extra parameters under noisy monocular observations. | On gold 3D, compare parabola, gravity+drag, drag+nuisance fitted gravity, and current solver under the same calibration perturbations. | Kill every parameter that lowers reprojection residual without lowering 3D/landing/contact error. Never call fitted gravity physical gravity or spin. | High-value ablation; reproduce design, not noncommercial code. |
| 6 | Keeping both planar paddle solutions lets cross-lane cues resolve small/edge-on ambiguity. | True corners/normal/contact GT; RacketVision keys -> both IPPE solutions -> hand/time/ball/surface disambiguation. Compare GigaPose only on event crops after this baseline. | RKT gate, coverage, runtime, and no BALL/BODY regression. Repaired/hypothesis output must be labelled. | Immediate contract fix; model experiment after GT. |
| 7 | Classified audio and a refined pass improve contact timing. | Label paddle hit, bounce, net, stomp, and other; preserve raw/corrected times. Run coarse -> deep -> refined. Consider AdaSpot only after enough labels. | Source-disjoint contact p90 <=40 ms. Kill learned ROI path if actor/contact ROI recall <99% or timing does not improve at equal compute. | Architecture fix first, learned model later. |
| Watch | Lift the full rally before time segmentation, as in Uplifting/TT4D. | Wait for 3D GT and a reproducible implementation; compare full-rally lift-first with current event-first segmentation. | Must improve 3D position, landing, event timing, coverage, and calibration sensitivity. No spin claim without spin GT. | TT4D has a compelling May 2026 design but no official code/weights/dataset were found; do not schedule a rewrite yet. |

Why this order: current raw fusion already regressed versus WASB, and current
RKT/3D/events lack independent truth. Resolution, visibility, uncertainty, and
dead evidence wiring are more plausible bottlenecks than another ensemble.

### TRK and BODY

| Rank | Hypothesis and candidate | Prerequisite and exact test | Gate and kill rule | Posture |
|---:|---|---|---|---|
| 1 | Player-domain detection/segmentation, not another association threshold, fixes coverage and spectator errors. Test RF-DETR M/L detection and segmentation. | Freeze current YOLO detections/scorer. Fine-tune source-grouped boxes plus a small mask set, spectators, off-court people, truncation, blur, and random negatives. | Per-clip IDF1 >=0.85, zero switches/spectator/off-court FP, coverage >=0.95. Kill on any gate regression or >20% all-in TRK wall increase. | Highest TRK model priority; Apache-2.0 code/weights subject to final inventory. |
| 2 | Shared true masks improve association and BODY prompts. | Build one person-authority artifact. A/B fixed detections with McByte mask association; then, only if switches remain, CAMELTrack multi-cue association or an in-house constrained four-player graph. | Kill if any clip retains a switch, any FP/coverage/IDF1 regresses, or association adds >10% TRK wall. | McByte is a quick MIT training-free test; CAMELTrack is a later learned Apache-2.0 comparison, not a reason to skip detector work. |
| 3 | Masklet conditioning helps current SAM-3D-Body under occlusion. | Resolve the 23-27 mm decode discrepancy and collect fast-athletic GT. Feed track-conditioned SAM masklets through the existing input-prep seam; keep the current decoder. Disable diffusion completion. | Require >=10% visibility-stratified wrist/foot/core or jitter improvement, no BODY regression, and <=20% BODY wall increase. | Bounded A/B only. Official H800 five-target/90-frame profile is about 206 s without completion and about 26 min with it. |
| 4 | GEM-X supplies better temporal whole-body/hands than framewise SAM-3D-Body. | After stable tracks/intrinsics and BODY GT, run static-camera GEM-X per player. Treat local pose/hands as the initial comparison and retain calibrated court root. | World-MPJPE <=50 mm, >=15% p90 wrist/foot and jitter improvement, zero identity/grounding regression, and no slower than current BODY unless accuracy gain justifies measured SLA impact. | Strongest new BODY challenger. Apache-2.0 code, NVIDIA Open Model weights; synthetic-only training and no uncertainty make local GT mandatory. |
| 5 | Part-visible ReID helps only the hard crossing/occlusion windows. | If mask/multi-cue association still fails, swap only appearance distance for KPR and freeze every other input. | Zero switches in named windows, no global regression, <=10% TRK wall; kill on license failure. | Diagnostic/research only; Hippocratic license and cross-domain risk. |
| 6 | Specialist hands help contact windows without full-video cost. | After BODY passes, run HaMeR only on original-resolution event-window crops and bin by hand pixels. | >=20% p90 hand/wrist improvement with no contact regression; abstain below supported pixel size; MANO terms must pass. | Conditional late challenger. |

Keep SAM-3D-Body as the selected default until a candidate wins independent
pickleball GT. Full SAM-Body4D diffusion completion, ActionMesh, Human3R,
OpenCap Monocular, GVHMR, and Multi-HMR are not production candidates today due
to cost, task mismatch, and/or noncommercial terms. They may inform evaluation,
not user-facing biomechanical authority.

### CAL, camera, mobile capture, fusion, and visual reconstruction

| Rank | Hypothesis and candidate | Prerequisite and exact test | Gate and kill rule | Posture |
|---:|---|---|---|---|
| 1 | Encoded-frame truth, native intrinsics, drop telemetry, and clock-aligned sensors improve every downstream lane. | Physical 30 s and 5 min tests at 1080p60 plus supported high-speed modes; align encoded PTS, tap/sensor/ARKit samples, reference dimensions/crop/orientation, exposure/focus/thermal/pressure. | 100% strict sidecar acceptance; no silent truncation; every encoded frame has aligned evidence or an explicit missing/drop reason. Kill any mode that worsens drops, BALL/CAL tails, blur, or thermal behavior. | Adopt capture contract now. Use native YUV for analysis where possible; BGRA conversion is not free. |
| 2 | Known-device profiles plus guided court confirmation beat another synthetic-only auto-CAL cycle. | Offline Kalibr target sweeps per phone/lens/format for intrinsics, distortion, camera-IMU offset, rolling shutter; user confirms semantic court points. Score held-out sweeps, owner viewpoints, and handheld perturbations. | PCK@5 >=0.95, net-height <=2 cm, distortion/reprojection/world gates. Learned output remains a seed until it passes. | Product default direction; Kalibr is an offline lab/profile tool. |
| 3 | AnyCalib improves intrinsics/distortion initialization on cropped/edited imports. | Compare profile/native, current FOV estimate, AnyCalib-only, and AnyCalib-seeded known-court optimization while holding the court solver fixed. | Kill after two iterations without held-out p95/worst-fold downstream improvement. Never use it as court authority. | Runnable Apache-2.0 prior. |
| 4 | An ARKit-owned 60 fps capture path can align pixels, pose, intrinsics, and time more cleanly. | Bounded `ARFrame` + audio + `AVAssetWriter` spike; retain AVFoundation for 120/240 modes. | >=99% pose coverage within half a frame, better pose/court reprojection, and no worse drops/thermal/blur/BALL/CAL tail. | Bounded mobile spike, not an assumed replacement. |
| 5 | Dynamic-scene VO helps labeled handheld imports only. | After camera GT, test DPVO/DPV-SLAM or MegaSaM on failing imports; mask dynamic players and retain metric court scale. Native static captures keep profile/ARKit/known-court preference. | Camera pose/focal/reprojection and downstream court/BODY/BALL gates must improve. Kill if court geometry worsens or compute breaks the SLA. | Deferred fallback. BroadTrack/VGGT-Omega are method/watch items until commercial-clean assets and local gains exist. |
| 6 | Robust progressive Ceres fusion can resolve, not hide, cross-lane disagreement. | Start with camera deltas + ball XYZ on reviewed 2-5 s windows; then paddle/contact; unlock player roots/feet only after those pass. Preserve all raw hypotheses and run modality/fixed-anchor ablations. | Independent world-MPJPE, paddle contact, bounce/landing, sole/floor, event, and reprojection all improve without standalone regressions. Objective reduction alone fails. | Core NS-04 architecture; use GTSAM only if later incremental IMU fusion actually needs it. |
| 7 | MoVieS can make replay more visually rich without becoming measurement authority. | Only after metric MHR/world gates, run a short fixed-camera appearance spike with camera poses and compare source-view fidelity, temporal identity, hallucinations, memory, and render time. | Kill on identity/detail hallucination, low player pixel support, or SLA failure. Never source stats/geometry. | Optional presentation layer; MIT code, checkpoint/data inventory still required. |

### Latency and cost

Do not optimize against copied paper FPS. Measure the integrated path, including
upload, cold start, model load/compile, decode, transfer, asset build, and replay
delivery. Apply in this order:

1. one reproducible cold/warm stage profile with GPU utilization, VRAM, bytes,
   compile buckets, and $/game-hour;
2. decode once and share frames/crops/PTS rather than each lane reopening video;
3. persistent GPU workers and pinned model processes;
4. stable shape/length buckets, then compile/ONNX/TensorRT where supported;
5. batch players and adjacent contact windows while preserving identities;
6. keep frames GPU-resident across compatible stages and overlap independent
   decode/upload/compute/asset work;
7. run expensive high-resolution hands/paddle/appearance only on uncertainty or
   event windows, while retaining every full-mesh frame promised by policy;
8. expose L2 while L3 continues, without duplicating more compute than it saves.

Each lever is one ablation. Revert it if any frozen metric changes outside its
declared tolerance. The target remains first a reliable useful wait, then <=2x
source duration, and only then <=1x if the measured system supports it.

## Shared data contract

The most important cross-lane artifact to add is a versioned person authority
per frame:

```text
source/frame/PTS
  stable player_id + identity confidence/hypotheses
  bbox + true mask + visibility/occlusion
  cheap 2D joints/hands + confidence
  court/world footpoint + transform/covariance
  appearance/part embeddings + model/checkpoint identity
  raw/refined provenance and coordinate space
```

Required reuse:

- masks -> spectator/camera-motion exclusion, mask-aware association, BODY
  prompts/masklets, occlusion quality, and replay;
- cheap joints -> part-aware ReID, BODY/paddle crops, contact proposals, and foot
  phases;
- court points -> association constraints, BODY root placement, and placement;
- ball/audio proposals -> high-resolution BALL/BODY/hand/paddle windows;
- BODY wrists/hands -> paddle pose and refined contact/hitter;
- paddle surface and ball size/blur -> contact and 3D arc factors;
- identity/camera/contact uncertainty -> multiple hypotheses or abstention, never
  a silent assignment;
- user corrections -> lane-specific reviewed queues without overwriting raw
  observations or protected evaluation.

## License and adoption triage

| Candidate | Current posture | Why |
|---|---|---|
| RacketVision | Test after frozen BALL/RKT prerequisites | Released code/weights/dataset and MIT metadata; pickleball/domain and video provenance remain unproven |
| TOTNet | Test now on the frozen BALL card | MIT code and existing adapter; dataset terms and domain transfer still gate use |
| RF-DETR | Test after TRK labels/scorer freeze | Permissive architecture-diverse detection/segmentation candidate |
| McByte | Quick bounded TRK test after masks exist | Training-free mask cue and MIT code; academic implementation, not optimized product code |
| CAMELTrack | Second association challenger only | Multi-cue fit, but learned association must not distract from detector recall |
| GEM-X/SOMA | Test after BODY decode/GT | Commercial-oriented terms and temporal hands; synthetic-only/no uncertainty/local accuracy unknown |
| SAM-Body4D masklets | Bounded current-BODY input A/B | Full path is too slow; masklets may still help identity/occlusion |
| AnyCalib | Test as prior | Apache-2.0 and direct intrinsics/distortion fit; not semantic court calibration |
| MegaSaM | Moving/import fallback test | Permissive camera/structure code; duplicates stronger native/static priors |
| VGGT-Omega | Watch until full release inventory | Promising 2026 result; project claims alone are not an integration contract |
| MoVieS | Late render-only spike | Fast dynamic appearance, but not a metric articulated mesh |
| Uplifting/ball3d/KPR/Human3R/OpenCap Monocular/CARI4D/ActionMesh | Research design or reject for production | GPL/noncommercial/research terms, task mismatch, cost, or missing direct product evidence |

Licensing statements here are engineering triage, not legal advice. NS-07 still
requires a pinned code/weight/data/dependency inventory before launch.

## Primary sources

- [RacketVision official repository](https://github.com/OrcustD/RacketVision)
  and [dataset card](https://huggingface.co/datasets/linfeng302/RacketVision)
- [TOTNet official repository](https://github.com/AugustRushG/TOTNet)
- [Uplifting Table Tennis official repository](https://github.com/KieDani/UpliftingTableTennis)
  and [TT4D paper](https://arxiv.org/abs/2605.01234)
- [BlurBall paper](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Gossard_BlurBall_Joint_Ball_and_Motion_Blur_Estimation_for_Table_Tennis_CVPRW_2026_paper.html)
- [Physics-based ball3d benchmark](https://github.com/lukaszgrad/ball3d)
- [IPPE official repository](https://github.com/tobycollins/IPPE)
- [AdaSpot official repository](https://github.com/arturxe2/AdaSpot)
- [RF-DETR official repository](https://github.com/roboflow/rf-detr)
- [McByte official repository](https://github.com/tstanczyk95/McByte)
- [CAMELTrack official repository](https://github.com/TrackingLaboratory/cameltrack)
- [GEM-X official repository](https://github.com/NVlabs/GEM-X) and
  [official model card](https://huggingface.co/nvidia/GEM-X)
- [SAM-Body4D official resource profile](https://github.com/gaomingqi/sam-body4d/blob/master/assets/doc/resources.md)
- [SAM 3D Body official repository](https://github.com/facebookresearch/sam-3d-body)
- [AnyCalib official repository](https://github.com/javrtg/AnyCalib)
- [Kalibr official repository](https://github.com/ethz-asl/kalibr)
- [DPVO/DPV-SLAM official repository](https://github.com/princeton-vl/DPVO)
- [MegaSaM official repository](https://github.com/mega-sam/mega-sam)
- [VGGT-Omega project](https://vggt-omega.github.io/)
- [MoVieS official repository](https://github.com/chenguolin/MoVieS)
- [Ceres Solver documentation](https://ceres-solver.readthedocs.io/latest/)
- [CARI4D official repository](https://github.com/NVlabs/CARI4D)
- [OpenCap Monocular official repository](https://github.com/utahmobl/opencap-monocular)

## Bottom line

The immediate direction is not to replace the stack. It is to make the current
stack measurable, stop throwing away available evidence, and collect the GT
that lets us reject attractive but wrong answers. After that, the highest-value
bounded challengers are: high-resolution BALL plus explicit visibility/blur,
RacketVision for paddle keypoints, RF-DETR segmentation for player authority,
mask-aware multi-cue association, GEM-X for temporal whole-body/hands, AnyCalib
as an import prior, and a two-pass robust world fusion. Everything else remains
watch/research-only until those experiments say otherwise.
