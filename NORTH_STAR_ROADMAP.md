# DinkVision North Star

Last updated: 2026-07-28.
Status: `VERIFIED=0`.

## Authority and reading rule

This is the sole authority for:

- the product we are building;
- what is actually true today;
- the order of future work;
- promotion gates, stop rules, and the active agent queue.

If another narrative document conflicts with this file, this file wins. The
other active documents have narrower roles:

- `AGENTS.md`: durable repository rules and code navigation;
- `RUNBOOK.md`: commands, flags, actual stage order, artifacts, and failure diagnosis;
- `BALL_TRACKING_PIPELINE.md`: stable numbered BALL interface contract;
- `configs/racketsport/best_stack.json` and `models/MANIFEST.json`: selected defaults and checkpoint identity;
- `runs/`: dated evidence and history, never current truth merely because a file exists.

No root checklist, separate master plan, capability matrix, wave narrative, or
technical blueprint may become a second roadmap. Historical versions are
preserved under `runs/archive/root_docs_20260709/`.

## 1. The product

### 1.1 Promise

A player opens the iPhone app, records or imports one full pickleball game, and
receives the most accurate practical single-camera reconstruction we can make:

- synchronized original video and a metric 3D court;
- four persistent players with articulated 3D meshes;
- ball flight, bounces, contacts, landings, and in/out uncertainty;
- paddle pose at the moments where it can be supported;
- rally, shot, movement, positioning, and recovery facts;
- a short coaching plan whose claims jump back to the exact supporting moments.

The product must be fast enough to remain useful, but correctness and honesty
come before latency. Product inference is single-camera for v1 and assumes a
fixed, non-moving camera: court geometry is solved once and reused for the whole
clip. Moving-camera support is out of scope for v1. Extra cameras, markers, and
surveyed geometry are allowed for training and independent ground truth, not as
a hidden product requirement.

### 1.2 End-to-end user experience

| Step | User experience | Product requirement |
|---|---|---|
| 1. Onboard | Minimal account, consent, handedness, optional player/paddle/court profile | Profiles accelerate later sessions but the generic path must still work. Non-owner biometric persistence is opt-in only. |
| 2. Record/import | Record-first landscape camera or camera-roll import; full-court framing, stability, exposure, FPS, storage, and court-lock guidance | Recording never stalls because an advisory model is slow. Imports disclose reduced sensor confidence. |
| 3. Upload | Clear queued/uploading/processing/partial/failed/ready states | The app uploads the exact video plus one versioned sidecar and can prove which run belongs to it. |
| 4. Fast result | A trust-banded court map, rally segmentation, obvious contacts/bounces, and review clips when supported | Fast results are advisory and may abstain. They never masquerade as deep-world authority. |
| 5. Deep result | Synchronized video and 3D replay with court, net, four players, ball, paddles, contacts, and free-camera controls | Missing entities remain missing; predicted or preview geometry is visibly distinguished from measured evidence. |
| 6. Learn | A concise strengths card, the top three changes, one drill, and evidence-linked comparisons | Deterministic facts first. Language may phrase facts but cannot invent measurements or causality. |
| 7. Correct | User can correct ball/contact/shot outcomes and see “how measured” lineage | Corrections enter reviewed lane-specific datasets; they never silently overwrite raw observations. |
| 8. Improve | Session history and self-relative trends | Compare the player with themself under compatible setups before making population claims. |

### 1.3 Product surfaces and visual direction

The app is record-first and playful but clean: five tabs (Replays, Stats, raised
Record, Coach, Profile) with Record as the cold-launch default and its raised
ball-yellow control turning into a red stop state with an elapsed-time pill. The
visual system is the existing DinkVision ink-on-cream identity — court green,
ball yellow, trail blue/red, rounded cards, restrained hand-drawn accents,
reduced-motion fallbacks. Accents belong on onboarding and empty states, never on
measured-data surfaces.

The deep-result screen must make these obvious without crowding: one shared
timeline across video and 3D/court-map; directly seekable rally, contact, bounce
and shot markers; court, follow-player and free-orbit camera presets; entity
toggles for player meshes/skeletons, ball trail, paddles, contact surfaces,
target zones and ghost positioning; a compact trust badge on every visible
entity; “jump to evidence” and “how measured” on coaching cards; and sample or
fixture content watermarked and never mixed with a real session. Brand
implementation detail lives in `ios/README.md`, which owns neither product scope
nor sequencing.

### 1.4 Trust contract

Evidence provenance and product authority are separate axes. A directly
measured sample may still carry a preview badge when its pipeline has not
passed promotion.

| Evidence provenance | Meaning |
|---|---|
| `measured` | Direct observation or reviewed input, preserved with source identity. |
| `model_estimated` | Model-derived observation with confidence/covariance. |
| `physics_predicted` | Physics or temporal interpolation; never detection truth. |
| `missing` | No defensible evidence. |

| Authority badge | Meaning |
|---|---|
| `verified` | The named capability gate passed on independent preregistered data. |
| `preview` | Useful output from an unpromoted/scaffold path, including `estimated_preview`. |
| `low_confidence` | Evidence exists but is outside the trusted operating band. |
| `too_close_to_call` | Uncertainty crosses a decision boundary; the product abstains. |

`complete` means the minimum product bundle exists and every advertised URL
resolves. It does not mean the underlying CV is accurate. Accuracy is earned
only by the named independent-data gates below.

### 1.5 Product tiers

| Tier | Timing and surface | Authority |
|---|---|---|
| L0: live in-rally | On-device, sub-second capture guidance and sparse advisory overlays | Never promotes, trains, or issues officiating-grade calls. Recording is the priority. |
| L1: between-rally | On-device seconds after a rally/recording | Instant replay and broad advisory cues with abstention. |
| L2: server fast | Target roughly 1-2 minutes after upload, without deep BODY | Trust-banded court/ball/events/placement preview; no deep-world promotion. |
| L3: server deep world | Asynchronous full BODY, paddle, fusion, replay, stats, and coaching | The only product tier that may expose independently gated components as authoritative output or call the integrated result `VERIFIED`. Components may pass frozen NS-03 gates in isolation, but those remain scoped passes until L3 integration succeeds. |

Latency targets are measured end to end, including cold start, upload, compile,
transfer, asset build, and delivery. No named VM is a permanent runtime.

### 1.6 v1 Definition of Done

v1 is done only when three consecutive fresh, preregistered owner/friend games
complete the physical app-to-replay route and satisfy all of the following:

1. The video/sidecar/run identity is exact and reproducible.
2. Minimum replay assets load on native and web surfaces with every URL valid.
3. CAL, four-player TRK, BALL, BODY, contact, RKT, and world-fusion gates all
   pass. Optional presentation features may remain absent, but the core court,
   players, ball, paddle, contact, and fused-world product may not.
4. The replay preserves four player identities, metric court placement, ball
   and paddle relationships, and trust bands without visually convincing
   contradictions.
5. Coaching facts are deterministic, evidence-linked, and pass a zero-
   fabrication audit before language generation.
6. Privacy, deletion, authorization, security, and commercial-license gates
   pass for non-owner use.
7. L3 is delivered in ≤2× source-video duration end to end across all three
   games, including upload, cold start, compile, transfer, asset build, and
   delivery. Accuracy/full-mesh gates are not weakened to reach the SLA.

## 2. Current truth

### 2.1 Product-level blockers

These are more important than another isolated model campaign:

| ID | Current defect | Consequence | Exit gate |
|---|---|---|---|
| P0-A | Swift and Python v1 sidecars disagree | A real capture can fail before CV begins. | Swift-encoded golden fixtures and one physical sidecar validate on the server. |
| P0-B | Presigned video+sidecar upload and clip-status refresh are code-wired, but ready job → manifest → matching replay is unwired and multipart attempt state is not restart-safe | A relaunch can duplicate/orphan an upload, and “Open” still selects the local row rather than the uploaded run. | Mid-part and sidecar-failure relaunch tests preserve/abort one server attempt; one physical record/import → upload → GPU → own replay trace. |
| P0-C | CLOSED (engineering, 2026-07-12, `ns013_stale_reuse`): content-addressed identity and exact dependent-closure invalidation landed; unfingerprinted stale reuse is dead | Legacy run dirs need migration attestation. | Closed at engineering level; product promotion still gated by VERIFIED. |
| P0-D | ENGINEERING-WIRED (scoped pass, 2026-07-15, trackC `tbwire`/`coordwire`): typed coordinate API is now consumed by the real stage consumers (placement, ball court/in-out, ball arc projection, in/out uncertainty, virtual world, plus the earlier person/paddle/metric15 seams); parity-proven byte-identical, distorted-synthetic and real-clip tests pass | Corrected-vs-raw error improvement is still unmeasured; lens-edge accuracy claims wait on independent labels. | Engineering slice scoped-passed; corrected error beats raw path on NS-02 independent labels remains open. |
| P0-E | CLOSED (engineering, 2026-07-15, trackC audit): no live path upgrades `partial` — runner minimum-bundle policy, never-upgrading server bundle policy/gate, pre-display override of the runner's hardcoded `complete`, honest app display; scoped test proof through runner/worker/db/API plus Swift package | Physical end-to-end trace still open. | Closed at engineering level; physical trace under NS-01.2b; product promotion still gated by VERIFIED. |
| P0-F | CLOSED (engineering, 2026-07-15, trackC audit): recursive closure packaging incl. directory assets, atomic staging with manifest swapped last, stats/coaching enforced before manifest, every advertised URL checked runner+server, local/SSH agree via one evaluator; stale stage-order doc/test pin fixed | Physical bundle delivery trace still open. | Closed at engineering level; physical trace under NS-01.2b; product promotion still gated by VERIFIED. |
| P0-G | Explicit timed refined stages LANDED (scoped pass, 2026-07-16 `refinedstage`): events_refined + ball_arc_refined are first-class stages (the ~122s is out of `world`, guard timeouts typed-degrade) and the contact dependency-hash set is complete; `evidence17` earlier landed audio soft evidence (bounded, non-gating), BOTH IPPE poses w/ ambiguity flags, and repaired-confidence markers; size/diameter depth remains unused (design banked, needs a runner blur-sidecar lane) | Diameter evidence still cannot improve contacts/flight; independent-error proof absent. | Audio/diameter affect independent error on NS-02 labels; structural clauses (explicit stages, both poses, marked confidence, hashing) now wired at scoped-pass level. |
| P0-I | **The global association FABRICATES player positions and strips their provenance at export** (2026-07-17, trkL forensics). On wolverine it stitched two different GT players' tracklets into one track and synthesized a 42-frame linear bridge (f45-86, conf pinned 0.35) marching a footpoint 10.4 m across the court THROUGH THE NET at 7.4 m/s. `player_id_repair.py:550` stamps `conf_source="interpolated_endpoint_min_capped_0_35"`, but `tracks.json` frames export only bbox/conf/t/world_xy | **Trust-contract (§1.4) violation: `physics_predicted` interpolation reaches every downstream consumer indistinguishable from `measured`.** The "spectator FPs" on the frozen card are this fabrication, not detector output; synthetic frames also pad cov4 (~0.107 of wolverine's 0.7233 is fake coverage). Every defense is structurally blind: the margin gate runs pre-association on pool detections and interpolated footpoints are on-court by construction; the speed guard scales with gap length (48-frame gap buys 11.28 m); conf floors make it WORSE (more dropouts → more gaps → more bridges) | Fabricated positions never reach export unlabeled: `interpolated: true` survives to `tracks.json`; identity-ambiguous geometric bridging refused (stitch veto + slot re-bind, `runs/lanes/trkL_selection_20260717/DESIGN_selection_layer.md`); wolverine 0 spectator FP, 0 switches, IDF1 ≥0.8516 on the frozen card. Coverage is rebuilt from REAL pool detections or reported honestly lower. |
| P0-H | Typed timebase contract WIRED (scoped pass, 2026-07-15, trackC `tbwire`) through the decode/ingest/frames/events seams; 2026-07-16 `tbcam` added the representation/transform remainder: typed scale/rotate/crop intrinsics transforms (ad-hoc scalers routed parity-first), optional sidecar reference_crop + rolling_shutter fields, loud orientation-mismatch at the calibration seam, RollingShutterModel populated-or-explicitly-missing | Swift-side sidecar emission, physical timing truth, and row-time consumers are still unproven/absent. | Physical 30-second and 5-minute captures (owner-gated): no silent truncation, monotonic encoded PTS, and an aligned sample or explicit missing/drop reason per frame; Swift emission of the new sidecar fields rides NS-01.2b. |

### 2.2 Capability snapshot

Numbers from different protocols are not compared directly.

| Area | Built today | Best honest evidence | Binding next gate |
|---|---|---|---|
| DATA | Owner/public ingest, prelabel, CVAT review, dedup, PTS and protected-eval guards | 1,750 reviewed BALL rows prepared; only the 1,121 clip-folded/disagreement-selected card was scored. **THE BINDING REALITY (corrected 2026-07-24, content-verified): owner-shot pickleball footage totals 9.9 seconds and zero rallies.** `runs/owner_data/incoming/IMG_1605.MOV` (297 frames @ 29.99, sha256 `8a19340278…`) IS content-verified owner-shot pickleball, but it is a static pre-serve moment, not play: zero person boxes, zero BALL/EVENT labels, `trainer_forbidden: true`, all five component rulings FORBID, `train_eligible: false`. Its only product is a 15-pt owner court review (`owner_img_1605_court_review_20260721`, eval-only) which seeds no calibration today — the demo `metric_15pt_reviewed` seed at 1075cee57 is a different file on the pb.vision video. **Usable owner-shot TRAINING footage is therefore zero and no owner-shot rally exists** (correction + provenance: `runs/lanes/data_integrity_20260724/`). Separately, the "39 owned landscape clips" FAILED content verification — dance-rehearsal/personal footage (Sway), not pickleball; that staged label pack was revoked and deleted before any owner time was spent. Ledger integrity: `pbvision_gallery` `training_allowed_ids` overlapped `partitions.test`/`.val`; closed 2026-07-24 (90626da), the allowlist now equals `partitions.train`. It was never exercised, so no score is contaminated. The only human-reviewed person boxes (11,459) are on the four PROTECTED eval clips — 2 of which ARE the frozen card clips, the other 2 strict holdout — so usable person-box training data is zero. CVAT is closed at API level: no person-box tasks beyond those four. Every capability therefore rides on harvested/competitor video. Detector fine-tune arm PARKED blocked-on-data. Owner's 50-row event spot-check FAILED 29/50 vs the ≥47/50 bar → Tier-A auto-labels REJECTED for training; those 50 rows are protected eval seed | Uniform-random audit + true source groups + fresh untouched owner/HARVEST holdout with audio. **Any future owner-facing label pack requires decoded-thumbnail content verification of every source clip plus owner confirmation of the clip list BEFORE rendering** |
| CAL | Manual/metric/profile paths, distortion and ChArUco tools, preview auto-find, frozen GT-free precision harness, guarded refinement, hybrid paint/temporal-lock candidates (all PENDING); `line_evidence_solved_preview` ingestion policy (2026-07-16 Track C ruling, 5cb556fd2): line-intersection solves ingest with mandatory space/distortion/residual/provenance declarations, PERMANENTLY preview-band, structurally never satisfying `metric_15pt_reviewed` gates; owner 15-pt review is the only authority door. **v1 assumes a static camera (§1.1): one authoritative lock — owner 15-pt tap OR an aggregated empty-court/low-occlusion auto-solve — is solved once and reused for every frame, refined by cross-frame line-evidence pooling. Learned/auto court-corner finding is DEPRIORITIZED as an authority path: unsolved even at SOTA (owner PCK@5=0; TVCalib ~65-69% AC@5px even WITH GT segments), so the ≥0.95 gate stays but v1 does not wait on it.** | Corrected owner PCK@5 is 0 for learned candidates; synthetic-only transfer failed twice (~290px); harness M1 3.01/6.22px med/p90 Wolverine (3.81/8.86 Burlington); M4 ours 6.61px vs pb.vision 5.67px median (<1px apart), line-evidence solve 2.6px median. **Our calibration math is at/below pb.vision when seeded well; their ~5.67px edge is capture discipline (mandatory static mount, all 4 corners visible, no cuts), not solver superiority.** Owner completed the 15-pt review 2026-07-17 → demo now has a `metric_15pt_reviewed` seed (1075cee57); orchestrator ingests it, no correction task, calibration stage runs. **NAMED FIXABLE DEFECT: the reviewed 15-pt solve is 19.16px median — WORSE than the 2.61px line solve on the SAME video — because it was fit with a zero-distortion config on a k1=−0.28 camera (the metric solve undistorts with `calibration_intrinsics_from_sidecar`, whose `dist` was zero, so the homography fits distorted pixels); `metric_confidence` stays low → in/out ABSTAINS. Fitting k1 is a config fix, not a fundamental limit** (the two solves AGREE on the camera: fx 719.3 vs 743.0, ~3%). **2026-07-26: CALIBRATION IS NOW THE BINDING FLOOR ON BOUNCE ACCURACY, measured directly** — pushing each calibration's own reviewed correspondences back through the bounce ray-plane path (`pixel_ray_world` → `intersect_ray_z(z=0)`) gives a court-plane residual that a perfect bounce click still inherits: 0.101 m outdoor, 0.127 m wolverine, 0.232 m median / 0.928 m worst indoor. Undistorting first moves the indoor figure 0.232 → 0.199 m, so the distortion fit is a named, measured lever, not a hypothesis (`runs/lanes/ball_label_tool_20260726/`). Owner semantics pinned: unmarked review points = explicitly not-in-frame, never missing | v1: static single-lock reuse + profile/guided confirmation; auto-find still gated on owner-viewpoint PCK@5 ≥0.95 and handheld/distortion gates but off the v1 path |
| TRK | YOLO26m, BoT-SORT/ReID, raw-pool association, court placement; margin-1.0m+OSNet WIRED_DEFAULT (rev 12) preview-band. RF-DETR-L benchmarked but NOT landed; selection layer designed, not built | Mean IDF1 about 0.852; the rev-12 flip lifts worst-clip IDF1 0.6425→0.8516, cov4 0.0433→0.7117, 0 new switches (owner-directed default 2026-07-13, internal-use license ok; fresh full bar cov4≥0.95 unmet, stays preview). 2026-07-16/17 detector bench (GPU-class VM, frozen card): **all 4 zero-shot candidates REJECTED as drop-ins** (RF-DETR-Seg-L/D-FINE-L/DEIMv2-L on measured regressions; RF-DETR-L `adopt-next-step` only). RF-DETR-L @conf 0.18 is the **best burlington row ever measured** (IDF1 0.8831→0.9220, cov4 0.7117→0.9933, all FP axes 0) but REGRESSES wolverine (0.8516→0.8036, cov4 0.7600→0.7233, 0→1 switch, 0→4 "spectator" FP). Those 4 FPs are **P0-I's fabricated bridge, not detector output** — a preregistered conf-0.30 arm made wolverine worse on every axis, confirming threshold suppression is exhausted for the wrong reason. Flip proposal + integration spec are DISPATCH-READY, NOT LANDED (`runs/lanes/trk_rfdetr_integrate_20260717/spec.md`). Counterfactual (stitch veto + slot re-bind, GT-informed upper bound): IDF1 0.8519, 0 switches, 0 FPs, honest cov4 0.6167. Local Mac CPU is NOT score-faithful for association — card rows are GPU-class only | Every fresh clip: IDF1 ≥0.85, zero switches, zero spectator FP, zero far-off-court FP, coverage ≥0.95 |
| BALL | WASB default, candidate training, bounce/in-out, audio/events, arc/sanity chain; split-only `SoftSegmentBoundary` API adopted default-off (byte-identical unused) | Standing anchor F1@20 0.7248, recall@20 0.626, hFP 0.063; candidate A 0.6152/0.654/0.2506 (different internal card); 2D→3D wall (11-min pb.vision study 07-13) = MISSING TRAINED CONTACT/EVENT DETECTION, not solver/camera/candidate-density (all geometry-only paths killed; ~130k public hit/bounce events acquired for an event-head + 117 audio-bootstrap pickleball labels). **2026-07-16 definitive negative: all 3 pre-registered audio-anchor presets REJECTED on the 0-violation kill rule** — conservative 18.77% in-rally coverage / 16 violations; balanced 29.65% / 18; broad 43.69% / 18, vs a frozen baseline of 1/188 segments fit (~0% coverage, 0 violations). **Coverage and violations rise TOGETHER: splits buy coverage, not physics.** Taxonomy 52/52 classified → NEEDS-TYPED-ANCHORS (42/52 anchor-semantics-structural: an untyped onset cannot say *what* it is, so the solver splits in the wrong places); MOVE-1 #3 NOT fired, envelope withdrawn/closed; full taxonomy under `runs/lanes/ballarc_anchorfusion_20260716/`. Arc guard: 697s clip completes in 1493s CPU exit 0 with typed abstentions (was a 3h+ unbounded stall). **2026-07-26 external validation on TT3D (table tennis, NOT a pickleball promotion; `runs/lanes/tt3d_external_validation_20260726/`) — three findings that reorder the ball program:** (1) **REPROJECTION ERROR IS PROVABLY BLIND TO DEPTH.** Sliding a point 1.00 m along its own camera ray changes reprojection by ≤1.6e-13 px; one view carried 0.323 px median reprojection alongside 0.305 m median / 1.381 m p95 3D error. Monocular 3D error is essentially PURE DEPTH — 0.3118 m mean total vs 0.3115 m on the depth axis, image-plane 0.0047 m, 40× smaller. Every prior 3D ball number gated on reprojection was therefore unfalsifiable. (2) **OUR SOLVER MATH IS CORRECT**: camera model, `pixel_ray_world`, `intersect_ray_z` and `build_bounce_anchor` reproduce an independent external implementation at machine precision (5.4e-13 px), and 498/498 fits ran on external data with no core-math change. (3) **`anchor_sigma_for_bounce` IS OPTIMISTIC**: 1.65-2.97× understated on depth, 29-47% coverage against the 68.3% a 1σ implies, plus a systematic +0.068..0.124 m bias away from the camera — one isotropic scalar standing in for an error that is anisotropic AND biased. Confirmed three independent ways (TT3D, the label tool's calibration-residual floor, the 19 owner labels). (1) and (3) are correctness defects, both fixes in flight. **First owner ball labels exist** (2026-07-26): 19 human labels on wolverine — 7 bounce, 7 free-flight, 5 near-player, all `prefill_corrected` — at `runs/lanes/ball_label_tool_20260726/labels/wolverine/`. Review-only human labels, NOT verified ground truth, and already decisive: solver output carrying band `arc_weak` is CATASTROPHICALLY WRONG (ball placed at z = +21 to +23 m and at negative z; errors 2.5-24.8 m) while `anchored_measured` prefills sat within ~0.3 m. The band already separates them; nothing downstream may read `arc_weak` as a measurement | Same-protocol F1@20 ≥0.90, recall@20 ≥0.75, hFP ≤0.05 plus contact/in-out/tail gates on fresh data. No 3D quantity is gated or promoted on reprojection error (§2.3) |
| BODY | SAM-3D-Body runtime, mesh index, placement, grounding and foot-lock | External root-relative 59.7mm, PA 39.9mm, grounding-consistent 76.5mm; decode residual decomposed 2026-07-10 (FK-vs-head ~0, grounding exact, ~53mm = family-metric definition, intentional postchain 23.4mm p95; gate recalibration proposal owner-facing in runs/lanes/ns014_p22residual_20260709/REPORT.md); skeleton-direct foot-slide 20.8-48.4mm breaches 30mm on 3/4 clips (old gate-stream proxy passed — gate open); 2026-07-16 preview-band plant-aware rigid trajectory refiner ADOPTED (Track I CPU fusion of TRK footpoints + BODY root/foot + plant soft anchors, separate artifact, raw immutable; commit 0ec239325, wired as an opt-in default-OFF runner stage 02982d358) cuts skeleton-direct foot-slide 34.55/33.61/20.81/48.38mm → 6.72/5.60/6.26/6.75mm — **4/4 clips under the 30mm bar (baseline 1/4)**, frozen-window anti-gaming arm strictly better 4/4 (6.3-13.4mm), reprojection non-degrading (<1px worse), deterministic byte-identical rebuild, 0 clamps, raw immutable. Hyperparameters were tuned on the same 4 internal cards → **scoped evidence, not independent**; NS-02 GT is the promotion gate (runs/lanes/trackI_placefuse_20260716/) | Corrected decode gate; independent court-frame world-MPJPE ≤50mm; `grounding_metrics.max_foot_lock_slide_m` ≤0.03; no candidate-label promotion |
| RKT | Default-wired wrist/palm/grip `estimated_preview`; both IPPE poses retained and never resolved by reprojection alone | Rectangle IoU about 0.224-0.331; no true pose/contact GT. 2026-07-16 research (corroborated ×2) tightened the GT requirement: **no usable dataset exists** combining tiny/blurred paddle 6DoF + contact GT, and the NS-02.1 ≤0.5-frame sync bar is **insufficient for contact GT** (8.33ms @60fps = 8-17cm of ball travel at impact) — the rig needs ≤1ms audio/LED-verified sync and must prove its OWN held-out error. `racket_pose_hypotheses` has a writer + tests but no strict schema/`ARTIFACT_MODELS` entry — a real schema gap before fusion consumes it | Marker/corner GT; checked-in promotion gates: face-angle p90 ≤5° and contact-point p90 ≤3cm. The old 30° bar is an interim candidate milestone only. |
| EVENTS/PHYS | Ball/audio/wrist fusion, fill, confidence bands and foot postprocessing | Useful internal slide reductions; no reviewed product event gate; audio onset chain + below-threshold candidates landed; ~130k public event labels on disk; owner 50-row spot-check FAILED (2026-07-15): 29/50 true contacts vs the >=47/50 bar, every source fails broadly, 15/29 true windows mistimed \|dt\|>=0.2s — Tier-A audio-x-track bootstrap REJECTED as a training-label source at current thresholds; the 50 owner-reviewed rows (29 typed contacts w/ 2D clicks + dt, 21 hard negatives) are the first owner-verified pickleball event labels, reserved as PROTECTED EVAL SEED, never training. **2026-07-16/17 event head — a real checkpoint and an honest failure:** T4 pretrain ran 3956 steps, best val F1@±2 **0.3631** (step 1976; the late collapse to 0.009 is preserved — the overfit-then-diverge signature of a starved set). Two findings: (1) the repo's public-eval "0 TP" was a **HARNESS BUG** — `eval_event_head.py` hardcodes 15-frame windows against a 64-frame-context model; re-eval on the SAME checkpoint at the matched window gives **9 TP / 0 FP, max prob 0.937** (~20-22% recall), so the committed CLI measured an artifact, not a model verdict; (2) **zero-shot tennis→pickleball transfer FAILED** — 7.16 HIT/s over 697s against the ~200-400 contacts a real game holds, a HIT in 98% of seconds, and 70.2% audio agreement against a ~99.3% chance baseline = **zero discriminative information**; anchors ruled DO-NOT-INGEST, and the ≥0.9 tail is 123/123 HIT with zero BOUNCE. Root cause is DATA, not architecture: 2.4% label reach (1,793/74,546), 18.1% media coverage, one-window-per-row extraction → 226 train windows; `SCALE_UP_SPEC.md` = ~68× available for $2.2-4.5 / 2-5h. The checkpoint is RGB-only — directed track/wrist conditioning is NOT in it (booked); detail under `runs/lanes/event_head_scaffold_20260716/`. **E-v2 is now UNBLOCKED (2026-07-24, `runs/lanes/ev2_staging_rootcause_20260724/`): the three failed A100 dispatches were ONE divergent gate-proof assert** — `proof.get('pass') is True or proof.get('status') == 'pass'` against an emitter that only ever writes `status: "PASS"`, so a gate that had just passed aborted the run under `set -euo pipefail` before the evidence copy and before the ERR trap that would have explained it. Reproduced, not narrowed; fixed at f29145a with an on-VM log-pull leg added. The owner-facing "media-staging bug refused the run 3×" note is accurate for attempts 1-3 and stale for the final one | Contact timing p90 ≤40ms, bounce-vs-hit/in-out gates, corrected acoustic/A/V timing, no standalone regression |
| FUSION | `one_world_v1` preview module, now WIRED as a default-OFF runner stage at order 185 (2026-07-24, `runs/lanes/oneworld_wire_20260724/`) — integration progress, not capability progress: nothing downstream reads it and it still refuses every contact. The same lane renamed the `world` stage capability `fusion` → `composited_world`, because that stage composites and performs no joint refinement; the old name asserted a capability that never ran. Confidence-weighted staged refinement over Track I's contract; soft bounce priors that never snap; contact co-location as a likelihood product with a null hypothesis; both IPPE poses retained; always-emit display pose with an honest band; tiered ball continuity (arc_measured → physics_predicted → ray_projection → absent); typed paddle/floor/net events | Wolverine (300f): 4 players placed every rally frame (conf 0.82-0.92), ball 295 arc_measured + 5 physics_predicted / 0 absent, 28 typed events, M5 reprojection non-regressing (61.2683→61.2658px), independent rebuild byte-identical, 14/14 raw input hashes unchanged. **Headline is an honest refusal: 0 of 24 declared contacts CONFIRMED** (22 unsupported — ball >1.2m from every wrist; 2 too_close_to_call). Baseline contact→wrist residual median 7.97m / p90 11.17m (ball-center distance 8.13m / 11.32m) with exact 30-fps joins on all 24 — so the mismatch is real, not a timebase artifact; at frame 78 the stack declares a hitter whose wrist is 11.17m from the arc ball. The pass refused to move the ball and left the raw event immutable. Demo (697s) is an honest partial: 0 players/paddles/contacts because tracking/BODY never ran there | NS-04 independent gates + leave-one-modality/multi-init ablations; contacts require BODY+contact GT. Unreviewed fused output may never train or validate its own inputs |
| REPLAY/STATS | Web/native boundaries, ghost previews, movement stats; viewer usability wave ADOPTED (browser-verified) | Follow-camera playback fixed: FPS ratio 0.375 → 0.809 guarded / 0.933 segment-matched (follow 17.9→46.1fps); VM-written-manifest recovery with a loud counted banner; badge overlap zero; 280/280 tests. A measurement hazard was found and fixed mid-wave (an ended video lightens the render loop and inflates FPS — probes now assert playing-state and loop). Owner evidence pack at `~/Desktop/visual_evidence_20260716/` renders the fused world. Residual (non-blocking): the clip's last ~5s is heavier for BOTH presets — a shared, pre-existing render cost, not a preset collapse | Current full bundle, full computed-frame mesh policy, native/web visual/perf/every-URL proof |
| COACHING | Deterministic facts plus runner-enforced zero-fabrication audit before manifest (NS-05.1 core landed); **2026-07-26 a standalone dense-mesh Form Study preview replaced the mixed legacy dashboard for local review.** It loads two immutable MHR70 surfaces through the existing indexed-mesh path, supports translucent overlay or side-by-side views, exposes five aligned motion phases and one bounded preview cue, and contains no skeleton, rest-pose, court, ball, analytics, or legacy evidence fallback. The current compact example has 48/48 user and 43/43 senior-reference dense frames, 18,439 vertices and 36,874 faces per frame, identical topology, and explicit `skeleton_fallback=false`. The explicit route preserves old rubric links; query index overrides are removed; world/index/faces/chunks are hash-bound before rendering; and `PRO` requires separate trusted clearance. The senior reference is labeled `REFERENCE PLAYER` and works only on loopback because its broadcast likeness/derivative-display rights are not cleared | 141 focused comparison/viewer-data tests, production build, one real hash-verified two-chunk live-decode integration test, and the mesh-subset CLI round trip pass. The full viewer run is 326 passing with six unrelated missing historical fixture failures. This remains `PARTIAL`, `VERIFIED=0`, local R&D only: the similar soft-shot classification, contact timing, coaching causality, athlete identity/qualification, and public display rights are unverified | Capture a pro/5.0 coach under a signed likeness and derivative-motion release (or clear the PBVision `Pro Training` participant and derivative-display rights), run the same BODY/index path, manually review shot/contact/phase correspondence, and pass reference-faithfulness, owner, and ≥4.0-player audits before shipping a `PRO` comparison |
| E2E | A 20-stage CLI (21/22 with rally_gating/verify) plus code-wired video/sidecar upload and clip-status refresh | Swift package tests are scoped code proof; ready-manifest routing, restart safety, and a current physical-app bundle remain unproved | Complete NS-01 through NS-05, then one clean current-stack reproduction |

The active BODY stack is SAM-3D-Body only. RTMW, RTMW3D, RTMPose, and MMPose
are retired from the pickleball pipeline. The separately tested
Fast-SAM-3D-Body challenger regressed end-to-end speed/fast-swing accuracy and
remains rejected unless a new bounded hypothesis directly addresses that miss.

`VERIFIED=0` remains binding. Test green, schema green, a browser load, a
partial run, or an attractive overlay is not a capability promotion.

Capability/evidence status is separate from per-object trust:

| Status | Meaning |
|---|---|
| `VERIFIED` | Named independent-data promotion gate passed. |
| `scoped pass` | A named slice passed inside its declared scope only. |
| `smoke-verified` | Execution/presence proof, not accuracy. |
| `partial` | Some declared outputs are missing or degraded. |
| `review-only` | Intended for human inspection, never authority. |
| `rejected` | Measured candidate failed or regressed; preserve the result. |
| `no-attempt` | Candidate was not run because prerequisites/access were absent; not negative model evidence. |

### 2.3 What not to repeat without new evidence

- naive detector voting for BALL;
- another synthetic-only CAL retrain without real viewpoint supervision;
- chasing learned/auto court-corner finding as an authority path (owner-viewpoint PCK@5=0, SOTA <70% AC@5px even with GT segments) instead of a single static lock reused across the clip;
- more association-only TRK sweeps without detector/domain/ReID leverage;
- self-generated 3D used as validation truth;
- rectangle or box IoU promoted as paddle 6DoF;
- Fast-SAM-3D-Body replacing the current BODY path after the measured regression;
- scalar Magnus/spin claims before trusted contacts and flight GT;
- generic neural rendering used as measurement authority;
- threshold shopping on Outdoor or relabeling it as a fresh holdout;
- untyped audio-only anchor classes for BALL: onsets on a real game are ~3.3/s and near-uniform, so any single-signal audio anchor is at/below its own chance baseline and buys coverage only by splitting flight in the wrong places (42/52 failures were anchor-semantics-structural). Typed anchors first, then fusion — no single signal decides;
- conf-floor or threshold tuning aimed at association-FABRICATED false positives: the synthetic frames carry a pinned conf and are created AFTER every pool-stage gate, so raising the floor strictly worsens the failure (more dropouts → more gaps → more bridges). Fix the fabrication, not the threshold;
- pb.vision data now has OWNER-SIGNED FULL 100% USAGE RIGHTS (2026-07-20, training + commercial) — the advisory-D3 "compare-only, no training" ruling is SUPERSEDED for LEGAL reasons. Two NON-legal disciplines remain: (a) hold 2-3 pb.vision videos out as a compare-only benchmark so an honest head-to-head survives (a model that trained on a clip cannot fairly be scored against pb.vision on that clip); (b) their cv_export predictions are their MODEL's outputs (noisy), not human GT — use their VIDEOS as full training pixels, use their event/ball/court predictions as an agreement-filtered TEACHER signal (keep where our audio+wrist+physics independently agree), never as ground truth. Their 12 videos are IN-DOMAIN pickleball (unlike the tennis/TT pretrain corpus) — the highest-value fix for the event-head domain gap
- single-window-per-row training extraction, or reading an eval harness whose window/context disagrees with the checkpoint's: the first starves a corpus to a rounding error (2.4% label reach), the second manufactures a fake "0 TP" verdict. Assert the eval window against the checkpoint config at load;
- **never gate, band, or promote a 3D quantity on reprojection error.** It is provably blind to the axis that carries the error: a 1.00 m slide along the camera ray moves reprojection by ≤1.6e-13 px, and a measured 0.323 px median sat beside 0.305 m median / 1.381 m p95 3D error. A smaller residual is not a better world. Gate 3D on independent 3D, on depth-axis coverage against declared sigma, or abstain;
- reporting a single isotropic sigma for a monocular 3D point: the true error is anisotropic (40× larger along depth than in the image plane) and biased away from the camera, which a zero-mean isotropic scalar cannot represent at all. A sigma that covers 29-47% of its own errors is a false promise, not a conservative one;
- treating a solver band as decoration. `arc_weak` output was measured 2.5-24.8 m wrong, including impossible heights and negative z, on the same clip where `anchored_measured` sat within ~0.3 m. Bands that already separate good from catastrophic must be enforced at every consumer, not just displayed.

Indoor remains protected. Outdoor remains protected from further leakage but is
a historical benchmark, not statistically fresh promotion evidence.

## 3. Target CV architecture and data reuse

The pipeline is a provenance-aware two-pass DAG. Separate lanes are valuable,
but their data must meet again before a world or coaching claim is produced.

```mermaid
flowchart LR
    A["Video + audio + PTS + sensors + profile"] --> B["Versioned schema, timebase, coordinates"]
    B --> C["Content-addressed ingest + quality"]
    C --> D["Court, intrinsics, distortion, camera"]
    C --> E["BALL + audio prepass"]
    D --> F["TRK, identity, court placement"]
    F --> G["Cheap full-rate joints, hands, feet"]
    D --> H["Coarse events + initial 3D arcs"]
    E --> H
    G --> H
    H --> I["Full BODY + high-res paddle"]
    D --> J["Refined contacts, arcs, placement"]
    E --> J
    I --> J
    D --> K["Robust global fusion"]
    F --> K
    I --> K
    J --> K
    K --> L["Stats + evidence-linked coaching facts"]
    L --> M["Confidence gate + recursive assets + manifest last"]
    M --> N["Native/web replay"]
    N --> O["User correction + reviewed lane datasets"]
    O --> P["Grouped evaluation + gated registry"]
    P -. "approved next stack only" .-> C
```

### 3.1 Reuse contract

| Producer | Required consumers | Never allowed |
|---|---|---|
| Capture/timebase | encoded PTS, native intrinsics/crop, drop reasons, every frame-aligned stage, audio correction, rolling-shutter model | Assume CFR, silently truncate sensors, or align “latest” ARKit/tap samples to the movie. |
| Court/camera | membership, placement, ball 3D/in-out, BODY grounding, paddle, net, fusion, metrics | Publish a homography without coordinate space, distortion state, and covariance. |
| TRK/person authority | stable ID plus bbox, true mask, cheap joints, court footpoint, visibility, embeddings and confidence feed BODY, camera exclusion, paddle, hitter, events and replay | Import lexical “latest,” equate role/side with identity, or use BODY translation as identity truth. |
| BALL/audio prepass | top-K points, visibility, blur/diameter, rally spans, contact proposals, mesh schedule, initial arcs | Become contact authority by itself or drop raw/corrected timing. |
| Cheap joints/hands | event refinement, mesh schedule, paddle initialization, foot phases | Arrive only after events are frozen. |
| BODY | paddle grip, stance/sole, biomechanics, placement refinement, fusion | Relabel placement-derived coordinates as BODY-derived because a skeleton file exists. |
| Paddle | all plausible planar poses, contact refinement, outgoing-ball constraints, swing facts, fusion | Claim 6DoF from rectangle IoU or discard the second IPPE pose by reprojection alone. |
| Refined events/arcs | shot taxonomy, slow motion, landing/in-out, fusion | Validate only by a smaller optimizer residual. |
| Global fusion | one refined world with covariance/provenance | Overwrite immutable observations or train/evaluate on unreviewed fused output. |
| Product gate | replay visibility, wording strength, review queue | Turn “artifact exists” into “accurate.” |

### 3.2 Minimum inspectable deep-result bundle

A deep job is inspectable and status-reportable only when it owns and validates:

- source identity and versioned capture sidecar;
- court/camera calibration plus coordinate/time metadata;
- persistent player tracks and declared BODY coverage;
- ball, event, arc, paddle, and fusion artifacts or explicit missing reasons;
- deterministic stats/coaching facts built before the manifest;
- recursive replay assets, trust bands, and a manifest whose every URL resolves;
- a summary that reports `complete`, `partial`, or `failed` without translation.

`complete` additionally requires every mandatory v1 artifact and gate named in
Section 1.6. An explicit missing reason makes a bundle inspectable, not complete;
the job must remain `partial`.

### 3.3 Coaching safety boundary

Before upstream gates pass, user-facing authority is limited to current
placement/court movement facts with their preview lineage. Contact height and
stance require BODY+contact gates; apex/net clearance/landing require BALL+
CAL+event gates; paddle facts require RKT. Even after those gates, do not ship
torque, muscle load, injury risk, exact high-speed angular velocity, or causal
shot-error attribution without separate athletic validation.

## 4. Ordered execution program

Phases execute in order. Work explicitly marked parallel may overlap only after
its dependencies exist. Every task must save a report under `runs/`, identify
source/code/model/config versions, score the frozen gate, and state
`adopt`, `reject`, `partial`, or `no-attempt`.

### NS-01 — Make the real product route correct

Nothing else can produce trustworthy user evidence until NS-01 is complete.

| Task | Outcome and owned surfaces | Acceptance gate | Stop/kill rule and unlock |
|---|---|---|---|
| NS-01.1 Capture/sidecar truth | One versioned schema across Swift/import/Python/server; stream sensor samples; enumerate encoded PTS; store native intrinsics with reference crop/orientation, drops, clocks and rolling shutter | Golden fixtures plus physical 30-second/5-minute and supported high-FPS captures: no silent truncation, monotonic PTS, every frame aligned or explicitly missing | Do not loosen Python or derive authority from an optional late-discard tap. Unlocks sensor/CAL truth. |
| NS-01.2a Complete production upload lifecycle | Finish the code-wired video+sidecar path: persisted multipart identity/ETags, restart/abort semantics, honest job polling, ready manifest, and matching replay routing | Focused Swift/server tests cover death after a part and after video-before-sidecar; one capture ID survives clip/job/manifest/replay | A new clip on relaunch or local-row replay fails. Mocked tests are not physical proof. |
| NS-01.2b Prove physical upload | After NS-01.3-01.5 settle identity/status/assets, run final device record and camera-roll traces | Saved 30-second traces prove record/import → upload → job → artifact → own replay with auth | Any manual artifact substitution fails the gate. Unlocks real E2E evidence. |
| NS-01.3 Content-addressed run DAG | Source SHA-256/size/timing identity; code/model/config/upstream fingerprints; explicit inputs win; atomic transactional stage dirs | Same clip ID with different video cannot reuse pixels; changed dependency rebuilds exact closure; identical dependency reuses safely | Do not expand `--force` into another manual deletion list. Unlocks reproducible evaluation. |
| NS-01.4 Coordinate/time convention | Typed encoded/raw→undistorted→reference→court/world transforms; PTS/VFR, native-intrinsics scaling, A/V mux, acoustic propagation, sensor clocks and rolling shutter | Distorted synthetic and real iPhone tests; corrected event/geometry error beats raw path on independent labels | Do not mix coordinate spaces, align latest samples, or correct audio destructively; preserve raw values. Unlocks trustworthy fusion. |
| NS-01.5 Honest status and packaging | Minimum bundle policy; partial propagation through runner/worker/API/app; recursive atomic copy; stats/coaching before manifest | Missing BODY/ball/paddle/assets remains `partial`; complete requires every advertised URL; local and SSH paths agree | Exit 0 is not sufficient. Unlocks meaningful product-ready state. |
| NS-01.6 Current spine cleanup | Remove duplicate legacy stage graph, type expected optional failures, fail on programming/schema errors, validate complete frame schedules | One authoritative stage graph and tests for cold, reused, partial, and failure paths | Do not hide arbitrary exceptions as degraded stages. Unlocks two-pass integration. |
| NS-01.7 Evidence plumbing | Make classified audio affect events; hash contact dependencies; pass blur/diameter; retain both IPPE poses; mark repaired confidence; run one post-BODY/RKT refinement | Focused tests plus independent contact/3D/RKT ablations prove each new consumer; unsupported evidence remains missing | Do not raw-average modalities or promote on residual/overlay gains. Unlocks NS-04 fusion. |

**NS-01 exit:** one physical capture opens its own correctly identified replay,
with honest missing capabilities and no stale artifact path.

### NS-02 — Build independent truth and reset evaluation

| Task | Outcome and owned surfaces | Acceptance gate | Stop/kill rule and unlock |
|---|---|---|---|
| NS-02.1 Gold capture protocol | Product phone plus two auxiliary high-FPS phones, surveyed court/net, ChArUco, LED/audio sync, paddle markers, scripted shots/occlusions | Static points within 2-3cm, inter-camera sync ≤0.5 frame, uncertainty saved for dynamic labels | Extra cameras are GT only. Unlocks CAL/BODY/BALL-3D/RKT/contact truth. |
| NS-02.2 Lane-specific GT | Versioned CAL points, person IDs/boxes, 3D joints/sole contacts, ball centers/events, paddle face/markers | Each label has source, frame/PTS, reviewer, uncertainty, and immutable raw reference | Candidate predictions cannot become independent GT. |
| NS-02.3 Representative audit | Add uniform-random reviewed frames alongside disagreement-selected examples | Report performance separately on random, hard/occluded, seen, and unseen strata | Do not average away the unseen-source gap. |
| NS-02.4 True source grouping | Group by source game/session/court/device before any candidate selection | No frames/clips from one source cross train/selection/test; 1,750 BALL folds scored source-disjoint | Leave-one-clip is not source-LoSO. |
| NS-02.5 Fresh promotion ledger | Pre-register metrics, thresholds, candidate, code/checkpoint/training-data provenance, transitive licenses, and untouched owner/HARVEST sources | Ledger/license card exists before inference; one-shot result retained whether pass or fail | No threshold shopping or noncommercial candidate in the selected product stack. Unlocks component promotion. |

**NS-02 exit:** every component has an independent, source-disjoint route to a
frozen gate. This phase should run immediately after NS-01 design stabilizes;
capture preparation can overlap NS-01 implementation.

### NS-03 — Improve components in parallel against the same truth policy

The model-improvement lanes may run in parallel after NS-02 supplies their
required labels. LIVE infrastructure, capture guidance, and thermal-soak work
are explicitly exempt and may begin after NS-01.1/01.2a; deploying a BALL
student still waits for the NS-03.BALL gate.

| Lane | Exact next sequence | Promotion gate | Kill/defer rule | Downstream unlock |
|---|---|---|---|---|
| NS-03.CAL | **v1 static single-lock FIRST: solve one authoritative court (owner 15-pt tap OR aggregated empty-court auto-solve) and reuse it for the whole clip, refined by cross-frame line-evidence pooling; fix the zero-distortion 15-pt config (fit k1) so `metric_confidence` rises and in/out stops abstaining** (spec `runs/lanes/static_cal_firstlock_20260717/spec.md`). Then ship device/lens profiles + guided confirmation; validate native intrinsics/distortion/rolling shutter; AnyCalib only as an import prior; DPVO/MegaSaM only after labeled moving-import failure | Reuse-lock residual within ~1px of the per-frame solve, owner-viewpoint PCK@5 ≥0.95, net-height error ≤2cm, reprojection/distortion/handheld gates | Auto court-corner finding DEPRIORITIZED as authority (PCK@5=0, SOTA <70% even with GT) — the static lock is the v1 authority; no third synthetic-only retrain; AnyCalib/SfM never court authority; native static path stays profile/ARKit/known-court first | Metric world, in/out, grounding, placement, fusion |
| NS-03.TRK | Freeze scorer/provenance → fix detector/domain/off-court errors; benchmark RF-DETR det/seg → ReID → McByte mask cue on worst clips → only if needed CAMELTrack/constrained tracklets | All fresh clips IDF1 ≥0.85, 0 switches, 0 spectator FP, 0 far-off-court FP, coverage ≥0.95 | Stop association-only sweeps; each later step keeps detections fixed; reject >20% detector or >10% association wall increase without full-gate gain | Shared person authority for BODY/camera/paddle/hitter/stats |
| NS-03.BALL | Freeze source groups/strata → score WASB, Candidate A, existing TOTNet adapter and RacketVision checkpoint on all 1,750 rows → train high-res/court-crop visibility+blur challenger → after 3D GT add diameter and simple-physics/lift-first ablations | F1@20 ≥0.90, recall@20 ≥0.75, hFP ≤0.05 plus p95/p99, teleport, contact, bounce, 3D landing and in/out gates | No duplicate TrackNet integration, raw voting, tuned holdout, or spin claim; reject subset gains that worsen global hFP/tails/runtime | Events, arcs, mesh schedule, shots, replay |
| NS-03.BODY | Fix the 23-27mm decode residual → collect fast-athletic GT → score current path → severe-occlusion masklet-only SAM-Body4D with completion disabled → bounded GEM-X temporal whole-body/hands challenger | Court-frame world-MPJPE ≤50mm, `grounding_metrics.max_foot_lock_slide_m` ≤0.03, ≥15% p90 wrist/foot/jitter gain for replacement, mesh/joint/identity consistency | Keep SAM-3D-Body default on no-attempt/regression; full SAM-Body4D completion is killed for latency; synthetic/internal metrics do not promote GEM-X | Paddle, biomechanics, placement refine, fusion |
| NS-03.RKT | Capture 4-corner/normal/contact GT → released RacketVision 5-keypoint zero-shot → pickleball high-res fine-tune → retain both IPPE poses → resolve with hand/time/ball/surface; only then trajectory cross-attention or GigaPose | Interim candidate milestone: face-angle p90 ≤30°. Promotion: face-angle p90 ≤5°, contact-point p90 ≤3cm, no BALL/BODY regression | Rectangle IoU, one-solution reprojection and box orientation remain preview; candidate must beat current preview and local supervised baseline | Refined contacts, swing facts, ball impulse, fusion |
| NS-03.EVENTS | Fix audio/time plumbing → label hit/bounce/net/stomp/other → coarse proposals → deep BALL/BODY/RKT → refined global assignment and arc/schedule once; AdaSpot only after labels | Source-disjoint contact timing p90 ≤40ms plus bounce-vs-hit/hitter/coverage and no standalone regression | A loose published collar, raw confidence average, or improved proposal recall alone does not promote | Correct deep windows, contacts, arcs, slow motion, shots |
| NS-03.LIVE | After NS-01.1/01.2a: ship capture guidance/live court lock, person model and record+infer thermal soak; test ARKit-owned 60fps capture, retain AVFoundation high-speed; BALL student waits for BALL gate | Record never drops/stalls; sustained cadence/thermal/pressure/drop budget; ≥99% aligned ARKit pose if selected; every call advisory/abstaining | Live work may parallel server lanes but never weakens recording/server gates; unsupported modes and unpromoted models stay kill-switched | Trustworthy L0/L1 capture and between-rally product |

Every lane must score its baseline first, use the same scorer for each candidate,
record runtime without making it the accuracy gate, and update the selected
stack only after a named pass. Before execution, pin code, checkpoint,
training-data provenance and transitive licenses; research-only candidates may
produce diagnostics but cannot enter the selected product stack.

### NS-04 — Join the lanes into one world

| Task | Outcome | Acceptance gate | Stop/kill rule |
|---|---|---|---|
| NS-04.1 Coarse pass | BALL/audio + cheap joints propose rallies, contacts, bounces, hitters, initial arcs, and BODY compute windows | High-recall proposals with uncertainty; no authority claim | Do not freeze or promote coarse contacts. |
| NS-04.2 Deep pass | Run full BODY and high-resolution paddle on the planned cadence, including every computed frame required by the full-mesh policy | Coverage/provenance complete; no byte budget silently removes required frames | Fail partial if required deep inputs are absent. |
| NS-04.3 Refined pass | Recompute contacts, bounce/hit class, hitter, arcs, landing, and placement from same-run wrists/paddle/BODY/audio | Contact p90 ≤40ms and arc/landing error improve on independent GT without standalone regression | Never reuse no-wrist contacts as current after BODY changes. |
| NS-04.4 Surface priors | Ball center one radius from paddle/court surfaces; projected paddle contact inside face polygon; bounded impulse/friction; sole/mesh on court | Independent contact/bounce/floor errors improve | Never snap ball center to a plane or ankle centers to the floor. |
| NS-04.5 Robust global fusion | Progressively optimize camera/time, player root/pose, ball segments, multiple paddle/identity hypotheses and contacts with robust/switchable factors | Independent world-MPJPE, paddle-surface contact, bounce/landing, sole/floor, event and reprojection improve; multiple-initialization and leave-one-modality/fixed-anchor ablations pass | Raw observations immutable; one early hypothesis, residual reduction and visual plausibility never promote. |
| NS-04.6 World output | One refined world candidate with covariance, provenance, trust bands, and raw/refined separation | Viewer and artifact checks pass; unsupported elements absent/banded | Unreviewed fused output cannot train or validate its own inputs. |

### NS-05 — Turn the world into a useful product

| Task | Outcome | Acceptance gate | Stop/kill rule |
|---|---|---|---|
| NS-05.1 Deterministic facts | Generate rally, shot, movement, positioning, recovery, landing and contact facts before the manifest | Reviewed correctness and complete lineage; shot macro-F1 ≥0.65 and top-2 accuracy ≥0.85 | No claim whose source cannot be opened. |
| NS-05.2 Coaching comparator | Convert facts into reference/self-relative comparisons and rank the top three actionable changes | Expert rubric and user audit; facts unchanged by wording layer | Do not feed free-form raw numbers to the language model. |
| NS-05.3 Language layer | Phrase approved facts, one drill, and evidence links | Usefulness ≥8/10 and fabrication 0/300; every claim opens its evidence; owner + ≥4.0 player review | Language cannot invent injury/torque/load claims. |
| NS-05.4 Replay assets | Metric MHR meshes first, banded contact/ball/paddle overlays and free-camera comparisons; optional MoVieS-style appearance only after metric gates | Native/web visual QA, target FPS/size measured, every URL valid; appearance checked for temporal identity/hallucination | No fixture as user data; render-only appearance and unseen surfaces are predicted, never measurement/stat authority. |
| NS-05.5 Correction flywheel | User edits route to lane-specific reviewed queues with provenance | Round-trip correction tests and dataset version increments | Product corrections never mutate protected test labels or raw artifacts. |

### NS-06 — Optimize speed, cost, and reliability with metric parity

Start only after NS-01 correctness and stable NS-04/05 outputs make timing
meaningful. Current runtime evidence says cold start, compile, decode/I/O, and
transfer dominate more than steady inference in some BODY buckets.

| Task | Sequence | Gate |
|---|---|---|
| NS-06.1 Profile | Measure cold/warm stage time, GPU utilization, compile buckets, decode, upload/download, asset build, size, and cost | One reproducible trace on the current stack; no reused/stale timing artifacts. |
| NS-06.2 Remove waste | Shared decode/PTS, GPU-resident frames where appropriate, persistent workers, persisted stable compile buckets, batched players/windows, stage overlap; then ONNX/TensorRT/DALI/quantization only when the trace justifies each | Each lever improves full p95 wall time and preserves all frozen metrics/timestamps; revert independently on drift. |
| NS-06.3 Tier delivery | Make L2 fast result useful while L3 continues; stream honest progress/capabilities | No duplicate inference that outweighs latency; partial semantics preserved. |
| NS-06.4 Reliability/cost | Preemption-safe jobs, resume, idempotency, observability, teardown, fully loaded $/game-hour | Three fresh runs without orphaned resources or status drift; measured cost reported as a range until invoice-backed. |

Target progression is first a reliable useful wait, then ≤2× video duration,
then ≤1× only if measured levers support it. Accuracy and full-mesh requirements
are not weakened to hit a headline.

### NS-07 — Launch safely and prove repeatability

| Task | Outcome and gate |
|---|---|
| NS-07.1 Security/auth | Close the three HIGH findings, enforce user/job/artifact authorization, secrets/dependency scans, abuse limits, and audit logs. |
| NS-07.2 Privacy/deletion | Explicit biometric/video consent, retention policy, export, delete cascade, and session-only default for non-owner data until opted in. |
| NS-07.3 Commercial path | Model/code/data/license inventory and commercial-clean selected stack. Private development may continue; monetized launch may not bypass this gate. |
| NS-07.4 Friend onboarding | A non-owner completes setup, capture/import, upload, replay, correction, and delete without developer intervention. |
| NS-07.5 Repeatability | Three consecutive fresh preregistered games pass the v1 Definition of Done. Any failed game resets the consecutive count after the defect is fixed. |

## 5. Active queue for the next agents

Do not start another broad model search. The 2026-07-26 measurements (§2.2
BALL/CAL, §2.3) reordered this program: two correctness defects now outrank every
accuracy campaign, and calibration — not the solver — is the binding floor on
bounce accuracy. `VERIFIED=0`, `authority_state=review_only`, and
`measurement_valid=false` remain binding on every row below.

**2026-07-28 overnight amendment (owner full-access directive, evidence under
`runs/alwayson_fresh_wave_20260728/` and `runs/lanes/*_20260728/`):** rows 1–4
below are ENGINEERING-COMPLETE on main (sigma anisotropic fix, reprojection
retirement, calibration k1 fit, sub-frame bounce timing) — their remaining work
is scoring against independent labels, not building. Row 6 (E-v2) EXECUTED,
then completed to the full 12k-step design the same day
(`runs/lanes/ev2_cont_20260728/`): **HIT F1@±2 0.541 / precision 0.923@0.5 /
~30 ms mean timing on the 50-clip public sweep + first nonzero BOUNCE TPs —
the strongest event-head evidence this program has produced; the trained-event
wall is broken on the PUBLIC domain.** The spec'd 400-step owner fine-tune was
run and REJECTED with evidence (destroys the signal). Zero-shot pickleball
firing is zero → the binding gap is now DOMAIN transfer; the named lever is
pb.vision in-domain training pixels (§2.2 DATA). RD_ONLY license posture on
the public corpus — research diagnostic, not a product-stack candidate before
license review. No anchors ingested; firing-rate/typed-anchor gates unchanged. Row 7's always-on directive is
DONE by owner order: post-BODY foot anchoring + planted-foot trajectory
refinement are default-ON in both presets (`1e4ab2a`), proven on a fresh
six-clip GPU wave (foot-slide 4/6 pass; wolverine 0.0378 NEW FLAG,
indoor-diagonal 0.0546 known). NS-06 speed levers measured/landed: warm BODY
worker (cold 174–222 s → 110 s warm remote command, default-OFF) and the
`--body-local` co-located silent-degrade bug FIXED (`757da51`) after the
co-located pipeline measured median 94.6 s (no BODY) vs ~350–500 s split.
Models are durable: warm snapshot `pickleball-court23-warm-20260728` + S3 store
`s3://sway-videos/pickleball-models/20260728/`. H100: zero a3 quota in every
region — owner quota request queued.

| Order | Exact action | Pass gate | Failure/stop rule |
|---:|---|---|---|
| 1 | **Fix the uncertainty model (in flight).** Replace the single isotropic `anchor_sigma_for_bounce` scalar with an along-ray sigma that is anisotropic and bias-corrected, and floor every bounce sigma at that clip's own measured calibration plane residual. | Re-scored on the same TT3D protocol: depth-axis coverage inside 0.55-0.80 against the 68.3% a 1σ implies (today 0.29-0.47), residual depth bias \|μ\| ≤0.02 m (today +0.068..0.124 m), image-plane coverage not degraded. | A larger sigma is not a fix; buying depth coverage by over-covering the image plane fails. Do not fit on the 19 owner labels — they are the independent cross-check. |
| 2 | **Retire reprojection gating (in flight).** Remove reprojection error from every gate, band assignment, promotion and kill rule that decides a 3D quantity; keep it only as a 2D-consistency diagnostic, explicitly labelled as one. | An enumeration of every reprojection consumer in the tree, each removed, replaced with a depth-aware criterion, or annotated 2D-diagnostic-only; no 3D promotion path reads it. | Do not swap in a different residual on the same blind axis. Where no depth-aware evidence exists the answer is abstain, not a smaller number. |
| 3 | **Fix the calibration distortion fit — now the measured binding floor.** Fit k1 on the reviewed-15pt path (the zero-distortion config that produced the 19.16px solve), then re-measure the floor by pushing each calibration's own reviewed correspondences through the bounce ray-plane path. | Plane residual improves from the 0.101 m outdoor / 0.127 m wolverine / 0.232 m indoor baseline with no clip regressing; indoor beats its 0.199 m undistort-first figure; `metric_confidence` rises and in/out stops abstaining on the pb.vision demo; reuse-lock residual within ~1px of the per-frame solve. | Raw solves stay immutable. No third synthetic-only retrain; auto court-corner finding stays deprioritized as an authority path (§2.3). |
| 4 | **Sub-frame bounce timing.** The ray-plane anchor is exact to ~1e-15 m when the ball genuinely lies on the plane, yet measured bounce error is 0.091 m median — that residual is frame quantisation, not geometry. Interpolate the bounce instant between frames from the fitted arc and anchor there. | Bounce error improves on the same TT3D protocol with row 1's sigma in place, reported separately from any calibration gain so the two levers stay distinguishable. | Never report a combined improvement. If sub-frame timing does not beat the frame-quantised anchor, say so and stop; the remaining error is then calibration or depth. |
| 5 | **Owner bounce labelling at scale** on the existing tool (`scripts/racketsport/ball_label_studio.py`). Bounce is the only kind with a solved depth, so it is the only kind that can become truth; free-flight and near-player stay review-only estimates. | ≥150 bounce labels across ≥4 source-disjoint clips, each carrying its calibration floor, click sensitivity and realised sigma; a source-disjoint held-out split declared before any scoring. | Human labels stay review-only until an independent capture backs them (NS-02). Never fit a solver on labels it prefilled without reporting the prefill-corrected fraction. |
| 6 | **Event-head scale-up and pickleball fine-tune (E-v2 unblocked).** `SCALE_UP_SPEC.md` levers in order — stage videos → multi-window extraction → dataloader workers; ~68×, $2.2-4.5, 2-5h — then fine-tune on the owner's 102 banked labels. The gate-proof assert that killed three A100 dispatches is fixed (f29145a). | Matched-window eval on ≥50 clips with a threshold sweep; a plausible firing rate (~0.3-1.0/s, not 7.16) BEFORE any anchor is ingested; the protected 50-row owner seed stays eval-only. | Assert the eval window against the checkpoint config at load. No untyped audio anchor is ingested at any coverage (§2.3). |
| 7 | **Finish the court + people skeleton closeout.** Score the committed candidate on the locked 24-moment review (exact player, foot and semantic point), rerun six supported + two refusal videos from source through `process_video.py --force` with `pipeline_preset=court_skeletons` and no upstream reuse, then freeze the winning hashes and evidence index. | Zero false `confirmed_inside_or_on` and zero false `confirmed_outside`, ambiguous cases `unknown`, ≥75% decisive coverage on reviewer-clear planted cases; zero invalid/collapsed courts; four indoor-doubles players retained; zero fabricated/interpolated measured samples; BODY ≥98% of eligible samples; foot slide ≤0.03m per supported clip; non-BODY overhead increase ≤15%. | Never tune on the locked 24; a false decisive call widens abstention, it never moves a skeleton out of the kitchen. Any court/identity fabrication rejects the candidate. External blocker: expired `gcloud` auth (owner action `gcloud auth login`) — do not substitute copied upstream artifacts. |
| 8 | **Fix the association fabrication (P0-I) and build selection layers A/B/C.** Stitch veto + identity re-bind + stop stripping `interpolated` at export, then soft court-presence prior, 4-slot enrollment/open-set rejection, and identity-conditioned recovery from the raw pool. `DESIGN_selection_layer.md` carries the pre-declared thresholds; `player_global_association.py` and `player_id_repair.py` are diagnosed READ-ONLY. | Wolverine 0 spectator FP, 0 switches, IDF1 ≥0.8516; `interpolated: true` survives to `tracks.json`; burlington non-degraded; selection-OFF is byte-identical; cov4 rebuilt from REAL detections or reported honestly lower. | Card rows are GPU-class only — local Mac CPU is not score-faithful for association. No conf-floor or threshold tuning aimed at fabricated FPs (§2.3). |
| 9 | **RF-DETR-L integration lane** (`runs/lanes/trk_rfdetr_integrate_20260717/spec.md`, branch 2b, conf 0.18): detector-injection seam only, zero association changes, kill-switch to yolo26m, preview + `do_not_promote`. Correct the mandated best_stack note first — it calls the wolverine ghosts "HIGH-CONFIDENCE detections", which P0-I forensics disprove. | Four card rows reproduced within 0.0001 through the real production entrypoint on a GPU-class environment. | Runs after row 8: scoring a detector against fabricated FPs measures the wrong thing. |
| Parked | **PADDLE (NS-03.RKT).** No scoreable gate exists. 2026-07-16 research found no dataset combining tiny/blurred paddle 6DoF with contact GT, and the NS-02.1 ≤0.5-frame sync bar is insufficient for contact GT (8.33ms @60fps = 8-17cm of ball travel). | Unparks only when the NS-02.1 metrology capture exists at ≤1ms audio/LED-verified sync with its own held-out error proven. | Do not substitute rectangle IoU, one-solution reprojection, or box orientation for 6DoF (§2.3). The `racket_pose_hypotheses` schema gap is closed before fusion consumes it. |
| Parallel now | **NS-01.2b physical upload proof** plus the P0-H physical 30-second and 5-minute timebase captures. Production app, real device, owner-gated. | A saved record/import → upload → job → manifest → matching replay trace, and monotonic encoded PTS with an aligned sample or explicit drop reason per frame. | Mocked tests are not physical proof; manual artifact substitution fails the gate. |
| Parallel now | **NS-02.1/02.2 gold capture and owner labeling.** Owner half-day per the ns021 checklist; extra cameras remain GT-only. | Surveyed points within 2-3cm, ≤1ms audio/LED sync proof, versioned lane-specific labels with source, frame/PTS, reviewer and uncertainty. | The rig must prove its own held-out error before its labels gate anything. Candidate predictions never become independent GT. |

In-flight background lanes may finish and save their reports, but they do not
supersede this queue. A running process, incomplete report, or speed number does
not change the order above. After NS-02 gates exist, dispatch CAL/TRK/BALL/BODY
as file-fenced parallel NS-03 lanes; NS-04 remains one serialized owner of
`process_video.py`, and NS-05 follows an improved world.

### Superseded stop points — rulings stand, detail lives under `runs/`

Compressed 2026-07-26 under standing rule 13. Every ruling below still binds; its
evidence is unchanged in its lane directory.

- **2026-07-25 court + people skeleton closeout** — the narrow slice is source video → static court/camera lock → measured identities → Fast-SAM-3D-Body MHR70 joints → support-foot anchoring → guarded stabilization → conservative NVZ occupancy → skeleton replay, with no meshes, ball, paddle, audio, events, stats or coaching. Six 10.0-14.8 s clips ran 290.2-447.0 s end to end (median 352.5 s) at revision `60631f1d` while BODY inference itself hit a median 62.1 crops/s, so tracking and cold/remote orchestration dominate, not joint inference. The owner's 24-moment foot/NVZ review is structurally complete (24/24 accepted, 15 foot and 8 court edits) and is a locked selection set, never training data; untouched prelabels are acceptances, not redrawn labels. The July 25 foot/toe repairs still need a fresh timed reproduction and an executable focused test environment. `runs/court_skeleton_runtime_20260725/REPORT.md`, `runs/foot_anchor_stabilization_20260725/`.
- **2026-07-23 court v3.1 selection** — the selected running method is the v2 evidence checkpoint with DARK decoding in legacy stride coordinates feeding the v3.1 regulation-template solver: fold-0 validation structured PCK@5 `0.9516686`, median `1.6673 px`, p95 `4.9310 px`, 74/74 valid topologies over 869 visible labels, zero exact/perceptual/source-group leakage across 370 rows in 67 groups. The 30-point v3 seed-13 screen REGRESSED (PCK@5 `0.2382048` then `0.2094361`) and did not replace the incumbent; seeds 29/47 stay forbidden until a seed-13 candidate beats it. Whole-court confidence ECE is `0.6689`, so the numeric confidence is explicitly uncalibrated — calibrate only from out-of-fold source-disjoint predictions, then build the independent 20-camera-setup pack. Best measured development result, not authority evidence.
- **2026-07-22 spend-limit recovery** — BALL B1 `REJECT` with `DO_NOT_DISPATCH_B1_GPU_RESUME`, B2 `DO_NOT_ARM`; training-data enforcement locally green but unlanded against a cache image baked from older `e1e2184d`; COURT candidate review-only with uncommitted default-deny loader changes; PERSON/BODY needs the selection ultra review, a watchdog fix and one cheap real-Linux Step-0; Track E method rules 7-8 have bounded adopt authority while data-debt and data-hygiene do not. The cache marks protected `83gyqyc10y8f` both `COMPARE_ONLY_NEVER_TRAIN` and `SHA256_MISMATCH`, contrary to its own prose report — staged infrastructure, not clean training provenance. `runs/handoff_20260722/LIMIT_RECOVERY.md`.
- **Scoped passes since 2026-07-09** (named lane dirs; none are promotions): NS-01.1 sidecar contract; NS-01.2a upload route; NS-01.3 run identity and the P0-C close; NS-02 gold-capture and eval reset; NS-01.4 typed coordinates/timebase (P0-D/P0-H); NS-01.5 status and packaging (P0-E/P0-F); NS-01.6 one authoritative stage graph with `pipeline_cli` deleted and exceptions failing loudly; NS-01.7 explicit timed refined stages plus audio soft evidence; NS-05.1 facts core and runner enforcement; Track I placement fusion opt-in; Track K `one_world` stage-185 wiring (2026-07-24).
- **Ball 2D→3D history** — every geometry-only path is killed same-protocol (TT3D, gap-bridging, size-proxy, anchor-fusion, global-track-on-real-pools, direction-break, audio-anchor fusion). The 11-minute pb.vision study decoded their edge as a per-rally global ballistic track plus TRAINED event/radius heads; our 2D coverage already beats theirs, and the wall is trained contact detection. The 41-rally head-to-head is still INCOMPLETE, reattributed to a now-guarded `ball_arc` scaling stall. Cross-reference of Track K's 24 refusals against Track L's fabricated bridges ruled NO-REORDER on 2026-07-19: only 5/24 refusals overlap synthetic frames, 19/24 stand on fully-real frames at 1.6-15.7 m wrist residuals, so the trained-event wall stays the binding ball blocker while the P0-I fix proceeds for trust and honest cov4 (true wolverine cov4 0.520, padding 0.203). `runs/HANDOFF_20260717.md`, `runs/HANDOFF_20260714.md`, `runs/lanes/oneworld_bridge_xref_20260719/XREF_RULING.md`.
- **Speed and labeling** — Wolverine promoted-stack wall 489.4s (×6) with BODY at 78.6% of wall and ~122s of refined-arc solve now an explicit timed stage; a reuse-aware solve is NS-06's biggest lever. Confirm-heavy labeling stopped paying (0.361 → 0.614 at 1k → 0.571 at 3k; ~74.8% byte-identical prelabels inflate scores), so the pivot is uniform-random scratch audit plus hard frames plus venue diversity. The court GT-free harness is frozen and aggressive refinement is PERMANENTLY killed on stability.
- **The through-line still binds:** our cleaning layer fabricates and our own fusion caught it. P0-I is the first time an internal layer was proven to invent measured-looking data; `one_world_v1` is the first component that refused to believe the stack. Fix the fabrication before judging what sits on top.

### Owner-only asks

| Rank | Ask | Why | Safe default while waiting |
|---:|---|---|---|
| 1 | **RECORD A REAL GAME.** One full pickleball game on the product phone, owner-shot | **We own 9.9 seconds of owner-shot pickleball — one static pre-serve clip (`IMG_1605.MOV`), quarantined eval-only (`trainer_forbidden: true`), zero rallies — and nothing else.** Usable owner-shot training footage is zero. Every capability rides on harvested/competitor video; the "39 owned clips" are non-pickleball (content-verified). Owner-shot footage is commercial-clean and provably source-disjoint from both frozen eval clips (which are YouTube-derived) — it unblocks the detector fine-tune, real person labels, and the first honest end-to-end product trace. Nothing else the owner can do is worth more | Every lane keeps scoring on 2 historical-internal clips and calling it scoped; no fresh evidence exists. |
| 2 | **60-second phone test** — record, stop, confirm the file (signed build STAGED at `runs/lanes/ios_recordpath_20260715/device_build/` + `MORNING_SCRIPT.md`) | The dead record button is fixed but UNPROVEN on device: landscape-tap→recording-starts has never run on real hardware. Also carries the NS-01.2b upload trace and P0-H physical timing proof | Simulator/golden fixtures proceed; physical proof stays blocked and the fix stays unverified. |
| 2b | **File the GCP a3/H100 quota increase request** (or approve RunPod H100s) | The project has ZERO H100 quota in every region (swept 2026-07-28); the owner's "prioritize H100" speed directive is structurally impossible on GCP until quota exists. A100 spot fleet works but caps throughput | Fleet continues on A100 spot (16 + 64 preemptible quota) with the warm snapshot + S3 model store for zero-setup boots. |
| 3 | **Bounce labelling round** — ~150 bounce clicks across ≥4 clips in the existing local tool, ~1-2 hours | Bounce is the only ball label kind whose depth is solved rather than estimated, so it is the only route to a falsifiable 3D ball number that does not wait on the gold capture. The 19 pilot labels already caught `arc_weak` output 2.5-24.8 m wrong | Rows 1-4 proceed on TT3D external validation and 19 pilot labels; pickleball 3D accuracy stays unmeasured. |
| 4 | **Event labeling — DONE for now, do NOT continue.** 102 rows banked 2026-07-19 (60 typed + 42 neg; 61 train / 41 val), `data/event_labels_owner_20260719/` | Advisory 2026-07-20 D4: ask for one 50-row uncertainty round ONLY IF the first fine-tune shows median G_val ≥ +0.10 but stays < 0.80 macro-F1; otherwise stop. Fresh owner-shot footage outranks another old-pack round | Fine-tune proceeds on 102; no more owner label time unless the measured gate fires. |
| 5 | **Gold capture half-day** — product phone + 2 high-speed phones, surveyed court/net, ChArUco, paddle markers, **≤1ms audio/LED sync** | Independent truth for CAL/BODY/BALL-3D/RKT/contact, and the only thing that unparks paddle. 2026-07-16 research TIGHTENED the spec: the old ≤0.5-frame bar is insufficient for contact GT (8.33ms @60fps = 8-17cm of ball travel); the rig must prove its own held-out error | All affected lanes remain unverified; engineering prepares tooling. |
| 6 | Approve biometric/video retention and deletion behavior before non-owner persistence; commit owner + one ≥4.0-rated player to the later coaching audit; supply invoice-backed cloud cost during NS-06 | Friend launch and profile reuse; the Usefulness ≥8/10 and fabrication 0/300 gate; honest economics | Session-only non-owner processing with no biometric profile; deterministic facts and rubric only; cost reported as conservative ranges. |

## 6. Standing rules

1. Stay on `main` unless the user explicitly asks otherwise; preserve unrelated dirty work.
2. Read this file, `AGENTS.md`, and the relevant `RUNBOOK.md` section before changing direction.
3. `VERIFIED`, `smoke-verified`, `scoped pass`, `partial`, `review-only`, `rejected`, and `no-attempt` are distinct.
4. Baseline first; score every candidate with the same scorer and frozen dataset/gate.
5. Protected data never becomes training data. Outdoor is historical, not fresh.
6. Raw observations are immutable. Refinements are separate artifacts with provenance/covariance.
7. Explicit user inputs must win over cache. No lexical “latest” imports.
8. Every stage declares coordinate space, timebase, source/model/config identity, and trust band.
9. One integration owner serializes `process_video.py` changes; separate lanes own CAL/TRK/BALL/BODY/RKT files.
10. Expected missing optional evidence may degrade; schema/programming errors fail loudly.
11. Best-stack changes require a named gate pass plus pinned code/checkpoint/training-data provenance and transitive commercial-license review.
12. Visual overlays, smaller residuals, internal validation, copied fixtures, and test green are not accuracy proof.
13. Every completed task writes a dated `runs/` report; root docs do not accumulate wave logs.
14. Volatile coordination lives in `runs/manager/inflight_lanes.md` and `runs/manager/gpu_fleet.md`, not a root checklist.
15. Update this file only when product truth, sequencing, gates, or the active queue materially changes.

## 7. Evidence and history

- Deep code/results/research review and detailed flowcharts: `runs/CV_PIPELINE_DEEP_REVIEW_20260709.md`.
- Multi-agent SOTA review, ranked experiment register, licenses, and kill rules:
  `runs/CV_SOTA_RESEARCH_20260709.md`.
- Exact pre-consolidation plans, checklists, capability tables, blueprints, owner check-ins, and goal
  documents: `runs/archive/root_docs_20260709/INDEX.md`.
- Current selected defaults: `configs/racketsport/best_stack.json`. Checkpoint identity:
  `models/MANIFEST.json`. Held-out preregistration: `runs/manager/heldout_eval_ledger.md`.
- Current transient work: `runs/manager/inflight_lanes.md` and `runs/manager/gpu_fleet.md`.
- Historical research: `runs/research_sota_20260705/` and `runs/research_w6refresh_20260709/`.

Historical files are evidence, not instructions. A future agent should be able
to determine the product goal, current blocker, next task, gate, and stop rule
from this North Star without reading the archive.
