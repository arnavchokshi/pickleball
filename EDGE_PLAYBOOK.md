# EDGE PLAYBOOK — profile-first advantages, domain logic-hacks, exact stack, exact data sources

Owner-requested second pass on `NORTH_STAR_ROADMAP.md` (2026-07-05, refined same day). Purpose:
squeeze accuracy and speed out of things that are TRUE FOR OUR USERS SPECIFICALLY — known courts,
known gear, known players, offline cloud processing, private use.

**Owner ruling (2026-07-05, refined):** the product starts as OWNER-ONLY but must be built so
friends can use it too — **not hardcoded to one person**. The mechanism: every person-specific
advantage below is delivered through a per-user/per-court/per-gear PROFILE captured in an explicit
SETUP PHASE and stored per account; the owner is simply profile #1, and any upload without a
matching profile degrades gracefully to the generic pipeline paths. Licenses remain a NON-constraint
for now (private use; revisit only IF the product ever expands beyond friends — inventory-only,
zero effort today). ⚠️ Unchanged: **held-out eval discipline** (Outdoor/Indoor + ledger
pre-registration). That rule is about not lying to ourselves, not about law — it stays.

Planning artifact; `CAPABILITIES.md` canonical on truth conflicts; `VERIFIED=0` today.

---

# 1. The three unfair advantages (the frame for every hack below)

**A. Profile-first — everything measurable becomes a profile, not a problem (N starts at 1).**
Each user plays on a small set of known courts, with one phone, one-ish paddle, regular partners,
and a consistent ball. Generic-vision problems (court detection, camera calibration, player re-ID,
paddle appearance, ball color) collapse into *capture-once-in-setup, reuse-every-clip* profiles.
Most of the industry's hard problems are things a 15-minute onboarding can simply measure. The
registry that makes this multi-user instead of owner-hardcoded is **H0** below; unknown-profile
uploads (a friend at a new court) fall back to the generic solvers — which is exactly why the
generic lanes in NORTH_STAR (court auto-find Wave B, default paddle template, generic ball prior)
still get built.

**B. Pickleball's rulebook is a free constraint engine.** Court is exactly 20×44ft with 2in-wide
same-color lines; net is 36in at posts / 34in center; the serve is underhand, cross-court, must
bounce; the return must bounce (double-bounce rule); play alternates sides over the net; volleys
are illegal in the kitchen. Every one of these prunes hypotheses in calibration, ball tracking,
event detection, and placement — for free.

**C. We are offline.** No causal filters (no lag), whole-clip global optimization, unlimited
lookahead, compute concentrated where coaching value lives (contact windows), everything cacheable
across runs. Real-time products (SwingVision) can't do any of this; we should never leave it on the
table.

Deep nets only where they pay: after one-time calibration the court/background never sees a deep
model again; the ball gets tiny heatmap nets on motion ROIs; only PEOPLE get big models — and only
the HITTER gets them densely (contact windows), which is already landed as `ball_aware` scheduling.

---

# 2. The logic-hack arsenal

Each hack: what → why it works → where it plugs in (roadmap task ID) → cost → expected win.
Tiers: **S = do first (high win / low cost)**, **A = do soon**, **B = spike/speculative (time-boxed)**.
Expected wins are engineering estimates, not measurements — every hack still passes through normal
lane gates.

## Tier S — profile-once hacks (populated by H0's setup phases)

- [ ] **H0. Profile registry + onboarding setup wizard (the umbrella — makes all of this multi-user).**
  First-class, per-ACCOUNT data model with five profile types, each versioned + trust-banded like
  every other artifact:
  - **Court profile** (per registered court): calibration + distortion pairing, line paint color,
    background model, net geometry, lighting variants (H1/H2/H6 populate it).
  - **Device profile** (per phone/lens/zoom preset): intrinsics + distortion from the ChArUco sweep,
    effective-exposure constant for the blur speedometer (H3/H13).
  - **Player profile** (per person, shareable across accounts for regular partners): measured
    height, frozen MHR shape betas, ReID embedding gallery, handedness, grip seeds (H4).
  - **Gear profile**: paddle scan (mesh+texture+dims, H7) + ball SKU (color window + diameter, H5).
  - **Session cache**: per-session background/lighting/apparel colors, auto-refreshed (H6).
  Delivery = a guided SETUP PHASE in the app ("film the empty court", "film this printed board",
  "orbit your paddle", "enter heights", "pick your ball") — one-time ~15 min per user/court, each
  step optional. The pipeline consumes profiles opportunistically: present → precision path;
  absent → generic path with the corresponding trust band. Storage: per-account server-side (we
  keep and accumulate each user's data — every processed game enriches their profiles: ReID
  galleries grow, backgrounds refresh, grip transforms re-fit).
  → Plugs into: schema in `docs/racketsport/` + ingest (`ingest_owner_capture.py` generalizes to
  `ingest_user_capture`), P0-3, P7-1 (accounts). Cost: ~1-2 lanes for registry+fallback plumbing;
  wizard UI later (owner can populate profiles via CLI first). Win: every hack below becomes a
  product feature instead of an owner-special.
- [ ] **H1. Court profile library (court auto-find becomes court RE-IDENTIFICATION).**
  For each registered court in a user's H0 profile (owner's 2-3 to start), build once: precise
  homography (meticulous manual taps, all
  15 points + net), lens-distortion pairing (H3), dominant line-paint color (H2), background median
  frame (H6), net geometry, GPS/Wi-Fi hint from capture sidecar. On upload: classify which known
  court (image-retrieval embedding + color histogram — trivially separable at N=3), load the frozen
  profile, verify with a cheap 4-line reprojection check, done in seconds. Generic no-tap solving
  (P4-2/P4-3) remains only for NEW courts — i.e., rare.
  → Plugs into: P4 (new task P4-0, do BEFORE Wave B training matters for the owner's own videos).
  Cost: days. Win: near-zero calibration error on ~100% of owner uploads; removes the single
  biggest accuracy tax (calibration noise floor p95 ≈ 19.8px ≈ the whole ball F1@20 radius).
- [ ] **H2. Line-color prior + line-family separation (owner's example, weaponized).**
  All pickleball lines on one court are the SAME paint color (USAP standard, 2in wide); on
  multi-use courts the overlaid tennis/badminton lines are typically a CONTRASTING color. So:
  sample paint color along candidate line segments → cluster → keep the family matching the court
  profile's stored line color; score candidate homographies by color-consistency of all 8 expected
  lines. Directly attacks the two open CAL walls: tennis-overlay confusion (IMG_1605) and
  adjacent-identical-court lock-on (neighbor court = same color but wrong geometry → the color
  test doesn't help there, the profile H1 + top-3 cross-frame vote does; color kills the OVERLAY
  case). Same-color overlays fall back to geometry — honest degrade.
  → Plugs into: P4-2/P4-3 (add color-evidence channel to solver scoring; synth generator v2
  already renders tennis_overlay dual line families — add color randomization).
  Cost: days. Win: IMG_1605-class clips stop failing; overlay courts become tractable.
- [ ] **H3. One-time lens calibration per phone lens (kills a whole failure class).**
  Print a checkerboard/ChArUco board; owner films one 30s sweep per lens/zoom preset; OpenCV
  `calibrateCamera` → intrinsics + k1/k2 distortion stored in the capture profile and applied at
  ingest. IMG_1605's foot-slide FAIL (330mm) was *edge-of-frame zero-distortion* error — this
  deletes the class for every future owner capture (P4-4 then only handles legacy/third-party clips).
  → Plugs into: P0-3 ingest (apply profile), P4-4. Cost: hours + one owner errand. Win: measured
  330mm→expected ≤30mm on edge-of-frame placement; better everything (calibration feeds all stages).
- [ ] **H4. Per-player identity + body profiles (re-ID becomes a lookup).**
  Per player profile (H0; owner + regulars first, any future friend the same way): store OSNet ReID
  embedding galleries, measured real heights (tape measure once),
  per-person locked MHR shape (fit betas from hundreds of frames once, freeze), handedness, typical
  apparel colors per session. Effects: (a) tracking identity = gallery match + court-half prior +
  between-rally re-anchor (identities only need to survive within a rally — between rallies players
  separate and re-ID is easy); (b) measured height resolves monocular scale per person → far-player
  placement depth stops wobbling; (c) frozen betas = constant bone lengths across ALL sessions →
  shape flicker gone, and P2-2 latent smoothing gets a fixed-shape manifold to work on; (d)
  handedness locks the paddle-hand prior.
  → Plugs into: P2-5 (TRK), P2-2, P2-3, P3-1. Cost: days + one measuring tape. Win: ID switches →
  ~0 on owner footage; far-player scale error down; permanent shape stability.
- [ ] **H5. One ball SKU, chosen for contrast (buy the accuracy).**
  Each user/group standardizes on ONE ball model (gear profile, H0), picked to maximize color
  contrast against their registered courts' surface colors (pb.vision's own #1 documented failure is yellow-ball-on-tan-court —
  don't inherit it; e.g. orange/pink on green/blue courts). Store exact diameter (74mm) + HSV/LAB
  color window in the profile → tight color gating for detection, hard-negative suppression, and
  the H10 diameter-depth cue. Cost: $30 of balls. Win: detector recall/precision on owner footage;
  this is the cheapest accuracy purchase available.
- [ ] **H6. Empty-court background models (30s of filming → four superpowers).**
  Owner films ~30-60s of each EMPTY court per session start (or we median-filter it from dead time
  automatically — rally gating already finds dead time). Yields: (1) background subtraction →
  motion masks that boost small-ball detection (RacketVision: background modeling −54-61% error);
  (2) line detection on the CLEAN background (players never occlude lines) → better per-session
  calibration verification; (3) the compositing canvas for H11 synthetic ball data; (4) per-session
  lighting normalization reference.
  → Plugs into: P0-3 capture protocol, P1-3, H11, P4. Cost: trivial. Win: broad.
- [ ] **H7. Paddle = known object: scan it, then match its face texture.**
  The owner's paddle face has a fixed graphic. Photogrammetry-scan the paddle once (P3-7 already
  planned): exact dims + textured mesh. Then on sharp wrist-crop frames, feature-match the face
  graphic (SuperPoint+LightGlue) → planar homography → DIRECT 6-DOF paddle pose from texture — no
  learning, no ambiguity, exactly when the face is visible and unblurred. Use those frames as (a)
  gold pose anchors that re-calibrate the per-segment grip transform continuously (fixes the
  constant-grip assumption), and (b) free training labels for the P3-4 keypoint detector
  (auto-labeling: texture-match frames label themselves).
  → Plugs into: P3-2/P3-3/P3-4 (new sub-task P3-4b). Cost: ~1 lane. Win: absolute pronation
  accuracy where it's measurable + a self-labeling data engine for the frames where it isn't.
- [ ] **H8. Background-keypoint micro-stabilization (tripod ≠ perfectly static).**
  Track fence/pole/net-post corners on the background mask; per-frame ECC/RAFT micro-homography
  update to the court mapping. Handles wind bumps and small tripod drift WITHOUT full SLAM (full
  masked-SLAM per P2-1 stays for genuinely handheld clips). pb.vision breaks on ANY movement —
  this is differentiator #2 made cheap.
  → Plugs into: P2-1(a) as the lightweight mode. Cost: days. Win: silent robustness on every clip.

## Tier S — rules-of-the-game hacks

- [ ] **H9. The double-bounce rule is free ground truth.**
  EVERY rally guarantees: serve → MUST bounce → return → MUST bounce. So the first two shots of
  every rally each contain a certain, roughly-localizable court-plane bounce. Exploit: (a)
  auto-calibrate the bounce detector per session on these guaranteed events; (b) they're perfect
  anchors for the arc solver at rally start; (c) violations = event-detection errors → self-check
  metric; (d) mining owner videos yields hundreds of free bounce labels per session with zero
  human labeling.
  → Plugs into: P1-6 (train/calibrate), P1-4 (anchors), P0-4 (free labels). Cost: days. Win:
  bounce detection stops being heuristic exactly where rallies start; free contact corpus.
- [ ] **H10. Rally grammar decoding (events form a regular language).**
  Legal event sequences are tightly constrained: serve(behind baseline, cross-court, underhand) →
  bounce(receiver's service box) → return → bounce(server side) → then alternating hits with
  optional single bounces, ball crossing the net between posts each exchange, hitter = player
  nearest the trajectory on that side, kitchen volley restrictions. Implement event fusion as
  Viterbi/HMM decoding over (trajectory-kink, audio-onset, wrist-peak) candidates under this
  grammar instead of independent thresholding. False events get grammar-pruned; missed events get
  grammar-inferred with wide bands. Also falls out for free: serve side/score inference, rally
  segmentation, shot counting.
  → Plugs into: P1-6 (this IS the fusion architecture), P6-1/P6-2. Cost: ~1-2 lanes. Win: event
  precision jumps without any new model; the structure is the model.
- [ ] **H11. Composite in-domain ball training data (infinite labels, zero labeling).**
  Render P0-7 simulator trajectories as motion-blurred ball sprites (correct size from H10-known
  diameter + calibration; correct blur from velocity; correct color from H5) composited onto H6
  REAL empty-court backgrounds of the owner's actual courts, plus hard negatives (shoes, line
  intersections, fence glints sampled from his real footage). This is domain randomization with a
  REAL domain — it directly attacks the distractor-lock failure that killed the public-data
  fine-tunes, because the negatives are his court's actual distractors.
  → Plugs into: P1-1 (pre-train on composite → fine-tune on owner labels), P1-2. Cost: ~1 lane on
  top of P0-7. Win: the in-domain data wall gets a second, label-free ladder over it.
- [ ] **H12. Audio is a contact sensor, not a nice-to-have.**
  The pickleball pop is loud, sharp, and distinctive. Onset detection (librosa/madmom) on owner
  captures (audio ON is already protocol) gives near-free contact TIMES; classify paddle-pop vs
  bounce-thud vs voice with a tiny classifier trained on ~500 owner events (P0-4). Snap trajectory
  kinks to onsets for sub-frame contact timing; correct per-side audio lag (~30-60ms across court)
  using the ball's court side. IMG_1605's 30 real onsets are the seed test bed (already banked).
  → Plugs into: P1-6, P3-5 (impact frames), P6 (rally stats). Cost: days. Win: contact timing to
  ≤1 frame; the 40ms gate becomes reachable.
- [ ] **H13. Physics constants measured, not assumed — including a blur-speedometer.**
  Nobody publishes wiffle-ball aerodynamics; measure OURS once: (a) drop the ball from a measured
  height on camera → drag coefficient + restitution vs HIS court surface; (b) one drill clip with
  a radar app or known-speed machine → calibrate the blur-length→speed constant (blur streak
  length = v·t_exposure; BlurBall already outputs blur length+angle) → thereafter EVERY sufficiently
  blurred detection carries its own per-frame speed + direction estimate = a direct 3D-lift
  constraint the arc solver consumes; (c) paddle/ball restitution + friction from slow-mo drill
  impacts → feeds the P3-5 impact-inversion factor with true constants (TT4D's residual error was
  exactly "unmodeled restitution" — we can model ours).
  → Plugs into: P0-7 (simulator constants), P1-4 (per-frame speed factors), P3-5. Cost: one owner
  drill session + days. Win: the physics stack runs on measured truth; spin/speed stats gain a
  calibration story.

## Tier A — compute-shaping hacks (the owner's "double the frame rate at contact" idea, generalized)

- [ ] **H14. Contact-centric fidelity pyramid (formalize what's half-landed).**
  All heavy artifacts concentrate in contact windows ±0.5-1s: dense per-frame hitter meshes
  (landed: `ball_aware`), paddle refinement (P3-6 keyframes), RIFE frame interpolation (H15), 3D
  ball solve refinement. Elsewhere: sparse meshes + skeleton interpolation, cheap detectors, no
  paddle refine. Rationale: 100% of coaching value lives at contacts + positioning between them;
  positioning needs only skeletons/placement, which are cheap.
  → Plugs into: P0-6 (live-prove), P5 (budgets), P6-5 (viewer streams fidelity tiers). Cost:
  mostly landed, needs wiring + proof. Win: accuracy where it matters at flat compute.
- [ ] **H15. Learned frame interpolation at contact windows (RIFE), three uses.**
  Run RIFE (Practical-RIFE) on contact-window crops: (1) DISPLAY: 30→60/120fps buttery replay of
  the hit (extends the landed 2x-FPS viewer button with real interpolated pixels, not just pose
  lerp); (2) ANALYSIS: sub-frame ball positions around the kink → sharper bounce/contact
  localization + better impact-normal estimates; (3) BALL RECALL: run the detector on interpolated
  frames where the real frame missed (ball ghosted by blur) — recovered detections marked
  render-only/derived (trust-band honesty: interpolated evidence never counts as measured).
  → Plugs into: P1-3(e), P1-6, P6-5. Cost: ~1 lane (RIFE is plug-and-play on crops). Win: recall +
  event timing + visible product wow at contacts.
- [ ] **H16. Zero-lag everything (we are offline — never pay causal lag again).**
  Audit every temporal filter for causality lag (one-euro is causal; body-lag-behind-feet was a
  measured symptom). Replace with zero-phase equivalents (forward-backward filtering, RTS
  smoothing, spline fits) everywhere offline. The stance-aware chain already moved this way —
  finish the audit, especially wrists (swing timing must stay 0-frame-delta — harness exists).
  → Plugs into: P2-1(d)/P2-2. Cost: days. Win: lag class of artifacts deleted.
- [ ] **H17. Cascade + cache: cheap-first inference, per-court warm starts.**
  (a) Rally gating already skips dead time on owner captures (eval clips had none — owner footage
  is where it pays); (b) within rallies, run the tiny detector every frame and the full ensemble
  only on uncertainty/disagreement frames; (c) cache per-court/per-session immutables (H1 profile,
  H6 background, H4 galleries, TensorRT engines) so a clip's marginal work is only its rallies.
  → Plugs into: P5-2/P5-3 (+ new P5 task). Cost: days. Win: owner-clip wall time drops well below
  the eval-clip numbers (dead time is 50-70% of real game video).
- [ ] **H18. Whole-rally joint refinement pass (the offline endgame).**
  Final polish per rally: one factor-graph/bundle optimization (ceres/GTSAM) jointly over ball arc
  segments + contact times (audio-anchored) + paddle poses at impacts + hitter wrist trajectory,
  with the physics + grammar constraints. Each subsystem already produces the factors; the joint
  solve reconciles them (e.g., contact time moves 1 frame → arc + paddle + wrist agree). This is
  the "everything pairs hand-in-hand" owner principle as an optimizer.
  → Plugs into: new P1/P3 integration task after P1-4 + P3-5 land. Cost: 1-2 lanes. Win:
  consistency users can SEE (ball meets paddle exactly); error stops compounding stage-to-stage.

## Tier A — body/placement extras

- [ ] **H19. Far-player high-res re-crop pass** (roadmap P2-3, kept here for completeness: upscale
  far-player crops ≥2× before SAM-3D; optionally Real-ESRGAN — evaluate hallucination risk with
  A/B on wrist-swing timing). Combined with H4 height locks, far players stop being "worst".
- [ ] **H20. Kitchen-line + court-region priors as placement sanity.** Players are ~never inside
  the kitchen during volleys and mostly behind the baseline at serve; feet within court+margin
  bounds during rallies (spectator/adjacent-court exclusion already uses membership — generalize to
  a soft court-region prior on placement, flagging violations instead of hard-forcing).
  → Plugs into: P2-5, P6-2 (kitchen stats reuse the same computation). Cost: small.
- [ ] **H21. Shot-type-conditioned pose priors (later, once P6-1 classifies).** A dink, drive, and
  serve have distinct wrist/torso envelopes; feeding the classified shot type back as a weak prior
  on pose smoothing at contact windows stabilizes exactly the frames that matter. Circular
  dependency handled by two-pass processing (classify on pass 1, refine on pass 2 — we're offline).
  → Plugs into: P2-2/P6-1 second pass. Tier A only after P6-1 exists.

## Tier B — spikes (time-boxed, adopt only on measured wins)

- [ ] **H22. Paddle blur-streak axis as swing-plane cue** (novel — nobody has published it for
  elongated implements; P3-8). One-lane spike on contact-window frames.
- [ ] **H23. Ball diameter-depth cue [ALREADY LANDED — tech-audit: `enable_size_depth_residual=True` is default-on in all three arc-solver fit paths; keep for reference].** SoccerNet-v3D (arXiv:2504.10106) does exactly this and improved detector IoU 0.57→0.66 (size error 19%→7.3%) — reuse its Eq.5-7. Known 74mm diameter + calibrated focal length → apparent
  diameter gives depth along the ray. Noisy at 10m+ (ball ≈ 10-15px at 1080p) but integrated over
  an arc it's a real monocular-3D constraint; at 4K near-court it's strong. Feed as a weak
  per-detection depth factor into P1-4 (the solver weighs it by pixel size).
- [ ] **H24. Shadow as a second view.** Sunny outdoor sessions: the ball's/player's shadow on the
  known court plane is a projection from a second "camera" (the sun, direction constant per
  session and computable from time+GPS). Ball height = f(ball px, shadow px, sun angle). Fragile
  (shadow detection), but on clean sunny clips it's free 3D. Spike after P1-4 to measure value.
- [ ] **H25. Slow-mo drill mode as a data product.** Owner records 1080p240 drills monthly:
  feeds H13 constants, spin studies (P1-5), contact GT, paddle-impact GT — 10 minutes of drills =
  a quarter's worth of precision labels. (Protocol addition, not runtime.)
- [ ] **H26. Score/serve-side inference from positions** (double-bounce + side-switching rules +
  server position determine score progression) → free scoreboard + rally importance weighting for
  coaching ("game point errors"). After P6-1.

---

# 2b. iPhone capture hacks — ALL video comes from iPhones; mine the device

**We have our own app — build the platform around that assumption.** `ios/` is a real 110-Swift-file
app (7 modules: Capture, Core, Calibration, FastTier, Guidance, Upload, Replay) and the capture
sidecar contract (`CaptureSidecar.swift`) ALREADY carries per-frame **intrinsics, ARKit camera pose,
gravity, court plane, locked exposure/ISO/focus/WB, LiDAR refs, and capture modes incl. `ballPhysics240`
(240fps) + `swing120`** — plus a CoreMotion gravity sampler, camera-roll importer, and live overlays.
The server ingest already reads the sidecar (provenance, intrinsics fingerprint, taps) and
`metric_calibration_from_sidecar_and_keypoints` already consumes ARKit pose/plane/intrinsics. BUT
(tech-audit 2026-07-06): zero `import ARKit` exists in ios/ — the ARSession itself is UNBUILT
(schema-only); only CoreMotion gravity is real. So the work is (a) BUILD the ARKit session +
real-device proof, (b) profile-capture flows, (c) per-frame-ify the sidecar (P0-10).
Two tiers: **Tier 1** = any file including stock-Camera video from friends (parse metadata, correct
VFR); **Tier 2** = video through our app (the rich sidecar). Both matter — friends may sideload stock
clips before they install the app. Consolidated as roadmap task **P0-10**.

## Tier 1 — any iPhone video file (parse at ingest; mostly free)

- [ ] **H27. QuickTime metadata harvest at ingest.** Parse per-file: GPS (`com.apple.quicktime.location`)
  → auto-select H1 court profile; creation timestamp + timezone → sun position for H24 shadows +
  lighting-variant selection in the court profile; camera model + lens type → device-profile lookup
  (H3 intrinsics); HDR flag (HLG/Dolby Vision) → correct color-transfer conversion BEFORE ball/line
  color gating (or it silently breaks H5/H2 color windows). → Plugs into: P0-3 ingest. Cost: days.
- [ ] **H27b. VFR correctness (not a hack — a latent bug class).** iPhone video is VARIABLE frame
  rate; any stage assuming constant fps drifts on timing (contact timestamps, velocity estimates,
  audio-video alignment). Ingest must emit a PTS-accurate frame-time table and every stage consumes
  real timestamps (audit: frame extraction, arc solver dt, wrist-peak timing, event fusion).
  iPhone A/V timestamps are mutually consistent — once on PTS, audio-contact fusion (H12) gets
  sub-frame alignment for free. → Plugs into: P0-3 + an audit lane. Cost: days. Win: correctness.
- [ ] **H27c. Capture guidance for stock-camera users (till the app exists).** Written protocol
  mirrors what our app will enforce: SDR (HDR off), Enhanced Stabilization OFF, no mid-clip zoom,
  4K60, AE/AF lock (long-press). pb.vision can only ASK users for this too — but our profiles +
  fallbacks tolerate violations instead of silently degrading.

## Tier 2 — recorded through our capture app (the real unlock)

- [ ] **H28. Capture sidecar (the container — CONTRACT ALREADY EXISTS, extend + consume).**
  `ios/Core/.../CaptureSidecar.swift` already defines intrinsics, ARKit camera pose, gravity, court
  plane, locked exposure/ISO/focus/WB, LiDAR refs, capture modes, provenance. Remaining: make it
  PER-FRAME where it's currently per-clip (exposure/intrinsics/pose can drift), add PTS + GPS +
  thermal if missing, and — the real gap — CONSUME the geometry fields server-side (today's ingest
  only uses intrinsics for a fingerprint; ARKit pose + gravity are written by the app and dropped by
  the server). → Plugs into: P0-10, consumed by P0-3/P2-1/P4. This one artifact supersedes chunks of
  FOUR server-side problems below.
- [ ] **H29. ARKit camera pose + gravity during recording → handheld solved AT THE SOURCE.**
  ARKit world tracking (VIO: fused IMU+vision; 4K video capture supported on recent devices) gives
  per-frame camera extrinsics + gravity direction + (via plane detection) the ground plane and
  camera height in METRIC scale. Effects: (a) handheld clips stop needing server-side masked-SLAM
  (P2-1's heavy path becomes the fallback for stock-camera video only); (b) gravity is EXACTLY the
  quantity world-grounded HMR methods struggle to estimate (GVHMR's core trick) — we get it from
  the IMU for free, feeding world grounding + placement directly; (c) camera height + ground plane
  = an independent check on court-line calibration. Fallback when ARKit degrades (fast pans):
  raw gyro/accel log still enables rolling-shutter correction + H8-style stabilization.
  → Plugs into: P2-1, P4, placement. Cost: the P0-10 app + ingest consumption. Win: the entire
  handheld failure class (IMG_1605, 330mm slide) becomes a non-event for app-captured video.
- [ ] **H30. Per-frame exposure → the blur speedometer reads exactly.** H13 calibrates
  blur-length→speed with an assumed shutter constant; the sidecar's per-frame `exposureDuration`
  makes it exact per frame (speed = blur_len_px × depth_scale / exposure_s), and ISO/exposure
  changes stop poisoning the estimate. Also: lock AE/AF/WB during rallies from the app (focus
  breathing changes intrinsics; WB drift breaks color gates) + stabilization mode `.off`
  (EIS warps frames non-rigidly — silently corrupts calibration; locked off = clean geometry).
  → Plugs into: H13/P1-4 factors, H5/H2 color stability. Cost: included in P0-10.
- [ ] **H31. Stereo/spatial audio lateralization (spike).** iPhone records stereo (spatial on
  recent models); a paddle pop's inter-channel delay/level gives a coarse LEFT/RIGHT side cue →
  disambiguates which side contacted when trajectory is ambiguous, tightens H10 grammar decoding.
  Mic baseline is small so treat as a weak prior only. → Plugs into: H12/P1-6. Tier B effort.
- [ ] **H32. Two-iPhone GT rig (friend's phone = second camera).** Our app on two phones, absolute
  timestamp sync (NTP/clock log in sidecar) → an ad-hoc synchronized stereo pair for SETUP/GT
  sessions only: triangulated ball 3D + joint GT for P2-6 world-MPJPE and P1-4 3D-lift validation —
  without any special hardware. Product stays single-camera; this is a measurement tool.
  → Plugs into: P2-6, P1-4 eval. Cost: small once P0-10 exists. Win: independent GT, the thing we
  currently lack most.
- [ ] **H33. LiDAR-assisted setup scans (Pro models).** During H0 onboarding: walk the court once →
  LiDAR plane + corner distances verify/refine the court profile in metric scale + net-height
  check; paddle scan gets true-scale geometry (photogrammetry scale ambiguity gone).
  → Plugs into: H0 wizard, H1, P3-7. Cost: small, Pro-only (graceful skip otherwise).
- [ ] **H34. Capture-mode playbook (what the app selects per activity).**
  | Mode | Setting | Used for |
  |---|---|---|
  | Game (default) | 4K60 SDR, EIS off, AE/AF/WB locked, ARKit+sidecar on | everything |
  | Drill / GT | 1080p240 slow-mo (4K120 on newest Pros) | H25 contact/spin GT, H13 constants |
  | Setup scans | 48MP stills + LiDAR walk | H0 profiles (court, paddle) |
  | Long sessions | monitor `thermalState` in sidecar; prompt shade/segment breaks | outdoor 4K60 throttling reality |

---

# 3. Exact technology bill of materials (per stage)

License column removed by owner ruling — "src" notes what's already local. Choices marked ✅ are
already vendored/landed; 🔜 = to bring in; the fallback column is the pre-registered plan-B.

| Stage | Choice (exact) | Where it lives / comes from | Fallback |
|---|---|---|---|
| Ball detect ensemble | ✅ WASB-SBDT (tennis ckpt anchor) + ✅ blurball fork (training + blur cues) + ✅ TOTNet (occlusion recipe: 4-level visibility-weighted BCE + occlusion aug TOGETHER) + TrackNetV4 vendored; motion-prompt re-implementation NOT started (tech-audit 2026-07-06) | `third_party/{WASB-SBDT,blurball,TOTNet,TrackNetV3}`; ckpts `models/checkpoints/` + ledger sha256s | RF-DETR-S (Roboflow, Apache) as an architecture-diversity check only |
| Ball data engine | H11 compositor (ours) + SST teacher-student (github.com/rvandeghen/SST recipe) + CVAT | P0-7 sim + H6 backgrounds; owner labels via `ingest/prelabel_owner_capture.py` | pure owner-label fine-tune (P1-1 without SST) |
| Ball 3D lift | Ours: P3-A BVP/IRLS solver + drag-Magnus ODE (scipy RK4, H13 constants) + H13 blur-speed factors + H23 diameter-depth + net/court constraints; then learned lift: UpliftingTableTennis-style transformer (github.com/KieDani/UpliftingTableTennis, code released) retrained on P0-7 pickleball sim | solver in `threed/racketsport/ball_arc_*`; sim = P0-7 | TT4D-style lift-first transformer (paper recipe; no code) |
| Events/contacts | H10 rally-grammar Viterbi (ours) + H12 audio onsets (librosa/madmom) + wrist-peak cues (landed) + H9 double-bounce calibration | new `threed/racketsport/` module; audio already in owner protocol | TTNet-style learned event head once labels accumulate |
| Court/CAL | H1 court-profile library (ours) + ✅ court_unet_v2 (staged trainer) + ✅ PnLCalib SV_kp/SV_lines + ✅ TennisCourtDetector + ✅ DeepLSD/ScaleLSD (all local in `models/checkpoints/court_external/`) + H2 color channel + H3 ChArUco intrinsics (OpenCV) | Wave A branch + `runs/lanes/cal_*_20260705/` specs | manual 15-pt tap path (v1, always works) |
| Person track | ✅ YOLO26m + BoT-SORT + OSNet ReID w/ H4 persistent galleries | in-repo | CoMotion (Apple) bench; RAM when code drops |
| Body | ✅ SAM-3D-Body / Fast-SAM-3D-Body (locked backbone) + P2-2 MHR-latent smoothing (build; blueprint arXiv 2512.21573) + SMART recipe: RAFT-small + MAD camera track, per-track shape lock (H4), MAD+Gaussian two-pass + H16 zero-phase audit | `third_party/Fast-SAM-3D-Body`; smoothing in `worldhmr.py`/`pose_temporal.py` | challenger bench P2-7: GVHMR (zju3dv), PromptHMR-Vid, WHAM — SMPL now unblocked by owner ruling |
| Hands | 🔜 WiLoR (github.com/rolpotamias/WiLoR, detector+MANO) on wrist crops | P3-3 lane pulls weights | HaMeR / keep MHR fingers |
| Paddle | ✅ YOLO26s boxes + ✅ seg ckpt (`runs/rkt_train_20260702T072800Z/seg_yolo_external_split/`) → oriented-quad → IPPE (github.com/tobycollins/IPPE) + P3-4 RTMPose-s 5-keypoint head (RacketVision schema) trained on H7 auto-labels + owner CVAT + H7 SuperPoint+LightGlue (github.com/cvg/LightGlue) face-texture homography + P3-5 impact inversion (ours, H13 constants) + nvdiffrast keyframe refine | fused estimator landed (`paddle_pose_fused.py`) | box+palm fusion (current, works today) |
| Frame interp | 🔜 Practical-RIFE (github.com/hzwer/Practical-RIFE) on contact-window crops | H15 lane | FILM (Google) |
| Far-player SR | bicubic 2× first; 🔜 Real-ESRGAN if A/B wins | P2-3 lane | none (crop-only) |
| Joint refine | 🔜 ceres-solver or GTSAM factor graph (H18) | after P1-4+P3-5 | keep per-stage solves |
| Speed | TensorRT FP16 engines (P5-3) for WASB/TrackNet/YOLO26; per-court engine+profile cache (H17) | VM build | ONNX Runtime CUDA |
| Coaching | Claude API (grounded 3-stage per P6-4) + THETIS-pretrained stroke classifier features + our 3D facts | P6 lanes | rule-only reports (no LLM) |
| iPhone capture | ✅ `ios/` capture/sidecar/upload scaffolding → P0-10 capture-logger app: AVFoundation per-frame intrinsics + exposure, ARKit world tracking (pose+gravity+planes), CoreMotion IMU, LiDAR (Pro), stereo audio; ffprobe/pymediainfo QuickTime metadata parse at ingest (H27) | `ios/` modules + P0-10 lane | stock Camera app + H27 metadata harvest + H27c written protocol |

---

# 4. Exactly where the data comes from

## 4.1 Capture protocol v2 (owner = first user; ⭐ items are exactly the H0 setup-wizard steps any
future user repeats once)

| Item | Spec | Feeds |
|---|---|---|
| Game video (the staple) | 4K60 (or 1080p60 min), landscape, tripod ≥5ft, whole court + all corners, **audio ON**, both courts, varied times of day | everything |
| ⭐ Lens calibration sweep | printed ChArUco board, 30s sweep per lens/zoom preset used (SUPERSEDED per-clip by H28 sidecar intrinsics once video comes via our P0-10 app — keep the board for stock-camera video) | H3 → all geometry |
| Empty-court clip | 30-60s per court per session start (or auto-mined from dead time) | H6 → H11, P1-3, P4 |
| ⭐ Player heights | tape-measure owner + regulars once | H4 → P2-3, placement |
| ⭐ Paddle scan + GT | photo orbit of paddle (texture+dims) + 4 corner-marker clips + a few slow-mo impacts | H7, P3-5, P3-7, RKT gate |
| ⭐ Physics constants | ball drop from measured height; one known-speed drill (radar app); bounce on each court surface | H13 → P0-7, P1-4/5, P3-5 |
| Slow-mo drills (monthly) | 1080p240, 10 min: serves, drops, drives, dinks | H25 → contact/spin GT |
| Handheld clips (few) | deliberately handheld games | motion-tolerance eval set (P0-5) |
| Deliberate negatives | clips with: second ball on ground, spectators, adjacent-court play, bags/shoes near lines | hard negatives for H11/P1-1; TRK spectator gates |

Ingest: `scripts/racketsport/ingest_owner_capture.py` → role assignment AT INGEST
(train / internal-val / held-out; ≥2 new held-out WITH AUDIO) → `prelabel_owner_capture.py` → CVAT.

## 4.2 External corpora (owner ruling: pull freely; keep the one-line inventory)

| Corpus | What we take | Feeds |
|---|---|---|
| **Online pickleball video (owner 2026-07-06: harvest instead of only self-recording) — YouTube PPA/MLP/APP VODs + amateur/vlog TRIPOD games** (amateur camera profiles match ours better than broadcast) | yt-dlp bulk pull → auto-clip rallies → prelabel (P0-1b) | training (SST P1-2, aux fine-tune, stroke-classifier P6-1), reference-range stats (P6-3), AND broad testing. NOT in-domain (their cameras) → diversity/pretrain/test, owner captures still finish. Exclude competitor-processed clips; reserve a couple as held-out |
| RacketVision (github.com/OrcustD/RacketVision; 435k frames, ball + 5-kp racket, 3 sports) | multi-sport AUX joint-training data (+14-19% mAP evidence); racket-keypoint pretraining | P1-1, P3-4 |
| TrackNet tennis/badminton sets + WASB 5-sport zoo data | aux ball data | P1-1 |
| Roboflow — ALL pickleball projects (ball/court/player/paddle), aggregated + deduped (licenses no longer a gate). NOT the old 8.6k-only pull that distractor-locked; the fix is MORE sources for diversity. Measured (`runs/lanes/ball_t4_train_20260704/`): public-only fine-tunes DEGRADE held-out → use as PRETRAIN/aux only, owner data is the finisher | ball detect pretrain (P1-0→P1-1), court kp (P4-2) | P1-0 harvest (owner: re-issue API key) |
| SoccerNet-Calibration + WorldCup2014 | line/kp heatmap pretraining for court_unet_v2 | P4-2 |
| THETIS (tennis strokes) + OpenTTGames/TTNet | stroke + event pretraining | P6-1, P1-6 |
| TT3D/TT4D/UpliftingTT synthetic + code | 3D-lift architecture + training patterns | P1-4 |
| AMASS (+ SMPL now unblocked) | motion prior for challenger bench + any learned smoothing experiments | P2-2/P2-7 |
| FreiHAND/InterHand2.6M via WiLoR weights | hands (pretrained — just use the weights) | P3-3 |
| P0-7 simulator + H11 compositor (OURS) | infinite labeled ball 2D↔3D+spin pairs on real backgrounds | P1-1/P1-4/P1-5 |
| CAL-SYNTH generator v2 (OURS, landed) | court-line synthetic incl. tennis overlays (+ H2 color randomization) | P4-2 |
| Our own product outputs over time | every processed owner game accumulates pseudo-labels, contact events (H9/H12), reference ranges | the flywheel: P1-2, P6-3 |

## 4.3 What we deliberately do NOT source
Public "pickleball datasets" that are single-match/unlicensed-blog artifacts (Rochester-class —
proven wall); TrackNetV4 upstream weights (undeserializable); anything that would put Outdoor/
Indoor/held-out content into training (discipline unchanged).

---

# 5. Roadmap deltas this playbook creates

Applied to `NORTH_STAR_ROADMAP.md` thinking; execute via these task adjustments:
0. **P0-9 (new):** H0 profile registry — schemas (`docs/racketsport/`), per-account storage,
   opportunistic profile consumption + generic-path fallback plumbing, `ingest_user_capture`
   generalization; wizard UI deferred to P7-1 (CLI-populated profiles until then).
0b. **P0-10 (our app EXISTS — prove + wire, not build):** `ios/` (110 Swift files) already records
   the H28 sidecar (intrinsics, ARKit pose, gravity, court plane, locked exposure, LiDAR, 240fps).
   Work: (a) physical-device capture proof; (b) SERVER consumption of the ARKit-pose/gravity/exposure
   fields it currently drops (H29/H30); (c) per-frame-ify the sidecar; (d) add H0 profile-capture
   flows to the capture UI. H27b PTS/VFR-correctness audit is MANDATORY server-side regardless.
1. **P4-0 (new, before P4-2 matters for owner clips):** H1 court-profile library + H2 color channel
   + H3 intrinsics-at-ingest. Wave B training (P4-2/3) remains for new/unknown courts + IMG_1605-class.
2. **P0-3/P0-4:** capture protocol v2 (§4.1) replaces the sketch; add H9 free-bounce mining and H12
   audio-onset seeding to the labeling factory.
3. **P0-7:** simulator gains H13 measured constants + H11 compositor as an explicit deliverable.
4. **P1-3:** add (e) RIFE-interpolated-frame recovery (H15) to the recall levers.
5. **P1-6:** implement as H10 rally-grammar Viterbi fusion + H12 audio + H9 self-calibration.
6. **P2-1:** lightweight H8 background-ECC mode as the default static-clip path; masked-SLAM only
   for handheld.
7. **P2-5:** H4 galleries + H20 court-region priors.
8. **P3-4b (new):** H7 face-texture homography anchors + auto-labeling.
9. **P5-x (new):** H17 per-court cache + cascade inference.
10. **P1/P3 integration (new, after P1-4+P3-5):** H18 whole-rally factor-graph polish.
11. **Owner list additions:** ball SKU choice (H5), ChArUco sweep (H3), heights (H4), empty-court
    +physics-constants+slow-mo drill sessions (H6/H13/H25). SMPL action item: deleted (ruling).

*Maintained alongside NORTH_STAR_ROADMAP.md; same rules: checkboxes = work items, evidence decides.*
