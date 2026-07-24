# Pose-Conditioned Synthetic Rally Generator — Interface Specification (design only)

Date: 2026-07-23. Status: DESIGN DOCUMENT, no code, no model, no solver change.
`VERIFIED=0`. Freeze-safe prep item 2 of 2 (owner-approved 2026-07-23) under the
program freeze in `runs/ball3d_lifting_plan_20260723/PLAN.md` (v2 reframe).

HOLD conditions (both must clear before any implementation beyond this document):
1. **Codex K,D,R,t** — the court lane's full metric calibration artifact
   (K, D, R, t, Σ_camera per PLAN A-2). The generator's projection stage is
   specified against it; no substitute homography-only path will be built.
2. **Gate 2.2 / Phase-A exit** — the reproducible metric report for the CURRENT
   physics-only solver (PLAN v2 "PROGRAM DECISION"). No training use of any
   generator output before Phase A exit gates hold.

## 0. Contract baseline

This spec is written against the landed A-3 contract module, present in this
worktree: `threed/racketsport/ball_metric3d_contract.py` (schema_version 1,
`WORLD_FRAME = "court_netcenter_z_up_m"`, artifact types
`racketsport_ball_metric3d_gt_observations` and
`racketsport_ball_solver_observation_log`). All generator outputs serialize
through that module's shapes and its deterministic serializer
(`dumps_contract_json`: sorted keys, 6-decimal floats, no timestamps). If the
contract moves, this spec re-syncs to it, not the reverse.

## 1. Purpose and position in the plan

PLAN §5.B1 pose-conditioned simulator: use REAL player tracks → feasible contact
volumes → sample contact + outgoing velocity → simulate flight → project through
REAL calibration → inject MEASURED detector noise/occlusion. The generator is an
**observation-space** generator (it emits tracks, rays, confidences, and event
labels — never pixels/rendering). Its sole customer is the Phase-B ray-to-height
lifter's training pipeline (B4 steps 1–2); it is not a 2D-detector data source.

## 2. Inputs (all versioned, all recorded as `SourceArtifact{kind, path, sha256}`)

| # | Input | Concrete form | Notes |
|---|---|---|---|
| I1 | Real player tracks | Production-stack player root-on-court + joints (wrist/elbow/shoulder minimum) and paddle 6-DOF (`paddle_pose_fused` family), time-aligned to native frame timestamps | Real motion is the point: contact volumes and occluder silhouettes come from these, never from synthetic humans |
| I2 | Real calibration artifact | Court-lane K, D, R, t + Σ_camera (**HOLD 1**; the existing `court_calibration_metric15pt.json` homography-level artifacts are NOT sufficient for ray generation and are not an accepted substitute) | Frozen before use; the generator never optimizes calibration |
| I3 | Court/net template | Metric court + net model in `court_netcenter_z_up_m`: x width, y baseline (net plane y=0), z up, surface z=0, measured net-top curve h_net(x), ball radius r_ball (`BALL_RADIUS_M = 0.0371` in `ball_arc_solver.py`; per-ball override allowed) | Same frame the contract validates |
| I4 | Measured detector-noise model | `runs/ball_lane_20260723/lifter_prep/detector_noise_model/residual_analysis.json` (artifact_type `racketsport_ball_detector_noise_model`) | See §4 for exactly which fields drive injection |
| I5 | Generator config + seed | Single JSON config (shot-family mix, fps set, rally-length distribution, regime mix, noise-model version pin) + integer seed | Same (config, seed) → byte-identical outputs |

## 3. Sampling pipeline (stage contract, not implementation)

1. **Contact sampling** — from I1: candidate hitter at time t, reachable-contact
   volume from root+shoulder+elbow+wrist+paddle pose; sample contact point and
   outgoing velocity (speed/elevation/azimuth per shot family). No spin state.
2. **Flight simulation** — gravity + quadratic drag ODE only, using the existing
   solver's constants (`PhysicsParameters`: gravity 9.80665 m/s²; drag_cd 0.33
   outdoor / 0.45 indoor; drag_k_per_m from ρ_air, Cd, area, mass), RK4, plus
   simple restitution/friction bounce at z=r_ball and optional per-clip constant
   wind. This matches PLAN §5.C6 V1 physics exactly — the generator must never
   embody physics the solver does not.
3. **Event labeling** — continuous-time hits, bounces, net crossings/contacts,
   apex, dead-ball, each with the sub-frame time and world position.
4. **Projection** — through I2 (K, D, R, t including distortion) at the target
   fps with timestamp irregularity; optional calibration perturbation sampled
   from Σ_camera.
5. **Noise/occlusion injection** — per §4, driven only by I4 measurements.
6. **Serialization** — per §5.

## 4. Noise injection ⇄ deliverable-1 field mapping (binding)

| Injected effect | Source field in `residual_analysis.json` |
|---|---|
| Inlier localization jitter (per-axis bias + σ, near-Gaussian) | `by_set.A_eval_clips_independent_clicks.inliers_le_20px.{dx,dy}` |
| False-peak / mis-association mode (rate + court-scale displacement quantiles) | `...outliers_gt_20px.{rate, displacement_px}` |
| Miss (visible ball, no detection), per regime | `by_regime.{indoor,outdoor_day,outdoor_night}.rates.miss_rate_at_labeled_visible` |
| Confident false positives while ball not visible, per regime | `by_regime.*.rates.{fp_rate_at_labeled_hidden, fp_confidence}` |
| Confidence values on emitted detections | `pooled_all.confidence_bins_independent_pairs` (inverse sampling) |
| Missing-detection gap lengths (short mode + long tail), per clip family | `per_clip.*.missing_detection_gaps.{gap_frames_histogram, gap_ms}` |
| Structured occlusion (player/paddle/net-projected) | Geometry from I1 silhouette capsules; DURATIONS constrained to the measured gap histograms above (no occlusion-cause labels exist yet — flagged gap in RESIDUALS.md §7) |
| Blur marginal (feature channel only, not residual conditioning) | `blur_association.blur_length_px` (n=36 showed no residual association; do not couple jitter to blur from current evidence) |

FP placement follows PLAN §5.B1's structured list (near shoes/lines/paddle
edges/lights/other courts) using I1 geometry; FP *rates and confidences* come
only from I4.

## 5. Outputs (A-3-compatible, one synthetic rally = one clip id `synth/<config_sha8>/<seed>/<n>`)

- **O1 Ground truth**: `GroundTruthObservationSet` — strictly increasing
  `timestamp_s`, `xyz_world_m` from stage 2, `sigma_xyz_m` = fixed small
  simulation floor (>0 as the contract requires), `cameras_used =
  ["synth_cam_0"]`, `triangulation_residual_px = 0.0`, `quality_flags = []`
  (empty is contract-valid; real-review flags such as `reviewed`/`gold` are
  never claimed). Synthetic provenance is carried by the clip-id namespace and
  the sidecar manifest, since `KNOWN_QUALITY_FLAGS` has no `synthetic` entry.
  **Contract extension request (non-blocking): add a `synthetic` quality flag
  in schema v2.**
- **O2 Observation stream**: `SolverObservationLog` — per frame:
  `pixel_xy`/`pixel_confidence` after §4 injection; `observation_status`
  mapping: clean detection → `observed`; jitter-inlier low-conf (<0.7) →
  `weakly_observed`; injected FP or false-peak coincident with a true-ball
  frame → `ambiguous`; miss/gap → `missing` (never 2D-splined — PLAN A3);
  `inferred` unused by the generator. `ray` computed from I2 with `ray_status =
  "computed"` on observed frames, else `"no_pixel"`; `candidate_summary.candidate_count`
  = 1 + injected FPs that frame; `inputs` = the full I1–I5 `SourceArtifact`
  list. `solver_verdict` is semantically "no solver ran" — schema v1 has no such
  verdict, so v1 emission uses `"hidden"` on every frame with the meaning
  documented in the sidecar manifest. **Contract extension request
  (non-blocking): add `not_evaluated` to `SOLVER_VERDICTS` in v2.**
- **O3 Event labels**: emitted as `AnchorEvent` rows on the frames they occur
  (`anchor_id` = deterministic non-empty string `"synth_<rally_id>_<event_seq>"`
  per the contract's required field, `kind ∈ {paddle_hit, bounce, net_crossing,
  net_contact, apex, dead}`,
  `status = "synthetic_truth"`, `source = "pose_conditioned_generator"`) plus a
  sidecar `events.json` with continuous sub-frame times, hitter id, and contact
  point — the per-event supervision (hit logits, sub-frame offsets) B3 needs.
- **O4 Conditioning tracks**: the I1 player/paddle tokens re-emitted time-aligned
  to O2 timestamps (pass-through + resampling record), so training consumes one
  bundle. Sidecar `conditioning.json`, shapes in §6.
- **O5 Manifest**: config, seed, noise-model sha, calibration sha, counts, and
  the deterministic-serialization statement. No timestamps.

## 6. Gated cross-attention I/O contract (PLAN §5.B3, RacketVision-style)

The generator must emit exactly the tensors this fusion consumes:

- **Ball tokens (queries)** — per frame t: [ray direction or (Δu,Δv)-normalized
  pixel, exact-time rotary phase from `timestamp_s`, pixel_confidence,
  observation-status one-hot (with learnable missing token substituting the
  ball embedding ONLY — camera/court context embeddings persist through gaps),
  blur feature channel]. From O2.
- **Player/paddle tokens (keys/values)** — per player p, frame t:
  [root-on-court xy, wrist/elbow/shoulder positions, paddle 6-DOF pose +
  angular velocity, contact-centered features: ball-ray-to-paddle-plane
  distance, ball-ray-vs-paddle-center offset, reachable-contact probability,
  time-from-nearest-audio-onset (real data only; null channel on synthetic),
  time-from-inflection]. From O4.
- **Gate** — scalar g_{t,p} ∈ [0,1] = hit-probability for player p near frame t;
  fused ball token = ball_token_t + Σ_p g_{t,p} · CrossAttn(Q=ball_token_t,
  K/V=player_p tokens in a ±250 ms window). **Not raw concatenation.** Gate
  supervision target = O3 `paddle_hit` events (1 within the event's sub-frame
  window, 0 elsewhere); the gate must be able to drive to 0 so a biased paddle
  estimate cannot drag the ball (PLAN v2 caution 2).

## 7. Non-goals (binding)

- **NO spin, NO Magnus, no aero beyond gravity+drag** — killed on pre-registered
  RMSE regression; Phase E owns any controlled revisit.
- **No training of anything until Phase A exit gates hold** (HOLD 2). This
  document authorizes zero training runs.
- No pixel rendering, no synthetic humans, no camera-mount optimization, no
  2D-detector augmentation, no substitute for A-1 real multi-view ground truth
  (synthetic data never enters the frozen eval).

## 8. Acceptance checks before ANY training use

All run by a frozen checker script (to be built with the generator), each gate
pass/fail, no partial credit:

1. **Noise-distribution match vs deliverable 1**: for each mapped effect in §4,
   the synthetic marginal must match the measured statistic (inlier per-axis
   MAD-σ and bias, outlier rate + displacement quantiles, per-regime miss/FP
   rates, FP-confidence median, gap-length histogram) within tolerances declared
   in the generator config BEFORE generation; comparison uses the same stats
   code path as `analyze_residuals.py`.
2. **Physics sanity via the existing solver constants**: re-fitting each
   synthetic flight with the production gravity+drag ODE (`ball_arc_solver.py`
   `PhysicsParameters`, RK4) recovers it to a declared residual floor; zero
   underground/through-net trajectories; bounces at z=r_ball; speeds/apex
   heights inside declared per-shot-family envelopes.
3. **Projection consistency**: noiseless reprojection of O1 through I2 matches
   O2 pre-noise pixels to < 0.1 px.
4. **Contract validity**: every O1/O2 artifact round-trips through
   `ball_metric3d_contract.read_*` without `ContractValidationError`.
5. **Determinism**: same (config, seed) → byte-identical O1–O5.
6. **Coverage report**: near/far side, shot families, fps variants, regime mix —
   reported, with declared minimums, before any batch is admitted to training.
