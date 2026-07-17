# one_world_v1: confidence-weighted joint world refinement

Status: buildable design, 2026-07-16. **PREVIEW BAND, `VERIFIED=0`.** This
pass cannot become verified, train an input, or validate its own inputs until
the independent NS-04 gates below pass. There is no best-stack delta: this is
documents plus probes only; no model, weight, default, or policy changed.

## 1. Purpose and governing gates

Owner directive:

> everything goes hand in hand working with each other. we don't fully trust
> single things, but we use all info together to produce better results in the
> end, combining things we are most confident in.

`one_world_v1` makes that concrete without pretending to be the full global
factor graph. It reads immutable same-run modality artifacts, performs a fixed
sequence of bounded local refinements, abstains when independent evidence is
weak or contradictory, and writes one new artifact.

NS-04.4 acceptance gate, verbatim:

> Independent contact/bounce/floor errors improve

NS-04.4 stop/kill rule, verbatim:

> Never snap ball center to a plane or ankle centers to the floor.

NS-04.5 acceptance gate, verbatim:

> Independent world-MPJPE, paddle-surface contact, bounce/landing, sole/floor,
> event and reprojection improve; multiple-initialization and
> leave-one-modality/fixed-anchor ablations pass

NS-04.5 stop/kill rule, verbatim:

> Raw observations immutable; one early hypothesis, residual reduction and
> visual plausibility never promote.

Mapping:

| Gate language | v1 mechanism |
|---|---|
| independent errors improve | frozen baseline/after scorers; protected review surfaces are evaluation-only |
| never snap | finite soft surface weights, Huber influence, correction caps, before/after residuals retained |
| all error families and reprojection improve | five headline metrics; per-element reprojection regression suppresses that refinement |
| multiple initialization / leave-one / fixed anchor | harness is specified now; required before promotion, not claimed by this design |
| raw immutable | only `one_world_v1.json` is written; every input hash is retained |
| one early hypothesis never promotes | both IPPE poses survive to an independent-evidence resolver; ties abstain |
| residual/visual plausibility never promote | optimizer residuals are diagnostics; independent surfaces decide gates |

The full NS-04.5 system would jointly optimize camera/time, player root and
pose, ball segments, identity switches, paddle branches, and events with
switchable factors across the whole rally. v1 deliberately defers camera/time
optimization, re-posing BODY, global identity reassignment, ball-arc re-solving,
and a nonlinear all-rally solve. Those require independent ground truth and
multiple-initialization evidence that do not exist today. v1 is deterministic
staged fusion, not that factor graph.

## 2. Input contract and generation selection

### 2.1 Identity and time rules

Every file is selected from an explicit run directory; lexical “latest” is
forbidden. An external artifact is accepted only when its recorded input
SHA-256s match files in the run, its clip/source identity matches, player IDs
match, fps agrees within `1e-9`, and every `(frame_idx,t)` satisfies
`abs(t-frame_idx/fps) <= max(1e-6, 0.25/fps)`. A mismatch fails that artifact
closed and activates its documented fallback. It is never silently relabeled.

Frame joins use integer `frame_idx`/`frame` first. A time-only sample joins to
`round(t*fps)` only within half a frame. Raw observations are never
interpolated for display. The only interpolation is a latent wrist likelihood
described in §3.2; if its checks fail, the wrist stays missing.

### 2.2 Exact artifacts and fields

| Artifact / selection | Fields consumed | space, units, time, confidence and trust |
|---|---|---|
| `court_calibration.json` + `trust_bands.json.court` fallback | `schema_version`, `coordinate_frame`, `homography`, `intrinsics.{fx,fy,cx,cy,dist,source}`, `extrinsics.{R,t,camera_height_m}`, `image_size`, `capture_quality.{grade,reasons}`, `metric_confidence`, `source`, optional `coordinate_contract`, optional `trust_band` | canonical court net-center z-up metres; OpenCV world-to-camera extrinsics; native pixels; static. Inline band wins only if not stronger than companion. `corrected_unverified` and `line_evidence_solved_preview` remain preview/low and are inherited by every fused metric entity. |
| `tracks.json` | `fps`, `players[].{id,side,role}`, `frames[].{frame_idx?,t,bbox,world_xy,conf}`, `placement_provenance` | `world_xy` court metres; bbox native pixels; seconds/frames; `conf` [0,1]. Optional repair markers are joined from `repair_summary.json.summary.confidence_repairs[]` only when its `input_paths.tracks_path` hash matches. Legacy absence is recorded as unknown provenance. |
| preferred `placement_trajectory_refined.json` | top `artifact_type`, `fps`, `coordinate_space`, `world_frame`, `preview_band`, `VERIFIED`, `provenance.inputs.*.{path,sha256}`; per frame `frame_idx,t,transl_world,joints_world,joint_conf,placement_trajectory_refinement.{refined_transl_world,rigid_correction_xyz_m,covariance_m2,correction_magnitude_m,provenance.evidence}` | court metres, seconds, 3x3 covariance m², explicit effective weights. It is used only after the identity checks above. Its own TRK/BODY fusion is not repeated. Provenance records tier `trackI_refined`. |
| fallback `placement.json` | `fps`, `undistort_applied`, `homography_pixel_convention?`, `players[].id`, `frames[].{frame_idx,t,fused_world_xy,smoothed_world_xy,covariance_m2,signals[],source_counts,stance}`; each signal `{name,xy,sigma_m,covariance_m2,used,reason}` | court XY metres, covariance m², seconds/frames. Provenance tier `placement_fused`; uses `smoothed_world_xy` for output and `fused_world_xy` as raw diagnostic. |
| final fallback `tracks.json` | `world_xy`, `conf` | provenance tier `tracks_world_xy`; covariance is explicitly unavailable and a conservative 0.25² m² diagonal is attached as an assumption, not an observation. |
| `smpl_motion.json` (or schema-equivalent `skeleton3d.json` fallback) | `fps`, `model`, `world_frame`, `players[].id`, `frames[].{frame_idx,t,joints_world,joint_conf,transl_world,track_world_xy?}`, optional `skeleton_stride` | `court_Z0` metres, seconds/frames. BODY_17 wrists are indices 9/10. The real artifact lacks `skeleton_stride`; v1 records `observed_stride=median(diff(frame_idx))` per player. |
| `ball_track.json` | `fps`, `source`, `frames[].{t,xy,conf,visible,visibility_level?,world_xyz?,approx}`, `bounces[].{t,frame?,world_xy,p_bounce?,confidence?,uncertainty_m?,source,render_only,not_for_detection_metrics}` | WASB 2D native pixels, optional court-world metres, seconds; `conf` [0,1]. `approx=true` discounts. Exact repair metadata comes from a hash-matched filter summary when present; otherwise unknown. |
| optional `ball_candidates.json` | top `fps,source,source_mode,input_preprocessing,primary_output,max_candidates_per_frame,not_ground_truth,candidate_prediction,provenance`; `frames[].frame,candidates[].{xy,score,source_detector}` | native detector pixels; frame clock; scores [0,1]. Used only to estimate 2D ambiguity/covariance, never to replace the selected observation or create an event. |
| generation selector `{ball_arc_render.json | ball_track_arc_solved.json}` | render: `solver_status,solver_trusted_for_render,policy,segments[].{segment_id,t0,t1,frame_start,frame_end,anchor_types,anchor_frames,confidence,flight_sanity_verdict,reasons,fit_status,reprojection_rmse_px,endpoint_error_m,net_clearance_m,net_clearance_ok},samples[].{t,frame_float,segment_id,world_xyz,confidence,band,bridge}`; solved: top `status,kill_reasons,render_only,not_for_detection_metrics,policy`; `frames[].{t,xy,world_xyz,conf,sigma_m,band,approx,source}`; `anchors[].{anchor_id,kind,t,frame,world_xyz,sigma_m,status,source,immovable}`; `segments[].{segment_id,t0,t1,frame_start,frame_end,anchors_used,reprojection_rmse_px,physical_sanity,status,net_clearance_m,net_clearance_ok}` | court metres, seconds/frames, px RMSE. Prefer render generation when present and internally hash-linked; otherwise solved. Both remain render-only/not for detection metrics. `segment_budget_exceeded` makes affected segments missing and low trust; consume as-is. Generation recorded. |
| `audio_onsets_v2.json` | `frame_rate,status,not_gate_verified,trusted_for_contact,onsets[].{time_s,raw_time_s,analysis_time_s,score,onset_strength,window_start_s,window_end_s,features.{spectral_flux,high_frequency_content,band_energy_delta,pop_band_ratio}}` | seconds; feature/score unitless. Always review-only soft evidence. `trusted_for_contact=false` and `not_gate_verified=true` are binding. |
| selector `{contact_windows_refined_v1.json | contact_windows.json}` | `events[].{type,t,frame,player_id,confidence,sources.{audio?,wrist_vel,ball_inflection,human_review?},window.{t0,t1,importance},trust_band_note}` | seconds/frames, confidences [0,1]. Tier recorded. Refined absence falls back to raw; no event is invented. |
| racket generation 2: `racket_pose.json` + `racket_pose_hypotheses.json` | pose `fps,world_frame,translation_unit,players[].{id,paddle_dims_in,frames[].{t,pose_se3.{R,t},conf,source,reprojection_error_px,ambiguous}}`; hypotheses frame `{t,primary_pose,alt_pose,candidate_reprojection_errors_px,ambiguity_margin_px,ambiguous}` with each pose `{pose_se3,confidence,frame_conf,reprojection_error_px,source}` | camera frame, translations cm, seconds, px error, confidence [0,1]. Both files are required for resolution. |
| racket generation 1: `racket_pose_estimate.json` | top `world_frame,translation_unit,render_only,not_for_detection_metrics,trust`; frames `frame,t,pose_se3,conf,reprojection_error_px,ambiguous,source,confidence_provenance,trust_band` | current Wolverine is court-world metres and `wrist_proxy`. It is accepted for display continuity but cannot claim IPPE resolution: output status `unresolved_legacy_wrist_proxy`. |
| `net_plane.json` | `plane.{point,normal}`, `endpoints`, `center_height_in`, `post_height_in` | court metres for plane/endpoints; stated heights inches converted once to metres. Static. |
| `court_zones.json` | `zones` polygons, especially `zones.court` | court XY metres. Used for flags only; never clamps a landing. |
| `rally_spans.json` (fallback `tracks.rally_spans`) | `spans[].{t0,t1,sources}`, `not_ground_truth` | half-open seconds. Defines coverage denominator. |
| `virtual_world.json` | `schema_version,artifact_type,world_frame,fps,court,players,ball,paddles,summary` | immutable no-fusion baseline only. It is never rewritten. |

Generation tolerance is strict: no target run currently has
`racket_pose_hypotheses.json` or `ball_arc_render.json`; v1 still accepts their
verified code schemas. Missing generation-2 inputs select the legacy branch and
record it. Track-K outputs use only `one_world_v1*`; they never claim the
Track-I `placement_*` namespace.

### 2.3 Camera-pose lift

For camera hypothesis `(R_co,t_co)` and calibration `camera = R_cw world +
t_cw`, call `translation_to_metres(t_co,input_unit=...)`, then
`camera_to_world_points(t_co_m,R_cw,t_cw)`. Orientation is
`R_wo = R_cw.T @ R_co`. Use `invert_extrinsics` for the checked inverse and
`project_world_points` for evaluation, with explicit
`WORLD_COURT_NETCENTER_Z_UP_M -> PIXELS_UNDISTORTED_NATIVE` and declared
reference raster. No handwritten convention is permitted.

## 3. Deterministic pass

### 3.1 Common confidence and robust influence

For observation confidence `c`, variance `sigma²`, and markers `m`, define

`quality(c,m) = clip(c,0,1)^2 * product(reliability(m))`

`weight = quality / max(sigma², 0.005²)`.

Reliability multipliers are multiplicative and recorded: measured `1.0`;
player repaired `0.25`; ball `conf_source=interpolated_endpoint_min_half`
`0.25`; any `approx=true` `0.25`; render-only arc `0.35`; legacy repair
provenance unavailable `0.80`; audio review-only/not-gate-verified `0.20` and
is additionally bounded below; calibration high/med/low `1.0/0.60/0.30`,
`corrected_unverified=0.30`, `line_evidence_solved_preview=0.20`, missing band
`0.20`. Trust multipliers affect weights, never upgrade the inherited band.

For scalar normalized residual `u=r/sigma`, Huber loss is
`rho(u)=0.5u²` for `|u|<=1.5`, otherwise `1.5(|u|-0.75)`. IRLS influence is
`h(u)=min(1,1.5/|u|)`. All updates also have the explicit physical caps below.
Input order is canonical `(frame,entity id,source name)`; sums use float64;
three IRLS iterations and tie-breaks by stable ID make bytes deterministic.

### 3.2 BODY wrist availability

Exact BODY frames win. If contact frame `k` is missing, latent inference may
linearly interpolate wrists only when bracketing frames exist, both endpoint
joint confidences are at least `0.5`, endpoint speed is at most `15 m/s`, and
the gap is at most `min(0.10*fps, 2*observed_stride)` frames. Define

`w(k)=(1-a)w0+a w1`,

`c_interp=min(c0,c1)*exp(-gap/(2*observed_stride))`,

`sigma_interp²=(0.02+0.10(1-c_interp))² + (0.04*gap/observed_stride)²`.

It is marked `interpolated_latent_wrist`, used only in event/paddle scoring,
and never emitted as a per-frame observed wrist. No brackets or a failed guard
means missing.

### 3.3 Stage A — player placement reconciliation

If Track I refined input passes identity checks, consume its
`refined_transl_world`, corrected joints, covariance, and weights as-is. BODY
roots only produce a consistency residual; v1 does not fuse the same signals
again.

With `placement.json`, solve only a small rigid root reconciliation, not the
producer's per-signal placement problem. Let `p` be `smoothed_world_xy` with
`C_p`, `b` the BODY root XY, BODY confidence the median hip confidence, and
`C_b=(0.15+0.25(1-c_body))² I`. Because BODY grounding often came from the
track point, its nominal weight is multiplied by `0.25` for correlation.
Minimize

`rho(||x-p||_{C_p^-1}) + 0.25*rho(||x-b||_{C_b^-1})`

for three IRLS steps, then cap `||x-p|| <= 0.15 m`. Posterior covariance is
the inverse weighted normal matrix. Apply the resulting XY rigid translation
to BODY root/joints only in the new output. With tracks fallback, `p=world_xy`,
`C_p=0.25²I`. Missing BODY emits a placed root with no joints; missing placement
emits no player state. This reconciles already-fused placement and BODY roots;
it does not re-derive Track I placement.

### 3.4 Stage B — ball surface priors

The repository ball radius is `r_b=0.0371 m`. For a bounce with ball `x`, court
plane `(p,n)`, raw residual is `r0=n·(x-p)-r_b`. Let
`sigma_cal={0.02,0.05,0.12,0.20} m` for high/med/low-or-preview/missing,
`sigma_event=max(event.uncertainty_m,0.05)` when present else `0.12`,
`w_s=quality(c_bounce,calibration_markers)/(sigma_cal²+sigma_event²)`, and
`w_b=quality(c_ball,ball_markers)/max(sigma_ball²,0.03²)`.

Update only along `n`:

`delta = -n * h(r0/sqrt(sigma_cal²+sigma_event²)) * r0 * w_s/(w_s+w_b)`.

Cap `||delta|| <= min(0.15 m,2*sigma_ball)` and never assign `z=r_b`.
Report signed/absolute before and after residual and all weights. A result that
lands numerically on zero still reports the floating residual and
`soft_constraint=true`; tests use nonzero synthetic inputs to prove no clamp.

For `into_net`, use the same formula on
`r0=abs(n_net·(x-p_net))-r_b`, gradient `sign(d)n_net`. For `net_cross`, do not
treat the net as a contacted surface: report the zero-crossing time residual
and apply no positional pull. Missing event, ball world point, plane, or killed
arc segment means no refinement and a reason. Landing/bounce XY outside the
`zones.court` polygon sets `out_of_court_bounds=true`; it is never clamped.

### 3.5 Stage C — contact co-location and hitter inference

For every contact, form candidates for both wrists of every player. Wrist
uncertainty radius is `r_w=0.12 m`; metric volume radius is `r_w+r_b`. For ball
center `b`, wrist `w_pj`, covariance sum `S=C_ball+C_wrist+(r_w+r_b)²I/3`,
define capped Mahalanobis distance `D=min((b-w)^T S^-1(b-w),25)`. Candidate
likelihood is

`L_pj = exp(-0.5D) * (c_ball*c_joint*c_event)² * marker_reliability`.

The event's prior `player_id`, when present, receives only a bounded `1.25`
multiplier. Add null likelihood `L_null=0.05`. Player probability is
`P_p=sum_j L_pj / (L_null+sum_all L)`. The hitter is the maximum only if
`P_best>=0.55` and `P_best-P_second>=0.15`; otherwise ID is null with band
`too_close_to_call`. Both wrist likelihoods and the declared-ID disagreement
are output.

For the winning wrist, define `q_w=(c_ball*c_joint*c_event)²*reliability`,
`W_w=q_w*S^-1`, and `W_b=quality(c_ball)/C_ball`. The unconstrained combination
is

`b* = (W_b+W_w)^-1 (W_b b + W_w w)`.

Apply Huber influence to `(b*-b)`, and cap displacement at
`min(0.35 m,2*sigma_ball)`. If the raw volume residual exceeds `1.2 m`, hitter
confidence fails, ball/wrist is missing, or reprojection guard §4.5 fails,
abstain: refined contact ball is absent and raw ball reference remains. Thus
the real 7.97 m median mismatch cannot drag the ball across the court. Every
event outputs status, confidence, weights, before/after distance, and reason.

Audio can only multiply an existing visual event's odds by
`exp(clip(0.20*(2*pop_band_ratio-1),-0.20,0.20))`, matching the repository's
bounded `POP_LIKELIHOOD_LOG_BOUND=0.20` policy. It cannot create, veto, classify,
or gate a contact. Zero/blocked audio is neutral and recorded.

### 3.6 Stage D — paddle two-hypothesis resolution

Generation-2 swings use all ambiguous frames in the event window. Convert both
IPPE poses to world. Reprojection error and `ambiguity_margin_px` are carried
but **not** used to choose or discard a hypothesis.

For hypothesis `h` at frame `k`, compute independent unary energy:

`E_w=quality_w*rho(d(handle_h,nearest wrist)/0.15)`;

`E_c=quality_event*rho(d(face volume_h,ball_contact)/0.10)` at contact;

`E_m=quality_ball*rho(angle(oriented n_face_h, normalize(v_out-v_in))/30deg)`.

The normal sign is oriented toward the observed momentum change before the
angle. `v_in/v_out` use 3D arc differences over the nearest two supported
samples. If 3D is unavailable, use 2D direction change and the projected world
normal with weight multiplied by `0.25`; if neither side exists, `E_m` is
missing. Pairwise energy is
`E_t=0.5*rho(rotation_geodesic/25deg)+0.5*rho(translation_step/0.20m)`.
A two-state Viterbi pass minimizes summed unary+pairwise energy over the swing.

Let normalized margin `M=(E_second-E_best)/max(independent_term_count,1)`.
Resolve only if `M>=0.25`, at least three wrist frames exist, the contact term
exists, and at least one of momentum or multi-frame wrist consistency is
independent of reprojection. Otherwise emit both hypotheses with
`status=unresolved`, no canonical paddle pose, and the reason. Ties within
`1e-12` are unresolved, never index-selected. Output world pose, winning
hypothesis IDs, score components, `M`, carried `ambiguity_margin_px`, evidence
counts, and trust. Legacy `racket_pose_estimate` is accepted but always
`unresolved_legacy_wrist_proxy`.

### 3.7 Assembly and absence

Assemble only supported elements. Every element includes its confidence,
inherited weakest trust band, source hashes, rule ID, weights, residuals, and
marker discounts. Upstream bytes are checked before/after in tests. There is no
display fill: missing ball, wrist, player, or paddle stays missing. Latent
interpolation is event evidence only. Output is always `preview`, render-only,
not for detection metrics, not for training, and `VERIFIED=0`.

`physics_world_refinement.py` is not reused as the solver: it only packages
floor/contact windows from `virtual_world` and explicitly is not a global
optimizer. v1 may feed its output later as a diagnostic, but supersedes none
of its files. `confidence_gate.py` remains downstream display banding; v1
reuses its `ConfidenceProvenance` vocabulary and never weakens its bands.

## 4. Output schema draft

All models use `ConfigDict(extra="forbid")`; numeric vectors/matrices reuse
the schema aliases. The concrete implementation should add these classes to
`schemas/__init__.py` and `ARTIFACT_MODELS["one_world_v1"]`.

```python
class OneWorldInputRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact: str; path: str; sha256: str; generation: str
    schema_version: int | None; consumed_fields: list[str]
    trust_band: TrustBand | None; missing_reason: str | None

class OneWorldRuleProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: str; input_refs: list[str]; nominal_weights: dict[str, FiniteFloat]
    effective_weights: dict[str, FiniteFloat]; discounts: list[str]
    robust_kernel: str; correction_cap: str | None; degraded_reasons: list[str]

class OneWorldPlayerState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_id: int; root_world: Vector3; covariance_m2: Matrix3
    joints_world: list[Vector3] | None; joint_conf: list[FiniteFloat] | None
    placement_tier: Literal["trackI_refined","placement_fused","tracks_world_xy"]
    confidence: FiniteFloat; trust_band: TrustBand
    confidence_provenance: ConfidenceProvenance; provenance: OneWorldRuleProvenance

class OneWorldBallState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world_xyz: Vector3; covariance_m2: Matrix3; xy_observed_px: Vector2 | None
    confidence: FiniteFloat; source_generation: str; approx: bool
    render_only: Literal[True]; not_for_detection_metrics: Literal[True]
    trust_band: TrustBand; confidence_provenance: ConfidenceProvenance
    provenance: OneWorldRuleProvenance

class OneWorldPaddleState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_id: int; status: Literal["resolved","unresolved","unresolved_legacy_wrist_proxy"]
    pose_world: SE3 | None; winning_hypothesis: str | None
    retained_hypotheses: list[dict[str, Any]]; score_components: dict[str, FiniteFloat]
    score_margin: FiniteFloat | None; ambiguity_margin_px: FiniteFloat | None
    confidence: FiniteFloat; trust_band: TrustBand; provenance: OneWorldRuleProvenance

class OneWorldFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_idx: int; t: FiniteFloat; players: list[OneWorldPlayerState]
    ball: OneWorldBallState | None; paddles: list[OneWorldPaddleState]
    missing: list[str]

class OneWorldContactRefinement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_index: int; frame: int; t: FiniteFloat; status: str
    raw_player_id: int | None; hitter_id: int | None; hitter_confidence: FiniteFloat
    hitter_band: Literal["resolved","too_close_to_call","unsupported"]
    per_player_wrist_likelihoods: dict[str, list[FiniteFloat]]
    raw_ball_world: Vector3 | None; refined_ball_world: Vector3 | None
    raw_wrist_volume_residual_m: FiniteFloat | None
    refined_wrist_volume_residual_m: FiniteFloat | None
    displacement_m: FiniteFloat | None; confidence: FiniteFloat
    trust_band: TrustBand; provenance: OneWorldRuleProvenance

class OneWorldBounceRefinement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_index: int; frame: int; t: FiniteFloat; status: str
    raw_ball_world: Vector3 | None; refined_ball_world: Vector3 | None
    signed_plane_residual_before_m: FiniteFloat | None
    signed_plane_residual_after_m: FiniteFloat | None
    out_of_court_bounds: bool; confidence: FiniteFloat
    trust_band: TrustBand; provenance: OneWorldRuleProvenance

class OneWorldSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    placement_tier_counts: dict[str,int]; missing_counts: dict[str,int]
    ball_contact_distance_m: dict[str,FiniteFloat | int | None]
    bounce_plane_residual_m: dict[str,FiniteFloat | int | None]
    world_coverage: dict[str,FiniteFloat | int]
    paddle_resolution: dict[str,FiniteFloat | int]
    reprojection_consistency: dict[str,Any]
    regression_kills: list[str]; warnings: list[str]

class OneWorldV1(StrictArtifact):
    artifact_type: Literal["racketsport_one_world_v1"]
    world_frame: Literal["court_Z0"]
    coordinate_space: Literal["world_court_netcenter_z_up_m"]
    fps: FiniteFloat; VERIFIED: Literal[0]; preview_only: Literal[True]
    render_only: Literal[True]; not_for_detection_metrics: Literal[True]
    not_for_training: Literal[True]; raw_inputs_mutated: Literal[False]
    inputs: list[OneWorldInputRef]; frames: list[OneWorldFrame]
    contacts: list[OneWorldContactRefinement]
    bounces: list[OneWorldBounceRefinement]
    summary: OneWorldSummary; trust_band: TrustBand
```

## 5. Metrics and frozen baseline procedure

Baseline is current `virtual_world.json` plus its immutable inputs, with no
one-world correction. After uses `one_world_v1.json`. Identical events, rally
frames, projection code, thresholds, and quantile convention are used.

1. **Ball-at-contact distance to hitter wrist volume.** For supported contact
   `e`, `d_e=max(0,min_j ||b_e-w_ej||-(0.12+0.0371))` metres. Report count,
   median, and nearest-rank p90 plus center-distance diagnostics. Baseline uses
   raw arc ball and raw BODY wrists; after uses refined event ball and inferred
   hitter. Abstentions are separately counted, never dropped from coverage.
   Wolverine today: 24/24 computable, baseline median `7.9737214557 m`, p90
   `11.1650654218 m` (`VERIFIED=0`, plainly implausible).

2. **Bounce-to-court-plane residual.** For plane `(p,n)`,
   `r_e=abs(n·(b_e-p)-0.0371)` metres. Report count/median/p90 before and after.
   Soft priors should leave honest nonzero residual. Wolverine can measure it
   at arc bounce anchors/events; reviewed label comparison is evaluation-only.

3. **WORLD COVERAGE.** Rally frames are integer frames in half-open `[t0,t1)`.
   A player qualifies when a placement tier emits the frame and its source
   track confidence is `>=0.5`; posterior uncertainty is reported separately
   and cannot increase source confidence. Ball qualifies when supported
   world XYZ exists and source ball/arc confidence is `>=0.5`. Coverage is
   `#frames(all four expected player IDs and ball qualify)/#rally frames`.
   Trust-band-stratified coverage is also reported. Wolverine baseline is
   `117/300 = 0.39`.

4. **Paddle ambiguity resolution.** Denominator is ambiguous-flagged swing
   windows with two valid IPPE hypotheses. Resolved numerator has `M>=0.25`
   and all evidence requirements in §3.6. Report resolved fraction and
   `1-resolved` honestly unresolved fraction; legacy proxies are a separate
   unsupported count. No target-run denominator exists today.

5. **Reprojection consistency.** Use typed `project_world_points`. BALL error
   is distance to raw `ball_track.xy`; player error is grounded root projection
   to track bbox bottom-center; paddle error is mean four-corner distance when
   true corners exist. For each entity and median/p90 distribution, require
   `after <= baseline + max(1 px,0.05*baseline)`. Per sample,
   `after-baseline > max(2 px,0.10*baseline)` is a kill signal: suppress that
   element's refinement and record `reprojection_regression`. Smaller optimizer
   residual alone never counts as improvement.

The later harness runs the same scorer under leave-one-modality arms
`{-BODY,-arc,-paddle,-audio,-placement-covariance}`, multiple initial states
`{raw, bounded +5cm/-5cm XY/Z perturbations}`, and fixed-anchor arms
`{calibration fixed, surface priors off, contact priors off}`. v1 itself fixes
camera/time; convergence to identical bytes across allowed initializations and
no independent-metric regression are required before an NS-04.5 claim.

## 6. What is measurable today

### Wolverine manager v5.1

This is the fullest same-run stack: calibration/trust, four tracks, placement,
SMPL/BODY wrists, 2D ball, 3D arc, 24 contacts, net/zones/rallies, wrist proxy,
and virtual world. It can compute metrics 1, 2, 3, and BALL/player reprojection.
It cannot measure true two-IPPE resolution. Gaps: no same-run Track-I refined
placement, ball candidates, refined contacts, audio onsets (valid file but
blocked/zero), true-corner racket candidates/poses/hypotheses, or arc render.

The exact CPU-safe regeneration recipe is to clone, never mutate, the banked
run and reuse its inputs:

```bash
cp -a runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z \
  runs/lanes/<implementation_lane>/wolverine_regen
python3 scripts/racketsport/process_video.py \
  --video eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4 \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --out runs/lanes/<implementation_lane>/wolverine_regen \
  --court-calibration runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/court_calibration.json \
  --tracks runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/tracks.json \
  --ball-track runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/ball_track.json \
  --no-gpu --json
```

Valid existing artifacts reuse; the new explicit `events_refined` and
`ball_arc_refined` stages mint `contact_windows_refined_v1.json` and
`ball_arc_render.json` in the clone. Do not use `--force`, which deletes banked
BODY inputs.

There is **no valid command today** to mint both-IPPE hypotheses from this run:
`racket_candidates.json` is absent and the wrist proxy is not four true
corners. Once true-corner candidates exist, the CPU module command is:

```bash
python3 - <<'PY'
from pathlib import Path
from types import SimpleNamespace
from threed.racketsport.racket_stage_runner import RacketStageRunner
r=Path("runs/lanes/<implementation_lane>/wolverine_regen")
print(RacketStageRunner(candidate_path=r/"racket_candidates.json", reject_ambiguous=False).run(SimpleNamespace(run_dir=r,clip="wolverine_mixed_0200_mid_steep_corner")).as_dict())
PY
```

It fail-closes without real corners; no box/wrist conversion is an acceptable
substitute.

### Protected evaluation cards

The four cards' labels are evaluation-only and never training inputs. Reviewed
events, 15-point metric calibration, and ball review points can independently
check timing/reprojection/bounce surfaces, but their own files say
`not_ground_truth=true`; call them independent-ish review surfaces, not GT.
Outdoor and Indoor labels are prohibited for this lane and for tuning. Run the
frozen scorer only under manager authorization; never feed fused outputs back
as labels.

### pb.vision 11-minute demo

Current disk truth is 2D WASB ball + court/timebase only: no tracks, BODY,
contacts, audio artifact, or ball world XYZ; calibration lacks metric confidence
and is poor/corrected-unverified. CPU work can derive audio from
`data/pbvision_11min_20260713/source_video.mp4`, propose bounce/audio windows,
and run a preview ball-on-court/soft-bounce arc. It cannot co-locate contacts,
identify hitters, resolve paddles, or place four players. Four-player world
coverage is honestly `0`; no GPU regeneration is part of v1.

## 7. Standalone first, then Track-C wiring request

First ship `scripts/racketsport/build_one_world_v1.py --run-dir RUN --out
RUN/one_world_v1.json`. It reads only the explicit run and fails loudly on
schema/coordinate contradictions while optional evidence degrades.
`report_one_world_metrics.py` scores baseline/after, and
`validate_one_world_v1.py` validates schema, hashes, absence, band inheritance,
and reprojection kills.

Exact request to Track C (Track K never edits `process_video.py`):

> Add `PipelineStageDefinition("one_world_v1", 175, 175)` immediately after
> `ball_arc_refined` (170) and before `world` (180). Add
> `RUN_IDENTITY_DEPENDENCIES["one_world_v1"] = ("calibration", "tracking",
> "placement_refine", "grounding_refine", "paddle_pose", "events_refined",
> "ball_arc_refined")`; change `world` to depend on `one_world_v1` in addition
> to its existing raw dependencies so raw world assembly remains available.
> Add `RUN_IDENTITY_CONFIG_KEYS["one_world_v1"] = ("world.one_world_v1",)`.
> Add `RUN_IDENTITY_OUTPUTS["one_world_v1"] = ("one_world_v1.json",
> "one_world_v1_metrics.json")`. Wire `_stage_one_world_v1` to call the same
> module as `build_one_world_v1.py`, preserve optional-degrade vs malformed-fail
> semantics, expose real wall time, and add cold/reuse/hash-change tests. Do not
> rename upstream artifacts or make `virtual_world.json` mutable.

## 8. Implementation slicing (maximum three lanes)

1. **Core/schema lane:** owns new `threed/racketsport/one_world_v1.py`, additive
   OneWorld models/registry entry in `schemas/__init__.py`,
   `docs/racketsport/one_world_v1.schema.json`, and
   `tests/racketsport/test_one_world_core.py`. Tests: both generations,
   same-run identity, covariance weighting, repaired/approx/audio discounts,
   no-snap/out-of-bounds, huge-outlier abstention, wrist interpolation limits,
   hitter ties, IPPE Viterbi/unresolved behavior, camera lift, determinism, and
   byte hashes of every raw input.

2. **CLI/metrics lane:** owns `build_one_world_v1.py`,
   `report_one_world_metrics.py`, `validate_one_world_v1.py`,
   `tests/racketsport/test_one_world_clis.py`, and additive scaffold-index rows.
   Each CLI gets the three required rows:

   ```python
   RELATED_TEST_OVERRIDES.update({
     "build_one_world_v1":"test_one_world_clis.py",
     "report_one_world_metrics":"test_one_world_clis.py",
     "validate_one_world_v1":"test_one_world_clis.py"})
   SCHEMA_OVERRIDES.update({
     "build_one_world_v1":"one_world_v1.schema.json",
     "report_one_world_metrics":"one_world_v1_metrics.schema.json",
     "validate_one_world_v1":"one_world_v1_validation.schema.json"})
   TASK_HINTS.update({
     "build_one_world_v1":("WORLD","NS-04.6"),
     "report_one_world_metrics":("EVAL","NS-04.5"),
     "validate_one_world_v1":("WORLD","NS-04.5")})
   ```

   The same-lane direct-CLI test contains literal references to all three
   command paths, executes `--help` and a synthetic run, checks exact exit
   codes, and asserts the scaffold index has related/direct/schema entries.

3. **Track-C integration lane:** sole owner of `process_video.py` and its
   stage-contract tests. It applies the request in §7 only after lanes 1/2 land,
   verifies serial/overlap order, cold/reuse/partial/malformed behavior,
   dependency hashes, output declarations, and the wide suite.

## 8.5 Strategic rationale: the anchor-source hierarchy (owner framing, 2026-07-16)

Why contact co-location is the structural point of this pass, not a fusion nicety.

pb.vision's ball-3D works because trained event heads give them **an anchor at every event**:
bounces pinned to z=radius (154/154), net interactions pinned to the net plane, and a trained
radius head (R²=0.71) supplying depth. Gravity then determines the arc between anchors. Our arcs
fail for the same reason from the other side: **anchor sparsity** (~20 auto-anchors across 697s
-> game-scale segments balloon; see runs/lanes/ballarc_scale_guard_20260715). Anchors are the
whole ballgame.

But pb.vision has **no 3D players** — their world is ball/events/court only. We have accurate,
correctly-placed SAM-3D-Body meshes (foot-slide 34mm -> 7mm as of Track I's placefuse). That
asymmetry creates an anchor class they structurally cannot copy:

**Anchor-source hierarchy (ranked by what each is worth to us):**
1. **Contact co-located with the hitter's hand — 3D, MEASURED, OURS ALONE.** A paddle contact
   happens at the hitter's hand, and we know that hand's position in 3D. So a confirmed contact
   yields a *measured* 3D ball anchor, not an inferred-depth 2D one. It needs **zero trained
   event heads** — BODY + court + ball-2D proximity is enough to propose it — and it compounds
   when the event head lands (better proposals -> more co-locations).
2. **Bounce at the court plane — 3D, both of us.** z = ball radius, soft-weighted (§3.4).
3. **Net interaction at the net plane — 3D, both of us.** (§3.4; net_cross carries time residual
   only, no positional pull.)
4. **Trained-event / radius-head depth — pb.vision has it, we're building it** (Track G).

### 8.5.1 Measured evidence on the Wolverine bundle (manager probe, 2026-07-16)

Honest status: **v1 emits ZERO co-location anchors today** — all 24 declared contacts abstain
(22 at >1.2m, 2 too_close_to_call), because §3.5 evaluates co-location **at the declared event
frame** and those frames are mistimed. The class is not disproven by that; it is blocked by
upstream event timing. The manager probe
(runs/lanes/oneworld_impl_20260716/anchor_window_probe.py, output *_output.json, EXIT 0)
searched +/-15 frames (+/-0.5s) around each declared contact for the closest approach of the 3D
ball to any BODY wrist, **with a chance baseline at random non-event frames**:

| statistic | declared contacts (n=24) | chance / random non-event frames (n=24) |
|---|---|---|
| distance at the declared frame, median | 3.126 m | — |
| windowed closest-approach, median | **1.167 m** | **4.499 m** |
| closest approach <= 0.50 m (paddle-length band) | **6/24** | **0/24** |
| closest approach <= 1.20 m | **15/24** | 6/24 |

The signal is **real and ~4x above chance**: declared contacts have the ball passing near a
wrist within +/-0.5s far more often than random frames do. A ball at true contact should sit
~0.3-0.5m from the wrist (paddle length), so the 6/24 in that band are plausibly true
co-locations; 0/24 within 0.30m is the physically *expected* result, not a failure.

Two upstream defects, now quantified, block harvest:
- **Event timing**: best-approach offsets spread across the whole window
  (-15,-13,-13,-12,-11,-11,-10,-9,-7,-6,-5,-5,-4,-2,0,2,3,3,4,5,6,10,10,13 frames), several
  pegged at the window edge — so some true offsets are >0.5s and this window is too narrow to
  measure them.
- **Attribution**: only **9/24** declared hitters are the nearest player — roughly a coin flip.

### 8.5.2 The unlock (design change proposed, NOT silently applied)

The anchor class needs a **bounded closest-approach search inside the event window** instead of
trusting the declared frame: propose the co-location anchor at argmin distance over the window,
gate it on (a) the paddle-length band, (b) a chance-baseline margin recomputed per clip, (c)
agreement between the co-located player and independent wrist-speed evidence, and emit it as a
*proposed measured anchor* with its dt recorded as an honest event-timing correction — never
rewriting the raw event. This is exactly the anchor class Track A's arc solver is starving for,
and it arrives with no trained event head. It is a **v2 behavior**: it changes §3.5's join rule,
so it needs its own failing-first tests, a per-clip chance baseline (a wide window will always
find a spurious nearest wrist), and a pre-registered kill rule — the same discipline that made
this probe trustworthy. Not adopted here; specified for the next window.

## 9. Honest limitations and promotion wall

- Arc segments are sparse/weak today; Track-A anchors may improve them, but v1
  must not anticipate fields or fill killed segments.
- BODY wrists are absent on some frames and `skeleton_stride` is not serialized;
  observed cadence plus inflated latent interpolation is the only honest path.
- Handedness is unknown. Both wrists compete; ties are `too_close_to_call`.
- Demo calibration is poor/corrected-unverified and has no metric confidence;
  every derived world result inherits that band.
- Wolverine's real contact/wrist baseline is catastrophically inconsistent.
  Bounded abstention is mandatory; a lower fused residual by itself proves
  nothing.
- True racket corners and generation-2 hypotheses do not exist for the target
  run. The wrist proxy is never promoted to 6DoF.
- Placement Track-I output exists only for a different run/ID domain. Cross-run
  import is forbidden even though its schema-shaped fields are useful evidence.
- “one early hypothesis ... never promote[s]”: v1 retains both and abstains.
- “Validate only by a smaller optimizer residual” is banned by the reuse
  contract: improvement requires the independent metrics and protected posture
  in §§5-6, plus leave-one/multiple-init/fixed-anchor ablations.
- All output is permanently preview-band, render-only, not for detection
  metrics/training, `VERIFIED=0`, until a named independent-data gate passes.
