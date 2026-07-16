# LANE oneworld_design_20260716 — Track K: "one world v1" fusion pass DESIGN (docs-only)

You are a Codex design lane for DinkVision (repo root /Users/arnavchokshi/Desktop/pickleball).
Your job: write the complete, disk-verified design for "one world v1" — a confidence-weighted
joint fusion pass that consumes per-modality artifacts and produces ONE coherent refined world.
You write documents + throwaway probe scripts in YOUR LANE DIR ONLY. Zero repo-file edits.

OWNER DIRECTIVE (2026-07-16, the product's center of gravity — design to THIS):
"everything goes hand in hand working with each other. we don't fully trust single things, but we
use all info together to produce better results in the end, combining things we are most
confident in."

## 1. HARD RULES
- NO branches, NO commits (manager commits fence-only after ruling).
- Read NORTH_STAR_ROADMAP.md FIRST: §3.1 reuse contract (lines ~245-258), NS-04 table
  (lines ~347-356, especially NS-04.4 + NS-04.5 kill language), standing rules (~462-478).
- 4 protected clips are EVAL-ONLY. Wolverine/Burlington internal-val reads are OK. Outdoor/Indoor
  labels: NEVER. You are read-only everywhere outside your lane dir anyway.
- Honest reporting. If an artifact lacks a field your design needs, the design changes (absence
  semantics / degrade path) — you never assume the field into existence.
- Every claim about an artifact field MUST be backed by a real command + real output in
  FIELD_VERIFICATION.md (unpiped exit codes; `python3 -c` json probes are fine).
- All outputs under /Users/arnavchokshi/Desktop/pickleball/runs/lanes/oneworld_design_20260716/.
  Other lanes' run dirs are READ-ONLY evidence. NO .patch files. No new root .md files.
- WIDE SUITE: NOT run by this lane — deviation stated by the manager: this lane makes ZERO
  repo-file edits. Substitute proof required in your report: `git status --porcelain` output
  showing you touched nothing outside runs/lanes/oneworld_design_20260716/ (pre-existing dirty
  files from other tracks listed as untouched-by-you).
- VERIFIED=0 is binding. The fused world is PREVIEW BAND, permanently, until a named
  independent-data gate passes. Your design must say this explicitly.

## 2. EXPLICIT FILE OWNERSHIP
- OWNED (create/write): runs/lanes/oneworld_design_20260716/** (DESIGN.md,
  FIELD_VERIFICATION.md, baseline_probe*.py, probe outputs, report.json).
- READ-ONLY: everything else. In-flight lanes own: process_video.py (refinedstage),
  orchestrator.py (calpolicy), ios/** (recordvis), threed/racketsport/event_head/** (Track G).
  You do not need any of those files writable.

## 3. OBJECTIVE + ACCEPTANCE
Produce DESIGN.md — the buildable design for the one_world_v1 pass — plus disk verification that
every input it consumes and every metric it defines is real and computable TODAY.

### DESIGN.md required content (acceptance item D1 — ALL sub-items present and internally consistent):
(a) Problem statement anchored to the owner directive + NS-04.4/NS-04.5 (quote their gate/kill
    language verbatim and map each design mechanism to it).
(b) INPUTS: exact artifact filenames + the exact fields consumed, each with coordinate space,
    units, timebase, confidence field, and trust/provenance markers. The known input set:
    - court_calibration.json (+ intrinsics/extrinsics + trust band; corrected_unverified and
      line_evidence_solved_preview bands exist — fused outputs must inherit calibration band).
    - tracks.json (TrackFrame.world_xy footpoints + conf; player_id_repair "repaired" markers).
    - PLAYER TRAJECTORIES (Track I seam, LIVE lane trackI_placefuse_20260716): preferred input =
      placement_trajectory_refined.json (Track I's NEW artifact w/ covariance+provenance; their
      handoff schema lands at runs/lanes/trackI_placefuse_20260716/SCHEMA.md — READ their spec.md
      + SCHEMA.md if present and design against it); fallback chain when absent:
      placement.json (PlacementFrame.fused_world_xy / smoothed_world_xy / covariance_m2 +
      signals[] provenance) -> tracks.json world_xy. Record which tier was consumed in
      provenance. NAMING FENCE: Track K artifacts must NOT claim placement_* filenames
      (Track I owns that namespace) — use one_world_v1* naming.
    - smpl_motion.json (SmplPlayer.frames[].joints_world w/ BODY_17 wrists idx 9/10, joint_conf,
      transl_world, court_Z0 frame, skeleton_stride — wrists NOT present every frame; design the
      interpolation/uncertainty-inflation policy explicitly).
    - ball_track.json + ball_candidates.json (2D WASB, conf, visibility; temporal-filter repair
      markers conf_source=interpolated_endpoint_min_half, approx=True).
    - ball_track_arc_solved.json / ball_arc_render.json (3D segments w/ anchors, confidence,
      reprojection_rmse_px, flight_sanity; segment_budget_exceeded LOUD degrades — consume as-is;
      sparse today, Track A anchor recovery in flight, design for anchors improving).
    - audio_onsets_v2.json (per-onset features incl. pop_band_ratio; review_only,
      not_gate_verified, trusted_for_contact=false — soft evidence ONLY, non-gating, bounded).
    - contact_windows.json (+ contact_windows_refined_v1.json when present): ContactEvent
      {type, t, frame, player_id, confidence, sources{audio,wrist_vel,ball_inflection}, window}.
    - racket_pose.json + racket_pose_hypotheses.json (evidence17: BOTH IPPE poses, alt_pose,
      ambiguity_margin_px, ambiguous; world_frame="camera", translation_unit="cm" — design the
      camera->world lift via calibration extrinsics with the typed coordinates.py API).
    - net_plane.json, court_zones.json.
    GENERATION TOLERANCE (mandatory): NO run on disk currently has racket_pose_hypotheses.json or
    ball_arc_render.json (code landed 2026-07-16, unexercised). v1 must accept BOTH artifact
    generations: {racket_pose_estimate.json | racket_pose.json + hypotheses} and
    {ball_track_arc_solved.json | ball_arc_render.json}, with provenance recording which it got.
(c) THE PASS: a staged, deterministic, confidence-weighted joint refinement (v1 — NOT the full
    NS-04.5 factor graph; say what v1 defers and why that is honest). Required behaviors:
    1. PLAYER PLACEMENT: consume placement fused trajectories + BODY roots; reconcile with
       per-signal covariance weighting; do NOT re-derive placement (that pass owns its fusion).
    2. BALL-SURFACE PRIORS (NS-04.4): bounce events pull ball center toward court plane at one
       ball radius as a SOFT constraint — weight from bounce confidence + calibration band —
       with the residual REPORTED on the output; net interactions vs net_plane the same way;
       out-of-range landings FLAGGED (out_of_court_bounds) never clamped. Quote and honor:
       "Never snap ball center to a plane or ankle centers to the floor."
    3. CONTACT CO-LOCATION: each contact event co-locates the ball position with the hitter's
       paddle-side wrist volume (from BODY joints_world; paddle-side = handedness-unknown -> use
       both wrists w/ per-wrist likelihood). Output BOTH: refined ball position at contact
       (weighted combination, weights from ball conf x wrist joint_conf x event confidence,
       combination rule documented + bounded) AND hitter identification w/ confidence
       (which player's wrist volume best explains the contact; ties -> too_close_to_call band).
    4. PADDLE 6DoF RESOLUTION: per-swing, resolve the two-IPPE ambiguity using (i) wrist
       trajectory proximity/consistency over the swing window, (ii) contact timing, (iii) face
       normal vs ball momentum change (incoming/outgoing ball direction from 2D track + arc
       segments where available; the chosen face normal must be compatible with the ball's
       momentum change direction at contact). Output paddle pose in WORLD frame WITH: which
       hypothesis won, the score margin, the ambiguity_margin_px carried through, and an
       unresolved band when evidence is insufficient (NEVER silently pick). Reuse contract:
       "discard the second IPPE pose by reprojection alone" is BANNED — your resolver uses
       independent evidence, not reprojection.
    5. EVERY OUTPUT: provenance (which inputs, which rule, which weights), confidence,
       trust band; raw observations IMMUTABLE (one_world_v1.json is a NEW artifact; nothing
       upstream is rewritten); unsupported elements STAY MISSING (absence semantics, no
       interpolation-to-fill for display).
    For each behavior: the exact residual/score formula, the confidence->weight mapping
    (including how repaired/approx/review_only markers DISCOUNT weight), robust kernel or
    bounded-influence rule, and the degrade path when an input is absent.
(d) OUTPUT SCHEMA: one_world_v1.json draft as a Pydantic-style field list consistent with
    schemas/__init__.py house style (extra="forbid", schema_version, artifact_type,
    TrustBand + ConfidenceProvenance reuse, render_only/not_for_detection_metrics flags where
    honest). Include per-frame world state (players, ball, paddles), per-event refinements
    (contacts w/ hitter id + residuals, bounces w/ plane residuals), per-element provenance,
    and a summary block with the headline metrics. State where it slots: standalone CLI first
    (consumes a run dir), later stage between ball_arc_refined(170) and world(180) via a Track C
    wiring request (draft that request's exact text: stage node, RUN_IDENTITY_DEPENDENCIES /
    _CONFIG_KEYS / _OUTPUTS entries — Track K never edits process_video.py).
(e) TARGET METRICS (all 5, exact formulas + baseline procedure + which clips measurable TODAY):
    1. ball-at-contact distance to hitter wrist volume (m) — per contact event, distribution +
       median/p90; baseline = raw ball vs raw wrists; after = fused.
    2. bounce-to-court-plane residual (m) — |z - ball_radius| at bounce events (ball radius:
       pickleball diameter 73-75.5mm -> radius ~0.0368m; cite what the repo already uses if a
       constant exists); baseline vs fused. REPORTED not zeroed (soft prior means residual>0 is
       expected and honest).
    3. WORLD COVERAGE (owner headline): fraction of rally frames where ALL FOUR players + ball
       are simultaneously world-placed with confidence >= threshold (define the threshold and
       the per-entity confidence source; define rally frames from rally_spans).
    4. paddle-pose ambiguity resolution rate: fraction of ambiguous-flagged swing windows
       resolved with margin above threshold + fraction honestly left unresolved.
    5. reprojection consistency: project fused world entities back to 2D via typed coordinates
       API; must NOT degrade vs raw (define per-entity px deltas; fused-worse-than-raw beyond
       tolerance = regression, a kill signal for that element's fusion rule).
    BASELINE = current world stage output (virtual_world.json + its inputs, no fusion).
    Ablations (design the harness now, run later per NS-04.5 gate): leave-one-modality,
    multiple-initialization, fixed-anchor.
(f) MEASUREMENT REALITY: which runs/clips can produce baseline+fused numbers TODAY:
    - runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/ = fullest real stack (verify
      every consumed field on these actual files).
    - eval_clips/ball/<4 cards>/labels/ = reviewed events + metric-15pt calibration + ball
      points (independent-ish check surfaces; note their protection posture).
    - demo 11-min (runs/lanes/pbv11_headtohead_20260713/rerun_20260715/cpu_events_full/
      pbvision_11min_20260713/) = PARTIAL: ball 2D + court + timebase only. Design the demo
      story honestly: what fusion CAN do there (ball-on-court w/ bounce priors + audio derivable
      from source.mp4) and what it CANNOT (no players/wrists -> no contact co-location; world
      coverage honestly ~0 for players). NO GPU regeneration in v1.
    - State which cheap CPU stages must be re-run on the Wolverine run dir to mint
      new-generation inputs (paddle_pose w/ hypotheses; events_refined/ball_arc_refined) and
      how to do it WITHOUT editing the runner (reuse mechanics / module CLIs) — implementation
      lane's job, but the design names the exact commands.
(g) IMPLEMENTATION SLICING: <=3 lanes with file fences, test plan per lane (fenced new-module
    tests under tests/racketsport/test_one_world_*.py), CLI naming that satisfies the scaffold
    index prefixes (build_/report_/validate_ — e.g. scripts/racketsport/build_one_world_v1.py +
    report_one_world_metrics.py) + the three list_scaffold_tools.py dict entries each CLI needs
    + direct-CLI reference test same-lane.
(h) HONEST LIMITATIONS section: sparse arc segments today; skeleton stride; handedness unknown;
    calibration bands on the demo; single early hypothesis ban ("one early hypothesis ...
    never promote[s]"); "Validate only by a smaller optimizer residual" ban -> improvement
    claims require the independent surfaces in (e)/(f).

### Acceptance item D2 — FIELD_VERIFICATION.md:
For EVERY input field named in (b): one real probe command + trimmed real output from an
on-disk artifact (Wolverine v5.1 run, demo run, or eval_clips labels), confirming presence,
type, units, and coordinate frame. Where a field exists only in code (hypotheses artifact,
ball_arc_render), probe the PRODUCING CODE instead (file:line of the writer + the schema class)
and mark it generation-2-unexercised. Real unpiped exit codes shown.

### Acceptance item D3 — baseline feasibility probe (numbers, not promises):
Write baseline_probe.py in your lane dir and RUN it against the Wolverine v5.1 run: compute at
least TWO of the five metrics as BASELINE numbers end-to-end from the real artifacts —
(1) ball-at-contact distance to nearest wrist across contact_windows events, and (2) world
coverage at a stated threshold across rally frames (players from tracks/placement + ball from
ball_track world_xyz or arc_solved). Report the actual numbers + event/frame counts in
report.json acceptance rows. If a metric is genuinely uncomputable from that run, prove why
(missing field probe) and compute a named substitute. These numbers become the design's
baseline column — mark them internal, preview, VERIFIED=0.

### KILL / honesty criteria:
- A behavior whose required inputs cannot support it on ANY on-disk run -> design records the
  degrade path + what future artifact unlocks it. Do not fake support.
- If placement/smpl artifacts disagree on frames/timebase in ways that break co-location, that
  is a FINDING (report it precisely), not something to paper over.

## 4. EVIDENCE TO READ FIRST (exact paths)
- NORTH_STAR_ROADMAP.md (§3.1 ~245; NS-04 ~347-356; queue ~412; rules ~462-478).
- threed/racketsport/schemas/__init__.py: Tracks ~645, PlacementArtifact ~725, SmplMotion ~1266,
  BallTrack ~1413, BallArcRender ~1508, RacketPose ~1758, ContactWindows ~1893, VirtualWorld
  ~2324, TrustBand ~1282ish / ConfidenceProvenance ~2128, ARTIFACT_MODELS ~2736.
- threed/racketsport/coordinates.py (typed spaces; project_world_points,
  unproject_image_points_to_world, invert_extrinsics, translation_to_metres).
- threed/racketsport/joint_schema.py (BODY_17 wrists idx 9/10).
- threed/racketsport/virtual_world.py (BASELINE assembler — your pass refines, never rewrites),
  confidence_gate.py, physics_world_refinement.py (existing scaffold: state explicitly whether
  v1 reuses, feeds, or honestly supersedes it and why), event_fusion.py (AudioOnsetCandidate,
  bounded audio rules POP_LIKELIHOOD_LOG_BOUND=0.20), racket6dof.py + racket_stage_runner.py
  (hypotheses writer), ball_temporal_filter.py (~448-498 repair markers), ball_arc_solver.py
  (segment_budget_exceeded ~40/2431/2529), player_id_repair.py (~475 repaired marker).
- scripts/racketsport/process_video.py: AUTHORITATIVE_STAGE_GRAPH ~329 (events_refined 160,
  ball_arc_refined 170, world 180, confidence_gate 190), RUN_IDENTITY_DEPENDENCIES ~456,
  _CONFIG_KEYS ~493, _OUTPUTS ~518.
- runs/lanes/evidence17_20260716/report.json + runs/lanes/refinedstage_20260716/spec.md
  (stage outputs you will consume; refinedstage lane is LIVE — do not touch its files).
- Real artifacts: runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/*.json;
  runs/lanes/pbv11_headtohead_20260713/rerun_20260715/cpu_events_full/pbvision_11min_20260713/;
  eval_clips/ball/*/labels/.

## 5. MANDATORY STRUCTURED REPORT (report.json via --output-schema; manager rules on THIS)
- objective_result: PASS only if D1 (all sub-items) + D2 + D3 delivered.
- acceptance rows: one per D1..D3, plus one row per D3 baseline number
  (metric/baseline=null/after=<number>/target="computed"/verdict).
- changes: [] plus a listing of lane-dir files created.
- full_suite: command="NOT RUN — docs-only lane, zero repo edits (manager deviation);
  git status proof in honest_issues", passed=0, failed=0, skipped=0,
  failures_all_preexisting=true, failure_notes=<the git status --porcelain evidence>.
- honest_issues: every gap, disagreement, uncomputable metric, generation-2-unexercised input.
- next: the implementation lane slicing in one paragraph.
- session_id: your codex session id.

## 6. ANTI-PASSIVE-WAIT
All work is CPU-local; finish in one session (~90-120 min). Ending your turn to wait = lane
death; you will NOT be re-woken. If a probe is slow, bound it and move on.

## 7. BEST-STACK DELTA
(c) No stack delta — design documents + probe scripts only; no model/weights/policy change.
State this in the report.
