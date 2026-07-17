# Track K rulings — 2026-07-16 (one-world fusion, NS-04.4/04.5)

## oneworld_design_20260716 — RULED: ADOPT (as-design, scoped pass), lane-honest PARTIAL accepted
Lane: codex gpt-5.6-sol high, session 019f6bb1-b2f9-7723-917e-c7fe34ce1235, nohup-detached.
Deliverables: DESIGN.md (buildable one_world_v1 design), FIELD_VERIFICATION.md, field_probe.py/
baseline_probe.py + outputs, report.json (objective PARTIAL, D1/D2/D3 all PASS).

Manager verification (real unpiped exit codes, personally executed):
- baseline_probe re-run: EXIT 0, output BYTE-IDENTICAL to banked
  (rerun==banked True). Baseline numbers CONFIRMED on Wolverine v5.1:
  ball-at-contact center distance median 8.1308 m / p90 11.3222 m;
  wrist-volume residual median 7.9737 m / p90 11.1651 m (24/24 events);
  world coverage @conf>=0.5 = 0.39 (117/300 rally frames).
- Forensic root-cause check (frame 78, event 10): 2D ball (578,202) conf 0.94
  sits INSIDE player 4's bbox while contact_windows declares player_id=1
  (bbox 1151-1588 x 397-1080, across the image); arc ball ~3m from p4, ~9m
  from p1. The baseline world asserts "p1 hit a ball visibly at p4" — the
  incoherence is REAL (event attribution + sparse-anchor arc), the probe join
  is CORRECT. This is the owner directive's exact target: single-modality
  outputs contradict; confidence-weighted fusion (bounded hitter inference,
  declared-ID prior capped at 1.25x) is the cure.
- Field spot-checks: racket_pose_estimate.json = court_Z0/m/render_only/
  wrist_proxy/ambiguous:false (matches gen-1 "unresolved_legacy_wrist_proxy"
  treatment); audio_onsets_v2.json = blocked/0 onsets/no_audio_stream.
- Label incident BOUNDED: an early broad rg printed protected Outdoor/Indoor
  label FILE PATHS/status lines into the transcript (log.txt:9231-9235). Zero
  label content in any deliverable (probes attest outdoor_indoor_labels_read:
  false; deliverable grep clean). Honest self-report; lane correctly refused
  to claim PASS. LESSON BOOKED: all future Track K specs mandate explicit
  eval_clips/ball/{outdoor,indoor}* exclusion globs for every search sweep.
  log.txt intentionally NOT committed (transcript; contains the printed paths).
- Fence: PASS — lane wrote only its lane dir; every dirty tracked file
  belongs to other tracks' live lanes (refinedstage/anchorfusion/calpolicy/
  trackI/GPU lanes).

Design compliance highlights (verified against NS-04.4/04.5 + reuse contract):
finite soft surface weights + caps, never snap (z=r_b never assigned; tests
must prove no clamp); raw observations immutable (new artifact only, input
sha256s retained); both IPPE poses survive to an independent-evidence resolver
that NEVER uses reprojection to choose (reuse-contract ban honored); bounded
abstention (contact refinement abstains >1.2m residual — the 7.97m baseline
cannot drag the ball); absence semantics everywhere; permanently preview-band,
render-only, not_for_detection_metrics, not_for_training, VERIFIED=0.

Cross-track confirmations folded into implementation:
- Track I SCHEMA.md (landed after design): consumption contract ALIGNED —
  per-frame placement_trajectory_refinement.{rigid_correction_xyz_m,
  refined_transl_world,covariance_m2,provenance.evidence} + top-level
  provenance.inputs.*.sha256 feed the design's identity gate as-is. No change
  requests from Track K. Their rule ("consume correction+covariance as a
  candidate factor; never overwrite/relabel BODY/TRK") = Stage A verbatim.
- Track H FUSED_WORLD_VIEWER_READINESS.md: schema asks satisfied by
  OneWorldV1 draft (per-entity confidence+band+source refs, absence sentinels,
  artifact-level trust_band). Track K decision on their open question: a
  DISTINCT "fused" evidence-provenance origin class — physics_predicted is
  NOT overloaded. fused_world_url manifest wiring = post-v1, with Track C/H.
- Shared-file contention ruling: schemas/__init__.py is live-owned (calpolicy)
  -> OneWorld Pydantic models live in the NEW module; the ARTIFACT_MODELS
  registry hunk is DEFERRED as an inline diff in the implementation report
  (evidence17 deferred-hunk precedent). list_scaffold_tools.py additive dict
  entries only, fresh-read, Track I/G lines untouched, manager commits
  hunk-selectively.

OWNER DIRECTIVE COMPLIANCE CHECK (2026-07-16, relayed by coordinator: contacts must be a
combination of audio + joint/wrist swings + ball tracking + proximity; no single signal may
declare a contact — neighbor-court audio bleed): design VERIFIED COMPLIANT, two layers.
(1) Contact EXISTENCE is upstream-fused (event_fusion.fuse_contact_windows fail-closes without
wrist_vel + ball_inflection; evidence17 audio is non-gating soft evidence that cannot
create/gate/veto, bounded +/-0.20 log-odds) — bled audio cannot create an event, and every
event carries a sources{audio,wrist_vel,ball_inflection,human_review} evidence vector.
(2) The design's Stage C confirmation likelihood is a PRODUCT (ball term x wrist term x event
confidence x marker reliability) with a null hypothesis — any missing modality means
unsupported/abstain, and a declared contact with no wrist near the ball is flagged
unsupported in the fused world (raw event immutable). Implementation spec AMENDED (A8b) to
make this testable: per-event contact_evidence_vector emission + audio-only-cannot-confirm +
neighbor-court-bleed-creates-nothing + co-location-discount tests, all failing-first.

BEST-STACK DELTA: (c) none — documents + probes only.
VERIFIED=0 binding; nothing promoted.

NEXT: oneworld_impl_20260716 dispatched (slices 1+2 of the design's
lane plan: core module + CLIs + tests + scaffold rows + Wolverine
baseline->fused metric table + honest demo partial). Slice 3 (runner stage
175 wiring) remains a Track C-owned integration lane; request text is staged
verbatim in DESIGN.md §7.

## DESIGN AMENDMENTS 1+2 (owner win condition, relayed 2026-07-16) — RULED: ADOPT into impl
Owner: demo videos must show 4 placed players, each w/ semi-decent paddle orientation, and
continuous well-placed 3D ball w/ visible paddle/floor/net events; "then I say what I see wrong
and we fix." Priority: demo-visible completeness w/ honest bands > per-metric polish.
Injected mid-flight into oneworld_impl_20260716 as spec_addendum_A10.md (session resumed with
addendum; old PID killed cleanly, resume PID recorded in codex.pid):
- AMENDMENT 1 (A10): paddle ALWAYS emits display_pose_world + display_tier w/ honest band even
  when unresolved (best-evidence hypothesis below resolve bar; display-only reprojection
  tie-break allowed and recorded; gen-1 wrist proxy carried as unresolved_legacy_wrist_proxy).
  NOT promotion: status/flags/both retained hypotheses unchanged. NS-04.5 "one early hypothesis
  never promotes" preserved — this is band-carried display, resolve bar untouched.
- A11: consolidated first-class typed events[] (paddle_contact/floor_bounce/net_contact/
  net_cross w/ t, frame, locations raw+refined, hitter_id, confidence, band, evidence,
  provenance) for the viewer; refinement lists remain.
- AMENDMENT 2 (A12): ball continuity chain via estimate_tier (arc_measured -> physics_predicted
  bounded 0.5s-from-support -> ray_court_projection altitude_unknown -> absent only when no 2D).
  HONESTY RAILS: metrics computed ONLY on tier-1; M3 coverage reported STRATIFIED
  (coverage_measured comparable to frozen 0.39 baseline; coverage_with_predicted labeled);
  tier counts reported to the owner. Supersedes the design's strict display-absence for the
  ball/paddle DISPLAY surfaces only; absence sentinels + raw immutability + preview band +
  VERIFIED=0 unchanged.
