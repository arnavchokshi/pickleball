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

## oneworld_impl_20260716 — RULED: ADOPT (scoped pass), COMMITTED a54b7c451 (wrap-up window)
Lane-honest PARTIAL accepted; the fused world is REAL, reproducible, and watchable.

Manager verification, personally executed, REAL unpiped exit codes:
- build EXIT 0; INDEPENDENT MANAGER REBUILD BYTE-IDENTICAL to the lane's artifact
  (sha256 2c58c4a8bc8461f669f2bd3890d51d9fa50ce4d1fb1663d6ff96e7cd43f4f721) — reproducibility
  proven, not asserted; validate EXIT 0 (valid:true, 0 warnings); focused 18 passed EXIT 0;
  test_scaffold_tool_index 3 passed REAL_EXIT=0 (A13 CLOSED); raw immutability 14/14.
- Artifact inspected directly at frame 78: 4 players (placement_fused, conf .82-.92), 4 paddles
  each with display_pose_world (status unresolved_legacy_wrist_proxy), ball arc_measured
  [0.97,3.64,1.75] conf .94. 300 frames, 28 typed events. VERIFIED=0/preview_only/render_only/
  not_for_detection_metrics/raw_inputs_mutated=false all present.

THE FINDING THAT MATTERS (owner-facing): ZERO of 24 declared contacts confirmed — 22 unsupported
(>1.2m from every wrist), 2 too_close_to_call, 0 hitter attributions. Frame 78: declared hitter's
wrist 11.17m from the arc ball. The pass abstained instead of dragging the ball 11m to satisfy a
declared event, and left the raw event immutable. This is the design's bounded-abstention rail
doing exactly its job, and it converts a vague "the world looks wrong" into a precise, ranked
upstream question: on this bundle, contact events + arc ball + player placement genuinely
disagree. That is the honest state of the stack, now measurable per event.

Metrics vs frozen baselines: M1 <=0.60m NOT met and NOT claimed (0 supported events — a median
over zero is undefined and the lane correctly refused to invent one); M3 coverage_measured 0.39
= baseline (no coverage lost, none fabricated); M5 non-regressing (player p90 61.2683 ->
61.2658px); M2 honestly 0/0 (upstream anchors arrive exactly at z=r_b — synthetic nonzero tests
prove the pass itself never snaps); M4 denominator 0 (no gen-2 hypotheses exist on any run).
Priority ruling honored: A10/A11/A12 all PASS — watchability delivered over metric polish.

Accepted debts (stated, not hidden): A9 wide suite time-boxed at ~34% with 15 unrelated
dirty-tree failures attributed but NOT proven against clean HEAD — the one real gap in this
ruling; concurrent-tree churn made a clean wide run impossible in the window. Gen-2 regen EXIT 1
(tracking reuse refused without migration attestation) = a genuine finding routed to Track C.
Lane again printed protected label paths in a git-status diagnostic (paths only, no content) —
second occurrence of the same class; the exclusion rule is now a standing Track K spec line.

## ANCHOR-SOURCE HIERARCHY + the probe that tested it (owner framing, 2026-07-16 wrap-up)
Owner/coordinator framing folded into DESIGN.md §8.5 so it survives the session: pb.vision's
ball-3D works on ANCHOR DENSITY (trained event heads -> bounce z=radius 154/154, net plane,
radius head R²=0.71 depth; gravity fills between). Our arc failure = anchor sparsity (20/697s) —
same fact, other side. They have NO 3D players; we have correctly-placed BODY (foot-slide
34->7mm). Therefore contact co-location = an anchor class they structurally cannot copy:
(1) contact at hitter's hand [3D MEASURED, ours alone, zero trained heads needed],
(2) bounce at court plane [3D, both], (3) net plane [3D, both], (4) trained-event/radius depth
[theirs today, Track G building].

MANAGER PROBE (anchor_window_probe.py, EXIT 0, read-only) — tested whether the class yields on
Wolverine, WITH a chance baseline (the check that made it trustworthy rather than wishful):
| stat | declared contacts n=24 | chance (random non-event frames) n=24 |
| dist at declared frame, median | 3.126m | — |
| windowed closest-approach, median | 1.167m | 4.499m |
| <=0.50m (paddle band) | 6/24 | 0/24 |
| <=1.20m | 15/24 | 6/24 |
VERDICT: the co-location signal is REAL, ~4x above chance — but v1 emits ZERO anchors today
because §3.5 evaluates at the DECLARED event frame, and those frames are mistimed (best-approach
offsets spread -15..+13 frames, several pegged at the window edge = some true offsets exceed
0.5s) with attribution near coin-flip (9/24 declared hitter == nearest player). The class is
BLOCKED UPSTREAM, not disproven. 0/24 within 0.30m is physically EXPECTED (ball sits ~0.3-0.5m
from the wrist at true contact — paddle length), so it is not evidence against.
UNLOCK (specified, deliberately NOT applied): bounded closest-approach search inside the event
window w/ per-clip chance-margin gate + independent wrist-speed agreement, emitting a proposed
measured anchor + its dt as an honest timing correction, raw events immutable. v2 behavior —
needs failing-first tests + pre-registered kill rule. This is the anchor class Track A's solver
is starving for, available with no trained event head.

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
