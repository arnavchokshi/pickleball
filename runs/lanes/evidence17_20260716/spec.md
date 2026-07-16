# Lane evidence17_20260716 — NS-01.7 non-ball_arc evidence plumbing: audio soft evidence into fusion, both IPPE poses, repaired-confidence markers

You are a Codex implementation lane for the DinkVision pickleball repo at
/Users/arnavchokshi/Desktop/pickleball. VERIFIED=0 binding; "wired"/"scoped pass" at most.

## HARD RULES
- NO branches/commits/pushes; manager rules and commits.
- Read first: NORTH_STAR_ROADMAP.md (§4 NS-01.7 row + §3.1 reuse contract rows for
  BALL/audio and Paddle, §6 standing rules), runs/manager/inflight_lanes.md (live fences).
- CONCURRENT-LANE FENCES (hard): spine16_20260716 owns scripts/racketsport/process_video.py
  (+ process_video_body_frames, pipeline_cli/contracts, AGENTS/RUNBOOK,
  test_process_video.py, test_truthful_capabilities.py). tbcam_20260716 owns
  threed/racketsport/{schemas/__init__.py,coordinates,court_calibration,io_decode,timebase,
  sam3d_body_input_prep,court_auto_evidence}.py + their tests. Track A owns
  threed/racketsport/ball_arc_*.py + tests. You may NOT touch ANY of those files. Runner
  changes you need (e.g. contact dependency hashing) = exact diff hunks INLINE in your
  report as DEFERRED changes; do not apply.
- NS-01.7 stop rule (quote it): "Do not raw-average modalities or promote on
  residual/overlay gains." Audio remains NON-GATING on the normal path; review_only /
  not_classifier warnings stay.
- Preserve unrelated dirty work (configs/ssh/a100_known_hosts, ios/*,
  scripts/racketsport/build_event_review_session.py, brand-exploration/, cvat_upload/, data/).
- PYTEST EXIT-CODE TRAP (3b639768c): no pipes; literal `$?`; report numbers.
- Wide suite mandatory; manager baseline 3684/24/1 EXIT 1 (single failure = a concurrent
  session's untracked build_event_review_session.py — attribute, don't fix); expect
  concurrent-lane noise; attribute per-file, never edit other lanes' files.
- Artifacts under runs/lanes/evidence17_20260716/. Raw observations immutable.

## EXPLICIT FILE OWNERSHIP (edit ONLY these)
- threed/racketsport/event_fusion.py
- threed/racketsport/racket6dof.py
- threed/racketsport/racket_stage_runner.py (and racket_pose_preview.py if its call
  signature must follow)
- threed/racketsport/ball_temporal_filter.py
- threed/racketsport/player_id_repair.py
- threed/racketsport/pose_temporal.py
- Tests: test_event_fusion*.py, test_racket6dof*.py / racket stage tests,
  test_ball_temporal_filter*.py, test_player_id_repair*.py, test_pose_temporal*.py,
  new test_evidence17_*.py.
- FORBIDDEN even though adjacent: audio_onsets*.py (tbwire-landed, read-only),
  paddle_pose_fused.py (separate wrist/grip pipeline — out of scope), ball_physics_fill.py,
  ball_anchor_evidence.py, ball_global_track.py.

## OBJECTIVE (North Star NS-01.7 row, quote in report)
"Make classified audio affect events; hash contact dependencies; pass blur/diameter; retain
both IPPE poses; mark repaired confidence" — the non-ball_arc, non-runner slices, honest
about what waits for a trained classifier.

Manager-verified ground truth:
1. AUDIO: fusion is advisory-only on the normal path (event_fusion.py:168-235 — audio only
   augments visual proposals; audio-primary loop needs require_audio=True). The payload
   already carries per-onset soft evidence (features: spectral_flux,
   high_frequency_content, band_energy_delta, pop_band_ratio — audio_onsets_v2.py:385-401)
   but _coerce_audio_onset (event_fusion.py:444-466) and AudioOnsetCandidate (:66-75) keep
   ONLY time_s + confidence — the soft evidence is discarded before fusion.
2. IPPE: racket6dof.estimate_planar_paddle_pose_with_diagnostics gets BOTH solutions from
   solvePnPGeneric (:251-253) but keeps only scored[0] (:268-278); the second rvec/tvec is
   discarded (only scalar errors survive); PlanarPaddlePoseEstimate (:64-96) has no second-
   pose slot; racket_stage_runner defaults reject_ambiguous=True (:54) and DROPS ambiguous
   frames.
3. REPAIRED CONFIDENCE: ball_temporal_filter.py:475-488 synthesizes frame["conf"] =
   min(left,right)*0.5 on gap interpolation with only approx=True (no conf_source/repaired
   marker); player_id_repair.py:454/:524/:530 rebuilds track conf unmarked;
   pose_temporal.py:2060-2075 recomputes joint confidence unmarked. Good precedents to
   follow: ball_physics_fill source="physics_interpolated" provenance block;
   ball_on_device_gate confidence_source.

## DELIVERABLES (numbered; honest PARTIAL allowed)
1. AUDIO SOFT EVIDENCE INTO FUSION (weak-heuristic "classified", non-gating): extend
   AudioOnsetCandidate + _coerce_audio_onset to carry the existing features (at minimum
   pop_band_ratio); when an onset augments a visual proposal, let pop-likeness modulate the
   fused confidence and window tightness in a bounded, documented way (NO raw averaging —
   justify the combination rule; audio still cannot create/gate/veto a contact). Raw onset
   values preserved unchanged. Tests: augmentation with high vs low pop_band_ratio differs
   as documented; absence of features field behaves exactly as today (byte-parity on the
   no-features path); review_only warnings untouched. What must WAIT for the trained
   classifier (typing, gating, audio-primary) — state it explicitly as PENDING.
2. RETAIN BOTH IPPE POSES: PlanarPaddlePoseEstimate gains an alt_pose (full second
   SE3+confidence) populated from solvePnPGeneric's second solution; primary selection
   UNCHANGED (parity on every existing test/fixture digest); racket_stage_runner carries
   ambiguous frames WITH both poses and an ambiguous flag instead of dropping them
   (reject_ambiguous default flip documented as the North Star-mandated behavior change —
   "retain both IPPE poses" / never "discard the second IPPE pose by reprojection alone");
   the per-frame racket pose artifact carries both hypotheses + ambiguity_margin_px.
   Fail-closed if the second solution is degenerate. Tests: both poses survive to the
   artifact; primary identical to before; downstream readers of the artifact unaffected
   (additive fields).
3. REPAIRED-CONFIDENCE MARKERS (additive, values unchanged): ball_temporal_filter gap
   interpolation marks conf_source="interpolated_min_half" (or similar typed marker) per
   repaired sample; player_id_repair marks repaired track confidences; pose_temporal marks
   interpolated joint confidences. Follow the ball_physics_fill/ball_on_device_gate marker
   precedents. Tests: markers present exactly on repaired values, absent on measured ones;
   downstream consumers unaffected (additive).
4. DEFERRED RUNNER HUNKS (report-only, do NOT apply): the contact-dependency-hashing
   additions to process_video.py _contact_dependency_paths — frame_times.json, tracks.json
   (fps source), wrist_velocity_peaks*.json, ball_candidates.json, coarse
   ball_inflections.json, and the audio-config identity question — as exact inline diffs
   with rationale, for the spine16/manager integration window. Similarly any blur/diameter-
   into-events proposal: ANALYSIS ONLY this lane (the honest events-side wire for
   blur/diameter needs the ball_blur sidecar generated on the normal path, which is a
   runner change — describe the design, do not implement).

## MANDATORY VERIFICATION (literal exit codes, no pipes)
- All owned-file focused tests EXIT 0; parity digests pinned before/after for the
  racket-pose primary path and the no-features audio path.
- Full wide suite with per-failure attribution vs the manager baseline.

## MANDATORY STRUCTURED REPORT at runs/lanes/evidence17_20260716/report.json (write it
yourself; schema docs/racketsport/lane_report.schema.json): objective_result per
deliverable; full_suite counts + literal exit codes + attribution; HONEST ISSUES (classifier
PENDING, deferred runner hunks); BEST-STACK DELTA (expected "(c) none" — the
reject_ambiguous flip is a correctness/contract change, not a model/policy selection; state
it); dated inflight note paragraph.
