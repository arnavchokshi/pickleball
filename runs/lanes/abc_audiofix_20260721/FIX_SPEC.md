# FIX ROUND for abc_audiofix_20260721 — address ultra-review REJECT (runs/lanes/abc_audiofix_20260721_review/review.json)

Same HARD RULES + FILE OWNERSHIP as spec.md (build_abc_arm_manifests.py + test_abc_arm_manifests.py
ONLY). Read the full review JSON first; implement EVERY required_fix for all four findings:

- BALL_FAMILY_SPOOF (CRITICAL): the CLI slot must not determine family identity. Require ball
  artifacts to be schema-v1 racketsport_ball_inflections with source=ball_track_image_motion and
  world_frame=image_xy; require audio artifacts to declare artifact_type=racketsport_audio_onsets,
  detector_version=audio_onset_pop_v2, source=video_audio_pop_v2; validate + record upstream
  provenance (ball_track_source SHA/path, pts identity). Rejection tests: exotic artifact,
  pb.vision-derived, audio-disguised-as-ball, case-variant family strings.
- DUPLICATE_EVENT_ID_ALIAS (CRITICAL): require nonempty string event IDs unique per video BEFORE
  matching; reject string-coercion collisions (1 vs "1"); key matching by internal unique index;
  recheck emitted cue-to-event delta. Duplicate-ID and distant-alias regression tests.
- NULL_DURATION_GAMING (CRITICAL): derive the circular period from verified frame PTS + validated
  terminal-frame duration (or tightly reject inconsistent declared duration); require
  racketsport_frame_times artifact type + frame_count + media declaration. Inflated-duration
  refusal test (the reviewer's repro: duration=1000s flipped beats_null false->true — must refuse).
- FRAME_TIMES_IDENTITY_OMISSION (HIGH): frame-times must declare media identity equal to the staged
  media SHA (hard refusal per VM_ABC_RUN.md §2); record declared_media_sha256 separately from the
  inferred staged value; missing-identity test.

Invariants that must SURVIVE unchanged (regression-check them): audio-only accepted = 0; corrected
real-decisions recount = 1,189; kink-only 0.25 / both+null-pass 0.5 / both+null-fail 0.25;
C mirrors corrected B; byte-identical determinism; CLI flags unchanged.
CAUTION: the real VM artifacts must still pass — the new validation requirements must match the
REAL artifact schemas (inspect real samples under runs/lanes/abc_experiment_20260721/vm_pull/ and
the builders build_ball_inflections.py / build_audio_onsets_v2.py to pin exact field names — do not
invent field values the real artifacts lack; if a real artifact lacks a required field, report it
as a HARD BLOCKER rather than weakening the check).
Report to report_fix1.json (schema-valid). Wide suite: no NEW failures vs the known environmental set.
