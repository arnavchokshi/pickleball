# Track C rulings — NS-01.6/01.7 spine + P0-H remainder wave (2026-07-16)

Manager: Track C (Fable), serialized owner of `scripts/racketsport/process_video.py` this
window. VERIFIED=0 binding; scoped passes only. All exit codes below manager-run.

Context: all three lanes were killed overnight by Mac sleep AFTER writing report.json;
work reconciled from disk, verified on the settled tree, ruled, committed by explicit path
(Track A ball_arc files, Track D ios/ files, and a concurrent-session review CLI preserved
untouched).

## Lane rulings

### spine16_20260716 — ADOPT (scoped pass), commit ffb7e0975
One authoritative stage graph (canonical definition drives serial+overlap); legacy
`threed/racketsport/pipeline_cli.py` duplicate DELETED with readiness migrated
(pipeline_contracts + validate_pipeline_artifacts --public-contracts) and doc/test pins
updated; typed ExpectedOptionalAbsence + master catch-all rewritten so unexpected
exceptions FAIL loudly (the old hide-as-degraded behavior and its two codifying tests
retired); frame-schedule completeness (silent equal:True defaults killed, loud-path
runner test, plan-coverage cross-check); cold/reuse/partial/failure families under the new
contract. Micro-fix (same session): out-of-fence test_best_stack_resolution fake
materializer updated to the new mandatory validation shape — contract not weakened.
Manager verification: focused+hygiene quartet 44 EXIT 0; test_process_video 168 EXIT 0;
--help EXIT 0; best_stack_resolution 9 EXIT 0.

### tbcam_20260716 — ADOPT (scoped pass), commit 1685a8878
P0-H remainder representation/transform slices: typed scale/rotate/crop intrinsics
transforms (pinhole-only policy explicit); both ad-hoc scalers routed parity-first;
optional cross-language-safe CaptureSidecar reference_crop + rolling_shutter fields
(goldens valid); loud orientation-mismatch at the calibration seam; RollingShutterModel
populated-or-explicitly-missing in the timebase contract. Manager verification: 93 EXIT 0.
PENDING: Swift emission (iOS), physical device evidence (owner), row_time consumers.

### evidence17_20260716 — ADOPT (scoped pass), commit 8a282d4db
Audio soft evidence (pop_band_ratio et al) into fusion, bounded and non-gating, no raw
averaging, no-features path byte-parity; BOTH IPPE poses retained (alternate hypothesis +
carry-ambiguous-flagged, primary parity-pinned); repaired-confidence markers (values
unchanged) in ball_temporal_filter/player_id_repair/pose_temporal. Manager verification:
42 EXIT 0. DEFERRED (booked): contact-dependency-hashing runner hunks + blur/diameter-
into-events design (inline in its report.json) — next runner integration window.
Classifier-gated audio slices PENDING (trained classifier does not exist; spot-check FAIL
2026-07-15 keeps bootstrap labels blocked).

## Held work
- Explicit timed post-BODY refined events/arc stages: HELD pending Track A
  ballarc_scale_guard ruling (coordinator sequencing); scout map banked in the 07-15/16
  session records (world-stage seams at process_video.py _stage_world).
- evidence17 deferred runner hunks: booked for the next process_video.py window.

## Wave-close verification
- Wide suite: see wave2_wide_suite.log (result + attribution recorded at close).
- Per-lane focused: 44 / 93 / 42 / 168 / 9 — all EXIT 0, manager-run.
