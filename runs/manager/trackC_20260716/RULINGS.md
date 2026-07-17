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

## Wave 3 (2026-07-16 morning, post-thaw)

### refinedstage_20260716 — ADOPT (scoped pass), commit d941b0d7d
events_refined + ball_arc_refined are explicit first-class timed stages (the ~122s out of
`world`; guard timeouts typed-degrade per Track A af6b8d40f; reuse gating unchanged; one-time
refined-artifact rebuild noted); stage counts/docs coherent; evidence17's booked dependency-
hashing hunks re-derived + applied w/ reuse-refusal test. Dispatched only after the Track A
gate opened. Manager verification: focused 195 passed; the ONLY 2 failures are Track G's
unregistered large tracked artifacts (runs/lanes/event_head_scaffold_20260716/dataset/
manifest_{a,b}.json, commit 40b013ab2) — proven via the storage audit; --help EXIT 0.

### calpolicy_20260716 — ADOPT (scoped pass), commit 5cb556fd2
Implements the Track C policy ruling: `line_evidence_solved_preview` external-calibration
source class — ingestion only w/ space/distortion/residual/provenance declarations,
permanently preview-band, adversarially pinned to never satisfy metric_15pt_reviewed gates,
reviewed-class byte-parity, real pbv11 solved artifact as read-only fixture. Manager
verification: 47 passed EXIT 0. Ruling grounds recorded in the commit and the North Star
CAL row (§1.4 two-axis; preview-seed precedent; rule 12 honored by band; NS-03.CAL
authority kill-rule untouched).

### Cross-track flag (not Track C's to fix)
Track G commit 40b013ab2 tracks two >threshold artifacts without storage-allowlist
registration -> test_truthful_capabilities storage tests fail repo-wide (2 failures in every
suite run since). Routed to the coordinator for Track G.

### North Star notes landed with this wave
P0-G structural clauses wired (scoped); CAL row policy note (5cb556fd2); queue row 1
reduced to diameter-wire + ablations. Doc tests 12/14 w/ the two Track-G failures attributed.

## Wave-3 close verification (manager-run, real exit codes)
Wide suite ran during a heavy cross-track landing window (5+ commits mid-run): 3741 passed /
33 failed / 24 skipped EXIT 1. Post-settlement attribution, every failure accounted:
- 6 CLI-help + most others: mid-run contamination (Track I CLIs committed mid-suite) —
  green solo (318/319 with scaffold the lone survivor).
- 1 scaffold-index: Track K's unregistered one-world CLIs (their A13 fix injected, 71477b4ee).
- 2 storage-policy: Track G's manifests — FIXED by them same-day (fd5bc1da5); doc tests
  re-verified 14/14 EXIT 0 after their fix.
- 24 court-family: fail solo BUT attributed to ANOTHER track's in-progress court-label
  landing in the worktree (untracked owner_IMG_1605 partial frames incl. the exact
  frame_000060 the scanner now discovers, new pbvision court_keypoint_frames dir = the
  6==5 sixth sample, five modified court_keypoints*.json) — data-count pins, zero
  schema/source-class signatures, calpolicy modules untouched by the failing tests.
  Routed to the CAL/MOVE-1 owner to land data + test pins together.
Track C-owned surfaces: all green (focused 195+47, doc pins, --help, 318/319 CLI).
