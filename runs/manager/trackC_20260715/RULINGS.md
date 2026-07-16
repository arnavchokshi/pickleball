# Track C rulings — NS-01.4/NS-01.5 core wiring window (2026-07-15)

Manager: Track C (Fable), serialized integration owner of `scripts/racketsport/process_video.py`
for this window (North Star rule 9). VERIFIED=0 binding throughout; nothing here is a promotion.
"Wired" and "scoped pass" at most. All exit codes below were produced by the manager directly
(not lane-reported): the piped-pytest trap (commit 3b639768c) gates every ruling.

## Baselines (HEAD ac0b14ab0 before this window)

- tests/server + tests/render_service: 150 passed, EXIT 0
  (runs/manager/trackC_20260715/baseline_server_suite.log)
- tests/racketsport wide suite (3,687 collected): 3662 passed, 24 skipped, 1 failed, EXIT 1
  (baseline_wide_suite.log). The single failure
  (test_runbook_documents_current_process_video_entrypoint, assert 8284 > 8604) is a
  MID-RUN CROSS-CONTAMINATION artifact, not a pre-existing or current failure: pytest
  imported the OLD expected_order pin at collection (15:42), statusdocs rewrote RUNBOOK.md
  mid-run, and the stale pin asserted the obsolete manifest->match_stats order against the
  corrected doc. Manager re-verified the adopted pair in isolation TWICE on the settled
  tree: 14 passed, EXIT 0 both times. Effective HEAD baseline for lane comparisons:
  3663 passed / 24 skipped equivalents.
- ios Swift package (Upload-filtered): 42 executed, 0 failures, EXIT 0 — includes
  testPartialJobPreservesMissingCapabilitiesTrustBandsAndUnknownProvenance (the exact P0-E
  app clause)

## Pre-lane audit finding (manager + read-only scouts)

NS-01.5 was already substantially landed at engineering level on HEAD, contrary to the queue
row's framing:
- Runner: `_minimum_bundle_missing_capabilities` (North Star §3.2 list) + honest
  failed/partial/complete + `_missing_local_manifest_urls` (every advertised local URL).
- Server: `server/bundle_policy.py::evaluate_bundle` never upgrades; `gate_reported_status`
  fail-closed; the hardcoded `complete` from `gpu_runner.py` (L278/L446) is overridden in
  `render_app._execute_job` before any "Replay ready" string; worker daemon re-evaluates
  post-publish with `s3_url_exists`; local vs SSH agree via one evaluator + two URL resolvers.
- Packaging: `stage_manifest_delivery_bundle` — recursive closure (rglob incl.
  body_mesh_index/), temp-dir staging, atomic exchange (renamex_np / RENAME_EXCHANGE,
  loud failure if unavailable), manifest written+swapped last.
- App: partial never displays "Replay ready"; missingCapabilities carried.
The remaining honest NS-01.5 gaps were the stale doc/test pin (below) and the owner-gated
physical trace (NS-01.2b).

## Lane rulings

### statusdocs_20260715 — RULED ADOPT, committed 9bf8eef75
Booked docfix follow-up executed: RUNBOOK numbered Stage Order block now matches the
enforcing code (20 default stage outcomes; 18 match_stats -> 19 coaching_facts ->
20 manifest; obsolete "match_stats after manifest" sentence removed); `expected_order` in
test_truthful_capabilities.py guards the true tail including coaching_facts; Status
Interpretation honestly split (scoped runner/worker/db/API test proof; app code-wired;
physical trace owner-gated open). Manager verification: truthful 14/14 EXIT 0;
tests/server + tests/render_service 150 passed EXIT 0. Fence held (2 files only).
No stack delta.

### tbwire_20260715 — RULED ADOPT (scoped pass, wired), committed bd99c6d11
Typed timebase contract wired through the real decode seam, canonical-beside-legacy:
- ingest writes schema-valid `timebase_contract.json` (exact ffprobe integer ticks +
  timescale, strict monotonicity, per-frame typed availability/absence) +
  `timebase_decode_evidence.json`; legacy `frame_times.json` values byte-identical
  (Wolverine frame-value SHA pinned unchanged; 300/300 typed accounting vs legacy 299 with
  an explicit compatibility declaration).
- `time_for_frame` silent index/fps fallback removed: missing table entries fail loudly;
  CFR only via explicit fps argument with recorded provenance.
- `SensorClockMapping` outcomes from sidecar dual-clock pairs (>=2-pair offset+drift fit
  w/ residual; typed missing reasons proven on all three golden sidecar fixtures).
- events dependency identity + audio-onset ordering provenance reference the same contract;
  no-contract audio bytes unchanged.
Manager verification (real exit codes, local env): focused
timebase/io_decode/audio_onsets/process_video = 216 passed EXIT 0; the lane's 8 wide-suite
"bind" failures = 57 passed EXIT 0 locally (codex-sandbox-only artifact); the 4
test_ball_physics_fill failures on the current tree are REAL but attributed to the
concurrent coordwire lane's in-progress edits (pass at HEAD) — carried to coordwire's ruling.
Lane survived an external 1h background-task kill; resumed via detached codex session.
PENDING honestly: P0-H physical 30s/5min captures (owner); native-intrinsics scaling +
rolling-shutter device slices (out of fence — NS-01.6/01.7 follow-on); corrected-beats-raw
(NS-02 labels). No stack delta.

### statusdocs third command (deferred at its ruling): test_process_video.py re-verified
inside the tbwire focused run above (216 passed EXIT 0 includes the full file).

### coordwire_20260715 — RULED ADOPT (scoped pass, wired), committed aab8c3098
Typed coordinate API adopted by the five remaining raw-math stage consumers (placement,
ball_court_filter, ball_physics3d, ball_inout_uncertainty, virtual_world), canonical-beside-
legacy: fail-closed HomographyPixelConvention resolution, typed undistort/H^-1/planar
project/unproject/pinhole adapters, declared raster/world spaces around unchanged cv2 calls.
Zero-tolerance parity: six SHA-pinned Wolverine real-fixture digests byte-identical pre/post;
nonzero-distortion synthetic parity + wrong-space rejection per seam (real Wolverine
calibration is zero-distortion — distortion coverage is synthetic; honestly declared).
Manager verification (real exit codes): 22 parity/adoption EXIT 0; 165 touched-stage EXIT 0;
57 virtual-world EXIT 0. No process_video.py hunk needed. Fence held. No stack delta.
The lane also isolated the wide-suite ball_physics_fill failures to tbwire (below).

### tbwire empty-table regression fix — RULED ADOPT, committed c4dfb2d8b
coordwire's controlled experiment (ball_fill_timebase_attribution.log) overturned tbwire's
initial cross-lane attribution: the 4 test_ball_physics_fill failures were caused by the
adopted tbwire guard raising on an EMPTY frame-times mapping (historically CFR-with-fps).
Manager confirmed the mechanism by reading ball_physics_fill.py:297 (eager .get() default)
and the new time_for_frame guard. Fix (tbwire session, one line + regression test): raise
only for a NON-EMPTY table missing the frame; empty lookup behaves as no-table (explicit-fps
CFR with recorded provenance). Manager verification: test_ball_physics_fill 11 passed EXIT 0;
focused io_decode/timebase/process_video/audio_onsets 217 passed EXIT 0.

### Lesson booked (process): both heavy lanes were killed at exactly 1h wall-age by a
background-task cap and were resumed via detached `codex exec resume` under nohup; Mac
sleep also kills waits — reconcile from disk on every resume. Lane-report attributions of
cross-lane failures must be re-verified by the manager (tbwire's first attribution was
wrong; coordwire's controlled experiment was right).

## Wave-close verification (manager-run, real exit codes)

- Wave-close wide suite (post-adoption tree): 3684 passed, 24 skipped, 1 failed, EXIT 1
  (waveclose_wide_suite.log). The single failure
  (test_real_scaffold_tool_index_matches_checked_in_schema) is caused by a CONCURRENT
  SESSION's brand-new UNTRACKED CLI (`scripts/racketsport/build_event_review_session.py`,
  created 20:11 mid-suite, not yet registered in the scaffold index). Proof it is not Track
  C's: the test passes 3/3 EXIT 0 on a pristine `git archive HEAD` export
  (/tmp/scaffold_head_verify.log). File preserved untouched per the unrelated-dirty-work
  rule; registration belongs to its author's session.
- Per-family manager verifications this window (all EXIT 0): fill 11/11; tbwire focused 216
  and 217 post-fix; coordwire parity/adoption 22 + touched-stage 165 + virtual-world 57;
  server/render 150 (twice); Swift Upload 42; truthful doc tests 14 (four times, incl.
  after every North Star edit).

## Owner Tier-A spot-check ruling (2026-07-15, committed d74897203)

Gate FAIL 29/50 vs pre-registered >=47/50 (manager independently re-tallied the owner
results file, sha dcb0c00e, before transcription; per-source fail across all 6 sources;
15/29 true contacts mistimed |dt|>=0.2s, 8 near the +/-0.3s edge). Training on Tier-A
bootstrap labels REMAINS BLOCKED; the audio-x-track auto-labeler is REJECTED as a
training-label source at current thresholds. The 50 owner-reviewed rows are the first
owner-verified pickleball event labels — PROTECTED EVAL SEED, never training. Public-data
pretrain leg unaffected. Proven label channel: ~20 min per 50 rich rows via the clip-review
page. Full detail: runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.md.

## Gate-clause ledger (finalized at wave close)

- NS-01.5 "Missing BODY/ball/paddle/assets remains partial": scoped test proof
  (tests/server/test_ns015_bundle_policy.py parametrized body/ball/paddle/assets), manager-run
  EXIT 0. Physical-trace slice PENDING (owner, NS-01.2b).
- NS-01.5 "complete requires every advertised URL": runner + server checks with tests,
  manager-run EXIT 0.
- NS-01.5 "local and SSH paths agree": same evaluator, resolver-parameterized; scoped tests.
- NS-01.4 P0-D "stage-adopted transform API ... distorted synthetic and real iPhone tests":
  MET at scoped-pass level (coordwire aab8c3098 + earlier parity seams; distorted-synthetic
  + Wolverine real-clip byte parity; caveat: the frozen Wolverine calibration is
  zero-distortion, so distortion coverage is synthetic-only until NS-02 capture).
- NS-01.4/P0-H "wired contract ... monotonic encoded PTS ... explicit missing/drop reason":
  wired-contract clause MET at scoped-pass level (tbwire bd99c6d11 + fix c4dfb2d8b);
  physical 30s/5min captures PENDING (owner); native-intrinsics scaling, crop/orientation,
  rolling-shutter device slices PENDING (NS-01.6/01.7).
- NS-01.4 "corrected event/geometry error beats raw path on independent labels": PENDING —
  no independent label set exists (NS-02 owner-gated). Not fabricated.
