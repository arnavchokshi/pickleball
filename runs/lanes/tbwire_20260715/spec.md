# Lane tbwire_20260715 — NS-01.4/P0-H: wire the typed timebase contract through the real decode seam

You are a Codex implementation lane for the DinkVision pickleball repo at
/Users/arnavchokshi/Desktop/pickleball. Work only inside that repo. VERIFIED=0 is binding:
no promotion language anywhere — "wired" and "scoped pass" at most.

## HARD RULES
- NO branches, NO commits, NO pushes. Leave your changes in the working tree on `main`.
  The manager rules and commits.
- Read first: `NORTH_STAR_ROADMAP.md` (§2.1 rows P0-D/P0-H, §4 NS-01.4, §6 standing rules),
  `runs/HANDOFF_20260714.md` §5, `RUNBOOK.md` (Stage Order + Focused Verification),
  `runs/manager/inflight_lanes.md` (last note).
- Protected data: Outdoor/Indoor labels NEVER. Burlington/Wolverine internal-val use is OK
  (you may ffprobe/decode `eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4`
  for timebase tests; you may NOT use its labels as accuracy truth).
- Preserve unrelated dirty work: `configs/ssh/a100_known_hosts` is modified by someone else —
  do not touch, revert, or stage it. Do not touch `brand-exploration/`, `cvat_upload/`,
  `data/`, `runs/manager/gpu_fleet.md`, or other lanes' run dirs.
- Raw observations are immutable (standing rule 6). Schema/programming errors fail loudly
  (rule 10). Every stage declares timebase + provenance (rule 8).
- PYTEST EXIT-CODE TRAP (this repo committed past a piped-away pytest exit code TWICE — see
  commit 3b639768c): run pytest with output redirected to a file and capture `$?` of the
  pytest command itself, no pipes. Report the literal numeric exit codes in your report.
- Wide blast-radius suite is mandatory before you claim anything:
  `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q` (full, not a subset).
  Baseline on HEAD is being measured by the manager; report your full passed/failed counts
  and list every failure with proof it is pre-existing (reproduce on stash if needed).
- All lane artifacts (logs, evidence JSON) under `runs/lanes/tbwire_20260715/`.
- No new root .md files. No new CLI without a same-lane direct-CLI reference test.

## EXPLICIT FILE OWNERSHIP (you may edit ONLY these; everything else read-only)
- `threed/racketsport/io_decode.py`
- `threed/racketsport/timebase.py` (additive adapters/builders only; existing dataclass
  semantics and the byte-stable JSON contract MUST NOT change — 18 contract tests are frozen
  behavior)
- `threed/racketsport/audio_onsets.py` (only if needed to accept/record the typed contract;
  keep the existing raw+corrected non-destructive behavior byte-identical when no new input
  is supplied)
- `scripts/racketsport/process_video.py` — you are the ONLY lane allowed to edit this file
  in this window (manager-serialized). Keep your hunks minimal and confined to the ingest
  stage, the frames-stage `time_for_frame` seam, and events-stage plumbing of the contract.
- `docs/racketsport/timebase_schema.json` (additive only, if the artifact grows fields)
- Tests: `tests/racketsport/test_timebase_contract.py` (additive), `tests/racketsport/test_io_decode.py`
  (or create it), `tests/racketsport/test_process_video.py` (only ingest/frames/events-seam tests),
  new test files with `timebase` in the name.
- FORBIDDEN even though related: `threed/racketsport/coordinates.py`, `threed/racketsport/placement.py`,
  `threed/racketsport/ball_*.py`, `threed/racketsport/virtual_world.py`, `RUNBOOK.md`,
  `tests/racketsport/test_truthful_capabilities.py` (two concurrent lanes own those).

## OBJECTIVE
The typed timebase core (`threed/racketsport/timebase.py`, commit f3cfcb932) landed PURE
UNWIRED. Wire it through the real decode seam so the pipeline's frame-time evidence carries
raw-PTS authority with typed provenance, canonical-beside-legacy.

North Star acceptance gates you serve (quote in your report):
- P0-H exit gate: "Wired contract; physical 30-second and 5-minute captures have no silent
  truncation, monotonic encoded PTS, and an aligned sample or explicit missing/drop reason
  per frame." (The physical-capture slice is OWNER-GATED — you cannot close it; mark it
  PENDING honestly. Your job is the "wired contract" + fixture/synthetic/real-clip slices.)
- NS-01.4 gate (timebase slice): "PTS/VFR, native-intrinsics scaling, A/V mux, acoustic
  propagation, sensor clocks and rolling shutter" wired with the stop rule "Do not mix
  coordinate spaces, align latest samples, or correct audio destructively; preserve raw values."

Current defects to fix (verified by the manager, exact locations):
1. `io_decode.build_frame_time_table` (L199-253): discards raw integer PTS (no ticks/timescale),
   rebases to `pts - source_start_pts_s` and rounds to 9dp WITHOUT recording that as a typed
   derived correction; silently drops non-monotonic PTS runs inside
   `_frame_pts_from_ffprobe_frames` (L162-187) with no per-frame absence reason; the
   `constant_fps_assumed` fallback produces index/fps times with only a coarse trust band.
2. `io_decode.time_for_frame` (L320-326): silent index/fps fallback when a frame is missing
   from the table — a silent-CFR seam.
3. The capture sidecar's dual-clock samples (`arkit_timestamp_s` vs `video_pts_s`, see
   `tests/racketsport/fixtures/capture_sidecar/full_sensors.json`) are never turned into a
   `SensorClockMapping`; `missing_sensors.json` / `camera_roll_import.json` carry explicit
   unavailable-reasons that are never surfaced as typed alignment absence.
4. `audio_onsets.finalize_audio_onset_timing` already builds `AcousticPropagationModel`
   (identity when mic distance unknown) — the ordering provenance exists; do not regress it.

## REQUIRED DESIGN (canonical-beside-legacy, additive)
- Ingest produces a schema-valid typed timebase artifact (either `timebase_contract.json`
  beside `frame_times.json`, or raw-tick fields added inside `frame_times.json` — your call,
  but existing keys and their VALUES must stay byte-identical so current consumers and
  content-addressed reuse semantics are not silently perturbed; document the choice).
- Raw encoded PTS preserved as integer ticks + timescale (from ffprobe; if ffprobe cannot
  give exact ticks, preserve the exact decimal strings it reports and record the conversion
  method — never round-trip through float silently).
- The existing rebased/rounded `pts_s` values become a DERIVED field with typed
  `CorrectionProvenance` (method e.g. `rebase_to_first_frame_round9`) — values unchanged.
- Monotonicity validated; any frame ffprobe reported but the table drops/cannot time gets an
  explicit `FrameAvailability`/absence-reason entry — no silent truncation. Count consistency
  (stream frame count vs table) asserted or explained per frame.
- CFR fallback stays available for legacy compatibility but may NOT emit a typed contract
  claiming raw-PTS authority; it must be a typed, explicit `constant_fps_assumed` declaration.
- `SensorClockMapping` built from sidecar `arkit_frame_samples` when >=2 dual-clock pairs
  exist (fit offset+drift, record residual); explicit typed missing reason otherwise. All
  three golden fixtures must produce the correct typed outcome.
- Real stage consumers adopt it: (a) ingest writes it + stage notes/artifacts list it;
  (b) the frames stage timestamp seam (`process_video.py` ~L2641-2723 `time_for_frame` use)
  reads times through the typed table and records `time_basis` provenance in its existing
  normalization metadata (values unchanged for PTS-present clips); (c) `time_for_frame`'s
  silent fps fallback becomes loud-or-declared: consumers that hit the fallback must get a
  recordable signal (e.g. returned provenance or a required opt-in flag) — choose the least
  invasive honest design and document it. (d) events stage passes the contract (or its audio
  fields) into the audio-onset chain so ordering provenance references the same artifact.
- DO NOT rewrite the dozens of index/fps call sites across body/tracking/stats modules —
  that is NS-01.6/01.7 territory. Fence: decode seam + ingest + frames seam + events plumbing.

## MANDATORY TESTS (all must pass with real exit codes)
- Golden-fixture tests: all three capture sidecars produce correct typed clock-mapping /
  absence outcomes.
- Synthetic VFR test: a constructed PTS stream with jitter + one non-monotonic run + one gap
  produces (i) monotonic raw sequence or typed per-frame absence reasons, (ii) no silent
  truncation (every source frame accounted for), (iii) byte-stable round-trip through the
  contract.
- Real-clip test: ffprobe the Wolverine `source.mp4`; assert provenance `ffprobe_pts`,
  monotonic raw PTS, count consistency, and that derived `pts_s` values are byte-identical
  to the pre-change table (parity proof — regenerate with old code path or pin the digest).
- Existing 18 timebase contract tests: unchanged and green.
- Full wide suite: report exact counts + exit code.

## KILL / STOP RULES
- If ffprobe cannot yield raw tick-exact PTS on this platform at all, STOP that sub-slice and
  report honestly (typed decimal-string preservation is the fallback design, not silent floats).
- If you cannot keep legacy `frame_times.json` values byte-identical, STOP and report why
  rather than shipping a silent value change.
- No accuracy claims. The "aligned sample per frame on physical 30s/5min captures" clause is
  PENDING (owner-gated) — say so in the report.

## MANDATORY STRUCTURED REPORT (report.json via the output schema; also readable summary)
- `objective_result`: PASS/FAIL against each numbered defect above + which gate clauses are
  now "wired (scoped pass)" vs PENDING and why.
- `full_suite`: exact passed/failed counts + the literal pytest exit codes; every failure
  proven pre-existing or fixed.
- HONEST ISSUES: anything weaker than it looks.
- Artifacts: paths under runs/lanes/tbwire_20260715/.
- BEST-STACK DELTA: expected "(c) no stack delta" — this is correctness wiring, no model or
  policy selection; state it explicitly and justify.
- A dated one-paragraph note the manager can paste into runs/manager/inflight_lanes.md.
