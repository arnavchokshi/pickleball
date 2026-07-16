# tbwire_20260715 readable summary

Result: **PARTIAL**. The requested decode timebase contract is wired as a scoped pass, but the mandatory full suite is non-green and the physical/device gate is pending. `VERIFIED=0` remains binding.

## Wired scope

- Ingest writes byte-stable legacy `frame_times.json` beside schema-valid `timebase_contract.json` and `timebase_decode_evidence.json` when raw PTS exists. CFR fallback emits no raw-authority contract and is explicitly declared `constant_fps_assumed`.
- Raw ffprobe timestamps preserve integer ticks/timescale; decimal-only fallback preserves exact strings and records the conversion method. Derived legacy timing carries `CorrectionProvenance`.
- Non-monotonic or unavailable source-frame PTS becomes typed per-frame absence instead of silent truncation. `time_for_frame` refuses a missing supplied-table entry; explicit `fps` with no table returns recordable CFR provenance.
- Sidecar ARKit/video pairs fit `SensorClockMapping` when at least two pairs exist; otherwise a typed missing reason is retained. Events/audio ordering references the same timebase contract without destructively replacing raw audio times.
- The process normalization seam reads typed `corrected_pts` and records `time_basis=corrected_pts` plus the contract path.

## Evidence

- Final focused command: `209 passed`, `0 failed`, literal exit code `0`.
- Wolverine: exact raw PTS `300/300`, timescale `15360`, strictly monotonic; contract schema-valid and byte-stable. The legacy table remains `299` entries with unchanged frame-value SHA-256 `44e416b50db01bb6bbc38d583d8134ee4235421c3e13b1bb7f1d2d76ba8fcbb5`.
- Synthetic VFR: all 6 source frames accounted; non-monotonic frame typed dropped, missing-PTS frame typed missing, retained raw PTS monotonic, contract round-trip byte-stable.
- Golden sidecars: `full_sensors=insufficient_dual_clock_pairs` (the fixture has one pair), `missing_sensors=sidecar_declares_unavailable`, `camera_roll_import=sidecar_declares_unavailable`; a 3-pair synthetic case proves offset/drift/residual fitting.
- Mandatory full suite: `3670 passed`, `12 failed`, `24 skipped`, literal exit code `1`. Six TCP-bind and two AF_UNIX-bind failures reproduce at HEAD under this sandbox. Four current-tree BALL failures pass at HEAD and are concurrent cross-lane contamination, not tbwire edits. No owned timebase/audio/process test failed.

## Pending and honest limits

P0-H exit gate: "Wired contract; physical 30-second and 5-minute captures have no silent truncation, monotonic encoded PTS, and an aligned sample or explicit missing/drop reason per frame." The wired contract/fixture/real-clip slice is a scoped pass; physical 30-second/5-minute proof is owner-gated and PENDING.

The wider NS-01.4 native-intrinsics scaling and rolling-shutter/device slices were outside this lane fence and remain PENDING. No accuracy or promotion claim is made. No protected labels were used.

BEST-STACK DELTA: **(c) no stack delta**. This changed timing correctness/provenance only; no model, checkpoint, config, threshold, or policy selection changed.

## Paste-ready manager note

2026-07-15 tbwire_20260715: NS-01.4/P0-H typed decode timebase is wired (scoped pass) through ingest, the frame-time normalization seam, and events/audio provenance; canonical exact ffprobe ticks account for 300/300 Wolverine frames while legacy frame_times values remain unchanged, synthetic VFR gaps/non-monotonic PTS are explicit, and sidecar clock mapping has typed fit/missing outcomes. Final focused suite: 209 passed, exit 0. Mandatory full suite: 3670 passed/12 failed/24 skipped, exit 1; 8 failures are sandbox TCP/AF_UNIX bind restrictions reproduced at HEAD and 4 are concurrent BALL dirty-tree regressions that pass at HEAD, with no owned-test failures. Physical 30s/5min capture proof and broader native-intrinsics/rolling-shutter clauses remain owner-gated PENDING; VERIFIED=0, no stack delta, no commit.
