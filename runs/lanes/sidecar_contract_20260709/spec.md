# LANE: sidecar_contract_20260709 — NS-01.1 Swift↔Python capture-sidecar contract (P0-A exit)

## HARD RULES
- Stay on `main`. NO branches, NO commits, NO pushes — the manager commits after ruling.
- Read `NORTH_STAR_ROADMAP.md` §2.1 (P0-A), §4 NS-01.1, and `AGENTS.md` first.
- Protected eval clips are EVAL-ONLY. This lane needs NO eval labels; do not touch `eval_clips/` labels.
- Honest reporting. A failing acceptance item is a FAIL — report it, never soften it.
- Run the WIDE blast-radius suites before reporting: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport tests/ios tests/server tests/render_service -q` AND `swift test --package-path ios --scratch-path .build-laneA`. If `swift test` fails for sandbox-only reasons (cache/xcodebuild), fall back to `xcrun swiftc -parse` on every touched Swift file and SAY SO in the report — the manager re-runs swift test locally.
- Artifacts, notes, and your report go under `runs/lanes/sidecar_contract_20260709/`.
- Do not create/edit any root `.md` except the RUNBOOK note listed below. No new CLIs.
- Preserve unrelated dirty worktree changes — the tree has other in-flight edits; `git diff` only YOUR owned files at the end and list every touched path in the report.

## FILE OWNERSHIP (exclusive — do not edit anything else)
- `ios/Core/Sources/PickleballCore/CaptureSidecar.swift`, `CaptureSidecarPayloads.swift`
- `ios/Core/Tests/PickleballCoreTests/**`
- `ios/Capture/Sources/PickleballCapture/CaptureSidecarWriter.swift`
- `ios/Capture/Tests/PickleballCaptureTests/CaptureSidecarWriterTests*.swift` (create if absent; do NOT touch other Capture test files)
- `threed/racketsport/schemas/__init__.py` — ONLY `CaptureSidecar` and new payload models it references
- `tests/racketsport/test_sidecar_intrinsics.py` and new `tests/racketsport/test_capture_sidecar_contract.py`
- New golden fixtures under `tests/racketsport/fixtures/capture_sidecar/` (create dir)
- `RUNBOOK.md` ONLY the warning paragraph at lines ~38-41 ("Do not feed a production Swift-encoded sidecar…") — update it to describe the fixed contract once acceptance passes.
- A concurrent lane owns `ios/App/**`, `ios/Upload/**`, `ios/AppUITests/**` — NEVER edit those.

## CONTEXT (verified today by the manager — trust this)
- Swift `CaptureSidecar.encode(to:)` (ios/Core/.../CaptureSidecar.swift:220-249) always writes `provenance`, and real device recordings also carry `setup_pass`, `arkit_frame_samples`, `unavailable_sensor_reasons`, `policy_enforcement` (writer: ios/Capture/.../CameraCaptureController.swift:443-482 constructs them non-nil; the writer builder is `CaptureSidecarWriter.makeSidecar`).
- Python `CaptureSidecar(StrictArtifact)` (threed/racketsport/schemas/__init__.py:112-156) is `extra="forbid"`, `schema_version: Literal[1]`, and has NONE of those five keys → a real device sidecar raises pydantic ValidationError at `validate_artifact_file("capture_sidecar", …)` (consumed at threed/racketsport/orchestrator.py:~2877 `_tracking_fps`, fed verbatim from `--capture-sidecar`).
- Python requires (no default): `device_tier, device_model, fps, format, resolution, orientation, locked, intrinsics, gravity, capture_quality`. Python-only optionals Swift never writes: `hdr_enabled, video_stabilization_enabled, exposure_locked, focus_locked, tripod_height_m, full_court_visible, court_lock_passed, ball_high_contrast, audio_recorded`.
- Audio input is mandatory in the capture session (CameraCaptureController throws `cannotAddAudioInput`), so `audio_recorded=true` is knowable at write time.

## OBJECTIVE (NS-01.1, verbatim gate)
"One canonical versioned schema across CaptureSidecar.swift, camera-roll import, Python strict schemas, server ingest. Swift-encoded golden fixtures cover present/missing sensors and validate in Python." Do NOT loosen Python to arbitrary extras.

## DESIGN (manager's architecture — follow it)
1. Canonical contract = schema_version 1 = the UNION, explicitly typed on both sides. Add to the Python model, each as a typed pydantic model/enum with `extra="forbid"` (mirror the Swift types field-for-field; snake_case keys exactly as Swift encodes them):
   - `provenance`: Literal of the Swift `CaptureProvenance` raw values (enumerate them from the Swift enum; includes "camera_roll_import"). Optional with a sensible default is NOT allowed — Swift always writes it; make it required? NO: make it required only if every historical Python-built sidecar fixture in the repo has it; otherwise `provenance: <Literal…> | None = None` and document why. Decide from evidence, state the decision in the report.
   - `setup_pass`, `arkit_frame_samples` (list, typed sample model), `unavailable_sensor_reasons`, `policy_enforcement` — all Optional (None/empty default) since camera-roll imports may omit them.
2. Swift writer (`CaptureSidecarWriter.makeSidecar` / its call path): populate `audio_recorded` (true on live recording), and `hdr_enabled`, `video_stabilization_enabled`, `exposure_locked`, `focus_locked` where the session config knows them. NEVER fabricate a value: if unknowable, leave nil and add an entry to `unavailable_sensor_reasons`. Do not change file layout/paths of the package.
3. If Swift can legitimately produce nil for a Python-required field (`locked`, `intrinsics`, `gravity`, `capture_quality`) in any real path (esp. camera-roll import — check `CameraRollVideoImporter`'s sidecar), make that Python field `| None = None` explicitly (typed, justified in report) rather than inventing values. Camera-roll-import sidecars MUST validate too — they are a first-class product path.
4. Golden fixtures: add a Swift test (macOS-runnable, PickleballCore or PickleballCapture tests) that ENCODES canonical sidecar instances via the real encoder and writes/compares them against checked-in fixtures at `tests/racketsport/fixtures/capture_sidecar/`: at minimum `full_sensors.json` (live recording, all sensors), `missing_sensors.json` (no ARKit/LiDAR, reasons populated), `camera_roll_import.json`. The Swift test asserts byte-stable encoding against the fixture (sorted keys or stable ordering — if the encoder isn't deterministic, compare decoded-normalized JSON, and say so).
5. Python contract test (`tests/racketsport/test_capture_sidecar_contract.py`): every fixture in that dir MUST pass `validate_artifact_file("capture_sidecar", path)`. Also keep/extend the existing intrinsics test. Also add a negative test: an unknown extra key still FAILS (proves extra="forbid" survived).
6. Also check `runs/` for any real device-pulled `capture_sidecar.json` (e.g. under runs/ios_device_build_*); if one exists, copy it into your lane dir as evidence and test whether it validates; report the result either way (it may predate the current writer — do not force it green, just report).
7. Search server ingest (`server/` routes that accept the sidecar, e.g. clips/jobs paths) for any duplicate sidecar schema assumptions; report (do NOT edit server/) if it re-validates with a different shape.

## ACCEPTANCE (all must pass)
1. All golden fixtures (incl. camera_roll_import) validate via `validate_artifact_file("capture_sidecar", …)` unchanged.
2. Swift round-trip: encoder output == fixtures; decoder reads every fixture back.
3. `extra="forbid"` retained; negative-key test passes.
4. Wide pytest suites green (0 new failures — diff any failures against a pre-change baseline run you record first).
5. `swift test` green locally OR documented sandbox fallback (swiftc -parse all touched files) with zero parse errors.

## KILL CRITERIA
If mirroring `arkit_frame_samples` typing becomes unbounded churn (>~200 lines of pydantic modeling) or you'd need `extra="allow"` anywhere, STOP, write the report with the blocker, and end.

## BEST-STACK DELTA
(c) No stack delta — product-route contract work, no model/weights/policy change. State this in the report.

## REPORT (structured, schema-enforced via --output-schema)
objective_result PASS/FAIL per acceptance item; full_suite counts (before AND after); HONEST_ISSUES; every touched file; fixture paths; the provenance-required decision + evidence; dated handoff bullet text (manager will place it).
