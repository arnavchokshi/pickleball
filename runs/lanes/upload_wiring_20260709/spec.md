# LANE: upload_wiring_20260709 — NS-01.2a record→upload production wiring (P0-B call path)

## HARD RULES
- Stay on `main`. NO branches, NO commits, NO pushes — the manager commits after ruling.
- Read `NORTH_STAR_ROADMAP.md` §1.2 (steps 2-3), §2.1 (P0-B), §4 NS-01.2a, and `AGENTS.md` first.
- Honest reporting; no fixture-only shortcut may masquerade as the production call path.
- Run before reporting: `swift test --package-path ios --scratch-path .build-laneB` and `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ios tests/server tests/render_service -q`. If swift test fails for sandbox-only reasons, fall back to `xcrun swiftc -parse` on every touched Swift file and SAY SO — the manager re-runs locally (xcodebuild sim build + UITest compile).
- Artifacts + report under `runs/lanes/upload_wiring_20260709/`.
- Preserve unrelated dirty worktree changes; list every touched path in the report.

## FILE OWNERSHIP (exclusive — do not edit anything else)
- `ios/App/**` (all Swift + Info.plist), `ios/Upload/**`, `ios/AppUITests/**`, `ios/AppTests/**`
- A concurrent lane owns `ios/Core/**`, `ios/Capture/Sources/**`, `ios/Capture/Tests/**`, `threed/racketsport/schemas/__init__.py`, `tests/racketsport/**` — NEVER edit those. Consume `PickleballCapture`/`PickleballCore` public APIs as-is; if an API gap blocks you, REPORT it, don't patch their files.
- Do NOT edit `server/**`. The server contract is fixed: `server/routes/auth.py` (register/login/refresh, native:true → refresh token in body), `server/routes/clips.py` (`POST /api/clips` presigned multipart+sidecar, `POST /api/clips/{id}/complete`, `GET /api/clips`). Read them to match shapes exactly.

## CONTEXT (verified today by the manager — trust this)
- Recording works and is wired: Record tab is cold-launch default; `CaptureViewModel.toggleRecording()` → `CameraCaptureController` (AVCaptureMovieFileOutput, landscape-enforced, audio mandatory); on stop, package lands at `Documents/captures/<sessionID>/{clip.mov,capture_sidecar.json}`; `CaptureLibrary.listPackages` feeds the Replays tab.
- Upload clients EXIST and are package-tested but have ZERO production call sites: `AuthApiClient` (used only by SignInView/SignInViewModel), `PresignedUploadClient` (createClip → uploadParts → completeClip → uploadSidecar; comment cites server/routes/clips.py::presign_multipart_put), `RenderGatewayClient` (legacy). Base URLs are hardcoded statics = https://pickleball-gpu-gateway.onrender.com.
- `dinkVisionAuthGateEnabled = false` (ios/App/AppRootView.swift:22) ships the auth gate DARK; RUNBOOK notes flipping it would break RecordStopUITests (no bypass arg).
- `RecordStopUITests.swift` is STALE: taps a button "Open Camera" that no longer exists (current shell = 5-tab, record default, button id `DinkVisionRecordButton`, labels "Start recording"/"Stop recording").
- `CameraRollVideoImporter` (PickleballCapture) is complete + tested but unreachable from the app UI.
- Live server today: Render gateway healthy; `PICKLEBALL_ACCOUNTS_ENABLED=0` right now — the manager flips it to 1 during verification. Your tests must run against an injected local stub (URLProtocol), never the live service.

## OBJECTIVE (NS-01.2a, verbatim gate)
"The app constructs the upload request, sends video+sidecar, polls honest job status, and routes to replay. Production call path has focused Swift/server tests and no fixture-only shortcut." (Job polling may honestly report the server's answer; job SUBMISSION stays out of scope while the queue is dark — an uploaded clip visible in `GET /api/clips` with an honest "uploaded, processing not yet started" state satisfies this lane.)

## DESIGN (manager's architecture — follow it)
1. **Auth never gates recording.** Signed-out: record + Replays library fully work. Sign-in lives in the Profile tab and is prompted contextually at first upload attempt. Restructure the `dinkVisionAuthGateEnabled` gate accordingly and TURN IT ON (`true`): with the new structure it must not block the record path, so UITests keep working without bypass hacks. Keep a launch-arg escape hatch for UI tests only if genuinely needed.
2. **UploadQueue** (new, `ios/App` or `ios/Upload` — your choice, report it): serial, one capture at a time, per-package persisted state file `upload_state.json` INSIDE the capture package dir (fields: state ∈ queued|uploading|uploaded|failed, clip_id, uploaded_parts, byte counts, updated_at, last_error). Restart-safe: on relaunch, queued/uploading resumes or re-queues idempotently (re-presign is fine). Foreground URLSession v1 is acceptable; note background-URLSession as a follow-up in the report.
3. **UX:** each Replays row shows an honest state chip (Local / Queued / Uploading %/ Uploaded / Failed+Retry). After a recording finishes, offer Upload (and an "auto-upload after recording" toggle, default OFF, in Profile). Uploaded rows keep their server clip_id visible (identity trace, P0-B).
4. **Sidecar:** upload the exact on-disk `capture_sidecar.json` bytes with the video (uploadSidecar). Never regenerate/mutate it.
5. **Camera-roll import:** PHPickerViewController (videos) entry point on the Replays tab → `CameraRollVideoImporter.importVideo` → library row → uploadable like any capture.
6. **Base URL override:** runtime override (env/launch arg/Info.plist key, e.g. `DINKVISION_API_BASE_URL`) threaded to all three clients; default unchanged.
7. **Status polling:** for uploaded clips, poll/query honestly (`GET /api/clips` or job status when it exists) and render the server's answer verbatim states; NEVER map "uploaded" to "processing"/"ready". Missing processing = show "Uploaded — processing not started".
8. **Tests:** focused Swift tests for UploadQueue state machine (fake clients), SignIn-at-upload prompt flow, importer wiring, base-URL override; URLProtocol-stub integration test covering createClip→parts→complete→sidecar with a simulated 5xx mid-part retry. FIX `RecordStopUITests` for the current shell (record tab default, `DinkVisionRecordButton`), keeping its device-oriented structure. Do not weaken WorldViewerUITests.

## ACCEPTANCE (all must pass)
1. Production call path exists: recording-finish/library → UploadQueue → PresignedUploadClient (createClip, uploadParts, completeClip, uploadSidecar) with real on-disk bytes — verified by the stub integration test, no fixture shortcut.
2. Signed-out record path unaffected with the gate ON (state-machine test proves record tab reachable without auth).
3. Upload state machine: queued→uploading→uploaded happy path; failure → failed+retry preserving clip_id; relaunch resume test.
4. Camera-roll import reachable in UI and produces an uploadable package.
5. RecordStopUITests compiles and targets the real current shell (manager runs it on sim/device).
6. Swift package tests green (or documented sandbox fallback with 0 parse errors); pytest suites listed above green (0 new failures vs your recorded baseline).

## KILL CRITERIA
If the server clip contract cannot be satisfied without editing `server/**`, STOP and report the exact gap (endpoint, shape) — the manager rules on the server change.

## BEST-STACK DELTA
(c) No stack delta — product-route wiring. State this in the report.

## REPORT (structured, schema-enforced via --output-schema)
objective_result PASS/FAIL per acceptance item; full_suite counts (before/after); HONEST_ISSUES (incl. what only a real device can prove); every touched file; where UploadQueue lives; follow-ups (background URLSession, job submission when queue un-darks); dated handoff bullet text.
