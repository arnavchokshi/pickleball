# LANE: record_ux_fix_20260709 — dead record button + hardcoded focus lock (physical-device P1 bugs)

## HARD RULES
- Stay on `main`. NO branches, NO commits — manager commits after ruling.
- Read `NORTH_STAR_ROADMAP.md` §1.2 step 2, §NS-03.LIVE and `AGENTS.md` first.
- Honest reporting. Run before reporting: `swift test --package-path ios` (if sandbox-blocked, `xcrun swiftc -parse` every touched file and SAY SO; manager reruns locally). Do NOT create .build-lane* scratch dirs in the repo — if you need a scratch path use $TMPDIR.
- Artifacts + structured report.json (schema docs/racketsport/lane_report.schema.json) under `runs/lanes/record_ux_fix_20260709/`.
- Preserve unrelated dirty worktree changes; list every touched path.

## FILE OWNERSHIP (exclusive)
- `ios/Capture/Sources/PickleballCapture/CameraCaptureController.swift`, `CapturePolicy.swift` (focus/exposure semantics only)
- `ios/App/CaptureViewModel.swift`, `ios/App/AppRootView.swift` (record-button recovery + blocked-reason banner only)
- Their test files under `ios/Capture/Tests/**`, `ios/AppTests/**`
- NOTHING else. Concurrent lanes own threed/scripts/server Python — never touch. Do not touch ios/Upload or ios/Core.

## BUGS (manager-diagnosed on a physical iPhone 14 Pro today — trust this)
1. DEAD BUTTON: `CaptureViewModel.toggleRecording()` sets `status = .blocked(msg)` on ANY start error
   (e.g. `.landscapeRequired` when portrait). `canRecordFromTab` (AppRootView) returns false for
   `.blocked` and the button is `.disabled` — so ONE failed tap (portrait) bricks recording for the
   entire app session, on every tab, with the blocked message not prominently shown on the Record tab.
   Owner hit exactly this.
2. FOCUS: `lockExposureFocusAndWhiteBalance(on:)` (CameraCaptureController ~L385-397) runs at session
   CONFIGURE time and calls `camera.setFocusModeLocked(lensPosition: 0.7)` + `exposureMode = .locked`.
   Hardcoded lens position = permanently blurry preview/recording at arbitrary distances and no
   tap-to-refocus. Owner reports "insanely out of focus, won't refocus".

## DESIGN (manager's architecture — follow it)
1. Focus/exposure: during PREVIEW use `.continuousAutoFocus` + `.continuousAutoExposure` (+ subject-area
   change monitoring default behavior). At `startRecording()` — after orientation gate passes — lock:
   focus via `setFocusModeLocked(lensPosition: camera.lensPosition)` (lock AT current converged position,
   never a constant), exposure via `.locked`, white balance as currently. On stop/preview-restart, return
   to continuous. The sidecar `locked` block must snapshot values AT record-start (it already reads the
   device at write time — verify semantics still true). If the device doesn't support locked modes, keep
   continuous and record that in policy achieved/unavailable reasons — never fabricate.
2. Blocked-state recovery: `.blocked` must be retryable. `handleRecordTap` treats `.blocked` like `.idle`:
   re-run `prepare()` then `toggleRecording()`. The tab-bar button stays ENABLED for `.blocked` (it is the
   retry affordance). Show the blocked reason as a visible banner/pill on the Record screen (reuse existing
   DinkVision status UI patterns; ink-on-cream style), with the landscape case worded "Rotate to landscape
   to record". A tap while portrait must NOT permanently block: after the error, returning to landscape and
   tapping again must start cleanly (add a ViewModel test for exactly this sequence).
3. Keep changes minimal — no redesign of the state machine beyond making `.blocked` recoverable.

## ACCEPTANCE
1. New ViewModel test: portrait tap → blocked(landscape msg) → landscape tap → recording. Passes.
2. New/updated controller tests: focus/exposure continuous in preview, locked at record start at current
   lens position (fake/policy-level where device APIs can't run on macOS — test the decision logic, and
   parse-verify the AVFoundation calls).
3. Existing 236-test suite: 0 new failures.
4. Record button never renders disabled due to `.blocked`; blocked reason visible on Record tab (assert
   via state/view-model, not screenshots).

## KILL CRITERIA
If recoverable-blocked requires restructuring beyond CaptureViewModel/AppRootView button gating, STOP and report.

## BEST-STACK DELTA
(c) None — device UX/correctness fix. State in report.

## REPORT
objective_result per acceptance; suites before/after; HONEST_ISSUES (esp. what only a physical device can confirm); touched files; dated handoff bullet text.
