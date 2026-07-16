# LANE ios_recordpath_20260715 — dead record button on real iPhone: root-cause, loud-state contract, device-ready fix (Track D)

Ground truth READ FIRST: NORTH_STAR_ROADMAP.md §1.2 step 2 ("Recording never stalls because an
advisory model is slow" — a silently dead record button violates the product truth contract), §2.1
P0 rows, §4 NS-03.LIVE boundaries, §6 standing rules; ios/README.md (Brand V4 record control +
"camfix ARKit setup-pass ordering stays intact" + Local Verification commands); prior lane reports
runs/lanes/record_ux_fix_20260709/report.json (blockedReason pill + portrait→landscape recovery —
device claims were left UNPROVEN), runs/lanes/record_button_landscape_20260709/report.json (hit
geometry — hittability device-unproven), runs/lanes/ios_product_ui2_20260712/report.json +
simverify/report.json (simulator xcodebuild verification pattern; DinkVision-Test sim).

## SYMPTOM (owner report, 2026-07-15 19:42 PDT)
Tapping the yellow record button does NOTHING on the owner's real iPhone 14 Pro (device id
B03696B6-6481-5FCD-A79C-105DA3F08F98, bundle com.arnavchokshi.pickleball). Fresh Debug build of
current main installed+launched fine today; camera/mic/motion/photo usage keys ARE present in the
built Info.plist; blockedReason banner IS wired (ios/App/AppRootView.swift:817). The phone is GONE
for the next several hours — you have NO device access. All device claims stay unproven; your job
is code-level root cause, durable loud-state fixes with tests, and a ready-to-install fixed build.

## MANDATORY SKILL (owner requirement — non-negotiable)
Load and follow the Codex-installed iOS skill **`ios-debugger-agent`** (plugin
`build-ios-apps@openai-curated`, installed at
/Users/arnavchokshi/.codex/plugins/cache/openai-curated/build-ios-apps/2f1a8948/skills/ios-debugger-agent/SKILL.md).
It drives XcodeBuildMCP (mcp__XcodeBuildMCP__* tools: list_sims, build/run, UI drive, logs,
screenshots) for the simulator work below. If the MCP tools are not exposed in your session, you
must still READ that SKILL.md and follow its workflow with direct xcodebuild/xcrun simctl CLI.
Your report MUST state that the skill was loaded and how it was used. You may additionally use the
sibling `ios-simulator-browser` skill for interactive UI proof if helpful.

## HARD RULES
- NO branches, NO commits, NO `git add` (manager commits after ruling). Preserve unrelated dirty
  files (configs/ssh/a100_known_hosts, runs/manager/inflight_lanes.md) and ALL untracked owner dirs
  (brand-exploration/, cvat_upload/, data/) — never touch them.
- FILE OWNERSHIP (edits allowed ONLY here): ios/App/**, ios/Capture/**, ios/AppTests/**,
  runs/lanes/ios_recordpath_20260715/**. FORBIDDEN: ios/Upload/**, ios/Replay/** (routing
  semantics), ios/Core/**, ios/Pickleball.xcodeproj/** (project structure; scheme/plist edits
  ONLY if strictly required — call them out loudly in the report), threed/**, scripts/**,
  server/**, web/**, configs/**, docs/**, other lanes' run dirs (READ-ONLY evidence). If the true
  root cause requires an out-of-fence edit: do NOT make it; put inline diff hunks in the report.
- Honest reporting. VERIFIED=0 is binding. No promotion language. Simulator green is NOT device
  proof — say so explicitly in the report. The prior five-tab "green" pass (f824a81e7) ran the
  WALKER FAKE controller; do not repeat that mistake as evidence.
- The 4 protected clips are EVAL-ONLY (should be irrelevant to this lane — do not touch data/).
- Artifacts (logs, screenshots, xcresult summaries, built .app, INSTALL.md) go under
  runs/lanes/ios_recordpath_20260715/.
- Wide blast radius for this iOS-fenced lane = the FULL SwiftPM package suite + the FULL hosted
  AppTests run on simulator (commands below), not a hand-picked subset. The Python wide suite is
  not required (fence contains zero Python), state that in the report.

## PRE-TRACED DIAGNOSIS (manager code trace, 2026-07-15 — verify or refute EACH with evidence)
Tap chain: AppRootView.swift:390-393 Button → Task { handleRecordTap() }, gated
`.disabled(!canRecordFromTab)` at :416 → CaptureViewModel.isRecordButtonEnabled (:84-89, false iff
status == .requestingAccess) → DinkVisionTabBar.handleRecordTap (:518-522) →
CaptureViewModel.handleRecordTap (:208-216) → prepare() (:201-206: status=.requestingAccess →
requestPermissions → configure) → configure (:218-270 → controller.configure →
refreshSetupPassIfNeeded(force:true) :394-429 → startPreview → status=.ready) → toggleRecording
(:431-463, guard canStartRecording :593-600). Real controller: CameraCaptureController.swift
configure :86-169, startPreview :171-185, startRecording :253-289, serialized through
CameraSessionQueue.swift:10-22 via QueuedCameraCaptureController (CameraCaptureControlling.swift:98-160).
blockedReason banner: AppRootView.swift:817-820/:940-950; non-nil ONLY in .blocked
(CaptureViewModel.swift:91-96).

Ranked candidates (C1 is the structural mechanism; C2-C4 are concrete causes behind it):
- **C1 — button disabled while stuck in .requestingAccess, no banner (PRIMARY MECHANISM).** Cold
  launch defaults to Record; DinkVisionRecordScreen.task auto-runs prepare() (AppRootView.swift:857-866).
  If configure() never reaches .ready, status pins at .requestingAccess forever: button disabled,
  .blocked never set, blockedReason nil, banner never shows, every tap silently swallowed by
  SwiftUI. THERE IS NO TIMEOUT anywhere in the chain.
- **C2 — real-hardware configure() chain stalls.** Real ARKit setup pass grabs the back camera
  (CameraCaptureController.swift:201-218, ARKitSetupPassRunner.swift:22-74,
  ARKitSessionProvider.swift:12-20) then startPreview does session.startRunning()
  (CameraCaptureController.swift:183) on the shared serial CameraSessionQueue; ARKit→AVCapture
  camera-ownership handoff (CameraResourceOwnership.swift) can contend; a non-resuming queue
  continuation pins .requestingAccess (→C1). All of this is an instant no-op on the walker fake.
- **C3 — silent camera-ownership failure in startPreview.**
  `guard activeAVCaptureToken == nil, let token = try? cameraOwnership.beginAVCapture() else { return }`
  (CameraCaptureController.swift:175-181) + `_ = try? await` swallowing in
  QueuedCameraCaptureController (CameraCaptureControlling.swift:118-128,134-136,154-160). Ownership
  still held by the just-run ARKit pass → preview never starts, NO error surfaced, no banner.
- **C4 — double-prepare race amplifier.** prepare() can fire twice at cold launch (screen .task +
  tap; AppRootView.swift:861-862 + CaptureViewModel.swift:210-211); the second ARKit pass hits
  cameraAlreadyOwned (ARKitSetupPassRunner.swift:27-40) and startPreview early-returns (C3).
- **C5 — TCC/permission seam (real-device-only).** On a real device, previously-denied camera/mic
  persists across REINSTALLS (TCC state): AVCaptureDevice.requestAccess returns false immediately
  with NO prompt. Trace what requestPermissions=false does to the state machine
  (CaptureViewModel.swift:201-206 — does a denial produce .blocked with an actionable
  "enable in Settings" banner, or pin .requestingAccess/misconfigure?). Also verify first-run
  prompt sequencing (video then audio) cannot deadlock the chain.
- **C6 (bonus correctness bug — fix it):** updateOrientation only runs on viewport .onChange
  (AppRootView.swift:867-871), never on first appearance; captureDeviceOrientation defaults to
  .landscapeRight (CaptureViewModel.swift:23) → a portrait phone at cold launch still configures
  as landscape (stale orientation state; landscape gate produces a VISIBLE banner only after a
  rotation event actually fires).
Gesture layer was audited clean (contentShape/zIndex/allowsHitTesting all correct) — the only tap
gate is `.disabled` (C1). Do not chase layout ghosts unless you find new evidence.

## MISSION
1. **Reproduce the silent-dead mechanism in tests (baseline first).** Measure the current full
   SwiftPM + AppTests baselines on unmodified main BEFORE editing. Then write failing-first hosted
   tests driving CaptureViewModel with stub controllers for: (a) configure() that never completes
   → assert the button cannot stay disabled forever (bounded-time loud state); (b) startPreview
   silent ownership failure → assert surfaced .blocked; (c) double-prepare race → assert single
   coalesced preparation; (d) requestPermissions=false (TCC denied) → assert .blocked with
   actionable reason; (e) first-appearance orientation staleness. Also REPRODUCE THE REAL PATH ON
   SIMULATOR: launch the app with captureState == .live (NOT the walker) on the DinkVision-Test
   sim (iOS 26.5, id C719AFE6-FE07-48BF-A042-534A6FEF8748) via the ios-debugger-agent workflow;
   tap Record; capture screenshots + logs. On simulator the real chain cannot get a camera — the
   pre-fix build shows the dead-button symptom (or a stuck state); the post-fix build must show a
   LOUD blocked banner. Save before/after artifacts.
2. **Durable fixes — the LOUD-STATE CONTRACT (§1.2).** After your fix, EVERY exit path of
   handleRecordTap/prepare/configure/toggleRecording must, within a bounded time (pick and justify
   a watchdog, ~8s order), land in .ready/.recording OR .blocked(actionable reason) with the
   banner visible and a working Retry. Specifically: kill every silent `try?`/guard-return in the
   chain (route typed errors to .blocked); add a configure/prepare watchdog timeout; make prepare
   idempotent (coalesce concurrent calls); map TCC-denied to .blocked with "enable camera in
   Settings" wording + retry; fix first-appearance orientation refresh; the record button must
   never be indefinitely disabled (disabled is acceptable only while a bounded-time preparation is
   in flight). PRESERVE the camfix ARKit setup-pass ordering and the CameraResourceOwnership
   design — surface failures loudly, do not delete the pass. Add os.Logger diagnostics (subsystem
   "com.arnavchokshi.pickleball", category "RecordPath") at every state transition + guard exit in
   the chain so the next device session diagnoses instantly via `log stream`/devicectl.
3. **Device-ready deliverable.** Build a SIGNED Debug iphoneos .app of the fixed tree
   (xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball -sdk iphoneos
   -destination 'generic/platform=iOS' -allowProvisioningUpdates; 2 valid Apple Development
   identities exist on this Mac; the owner installed a Debug build today so signing works). Stage
   the built .app under runs/lanes/ios_recordpath_20260715/device_build/ and write INSTALL.md with
   the EXACT command set: `xcrun devicectl device install app --device
   B03696B6-6481-5FCD-A79C-105DA3F08F98 <staged .app path>`, `xcrun devicectl device process
   launch --device B03696B6-6481-5FCD-A79C-105DA3F08F98 com.arnavchokshi.pickleball`, the exact
   `log stream`/`devicectl` command to capture RecordPath logs, and a 2-minute on-device checklist
   (expected banner/state per scenario incl. Settings > Privacy > Camera check). Device is
   currently unavailable — do NOT attempt a device install; stage everything.
4. **Guard audit table (report + lane dir).** Enumerate EVERY early-return/guard in the tap chain
   post-fix with its user-visible consequence. Acceptance: zero rows read "silent".

## ACCEPTANCE (all with REAL exit codes — no piped-command exit-code masking; `set -o pipefail` or
capture $? immediately; report each verbatim)
1. Full SwiftPM suite: `swift test --package-path ios` (caches to /tmp, --disable-sandbox pattern
   from prior lanes OK) — 0 failures, count ≥ measured baseline, EXIT 0 reported.
2. Hosted AppTests on the simulator via xcodebuild build-for-testing + test-without-building
   (ios_product_ui2 simverify pattern): the ONLY allowed failure is the pre-existing device-only
   ANELatencyBenchmarkTests.testMeasureOnDeviceLatencyForConvertedModels (file untouched). Report
   executed/passed/failed counts + real exit code, and prove any failure pre-exists on baseline.
3. New tests: ≥1 per candidate C1-C6 (CONFIRMED candidates get a regression test; REFUTED get the
   refuting evidence documented; device-only-untestable seams get a stub-level contract test +
   explicit UNPROVEN-on-device note). All new tests green in both runs above.
4. Simulator real-path (live controller) before/after artifacts: pre-fix dead/stuck evidence,
   post-fix loud banner + Retry evidence (screenshots + logs saved under the lane dir).
5. Guard audit table complete; zero silent rows.
6. Staged signed .app + INSTALL.md exact commands (verify the .app bundle exists, is iphoneos
   arm64, signed — `codesign -dv` output saved).
7. Per-candidate verdict table C1-C6: CONFIRMED / REFUTED / UNTESTABLE-WITHOUT-DEVICE + evidence.
KILL criteria (any → FAIL yourself honestly): weakening landscape enforcement or capture
correctness to make the button "work"; deleting/reordering the ARKit setup pass; touching
upload/replay routing; fabricating device claims; leaving any silent no-op path in the tap chain.

## REPORT (schema-validated report.json via --output-schema; Fable rules on this, never transcript)
objective_result PASS/FAIL vs the 7 acceptance items; full_suite counts + REAL exit codes for both
suites; per-candidate C1-C6 verdict table; guard audit table; changes with file:line; HONEST
ISSUES (must include: real-device root cause remains UNPROVEN until the phone returns — the fix
guarantees loudness and removes the identified stall/swallow mechanisms); artifacts list;
skill-usage statement (ios-debugger-agent loaded + how); session_id;
**BEST-STACK DELTA: expected (c) none — this is app-shell/capture-path correctness, no model/stack
change; state it explicitly.**

## SANDBOX REALITY (workspace-write)
You run under codex `--sandbox workspace-write` (workdir + /tmp writable). Proven in prior lanes:
full SwiftPM suite WORKS with caches redirected to /tmp + `swift test --disable-sandbox`
(ios_product_ui2 pattern). xcodebuild/CoreSimulatorService MAY be blocked
(`sandbox-exec: sandbox_apply: Operation not permitted` seen 2026-07-12). Attempt the simulator
and device-build steps; if a step is sandbox-blocked, do NOT fake it and do NOT fail the whole
lane: mark that acceptance row SANDBOX-BLOCKED with the verbatim error, and write
runs/lanes/ios_recordpath_20260715/MANAGER_VERIFY.md containing the EXACT ready-to-run command
sequence (per the ios-debugger-agent skill workflow) for the manager to execute at ruling:
simulator build+install+launch+UI-drive proof, hosted AppTests run, signed iphoneos Debug build +
codesign check, and the devicectl install/launch/log commands. Everything that CAN run in-sandbox
(all unit/hosted test code, state-machine fixes, guard audit, os.Logger diagnostics, INSTALL.md)
MUST still be delivered complete.

## ANTI-PASSIVE-WAIT
Ending your turn to wait = lane death; you will NOT be re-woken. Poll any >10min build/test with a
bounded foreground until-loop. End only with the final report.json written or a hard blocker
stated in it. Budget: this should complete in one session; 1-2 manager resumes available via
session id if needed.
