# record_ux_fix_20260709 handoff

- 2026-07-09: Made the blocked Record state retryable, kept the raised Record control enabled for retry, and added an ink-on-cream blocked-reason pill on the Record screen. The landscape message is now “Rotate to landscape to record.”
- 2026-07-09: Replaced configure-time focus/exposure locks with continuous autofocus/autoexposure during preview; recording now locks focus at the device's current lens position and locks exposure only after the landscape gate passes. Stop and preview restart restore continuous modes.
- 2026-07-09: Preserved the record-start exposure/ISO/focus/white-balance readback for the sidecar. Unsupported focus/exposure lock modes remain continuous when supported and surface existing `focus_not_locked` / `exposure_not_locked` policy violations.
- 2026-07-09: BEST-STACK DELTA (c) None — device UX/correctness only.

## Verification

- Baseline exact `swift test --package-path ios`: sandbox-blocked before edits by the unwritable `~/.cache/clang/ModuleCache`.
- After exact `swift test --package-path ios`: same sandbox blocker.
- Full sandbox-safe retry: 238 executed, 237 passed, 1 skipped, 0 failed. This is the existing 236-test suite plus two new capture-policy tests.
- All six touched Swift files passed `xcrun swiftc -parse`.
- The Swift package cross-built successfully for `arm64-apple-ios18.0-simulator`, type-checking the iOS-only AVFoundation controller calls.
- `CaptureViewModel.swift` plus its dependencies emitted an iOS Simulator module, and the complete `CaptureViewModelTests.swift` type-checked against it.
- Hosted AppTests execution was blocked because the sandbox cannot connect to `CoreSimulatorService`; the manager must run the test below locally.

## Manager rerun

```bash
swift test --package-path ios
xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17 Pro,OS=26.5' -only-testing:PickleballAppTests/CaptureViewModelTests/testPortraitRecordTapCanRetryInLandscapeAndStartRecording CODE_SIGNING_ALLOWED=NO test
```

Physical iPhone 14 Pro confirmation still required: preview converges focus/exposure, recording locks at that converged lens position, stop returns to continuous refocus, and portrait-blocked then landscape-retry records without relaunch.
