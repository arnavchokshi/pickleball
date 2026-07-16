# Failing-first evidence

Tests were authored before the implementation in both required test surfaces.

## SwiftPM RED

Command (no pipe):

```bash
HOME=/tmp/ios_recordvis_red_home CFFIXED_USER_HOME=/tmp/ios_recordvis_red_home XDG_CACHE_HOME=/tmp/ios_recordvis_red_home/.cache CLANG_MODULE_CACHE_PATH=/tmp/ios_recordvis_red_clang SWIFTPM_MODULECACHE_OVERRIDE=/tmp/ios_recordvis_red_clang swift test --package-path ios --scratch-path /tmp/ios_recordvis_swiftpm_red --disable-sandbox --filter RecordControlInteractionPolicyTests
```

Real exit: `1`. The four new tests failed to compile because `RecordControlInteractionPolicy`,
`RecordControlState`, and the reaction/accessibility types did not exist. This is the intentional
pre-implementation red state.

## SwiftPM GREEN

The same focused command with the final scratch/cache names executed 4 tests, 4 passed, 0 failed,
EXIT 0.

## Hosted AppTests

Four hosted integration tests were also authored before implementation in
`ios/AppTests/CaptureViewModelTests.swift`, covering persistent guidance, blocked/preparing tap
feedback, enabled accessibility in idle/requestingAccess/blocked/ready/recording, and repeated
blocked-entry announcements. Runtime RED/GREEN could not be collected in this sandbox:
`xcodebuild build-for-testing` exited 74 before compilation because CoreSimulatorService and the
Xcode package/module cache were sandbox-inaccessible. All AppTests sources later typechecked
together for arm64 iOS Simulator with EXIT 0. Runtime execution remains required via
`MANAGER_VERIFY2.md`.

