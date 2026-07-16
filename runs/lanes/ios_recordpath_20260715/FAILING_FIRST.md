# Failing-first record

The unmodified baseline was measured first: full SwiftPM executed 245 tests with 244 passed, 0 failed, 1 skipped, real exit 0. Hosted AppTests build-for-testing was attempted before edits and was sandbox-blocked at Xcode package resolution with real exit 74.

Before implementation, seven hosted regressions were authored in `ios/AppTests/CaptureViewModelTests.swift`: one each for C1-C6 plus a recording-start watchdog. Against the pre-fix interfaces, the C1/C3/C6 tests intentionally referenced missing timeout, typed preview-error, and initial-orientation contracts; C4/C5 asserted behavior the old implementation did not satisfy. A red hosted execution could not be collected because the sandbox failed before compilation/launch. This is reported as SANDBOX-BLOCKED, not as an executed red test.

After implementation, all 54 hosted AppTests source functions typecheck together for arm64 iOS Simulator with real exit 0. Runtime execution still requires the manager command in `MANAGER_VERIFY.md`.
