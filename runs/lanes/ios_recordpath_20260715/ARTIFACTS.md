# Lane artifacts

- `baseline/swift_test.log` and `.exit`: unmodified full SwiftPM baseline, 245 executed, real exit 0.
- `baseline/build_for_testing*.log` and `.exit`: unmodified hosted AppTests attempts, sandbox-blocked, real exit 74.
- `final/swift_test.log` and `.exit`: final full SwiftPM suite, 245 executed, real exit 0.
- `final/build_for_testing.log` and `.exit`: final hosted AppTests attempt, sandbox-blocked, real exit 74.
- `final/simulator_cross_build.log`, `simulator_app_emit_module.log`, `simulator_apptests_typecheck.log`, and `static_verification.exit`: arm64 iOS Simulator package/app/test static verification, all real exit 0.
- `final/simctl_list.log` and `.exit`: CoreSimulatorService sandbox failure, real exit 1.
- `device_build/xcodebuild.log` and `.exit`: signed iphoneos build attempt, sandbox-blocked, real exit 74.
- `GUARD_AUDIT.md`: every tap-chain guard and its visible/logged consequence; zero silent rows.
- `CANDIDATE_VERDICTS.md`: C1-C6 verdict/evidence table.
- `FAILING_FIRST.md`: baseline-first and failing-first limitations.
- `INSTALL.md`: exact device install/launch/log commands and two-minute checklist.
- `MANAGER_VERIFY.md`: exact normal-Terminal commands for simulator evidence, hosted tests, signed build, codesign, and staging.
- `report.json`: schema-shaped lane ruling input.

Absent by honest design: simulator screenshots/logs, `.xcresult`, staged `.app`, `file`/`lipo` proof, and `codesign` output. The sandbox prevented their creation; manager destinations are predeclared in `MANAGER_VERIFY.md`.
