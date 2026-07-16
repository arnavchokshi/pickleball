# Manager verification commands

These commands are required because CoreSimulatorService and Xcode's SwiftPM sandbox were unavailable in the Codex workspace sandbox. Run from `/Users/arnavchokshi/Desktop/pickleball` in a normal Terminal.

## 1. Boot the named simulator

```bash
xcrun simctl shutdown C719AFE6-FE07-48BF-A042-534A6FEF8748
xcrun simctl boot C719AFE6-FE07-48BF-A042-534A6FEF8748
xcrun simctl bootstatus C719AFE6-FE07-48BF-A042-534A6FEF8748 -b
open -a Simulator
```

## 2. Pre-fix live-controller evidence from HEAD

The lane source edits are uncommitted, so `HEAD` is the pre-fix source snapshot. This uses a detached archive, not a branch or commit.

```bash
rm -rf /tmp/ios_recordpath_prefixtree /tmp/ios_recordpath_prefix_dd
mkdir -p /tmp/ios_recordpath_prefixtree
git archive HEAD | tar -x -C /tmp/ios_recordpath_prefixtree
xcodebuild -project /tmp/ios_recordpath_prefixtree/ios/Pickleball.xcodeproj -scheme Pickleball -configuration Debug -destination 'platform=iOS Simulator,id=C719AFE6-FE07-48BF-A042-534A6FEF8748' -derivedDataPath /tmp/ios_recordpath_prefix_dd CODE_SIGNING_ALLOWED=NO build
xcrun simctl install C719AFE6-FE07-48BF-A042-534A6FEF8748 /tmp/ios_recordpath_prefix_dd/Build/Products/Debug-iphonesimulator/Pickleball.app
xcrun simctl privacy C719AFE6-FE07-48BF-A042-534A6FEF8748 reset all com.arnavchokshi.pickleball
xcrun simctl launch --terminate-running-process C719AFE6-FE07-48BF-A042-534A6FEF8748 com.arnavchokshi.pickleball -dinkvision.skipSplash -dinkvision.captureState live
xcrun simctl spawn C719AFE6-FE07-48BF-A042-534A6FEF8748 log show --last 2m --style compact --info --debug --predicate 'subsystem == "com.arnavchokshi.pickleball" AND category == "RecordPath"' > runs/lanes/ios_recordpath_20260715/baseline/simulator_recordpath.log
xcrun simctl io C719AFE6-FE07-48BF-A042-534A6FEF8748 screenshot runs/lanes/ios_recordpath_20260715/baseline/live_before.png
```

Confirm the launch command has `captureState live` and does not contain `-dinkvision.walker`. In Simulator, tap the center Record control before the log/screenshot commands. Record whether the pre-fix app is stuck/disabled or already reaches an existing blocked state; do not force a C1 claim if the simulator follows a permission-denied path instead.

## 3. Post-fix live-controller banner and Retry evidence

```bash
rm -rf /tmp/ios_recordpath_postfix_dd
xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball -configuration Debug -destination 'platform=iOS Simulator,id=C719AFE6-FE07-48BF-A042-534A6FEF8748' -derivedDataPath /tmp/ios_recordpath_postfix_dd CODE_SIGNING_ALLOWED=NO build
xcrun simctl install C719AFE6-FE07-48BF-A042-534A6FEF8748 /tmp/ios_recordpath_postfix_dd/Build/Products/Debug-iphonesimulator/Pickleball.app
xcrun simctl privacy C719AFE6-FE07-48BF-A042-534A6FEF8748 reset all com.arnavchokshi.pickleball
xcrun simctl launch --terminate-running-process C719AFE6-FE07-48BF-A042-534A6FEF8748 com.arnavchokshi.pickleball -dinkvision.skipSplash -dinkvision.captureState live
xcrun simctl spawn C719AFE6-FE07-48BF-A042-534A6FEF8748 log show --last 2m --style compact --info --debug --predicate 'subsystem == "com.arnavchokshi.pickleball" AND category == "RecordPath"' > runs/lanes/ios_recordpath_20260715/final/simulator_recordpath.log
xcrun simctl io C719AFE6-FE07-48BF-A042-534A6FEF8748 screenshot runs/lanes/ios_recordpath_20260715/final/live_blocked_after.png
```

Tap Record, confirm a loud blocked banner with visible Retry, tap Retry once, then capture a second screenshot:

```bash
xcrun simctl io C719AFE6-FE07-48BF-A042-534A6FEF8748 screenshot runs/lanes/ios_recordpath_20260715/final/live_retry_after.png
```

## 4. Full hosted AppTests

```bash
rm -rf /tmp/ios_recordpath_manager_tests_dd
xcodebuild build-for-testing -project ios/Pickleball.xcodeproj -scheme Pickleball -destination 'platform=iOS Simulator,id=C719AFE6-FE07-48BF-A042-534A6FEF8748' -derivedDataPath /tmp/ios_recordpath_manager_tests_dd CODE_SIGNING_ALLOWED=NO
xctestrun=$(find /tmp/ios_recordpath_manager_tests_dd/Build/Products -name '*.xctestrun' -print -quit)
xcodebuild test-without-building -xctestrun "$xctestrun" -destination 'platform=iOS Simulator,id=C719AFE6-FE07-48BF-A042-534A6FEF8748' -only-testing:PickleballAppTests -resultBundlePath /tmp/ios_recordpath_manager_tests_dd/AppTests.xcresult | tee runs/lanes/ios_recordpath_20260715/final/test_without_building_manager.log
apptests_rc=${pipestatus[1]}
print "TEST_WITHOUT_BUILDING_REAL_EXIT_CODE=$apptests_rc" | tee runs/lanes/ios_recordpath_20260715/final/test_without_building_manager.exit
```

The only allowed simulator failure is the pre-existing `ANELatencyBenchmarkTests.testMeasureOnDeviceLatencyForConvertedModels`. Record all executed/passed/failed counts and preserve the `.xcresult` summary.

## 5. Signed iphoneos Debug app, staging, and signature proof

```bash
rm -rf /tmp/ios_recordpath_manager_device_dd runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app
xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball -configuration Debug -sdk iphoneos -destination 'generic/platform=iOS' -derivedDataPath /tmp/ios_recordpath_manager_device_dd -allowProvisioningUpdates build | tee runs/lanes/ios_recordpath_20260715/device_build/xcodebuild_manager.log
test ${pipestatus[1]} -eq 0
ditto /tmp/ios_recordpath_manager_device_dd/Build/Products/Debug-iphoneos/Pickleball.app runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app
file runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app/Pickleball | tee runs/lanes/ios_recordpath_20260715/device_build/file.txt
lipo -info runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app/Pickleball | tee runs/lanes/ios_recordpath_20260715/device_build/lipo.txt
codesign -dv --verbose=4 runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app 2>&1 | tee runs/lanes/ios_recordpath_20260715/device_build/codesign.txt
codesign --verify --deep --strict --verbose=4 runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app
```

Then use the exact install, launch, log, and checklist commands in `INSTALL.md`. Do not install until the phone returns.
