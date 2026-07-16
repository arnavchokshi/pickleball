# Manager verification 2

Run from `/Users/arnavchokshi/Desktop/pickleball` in a normal Terminal. These commands intentionally
contain no output pipes, so each printed status is the real exit of the command immediately above.

## Boot simulator

```bash
xcrun simctl shutdown C719AFE6-FE07-48BF-A042-534A6FEF8748
xcrun simctl boot C719AFE6-FE07-48BF-A042-534A6FEF8748
xcrun simctl bootstatus C719AFE6-FE07-48BF-A042-534A6FEF8748 -b
open -a Simulator
```

Keep Simulator portrait for the cold-launch checks.

## Before: wave-1 HEAD

```bash
rm -rf /tmp/ios_recordvis_before_tree /tmp/ios_recordvis_before_dd /tmp/ios_recordvis_before.tar
mkdir -p /tmp/ios_recordvis_before_tree
git archive --format=tar --output=/tmp/ios_recordvis_before.tar HEAD
tar -x -f /tmp/ios_recordvis_before.tar -C /tmp/ios_recordvis_before_tree
xcodebuild -project /tmp/ios_recordvis_before_tree/ios/Pickleball.xcodeproj -scheme Pickleball -configuration Debug -destination 'platform=iOS Simulator,id=C719AFE6-FE07-48BF-A042-534A6FEF8748' -derivedDataPath /tmp/ios_recordvis_before_dd CODE_SIGNING_ALLOWED=NO build
print "BEFORE_BUILD_REAL_EXIT_CODE=$?"
xcrun simctl install C719AFE6-FE07-48BF-A042-534A6FEF8748 /tmp/ios_recordvis_before_dd/Build/Products/Debug-iphonesimulator/Pickleball.app
print "BEFORE_INSTALL_REAL_EXIT_CODE=$?"
xcrun simctl privacy C719AFE6-FE07-48BF-A042-534A6FEF8748 grant camera com.arnavchokshi.pickleball
xcrun simctl privacy C719AFE6-FE07-48BF-A042-534A6FEF8748 grant microphone com.arnavchokshi.pickleball
xcrun simctl launch --terminate-running-process C719AFE6-FE07-48BF-A042-534A6FEF8748 com.arnavchokshi.pickleball -dinkvision.skipSplash -dinkvision.captureState live
print "BEFORE_LAUNCH_REAL_EXIT_CODE=$?"
xcrun simctl io C719AFE6-FE07-48BF-A042-534A6FEF8748 screenshot runs/lanes/ios_recordvis_20260716/before_portrait_cold.png
```

## After: working tree

```bash
rm -rf /tmp/ios_recordvis_after_dd
xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball -configuration Debug -destination 'platform=iOS Simulator,id=C719AFE6-FE07-48BF-A042-534A6FEF8748' -derivedDataPath /tmp/ios_recordvis_after_dd CODE_SIGNING_ALLOWED=NO build
print "AFTER_BUILD_REAL_EXIT_CODE=$?"
xcrun simctl install C719AFE6-FE07-48BF-A042-534A6FEF8748 /tmp/ios_recordvis_after_dd/Build/Products/Debug-iphonesimulator/Pickleball.app
print "AFTER_INSTALL_REAL_EXIT_CODE=$?"
xcrun simctl privacy C719AFE6-FE07-48BF-A042-534A6FEF8748 grant camera com.arnavchokshi.pickleball
xcrun simctl privacy C719AFE6-FE07-48BF-A042-534A6FEF8748 grant microphone com.arnavchokshi.pickleball
xcrun simctl launch --terminate-running-process C719AFE6-FE07-48BF-A042-534A6FEF8748 com.arnavchokshi.pickleball -dinkvision.skipSplash -dinkvision.captureState live
print "AFTER_LAUNCH_REAL_EXIT_CODE=$?"
xcrun simctl io C719AFE6-FE07-48BF-A042-534A6FEF8748 screenshot runs/lanes/ios_recordvis_20260716/after_portrait_cold.png
```

Follow `SIM_PROOF_PLAN.md` for the manual tap, Retry, Reduce Motion, landscape-clear screenshots,
and RecordPath log capture.

## Full hosted AppTests

```bash
rm -rf /tmp/ios_recordvis_manager_tests_dd
xcodebuild build-for-testing -project ios/Pickleball.xcodeproj -scheme Pickleball -destination 'platform=iOS Simulator,id=C719AFE6-FE07-48BF-A042-534A6FEF8748' -derivedDataPath /tmp/ios_recordvis_manager_tests_dd CODE_SIGNING_ALLOWED=NO
print "BUILD_FOR_TESTING_REAL_EXIT_CODE=$?"
find /tmp/ios_recordvis_manager_tests_dd/Build/Products -name '*.xctestrun' -print
```

Copy the single printed `.xctestrun` path into the next command without changing the destination:

```bash
xcodebuild test-without-building -xctestrun /tmp/ios_recordvis_manager_tests_dd/Build/Products/REPLACE_WITH_PRINTED_FILE.xctestrun -destination 'platform=iOS Simulator,id=C719AFE6-FE07-48BF-A042-534A6FEF8748' -only-testing:PickleballAppTests -resultBundlePath /tmp/ios_recordvis_manager_tests_dd/AppTests.xcresult
print "APPTESTS_REAL_EXIT_CODE=$?"
```

Expected post-change count is 58 hosted tests. The only allowed failure is the pre-existing
`ANELatencyBenchmarkTests.testMeasureOnDeviceLatencyForConvertedModels`; all seven wave-1 C1-C6/
watchdog regressions and the four new wave-2 integration tests must be green.
