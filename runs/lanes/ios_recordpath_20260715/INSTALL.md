# iPhone 14 Pro install and two-minute RecordPath check

Device verification is not claimed. The phone was unavailable, and this sandbox could not produce the signed app because Xcode package resolution failed with `sandbox-exec: sandbox_apply: Operation not permitted` (real exit 74). After the manager runs the build/stage commands in `MANAGER_VERIFY.md`, the expected staged bundle is:

`/Users/arnavchokshi/Desktop/pickleball/runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app`

## Install and launch

```bash
xcrun devicectl device install app --device B03696B6-6481-5FCD-A79C-105DA3F08F98 /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app
xcrun devicectl device process launch --device B03696B6-6481-5FCD-A79C-105DA3F08F98 --terminate-existing com.arnavchokshi.pickleball
```

## Capture RecordPath diagnostics

Terminal A launches the app with an attached console and saves the CoreDevice transcript:

```bash
xcrun devicectl device process launch --device B03696B6-6481-5FCD-A79C-105DA3F08F98 --terminate-existing --console --log-output /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ios_recordpath_20260715/device_build/devicectl_console.log com.arnavchokshi.pickleball
```

After the two-minute check, collect and render the device unified log. `RecordPath` messages use subsystem `com.arnavchokshi.pickleball` and category `RecordPath`.

```bash
sudo /usr/bin/log collect --device-udid B03696B6-6481-5FCD-A79C-105DA3F08F98 --last 5m --output /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ios_recordpath_20260715/device_build/RecordPath.logarchive --predicate 'subsystem == "com.arnavchokshi.pickleball" AND category == "RecordPath"'
/usr/bin/log show --archive /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ios_recordpath_20260715/device_build/RecordPath.logarchive --style compact --info --debug --predicate 'subsystem == "com.arnavchokshi.pickleball" AND category == "RecordPath"' > /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ios_recordpath_20260715/device_build/RecordPath.log
```

## Two-minute checklist

1. In Settings > Privacy & Security > Camera and Microphone, note DinkVision's current grants. Do not infer that reinstall reset them.
2. With both grants on, cold-launch in portrait. Expect a visible `Rotate to landscape to record` banner with a visible `Retry`; the record control must be enabled for retry.
3. Rotate to landscape. Expect `requestingAccess` only briefly, then Ready, or within 8 seconds a visible actionable blocked banner. No indefinite disabled yellow button is allowed.
4. Tap Record once. Expect the red Stop control and elapsed pill, or within 8 seconds a visible blocked banner with Retry.
5. Tap Stop. Expect Saving, then Capture saved, or within 8 seconds a visible blocked banner. Confirm no background recording remains if a watchdog fires.
6. Rapidly double-tap Retry/Record once. Expect one coalesced preparation/action in `RecordPath.log`, not duplicate ARKit setup passes.
7. Turn Camera permission off in Settings, relaunch, and tap Retry. Expect `Enable Camera in Settings, then tap Retry.`; restore the grant afterward.
8. Save the console/unified logs and screenshots. Physical camera handoff, actual recording, and device UX remain UNPROVEN until this checklist passes.
