# Real-device evidence — iPhone 14 Pro B03696B6, 2026-07-16 00:34-00:59 PDT

Staged fixed Debug build installed+launched by coordinator 00:34 (devicectl, both EXIT 0).
Owner live test ~00:3x: "tapping the yellow button does nothing, no banner" (relayed).

## 1. Console-attached relaunch (manager, 00:43:55) — decisive state-machine capture
`xcrun devicectl device process launch --console` with OS_ACTIVITY_DT_MODE=YES; full log:
`device_build/device_console_diag.log`. RecordPath story (1.4 s from launch to terminal state):

- First-appearance orientation refresh landscapeRight -> portrait  (C6 fix live)
- State idle -> requestingAccess; reason=prepare
- Permission request entered: camera then microphone
- Camera permission already determined / Microphone already determined
- Permission request terminal camera=authorized microphone=authorized  (TCC CLEAN — C5 refuted on-device)
- Configure entered mode=standard60 orientation=portrait
- Configure blocked: landscape required
- State requestingAccess -> blocked("Rotate to landscape to record"); reason=preparation error
- Preparation attempt reached terminal state=blocked (loud, bounded — no watchdog needed)

READING: the record path WORKS on-device with the fix. The phone was in PORTRAIT (desk/plugged-in).
The product failure reframes to: blocked states were not visible/salient enough for the owner
(pre-fix build additionally had the true silent mechanisms C1/C3/C4/C6 — no banner ever).

## 2. Signal-9 attribution
Coordinator's 00:40:50 console launch ended "terminated due to signal 9" — that SIGKILL was this
manager's own 00:43:55 `--terminate-existing` relaunch (timeline exact). No jetsam/watchdog kill.
Stray stdout lines ("warning: using linearization / solving fallback.", bare "0.5") have ZERO
grep hits in ios/ Swift sources — framework-internal chatter, not app code.

## 3. On-device UI test attempt (00:48) — INFRASTRUCTURE ARTIFACT, not product evidence
`xcodebuild test RecordStopUITests` on the device: FAILED "DinkVisionRecordButton never became
hittable" after 31.2 s. The xcresult screen recording (uitest_attachments/F2B30051-....mp4, frames
device_build/devframe_*.png) shows the phone on the HOME SCREEN the whole window with the
DinkVision icon in "Installing..." state — xcodebuild's reinstall raced the launch; the app never
foregrounded; all 57 accessibility snapshots are springboard-empty. DO NOT count this run as
evidence about the record UI. Rerun required once install settles (best-effort overnight; phone
may lock).

## 3b. UI test retry (01:09) — device LOCKED
Retry of RecordStopUITests failed pre-launch: "Xcode cannot launch PickleballAppUITests on arnav
because the device is locked" (owner asleep). Inconclusive by lock, not a product signal. Device
work resumes with the owner's 60-second morning script (device_build/MORNING_SCRIPT.md), whose
steps 1-2 answer the banner-visibility question directly.

## 4. Open question carried to wave 2 (ios_recordvis_20260716)
On-device visual confirmation that the blocked banner RENDERS (state machine provably reaches it;
simulator portrait screenshot proves the layout renders it). Wave 2 makes blocked states
impossible to miss regardless: persistent portrait rotate prompt on the Record tab (pre-tap),
visible per-tap reaction while blocked/disabled, banner prominence per §1.3 + reduced-motion.
