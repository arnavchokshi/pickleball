# Simulator live-controller before/after proof plan

Simulator execution was unavailable in the Codex sandbox (`simctl` EXIT 1; CoreSimulatorService
connection refused). Run this plan outside the sandbox. Use the real `captureState live` path;
never substitute the walker controller.

1. Boot `DinkVision-Test` (`C719AFE6-FE07-48BF-A042-534A6FEF8748`) in portrait.
2. Build/install the detached `HEAD` archive (wave 1) and grant camera + microphone permission.
   Cold-launch Record in portrait and capture `before_portrait_cold.png`. This is the before layout;
   do not claim a persistent prompt if only the wave-1 banner appears.
3. Build/install the working tree, grant camera + microphone permission, and cold-launch Record in
   portrait. Wait for RecordPath to reach `blocked("Rotate to landscape to record")`; capture
   `after_portrait_cold.png`. Required visual: persistent rotate card plus prominent banner/Retry.
4. Tap `DinkVisionRecordButton` while still portrait and immediately capture
   `after_portrait_tap.png`. Visually confirm press/depress plus wobble/highlight; repeat with Reduce
   Motion enabled and confirm no wobble/pulse, only the static thick highlight.
5. Tap the banner Retry. Capture `after_retry.png`; confirm `Setting up camera…` appears immediately
   and the path returns to the persistent portrait blocker if still portrait.
6. Rotate landscape. Confirm the portrait blocker disappears as soon as the blocker clears. Do not
   infer real camera recording success from Simulator.
7. Save RecordPath logs and screenshots under this lane directory and record real exit codes.

