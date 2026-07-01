# Pickleball iOS

This scaffold is the native iOS entry point for the Pickleball capture client.
It contains the app shell, partial AVFoundation capture scaffolding, sensor
sidecar contracts, and module boundaries for the video-to-pipeline workflow.

Canonical split: the iOS app owns ON-DEVICE LIVE / fast tier work only
(cached ARKit court seed, person detect+track+N-lock, 2D pose/joints, the
~288p CoreML ball heatmap spike, cheap line/contact cues, court map, one cue,
and capture-quality guidance). SERVER OFFLINE / deep tier work is async GPU
(mesh, world grounding, foot-lock/physics, paddle 6DoF, full biomech, replay
render, LLM copy, week-over-week). Camera-space mesh preview is `server-fast`,
not phone-real-time; LiDAR is a near-field (~5 m) bonus only.

## Modules

- `PickleballCore`: shared Codable contracts that mirror server JSON artifacts.
- `PickleballCapture`: capture-mode and session setup boundaries for IOS-1.
- `PickleballCalibration`: ARKit/manual calibration seed boundaries for IOS-2.
- `PickleballFastTier`: ON-DEVICE LIVE Vision/Core ML boundaries for IOS-3.
- `PickleballGuidance`: capture-quality guidance flags for IOS-4.
- `PickleballUpload`: upload manifest boundaries for IOS-5.
- `PickleballReplay`: playback boundaries for server-rendered replay assets in IOS-6.

## Local Verification

`swift test` covers the Swift package modules. The shared Xcode scheme now has a
minimal hosted `PickleballAppTests` target for camera-free app state coverage.
This proves the app target can be built for XCTest, but it is not physical-device
capture validation.
Capture landscape enforcement, ARKit setup, Core ML live tier, upload, and
RealityKit playback remain scaffold/partial unless promoted in the root
checklist.

```bash
swift package --package-path ios describe
swift test --package-path ios
xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17 Pro,OS=26.5' CODE_SIGNING_ALLOWED=NO test
```
