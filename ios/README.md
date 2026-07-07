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

## DinkVision Design System

The app-facing display name is `DinkVisionBrand.displayName` in
`ios/App/DinkVisionDesignSystem.swift`. Bundle identifiers remain unchanged.

Tokens are mirrored from the 2026-07-07 manager mockups:

- Colors: cream `#F4EEE3`, court green `#2E5B3F`, deep green `#234731`, ink
  `#141414` from `tokens.css`, ball yellow `#F2C63F`, trail blue `#3E8EF0`,
  trail red `#E8503A`, card white `#FFFFFF`, and line `#E7DFD1`.
- Shapes: 24 pt card radius, 32 pt top tab-bar radius, black bottom tab rail,
  and thick rounded strokes for the paddle-eye mark.
- Type: SF rounded/system heavy numerals and sentence-case labels. Stat cards
  use dynamic SwiftUI text with scale limits rather than fixed bitmap copy.
- Iconography: the paddle-eye mark is pure SwiftUI vector drawing in
  `PaddleEyeMark`; empty states and the loader reuse the perforated-ball motif.

Signature animations:

- Splash: cream launch screen, paddle-eye blink, then an ink perforated-ball iris
  expansion that reveals the Record tab. Reduced-motion users get a short
  crossfade with no iris expansion.
- Loader: `BallTrailLoadingView` is the reusable black-card speed-streak loader
  used for capture save/sidecar processing and replay loading states.

Screen map:

- Record is the cold-launch default tab. It wraps `CaptureViewModel` and the
  P0-10a `CameraCaptureControlling` path, so start/stop still lands the
  capture package and sidecar through the existing controller. Policy chips are
  mapped from `CapturePolicyEnforcementReport`; tapping a chip shows the
  one-line fix hint. The court overlay toggle only shows/hides the existing
  live overlay surfaces.
- Replays lists real local capture packages through `CaptureLibrary` via
  `DinkVisionReplayListDataSource`. Opening a row wraps the existing replay
  viewer module; until a local capture has server replay output, the viewer
  labels that it is showing the bundled sample fixture.
- Stats is explicitly sample placeholder UI. No stat is claimed as measured
  until server wiring exists.
- Profile/Settings exposes the H0 profile checklist steps, app info, and the
  capture-policy explainer.

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
