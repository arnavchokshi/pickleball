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

## DinkVision Brand V3

The app-facing display name is `DinkVisionBrand.displayName` in
`ios/App/DinkVisionDesignSystem.swift`. Bundle identifiers remain unchanged.

Tokens are mirrored from `runs/lanes/ios_brand_v2_20260707/mockups/tokens.css`:

- Colors: cream `#F4EEE3`, court green `#2E5B3F`, deep green `#234731`, ink
  `#141414`, ball yellow `#F2C63F`, trail blue `#3E8EF0`, trail red
  `#E8503A`, card white `#FFFFFF`, and line `#E7DFD1`.
- Shapes: 24 pt card radius, 32 pt top tab-bar radius, black bottom tab rail,
  and owner raster artwork for all app-icon and in-app logo placements.
- Type: SF rounded/system heavy numerals and sentence-case labels. Stat cards
  use dynamic SwiftUI text with scale limits rather than fixed bitmap copy.
- Iconography: the app icon and in-app logo use the owner's actual raster
  masters from `runs/lanes/ios_brand_v3_20260707/assets/`: `app_icon_1024.png`,
  `mark_master.png`, and `lockup_master.png`. Do not redraw, re-vector, trace,
  recolor, or "improve" those files. `PaddleEyeMark` remains only for tiny
  template glyph use such as the tab bar.

Owner taste board, 2026-07-07: ink-on-cream raster mark; aligned slow-blink
splash with no extra layers; speed-streak trail card as the motion motif;
perforation panels as texture; hand-drawn double-slashes, dot grids, and curvy
orange arrow around empty/onboarding content only. The tone is playful but super
clean.

Signature animations:

- Splash v3: the app shell renders behind the overlay. Phase A `settle` is
  150 ms on full cream, with the raster mark centered at 34% screen width and a
  1.06 -> 1.00 spring scale. Phase B `blink` draws only cream lids plus ink edge
  strokes over the raster eye: close 260 ms, hold 120 ms, open 340 ms,
  ease-in-out. Phase C `openUp` is 380 ms: the mark/lids scale up from the eye
  center while the cream overlay and mark fade to reveal the live app. Total
  duration is 1.25 s. Reduced-motion users get a 180 ms crossfade.
- Splash v3 eye constants: eye center is `(0.50, 0.361)` of the mark frame,
  aperture half-height is `0.145 * markHeight`, lid span is `0.86 * markWidth`,
  and ink stroke width is `0.075 * markWidth`. Geometry helpers are nonisolated
  so SwiftUI `Shape.path(in:)` does not call MainActor-isolated code.
- Loader: `BallTrailLoadingView` is the reusable black-card speed-streak loader
  used for capture save/sidecar processing, replay loading states, and the
  brief replay-open transition.

Accent usage rules:

- `SketchSlashes`, `DotGrid`, `HandArrow`, and `PerforationPanel` live in
  `ios/App/DinkVisionAccents.swift`.
- Approved screen sites are limited to Replays empty state, Stats sample
  watermark, Profile completed checklist rows, and the permission primer.
- Keep accents off measured data surfaces and use at most one accent cluster per
  screen. Perforation panels may be ink/white dots or yellow embossed only.

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
