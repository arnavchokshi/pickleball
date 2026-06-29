# Pickleball iOS

This scaffold is the native iOS entry point for the Pickleball capture client.
It contains the app shell, partial AVFoundation capture scaffolding, sensor
sidecar contracts, and module boundaries for the video-to-pipeline workflow.

## Modules

- `PickleballCore`: shared Codable contracts that mirror server JSON artifacts.
- `PickleballCapture`: capture-mode and session setup boundaries for IOS-1.
- `PickleballCalibration`: ARKit/manual calibration seed boundaries for IOS-2.
- `PickleballFastTier`: on-device Vision/Core ML preview output boundaries for IOS-3.
- `PickleballGuidance`: capture-quality guidance flags for IOS-4.
- `PickleballUpload`: upload manifest boundaries for IOS-5.
- `PickleballReplay`: replay asset boundaries for IOS-6.

## Local Verification

`swift test` covers the Swift package modules. The shared Xcode scheme now has a
minimal hosted `PickleballAppTests` target for camera-free app state coverage.
This proves the app target can be built for XCTest, but it is not physical-device
capture validation.
Capture landscape enforcement, ARKit setup, Core ML fast tier, upload, and
RealityKit replay remain scaffold/partial unless promoted in the root checklist.

```bash
swift package --package-path ios describe
swift test --package-path ios
xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
xcodebuild -project ios/Pickleball.xcodeproj -scheme Pickleball -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17 Pro,OS=26.5' CODE_SIGNING_ALLOWED=NO test
```
