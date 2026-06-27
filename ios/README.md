# Sway Body iOS

This scaffold is the native iOS entry point for the racket-sport capture client.
It intentionally contains module boundaries and shared data contracts only; real
AVFoundation, ARKit, Vision, upload, and RealityKit behavior lands in the IOS
tasks that depend on `ENV-3`.

## Modules

- `SwayCore`: shared Codable contracts that mirror server JSON artifacts.
- `SwayCapture`: capture-mode and session setup boundaries for IOS-1.
- `SwayCalibration`: ARKit/manual calibration seed boundaries for IOS-2.
- `SwayFastTier`: on-device Vision/Core ML preview output boundaries for IOS-3.
- `SwayGuidance`: capture-quality guidance flags for IOS-4.
- `SwayUpload`: upload manifest boundaries for IOS-5.
- `SwayReplay`: replay asset boundaries for IOS-6.

## Local Verification

```bash
swift package --package-path ios describe
swift test --package-path ios
xcodebuild -project ios/SwayBody.xcodeproj -scheme SwayBody -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```
