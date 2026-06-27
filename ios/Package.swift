// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwayBodyIOS",
    platforms: [.iOS(.v18)],
    products: [
        .library(name: "SwayCore", targets: ["SwayCore"]),
        .library(name: "SwayCapture", targets: ["SwayCapture"]),
        .library(name: "SwayCalibration", targets: ["SwayCalibration"]),
        .library(name: "SwayFastTier", targets: ["SwayFastTier"]),
        .library(name: "SwayGuidance", targets: ["SwayGuidance"]),
        .library(name: "SwayUpload", targets: ["SwayUpload"]),
        .library(name: "SwayReplay", targets: ["SwayReplay"]),
    ],
    targets: [
        .target(name: "SwayCore", path: "Core/Sources/SwayCore"),
        .target(name: "SwayCapture", dependencies: ["SwayCore"], path: "Capture/Sources/SwayCapture"),
        .target(name: "SwayCalibration", dependencies: ["SwayCore"], path: "Calibration/Sources/SwayCalibration"),
        .target(name: "SwayFastTier", dependencies: ["SwayCore"], path: "FastTier/Sources/SwayFastTier"),
        .target(name: "SwayGuidance", dependencies: ["SwayCore"], path: "Guidance/Sources/SwayGuidance"),
        .target(name: "SwayUpload", dependencies: ["SwayCore"], path: "Upload/Sources/SwayUpload"),
        .target(name: "SwayReplay", dependencies: ["SwayCore"], path: "Replay/Sources/SwayReplay"),
        .testTarget(name: "SwayCoreTests", dependencies: ["SwayCore"], path: "Core/Tests/SwayCoreTests"),
        .testTarget(name: "SwayCaptureTests", dependencies: ["SwayCapture"], path: "Capture/Tests/SwayCaptureTests"),
        .testTarget(name: "SwayFastTierTests", dependencies: ["SwayFastTier"], path: "FastTier/Tests/SwayFastTierTests"),
        .testTarget(name: "SwayGuidanceTests", dependencies: ["SwayGuidance"], path: "Guidance/Tests/SwayGuidanceTests"),
        .testTarget(name: "SwayCalibrationTests", dependencies: ["SwayCalibration", "SwayCore"], path: "Calibration/Tests/SwayCalibrationTests"),
        .testTarget(name: "SwayUploadTests", dependencies: ["SwayUpload"], path: "Upload/Tests/SwayUploadTests"),
        .testTarget(name: "SwayReplayTests", dependencies: ["SwayReplay"], path: "Replay/Tests/SwayReplayTests"),
    ]
)
