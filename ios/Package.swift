// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PickleballIOS",
    platforms: [.iOS(.v18), .macOS(.v13)],
    products: [
        .library(name: "PickleballCore", targets: ["PickleballCore"]),
        .library(name: "PickleballCapture", targets: ["PickleballCapture"]),
        .library(name: "PickleballCalibration", targets: ["PickleballCalibration"]),
        .library(name: "PickleballFastTier", targets: ["PickleballFastTier"]),
        .library(name: "PickleballGuidance", targets: ["PickleballGuidance"]),
        .library(name: "PickleballUpload", targets: ["PickleballUpload"]),
        .library(name: "PickleballReplay", targets: ["PickleballReplay"]),
    ],
    targets: [
        .target(name: "PickleballCore", path: "Core/Sources/PickleballCore"),
        .target(
            name: "PickleballCapture",
            dependencies: ["PickleballCore", "PickleballGuidance", "PickleballFastTier"],
            path: "Capture/Sources/PickleballCapture"
        ),
        .target(name: "PickleballCalibration", dependencies: ["PickleballCore"], path: "Calibration/Sources/PickleballCalibration"),
        .target(name: "PickleballFastTier", dependencies: ["PickleballCore"], path: "FastTier/Sources/PickleballFastTier"),
        .target(name: "PickleballGuidance", dependencies: ["PickleballCore"], path: "Guidance/Sources/PickleballGuidance"),
        .target(name: "PickleballUpload", dependencies: ["PickleballCore"], path: "Upload/Sources/PickleballUpload"),
        .target(
            name: "PickleballReplay",
            dependencies: ["PickleballCore", "PickleballFastTier"],
            path: "Replay/Sources/PickleballReplay",
            resources: [
                .copy("Resources/WorldFixture"),
                .copy("Resources/RealityReplayFixture"),
            ]
        ),
        .testTarget(name: "PickleballCoreTests", dependencies: ["PickleballCore"], path: "Core/Tests/PickleballCoreTests"),
        .testTarget(name: "PickleballCaptureTests", dependencies: ["PickleballCapture"], path: "Capture/Tests/PickleballCaptureTests"),
        .testTarget(name: "PickleballFastTierTests", dependencies: ["PickleballFastTier"], path: "FastTier/Tests/PickleballFastTierTests"),
        .testTarget(name: "PickleballGuidanceTests", dependencies: ["PickleballGuidance"], path: "Guidance/Tests/PickleballGuidanceTests"),
        .testTarget(name: "PickleballCalibrationTests", dependencies: ["PickleballCalibration", "PickleballCore"], path: "Calibration/Tests/PickleballCalibrationTests"),
        .testTarget(name: "PickleballUploadTests", dependencies: ["PickleballUpload"], path: "Upload/Tests/PickleballUploadTests"),
        .testTarget(
            name: "PickleballReplayTests",
            dependencies: ["PickleballReplay"],
            path: "Replay/Tests/PickleballReplayTests",
            resources: [.copy("Resources")]
        ),
    ]
)
