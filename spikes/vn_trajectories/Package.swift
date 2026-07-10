// swift-tools-version:5.9
import PackageDescription

// W0-BALL-SPIKE (see NORTH_STAR_ROADMAP.md): rung-1 spike of Apple's native
// VNDetectTrajectoriesRequest (Vision framework) as an on-device ball tracker
// candidate, evaluated offline on this Mac against the committed eval clips
// while the paired iPhone is unavailable. No third-party dependencies -
// everything here is AVFoundation/Vision/Foundation so it builds without
// network access to the Swift package registry.
let package = Package(
    name: "VNTrajectorySpike",
    platforms: [
        .macOS(.v12)
    ],
    targets: [
        .executableTarget(
            name: "VNTrajectorySpike",
            path: "Sources/VNTrajectorySpike"
        )
    ]
)
