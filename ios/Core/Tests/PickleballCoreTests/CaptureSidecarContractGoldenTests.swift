import Foundation
import XCTest
@testable import PickleballCore

final class CaptureSidecarContractGoldenTests: XCTestCase {
    func testCanonicalSidecarsMatchGoldenFixturesAndDecodeBack() throws {
        let fixtures: [(name: String, sidecar: CaptureSidecar)] = [
            ("full_sensors.json", fullSensorsSidecar()),
            ("missing_sensors.json", missingSensorsSidecar()),
            ("camera_roll_import.json", cameraRollImportSidecar()),
        ]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let decoder = JSONDecoder()

        for fixture in fixtures {
            let fixtureURL = goldenFixtureRootURL.appendingPathComponent(fixture.name)
            let fixtureData = try Data(contentsOf: fixtureURL)
            let encodedData = try encoder.encode(fixture.sidecar)

            XCTAssertEqual(
                try normalizedJSON(encodedData),
                try normalizedJSON(fixtureData),
                "Swift encoder output drifted from \(fixture.name)"
            )
            XCTAssertEqual(
                try decoder.decode(CaptureSidecar.self, from: fixtureData),
                fixture.sidecar,
                "Swift decoder could not round-trip \(fixture.name)"
            )
        }
    }

    private var goldenFixtureRootURL: URL {
        var url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<4 {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("tests/racketsport/fixtures/capture_sidecar", isDirectory: true)
    }

    private func normalizedJSON(_ data: Data) throws -> Data {
        let object = try JSONSerialization.jsonObject(with: data)
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private func fullSensorsSidecar() -> CaptureSidecar {
        let intrinsics = CameraIntrinsics(
            fx: 1180,
            fy: 1190,
            cx: 960,
            cy: 540,
            dist: [0.01, -0.02, 0, 0],
            source: "arkit"
        )
        let pose = RigidPose(
            R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            t: [0.2, 1.4, 0.1]
        )
        let plane = Plane(point: [0, 0, 0], normal: [0, 1, 0])
        let frame = ARKitFrameSample(
            videoPTSS: 0.041,
            arkitTimestampS: 10.04,
            cameraPose: pose,
            intrinsics: intrinsics,
            tracking: ARTrackingSnapshot(state: .normal, quality: .good),
            gravity: [0, -1, 0]
        )
        let setupPass = ARKitSetupPassSidecar(
            intrinsics: intrinsics,
            cameraPose: pose,
            courtPlane: plane,
            trackingState: .normal,
            timestampS: 10,
            durationS: 3.5,
            gravity: [0, -1, 0]
        )
        let enforcement = CapturePolicyEnforcementReport(
            requested: CapturePolicyRequestedState(
                fps: 120,
                resolution: [1920, 1080],
                format: .hevc,
                orientation: .landscape,
                electronicStabilizationEnabled: false,
                exposureLocked: true,
                focusLocked: true,
                whiteBalanceLocked: true
            ),
            achieved: CapturePolicyAchievedState(
                fps: 120,
                resolution: [1920, 1080],
                format: .hevc,
                orientation: .landscape,
                electronicStabilizationEnabled: false,
                exposureLocked: true,
                focusLocked: true,
                whiteBalanceLocked: true
            ),
            violations: []
        )

        return CaptureSidecar(
            provenance: .liveRecording,
            deviceTier: .lidar,
            deviceModel: "iPhone16,2",
            fps: 120,
            format: .hevc,
            resolution: [1920, 1080],
            orientation: .landscape,
            captureDeviceOrientation: .landscapeLeft,
            videoRotationAngleDegrees: 270,
            recordingStartedAt: "2026-07-09T12:00:00.000Z",
            recordingDurationS: 12.5,
            cameraPosition: "back",
            cameraLens: "builtInWideAngleCamera",
            locked: LockedCapture(exposureS: 0.001, iso: 200, focus: 0.7, wbLocked: true),
            intrinsics: intrinsics,
            arkitCameraPose: pose,
            courtPlane: plane,
            setupPass: setupPass,
            manualCourtTaps: [[100, 100], [1820, 100], [1820, 980], [100, 980]],
            gravity: [0, -1, 0],
            arkitFrameSamples: [frame],
            lidarDepthRefs: ["lidar/depth_0001.bin"],
            ondevicePoseTrack: "ondevice_pose.json",
            unavailableSensorReasons: [:],
            policyEnforcement: enforcement,
            profileCapture: ProfileCapturePayload(steps: [
                ProfileCaptureStepRecord(
                    kind: .emptyCourtClip,
                    status: .complete,
                    artifactRef: "captures/empty/clip.mov"
                ),
            ]),
            captureQuality: CaptureQuality(grade: .good),
            hdrEnabled: true,
            videoStabilizationEnabled: false,
            exposureLocked: true,
            focusLocked: true,
            tripodHeightM: 1.4,
            fullCourtVisible: true,
            courtLockPassed: true,
            ballHighContrast: true,
            audioRecorded: true
        )
    }

    private func missingSensorsSidecar() -> CaptureSidecar {
        CaptureSidecar(
            provenance: .liveRecording,
            deviceTier: .standard,
            deviceModel: "iPhone15,2",
            fps: 60,
            format: .hevc,
            resolution: [1920, 1080],
            orientation: .landscape,
            locked: LockedCapture(exposureS: 0.002, iso: 320, focus: 0.65, wbLocked: true),
            intrinsics: CameraIntrinsics(
                fx: 1277.9,
                fy: 1277.9,
                cx: 960,
                cy: 540,
                source: "avfoundation_fov_estimate"
            ),
            setupPass: .unavailable(
                reason: "arkit_not_supported",
                gravity: [0, -1, 0],
                durationS: 0.25
            ),
            gravity: [0, -1, 0],
            arkitFrameSamples: [
                ARKitFrameSample(
                    videoPTSS: 0.033,
                    gravity: [0, -1, 0],
                    provenance: .coreMotionOnly,
                    unavailableReason: "arkit_not_supported"
                ),
            ],
            unavailableSensorReasons: [
                "arkit_camera_pose": "arkit_not_supported",
                "court_plane": "arkit_not_supported",
                "hdr_enabled": "hdr_state_not_recorded",
                "lidar_depth": "device_has_no_lidar",
            ],
            captureQuality: CaptureQuality(
                grade: .warn,
                reasons: ["arkit_seed_missing", "court_plane_missing"]
            ),
            videoStabilizationEnabled: false,
            exposureLocked: true,
            focusLocked: true,
            audioRecorded: true
        )
    }

    private func cameraRollImportSidecar() -> CaptureSidecar {
        CaptureSidecar(
            provenance: .cameraRollImport,
            deviceTier: .standard,
            deviceModel: "camera_roll",
            fps: 30,
            format: .hevc,
            resolution: [1920, 1080],
            orientation: .landscape,
            recordingStartedAt: "2026-07-09T12:30:00.000Z",
            recordingDurationS: 8,
            locked: nil,
            intrinsics: nil,
            gravity: nil,
            unavailableSensorReasons: [
                "arkit_camera_pose": "camera_roll_import_has_no_live_arkit_tracking",
                "camera_intrinsics": "camera_roll_import_has_no_live_calibration_intrinsics",
                "core_motion_gravity": "camera_roll_import_has_no_live_core_motion",
                "court_plane": "camera_roll_import_has_no_live_court_lock",
                "locked_camera_settings": "camera_roll_import_has_no_live_exposure_focus_or_white_balance",
            ],
            captureQuality: CaptureQuality(
                grade: .warn,
                reasons: ["imported_no_live_sensors"]
            )
        )
    }
}
