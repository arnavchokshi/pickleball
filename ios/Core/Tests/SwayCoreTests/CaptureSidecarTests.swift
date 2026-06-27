import XCTest
@testable import SwayCore

final class CaptureSidecarTests: XCTestCase {
    func testCaptureSidecarDecodesServerPayload() throws {
        let json = """
        {
          "schema_version": 1,
          "device_tier": "B_standard",
          "device_model": "iPhone16,2",
          "fps": 120,
          "format": "hevc",
          "resolution": [1920, 1080],
          "orientation": "landscape",
          "locked": {
            "exposure_s": 0.001,
            "iso": 320,
            "focus": 0.7,
            "wb_locked": true
          },
          "intrinsics": {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0,
            "dist": [0.0, 0.0, 0.0, 0.0],
            "source": "arkit"
          },
          "arkit_camera_pose": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 1.5, 0.0]
          },
          "court_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
          "manual_court_taps": [[10.0, 10.0], [100.0, 10.0], [100.0, 80.0], [10.0, 80.0]],
          "gravity": [0.0, -1.0, 0.0],
          "lidar_depth_refs": [],
          "ondevice_pose_track": "ondevice_pose.json",
          "capture_quality": {"grade": "good", "reasons": []}
        }
        """

        let sidecar = try JSONDecoder().decode(CaptureSidecar.self, from: Data(json.utf8))

        XCTAssertEqual(sidecar.schemaVersion, 1)
        XCTAssertEqual(sidecar.deviceTier, .standard)
        XCTAssertEqual(sidecar.format, .hevc)
        XCTAssertEqual(sidecar.orientation, .landscape)
        XCTAssertEqual(sidecar.locked.wbLocked, true)
        XCTAssertEqual(sidecar.captureQuality.grade, .good)
        XCTAssertEqual(sidecar.ondevicePoseTrack, "ondevice_pose.json")
    }

    func testCaptureSidecarEncodesSnakeCasePayload() throws {
        let sidecar = CaptureSidecar(
            deviceTier: .standard,
            deviceModel: "iPhone16,2",
            fps: 60,
            format: .hevc,
            resolution: [1920, 1080],
            locked: LockedCapture(exposureS: 0.001, iso: 200, focus: 0.8, wbLocked: true),
            intrinsics: CameraIntrinsics(fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0, source: "arkit"),
            gravity: [0.0, -1.0, 0.0],
            captureQuality: CaptureQuality(grade: .warn, reasons: ["motion_blur_marginal"])
        )

        let data = try JSONEncoder().encode(sidecar)
        let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        XCTAssertEqual(payload?["schema_version"] as? Int, 1)
        XCTAssertEqual(payload?["device_tier"] as? String, "B_standard")
        XCTAssertEqual(payload?["capture_quality"] as? [String: Any]? != nil, true)
    }
}
