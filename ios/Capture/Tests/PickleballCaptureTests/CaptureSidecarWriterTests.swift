import XCTest
import PickleballCore
@testable import PickleballCapture

final class CaptureSidecarWriterTests: XCTestCase {
    func testWriterPersistsSidecarAtDescriptorPathWithRuntimeMetadata() throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("capture-sidecar-writer-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: rootURL)
        }
        let descriptor = try CapturePackageDescriptor(
            sessionID: "write-sidecar",
            policy: CapturePolicy.recommended(
                for: .swing120,
                deviceTier: .standard,
                capabilities: .hevcOnly,
                orientation: .landscape
            ),
            startedAt: Date(timeIntervalSince1970: 10),
            captureDeviceOrientation: .landscapeLeft
        )
        let context = CaptureSidecarWriteContext(
            deviceTier: .standard,
            deviceModel: "iPhone15,2",
            cameraPosition: "back",
            cameraLens: "builtInWideAngleCamera",
            locked: LockedCapture(exposureS: 0.001, iso: 220, focus: 0.7, wbLocked: true),
            intrinsics: CameraIntrinsics(fx: 1000, fy: 1000, cx: 960, cy: 540, source: "avfoundation_fov_estimate"),
            gravity: [0.0, -1.0, 0.0],
            captureQuality: CaptureQuality(
                grade: .warn,
                reasons: ["arkit_seed_missing", "court_plane_missing"]
            )
        )

        let sidecarURL = try CaptureSidecarWriter.write(
            descriptor: descriptor,
            packageRootURL: rootURL,
            recordingStartedAt: Date(timeIntervalSince1970: 12),
            finishedAt: Date(timeIntervalSince1970: 15.25),
            context: context
        )

        XCTAssertEqual(sidecarURL, rootURL.appendingPathComponent("captures/write-sidecar/capture_sidecar.json"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: sidecarURL.path))

        let sidecar = try JSONDecoder().decode(CaptureSidecar.self, from: Data(contentsOf: sidecarURL))
        XCTAssertEqual(sidecar.deviceTier, .standard)
        XCTAssertEqual(sidecar.deviceModel, "iPhone15,2")
        XCTAssertEqual(sidecar.fps, 120)
        XCTAssertEqual(sidecar.resolution, [1920, 1080])
        XCTAssertEqual(sidecar.orientation, .landscape)
        XCTAssertEqual(sidecar.captureDeviceOrientation, .landscapeLeft)
        XCTAssertEqual(sidecar.videoRotationAngleDegrees, 270)
        XCTAssertEqual(sidecar.recordingStartedAt, "1970-01-01T00:00:12.000Z")
        XCTAssertEqual(sidecar.recordingDurationS, 3.25)
        XCTAssertEqual(sidecar.cameraPosition, "back")
        XCTAssertEqual(sidecar.cameraLens, "builtInWideAngleCamera")
        XCTAssertEqual(sidecar.locked, context.locked)
        XCTAssertEqual(sidecar.intrinsics, context.intrinsics)
        XCTAssertEqual(sidecar.gravity, [0.0, -1.0, 0.0])
        XCTAssertEqual(sidecar.captureQuality, context.captureQuality)
    }
}
