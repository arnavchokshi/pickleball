import XCTest
import PickleballCore
@testable import PickleballCapture

final class CameraRollVideoImportTests: XCTestCase {
    func testImporterCopiesPickedVideoIntoPipelineCapturePackage() async throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("camera-roll-import-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: rootURL)
        }
        let sourceURL = rootURL.appendingPathComponent("picked.mov")
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        try Data("old camera roll movie".utf8).write(to: sourceURL)

        let importer = CameraRollVideoImporter(
            videoProbe: StubVideoProbe(metadata: .init(
                resolution: [1920, 1080],
                fps: 59.94,
                durationSeconds: 12.4,
                format: .hevc,
                orientation: .landscape,
                warnings: ["fps_from_asset_nominal_rate"]
            )),
            sessionIDFactory: { _ in "import-20260702-120000-000" }
        )

        let result = try await importer.importVideo(
            sourceURL: sourceURL,
            packageRootURL: rootURL,
            importedAt: Date(timeIntervalSince1970: 1_783_014_400)
        )

        XCTAssertEqual(result.descriptor.directoryRelativePath, "captures/import-20260702-120000-000")
        XCTAssertEqual(result.descriptor.clipRelativePath, "captures/import-20260702-120000-000/clip.mov")
        XCTAssertEqual(result.descriptor.sidecarRelativePath, "captures/import-20260702-120000-000/capture_sidecar.json")
        XCTAssertEqual(try Data(contentsOf: result.clipURL), try Data(contentsOf: sourceURL))

        let sidecar = try JSONDecoder().decode(CaptureSidecar.self, from: Data(contentsOf: result.sidecarURL))
        XCTAssertEqual(sidecar.provenance, .cameraRollImport)
        XCTAssertEqual(sidecar.fps, 60)
        XCTAssertEqual(sidecar.resolution, [1920, 1080])
        XCTAssertEqual(sidecar.recordingDurationS, 12.4)
        XCTAssertEqual(sidecar.format, .hevc)
        XCTAssertEqual(sidecar.captureQuality.grade, .warn)
        XCTAssertTrue(sidecar.captureQuality.reasons.contains("imported_no_live_sensors"))
        XCTAssertTrue(sidecar.captureQuality.reasons.contains("fps_from_asset_nominal_rate"))
    }

    func testImportedSidecarLeavesLiveOnlySensorFieldsAbsentWithReasons() async throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("camera-roll-import-honesty-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: rootURL)
        }
        let sourceURL = rootURL.appendingPathComponent("picked.mov")
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        try Data("movie".utf8).write(to: sourceURL)

        let result = try await CameraRollVideoImporter(
            videoProbe: StubVideoProbe(metadata: .init(
                resolution: [1280, 720],
                fps: 30,
                durationSeconds: 4.0,
                format: .hevc,
                orientation: .landscape,
                warnings: []
            )),
            sessionIDFactory: { _ in "import-honest" }
        ).importVideo(
            sourceURL: sourceURL,
            packageRootURL: rootURL,
            importedAt: Date(timeIntervalSince1970: 1_783_014_401)
        )

        let sidecar = result.sidecar
        XCTAssertEqual(sidecar.provenance, .cameraRollImport)
        XCTAssertNil(sidecar.locked)
        XCTAssertNil(sidecar.intrinsics)
        XCTAssertNil(sidecar.gravity)
        XCTAssertNil(sidecar.arkitCameraPose)
        XCTAssertNil(sidecar.courtPlane)
        XCTAssertEqual(sidecar.manualCourtTaps, [])
        XCTAssertEqual(sidecar.unavailableSensorReasons["locked_camera_settings"], "camera_roll_import_has_no_live_exposure_focus_or_white_balance")
        XCTAssertEqual(sidecar.unavailableSensorReasons["camera_intrinsics"], "camera_roll_import_has_no_live_calibration_intrinsics")
        XCTAssertEqual(sidecar.unavailableSensorReasons["core_motion_gravity"], "camera_roll_import_has_no_live_core_motion")
        XCTAssertEqual(sidecar.unavailableSensorReasons["arkit_camera_pose"], "camera_roll_import_has_no_live_arkit_tracking")
        XCTAssertEqual(sidecar.unavailableSensorReasons["court_plane"], "camera_roll_import_has_no_live_court_lock")
        XCTAssertEqual(sidecar.captureQuality.grade, .poor)
        XCTAssertTrue(sidecar.captureQuality.reasons.contains("resolution_below_1080p_floor"))
        XCTAssertTrue(sidecar.captureQuality.reasons.contains("fps_below_60_floor"))
    }
}

private struct StubVideoProbe: CameraRollVideoProbing {
    var metadata: CameraRollVideoMetadata

    func metadata(for _: URL) async throws -> CameraRollVideoMetadata {
        metadata
    }
}
