import XCTest
import PickleballCore
@testable import PickleballCapture

final class CaptureLibraryTests: XCTestCase {
    func testCaptureLibraryListsRecordedAndImportedPackagesWithImportBadgeState() throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("capture-library-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: rootURL)
        }

        try writePackage(
            rootURL: rootURL,
            sessionID: "recorded-001",
            sidecar: recordedSidecar(startedAt: "2026-07-02T12:00:00.000Z")
        )
        try writePackage(
            rootURL: rootURL,
            sessionID: "imported-001",
            sidecar: importedSidecar(startedAt: "2026-07-02T12:05:00.000Z")
        )

        let items = try CaptureLibrary.listPackages(packageRootURL: rootURL)

        XCTAssertEqual(items.map(\.sessionID), ["imported-001", "recorded-001"])
        XCTAssertEqual(items[0].clipRelativePath, "captures/imported-001/clip.mov")
        XCTAssertEqual(items[0].sidecarRelativePath, "captures/imported-001/capture_sidecar.json")
        XCTAssertEqual(items[0].provenance, .cameraRollImport)
        XCTAssertTrue(items[0].isImported)
        XCTAssertEqual(items[0].badgeText, "imported")
        XCTAssertEqual(items[0].durationSeconds, 8.5)
        XCTAssertEqual(items[0].fps, 60)
        XCTAssertEqual(items[0].resolution, [1920, 1080])

        XCTAssertEqual(items[1].provenance, .liveRecording)
        XCTAssertFalse(items[1].isImported)
        XCTAssertNil(items[1].badgeText)
    }

    private func writePackage(rootURL: URL, sessionID: String, sidecar: CaptureSidecar) throws {
        let directory = rootURL.appendingPathComponent("captures/\(sessionID)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("clip".utf8).write(to: directory.appendingPathComponent("clip.mov"))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(sidecar).write(to: directory.appendingPathComponent("capture_sidecar.json"))
    }

    private func recordedSidecar(startedAt: String) -> CaptureSidecar {
        CaptureSidecar(
            provenance: .liveRecording,
            deviceTier: .standard,
            deviceModel: "iPhone15,2",
            fps: 60,
            format: .hevc,
            resolution: [1920, 1080],
            recordingStartedAt: startedAt,
            recordingDurationS: 6.0,
            locked: LockedCapture(exposureS: 0.001, iso: 200, focus: 0.7, wbLocked: true),
            intrinsics: CameraIntrinsics(fx: 1000, fy: 1000, cx: 960, cy: 540, source: "avfoundation_fov_estimate"),
            gravity: [0, -1, 0],
            captureQuality: CaptureQuality(grade: .warn)
        )
    }

    private func importedSidecar(startedAt: String) -> CaptureSidecar {
        CaptureSidecar(
            provenance: .cameraRollImport,
            deviceTier: .standard,
            deviceModel: "camera_roll",
            fps: 60,
            format: .hevc,
            resolution: [1920, 1080],
            recordingStartedAt: startedAt,
            recordingDurationS: 8.5,
            locked: nil,
            intrinsics: nil,
            gravity: nil,
            unavailableSensorReasons: [
                "locked_camera_settings": "camera_roll_import_has_no_live_exposure_focus_or_white_balance",
                "core_motion_gravity": "camera_roll_import_has_no_live_core_motion",
            ],
            captureQuality: CaptureQuality(grade: .warn, reasons: ["imported_no_live_sensors"])
        )
    }
}
