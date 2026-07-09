import Foundation
import PickleballCapture
import PickleballUpload
import XCTest
@testable import Pickleball

final class UploadWiringTests: XCTestCase {
    @MainActor
    func testRuntimeBaseURLOverrideThreadsToAllNetworkClients() {
        let configuration = DinkVisionRuntimeConfiguration.current(
            arguments: ["app", "-dinkvision.apiBaseURL", "http://127.0.0.1:8765"],
            environment: [:],
            infoDictionary: [:]
        )

        XCTAssertEqual(configuration.apiBaseURL.absoluteString, "http://127.0.0.1:8765")
        XCTAssertEqual(configuration.makeAuthApiClient().baseURL, configuration.apiBaseURL)
        XCTAssertEqual(configuration.makePresignedUploadClient().baseURL, configuration.apiBaseURL)
        XCTAssertEqual(configuration.makeRenderGatewayClient().baseURL, configuration.apiBaseURL)
    }

    @MainActor
    func testCameraRollSelectionCoordinatorProducesUploadableLibraryItem() async {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("camera-roll-wiring-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("selected.mov")
        let expected = CaptureLibraryItem(
            sessionID: "import-selection",
            clipRelativePath: "captures/import-selection/clip.mov",
            sidecarRelativePath: "captures/import-selection/capture_sidecar.json",
            provenance: .cameraRollImport,
            durationSeconds: 12,
            fps: 60,
            resolution: [1920, 1080],
            captureQualityGrade: .warn,
            recordedAt: Date(timeIntervalSince1970: 1_783_440_000)
        )
        var receivedSource: URL?
        var receivedRoot: URL?
        let coordinator = CameraRollImportCoordinator(packageRootURL: root) { selected, packageRoot in
            receivedSource = selected
            receivedRoot = packageRoot
            return expected
        }

        let item = await coordinator.importVideo(at: source)

        XCTAssertEqual(receivedSource, source)
        XCTAssertEqual(receivedRoot, root)
        XCTAssertEqual(item, expected)
        XCTAssertEqual(item?.clipRelativePath, "captures/import-selection/clip.mov")
        XCTAssertEqual(item?.sidecarRelativePath, "captures/import-selection/capture_sidecar.json")
        XCTAssertNil(coordinator.errorMessage)
        XCTAssertFalse(coordinator.isImporting)
    }
}
