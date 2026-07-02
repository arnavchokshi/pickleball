import XCTest
import PickleballCore
@testable import PickleballUpload

final class UploadManifestTests: XCTestCase {
    func testManifestValidationRejectsUnsafeRelativePaths() {
        let manifest = UploadManifest(
            clipRelativePath: "../clips/session.mov",
            sidecar: Self.sidecar(ondevicePoseTrack: "../sidecar_pose.json", lidarDepthRefs: ["depth/../sidecar.bin"]),
            onDevicePoseTrack: "https://example.com/pose.json",
            onDevicePersonTracks: "../tracks.json",
            onDevicePersonTiming: "https://example.com/timing.json",
            lidarDepthRefs: ["depth/../frame.bin"]
        )

        let report = UploadManifestValidator.validate(manifest)

        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "clipRelativePath", value: "../clips/session.mov")))
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "onDevicePoseTrack", value: "https://example.com/pose.json")))
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "sidecar.ondevicePoseTrack", value: "../sidecar_pose.json")))
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "onDevicePersonTracks", value: "../tracks.json")))
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "onDevicePersonTiming", value: "https://example.com/timing.json")))
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "sidecar.lidarDepthRefs[0]", value: "depth/../sidecar.bin")))
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "lidarDepthRefs[0]", value: "depth/../frame.bin")))
    }

    func testUploadPlanRejectsUnsafeSidecarPoseFallback() {
        let manifest = UploadManifest(
            clipRelativePath: "clips/session.mov",
            sidecar: Self.sidecar(ondevicePoseTrack: "../pose.json", lidarDepthRefs: []),
            onDevicePoseTrack: nil,
            onDevicePersonTracks: nil,
            onDevicePersonTiming: nil,
            lidarDepthRefs: []
        )

        XCTAssertThrowsError(
            try UploadPlan.sidecarFirstParts(for: manifest, sidecarRelativePath: "sidecars/session.json")
        ) { error in
            guard case UploadPlanningError.invalidManifest(let errors) = error else {
                return XCTFail("expected invalid manifest, got \(error)")
            }
            XCTAssertTrue(errors.contains(.unsafeRelativePath(field: "sidecar.ondevicePoseTrack", value: "../pose.json")))
        }
    }

    func testSidecarFirstOrderingUsesPoseAndDepthBeforeClip() throws {
        let manifest = UploadManifest(
            clipRelativePath: "clips/session.mov",
            sidecar: Self.sidecar(ondevicePoseTrack: "pose/sidecar_pose.json", lidarDepthRefs: ["depth/from_sidecar.bin"]),
            onDevicePoseTrack: "pose/ondevice_pose.json",
            onDevicePersonTracks: "tracks/on_device_person_tracks.json",
            onDevicePersonTiming: "tracks/timing.json",
            lidarDepthRefs: ["depth/frame_0002.bin", "depth/frame_0001.bin"]
        )

        let parts = try UploadPlan.sidecarFirstParts(for: manifest, sidecarRelativePath: "sidecars/session.json")

        XCTAssertEqual(parts.map(\.kind), [
            .captureSidecar,
            .posePrior,
            .personTracks,
            .personTiming,
            .lidarDepth,
            .lidarDepth,
            .lidarDepth,
            .clip
        ])
        XCTAssertEqual(
            parts.map(\.relativePath),
            [
                "sidecars/session.json",
                "pose/ondevice_pose.json",
                "tracks/on_device_person_tracks.json",
                "tracks/timing.json",
                "depth/frame_0001.bin",
                "depth/frame_0002.bin",
                "depth/from_sidecar.bin",
                "clips/session.mov",
            ]
        )
    }

    func testSidecarFirstOrderingOmitsLidarPartsWhenNoDepthRefsExist() throws {
        let manifest = UploadManifest(
            clipRelativePath: "clips/session.mov",
            sidecar: Self.sidecar(ondevicePoseTrack: "pose/sidecar_pose.json", lidarDepthRefs: []),
            onDevicePoseTrack: nil,
            onDevicePersonTracks: nil,
            onDevicePersonTiming: nil,
            lidarDepthRefs: []
        )

        let parts = try UploadPlan.sidecarFirstParts(for: manifest, sidecarRelativePath: "sidecars/session.json")

        XCTAssertEqual(parts.map(\.kind), [.captureSidecar, .posePrior, .clip])
        XCTAssertFalse(parts.contains { $0.kind == .lidarDepth })
    }

    func testUploadValidationAllowsCameraRollImportWithoutGravityWhenReasonIsPresent() {
        let manifest = UploadManifest(
            clipRelativePath: "captures/imported/clip.mov",
            sidecar: CaptureSidecar(
                provenance: .cameraRollImport,
                deviceTier: .standard,
                deviceModel: "camera_roll",
                fps: 60,
                format: .hevc,
                resolution: [1920, 1080],
                locked: nil,
                intrinsics: nil,
                gravity: nil,
                unavailableSensorReasons: [
                    "core_motion_gravity": "camera_roll_import_has_no_live_core_motion",
                ],
                captureQuality: CaptureQuality(grade: .warn, reasons: ["imported_no_live_sensors"])
            )
        )

        XCTAssertTrue(UploadManifestValidator.validate(manifest).isValid)
    }

    func testChunkPlanComputesStableOffsetsLengthsAndIdentifiers() throws {
        let plan = try ResumableChunkPlan(
            relativePath: "clips/session.mov",
            fileSizeBytes: 10_485_761,
            chunkSizeBytes: 5_242_880
        )

        XCTAssertEqual(plan.chunkSizeBytes, 5_242_880)
        XCTAssertEqual(plan.chunks.count, 3)
        XCTAssertEqual(plan.chunks.map(\.index), [0, 1, 2])
        XCTAssertEqual(plan.chunks.map(\.offsetBytes), [0, 5_242_880, 10_485_760])
        XCTAssertEqual(plan.chunks.map(\.lengthBytes), [5_242_880, 5_242_880, 1])
        XCTAssertEqual(
            plan.chunks.map(\.identifier),
            [
                "clips/session.mov:0:0:5242880",
                "clips/session.mov:1:5242880:5242880",
                "clips/session.mov:2:10485760:1",
            ]
        )
    }

    func testRenderGatewayMultipartBodyIncludesVideoSidecarClipAndFrameCap() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("render-gateway-upload-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let videoURL = directory.appendingPathComponent("clip.mov")
        let sidecarURL = directory.appendingPathComponent("capture_sidecar.json")
        try Data("video-bytes".utf8).write(to: videoURL)
        try Data("{\"schema_version\":1}".utf8).write(to: sidecarURL)

        let body = try RenderGatewayMultipartBodyWriter.write(
            RenderGatewayUploadRequest(
                videoURL: videoURL,
                captureSidecarURL: sidecarURL,
                clip: "capture_001",
                maxFrames: 12
            ),
            boundary: "test-boundary",
            directory: directory
        )

        let payload = try String(contentsOf: body.fileURL, encoding: .utf8)
        XCTAssertEqual(body.contentType, "multipart/form-data; boundary=test-boundary")
        XCTAssertTrue(payload.contains(#"name="video"; filename="clip.mov""#))
        XCTAssertTrue(payload.contains(#"name="capture_sidecar"; filename="capture_sidecar.json""#))
        XCTAssertTrue(payload.contains(#"name="clip""#))
        XCTAssertTrue(payload.contains("capture_001"))
        XCTAssertTrue(payload.contains(#"name="max_frames""#))
        XCTAssertTrue(payload.contains("12"))
    }

    func testRenderGatewayJobDecodesProgressAndReplayManifestURL() throws {
        let data = Data(
            """
            {
              "id": "job_1",
              "clip": "capture_001",
              "status": "running",
              "progress": {
                "percent": 42,
                "stage": "Running pipeline on GPU",
                "message": "Tracking and body stages are active.",
                "eta_seconds": 118
              },
              "result": {
                "manifest_url": "/api/jobs/job_1/manifest"
              },
              "links": {
                "status": "/api/jobs/job_1"
              }
            }
            """.utf8
        )

        let job = try RenderGatewayJob.decode(data)

        XCTAssertEqual(job.id, "job_1")
        XCTAssertEqual(job.status, .running)
        XCTAssertEqual(job.progress?.percent, 42)
        XCTAssertEqual(job.progress?.etaText, "about 2 min")
        XCTAssertEqual(job.result?.manifestUrl, "/api/jobs/job_1/manifest")
        XCTAssertTrue(job.isActive)
    }

    private static func sidecar(
        ondevicePoseTrack: String?,
        lidarDepthRefs: [String]
    ) -> CaptureSidecar {
        CaptureSidecar(
            deviceTier: .lidar,
            deviceModel: "iPhone17,1",
            fps: 60,
            format: .hevc,
            resolution: [1920, 1080],
            locked: LockedCapture(exposureS: 0.001, iso: 200, focus: 0.8, wbLocked: true),
            intrinsics: CameraIntrinsics(fx: 1000, fy: 1000, cx: 960, cy: 540, source: "arkit"),
            gravity: [0, -1, 0],
            lidarDepthRefs: lidarDepthRefs,
            ondevicePoseTrack: ondevicePoseTrack,
            captureQuality: CaptureQuality(grade: .good)
        )
    }
}
