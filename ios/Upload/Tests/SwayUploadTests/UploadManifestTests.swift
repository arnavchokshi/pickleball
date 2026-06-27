import XCTest
import SwayCore
@testable import SwayUpload

final class UploadManifestTests: XCTestCase {
    func testManifestValidationRejectsUnsafeRelativePaths() {
        let manifest = UploadManifest(
            clipRelativePath: "../clips/session.mov",
            sidecar: Self.sidecar(ondevicePoseTrack: "pose/ondevice_pose.json", lidarDepthRefs: ["depth/frame_0001.bin"]),
            onDevicePoseTrack: "https://example.com/pose.json",
            lidarDepthRefs: ["depth/../frame.bin"]
        )

        let report = UploadManifestValidator.validate(manifest)

        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "clipRelativePath", value: "../clips/session.mov")))
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "onDevicePoseTrack", value: "https://example.com/pose.json")))
        XCTAssertTrue(report.errors.contains(.unsafeRelativePath(field: "lidarDepthRefs[0]", value: "depth/../frame.bin")))
    }

    func testSidecarFirstOrderingUsesPoseAndDepthBeforeClip() throws {
        let manifest = UploadManifest(
            clipRelativePath: "clips/session.mov",
            sidecar: Self.sidecar(ondevicePoseTrack: "pose/sidecar_pose.json", lidarDepthRefs: ["depth/from_sidecar.bin"]),
            onDevicePoseTrack: "pose/ondevice_pose.json",
            lidarDepthRefs: ["depth/frame_0002.bin", "depth/frame_0001.bin"]
        )

        let parts = try UploadPlan.sidecarFirstParts(for: manifest, sidecarRelativePath: "sidecars/session.json")

        XCTAssertEqual(parts.map(\.kind), [.captureSidecar, .posePrior, .lidarDepth, .lidarDepth, .clip])
        XCTAssertEqual(
            parts.map(\.relativePath),
            [
                "sidecars/session.json",
                "pose/ondevice_pose.json",
                "depth/frame_0001.bin",
                "depth/frame_0002.bin",
                "clips/session.mov",
            ]
        )
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
