import XCTest
@testable import SwayReplay

final class ReplayAssetTests: XCTestCase {
    func testReplayAssetValidationRequiresUsdzOrGlbAndPositiveDuration() {
        let asset = ReplayAsset(usdzURL: nil, glbURL: nil, durationSeconds: 0)

        let report = ReplayAssetValidator.validate(asset)

        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains(.missingRenderableAsset))
        XCTAssertTrue(report.errors.contains(.invalidDuration(0)))
    }

    func testReplayAssetReferencesExposeUsdzAndGlbMetadata() throws {
        let asset = ReplayAsset(
            usdzURL: try XCTUnwrap(URL(string: "https://cdn.example.com/replay/session.usdz")),
            glbURL: try XCTUnwrap(URL(string: "point_3.glb")),
            durationSeconds: 10
        )

        let report = ReplayAssetValidator.validate(asset)
        let references = ReplayAssetReference.references(for: asset)

        XCTAssertTrue(report.isValid)
        XCTAssertEqual(references.map(\.format), [.usdz, .glb])
        XCTAssertEqual(references.map(\.role), [.nativeRealityKit, .webShare])
        XCTAssertEqual(references.map(\.pathExtension), ["usdz", "glb"])
    }

    func testTimelineAndCapabilityDescriptorAreMetadataOnly() throws {
        let asset = ReplayAsset(
            usdzURL: try XCTUnwrap(URL(string: "https://cdn.example.com/replay/session.usdz")),
            glbURL: try XCTUnwrap(URL(string: "point_3.glb")),
            durationSeconds: 12
        )
        let timeline = ReplayTimelineDescriptor(
            schemaVersion: 1,
            worldFrame: "court_Z0",
            fps: 30,
            durationSeconds: 12,
            points: [
                ReplayTimelinePoint(
                    id: 3,
                    startSeconds: 1.2,
                    endSeconds: 11.2,
                    glbURL: try XCTUnwrap(URL(string: "point_3.glb")),
                    sizeMB: 9.4
                )
            ]
        )

        let capabilities = ReplayCapabilityDescriptor.describe(asset: asset, timeline: timeline)

        XCTAssertTrue(capabilities.supportsNativeUSDZ)
        XCTAssertTrue(capabilities.supportsWebGLB)
        XCTAssertTrue(capabilities.supportsTimelineScrubbing)
        XCTAssertTrue(capabilities.supportsFreeViewpointMetadata)
        XCTAssertFalse(capabilities.hasRealityKitRuntimeValidation)
        XCTAssertEqual(capabilities.validationScope, .metadataOnly)
    }
}
