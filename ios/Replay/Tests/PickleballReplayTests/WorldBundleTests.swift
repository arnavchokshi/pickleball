import XCTest
@testable import PickleballReplay

final class WorldBundleTests: XCTestCase {
    func testBundledSampleLoadsVerifiedWolverineGlueRun() throws {
        let bundle = try WorldBundle.loadBundledSample()

        XCTAssertEqual(bundle.manifest.clip, "wolverine_mixed_0200_mid_steep_corner")
        XCTAssertEqual(bundle.world.players.count, 4)
        XCTAssertEqual(bundle.world.court.sport, "pickleball")
        XCTAssertGreaterThan(bundle.world.ball.frames.count, 0)
        XCTAssertGreaterThan(bundle.world.summary.jointPlayerFrameCount, 0)
        XCTAssertEqual(bundle.world.summary.meshPlayerFrameCount, 0)
        XCTAssertNil(bundle.bodyMesh, "bundled fixture intentionally omits the 570 MB verified BODY mesh artifact")
        XCTAssertNotNil(bundle.contactWindows, "verified process_video run regenerated contact windows after remote skeleton fallback")
        XCTAssertEqual(bundle.contactWindows?.events.count, 24)
    }

    func testBundledSamplePlayersHaveHonestRepresentationsAndTrustBands() throws {
        let bundle = try WorldBundle.loadBundledSample()
        XCTAssertEqual(Set(bundle.world.players.map(\.representation)), [.joints])
        XCTAssertTrue(bundle.world.players.allSatisfy { $0.trustBand?.badge == .preview })

        XCTAssertEqual(bundle.world.ball.trustBand?.badge, .lowConfidence)
        XCTAssertEqual(bundle.world.court.trustBand?.badge, .preview)
    }

    func testBundledSampleDoesNotFabricateBundledMeshArtifacts() throws {
        let bundle = try WorldBundle.loadBundledSample()
        XCTAssertNil(bundle.bodyMesh)
        XCTAssertNotNil(bundle.contactWindows)
        for player in bundle.world.players {
            XCTAssertTrue(player.frames.allSatisfy { $0.meshVerticesWorld.isEmpty })
        }
    }

    func testMissingManifestResourceThrowsRatherThanFabricatingAWorld() {
        let emptyBundle = Bundle(for: WorldBundleTests.self)
        XCTAssertThrowsError(try WorldBundle.loadBundledSample(bundle: emptyBundle)) { error in
            XCTAssertEqual(error as? WorldBundleError, .missingManifestResource)
        }
    }
}
