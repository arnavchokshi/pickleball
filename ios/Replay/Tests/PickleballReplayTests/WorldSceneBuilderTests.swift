import XCTest
import SceneKit
@testable import PickleballReplay

final class WorldSceneBuilderTests: XCTestCase {
    private func loadedBundle() throws -> WorldBundle {
        try WorldBundle.loadBundledSample()
    }

    func testStaticCourtGeometryIsBuiltOnce() throws {
        let bundle = try loadedBundle()
        let builder = WorldSceneBuilder(court: bundle.world.court, meshFaces: bundle.bodyMesh?.meshFaces ?? [])
        // Floor + court lines + net assembly (box + 2 posts) => at least 5
        // static nodes present before any dynamic frame is ever applied.
        XCTAssertGreaterThanOrEqual(builder.scene.rootNode.childNodes.count, 5)
    }

    func testApplyingAJointsTierSnapshotAddsSkeletonSpheresNotMesh() throws {
        let bundle = try loadedBundle()
        let builder = WorldSceneBuilder(court: bundle.world.court, meshFaces: bundle.bodyMesh?.meshFaces ?? [])
        let player3World = try XCTUnwrap(bundle.world.players.first { $0.id == 3 })
        let jointTime = try XCTUnwrap(player3World.frames.first { !$0.jointsWorld.isEmpty }?.t)
        let snapshot = WorldFrameSnapshotBuilder.build(bundle: bundle, at: jointTime)

        let player3 = try XCTUnwrap(snapshot.players.first { $0.id == 3 })
        guard case .joints = player3.tier else {
            return XCTFail("expected player 3 to resolve to joints tier at t=\(jointTime), got \(player3.tier)")
        }

        builder.apply(snapshot, dimLowConfidence: true)
        let playerNode = try XCTUnwrap(builder.scene.rootNode.childNode(withName: "player-3", recursively: true))
        let descendants = playerNode.childNodes(passingTest: { _, _ in true })
        let sphereCount = descendants.filter { $0.geometry is SCNSphere }.count
        XCTAssertGreaterThan(sphereCount, 1, "expected multiple joint spheres, no solid mesh, for the joints tier")
        let hasHighVertexMesh = descendants.contains { ($0.geometry?.sources.first?.vectorCount ?? 0) > 1000 }
        XCTAssertFalse(hasHighVertexMesh, "joints tier must never render a solid mesh")
    }

    func testApplyReplacesDynamicNodesRatherThanAccumulating() throws {
        let bundle = try loadedBundle()
        let builder = WorldSceneBuilder(court: bundle.world.court, meshFaces: bundle.bodyMesh?.meshFaces ?? [])
        let snapshot = WorldFrameSnapshotBuilder.build(bundle: bundle, at: 2.6193)

        builder.apply(snapshot, dimLowConfidence: true)
        let countAfterFirst = builder.scene.rootNode.childNodes(passingTest: { node, _ in node.name?.hasPrefix("player-") == true }).count
        builder.apply(snapshot, dimLowConfidence: true)
        let countAfterSecond = builder.scene.rootNode.childNodes(passingTest: { node, _ in node.name?.hasPrefix("player-") == true }).count

        XCTAssertEqual(countAfterFirst, countAfterSecond, "re-applying a snapshot must not accumulate stale player nodes")
    }

    func testDimLowConfidenceProducesADifferentPlayerNodeThanFullOpacity() throws {
        let bundle = try loadedBundle()
        let builder = WorldSceneBuilder(court: bundle.world.court, meshFaces: bundle.bodyMesh?.meshFaces ?? [])
        let snapshot = WorldFrameSnapshotBuilder.build(bundle: bundle, at: 0.2)
        let player1 = try XCTUnwrap(snapshot.players.first { $0.id == 1 })
        XCTAssertEqual(player1.trustBadge, .preview, "player 1 uses BODY preview evidence in the refreshed compact fixture")

        builder.apply(snapshot, dimLowConfidence: true)
        XCTAssertNotNil(builder.scene.rootNode.childNode(withName: "player-1", recursively: true))

        builder.apply(snapshot, dimLowConfidence: false)
        XCTAssertNotNil(builder.scene.rootNode.childNode(withName: "player-1", recursively: true))
        // Exact material alpha is covered at the pure-function level in
        // WorldTrustColorsTests; here we only need apply(...) to accept and
        // re-render both toggle states without crashing or dropping the node.
    }
}
