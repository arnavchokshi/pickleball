import XCTest
@testable import PickleballReplay

final class WorldTierSelectionTests: XCTestCase {
    private func loadedBundle() throws -> WorldBundle {
        try WorldBundle.loadBundledSample()
    }

    func testCompactVerifiedFixtureRendersJointsWithoutBundledMeshArtifacts() throws {
        let bundle = try loadedBundle()
        let player3 = try XCTUnwrap(bundle.world.players.first { $0.id == 3 })
        let jointTime = try XCTUnwrap(player3.frames.first { !$0.jointsWorld.isEmpty }?.t)

        let tier = WorldFrameSelection.tier(
            for: player3,
            at: jointTime,
            world: bundle.world,
            bodyMesh: bundle.bodyMesh,
            contactWindows: bundle.contactWindows
        )

        guard case .joints(let frame) = tier else {
            return XCTFail("expected .joints tier, got \(tier)")
        }
        XCTAssertFalse(frame.jointsWorld.isEmpty)
        XCTAssertTrue(frame.meshVerticesWorld.isEmpty)
    }

    func testNoMeshTierIsFabricatedWhenBodyMeshIsAbsent() throws {
        let bundle = try loadedBundle()
        XCTAssertNil(bundle.bodyMesh)
        for player in bundle.world.players {
            let tier = WorldFrameSelection.tier(
                for: player,
                at: 2.6193,
                world: bundle.world,
                bodyMesh: bundle.bodyMesh,
                contactWindows: bundle.contactWindows
            )
            if case .mesh = tier {
                XCTFail("skeleton-level fixture must not render mesh for player \(player.id)")
            }
        }
    }

    func testOutOfPlayerCoverageYieldsNone() throws {
        let bundle = try loadedBundle()
        let player4 = try XCTUnwrap(bundle.world.players.first { $0.id == 4 })
        let tier = WorldFrameSelection.tier(
            for: player4,
            at: 0.2,
            world: bundle.world,
            bodyMesh: bundle.bodyMesh,
            contactWindows: bundle.contactWindows
        )
        XCTAssertEqual(tier, .none)
    }

    func testBundledPlayersNeverGetMeshWhenBodyMeshIsOmitted() throws {
        let bundle = try loadedBundle()

        for player in bundle.world.players {
            for frame in player.frames {
                let tier = WorldFrameSelection.tier(
                    for: player,
                    at: frame.t,
                    world: bundle.world,
                    bodyMesh: bundle.bodyMesh,
                    contactWindows: bundle.contactWindows
                )
                if case .mesh = tier {
                    XCTFail("compact fixture player \(player.id) must not render mesh at t=\(frame.t) without body_mesh.json")
                }
            }
        }
    }

    func testOutOfRangeTimeYieldsNoneRatherThanStaleFrame() throws {
        let bundle = try loadedBundle()
        let player1 = try XCTUnwrap(bundle.world.players.first { $0.id == 1 })
        let tier = WorldFrameSelection.tier(
            for: player1,
            at: 10_000,
            world: bundle.world,
            bodyMesh: bundle.bodyMesh,
            contactWindows: bundle.contactWindows
        )
        XCTAssertEqual(tier, .none)
    }

    func testVisiblePlayerCountMatchesManualTierScan() throws {
        let bundle = try loadedBundle()
        let time = 1.4014
        let expected = bundle.world.players.filter {
            WorldFrameSelection.tier(for: $0, at: time, world: bundle.world, bodyMesh: bundle.bodyMesh, contactWindows: bundle.contactWindows) != .none
        }.count
        let actual = WorldFrameSelection.visiblePlayerCount(world: bundle.world, bodyMesh: bundle.bodyMesh, contactWindows: bundle.contactWindows, at: time)
        XCTAssertEqual(actual, expected)
        XCTAssertGreaterThan(actual, 0)
    }

    /// Real, honest state of this bundle: BALL has 0/8 M1-M8 gates
    /// (`MASTER_PLAN.md`) and this run never projected ball pixel tracks
    /// into 3D, so every ball frame has `world_xyz == nil`. The tier
    /// selector must report that as "no renderable ball frame" rather than
    /// inventing a 3D position -- exercised against synthetic data below to
    /// prove the nearest-frame logic itself still works once real 3D ball
    /// data exists.
    func testBundledBallHasNoRenderable3DFrame() throws {
        let bundle = try loadedBundle()
        XCTAssertGreaterThan(bundle.world.ball.frames.count, 0)
        XCTAssertNil(WorldFrameSelection.ballFrame(in: bundle.world, at: bundle.world.ball.frames.first!.t))
        let info = WorldBallRendering.renderInfo(world: bundle.world, at: bundle.world.ball.frames.first!.t)
        XCTAssertEqual(info.mode, .missing)
        XCTAssertFalse(info.shouldRender3D)
    }

    func testBallFrameForTimeFindsNearestSyntheticVisibleFrame() throws {
        let world = SyntheticWorld.withBall(frames: [
            (t: 0.0, xy: WorldVec2(0, 0), worldXYZ: WorldVec3(0, 1, 0.1), visible: true, approx: false),
            (t: 0.5, xy: WorldVec2(0, 0), worldXYZ: WorldVec3(0, 2, 0.2), visible: true, approx: false),
            (t: 1.0, xy: WorldVec2(0, 0), worldXYZ: nil, visible: false, approx: false),
        ])
        let found = WorldFrameSelection.ballFrame(in: world, at: 0.4)
        XCTAssertEqual(found?.t, 0.5)

        let farOutside = WorldFrameSelection.ballFrame(in: world, at: 50.0)
        XCTAssertNil(farOutside, "times far outside the visible-frame coverage range must not resolve to a stale frame")
    }

    func testBallRenderModeDistinguishesCalibratedApproxAndOffCourt() throws {
        let court = SyntheticWorld.samplePickleballCourt()
        let world = SyntheticWorld.withBall(
            court: court,
            frames: [
                (t: 0.0, xy: WorldVec2(0, 0), worldXYZ: WorldVec3(0, 6, 1.0), visible: true, approx: false),
                (t: 1.0, xy: WorldVec2(0, 0), worldXYZ: WorldVec3(0, 6, 0.2), visible: true, approx: true),
                (t: 2.0, xy: WorldVec2(0, 0), worldXYZ: WorldVec3(500, 500, 0), visible: true, approx: true),
            ]
        )
        XCTAssertEqual(WorldBallRendering.renderInfo(world: world, at: 0.0).mode, .calibrated3D)
        XCTAssertEqual(WorldBallRendering.renderInfo(world: world, at: 1.0).mode, .courtPlaneProjection)
        XCTAssertEqual(WorldBallRendering.renderInfo(world: world, at: 2.0).mode, .offCourtProjection)
    }
}
