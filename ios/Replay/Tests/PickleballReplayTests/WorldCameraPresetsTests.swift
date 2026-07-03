import XCTest
@testable import PickleballReplay

final class WorldCameraPresetsTests: XCTestCase {
    private func sampleCourt() -> VirtualWorld.Court {
        VirtualWorld.Court(
            sport: "pickleball",
            coordinateFrame: "origin_net_center_x_width_y_length_z_up_m",
            lengthM: 13.41,
            widthM: 6.1,
            lineSegments: [
                "near_baseline": [WorldVec3(-3.05, 0, 0), WorldVec3(3.05, 0, 0)],
                "far_baseline": [WorldVec3(-3.05, 13.41, 0), WorldVec3(3.05, 13.41, 0)],
            ],
            net: VirtualWorld.Court.Net(endpoints: [WorldVec3(-3.05, 6.705, 0.91), WorldVec3(3.05, 6.705, 0.91)], centerHeightM: 0.86, postHeightM: 0.91),
            trustBand: nil
        )
    }

    func testCourtBoundsMatchRegulationDimensions() {
        let bounds = WorldCameraPlanner.courtBounds(for: sampleCourt())
        XCTAssertEqual(bounds.width, 6.1, accuracy: 1e-9)
        XCTAssertEqual(bounds.length, 13.41, accuracy: 1e-9)
        XCTAssertEqual(bounds.centerX, 0, accuracy: 1e-9)
        XCTAssertEqual(bounds.centerY, 13.41 / 2, accuracy: 1e-9)
    }

    func testEachPresetProducesADistinctCameraPose() {
        let court = sampleCourt()
        let poses = WorldCameraPreset.allCases.map { WorldCameraPlanner.pose(for: $0, court: court) }
        XCTAssertEqual(Set(poses.map { "\($0.position)" }).count, poses.count, "each preset should be a distinct camera position")
    }

    func testTopDownPresetLooksStraightDownAtCourtCenter() {
        let court = sampleCourt()
        let pose = WorldCameraPlanner.pose(for: .topDown, court: court)
        XCTAssertEqual(pose.position.x, pose.target.x, accuracy: 1e-9)
        XCTAssertEqual(pose.position.y, pose.target.y, accuracy: 1e-9)
        XCTAssertGreaterThan(pose.position.z, pose.target.z)
    }

    func testBroadcastAndBehindBaselineSitBehindTheNearBaseline() {
        let court = sampleCourt()
        let bounds = WorldCameraPlanner.courtBounds(for: court)
        for preset in [WorldCameraPreset.broadcast, .behindBaseline] {
            let pose = WorldCameraPlanner.pose(for: preset, court: court)
            XCTAssertLessThan(pose.position.y, bounds.minY, "\(preset) camera should sit behind the near baseline")
            XCTAssertGreaterThan(pose.position.z, 0)
        }
    }

    func testPresetDisplayNamesAreHumanReadable() {
        XCTAssertEqual(WorldCameraPreset.broadcast.displayName, "Broadcast")
        XCTAssertEqual(WorldCameraPreset.behindBaseline.displayName, "Behind Baseline")
        XCTAssertEqual(WorldCameraPreset.topDown.displayName, "Top Down")
    }
}
