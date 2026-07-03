import Foundation
@testable import PickleballReplay

/// Small synthetic `VirtualWorld` builders for tests that need real-shaped
/// (but not real-run) data -- e.g. ball 3D positions, which the currently
/// bundled fixture legitimately never has (BALL is 0/8 gates).
enum SyntheticWorld {
    static func samplePickleballCourt() -> VirtualWorld.Court {
        VirtualWorld.Court(
            sport: "pickleball",
            coordinateFrame: "origin_net_center_x_width_y_length_z_up_m",
            lengthM: 13.41,
            widthM: 6.1,
            lineSegments: [
                "near_baseline": [WorldVec3(-3.05, 0, 0), WorldVec3(3.05, 0, 0)],
                "far_baseline": [WorldVec3(-3.05, 13.41, 0), WorldVec3(3.05, 13.41, 0)],
                "left_sideline": [WorldVec3(-3.05, 0, 0), WorldVec3(-3.05, 13.41, 0)],
                "right_sideline": [WorldVec3(3.05, 0, 0), WorldVec3(3.05, 13.41, 0)],
            ],
            net: VirtualWorld.Court.Net(endpoints: [WorldVec3(-3.05, 6.705, 0.91), WorldVec3(3.05, 6.705, 0.91)], centerHeightM: 0.86, postHeightM: 0.91),
            trustBand: nil
        )
    }

    static func withBall(
        court: VirtualWorld.Court = samplePickleballCourt(),
        players: [VirtualWorld.Player] = [],
        frames: [(t: Double, xy: WorldVec2, worldXYZ: WorldVec3?, visible: Bool, approx: Bool)]
    ) -> VirtualWorld {
        let ballFrames = frames.map { entry in
            VirtualWorld.Ball.Frame(t: entry.t, xy: entry.xy, conf: 1, visible: entry.visible, worldXYZ: entry.worldXYZ, approx: entry.approx)
        }
        return VirtualWorld(
            schemaVersion: 1,
            artifactType: "racketsport_virtual_world",
            worldFrame: "court_Z0",
            fps: 30,
            court: court,
            players: players,
            ball: VirtualWorld.Ball(source: "tracknet", frames: ballFrames, trustBand: nil),
            paddles: [],
            summary: VirtualWorld.Summary(
                playerCount: players.count,
                meshPlayerFrameCount: 0,
                jointPlayerFrameCount: 0,
                trackOnlyPlayerFrameCount: 0,
                ballFrameCount: ballFrames.count,
                approxBallFrameCount: ballFrames.filter(\.approx).count
            )
        )
    }
}
