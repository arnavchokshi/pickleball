import Foundation

/// Mirrors `ballRenderInfoForTime` in `web/replay/src/viewerData.ts`: the
/// ball only ever renders in 3D when a real `world_xyz` exists, and even
/// then callers must know whether it's a real calibrated 3D point or a
/// flattened court-plane approximation, versus genuinely missing/off-court.
/// For the bundled Burlington fixture every ball frame currently has
/// `world_xyz == nil` (BALL has 0/8 M1-M8 gates per `NORTH_STAR_ROADMAP.md`, and
/// this particular run never ran ball/court projection), so `.missing` is
/// the honest, expected mode there -- this type exists so that state is
/// surfaced, not silently hidden.
public enum WorldBallRenderMode: Equatable, Sendable {
    case missing
    case calibrated3D
    case courtPlaneProjection
    case offCourtProjection

    public var readoutText: String {
        switch self {
        case .calibrated3D: return "ball: calibrated 3D"
        case .courtPlaneProjection: return "ball: court-plane approx"
        case .offCourtProjection: return "ball: off-court hidden"
        case .missing: return "ball: missing"
        }
    }
}

public struct WorldBallRenderInfo: Equatable, Sendable {
    public var frame: VirtualWorld.Ball.Frame?
    public var mode: WorldBallRenderMode
    public var shouldRender3D: Bool
}

public enum WorldBallRendering {
    public static func renderInfo(world: VirtualWorld, at timeSeconds: Double, courtMarginM: Double = 0.35) -> WorldBallRenderInfo {
        guard let frame = WorldFrameSelection.ballFrame(in: world, at: timeSeconds), let position = frame.worldXYZ else {
            return WorldBallRenderInfo(frame: nil, mode: .missing, shouldRender3D: false)
        }
        if !frame.approx {
            return WorldBallRenderInfo(frame: frame, mode: .calibrated3D, shouldRender3D: true)
        }
        if !isInsideCourt(world: world, point: position, marginM: courtMarginM) {
            return WorldBallRenderInfo(frame: frame, mode: .offCourtProjection, shouldRender3D: false)
        }
        return WorldBallRenderInfo(frame: frame, mode: .courtPlaneProjection, shouldRender3D: true)
    }

    private static func isInsideCourt(world: VirtualWorld, point: WorldVec3, marginM: Double) -> Bool {
        let points = world.court.lineSegments.values.flatMap { $0 }
        let xs = points.map(\.x) + [-world.court.widthM / 2, world.court.widthM / 2]
        let ys = points.map(\.y) + [0, world.court.lengthM]
        let minX = (xs.min() ?? -world.court.widthM / 2) - marginM
        let maxX = (xs.max() ?? world.court.widthM / 2) + marginM
        let minY = (ys.min() ?? 0) - marginM
        let maxY = (ys.max() ?? world.court.lengthM) + marginM
        return point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY
    }
}
