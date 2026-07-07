import Foundation

/// Fixed starting camera poses a viewer can jump to before continuing to
/// orbit/pan/zoom freely -- mirrors `CameraPreset` /
/// `cameraPresetPose` in `web/replay/src/App.tsx`.
public enum WorldCameraPreset: String, CaseIterable, Equatable, Sendable {
    case broadcast
    case behindBaseline
    case topDown
    case ballFollow

    public var displayName: String {
        switch self {
        case .broadcast: return "Broadcast"
        case .behindBaseline: return "Behind"
        case .topDown: return "Top"
        case .ballFollow: return "Ball-follow"
        }
    }
}

public struct WorldCameraPose: Equatable, Sendable {
    public var position: WorldVec3
    public var target: WorldVec3

    public init(position: WorldVec3, target: WorldVec3) {
        self.position = position
        self.target = target
    }
}

public struct WorldCourtBounds: Equatable, Sendable {
    public var centerX: Double
    public var centerY: Double
    public var minY: Double
    public var maxY: Double
    public var width: Double
    public var length: Double

    public init(centerX: Double, centerY: Double, minY: Double, maxY: Double, width: Double, length: Double) {
        self.centerX = centerX
        self.centerY = centerY
        self.minY = minY
        self.maxY = maxY
        self.width = width
        self.length = length
    }
}

public enum WorldCameraPlanner {
    /// Court bounding box in world meters, mirroring `courtBounds` in
    /// `web/replay/src/App.tsx`: the union of every line-segment endpoint
    /// and the regulation half-width/full-length, so a slightly
    /// mis-calibrated court never clips a preset camera.
    public static func courtBounds(for court: VirtualWorld.Court) -> WorldCourtBounds {
        let points = court.lineSegments.values.flatMap { $0 }
        let xs = points.map(\.x) + [-court.widthM / 2, court.widthM / 2]
        let ys = points.map(\.y) + [0, court.lengthM]
        let minX = xs.min() ?? -court.widthM / 2
        let maxX = xs.max() ?? court.widthM / 2
        let minY = ys.min() ?? 0
        let maxY = ys.max() ?? court.lengthM
        return WorldCourtBounds(
            centerX: (minX + maxX) / 2,
            centerY: (minY + maxY) / 2,
            minY: minY,
            maxY: maxY,
            width: max(1, maxX - minX),
            length: max(1, maxY - minY)
        )
    }

    /// Camera position/target for one preset, mirroring
    /// `cameraPresetPose` in `web/replay/src/App.tsx` (Z-up world frame).
    public static func pose(for preset: WorldCameraPreset, court: VirtualWorld.Court) -> WorldCameraPose {
        let bounds = courtBounds(for: court)
        let groundTarget = WorldVec3(bounds.centerX, bounds.centerY, 0.35)
        switch preset {
        case .topDown:
            return WorldCameraPose(
                position: WorldVec3(bounds.centerX, bounds.centerY, max(10, bounds.length * 1.1)),
                target: WorldVec3(bounds.centerX, bounds.centerY, 0)
            )
        case .ballFollow:
            return WorldCameraPose(
                position: WorldVec3(bounds.centerX, bounds.minY - bounds.length * 0.24, 1.9),
                target: groundTarget
            )
        case .behindBaseline:
            return WorldCameraPose(
                position: WorldVec3(bounds.centerX, bounds.minY - bounds.length * 0.32, 2.1),
                target: groundTarget
            )
        case .broadcast:
            return WorldCameraPose(
                position: WorldVec3(bounds.centerX, bounds.minY - bounds.length * 0.86, max(6.5, bounds.length * 0.64)),
                target: groundTarget
            )
        }
    }
}
