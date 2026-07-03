import Foundation

/// Everything needed to render one instant of a `WorldBundle`: the chosen
/// tier per player (locked mesh/joints/track rule), the ball's render mode,
/// and every entity's trust badge. Pure data, computed once per scrub tick
/// so the SceneKit layer only has to draw it -- no gate logic lives in the
/// rendering code.
public struct WorldFrameSnapshot: Equatable, Sendable {
    public var timeSeconds: Double
    public var players: [PlayerSnapshot]
    public var ball: WorldBallRenderInfo
    public var ballTrustBadge: TrustBadge
    public var courtTrustBadge: TrustBadge
    public var visiblePlayerCount: Int

    public struct PlayerSnapshot: Equatable, Sendable {
        public var id: Int
        public var tier: PlayerRenderTier
        public var trustBadge: TrustBadge
        public var trustBand: TrustBand?
        public var floorPosition: WorldVec3?
        public var floorTrail: [WorldVec3]
    }
}

public enum WorldFrameSnapshotBuilder {
    public static func build(bundle: WorldBundle, at timeSeconds: Double) -> WorldFrameSnapshot {
        let players = bundle.world.players.map { player -> WorldFrameSnapshot.PlayerSnapshot in
            let tier = WorldFrameSelection.tier(
                for: player,
                at: timeSeconds,
                world: bundle.world,
                bodyMesh: bundle.bodyMesh,
                contactWindows: bundle.contactWindows
            )
            let floor = WorldFrameSelection.frame(for: player, at: timeSeconds, fps: bundle.world.fps)?.floorPosition
            let trail = player.frames.filter { $0.t <= timeSeconds }.suffix(90).compactMap(\.floorPosition)
            return WorldFrameSnapshot.PlayerSnapshot(
                id: player.id,
                tier: tier,
                trustBadge: TrustBandPresentation.badge(for: player.trustBand),
                trustBand: player.trustBand,
                floorPosition: floor,
                floorTrail: Array(trail)
            )
        }
        let ballInfo = WorldBallRendering.renderInfo(world: bundle.world, at: timeSeconds)
        return WorldFrameSnapshot(
            timeSeconds: timeSeconds,
            players: players,
            ball: ballInfo,
            ballTrustBadge: TrustBandPresentation.badge(for: bundle.world.ball.trustBand),
            courtTrustBadge: TrustBandPresentation.badge(for: bundle.world.court.trustBand),
            visiblePlayerCount: players.filter { $0.tier != .none }.count
        )
    }
}
