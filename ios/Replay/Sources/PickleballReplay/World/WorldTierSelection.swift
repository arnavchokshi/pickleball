import Foundation

/// The locked tier rule (`NORTH_STAR_ROADMAP.md`, owner directive): mesh people at
/// contact windows only, joints otherwise, and a bare floor dot/path when
/// neither is available. Exactly one tier is chosen per player per frame --
/// never blended, never a guess.
public enum PlayerRenderTier: Equatable, Sendable {
    case mesh(BodyMesh.Frame)
    case joints(VirtualWorld.Frame)
    case trackOnly(VirtualWorld.Frame)
    case none
}

public enum WorldFrameSelection {
    /// Nearest frame to `timeSeconds` for one player, or `nil` outside the
    /// player's covered time range -- mirrors `frameForTime` in
    /// `web/replay/src/viewerData.ts`.
    public static func frame(for player: VirtualWorld.Player, at timeSeconds: Double, fps: Double) -> VirtualWorld.Frame? {
        guard !player.frames.isEmpty else { return nil }
        let times = player.frames.map(\.t).sorted()
        guard let first = times.first, let last = times.last else { return nil }
        let positiveGaps = zip(times, times.dropFirst()).map { $1 - $0 }.filter { $0 > 0 }
        let tolerance = positiveGaps.isEmpty ? (1 / max(fps, 1)) : positiveGaps.min()! * 1.5
        guard timeSeconds >= first - tolerance, timeSeconds <= last + tolerance else { return nil }
        return player.frames.min { abs($0.t - timeSeconds) < abs($1.t - timeSeconds) }
    }

    /// Nearest visible ball frame with a resolved 3D position, mirroring
    /// `ballFrameForTime` in `web/replay/src/viewerData.ts`.
    public static func ballFrame(in world: VirtualWorld, at timeSeconds: Double) -> VirtualWorld.Ball.Frame? {
        let candidates = world.ball.frames.filter { $0.visible && $0.worldXYZ != nil }
        guard !candidates.isEmpty else { return nil }
        let times = candidates.map(\.t).sorted()
        guard let first = times.first, let last = times.last else { return nil }
        let positiveGaps = zip(times, times.dropFirst()).map { $1 - $0 }.filter { $0 > 0 }
        let tolerance = positiveGaps.isEmpty ? (1.0 / 30.0) : positiveGaps.min()! * 1.5
        guard timeSeconds >= first - tolerance, timeSeconds <= last + tolerance else { return nil }
        return candidates.min { abs($0.t - timeSeconds) < abs($1.t - timeSeconds) }
    }

    /// Whether `playerID` has an active reviewed contact event at
    /// `timeSeconds` per `contact_windows.json`. `nil` contact windows (no
    /// artifact for this clip) never gate MESH on -- mirrors
    /// `solidBodyMeshFramesForTime`'s "no contactWindows artifact at all"
    /// case in `web/replay/src/viewerData.ts`, which still shows mesh if a
    /// body-mesh frame is close enough. Here we're explicit: pass `nil`
    /// contactWindows to allow MESH purely from body-mesh frame proximity.
    private static func hasActiveContact(contactWindows: ContactWindows?, playerID: Int, at timeSeconds: Double) -> Bool {
        guard let contactWindows else { return true }
        let active = contactWindows.activeContactEvents(at: timeSeconds)
        if active.isEmpty { return contactWindows.events.isEmpty }
        return active.contains { $0.playerID == nil || $0.playerID == playerID }
    }

    /// Choose the one render tier for `player` at `timeSeconds`, applying
    /// the locked rule: MESH only when both a nearby `body_mesh.json` frame
    /// exists AND (if a contact-windows artifact exists) a contact window
    /// for this player is active right now; otherwise JOINTS if BODY joint
    /// data covers this time; otherwise a bare floor TRACK point; otherwise
    /// nothing renders for this player at this time.
    public static func tier(
        for player: VirtualWorld.Player,
        at timeSeconds: Double,
        world: VirtualWorld,
        bodyMesh: BodyMesh?,
        contactWindows: ContactWindows?
    ) -> PlayerRenderTier {
        if let bodyMesh, let meshFrame = bodyMesh.frame(forPlayer: player.id, at: timeSeconds),
           hasActiveContact(contactWindows: contactWindows, playerID: player.id, at: timeSeconds) {
            return .mesh(meshFrame)
        }
        guard let frame = frame(for: player, at: timeSeconds, fps: world.fps) else { return .none }
        if !frame.jointsWorld.isEmpty {
            return .joints(frame)
        }
        if frame.floorPosition != nil {
            return .trackOnly(frame)
        }
        return .none
    }

    /// Count of players with any renderable tier at `timeSeconds` -- backs
    /// the on-screen "players visible" count, independent of the
    /// low-confidence dim toggle (dimming is a rendering style, not a
    /// visibility gate).
    public static func visiblePlayerCount(
        world: VirtualWorld,
        bodyMesh: BodyMesh?,
        contactWindows: ContactWindows?,
        at timeSeconds: Double
    ) -> Int {
        world.players.reduce(into: 0) { count, player in
            let tier = tier(for: player, at: timeSeconds, world: world, bodyMesh: bodyMesh, contactWindows: contactWindows)
            if tier != .none { count += 1 }
        }
    }
}
