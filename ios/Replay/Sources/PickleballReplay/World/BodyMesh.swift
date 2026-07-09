import Foundation

/// Swift mirror of `body_mesh.json` (`racketsport_body_mesh`, the
/// W3-REPLAY-NATIVE solid-mesh artifact). This is the MESH tier of the
/// GLUE-4 world viewer's data contract: full indexed SMPL-family meshes,
/// but only for the frames the pipeline actually computed a solid mesh for
/// -- in practice, contact windows, per `NORTH_STAR_ROADMAP.md`'s "mesh people at
/// contact windows only, joints otherwise" rule. Mirrors `BodyMesh` in
/// `web/replay/src/viewerData.ts`.
public struct BodyMesh: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var artifactType: String
    public var clip: String
    public var model: String
    public var fps: Double
    public var worldFrame: String
    public var meshFaces: [WorldMeshFace]
    public var players: [Player]
    public var summary: Summary

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifactType = "artifact_type"
        case clip, model, fps
        case worldFrame = "world_frame"
        case meshFaces = "mesh_faces"
        case players, summary
    }

    public init(
        schemaVersion: Int,
        artifactType: String,
        clip: String,
        model: String,
        fps: Double,
        worldFrame: String,
        meshFaces: [WorldMeshFace],
        players: [Player],
        summary: Summary
    ) {
        self.schemaVersion = schemaVersion
        self.artifactType = artifactType
        self.clip = clip
        self.model = model
        self.fps = fps
        self.worldFrame = worldFrame
        self.meshFaces = meshFaces
        self.players = players
        self.summary = summary
    }

    public struct Player: Codable, Equatable, Sendable {
        public var id: Int
        public var frames: [Frame]

        public init(id: Int, frames: [Frame]) {
            self.id = id
            self.frames = frames
        }
    }

    public struct Frame: Codable, Equatable, Sendable {
        public var frameIndex: Int
        public var t: Double
        public var blendWeight: Double
        public var meshVerticesWorld: [WorldVec3]
        public var jointsWorld: [WorldVec3]
        public var reasons: [String]

        private enum CodingKeys: String, CodingKey {
            case frameIndex = "frame_idx"
            case t
            case blendWeight = "blend_weight"
            case meshVerticesWorld = "mesh_vertices_world"
            case jointsWorld = "joints_world"
            case reasons
        }

        public init(frameIndex: Int, t: Double, blendWeight: Double, meshVerticesWorld: [WorldVec3], jointsWorld: [WorldVec3], reasons: [String]) {
            self.frameIndex = frameIndex
            self.t = t
            self.blendWeight = blendWeight
            self.meshVerticesWorld = meshVerticesWorld
            self.jointsWorld = jointsWorld
            self.reasons = reasons
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            frameIndex = try container.decode(Int.self, forKey: .frameIndex)
            t = try container.decode(Double.self, forKey: .t)
            blendWeight = try container.decodeIfPresent(Double.self, forKey: .blendWeight) ?? 1
            meshVerticesWorld = try container.decodeIfPresent([WorldVec3].self, forKey: .meshVerticesWorld) ?? []
            jointsWorld = try container.decodeIfPresent([WorldVec3].self, forKey: .jointsWorld) ?? []
            reasons = try container.decodeIfPresent([String].self, forKey: .reasons) ?? []
        }

        /// Solid mesh opacity from blend weight, mirroring
        /// `bodyMeshOpacityFromBlendWeight` in `web/replay/src/App.tsx`
        /// (`clamp(blend_weight, 0, 1) * 0.68`).
        public var meshOpacity: Double {
            max(0, min(1, blendWeight)) * 0.68
        }
    }

    public struct Summary: Codable, Equatable, Sendable {
        public var meshFrameCount: Int
        public var playerCount: Int
        public var contactWindowCount: Int

        private enum CodingKeys: String, CodingKey {
            case meshFrameCount = "mesh_frame_count"
            case playerCount = "player_count"
            case contactWindowCount = "contact_window_count"
        }

        public init(meshFrameCount: Int, playerCount: Int, contactWindowCount: Int) {
            self.meshFrameCount = meshFrameCount
            self.playerCount = playerCount
            self.contactWindowCount = contactWindowCount
        }
    }
}

extension BodyMesh {
    public static func load(from url: URL, decoder: JSONDecoder = JSONDecoder()) throws -> BodyMesh {
        let data = try Data(contentsOf: url)
        return try decoder.decode(BodyMesh.self, from: data)
    }

    /// Nearest body-mesh frame to `timeSeconds` for one player, or `nil` if
    /// outside the frame-spacing tolerance -- mirrors
    /// `bodyMeshFrameForTime` in `web/replay/src/viewerData.ts`.
    public func frame(forPlayer playerID: Int, at timeSeconds: Double) -> Frame? {
        guard let player = players.first(where: { $0.id == playerID }), !player.frames.isEmpty else { return nil }
        let sorted = player.frames.sorted { $0.t < $1.t }
        let tolerance = max(1 / max(fps, 1) * 1.5, 0.04)
        guard let first = sorted.first?.t, let last = sorted.last?.t else { return nil }
        guard timeSeconds >= first - tolerance, timeSeconds <= last + tolerance else { return nil }
        return sorted.min { abs($0.t - timeSeconds) < abs($1.t - timeSeconds) }
    }
}
