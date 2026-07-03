import Foundation

/// Swift mirror of `virtual_world.json` (`racketsport_virtual_world`,
/// `threed/racketsport/virtual_world.py` / `web/replay/src/viewerData.ts`
/// `VirtualWorld`). This is the joints/track tier of the GLUE-4 iOS world
/// viewer's data contract: full-clip player floor tracks plus BODY joint
/// coverage where it exists, court + net regulation geometry, ball path,
/// and one `TrustBand` per entity.
public struct VirtualWorld: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var artifactType: String
    public var worldFrame: String
    public var fps: Double
    public var court: Court
    public var players: [Player]
    public var ball: Ball
    public var paddles: [Paddle]
    public var summary: Summary

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifactType = "artifact_type"
        case worldFrame = "world_frame"
        case fps, court, players, ball, paddles, summary
    }

    public init(
        schemaVersion: Int,
        artifactType: String,
        worldFrame: String,
        fps: Double,
        court: Court,
        players: [Player],
        ball: Ball,
        paddles: [Paddle],
        summary: Summary
    ) {
        self.schemaVersion = schemaVersion
        self.artifactType = artifactType
        self.worldFrame = worldFrame
        self.fps = fps
        self.court = court
        self.players = players
        self.ball = ball
        self.paddles = paddles
        self.summary = summary
    }

    public struct Court: Codable, Equatable, Sendable {
        public var sport: String
        public var coordinateFrame: String
        public var lengthM: Double
        public var widthM: Double
        public var lineSegments: [String: [WorldVec3]]
        public var net: Net
        public var trustBand: TrustBand?

        private enum CodingKeys: String, CodingKey {
            case sport
            case coordinateFrame = "coordinate_frame"
            case lengthM = "length_m"
            case widthM = "width_m"
            case lineSegments = "line_segments"
            case net
            case trustBand = "trust_band"
        }

        public init(
            sport: String,
            coordinateFrame: String,
            lengthM: Double,
            widthM: Double,
            lineSegments: [String: [WorldVec3]],
            net: Net,
            trustBand: TrustBand?
        ) {
            self.sport = sport
            self.coordinateFrame = coordinateFrame
            self.lengthM = lengthM
            self.widthM = widthM
            self.lineSegments = lineSegments
            self.net = net
            self.trustBand = trustBand
        }

        public struct Net: Codable, Equatable, Sendable {
            public var endpoints: [WorldVec3]
            public var centerHeightM: Double
            public var postHeightM: Double

            private enum CodingKeys: String, CodingKey {
                case endpoints
                case centerHeightM = "center_height_m"
                case postHeightM = "post_height_m"
            }

            public init(endpoints: [WorldVec3], centerHeightM: Double, postHeightM: Double) {
                self.endpoints = endpoints
                self.centerHeightM = centerHeightM
                self.postHeightM = postHeightM
            }
        }
    }

    public struct Player: Codable, Equatable, Sendable {
        public var id: Int
        public var side: String?
        public var role: String?
        public var representation: Representation
        public var frames: [Frame]
        public var trustBand: TrustBand?

        private enum CodingKeys: String, CodingKey {
            case id, side, role, representation, frames
            case trustBand = "trust_band"
        }

        public init(id: Int, side: String?, role: String?, representation: Representation, frames: [Frame], trustBand: TrustBand?) {
            self.id = id
            self.side = side
            self.role = role
            self.representation = representation
            self.frames = frames
            self.trustBand = trustBand
        }

        public enum Representation: String, Codable, Equatable, Sendable {
            case trackOnly = "track_only"
            case joints
            case mesh
        }
    }

    public struct Frame: Codable, Equatable, Sendable {
        public var t: Double
        public var trackWorldXY: WorldVec2?
        public var jointsWorld: [WorldVec3]
        public var jointConf: [Double]
        public var meshVerticesWorld: [WorldVec3]
        public var floorWorldXYZ: WorldVec3?

        private enum CodingKeys: String, CodingKey {
            case t
            case trackWorldXY = "track_world_xy"
            case jointsWorld = "joints_world"
            case jointConf = "joint_conf"
            case meshVerticesWorld = "mesh_vertices_world"
            case floorWorldXYZ = "floor_world_xyz"
        }

        public init(
            t: Double,
            trackWorldXY: WorldVec2?,
            jointsWorld: [WorldVec3],
            jointConf: [Double],
            meshVerticesWorld: [WorldVec3],
            floorWorldXYZ: WorldVec3?
        ) {
            self.t = t
            self.trackWorldXY = trackWorldXY
            self.jointsWorld = jointsWorld
            self.jointConf = jointConf
            self.meshVerticesWorld = meshVerticesWorld
            self.floorWorldXYZ = floorWorldXYZ
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            t = try container.decode(Double.self, forKey: .t)
            trackWorldXY = try container.decodeIfPresent(WorldVec2.self, forKey: .trackWorldXY)
            jointsWorld = try container.decodeIfPresent([WorldVec3].self, forKey: .jointsWorld) ?? []
            jointConf = try container.decodeIfPresent([Double].self, forKey: .jointConf) ?? []
            meshVerticesWorld = try container.decodeIfPresent([WorldVec3].self, forKey: .meshVerticesWorld) ?? []
            floorWorldXYZ = try container.decodeIfPresent(WorldVec3.self, forKey: .floorWorldXYZ)
        }

        /// Best-available floor position: real BODY/TRK floor placement if
        /// present, else the raw court-plane track point at z=0. Mirrors
        /// `floorWorldForFrame` in `web/replay/src/replayScene.ts`/`App.tsx`.
        public var floorPosition: WorldVec3? {
            if let floorWorldXYZ { return floorWorldXYZ }
            if let trackWorldXY { return trackWorldXY.asFloorVec3 }
            return nil
        }
    }

    public struct Ball: Codable, Equatable, Sendable {
        public var source: String?
        public var frames: [Frame]
        public var trustBand: TrustBand?

        private enum CodingKeys: String, CodingKey {
            case source, frames
            case trustBand = "trust_band"
        }

        public init(source: String?, frames: [Frame], trustBand: TrustBand?) {
            self.source = source
            self.frames = frames
            self.trustBand = trustBand
        }

        public struct Frame: Codable, Equatable, Sendable {
            public var t: Double
            public var xy: WorldVec2
            public var conf: Double
            public var visible: Bool
            public var worldXYZ: WorldVec3?
            public var approx: Bool

            private enum CodingKeys: String, CodingKey {
                case t, xy, conf, visible
                case worldXYZ = "world_xyz"
                case approx
            }

            public init(t: Double, xy: WorldVec2, conf: Double, visible: Bool, worldXYZ: WorldVec3?, approx: Bool) {
                self.t = t
                self.xy = xy
                self.conf = conf
                self.visible = visible
                self.worldXYZ = worldXYZ
                self.approx = approx
            }

            public init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                t = try container.decode(Double.self, forKey: .t)
                xy = try container.decode(WorldVec2.self, forKey: .xy)
                conf = try container.decode(Double.self, forKey: .conf)
                visible = try container.decode(Bool.self, forKey: .visible)
                worldXYZ = try container.decodeIfPresent(WorldVec3.self, forKey: .worldXYZ)
                approx = try container.decodeIfPresent(Bool.self, forKey: .approx) ?? false
            }
        }
    }

    public struct Paddle: Codable, Equatable, Sendable {
        public var playerID: Int
        public var trustBand: TrustBand?

        private enum CodingKeys: String, CodingKey {
            case playerID = "player_id"
            case trustBand = "trust_band"
        }

        public init(playerID: Int, trustBand: TrustBand?) {
            self.playerID = playerID
            self.trustBand = trustBand
        }
    }

    public struct Summary: Codable, Equatable, Sendable {
        public var playerCount: Int
        public var meshPlayerFrameCount: Int
        public var jointPlayerFrameCount: Int
        public var trackOnlyPlayerFrameCount: Int
        public var ballFrameCount: Int
        public var approxBallFrameCount: Int

        private enum CodingKeys: String, CodingKey {
            case playerCount = "player_count"
            case meshPlayerFrameCount = "mesh_player_frame_count"
            case jointPlayerFrameCount = "joint_player_frame_count"
            case trackOnlyPlayerFrameCount = "track_only_player_frame_count"
            case ballFrameCount = "ball_frame_count"
            case approxBallFrameCount = "approx_ball_frame_count"
        }

        public init(
            playerCount: Int,
            meshPlayerFrameCount: Int,
            jointPlayerFrameCount: Int,
            trackOnlyPlayerFrameCount: Int,
            ballFrameCount: Int,
            approxBallFrameCount: Int
        ) {
            self.playerCount = playerCount
            self.meshPlayerFrameCount = meshPlayerFrameCount
            self.jointPlayerFrameCount = jointPlayerFrameCount
            self.trackOnlyPlayerFrameCount = trackOnlyPlayerFrameCount
            self.ballFrameCount = ballFrameCount
            self.approxBallFrameCount = approxBallFrameCount
        }
    }
}

extension VirtualWorld {
    public static func load(from url: URL, decoder: JSONDecoder = JSONDecoder()) throws -> VirtualWorld {
        let data = try Data(contentsOf: url)
        return try decoder.decode(VirtualWorld.self, from: data)
    }
}
