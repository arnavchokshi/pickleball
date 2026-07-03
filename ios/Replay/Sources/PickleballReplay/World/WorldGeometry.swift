import Foundation

/// A world-frame point, meters, `court_Z0` convention
/// (origin at net center, +X = width, +Y = length, +Z = up). Mirrors the
/// `Vec3` tuple type used by the web viewer's `viewerData.ts`, decoded from
/// the same `[x, y, z]` JSON arrays.
public struct WorldVec3: Codable, Equatable, Sendable {
    public var x: Double
    public var y: Double
    public var z: Double

    public init(_ x: Double, _ y: Double, _ z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }

    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        guard container.count == 3 else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "WorldVec3 requires exactly 3 elements")
            )
        }
        x = try container.decode(Double.self)
        y = try container.decode(Double.self)
        z = try container.decode(Double.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(x)
        try container.encode(y)
        try container.encode(z)
    }

    public func distance(to other: WorldVec3) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        let dz = z - other.z
        return (dx * dx + dy * dy + dz * dz).squareRoot()
    }

    public func floorDistance(to other: WorldVec3) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        return (dx * dx + dy * dy).squareRoot()
    }
}

/// A court-plane point, meters (`[x, y]`).
public struct WorldVec2: Codable, Equatable, Sendable {
    public var x: Double
    public var y: Double

    public init(_ x: Double, _ y: Double) {
        self.x = x
        self.y = y
    }

    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        guard container.count == 2 else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "WorldVec2 requires exactly 2 elements")
            )
        }
        x = try container.decode(Double.self)
        y = try container.decode(Double.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(x)
        try container.encode(y)
    }

    public var asFloorVec3: WorldVec3 { WorldVec3(x, y, 0) }
}

/// A triangle face, vertex indices into a `mesh_vertices_world` array.
public struct WorldMeshFace: Codable, Equatable, Sendable {
    public var a: Int
    public var b: Int
    public var c: Int

    public init(_ a: Int, _ b: Int, _ c: Int) {
        self.a = a
        self.b = b
        self.c = c
    }

    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        guard container.count == 3 else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "WorldMeshFace requires exactly 3 elements")
            )
        }
        a = try container.decode(Int.self)
        b = try container.decode(Int.self)
        c = try container.decode(Int.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(a)
        try container.encode(b)
        try container.encode(c)
    }
}
