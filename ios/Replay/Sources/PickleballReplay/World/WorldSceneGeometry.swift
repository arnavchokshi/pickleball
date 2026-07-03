import SceneKit
import simd

/// Pure SceneKit geometry/node factories used by `WorldSceneBuilder`. Split
/// out so the small amount of vector math (line building, cylinder
/// orientation) is easy to find and doesn't clutter the scene-graph
/// assembly logic.
enum WorldSceneGeometry {
    static func scnVector(_ v: WorldVec3) -> SCNVector3 {
        SCNVector3(Float(v.x), Float(v.y), Float(v.z))
    }

    /// One merged line-segment geometry from an arbitrary list of
    /// `(start, end)` pairs -- used for court lines, the net cable, and
    /// player floor trails. A single draw call rather than N nodes.
    static func lineGeometry(segments: [(WorldVec3, WorldVec3)]) -> SCNGeometry? {
        guard !segments.isEmpty else { return nil }
        var vertices: [SCNVector3] = []
        vertices.reserveCapacity(segments.count * 2)
        var indices: [Int32] = []
        indices.reserveCapacity(segments.count * 2)
        for (start, end) in segments {
            let base = Int32(vertices.count)
            vertices.append(scnVector(start))
            vertices.append(scnVector(end))
            indices.append(base)
            indices.append(base + 1)
        }
        let source = SCNGeometrySource(vertices: vertices)
        let element = SCNGeometryElement(indices: indices, primitiveType: .line)
        return SCNGeometry(sources: [source], elements: [element])
    }

    /// A polyline (connected strip) through `points`, e.g. a player's floor
    /// trail.
    static func polylineSegments(points: [WorldVec3]) -> [(WorldVec3, WorldVec3)] {
        guard points.count > 1 else { return [] }
        return zip(points, points.dropFirst()).map { ($0, $1) }
    }

    /// Solid indexed mesh geometry from `body_mesh.json` vertices + the
    /// artifact's shared face topology.
    static func meshGeometry(vertices: [WorldVec3], faces: [WorldMeshFace]) -> SCNGeometry? {
        guard !vertices.isEmpty, !faces.isEmpty else { return nil }
        let source = SCNGeometrySource(vertices: vertices.map(scnVector))
        var indices: [Int32] = []
        indices.reserveCapacity(faces.count * 3)
        for face in faces {
            indices.append(Int32(face.a))
            indices.append(Int32(face.b))
            indices.append(Int32(face.c))
        }
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
        let geometry = SCNGeometry(sources: [source], elements: [element])
        return geometry
    }

    /// A thin cylinder ("capsule" bone) spanning `start`..`end`, aligned via
    /// a from/to quaternion since `SCNCylinder`'s default axis is local Y.
    static func capsuleNode(from start: WorldVec3, to end: WorldVec3, radius: CGFloat, color: CGColor) -> SCNNode {
        let a = SIMD3<Float>(Float(start.x), Float(start.y), Float(start.z))
        let b = SIMD3<Float>(Float(end.x), Float(end.y), Float(end.z))
        let delta = b - a
        let length = max(0.001, simd_length(delta))
        let cylinder = SCNCylinder(radius: radius, height: CGFloat(length))
        cylinder.firstMaterial?.diffuse.contents = color
        cylinder.firstMaterial?.lightingModel = .constant
        let node = SCNNode(geometry: cylinder)
        node.simdPosition = (a + b) / 2
        node.simdOrientation = rotation(from: SIMD3<Float>(0, 1, 0), to: delta / length)
        return node
    }

    static func rotation(from: SIMD3<Float>, to: SIMD3<Float>) -> simd_quatf {
        let fromN = simd_normalize(from)
        let toN = simd_normalize(to)
        let dot = max(-1, min(1, simd_dot(fromN, toN)))
        if dot > 0.99999 { return simd_quatf(angle: 0, axis: SIMD3<Float>(1, 0, 0)) }
        if dot < -0.99999 {
            var axis = simd_cross(SIMD3<Float>(1, 0, 0), fromN)
            if simd_length(axis) < 1e-6 { axis = simd_cross(SIMD3<Float>(0, 0, 1), fromN) }
            return simd_quatf(angle: .pi, axis: simd_normalize(axis))
        }
        let axis = simd_normalize(simd_cross(fromN, toN))
        return simd_quatf(angle: acos(dot), axis: axis)
    }
}
