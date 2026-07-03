import Foundation

/// The bundled `body_mesh.json`/`virtual_world.json` joint arrays carry
/// generic `sam3dbody_joint_NNN` names (see `BodyMesh.joint_names`), not a
/// documented kinematic tree/parent index, so there is no honest way to
/// hardcode "this index is the elbow" bone pairs. Instead this builds a
/// minimum-spanning-tree over the joint point cloud each frame: nearest
/// real 3D distances only, no invented anatomy, and it degrades gracefully
/// (still one connected skeleton) for any joint ordering.
public enum WorldSkeletonBones {
    /// Prim's algorithm over `joints` distances. Returns `joints.count - 1`
    /// edges (or fewer if `joints` has 0-1 elements), each `(a, b)` an index
    /// pair with `a < b`.
    public static func minimumSpanningTree(joints: [WorldVec3]) -> [(Int, Int)] {
        guard joints.count > 1 else { return [] }
        var inTree = [Bool](repeating: false, count: joints.count)
        var bestDistance = [Double](repeating: .infinity, count: joints.count)
        var bestParent = [Int?](repeating: nil, count: joints.count)
        bestDistance[0] = 0
        var edges: [(Int, Int)] = []
        edges.reserveCapacity(joints.count - 1)

        for _ in 0..<joints.count {
            guard let next = nextVertex(inTree: inTree, bestDistance: bestDistance) else { break }
            inTree[next] = true
            if let parent = bestParent[next] {
                edges.append(parent < next ? (parent, next) : (next, parent))
            }
            for candidate in 0..<joints.count where !inTree[candidate] {
                let distance = joints[next].distance(to: joints[candidate])
                if distance < bestDistance[candidate] {
                    bestDistance[candidate] = distance
                    bestParent[candidate] = next
                }
            }
        }
        return edges
    }

    private static func nextVertex(inTree: [Bool], bestDistance: [Double]) -> Int? {
        var chosen: Int?
        var chosenDistance = Double.infinity
        for index in bestDistance.indices where !inTree[index] && bestDistance[index] < chosenDistance {
            chosen = index
            chosenDistance = bestDistance[index]
        }
        return chosen
    }
}
