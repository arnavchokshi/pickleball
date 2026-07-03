import XCTest
@testable import PickleballReplay

final class WorldSkeletonBonesTests: XCTestCase {
    func testEmptyAndSingleJointProduceNoEdges() {
        XCTAssertEqual(WorldSkeletonBones.minimumSpanningTree(joints: []).count, 0)
        XCTAssertEqual(WorldSkeletonBones.minimumSpanningTree(joints: [WorldVec3(0, 0, 0)]).count, 0)
    }

    func testChainOfPointsProducesNMinusOneEdgesAndStaysConnected() {
        let joints = (0..<8).map { WorldVec3(Double($0), 0, 0) }
        let edges = WorldSkeletonBones.minimumSpanningTree(joints: joints)
        XCTAssertEqual(edges.count, joints.count - 1)
        assertConnected(edges: edges, vertexCount: joints.count)
    }

    func testPicksNearestNeighborOverFartherOne() {
        // 0 at origin, 1 very close to 0, 2 far away: MST must connect 2
        // through 1 (nearest available), not directly to 0.
        let joints = [WorldVec3(0, 0, 0), WorldVec3(0.01, 0, 0), WorldVec3(5, 0, 0)]
        let edges = WorldSkeletonBones.minimumSpanningTree(joints: joints)
        XCTAssertEqual(edges.count, 2)
        assertConnected(edges: edges, vertexCount: joints.count)
        XCTAssertTrue(edges.contains { $0 == (0, 1) })
    }

    func testStarShapedJointsStayFullyConnectedWithNoCycles() {
        let center = WorldVec3(0, 0, 1)
        let joints = [center] + (0..<12).map { index -> WorldVec3 in
            let angle = Double(index) / 12 * 2 * .pi
            return WorldVec3(cos(angle), sin(angle), 1)
        }
        let edges = WorldSkeletonBones.minimumSpanningTree(joints: joints)
        XCTAssertEqual(edges.count, joints.count - 1)
        assertConnected(edges: edges, vertexCount: joints.count)
    }

    func testRealBundledJointCloudsProduceAValidSkeleton() throws {
        let bundle = try WorldBundle.loadBundledSample()
        guard let frame = bundle.world.players.flatMap(\.frames).first(where: { !$0.jointsWorld.isEmpty }) else {
            return XCTFail("fixture should have at least one player frame with joints")
        }
        let edges = WorldSkeletonBones.minimumSpanningTree(joints: frame.jointsWorld)
        XCTAssertEqual(edges.count, frame.jointsWorld.count - 1)
        assertConnected(edges: edges, vertexCount: frame.jointsWorld.count)
    }

    private func assertConnected(edges: [(Int, Int)], vertexCount: Int, file: StaticString = #filePath, line: UInt = #line) {
        guard vertexCount > 0 else { return }
        var adjacency = [Int: [Int]](minimumCapacity: vertexCount)
        for (a, b) in edges {
            adjacency[a, default: []].append(b)
            adjacency[b, default: []].append(a)
        }
        var visited = Set<Int>([0])
        var stack = [0]
        while let node = stack.popLast() {
            for neighbor in adjacency[node] ?? [] where !visited.contains(neighbor) {
                visited.insert(neighbor)
                stack.append(neighbor)
            }
        }
        XCTAssertEqual(visited.count, vertexCount, "skeleton graph must be a single connected tree", file: file, line: line)
    }
}
