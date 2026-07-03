import SceneKit

/// Builds and updates the SceneKit world for the GLUE-4 iOS viewer: a
/// static regulation court + net built once from `VirtualWorld.court`, plus
/// a per-scrub-tick dynamic layer (players at their locked tier, ball) that
/// is fully rebuilt from a `WorldFrameSnapshot`. World units are meters in
/// the `court_Z0` frame (Z-up); the camera is oriented with `up = (0,0,1)`
/// to match rather than remapping every coordinate.
public final class WorldSceneBuilder {
    public let scene = SCNScene()
    public let cameraNode = SCNNode()

    private let dynamicRoot = SCNNode()
    private let meshFaces: [WorldMeshFace]

    public init(court: VirtualWorld.Court, meshFaces: [WorldMeshFace]) {
        self.meshFaces = meshFaces
        scene.background.contents = CGColor(red: 0.067, green: 0.075, blue: 0.082, alpha: 1)
        scene.rootNode.addChildNode(dynamicRoot)
        buildCamera()
        buildLighting()
        buildCourt(court)
    }

    // MARK: - Static geometry

    private func buildCamera() {
        let camera = SCNCamera()
        camera.fieldOfView = 50
        camera.zNear = 0.05
        camera.zFar = 100
        cameraNode.camera = camera
        scene.rootNode.addChildNode(cameraNode)
    }

    private func buildLighting() {
        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 700
        scene.rootNode.addChildNode(ambient)

        let directional = SCNNode()
        directional.light = SCNLight()
        directional.light?.type = .directional
        directional.light?.intensity = 900
        directional.position = SCNVector3(0, -4, 8)
        directional.look(at: SCNVector3(0, 0, 0), up: SCNVector3(0, 0, 1), localFront: SCNVector3(0, 0, -1))
        scene.rootNode.addChildNode(directional)
    }

    private func buildCourt(_ court: VirtualWorld.Court) {
        let bounds = WorldCameraPlanner.courtBounds(for: court)
        let floor = SCNPlane(width: CGFloat(bounds.width), height: CGFloat(bounds.length))
        floor.firstMaterial?.diffuse.contents = CGColor(red: 0.11, green: 0.45, blue: 0.31, alpha: 1)
        floor.firstMaterial?.isDoubleSided = true
        let floorNode = SCNNode(geometry: floor)
        floorNode.simdPosition = SIMD3(Float(bounds.centerX), Float(bounds.centerY), -0.01)
        scene.rootNode.addChildNode(floorNode)

        var lineSegments: [(WorldVec3, WorldVec3)] = court.lineSegments.values.compactMap { pair in
            guard pair.count == 2 else { return nil }
            return (pair[0], pair[1])
        }
        if court.net.endpoints.count == 2 {
            let left = court.net.endpoints[0]
            let right = court.net.endpoints[1]
            let topLeft = WorldVec3(left.x, left.y, court.net.postHeightM)
            let topRight = WorldVec3(right.x, right.y, court.net.postHeightM)
            let centerX = (left.x + right.x) / 2
            let centerY = (left.y + right.y) / 2
            let centerTop = WorldVec3(centerX, centerY, court.net.centerHeightM)
            lineSegments.append((topLeft, centerTop))
            lineSegments.append((centerTop, topRight))
        }
        if let lineGeometry = WorldSceneGeometry.lineGeometry(segments: lineSegments) {
            lineGeometry.firstMaterial?.diffuse.contents = CGColor(red: 0.91, green: 0.96, blue: 0.91, alpha: 1)
            lineGeometry.firstMaterial?.lightingModel = .constant
            scene.rootNode.addChildNode(SCNNode(geometry: lineGeometry))
        }

        if court.net.endpoints.count == 2 {
            scene.rootNode.addChildNode(netAssemblyNode(court: court))
        }
    }

    private func netAssemblyNode(court: VirtualWorld.Court) -> SCNNode {
        let root = SCNNode()
        let left = court.net.endpoints[0]
        let right = court.net.endpoints[1]
        let width = left.floorDistance(to: right)
        let net = SCNBox(width: CGFloat(width), height: 0.045, length: CGFloat(court.net.postHeightM), chamferRadius: 0)
        net.firstMaterial?.diffuse.contents = CGColor(red: 0.62, green: 0.83, blue: 0.84, alpha: 0.28)
        net.firstMaterial?.isDoubleSided = true
        net.firstMaterial?.lightingModel = .constant
        let netNode = SCNNode(geometry: net)
        netNode.simdPosition = SIMD3(Float((left.x + right.x) / 2), Float((left.y + right.y) / 2), Float(court.net.postHeightM / 2))
        root.addChildNode(netNode)

        for endpoint in [left, right] {
            let post = SCNBox(width: 0.075, height: 0.075, length: CGFloat(court.net.postHeightM), chamferRadius: 0)
            post.firstMaterial?.diffuse.contents = CGColor(red: 0.95, green: 0.95, blue: 0.91, alpha: 1)
            let postNode = SCNNode(geometry: post)
            postNode.simdPosition = SIMD3(Float(endpoint.x), Float(endpoint.y), Float(court.net.postHeightM / 2))
            root.addChildNode(postNode)
        }
        return root
    }

    // MARK: - Dynamic per-frame layer

    public func apply(_ snapshot: WorldFrameSnapshot, dimLowConfidence: Bool) {
        dynamicRoot.childNodes.forEach { $0.removeFromParentNode() }
        for player in snapshot.players {
            dynamicRoot.addChildNode(playerNode(player, dimLowConfidence: dimLowConfidence))
        }
        if snapshot.ball.shouldRender3D, let frame = snapshot.ball.frame, let position = frame.worldXYZ {
            dynamicRoot.addChildNode(ballNode(at: position, badge: snapshot.ballTrustBadge, approx: frame.approx, dimLowConfidence: dimLowConfidence))
        }
    }

    private func playerNode(_ player: WorldFrameSnapshot.PlayerSnapshot, dimLowConfidence: Bool) -> SCNNode {
        let root = SCNNode()
        root.name = "player-\(player.id)"
        let opacity = WorldTrustColors.opacity(for: player.trustBadge, dimLowConfidence: dimLowConfidence)
        let color = WorldTrustColors.cgColor(for: player.trustBadge, opacity: opacity)

        if let trailSegments = WorldSceneGeometry.lineGeometry(segments: WorldSceneGeometry.polylineSegments(points: player.floorTrail)) {
            trailSegments.firstMaterial?.diffuse.contents = color
            trailSegments.firstMaterial?.lightingModel = .constant
            root.addChildNode(SCNNode(geometry: trailSegments))
        }

        if let floor = player.floorPosition {
            let dot = SCNCylinder(radius: 0.16, height: 0.025)
            dot.firstMaterial?.diffuse.contents = color
            let dotNode = SCNNode(geometry: dot)
            dotNode.simdPosition = SIMD3(Float(floor.x), Float(floor.y), Float(floor.z) + 0.0125)
            root.addChildNode(dotNode)
        }

        switch player.tier {
        case .mesh(let frame):
            root.addChildNode(meshNode(frame: frame, opacity: opacity))
        case .joints(let frame):
            root.addChildNode(skeletonNode(joints: frame.jointsWorld, color: color))
        case .trackOnly, .none:
            break
        }
        return root
    }

    private func meshNode(frame: BodyMesh.Frame, opacity: Double) -> SCNNode {
        guard let geometry = WorldSceneGeometry.meshGeometry(vertices: frame.meshVerticesWorld, faces: meshFaces) else {
            return SCNNode()
        }
        geometry.firstMaterial?.diffuse.contents = CGColor(red: 0.71, green: 0.95, blue: 0.75, alpha: 1)
        geometry.firstMaterial?.emission.contents = CGColor(red: 0.06, green: 0.18, blue: 0.09, alpha: 1)
        geometry.firstMaterial?.isDoubleSided = true
        geometry.firstMaterial?.transparency = CGFloat(max(0.05, min(1, frame.meshOpacity)))
        geometry.firstMaterial?.writesToDepthBuffer = false
        let node = SCNNode(geometry: geometry)
        node.renderingOrder = 20
        return node
    }

    private func skeletonNode(joints: [WorldVec3], color: CGColor) -> SCNNode {
        let root = SCNNode()
        for joint in joints {
            let sphere = SCNSphere(radius: 0.028)
            sphere.firstMaterial?.diffuse.contents = color
            sphere.firstMaterial?.lightingModel = .constant
            let node = SCNNode(geometry: sphere)
            node.simdPosition = SIMD3(Float(joint.x), Float(joint.y), Float(joint.z))
            root.addChildNode(node)
        }
        let bones = WorldSkeletonBones.minimumSpanningTree(joints: joints)
        for (a, b) in bones {
            root.addChildNode(WorldSceneGeometry.capsuleNode(from: joints[a], to: joints[b], radius: 0.012, color: color))
        }
        return root
    }

    private func ballNode(at position: WorldVec3, badge: TrustBadge, approx: Bool, dimLowConfidence: Bool) -> SCNNode {
        let sphere = SCNSphere(radius: 0.055)
        let opacity = WorldTrustColors.opacity(for: badge, dimLowConfidence: dimLowConfidence)
        let color: CGColor
        if badge == .lowConfidence {
            color = WorldTrustColors.cgColor(for: .lowConfidence, opacity: opacity)
        } else if approx {
            color = CGColor(red: 1, green: 0.81, blue: 0.35, alpha: opacity)
        } else {
            color = CGColor(red: 0.91, green: 1, blue: 0.2, alpha: opacity)
        }
        sphere.firstMaterial?.diffuse.contents = color
        sphere.firstMaterial?.lightingModel = .constant
        let node = SCNNode(geometry: sphere)
        node.name = "ball"
        node.simdPosition = SIMD3(Float(position.x), Float(position.y), Float(position.z))
        return node
    }
}
