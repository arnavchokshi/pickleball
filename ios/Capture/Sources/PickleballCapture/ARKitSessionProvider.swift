#if os(iOS) && canImport(ARKit)
import ARKit
import Foundation
import PickleballCore

public final class ARKitSessionProvider: NSObject, ARSessionProviding, ARSessionDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var session: ARSession?
    private var latestSnapshot: ARFrameSnapshot?
    private var latestCourtPlane: Plane?

    public func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            return
        }
        let session = makeSession()
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal]
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    public func stopSession() {
        session?.pause()
        session?.delegate = nil
        session = nil
        lock.lock()
        latestSnapshot = nil
        latestCourtPlane = nil
        lock.unlock()
    }

    public func snapshot(forVideoPTS _: Double) -> ARFrameSnapshot? {
        lock.lock()
        let snapshot = latestSnapshot
        lock.unlock()
        return snapshot
    }

    public func session(_: ARSession, didUpdate frame: ARFrame) {
        let snapshot = Self.snapshot(from: frame, courtPlane: currentCourtPlane())
        lock.lock()
        latestSnapshot = snapshot
        lock.unlock()
    }

    public func session(_: ARSession, didAdd anchors: [ARAnchor]) {
        updateCourtPlane(from: anchors)
    }

    public func session(_: ARSession, didUpdate anchors: [ARAnchor]) {
        updateCourtPlane(from: anchors)
    }

    private func updateCourtPlane(from anchors: [ARAnchor]) {
        guard let planeAnchor = anchors.compactMap({ $0 as? ARPlaneAnchor }).first(where: { $0.alignment == .horizontal }) else {
            return
        }
        let transform = planeAnchor.transform
        let point = [
            Double(transform.columns.3.x),
            Double(transform.columns.3.y),
            Double(transform.columns.3.z),
        ]
        let normal = [
            Double(transform.columns.1.x),
            Double(transform.columns.1.y),
            Double(transform.columns.1.z),
        ]
        lock.lock()
        latestCourtPlane = Plane(point: point, normal: normal)
        lock.unlock()
    }

    private func currentCourtPlane() -> Plane? {
        lock.lock()
        let plane = latestCourtPlane
        lock.unlock()
        return plane
    }

    private func makeSession() -> ARSession {
        if let session {
            session.delegate = self
            return session
        }
        let session = ARSession()
        session.delegate = self
        self.session = session
        return session
    }

    private static func snapshot(from frame: ARFrame, courtPlane: Plane?) -> ARFrameSnapshot {
        let transform = frame.camera.transform
        let intrinsics = frame.camera.intrinsics
        return ARFrameSnapshot(
            timestampS: frame.timestamp,
            cameraPose: RigidPose(
                R: [
                    [Double(transform.columns.0.x), Double(transform.columns.1.x), Double(transform.columns.2.x)],
                    [Double(transform.columns.0.y), Double(transform.columns.1.y), Double(transform.columns.2.y)],
                    [Double(transform.columns.0.z), Double(transform.columns.1.z), Double(transform.columns.2.z)],
                ],
                t: [
                    Double(transform.columns.3.x),
                    Double(transform.columns.3.y),
                    Double(transform.columns.3.z),
                ]
            ),
            intrinsics: CameraIntrinsics(
                fx: Double(intrinsics.columns.0.x),
                fy: Double(intrinsics.columns.1.y),
                cx: Double(intrinsics.columns.2.x),
                cy: Double(intrinsics.columns.2.y),
                source: "arkit"
            ),
            tracking: trackingSnapshot(from: frame.camera.trackingState),
            courtPlane: courtPlane
        )
    }

    private static func trackingSnapshot(from state: ARCamera.TrackingState) -> ARTrackingSnapshot {
        switch state {
        case .normal:
            return ARTrackingSnapshot(state: .normal, quality: .good)
        case .notAvailable:
            return ARTrackingSnapshot(state: .unavailable, quality: .unavailable)
        case .limited(let reason):
            return ARTrackingSnapshot(state: .limited, quality: .limited, reason: "\(reason)")
        }
    }
}
#endif
