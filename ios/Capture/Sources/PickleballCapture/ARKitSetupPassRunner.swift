import Foundation
import PickleballCore

public final class ARKitSetupPassRunner: @unchecked Sendable {
    private let provider: ARSessionProviding
    private let ownership: CameraResourceOwnership
    private let gravityProvider: @Sendable () -> [Double]
    private let pollIntervalNanoseconds: UInt64

    public init(
        provider: ARSessionProviding,
        ownership: CameraResourceOwnership,
        gravityProvider: @escaping @Sendable () -> [Double],
        pollIntervalNanoseconds: UInt64 = 100_000_000
    ) {
        self.provider = provider
        self.ownership = ownership
        self.gravityProvider = gravityProvider
        self.pollIntervalNanoseconds = pollIntervalNanoseconds
    }

    public func run(timeoutSeconds: Double = 4.0) async -> ARKitSetupPassSidecar {
        let startedAt = Date()
        let gravity = gravityProvider()
        let token: ARKitSetupCameraOwnershipToken
        do {
            token = try ownership.beginARKitSetup()
        } catch let error as CameraResourceOwnershipError {
            return .unavailable(
                reason: unavailableReason(for: error),
                gravity: gravity,
                durationS: Date().timeIntervalSince(startedAt)
            )
        } catch {
            return .unavailable(
                reason: "arkit_setup_pass_camera_ownership_failed",
                gravity: gravity,
                durationS: Date().timeIntervalSince(startedAt)
            )
        }

        provider.startSession()
        defer {
            provider.stopSession()
            token.release()
        }

        let deadline = startedAt.addingTimeInterval(max(0, timeoutSeconds))
        repeat {
            if let snapshot = provider.snapshot(forVideoPTS: 0), let courtPlane = snapshot.courtPlane {
                return ARKitSetupPassSidecar(
                    intrinsics: snapshot.intrinsics,
                    cameraPose: snapshot.cameraPose,
                    courtPlane: courtPlane,
                    trackingState: setupTrackingState(from: snapshot.tracking.state),
                    timestampS: snapshot.timestampS,
                    durationS: Date().timeIntervalSince(startedAt),
                    gravity: gravityProvider()
                )
            }

            if pollIntervalNanoseconds > 0 {
                try? await Task.sleep(nanoseconds: pollIntervalNanoseconds)
            } else {
                await Task.yield()
            }
        } while Date() < deadline

        return .unavailable(
            reason: "arkit_setup_pass_timeout",
            gravity: gravityProvider(),
            durationS: Date().timeIntervalSince(startedAt)
        )
    }

    private func setupTrackingState(from state: ARTrackingState) -> ARKitSetupTrackingState {
        switch state {
        case .normal:
            return .normal
        case .limited:
            return .limited
        case .unavailable:
            return .unavailable
        }
    }

    private func unavailableReason(for error: CameraResourceOwnershipError) -> String {
        switch error {
        case .cameraAlreadyOwned(.avCapture):
            return "camera_owned_by_avcapture"
        case .cameraAlreadyOwned(.arKitSetup):
            return "camera_owned_by_arkit_setup"
        }
    }
}

public enum ARKitSetupPassRefreshPolicy {
    public static func shouldRefresh(
        now: Date,
        lastCompletedAt: Date?,
        lastGravity: [Double]?,
        currentGravity: [Double],
        staleAfterSeconds: TimeInterval = 120,
        gravityDeltaThreshold: Double = 0.08
    ) -> Bool {
        guard let lastCompletedAt else {
            return true
        }
        if now.timeIntervalSince(lastCompletedAt) > staleAfterSeconds {
            return true
        }
        guard let lastGravity else {
            return false
        }
        return gravityDelta(lastGravity, currentGravity) > gravityDeltaThreshold
    }

    public static func gravityDelta(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == 3, rhs.count == 3 else {
            return .infinity
        }
        let squared = zip(lhs, rhs).reduce(0.0) { partial, pair in
            let delta = pair.0 - pair.1
            return partial + delta * delta
        }
        return squared.squareRoot()
    }
}
