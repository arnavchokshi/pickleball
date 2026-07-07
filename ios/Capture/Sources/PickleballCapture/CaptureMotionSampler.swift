#if os(iOS)
@preconcurrency import CoreMotion
import Foundation

public final class CaptureMotionSampler: @unchecked Sendable {
    private let manager = CMMotionManager()

    public init() {}

    public var isAvailable: Bool {
        manager.isDeviceMotionAvailable
    }

    public var latestGravity: [Double] {
        guard let gravity = manager.deviceMotion?.gravity else {
            return [0.0, -1.0, 0.0]
        }

        return [gravity.x, gravity.y, gravity.z]
    }

    public func start(sampleRateHz: Double = 30.0) {
        guard manager.isDeviceMotionAvailable else {
            return
        }

        manager.deviceMotionUpdateInterval = 1.0 / sampleRateHz
        manager.startDeviceMotionUpdates()
    }

    public func stop() {
        guard manager.isDeviceMotionActive else {
            return
        }

        manager.stopDeviceMotionUpdates()
    }
}
#endif
