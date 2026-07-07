import Foundation
import PickleballCore

public struct ARFrameSnapshot: Equatable, Sendable {
    public var timestampS: Double
    public var cameraPose: RigidPose
    public var intrinsics: CameraIntrinsics
    public var tracking: ARTrackingSnapshot
    public var courtPlane: Plane?

    public init(
        timestampS: Double,
        cameraPose: RigidPose,
        intrinsics: CameraIntrinsics,
        tracking: ARTrackingSnapshot,
        courtPlane: Plane? = nil
    ) {
        self.timestampS = timestampS
        self.cameraPose = cameraPose
        self.intrinsics = intrinsics
        self.tracking = tracking
        self.courtPlane = courtPlane
    }
}

public protocol ARSessionProviding: AnyObject, Sendable {
    func startSession()
    func stopSession()
    func snapshot(forVideoPTS ptsS: Double) -> ARFrameSnapshot?
}

public struct ARCaptureSidecarPayload: Equatable, Sendable {
    public var frameSamples: [ARKitFrameSample]
    public var courtPlane: Plane?

    public var latestFrame: ARKitFrameSample? {
        frameSamples.last
    }

    public init(frameSamples: [ARKitFrameSample], courtPlane: Plane? = nil) {
        self.frameSamples = frameSamples
        self.courtPlane = courtPlane
    }
}

public final class ARFrameSidecarRecorder: @unchecked Sendable {
    private let provider: ARSessionProviding
    private let maxSamples: Int
    private let lock = NSLock()
    private var isRecording = false
    private var samples: [ARKitFrameSample] = []
    private var latestCourtPlane: Plane?

    public init(provider: ARSessionProviding, maxSamples: Int = 4_800) {
        self.provider = provider
        self.maxSamples = max(1, maxSamples)
    }

    public func start() {
        startSession()
        beginRecording()
    }

    public func stop() {
        endRecording()
        stopSession()
    }

    public func startSession() {
        provider.startSession()
    }

    public func stopSession() {
        provider.stopSession()
    }

    public func beginRecording() {
        lock.lock()
        samples = []
        latestCourtPlane = nil
        isRecording = true
        lock.unlock()
    }

    public func endRecording() {
        lock.lock()
        isRecording = false
        lock.unlock()
    }

    public func recordVideoFrame(ptsS: Double) {
        lock.lock()
        let shouldRecord = isRecording
        lock.unlock()
        guard shouldRecord else {
            return
        }
        guard let snapshot = provider.snapshot(forVideoPTS: ptsS) else {
            return
        }
        let sample = ARKitFrameSample(
            videoPTSS: ptsS,
            arkitTimestampS: snapshot.timestampS,
            cameraPose: snapshot.cameraPose,
            intrinsics: snapshot.intrinsics,
            tracking: snapshot.tracking
        )

        lock.lock()
        defer {
            lock.unlock()
        }
        guard isRecording else {
            return
        }
        samples.append(sample)
        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }
        if let courtPlane = snapshot.courtPlane {
            latestCourtPlane = courtPlane
        }
    }

    public func sidecarPayload() -> ARCaptureSidecarPayload {
        lock.lock()
        let payload = ARCaptureSidecarPayload(frameSamples: samples, courtPlane: latestCourtPlane)
        lock.unlock()
        return payload
    }
}

public final class DeterministicARSessionProvider: ARSessionProviding, @unchecked Sendable {
    private let samples: [ARFrameSnapshot]
    private let lock = NSLock()
    private var nextIndex = 0
    public private(set) var isRunning = false

    public init(samples: [ARFrameSnapshot]) {
        self.samples = samples
    }

    public func startSession() {
        lock.lock()
        nextIndex = 0
        isRunning = true
        lock.unlock()
    }

    public func stopSession() {
        lock.lock()
        isRunning = false
        lock.unlock()
    }

    public func snapshot(forVideoPTS _: Double) -> ARFrameSnapshot? {
        lock.lock()
        defer {
            lock.unlock()
        }
        guard isRunning, !samples.isEmpty else {
            return nil
        }
        let index = min(nextIndex, samples.count - 1)
        nextIndex += 1
        return samples[index]
    }
}

public final class NoOpARSessionProvider: ARSessionProviding, @unchecked Sendable {
    public init() {}

    public func startSession() {}
    public func stopSession() {}
    public func snapshot(forVideoPTS _: Double) -> ARFrameSnapshot? { nil }
}

public enum DefaultARSessionProviderFactory {
    public static func make() -> ARSessionProviding {
        #if os(iOS) && canImport(ARKit)
        return ARKitSessionProvider()
        #else
        return NoOpARSessionProvider()
        #endif
    }
}
