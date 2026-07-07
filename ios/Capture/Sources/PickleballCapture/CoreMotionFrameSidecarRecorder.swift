import Foundation
import PickleballCore

public final class CoreMotionFrameSidecarRecorder: @unchecked Sendable {
    private let maxSamples: Int
    private let lock = NSLock()
    private var isRecording = false
    private var samples: [ARKitFrameSample] = []

    public init(maxSamples: Int = 4_800) {
        self.maxSamples = max(1, maxSamples)
    }

    public func beginRecording() {
        lock.lock()
        samples = []
        isRecording = true
        lock.unlock()
    }

    public func endRecording() {
        lock.lock()
        isRecording = false
        lock.unlock()
    }

    public func recordVideoFrame(ptsS: Double, gravity: [Double]) {
        lock.lock()
        defer {
            lock.unlock()
        }
        guard isRecording else {
            return
        }
        samples.append(ARKitFrameSample(
            videoPTSS: ptsS,
            gravity: gravity,
            provenance: .coreMotionOnly,
            unavailableReason: "arkit_not_running_during_avcapture_recording"
        ))
        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }
    }

    public func frameSamples() -> [ARKitFrameSample] {
        lock.lock()
        let currentSamples = samples
        lock.unlock()
        return currentSamples
    }
}
