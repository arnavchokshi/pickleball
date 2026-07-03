import Foundation

/// Decides which live camera frames actually get run through the on-device
/// CoreML person detector (W3-LIVE-MLP, surface 2). Running the detector on
/// every frame is unnecessary: `runs/ios_device_gate_20260702T025809Z/LATENCY_TABLE_DEVICE.md`
/// measured the real iPhone 14 Pro ANE at ~3.19ms/frame for `yolo26n_640`
/// (~218fps headroom for the detector+ball-student compute alone), but that
/// number excludes RTMPose, person tracking, rendering, and thermal
/// throttling under sustained load -- none of which are proven yet on a real
/// camera+render loop. Running detection every 3rd-5th frame (not every
/// frame) leaves a wide, honest safety margin for everything not yet
/// measured, per the milestone's explicit cadence budget.
public struct LiveDetectionCadenceScheduler: Equatable, Sendable {
    /// Run the detector once every `everyNFrames` frames; frames in between
    /// reuse the last linked detections (still rendered, just not
    /// re-detected) via the caller's own track propagation.
    public var everyNFrames: Int

    public init(everyNFrames: Int = 4) {
        self.everyNFrames = max(1, everyNFrames)
    }

    public func shouldRunDetection(forFrameIndex frameIndex: Int) -> Bool {
        frameIndex % everyNFrames == 0
    }

    /// Budgeted default per the milestone spec ("every 3rd-5th frame").
    /// Picks the middle of that explicit range rather than the fastest
    /// extreme, keeping headroom for the still-unmeasured RTMPose/tracking/
    /// render/thermal cost documented above.
    public static let budgeted = LiveDetectionCadenceScheduler(everyNFrames: 4)
}
