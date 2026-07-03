import AVFoundation
import Vision
import Foundation

/// Thin, platform-agnostic wrapper around `VNDetectTrajectoriesRequest`.
///
/// This type is deliberately the ONLY place that touches the Vision API, and
/// it only depends on `CMSampleBuffer` (not on `AVAssetReader` or any other
/// macOS-file-reading machinery). That is what makes it "device-ready": the
/// exact same `TrajectoryDetector` instance and `process(sampleBuffer:...)`
/// call can be reused verbatim inside an iOS
/// `AVCaptureVideoDataOutputSampleBufferDelegate.captureOutput(_:didOutput:from:)`
/// callback if this candidate is ever reopened. The current policy in
/// `BALL_TRACKING_PIPELINE.md` keeps this as a killed BALL path.
///
/// VNDetectTrajectoriesRequest is a *stateful* Vision request
/// (`VNStatefulRequest`): the analysis window (last `trajectoryLength`
/// frames) lives inside the request object itself, not inside whatever
/// handler performs it. Per Apple's own sample pattern (WWDC 2020 "Explore
/// the Action & Vision app" / the request's own doc comments), the request
/// object must be created ONCE and reused across every frame, while a fresh
/// `VNImageRequestHandler` is created per frame around that frame's sample
/// buffer.
final class TrajectoryDetector {
    let request: VNDetectTrajectoriesRequest
    let trajectoryLength: Int
    let objectMinimumNormalizedRadius: Float
    let objectMaximumNormalizedRadius: Float

    private(set) var performCallCount: Int = 0

    init(
        trajectoryLength: Int,
        objectMinimumNormalizedRadius: Float,
        objectMaximumNormalizedRadius: Float
    ) {
        self.trajectoryLength = trajectoryLength
        self.objectMinimumNormalizedRadius = objectMinimumNormalizedRadius
        self.objectMaximumNormalizedRadius = objectMaximumNormalizedRadius
        // .zero == analyze every fed frame (no thinning). This matches the
        // 10s smoke-clip eval intent: we want maximum recall data first,
        // and can thin frames later once running live on-device if latency
        // demands it.
        self.request = VNDetectTrajectoriesRequest(
            frameAnalysisSpacing: .zero,
            trajectoryLength: trajectoryLength
        )
        self.request.objectMinimumNormalizedRadius = objectMinimumNormalizedRadius
        self.request.objectMaximumNormalizedRadius = objectMaximumNormalizedRadius
    }

    var requestRevision: Int {
        request.revision
    }

    /// Feeds one frame and returns whatever trajectory observations Vision
    /// produced for it (empty most of the time while the sliding window
    /// warms up, and after every subsequent frame once it has enough
    /// history). `frameIndex`/`ptsSeconds` are caller-tracked bookkeeping,
    /// not Vision inputs — they exist purely so the caller can stamp
    /// emissions with absolute video-frame indices for the JSON schema.
    func process(
        sampleBuffer: CMSampleBuffer,
        orientation: CGImagePropertyOrientation = .up
    ) throws -> [VNTrajectoryObservation] {
        let handler = VNImageRequestHandler(
            cmSampleBuffer: sampleBuffer,
            orientation: orientation,
            options: [:]
        )
        try handler.perform([request])
        performCallCount += 1
        guard let results = request.results else { return [] }
        return results
    }
}
