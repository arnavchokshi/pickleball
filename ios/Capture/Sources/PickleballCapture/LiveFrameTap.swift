#if os(iOS)
@preconcurrency import AVFoundation
import CoreVideo
import Foundation

/// Best-effort, additive video-frame tap for the live court-dot map
/// (W3-LIVE-MLP surface 2). This adds a SECOND output
/// (`AVCaptureVideoDataOutput`) to the same session the existing
/// `AVCaptureMovieFileOutput` already records from -- it never replaces or
/// reconfigures the movie output, and `attach(to:)` is a no-op (returns
/// `false`) if the session refuses the extra output for any reason, so a
/// live-overlay failure can never block or degrade the proven record path.
public final class LiveFrameTap: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    public let output = AVCaptureVideoDataOutput()
    /// `(pixelBuffer, frameIndex, presentationSeconds)`. Called on
    /// `queue`, NOT the main thread -- callers must hop to the main actor
    /// themselves before touching UI state.
    public var onFrame: (@Sendable (CVPixelBuffer, Int, Double) -> Void)?

    private let queue: DispatchQueue
    private var frameIndex = 0

    public init(label: String = "com.arnavchokshi.pickleball.live-frame-tap") {
        self.queue = DispatchQueue(label: label, qos: .userInitiated)
        super.init()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [String(kCVPixelBufferPixelFormatTypeKey): kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: queue)
    }

    /// Must be called inside the session's own
    /// `beginConfiguration()`/`commitConfiguration()` block, after the movie
    /// output has already been added, so the movie output always wins any
    /// resource contention.
    @discardableResult
    public func attach(to session: AVCaptureSession) -> Bool {
        guard session.canAddOutput(output) else {
            return false
        }
        session.addOutput(output)
        return true
    }

    public func resetFrameIndex() {
        queue.async { [weak self] in
            self?.frameIndex = 0
        }
    }

    public func captureOutput(
        _: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from _: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let seconds = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let index = frameIndex
        frameIndex += 1
        onFrame?(pixelBuffer, index, seconds)
    }
}
#endif
