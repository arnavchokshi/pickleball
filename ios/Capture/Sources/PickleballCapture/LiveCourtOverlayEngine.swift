#if os(iOS) && canImport(CoreML)
@preconcurrency import AVFoundation
import CoreVideo
import Foundation
import PickleballFastTier

/// One rendered live-overlay frame: the (possibly stale, propagated) dot
/// positions plus whether the detector actually ran this frame.
public struct LiveCourtOverlayFrame: Equatable, Sendable {
    public var points: [CourtDotMapPoint]
    public var frameIndex: Int
    public var detectorInvoked: Bool
    public var videoAspectRatio: Double
    public var playerFootRings: [LivePlayerFootRing]
    public var ballState: LiveBallIndicatorState?

    public init(
        points: [CourtDotMapPoint],
        frameIndex: Int,
        detectorInvoked: Bool,
        videoAspectRatio: Double = 16.0 / 9.0,
        playerFootRings: [LivePlayerFootRing] = [],
        ballState: LiveBallIndicatorState? = nil
    ) {
        self.points = points
        self.frameIndex = frameIndex
        self.detectorInvoked = detectorInvoked
        self.videoAspectRatio = videoAspectRatio
        self.playerFootRings = playerFootRings
        self.ballState = ballState
    }
}

public enum LiveCourtOverlayStatus: Equatable, Sendable {
    case idle
    case running
    /// Real, honest state when the compiled detector model is not present
    /// on-device -- the overlay must show this rather than silently
    /// rendering nothing with no explanation. See
    /// `LiveCourtOverlayEngine.defaultModelURL` for where it looks.
    case modelUnavailable(String)
    case failed(String)
}

/// Wires the live camera feed into the on-device player detector at reduced
/// cadence (W3-LIVE-MLP surface 2). Reuses `CoreMLPersonDetector` (which
/// itself reuses `PersonTrackLinker`) verbatim from `PickleballFastTier` --
/// no new detection/tracking logic here, only cadence decimation (via
/// `LiveDetectionCadenceScheduler`, unit-tested in `PickleballFastTierTests`)
/// and screen-space dot-map projection (`CourtDotMapBuilder`, also
/// unit-tested there).
///
/// Looks for the compiled model at the SAME `Documents/benchmark_models/`
/// path `ios/AppTests/ANELatencyBenchmarkTests.swift` already pushes to via
/// `devicectl device copy to` (see `runs/ios_device_gate_20260702T025809Z/`)
/// -- if a previous session's push is still on the paired device, this
/// reuses it directly; if not, the engine reports `.modelUnavailable`
/// honestly instead of guessing or crashing.
@available(iOS 15.0, *)
public final class LiveCourtOverlayEngine: @unchecked Sendable {
    public var onFrame: (@Sendable (LiveCourtOverlayFrame) -> Void)?
    public var onStatusChange: (@Sendable (LiveCourtOverlayStatus) -> Void)?
    public var onFramePresentationTimestamp: (@Sendable (Int, Double) -> Void)?

    private let tap: LiveFrameTap
    private let cadence: LiveDetectionCadenceScheduler
    private let modelURLProvider: @Sendable () -> URL?
    private let stateQueue = DispatchQueue(label: "com.arnavchokshi.pickleball.live-court-overlay-state")
    private var detector: CoreMLPersonDetector?
    private var didAttemptLoad = false
    private var lastDetections: [OnDevicePersonDetection] = []
    private var lastDetectionFrameIndex: Int?
    private var rotationDegrees: Int = 0
    private var status: LiveCourtOverlayStatus = .idle {
        didSet {
            guard status != oldValue else {
                return
            }
            onStatusChange?(status)
        }
    }

    public init(
        cadence: LiveDetectionCadenceScheduler = .budgeted,
        tap: LiveFrameTap = LiveFrameTap(),
        modelURLProvider: @escaping @Sendable () -> URL? = LiveCourtOverlayEngine.defaultModelURL
    ) {
        self.cadence = cadence
        self.tap = tap
        self.modelURLProvider = modelURLProvider
        self.tap.onFrame = { [weak self] pixelBuffer, frameIndex, presentationSeconds in
            self?.onFramePresentationTimestamp?(frameIndex, presentationSeconds)
            self?.stateQueue.async {
                self?.process(pixelBuffer: pixelBuffer, frameIndex: frameIndex)
            }
        }
    }

    /// Same `Documents/benchmark_models/yolo26n_640.mlmodelc` path used by
    /// `ANELatencyBenchmarkTests` -- the live app and the host-app unit test
    /// share the same app sandbox/Documents container.
    public static func defaultModelURL() -> URL? {
        guard let documentsURL = try? FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        ) else {
            return nil
        }
        let modelURL = documentsURL
            .appendingPathComponent("benchmark_models", isDirectory: true)
            .appendingPathComponent("yolo26n_640.mlmodelc", isDirectory: true)
        return FileManager.default.fileExists(atPath: modelURL.path) ? modelURL : nil
    }

    /// Best-effort attach: returns `false` (and never throws) if the session
    /// declines the extra output, so a live-overlay failure never blocks
    /// recording.
    @discardableResult
    public func attach(to session: AVCaptureSession, rotationDegrees: Int) -> Bool {
        stateQueue.sync {
            self.rotationDegrees = rotationDegrees
        }
        return tap.attach(to: session)
    }

    public func updateRotation(_ rotationDegrees: Int) {
        stateQueue.async {
            self.rotationDegrees = rotationDegrees
        }
    }

    private func process(pixelBuffer: CVPixelBuffer, frameIndex: Int) {
        if !didAttemptLoad {
            didAttemptLoad = true
            loadDetectorIfPossible()
        }
        guard let detector else {
            return
        }

        let sourceWidth = Double(CVPixelBufferGetWidth(pixelBuffer))
        let sourceHeight = Double(CVPixelBufferGetHeight(pixelBuffer))
        var detectorInvoked = false
        if cadence.shouldRunDetection(forFrameIndex: frameIndex) {
            do {
                lastDetections = try detector.process(pixelBuffer: pixelBuffer, frameIndex: frameIndex)
                detectorInvoked = true
                lastDetectionFrameIndex = frameIndex
                status = .running
            } catch {
                status = .failed(String(describing: error))
                return
            }
        }

        let points = CourtDotMapBuilder.build(
            detections: lastDetections,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            rotationDegrees: rotationDegrees
        )
        let playerFootRings = LivePlayerFootRingBuilder.build(
            detections: lastDetections,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            rotationDegrees: rotationDegrees,
            frameIndex: frameIndex,
            lastDetectionFrameIndex: lastDetectionFrameIndex
        )
        let rotatedDimensions = Self.rotatedDimensions(
            width: sourceWidth,
            height: sourceHeight,
            rotationDegrees: rotationDegrees
        )
        let videoAspectRatio: Double
        if rotatedDimensions.height > 0 {
            videoAspectRatio = rotatedDimensions.width / rotatedDimensions.height
        } else {
            videoAspectRatio = 16.0 / 9.0
        }
        onFrame?(
            LiveCourtOverlayFrame(
                points: points,
                frameIndex: frameIndex,
                detectorInvoked: detectorInvoked,
                videoAspectRatio: videoAspectRatio,
                playerFootRings: playerFootRings
            )
        )
    }

    private static func rotatedDimensions(
        width: Double,
        height: Double,
        rotationDegrees: Int
    ) -> (width: Double, height: Double) {
        switch ((rotationDegrees % 360) + 360) % 360 {
        case 90, 270:
            return (height, width)
        default:
            return (width, height)
        }
    }

    private func loadDetectorIfPossible() {
        guard let modelURL = modelURLProvider() else {
            status = .modelUnavailable(
                "No compiled detector at Documents/benchmark_models/yolo26n_640.mlmodelc -- push one via "
                    + "`devicectl device copy to` (see spikes/coreml_conversion) to enable the live court-dot map."
            )
            return
        }
        do {
            detector = try CoreMLPersonDetector(
                configuration: CoreMLPersonDetectorConfiguration(
                    candidate: .yolo26nInt8Img640LiveCadence4,
                    modelURL: modelURL,
                    outputFormat: .yolo26EndToEnd,
                    inputWidth: 640,
                    inputHeight: 640,
                    maxTracks: 4,
                    minConfidence: 0.10,
                    detectionIntervalFrames: cadence.everyNFrames
                )
            )
        } catch {
            status = .failed("Detector failed to load: \(error)")
        }
    }
}
#endif
