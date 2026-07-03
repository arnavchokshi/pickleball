#if os(iOS)
@preconcurrency import AVFoundation
import Foundation
import PickleballCore
import PickleballGuidance

public protocol CameraCaptureControlling: AnyObject, Sendable {
    var session: AVCaptureSession { get }
    var onRecordingFinished: ((Result<CameraRecordingResult, Error>) -> Void)? { get set }

    func configure(
        mode: CaptureMode,
        deviceTier: DeviceTier,
        capabilities: CaptureCodecCapabilities,
        captureDeviceOrientation: CaptureDeviceOrientation,
        sessionID: String,
        packageRootURL: URL
    ) async throws -> CapturePackageDescriptor

    func startPreview() async
    func stopPreview() async
    func startRecording() async throws -> CapturePackageDescriptor
    func stopRecording() async throws

    /// Real, live-readback signals for the pre-record capture-quality
    /// guidance screen (W3-LIVE-MLP surface 1). Default implementation
    /// returns an all-`nil` sample (every check renders `.unavailable`) so
    /// test fakes that don't override this stay honest by default rather
    /// than needing a stub.
    func currentLiveGuidanceSample() async -> LiveGuidanceSample

    /// Wires the live court-dot map (W3-LIVE-MLP surface 2). Default
    /// implementation is a no-op for test fakes/controllers that don't have
    /// a live overlay engine.
    func setLiveCourtOverlayHandlers(
        onFrame: (@Sendable (LiveCourtOverlayFrame) -> Void)?,
        onStatusChange: (@Sendable (LiveCourtOverlayStatus) -> Void)?
    )
}

extension CameraCaptureControlling {
    public func currentLiveGuidanceSample() async -> LiveGuidanceSample {
        LiveGuidanceSample()
    }

    public func setLiveCourtOverlayHandlers(
        onFrame _: (@Sendable (LiveCourtOverlayFrame) -> Void)?,
        onStatusChange _: (@Sendable (LiveCourtOverlayStatus) -> Void)?
    ) {}
}

public final class QueuedCameraCaptureController: CameraCaptureControlling, @unchecked Sendable {
    public let controller: CameraCaptureController
    private let sessionQueue: CameraSessionQueue

    public init(
        controller: CameraCaptureController = CameraCaptureController(),
        sessionQueue: CameraSessionQueue = CameraSessionQueue()
    ) {
        self.controller = controller
        self.sessionQueue = sessionQueue
    }

    public var session: AVCaptureSession {
        controller.session
    }

    public var onRecordingFinished: ((Result<CameraRecordingResult, Error>) -> Void)? {
        get {
            controller.onRecordingFinished
        }
        set {
            controller.onRecordingFinished = newValue
        }
    }

    public func configure(
        mode: CaptureMode,
        deviceTier: DeviceTier = .standard,
        capabilities: CaptureCodecCapabilities = .hevcOnly,
        captureDeviceOrientation: CaptureDeviceOrientation = .landscapeRight,
        sessionID: String = CameraCaptureController.defaultSessionID(),
        packageRootURL: URL = CameraCaptureController.defaultPackageRootURL()
    ) async throws -> CapturePackageDescriptor {
        try await sessionQueue.run {
            try self.controller.configure(
                mode: mode,
                deviceTier: deviceTier,
                capabilities: capabilities,
                captureDeviceOrientation: captureDeviceOrientation,
                sessionID: sessionID,
                packageRootURL: packageRootURL
            )
        }
    }

    public func startPreview() async {
        _ = try? await sessionQueue.run {
            self.controller.startPreview()
        }
    }

    public func stopPreview() async {
        _ = try? await sessionQueue.run {
            self.controller.stopPreview()
        }
    }

    public func startRecording() async throws -> CapturePackageDescriptor {
        try await sessionQueue.run {
            try self.controller.startRecording()
            guard let descriptor = self.controller.activeDescriptor else {
                throw CameraCaptureControllerError.noConfiguredPackage
            }
            return descriptor
        }
    }

    public func stopRecording() async throws {
        try await sessionQueue.run {
            try self.controller.stopRecording()
        }
    }

    public func currentLiveGuidanceSample() async -> LiveGuidanceSample {
        (try? await sessionQueue.run { self.controller.currentLiveGuidanceSample() }) ?? LiveGuidanceSample()
    }

    public func setLiveCourtOverlayHandlers(
        onFrame: (@Sendable (LiveCourtOverlayFrame) -> Void)?,
        onStatusChange: (@Sendable (LiveCourtOverlayStatus) -> Void)?
    ) {
        controller.liveOverlayEngine.onFrame = onFrame
        controller.liveOverlayEngine.onStatusChange = onStatusChange
    }
}
#endif
