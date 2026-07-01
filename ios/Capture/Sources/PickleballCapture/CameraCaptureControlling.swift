#if os(iOS)
@preconcurrency import AVFoundation
import Foundation
import PickleballCore

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
}
#endif
