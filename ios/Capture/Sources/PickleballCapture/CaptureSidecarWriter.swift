import Foundation
import PickleballCore

public struct CaptureSidecarWriteContext: Equatable, Sendable {
    public var deviceTier: DeviceTier
    public var deviceModel: String
    public var cameraPosition: String?
    public var cameraLens: String?
    public var locked: LockedCapture
    public var intrinsics: CameraIntrinsics
    public var gravity: [Double]
    public var captureQuality: CaptureQuality

    public init(
        deviceTier: DeviceTier,
        deviceModel: String,
        cameraPosition: String?,
        cameraLens: String?,
        locked: LockedCapture,
        intrinsics: CameraIntrinsics,
        gravity: [Double],
        captureQuality: CaptureQuality
    ) {
        self.deviceTier = deviceTier
        self.deviceModel = deviceModel
        self.cameraPosition = cameraPosition
        self.cameraLens = cameraLens
        self.locked = locked
        self.intrinsics = intrinsics
        self.gravity = gravity
        self.captureQuality = captureQuality
    }
}

public enum CaptureSidecarWriter {
    @discardableResult
    public static func write(
        descriptor: CapturePackageDescriptor,
        packageRootURL: URL,
        recordingStartedAt: Date,
        finishedAt: Date,
        context: CaptureSidecarWriteContext,
        fileManager: FileManager = .default
    ) throws -> URL {
        let sidecarURL = packageRootURL.appendingPathComponent(descriptor.sidecarRelativePath, isDirectory: false)
        try fileManager.createDirectory(
            at: sidecarURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let sidecar = makeSidecar(
            descriptor: descriptor,
            recordingStartedAt: recordingStartedAt,
            finishedAt: finishedAt,
            context: context
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(sidecar).write(to: sidecarURL, options: .atomic)
        return sidecarURL
    }

    public static func makeSidecar(
        descriptor: CapturePackageDescriptor,
        recordingStartedAt: Date,
        finishedAt: Date,
        context: CaptureSidecarWriteContext
    ) -> CaptureSidecar {
        CaptureSidecar(
            deviceTier: context.deviceTier,
            deviceModel: context.deviceModel,
            fps: descriptor.expectedFPS,
            format: descriptor.expectedFormat,
            resolution: descriptor.expectedResolution,
            orientation: descriptor.expectedOrientation,
            captureDeviceOrientation: descriptor.captureDeviceOrientation,
            videoRotationAngleDegrees: descriptor.videoRotationAngleDegrees,
            recordingStartedAt: iso8601String(from: recordingStartedAt),
            recordingDurationS: max(0.0, finishedAt.timeIntervalSince(recordingStartedAt)),
            cameraPosition: context.cameraPosition,
            cameraLens: context.cameraLens,
            locked: context.locked,
            intrinsics: context.intrinsics,
            gravity: context.gravity,
            ondevicePoseTrack: nil,
            captureQuality: context.captureQuality
        )
    }

    private static func iso8601String(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}
