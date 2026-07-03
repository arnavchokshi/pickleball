import AVFoundation
import Foundation
import PickleballCore

public struct CameraRollVideoMetadata: Equatable, Sendable {
    public var resolution: [Int]
    public var fps: Double
    public var durationSeconds: Double
    public var format: CaptureFormat
    public var orientation: CaptureOrientation
    public var warnings: [String]

    public init(
        resolution: [Int],
        fps: Double,
        durationSeconds: Double,
        format: CaptureFormat,
        orientation: CaptureOrientation,
        warnings: [String]
    ) {
        self.resolution = resolution
        self.fps = fps
        self.durationSeconds = durationSeconds
        self.format = format
        self.orientation = orientation
        self.warnings = warnings
    }
}

public protocol CameraRollVideoProbing: Sendable {
    func metadata(for url: URL) async throws -> CameraRollVideoMetadata
}

public enum CameraRollVideoImportError: Error, Equatable, Sendable {
    case missingVideoTrack
    case invalidResolution([Int])
    case invalidFPS(Double)
    case invalidDuration(Double)
    case unreadableSource(URL)
}

public struct AVAssetCameraRollVideoProbe: CameraRollVideoProbing {
    public init() {}

    public func metadata(for url: URL) async throws -> CameraRollVideoMetadata {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw CameraRollVideoImportError.missingVideoTrack
        }

        let naturalSize = try await track.load(.naturalSize)
        let preferredTransform = try await track.load(.preferredTransform)
        let transformedSize = naturalSize.applying(preferredTransform)
        let width = Int(abs(transformedSize.width).rounded())
        let height = Int(abs(transformedSize.height).rounded())
        let durationSeconds = try await asset.load(.duration).seconds
        let fps = Double(try await track.load(.nominalFrameRate))

        guard width > 0, height > 0 else {
            throw CameraRollVideoImportError.invalidResolution([width, height])
        }
        guard fps.isFinite, fps > 0 else {
            throw CameraRollVideoImportError.invalidFPS(fps)
        }
        guard durationSeconds.isFinite, durationSeconds > 0 else {
            throw CameraRollVideoImportError.invalidDuration(durationSeconds)
        }

        return CameraRollVideoMetadata(
            resolution: [width, height],
            fps: fps,
            durationSeconds: durationSeconds,
            format: Self.format(for: url),
            orientation: width >= height ? .landscape : .portrait,
            warnings: ["fps_from_asset_nominal_rate"]
        )
    }

    private static func format(for url: URL) -> CaptureFormat {
        let ext = url.pathExtension.lowercased()
        if ext.contains("prores") {
            return .prores422lt
        }
        return .hevc
    }
}

public struct CameraRollVideoImportResult: Equatable, Sendable {
    public var descriptor: CapturePackageDescriptor
    public var clipURL: URL
    public var sidecarURL: URL
    public var sidecar: CaptureSidecar

    public init(
        descriptor: CapturePackageDescriptor,
        clipURL: URL,
        sidecarURL: URL,
        sidecar: CaptureSidecar
    ) {
        self.descriptor = descriptor
        self.clipURL = clipURL
        self.sidecarURL = sidecarURL
        self.sidecar = sidecar
    }
}

public struct CameraRollVideoImporter {
    public typealias SessionIDFactory = @Sendable (Date) -> String

    private let videoProbe: any CameraRollVideoProbing
    private let sessionIDFactory: SessionIDFactory
    private let fileManager: FileManager

    public init(
        videoProbe: any CameraRollVideoProbing = AVAssetCameraRollVideoProbe(),
        sessionIDFactory: @escaping SessionIDFactory = { date in
            "import-\(CaptureSessionIDFactory.uniqueSessionID(now: date).replacingOccurrences(of: "capture-", with: ""))"
        },
        fileManager: FileManager = .default
    ) {
        self.videoProbe = videoProbe
        self.sessionIDFactory = sessionIDFactory
        self.fileManager = fileManager
    }

    public func importVideo(
        sourceURL: URL,
        packageRootURL: URL,
        importedAt: Date = Date()
    ) async throws -> CameraRollVideoImportResult {
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw CameraRollVideoImportError.unreadableSource(sourceURL)
        }

        let metadata = try await videoProbe.metadata(for: sourceURL)
        let descriptor = try CapturePackageDescriptor(
            sessionID: sessionIDFactory(importedAt),
            fps: Int(metadata.fps.rounded()),
            resolution: metadata.resolution,
            format: metadata.format,
            orientation: metadata.orientation,
            startedAt: importedAt
        )
        let clipURL = packageRootURL.appendingPathComponent(descriptor.clipRelativePath)
        let sidecarURL = packageRootURL.appendingPathComponent(descriptor.sidecarRelativePath)

        try fileManager.createDirectory(at: clipURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fileManager.fileExists(atPath: clipURL.path) {
            try fileManager.removeItem(at: clipURL)
        }
        try fileManager.copyItem(at: sourceURL, to: clipURL)

        let sidecar = Self.makeImportedSidecar(
            descriptor: descriptor,
            metadata: metadata,
            importedAt: importedAt
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(sidecar).write(to: sidecarURL, options: .atomic)

        return CameraRollVideoImportResult(
            descriptor: descriptor,
            clipURL: clipURL,
            sidecarURL: sidecarURL,
            sidecar: sidecar
        )
    }

    private static func makeImportedSidecar(
        descriptor: CapturePackageDescriptor,
        metadata: CameraRollVideoMetadata,
        importedAt: Date
    ) -> CaptureSidecar {
        CaptureSidecar(
            provenance: .cameraRollImport,
            deviceTier: .standard,
            deviceModel: "camera_roll",
            fps: descriptor.expectedFPS,
            format: descriptor.expectedFormat,
            resolution: descriptor.expectedResolution,
            orientation: descriptor.expectedOrientation,
            captureDeviceOrientation: nil,
            videoRotationAngleDegrees: nil,
            recordingStartedAt: iso8601String(from: importedAt),
            recordingDurationS: metadata.durationSeconds,
            cameraPosition: nil,
            cameraLens: nil,
            locked: nil,
            intrinsics: nil,
            arkitCameraPose: nil,
            courtPlane: nil,
            manualCourtTaps: [],
            gravity: nil,
            unavailableSensorReasons: [
                "locked_camera_settings": "camera_roll_import_has_no_live_exposure_focus_or_white_balance",
                "camera_intrinsics": "camera_roll_import_has_no_live_calibration_intrinsics",
                "core_motion_gravity": "camera_roll_import_has_no_live_core_motion",
                "arkit_camera_pose": "camera_roll_import_has_no_live_arkit_tracking",
                "court_plane": "camera_roll_import_has_no_live_court_lock",
            ],
            captureQuality: importedCaptureQuality(metadata: metadata)
        )
    }

    private static func importedCaptureQuality(metadata: CameraRollVideoMetadata) -> CaptureQuality {
        var reasons = ["imported_no_live_sensors"] + metadata.warnings
        let width = metadata.resolution.first ?? 0
        let height = metadata.resolution.dropFirst().first ?? 0
        if min(width, height) < 1080 {
            reasons.append("resolution_below_1080p_floor")
        }
        if metadata.fps < 59.5 {
            reasons.append("fps_below_60_floor")
        }

        let grade: CaptureQuality.Grade = reasons.contains("resolution_below_1080p_floor")
            || reasons.contains("fps_below_60_floor")
            ? .poor
            : .warn
        return CaptureQuality(grade: grade, reasons: reasons)
    }

    private static func iso8601String(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}
