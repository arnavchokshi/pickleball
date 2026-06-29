#if os(iOS)
@preconcurrency import AVFoundation
import Darwin
import Foundation
import PickleballCore

public struct CameraRecordingResult: Equatable, Sendable {
    public var descriptor: CapturePackageDescriptor
    public var clipURL: URL

    public init(descriptor: CapturePackageDescriptor, clipURL: URL) {
        self.descriptor = descriptor
        self.clipURL = clipURL
    }
}

public enum CameraCaptureControllerError: Error, Equatable, Sendable {
    case permissionDenied(CapturePermissionSnapshot)
    case cameraUnavailable
    case cannotAddVideoInput
    case cannotAddAudioInput
    case cannotAddMovieOutput
    case unsupportedFrameRate(Int)
    case noConfiguredPackage
    case landscapeRequired
    case alreadyRecording
    case notRecording
    case fileSystem(String)
}

public final class CameraCaptureController: NSObject {
    public let session = AVCaptureSession()
    public private(set) var activeDescriptor: CapturePackageDescriptor?
    public private(set) var lastRecordingResult: CameraRecordingResult?
    public var onRecordingFinished: ((Result<CameraRecordingResult, Error>) -> Void)?

    private let movieOutput = AVCaptureMovieFileOutput()
    private let motionSampler = CaptureMotionSampler()
    private var activeDevice: AVCaptureDevice?
    private var activePolicy: CapturePolicy?
    private var activePackageRootURL: URL?
    private var activeCaptureDeviceOrientation: CaptureDeviceOrientation?
    private var activeClipURL: URL?
    private var activeSidecarURL: URL?
    private var recordingStartedAt: Date?
    private var sessionIDFactory = CaptureSessionIDFactory()

    public override init() {
        super.init()
    }

    public static func permissionSnapshot() -> CapturePermissionSnapshot {
        CapturePermissionSnapshot(
            camera: permissionState(for: .video),
            microphone: permissionState(for: .audio)
        )
    }

    public static func requestPermissions() async -> CapturePermissionSnapshot {
        async let cameraGranted = AVCaptureDevice.requestAccess(for: .video)
        async let microphoneGranted = AVCaptureDevice.requestAccess(for: .audio)
        _ = await (cameraGranted, microphoneGranted)
        return permissionSnapshot()
    }

    public func configure(
        mode: CaptureMode,
        deviceTier: DeviceTier = .standard,
        capabilities: CaptureCodecCapabilities = .hevcOnly,
        captureDeviceOrientation: CaptureDeviceOrientation = .landscapeRight,
        sessionID: String = defaultSessionID(),
        packageRootURL: URL = defaultPackageRootURL()
    ) throws -> CapturePackageDescriptor {
        let permissions = Self.permissionSnapshot()
        let captureOrientation = CaptureOrientationPolicy.captureOrientation(for: captureDeviceOrientation)
        let policy = CapturePolicy.recommended(
            for: mode,
            deviceTier: deviceTier,
            capabilities: capabilities,
            orientation: captureOrientation
        )
        let descriptor = try CapturePackageDescriptor(
            sessionID: sessionID,
            policy: policy,
            startedAt: Date(),
            captureDeviceOrientation: captureDeviceOrientation
        )
        let readiness = CaptureReadinessEvaluator.evaluate(permissions: permissions, descriptor: descriptor)
        guard readiness.isReady else {
            if readiness.blockers == [.landscapeRequired] {
                throw CameraCaptureControllerError.landscapeRequired
            }
            throw CameraCaptureControllerError.permissionDenied(permissions)
        }

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CameraCaptureControllerError.cameraUnavailable
        }
        try configureAudioSession()

        session.beginConfiguration()
        var didCommitConfiguration = false
        defer {
            if !didCommitConfiguration {
                session.commitConfiguration()
            }
        }
        session.inputs.forEach(session.removeInput)
        session.outputs.forEach(session.removeOutput)
        session.sessionPreset = sessionPreset(for: policy.resolution)

        try configure(camera: camera, with: policy)

        let videoInput = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(videoInput) else {
            throw CameraCaptureControllerError.cannotAddVideoInput
        }
        session.addInput(videoInput)

        guard let microphone = AVCaptureDevice.default(for: .audio) else {
            throw CameraCaptureControllerError.cannotAddAudioInput
        }
        let audioInput = try AVCaptureDeviceInput(device: microphone)
        guard session.canAddInput(audioInput) else {
            throw CameraCaptureControllerError.cannotAddAudioInput
        }
        session.addInput(audioInput)

        guard session.canAddOutput(movieOutput) else {
            throw CameraCaptureControllerError.cannotAddMovieOutput
        }
        session.addOutput(movieOutput)
        configureMovieOutput(policy: policy, videoRotationAngleDegrees: descriptor.videoRotationAngleDegrees)
        session.commitConfiguration()
        didCommitConfiguration = true

        activeDevice = camera
        activePolicy = policy
        activePackageRootURL = packageRootURL
        activeCaptureDeviceOrientation = captureDeviceOrientation
        activeDescriptor = descriptor
        activeClipURL = packageRootURL.appendingPathComponent(descriptor.clipRelativePath, isDirectory: false)
        activeSidecarURL = packageRootURL.appendingPathComponent(descriptor.sidecarRelativePath, isDirectory: false)
        return descriptor
    }

    public func startPreview() {
        guard !session.isRunning else {
            return
        }
        motionSampler.start()
        session.startRunning()
    }

    public func stopPreview() {
        guard session.isRunning else {
            return
        }
        session.stopRunning()
        motionSampler.stop()
    }

    public var latestGravity: [Double] {
        motionSampler.latestGravity
    }

    public func startRecording() throws {
        guard !movieOutput.isRecording else {
            throw CameraCaptureControllerError.alreadyRecording
        }
        guard let policy = activePolicy else {
            throw CameraCaptureControllerError.noConfiguredPackage
        }
        guard policy.orientation == .landscape else {
            throw CameraCaptureControllerError.landscapeRequired
        }
        if lastRecordingResult != nil {
            try rotateRecordingPackage()
        }
        guard let clipURL = activeClipURL else {
            throw CameraCaptureControllerError.noConfiguredPackage
        }

        try FileManager.default.createDirectory(
            at: clipURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: clipURL.path) {
            try FileManager.default.removeItem(at: clipURL)
        }

        recordingStartedAt = Date()
        lastRecordingResult = nil
        movieOutput.startRecording(to: clipURL, recordingDelegate: self)
    }

    public func stopRecording() throws {
        guard movieOutput.isRecording else {
            throw CameraCaptureControllerError.notRecording
        }

        movieOutput.stopRecording()
    }

    public static func defaultPackageRootURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    public static func defaultSessionID(now: Date = Date()) -> String {
        CaptureSessionIDFactory.uniqueSessionID(now: now)
    }

    private func rotateRecordingPackage() throws {
        guard let policy = activePolicy,
              let packageRootURL = activePackageRootURL,
              let captureDeviceOrientation = activeCaptureDeviceOrientation else {
            throw CameraCaptureControllerError.noConfiguredPackage
        }
        let descriptor = try CapturePackageDescriptor(
            sessionID: sessionIDFactory.nextSessionID(),
            policy: policy,
            startedAt: Date(),
            captureDeviceOrientation: captureDeviceOrientation
        )
        activeDescriptor = descriptor
        activeClipURL = packageRootURL.appendingPathComponent(descriptor.clipRelativePath, isDirectory: false)
        activeSidecarURL = packageRootURL.appendingPathComponent(descriptor.sidecarRelativePath, isDirectory: false)
    }

    private static func permissionState(for mediaType: AVMediaType) -> CapturePermissionState {
        switch AVCaptureDevice.authorizationStatus(for: mediaType) {
        case .authorized:
            return .authorized
        case .denied:
            return .denied
        case .restricted:
            return .restricted
        case .notDetermined:
            return .notDetermined
        @unknown default:
            return .restricted
        }
    }

    private func configure(camera: AVCaptureDevice, with policy: CapturePolicy) throws {
        try camera.lockForConfiguration()
        defer {
            camera.unlockForConfiguration()
        }

        guard let format = bestFormat(on: camera, policy: policy) else {
            throw CameraCaptureControllerError.unsupportedFrameRate(policy.fps)
        }

        camera.activeFormat = format
        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(policy.fps))
        camera.activeVideoMinFrameDuration = frameDuration
        camera.activeVideoMaxFrameDuration = frameDuration

        lockExposureFocusAndWhiteBalance(on: camera)
    }

    private func configureAudioSession() throws {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .videoRecording, options: [.allowBluetooth])
            try audioSession.setActive(true)
        } catch {
            throw CameraCaptureControllerError.cannotAddAudioInput
        }
    }

    private func bestFormat(on camera: AVCaptureDevice, policy: CapturePolicy) -> AVCaptureDevice.Format? {
        camera.formats
            .filter { format in
                let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let supportsResolution = Int(dimensions.width) >= policy.resolution.dimensions[0]
                    && Int(dimensions.height) >= policy.resolution.dimensions[1]
                let supportsFPS = format.videoSupportedFrameRateRanges.contains { range in
                    range.maxFrameRate >= Double(policy.fps)
                }
                return supportsResolution && supportsFPS
            }
            .sorted { lhs, rhs in
                let left = CMVideoFormatDescriptionGetDimensions(lhs.formatDescription)
                let right = CMVideoFormatDescriptionGetDimensions(rhs.formatDescription)
                let leftPixels = Int(left.width) * Int(left.height)
                let rightPixels = Int(right.width) * Int(right.height)
                return leftPixels < rightPixels
            }
            .first
    }

    private func lockExposureFocusAndWhiteBalance(on camera: AVCaptureDevice) {
        if camera.isExposureModeSupported(.custom) {
            let requestedDuration = CMTime(seconds: 1.0 / 1_000.0, preferredTimescale: 1_000_000)
            let duration = max(camera.activeFormat.minExposureDuration, min(camera.activeFormat.maxExposureDuration, requestedDuration))
            let iso = max(camera.activeFormat.minISO, min(camera.activeFormat.maxISO, 200))
            camera.setExposureModeCustom(duration: duration, iso: iso)
        } else if camera.isExposureModeSupported(.locked) {
            camera.exposureMode = .locked
        }

        if camera.isFocusModeSupported(.locked) {
            camera.setFocusModeLocked(lensPosition: 0.7)
        }

        if camera.isWhiteBalanceModeSupported(.locked) {
            camera.whiteBalanceMode = .locked
        }
    }

    private func configureMovieOutput(policy: CapturePolicy, videoRotationAngleDegrees: Int) {
        guard let connection = movieOutput.connection(with: .video) else {
            return
        }

        let rotationAngle = CGFloat(videoRotationAngleDegrees)
        if connection.isVideoRotationAngleSupported(rotationAngle) {
            connection.videoRotationAngle = rotationAngle
        }

        let codec = videoCodec(for: policy.codec)
        if movieOutput.availableVideoCodecTypes.contains(codec) {
            movieOutput.setOutputSettings([AVVideoCodecKey: codec], for: connection)
        }
    }

    private func videoCodec(for format: CaptureFormat) -> AVVideoCodecType {
        switch format {
        case .hevc:
            return .hevc
        case .prores422lt:
            return .proRes422LT
        }
    }

    private func sessionPreset(for resolution: CaptureResolution) -> AVCaptureSession.Preset {
        switch resolution {
        case .hd720p:
            return .hd1280x720
        case .hd1080p:
            return .hd1920x1080
        case .uhd4K:
            return .hd4K3840x2160
        }
    }

    private func writeSidecar(for descriptor: CapturePackageDescriptor, outputFileURL _: URL, finishedAt: Date) throws {
        guard let sidecarURL = activeSidecarURL else {
            throw CameraCaptureControllerError.noConfiguredPackage
        }

        try FileManager.default.createDirectory(
            at: sidecarURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let startedAt = recordingStartedAt ?? descriptor.startedAt
        let sidecar = CaptureSidecar(
            deviceTier: .standard,
            deviceModel: Self.deviceModelIdentifier(),
            fps: descriptor.expectedFPS,
            format: descriptor.expectedFormat,
            resolution: descriptor.expectedResolution,
            orientation: descriptor.expectedOrientation,
            captureDeviceOrientation: descriptor.captureDeviceOrientation,
            videoRotationAngleDegrees: descriptor.videoRotationAngleDegrees,
            recordingStartedAt: Self.iso8601String(from: startedAt),
            recordingDurationS: max(0.0, finishedAt.timeIntervalSince(startedAt)),
            cameraPosition: "back",
            cameraLens: activeDevice?.deviceType.rawValue,
            locked: lockedCaptureSnapshot(),
            intrinsics: estimatedIntrinsics(for: descriptor.expectedResolution),
            gravity: latestGravity,
            ondevicePoseTrack: descriptor.onDevicePoseTrackRelativePath,
            captureQuality: CaptureQuality(
                grade: .warn,
                reasons: [
                    "arkit_seed_missing",
                    "court_plane_missing",
                    "intrinsics_estimated_from_fov",
                ]
            )
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(sidecar)
        try data.write(to: sidecarURL, options: .atomic)
    }

    private func lockedCaptureSnapshot() -> LockedCapture {
        guard let device = activeDevice else {
            return LockedCapture(exposureS: 0.001, iso: 0.0, focus: 0.0, wbLocked: false)
        }

        return LockedCapture(
            exposureS: device.exposureDuration.seconds,
            iso: Double(device.iso),
            focus: Double(device.lensPosition),
            wbLocked: device.whiteBalanceMode == .locked
        )
    }

    private func estimatedIntrinsics(for resolution: [Int]) -> CameraIntrinsics {
        let width = Double(resolution.first ?? 0)
        let height = Double(resolution.dropFirst().first ?? 0)
        guard let activeDevice, width > 0.0, height > 0.0 else {
            return CameraIntrinsics(fx: 0.0, fy: 0.0, cx: width / 2.0, cy: height / 2.0, source: "unavailable")
        }

        let horizontalFOVRadians = Double(activeDevice.activeFormat.videoFieldOfView) * .pi / 180.0
        let focalLengthPixels = (width / 2.0) / tan(horizontalFOVRadians / 2.0)
        return CameraIntrinsics(
            fx: focalLengthPixels,
            fy: focalLengthPixels,
            cx: width / 2.0,
            cy: height / 2.0,
            source: "avfoundation_fov_estimate"
        )
    }

    private static func iso8601String(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }

    private static func deviceModelIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafePointer(to: &systemInfo.machine) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: 1) { machinePointer in
                String(validatingUTF8: machinePointer) ?? "unknown"
            }
        }
    }
}

extension CameraCaptureController: AVCaptureFileOutputRecordingDelegate {
    public func fileOutput(
        _: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from _: [AVCaptureConnection],
        error: Error?
    ) {
        if let error {
            onRecordingFinished?(.failure(error))
            return
        }

        guard let descriptor = activeDescriptor else {
            onRecordingFinished?(.failure(CameraCaptureControllerError.noConfiguredPackage))
            return
        }

        do {
            try writeSidecar(for: descriptor, outputFileURL: outputFileURL, finishedAt: Date())
        } catch {
            onRecordingFinished?(.failure(error))
            return
        }

        let result = CameraRecordingResult(descriptor: descriptor, clipURL: outputFileURL)
        lastRecordingResult = result
        recordingStartedAt = nil
        onRecordingFinished?(.success(result))
    }
}
#endif
