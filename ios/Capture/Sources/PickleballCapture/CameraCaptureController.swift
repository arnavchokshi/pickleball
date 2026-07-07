#if os(iOS)
@preconcurrency import AVFoundation
import Darwin
import Foundation
import PickleballCore
import PickleballGuidance

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
    /// Live court-dot map engine (W3-LIVE-MLP surface 2). Attached
    /// best-effort alongside the movie output in `configure()`; a failure to
    /// attach or to find a compiled detector model never blocks recording.
    public let liveOverlayEngine = LiveCourtOverlayEngine()

    private let movieOutput = AVCaptureMovieFileOutput()
    private let motionSampler = CaptureMotionSampler()
    private let arSessionProvider: ARSessionProviding
    private let cameraOwnership = CameraResourceOwnership()
    private let frameMotionRecorder = CoreMotionFrameSidecarRecorder()
    private var activeDevice: AVCaptureDevice?
    private var activePolicy: CapturePolicy?
    private var activePolicyEnforcement: CapturePolicyEnforcementReport?
    private var activeProfileCapture: ProfileCapturePayload?
    private var activeAVCaptureToken: AVCaptureCameraOwnershipToken?
    private var latestSetupPass: ARKitSetupPassSidecar?
    private var activePackageRootURL: URL?
    private var activeCaptureDeviceOrientation: CaptureDeviceOrientation?
    private var activeClipURL: URL?
    private var activeSidecarURL: URL?
    private var recordingStartedAt: Date?
    private var sessionIDFactory = CaptureSessionIDFactory()

    public init(arSessionProvider: ARSessionProviding = DefaultARSessionProviderFactory.make()) {
        self.arSessionProvider = arSessionProvider
        super.init()
        liveOverlayEngine.onFramePresentationTimestamp = { [frameMotionRecorder, motionSampler] _, presentationSeconds in
            frameMotionRecorder.recordVideoFrame(
                ptsS: presentationSeconds,
                gravity: motionSampler.latestGravity
            )
        }
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
        // Best-effort: never throws, never blocks recording if the session
        // declines the extra output or no detector model is installed yet.
        liveOverlayEngine.attach(to: session, rotationDegrees: descriptor.videoRotationAngleDegrees)
        session.commitConfiguration()
        didCommitConfiguration = true

        activeDevice = camera
        activePolicy = policy
        activePolicyEnforcement = policyEnforcementReport(policy: policy, descriptor: descriptor)
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
        guard activeAVCaptureToken == nil,
              let token = try? cameraOwnership.beginAVCapture() else {
            return
        }
        motionSampler.start()
        session.startRunning()
        activeAVCaptureToken = token
    }

    public func stopPreview() {
        guard session.isRunning else {
            return
        }
        session.stopRunning()
        activeAVCaptureToken?.release()
        activeAVCaptureToken = nil
        motionSampler.stop()
    }

    public var latestGravity: [Double] {
        motionSampler.latestGravity
    }

    public func performARKitSetupPass(timeoutSeconds: Double = 4.0) async -> ARKitSetupPassSidecar {
        if session.isRunning {
            session.stopRunning()
            activeAVCaptureToken?.release()
            activeAVCaptureToken = nil
        }
        motionSampler.start()
        let runner = ARKitSetupPassRunner(
            provider: arSessionProvider,
            ownership: cameraOwnership,
            gravityProvider: { [motionSampler] in
                motionSampler.latestGravity
            }
        )
        let setupPass = await runner.run(timeoutSeconds: timeoutSeconds)
        latestSetupPass = setupPass
        return setupPass
    }

    /// Real, live-readback signals for the pre-record capture-quality
    /// guidance screen (W3-LIVE-MLP surface 1) -- see `LiveGuidanceEvaluator`
    /// in `PickleballGuidance` for how these become pass/warn/unavailable
    /// checks. Every field here is a direct AVFoundation/CoreMotion
    /// readback or a value already computed from the active policy; nothing
    /// is estimated or fabricated. Returns an all-`nil` sample (every check
    /// renders `.unavailable`) if the session has not been configured yet.
    public func currentLiveGuidanceSample() -> LiveGuidanceSample {
        guard let device = activeDevice, let policy = activePolicy, let descriptor = activeDescriptor else {
            return LiveGuidanceSample()
        }

        let dimensions = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        let minFrameDurationSeconds = device.activeVideoMinFrameDuration.seconds
        let tilt = LiveTiltEstimator.tiltDegrees(
            gravity: latestGravity,
            expectedLevelAxis: LiveTiltEstimator.expectedLevelAxis(for: descriptor.captureDeviceOrientation)
        )

        return LiveGuidanceSample(
            exposureTargetOffsetEV: Double(device.exposureTargetOffset),
            isExposureLocked: device.exposureMode == .locked || device.exposureMode == .custom,
            shutterSeconds: device.exposureDuration.seconds,
            minimumSharpShutterSeconds: LockedCapturePolicy.slowestAllowedShutterSeconds,
            tiltFromLevelDegrees: tilt,
            requestedFPS: policy.fps,
            configuredFPS: minFrameDurationSeconds > 0 ? 1.0 / minFrameDurationSeconds : nil,
            expectedResolution: policy.resolution.dimensions(for: policy.orientation),
            configuredResolution: [Int(dimensions.width), Int(dimensions.height)],
            setupTipReasons: setupTipReasons()
        )
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
        frameMotionRecorder.beginRecording()
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

    public func currentPolicyEnforcementReport() -> CapturePolicyEnforcementReport? {
        activePolicyEnforcement
    }

    public func setProfileCapturePayload(_ payload: ProfileCapturePayload?) {
        activeProfileCapture = payload
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
        if connection.isVideoStabilizationSupported {
            connection.preferredVideoStabilizationMode = .off
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
        guard activeSidecarURL != nil, let packageRootURL = activePackageRootURL else {
            throw CameraCaptureControllerError.noConfiguredPackage
        }

        let startedAt = recordingStartedAt ?? descriptor.startedAt
        let context = CaptureSidecarWriteContext(
            deviceTier: .standard,
            deviceModel: Self.deviceModelIdentifier(),
            cameraPosition: "back",
            cameraLens: activeDevice?.deviceType.rawValue,
            locked: lockedCaptureSnapshot(),
            intrinsics: estimatedIntrinsics(for: descriptor.expectedResolution),
            gravity: latestGravity,
            arkit: ARCaptureSidecarPayload(
                setupPass: latestSetupPass ?? .unavailable(
                    reason: "arkit_setup_pass_not_run",
                    gravity: latestGravity
                ),
                frameSamples: frameMotionRecorder.frameSamples()
            ),
            policyEnforcement: activePolicyEnforcement,
            profileCapture: activeProfileCapture,
            captureQuality: CaptureQuality(
                grade: .warn,
                reasons: [
                    "arkit_seed_missing",
                    "court_plane_missing",
                    "intrinsics_estimated_from_fov",
                ]
            )
        )
        try CaptureSidecarWriter.write(
            descriptor: descriptor,
            packageRootURL: packageRootURL,
            recordingStartedAt: startedAt,
            finishedAt: finishedAt,
            context: context
        )
    }

    private func setupTipReasons() -> [String] {
        guard let latestSetupPass else {
            return [
                "arkit_seed_missing",
                "court_plane_missing",
                "intrinsics_estimated_from_fov",
            ]
        }
        guard latestSetupPass.status == .available else {
            return [
                "arkit_seed_missing",
                "court_plane_missing",
                "intrinsics_estimated_from_fov",
                latestSetupPass.unavailableReason ?? "arkit_setup_pass_unavailable",
            ]
        }
        return latestSetupPass.courtPlane == nil ? ["court_plane_missing"] : []
    }

    private func policyEnforcementReport(
        policy: CapturePolicy,
        descriptor: CapturePackageDescriptor
    ) -> CapturePolicyEnforcementReport {
        CapturePolicyEnforcer.evaluate(
            policy: policy,
            achieved: CapturePolicyAchievedState(
                fps: achievedFPS(),
                resolution: activeResolution(),
                format: policy.codec,
                orientation: descriptor.expectedOrientation,
                electronicStabilizationEnabled: electronicStabilizationEnabled(),
                exposureLocked: isExposureLocked(),
                focusLocked: activeDevice?.focusMode == .locked,
                whiteBalanceLocked: activeDevice?.whiteBalanceMode == .locked
            )
        )
    }

    private func achievedFPS() -> Int? {
        guard let device = activeDevice else {
            return nil
        }
        let seconds = device.activeVideoMinFrameDuration.seconds
        guard seconds > 0 else {
            return nil
        }
        return Int((1.0 / seconds).rounded())
    }

    private func activeResolution() -> [Int]? {
        guard let device = activeDevice else {
            return nil
        }
        let dimensions = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        return [Int(dimensions.width), Int(dimensions.height)]
    }

    private func electronicStabilizationEnabled() -> Bool? {
        guard let connection = movieOutput.connection(with: .video) else {
            return nil
        }
        return connection.activeVideoStabilizationMode != .off
    }

    private func isExposureLocked() -> Bool? {
        guard let device = activeDevice else {
            return nil
        }
        return device.exposureMode == .locked || device.exposureMode == .custom
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
            frameMotionRecorder.endRecording()
            onRecordingFinished?(.failure(error))
            return
        }

        guard let descriptor = activeDescriptor else {
            frameMotionRecorder.endRecording()
            onRecordingFinished?(.failure(CameraCaptureControllerError.noConfiguredPackage))
            return
        }

        do {
            defer {
                frameMotionRecorder.endRecording()
            }
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
