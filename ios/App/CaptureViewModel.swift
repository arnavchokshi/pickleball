import Foundation
import SwiftUI
import UIKit
@preconcurrency import AVFoundation
import PickleballCapture
import PickleballCore
import PickleballFastTier
import PickleballGuidance
import PickleballReplay

@MainActor
final class CaptureViewModel: ObservableObject {
    enum Status: Equatable {
        case idle
        case requestingAccess
        case ready
        case recording
        case finished(String)
        case blocked(String)
    }

    @Published var selectedMode: CaptureMode = .standard60
    @Published private(set) var captureDeviceOrientation: CaptureDeviceOrientation = .landscapeRight
    @Published private(set) var descriptor: CapturePackageDescriptor?
    @Published private(set) var permissions = CapturePermissionSnapshot(camera: .notDetermined, microphone: .notDetermined)
    @Published private(set) var replayBenchmarkStatus: ReplayBenchmarkStatus = .idle
    @Published private(set) var status: Status = .idle
    @Published private(set) var capturePolicyEnforcement: CapturePolicyEnforcementReport?
    @Published private(set) var recordFlowPhase: DinkVisionRecordFlowPhase = .idle
    @Published private(set) var recordingStartedAt: Date?
    @Published private(set) var profileFlow = ProfileCaptureFlowState.h0Checklist()
    @Published var profilePlayerHeightCM: Double = 180
    @Published var profileBallSKU: String = "outdoor_yellow"

    // W3-LIVE-MLP live overlay state -- see PickleballGuidance.LiveGuidanceEvaluator,
    // PickleballFastTier.CourtDotMapBuilder/LiveBallIndicatorPolicy/PostStopPreviewBuilder.
    @Published private(set) var liveGuidanceState: LiveGuidanceState?
    @Published private(set) var courtDotMapPoints: [CourtDotMapPoint] = []
    @Published private(set) var playerFootRings: [LivePlayerFootRing] = []
    @Published private(set) var liveOverlayVideoAspectRatio: Double = 16.0 / 9.0
    @Published private(set) var courtOverlayStatusText: String = "Court map: starting…"
    @Published private(set) var courtOverlayDetailText: String = ""
    @Published private(set) var ballIndicatorState: LiveBallIndicatorState = .comingSoon
    @Published private(set) var ballTrailPoints: [LiveBallTrailPoint] = []
    @Published private(set) var ballContactMarkers: [LiveBallContactMarker] = []
    @Published private(set) var postStopSummary: PostStopPreviewSummary?

    let controller: CameraCaptureControlling
    private let requestPermissions: () async -> CapturePermissionSnapshot
    private var didAutoStartReplayBenchmark = false
    private var guidancePollingTask: Task<Void, Never>?
    private var recentPlayerCountSamples: [Int] = []
    private var lastFootRingDetectionFrameIndex: Int?
    private var ballOverlayTracker = LiveBallOverlayTracker()

    static let modes: [CaptureMode] = [.standard60, .swing120, .ballPhysics240, .quality4K60]

    init(
        controller: CameraCaptureControlling = QueuedCameraCaptureController(),
        requestPermissions: @escaping () async -> CapturePermissionSnapshot = {
            await CameraCaptureController.requestPermissions()
        }
    ) {
        self.controller = controller
        self.requestPermissions = requestPermissions
    }

    deinit {
        guidancePollingTask?.cancel()
    }

    var session: AVCaptureSession {
        controller.session
    }

    var isRecording: Bool {
        status == .recording
    }

    var modeSummary: String {
        let capture = CaptureSessionScaffold(
            mode: selectedMode,
            deviceTier: .standard,
            capabilities: .hevcOnly,
            orientation: CaptureOrientationPolicy.captureOrientation(for: captureDeviceOrientation)
        )
        let dimensions = capture.resolution.dimensions(for: capture.orientation)
        return "\(capture.requestedFPS) fps · \(dimensions[0])x\(dimensions[1]) · \(capture.format.rawValue)"
    }

    var captureOrientationTitle: String {
        switch CaptureOrientationPolicy.captureOrientation(for: captureDeviceOrientation) {
        case .portrait:
            return "Portrait"
        case .landscape:
            return "Landscape"
        }
    }

    var videoRotationTitle: String {
        "\(CaptureOrientationPolicy.rotationAngleDegrees(for: captureDeviceOrientation))°"
    }

    var isReplayBenchmarkRunning: Bool {
        replayBenchmarkStatus == .running
    }

    var replayBenchmarkTitle: String {
        switch replayBenchmarkStatus {
        case .idle:
            return "Replay"
        case .missingInputs:
            return "No inputs"
        case .running:
            return "Running"
        case .finished:
            return "Done"
        case .failed:
            return "Failed"
        }
    }

    var replayBenchmarkDetail: String {
        switch replayBenchmarkStatus {
        case .idle:
            return "Vision"
        case .missingInputs:
            return "Manifest"
        case .running:
            return "On phone"
        case .finished(let summaries):
            let frames = summaries.reduce(0) { $0 + $1.processedFrameCount }
            return "\(summaries.count) clips · \(frames) frames"
        case .failed(let message):
            return message
        }
    }

    var replayBenchmarkOutputPath: String? {
        switch replayBenchmarkStatus {
        case .finished(let summaries):
            return summaries.first?.tracksPath.components(separatedBy: "/Documents/").last
        case .idle, .missingInputs, .running, .failed:
            return nil
        }
    }

    var previewRotationAngle: Int {
        CaptureOrientationPolicy.rotationAngleDegrees(for: captureDeviceOrientation)
    }

    var capturePolicyStatusText: String {
        guard let capturePolicyEnforcement else {
            return "Policy pending"
        }
        if capturePolicyEnforcement.isCompliant {
            return "Capture policy locked"
        }
        return "Policy issue: \(capturePolicyEnforcement.violations.first ?? "unknown")"
    }

    var policyChips: [DinkVisionPolicyChip] {
        DinkVisionPolicyChipMapper.chips(for: capturePolicyEnforcement)
    }

    func prepare() async {
        status = .requestingAccess
        permissions = await requestPermissions()
        await configure()
        await autoStartReplayBenchmarkIfRequested()
    }

    func configure() async {
        do {
            descriptor = try await controller.configure(
                mode: selectedMode,
                deviceTier: .standard,
                capabilities: .hevcOnly,
                captureDeviceOrientation: captureDeviceOrientation,
                sessionID: CameraCaptureController.defaultSessionID(),
                packageRootURL: CameraCaptureController.defaultPackageRootURL()
            )
            controller.onRecordingFinished = { [weak self] result in
                switch result {
                case .success(let recording):
                    Task { @MainActor [weak self] in
                        self?.status = .finished(recording.descriptor.clipRelativePath)
                        self?.recordFlowPhase = .done(sessionID: recording.descriptor.sessionID)
                        self?.recordingStartedAt = nil
                        await self?.buildPostStopSummary(for: recording)
                    }
                case .failure(let error):
                    let message = String(describing: error)
                    Task { @MainActor [weak self] in
                        self?.status = .blocked(message)
                        self?.recordFlowPhase = .blocked(message)
                        self?.recordingStartedAt = nil
                    }
                }
            }
            controller.setLiveCourtOverlayHandlers(
                onFrame: { [weak self] frame in
                    Task { @MainActor [weak self] in
                        self?.handleLiveOverlayFrame(frame)
                    }
                },
                onStatusChange: { [weak self] overlayStatus in
                    Task { @MainActor [weak self] in
                        self?.handleLiveOverlayStatus(overlayStatus)
                    }
                }
            )
            await controller.startPreview()
            capturePolicyEnforcement = await controller.currentPolicyEnforcementReport()
            status = .ready
            recordFlowPhase = .ready
            startLiveGuidancePollingIfNeeded()
        } catch {
            let message = Self.message(for: error)
            status = .blocked(message)
            recordFlowPhase = Self.recordFlowBlockedPhase(for: error, message: message)
        }
    }

    /// Pre-record capture-quality guidance (W3-LIVE-MLP surface 1): polls
    /// real device readbacks every 0.5s and re-evaluates them through
    /// `LiveGuidanceEvaluator`. Runs continuously once configured (harmless
    /// while recording too, since none of the polled signals are
    /// recording-destructive reads).
    private func startLiveGuidancePollingIfNeeded() {
        guard guidancePollingTask == nil else {
            return
        }
        guidancePollingTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                let sample = await self.controller.currentLiveGuidanceSample()
                self.liveGuidanceState = LiveGuidanceEvaluator.evaluate(sample)
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
    }

    private func handleLiveOverlayFrame(_ frame: LiveCourtOverlayFrame) {
        ingestLiveOverlayFrame(frame)
    }

    func ingestLiveOverlayFrame(_ frame: LiveCourtOverlayFrame) {
        courtDotMapPoints = frame.points
        liveOverlayVideoAspectRatio = frame.videoAspectRatio
        if frame.detectorInvoked {
            lastFootRingDetectionFrameIndex = frame.frameIndex
            recentPlayerCountSamples.append(frame.points.count)
            if recentPlayerCountSamples.count > 20 {
                recentPlayerCountSamples.removeFirst(recentPlayerCountSamples.count - 20)
            }
        }
        if frame.playerFootRings.isEmpty {
            playerFootRings = LivePlayerFootRingBuilder.build(
                points: frame.points,
                frameIndex: frame.frameIndex,
                lastDetectionFrameIndex: lastFootRingDetectionFrameIndex
            )
        } else {
            playerFootRings = frame.playerFootRings
        }
        // v0: the live engine does not emit a trained ball state, so this
        // remains "coming soon" and produces no trail/contact visuals.
        // Future trained live-ball output must still pass through
        // LiveBallIndicatorPolicy before it is attached to `LiveCourtOverlayFrame`.
        let nextBallState = frame.ballState ?? LiveBallIndicatorPolicy.evaluate(rawConfidence: nil, rawNormalizedX: nil, rawNormalizedY: nil)
        ballIndicatorState = nextBallState
        let ballOverlay = ballOverlayTracker.update(frameIndex: frame.frameIndex, ballState: nextBallState)
        ballTrailPoints = ballOverlay.trailPoints
        ballContactMarkers = ballOverlay.contactMarkers
    }

    private func handleLiveOverlayStatus(_ overlayStatus: LiveCourtOverlayStatus) {
        switch overlayStatus {
        case .idle:
            courtOverlayStatusText = "Court map: idle"
            courtOverlayDetailText = ""
        case .running:
            courtOverlayStatusText = "Court map: live (screen-space proxy)"
            courtOverlayDetailText = ""
        case .modelUnavailable(let message):
            courtOverlayStatusText = "Court map: detector not installed"
            courtOverlayDetailText = message
            playerFootRings = []
            lastFootRingDetectionFrameIndex = nil
            resetBallOverlay()
        case .failed(let message):
            courtOverlayStatusText = "Court map: error"
            courtOverlayDetailText = message
            playerFootRings = []
            lastFootRingDetectionFrameIndex = nil
            resetBallOverlay()
        }
    }

    private func resetBallOverlay() {
        ballOverlayTracker.reset()
        ballIndicatorState = .comingSoon
        ballTrailPoints = []
        ballContactMarkers = []
    }

    /// Post-stop preview (W3-LIVE-MLP surface 4, <10s gate). Reads back the
    /// REAL just-written `capture_sidecar.json` (duration, fps, capture
    /// quality are all real device readbacks already persisted by
    /// `CameraCaptureController.writeSidecar`) rather than re-deriving them
    /// client-side, and pairs it with the player-count samples collected
    /// from actual live cadence-scheduled detections during the just-finished
    /// recording.
    private func buildPostStopSummary(for recording: CameraRecordingResult) async {
        let startedAt = CFAbsoluteTimeGetCurrent()
        let sidecarURL = CameraCaptureController.defaultPackageRootURL()
            .appendingPathComponent(recording.descriptor.sidecarRelativePath)
        guard let data = try? Data(contentsOf: sidecarURL),
              let sidecar = try? JSONDecoder().decode(CaptureSidecar.self, from: data) else {
            return
        }
        let elapsedBuildSeconds = max(0, CFAbsoluteTimeGetCurrent() - startedAt)
        postStopSummary = PostStopPreviewBuilder.summarize(
            durationSeconds: sidecar.recordingDurationS ?? 0,
            requestedFPS: sidecar.fps,
            measuredFPS: nil,
            captureQuality: sidecar.captureQuality,
            sampledFrameDetectionCounts: recentPlayerCountSamples,
            elapsedBuildSeconds: elapsedBuildSeconds
        )
    }

    func updateOrientation(isLandscapeViewport: Bool) async {
        let updatedOrientation = Self.deviceOrientation(fallbackIsLandscape: isLandscapeViewport)
        guard updatedOrientation != captureDeviceOrientation else {
            return
        }

        captureDeviceOrientation = updatedOrientation
        guard !isRecording else {
            return
        }

        await configure()
    }

    func toggleRecording() async {
        do {
            if isRecording {
                recordFlowPhase = .saving
                try await controller.stopRecording()
                return
            }

            guard canStartRecording else {
                if case .blocked = status {
                    return
                }
                status = .blocked("Camera is not ready")
                recordFlowPhase = .blocked("Camera is not ready")
                return
            }

            recentPlayerCountSamples = []
            resetBallOverlay()
            postStopSummary = nil
            controller.setProfileCapturePayload(profileFlow.payload)
            descriptor = try await controller.startRecording()
            let startedAt = Date()
            recordingStartedAt = startedAt
            status = .recording
            recordFlowPhase = .recording(startedAt: startedAt)
        } catch {
            let message = Self.message(for: error)
            status = .blocked(message)
            recordFlowPhase = Self.recordFlowBlockedPhase(for: error, message: message)
        }
    }

    func recordCurrentProfileStep(artifactRef: String? = nil, metadata: [String: String] = [:]) {
        profileFlow.recordCurrentStep(artifactRef: artifactRef, metadata: metadata)
        controller.setProfileCapturePayload(profileFlow.payload)
    }

    func completeCurrentProfileStepFromUI() {
        guard let kind = profileFlow.currentStep?.kind else {
            return
        }
        let directory = descriptor?.directoryRelativePath ?? "captures/profile_setup_pending"
        switch kind {
        case .emptyCourtClip:
            recordProfileStep(
                kind,
                artifactRef: descriptor?.clipRelativePath ?? "\(directory)/empty_court_clip.mov",
                metadata: ["clip_type": "empty_court"]
            )
        case .calibrationGridSweep:
            recordProfileStep(
                kind,
                artifactRef: "\(directory)/calibration_grid_sweep.json",
                metadata: ["pattern": "charuco_or_aprilgrid"]
            )
        case .paddleOrbit:
            recordProfileStep(
                kind,
                artifactRef: "\(directory)/paddle_orbit.mov",
                metadata: ["orbit": "single_paddle"]
            )
        case .playerHeightEntry:
            recordProfileStep(
                kind,
                metadata: ["height_cm": "\(Int(profilePlayerHeightCM.rounded()))"]
            )
        case .ballPick:
            recordProfileStep(
                kind,
                metadata: ["sku": profileBallSKU]
            )
        }
    }

    func recordProfileStep(
        _ kind: ProfileCaptureStepKind,
        artifactRef: String? = nil,
        metadata: [String: String] = [:]
    ) {
        profileFlow.recordStep(kind, artifactRef: artifactRef, metadata: metadata)
        controller.setProfileCapturePayload(profileFlow.payload)
    }

    func runReplayBenchmarkFromStagedManifest() async {
        guard !isReplayBenchmarkRunning else {
            return
        }

        do {
            let documentsURL = try Self.documentsDirectoryURL()
            let manifestURL = documentsURL.appendingPathComponent("manifest.json")
            guard FileManager.default.fileExists(atPath: manifestURL.path) else {
                replayBenchmarkStatus = .missingInputs
                return
            }

            replayBenchmarkStatus = .running
            let outputRootURL = documentsURL
                .appendingPathComponent("replay_benchmarks", isDirectory: true)
                .appendingPathComponent(Self.replayRunID(), isDirectory: true)
            let processors = Self.replayBenchmarkProcessors(documentsURL: documentsURL)
            let summaries = try await Task.detached(priority: .userInitiated) {
                let manifest = try ReplayInputManifest.load(from: manifestURL)
                let clips = manifest.resolvedClips(relativeTo: manifestURL.deletingLastPathComponent())
                var summaries: [ReplayPersonBenchmarkRunSummary] = []
                for processor in processors {
                    let runner = ReplayPersonBenchmarkRunner(processor: processor)
                    for clip in clips {
                        do {
                            let summary = try await runner.run(clip: clip, outputRootURL: outputRootURL)
                            summaries.append(summary)
                        } catch {
                            try Self.writeReplayBenchmarkFailure(
                                candidate: processor.candidate.rawValue,
                                clipID: clip.clipID,
                                error: error,
                                outputRootURL: outputRootURL
                            )
                        }
                    }
                }
                if summaries.isEmpty {
                    throw ReplayBenchmarkBatchError.noSuccessfulRuns
                }
                return summaries
            }.value
            replayBenchmarkStatus = .finished(summaries)
        } catch is CancellationError {
            replayBenchmarkStatus = .idle
        } catch {
            replayBenchmarkStatus = .failed(Self.message(for: error))
        }
    }

    nonisolated private static func message(for error: Error) -> String {
        switch error {
        case CameraCaptureControllerError.cameraUnavailable:
            return "Camera unavailable"
        case CameraCaptureControllerError.permissionDenied:
            return "Camera or microphone access needed"
        case CameraCaptureControllerError.unsupportedFrameRate(let fps):
            return "\(fps) fps unavailable on this camera"
        case CameraCaptureControllerError.landscapeRequired:
            return "Rotate to landscape"
        case CameraCaptureControllerError.alreadyRecording:
            return "Already recording"
        case CameraCaptureControllerError.notRecording:
            return "Not recording"
        default:
            return String(describing: error)
        }
    }

    nonisolated private static func recordFlowBlockedPhase(for error: Error, message: String) -> DinkVisionRecordFlowPhase {
        if case CameraCaptureControllerError.permissionDenied = error {
            return .permissionDenied(message)
        }
        return .blocked(message)
    }

    private var canStartRecording: Bool {
        switch status {
        case .ready, .finished:
            return true
        case .idle, .requestingAccess, .recording, .blocked:
            return false
        }
    }

    private func autoStartReplayBenchmarkIfRequested() async {
        guard !didAutoStartReplayBenchmark else {
            return
        }
        guard ProcessInfo.processInfo.arguments.contains("--run-replay-benchmark") else {
            return
        }
        didAutoStartReplayBenchmark = true
        await runReplayBenchmarkFromStagedManifest()
    }

    private static func deviceOrientation(fallbackIsLandscape: Bool) -> CaptureDeviceOrientation {
        switch UIDevice.current.orientation {
        case .portrait:
            return .portrait
        case .portraitUpsideDown:
            return .portraitUpsideDown
        case .landscapeLeft:
            return .landscapeLeft
        case .landscapeRight:
            return .landscapeRight
        default:
            return fallbackIsLandscape ? .landscapeRight : .portrait
        }
    }

    private static func documentsDirectoryURL() throws -> URL {
        try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
    }

    private static func replayRunID() -> String {
        "person_replay_benchmark_\(Int(Date().timeIntervalSince1970))"
    }

    private static func replayBenchmarkProcessors(documentsURL: URL) -> [ReplayPersonBenchmarkProcessor] {
        let modelsRoot = documentsURL.appendingPathComponent("models_coreml", isDirectory: true)
        var processors: [ReplayPersonBenchmarkProcessor] = []

        Self.appendYolo26Processor(
            to: &processors,
            modelsRoot: modelsRoot,
            modelDirectory: "yolo26n_img416_int8",
            modelName: "yolo26n",
            candidate: .yolo26nInt8EveryFrame,
            imageSize: 416
        )
        Self.appendYolo26Processor(
            to: &processors,
            modelsRoot: modelsRoot,
            modelDirectory: "yolo26n_img416_int8",
            modelName: "yolo26n",
            candidate: .yolo26nInt8Detect2Track30,
            imageSize: 416,
            maxTrackAgeFrames: 30,
            detectionIntervalFrames: 2
        )
        Self.appendYolo26Processor(
            to: &processors,
            modelsRoot: modelsRoot,
            modelDirectory: "yolo26n_img512_int8",
            modelName: "yolo26n",
            candidate: .yolo26nInt8Img512EveryFrame,
            imageSize: 512
        )
        Self.appendYolo26Processor(
            to: &processors,
            modelsRoot: modelsRoot,
            modelDirectory: "yolo26n_img640_int8",
            modelName: "yolo26n",
            candidate: .yolo26nInt8Img640EveryFrame,
            imageSize: 640
        )
        Self.appendYolo26Processor(
            to: &processors,
            modelsRoot: modelsRoot,
            modelDirectory: "yolo26s_img416_int8",
            modelName: "yolo26s",
            candidate: .yolo26sInt8EveryFrame,
            imageSize: 416
        )
        Self.appendYolo26Processor(
            to: &processors,
            modelsRoot: modelsRoot,
            modelDirectory: "yolo26m_img416_int8",
            modelName: "yolo26m",
            candidate: .yolo26mInt8Img416EveryFrame,
            imageSize: 416
        )

        if let yolo11n = Self.replayModelURL(
            modelsRoot: modelsRoot,
            modelDirectory: "yolo11n_img416_int8_nms",
            modelName: "yolo11n"
        ) {
            processors.append(
                .coreML(
                    CoreMLPersonDetectorConfiguration(
                        candidate: .yolo11nInt8Fallback,
                        modelURL: yolo11n,
                        outputFormat: .yolo11NMS,
                        inputWidth: 416,
                        inputHeight: 416,
                        minConfidence: 0.10
                    )
                )
            )
        }

        processors.append(.visionHumanRectangles)
        return processors
    }

    private static func appendYolo26Processor(
        to processors: inout [ReplayPersonBenchmarkProcessor],
        modelsRoot: URL,
        modelDirectory: String,
        modelName: String,
        candidate: OnDevicePersonCandidate,
        imageSize: Int,
        maxTrackAgeFrames: Int = 8,
        detectionIntervalFrames: Int = 1
    ) {
        guard let modelURL = Self.replayModelURL(
            modelsRoot: modelsRoot,
            modelDirectory: modelDirectory,
            modelName: modelName
        ) else {
            return
        }
        processors.append(
            .coreML(
                CoreMLPersonDetectorConfiguration(
                    candidate: candidate,
                    modelURL: modelURL,
                    outputFormat: .yolo26EndToEnd,
                    inputWidth: imageSize,
                    inputHeight: imageSize,
                    minConfidence: 0.10,
                    maxTrackAgeFrames: maxTrackAgeFrames,
                    detectionIntervalFrames: detectionIntervalFrames
                )
            )
        )
    }

    private static func replayModelURL(modelsRoot: URL, modelDirectory: String, modelName: String) -> URL? {
        let root = modelsRoot.appendingPathComponent(modelDirectory, isDirectory: true)
        let compiled = root.appendingPathComponent("\(modelName).mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: compiled.path) {
            return compiled
        }
        let package = root.appendingPathComponent("\(modelName).mlpackage", isDirectory: true)
        if FileManager.default.fileExists(atPath: package.path) {
            return package
        }
        return nil
    }

    nonisolated private static func writeReplayBenchmarkFailure(
        candidate: String,
        clipID: String,
        error: Error,
        outputRootURL: URL
    ) throws {
        let failureURL = outputRootURL
            .appendingPathComponent(clipID, isDirectory: true)
            .appendingPathComponent(candidate, isDirectory: true)
            .appendingPathComponent("error.json")
        try FileManager.default.createDirectory(
            at: failureURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let payload: [String: Any] = [
            "schema_version": 1,
            "artifact_type": "pickleball_replay_person_benchmark_failure",
            "clip_id": clipID,
            "candidate": candidate,
            "message": Self.message(for: error),
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: failureURL, options: [.atomic])
    }
}

private enum ReplayBenchmarkBatchError: Error {
    case noSuccessfulRuns
}

enum ReplayBenchmarkStatus: Equatable {
    case idle
    case missingInputs
    case running
    case finished([ReplayPersonBenchmarkRunSummary])
    case failed(String)
}

extension CaptureMode {
    var title: String {
        switch self {
        case .standard60:
            return "Standard"
        case .swing120:
            return "Swing"
        case .ballPhysics240:
            return "Ball"
        case .quality4K60:
            return "Quality"
        }
    }
}
