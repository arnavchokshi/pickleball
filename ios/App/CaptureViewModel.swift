import Foundation
import OSLog
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
    private static let recordPathLogger = Logger(
        subsystem: "com.arnavchokshi.pickleball",
        category: "RecordPath"
    )

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
    @Published private(set) var recordTapFeedback = RecordControlTapFeedback.initial
    @Published private(set) var capturePolicyEnforcement: CapturePolicyEnforcementReport?
    @Published private(set) var recordFlowPhase: DinkVisionRecordFlowPhase = .idle
    @Published private(set) var setupPassStatus: DinkVisionSetupPassStatus = .idle
    @Published private(set) var recordingStartedAt: Date?
    @Published private(set) var lastFinishedCapture: CameraRecordingResult?
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
    private var lastSetupPassCompletedAt: Date?
    private var lastSetupPassGravity: [Double]?
    private var preparationTask: Task<Void, Never>?
    private var preparationWatchdogTask: Task<Void, Never>?
    private var activePreparationID: UUID?
    private var recordActionTask: Task<Void, Never>?
    private var recordActionWatchdogTask: Task<Void, Never>?
    private var activeRecordActionID: UUID?
    private let preparationTimeoutNanoseconds: UInt64
    private let orientationResolver: (Bool) -> CaptureDeviceOrientation
    private let announceBlockedState: (String) -> Void

    static let modes: [CaptureMode] = [.standard60, .swing120, .ballPhysics240, .quality4K60]
    static let preparationTimeoutMessage = "Camera setup took too long. Tap Retry."
    static let recordActionTimeoutMessage = "Recording action took too long. Tap Retry."

    init(
        controller: CameraCaptureControlling = QueuedCameraCaptureController(),
        requestPermissions: @escaping () async -> CapturePermissionSnapshot = {
            await CameraCaptureController.requestPermissions()
        },
        preparationTimeoutNanoseconds: UInt64 = 8_000_000_000,
        orientationResolver: ((Bool) -> CaptureDeviceOrientation)? = nil,
        announceBlockedState: @escaping (String) -> Void = { announcement in
            UIAccessibility.post(notification: .announcement, argument: announcement)
        }
    ) {
        self.controller = controller
        self.requestPermissions = requestPermissions
        self.preparationTimeoutNanoseconds = preparationTimeoutNanoseconds
        self.orientationResolver = orientationResolver ?? { fallbackIsLandscape in
            Self.deviceOrientation(fallbackIsLandscape: fallbackIsLandscape)
        }
        self.announceBlockedState = announceBlockedState
    }

    deinit {
        guidancePollingTask?.cancel()
        preparationTask?.cancel()
        preparationWatchdogTask?.cancel()
        recordActionTask?.cancel()
        recordActionWatchdogTask?.cancel()
    }

    var session: AVCaptureSession {
        controller.session
    }

    var isRecording: Bool {
        status == .recording
    }

    var isRecordButtonEnabled: Bool {
        recordButtonAccessibility.isEnabled
    }

    var recordControlState: RecordControlState {
        Self.recordControlState(for: status)
    }

    var recordButtonAccessibility: RecordControlAccessibilityState {
        RecordControlInteractionPolicy.accessibility(for: recordControlState)
    }

    var recordGuidancePresentation: RecordGuidancePresentation? {
        RecordControlInteractionPolicy.guidance(for: recordControlState)
    }

    var blockedReason: String? {
        guard case .blocked(let message) = status else {
            return nil
        }
        return message
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

    var setupPassStatusText: String {
        switch setupPassStatus {
        case .idle:
            return "Aligning"
        case .aligning:
            return "Aligning…"
        case .aligned:
            return "Aligned ✓"
        case .unavailable:
            return "Align skipped"
        }
    }

    var setupPassChipStatus: DinkVisionPolicyChipStatus {
        setupPassStatus == .aligned ? .pass : .warning
    }

    var policyChips: [DinkVisionPolicyChip] {
        DinkVisionPolicyChipMapper.chips(for: capturePolicyEnforcement)
    }

    func prepare(isLandscapeViewport: Bool? = nil) async {
        if let isLandscapeViewport {
            let firstAppearanceOrientation = orientationResolver(isLandscapeViewport)
            if firstAppearanceOrientation != captureDeviceOrientation {
                Self.recordPathLogger.info("First-appearance orientation refresh \(String(describing: self.captureDeviceOrientation), privacy: .public) -> \(String(describing: firstAppearanceOrientation), privacy: .public)")
                captureDeviceOrientation = firstAppearanceOrientation
            } else {
                Self.recordPathLogger.info("First-appearance orientation already current: \(String(describing: firstAppearanceOrientation), privacy: .public)")
            }
        }
        await beginPreparation(requestAccess: true)
    }

    @discardableResult
    func noteRecordControlTap() -> RecordControlTapReaction {
        let reaction = RecordControlInteractionPolicy.tapReaction(for: recordControlState)
        recordTapFeedback = RecordControlTapFeedback(
            sequence: recordTapFeedback.sequence + 1,
            reaction: reaction
        )
        Self.recordPathLogger.info("Record tap visible reaction sequence=\(self.recordTapFeedback.sequence, privacy: .public) kind=\(String(describing: reaction.kind), privacy: .public)")
        return reaction
    }

    func handleRecordTap(registerVisibleFeedback: Bool = true) async {
        if registerVisibleFeedback {
            noteRecordControlTap()
        }
        Self.recordPathLogger.info("Record tap entered state=\(String(describing: self.status), privacy: .public)")
        switch status {
        case .idle, .blocked:
            await prepare()
        case .requestingAccess:
            Self.recordPathLogger.info("Record tap coalescing with bounded preparation")
            await prepare()
        case .ready, .recording, .finished:
            break
        }
        await toggleRecording()
    }

    func configure() async {
        await beginPreparation(requestAccess: false)
    }

    private func beginPreparation(requestAccess: Bool) async {
        if let preparationTask {
            Self.recordPathLogger.info("Preparation request coalesced with active attempt")
            await preparationTask.value
            return
        }

        let attemptID = UUID()
        activePreparationID = attemptID
        transition(to: .requestingAccess, phase: .idle, reason: requestAccess ? "prepare" : "reconfigure")

        let task = Task { @MainActor [weak self] in
            guard let self else {
                return
            }
            await self.runPreparation(attemptID: attemptID, requestAccess: requestAccess)
        }
        preparationTask = task
        let timeoutNanoseconds = preparationTimeoutNanoseconds
        preparationWatchdogTask = Task { @MainActor [weak self] in
            guard let self else {
                return
            }
            do {
                try await Task.sleep(nanoseconds: timeoutNanoseconds)
            } catch {
                return
            }
            self.expirePreparation(attemptID: attemptID)
        }

        await task.value
        finishPreparationIfCurrent(attemptID: attemptID)
    }

    private func runPreparation(attemptID: UUID, requestAccess: Bool) async {
        if requestAccess {
            permissions = await requestPermissions()
            guard isCurrentPreparation(attemptID, exit: "permission request completed after timeout or replacement") else {
                return
            }
            if let permissionMessage = Self.permissionBlockedMessage(for: permissions) {
                block(permissionMessage, phase: .permissionDenied(permissionMessage), reason: "permissions unavailable")
                return
            }
        }

        do {
            let configuredDescriptor = try await controller.configure(
                mode: selectedMode,
                deviceTier: .standard,
                capabilities: .hevcOnly,
                captureDeviceOrientation: captureDeviceOrientation,
                sessionID: CameraCaptureController.defaultSessionID(),
                packageRootURL: CameraCaptureController.defaultPackageRootURL()
            )
            try Task.checkCancellation()
            guard isCurrentPreparation(attemptID, exit: "configure completed after timeout or replacement") else {
                return
            }
            descriptor = configuredDescriptor
            controller.onRecordingFinished = { [weak self] result in
                switch result {
                case .success(let recording):
                    Task { @MainActor [weak self] in
                        self?.lastFinishedCapture = recording
                        self?.transition(
                            to: .finished(recording.descriptor.clipRelativePath),
                            phase: .done(sessionID: recording.descriptor.sessionID),
                            reason: "recording finished"
                        )
                        self?.recordingStartedAt = nil
                        await self?.buildPostStopSummary(for: recording)
                    }
                case .failure(let error):
                    let message = String(describing: error)
                    Task { @MainActor [weak self] in
                        self?.block(message, phase: .blocked(message), reason: "recording callback failure")
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
            try await performSetupPassIfNeeded(force: true, attemptID: attemptID)
            try Task.checkCancellation()
            guard isCurrentPreparation(attemptID, exit: "setup pass completed after timeout or replacement") else {
                return
            }
            try await controller.startPreview()
            try Task.checkCancellation()
            guard isCurrentPreparation(attemptID, exit: "preview start completed after timeout or replacement") else {
                return
            }
            capturePolicyEnforcement = await controller.currentPolicyEnforcementReport()
            guard isCurrentPreparation(attemptID, exit: "policy read completed after timeout or replacement") else {
                return
            }
            transition(to: .ready, phase: .ready, reason: "configure and preview completed")
            startLiveGuidancePollingIfNeeded()
            await autoStartReplayBenchmarkIfRequested()
        } catch is CancellationError {
            guard activePreparationID == attemptID else {
                Self.recordPathLogger.info("Cancelled preparation exited after watchdog already made state loud")
                return
            }
            block("Camera setup was interrupted. Tap Retry.", phase: .blocked("Camera setup was interrupted. Tap Retry."), reason: "preparation cancelled")
        } catch {
            guard activePreparationID == attemptID else {
                Self.recordPathLogger.error("Late preparation error ignored after loud terminal state: \(String(describing: error), privacy: .public)")
                return
            }
            let message = Self.message(for: error)
            block(message, phase: Self.recordFlowBlockedPhase(for: error, message: message), reason: "preparation error")
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

    private func expirePreparation(attemptID: UUID) {
        guard activePreparationID == attemptID else {
            Self.recordPathLogger.info("Preparation watchdog ignored: attempt already terminal")
            return
        }
        Self.recordPathLogger.error("Preparation watchdog expired after \(Double(self.preparationTimeoutNanoseconds) / 1_000_000_000, privacy: .public) seconds")
        activePreparationID = nil
        let expiredTask = preparationTask
        preparationTask = nil
        preparationWatchdogTask = nil
        block(
            Self.preparationTimeoutMessage,
            phase: .blocked(Self.preparationTimeoutMessage),
            reason: "bounded preparation watchdog"
        )
        expiredTask?.cancel()
    }

    private func finishPreparationIfCurrent(attemptID: UUID) {
        guard activePreparationID == attemptID else {
            Self.recordPathLogger.info("Preparation completion observed after watchdog or replacement")
            return
        }
        preparationWatchdogTask?.cancel()
        preparationWatchdogTask = nil
        preparationTask = nil
        activePreparationID = nil
        Self.recordPathLogger.info("Preparation attempt reached terminal state=\(String(describing: self.status), privacy: .public)")
    }

    private func isCurrentPreparation(_ attemptID: UUID, exit: String) -> Bool {
        guard activePreparationID == attemptID, !Task.isCancelled else {
            Self.recordPathLogger.error("Preparation guard exit: \(exit, privacy: .public); visible state=\(String(describing: self.status), privacy: .public)")
            return false
        }
        return true
    }

    private func transition(to newStatus: Status, phase: DinkVisionRecordFlowPhase, reason: String) {
        let previousStatus = status
        status = newStatus
        recordFlowPhase = phase
        Self.recordPathLogger.info("State \(String(describing: previousStatus), privacy: .public) -> \(String(describing: newStatus), privacy: .public); reason=\(reason, privacy: .public)")
        if let announcement = RecordControlInteractionPolicy.blockedAnnouncement(
            previous: Self.recordControlState(for: previousStatus),
            current: Self.recordControlState(for: newStatus)
        ) {
            Self.recordPathLogger.info("Posting blocked-state accessibility announcement")
            announceBlockedState(announcement)
        }
    }

    private func block(_ message: String, phase: DinkVisionRecordFlowPhase, reason: String) {
        transition(to: .blocked(message), phase: phase, reason: reason)
    }

    func updateOrientation(isLandscapeViewport: Bool) async {
        let updatedOrientation = orientationResolver(isLandscapeViewport)
        guard updatedOrientation != captureDeviceOrientation else {
            Self.recordPathLogger.info("Orientation update no-op: orientation already current")
            return
        }

        captureDeviceOrientation = updatedOrientation
        guard !isRecording else {
            Self.recordPathLogger.info("Orientation changed during recording; current recording continues and reconfigure is deferred")
            return
        }

        await configure()
    }

    func refreshSetupPassIfNeeded(force: Bool = false) async {
        do {
            try await performSetupPassIfNeeded(force: force, attemptID: nil)
        } catch {
            let message = Self.message(for: error)
            block(message, phase: Self.recordFlowBlockedPhase(for: error, message: message), reason: "setup refresh failed")
        }
    }

    private func performSetupPassIfNeeded(force: Bool, attemptID: UUID?) async throws {
        guard !isRecording else {
            Self.recordPathLogger.info("Setup pass skipped: recording is active")
            return
        }
        let currentGravity = await controller.latestGravity()
        if let attemptID {
            try Task.checkCancellation()
            guard isCurrentPreparation(attemptID, exit: "gravity read completed after timeout or replacement") else {
                throw CancellationError()
            }
        }
        let shouldRefresh = force || ARKitSetupPassRefreshPolicy.shouldRefresh(
            now: Date(),
            lastCompletedAt: lastSetupPassCompletedAt,
            lastGravity: lastSetupPassGravity,
            currentGravity: currentGravity
        )
        guard shouldRefresh else {
            Self.recordPathLogger.info("Setup pass skipped: fresh alignment remains valid")
            return
        }

        let shouldRestartPreviewAfterSetup: Bool
        switch status {
        case .ready, .finished:
            shouldRestartPreviewAfterSetup = true
        case .idle, .requestingAccess, .recording, .blocked:
            shouldRestartPreviewAfterSetup = false
        }
        setupPassStatus = .aligning
        Self.recordPathLogger.info("ARKit setup pass started before AVCapture preview")
        let setupPass = await controller.performARKitSetupPass(timeoutSeconds: 4.0)
        if let attemptID {
            try Task.checkCancellation()
            guard isCurrentPreparation(attemptID, exit: "ARKit setup pass completed after timeout or replacement") else {
                throw CancellationError()
            }
        }
        lastSetupPassCompletedAt = Date()
        lastSetupPassGravity = setupPass.gravity ?? currentGravity
        switch setupPass.status {
        case .available:
            setupPassStatus = .aligned
            Self.recordPathLogger.info("ARKit setup pass available")
        case .unavailable:
            let unavailableReason = setupPass.unavailableReason ?? "arkit_setup_pass_unavailable"
            setupPassStatus = .unavailable(unavailableReason)
            Self.recordPathLogger.notice("ARKit setup pass unavailable but advisory: \(unavailableReason, privacy: .public)")
        }
        if shouldRestartPreviewAfterSetup {
            try await controller.startPreview()
        }
    }

    func toggleRecording() async {
        if let recordActionTask {
            Self.recordPathLogger.info("Recording toggle coalesced with active bounded action")
            await recordActionTask.value
            return
        }

        let actionID = UUID()
        activeRecordActionID = actionID
        let actionTask = Task { @MainActor [weak self] in
            guard let self else {
                return
            }
            await self.runRecordingToggle(actionID: actionID)
        }
        recordActionTask = actionTask
        let timeoutNanoseconds = preparationTimeoutNanoseconds
        recordActionWatchdogTask = Task { @MainActor [weak self] in
            guard let self else {
                return
            }
            do {
                try await Task.sleep(nanoseconds: timeoutNanoseconds)
            } catch {
                return
            }
            self.expireRecordAction(actionID: actionID)
        }

        await actionTask.value
        finishRecordActionIfCurrent(actionID: actionID)
    }

    private func runRecordingToggle(actionID: UUID) async {
        Self.recordPathLogger.info("Recording toggle entered state=\(String(describing: self.status), privacy: .public)")
        do {
            if isRecording {
                recordFlowPhase = .saving
                Self.recordPathLogger.info("State remains recording while stop is submitted; saving UI is visible")
                try await controller.stopRecording()
                try Task.checkCancellation()
                guard isCurrentRecordAction(actionID, exit: "recording stop completed after timeout or replacement") else {
                    return
                }
                return
            }

            guard canStartRecording else {
                if case .blocked = status {
                    Self.recordPathLogger.info("Recording toggle retained existing loud blocked reason")
                    return
                }
                block("Camera is not ready. Tap Retry.", phase: .blocked("Camera is not ready. Tap Retry."), reason: "record start readiness guard")
                return
            }

            transition(to: .requestingAccess, phase: .ready, reason: "bounded recording start")
            recentPlayerCountSamples = []
            resetBallOverlay()
            postStopSummary = nil
            controller.setProfileCapturePayload(profileFlow.payload)
            let recordingDescriptor = try await controller.startRecording()
            try Task.checkCancellation()
            guard isCurrentRecordAction(actionID, exit: "recording start completed after timeout or replacement") else {
                await stopLateRecordingAfterTimeout()
                return
            }
            descriptor = recordingDescriptor
            capturePolicyEnforcement = await controller.currentPolicyEnforcementReport()
            try Task.checkCancellation()
            guard isCurrentRecordAction(actionID, exit: "recording policy read completed after timeout or replacement") else {
                await stopLateRecordingAfterTimeout()
                return
            }
            let startedAt = Date()
            recordingStartedAt = startedAt
            transition(to: .recording, phase: .recording(startedAt: startedAt), reason: "controller accepted recording start")
        } catch is CancellationError {
            guard activeRecordActionID == actionID else {
                Self.recordPathLogger.info("Cancelled recording action exited after watchdog already made state loud")
                return
            }
            block("Recording action was interrupted. Tap Retry.", phase: .blocked("Recording action was interrupted. Tap Retry."), reason: "recording action cancelled")
        } catch {
            guard activeRecordActionID == actionID else {
                Self.recordPathLogger.error("Late recording action error ignored after loud terminal state: \(String(describing: error), privacy: .public)")
                return
            }
            let message = Self.message(for: error)
            block(message, phase: Self.recordFlowBlockedPhase(for: error, message: message), reason: "recording toggle error")
        }
    }

    private func expireRecordAction(actionID: UUID) {
        guard activeRecordActionID == actionID else {
            Self.recordPathLogger.info("Recording-action watchdog ignored: action already terminal")
            return
        }
        Self.recordPathLogger.error("Recording-action watchdog expired after \(Double(self.preparationTimeoutNanoseconds) / 1_000_000_000, privacy: .public) seconds")
        activeRecordActionID = nil
        let expiredTask = recordActionTask
        recordActionTask = nil
        recordActionWatchdogTask = nil
        block(
            Self.recordActionTimeoutMessage,
            phase: .blocked(Self.recordActionTimeoutMessage),
            reason: "bounded recording-action watchdog"
        )
        expiredTask?.cancel()
        Task { [controller] in
            do {
                try await controller.stopRecording()
                Self.recordPathLogger.notice("Watchdog cleanup submitted a stop for any late recording")
            } catch {
                Self.recordPathLogger.info("Watchdog cleanup found no stoppable recording: \(String(describing: error), privacy: .public)")
            }
        }
    }

    private func finishRecordActionIfCurrent(actionID: UUID) {
        guard activeRecordActionID == actionID else {
            Self.recordPathLogger.info("Recording-action completion observed after watchdog or replacement")
            return
        }
        recordActionWatchdogTask?.cancel()
        recordActionWatchdogTask = nil
        recordActionTask = nil
        activeRecordActionID = nil
        Self.recordPathLogger.info("Recording action reached terminal state=\(String(describing: self.status), privacy: .public)")
    }

    private func isCurrentRecordAction(_ actionID: UUID, exit: String) -> Bool {
        guard activeRecordActionID == actionID, !Task.isCancelled else {
            Self.recordPathLogger.error("Recording-action guard exit: \(exit, privacy: .public); visible state=\(String(describing: self.status), privacy: .public)")
            return false
        }
        return true
    }

    private func stopLateRecordingAfterTimeout() async {
        do {
            try await controller.stopRecording()
            Self.recordPathLogger.notice("Late recording start was stopped after watchdog")
        } catch {
            Self.recordPathLogger.error("Late recording cleanup failed: \(String(describing: error), privacy: .public)")
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
            return "Back camera is unavailable. Close other camera apps, then tap Retry."
        case CameraCaptureControllerError.permissionDenied(let snapshot):
            return permissionBlockedMessage(for: snapshot) ?? "Enable Camera and Microphone in Settings, then tap Retry."
        case CameraCaptureControllerError.cannotAddVideoInput:
            return "Camera input could not start. Close other camera apps, then tap Retry."
        case CameraCaptureControllerError.cannotAddAudioInput:
            return "Microphone input could not start. Enable Microphone in Settings, then tap Retry."
        case CameraCaptureControllerError.cannotAddMovieOutput:
            return "Recording output could not start. Tap Retry."
        case CameraCaptureControllerError.movieOutputVideoConnectionUnavailable:
            return "Camera video output is unavailable. Tap Retry."
        case CameraCaptureControllerError.unsupportedFrameRate(let fps):
            return "\(fps) fps is unavailable on this camera. Choose another mode, then tap Retry."
        case CameraCaptureControllerError.noConfiguredPackage:
            return "Camera setup is incomplete. Tap Retry."
        case CameraCaptureControllerError.landscapeRequired:
            return "Rotate to landscape to record"
        case CameraCaptureControllerError.alreadyRecording:
            return "Recording is already active. Tap Stop before retrying."
        case CameraCaptureControllerError.notRecording:
            return "No recording is active. Tap Retry."
        case CameraCaptureControllerError.cameraResourceBusy(.arKitSetup):
            return "Camera is still busy finishing alignment. Tap Retry."
        case CameraCaptureControllerError.cameraResourceBusy(.avCapture):
            return "Camera preview is busy. Tap Retry."
        case CameraCaptureControllerError.previewFailedToStart:
            return "Camera preview did not start. Close other camera apps, then tap Retry."
        case CameraCaptureControllerError.fileSystem:
            return "Recording storage is unavailable. Free space, then tap Retry."
        default:
            return String(describing: error)
        }
    }

    nonisolated private static func permissionBlockedMessage(for snapshot: CapturePermissionSnapshot) -> String? {
        let cameraUnavailable = snapshot.camera != .authorized
        let microphoneUnavailable = snapshot.microphone != .authorized
        switch (cameraUnavailable, microphoneUnavailable) {
        case (false, false):
            return nil
        case (true, false):
            return "Enable Camera in Settings, then tap Retry."
        case (false, true):
            return "Enable Microphone in Settings, then tap Retry."
        case (true, true):
            return "Enable Camera and Microphone in Settings, then tap Retry."
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

    nonisolated private static func recordControlState(for status: Status) -> RecordControlState {
        switch status {
        case .idle:
            return .idle
        case .requestingAccess:
            return .preparing
        case .ready, .finished:
            return .ready
        case .recording:
            return .recording
        case .blocked(let reason):
            return .blocked(reason: reason)
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
