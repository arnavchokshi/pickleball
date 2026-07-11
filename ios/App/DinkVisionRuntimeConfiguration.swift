import Foundation
import PickleballCapture
import PickleballCore
import PickleballGuidance
import PickleballReplay
import PickleballUpload
@preconcurrency import AVFoundation

enum DinkVisionWalkerCaptureState: String, Equatable {
    case live
    case permissionPrimer
    case granted
}

enum DinkVisionWalkerReplayState: String, Equatable {
    case live
    case empty
    case seeded
}

struct DinkVisionRuntimeConfiguration: Equatable {
    var isWalker: Bool
    var skipSplash: Bool
    var captureState: DinkVisionWalkerCaptureState
    var replayState: DinkVisionWalkerReplayState
    var forceRecordPressed: Bool
    var forceWorldCoachMark: Bool
    var apiBaseURL: URL

    static func current(
        arguments: [String] = ProcessInfo.processInfo.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        infoDictionary: [String: Any] = Bundle.main.infoDictionary ?? [:]
    ) -> DinkVisionRuntimeConfiguration {
        let isWalker = arguments.contains("-dinkvision.walker")
        return DinkVisionRuntimeConfiguration(
            isWalker: isWalker,
            skipSplash: arguments.contains("-dinkvision.skipSplash"),
            captureState: DinkVisionWalkerCaptureState(rawValue: value(after: "-dinkvision.captureState", in: arguments) ?? "") ?? (isWalker ? .granted : .live),
            replayState: DinkVisionWalkerReplayState(rawValue: value(after: "-dinkvision.replays", in: arguments) ?? "") ?? .live,
            forceRecordPressed: arguments.contains("-dinkvision.recordPressed"),
            forceWorldCoachMark: arguments.contains("-dinkvision.forceWorldCoachMark"),
            apiBaseURL: APIBaseURLResolver.resolve(
                arguments: arguments,
                environment: environment,
                infoDictionary: infoDictionary
            )
        )
    }

    func makeCaptureViewModel() -> CaptureViewModel {
        guard captureState != .live else {
            return CaptureViewModel()
        }
        let permissionSnapshot = captureState == .permissionPrimer
            ? CapturePermissionSnapshot(camera: .notDetermined, microphone: .notDetermined)
            : CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
        let controller = DinkVisionWalkerCameraController(captureState: captureState)
        return CaptureViewModel(
            controller: controller,
            requestPermissions: {
                permissionSnapshot
            }
        )
    }

    func makeReplayDataSource() -> DinkVisionReplayListDataSource {
        switch replayState {
        case .live:
            return DinkVisionReplayListDataSource()
        case .empty:
            return DinkVisionReplayListDataSource(loadPackages: { _ in [] })
        case .seeded:
            return DinkVisionReplayListDataSource(
                loadPackages: { _ in [Self.seededReplayItem] },
                bundledSamplePackageIDs: [Self.seededReplayItem.sessionID]
            )
        }
    }

    func makeAuthApiClient(tokenStore: AuthTokenStore = AuthTokenStore()) -> AuthApiClient {
        AuthApiClient(baseURL: apiBaseURL, tokenStore: tokenStore)
    }

    func makeRenderGatewayClient(tokenStore: AuthTokenStore = AuthTokenStore()) -> RenderGatewayClient {
        RenderGatewayClient(baseURL: apiBaseURL, accessTokenProvider: { try? tokenStore.readAccessToken() })
    }

    func makePresignedUploadClient(tokenStore: AuthTokenStore = AuthTokenStore()) -> PresignedUploadClient {
        PresignedUploadClient(baseURL: apiBaseURL, accessTokenProvider: { try? tokenStore.readAccessToken() })
    }

    @MainActor
    func makeUploadCoordinator(
        tokenStore: AuthTokenStore = AuthTokenStore(),
        preferences: UserDefaults = .standard
    ) -> DinkVisionUploadCoordinator {
        let client = makePresignedUploadClient(tokenStore: tokenStore)
        let jobClient = makeRenderGatewayClient(tokenStore: tokenStore)
        return DinkVisionUploadCoordinator(
            queue: UploadQueue(client: client, jobClient: jobClient),
            packageRootURL: CameraCaptureController.defaultPackageRootURL(),
            hasAccessToken: { tokenStore.hasAccessToken },
            preferences: preferences
        )
    }

    private static var seededReplayItem: CaptureLibraryItem {
        CaptureLibraryItem(
            sessionID: "walker-seeded-rally",
            clipRelativePath: "captures/walker-seeded-rally/clip.mov",
            sidecarRelativePath: "captures/walker-seeded-rally/capture_sidecar.json",
            provenance: .liveRecording,
            durationSeconds: 84,
            fps: 60,
            resolution: [1920, 1080],
            captureQualityGrade: .good,
            recordedAt: Date(timeIntervalSince1970: 1_783_440_000)
        )
    }

    private static func value(after key: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: key) else {
            return nil
        }
        let valueIndex = arguments.index(after: index)
        guard valueIndex < arguments.endIndex else {
            return nil
        }
        return arguments[valueIndex]
    }
}

final class DinkVisionWalkerCameraController: CameraCaptureControlling, @unchecked Sendable {
    let session = AVCaptureSession()
    var onRecordingFinished: ((Result<CameraRecordingResult, Error>) -> Void)?

    private let captureState: DinkVisionWalkerCaptureState
    private var descriptor: CapturePackageDescriptor?

    init(captureState: DinkVisionWalkerCaptureState) {
        self.captureState = captureState
    }

    func configure(
        mode: CaptureMode,
        deviceTier: DeviceTier,
        capabilities: CaptureCodecCapabilities,
        captureDeviceOrientation: CaptureDeviceOrientation,
        sessionID: String,
        packageRootURL _: URL
    ) async throws -> CapturePackageDescriptor {
        if captureState == .permissionPrimer {
            throw CameraCaptureControllerError.permissionDenied(
                CapturePermissionSnapshot(camera: .notDetermined, microphone: .notDetermined)
            )
        }
        let descriptor = try CapturePackageDescriptor(
            sessionID: sessionID,
            policy: CapturePolicy.recommended(
                for: mode,
                deviceTier: deviceTier,
                capabilities: capabilities,
                orientation: .landscape
            ),
            startedAt: Date(timeIntervalSince1970: 1_783_440_000),
            captureDeviceOrientation: captureDeviceOrientation
        )
        self.descriptor = descriptor
        return descriptor
    }

    func startPreview() async {}

    func stopPreview() async {}

    func performARKitSetupPass(timeoutSeconds _: Double) async -> ARKitSetupPassSidecar {
        .unavailable(reason: "walker_fixture_no_arkit", gravity: [0, -1, 0], durationS: 0)
    }

    func startRecording() async throws -> CapturePackageDescriptor {
        if let descriptor {
            return descriptor
        }
        let descriptor = try CapturePackageDescriptor(
            sessionID: "walker-recording",
            policy: CapturePolicy.recommended(
                for: .standard60,
                deviceTier: .standard,
                capabilities: .hevcOnly,
                orientation: .landscape
            ),
            startedAt: Date(timeIntervalSince1970: 1_783_440_000),
            captureDeviceOrientation: .landscapeRight
        )
        self.descriptor = descriptor
        return descriptor
    }

    func stopRecording() async throws {
        guard let descriptor else {
            throw CameraCaptureControllerError.notRecording
        }
        onRecordingFinished?(.success(CameraRecordingResult(descriptor: descriptor, clipURL: URL(fileURLWithPath: "/tmp/walker/clip.mov"))))
    }

    func latestGravity() async -> [Double] {
        [0, -1, 0]
    }

    func currentLiveGuidanceSample() async -> LiveGuidanceSample {
        LiveGuidanceSample(
            exposureTargetOffsetEV: 0,
            isExposureLocked: true,
            shutterSeconds: 1.0 / 240.0,
            minimumSharpShutterSeconds: 1.0 / 120.0,
            tiltFromLevelDegrees: 1.4,
            requestedFPS: 60,
            configuredFPS: 60,
            expectedResolution: [1920, 1080],
            configuredResolution: [1920, 1080],
            setupTipReasons: ["walker_fixture"]
        )
    }

    func currentPolicyEnforcementReport() async -> CapturePolicyEnforcementReport? {
        CapturePolicyEnforcementReport(
            requested: CapturePolicyRequestedState(
                fps: 60,
                resolution: [1920, 1080],
                format: .hevc,
                orientation: .landscape,
                electronicStabilizationEnabled: false,
                exposureLocked: true,
                focusLocked: true,
                whiteBalanceLocked: true
            ),
            achieved: CapturePolicyAchievedState(
                fps: 60,
                resolution: [1920, 1080],
                format: .hevc,
                orientation: .landscape,
                electronicStabilizationEnabled: false,
                exposureLocked: true,
                focusLocked: true,
                whiteBalanceLocked: true
            ),
            violations: []
        )
    }
}

final class DinkVisionForcedCoachMarkStore: WorldViewerCoachMarkStoring {
    var didShowViewerCoachMark: Bool = false
}
