import Foundation
import SwiftUI
import UIKit
@preconcurrency import AVFoundation
import PickleballCapture
import PickleballCore
import PickleballFastTier
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

    let controller = CameraCaptureController()
    private var didAutoStartReplayBenchmark = false

    static let modes: [CaptureMode] = [.standard60, .swing120, .ballPhysics240, .quality4K60]

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

    func prepare() async {
        status = .requestingAccess
        permissions = await CameraCaptureController.requestPermissions()
        configure()
        await autoStartReplayBenchmarkIfRequested()
    }

    func configure() {
        do {
            descriptor = try controller.configure(
                mode: selectedMode,
                captureDeviceOrientation: captureDeviceOrientation
            )
            controller.onRecordingFinished = { [weak self] result in
                switch result {
                case .success(let recording):
                    Task { @MainActor [weak self] in
                        self?.status = .finished(recording.descriptor.clipRelativePath)
                    }
                case .failure(let error):
                    let message = String(describing: error)
                    Task { @MainActor [weak self] in
                        self?.status = .blocked(message)
                    }
                }
            }
            controller.startPreview()
            status = .ready
        } catch {
            status = .blocked(Self.message(for: error))
        }
    }

    func updateOrientation(isLandscapeViewport: Bool) {
        let updatedOrientation = Self.deviceOrientation(fallbackIsLandscape: isLandscapeViewport)
        guard updatedOrientation != captureDeviceOrientation else {
            return
        }

        captureDeviceOrientation = updatedOrientation
        guard !isRecording else {
            return
        }

        configure()
    }

    func toggleRecording() {
        do {
            if isRecording {
                try controller.stopRecording()
            } else {
                try controller.startRecording()
                descriptor = controller.activeDescriptor
                status = .recording
            }
        } catch {
            status = .blocked(Self.message(for: error))
        }
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
