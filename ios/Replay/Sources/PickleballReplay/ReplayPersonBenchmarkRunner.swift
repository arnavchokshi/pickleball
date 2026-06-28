import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import PickleballFastTier

public struct ReplayPersonBenchmarkOutputPaths: Equatable, Sendable {
    public var sessionURL: URL
    public var tracksURL: URL
    public var timingURL: URL
    public var summaryURL: URL

    public init(rootURL: URL, clipID: String, candidate: String) {
        self.sessionURL = rootURL
            .appendingPathComponent(clipID, isDirectory: true)
            .appendingPathComponent(candidate, isDirectory: true)
        self.tracksURL = sessionURL.appendingPathComponent("on_device_person_tracks.json")
        self.timingURL = sessionURL.appendingPathComponent("timing.json")
        self.summaryURL = sessionURL.appendingPathComponent("run_summary.json")
    }
}

public struct ReplayPersonBenchmarkRunSummary: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var artifactType: String
    public var clipID: String
    public var candidate: String
    public var videoPath: String
    public var tracksPath: String
    public var timingPath: String
    public var processedFrameCount: Int
    public var wallClockSeconds: Double

    public init(
        schemaVersion: Int = 1,
        artifactType: String = "pickleball_replay_person_benchmark_run",
        clipID: String,
        candidate: String,
        videoPath: String,
        tracksPath: String,
        timingPath: String,
        processedFrameCount: Int,
        wallClockSeconds: Double
    ) {
        self.schemaVersion = schemaVersion
        self.artifactType = artifactType
        self.clipID = clipID
        self.candidate = candidate
        self.videoPath = videoPath
        self.tracksPath = tracksPath
        self.timingPath = timingPath
        self.processedFrameCount = processedFrameCount
        self.wallClockSeconds = wallClockSeconds
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifactType = "artifact_type"
        case clipID = "clip_id"
        case candidate
        case videoPath = "video_path"
        case tracksPath = "tracks_path"
        case timingPath = "timing_path"
        case processedFrameCount = "processed_frame_count"
        case wallClockSeconds = "wall_clock_seconds"
    }
}

public enum ReplayPersonBenchmarkRunnerError: Error, Equatable, Sendable {
    case missingVideoTrack(String)
    case cannotAddReaderOutput(String)
    case cannotStartReader(String)
    case missingPixelBuffer(Int)
    case unsupportedProcessor(String)
}

public enum ReplayPersonBenchmarkProcessor: Equatable, Sendable {
    case vision(VisionPersonBenchmarkConfiguration)
    case coreML(CoreMLPersonDetectorConfiguration)

    public static var visionHumanRectangles: ReplayPersonBenchmarkProcessor {
        .vision(VisionPersonBenchmarkConfiguration())
    }

    public var candidate: OnDevicePersonCandidate {
        switch self {
        case .vision(let configuration):
            return configuration.candidate
        case .coreML(let configuration):
            return configuration.candidate
        }
    }
}

@available(iOS 13.0, macOS 10.15, *)
public final class ReplayPersonBenchmarkRunner {
    private let processor: ReplayPersonBenchmarkProcessor
    private let encoder: JSONEncoder

    public init(processor: ReplayPersonBenchmarkProcessor = .visionHumanRectangles) {
        self.processor = processor
        self.encoder = JSONEncoder()
        self.encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    }

    public convenience init(configuration: VisionPersonBenchmarkConfiguration) {
        self.init(processor: .vision(configuration))
    }

    public func runAll(
        manifestURL: URL,
        outputRootURL: URL,
        maxFramesPerClip: Int? = nil
    ) async throws -> [ReplayPersonBenchmarkRunSummary] {
        let manifest = try ReplayInputManifest.load(from: manifestURL)
        let clips = manifest.resolvedClips(relativeTo: manifestURL.deletingLastPathComponent())
        var summaries: [ReplayPersonBenchmarkRunSummary] = []
        for clip in clips {
            let summary = try await run(clip: clip, outputRootURL: outputRootURL, maxFrames: maxFramesPerClip)
            summaries.append(summary)
        }
        return summaries
    }

    public func run(
        clip: ResolvedReplayInputClip,
        outputRootURL: URL,
        maxFrames: Int? = nil
    ) async throws -> ReplayPersonBenchmarkRunSummary {
        let runtime = try makeRuntime()
        let candidate = runtime.candidate
        let paths = ReplayPersonBenchmarkOutputPaths(
            rootURL: outputRootURL,
            clipID: clip.clipID,
            candidate: candidate.rawValue
        )
        try FileManager.default.createDirectory(at: paths.sessionURL, withIntermediateDirectories: true)

        let asset = AVURLAsset(url: clip.videoURL)
        guard let videoTrack = asset.tracks(withMediaType: .video).first else {
            throw ReplayPersonBenchmarkRunnerError.missingVideoTrack(clip.videoURL.path)
        }

        let outputSettings: [String: Any] = [
            String(kCVPixelBufferPixelFormatTypeKey): kCVPixelFormatType_32BGRA,
        ]
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw ReplayPersonBenchmarkRunnerError.cannotAddReaderOutput(clip.videoURL.path)
        }
        reader.add(output)
        guard reader.startReading() else {
            throw ReplayPersonBenchmarkRunnerError.cannotStartReader(reader.error?.localizedDescription ?? clip.videoURL.path)
        }

        let startedThermalState = thermalStateLabel()
        let started = CFAbsoluteTimeGetCurrent()
        let fps = videoTrack.nominalFrameRate > 0 ? Double(videoTrack.nominalFrameRate) : 30.0
        var frameIndex = 0
        var resolution: [Int]?
        var frames: [OnDevicePersonFrame] = []
        var timingSamples: [OnDevicePersonTimingSample] = []

        while reader.status == .reading {
            if let maxFrames, frameIndex >= maxFrames {
                reader.cancelReading()
                break
            }
            guard let sampleBuffer = output.copyNextSampleBuffer() else {
                break
            }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                throw ReplayPersonBenchmarkRunnerError.missingPixelBuffer(frameIndex)
            }
            if resolution == nil {
                resolution = [CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer)]
            }

            let frameStarted = CFAbsoluteTimeGetCurrent()
            let detections = try runtime.process(pixelBuffer: pixelBuffer, frameIndex: frameIndex)
            let latencyMs = max(0, (CFAbsoluteTimeGetCurrent() - frameStarted) * 1000.0)
            frames.append(OnDevicePersonFrame(frameIndex: frameIndex, detections: detections))
            timingSamples.append(OnDevicePersonTimingSample(frameIndex: frameIndex, latencyMs: latencyMs, processed: true))
            frameIndex += 1
        }

        if reader.status == .failed, let error = reader.error {
            throw error
        }

        let wallClockSeconds = max(0, CFAbsoluteTimeGetCurrent() - started)
        let tracks = OnDevicePersonTracks(
            clipID: clip.clipID,
            candidate: candidate,
            deviceModel: deviceModelIdentifier(),
            resolution: resolution,
            fps: fps,
            frames: frames
        )
        let timing = OnDevicePersonTiming(
            clipID: clip.clipID,
            candidate: candidate,
            mode: .replay,
            deviceModel: deviceModelIdentifier(),
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            wallClockSeconds: wallClockSeconds,
            droppedFrameCount: 0,
            modelLoadMs: runtime.modelLoadMs,
            mlpackageSizeMB: runtime.mlpackageSizeMB,
            startedThermalState: startedThermalState,
            endedThermalState: thermalStateLabel(),
            samples: timingSamples
        )
        try writeJSON(tracks, to: paths.tracksURL)
        try writeJSON(timing, to: paths.timingURL)

        let summary = ReplayPersonBenchmarkRunSummary(
            clipID: clip.clipID,
            candidate: candidate.rawValue,
            videoPath: clip.videoURL.path,
            tracksPath: paths.tracksURL.path,
            timingPath: paths.timingURL.path,
            processedFrameCount: frames.count,
            wallClockSeconds: wallClockSeconds
        )
        try writeJSON(summary, to: paths.summaryURL)
        return summary
    }

    private func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
        let data = try encoder.encode(value)
        try data.write(to: url, options: [.atomic])
    }

    private func makeRuntime() throws -> ReplayPersonFrameProcessor {
        switch processor {
        case .vision(let configuration):
            return VisionReplayPersonFrameProcessor(configuration: configuration)
        case .coreML(let configuration):
            if #available(iOS 15.0, macOS 12.0, *) {
                return try CoreMLReplayPersonFrameProcessor(configuration: configuration)
            }
            throw ReplayPersonBenchmarkRunnerError.unsupportedProcessor("Core ML person detector requires iOS 15 or newer")
        }
    }
}

private protocol ReplayPersonFrameProcessor {
    var candidate: OnDevicePersonCandidate { get }
    var modelLoadMs: Double? { get }
    var mlpackageSizeMB: Double? { get }

    func process(pixelBuffer: CVPixelBuffer, frameIndex: Int) throws -> [OnDevicePersonDetection]
}

@available(iOS 13.0, macOS 10.15, *)
private final class VisionReplayPersonFrameProcessor: ReplayPersonFrameProcessor {
    let candidate: OnDevicePersonCandidate
    let modelLoadMs: Double? = 0
    let mlpackageSizeMB: Double? = nil
    private let processor: VisionHumanRectanglePersonProcessor

    init(configuration: VisionPersonBenchmarkConfiguration) {
        self.candidate = configuration.candidate
        self.processor = VisionHumanRectanglePersonProcessor(configuration: configuration)
    }

    func process(pixelBuffer: CVPixelBuffer, frameIndex: Int) throws -> [OnDevicePersonDetection] {
        try processor.process(pixelBuffer: pixelBuffer, frameIndex: frameIndex)
    }
}

@available(iOS 15.0, macOS 12.0, *)
private final class CoreMLReplayPersonFrameProcessor: ReplayPersonFrameProcessor {
    let candidate: OnDevicePersonCandidate
    let modelLoadMs: Double?
    let mlpackageSizeMB: Double?
    private let detector: CoreMLPersonDetector

    init(configuration: CoreMLPersonDetectorConfiguration) throws {
        let detector = try CoreMLPersonDetector(configuration: configuration)
        self.candidate = configuration.candidate
        self.modelLoadMs = detector.modelLoadMs
        self.mlpackageSizeMB = detector.mlpackageSizeMB
        self.detector = detector
    }

    func process(pixelBuffer: CVPixelBuffer, frameIndex: Int) throws -> [OnDevicePersonDetection] {
        try detector.process(pixelBuffer: pixelBuffer, frameIndex: frameIndex)
    }
}

private func thermalStateLabel() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal:
        return "nominal"
    case .fair:
        return "fair"
    case .serious:
        return "serious"
    case .critical:
        return "critical"
    @unknown default:
        return "unknown"
    }
}

private func deviceModelIdentifier() -> String? {
    var systemInfo = utsname()
    uname(&systemInfo)
    let mirror = Mirror(reflecting: systemInfo.machine)
    let identifier = mirror.children.reduce(into: "") { result, element in
        guard let value = element.value as? Int8, value != 0 else {
            return
        }
        result.append(String(UnicodeScalar(UInt8(value))))
    }
    return identifier.isEmpty ? nil : identifier
}
