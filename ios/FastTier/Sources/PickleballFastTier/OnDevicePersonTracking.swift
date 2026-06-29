import Foundation

public enum OnDevicePersonCandidate: String, Codable, Sendable {
    case visionHumanRectanglesIouV1 = "vision_human_rectangles_iou_v1"
    case visionPoseRoleLock = "vision_pose_rolelock"
    case yolo26nInt8EveryFrame = "yolo26n_int8_every_frame"
    case yolo26nInt8Img512EveryFrame = "yolo26n_int8_img512_every_frame"
    case yolo26nInt8Img640EveryFrame = "yolo26n_int8_img640_every_frame"
    case yolo26nInt8Detect2Track30 = "yolo26n_int8_detect2_track30"
    case yolo26sInt8EveryFrame = "yolo26s_int8_every_frame"
    case yolo26mInt8Img416EveryFrame = "yolo26m_int8_img416_every_frame"
    case yolo11nInt8Fallback = "yolo11n_int8_fallback"
}

public struct OnDevicePersonDetection: Codable, Equatable, Sendable {
    public var trackID: Int
    public var bboxXYWH: [Double]
    public var confidence: Double
    public var source: String
    public var role: String?

    public init(trackID: Int, bboxXYWH: [Double], confidence: Double, source: String, role: String? = nil) {
        self.trackID = trackID
        self.bboxXYWH = bboxXYWH
        self.confidence = confidence
        self.source = source
        self.role = role
    }

    private enum CodingKeys: String, CodingKey {
        case trackID = "track_id"
        case bboxXYWH = "bbox_xywh"
        case confidence
        case source
        case role
    }
}

public struct OnDevicePersonFrame: Codable, Equatable, Sendable {
    public var frameIndex: Int
    public var detections: [OnDevicePersonDetection]

    public init(frameIndex: Int, detections: [OnDevicePersonDetection]) {
        self.frameIndex = frameIndex
        self.detections = detections
    }

    private enum CodingKeys: String, CodingKey {
        case frameIndex = "frame_index"
        case detections
    }
}

public struct OnDevicePersonTracksSummary: Codable, Equatable, Sendable {
    public var frameCount: Int
    public var detectionCount: Int
    public var trackIDs: [Int]

    public init(frameCount: Int, detectionCount: Int, trackIDs: [Int]) {
        self.frameCount = frameCount
        self.detectionCount = detectionCount
        self.trackIDs = trackIDs
    }

    private enum CodingKeys: String, CodingKey {
        case frameCount = "frame_count"
        case detectionCount = "detection_count"
        case trackIDs = "track_ids"
    }
}

public struct OnDevicePersonTracks: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var artifactType: String
    public var clipID: String
    public var candidate: OnDevicePersonCandidate
    public var deviceModel: String?
    public var coordinateSpace: String
    public var resolution: [Int]?
    public var fps: Double
    public var frames: [OnDevicePersonFrame]
    public var summary: OnDevicePersonTracksSummary

    public init(
        schemaVersion: Int = 1,
        artifactType: String = "racketsport_on_device_person_tracks",
        clipID: String,
        candidate: OnDevicePersonCandidate,
        deviceModel: String? = nil,
        coordinateSpace: String = "source_video_pixels",
        resolution: [Int]? = nil,
        fps: Double,
        frames: [OnDevicePersonFrame]
    ) {
        self.schemaVersion = schemaVersion
        self.artifactType = artifactType
        self.clipID = clipID
        self.candidate = candidate
        self.deviceModel = deviceModel
        self.coordinateSpace = coordinateSpace
        self.resolution = resolution
        self.fps = fps
        self.frames = frames
        self.summary = OnDevicePersonTracksSummary(
            frameCount: frames.count,
            detectionCount: frames.reduce(0) { $0 + $1.detections.count },
            trackIDs: Array(Set(frames.flatMap { frame in frame.detections.map(\.trackID) })).sorted()
        )
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifactType = "artifact_type"
        case clipID = "clip_id"
        case candidate
        case deviceModel = "device_model"
        case coordinateSpace = "coordinate_space"
        case resolution
        case fps
        case frames
        case summary
    }
}

public enum OnDevicePersonBenchmarkMode: String, Codable, Sendable {
    case replay
    case live
}

public struct OnDevicePersonTimingSample: Codable, Equatable, Sendable {
    public var frameIndex: Int
    public var latencyMs: Double
    public var processed: Bool

    public init(frameIndex: Int, latencyMs: Double, processed: Bool) {
        self.frameIndex = frameIndex
        self.latencyMs = latencyMs
        self.processed = processed
    }

    private enum CodingKeys: String, CodingKey {
        case frameIndex = "frame_index"
        case latencyMs = "latency_ms"
        case processed
    }
}

public struct OnDevicePersonTimingSummary: Codable, Equatable, Sendable {
    public var processedFrameCount: Int
    public var droppedFrameCount: Int
    public var sustainedProcessedFPS: Double
    public var p50LatencyMs: Double
    public var p95LatencyMs: Double

    public init(
        processedFrameCount: Int,
        droppedFrameCount: Int,
        sustainedProcessedFPS: Double,
        p50LatencyMs: Double,
        p95LatencyMs: Double
    ) {
        self.processedFrameCount = processedFrameCount
        self.droppedFrameCount = droppedFrameCount
        self.sustainedProcessedFPS = sustainedProcessedFPS
        self.p50LatencyMs = p50LatencyMs
        self.p95LatencyMs = p95LatencyMs
    }

    private enum CodingKeys: String, CodingKey {
        case processedFrameCount = "processed_frame_count"
        case droppedFrameCount = "dropped_frame_count"
        case sustainedProcessedFPS = "sustained_processed_fps"
        case p50LatencyMs = "p50_latency_ms"
        case p95LatencyMs = "p95_latency_ms"
    }
}

public struct OnDevicePersonTiming: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var artifactType: String
    public var clipID: String
    public var candidate: OnDevicePersonCandidate
    public var mode: OnDevicePersonBenchmarkMode
    public var deviceModel: String?
    public var osVersion: String?
    public var wallClockSeconds: Double
    public var droppedFrameCount: Int
    public var modelLoadMs: Double?
    public var mlpackageSizeMB: Double?
    public var startedThermalState: String?
    public var endedThermalState: String?
    public var samples: [OnDevicePersonTimingSample]
    public var summary: OnDevicePersonTimingSummary

    public init(
        schemaVersion: Int = 1,
        artifactType: String = "racketsport_on_device_person_timing",
        clipID: String,
        candidate: OnDevicePersonCandidate,
        mode: OnDevicePersonBenchmarkMode,
        deviceModel: String? = nil,
        osVersion: String? = nil,
        wallClockSeconds: Double,
        droppedFrameCount: Int,
        modelLoadMs: Double? = nil,
        mlpackageSizeMB: Double? = nil,
        startedThermalState: String? = nil,
        endedThermalState: String? = nil,
        samples: [OnDevicePersonTimingSample]
    ) {
        let processedSamples = samples.filter(\.processed)
        let latencies = processedSamples.map(\.latencyMs).sorted()
        self.schemaVersion = schemaVersion
        self.artifactType = artifactType
        self.clipID = clipID
        self.candidate = candidate
        self.mode = mode
        self.deviceModel = deviceModel
        self.osVersion = osVersion
        self.wallClockSeconds = wallClockSeconds
        self.droppedFrameCount = droppedFrameCount
        self.modelLoadMs = modelLoadMs
        self.mlpackageSizeMB = mlpackageSizeMB
        self.startedThermalState = startedThermalState
        self.endedThermalState = endedThermalState
        self.samples = samples
        self.summary = OnDevicePersonTimingSummary(
            processedFrameCount: processedSamples.count,
            droppedFrameCount: droppedFrameCount,
            sustainedProcessedFPS: wallClockSeconds > 0 ? Double(processedSamples.count) / wallClockSeconds : 0,
            p50LatencyMs: percentile(latencies, fraction: 0.50),
            p95LatencyMs: percentile(latencies, fraction: 0.95)
        )
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifactType = "artifact_type"
        case clipID = "clip_id"
        case candidate
        case mode
        case deviceModel = "device_model"
        case osVersion = "os_version"
        case wallClockSeconds = "wall_clock_seconds"
        case droppedFrameCount = "dropped_frame_count"
        case modelLoadMs = "model_load_ms"
        case mlpackageSizeMB = "mlpackage_size_mb"
        case startedThermalState = "started_thermal_state"
        case endedThermalState = "ended_thermal_state"
        case samples
        case summary
    }
}

public struct OnDevicePersonBenchmarkArtifactPaths: Equatable, Sendable {
    public var sessionRelativePath: String
    public var tracksRelativePath: String
    public var timingRelativePath: String

    public init(sessionRelativePath: String) {
        let normalized = sessionRelativePath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.sessionRelativePath = normalized
        self.tracksRelativePath = "\(normalized)/on_device_person_tracks.json"
        self.timingRelativePath = "\(normalized)/timing.json"
    }
}

private func percentile(_ sortedValues: [Double], fraction: Double) -> Double {
    guard !sortedValues.isEmpty else {
        return 0
    }
    let clamped = max(0, min(1, fraction))
    let index = Int((Double(sortedValues.count - 1) * clamped).rounded(.up))
    return sortedValues[index]
}
