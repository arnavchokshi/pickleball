import Foundation

// Canonical tier split: this module describes ON-DEVICE LIVE preview/guidance
// contracts only. 2D pose/joints are canonical for the live tier. The `pose3D`
// cases below are legacy/debug preview payloads and must not be treated as
// phone-real-time mesh, metric coaching truth, or SERVER OFFLINE deep output.
public enum PreviewTrackKind: String, Codable, Sendable {
    case pose2D = "pose_2d"
    case pose3D = "pose_3d"
    case handPose = "hand_pose"
    case segmentation
    case ball
    case racket
}

public enum PreviewTrackSource: String, Codable, Sendable {
    case appleVisionBody2D = "apple_vision_body_2d"
    case appleVisionBody3D = "apple_vision_body_3d"
    case appleVisionHandPose = "apple_vision_hand_pose"
    case appleVisionSegmentation = "apple_vision_segmentation"
    case yoloCoreMLBall = "yolo_coreml_ball"
    case yoloCoreMLRacket = "yolo_coreml_racket"
}

public struct PreviewTrack: Codable, Equatable, Sendable {
    public var id: String
    public var kind: PreviewTrackKind
    public var source: PreviewTrackSource
    public var relativePath: String
    public var fps: Double
    public var subjectLimit: Int
    public var previewOnly: Bool

    public init(
        id: String,
        kind: PreviewTrackKind,
        source: PreviewTrackSource,
        relativePath: String,
        fps: Double,
        subjectLimit: Int,
        previewOnly: Bool = true
    ) {
        self.id = id
        self.kind = kind
        self.source = source
        self.relativePath = relativePath
        self.fps = fps
        self.subjectLimit = subjectLimit
        self.previewOnly = previewOnly
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case kind
        case source
        case relativePath = "relative_path"
        case fps
        case subjectLimit = "subject_limit"
        case previewOnly = "preview_only"
    }
}

public struct PreviewMetricLabel: Codable, Equatable, Sendable {
    public var metricID: String
    public var title: String
    public var valueText: String
    public var previewOnly: Bool
    public var displayTitle: String

    public init(metricID: String, title: String, valueText: String, previewOnly: Bool = true) {
        self.metricID = metricID
        self.title = title
        self.valueText = valueText
        self.previewOnly = previewOnly
        self.displayTitle = previewOnly ? "Preview only: \(title)" : title
    }

    private enum CodingKeys: String, CodingKey {
        case metricID = "metric_id"
        case title
        case valueText = "value_text"
        case previewOnly = "preview_only"
        case displayTitle = "display_title"
    }
}

public enum FastTierPreviewStatus: String, Codable, Sendable {
    case previewAvailable = "preview_available"
    case failClosed = "fail_closed"
}

public struct FastTierPreviewPayload: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var clipID: String
    public var tracks: [PreviewTrack]
    public var metricLabels: [PreviewMetricLabel]
    public var status: FastTierPreviewStatus
    public var failureReasons: [String]

    public init(
        schemaVersion: Int = 1,
        clipID: String,
        tracks: [PreviewTrack],
        metricLabels: [PreviewMetricLabel]
    ) {
        let failureReasons = Self.failureReasons(for: tracks, metricLabels: metricLabels)

        self.schemaVersion = schemaVersion
        self.clipID = clipID
        self.tracks = tracks
        self.metricLabels = metricLabels
        self.status = failureReasons.isEmpty ? .previewAvailable : .failClosed
        self.failureReasons = failureReasons
    }

    private static func failureReasons(for tracks: [PreviewTrack], metricLabels: [PreviewMetricLabel]) -> [String] {
        var reasons: [String] = []
        let hasPoseTrack = tracks.contains { track in
            track.kind == .pose2D || track.kind == .pose3D || track.kind == .handPose
        }

        if !hasPoseTrack {
            reasons.append("missing_preview_pose_track")
        }

        // ON-DEVICE LIVE payloads are guidance only; server-deep artifacts and
        // authoritative coaching metrics must stay outside this contract.
        if tracks.contains(where: { !$0.previewOnly }) || metricLabels.contains(where: { !$0.previewOnly }) {
            reasons.append("non_preview_data_in_fast_tier")
        }

        return reasons
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case clipID = "clip_id"
        case tracks
        case metricLabels = "metric_labels"
        case status
        case failureReasons = "failure_reasons"
    }
}

public struct OnDevicePoseTrack: Codable, Equatable, Sendable {
    public var relativePath: String
    public var fps: Double
    public var previewOnly: Bool

    public init(relativePath: String, fps: Double, previewOnly: Bool = true) {
        self.relativePath = relativePath
        self.fps = fps
        self.previewOnly = previewOnly
    }
}
