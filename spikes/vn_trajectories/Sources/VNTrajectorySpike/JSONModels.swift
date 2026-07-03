import Foundation

// MARK: - Harness output schema
//
// This is the RAW candidate-trajectory schema written by the harness. It is
// deliberately close to, but not identical to, the repo's `ball_track.json`
// schema (`threed/racketsport/schemas/__init__.py::BallTrack`): Vision emits
// zero or more *candidate parabolic trajectories* per analyzed frame (a
// sliding window over the last `trajectory_length` frames), while
// `ball_track.json` wants exactly one (x, y, conf, visible) sample per video
// frame. `scripts/racketsport/convert_vn_trajectories.py` does that
// candidate -> single-track reduction. Keeping the raw multi-candidate
// output around (rather than collapsing it in Swift) is what makes this
// schema honestly "convertible to ball_track.json" rather than already-lossy.

struct HarnessOutput: Codable {
    let schemaVersion: Int
    let artifactType: String
    let status: String
    let sourceVideo: String
    let video: VideoMeta
    let requestConfig: RequestConfig
    let run: RunMeta
    /// Presentation timestamp (seconds, video-relative) for every frame fed
    /// into the request, indexed by frame index. This is the source of
    /// truth the converter uses to turn a trajectory-window frame offset
    /// back into an absolute `t` value.
    let framePtsS: [Double]
    let trajectories: [TrajectoryEmission]
    let notes: [String]
}

struct VideoMeta: Codable {
    let width: Int
    let height: Int
    let fps: Double
    let frameCount: Int
    let durationS: Double
}

struct RequestConfig: Codable {
    /// "zero" (analyze every fed frame) is the only mode this harness
    /// drives today; recorded as a string so a future device variant can
    /// also record fractional CMTime spacing without changing the schema.
    let frameAnalysisSpacing: String
    let trajectoryLength: Int
    let objectMinimumNormalizedRadius: Double
    let objectMaximumNormalizedRadius: Double
    let visionRequestRevision: Int
}

struct RunMeta: Codable {
    let framesFed: Int
    let performCallCount: Int
    let emissionCount: Int
    let wallClockSeconds: Double
    let framesPerSecondProcessed: Double
}

struct TrajectoryEmission: Codable {
    let observationUuid: String
    /// Frame index (0-based, matches `framePtsS`) of the most-recently-fed
    /// frame at the moment Vision produced this observation. Because
    /// VNDetectTrajectoriesRequest is a sliding-window stateful request,
    /// this is emitted once per fed frame once the internal window has
    /// enough history (>= 2 frames; it warms up towards trajectoryLength).
    let emittedAtFrameIndex: Int
    let confidence: Double
    let movingAverageRadiusNormalized: Double
    /// Parabola fit coefficients [a, b, c] (VNTrajectoryObservation.equationCoefficients).
    let equationCoefficients: [Double]
    let timeRangeStartS: Double
    let timeRangeDurationS: Double
    let detectedPoints: [TrajPoint]
    let projectedPoints: [TrajPoint]
}

struct TrajPoint: Codable {
    /// Absolute video frame index this point corresponds to, when known.
    let frameIndex: Int?
    let tS: Double?
    /// Vision normalized coordinates, ORIGIN BOTTOM-LEFT, range [0, 1].
    /// The converter flips Y and scales by video width/height to match
    /// ball_track.json's top-left pixel-space convention.
    let xNorm: Double
    let yNorm: Double
}

struct BlockerReport: Codable {
    let schemaVersion: Int
    let artifactType: String
    let status: String
    let blockedReason: String
    let detail: String
    let sourceVideo: String
    let osVersion: String
    let notes: [String]
}

enum JSONWriter {
    static func write<T: Encodable>(_ value: T, to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(value)
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: .atomic)
    }
}
