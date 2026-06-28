import Foundation

public struct ReplayInputManifest: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var artifactType: String
    public var clips: [ReplayInputClip]

    public init(
        schemaVersion: Int = 1,
        artifactType: String = "pickleball_replay_input_manifest",
        clips: [ReplayInputClip]
    ) {
        self.schemaVersion = schemaVersion
        self.artifactType = artifactType
        self.clips = clips
    }

    public static func load(from url: URL, decoder: JSONDecoder = JSONDecoder()) throws -> ReplayInputManifest {
        let data = try Data(contentsOf: url)
        return try decoder.decode(ReplayInputManifest.self, from: data)
    }

    public func resolvedClips(relativeTo baseURL: URL) -> [ResolvedReplayInputClip] {
        clips.map { clip in
            ResolvedReplayInputClip(
                clipID: clip.clipID,
                name: clip.name,
                videoURL: baseURL.appendingPathComponent(clip.video),
                groundTruthURL: baseURL.appendingPathComponent(clip.groundTruth),
                expectedPlayers: clip.expectedPlayers
            )
        }
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifactType = "artifact_type"
        case clips
    }
}

public struct ReplayInputClip: Codable, Equatable, Sendable {
    public var clipID: String
    public var name: String
    public var video: String
    public var groundTruth: String
    public var expectedPlayers: Int

    public init(clipID: String, name: String, video: String, groundTruth: String, expectedPlayers: Int) {
        self.clipID = clipID
        self.name = name
        self.video = video
        self.groundTruth = groundTruth
        self.expectedPlayers = expectedPlayers
    }

    private enum CodingKeys: String, CodingKey {
        case clipID = "clip_id"
        case name
        case video
        case groundTruth = "ground_truth"
        case expectedPlayers = "expected_players"
    }
}

public struct ResolvedReplayInputClip: Equatable, Sendable {
    public var clipID: String
    public var name: String
    public var videoURL: URL
    public var groundTruthURL: URL
    public var expectedPlayers: Int

    public init(clipID: String, name: String, videoURL: URL, groundTruthURL: URL, expectedPlayers: Int) {
        self.clipID = clipID
        self.name = name
        self.videoURL = videoURL
        self.groundTruthURL = groundTruthURL
        self.expectedPlayers = expectedPlayers
    }
}
