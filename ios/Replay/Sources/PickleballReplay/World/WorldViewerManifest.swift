import Foundation

/// Swift mirror of `replay_viewer_manifest.json`
/// (`racketsport_replay_viewer_manifest`), mirroring `ViewerManifest` in
/// `web/replay/src/viewerData.ts`. Points at the sibling artifact files
/// (relative to the manifest's own directory) that make up one world
/// bundle.
public struct WorldViewerManifest: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var artifactType: String
    public var clip: String
    public var videoURL: String
    public var virtualWorldURL: String
    public var bodyMeshURL: String?
    public var physicsRefinementURL: String?
    public var contactWindowsURL: String?
    public var ballInflectionsURL: String?
    public var notes: [String]

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifactType = "artifact_type"
        case clip
        case videoURL = "video_url"
        case virtualWorldURL = "virtual_world_url"
        case bodyMeshURL = "body_mesh_url"
        case physicsRefinementURL = "physics_refinement_url"
        case contactWindowsURL = "contact_windows_url"
        case ballInflectionsURL = "ball_inflections_url"
        case notes
    }

    public init(
        schemaVersion: Int,
        artifactType: String,
        clip: String,
        videoURL: String,
        virtualWorldURL: String,
        bodyMeshURL: String?,
        physicsRefinementURL: String?,
        contactWindowsURL: String?,
        ballInflectionsURL: String?,
        notes: [String]
    ) {
        self.schemaVersion = schemaVersion
        self.artifactType = artifactType
        self.clip = clip
        self.videoURL = videoURL
        self.virtualWorldURL = virtualWorldURL
        self.bodyMeshURL = bodyMeshURL
        self.physicsRefinementURL = physicsRefinementURL
        self.contactWindowsURL = contactWindowsURL
        self.ballInflectionsURL = ballInflectionsURL
        self.notes = notes
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        artifactType = try container.decode(String.self, forKey: .artifactType)
        clip = try container.decode(String.self, forKey: .clip)
        videoURL = try container.decode(String.self, forKey: .videoURL)
        virtualWorldURL = try container.decode(String.self, forKey: .virtualWorldURL)
        bodyMeshURL = try container.decodeIfPresent(String.self, forKey: .bodyMeshURL)
        physicsRefinementURL = try container.decodeIfPresent(String.self, forKey: .physicsRefinementURL)
        contactWindowsURL = try container.decodeIfPresent(String.self, forKey: .contactWindowsURL)
        ballInflectionsURL = try container.decodeIfPresent(String.self, forKey: .ballInflectionsURL)
        notes = try container.decodeIfPresent([String].self, forKey: .notes) ?? []
    }
}

extension WorldViewerManifest {
    public static func load(from url: URL, decoder: JSONDecoder = JSONDecoder()) throws -> WorldViewerManifest {
        let data = try Data(contentsOf: url)
        return try decoder.decode(WorldViewerManifest.self, from: data)
    }
}
