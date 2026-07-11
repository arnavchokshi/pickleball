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
    public var replaySceneURL: String?
    public var bodyMeshURL: String?
    public var bodyMeshIndexURL: String?
    public var physicsRefinementURL: String?
    public var contactWindowsURL: String?
    public var reviewedBouncesURL: String?
    public var ballInflectionsURL: String?
    public var eventsSelectedURL: String?
    public var shotsURL: String?
    public var ballArcSolvedURL: String?
    public var ballArcRenderURL: String?
    public var autoBounceCandidatesURL: String?
    public var ballBounceCandidatesURL: String?
    public var ballFlightSanityURL: String?
    public var rallySpansURL: String?
    public var rallyMetricsURL: String?
    public var coachingCardFactsURL: String?
    public var labelOverlays: [LabelOverlay]
    public var annotationSources: [AnnotationSource]
    public var notes: [String]

    public struct LabelOverlay: Codable, Equatable, Sendable {
        public var kind: String
        public var label: String
        public var url: String
        public var trustedForMetrics: Bool
        public var notGroundTruth: Bool

        private enum CodingKeys: String, CodingKey {
            case kind, label, url
            case trustedForMetrics = "trusted_for_metrics"
            case notGroundTruth = "not_ground_truth"
        }
    }

    public struct AnnotationSource: Codable, Equatable, Sendable {
        public var kind: String
        public var clipId: String
        public var url: String
        public var trustedForMetrics: Bool

        private enum CodingKeys: String, CodingKey {
            case kind
            case clipId = "clip_id"
            case url
            case trustedForMetrics = "trusted_for_metrics"
        }
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifactType = "artifact_type"
        case clip
        case videoURL = "video_url"
        case virtualWorldURL = "virtual_world_url"
        case replaySceneURL = "replay_scene_url"
        case bodyMeshURL = "body_mesh_url"
        case bodyMeshIndexURL = "body_mesh_index_url"
        case physicsRefinementURL = "physics_refinement_url"
        case contactWindowsURL = "contact_windows_url"
        case reviewedBouncesURL = "reviewed_bounces_url"
        case ballInflectionsURL = "ball_inflections_url"
        case eventsSelectedURL = "events_selected_url"
        case shotsURL = "shots_url"
        case ballArcSolvedURL = "ball_arc_solved_url"
        case ballArcRenderURL = "ball_arc_render_url"
        case autoBounceCandidatesURL = "auto_bounce_candidates_url"
        case ballBounceCandidatesURL = "ball_bounce_candidates_url"
        case ballFlightSanityURL = "ball_flight_sanity_url"
        case rallySpansURL = "rally_spans_url"
        case rallyMetricsURL = "rally_metrics_url"
        case coachingCardFactsURL = "coaching_card_facts_url"
        case labelOverlays = "label_overlays"
        case annotationSources = "annotation_sources"
        case notes
    }

    public init(
        schemaVersion: Int,
        artifactType: String,
        clip: String,
        videoURL: String,
        virtualWorldURL: String,
        replaySceneURL: String? = nil,
        bodyMeshURL: String? = nil,
        bodyMeshIndexURL: String? = nil,
        physicsRefinementURL: String? = nil,
        contactWindowsURL: String? = nil,
        reviewedBouncesURL: String? = nil,
        ballInflectionsURL: String? = nil,
        eventsSelectedURL: String? = nil,
        shotsURL: String? = nil,
        ballArcSolvedURL: String? = nil,
        ballArcRenderURL: String? = nil,
        autoBounceCandidatesURL: String? = nil,
        ballBounceCandidatesURL: String? = nil,
        ballFlightSanityURL: String? = nil,
        rallySpansURL: String? = nil,
        rallyMetricsURL: String? = nil,
        coachingCardFactsURL: String? = nil,
        labelOverlays: [LabelOverlay] = [],
        annotationSources: [AnnotationSource] = [],
        notes: [String] = []
    ) {
        self.schemaVersion = schemaVersion
        self.artifactType = artifactType
        self.clip = clip
        self.videoURL = videoURL
        self.virtualWorldURL = virtualWorldURL
        self.replaySceneURL = replaySceneURL
        self.bodyMeshURL = bodyMeshURL
        self.bodyMeshIndexURL = bodyMeshIndexURL
        self.physicsRefinementURL = physicsRefinementURL
        self.contactWindowsURL = contactWindowsURL
        self.reviewedBouncesURL = reviewedBouncesURL
        self.ballInflectionsURL = ballInflectionsURL
        self.eventsSelectedURL = eventsSelectedURL
        self.shotsURL = shotsURL
        self.ballArcSolvedURL = ballArcSolvedURL
        self.ballArcRenderURL = ballArcRenderURL
        self.autoBounceCandidatesURL = autoBounceCandidatesURL
        self.ballBounceCandidatesURL = ballBounceCandidatesURL
        self.ballFlightSanityURL = ballFlightSanityURL
        self.rallySpansURL = rallySpansURL
        self.rallyMetricsURL = rallyMetricsURL
        self.coachingCardFactsURL = coachingCardFactsURL
        self.labelOverlays = labelOverlays
        self.annotationSources = annotationSources
        self.notes = notes
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        artifactType = try container.decode(String.self, forKey: .artifactType)
        clip = try container.decode(String.self, forKey: .clip)
        videoURL = try container.decode(String.self, forKey: .videoURL)
        virtualWorldURL = try container.decode(String.self, forKey: .virtualWorldURL)
        replaySceneURL = try container.decodeIfPresent(String.self, forKey: .replaySceneURL)
        bodyMeshURL = try container.decodeIfPresent(String.self, forKey: .bodyMeshURL)
        bodyMeshIndexURL = try container.decodeIfPresent(String.self, forKey: .bodyMeshIndexURL)
        physicsRefinementURL = try container.decodeIfPresent(String.self, forKey: .physicsRefinementURL)
        contactWindowsURL = try container.decodeIfPresent(String.self, forKey: .contactWindowsURL)
        reviewedBouncesURL = try container.decodeIfPresent(String.self, forKey: .reviewedBouncesURL)
        ballInflectionsURL = try container.decodeIfPresent(String.self, forKey: .ballInflectionsURL)
        eventsSelectedURL = try container.decodeIfPresent(String.self, forKey: .eventsSelectedURL)
        shotsURL = try container.decodeIfPresent(String.self, forKey: .shotsURL)
        ballArcSolvedURL = try container.decodeIfPresent(String.self, forKey: .ballArcSolvedURL)
        ballArcRenderURL = try container.decodeIfPresent(String.self, forKey: .ballArcRenderURL)
        autoBounceCandidatesURL = try container.decodeIfPresent(String.self, forKey: .autoBounceCandidatesURL)
        ballBounceCandidatesURL = try container.decodeIfPresent(String.self, forKey: .ballBounceCandidatesURL)
        ballFlightSanityURL = try container.decodeIfPresent(String.self, forKey: .ballFlightSanityURL)
        rallySpansURL = try container.decodeIfPresent(String.self, forKey: .rallySpansURL)
        rallyMetricsURL = try container.decodeIfPresent(String.self, forKey: .rallyMetricsURL)
        coachingCardFactsURL = try container.decodeIfPresent(String.self, forKey: .coachingCardFactsURL)
        labelOverlays = try container.decodeIfPresent([LabelOverlay].self, forKey: .labelOverlays) ?? []
        annotationSources = try container.decodeIfPresent([AnnotationSource].self, forKey: .annotationSources) ?? []
        notes = try container.decodeIfPresent([String].self, forKey: .notes) ?? []
    }
}

extension WorldViewerManifest {
    public static func load(from url: URL, decoder: JSONDecoder = JSONDecoder()) throws -> WorldViewerManifest {
        let data = try Data(contentsOf: url)
        return try decoder.decode(WorldViewerManifest.self, from: data)
    }

    public static func decode(_ data: Data, decoder: JSONDecoder = JSONDecoder()) throws -> WorldViewerManifest {
        try decoder.decode(WorldViewerManifest.self, from: data)
    }
}
