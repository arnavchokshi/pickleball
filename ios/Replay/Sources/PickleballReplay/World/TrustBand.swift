import Foundation

/// Mirrors `TrustBadge` in `web/replay/src/viewerData.ts`. Every renderable
/// world entity (court, ball, each player, each paddle) carries one of
/// these, straight from the pipeline's own gate state -- never inferred or
/// guessed on-device.
public enum TrustBadge: String, Codable, Equatable, Sendable {
    case verified
    case preview
    case lowConfidence = "low_confidence"
}

/// Mirrors `TrustBand` in `web/replay/src/viewerData.ts`: one gate-ladder
/// provenance record per entity (`stage`, `gate_id`, `gate_status`, `badge`,
/// human-readable `reason`, optional `evidence_path`).
public struct TrustBand: Codable, Equatable, Sendable {
    public var stage: String
    public var gateID: String
    public var gateStatus: String
    public var badge: TrustBadge
    public var reason: String
    public var evidencePath: String?

    private enum CodingKeys: String, CodingKey {
        case stage
        case gateID = "gate_id"
        case gateStatus = "gate_status"
        case badge
        case reason
        case evidencePath = "evidence_path"
    }

    public init(stage: String, gateID: String, gateStatus: String, badge: TrustBadge, reason: String, evidencePath: String?) {
        self.stage = stage
        self.gateID = gateID
        self.gateStatus = gateStatus
        self.badge = badge
        self.reason = reason
        self.evidencePath = evidencePath
    }
}

public enum TrustBandPresentation {
    /// Short "STAGE: badge" chip text, mirroring `trustBandChipText` in the
    /// web viewer. Absent trust bands render an explicit "no trust band"
    /// placeholder rather than silently showing nothing.
    public static func chipText(_ trustBand: TrustBand?) -> String {
        guard let trustBand else { return "no trust band" }
        return "\(trustBand.stage): \(trustBand.badge.rawValue.replacingOccurrences(of: "_", with: " "))"
    }

    /// A badge with no trust band at all is treated as `verified` for
    /// coloring purposes, mirroring `trustBadgeColor(undefined)` in the web
    /// viewer (used only for entities with no gate concept yet, e.g. court
    /// wireframe geometry before CAL trust bands existed).
    public static func badge(for trustBand: TrustBand?) -> TrustBadge {
        trustBand?.badge ?? .verified
    }
}
