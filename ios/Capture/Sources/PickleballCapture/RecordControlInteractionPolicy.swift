import Foundation

public enum RecordControlState: Equatable, Sendable {
    case idle
    case preparing
    case blocked(reason: String)
    case ready
    case recording
}

public enum RecordGuidanceKind: Equatable, Sendable {
    case preparing
    case rotateToLandscape
    case blocked
}

public struct RecordGuidancePresentation: Equatable, Sendable {
    public var kind: RecordGuidanceKind
    public var title: String
    public var message: String
    public var systemImage: String
    public var actionTitle: String?
    public var accentClusterCount: Int
    public var isPersistent: Bool
    public var usesHighContrastSurface: Bool

    public init(
        kind: RecordGuidanceKind,
        title: String,
        message: String,
        systemImage: String,
        actionTitle: String?,
        accentClusterCount: Int,
        isPersistent: Bool,
        usesHighContrastSurface: Bool
    ) {
        self.kind = kind
        self.title = title
        self.message = message
        self.systemImage = systemImage
        self.actionTitle = actionTitle
        self.accentClusterCount = accentClusterCount
        self.isPersistent = isPersistent
        self.usesHighContrastSurface = usesHighContrastSurface
    }
}

public enum RecordTapReactionKind: Equatable, Sendable {
    case standard
    case preparing
    case blocked
}

public enum RecordTapReducedMotionEmphasis: Equatable, Sendable {
    case pressDepress
    case staticHighlight
}

public struct RecordControlTapReaction: Equatable, Sendable {
    public var kind: RecordTapReactionKind
    public var message: String
    public var hasVisibleConsequence: Bool
    public var usesWarningHaptic: Bool
    public var reducedMotionEmphasis: RecordTapReducedMotionEmphasis

    public init(
        kind: RecordTapReactionKind,
        message: String,
        hasVisibleConsequence: Bool = true,
        usesWarningHaptic: Bool,
        reducedMotionEmphasis: RecordTapReducedMotionEmphasis
    ) {
        self.kind = kind
        self.message = message
        self.hasVisibleConsequence = hasVisibleConsequence
        self.usesWarningHaptic = usesWarningHaptic
        self.reducedMotionEmphasis = reducedMotionEmphasis
    }
}

public struct RecordControlTapFeedback: Equatable, Sendable {
    public var sequence: Int
    public var reaction: RecordControlTapReaction

    public init(sequence: Int, reaction: RecordControlTapReaction) {
        self.sequence = sequence
        self.reaction = reaction
    }

    public static let initial = RecordControlTapFeedback(
        sequence: 0,
        reaction: RecordControlInteractionPolicy.tapReaction(for: .idle)
    )
}

public struct RecordControlAccessibilityState: Equatable, Sendable {
    public var elementExists: Bool
    public var isEnabled: Bool

    public init(elementExists: Bool, isEnabled: Bool) {
        self.elementExists = elementExists
        self.isEnabled = isEnabled
    }
}

public enum RecordControlInteractionPolicy {
    public static func guidance(for state: RecordControlState) -> RecordGuidancePresentation? {
        switch state {
        case .preparing:
            return RecordGuidancePresentation(
                kind: .preparing,
                title: "Setting up camera…",
                message: "Approve Camera + Microphone if asked.",
                systemImage: "camera.badge.ellipsis",
                actionTitle: nil,
                accentClusterCount: 1,
                isPersistent: true,
                usesHighContrastSurface: true
            )
        case .blocked(let reason):
            let isLandscapeBlocker = reason.localizedCaseInsensitiveContains("landscape")
                || reason.localizedCaseInsensitiveContains("rotate")
            return RecordGuidancePresentation(
                kind: isLandscapeBlocker ? .rotateToLandscape : .blocked,
                title: isLandscapeBlocker ? "Rotate to landscape" : "Recording needs attention",
                message: isLandscapeBlocker
                    ? "Turn your iPhone sideways to unlock recording."
                    : reason,
                systemImage: isLandscapeBlocker ? "rectangle.landscape.rotate" : "exclamationmark.triangle.fill",
                actionTitle: "Retry",
                accentClusterCount: 1,
                isPersistent: true,
                usesHighContrastSurface: true
            )
        case .idle, .ready, .recording:
            return nil
        }
    }

    public static func tapReaction(for state: RecordControlState) -> RecordControlTapReaction {
        switch state {
        case .preparing:
            return RecordControlTapReaction(
                kind: .preparing,
                message: "Setting up camera…",
                usesWarningHaptic: true,
                reducedMotionEmphasis: .staticHighlight
            )
        case .blocked(let reason):
            return RecordControlTapReaction(
                kind: .blocked,
                message: reason,
                usesWarningHaptic: true,
                reducedMotionEmphasis: .staticHighlight
            )
        case .idle, .ready, .recording:
            return RecordControlTapReaction(
                kind: .standard,
                message: "Record control pressed",
                usesWarningHaptic: false,
                reducedMotionEmphasis: .pressDepress
            )
        }
    }

    public static func accessibility(for _: RecordControlState) -> RecordControlAccessibilityState {
        RecordControlAccessibilityState(elementExists: true, isEnabled: true)
    }

    public static func blockedAnnouncement(
        previous: RecordControlState,
        current: RecordControlState
    ) -> String? {
        guard case .blocked(let reason) = current, previous != current else {
            return nil
        }
        return "Recording blocked. \(reason)"
    }
}
