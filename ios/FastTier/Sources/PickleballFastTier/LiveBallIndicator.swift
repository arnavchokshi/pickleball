import Foundation

/// Confidence-gated live ball indicator, W3-LIVE-MLP surface 3.
///
/// `ball_student` (the CoreML model benchmarked at 1.41ms/frame on the real
/// iPhone 14 Pro in `runs/ios_device_gate_20260702T025809Z/LATENCY_TABLE_DEVICE.md`)
/// is UNTRAINED -- that benchmark exists to validate the CoreML
/// deploy/convert/run path, not to produce a real ball position. Showing its
/// raw output as a ball dot would be exactly the kind of confidently-wrong
/// signal the product must never render.
///
/// `LiveBallIndicatorPolicy.evaluate` is the single gate every live ball
/// signal must pass through. `modelIsTrained` is `false` everywhere in this
/// build (see `LiveBallIndicatorPolicy.modelIsTrainedInThisBuild`) and, as
/// long as it is `false`, the function ignores whatever confidence/position
/// values are passed in and always returns `.comingSoon` -- this is asserted
/// by `LiveBallIndicatorTests.testUntrainedModelIgnoresConfidenceEvenWhenHigh`.
/// The swap-in path for a real trained student is exactly one line: flip
/// `modelIsTrainedInThisBuild` to `true` once a trained `ball_student`
/// checkpoint clears its own gate (see MASTER_PLAN W1-BALL/W2-IOS-BALL).
public enum LiveBallIndicatorAvailability: String, Codable, Equatable, Sendable {
    /// v0 default: model exists on-device but is untrained, so its output is
    /// never shown. This is the ONLY state reachable while
    /// `modelIsTrainedInThisBuild == false`.
    case comingSoon = "coming_soon"
    /// Reachable only once a trained model ships: the model ran but its
    /// confidence did not clear `confidenceThreshold`.
    case lowConfidence = "low_confidence"
    /// Reachable only once a trained model ships and clears the threshold.
    case tracking
}

public struct LiveBallIndicatorState: Equatable, Sendable {
    public var availability: LiveBallIndicatorAvailability
    public var badgeText: String
    public var normalizedX: Double?
    public var normalizedY: Double?
    public var confidence: Double?

    public init(
        availability: LiveBallIndicatorAvailability,
        badgeText: String,
        normalizedX: Double? = nil,
        normalizedY: Double? = nil,
        confidence: Double? = nil
    ) {
        self.availability = availability
        self.badgeText = badgeText
        self.normalizedX = normalizedX
        self.normalizedY = normalizedY
        self.confidence = confidence
    }

    public static let comingSoon = LiveBallIndicatorState(
        availability: .comingSoon,
        badgeText: "Ball tracking: coming soon"
    )
}

public enum LiveBallIndicatorPolicy {
    /// Documented placeholder threshold for when a trained student exists;
    /// unused while `modelIsTrainedInThisBuild == false`.
    public static let confidenceThreshold: Double = 0.5

    /// v0 kill switch. Flip only once a trained `ball_student` checkpoint has
    /// cleared its own accuracy gate (MASTER_PLAN W1-BALL) -- never based on
    /// the deployment-path latency benchmark alone.
    public static let modelIsTrainedInThisBuild = false

    public static func evaluate(
        rawConfidence: Double?,
        rawNormalizedX: Double?,
        rawNormalizedY: Double?,
        modelIsTrained: Bool = modelIsTrainedInThisBuild
    ) -> LiveBallIndicatorState {
        guard modelIsTrained else {
            return .comingSoon
        }
        guard let rawConfidence, rawConfidence >= confidenceThreshold,
              let rawNormalizedX, let rawNormalizedY else {
            return LiveBallIndicatorState(
                availability: .lowConfidence,
                badgeText: "Ball: low confidence",
                confidence: rawConfidence
            )
        }
        return LiveBallIndicatorState(
            availability: .tracking,
            badgeText: "Ball tracking",
            normalizedX: rawNormalizedX,
            normalizedY: rawNormalizedY,
            confidence: rawConfidence
        )
    }
}
