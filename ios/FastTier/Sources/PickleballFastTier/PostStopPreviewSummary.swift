import Foundation
import PickleballCore

/// Quick post-stop summary shown while the real package saves in the
/// background, W3-LIVE-MLP surface 4. Gate: must be ready in under 10
/// seconds (`MASTER_PLAN` W3-LIVE-MLP gate row). Every field here is either
/// a direct readback (duration, requested fps, capture-quality grade) or an
/// explicitly-labeled estimate (player count from a handful of sampled
/// frames) -- never a number the pipeline can't defend.
public struct PostStopPreviewSummary: Equatable, Sendable {
    public var durationSeconds: Double
    public var requestedFPS: Int
    public var measuredFPS: Double?
    public var captureQualityGrade: CaptureQuality.Grade
    public var captureQualityReasons: [String]
    /// `nil` when zero frames were sampled (nothing to estimate from -- must
    /// render as "unavailable", never as 0 players).
    public var estimatedPlayerCount: Int?
    public var playerCountSampleFrameCount: Int
    public var elapsedBuildSeconds: Double
    public var isWithinPreviewBudget: Bool
    public var provenance: CaptureProvenance

    public init(
        durationSeconds: Double,
        requestedFPS: Int,
        measuredFPS: Double?,
        captureQualityGrade: CaptureQuality.Grade,
        captureQualityReasons: [String],
        estimatedPlayerCount: Int?,
        playerCountSampleFrameCount: Int,
        elapsedBuildSeconds: Double,
        isWithinPreviewBudget: Bool,
        provenance: CaptureProvenance = .liveRecording
    ) {
        self.durationSeconds = durationSeconds
        self.requestedFPS = requestedFPS
        self.measuredFPS = measuredFPS
        self.captureQualityGrade = captureQualityGrade
        self.captureQualityReasons = captureQualityReasons
        self.estimatedPlayerCount = estimatedPlayerCount
        self.playerCountSampleFrameCount = playerCountSampleFrameCount
        self.elapsedBuildSeconds = elapsedBuildSeconds
        self.isWithinPreviewBudget = isWithinPreviewBudget
        self.provenance = provenance
    }
}

public enum PostStopPreviewBuilder {
    public static let budgetSeconds: Double = 10.0

    /// `sampledFrameDetectionCounts` are player counts observed on a small
    /// number of frames sampled from the just-recorded clip (or from the
    /// live session's last few cadence-scheduled detections) -- NOT a full
    /// per-frame scan, hence "estimate."
    public static func summarize(
        durationSeconds: Double,
        requestedFPS: Int,
        measuredFPS: Double? = nil,
        captureQuality: CaptureQuality,
        sampledFrameDetectionCounts: [Int],
        elapsedBuildSeconds: Double,
        budgetSeconds: Double = PostStopPreviewBuilder.budgetSeconds,
        provenance: CaptureProvenance = .liveRecording
    ) -> PostStopPreviewSummary {
        PostStopPreviewSummary(
            durationSeconds: max(0, durationSeconds),
            requestedFPS: requestedFPS,
            measuredFPS: measuredFPS,
            captureQualityGrade: captureQuality.grade,
            captureQualityReasons: captureQuality.reasons,
            estimatedPlayerCount: medianCount(sampledFrameDetectionCounts),
            playerCountSampleFrameCount: sampledFrameDetectionCounts.count,
            elapsedBuildSeconds: max(0, elapsedBuildSeconds),
            isWithinPreviewBudget: elapsedBuildSeconds <= budgetSeconds,
            provenance: provenance
        )
    }

    /// Median (not mean) of sampled per-frame player counts: robust to a
    /// stray frame or two where the detector missed everyone or double
    /// counted, without needing a full-clip scan.
    static func medianCount(_ counts: [Int]) -> Int? {
        guard !counts.isEmpty else {
            return nil
        }
        let sorted = counts.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return Int((Double(sorted[middle - 1] + sorted[middle]) / 2.0).rounded())
        }
        return sorted[middle]
    }
}
