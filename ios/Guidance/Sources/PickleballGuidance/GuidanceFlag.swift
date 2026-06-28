import Foundation
import PickleballCore

public enum GuidanceFlag: String, Codable, Sendable {
    case framing
    case tracking
    case exposure
    case blur
    case level
    case shake
}

public enum GuidanceStatus: String, Codable, Sendable {
    case good
    case needsCorrection = "needs_correction"
    case failClosed = "fail_closed"
}

public struct GuidanceState: Codable, Equatable, Sendable {
    public var flags: [GuidanceFlag]
    public var captureQuality: CaptureQuality
    public var status: GuidanceStatus

    public init(flags: [GuidanceFlag] = [], captureQuality: CaptureQuality, status: GuidanceStatus? = nil) {
        self.flags = flags
        self.captureQuality = captureQuality
        self.status = status ?? (flags.isEmpty ? .good : .needsCorrection)
    }
}

public struct GuidanceSample: Codable, Equatable, Sendable {
    public var visibleCourtCornerRatio: Double?
    public var isTrackingNormal: Bool?
    public var exposureTargetOffset: Double?
    public var clippedPixelRatio: Double?
    public var shutterSeconds: Double?
    public var minimumSharpShutterSeconds: Double?
    public var rollDegrees: Double?
    public var pitchDegrees: Double?
    public var accelerationVarianceG: Double?

    public init(
        visibleCourtCornerRatio: Double? = nil,
        isTrackingNormal: Bool? = nil,
        exposureTargetOffset: Double? = nil,
        clippedPixelRatio: Double? = nil,
        shutterSeconds: Double? = nil,
        minimumSharpShutterSeconds: Double? = nil,
        rollDegrees: Double? = nil,
        pitchDegrees: Double? = nil,
        accelerationVarianceG: Double? = nil
    ) {
        self.visibleCourtCornerRatio = visibleCourtCornerRatio
        self.isTrackingNormal = isTrackingNormal
        self.exposureTargetOffset = exposureTargetOffset
        self.clippedPixelRatio = clippedPixelRatio
        self.shutterSeconds = shutterSeconds
        self.minimumSharpShutterSeconds = minimumSharpShutterSeconds
        self.rollDegrees = rollDegrees
        self.pitchDegrees = pitchDegrees
        self.accelerationVarianceG = accelerationVarianceG
    }
}

public struct GuidanceRuleThresholds: Codable, Equatable, Sendable {
    public var minimumVisibleCourtCornerRatio: Double
    public var maximumExposureTargetOffset: Double
    public var maximumClippedPixelRatio: Double
    public var maximumLevelDegrees: Double
    public var maximumAccelerationVarianceG: Double

    public init(
        minimumVisibleCourtCornerRatio: Double = 0.95,
        maximumExposureTargetOffset: Double = 0.75,
        maximumClippedPixelRatio: Double = 0.03,
        maximumLevelDegrees: Double = 5.0,
        maximumAccelerationVarianceG: Double = 0.06
    ) {
        self.minimumVisibleCourtCornerRatio = minimumVisibleCourtCornerRatio
        self.maximumExposureTargetOffset = maximumExposureTargetOffset
        self.maximumClippedPixelRatio = maximumClippedPixelRatio
        self.maximumLevelDegrees = maximumLevelDegrees
        self.maximumAccelerationVarianceG = maximumAccelerationVarianceG
    }
}

public enum CaptureGuidanceEvaluator {
    public static func evaluate(
        _ sample: GuidanceSample,
        thresholds: GuidanceRuleThresholds = GuidanceRuleThresholds()
    ) -> GuidanceState {
        var flags: [GuidanceFlag] = []
        var reasons: [String] = []
        var hasMissingMeasurement = false

        appendFramingResult(sample, thresholds, flags: &flags, reasons: &reasons, hasMissingMeasurement: &hasMissingMeasurement)
        appendTrackingResult(sample, flags: &flags, reasons: &reasons, hasMissingMeasurement: &hasMissingMeasurement)
        appendExposureResult(sample, thresholds, flags: &flags, reasons: &reasons, hasMissingMeasurement: &hasMissingMeasurement)
        appendBlurResult(sample, flags: &flags, reasons: &reasons, hasMissingMeasurement: &hasMissingMeasurement)
        appendLevelResult(sample, thresholds, flags: &flags, reasons: &reasons, hasMissingMeasurement: &hasMissingMeasurement)
        appendShakeResult(sample, thresholds, flags: &flags, reasons: &reasons, hasMissingMeasurement: &hasMissingMeasurement)

        let grade: CaptureQuality.Grade
        let status: GuidanceStatus
        if hasMissingMeasurement {
            grade = .poor
            status = .failClosed
        } else if flags.isEmpty {
            grade = .good
            status = .good
        } else if flags.count >= 3 {
            grade = .poor
            status = .needsCorrection
        } else {
            grade = .warn
            status = .needsCorrection
        }

        return GuidanceState(
            flags: flags,
            captureQuality: CaptureQuality(grade: grade, reasons: reasons),
            status: status
        )
    }

    private static func appendFramingResult(
        _ sample: GuidanceSample,
        _ thresholds: GuidanceRuleThresholds,
        flags: inout [GuidanceFlag],
        reasons: inout [String],
        hasMissingMeasurement: inout Bool
    ) {
        guard let ratio = sample.visibleCourtCornerRatio else {
            append(.framing, reason: "missing_framing_measurement", flags: &flags, reasons: &reasons)
            hasMissingMeasurement = true
            return
        }

        if ratio < thresholds.minimumVisibleCourtCornerRatio {
            append(.framing, reason: "framing_court_corners_not_visible", flags: &flags, reasons: &reasons)
        }
    }

    private static func appendTrackingResult(
        _ sample: GuidanceSample,
        flags: inout [GuidanceFlag],
        reasons: inout [String],
        hasMissingMeasurement: inout Bool
    ) {
        guard let isTrackingNormal = sample.isTrackingNormal else {
            append(.tracking, reason: "missing_tracking_measurement", flags: &flags, reasons: &reasons)
            hasMissingMeasurement = true
            return
        }

        if !isTrackingNormal {
            append(.tracking, reason: "tracking_not_normal", flags: &flags, reasons: &reasons)
        }
    }

    private static func appendExposureResult(
        _ sample: GuidanceSample,
        _ thresholds: GuidanceRuleThresholds,
        flags: inout [GuidanceFlag],
        reasons: inout [String],
        hasMissingMeasurement: inout Bool
    ) {
        guard let exposureTargetOffset = sample.exposureTargetOffset, let clippedPixelRatio = sample.clippedPixelRatio else {
            append(.exposure, reason: "missing_exposure_measurement", flags: &flags, reasons: &reasons)
            hasMissingMeasurement = true
            return
        }

        if abs(exposureTargetOffset) > thresholds.maximumExposureTargetOffset
            || clippedPixelRatio > thresholds.maximumClippedPixelRatio {
            append(.exposure, reason: "exposure_clipping_or_offset", flags: &flags, reasons: &reasons)
        }
    }

    private static func appendBlurResult(
        _ sample: GuidanceSample,
        flags: inout [GuidanceFlag],
        reasons: inout [String],
        hasMissingMeasurement: inout Bool
    ) {
        guard let shutterSeconds = sample.shutterSeconds,
              let minimumSharpShutterSeconds = sample.minimumSharpShutterSeconds else {
            append(.blur, reason: "missing_blur_measurement", flags: &flags, reasons: &reasons)
            hasMissingMeasurement = true
            return
        }

        if shutterSeconds > minimumSharpShutterSeconds {
            append(.blur, reason: "motion_blur_risk", flags: &flags, reasons: &reasons)
        }
    }

    private static func appendLevelResult(
        _ sample: GuidanceSample,
        _ thresholds: GuidanceRuleThresholds,
        flags: inout [GuidanceFlag],
        reasons: inout [String],
        hasMissingMeasurement: inout Bool
    ) {
        guard let rollDegrees = sample.rollDegrees, let pitchDegrees = sample.pitchDegrees else {
            append(.level, reason: "missing_level_measurement", flags: &flags, reasons: &reasons)
            hasMissingMeasurement = true
            return
        }

        if abs(rollDegrees) > thresholds.maximumLevelDegrees || abs(pitchDegrees) > thresholds.maximumLevelDegrees {
            append(.level, reason: "phone_not_level", flags: &flags, reasons: &reasons)
        }
    }

    private static func appendShakeResult(
        _ sample: GuidanceSample,
        _ thresholds: GuidanceRuleThresholds,
        flags: inout [GuidanceFlag],
        reasons: inout [String],
        hasMissingMeasurement: inout Bool
    ) {
        guard let accelerationVarianceG = sample.accelerationVarianceG else {
            append(.shake, reason: "missing_shake_measurement", flags: &flags, reasons: &reasons)
            hasMissingMeasurement = true
            return
        }

        if accelerationVarianceG > thresholds.maximumAccelerationVarianceG {
            append(.shake, reason: "camera_shake", flags: &flags, reasons: &reasons)
        }
    }

    private static func append(
        _ flag: GuidanceFlag,
        reason: String,
        flags: inout [GuidanceFlag],
        reasons: inout [String]
    ) {
        if !flags.contains(flag) {
            flags.append(flag)
        }
        reasons.append(reason)
    }
}
