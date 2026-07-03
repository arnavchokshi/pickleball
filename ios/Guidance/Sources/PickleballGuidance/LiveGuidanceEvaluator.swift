import Foundation
import PickleballCore

/// Pre-record live capture-quality guidance (W3-LIVE-MLP, surface 1).
///
/// This is deliberately a SEPARATE, lighter-weight evaluator from
/// `CaptureGuidanceEvaluator` above. That evaluator fails closed the moment
/// ANY of its six signals (framing/tracking/exposure/blur/level/shake) is
/// missing, which is the right behavior for the offline/full sidecar
/// validation it was built for, but it would make a *live* pre-record screen
/// permanently report "poor" today, since two of its inputs
/// (`isTrackingNormal` from ARKit and `clippedPixelRatio` from a live pixel
/// histogram) are not wired to any real signal the capture code currently
/// reads -- ARKit is not integrated yet (MASTER_PLAN: "ARKit seed/court-plane
/// are still absent"), and there is no live `AVCaptureVideoDataOutput` pixel
/// pipeline for a real clipped-pixel histogram in v0.
///
/// `LiveGuidanceEvaluator` instead evaluates each check independently: a
/// check with no real backing signal renders as `.unavailable` (honest, not
/// silently dropped, not faked good) and is excluded from the aggregate
/// grade, rather than poisoning the whole grade. Checks that DO have a real
/// signal today (exposure EV offset, phone tilt from CoreMotion gravity,
/// shutter-vs-light motion-blur risk, fps/resolution confirmation) are
/// evaluated for real. Court-corner framing has no live signal in v0 at all
/// (that would require live homography or a court detector) -- per the
/// v0 scope, it is surfaced ONLY as static manual guidance text, never as a
/// pass/fail check.
public enum LiveCheckStatus: String, Codable, Equatable, Sendable {
    case good
    case warn
    /// No real signal is wired for this check yet. Never rendered as good.
    case unavailable
}

public struct LiveGuidanceCheck: Codable, Equatable, Sendable {
    public var id: String
    public var title: String
    public var detail: String
    public var status: LiveCheckStatus

    public init(id: String, title: String, detail: String, status: LiveCheckStatus) {
        self.id = id
        self.title = title
        self.detail = detail
        self.status = status
    }
}

/// Real signals the capture code already reads, sampled just before/while
/// showing the pre-record guidance screen. Every field is optional because
/// an honest "not measured" must be representable -- `nil` renders as
/// `.unavailable`, never as `.good`.
public struct LiveGuidanceSample: Equatable, Sendable {
    /// `AVCaptureDevice.exposureTargetOffset`, in EV stops. Real per-frame
    /// AVFoundation photometric signal; doubles as the v0 "simple luminance"
    /// hint the milestone asks for (no per-pixel histogram in v0).
    public var exposureTargetOffsetEV: Double?
    public var isExposureLocked: Bool?
    public var shutterSeconds: Double?
    public var minimumSharpShutterSeconds: Double?
    /// Single combined "how far from level" angle derived from live
    /// CoreMotion gravity for the current capture orientation (v0
    /// simplification -- not decomposed into separate roll/pitch axes; see
    /// `LiveTiltEstimator`).
    public var tiltFromLevelDegrees: Double?
    public var requestedFPS: Int?
    /// fps actually configured on `AVCaptureDevice.activeVideoMinFrameDuration`
    /// -- a real device readback confirming the requested rate was honored,
    /// not a measured sustained frame rate under load.
    public var configuredFPS: Double?
    public var expectedResolution: [Int]?
    public var configuredResolution: [Int]?
    /// `capture_quality.reasons` style flags the capture pipeline already
    /// self-reports at record-stop today (e.g. "arkit_seed_missing"). Surfaced
    /// as static setup tips, never as a live pass/fail check.
    public var setupTipReasons: [String]

    public init(
        exposureTargetOffsetEV: Double? = nil,
        isExposureLocked: Bool? = nil,
        shutterSeconds: Double? = nil,
        minimumSharpShutterSeconds: Double? = nil,
        tiltFromLevelDegrees: Double? = nil,
        requestedFPS: Int? = nil,
        configuredFPS: Double? = nil,
        expectedResolution: [Int]? = nil,
        configuredResolution: [Int]? = nil,
        setupTipReasons: [String] = []
    ) {
        self.exposureTargetOffsetEV = exposureTargetOffsetEV
        self.isExposureLocked = isExposureLocked
        self.shutterSeconds = shutterSeconds
        self.minimumSharpShutterSeconds = minimumSharpShutterSeconds
        self.tiltFromLevelDegrees = tiltFromLevelDegrees
        self.requestedFPS = requestedFPS
        self.configuredFPS = configuredFPS
        self.expectedResolution = expectedResolution
        self.configuredResolution = configuredResolution
        self.setupTipReasons = setupTipReasons
    }
}

public struct LiveGuidanceState: Equatable, Sendable {
    public var checks: [LiveGuidanceCheck]
    /// Always-shown v0 manual reminder -- corner framing has no live signal.
    public var manualFramingTip: String
    /// Human-readable versions of `setupTipReasons`.
    public var setupTips: [String]
    /// Grade computed ONLY from checks that had a real signal. `.warn` (not
    /// `.good`) when nothing has been measured yet, so the screen never opens
    /// claiming a good setup with zero evidence.
    public var grade: CaptureQuality.Grade

    public init(checks: [LiveGuidanceCheck], manualFramingTip: String, setupTips: [String], grade: CaptureQuality.Grade) {
        self.checks = checks
        self.manualFramingTip = manualFramingTip
        self.setupTips = setupTips
        self.grade = grade
    }
}

public enum LiveGuidanceEvaluator {
    public static let manualFramingTipText =
        "Manual check (v0, not automatic yet): keep all 4 court corners inside the frame."

    private static let setupTipText: [String: String] = [
        "arkit_seed_missing": "No ARKit camera pose yet -- court geometry is FOV-estimated, not calibrated.",
        "court_plane_missing": "No ARKit court plane yet -- world-grounding will run offline instead of live.",
        "intrinsics_estimated_from_fov": "Camera intrinsics are estimated from the lens FOV, not device-calibrated.",
        "imported_no_live_sensors": "Imported footage has no live exposure, gravity, ARKit, or court-lock sensor stream.",
        "fps_from_asset_nominal_rate": "Frame rate is probed from the asset's nominal track rate.",
        "resolution_below_1080p_floor": "Resolution is below the 1080p capture-quality floor.",
        "fps_below_60_floor": "Frame rate is below the 60 fps capture-quality floor.",
    ]

    public static func evaluate(
        _ sample: LiveGuidanceSample,
        thresholds: GuidanceRuleThresholds = GuidanceRuleThresholds()
    ) -> LiveGuidanceState {
        let checks = [
            exposureCheck(sample, thresholds),
            levelCheck(sample, thresholds),
            blurRiskCheck(sample),
            frameRateCheck(sample),
            resolutionCheck(sample),
        ]

        let measured = checks.filter { $0.status != .unavailable }
        let grade: CaptureQuality.Grade
        if measured.isEmpty {
            grade = .warn
        } else if measured.allSatisfy({ $0.status == .good }) {
            grade = .good
        } else if measured.filter({ $0.status == .warn }).count >= 3 {
            grade = .poor
        } else {
            grade = .warn
        }

        return LiveGuidanceState(
            checks: checks,
            manualFramingTip: manualFramingTipText,
            setupTips: sample.setupTipReasons.map { humanReadableSetupTip(for: $0) },
            grade: grade
        )
    }

    public static func humanReadableSetupTip(for reason: String) -> String {
        setupTipText[reason] ?? reason
    }

    private static func exposureCheck(
        _ sample: LiveGuidanceSample,
        _ thresholds: GuidanceRuleThresholds
    ) -> LiveGuidanceCheck {
        guard let offset = sample.exposureTargetOffsetEV else {
            return LiveGuidanceCheck(
                id: "exposure",
                title: "Exposure",
                detail: "Exposure not read yet.",
                status: .unavailable
            )
        }
        let locked = sample.isExposureLocked ?? false
        if abs(offset) > thresholds.maximumExposureTargetOffset {
            let direction = offset > 0 ? "too bright" : "too dark"
            return LiveGuidanceCheck(
                id: "exposure",
                title: "Exposure",
                detail: "Scene reads \(direction) (\(formatted(offset)) EV from target). Move to more even light.",
                status: .warn
            )
        }
        return LiveGuidanceCheck(
            id: "exposure",
            title: "Exposure",
            detail: locked
                ? "Exposure locked and within \(formatted(thresholds.maximumExposureTargetOffset)) EV of target."
                : "Exposure within \(formatted(thresholds.maximumExposureTargetOffset)) EV of target (not yet locked).",
            status: .good
        )
    }

    private static func levelCheck(
        _ sample: LiveGuidanceSample,
        _ thresholds: GuidanceRuleThresholds
    ) -> LiveGuidanceCheck {
        guard let tilt = sample.tiltFromLevelDegrees else {
            return LiveGuidanceCheck(
                id: "level",
                title: "Level",
                detail: "Phone tilt not read yet.",
                status: .unavailable
            )
        }
        if abs(tilt) > thresholds.maximumLevelDegrees {
            return LiveGuidanceCheck(
                id: "level",
                title: "Level",
                detail: "Phone is tilted \(formatted(tilt))° from level. Aim for under \(formatted(thresholds.maximumLevelDegrees))°.",
                status: .warn
            )
        }
        return LiveGuidanceCheck(
            id: "level",
            title: "Level",
            detail: "Phone is level (\(formatted(tilt))° from level).",
            status: .good
        )
    }

    private static func blurRiskCheck(_ sample: LiveGuidanceSample) -> LiveGuidanceCheck {
        guard let shutterSeconds = sample.shutterSeconds,
              let minimumSharpShutterSeconds = sample.minimumSharpShutterSeconds else {
            return LiveGuidanceCheck(
                id: "blur",
                title: "Motion blur risk",
                detail: "Shutter speed not read yet.",
                status: .unavailable
            )
        }
        if shutterSeconds > minimumSharpShutterSeconds {
            return LiveGuidanceCheck(
                id: "blur",
                title: "Motion blur risk",
                detail: "Shutter is slower than the fast-swing-safe floor -- fast shots may blur in low light.",
                status: .warn
            )
        }
        return LiveGuidanceCheck(
            id: "blur",
            title: "Motion blur risk",
            detail: "Shutter speed is fast enough for fast swings.",
            status: .good
        )
    }

    private static func frameRateCheck(_ sample: LiveGuidanceSample) -> LiveGuidanceCheck {
        guard let requestedFPS = sample.requestedFPS, let configuredFPS = sample.configuredFPS else {
            return LiveGuidanceCheck(
                id: "frame_rate",
                title: "Frame rate",
                detail: "Frame rate not confirmed yet.",
                status: .unavailable
            )
        }
        if abs(configuredFPS - Double(requestedFPS)) > 0.5 {
            return LiveGuidanceCheck(
                id: "frame_rate",
                title: "Frame rate",
                detail: "Camera configured \(formatted(configuredFPS)) fps, requested \(requestedFPS) fps.",
                status: .warn
            )
        }
        return LiveGuidanceCheck(
            id: "frame_rate",
            title: "Frame rate",
            detail: "Camera confirmed at \(requestedFPS) fps.",
            status: .good
        )
    }

    private static func resolutionCheck(_ sample: LiveGuidanceSample) -> LiveGuidanceCheck {
        guard let expected = sample.expectedResolution, let configured = sample.configuredResolution else {
            return LiveGuidanceCheck(
                id: "resolution",
                title: "Resolution",
                detail: "Resolution not confirmed yet.",
                status: .unavailable
            )
        }
        if expected.count < 2 || configured.count < 2 || configured[0] < expected[0] || configured[1] < expected[1] {
            return LiveGuidanceCheck(
                id: "resolution",
                title: "Resolution",
                detail: "Camera configured \(describe(configured)), below the requested \(describe(expected)).",
                status: .warn
            )
        }
        return LiveGuidanceCheck(
            id: "resolution",
            title: "Resolution",
            detail: "Camera confirmed at \(describe(configured)).",
            status: .good
        )
    }

    private static func describe(_ resolution: [Int]) -> String {
        guard resolution.count >= 2 else {
            return "unknown"
        }
        return "\(resolution[0])x\(resolution[1])"
    }

    private static func formatted(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

/// v0 tilt estimator: collapses a raw CoreMotion gravity vector into a
/// single "degrees from level" magnitude for the given capture orientation,
/// rather than decomposing separate roll/pitch axes like the offline
/// `GuidanceSample.rollDegrees/pitchDegrees`. This keeps the live check
/// honest (it IS the real gravity vector, not a placeholder) while avoiding
/// a possibly-wrong sign convention on two decomposed axes; per-axis
/// decomposition is a documented follow-up once verified against a real
/// device in hand.
public enum LiveTiltEstimator {
    /// `gravity` is `[x, y, z]` in G units as read from
    /// `CMDeviceMotion.gravity` (see `CaptureMotionSampler.latestGravity`).
    /// `expectedLevelAxis` is the unit gravity vector expected when the
    /// phone is held level in the given capture orientation.
    public static func tiltDegrees(gravity: [Double], expectedLevelAxis: [Double]) -> Double? {
        guard gravity.count == 3, expectedLevelAxis.count == 3 else {
            return nil
        }
        let gravityMagnitude = magnitude(gravity)
        let axisMagnitude = magnitude(expectedLevelAxis)
        guard gravityMagnitude > 0.001, axisMagnitude > 0.001 else {
            return nil
        }
        let dot = zip(gravity, expectedLevelAxis).reduce(0) { $0 + $1.0 * $1.1 }
        let cosine = max(-1.0, min(1.0, dot / (gravityMagnitude * axisMagnitude)))
        return acos(cosine) * 180.0 / Double.pi
    }

    /// Expected level gravity axis per `CaptureDeviceOrientation`, matching
    /// the device-native frame `CaptureMotionSampler` reads gravity in
    /// (portrait-native axes, independent of interface rotation).
    public static func expectedLevelAxis(for orientation: CaptureDeviceOrientation) -> [Double] {
        switch orientation {
        case .portrait:
            return [0.0, -1.0, 0.0]
        case .portraitUpsideDown:
            return [0.0, 1.0, 0.0]
        case .landscapeLeft:
            return [1.0, 0.0, 0.0]
        case .landscapeRight:
            return [-1.0, 0.0, 0.0]
        }
    }

    private static func magnitude(_ vector: [Double]) -> Double {
        sqrt(vector.reduce(0) { $0 + $1 * $1 })
    }
}
