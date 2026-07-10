import Foundation
import CoreGraphics
import PickleballCapture
import PickleballCore

enum DinkVisionSplashPhase: Equatable {
    case settle
    case blink
    case openUp
    case done
}

enum DinkVisionSplashAnimationPlan: Equatable {
    case lidReveal
    case crossfade
}

enum DinkVisionSplashTiming {
    static let settleNanoseconds: UInt64 = 150_000_000
    static let blinkCloseNanoseconds: UInt64 = 260_000_000
    static let blinkHoldNanoseconds: UInt64 = 120_000_000
    static let blinkOpenNanoseconds: UInt64 = 340_000_000
    static let openUpNanoseconds: UInt64 = 380_000_000
    static let reducedMotionCrossfadeNanoseconds: UInt64 = 180_000_000

    static let totalDurationNanoseconds = settleNanoseconds
        + blinkCloseNanoseconds
        + blinkHoldNanoseconds
        + blinkOpenNanoseconds
        + openUpNanoseconds
}

struct DinkVisionSplashStateMachine: Equatable {
    private(set) var phase: DinkVisionSplashPhase = .settle
    let reducedMotion: Bool

    var animationPlan: DinkVisionSplashAnimationPlan {
        reducedMotion ? .crossfade : .lidReveal
    }

    mutating func advance() -> DinkVisionSplashPhase {
        if reducedMotion {
            phase = .done
            return phase
        }

        switch phase {
        case .settle:
            phase = .blink
        case .blink:
            phase = .openUp
        case .openUp, .done:
            phase = .done
        }
        return phase
    }
}

nonisolated struct DinkVisionSplashLidCurve: Equatable {
    let left: CGPoint
    let right: CGPoint
    let control: CGPoint

    var edgeY: CGFloat {
        (left.y + right.y) / 2
    }
}

nonisolated struct DinkVisionSplashLidCover: Equatable {
    let outerCurve: DinkVisionSplashLidCurve
    let innerCurve: DinkVisionSplashLidCurve
    let coverRect: CGRect
}

nonisolated enum DinkVisionSplashLidGeometry {
    static let markPixelWidth: CGFloat = 322
    static let markPixelHeight: CGFloat = 579
    static let markWidthToViewportRatio: CGFloat = 0.34
    static let eyeCenterXRatio: CGFloat = 0.50
    static let eyeCenterYRatio: CGFloat = 0.361
    static let apertureHalfHeightToMarkHeight: CGFloat = 0.145
    static let lidSpanToMarkWidth: CGFloat = 0.86
    static let strokeWidthToMarkWidth: CGFloat = 0.075

    static func markFrame(in viewport: CGSize) -> CGRect {
        let width = viewport.width * markWidthToViewportRatio
        let height = width * markPixelHeight / markPixelWidth
        return CGRect(
            x: (viewport.width - width) / 2,
            y: (viewport.height - height) / 2,
            width: width,
            height: height
        )
    }

    static func eyeCenter(in markFrame: CGRect) -> CGPoint {
        CGPoint(
            x: markFrame.minX + markFrame.width * eyeCenterXRatio,
            y: markFrame.minY + markFrame.height * eyeCenterYRatio
        )
    }

    static func apertureHalfHeight(in markFrame: CGRect) -> CGFloat {
        markFrame.height * apertureHalfHeightToMarkHeight
    }

    static func lidSpan(in markFrame: CGRect) -> CGFloat {
        markFrame.width * lidSpanToMarkWidth
    }

    static func strokeWidth(in markFrame: CGRect) -> CGFloat {
        markFrame.width * strokeWidthToMarkWidth
    }

    static func curve(isUpper: Bool, closure: CGFloat, markFrame: CGRect) -> DinkVisionSplashLidCurve {
        let clampedClosure = min(1, max(0, closure))
        let openAmount = 1 - clampedClosure
        let center = eyeCenter(in: markFrame)
        let halfSpan = lidSpan(in: markFrame) / 2
        let halfHeight = apertureHalfHeight(in: markFrame)
        let sign: CGFloat = isUpper ? -1 : 1
        let edgeY = center.y + sign * halfHeight * openAmount
        let controlY = center.y + sign * halfHeight * 1.52 * openAmount

        return DinkVisionSplashLidCurve(
            left: CGPoint(x: center.x - halfSpan, y: edgeY),
            right: CGPoint(x: center.x + halfSpan, y: edgeY),
            control: CGPoint(x: center.x, y: controlY)
        )
    }

    static func lidCover(isUpper: Bool, closure: CGFloat, markFrame: CGRect) -> DinkVisionSplashLidCover {
        let outerCurve = curve(isUpper: isUpper, closure: 0, markFrame: markFrame)
        let innerCurve = curve(isUpper: isUpper, closure: closure, markFrame: markFrame)
        let minY = min(outerCurve.edgeY, innerCurve.edgeY)
        let maxY = max(outerCurve.edgeY, innerCurve.edgeY)
        let minX = min(outerCurve.left.x, innerCurve.left.x)
        let maxX = max(outerCurve.right.x, innerCurve.right.x)
        return DinkVisionSplashLidCover(
            outerCurve: outerCurve,
            innerCurve: innerCurve,
            coverRect: CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        )
    }
}

enum DinkVisionReplayOpenTransition {
    static let durationNanoseconds: UInt64 = 550_000_000
    static let reducedMotionCrossfadeNanoseconds: UInt64 = 180_000_000

    enum Plan: Equatable {
        case diagonalSwoosh
        case crossfade
    }

    static func durationNanoseconds(reducedMotion: Bool) -> UInt64 {
        reducedMotion ? reducedMotionCrossfadeNanoseconds : durationNanoseconds
    }

    static func plan(reducedMotion: Bool) -> Plan {
        reducedMotion ? .crossfade : .diagonalSwoosh
    }
}

enum DinkVisionAccentSite: String, CaseIterable {
    case replaysEmptyState
    case statsSampleWatermark
    case profileCompletedStep
    case permissionPrimer
    case coachPlaceholder
}

enum DinkVisionTabKind: String, CaseIterable, Identifiable, Equatable {
    case replays
    case stats
    case record
    case coach
    case profile

    var id: String { rawValue }

    var title: String {
        switch self {
        case .replays: return "Replays"
        case .stats: return "Stats"
        case .record: return "Record"
        case .coach: return "Coach"
        case .profile: return "Profile"
        }
    }

    var symbolName: String {
        switch self {
        case .replays: return "play.rectangle"
        case .stats: return "chart.bar"
        case .record: return "record.circle"
        case .coach: return "sparkles"
        case .profile: return "person.crop.circle"
        }
    }
}

struct DinkVisionTabLayoutModel: Equatable {
    var tabs: [DinkVisionTabKind]
    var recordButtonDiameter: CGFloat
    var recordButtonCenterAboveBarTop: CGFloat
    var minimumHitTarget: CGFloat

    static let brandV4 = DinkVisionTabLayoutModel(
        tabs: [.replays, .stats, .record, .coach, .profile],
        recordButtonDiameter: 72,
        recordButtonCenterAboveBarTop: 26,
        minimumHitTarget: 44
    )

    var centerTab: DinkVisionTabKind? {
        guard !tabs.isEmpty else { return nil }
        return tabs[tabs.count / 2]
    }

    func totalOverlayHeight(tabBarHeight: CGFloat) -> CGFloat {
        tabBarHeight + recordButtonDiameter / 2 + recordButtonCenterAboveBarTop
    }

    func recordButtonCenterY(tabBarHeight _: CGFloat) -> CGFloat {
        recordButtonDiameter / 2
    }

    func recordButtonFrame(tabBarHeight: CGFloat) -> CGRect {
        CGRect(
            x: -recordButtonDiameter / 2,
            y: recordButtonCenterY(tabBarHeight: tabBarHeight) - recordButtonDiameter / 2,
            width: recordButtonDiameter,
            height: recordButtonDiameter
        )
    }

    func railEdge(for containerSize: CGSize) -> DinkVisionTabRailEdge {
        containerSize.width > containerSize.height ? .leading : .bottom
    }

    func recordButtonPlacement(
        in containerSize: CGSize,
        railEdge: DinkVisionTabRailEdge,
        barThickness: CGFloat,
        buttonDiameter: CGFloat
    ) -> DinkVisionRecordButtonPlacement {
        let buttonRadius = buttonDiameter / 2

        switch railEdge {
        case .bottom:
            let center = CGPoint(
                x: containerSize.width / 2,
                y: containerSize.height - barThickness - recordButtonCenterAboveBarTop
            )
            let exposedDiameter = containerSize.height - barThickness - center.y + buttonRadius
            return DinkVisionRecordButtonPlacement(
                center: center,
                exposedFraction: exposedDiameter / buttonDiameter
            )
        case .leading:
            return DinkVisionRecordButtonPlacement(
                center: CGPoint(x: barThickness + buttonRadius, y: containerSize.height / 2),
                exposedFraction: 1
            )
        case .trailing:
            return DinkVisionRecordButtonPlacement(
                center: CGPoint(x: containerSize.width - barThickness - buttonRadius, y: containerSize.height / 2),
                exposedFraction: 1
            )
        }
    }

    func contentBottomPadding(tabBarHeight: CGFloat) -> CGFloat {
        totalOverlayHeight(tabBarHeight: tabBarHeight) + 24
    }
}

enum DinkVisionTabRailEdge: Equatable {
    case bottom
    case leading
    case trailing
}

struct DinkVisionRecordButtonPlacement: Equatable {
    var center: CGPoint
    var exposedFraction: CGFloat
}

enum DinkVisionRecordButtonControlState: Equatable {
    case idle
    case pressed
    case recording
}

enum DinkVisionRecordButtonCenterShape: Equatable {
    case circle
    case roundedSquareStop
}

enum DinkVisionRecordButtonRing: Equatable {
    case ink
    case trailRed
}

struct DinkVisionRecordButtonVisual: Equatable {
    var scale: CGFloat
    var breathingScaleRange: ClosedRange<CGFloat>
    var breathingDurationSeconds: Double
    var centerShape: DinkVisionRecordButtonCenterShape
    var ring: DinkVisionRecordButtonRing
    var holeCount: Int
    var holeDiameterRatio: CGFloat
    var innerShadowStrength: Double

    static let idle = DinkVisionRecordButtonVisual(
        scale: 1.0,
        breathingScaleRange: 1.0...1.02,
        breathingDurationSeconds: 3.0,
        centerShape: .circle,
        ring: .ink,
        holeCount: 8,
        holeDiameterRatio: 0.13,
        innerShadowStrength: 0.26
    )

    static let pressed = DinkVisionRecordButtonVisual(
        scale: 0.94,
        breathingScaleRange: 0.94...0.94,
        breathingDurationSeconds: 0,
        centerShape: .circle,
        ring: .ink,
        holeCount: 8,
        holeDiameterRatio: 0.13,
        innerShadowStrength: 0.46
    )

    static let recording = DinkVisionRecordButtonVisual(
        scale: 1.0,
        breathingScaleRange: 1.0...1.0,
        breathingDurationSeconds: 0,
        centerShape: .roundedSquareStop,
        ring: .trailRed,
        holeCount: 8,
        holeDiameterRatio: 0.13,
        innerShadowStrength: 0.32
    )
}

struct DinkVisionRecordButtonStateMachine: Equatable {
    private(set) var state: DinkVisionRecordButtonControlState = .idle

    var visual: DinkVisionRecordButtonVisual {
        switch state {
        case .idle: return .idle
        case .pressed: return .pressed
        case .recording: return .recording
        }
    }

    mutating func press() {
        guard state != .recording else { return }
        state = .pressed
    }

    mutating func release() {
        guard state == .pressed else { return }
        state = .idle
    }

    mutating func startRecording() {
        state = .recording
    }

    mutating func stopRecording() {
        state = .idle
    }
}

struct DinkVisionStrokeDrawOnParameters: Equatable {
    var durationSeconds: Double
    var initialTrimEnd: CGFloat
    var finalTrimEnd: CGFloat

    static let `default` = DinkVisionStrokeDrawOnParameters(
        durationSeconds: 0.35,
        initialTrimEnd: 0,
        finalTrimEnd: 1
    )

    static func resolved(reducedMotion: Bool) -> DinkVisionStrokeDrawOnParameters {
        reducedMotion ? DinkVisionStrokeDrawOnParameters(durationSeconds: 0, initialTrimEnd: 1, finalTrimEnd: 1) : .default
    }
}

struct DinkVisionScreenMotionParameters: Equatable {
    var slidePoints: CGFloat
    var rotationDegrees: Double

    static let `default` = DinkVisionScreenMotionParameters(slidePoints: 8, rotationDegrees: 1.2)

    static func resolved(reducedMotion: Bool) -> DinkVisionScreenMotionParameters {
        reducedMotion ? DinkVisionScreenMotionParameters(slidePoints: 0, rotationDegrees: 0) : .default
    }
}

struct DinkVisionCoachPlaceholderModel: Equatable {
    var title: String
    var roadmapID: String
    var isComingSoon: Bool
    var fakeFeatureBullets: [String]

    static let brandV4 = DinkVisionCoachPlaceholderModel(
        title: "Your pocket coach is training...",
        roadmapID: "P6",
        isComingSoon: true,
        fakeFeatureBullets: []
    )
}

enum DinkVisionRecordFlowPhase: Equatable {
    case idle
    case ready
    case recording(startedAt: Date)
    case saving
    case done(sessionID: String)
    case permissionDenied(String)
    case blocked(String)
}

enum DinkVisionSetupPassStatus: Equatable {
    case idle
    case aligning
    case aligned
    case unavailable(String)
}

enum DinkVisionPolicyChipStatus: String, Equatable {
    case pass
    case warning
}

struct DinkVisionPolicyChip: Identifiable, Equatable {
    var id: String
    var title: String
    var status: DinkVisionPolicyChipStatus
    var hint: String
}

enum DinkVisionPolicyChipMapper {
    static func chips(for report: CapturePolicyEnforcementReport?) -> [DinkVisionPolicyChip] {
        let achieved = report?.achieved
        let violations = Set(report?.violations ?? [])

        let eisPass = achieved?.electronicStabilizationEnabled == false
            && !violations.contains("electronic_stabilization_enabled")
        let locksPass = achieved?.exposureLocked == true
            && achieved?.focusLocked == true
            && achieved?.whiteBalanceLocked == true
        let landscapePass = achieved?.orientation == .landscape
            && !violations.contains("orientation_not_landscape")

        return [
            DinkVisionPolicyChip(
                id: "eis",
                title: "EIS off",
                status: eisPass ? .pass : .warning,
                hint: eisPass ? "Enhanced Stabilization is off." : "Turn Enhanced Stabilization off before recording."
            ),
            DinkVisionPolicyChip(
                id: "camera_locks",
                title: "AE/AF/WB",
                status: locksPass ? .pass : .warning,
                hint: locksPass ? "Exposure, focus, and white balance are locked." : "Long-press to lock exposure, focus, and white balance."
            ),
            DinkVisionPolicyChip(
                id: "landscape",
                title: "Landscape",
                status: landscapePass ? .pass : .warning,
                hint: landscapePass ? "Landscape capture is active." : "Rotate to landscape before recording."
            ),
        ]
    }
}

struct DinkVisionReplayRow: Identifiable, Equatable {
    var id: String
    var title: String
    var subtitle: String
    var durationText: String
    var dateText: String
    var trustBadgeText: String
    var trusted3DText: String
    var ballTrustText: String
    var item: CaptureLibraryItem
}

struct DinkVisionReplayListDataSource {
    var packageRootURL: URL
    var loadPackages: (URL) throws -> [CaptureLibraryItem]

    init(
        packageRootURL: URL = CameraCaptureController.defaultPackageRootURL(),
        loadPackages: @escaping (URL) throws -> [CaptureLibraryItem] = { url in
            try CaptureLibrary.listPackages(packageRootURL: url)
        }
    ) {
        self.packageRootURL = packageRootURL
        self.loadPackages = loadPackages
    }

    func loadRows() throws -> [DinkVisionReplayRow] {
        try loadPackages(packageRootURL).map { row(for: $0) }
    }

    private func row(for item: CaptureLibraryItem) -> DinkVisionReplayRow {
        let sidecarTexts = Self.sidecarTrustTexts(
            sidecarURL: packageRootURL.appendingPathComponent(item.sidecarRelativePath)
        )
        return DinkVisionReplayRow(
            id: item.sessionID,
            title: item.isImported ? "Imported rally" : "Tuesday open play",
            subtitle: "\(Self.dateText(for: item.recordedAt)) · \(item.fps) fps · \(item.resolutionText)",
            durationText: Self.durationText(for: item.durationSeconds),
            dateText: Self.dateText(for: item.recordedAt),
            trustBadgeText: item.captureQualityGrade == .good ? "Trusted sidecar" : "Needs review",
            trusted3DText: sidecarTexts.trusted3D,
            ballTrustText: sidecarTexts.ball,
            item: item
        )
    }

    private static func durationText(for seconds: Double?) -> String {
        let value = max(0, Int((seconds ?? 0).rounded()))
        return String(format: "%02d:%02d", value / 60, value % 60)
    }

    private static func dateText(for date: Date?) -> String {
        guard let date else {
            return "Date unknown"
        }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    private static func sidecarTrustTexts(sidecarURL: URL) -> (trusted3D: String, ball: String) {
        guard let data = try? Data(contentsOf: sidecarURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return ("Trusted 3D —", "Ball —")
        }
        let trusted3D = boolValue(object["trusted_3d"])
            ?? boolValue(object["server_replay_trusted_3d"])
            ?? nestedBoolValue(object, path: ["replay", "trusted_3d"])
        let ballPercent = percentValue(object["ball_percent"])
            ?? percentValue(object["ball_confidence_percent"])
            ?? nestedPercentValue(object, path: ["ball", "percent"])
            ?? nestedPercentValue(object, path: ["replay", "ball_percent"])

        return (
            trusted3D.map { "Trusted 3D \($0 ? "yes" : "no")" } ?? "Trusted 3D —",
            ballPercent.map { "Ball \($0)%" } ?? "Ball —"
        )
    }

    private static func boolValue(_ value: Any?) -> Bool? {
        if let value = value as? Bool {
            return value
        }
        if let value = value as? String {
            switch value.lowercased() {
            case "true", "yes", "trusted":
                return true
            case "false", "no", "sample", "preview":
                return false
            default:
                return nil
            }
        }
        return nil
    }

    private static func percentValue(_ value: Any?) -> Int? {
        let raw: Double?
        if let value = value as? Double {
            raw = value
        } else if let value = value as? Int {
            raw = Double(value)
        } else if let value = value as? String {
            raw = Double(value.trimmingCharacters(in: CharacterSet(charactersIn: "%")))
        } else {
            raw = nil
        }
        guard let raw else {
            return nil
        }
        let percent = raw <= 1 ? raw * 100 : raw
        return min(100, max(0, Int(percent.rounded())))
    }

    private static func nestedBoolValue(_ object: [String: Any], path: [String]) -> Bool? {
        boolValue(nestedValue(object, path: path))
    }

    private static func nestedPercentValue(_ object: [String: Any], path: [String]) -> Int? {
        percentValue(nestedValue(object, path: path))
    }

    private static func nestedValue(_ object: [String: Any], path: [String]) -> Any? {
        var current: Any? = object
        for key in path {
            guard let dictionary = current as? [String: Any] else {
                return nil
            }
            current = dictionary[key]
        }
        return current
    }
}

private extension CaptureLibraryItem {
    var resolutionText: String {
        resolution.count == 2 ? "\(resolution[0])x\(resolution[1])" : "resolution unknown"
    }
}

extension ProfileCaptureStepKind {
    var dinkVisionTitle: String {
        switch self {
        case .emptyCourtClip:
            return "Empty-court clip"
        case .calibrationGridSweep:
            return "Calibration sweep"
        case .paddleOrbit:
            return "Paddle orbit"
        case .playerHeightEntry:
            return "Your height"
        case .ballPick:
            return "Ball pick"
        }
    }

    var dinkVisionDetail: String {
        switch self {
        case .emptyCourtClip:
            return "10s of your court, no players"
        case .calibrationGridSweep:
            return "Slow pan with the printed board"
        case .paddleOrbit:
            return "Circle your paddle for 15s"
        case .playerHeightEntry:
            return "One number scales everything"
        case .ballPick:
            return "Which ball do you play?"
        }
    }
}
