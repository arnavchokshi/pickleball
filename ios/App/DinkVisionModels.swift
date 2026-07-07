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

struct DinkVisionSplashLidCurve: Equatable {
    let left: CGPoint
    let right: CGPoint
    let control: CGPoint

    var edgeY: CGFloat {
        (left.y + right.y) / 2
    }
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
}

enum DinkVisionReplayOpenTransition {
    static let durationNanoseconds: UInt64 = 420_000_000

    static func durationNanoseconds(reducedMotion: Bool) -> UInt64 {
        reducedMotion ? 0 : durationNanoseconds
    }
}

enum DinkVisionAccentSite: String, CaseIterable {
    case replaysEmptyState
    case statsSampleWatermark
    case profileCompletedStep
    case permissionPrimer
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
        try loadPackages(packageRootURL).map(Self.row(for:))
    }

    private static func row(for item: CaptureLibraryItem) -> DinkVisionReplayRow {
        DinkVisionReplayRow(
            id: item.sessionID,
            title: item.isImported ? "Imported rally" : "Tuesday open play",
            subtitle: "\(dateText(for: item.recordedAt)) · \(item.fps) fps · \(item.resolutionText)",
            durationText: durationText(for: item.durationSeconds),
            dateText: dateText(for: item.recordedAt),
            trustBadgeText: item.captureQualityGrade == .good ? "Trusted sidecar" : "Needs review",
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
