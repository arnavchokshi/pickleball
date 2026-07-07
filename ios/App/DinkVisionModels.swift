import Foundation
import PickleballCapture
import PickleballCore

enum DinkVisionSplashPhase: Equatable {
    case zoomedClosed
    case lidsOpening
    case done
}

enum DinkVisionSplashAnimationPlan: Equatable {
    case lidReveal
    case crossfade
}

enum DinkVisionSplashTiming {
    static let closedHoldNanoseconds: UInt64 = 280_000_000
    static let lidOpeningNanoseconds: UInt64 = 620_000_000
    static let totalDurationNanoseconds = closedHoldNanoseconds + lidOpeningNanoseconds
}

struct DinkVisionSplashStateMachine: Equatable {
    private(set) var phase: DinkVisionSplashPhase = .zoomedClosed
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
        case .zoomedClosed:
            phase = .lidsOpening
        case .lidsOpening, .done:
            phase = .done
        }
        return phase
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
