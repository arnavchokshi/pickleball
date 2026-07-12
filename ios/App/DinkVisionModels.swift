import Foundation
import CoreFoundation
import CoreGraphics
import PickleballCapture
import PickleballCore
import PickleballUpload

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
    case statsEmptyState
    case profileCompletedStep
    case permissionPrimer
    case coachEmptyState
}

enum DinkVisionTabKind: String, CaseIterable, Identifiable, Equatable {
    case replays
    case stats
    case record
    case coach
    case profile

    var id: String { rawValue }

    static let coldLaunchDefault: DinkVisionTabKind = .record

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

struct DinkVisionRecordingPresentation: Equatable {
    var controlState: DinkVisionRecordButtonControlState
    var elapsedText: String?
    var accessibilityLabel: String
    var accessibilityValue: String

    init(isRecording: Bool, startedAt: Date?, now: Date) {
        guard isRecording else {
            controlState = .idle
            elapsedText = nil
            accessibilityLabel = "Start recording"
            accessibilityValue = "Not recording"
            return
        }

        let elapsedSeconds = max(0, Int(now.timeIntervalSince(startedAt ?? now)))
        let formatted = String(format: "%d:%02d", elapsedSeconds / 60, elapsedSeconds % 60)
        controlState = .recording
        elapsedText = formatted
        accessibilityLabel = "Stop recording"
        accessibilityValue = "Recording, elapsed time \(formatted)"
    }
}

enum DinkVisionReplayProductState: Equatable {
    case sample
    case local
    case queued
    case uploading(percent: Int)
    case processing
    case partial
    case ready
    case failed
}

struct DinkVisionReplayStatusPresentation: Equatable {
    var state: DinkVisionReplayProductState
    var title: String
    var detail: String?
    var missingCapabilities: [String]

    init(uploadState: CaptureUploadState?, isSample: Bool = false) {
        if isSample {
            state = .sample
            title = "Sample replay"
            detail = "Bundled fixture — not one of your sessions"
            missingCapabilities = []
            return
        }
        guard let uploadState else {
            state = .local
            title = "Local only"
            detail = "Not uploaded"
            missingCapabilities = []
            return
        }

        missingCapabilities = uploadState.missingCapabilities.map { capability in
            let name = capability.capability.replacingOccurrences(of: "_", with: " ")
            guard !capability.reason.isEmpty else { return name }
            return "\(name): \(capability.reason)"
        }

        switch uploadState.state {
        case .queued:
            state = .queued
            title = "Queued"
            detail = "Waiting to upload"
        case .uploading:
            let percent = min(100, max(0, Int((uploadState.fractionCompleted * 100).rounded())))
            state = .uploading(percent: percent)
            title = "Uploading \(percent)%"
            detail = "Video and capture sidecar"
        case .failed:
            state = .failed
            title = "Failed"
            detail = uploadState.lastError.map { "Upload or processing failed: \($0)" }
                ?? "Upload or processing failed"
        case .uploaded:
            switch uploadState.serverStatus {
            case RenderGatewayJobStatus.complete.rawValue where uploadState.manifestUrl != nil:
                state = .ready
                title = "Replay ready"
                detail = "Server output is available"
            case RenderGatewayJobStatus.partial.rawValue:
                state = .partial
                title = "Partial replay"
                detail = missingCapabilities.isEmpty
                    ? "Server marked this replay partial; missing capabilities were not listed"
                    : "Missing: \(missingCapabilities.joined(separator: "; "))"
            default:
                state = .processing
                title = "Processing"
                detail = uploadState.serverStatus.map { "Server status: \($0)" }
                    ?? "Waiting for server status"
            }
        }
    }
}

enum DinkVisionFactAuthority: String, Equatable {
    case verified
    case preview
    case lowConfidence = "low_confidence"
    case tooCloseToCall = "too_close_to_call"

    var displayName: String {
        switch self {
        case .verified: return "Verified"
        case .preview: return "Preview"
        case .lowConfidence: return "Low confidence"
        case .tooCloseToCall: return "Too close to call"
        }
    }
}

enum DinkVisionFactProvenance: String, Equatable {
    case measured
    case modelEstimated = "model_estimated"
    case physicsPredicted = "physics_predicted"
}

struct DinkVisionProductFact: Identifiable, Equatable {
    var id: String
    var sessionID: String
    var metric: String
    var label: String
    var valueText: String
    var authority: DinkVisionFactAuthority
    var provenance: DinkVisionFactProvenance
    var evidenceLocator: String
    var sourceArtifact: String
    var playerOrEntity: String?
}

enum DinkVisionFactsDocumentDecoder {
    static func decode(_ data: Data, sessionID: String) -> [DinkVisionProductFact] {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rawFacts = factsArray(in: root) else {
            return []
        }
        return rawFacts.compactMap { fact(in: $0, sessionID: sessionID) }
    }

    private static func factsArray(in root: [String: Any]) -> [[String: Any]]? {
        if let facts = root["audited_facts"] as? [[String: Any]] {
            return facts
        }
        if let container = root["coaching_card_facts"] as? [String: Any],
           let facts = container["audited_facts"] as? [[String: Any]] {
            return facts
        }
        return nil
    }

    private static func fact(in raw: [String: Any], sessionID: String) -> DinkVisionProductFact? {
        guard let id = nonemptyString(raw["fact_id"] ?? raw["id"]),
              let metric = nonemptyString(raw["metric"]),
              let authorityValue = trustString(raw["trust"], key: "authority_band")
                ?? nonemptyString(raw["authority"]),
              let authority = DinkVisionFactAuthority(rawValue: authorityValue),
              let provenanceValue = trustString(raw["trust"], key: "provenance_band")
                ?? provenanceString(raw["provenance"] ?? raw["evidence_provenance"]),
              let provenance = DinkVisionFactProvenance(rawValue: provenanceValue),
              let evidenceLocator = locatorString(raw["evidence_locator"]),
              let sourceArtifact = sourceArtifactsString(raw["source_artifacts"] ?? raw["source_artifact"]),
              let valueText = valueText(raw["value"], unit: nonemptyString(raw["unit"])) else {
            return nil
        }

        let entity = entityString(raw["entity"])
            ?? nonemptyString(raw["player_id"] ?? raw["entity_id"])
        return DinkVisionProductFact(
            id: "\(sessionID):\(id)",
            sessionID: sessionID,
            metric: metric,
            label: metric.replacingOccurrences(of: "_", with: " ").capitalized,
            valueText: valueText,
            authority: authority,
            provenance: provenance,
            evidenceLocator: evidenceLocator,
            sourceArtifact: sourceArtifact,
            playerOrEntity: entity
        )
    }

    private static func valueText(_ value: Any?, unit: String?) -> String? {
        if let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() {
            let double = number.doubleValue
            guard double.isFinite else { return nil }
            let numberText = double.rounded() == double
                ? String(Int(double))
                : String(format: "%.2f", double).replacingOccurrences(of: #"\.?0+$"#, with: "", options: .regularExpression)
            return [numberText, unit].compactMap { $0 }.joined(separator: " ")
        }
        if let zoneValue = value as? [String: Any],
           let zone = nonemptyString(zoneValue["zone"]),
           let fraction = zoneValue["fraction"] as? NSNumber {
            let percent = min(100, max(0, Int((fraction.doubleValue * 100).rounded())))
            return "\(zone.replacingOccurrences(of: "_", with: " ").capitalized) \(percent)%"
        }
        return nil
    }

    private static func provenanceString(_ value: Any?) -> String? {
        if let string = nonemptyString(value) { return string }
        if let object = value as? [String: Any] {
            return nonemptyString(object["kind"] ?? object["evidence"] ?? object["provenance"])
        }
        return nil
    }

    private static func trustString(_ value: Any?, key: String) -> String? {
        guard let object = value as? [String: Any] else { return nil }
        return nonemptyString(object[key])
    }

    private static func entityString(_ value: Any?) -> String? {
        guard let object = value as? [String: Any] else { return nil }
        return nonemptyString(object["id"])
    }

    private static func locatorString(_ value: Any?) -> String? {
        if let string = nonemptyString(value) { return string }
        if let object = value as? [String: Any] {
            return nonemptyString(object["uri"] ?? object["path"] ?? object["url"] ?? object["artifact_path"])
        }
        return nil
    }

    private static func sourceArtifactsString(_ value: Any?) -> String? {
        let objects: [[String: Any]]
        if let array = value as? [[String: Any]] {
            objects = array
        } else if let object = value as? [String: Any] {
            objects = [object]
        } else {
            return nil
        }
        guard !objects.isEmpty else { return nil }
        let references = objects.compactMap { object -> String? in
            guard let path = nonemptyString(object["path"] ?? object["artifact_path"]),
                  let sha256 = nonemptyString(object["sha256"]),
                  sha256.count == 64,
                  sha256.allSatisfy({ $0.isHexDigit }) else { return nil }
            return "\(path)#sha256=\(sha256.lowercased())"
        }
        guard references.count == objects.count else { return nil }
        return references.joined(separator: ", ")
    }

    private static func nonemptyString(_ value: Any?) -> String? {
        guard let string = value as? String else { return nil }
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

struct DinkVisionFactsLibraryDataSource {
    var replayDataSource: DinkVisionReplayListDataSource
    var readData: (URL) -> Data?

    init(
        replayDataSource: DinkVisionReplayListDataSource,
        readData: @escaping (URL) -> Data? = { try? Data(contentsOf: $0) }
    ) {
        self.replayDataSource = replayDataSource
        self.readData = readData
    }

    func loadFacts() -> [DinkVisionProductFact] {
        guard let rows = try? replayDataSource.loadRows() else { return [] }
        return rows.flatMap { row -> [DinkVisionProductFact] in
            guard case .capture = row.source else { return [] }
            let packageURL = replayDataSource.packageRootURL
                .appendingPathComponent(row.item.clipRelativePath)
                .deletingLastPathComponent()
            let candidates = [
                packageURL.appendingPathComponent("coaching_facts.json"),
                packageURL.appendingPathComponent("rally_metrics.json"),
                packageURL.appendingPathComponent("artifacts/coaching_facts.json"),
            ]
            guard let data = candidates.lazy.compactMap(readData).first else { return [] }
            return DinkVisionFactsDocumentDecoder.decode(data, sessionID: row.id)
        }
    }
}

struct DinkVisionProfileSettingsModel: Equatable {
    var accountTitle: String
    var uploadTitle: String
    var nonOwnerRetentionTitle: String
    var nonOwnerRetentionDetail: String

    static func current(isSignedIn: Bool, autoUploadAfterRecording: Bool) -> Self {
        Self(
            accountTitle: isSignedIn ? "Signed in" : "Local mode",
            uploadTitle: autoUploadAfterRecording ? "Auto-upload on" : "Auto-upload off",
            nonOwnerRetentionTitle: "Non-owner data: session only",
            nonOwnerRetentionDetail: "Non-owner biometric data is not retained beyond this session unless the owner explicitly opts in."
        )
    }
}

struct DinkVisionAccessibilityPolicy: Equatable {
    var supportsDynamicType: Bool
    var suppliesVoiceOverLabels: Bool
    var keepsDataInsideSafeAreas: Bool
    var usesAdaptiveDarkAndHighContrastColors: Bool
    var hasReducedMotionFallbacks: Bool

    static let productUI = Self(
        supportsDynamicType: true,
        suppliesVoiceOverLabels: true,
        keepsDataInsideSafeAreas: true,
        usesAdaptiveDarkAndHighContrastColors: true,
        hasReducedMotionFallbacks: true
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
    var source: DinkVisionReplaySource
}

enum DinkVisionReplaySource: Equatable {
    case capture(String)
    case bundledSample
}

struct DinkVisionReplayManifestRoute: Equatable {
    var captureId: String
    var clipId: String
    var jobId: String
    var manifestURL: URL
    var status: RenderGatewayJobStatus
    var missingCapabilities: [RenderGatewayMissingCapability]
    var trustBands: [String: RenderGatewayTrustBand?]
}

enum DinkVisionReplayRoute: Equatable {
    case bundledSample
    case manifest(DinkVisionReplayManifestRoute)
    case notReady(CaptureReplayNotReady)
}

enum DinkVisionReplayRouter {
    static func route(row: DinkVisionReplayRow, uploadState: CaptureUploadState?) -> DinkVisionReplayRoute {
        switch row.source {
        case .bundledSample:
            return .bundledSample
        case .capture(let captureId):
            guard let uploadState else {
                return .notReady(.notUploaded)
            }
            switch uploadState.replayAvailability(expectedCaptureId: captureId) {
            case .ready(let ready):
                return .manifest(DinkVisionReplayManifestRoute(
                    captureId: ready.captureId,
                    clipId: ready.clipId,
                    jobId: ready.jobId,
                    manifestURL: ready.manifestURL,
                    status: ready.status,
                    missingCapabilities: ready.missingCapabilities,
                    trustBands: ready.trustBands
                ))
            case .notReady(let reason):
                return .notReady(reason)
            }
        }
    }
}

struct DinkVisionReplaySelection: Identifiable, Equatable {
    var row: DinkVisionReplayRow
    var route: DinkVisionReplayRoute
    var id: String { row.id }
}

struct DinkVisionReplayListDataSource {
    var packageRootURL: URL
    var loadPackages: (URL) throws -> [CaptureLibraryItem]
    var bundledSamplePackageIDs: Set<String>

    init(
        packageRootURL: URL = CameraCaptureController.defaultPackageRootURL(),
        loadPackages: @escaping (URL) throws -> [CaptureLibraryItem] = { url in
            try CaptureLibrary.listPackages(packageRootURL: url)
        },
        bundledSamplePackageIDs: Set<String> = []
    ) {
        self.packageRootURL = packageRootURL
        self.loadPackages = loadPackages
        self.bundledSamplePackageIDs = bundledSamplePackageIDs
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
            title: item.isImported ? "Imported video" : "Recorded session",
            subtitle: "\(Self.dateText(for: item.recordedAt)) · \(item.fps) fps · \(item.resolutionText)",
            durationText: Self.durationText(for: item.durationSeconds),
            dateText: Self.dateText(for: item.recordedAt),
            trustBadgeText: item.captureQualityGrade == .good ? "Capture checks passed" : "Capture needs review",
            trusted3DText: sidecarTexts.trusted3D,
            ballTrustText: sidecarTexts.ball,
            item: item,
            source: bundledSamplePackageIDs.contains(item.sessionID) ? .bundledSample : .capture(item.sessionID)
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
