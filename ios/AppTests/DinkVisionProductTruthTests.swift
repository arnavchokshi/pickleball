import XCTest
import Foundation
import UIKit
import PickleballCapture
import PickleballCore
import PickleballUpload
@testable import Pickleball

@MainActor
final class DinkVisionProductTruthTests: XCTestCase {
    func testFiveTabOrderColdLaunchAndRecordingStopPresentation() {
        XCTAssertEqual(
            DinkVisionTabLayoutModel.brandV4.tabs,
            [.replays, .stats, .record, .coach, .profile]
        )
        XCTAssertEqual(DinkVisionTabKind.coldLaunchDefault, .record)

        let startedAt = Date(timeIntervalSince1970: 1_000)
        let presentation = DinkVisionRecordingPresentation(
            isRecording: true,
            startedAt: startedAt,
            now: startedAt.addingTimeInterval(125)
        )
        XCTAssertEqual(presentation.controlState, .recording)
        XCTAssertEqual(presentation.elapsedText, "2:05")
        XCTAssertEqual(presentation.accessibilityLabel, "Stop recording")
        XCTAssertEqual(presentation.accessibilityValue, "Recording, elapsed time 2:05")
        XCTAssertEqual(DinkVisionRecordButtonVisual.recording.centerShape, .roundedSquareStop)
        XCTAssertEqual(DinkVisionRecordButtonVisual.recording.ring, .trailRed)
    }

    func testReplayPresentationsCoverEveryHonestLifecycleStateAndDisclosePartialCapabilities() {
        let sample = DinkVisionReplayStatusPresentation(uploadState: nil, isSample: true)
        XCTAssertEqual(sample.state, .sample)
        XCTAssertEqual(sample.detail, "Bundled fixture — not one of your sessions")
        XCTAssertEqual(DinkVisionReplayStatusPresentation(uploadState: nil).state, .local)
        XCTAssertEqual(presentation(state: .queued).state, .queued)

        let uploading = presentation(state: .uploading, bytesUploaded: 25, totalBytes: 100)
        XCTAssertEqual(uploading.state, .uploading(percent: 25))
        XCTAssertEqual(uploading.title, "Uploading 25%")

        let processing = presentation(state: .uploaded, serverStatus: "processing")
        XCTAssertEqual(processing.state, .processing)
        XCTAssertEqual(processing.detail, "Server status: processing")

        let partial = presentation(
            state: .uploaded,
            serverStatus: "partial",
            missingCapabilities: [
                RenderGatewayMissingCapability(capability: "body_mesh", reason: "BODY output missing"),
                RenderGatewayMissingCapability(capability: "ball", reason: "confidence gate abstained"),
            ]
        )
        XCTAssertEqual(partial.state, .partial)
        XCTAssertEqual(partial.missingCapabilities.count, 2)
        XCTAssertTrue(partial.detail?.contains("body mesh: BODY output missing") == true)
        XCTAssertTrue(partial.detail?.contains("ball: confidence gate abstained") == true)

        let ready = presentation(
            state: .uploaded,
            serverStatus: "complete",
            manifestUrl: "https://example.test/manifest.json"
        )
        XCTAssertEqual(ready.state, .ready)

        let failed = presentation(state: .failed, lastError: "network unavailable")
        XCTAssertEqual(failed.state, .failed)
        XCTAssertEqual(failed.detail, "Upload or processing failed: network unavailable")
    }

    func testFactsDecoderRendersOnlySourceLinkedDecodedValuesAndRejectsFabricationPaths() throws {
        let valid: [String: Any] = [
            "audited_facts": [[
                "fact_id": "distance-r1-p1",
                "metric": "distance_covered_m",
                "value": 12.5,
                "unit": "m",
                "trust": ["authority_band": "preview", "provenance_band": "measured"],
                "evidence_locator": ["uri": "replay://session/rally/1?t=4.2"],
                "source_artifacts": [["path": "rally_metrics.json", "sha256": String(repeating: "a", count: 64)]],
                "player_id": "player-1",
            ]],
        ]
        let decoded = DinkVisionFactsDocumentDecoder.decode(
            try JSONSerialization.data(withJSONObject: valid),
            sessionID: "real-session"
        )
        XCTAssertEqual(decoded.count, 1)
        XCTAssertEqual(decoded[0].valueText, "12.5 m")
        XCTAssertEqual(decoded[0].authority, .preview)
        XCTAssertEqual(decoded[0].provenance, .measured)
        XCTAssertEqual(decoded[0].evidenceLocator, "replay://session/rally/1?t=4.2")

        for missingKey in ["fact_id", "trust", "evidence_locator", "source_artifacts", "value"] {
            var invalidFact = try XCTUnwrap((valid["audited_facts"] as? [[String: Any]])?.first)
            invalidFact.removeValue(forKey: missingKey)
            let invalidData = try JSONSerialization.data(withJSONObject: ["audited_facts": [invalidFact]])
            XCTAssertTrue(
                DinkVisionFactsDocumentDecoder.decode(invalidData, sessionID: "real-session").isEmpty,
                "facts missing \(missingKey) must not render"
            )
        }

        for missingTrustKey in ["authority_band", "provenance_band"] {
            var invalidFact = try XCTUnwrap((valid["audited_facts"] as? [[String: Any]])?.first)
            var trust = try XCTUnwrap(invalidFact["trust"] as? [String: Any])
            trust.removeValue(forKey: missingTrustKey)
            invalidFact["trust"] = trust
            XCTAssertTrue(
                DinkVisionFactsDocumentDecoder.decode(
                    try JSONSerialization.data(withJSONObject: ["audited_facts": [invalidFact]]),
                    sessionID: "real-session"
                ).isEmpty
            )
        }

        var missingSourceHash = try XCTUnwrap((valid["audited_facts"] as? [[String: Any]])?.first)
        missingSourceHash["source_artifacts"] = [["path": "rally_metrics.json"]]
        XCTAssertTrue(
            DinkVisionFactsDocumentDecoder.decode(
                try JSONSerialization.data(withJSONObject: ["audited_facts": [missingSourceHash]]),
                sessionID: "real-session"
            ).isEmpty
        )

        var invented = try XCTUnwrap((valid["audited_facts"] as? [[String: Any]])?.first)
        invented["value"] = "Great footwork!"
        XCTAssertTrue(
            DinkVisionFactsDocumentDecoder.decode(
                try JSONSerialization.data(withJSONObject: ["audited_facts": [invented]]),
                sessionID: "real-session"
            ).isEmpty
        )
    }

    func testSampleFactsNeverEnterRealStatsOrCoachDataSource() throws {
        let item = CaptureLibraryItem(
            sessionID: "explicit-sample",
            clipRelativePath: "captures/explicit-sample/clip.mov",
            sidecarRelativePath: "captures/explicit-sample/capture_sidecar.json",
            provenance: .liveRecording,
            durationSeconds: 30,
            fps: 60,
            resolution: [1920, 1080],
            captureQualityGrade: .good,
            recordedAt: Date()
        )
        let factData = try JSONSerialization.data(withJSONObject: [
            "audited_facts": [[
                "fact_id": "sample-fact",
                "metric": "distance_covered_m",
                "value": 99,
                "unit": "m",
                "trust": ["authority_band": "preview", "provenance_band": "measured"],
                "evidence_locator": ["uri": "fixture://sample"],
                "source_artifacts": [["path": "fixture.json", "sha256": String(repeating: "b", count: 64)]],
            ]],
        ])
        let replayDataSource = DinkVisionReplayListDataSource(
            packageRootURL: URL(fileURLWithPath: "/tmp/dinkvision-product-truth"),
            loadPackages: { _ in [item] },
            bundledSamplePackageIDs: ["explicit-sample"]
        )
        let facts = DinkVisionFactsLibraryDataSource(
            replayDataSource: replayDataSource,
            readData: { _ in factData }
        ).loadFacts()

        XCTAssertTrue(facts.isEmpty)
        let sampleRow = try XCTUnwrap(replayDataSource.loadRows().first)
        XCTAssertEqual(sampleRow.source, .bundledSample)
    }

    func testProfileSurfacesCurrentAccountUploadAndSessionOnlyNonOwnerPrivacyDefault() {
        let local = DinkVisionProfileSettingsModel.current(
            isSignedIn: false,
            autoUploadAfterRecording: false
        )
        XCTAssertEqual(local.accountTitle, "Local mode")
        XCTAssertEqual(local.uploadTitle, "Auto-upload off")
        XCTAssertEqual(local.nonOwnerRetentionTitle, "Non-owner data: session only")
        XCTAssertTrue(local.nonOwnerRetentionDetail.contains("explicitly opts in"))

        let signedIn = DinkVisionProfileSettingsModel.current(
            isSignedIn: true,
            autoUploadAfterRecording: true
        )
        XCTAssertEqual(signedIn.accountTitle, "Signed in")
        XCTAssertEqual(signedIn.uploadTitle, "Auto-upload on")
    }

    func testAccessibilityPolicyRequiresDynamicTypeVoiceOverSafeAreaContrastAndReducedMotion() {
        let policy = DinkVisionAccessibilityPolicy.productUI
        XCTAssertTrue(policy.supportsDynamicType)
        XCTAssertTrue(policy.suppliesVoiceOverLabels)
        XCTAssertTrue(policy.keepsDataInsideSafeAreas)
        XCTAssertTrue(policy.usesAdaptiveDarkAndHighContrastColors)
        XCTAssertTrue(policy.hasReducedMotionFallbacks)

        XCTAssertEqual(DinkVisionScreenMotionParameters.resolved(reducedMotion: true).slidePoints, 0)
        XCTAssertEqual(DinkVisionStrokeDrawOnParameters.resolved(reducedMotion: true).durationSeconds, 0)
        XCTAssertEqual(DinkVisionReplayOpenTransition.plan(reducedMotion: true), .crossfade)

        let cream = UIColor(DinkVisionColor.cream)
        let light = cream.resolvedColor(with: UITraitCollection(userInterfaceStyle: .light))
        let dark = cream.resolvedColor(with: UITraitCollection(userInterfaceStyle: .dark))
        let highContrastLight = cream.resolvedColor(with: UITraitCollection { traits in
            traits.userInterfaceStyle = .light
            traits.accessibilityContrast = .high
        })
        XCTAssertNotEqual(light, dark)
        XCTAssertNotEqual(light, highContrastLight)
    }

    private func presentation(
        state: CaptureUploadStateKind,
        bytesUploaded: Int64 = 0,
        totalBytes: Int64 = 100,
        serverStatus: String? = nil,
        manifestUrl: String? = nil,
        missingCapabilities: [RenderGatewayMissingCapability] = [],
        lastError: String? = nil
    ) -> DinkVisionReplayStatusPresentation {
        DinkVisionReplayStatusPresentation(uploadState: CaptureUploadState(
            state: state,
            captureId: "capture",
            clipId: "clip",
            bytesUploaded: bytesUploaded,
            totalBytes: totalBytes,
            lastError: lastError,
            serverStatus: serverStatus,
            jobId: "job",
            manifestUrl: manifestUrl,
            missingCapabilities: missingCapabilities
        ))
    }
}
