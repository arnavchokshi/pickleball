import XCTest
@testable import PickleballCapture

final class RecordControlInteractionPolicyTests: XCTestCase {
    func testBlockedCaptureStateHasPersistentHighContrastGuidance() {
        let rotate = RecordControlInteractionPolicy.guidance(
            for: .blocked(reason: "Rotate to landscape to record")
        )
        let camera = RecordControlInteractionPolicy.guidance(
            for: .blocked(reason: "Back camera is unavailable. Close other camera apps, then tap Retry.")
        )

        XCTAssertEqual(rotate?.kind, .rotateToLandscape)
        XCTAssertEqual(rotate?.title, "Rotate to landscape")
        XCTAssertEqual(rotate?.systemImage, "rectangle.landscape.rotate")
        XCTAssertEqual(rotate?.actionTitle, "Retry")
        XCTAssertEqual(rotate?.accentClusterCount, 1)
        XCTAssertTrue(rotate?.isPersistent == true)
        XCTAssertTrue(rotate?.usesHighContrastSurface == true)
        XCTAssertEqual(camera?.kind, .blocked)
        XCTAssertEqual(camera?.message, "Back camera is unavailable. Close other camera apps, then tap Retry.")
        XCTAssertNil(RecordControlInteractionPolicy.guidance(for: .ready))
        XCTAssertNil(RecordControlInteractionPolicy.guidance(for: .recording))
    }

    func testEveryCaptureStateMapsToAVisibleTapReaction() {
        let states: [RecordControlState] = [
            .idle,
            .preparing,
            .blocked(reason: "Rotate to landscape to record"),
            .ready,
            .recording,
        ]

        let reactions = states.map(RecordControlInteractionPolicy.tapReaction(for:))

        XCTAssertEqual(reactions.count, states.count)
        XCTAssertTrue(reactions.allSatisfy(\.hasVisibleConsequence))
        XCTAssertEqual(reactions[1].kind, .preparing)
        XCTAssertEqual(reactions[2].kind, .blocked)
        XCTAssertTrue(reactions[1].usesWarningHaptic)
        XCTAssertTrue(reactions[2].usesWarningHaptic)
        XCTAssertEqual(reactions[1].reducedMotionEmphasis, .staticHighlight)
        XCTAssertEqual(reactions[2].reducedMotionEmphasis, .staticHighlight)
    }

    func testRecordControlAccessibilityRemainsPresentAndEnabledInEveryState() {
        let states: [RecordControlState] = [
            .idle,
            .preparing,
            .blocked(reason: "Rotate to landscape to record"),
            .ready,
            .recording,
        ]

        for state in states {
            let accessibility = RecordControlInteractionPolicy.accessibility(for: state)
            XCTAssertTrue(accessibility.elementExists, "missing accessibility element for \(state)")
            XCTAssertTrue(accessibility.isEnabled, "disabled accessibility element for \(state)")
        }
    }

    func testBlockedEntryProducesVoiceOverAnnouncementPlumbing() {
        let reason = "Rotate to landscape to record"

        XCTAssertEqual(
            RecordControlInteractionPolicy.blockedAnnouncement(
                previous: .preparing,
                current: .blocked(reason: reason)
            ),
            "Recording blocked. \(reason)"
        )
        XCTAssertEqual(
            RecordControlInteractionPolicy.blockedAnnouncement(
                previous: .ready,
                current: .blocked(reason: reason)
            ),
            "Recording blocked. \(reason)"
        )
        XCTAssertNil(
            RecordControlInteractionPolicy.blockedAnnouncement(
                previous: .blocked(reason: reason),
                current: .blocked(reason: reason)
            )
        )
    }
}
