import XCTest
@testable import PickleballReplay

final class WorldTrustColorsTests: XCTestCase {
    func testLowConfidenceIsDimmedByDefaultToggle() {
        let dimmed = WorldTrustColors.opacity(for: .lowConfidence, dimLowConfidence: true)
        let full = WorldTrustColors.opacity(for: .lowConfidence, dimLowConfidence: false)
        XCTAssertLessThan(dimmed, full)
        XCTAssertEqual(full, 1)
    }

    func testPreviewAndVerifiedAreNeverDimmedByTheLowConfidenceToggle() {
        for badge: TrustBadge in [.preview, .verified] {
            XCTAssertEqual(WorldTrustColors.opacity(for: badge, dimLowConfidence: true), 1)
            XCTAssertEqual(WorldTrustColors.opacity(for: badge, dimLowConfidence: false), 1)
        }
    }

    func testCGColorAlphaChannelMatchesRequestedOpacity() {
        let color = WorldTrustColors.cgColor(for: .lowConfidence, opacity: 0.35)
        XCTAssertEqual(color.alpha, 0.35, accuracy: 1e-9)
    }

    func testEveryBadgeMapsToADistinctColor() {
        let colors = Set([TrustBadge.verified, .preview, .lowConfidence].map { badge -> String in
            let c = WorldTrustColors.cgColor(for: badge)
            return "\(c.components ?? [])"
        })
        XCTAssertEqual(colors.count, 3)
    }
}
