import XCTest
@testable import PickleballReplay

final class TrustBandTests: XCTestCase {
    func testChipTextForMissingTrustBandIsExplicitNotBlank() {
        XCTAssertEqual(TrustBandPresentation.chipText(nil), "no trust band")
    }

    func testChipTextFormatsStageAndBadgeWithSpaces() {
        let band = TrustBand(stage: "BODY", gateID: "g", gateStatus: "s", badge: .lowConfidence, reason: "r", evidencePath: nil)
        XCTAssertEqual(TrustBandPresentation.chipText(band), "BODY: low confidence")
    }

    func testBadgeDefaultsToVerifiedWhenNoTrustBandExists() {
        XCTAssertEqual(TrustBandPresentation.badge(for: nil), .verified)
    }

    func testBadgeReturnsTheBandsOwnBadgeWhenPresent() {
        let band = TrustBand(stage: "CAL", gateID: "g", gateStatus: "s", badge: .preview, reason: "r", evidencePath: nil)
        XCTAssertEqual(TrustBandPresentation.badge(for: band), .preview)
    }

    func testTrustBandDecodesRealSnakeCaseJSON() throws {
        let json = """
        {"stage":"TRK","gate_id":"trk_idf1_gate","gate_status":"do_not_promote","badge":"low_confidence","reason":"IDF1 below gate","evidence_path":"runs/foo/"}
        """.data(using: .utf8)!
        let band = try JSONDecoder().decode(TrustBand.self, from: json)
        XCTAssertEqual(band.stage, "TRK")
        XCTAssertEqual(band.gateID, "trk_idf1_gate")
        XCTAssertEqual(band.badge, .lowConfidence)
        XCTAssertEqual(band.evidencePath, "runs/foo/")
    }

    func testTrustBandEvidencePathIsOptional() throws {
        let json = """
        {"stage":"BALL","gate_id":"g","gate_status":"s","badge":"low_confidence","reason":"r","evidence_path":null}
        """.data(using: .utf8)!
        let band = try JSONDecoder().decode(TrustBand.self, from: json)
        XCTAssertNil(band.evidencePath)
    }
}
