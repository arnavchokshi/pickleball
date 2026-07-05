import XCTest
@testable import PickleballCalibration

final class AssistedCourtSeedTests: XCTestCase {
    func testOneInsideTapIsNotTrustedCalibration() throws {
        let seed = try AssistedCourtSeed(
            mode: .oneInsideTap,
            imageWidth: 1920,
            imageHeight: 1080,
            points: [.init(x: 500, y: 400)]
        )

        XCTAssertEqual(seed.mode, .oneInsideTap)
        XCTAssertEqual(seed.points, [.init(x: 500, y: 400)])
        XCTAssertFalse(seed.trustedCalibration)
    }

    func testTapOutsideFrameFails() {
        XCTAssertThrowsError(
            try AssistedCourtSeed(
                mode: .oneInsideTap,
                imageWidth: 1920,
                imageHeight: 1080,
                points: [.init(x: 2000, y: 400)]
            )
        ) { error in
            XCTAssertEqual(error as? AssistedCourtSeed.ValidationError, .courtTapOutsideFrame)
        }
    }

    func testTwoLineTapsRequireLineLabel() {
        XCTAssertThrowsError(
            try AssistedCourtSeed(
                mode: .twoLineTaps,
                imageWidth: 1920,
                imageHeight: 1080,
                points: [.init(x: 500, y: 400), .init(x: 800, y: 405)]
            )
        ) { error in
            XCTAssertEqual(error as? AssistedCourtSeed.ValidationError, .missingCalibrationAnchor)
        }
    }

    func testAssistedSeedEncodesStableJSON() throws {
        let seed = try AssistedCourtSeed(
            mode: .twoLineTaps,
            imageWidth: 1920,
            imageHeight: 1080,
            points: [.init(x: 500, y: 400), .init(x: 800, y: 405)],
            lineLabel: "near_nvz"
        )

        let encoded = try JSONEncoder().encode(seed)
        let decoded = try JSONDecoder().decode(AssistedCourtSeed.self, from: encoded)

        XCTAssertEqual(decoded, seed)
        XCTAssertFalse(decoded.trustedCalibration)
    }
}
