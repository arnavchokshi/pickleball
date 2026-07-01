import XCTest
@testable import Pickleball

final class AppFlowTests: XCTestCase {
    @MainActor
    func testLaunchFlowStartsOnSplashThenMovesHomeThenCamera() {
        let flow = PickleballAppFlow()

        XCTAssertEqual(flow.screen, .splash)

        flow.finishSplash()
        XCTAssertEqual(flow.screen, .home)

        flow.openCamera()
        XCTAssertEqual(flow.screen, .camera)
    }
}
