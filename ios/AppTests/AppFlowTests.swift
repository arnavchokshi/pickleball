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

    @MainActor
    func testOpenWorldViewerNavigatesFromHomeAndReturnHomeNavigatesBack() {
        let flow = PickleballAppFlow()
        flow.finishSplash()

        flow.openWorldViewer()
        XCTAssertEqual(flow.screen, .worldViewer)

        flow.returnHome()
        XCTAssertEqual(flow.screen, .home)
    }

    @MainActor
    func testOpenRealityReplayNavigatesFromHomeAndReturnHomeNavigatesBack() {
        let flow = PickleballAppFlow()
        flow.finishSplash()

        flow.openRealityReplay()
        XCTAssertEqual(flow.screen, .realityReplay)

        flow.returnHome()
        XCTAssertEqual(flow.screen, .home)
    }
}
