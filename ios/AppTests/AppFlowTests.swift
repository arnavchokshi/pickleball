import XCTest
@testable import Pickleball

final class AppFlowTests: XCTestCase {
    @MainActor
    func testSignedOutAuthGateLeavesRecordAndLocalReplaysReachable() {
        let state = DinkVisionLaunchAccessState(
            authGateEnabled: true,
            isSplashVisible: false,
            isSignedIn: false
        )

        XCTAssertTrue(state.recordTabReachable)
        XCTAssertTrue(state.localReplaysReachable)
        XCTAssertTrue(state.uploadRequiresSignIn)
        XCTAssertTrue(dinkVisionAuthGateEnabled)
    }

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
