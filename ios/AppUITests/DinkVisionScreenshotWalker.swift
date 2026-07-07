import XCTest

final class DinkVisionScreenshotWalker: XCTestCase {
    enum WalkerError: Error, CustomStringConvertible {
        case missingElement(String)

        var description: String {
            switch self {
            case let .missingElement(message):
                return message
            }
        }
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testWalkProductionScreens() throws {
        let app = XCUIApplication()

        launch(app, arguments: ["-dinkvision.walker", "1"])
        settle(0.05)
        attachScreenshot(app, name: "01_splash_settle")
        settle(0.32)
        attachScreenshot(app, name: "02_splash_blink")
        settle(0.48)
        attachScreenshot(app, name: "03_splash_openup")
        app.terminate()

        launch(app, arguments: walkerArguments(captureState: "permissionPrimer", replayState: "empty"))
        settle(1.0)
        attachScreenshot(app, name: "04_record_permission_primer")
        app.terminate()

        launch(app, arguments: walkerArguments(captureState: "granted", replayState: "empty"))
        settle(1.0)
        attachScreenshot(app, name: "05_record_granted")

        try tapTab("replays", in: app)
        settle(1.0)
        attachScreenshot(app, name: "06_replays_empty")

        try tapTab("stats", in: app)
        settle(1.0)
        attachScreenshot(app, name: "07_stats")

        try tapTab("coach", in: app)
        settle(1.0)
        attachScreenshot(app, name: "08_coach")

        try tapTab("profile", in: app)
        settle(1.0)
        attachScreenshot(app, name: "09_profile")
        app.terminate()

        launch(
            app,
            arguments: walkerArguments(
                captureState: "granted",
                replayState: "seeded",
                extra: ["-dinkvision.forceWorldCoachMark"]
            )
        )
        try tapTab("replays", in: app)
        let seededRow = app.buttons["DinkVisionReplayRow-walker-seeded-rally"]
        settle(1.0)
        attachScreenshot(app, name: "10_replays_seeded_fixture")

        seededRow.tap()
        settle(0.26)
        settle(1.0)
        attachScreenshot(app, name: "11_replay_open_swoosh_midframe")

        settle(1.0)
        attachScreenshot(app, name: "12_3d_viewer_coach_mark")

        let gotIt = app.buttons["Got it"]
        if gotIt.exists {
            gotIt.tap()
        }
        settle(1.0)
        settle(1.0)
        attachScreenshot(app, name: "13_3d_viewer_controls_visible")
        app.terminate()

        launch(
            app,
            arguments: walkerArguments(
                captureState: "granted",
                replayState: "empty",
                extra: ["-dinkvision.recordPressed"]
            )
        )
        settle(1.0)
        attachScreenshot(app, name: "14_record_pressed_state")
        app.terminate()
    }

    private func walkerArguments(captureState: String, replayState: String, extra: [String] = []) -> [String] {
        [
            "-dinkvision.walker", "1",
            "-dinkvision.skipSplash", "1",
            "-dinkvision.captureState", captureState,
            "-dinkvision.replays", replayState,
        ] + extra
    }

    private func launch(_ app: XCUIApplication, arguments: [String]) {
        app.launchArguments = arguments
        app.launch()
        settle(0.35)
    }

    private func tapTab(_ id: String, in app: XCUIApplication) throws {
        let tab = app.buttons["DinkVisionTab-\(id)"]
        settle(1.0)
        tab.tap()
        settle(0.45)
    }

    private func require(_ condition: Bool, _ message: String) throws {
        if !condition {
            throw WalkerError.missingElement(message)
        }
    }

    private func attachScreenshot(_ app: XCUIApplication, name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func waitFor(_ element: XCUIElement, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if element.exists {
                return true
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        }
        return element.exists
    }

    private func settle(_ seconds: TimeInterval) {
        RunLoop.current.run(until: Date().addingTimeInterval(seconds))
    }
}
