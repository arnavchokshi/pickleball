import XCTest

/// Drives a REAL Record/Stop session through the actual on-screen controls
/// (cold-launch Record tab -> Start recording -> wait -> Stop recording) using
/// XCUIApplication, so this exercises the exact same tap targets a human
/// finger would use. It intentionally does not assert on file contents --
/// that verification happens out-of-process by pulling the app container
/// with `devicectl` after this test runs. See
/// runs/ios_device_gate_20260702T*/ for the pulled evidence.
final class RecordStopUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testRecordStopProducesRealCapturePackage() throws {
        let app = XCUIApplication()
        app.launchArguments += ["-dinkvision.skipSplash"]

        // Camera/mic/motion permission alerts are presented by springboard on
        // first launch. XCUITest only checks interruption monitors when the
        // app under test receives an interaction, so we keep tapping a safe,
        // control-free spot while polling for the next expected element.
        addUIInterruptionMonitor(withDescription: "System permission alert") { alert in
            for label in ["OK", "Allow", "Allow While Using App"] {
                let button = alert.buttons[label]
                if button.exists {
                    button.tap()
                    return true
                }
            }
            return false
        }

        // NOTE: Recording is gated on CaptureOrientationPolicy landscape
        // (CameraCaptureController.startRecording() throws .landscapeRequired
        // otherwise). XCUIDevice.shared.orientation was tried here to spoof
        // landscape and was rejected: on this physical device it desynced the
        // accessibility coordinate space (window reported 852x393) from the
        // true physical panel (screen recording stayed portrait pixels), so
        // taps landed on the wrong physical spot and recording never started.
        // Real landscape requires physically rotating the phone; see the run
        // dir notes for the manual step.

        app.launch()

        let recordButton = app.buttons["DinkVisionRecordButton"]
        XCTAssertTrue(
            waitForHittable(recordButton, in: app, timeout: 25),
            "DinkVisionRecordButton never became hittable on the cold-launch Record tab"
        )
        XCTAssertEqual(recordButton.label, "Start recording")
        recordButton.tap()

        // Evidence for the run log regardless of pass/fail below: the exact
        // on-screen status pill text tells us whether the tap actually
        // registered (status flips away from "Ready") or whether the camera
        // pipeline rejected/stalled the start. Poll+log every ~3s for up to
        // 30s to distinguish "slow hardware bring-up" from "permanently
        // stuck".
        let stopButton = app.buttons["DinkVisionRecordButton"]
        var becameHittable = false
        for tick in 0..<10 {
            if stopButton.exists && stopButton.isHittable {
                becameHittable = true
                break
            }
            print("RECORD_STOP_UITEST_POLL_TICK_\(tick)_BEGIN")
            print(app.debugDescription)
            print("RECORD_STOP_UITEST_POLL_TICK_\(tick)_END")
            Thread.sleep(forTimeInterval: 3)
        }
        XCTAssertTrue(
            becameHittable,
            "Stop recording button never appeared after tapping Start recording (waited 30s)"
        )
        XCTAssertEqual(stopButton.label, "Stop recording")

        // Real 3-5s clip per the W2-IOS-DEVICE gate.
        Thread.sleep(forTimeInterval: 5)

        stopButton.tap()

        // Give AVCaptureMovieFileOutput's finish delegate and the sidecar
        // writer time to flush to disk before the test process exits.
        Thread.sleep(forTimeInterval: 4)
    }

    @discardableResult
    private func waitForHittable(_ element: XCUIElement, in app: XCUIApplication, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        // Center of the screen is clear of the header (back/status/orientation
        // pills) and the record button (pinned to a trailing/bottom edge) in
        // both portrait and landscape layouts, so tapping it here only serves
        // to give the interruption monitor a chance to fire.
        let safeSpot = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        while Date() < deadline {
            if element.exists && element.isHittable {
                return true
            }
            safeSpot.tap()
            usleep(400_000)
        }
        return element.exists && element.isHittable
    }
}
