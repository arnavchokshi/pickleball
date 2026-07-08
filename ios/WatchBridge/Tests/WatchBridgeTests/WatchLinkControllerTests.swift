import XCTest
@testable import WatchBridge

/// In-memory `WatchLinkTransport` used only by tests: no WCSession, no
/// paired device. `send` hands the message straight to whatever closure
/// `wireTo(_:)` recorded, so a pair of these can loop phone<->watch traffic
/// synchronously within a single test.
private final class FakeTransport: WatchLinkTransport {
    var onReceive: (([String: Any]) -> Void)?
    private(set) var sentMessages: [[String: Any]] = []
    private var deliverToPeer: (([String: Any]) -> Void)?

    func send(_ message: [String: Any]) {
        sentMessages.append(message)
        deliverToPeer?(message)
    }

    /// Cross-wires two fakes so `a.send` reaches `b.onReceive` and vice
    /// versa -- the synchronous loopback the spec calls for.
    static func makeLoopbackPair() -> (phone: FakeTransport, watch: FakeTransport) {
        let phone = FakeTransport()
        let watch = FakeTransport()
        phone.deliverToPeer = { [weak watch] message in watch?.onReceive?(message) }
        watch.deliverToPeer = { [weak phone] message in phone?.onReceive?(message) }
        return (phone, watch)
    }
}

final class WatchLinkControllerTests: XCTestCase {
    // MARK: Phone-side: incoming commands fire injected intents

    func testIncomingStartRecordingFiresInjectedStartIntent() {
        let transport = FakeTransport()
        let phoneController = WatchLinkController(transport: transport)
        var startCallCount = 0
        phoneController.onStartRecording = { startCallCount += 1 }

        transport.onReceive?(WatchCommand.startRecording.toMessage())

        XCTAssertEqual(startCallCount, 1)
    }

    func testIncomingStopRecordingFiresInjectedStopIntent() {
        let transport = FakeTransport()
        let phoneController = WatchLinkController(transport: transport)
        var stopCallCount = 0
        phoneController.onStopRecording = { stopCallCount += 1 }

        transport.onReceive?(WatchCommand.stopRecording.toMessage())

        XCTAssertEqual(stopCallCount, 1)
    }

    func testIncomingRequestStateTriggersAStatePush() {
        let transport = FakeTransport()
        let phoneController = WatchLinkController(transport: transport)
        let pushedState = PhoneRecordingState(isRecording: false, canRecord: true, clipName: nil, elapsedSeconds: 0)
        phoneController.onStateRequested = {
            phoneController.push(state: pushedState)
        }

        transport.onReceive?(WatchCommand.requestState.toMessage())

        XCTAssertEqual(transport.sentMessages.count, 1)
        let decoded = PhoneRecordingState(message: transport.sentMessages[0])
        XCTAssertEqual(decoded, pushedState)
    }

    func testMalformedIncomingMessageFiresNoIntentAndDoesNotCrash() {
        let transport = FakeTransport()
        let phoneController = WatchLinkController(transport: transport)
        var anyIntentFired = false
        phoneController.onStartRecording = { anyIntentFired = true }
        phoneController.onStopRecording = { anyIntentFired = true }
        phoneController.onStateRequested = { anyIntentFired = true }

        transport.onReceive?(["garbage": "not-a-known-message"])

        XCTAssertFalse(anyIntentFired)
    }

    // MARK: Watch-side: decodes a pushed state

    func testWatchSideDecodesAPushedPhoneState() {
        let transport = FakeTransport()
        let watchController = WatchLinkController(transport: transport)
        var receivedStates: [PhoneRecordingState] = []
        watchController.onStateReceived = { receivedStates.append($0) }

        let state = PhoneRecordingState(isRecording: true, canRecord: true, clipName: "clip_9.mov", elapsedSeconds: 3.25)
        transport.onReceive?(state.toMessage())

        XCTAssertEqual(receivedStates, [state])
        XCTAssertEqual(watchController.latestPhoneState, state)
    }

    // MARK: End-to-end loopback: both controllers wired via a fake transport pair

    func testEndToEndLoopbackStartRecordingThenStatePushReachesWatch() {
        let (phoneTransport, watchTransport) = FakeTransport.makeLoopbackPair()
        let phoneController = WatchLinkController(transport: phoneTransport)
        let watchController = WatchLinkController(transport: watchTransport)

        var phoneDidStartRecording = false
        phoneController.onStartRecording = {
            phoneDidStartRecording = true
            phoneController.push(state: PhoneRecordingState(isRecording: true, canRecord: false, clipName: nil, elapsedSeconds: 0))
        }

        var watchObservedStates: [PhoneRecordingState] = []
        watchController.onStateReceived = { watchObservedStates.append($0) }

        watchController.requestStartRecording()

        XCTAssertTrue(phoneDidStartRecording)
        XCTAssertEqual(watchObservedStates.map(\.isRecording), [true])
        XCTAssertEqual(watchController.latestPhoneState?.isRecording, true)
    }

    func testEndToEndLoopbackWatchRequestStateReceivesCurrentPhoneState() {
        let (phoneTransport, watchTransport) = FakeTransport.makeLoopbackPair()
        let phoneController = WatchLinkController(transport: phoneTransport)
        let watchController = WatchLinkController(transport: watchTransport)

        let currentState = PhoneRecordingState(isRecording: false, canRecord: true, clipName: "last_clip.mov", elapsedSeconds: 0)
        phoneController.onStateRequested = {
            phoneController.push(state: currentState)
        }

        var watchObservedStates: [PhoneRecordingState] = []
        watchController.onStateReceived = { watchObservedStates.append($0) }

        watchController.requestState()

        XCTAssertEqual(watchObservedStates, [currentState])
    }
}
