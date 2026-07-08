import XCTest
@testable import WatchBridge

final class WatchMessagesTests: XCTestCase {
    // MARK: WatchCommand round-trip

    func testStartRecordingRoundTripsThroughDictMessage() throws {
        let message = WatchCommand.startRecording.toMessage()
        let decoded = try XCTUnwrap(WatchCommand(message: message))
        XCTAssertEqual(decoded, .startRecording)
    }

    func testStopRecordingRoundTripsThroughDictMessage() throws {
        let message = WatchCommand.stopRecording.toMessage()
        let decoded = try XCTUnwrap(WatchCommand(message: message))
        XCTAssertEqual(decoded, .stopRecording)
    }

    func testRequestStateRoundTripsThroughDictMessage() throws {
        let message = WatchCommand.requestState.toMessage()
        let decoded = try XCTUnwrap(WatchCommand(message: message))
        XCTAssertEqual(decoded, .requestState)
    }

    func testAllWatchCommandCasesRoundTrip() throws {
        let allCases: [WatchCommand] = [.startRecording, .stopRecording, .requestState]
        for command in allCases {
            let decoded = try XCTUnwrap(WatchCommand(message: command.toMessage()))
            XCTAssertEqual(decoded, command)
        }
    }

    // MARK: PhoneRecordingState round-trip

    func testPhoneRecordingStateRoundTripsWithAllFieldsPresent() throws {
        let state = PhoneRecordingState(
            isRecording: true,
            canRecord: true,
            clipName: "session_042.mov",
            elapsedSeconds: 12.5
        )
        let message = state.toMessage()
        let decoded = try XCTUnwrap(PhoneRecordingState(message: message))
        XCTAssertEqual(decoded, state)
    }

    func testPhoneRecordingStateRoundTripsWithNilClipName() throws {
        let state = PhoneRecordingState(isRecording: false, canRecord: true, clipName: nil, elapsedSeconds: 0)
        let message = state.toMessage()
        let decoded = try XCTUnwrap(PhoneRecordingState(message: message))
        XCTAssertEqual(decoded, state)
        XCTAssertNil(decoded.clipName)
    }

    // MARK: Defensive decoding -- malformed input returns nil, never crashes

    func testWatchCommandDecodeReturnsNilForEmptyMessage() {
        XCTAssertNil(WatchCommand(message: [:]))
    }

    func testWatchCommandDecodeReturnsNilForUnknownTypeDiscriminator() {
        XCTAssertNil(WatchCommand(message: ["type": "notARealCommand"]))
    }

    func testWatchCommandDecodeReturnsNilForWrongValueType() {
        // "type" present but not a string -- must not crash the decoder.
        XCTAssertNil(WatchCommand(message: ["type": 42]))
    }

    func testWatchCommandDecodeReturnsNilForNonJSONValue() {
        // NSObject() cannot be represented as a JSON value at all;
        // WatchMessageCoding must detect this and bail out cleanly.
        XCTAssertNil(WatchCommand(message: ["type": NSObject()]))
    }

    func testPhoneRecordingStateDecodeReturnsNilForMissingRequiredField() {
        // Missing "canRecord" and "elapsedSeconds".
        XCTAssertNil(PhoneRecordingState(message: ["isRecording": true]))
    }

    func testPhoneRecordingStateDecodeReturnsNilForWrongFieldType() {
        XCTAssertNil(PhoneRecordingState(message: [
            "isRecording": "not-a-bool",
            "canRecord": true,
            "elapsedSeconds": 0,
        ]))
    }

    func testPhoneRecordingStateDecodeReturnsNilWhenGivenAWatchCommandMessage() {
        // Cross-type confusion must fail closed, not partially decode.
        XCTAssertNil(PhoneRecordingState(message: WatchCommand.startRecording.toMessage()))
    }
}
