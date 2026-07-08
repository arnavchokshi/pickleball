import Foundation

/// Transport-agnostic seam between `WatchLinkController` and whatever moves
/// bytes between phone and watch. The real conformer (`WCSessionTransport`)
/// wraps `WCSession`; tests use an in-memory fake. Class-bound so the
/// mutable `onReceive` callback can be wired without value-type copy
/// surprises -- every real conformer (WCSession delegates need `NSObject`)
/// and every practical fake are reference types anyway.
public protocol WatchLinkTransport: AnyObject {
    func send(_ message: [String: Any])
    var onReceive: (([String: Any]) -> Void)? { get set }
}

/// Pure, synchronous logic shared by both ends of the phone<->watch link.
/// No UI, no WatchConnectivity import -- fully unit-testable with a fake
/// `WatchLinkTransport`.
///
/// Phone side: wire `onStartRecording` / `onStopRecording` / `onStateRequested`
/// to `CaptureViewModel`'s intents, and call `push(state:)` whenever the
/// phone's recording state changes (including in reply to a `requestState`).
///
/// Watch side: call `requestStartRecording()` / `requestStopRecording()` /
/// `requestState()`, and read the latest phone state via `onStateReceived`
/// (or the `latestPhoneState` snapshot).
public final class WatchLinkController {
    private let transport: WatchLinkTransport

    /// Phone-side intents, fired when a `WatchCommand` arrives from the watch.
    public var onStartRecording: (() -> Void)?
    public var onStopRecording: (() -> Void)?
    public var onStateRequested: (() -> Void)?

    /// Watch-side notification, fired when a `PhoneRecordingState` arrives
    /// from the phone.
    public var onStateReceived: ((PhoneRecordingState) -> Void)?

    /// Watch-side snapshot of the most recently decoded phone state.
    public private(set) var latestPhoneState: PhoneRecordingState?

    public init(transport: WatchLinkTransport) {
        self.transport = transport
        self.transport.onReceive = { [weak self] message in
            self?.handleIncoming(message)
        }
    }

    // MARK: Watch-side sends

    public func requestStartRecording() {
        transport.send(WatchCommand.startRecording.toMessage())
    }

    public func requestStopRecording() {
        transport.send(WatchCommand.stopRecording.toMessage())
    }

    public func requestState() {
        transport.send(WatchCommand.requestState.toMessage())
    }

    // MARK: Phone-side sends

    /// Pushes the phone's current recording state to the watch. Callers
    /// should invoke this on every state change and whenever
    /// `onStateRequested` fires.
    public func push(state: PhoneRecordingState) {
        transport.send(state.toMessage())
    }

    // MARK: Incoming dispatch

    private func handleIncoming(_ message: [String: Any]) {
        if let command = WatchCommand(message: message) {
            dispatch(command)
            return
        }
        if let state = PhoneRecordingState(message: message) {
            latestPhoneState = state
            onStateReceived?(state)
        }
        // Malformed/unrecognized payloads are dropped defensively -- no
        // crash, no propagation.
    }

    private func dispatch(_ command: WatchCommand) {
        switch command {
        case .startRecording:
            onStartRecording?()
        case .stopRecording:
            onStopRecording?()
        case .requestState:
            onStateRequested?()
        }
    }
}
