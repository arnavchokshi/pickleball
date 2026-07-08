import Foundation

#if canImport(WatchConnectivity)
import WatchConnectivity

/// Real `WatchLinkTransport` adapter over `WCSession`. Isolated behind
/// `#if canImport(WatchConnectivity)` so `WatchBridge` still builds and
/// tests on platforms (macOS) where WatchConnectivity is unavailable --
/// `WatchLinkController` and the message contract never import this file's
/// framework.
///
/// `send` prefers `sendMessage` (low-latency, needs an active/reachable
/// counterpart) and falls back to `updateApplicationContext` (persisted,
/// delivered next time the counterpart wakes) whenever the counterpart is
/// unreachable or `sendMessage` itself errors -- the right behavior for
/// state pushes (`PhoneRecordingState`), where "eventually delivered" beats
/// "silently dropped."
public final class WCSessionTransport: NSObject, WatchLinkTransport {
    public var onReceive: (([String: Any]) -> Void)?

    private let session: WCSession

    public init(session: WCSession = .default) {
        self.session = session
        super.init()
        session.delegate = self
    }

    /// Activates the underlying `WCSession`. No-op if WatchConnectivity is
    /// unsupported on this device. Safe to call more than once.
    public func activate() {
        guard WCSession.isSupported() else {
            return
        }
        session.delegate = self
        session.activate()
    }

    public func send(_ message: [String: Any]) {
        guard WCSession.isSupported(), session.activationState == .activated else {
            return
        }
        guard session.isReachable else {
            updateApplicationContext(message)
            return
        }
        session.sendMessage(message, replyHandler: nil) { [weak self] _ in
            // Live send failed (e.g. counterpart went unreachable between
            // the isReachable check and delivery) -- fall back so the
            // payload isn't silently lost.
            self?.updateApplicationContext(message)
        }
    }

    private func updateApplicationContext(_ message: [String: Any]) {
        try? session.updateApplicationContext(message)
    }
}

extension WCSessionTransport: WCSessionDelegate {
    public func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        // No-op: callers observe link health via onReceive / their own
        // pushed state, not activation completion.
    }

    public func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        onReceive?(message)
    }

    public func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String: Any]) {
        onReceive?(applicationContext)
    }

    #if os(iOS)
    public func sessionDidBecomeInactive(_ session: WCSession) {}

    public func sessionDidDeactivate(_ session: WCSession) {
        // Re-activate for the next paired watch, per Apple's guidance for
        // multi-watch iPhone pairing.
        session.activate()
    }
    #endif
}

#endif
