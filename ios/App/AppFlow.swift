import SwiftUI

enum PickleballAppScreen: Equatable {
    case splash
    case signIn
    case home
    case camera
    case worldViewer
    case realityReplay
}

/// Authentication may protect upload, but it never protects local recording
/// or the local replay library. The root view uses this model directly so the
/// signed-out invariant is testable without a UI fixture bypass.
struct DinkVisionLaunchAccessState: Equatable {
    var authGateEnabled: Bool
    var isSplashVisible: Bool
    var isSignedIn: Bool

    var recordTabReachable: Bool { !isSplashVisible }
    var localReplaysReachable: Bool { !isSplashVisible }
    var uploadRequiresSignIn: Bool { authGateEnabled && !isSignedIn }
}

@MainActor
final class PickleballAppFlow: ObservableObject {
    @Published private(set) var screen: PickleballAppScreen = .splash

    /// Historical, unconditional splash exit -- always lands on `.home`.
    /// Kept exactly as-is (existing `AppFlowTests` depend on this) so this
    /// INFRA-4 addition stays purely additive.
    func finishSplash() {
        screen = .home
    }

    /// Token-aware variant for a real auth-gated launch: routes to `.signIn`
    /// when there's no stored session, `.home` otherwise. NOTE: as of
    /// INFRA-4, `PickleballAppFlow` is not yet wired into the live
    /// `DinkVisionAppRootView` shell (see AppRootView.swift) -- that shell
    /// keeps its own local splash/sign-in state today. This overload exists
    /// so a future refactor that does route AppRootView through
    /// `PickleballAppFlow` has the method ready; it is additive scaffolding,
    /// not yet a consumed code path.
    func finishSplash(hasStoredSession: Bool) {
        screen = hasStoredSession ? .home : .signIn
    }

    func finishSignIn() {
        screen = .home
    }

    func signOut() {
        screen = .signIn
    }

    func openCamera() {
        screen = .camera
    }

    func openWorldViewer() {
        screen = .worldViewer
    }

    func openRealityReplay() {
        screen = .realityReplay
    }

    func returnHome() {
        screen = .home
    }
}
