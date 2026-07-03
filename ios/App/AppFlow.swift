import SwiftUI

enum PickleballAppScreen: Equatable {
    case splash
    case home
    case camera
    case worldViewer
    case realityReplay
}

@MainActor
final class PickleballAppFlow: ObservableObject {
    @Published private(set) var screen: PickleballAppScreen = .splash

    func finishSplash() {
        screen = .home
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
