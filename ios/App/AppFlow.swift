import SwiftUI

enum PickleballAppScreen: Equatable {
    case splash
    case home
    case camera
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

    func returnHome() {
        screen = .home
    }
}
