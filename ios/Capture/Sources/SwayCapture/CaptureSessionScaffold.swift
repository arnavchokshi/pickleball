import Foundation
import SwayCore

public struct CaptureSessionScaffold: Equatable, Sendable {
    public var mode: CaptureMode
    public var format: CaptureFormat
    public var orientation: CaptureOrientation
    public var requestedFPS: Int

    public init(mode: CaptureMode) {
        self.mode = mode
        self.format = .hevc
        self.orientation = .landscape
        self.requestedFPS = mode == .swing120 ? 120 : 60
    }
}
