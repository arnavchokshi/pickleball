import Foundation
import SwayCore

public struct UploadManifest: Codable, Equatable, Sendable {
    public var clipRelativePath: String
    public var sidecar: CaptureSidecar
    public var onDevicePoseTrack: String?
    public var lidarDepthRefs: [String]

    public init(
        clipRelativePath: String,
        sidecar: CaptureSidecar,
        onDevicePoseTrack: String? = nil,
        lidarDepthRefs: [String] = []
    ) {
        self.clipRelativePath = clipRelativePath
        self.sidecar = sidecar
        self.onDevicePoseTrack = onDevicePoseTrack
        self.lidarDepthRefs = lidarDepthRefs
    }
}
