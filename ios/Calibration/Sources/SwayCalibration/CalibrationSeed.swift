import Foundation
import SwayCore

public struct CalibrationSeed: Codable, Equatable, Sendable {
    public var intrinsics: CameraIntrinsics
    public var arkitCameraPose: RigidPose?
    public var courtPlane: Plane?
    public var manualCourtTaps: [[Double]]

    public init(
        intrinsics: CameraIntrinsics,
        arkitCameraPose: RigidPose? = nil,
        courtPlane: Plane? = nil,
        manualCourtTaps: [[Double]] = []
    ) {
        self.intrinsics = intrinsics
        self.arkitCameraPose = arkitCameraPose
        self.courtPlane = courtPlane
        self.manualCourtTaps = manualCourtTaps
    }
}
