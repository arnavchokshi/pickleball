import Foundation
import PickleballCore

public struct ImageSize: Codable, Equatable, Sendable {
    public var width: Double
    public var height: Double

    public init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }

    public init(width: Int, height: Int) {
        self.width = Double(width)
        self.height = Double(height)
    }
}

public enum CalibrationSeedIssue: String, Codable, Equatable, Sendable {
    case invalidImageSize
    case implausibleIntrinsics
    case invalidCameraPose
    case implausibleCourtPlane
    case invalidManualTaps
    case missingCalibrationAnchor
}

public struct CalibrationSeedValidationReport: Codable, Equatable, Sendable {
    public var issues: [CalibrationSeedIssue]
    public var hasARKitSeed: Bool
    public var hasManualFallback: Bool

    public var isUsable: Bool {
        issues.isEmpty
    }

    public init(issues: [CalibrationSeedIssue], hasARKitSeed: Bool, hasManualFallback: Bool) {
        self.issues = issues
        self.hasARKitSeed = hasARKitSeed
        self.hasManualFallback = hasManualFallback
    }
}

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

    public func validationReport(imageSize: ImageSize) -> CalibrationSeedValidationReport {
        var issues: [CalibrationSeedIssue] = []

        if !imageSize.isPlausible {
            issues.append(.invalidImageSize)
        }

        if !intrinsics.isPlausible(for: imageSize) {
            issues.append(.implausibleIntrinsics)
        }

        let hasValidPose: Bool
        if let arkitCameraPose {
            hasValidPose = arkitCameraPose.isPlausibleRigidPose
            if !hasValidPose {
                issues.append(.invalidCameraPose)
            }
        } else {
            hasValidPose = false
        }

        let hasValidPlane: Bool
        if let courtPlane {
            hasValidPlane = courtPlane.isPlausibleCourtPlane
            if !hasValidPlane {
                issues.append(.implausibleCourtPlane)
            }
        } else {
            hasValidPlane = false
        }

        let hasARKitSeed = hasValidPose && hasValidPlane
        let hasManualFallback: Bool
        if manualCourtTaps.isEmpty {
            hasManualFallback = false
        } else {
            do {
                _ = try ManualCourtTaps(imagePoints: manualCourtTaps).orderedFourCorners(imageSize: imageSize)
                hasManualFallback = true
            } catch {
                hasManualFallback = false
                issues.append(.invalidManualTaps)
            }
        }

        if !hasARKitSeed && !hasManualFallback {
            issues.append(.missingCalibrationAnchor)
        }

        return CalibrationSeedValidationReport(
            issues: issues,
            hasARKitSeed: hasARKitSeed,
            hasManualFallback: hasManualFallback
        )
    }
}

extension ImageSize {
    var isPlausible: Bool {
        width.isFinite && height.isFinite && width > 0 && height > 0
    }
}

extension CameraIntrinsics {
    func isPlausible(for imageSize: ImageSize) -> Bool {
        guard imageSize.isPlausible else { return false }
        let largestDimension = max(imageSize.width, imageSize.height)
        let minFocal = largestDimension * 0.2
        let maxFocal = largestDimension * 5.0

        return fx.isFinite
            && fy.isFinite
            && cx.isFinite
            && cy.isFinite
            && fx >= minFocal
            && fy >= minFocal
            && fx <= maxFocal
            && fy <= maxFocal
            && cx >= 0
            && cx <= imageSize.width
            && cy >= 0
            && cy <= imageSize.height
            && dist.allSatisfy(\.isFinite)
    }
}

extension RigidPose {
    var isPlausibleRigidPose: Bool {
        R.count == 3
            && R.allSatisfy { $0.count == 3 && $0.allSatisfy(\.isFinite) }
            && t.count == 3
            && t.allSatisfy(\.isFinite)
    }
}

extension Plane {
    var isPlausibleCourtPlane: Bool {
        guard point.count == 3,
              normal.count == 3,
              point.allSatisfy(\.isFinite),
              normal.allSatisfy(\.isFinite)
        else {
            return false
        }

        let length = sqrt(normal.reduce(0) { $0 + ($1 * $1) })
        return (0.8...1.2).contains(length)
    }
}
