import Foundation

public struct AssistedCourtSeed: Codable, Equatable, Sendable {
    public enum Mode: String, Codable, Sendable {
        case oneInsideTap
        case twoLineTaps
        case twoKnownCorners
    }

    public enum ValidationError: Error, Equatable, Sendable {
        case invalidImageSize
        case missingCourtTaps
        case courtTapOutsideFrame
        case missingCalibrationAnchor
    }

    public struct Point: Codable, Equatable, Sendable {
        public let x: Double
        public let y: Double

        public init(x: Double, y: Double) {
            self.x = x
            self.y = y
        }
    }

    public let mode: Mode
    public let imageWidth: Int
    public let imageHeight: Int
    public let points: [Point]
    public let lineLabel: String?
    public let trustedCalibration: Bool

    public init(
        mode: Mode,
        imageWidth: Int,
        imageHeight: Int,
        points: [Point],
        lineLabel: String? = nil
    ) throws {
        guard imageWidth > 0, imageHeight > 0 else {
            throw ValidationError.invalidImageSize
        }
        guard !points.isEmpty else {
            throw ValidationError.missingCourtTaps
        }
        for point in points {
            guard point.x.isFinite,
                  point.y.isFinite,
                  point.x >= 0,
                  point.y >= 0,
                  point.x <= Double(imageWidth),
                  point.y <= Double(imageHeight)
            else {
                throw ValidationError.courtTapOutsideFrame
            }
        }
        if mode == .twoLineTaps {
            guard points.count == 2, lineLabel != nil else {
                throw ValidationError.missingCalibrationAnchor
            }
        }
        self.mode = mode
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.points = points
        self.lineLabel = lineLabel
        self.trustedCalibration = false
    }
}
