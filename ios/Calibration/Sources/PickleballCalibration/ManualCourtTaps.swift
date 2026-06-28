import Foundation

public struct ManualCourtTaps: Codable, Equatable, Sendable {
    public var imagePoints: [[Double]]

    public init(imagePoints: [[Double]]) {
        self.imagePoints = imagePoints
    }

    public enum ValidationError: Error, Equatable, Sendable {
        case expectedFourPoints
        case malformedPoint
        case nonFinitePoint
        case pointOutsideImage
        case duplicatePoint
        case degenerateQuadrilateral
    }

    public func orderedFourCorners(imageSize: ImageSize) throws -> ManualCourtTaps {
        guard imagePoints.count == 4 else {
            throw ValidationError.expectedFourPoints
        }
        guard imageSize.isPlausible else {
            throw ValidationError.pointOutsideImage
        }

        let points = try imagePoints.map { rawPoint -> ImagePoint in
            guard rawPoint.count == 2 else {
                throw ValidationError.malformedPoint
            }
            let point = ImagePoint(x: rawPoint[0], y: rawPoint[1])
            guard point.x.isFinite, point.y.isFinite else {
                throw ValidationError.nonFinitePoint
            }
            guard point.x >= 0,
                  point.x <= imageSize.width,
                  point.y >= 0,
                  point.y <= imageSize.height
            else {
                throw ValidationError.pointOutsideImage
            }
            return point
        }

        try validateMinimumPointDistance(points)
        let ordered = orderClockwiseStartingTopLeft(points)
        try validateConvexQuadrilateral(ordered, imageSize: imageSize)

        return ManualCourtTaps(imagePoints: ordered.map { [$0.x, $0.y] })
    }
}

private struct ImagePoint: Equatable {
    var x: Double
    var y: Double
}

private func validateMinimumPointDistance(_ points: [ImagePoint]) throws {
    for index in points.indices {
        for otherIndex in points.indices where otherIndex > index {
            let dx = points[index].x - points[otherIndex].x
            let dy = points[index].y - points[otherIndex].y
            if sqrt((dx * dx) + (dy * dy)) < 2.0 {
                throw ManualCourtTaps.ValidationError.duplicatePoint
            }
        }
    }
}

private func orderClockwiseStartingTopLeft(_ points: [ImagePoint]) -> [ImagePoint] {
    let centerX = points.reduce(0) { $0 + $1.x } / Double(points.count)
    let centerY = points.reduce(0) { $0 + $1.y } / Double(points.count)

    let clockwise = points.sorted {
        atan2($0.y - centerY, $0.x - centerX) < atan2($1.y - centerY, $1.x - centerX)
    }

    guard let startIndex = clockwise.indices.min(by: {
        (clockwise[$0].x + clockwise[$0].y) < (clockwise[$1].x + clockwise[$1].y)
    }) else {
        return clockwise
    }

    return Array(clockwise[startIndex...]) + Array(clockwise[..<startIndex])
}

private func validateConvexQuadrilateral(_ points: [ImagePoint], imageSize: ImageSize) throws {
    let area = abs(shoelaceArea(points))
    let minimumArea = max(100.0, imageSize.width * imageSize.height * 0.005)
    guard area >= minimumArea else {
        throw ManualCourtTaps.ValidationError.degenerateQuadrilateral
    }

    let crossProducts = points.indices.map { index in
        let a = points[index]
        let b = points[(index + 1) % points.count]
        let c = points[(index + 2) % points.count]
        let ab = ImagePoint(x: b.x - a.x, y: b.y - a.y)
        let bc = ImagePoint(x: c.x - b.x, y: c.y - b.y)
        return (ab.x * bc.y) - (ab.y * bc.x)
    }

    let hasPositive = crossProducts.contains { $0 > 0.0001 }
    let hasNegative = crossProducts.contains { $0 < -0.0001 }
    guard hasPositive != hasNegative else {
        throw ManualCourtTaps.ValidationError.degenerateQuadrilateral
    }
}

private func shoelaceArea(_ points: [ImagePoint]) -> Double {
    points.indices.reduce(0) { partial, index in
        let current = points[index]
        let next = points[(index + 1) % points.count]
        return partial + ((current.x * next.y) - (next.x * current.y))
    } / 2.0
}
