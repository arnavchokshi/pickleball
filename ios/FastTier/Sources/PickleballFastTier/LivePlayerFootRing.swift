import Foundation

/// The first PB Vision-style live overlay primitive: a preview-only ellipse
/// centered at the detected player foot point. This is intentionally still a
/// screen-space proxy until live court homography/ARKit court lock is wired in.
public enum LivePlayerFootRingSource: String, Codable, Equatable, Sendable {
    case screenSpaceProxy = "screen_space_proxy"
}

public struct LivePlayerFootRing: Equatable, Sendable {
    public var trackID: Int
    public var normalizedCenterX: Double
    public var normalizedCenterY: Double
    public var confidence: Double
    public var stalenessFrames: Int
    public var colorIndex: Int
    public var normalizedWidth: Double
    public var normalizedHeight: Double
    public var strokeOpacity: Double
    public var fillOpacity: Double
    public var source: LivePlayerFootRingSource

    public var isStale: Bool {
        stalenessFrames > 0
    }

    public init(
        trackID: Int,
        normalizedCenterX: Double,
        normalizedCenterY: Double,
        confidence: Double,
        stalenessFrames: Int,
        colorIndex: Int,
        normalizedWidth: Double,
        normalizedHeight: Double,
        strokeOpacity: Double,
        fillOpacity: Double,
        source: LivePlayerFootRingSource
    ) {
        self.trackID = trackID
        self.normalizedCenterX = normalizedCenterX
        self.normalizedCenterY = normalizedCenterY
        self.confidence = confidence
        self.stalenessFrames = stalenessFrames
        self.colorIndex = colorIndex
        self.normalizedWidth = normalizedWidth
        self.normalizedHeight = normalizedHeight
        self.strokeOpacity = strokeOpacity
        self.fillOpacity = fillOpacity
        self.source = source
    }
}

public enum LivePlayerFootRingBuilder {
    public static let paletteSize = 4
    private static let maxVisibleStalenessFrames = 12
    private struct StalenessStyle {
        var frames: Int
        var fade: Double
    }

    public static func build(
        points: [CourtDotMapPoint],
        frameIndex: Int,
        lastDetectionFrameIndex: Int?,
        maxPlayers: Int = 4
    ) -> [LivePlayerFootRing] {
        let style = stalenessStyle(frameIndex: frameIndex, lastDetectionFrameIndex: lastDetectionFrameIndex)

        return points
            .sorted { $0.trackID < $1.trackID }
            .prefix(maxPlayers)
            .map { point in
                let confidence = clamp01(point.confidence)
                let y = clamp01(point.normalizedY)
                let normalizedWidth = clamp(0.075 + y * 0.075, minValue: 0.07, maxValue: 0.16)
                return makeRing(
                    trackID: point.trackID,
                    normalizedCenterX: clamp01(point.normalizedX),
                    normalizedCenterY: y,
                    confidence: confidence,
                    normalizedWidth: normalizedWidth,
                    style: style
                )
            }
    }

    public static func build(
        detections: [OnDevicePersonDetection],
        sourceWidth: Double,
        sourceHeight: Double,
        rotationDegrees: Int = 0,
        frameIndex: Int,
        lastDetectionFrameIndex: Int?,
        maxPlayers: Int = 4
    ) -> [LivePlayerFootRing] {
        guard sourceWidth.isFinite,
              sourceHeight.isFinite,
              sourceWidth > 0,
              sourceHeight > 0 else {
            return []
        }
        let style = stalenessStyle(frameIndex: frameIndex, lastDetectionFrameIndex: lastDetectionFrameIndex)

        return detections
            .sorted { $0.trackID < $1.trackID }
            .prefix(maxPlayers)
            .compactMap { detection -> LivePlayerFootRing? in
                guard detection.bboxXYWH.count == 4,
                      detection.bboxXYWH.allSatisfy(\.isFinite) else {
                    return nil
                }
                let boxX = detection.bboxXYWH[0]
                let boxY = detection.bboxXYWH[1]
                let boxWidth = detection.bboxXYWH[2]
                let boxHeight = detection.bboxXYWH[3]
                guard boxWidth > 0, boxHeight > 0 else {
                    return nil
                }

                let footX = boxX + boxWidth / 2.0
                let footY = boxY + boxHeight
                let rotatedFoot = CourtDotMapBuilder.rotate(
                    x: footX,
                    y: footY,
                    width: sourceWidth,
                    height: sourceHeight,
                    degrees: rotationDegrees
                )
                let normalizedX = clamp01(rotatedFoot.x / rotatedFoot.width)
                let normalizedY = clamp01(rotatedFoot.y / rotatedFoot.height)
                let perspectiveWidth = 0.075 + normalizedY * 0.075
                let footprintWidth = rotatedBoxWidthRatio(
                    boxX: boxX,
                    boxY: boxY,
                    boxWidth: boxWidth,
                    boxHeight: boxHeight,
                    sourceWidth: sourceWidth,
                    sourceHeight: sourceHeight,
                    rotationDegrees: rotationDegrees
                )
                let normalizedWidth = clamp(
                    max(perspectiveWidth, footprintWidth * 1.15),
                    minValue: 0.07,
                    maxValue: 0.22
                )

                return makeRing(
                    trackID: detection.trackID,
                    normalizedCenterX: normalizedX,
                    normalizedCenterY: normalizedY,
                    confidence: clamp01(detection.confidence),
                    normalizedWidth: normalizedWidth,
                    style: style
                )
            }
    }

    private static func makeRing(
        trackID: Int,
        normalizedCenterX: Double,
        normalizedCenterY: Double,
        confidence: Double,
        normalizedWidth: Double,
        style: StalenessStyle
    ) -> LivePlayerFootRing {
        let normalizedHeight = normalizedWidth * 0.34
        let confidenceOpacity = 0.35 + confidence * 0.65
        let strokeOpacity = clamp(confidenceOpacity * style.fade, minValue: 0.30, maxValue: 1.0)

        return LivePlayerFootRing(
            trackID: trackID,
            normalizedCenterX: normalizedCenterX,
            normalizedCenterY: normalizedCenterY,
            confidence: confidence,
            stalenessFrames: style.frames,
            colorIndex: abs(trackID) % paletteSize,
            normalizedWidth: normalizedWidth,
            normalizedHeight: normalizedHeight,
            strokeOpacity: strokeOpacity,
            fillOpacity: strokeOpacity * 0.18,
            source: .screenSpaceProxy
        )
    }

    private static func stalenessStyle(frameIndex: Int, lastDetectionFrameIndex: Int?) -> StalenessStyle {
        let detectionFrameIndex = lastDetectionFrameIndex ?? frameIndex
        let stalenessFrames = max(0, frameIndex - detectionFrameIndex)
        let stalenessRatio = min(1.0, Double(stalenessFrames) / Double(maxVisibleStalenessFrames))
        let staleFade = max(0.55, 1.0 - stalenessRatio * 0.45)
        return StalenessStyle(frames: stalenessFrames, fade: staleFade)
    }

    private static func rotatedBoxWidthRatio(
        boxX: Double,
        boxY: Double,
        boxWidth: Double,
        boxHeight: Double,
        sourceWidth: Double,
        sourceHeight: Double,
        rotationDegrees: Int
    ) -> Double {
        let corners = [
            (x: boxX, y: boxY),
            (x: boxX + boxWidth, y: boxY),
            (x: boxX, y: boxY + boxHeight),
            (x: boxX + boxWidth, y: boxY + boxHeight),
        ].map { corner in
            CourtDotMapBuilder.rotate(
                x: corner.x,
                y: corner.y,
                width: sourceWidth,
                height: sourceHeight,
                degrees: rotationDegrees
            )
        }
        guard let renderedWidth = corners.first?.width,
              renderedWidth > 0 else {
            return 0
        }
        let minX = corners.map(\.x).min() ?? 0
        let maxX = corners.map(\.x).max() ?? 0
        return clamp((maxX - minX) / renderedWidth, minValue: 0, maxValue: 1)
    }

    private static func clamp01(_ value: Double) -> Double {
        guard value.isFinite else {
            return 0
        }
        return clamp(value, minValue: 0, maxValue: 1)
    }

    private static func clamp(_ value: Double, minValue: Double, maxValue: Double) -> Double {
        min(maxValue, max(minValue, value))
    }
}

public struct RenderedLivePlayerFootRing: Equatable, Sendable {
    public var ring: LivePlayerFootRing
    public var centerX: Double
    public var centerY: Double
    public var width: Double
    public var height: Double

    public init(ring: LivePlayerFootRing, centerX: Double, centerY: Double, width: Double, height: Double) {
        self.ring = ring
        self.centerX = centerX
        self.centerY = centerY
        self.width = width
        self.height = height
    }
}

public enum LivePlayerFootRingLayout {
    public static func layout(
        rings: [LivePlayerFootRing],
        viewportWidth: Double,
        viewportHeight: Double,
        videoAspectRatio: Double
    ) -> [RenderedLivePlayerFootRing] {
        guard viewportWidth > 0,
              viewportHeight > 0,
              videoAspectRatio.isFinite,
              videoAspectRatio > 0 else {
            return []
        }

        let viewportAspectRatio = viewportWidth / viewportHeight
        let renderedWidth: Double
        let renderedHeight: Double
        if viewportAspectRatio > videoAspectRatio {
            renderedWidth = viewportWidth
            renderedHeight = viewportWidth / videoAspectRatio
        } else {
            renderedHeight = viewportHeight
            renderedWidth = viewportHeight * videoAspectRatio
        }
        let offsetX = (viewportWidth - renderedWidth) / 2.0
        let offsetY = (viewportHeight - renderedHeight) / 2.0

        return rings.map { ring in
            RenderedLivePlayerFootRing(
                ring: ring,
                centerX: offsetX + ring.normalizedCenterX * renderedWidth,
                centerY: offsetY + ring.normalizedCenterY * renderedHeight,
                width: ring.normalizedWidth * renderedWidth,
                height: ring.normalizedHeight * renderedHeight
            )
        }
    }
}
