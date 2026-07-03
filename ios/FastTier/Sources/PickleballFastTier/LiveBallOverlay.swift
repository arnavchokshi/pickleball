import Foundation

public enum LiveBallContactMarkerSource: String, Codable, Equatable, Sendable {
    case kinematicInflectionCandidate = "kinematic_inflection_candidate"
}

public struct LiveBallTrailPoint: Equatable, Sendable {
    public var frameIndex: Int
    public var normalizedX: Double
    public var normalizedY: Double
    public var confidence: Double
    public var ageFrames: Int
    public var opacity: Double
    public var radius: Double

    public init(
        frameIndex: Int,
        normalizedX: Double,
        normalizedY: Double,
        confidence: Double,
        ageFrames: Int,
        opacity: Double,
        radius: Double
    ) {
        self.frameIndex = frameIndex
        self.normalizedX = normalizedX
        self.normalizedY = normalizedY
        self.confidence = confidence
        self.ageFrames = ageFrames
        self.opacity = opacity
        self.radius = radius
    }
}

public struct LiveBallContactMarker: Equatable, Sendable {
    public var frameIndex: Int
    public var normalizedX: Double
    public var normalizedY: Double
    public var confidence: Double
    public var ageFrames: Int
    public var opacity: Double
    public var radius: Double
    public var source: LiveBallContactMarkerSource

    public init(
        frameIndex: Int,
        normalizedX: Double,
        normalizedY: Double,
        confidence: Double,
        ageFrames: Int,
        opacity: Double,
        radius: Double,
        source: LiveBallContactMarkerSource
    ) {
        self.frameIndex = frameIndex
        self.normalizedX = normalizedX
        self.normalizedY = normalizedY
        self.confidence = confidence
        self.ageFrames = ageFrames
        self.opacity = opacity
        self.radius = radius
        self.source = source
    }
}

public struct LiveBallOverlayState: Equatable, Sendable {
    public var trailPoints: [LiveBallTrailPoint]
    public var contactMarkers: [LiveBallContactMarker]

    public init(trailPoints: [LiveBallTrailPoint] = [], contactMarkers: [LiveBallContactMarker] = []) {
        self.trailPoints = trailPoints
        self.contactMarkers = contactMarkers
    }

    public static let empty = LiveBallOverlayState()
}

public struct LiveBallOverlayTracker: Equatable, Sendable {
    private struct Sample: Equatable, Sendable {
        var frameIndex: Int
        var x: Double
        var y: Double
        var confidence: Double
    }

    private var samples: [Sample] = []
    private var contactMarkers: [LiveBallContactMarker] = []
    private let maxTrailAgeFrames: Int
    private let maxContactAgeFrames: Int
    private let maxContactSampleGapFrames: Int

    public init(maxTrailAgeFrames: Int = 24, maxContactAgeFrames: Int = 18, maxContactSampleGapFrames: Int = 1) {
        self.maxTrailAgeFrames = max(1, maxTrailAgeFrames)
        self.maxContactAgeFrames = max(1, maxContactAgeFrames)
        self.maxContactSampleGapFrames = max(1, maxContactSampleGapFrames)
    }

    public mutating func reset() {
        samples = []
        contactMarkers = []
    }

    /// Consumes only `LiveBallIndicatorAvailability.tracking` states. This is
    /// the honesty gate: `.comingSoon`, `.lowConfidence`, missing position, or
    /// non-finite coordinates never create new visible trail/contact evidence.
    public mutating func update(frameIndex: Int, ballState: LiveBallIndicatorState) -> LiveBallOverlayState {
        prune(currentFrameIndex: frameIndex)

        if ballState.availability == .tracking,
           let x = ballState.normalizedX,
           let y = ballState.normalizedY,
           x.isFinite,
           y.isFinite {
            samples.append(
                Sample(
                    frameIndex: frameIndex,
                    x: clamp01(x),
                    y: clamp01(y),
                    confidence: clamp01(ballState.confidence ?? 0)
                )
            )
            prune(currentFrameIndex: frameIndex)
            appendContactMarkerIfNeeded(currentFrameIndex: frameIndex)
        }

        return state(currentFrameIndex: frameIndex)
    }

    private mutating func appendContactMarkerIfNeeded(currentFrameIndex _: Int) {
        guard samples.count >= 3 else {
            return
        }
        let a = samples[samples.count - 3]
        let b = samples[samples.count - 2]
        let c = samples[samples.count - 1]
        guard b.frameIndex > a.frameIndex,
              c.frameIndex > b.frameIndex,
              b.frameIndex - a.frameIndex <= maxContactSampleGapFrames,
              c.frameIndex - b.frameIndex <= maxContactSampleGapFrames else {
            return
        }
        guard !contactMarkers.contains(where: { abs($0.frameIndex - b.frameIndex) <= 1 }) else {
            return
        }

        let incoming = (x: b.x - a.x, y: b.y - a.y)
        let outgoing = (x: c.x - b.x, y: c.y - b.y)
        let incomingSpeed = hypot(incoming.x, incoming.y)
        let outgoingSpeed = hypot(outgoing.x, outgoing.y)
        guard incomingSpeed >= 0.025, outgoingSpeed >= 0.025 else {
            return
        }

        let cosine = (incoming.x * outgoing.x + incoming.y * outgoing.y) / (incomingSpeed * outgoingSpeed)
        guard cosine <= 0.20 else {
            return
        }

        let confidence = (a.confidence + b.confidence + c.confidence) / 3.0
        contactMarkers.append(
            LiveBallContactMarker(
                frameIndex: b.frameIndex,
                normalizedX: b.x,
                normalizedY: b.y,
                confidence: confidence,
                ageFrames: 0,
                opacity: 1.0,
                radius: 0.050 + confidence * 0.020,
                source: .kinematicInflectionCandidate
            )
        )
    }

    private mutating func prune(currentFrameIndex: Int) {
        samples.removeAll { currentFrameIndex - $0.frameIndex > maxTrailAgeFrames }
        contactMarkers.removeAll { currentFrameIndex - $0.frameIndex > maxContactAgeFrames }
    }

    private func state(currentFrameIndex: Int) -> LiveBallOverlayState {
        LiveBallOverlayState(
            trailPoints: samples.map { sample in
                let ageFrames = max(0, currentFrameIndex - sample.frameIndex)
                let opacity = max(0.12, 1.0 - Double(ageFrames) / Double(maxTrailAgeFrames))
                return LiveBallTrailPoint(
                    frameIndex: sample.frameIndex,
                    normalizedX: sample.x,
                    normalizedY: sample.y,
                    confidence: sample.confidence,
                    ageFrames: ageFrames,
                    opacity: opacity,
                    radius: 0.010 + sample.confidence * 0.008
                )
            },
            contactMarkers: contactMarkers.map { marker in
                let ageFrames = max(0, currentFrameIndex - marker.frameIndex)
                let opacity = max(0.0, 1.0 - Double(ageFrames) / Double(maxContactAgeFrames))
                return LiveBallContactMarker(
                    frameIndex: marker.frameIndex,
                    normalizedX: marker.normalizedX,
                    normalizedY: marker.normalizedY,
                    confidence: marker.confidence,
                    ageFrames: ageFrames,
                    opacity: opacity,
                    radius: marker.radius,
                    source: marker.source
                )
            }
        )
    }

    private static func clamp01(_ value: Double) -> Double {
        guard value.isFinite else {
            return 0
        }
        return min(1, max(0, value))
    }

    private func clamp01(_ value: Double) -> Double {
        Self.clamp01(value)
    }
}

public struct RenderedLiveBallTrailPoint: Equatable, Sendable {
    public var point: LiveBallTrailPoint
    public var centerX: Double
    public var centerY: Double
    public var radius: Double

    public init(point: LiveBallTrailPoint, centerX: Double, centerY: Double, radius: Double) {
        self.point = point
        self.centerX = centerX
        self.centerY = centerY
        self.radius = radius
    }
}

public struct RenderedLiveBallContactMarker: Equatable, Sendable {
    public var marker: LiveBallContactMarker
    public var centerX: Double
    public var centerY: Double
    public var radius: Double

    public init(marker: LiveBallContactMarker, centerX: Double, centerY: Double, radius: Double) {
        self.marker = marker
        self.centerX = centerX
        self.centerY = centerY
        self.radius = radius
    }
}

public struct RenderedLiveBallTrailSegment: Equatable, Sendable {
    public var points: [RenderedLiveBallTrailPoint]

    public init(points: [RenderedLiveBallTrailPoint]) {
        self.points = points
    }
}

public struct RenderedLiveBallOverlayState: Equatable, Sendable {
    public var trailPoints: [RenderedLiveBallTrailPoint]
    public var trailSegments: [RenderedLiveBallTrailSegment]
    public var contactMarkers: [RenderedLiveBallContactMarker]

    public init(
        trailPoints: [RenderedLiveBallTrailPoint] = [],
        trailSegments: [RenderedLiveBallTrailSegment] = [],
        contactMarkers: [RenderedLiveBallContactMarker] = []
    ) {
        self.trailPoints = trailPoints
        self.trailSegments = trailSegments
        self.contactMarkers = contactMarkers
    }
}

public enum LiveBallOverlayLayout {
    public static func layout(
        overlay: LiveBallOverlayState,
        viewportWidth: Double,
        viewportHeight: Double,
        videoAspectRatio: Double
    ) -> RenderedLiveBallOverlayState {
        guard viewportWidth > 0,
              viewportHeight > 0,
              videoAspectRatio.isFinite,
              videoAspectRatio > 0 else {
            return RenderedLiveBallOverlayState()
        }

        let frame = aspectFillFrame(
            viewportWidth: viewportWidth,
            viewportHeight: viewportHeight,
            videoAspectRatio: videoAspectRatio
        )
        let radiusScale = min(frame.width, frame.height)
        let renderedTrailPoints = overlay.trailPoints.map { point in
            RenderedLiveBallTrailPoint(
                point: point,
                centerX: frame.offsetX + point.normalizedX * frame.width,
                centerY: frame.offsetY + point.normalizedY * frame.height,
                radius: point.radius * radiusScale
            )
        }
        return RenderedLiveBallOverlayState(
            trailPoints: renderedTrailPoints,
            trailSegments: segmentTrailPoints(renderedTrailPoints),
            contactMarkers: overlay.contactMarkers.map { marker in
                RenderedLiveBallContactMarker(
                    marker: marker,
                    centerX: frame.offsetX + marker.normalizedX * frame.width,
                    centerY: frame.offsetY + marker.normalizedY * frame.height,
                    radius: marker.radius * radiusScale
                )
            }
        )
    }

    private static func segmentTrailPoints(_ points: [RenderedLiveBallTrailPoint]) -> [RenderedLiveBallTrailSegment] {
        guard !points.isEmpty else {
            return []
        }

        var segments: [RenderedLiveBallTrailSegment] = []
        var currentSegment: [RenderedLiveBallTrailPoint] = []
        for point in points {
            if let previous = currentSegment.last,
               point.point.frameIndex - previous.point.frameIndex > 1 {
                segments.append(RenderedLiveBallTrailSegment(points: currentSegment))
                currentSegment = []
            }
            currentSegment.append(point)
        }
        if !currentSegment.isEmpty {
            segments.append(RenderedLiveBallTrailSegment(points: currentSegment))
        }
        return segments
    }

    private static func aspectFillFrame(
        viewportWidth: Double,
        viewportHeight: Double,
        videoAspectRatio: Double
    ) -> (width: Double, height: Double, offsetX: Double, offsetY: Double) {
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
        return (
            renderedWidth,
            renderedHeight,
            (viewportWidth - renderedWidth) / 2.0,
            (viewportHeight - renderedHeight) / 2.0
        )
    }
}
