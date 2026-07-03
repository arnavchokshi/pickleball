import Foundation

/// Top-down-styled mini court map dot, W3-LIVE-MLP surface 2.
///
/// HONESTY NOTE (read before touching normalizedX/normalizedY): v0 ships
/// WITHOUT live camera calibration -- there is no ARKit court plane and no
/// live tap-corner homography wired into the camera preview yet (manual
/// court taps exist as a data type in `PickleballCalibration` but are not
/// collected during live preview in this build). `CourtDotMapBuilder` is
/// therefore a **screen-space proxy**: it places each player's detected
/// foot-point at the same relative position it occupies in the camera
/// frame, normalized to [0, 1] x [0, 1]. This is NOT a real top-down court
/// projection -- a player standing at the back of the court and a player
/// standing at the net will not be spaced the way a true homography would
/// space them. It is an honest, cheap "where roughly is each player on
/// screen" map, and must be labeled as a proxy in any UI that renders it.
/// The real fix (tap-corner homography or ARKit court plane -> proper
/// image-to-court projection) is a documented follow-up.
public struct CourtDotMapPoint: Equatable, Sendable {
    public var trackID: Int
    public var normalizedX: Double
    public var normalizedY: Double
    public var confidence: Double
    public var role: String?

    public init(trackID: Int, normalizedX: Double, normalizedY: Double, confidence: Double, role: String? = nil) {
        self.trackID = trackID
        self.normalizedX = normalizedX
        self.normalizedY = normalizedY
        self.confidence = confidence
        self.role = role
    }
}

public enum CourtDotMapBuilder {
    /// `rotationDegrees` must be one of `0, 90, 180, 270` and should be the
    /// same `videoRotationAngleDegrees` already used to orient the camera
    /// preview (`CaptureOrientationPolicy.rotationAngleDegrees`), so the
    /// mini map's "up" matches what's on screen. `sourceWidth`/`sourceHeight`
    /// are the RAW (unrotated) pixel buffer dimensions the detector ran
    /// against, matching `CoreMLPersonDetector`'s `CVPixelBufferGetWidth/Height`
    /// convention.
    public static func build(
        detections: [OnDevicePersonDetection],
        sourceWidth: Double,
        sourceHeight: Double,
        rotationDegrees: Int = 0,
        maxPlayers: Int = 4
    ) -> [CourtDotMapPoint] {
        guard sourceWidth > 0, sourceHeight > 0 else {
            return []
        }

        return detections
            .sorted { $0.trackID < $1.trackID }
            .prefix(maxPlayers)
            .compactMap { detection -> CourtDotMapPoint? in
                guard detection.bboxXYWH.count == 4 else {
                    return nil
                }
                let footX = detection.bboxXYWH[0] + detection.bboxXYWH[2] / 2.0
                let footY = detection.bboxXYWH[1] + detection.bboxXYWH[3]
                let rotated = rotate(
                    x: footX,
                    y: footY,
                    width: sourceWidth,
                    height: sourceHeight,
                    degrees: rotationDegrees
                )
                return CourtDotMapPoint(
                    trackID: detection.trackID,
                    normalizedX: clamp01(rotated.x / rotated.width),
                    normalizedY: clamp01(rotated.y / rotated.height),
                    confidence: detection.confidence,
                    role: detection.role
                )
            }
    }

    /// Rotates a point clockwise by `degrees` (must be 0/90/180/270) within a
    /// `width x height` buffer, returning the point plus the resulting
    /// (possibly width/height-swapped) frame size. This mirrors the
    /// clockwise convention `AVCaptureConnection.videoRotationAngle` and
    /// `CaptureOrientationPolicy.rotationAngleDegrees` already use elsewhere
    /// in this codebase for `landscapeRight` == 90.
    static func rotate(
        x: Double,
        y: Double,
        width: Double,
        height: Double,
        degrees: Int
    ) -> (x: Double, y: Double, width: Double, height: Double) {
        switch ((degrees % 360) + 360) % 360 {
        case 90:
            return (height - y, x, height, width)
        case 180:
            return (width - x, height - y, width, height)
        case 270:
            return (y, width - x, height, width)
        default:
            return (x, y, width, height)
        }
    }

    private static func clamp01(_ value: Double) -> Double {
        guard value.isFinite else {
            return 0
        }
        return min(1, max(0, value))
    }
}
