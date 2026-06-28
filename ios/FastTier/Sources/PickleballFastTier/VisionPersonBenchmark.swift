import Foundation

#if canImport(CoreVideo)
import CoreVideo
#endif

#if canImport(ImageIO)
import ImageIO
#endif

#if canImport(Vision)
import Vision
#endif

public func normalizedVisionRectToPixelXYWH(
    x: Double,
    y: Double,
    width: Double,
    height: Double,
    imageWidth: Int,
    imageHeight: Int
) -> [Double] {
    let pixelX = x * Double(imageWidth)
    let pixelY = (1.0 - y - height) * Double(imageHeight)
    return [
        pixelX,
        pixelY,
        width * Double(imageWidth),
        height * Double(imageHeight),
    ]
}

public enum VisionPersonBenchmarkError: Error, Equatable, Sendable {
    case visionUnavailable
    case coreVideoUnavailable
}

public struct VisionPersonBenchmarkConfiguration: Equatable, Sendable {
    public var candidate: OnDevicePersonCandidate
    public var maxTracks: Int
    public var iouThreshold: Double
    public var maxTrackAgeFrames: Int

    public init(
        candidate: OnDevicePersonCandidate = .visionHumanRectanglesIouV1,
        maxTracks: Int = 4,
        iouThreshold: Double = 0.3,
        maxTrackAgeFrames: Int = 8
    ) {
        self.candidate = candidate
        self.maxTracks = maxTracks
        self.iouThreshold = iouThreshold
        self.maxTrackAgeFrames = maxTrackAgeFrames
    }
}

#if canImport(Vision) && canImport(CoreVideo) && canImport(ImageIO)
@available(iOS 13.0, macOS 10.15, *)
public final class VisionHumanRectanglePersonProcessor {
    private var linker: PersonTrackLinker
    private let orientation: CGImagePropertyOrientation

    public init(
        configuration: VisionPersonBenchmarkConfiguration = VisionPersonBenchmarkConfiguration(),
        orientation: CGImagePropertyOrientation = .up
    ) {
        self.linker = PersonTrackLinker(
            iouThreshold: configuration.iouThreshold,
            maxTrackAgeFrames: configuration.maxTrackAgeFrames,
            maxTracks: configuration.maxTracks
        )
        self.orientation = orientation
    }

    public func process(pixelBuffer: CVPixelBuffer, frameIndex: Int) throws -> [OnDevicePersonDetection] {
        let request = VNDetectHumanRectanglesRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation)
        try handler.perform([request])
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let observations = (request.results ?? []).map { detectedObject -> OnDevicePersonObservation in
            let rect = detectedObject.boundingBox
            return OnDevicePersonObservation(
                bboxXYWH: normalizedVisionRectToPixelXYWH(
                    x: rect.origin.x,
                    y: rect.origin.y,
                    width: rect.size.width,
                    height: rect.size.height,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight
                ),
                confidence: Double(detectedObject.confidence),
                source: "vision_human_rectangles"
            )
        }
        return linker.update(frameIndex: frameIndex, observations: observations)
    }
}
#else
public final class VisionHumanRectanglePersonProcessor {
    public init(configuration _: VisionPersonBenchmarkConfiguration = VisionPersonBenchmarkConfiguration()) {}

    public func process(pixelBuffer _: Any, frameIndex _: Int) throws -> [OnDevicePersonDetection] {
        throw VisionPersonBenchmarkError.visionUnavailable
    }
}
#endif
