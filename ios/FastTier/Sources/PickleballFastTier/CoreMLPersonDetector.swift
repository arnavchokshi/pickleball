import Foundation

#if canImport(CoreImage)
import CoreImage
#endif

#if canImport(CoreML)
import CoreML
#endif

#if canImport(CoreVideo)
import CoreVideo
#endif

public enum CoreMLPersonOutputFormat: Equatable, Sendable {
    case yolo11NMS
    case yolo26EndToEnd
}

public struct CoreMLPersonDetectorConfiguration: Equatable, Sendable {
    public var candidate: OnDevicePersonCandidate
    public var modelURL: URL
    public var outputFormat: CoreMLPersonOutputFormat
    public var inputWidth: Int
    public var inputHeight: Int
    public var maxTracks: Int
    public var minConfidence: Double
    public var iouThreshold: Double
    public var maxTrackAgeFrames: Int
    public var detectionIntervalFrames: Int

    public init(
        candidate: OnDevicePersonCandidate,
        modelURL: URL,
        outputFormat: CoreMLPersonOutputFormat,
        inputWidth: Int,
        inputHeight: Int,
        maxTracks: Int = 4,
        minConfidence: Double = 0.10,
        iouThreshold: Double = 0.30,
        maxTrackAgeFrames: Int = 8,
        detectionIntervalFrames: Int = 1
    ) {
        self.candidate = candidate
        self.modelURL = modelURL
        self.outputFormat = outputFormat
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.maxTracks = maxTracks
        self.minConfidence = minConfidence
        self.iouThreshold = iouThreshold
        self.maxTrackAgeFrames = maxTrackAgeFrames
        self.detectionIntervalFrames = max(1, detectionIntervalFrames)
    }
}

public enum CoreMLPersonDetectorError: Error, Equatable, Sendable {
    case coreMLUnavailable
    case coreVideoUnavailable
    case cannotCreatePixelBuffer
    case missingInputFeature
    case missingOutput(String)
    case unsupportedOutputShape(String)
}

public enum CoreMLPersonDetectionDecoder {
    public static func decodeYolo11NMS(
        coordinates: [[Double]],
        confidences: [[Double]],
        sourceWidth: Int,
        sourceHeight: Int,
        maxPlayers: Int,
        minConfidence: Double
    ) -> [OnDevicePersonObservation] {
        var observations: [OnDevicePersonObservation] = []
        for index in 0..<min(coordinates.count, confidences.count) {
            guard coordinates[index].count >= 4, let personConfidence = confidences[index].first else {
                continue
            }
            guard personConfidence >= minConfidence else {
                continue
            }
            let cx = coordinates[index][0]
            let cy = coordinates[index][1]
            let width = coordinates[index][2]
            let height = coordinates[index][3]
            let x = (cx - width / 2.0) * Double(sourceWidth)
            let y = (cy - height / 2.0) * Double(sourceHeight)
            observations.append(
                OnDevicePersonObservation(
                    bboxXYWH: [
                        clamp(x, lower: 0, upper: Double(sourceWidth)),
                        clamp(y, lower: 0, upper: Double(sourceHeight)),
                        clamp(width * Double(sourceWidth), lower: 0, upper: Double(sourceWidth)),
                        clamp(height * Double(sourceHeight), lower: 0, upper: Double(sourceHeight)),
                    ],
                    confidence: personConfidence,
                    source: "coreml_yolo11n_nms"
                )
            )
        }
        return topObservations(observations, maxPlayers: maxPlayers)
    }

    public static func decodeYolo26EndToEnd(
        rows: [[Double]],
        modelInputWidth: Int,
        modelInputHeight: Int,
        sourceWidth: Int,
        sourceHeight: Int,
        maxPlayers: Int,
        minConfidence: Double
    ) -> [OnDevicePersonObservation] {
        let scaleX = Double(sourceWidth) / Double(modelInputWidth)
        let scaleY = Double(sourceHeight) / Double(modelInputHeight)
        var observations: [OnDevicePersonObservation] = []
        for row in rows where row.count >= 6 {
            let confidence = row[4]
            let classID = Int(row[5].rounded())
            guard classID == 0, confidence >= minConfidence else {
                continue
            }
            let x1 = clamp(row[0] * scaleX, lower: 0, upper: Double(sourceWidth))
            let y1 = clamp(row[1] * scaleY, lower: 0, upper: Double(sourceHeight))
            let x2 = clamp(row[2] * scaleX, lower: 0, upper: Double(sourceWidth))
            let y2 = clamp(row[3] * scaleY, lower: 0, upper: Double(sourceHeight))
            let width = max(0, x2 - x1)
            let height = max(0, y2 - y1)
            guard width > 0, height > 0 else {
                continue
            }
            observations.append(
                OnDevicePersonObservation(
                    bboxXYWH: [x1, y1, width, height],
                    confidence: confidence,
                    source: "coreml_yolo26_end2end"
                )
            )
        }
        return topObservations(observations, maxPlayers: maxPlayers)
    }

    private static func topObservations(_ observations: [OnDevicePersonObservation], maxPlayers: Int) -> [OnDevicePersonObservation] {
        Array(
            observations
                .sorted {
                    if $0.confidence == $1.confidence {
                        return $0.bboxXYWH.lexicographicallyPrecedes($1.bboxXYWH)
                    }
                    return $0.confidence > $1.confidence
                }
                .prefix(maxPlayers)
        )
    }

    private static func clamp(_ value: Double, lower: Double, upper: Double) -> Double {
        min(max(value, lower), upper)
    }
}

#if canImport(CoreML) && canImport(CoreVideo) && canImport(CoreImage)
@available(iOS 15.0, macOS 12.0, *)
public final class CoreMLPersonDetector {
    public let candidate: OnDevicePersonCandidate
    public let modelLoadMs: Double
    public let mlpackageSizeMB: Double?
    private let configuration: CoreMLPersonDetectorConfiguration
    private let model: MLModel
    private let inputName: String
    private var linker: PersonTrackLinker
    private let ciContext = CIContext()

    public init(configuration: CoreMLPersonDetectorConfiguration) throws {
        let started = CFAbsoluteTimeGetCurrent()
        let modelConfiguration = MLModelConfiguration()
        if #available(iOS 16.0, macOS 13.0, *) {
            modelConfiguration.computeUnits = .cpuAndNeuralEngine
        } else {
            modelConfiguration.computeUnits = .all
        }
        let compiledURL = try Self.compiledModelURL(for: configuration.modelURL)
        let loadedModel = try MLModel(contentsOf: compiledURL, configuration: modelConfiguration)
        guard let inputName = loadedModel.modelDescription.inputDescriptionsByName.keys.first else {
            throw CoreMLPersonDetectorError.missingInputFeature
        }
        self.configuration = configuration
        self.candidate = configuration.candidate
        self.model = loadedModel
        self.inputName = inputName
        self.linker = PersonTrackLinker(
            iouThreshold: configuration.iouThreshold,
            maxTrackAgeFrames: configuration.maxTrackAgeFrames,
            maxTracks: configuration.maxTracks
        )
        self.modelLoadMs = max(0, (CFAbsoluteTimeGetCurrent() - started) * 1000.0)
        self.mlpackageSizeMB = Self.packageSizeMB(configuration.modelURL)
    }

    public func process(pixelBuffer: CVPixelBuffer, frameIndex: Int) throws -> [OnDevicePersonDetection] {
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let resized = try resizedPixelBuffer(pixelBuffer)
        var features: [String: MLFeatureValue] = [
            inputName: MLFeatureValue(pixelBuffer: resized),
        ]
        if model.modelDescription.inputDescriptionsByName["iouThreshold"] != nil {
            features["iouThreshold"] = MLFeatureValue(double: configuration.iouThreshold)
        }
        if model.modelDescription.inputDescriptionsByName["confidenceThreshold"] != nil {
            features["confidenceThreshold"] = MLFeatureValue(double: configuration.minConfidence)
        }
        let output = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: features))
        let observations = try observations(from: output, sourceWidth: sourceWidth, sourceHeight: sourceHeight)
        return linker.update(frameIndex: frameIndex, observations: observations)
    }

    private func observations(
        from output: MLFeatureProvider,
        sourceWidth: Int,
        sourceHeight: Int
    ) throws -> [OnDevicePersonObservation] {
        switch configuration.outputFormat {
        case .yolo11NMS:
            guard let coordinates = output.featureValue(for: "coordinates")?.multiArrayValue else {
                throw CoreMLPersonDetectorError.missingOutput("coordinates")
            }
            guard let confidence = output.featureValue(for: "confidence")?.multiArrayValue else {
                throw CoreMLPersonDetectorError.missingOutput("confidence")
            }
            return CoreMLPersonDetectionDecoder.decodeYolo11NMS(
                coordinates: rows(from: coordinates, columns: 4),
                confidences: rows(from: confidence, columns: 80),
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                maxPlayers: configuration.maxTracks,
                minConfidence: configuration.minConfidence
            )
        case .yolo26EndToEnd:
            let arrays = output.featureNames.compactMap { output.featureValue(for: $0)?.multiArrayValue }
            guard let array = arrays.first else {
                throw CoreMLPersonDetectorError.missingOutput("yolo26 end2end")
            }
            return CoreMLPersonDetectionDecoder.decodeYolo26EndToEnd(
                rows: rows(from: array, columns: 6),
                modelInputWidth: configuration.inputWidth,
                modelInputHeight: configuration.inputHeight,
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                maxPlayers: configuration.maxTracks,
                minConfidence: configuration.minConfidence
            )
        }
    }

    private func resizedPixelBuffer(_ pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        var output: CVPixelBuffer?
        let attrs: [String: Any] = [
            String(kCVPixelBufferCGImageCompatibilityKey): true,
            String(kCVPixelBufferCGBitmapContextCompatibilityKey): true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            configuration.inputWidth,
            configuration.inputHeight,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &output
        )
        guard status == kCVReturnSuccess, let output else {
            throw CoreMLPersonDetectorError.cannotCreatePixelBuffer
        }
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let image = CIImage(cvPixelBuffer: pixelBuffer).transformed(
            by: CGAffineTransform(
                scaleX: CGFloat(configuration.inputWidth) / CGFloat(sourceWidth),
                y: CGFloat(configuration.inputHeight) / CGFloat(sourceHeight)
            )
        )
        ciContext.render(image, to: output)
        return output
    }

    private static func compiledModelURL(for modelURL: URL) throws -> URL {
        if modelURL.pathExtension == "mlmodelc" {
            return modelURL
        }
        return try MLModel.compileModel(at: modelURL)
    }

    private static func packageSizeMB(_ url: URL) -> Double? {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return nil
        }
        if !isDirectory.boolValue {
            guard let bytes = (try? fileManager.attributesOfItem(atPath: url.path)[.size]) as? NSNumber else {
                return nil
            }
            return bytes.doubleValue / 1_000_000.0
        }
        guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return nil
        }
        var bytes = 0
        for case let itemURL as URL in enumerator {
            bytes += ((try? itemURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return Double(bytes) / 1_000_000.0
    }
}

private func rows(from multiArray: MLMultiArray, columns: Int) -> [[Double]] {
    guard columns > 0 else {
        return []
    }
    let count = multiArray.count / columns
    return (0..<count).map { rowIndex in
        (0..<columns).map { columnIndex in
            let flatIndex = rowIndex * columns + columnIndex
            return multiArray[flatIndex].doubleValue
        }
    }
}
#else
public final class CoreMLPersonDetector {
    public let candidate: OnDevicePersonCandidate
    public let modelLoadMs: Double = 0
    public let mlpackageSizeMB: Double? = nil

    public init(configuration: CoreMLPersonDetectorConfiguration) throws {
        self.candidate = configuration.candidate
        throw CoreMLPersonDetectorError.coreMLUnavailable
    }
}
#endif
