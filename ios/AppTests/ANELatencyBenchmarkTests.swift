import XCTest
import CoreML
import CoreVideo

/// Real iPhone ANE/CPU latency for the converted live-tier CoreML models.
///
/// This settles the Mac-proxy red flag in
/// runs/coreml_prep_20260702T023932Z/LATENCY_TABLE.md (a 9.5fps live-loop
/// projection measured on macOS CPU_ONLY/CPU_AND_NE via
/// coremltools.models.CompiledMLModel.predict, NOT the physical iPhone ANE).
///
/// Models are pushed ahead of time to the app's Documents/benchmark_models
/// directory via `devicectl device copy to` (see run dir for the exact
/// command) rather than bundled as build resources, since this is a
/// host-app unit test (PickleballAppTests) that shares the Pickleball app's
/// sandbox/Documents container -- no separate bundling/signing surface is
/// needed.
final class ANELatencyBenchmarkTests: XCTestCase {
    private struct BenchmarkResult: Codable {
        var model: String
        var computeUnits: String
        var iterations: Int
        var meanMs: Double
        var p90Ms: Double
        var minMs: Double
        var maxMs: Double
    }

    private struct ModelSpec {
        var fileName: String
        var displayName: String
    }

    func testMeasureOnDeviceLatencyForConvertedModels() throws {
        let documentsURL = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let modelsRoot = documentsURL.appendingPathComponent("benchmark_models", isDirectory: true)

        let specs = [
            ModelSpec(fileName: "yolo26n_640.mlmodelc", displayName: "yolo26n_player_detector_640"),
            ModelSpec(fileName: "yolo26n_960.mlmodelc", displayName: "yolo26n_player_detector_960"),
            ModelSpec(fileName: "ball_student_288x512.mlmodelc", displayName: "ball_student_288x512"),
        ]

        let computeUnitCases: [(name: String, units: MLComputeUnits)] = [
            ("all", .all),
            ("cpuOnly", .cpuOnly),
        ]

        var results: [BenchmarkResult] = []
        var skipped: [String] = []

        for spec in specs {
            let modelURL = modelsRoot.appendingPathComponent(spec.fileName, isDirectory: true)
            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                skipped.append("\(spec.displayName): missing at \(modelURL.path)")
                continue
            }

            for computeCase in computeUnitCases {
                let config = MLModelConfiguration()
                config.computeUnits = computeCase.units

                let model: MLModel
                do {
                    model = try MLModel(contentsOf: modelURL, configuration: config)
                } catch {
                    skipped.append("\(spec.displayName)/\(computeCase.name): load failed: \(error)")
                    continue
                }

                let input: MLFeatureProvider
                do {
                    input = try Self.makeZeroInput(for: model)
                } catch {
                    skipped.append("\(spec.displayName)/\(computeCase.name): input build failed: \(error)")
                    continue
                }

                // Warm up (excludes one-time graph compilation / ANE program
                // load from the timed loop).
                _ = try? model.prediction(from: input)

                let iterations = 50
                var samplesMs: [Double] = []
                samplesMs.reserveCapacity(iterations)
                for _ in 0..<iterations {
                    let start = DispatchTime.now()
                    _ = try model.prediction(from: input)
                    let end = DispatchTime.now()
                    let ms = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
                    samplesMs.append(ms)
                }

                let sorted = samplesMs.sorted()
                let mean = sorted.reduce(0, +) / Double(sorted.count)
                let p90Index = min(sorted.count - 1, Int(Double(sorted.count) * 0.9))
                let result = BenchmarkResult(
                    model: spec.displayName,
                    computeUnits: computeCase.name,
                    iterations: iterations,
                    meanMs: mean,
                    p90Ms: sorted[p90Index],
                    minMs: sorted.first ?? 0,
                    maxMs: sorted.last ?? 0
                )
                results.append(result)
                print(
                    "ANE_BENCHMARK_ROW model=\(result.model) computeUnits=\(result.computeUnits) "
                        + "n=\(result.iterations) meanMs=\(String(format: "%.3f", result.meanMs)) "
                        + "p90Ms=\(String(format: "%.3f", result.p90Ms)) minMs=\(String(format: "%.3f", result.minMs)) "
                        + "maxMs=\(String(format: "%.3f", result.maxMs))"
                )
            }
        }

        for message in skipped {
            print("ANE_BENCHMARK_SKIPPED \(message)")
        }

        let outputURL = documentsURL.appendingPathComponent("benchmark_results.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let payload: [String: AnyEncodable] = [
            "results": AnyEncodable(results),
            "skipped": AnyEncodable(skipped),
            "device_model": AnyEncodable(Self.deviceModelIdentifier()),
        ]
        let data = try encoder.encode(payload)
        try data.write(to: outputURL, options: .atomic)
        print("ANE_BENCHMARK_RESULTS_PATH \(outputURL.path)")

        XCTAssertFalse(results.isEmpty, "No benchmark rows were produced; see ANE_BENCHMARK_SKIPPED lines above")
    }

    private static func makeZeroInput(for model: MLModel) throws -> MLFeatureProvider {
        var values: [String: MLFeatureValue] = [:]
        for (name, description) in model.modelDescription.inputDescriptionsByName {
            switch description.type {
            case .image:
                guard let constraint = description.imageConstraint else {
                    throw BenchmarkInputError.unsupportedInput(name)
                }
                let pixelBuffer = try Self.makeZeroPixelBuffer(
                    width: constraint.pixelsWide,
                    height: constraint.pixelsHigh,
                    pixelFormat: constraint.pixelFormatType
                )
                values[name] = MLFeatureValue(pixelBuffer: pixelBuffer)
            case .multiArray:
                guard let constraint = description.multiArrayConstraint else {
                    throw BenchmarkInputError.unsupportedInput(name)
                }
                let shape = constraint.shape
                let array = try MLMultiArray(shape: shape, dataType: constraint.dataType)
                let count = array.count
                switch constraint.dataType {
                case .float32, .float64, .float16:
                    for i in 0..<count {
                        array[i] = 0
                    }
                default:
                    for i in 0..<count {
                        array[i] = 0
                    }
                }
                values[name] = MLFeatureValue(multiArray: array)
            default:
                throw BenchmarkInputError.unsupportedInput(name)
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: values)
    }

    private static func makeZeroPixelBuffer(
        width: Int,
        height: Int,
        pixelFormat: OSType
    ) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            pixelFormat,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw BenchmarkInputError.pixelBufferCreationFailed(status)
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            let byteCount = CVPixelBufferGetDataSize(buffer)
            memset(base, 0, byteCount)
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }

    private static func deviceModelIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafePointer(to: &systemInfo.machine) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: 1) { machinePointer in
                String(validatingUTF8: machinePointer) ?? "unknown"
            }
        }
    }
}

private enum BenchmarkInputError: Error {
    case unsupportedInput(String)
    case pixelBufferCreationFailed(CVReturn)
}

/// Minimal type-erased Encodable wrapper so a single dictionary literal can
/// mix [BenchmarkResult], [String], and String values when writing the
/// summary JSON.
private struct AnyEncodable: Encodable {
    private let encodeClosure: (Encoder) throws -> Void

    init<T: Encodable>(_ value: T) {
        encodeClosure = { encoder in
            try value.encode(to: encoder)
        }
    }

    func encode(to encoder: Encoder) throws {
        try encodeClosure(encoder)
    }
}
