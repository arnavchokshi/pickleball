import XCTest
@testable import PickleballFastTier

#if canImport(CoreML)
import CoreML
#endif

final class CoreMLPersonDetectorTests: XCTestCase {
    func testConfigurationClampsDetectionIntervalToAtLeastOne() {
        let configuration = CoreMLPersonDetectorConfiguration(
            candidate: .yolo26nInt8Detect2Track30,
            modelURL: URL(fileURLWithPath: "/tmp/model.mlmodelc", isDirectory: true),
            outputFormat: .yolo26EndToEnd,
            inputWidth: 416,
            inputHeight: 416,
            detectionIntervalFrames: 0
        )

        XCTAssertEqual(configuration.detectionIntervalFrames, 1)
    }

    func testYolo11NMSDecoderConvertsNormalizedCenterBoxesToSourcePixels() {
        let observations = CoreMLPersonDetectionDecoder.decodeYolo11NMS(
            coordinates: [
                [0.50, 0.25, 0.20, 0.10],
                [0.30, 0.40, 0.10, 0.20],
            ],
            confidences: [
                [0.80, 0.01],
                [0.05, 0.90],
            ],
            sourceWidth: 1000,
            sourceHeight: 500,
            maxPlayers: 4,
            minConfidence: 0.10
        )

        XCTAssertEqual(observations.count, 1)
        XCTAssertEqual(observations[0].bboxXYWH, [400, 100, 200, 50])
        XCTAssertEqual(observations[0].confidence, 0.80, accuracy: 0.001)
        XCTAssertEqual(observations[0].source, "coreml_yolo11n_nms")
    }

    func testYolo26EndToEndDecoderScalesInputPixelBoxesToSourcePixelsAndCapsPlayers() {
        let observations = CoreMLPersonDetectionDecoder.decodeYolo26EndToEnd(
            rows: [
                [10, 20, 110, 220, 0.70, 0],
                [50, 40, 150, 240, 0.95, 2],
                [30, 60, 130, 260, 0.60, 0],
                [40, 80, 140, 280, 0.50, 0],
            ],
            modelInputWidth: 200,
            modelInputHeight: 400,
            sourceWidth: 1000,
            sourceHeight: 800,
            maxPlayers: 2,
            minConfidence: 0.20
        )

        XCTAssertEqual(observations.count, 2)
        XCTAssertEqual(observations.map(\.confidence), [0.70, 0.60])
        XCTAssertEqual(observations[0].bboxXYWH, [50, 40, 500, 400])
        XCTAssertEqual(observations[1].bboxXYWH, [150, 120, 500, 400])
        XCTAssertEqual(observations[0].source, "coreml_yolo26_end2end")
    }

#if canImport(CoreML)
    func testMultiArrayRowsReadYolo11ConfidenceShapeWithStrides() throws {
        let confidence = try MLMultiArray(shape: [2, 80], dataType: .double)
        confidence[[0, 0] as [NSNumber]] = 0.80
        confidence[[1, 0] as [NSNumber]] = 0.25
        confidence[[1, 3] as [NSNumber]] = 0.95

        let rows = try CoreMLPersonMultiArrayRows.read(confidence, columns: 80, outputName: "confidence")

        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows[0][0], 0.80, accuracy: 0.001)
        XCTAssertEqual(rows[1][0], 0.25, accuracy: 0.001)
        XCTAssertEqual(rows[1][3], 0.95, accuracy: 0.001)
    }

    func testMultiArrayRowsRejectsScalarOrClassMajorConfidenceShapes() throws {
        let scalarConfidence = try MLMultiArray(shape: [2], dataType: .double)
        XCTAssertThrowsError(try CoreMLPersonMultiArrayRows.read(scalarConfidence, columns: 80, outputName: "confidence")) { error in
            guard case CoreMLPersonDetectorError.unsupportedOutputShape(let message) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains("confidence expected [N, 80], got [2]"))
        }

        let classMajorConfidence = try MLMultiArray(shape: [80, 2], dataType: .double)
        XCTAssertThrowsError(try CoreMLPersonMultiArrayRows.read(classMajorConfidence, columns: 80, outputName: "confidence")) { error in
            guard case CoreMLPersonDetectorError.unsupportedOutputShape(let message) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains("confidence expected [N, 80], got [80, 2]"))
        }
    }
#endif
}
