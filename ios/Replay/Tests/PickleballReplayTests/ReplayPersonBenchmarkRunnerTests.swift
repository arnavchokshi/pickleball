import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import PickleballFastTier
import XCTest
@testable import PickleballReplay

@available(iOS 13.0, macOS 10.15, *)
final class ReplayPersonBenchmarkRunnerTests: XCTestCase {
    func testDetectEveryTwoFramesCountsOnlyDetectorInvocationsAsProcessed() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("pickleball-replay-benchmark-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let videoURL = root.appendingPathComponent("input.mov")
        try makeVideo(url: videoURL, frameCount: 6)
        let outputRoot = root.appendingPathComponent("outputs", isDirectory: true)
        let processor = IntervalReplayPersonFrameProcessor(detectionIntervalFrames: 2)
        let runner = ReplayPersonBenchmarkRunner(runtimeFactory: { processor })

        let summary = try await runner.run(
            clip: ResolvedReplayInputClip(
                clipID: "clip_1",
                name: "Clip 1",
                videoURL: videoURL,
                groundTruthURL: root.appendingPathComponent("ground_truth.json"),
                expectedPlayers: 4
            ),
            outputRootURL: outputRoot
        )

        let paths = ReplayPersonBenchmarkOutputPaths(
            rootURL: outputRoot,
            clipID: "clip_1",
            candidate: ReplayPersonBenchmarkProcessor.visionHumanRectangles.candidate.rawValue
        )
        let timing = try JSONDecoder().decode(TimingArtifact.self, from: Data(contentsOf: paths.timingURL))
        let summaryArtifact = try JSONDecoder().decode(SummaryArtifact.self, from: Data(contentsOf: paths.summaryURL))

        XCTAssertEqual(timing.samples.count, 6)
        XCTAssertEqual(timing.samples.filter(\.processed).map(\.frameIndex), [0, 2, 4])
        XCTAssertEqual(timing.summary.processedFrameCount, 3)
        XCTAssertEqual(summary.processedFrameCount, 3)
        XCTAssertEqual(summaryArtifact.processedFrameCount, 3)
        XCTAssertEqual(summaryArtifact.frameCount, 6)
        XCTAssertEqual(summaryArtifact.detectorInvocationCount, 3)
        XCTAssertEqual(summaryArtifact.propagatedFrameCount, 3)
    }
}

private final class IntervalReplayPersonFrameProcessor: ReplayPersonFrameProcessor {
    let candidate: OnDevicePersonCandidate = .visionHumanRectanglesIouV1
    let modelLoadMs: Double? = 0
    let mlpackageSizeMB: Double? = nil
    private let detectionIntervalFrames: Int

    init(detectionIntervalFrames: Int) {
        self.detectionIntervalFrames = max(1, detectionIntervalFrames)
    }

    func process(pixelBuffer: CVPixelBuffer, frameIndex: Int) throws -> ReplayPersonFrameProcessingResult {
        ReplayPersonFrameProcessingResult(
            detections: [],
            detectorInvoked: frameIndex % detectionIntervalFrames == 0
        )
    }
}

private struct TimingArtifact: Decodable {
    var samples: [TimingSample]
    var summary: TimingSummary
}

private struct TimingSample: Decodable {
    var frameIndex: Int
    var processed: Bool

    private enum CodingKeys: String, CodingKey {
        case frameIndex = "frame_index"
        case processed
    }
}

private struct TimingSummary: Decodable {
    var processedFrameCount: Int

    private enum CodingKeys: String, CodingKey {
        case processedFrameCount = "processed_frame_count"
    }
}

private struct SummaryArtifact: Decodable {
    var processedFrameCount: Int
    var frameCount: Int?
    var detectorInvocationCount: Int?
    var propagatedFrameCount: Int?

    private enum CodingKeys: String, CodingKey {
        case processedFrameCount = "processed_frame_count"
        case frameCount = "frame_count"
        case detectorInvocationCount = "detector_invocation_count"
        case propagatedFrameCount = "propagated_frame_count"
    }
}

private enum TestVideoError: Error, Equatable {
    case cannotAddInput
    case cannotCreatePixelBuffer(OSStatus)
    case cannotAppendFrame(Int)
    case cannotStartWriting(String)
    case writerFailed(String)
}

private func makeVideo(url: URL, frameCount: Int) throws {
    let width = 16
    let height = 16
    let framesPerSecond: Int32 = 30
    let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
    let input = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
    )
    input.expectsMediaDataInRealTime = false
    guard writer.canAdd(input) else {
        throw TestVideoError.cannotAddInput
    }
    writer.add(input)
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            String(kCVPixelBufferPixelFormatTypeKey): kCVPixelFormatType_32BGRA,
            String(kCVPixelBufferWidthKey): width,
            String(kCVPixelBufferHeightKey): height,
            String(kCVPixelBufferCGImageCompatibilityKey): true,
            String(kCVPixelBufferCGBitmapContextCompatibilityKey): true,
        ]
    )

    guard writer.startWriting() else {
        throw TestVideoError.cannotStartWriting(writer.error?.localizedDescription ?? "unknown error")
    }
    writer.startSession(atSourceTime: .zero)

    for frameIndex in 0..<frameCount {
        while !input.isReadyForMoreMediaData {
            Thread.sleep(forTimeInterval: 0.001)
        }
        let pixelBuffer = try makePixelBuffer(width: width, height: height, fill: UInt8(frameIndex))
        let presentationTime = CMTime(value: CMTimeValue(frameIndex), timescale: framesPerSecond)
        guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
            throw TestVideoError.cannotAppendFrame(frameIndex)
        }
    }

    input.markAsFinished()
    let semaphore = DispatchSemaphore(value: 0)
    writer.finishWriting {
        semaphore.signal()
    }
    semaphore.wait()
    guard writer.status == .completed else {
        throw TestVideoError.writerFailed(writer.error?.localizedDescription ?? "unknown error")
    }
}

private func makePixelBuffer(width: Int, height: Int, fill: UInt8) throws -> CVPixelBuffer {
    let attributes: [String: Any] = [
        String(kCVPixelBufferCGImageCompatibilityKey): true,
        String(kCVPixelBufferCGBitmapContextCompatibilityKey): true,
    ]
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        attributes as CFDictionary,
        &pixelBuffer
    )
    guard status == kCVReturnSuccess, let pixelBuffer else {
        throw TestVideoError.cannotCreatePixelBuffer(status)
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    if let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) {
        memset(baseAddress, Int32(fill), CVPixelBufferGetDataSize(pixelBuffer))
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
    return pixelBuffer
}
