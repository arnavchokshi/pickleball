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

        let videoURL = try XCTUnwrap(
            Bundle.module.url(
                forResource: "six_frame_64x64",
                withExtension: "mov",
                subdirectory: "Resources/ReplayPersonBenchmarkFixtures"
            )
        )
        let outputRoot = root.appendingPathComponent("outputs", isDirectory: true)
        let processor = IntervalReplayPersonFrameProcessor(detectionIntervalFrames: 2)
        let runner = ReplayPersonBenchmarkRunner(runtimeFactory: { processor })

        let summary: ReplayPersonBenchmarkRunSummary
        do {
            summary = try await runner.run(
                clip: ResolvedReplayInputClip(
                    clipID: "clip_1",
                    name: "Clip 1",
                    videoURL: videoURL,
                    groundTruthURL: root.appendingPathComponent("ground_truth.json"),
                    expectedPlayers: 4
                ),
                outputRootURL: outputRoot
            )
        } catch ReplayPersonBenchmarkRunnerError.cannotStartReader(let message) where message == "Cannot Decode" {
            throw XCTSkip("Host AVAssetReader cannot decode the bundled synthetic video fixture.")
        }

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
