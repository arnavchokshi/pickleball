import AVFoundation
import Vision
import Foundation

enum HarnessError: Error, CustomStringConvertible {
    case noVideoTrack(String)
    case readerFailedToStart(String)
    case readerFailed(String)

    var description: String {
        switch self {
        case .noVideoTrack(let path):
            return "no video track found in \(path)"
        case .readerFailedToStart(let reason):
            return "AVAssetReader failed to start reading: \(reason)"
        case .readerFailed(let reason):
            return "AVAssetReader failed mid-read: \(reason)"
        }
    }
}

/// Reads an on-disk video file via AVFoundation and drives it frame-by-frame
/// through `TrajectoryDetector`. This is the macOS-only, offline-eval half
/// of the spike (rung-1 "run it on committed clips now"); the
/// `TrajectoryDetector` core it wraps is the device-ready half.
struct VideoTrajectoryHarness {
    let inputPath: String
    let trajectoryLength: Int
    let objectMinimumNormalizedRadius: Float
    let objectMaximumNormalizedRadius: Float
    let maxFrames: Int?
    let verbose: Bool

    func run() throws -> HarnessOutput {
        let url = URL(fileURLWithPath: inputPath)
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw HarnessError.noVideoTrack(inputPath)
        }

        let naturalSize = track.naturalSize
        let width = Int(abs(naturalSize.width.rounded()))
        let height = Int(abs(naturalSize.height.rounded()))

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw HarnessError.readerFailedToStart("\(error)")
        }

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        trackOutput.alwaysCopiesSampleData = false
        guard reader.canAdd(trackOutput) else {
            throw HarnessError.readerFailedToStart("cannot add track output")
        }
        reader.add(trackOutput)

        guard reader.startReading() else {
            throw HarnessError.readerFailedToStart(reader.error.map { "\($0)" } ?? "unknown")
        }

        let detector = TrajectoryDetector(
            trajectoryLength: trajectoryLength,
            objectMinimumNormalizedRadius: objectMinimumNormalizedRadius,
            objectMaximumNormalizedRadius: objectMaximumNormalizedRadius
        )

        var framePtsS: [Double] = []
        var emissions: [TrajectoryEmission] = []
        var frameIndex = 0
        var perFrameErrorCount = 0

        let clockStart = DispatchTime.now()

        while let sampleBuffer = trackOutput.copyNextSampleBuffer() {
            if let maxFrames = maxFrames, frameIndex >= maxFrames {
                break
            }
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let ptsSeconds = pts.isValid ? CMTimeGetSeconds(pts) : Double(frameIndex) / 30.0
            framePtsS.append(ptsSeconds)

            do {
                let results = try detector.process(sampleBuffer: sampleBuffer, orientation: .up)
                if !results.isEmpty {
                    for observation in results {
                        emissions.append(
                            makeEmission(
                                observation: observation,
                                emittedAtFrameIndex: frameIndex,
                                framePtsS: framePtsS
                            )
                        )
                    }
                    if verbose {
                        FileHandle.standardError.write(
                            "frame \(frameIndex): \(results.count) trajectory observation(s)\n".data(using: .utf8)!
                        )
                    }
                }
            } catch {
                perFrameErrorCount += 1
                if verbose {
                    FileHandle.standardError.write("frame \(frameIndex) perform() failed: \(error)\n".data(using: .utf8)!)
                }
            }

            frameIndex += 1
        }

        if reader.status == .failed {
            throw HarnessError.readerFailed(reader.error.map { "\($0)" } ?? "unknown")
        }

        let clockEnd = DispatchTime.now()
        let wallClockSeconds = Double(clockEnd.uptimeNanoseconds - clockStart.uptimeNanoseconds) / 1_000_000_000.0

        let durationS = framePtsS.count >= 2 ? (framePtsS.last! - framePtsS.first!) : CMTimeGetSeconds(asset.duration)
        var fps = Double(track.nominalFrameRate)
        if !fps.isFinite || fps <= 0, framePtsS.count >= 2, durationS > 0 {
            fps = Double(framePtsS.count - 1) / durationS
        }

        var notes: [String] = [
            "Vision normalized coordinates use a bottom-left origin; the converter script flips Y to match ball_track.json's top-left pixel convention.",
            "orientation is hardcoded to .up (no preferredTransform-based rotation correction) — all four committed eval clips are landscape 1920x1080 with no rotation metadata, verified via ffprobe before this run.",
        ]
        if perFrameErrorCount > 0 {
            notes.append("\(perFrameErrorCount) of \(frameIndex) perform() calls raised an error and were skipped (see stderr with --verbose).")
        }

        return HarnessOutput(
            schemaVersion: 1,
            artifactType: "vn_trajectories_spike_raw",
            status: "TESTED-ON-REAL-DATA",
            sourceVideo: inputPath,
            video: VideoMeta(
                width: width,
                height: height,
                fps: fps,
                frameCount: frameIndex,
                durationS: durationS
            ),
            requestConfig: RequestConfig(
                frameAnalysisSpacing: "zero",
                trajectoryLength: trajectoryLength,
                objectMinimumNormalizedRadius: Double(objectMinimumNormalizedRadius),
                objectMaximumNormalizedRadius: Double(objectMaximumNormalizedRadius),
                visionRequestRevision: detector.requestRevision
            ),
            run: RunMeta(
                framesFed: frameIndex,
                performCallCount: detector.performCallCount,
                emissionCount: emissions.count,
                wallClockSeconds: wallClockSeconds,
                framesPerSecondProcessed: wallClockSeconds > 0 ? Double(frameIndex) / wallClockSeconds : 0
            ),
            framePtsS: framePtsS,
            trajectories: emissions,
            notes: notes
        )
    }
}

private func makeEmission(
    observation: VNTrajectoryObservation,
    emittedAtFrameIndex: Int,
    framePtsS: [Double]
) -> TrajectoryEmission {
    let detected = indexedPoints(observation.detectedPoints, endingAtFrameIndex: emittedAtFrameIndex, framePtsS: framePtsS)
    // projectedPoints is Vision's smoothed re-fit of the same window; index
    // it against the same frame span as detectedPoints (they are produced
    // from the same window and are the same count in every observation
    // seen during local testing, but this does not assume that — points
    // beyond detectedPoints.count just get nil frame_index/t_s).
    let projected = indexedPoints(observation.projectedPoints, endingAtFrameIndex: emittedAtFrameIndex, framePtsS: framePtsS)

    let coeff = observation.equationCoefficients
    let timeRange = observation.timeRange

    return TrajectoryEmission(
        observationUuid: observation.uuid.uuidString,
        emittedAtFrameIndex: emittedAtFrameIndex,
        confidence: Double(observation.confidence),
        movingAverageRadiusNormalized: Double(observation.movingAverageRadius),
        equationCoefficients: [Double(coeff.x), Double(coeff.y), Double(coeff.z)],
        timeRangeStartS: timeRange.start.isValid ? CMTimeGetSeconds(timeRange.start) : 0,
        timeRangeDurationS: timeRange.duration.isValid ? CMTimeGetSeconds(timeRange.duration) : 0,
        detectedPoints: detected,
        projectedPoints: projected
    )
}

private func indexedPoints(
    _ points: [VNPoint],
    endingAtFrameIndex emittedAtFrameIndex: Int,
    framePtsS: [Double]
) -> [TrajPoint] {
    guard !points.isEmpty else { return [] }
    let startFrameIndex = emittedAtFrameIndex - points.count + 1
    return points.enumerated().map { offset, point in
        let frameIndex = startFrameIndex + offset
        let tS: Double? = (frameIndex >= 0 && frameIndex < framePtsS.count) ? framePtsS[frameIndex] : nil
        let validFrameIndex = frameIndex >= 0 ? frameIndex : nil
        let location = point.location
        return TrajPoint(
            frameIndex: validFrameIndex,
            tS: tS,
            xNorm: Double(location.x),
            yNorm: Double(location.y)
        )
    }
}
