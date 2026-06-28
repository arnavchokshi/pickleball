import XCTest
@testable import PickleballFastTier

final class OnDevicePersonTrackingTests: XCTestCase {
    func testOnDevicePersonTracksEncodeMobileBenchmarkSchema() throws {
        let tracks = OnDevicePersonTracks(
            clipID: "clip-a",
            candidate: .visionPoseRoleLock,
            deviceModel: "iPhone15,2",
            resolution: [1920, 1080],
            fps: 30,
            frames: [
                OnDevicePersonFrame(
                    frameIndex: 0,
                    detections: [
                        OnDevicePersonDetection(
                            trackID: 1,
                            bboxXYWH: [10, 20, 30, 40],
                            confidence: 0.91,
                            source: "apple_vision_body_2d",
                            role: "near_left"
                        )
                    ]
                )
            ]
        )

        XCTAssertEqual(tracks.artifactType, "racketsport_on_device_person_tracks")
        XCTAssertEqual(tracks.summary.frameCount, 1)
        XCTAssertEqual(tracks.summary.detectionCount, 1)
        XCTAssertEqual(tracks.summary.trackIDs, [1])

        let data = try JSONEncoder().encode(tracks)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["schema_version"] as? Int, 1)
        XCTAssertEqual(object["artifact_type"] as? String, "racketsport_on_device_person_tracks")
        XCTAssertEqual(object["clip_id"] as? String, "clip-a")
        XCTAssertEqual(object["candidate"] as? String, "vision_pose_rolelock")
        XCTAssertEqual(object["device_model"] as? String, "iPhone15,2")
        XCTAssertEqual(object["coordinate_space"] as? String, "source_video_pixels")
        XCTAssertEqual(object["resolution"] as? [Int], [1920, 1080])

        let frames = try XCTUnwrap(object["frames"] as? [[String: Any]])
        XCTAssertEqual(frames[0]["frame_index"] as? Int, 0)
        let detections = try XCTUnwrap(frames[0]["detections"] as? [[String: Any]])
        XCTAssertEqual(detections[0]["track_id"] as? Int, 1)
        XCTAssertEqual(detections[0]["bbox_xywh"] as? [Double], [10, 20, 30, 40])
    }

    func testOnDevicePersonTimingComputesLatencyAndDropSummaries() throws {
        let timing = OnDevicePersonTiming(
            clipID: "clip-a",
            candidate: .yolo26nInt8Detect15Track30,
            mode: .replay,
            deviceModel: "iPhone15,2",
            osVersion: "26.5",
            wallClockSeconds: 2.0,
            droppedFrameCount: 3,
            modelLoadMs: 120.0,
            mlpackageSizeMB: 18.5,
            startedThermalState: "nominal",
            endedThermalState: "fair",
            samples: [
                OnDevicePersonTimingSample(frameIndex: 0, latencyMs: 8.0, processed: true),
                OnDevicePersonTimingSample(frameIndex: 1, latencyMs: 12.0, processed: true),
                OnDevicePersonTimingSample(frameIndex: 2, latencyMs: 20.0, processed: true),
            ]
        )

        XCTAssertEqual(timing.artifactType, "racketsport_on_device_person_timing")
        XCTAssertEqual(timing.summary.processedFrameCount, 3)
        XCTAssertEqual(timing.summary.droppedFrameCount, 3)
        XCTAssertEqual(timing.summary.sustainedProcessedFPS, 1.5)
        XCTAssertEqual(timing.summary.p50LatencyMs, 12.0)
        XCTAssertEqual(timing.summary.p95LatencyMs, 20.0)

        let data = try JSONEncoder().encode(timing)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["artifact_type"] as? String, "racketsport_on_device_person_timing")
        XCTAssertEqual(object["candidate"] as? String, "yolo26n_int8_detect15_track30")
        XCTAssertEqual(object["mode"] as? String, "replay")
        XCTAssertEqual(object["os_version"] as? String, "26.5")
        XCTAssertEqual(object["model_load_ms"] as? Double, 120.0)
        XCTAssertEqual(object["mlpackage_size_mb"] as? Double, 18.5)
    }

    func testBenchmarkArtifactPathsUseStableFilenames() {
        let paths = OnDevicePersonBenchmarkArtifactPaths(sessionRelativePath: "captures/court-a-001")

        XCTAssertEqual(paths.tracksRelativePath, "captures/court-a-001/on_device_person_tracks.json")
        XCTAssertEqual(paths.timingRelativePath, "captures/court-a-001/timing.json")
    }
}
