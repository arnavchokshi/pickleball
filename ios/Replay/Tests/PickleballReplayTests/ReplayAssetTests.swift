import XCTest
@testable import PickleballReplay

final class ReplayAssetTests: XCTestCase {
    func testReplayAssetValidationRequiresUsdzOrGlbAndPositiveDuration() {
        let asset = ReplayAsset(usdzURL: nil, glbURL: nil, durationSeconds: 0)

        let report = ReplayAssetValidator.validate(asset)

        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains(.missingRenderableAsset))
        XCTAssertTrue(report.errors.contains(.invalidDuration(0)))
    }

    func testReplayAssetReferencesExposeUsdzAndGlbMetadata() throws {
        let asset = ReplayAsset(
            usdzURL: try XCTUnwrap(URL(string: "https://cdn.example.com/replay/session.usdz")),
            glbURL: try XCTUnwrap(URL(string: "point_3.glb")),
            durationSeconds: 10
        )

        let report = ReplayAssetValidator.validate(asset)
        let references = ReplayAssetReference.references(for: asset)

        XCTAssertTrue(report.isValid)
        XCTAssertEqual(references.map(\.format), [.usdz, .glb])
        XCTAssertEqual(references.map(\.role), [.nativeRealityKit, .webShare])
        XCTAssertEqual(references.map(\.pathExtension), ["usdz", "glb"])
    }

    func testTimelineAndCapabilityDescriptorAreMetadataOnly() throws {
        let asset = ReplayAsset(
            usdzURL: try XCTUnwrap(URL(string: "https://cdn.example.com/replay/session.usdz")),
            glbURL: try XCTUnwrap(URL(string: "point_3.glb")),
            durationSeconds: 12
        )
        let timeline = ReplayTimelineDescriptor(
            schemaVersion: 1,
            worldFrame: "court_Z0",
            fps: 30,
            durationSeconds: 12,
            points: [
                ReplayTimelinePoint(
                    id: 3,
                    startSeconds: 1.2,
                    endSeconds: 11.2,
                    glbURL: try XCTUnwrap(URL(string: "point_3.glb")),
                    sizeMB: 9.4
                )
            ]
        )

        let capabilities = ReplayCapabilityDescriptor.describe(asset: asset, timeline: timeline)

        XCTAssertTrue(capabilities.supportsNativeUSDZ)
        XCTAssertTrue(capabilities.supportsWebGLB)
        XCTAssertTrue(capabilities.supportsTimelineScrubbing)
        XCTAssertTrue(capabilities.supportsFreeViewpointMetadata)
        XCTAssertFalse(capabilities.hasRealityKitRuntimeValidation)
        XCTAssertEqual(capabilities.validationScope, .metadataOnly)
    }

    func testReplayInputManifestLoadsStagedVideoAndGroundTruthPaths() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("pickleball-replay-\(UUID().uuidString)", isDirectory: true)
        let clipDirectory = root.appendingPathComponent("task_1", isDirectory: true)
        try FileManager.default.createDirectory(at: clipDirectory, withIntermediateDirectories: true)
        let videoURL = clipDirectory.appendingPathComponent("input.mp4")
        let groundTruthURL = clipDirectory.appendingPathComponent("person_ground_truth.json")
        FileManager.default.createFile(atPath: videoURL.path, contents: Data())
        FileManager.default.createFile(atPath: groundTruthURL.path, contents: Data())
        let manifestURL = root.appendingPathComponent("manifest.json")
        try """
        {
          "schema_version": 1,
          "artifact_type": "pickleball_replay_input_manifest",
          "clips": [
            {
              "clip_id": "task_1",
              "name": "test clip",
              "video": "task_1/input.mp4",
              "ground_truth": "task_1/person_ground_truth.json",
              "expected_players": 4
            }
          ]
        }
        """.data(using: .utf8)!.write(to: manifestURL)

        let manifest = try ReplayInputManifest.load(from: manifestURL)
        let clips = manifest.resolvedClips(relativeTo: root)

        XCTAssertEqual(manifest.schemaVersion, 1)
        XCTAssertEqual(manifest.artifactType, "pickleball_replay_input_manifest")
        XCTAssertEqual(clips.map(\.clipID), ["task_1"])
        XCTAssertEqual(clips.first?.videoURL, videoURL)
        XCTAssertEqual(clips.first?.groundTruthURL, groundTruthURL)
        XCTAssertEqual(clips.first?.expectedPlayers, 4)
    }

    func testReplayPersonBenchmarkOutputPathsUseClipAndCandidateFolders() {
        let root = URL(fileURLWithPath: "/tmp/replay-benchmarks", isDirectory: true)

        let paths = ReplayPersonBenchmarkOutputPaths(
            rootURL: root,
            clipID: "task_1",
            candidate: "vision_human_rectangles_iou_v1"
        )

        XCTAssertEqual(paths.sessionURL, root.appendingPathComponent("task_1/vision_human_rectangles_iou_v1", isDirectory: true))
        XCTAssertEqual(paths.tracksURL.lastPathComponent, "on_device_person_tracks.json")
        XCTAssertEqual(paths.timingURL.lastPathComponent, "timing.json")
        XCTAssertEqual(paths.summaryURL.lastPathComponent, "run_summary.json")
        XCTAssertEqual(paths.progressURL.lastPathComponent, "progress.jsonl")
    }
}
