import Foundation
import XCTest
@testable import PickleballReplay

final class WorldBundleTests: XCTestCase {
    func testBundledSampleLoadsVerifiedWolverineGlueRun() throws {
        let bundle = try WorldBundle.loadBundledSample()

        XCTAssertEqual(bundle.manifest.clip, "wolverine_mixed_0200_mid_steep_corner")
        XCTAssertEqual(bundle.world.players.count, 4)
        XCTAssertEqual(bundle.world.court.sport, "pickleball")
        XCTAssertGreaterThan(bundle.world.ball.frames.count, 0)
        XCTAssertGreaterThan(bundle.world.summary.jointPlayerFrameCount, 0)
        XCTAssertEqual(bundle.world.summary.meshPlayerFrameCount, 0)
        XCTAssertNil(bundle.bodyMesh, "bundled fixture intentionally omits the 570 MB verified BODY mesh artifact")
        XCTAssertNotNil(bundle.contactWindows, "verified process_video run regenerated contact windows after remote skeleton fallback")
        XCTAssertEqual(bundle.contactWindows?.events.count, 24)
    }

    func testBundledSamplePlayersHaveHonestRepresentationsAndTrustBands() throws {
        let bundle = try WorldBundle.loadBundledSample()
        XCTAssertEqual(Set(bundle.world.players.map(\.representation)), [.joints])
        XCTAssertTrue(bundle.world.players.allSatisfy { $0.trustBand?.badge == .preview })

        XCTAssertEqual(bundle.world.ball.trustBand?.badge, .lowConfidence)
        XCTAssertEqual(bundle.world.court.trustBand?.badge, .preview)
    }

    func testBundledSampleDoesNotFabricateBundledMeshArtifacts() throws {
        let bundle = try WorldBundle.loadBundledSample()
        XCTAssertNil(bundle.bodyMesh)
        XCTAssertNotNil(bundle.contactWindows)
        for player in bundle.world.players {
            XCTAssertTrue(player.frames.allSatisfy { $0.meshVerticesWorld.isEmpty })
        }
    }

    func testMissingManifestResourceThrowsRatherThanFabricatingAWorld() {
        let emptyBundle = Bundle(for: WorldBundleTests.self)
        XCTAssertThrowsError(try WorldBundle.loadBundledSample(bundle: emptyBundle)) { error in
            XCTAssertEqual(error as? WorldBundleError, .missingManifestResource)
        }
    }

    func testTempFileManifestResolvesRelativeAssetsAndMissingOptionalDoesNotSubstitute() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("world-bundle-own-file-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let manifestURL = root.appendingPathComponent("replay_viewer_manifest.json")
        try Self.minimalWorldData(clip: "capture-own").write(to: root.appendingPathComponent("own_world.json"))
        try Self.manifestData(
            clip: "capture-own",
            virtualWorldURL: "own_world.json",
            bodyMeshURL: "missing_body.json"
        ).write(to: manifestURL)

        let bundle = try WorldBundle.load(manifestURL: manifestURL)

        XCTAssertEqual(bundle.manifest.clip, "capture-own")
        XCTAssertEqual(bundle.world.artifactType, "racketsport_virtual_world")
        XCTAssertNil(bundle.bodyMesh)
        XCTAssertEqual(bundle.assetIssues.map(\.manifestField), ["body_mesh_url"])
        XCTAssertNotEqual(bundle.manifest.clip, "wolverine_mixed_0200_mid_steep_corner")
    }

    func testHTTPManifestResolvesRelativeWorldAndOptional404RemainsOwnCapture() async throws {
        ReplayStubURLProtocol.reset()
        defer { ReplayStubURLProtocol.reset() }
        ReplayStubURLProtocol.handler = { request in
            switch request.url?.path {
            case "/api/jobs/job-own/manifest":
                return (200, Self.manifestData(
                    clip: "capture-http-own",
                    virtualWorldURL: "artifacts/own_world.json",
                    contactWindowsURL: "artifacts/missing_contacts.json"
                ))
            case "/api/jobs/job-own/artifacts/own_world.json":
                return (200, Self.minimalWorldData(clip: "capture-http-own"))
            case "/api/jobs/job-own/artifacts/missing_contacts.json":
                return (404, Data("missing".utf8))
            default:
                XCTFail("unexpected or fixture-fallback request: \(request.url?.absoluteString ?? "nil")")
                return (500, Data())
            }
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ReplayStubURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let manifestURL = URL(string: "https://api.example.test/api/jobs/job-own/manifest")!

        let bundle = try await WorldBundle.load(manifestURL: manifestURL) { url in
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                throw URLError(.fileDoesNotExist)
            }
            return data
        }

        XCTAssertEqual(bundle.manifest.clip, "capture-http-own")
        XCTAssertNil(bundle.contactWindows)
        XCTAssertEqual(bundle.assetIssues.map(\.manifestField), ["contact_windows_url"])
        XCTAssertEqual(
            Set(ReplayStubURLProtocol.requestedPaths),
            [
                "/api/jobs/job-own/manifest",
                "/api/jobs/job-own/artifacts/own_world.json",
                "/api/jobs/job-own/artifacts/missing_contacts.json",
            ]
        )
    }

    func testManifestDecodesEveryWebViewerURLFieldIncludingBallArcRender() throws {
        let data = Data(
            #"{"schema_version":1,"artifact_type":"racketsport_replay_viewer_manifest","clip":"own","video_url":"video.mp4","virtual_world_url":"world.json","replay_scene_url":"scene.json","body_mesh_url":"mesh.json","body_mesh_index_url":"mesh_index.json","physics_refinement_url":"physics.json","contact_windows_url":"contacts.json","reviewed_bounces_url":"reviewed.json","ball_inflections_url":"inflections.json","events_selected_url":"events.json","shots_url":"shots.json","ball_arc_solved_url":"solved.json","ball_arc_render_url":"render.json","auto_bounce_candidates_url":"auto.json","ball_bounce_candidates_url":"bounce.json","ball_flight_sanity_url":"sanity.json","rally_spans_url":"spans.json","rally_metrics_url":"metrics.json","coaching_card_facts_url":"coaching.json","label_overlays":[{"kind":"player_boxes","label":"boxes","url":"labels.json","trusted_for_metrics":false,"not_ground_truth":true}],"annotation_sources":[{"kind":"annotation","clip_id":"own","url":"annotations.json","trusted_for_metrics":false}],"notes":[]}"#.utf8
        )

        let manifest = try WorldViewerManifest.decode(data)

        XCTAssertEqual(manifest.replaySceneURL, "scene.json")
        XCTAssertEqual(manifest.bodyMeshIndexURL, "mesh_index.json")
        XCTAssertEqual(manifest.reviewedBouncesURL, "reviewed.json")
        XCTAssertEqual(manifest.eventsSelectedURL, "events.json")
        XCTAssertEqual(manifest.shotsURL, "shots.json")
        XCTAssertEqual(manifest.ballArcSolvedURL, "solved.json")
        XCTAssertEqual(manifest.ballArcRenderURL, "render.json")
        XCTAssertEqual(manifest.autoBounceCandidatesURL, "auto.json")
        XCTAssertEqual(manifest.ballBounceCandidatesURL, "bounce.json")
        XCTAssertEqual(manifest.ballFlightSanityURL, "sanity.json")
        XCTAssertEqual(manifest.rallySpansURL, "spans.json")
        XCTAssertEqual(manifest.rallyMetricsURL, "metrics.json")
        XCTAssertEqual(manifest.coachingCardFactsURL, "coaching.json")
        XCTAssertEqual(manifest.labelOverlays.first?.url, "labels.json")
        XCTAssertEqual(manifest.annotationSources.first?.url, "annotations.json")
    }

    private static func manifestData(
        clip: String,
        virtualWorldURL: String,
        bodyMeshURL: String? = nil,
        contactWindowsURL: String? = nil
    ) -> Data {
        var payload: [String: Any] = [
            "schema_version": 1,
            "artifact_type": "racketsport_replay_viewer_manifest",
            "clip": clip,
            "video_url": "video.mp4",
            "virtual_world_url": virtualWorldURL,
            "notes": [],
        ]
        payload["body_mesh_url"] = bodyMeshURL
        payload["contact_windows_url"] = contactWindowsURL
        return try! JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    }

    private static func minimalWorldData(clip _: String) -> Data {
        let payload: [String: Any] = [
            "schema_version": 1,
            "artifact_type": "racketsport_virtual_world",
            "world_frame": "court_Z0",
            "fps": 30,
            "court": [
                "sport": "pickleball",
                "coordinate_frame": "court_Z0",
                "length_m": 13.41,
                "width_m": 6.10,
                "line_segments": [:],
                "net": [
                    "endpoints": [[0, -3.05, 0], [0, 3.05, 0]],
                    "center_height_m": 0.86,
                    "post_height_m": 0.91,
                ],
            ],
            "players": [],
            "ball": ["source": "test", "frames": []],
            "paddles": [],
            "summary": [
                "player_count": 0,
                "mesh_player_frame_count": 0,
                "joint_player_frame_count": 0,
                "track_only_player_frame_count": 0,
                "ball_frame_count": 0,
                "approx_ball_frame_count": 0,
            ],
        ]
        return try! JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    }
}

private final class ReplayStubURLProtocol: URLProtocol, @unchecked Sendable {
    static let lock = NSLock()
    nonisolated(unsafe) static var handler: ((URLRequest) -> (Int, Data))?
    nonisolated(unsafe) private static var paths: [String] = []

    static var requestedPaths: [String] {
        lock.lock()
        defer { lock.unlock() }
        return paths
    }

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        handler = nil
        paths = []
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lock.lock()
        Self.paths.append(request.url?.path ?? "")
        let handler = Self.handler
        Self.lock.unlock()
        guard let handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        let (status, data) = handler(request)
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
