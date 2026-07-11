import Foundation
import XCTest
@testable import PickleballUpload

final class UploadQueueTests: XCTestCase {
    func testQueuedUploadingUploadedWritesExactSidecarAndServerIdentity() async throws {
        let fixture = try PackageFixture(video: Data("0123456789".utf8), sidecar: Data("{\"exact\":true}\n".utf8))
        defer { fixture.remove() }
        let client = FakeUploadQueueClient(observedPackage: fixture.package)
        let queue = UploadQueue(client: client, partSizeBytes: 4)

        let queued = try await queue.enqueue(fixture.package)
        XCTAssertEqual(queued.state, .queued)
        await queue.processPending()

        let uploaded = try UploadQueue.readState(for: fixture.package)
        XCTAssertEqual(uploaded.state, .uploaded)
        XCTAssertEqual(uploaded.clipId, "clip_1")
        XCTAssertEqual(uploaded.uploadedParts, [1, 2, 3])
        XCTAssertEqual(uploaded.bytesUploaded, 10)
        XCTAssertEqual(uploaded.totalBytes, 10)
        XCTAssertEqual(uploaded.serverStatus, "uploaded")
        XCTAssertNil(uploaded.jobId)
        XCTAssertTrue(uploaded.videoCompleted)
        XCTAssertTrue(uploaded.sidecarUploaded)
        let receivedSidecar = await client.receivedSidecar()
        let callOrder = await client.callOrder()
        let observedStates = await client.observedStates()
        XCTAssertEqual(receivedSidecar, fixture.sidecar)
        XCTAssertEqual(callOrder, ["create", "parts", "complete", "sidecar", "list"])
        XCTAssertTrue(observedStates.contains(.uploading))
    }

    func testFailureRetryPreservesClipIDAndCompletesMissingSidecarOnly() async throws {
        let fixture = try PackageFixture(video: Data("video".utf8), sidecar: Data("sidecar-exact".utf8))
        defer { fixture.remove() }
        let client = FakeUploadQueueClient(failSidecarCount: 1)
        let queue = UploadQueue(client: client, partSizeBytes: 3)

        _ = try await queue.enqueue(fixture.package)
        await queue.processPending()
        let failed = try UploadQueue.readState(for: fixture.package)
        XCTAssertEqual(failed.state, .failed)
        XCTAssertEqual(failed.clipId, "clip_1")
        XCTAssertTrue(failed.videoCompleted)
        XCTAssertFalse(failed.sidecarUploaded)

        let retried = try await queue.retry(fixture.package)
        XCTAssertEqual(retried.state, .queued)
        XCTAssertEqual(retried.clipId, "clip_1")
        await queue.processPending()

        let uploaded = try UploadQueue.readState(for: fixture.package)
        XCTAssertEqual(uploaded.state, .uploaded)
        XCTAssertEqual(uploaded.clipId, "clip_1")
        let createCount = await client.createCount()
        let completeCount = await client.completeCount()
        let sidecarAttemptCount = await client.sidecarAttemptCount()
        XCTAssertEqual(createCount, 1)
        XCTAssertEqual(completeCount, 1)
        XCTAssertEqual(sidecarAttemptCount, 2)
    }

    func testRelaunchResumesQueuedStateWithANewQueueInstance() async throws {
        let fixture = try PackageFixture(video: Data("resume-video".utf8), sidecar: Data("resume-sidecar".utf8))
        defer { fixture.remove() }
        let firstQueue = UploadQueue(client: FakeUploadQueueClient(), partSizeBytes: 4)
        _ = try await firstQueue.enqueue(fixture.package)

        let relaunchedClient = FakeUploadQueueClient()
        let relaunchedQueue = UploadQueue(client: relaunchedClient, partSizeBytes: 4)
        let resumed = try await relaunchedQueue.resume([fixture.package])
        XCTAssertEqual(resumed, [fixture.package.packageID])
        XCTAssertEqual(try UploadQueue.readState(for: fixture.package).state, .queued)

        await relaunchedQueue.processPending()

        XCTAssertEqual(try UploadQueue.readState(for: fixture.package).state, .uploaded)
        let relaunchedCreateCount = await relaunchedClient.createCount()
        XCTAssertEqual(relaunchedCreateCount, 1)
    }

    func testQueueProcessesTwoCapturesStrictlySerially() async throws {
        let first = try PackageFixture(packageID: "capture-1", video: Data("first".utf8), sidecar: Data("first-sidecar".utf8))
        let second = try PackageFixture(packageID: "capture-2", video: Data("second".utf8), sidecar: Data("second-sidecar".utf8))
        defer {
            first.remove()
            second.remove()
        }
        let client = FakeUploadQueueClient()
        let queue = UploadQueue(client: client, partSizeBytes: 3)

        _ = try await queue.enqueue(first.package)
        _ = try await queue.enqueue(second.package)
        await queue.processPending()

        XCTAssertEqual(try UploadQueue.readState(for: first.package).state, .uploaded)
        XCTAssertEqual(try UploadQueue.readState(for: second.package).state, .uploaded)
        let order = await client.callOrder()
        XCTAssertEqual(
            order,
            [
                "create", "parts", "complete", "sidecar", "list",
                "create", "parts", "complete", "sidecar", "list",
            ]
        )
    }

    func testAuthenticationGatePromptsThenReleasesOnlyThePendingUpload() {
        var gate = UploadAuthenticationGate()

        XCTAssertEqual(gate.requestUpload(packageID: "capture-1", hasAccessToken: false), .promptSignIn)
        XCTAssertEqual(gate.pendingPackageID, "capture-1")
        XCTAssertEqual(gate.completeSignIn(), "capture-1")
        XCTAssertNil(gate.pendingPackageID)
        XCTAssertEqual(gate.requestUpload(packageID: "capture-2", hasAccessToken: true), .enqueue("capture-2"))
    }

    func testBaseURLResolverUsesLaunchArgumentThenEnvironmentThenPlistThenDefault() {
        let fallback = URL(string: "https://default.example.test")!
        XCTAssertEqual(
            APIBaseURLResolver.resolve(
                arguments: ["app", "-dinkvision.apiBaseURL", "http://127.0.0.1:9090"],
                environment: [APIBaseURLResolver.environmentKey: "https://env.example.test"],
                infoDictionary: [APIBaseURLResolver.plistKey: "https://plist.example.test"],
                defaultURL: fallback
            ).absoluteString,
            "http://127.0.0.1:9090"
        )
        XCTAssertEqual(
            APIBaseURLResolver.resolve(
                arguments: [],
                environment: [APIBaseURLResolver.environmentKey: "https://env.example.test"],
                infoDictionary: [APIBaseURLResolver.plistKey: "https://plist.example.test"],
                defaultURL: fallback
            ).host,
            "env.example.test"
        )
        XCTAssertEqual(
            APIBaseURLResolver.resolve(
                arguments: [],
                environment: [:],
                infoDictionary: [APIBaseURLResolver.plistKey: "https://plist.example.test"],
                defaultURL: fallback
            ).host,
            "plist.example.test"
        )
        XCTAssertEqual(
            APIBaseURLResolver.resolve(
                arguments: [],
                environment: [APIBaseURLResolver.environmentKey: "not a url"],
                infoDictionary: [:],
                defaultURL: fallback
            ),
            fallback
        )
    }

    func testProductionClientCallPathRetriesMidPart5xxAndUsesOnDiskBytes() async throws {
        StubURLProtocol.box.reset()
        defer { StubURLProtocol.box.reset() }
        let fixture = try PackageFixture(
            video: Data("ABCDEF".utf8),
            sidecar: Data("{\"source\":\"on-disk\"}\n".utf8)
        )
        defer { fixture.remove() }
        let counter = LockedCounter()
        StubURLProtocol.box.handler = { request in
            let path = request.url?.path ?? ""
            switch (request.httpMethod, path) {
            case ("POST", "/api/clips"):
                return StubbedResponse(statusCode: 201, json: [
                    "id": "clip_integration",
                    "filename": "clip.mov",
                    "key": "raw/user/clip_integration/clip.mov",
                    "upload_id": "upload_integration",
                    "part_count": 2,
                    "part_urls": [
                        ["part_number": 1, "url": "https://s3.example.test/parts/1"],
                        ["part_number": 2, "url": "https://s3.example.test/parts/2"],
                    ],
                    "sidecar_upload_url": "https://s3.example.test/capture_sidecar.json",
                ])
            case ("PUT", "/parts/1"):
                return StubbedResponse(statusCode: 200, headers: ["ETag": "etag-1"])
            case ("PUT", "/parts/2"):
                if counter.incrementAndGet() == 1 {
                    return StubbedResponse(statusCode: 503, body: Data("try again".utf8))
                }
                return StubbedResponse(statusCode: 200, headers: ["ETag": "etag-2"])
            case ("POST", "/api/clips/clip_integration/complete"):
                return StubbedResponse(statusCode: 200, json: [
                    "id": "clip_integration",
                    "status": "uploaded",
                    "key": "raw/user/clip_integration/clip.mov",
                ])
            case ("PUT", "/capture_sidecar.json"):
                return StubbedResponse(statusCode: 200)
            case ("GET", "/api/clips"):
                return StubbedResponse(statusCode: 200, json: ["clips": [[
                    "id": "clip_integration",
                    "filename": "clip.mov",
                    "status": "uploaded",
                    "size_bytes": 6,
                    "key": "raw/user/clip_integration/clip.mov",
                    "job_id": NSNull(),
                    "created_at": "2026-07-09T12:00:00+00:00",
                ]]])
            default:
                XCTFail("unexpected request: \(request.httpMethod ?? "nil") \(path)")
                return StubbedResponse(statusCode: 500)
            }
        }
        let client = PresignedUploadClient(
            baseURL: URL(string: "https://api.example.test")!,
            session: makeStubbedSession(),
            accessTokenProvider: { "access-token" },
            maxPartAttempts: 2
        )
        let queue = UploadQueue(client: client, partSizeBytes: 3)

        _ = try await queue.enqueue(fixture.package)
        await queue.processPending()

        let state = try UploadQueue.readState(for: fixture.package)
        XCTAssertEqual(state.state, .uploaded)
        XCTAssertEqual(state.clipId, "clip_integration")
        XCTAssertEqual(counter.value, 2)
        let requests = StubURLProtocol.box.capturedRequests
        XCTAssertEqual(
            requests.map { "\($0.request.httpMethod ?? "") \($0.request.url?.path ?? "")" },
            [
                "POST /api/clips",
                "PUT /parts/1",
                "PUT /parts/2",
                "PUT /parts/2",
                "POST /api/clips/clip_integration/complete",
                "PUT /capture_sidecar.json",
                "GET /api/clips",
            ]
        )
        XCTAssertEqual(requests[1].bodyData, Data("ABC".utf8))
        XCTAssertEqual(requests[3].bodyData, Data("DEF".utf8))
        XCTAssertEqual(requests[5].bodyData, fixture.sidecar)
        XCTAssertEqual(requests[0].request.value(forHTTPHeaderField: "Authorization"), "Bearer access-token")
    }

    func testCompletedJobPersistsOwnCaptureClipJobManifestIdentity() async throws {
        StubURLProtocol.box.reset()
        defer { StubURLProtocol.box.reset() }
        let fixture = try PackageFixture(
            packageID: "capture-own",
            video: Data("own-video".utf8),
            sidecar: Data("own-sidecar".utf8)
        )
        defer { fixture.remove() }
        StubURLProtocol.box.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/jobs/job_own")
            return StubbedResponse(statusCode: 200, json: [
                "id": "job_own",
                "clip_id": "clip_1",
                "status": "partial",
                "missing_capabilities": [["capability": "body", "reason": "BODY output missing"]],
                "trust_bands": ["body": ["badge": "preview", "stage": "BODY"]],
                "result": ["manifest_url": "/api/jobs/job_own/manifest"],
                "links": ["status": "/api/jobs/job_own"],
            ])
        }
        let gateway = RenderGatewayClient(
            baseURL: URL(string: "https://api.example.test")!,
            session: makeStubbedSession()
        )
        let queue = UploadQueue(
            client: FakeUploadQueueClient(listedJobId: "job_own"),
            jobClient: gateway,
            partSizeBytes: 4
        )

        _ = try await queue.enqueue(fixture.package)
        await queue.processPending()

        let state = try UploadQueue.readState(for: fixture.package)
        XCTAssertEqual(state.captureId, "capture-own")
        XCTAssertEqual(state.clipId, "clip_1")
        XCTAssertEqual(state.jobId, "job_own")
        XCTAssertEqual(state.serverStatus, "partial")
        XCTAssertEqual(state.manifestUrl, "https://api.example.test/api/jobs/job_own/manifest")
        XCTAssertEqual(state.missingCapabilities.first?.capability, "body")
        guard case .ready(let ready) = state.replayAvailability(expectedCaptureId: "capture-own") else {
            return XCTFail("matching capture should be inspectable")
        }
        XCTAssertEqual(ready.jobId, "job_own")
        XCTAssertEqual(ready.manifestURL.absoluteString, state.manifestUrl)
        XCTAssertEqual(
            state.replayAvailability(expectedCaptureId: "capture-other"),
            .notReady(.identityMismatch(expectedCaptureId: "capture-other", persistedCaptureId: "capture-own"))
        )
    }

    func testUploadedRowWithoutReadyManifestHasTypedNotReadyState() {
        let state = CaptureUploadState(
            state: .uploaded,
            captureId: "capture-1",
            clipId: "clip-1",
            totalBytes: 10,
            serverStatus: "running",
            jobId: "job-1"
        )

        XCTAssertEqual(
            state.replayAvailability(expectedCaptureId: "capture-1"),
            .notReady(.processing("running"))
        )
    }
}

private struct PackageFixture {
    var root: URL
    var package: CaptureUploadPackage
    var sidecar: Data

    init(packageID: String = "capture-1", video: Data, sidecar: Data) throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("upload-queue-tests-\(UUID().uuidString)", isDirectory: true)
        let packageDirectory = root.appendingPathComponent("captures/\(packageID)", isDirectory: true)
        try FileManager.default.createDirectory(at: packageDirectory, withIntermediateDirectories: true)
        let videoURL = packageDirectory.appendingPathComponent("clip.mov")
        let sidecarURL = packageDirectory.appendingPathComponent("capture_sidecar.json")
        try video.write(to: videoURL)
        try sidecar.write(to: sidecarURL)
        self.sidecar = sidecar
        package = CaptureUploadPackage(
            packageID: packageID,
            packageDirectoryURL: packageDirectory,
            videoURL: videoURL,
            sidecarURL: sidecarURL
        )
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}

private actor FakeUploadQueueClient: UploadQueueClient {
    private var events: [String] = []
    private var createCalls = 0
    private var completeCalls = 0
    private var sidecarAttempts = 0
    private var remainingSidecarFailures: Int
    private var sidecarData: Data?
    private let observedPackage: CaptureUploadPackage?
    private let listedJobId: String?
    private var stateObservations: [CaptureUploadStateKind] = []

    init(
        failSidecarCount: Int = 0,
        observedPackage: CaptureUploadPackage? = nil,
        listedJobId: String? = nil
    ) {
        remainingSidecarFailures = failSidecarCount
        self.observedPackage = observedPackage
        self.listedJobId = listedJobId
    }

    func createClip(filename: String, sizeBytes: Int64, partSizeBytes: Int64) async throws -> PresignedClipUploadTarget {
        events.append("create")
        createCalls += 1
        let partCount = Int((sizeBytes + partSizeBytes - 1) / partSizeBytes)
        return PresignedClipUploadTarget(
            id: "clip_\(createCalls)",
            filename: filename,
            key: "raw/user/clip_\(createCalls)/\(filename)",
            uploadId: "upload_\(createCalls)",
            partCount: partCount,
            partUrls: (1...partCount).map { PresignedUploadPartURL(partNumber: $0, url: "https://s3.test/\($0)") },
            sidecarUploadUrl: "https://s3.test/sidecar"
        )
    }

    func uploadParts(
        plan: ResumableChunkPlan,
        fileURL: URL,
        partURLs: [PresignedUploadPartURL],
        contentType: String,
        onPartUploaded: @escaping @Sendable (CompletedUploadPart, Int64) async -> Void
    ) async throws -> [CompletedUploadPart] {
        events.append("parts")
        if let observedPackage,
           let state = try? UploadQueue.readState(for: observedPackage) {
            stateObservations.append(state.state)
        }
        XCTAssertEqual(try Data(contentsOf: fileURL).count, Int(plan.fileSizeBytes))
        XCTAssertEqual(partURLs.count, plan.chunks.count)
        XCTAssertTrue(contentType.hasPrefix("video/"))
        var completed: [CompletedUploadPart] = []
        for chunk in plan.chunks {
            let part = CompletedUploadPart(partNumber: chunk.index + 1, etag: "etag-\(chunk.index + 1)")
            completed.append(part)
            await onPartUploaded(part, chunk.lengthBytes)
        }
        return completed
    }

    func completeClip(clipID: String, uploadID: String, parts: [CompletedUploadPart]) async throws -> PresignedUploadCompleteResult {
        events.append("complete")
        completeCalls += 1
        return PresignedUploadCompleteResult(id: clipID, status: "uploaded", key: "raw/user/\(clipID)/clip.mov")
    }

    func uploadSidecar(data: Data, to sidecarUploadURL: String, contentType: String) async throws {
        events.append("sidecar")
        sidecarAttempts += 1
        if remainingSidecarFailures > 0 {
            remainingSidecarFailures -= 1
            throw PresignedUploadClientError.httpStatus(503, "sidecar unavailable")
        }
        sidecarData = data
    }

    func listClips() async throws -> [ServerClipRecord] {
        events.append("list")
        return [ServerClipRecord(
            id: "clip_1",
            filename: "clip.mov",
            status: "uploaded",
            sizeBytes: 10,
            key: "raw/user/clip_1/clip.mov",
            jobId: listedJobId,
            createdAt: "2026-07-09T12:00:00+00:00"
        )]
    }

    func receivedSidecar() -> Data? { sidecarData }
    func callOrder() -> [String] { events }
    func createCount() -> Int { createCalls }
    func completeCount() -> Int { completeCalls }
    func sidecarAttemptCount() -> Int { sidecarAttempts }
    func observedStates() -> [CaptureUploadStateKind] { stateObservations }
}

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = 0

    func incrementAndGet() -> Int {
        lock.lock()
        defer { lock.unlock() }
        storage += 1
        return storage
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}
