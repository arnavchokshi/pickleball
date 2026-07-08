import XCTest
@testable import PickleballUpload

final class PresignedUploadClientTests: XCTestCase {
    override func setUp() {
        super.setUp()
        StubURLProtocol.box.reset()
    }

    override func tearDown() {
        StubURLProtocol.box.reset()
        super.tearDown()
    }

    func testCreateClipSendsAuthHeaderAndDecodesPresignedTarget() async throws {
        StubURLProtocol.box.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/clips")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer token-abc")
            return StubbedResponse(statusCode: 201, json: [
                "id": "clip_1",
                "filename": "session.mov",
                "key": "raw/user_1/clip_1/session.mov",
                "upload_id": "upload_1",
                "part_count": 2,
                "part_urls": [
                    ["part_number": 1, "url": "https://s3.example.test/part1"],
                    ["part_number": 2, "url": "https://s3.example.test/part2"],
                ],
                "sidecar_upload_url": "https://s3.example.test/sidecar",
            ])
        }
        let client = PresignedUploadClient(
            baseURL: URL(string: "https://api.example.test")!,
            session: makeStubbedSession(),
            accessTokenProvider: { "token-abc" }
        )

        let target = try await client.createClip(filename: "session.mov", sizeBytes: 10, partSizeBytes: 5)

        XCTAssertEqual(target.id, "clip_1")
        XCTAssertEqual(target.uploadId, "upload_1")
        XCTAssertEqual(target.partCount, 2)
        XCTAssertEqual(target.partUrls.map(\.partNumber), [1, 2])
        XCTAssertEqual(target.sidecarUploadUrl, "https://s3.example.test/sidecar")

        let body = try XCTUnwrap(StubURLProtocol.box.capturedRequests.first?.bodyJSON)
        XCTAssertEqual(body["filename"] as? String, "session.mov")
        XCTAssertEqual(body["size_bytes"] as? Int, 10)
        XCTAssertEqual(body["part_size_bytes"] as? Int, 5)
    }

    func testUploadPartsPutsEachChunkInOrderWithCorrectBytesAndCollectsETags() async throws {
        // 10 bytes, 4-byte chunks -> 3 parts of length 4, 4, 2.
        let fileBytes = Data("0123456789".utf8)
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("upload-\(UUID().uuidString).bin")
        try fileBytes.write(to: fileURL)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let plan = try ResumableChunkPlan(relativePath: "clips/session.mov", fileSizeBytes: 10, chunkSizeBytes: 4)
        // Provide part URLs out of order to prove the client re-sorts by part number, not array order.
        let partURLs = [
            PresignedUploadPartURL(partNumber: 3, url: "https://s3.example.test/part3"),
            PresignedUploadPartURL(partNumber: 1, url: "https://s3.example.test/part1"),
            PresignedUploadPartURL(partNumber: 2, url: "https://s3.example.test/part2"),
        ]

        StubURLProtocol.box.handler = { request in
            XCTAssertEqual(request.httpMethod, "PUT")
            let path = request.url?.path ?? ""
            let etag = "etag-for-\(path.split(separator: "/").last ?? "")"
            return StubbedResponse(statusCode: 200, headers: ["ETag": etag])
        }

        let client = PresignedUploadClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession())
        let completed = try await client.uploadParts(plan: plan, fileURL: fileURL, partURLs: partURLs)

        // `capturedRequests` is itself thread-safe (lock-protected in StubURLProtocolBox)
        // and preserves call order, so we derive the observed PUT order from it rather
        // than capturing a mutable local inside the @Sendable stub closure.
        let seenPathsInOrder = StubURLProtocol.box.capturedRequests.map { $0.request.url?.path ?? "" }
        XCTAssertEqual(seenPathsInOrder, ["/part1", "/part2", "/part3"], "parts must be PUT in ascending part-number order")
        XCTAssertEqual(completed.map(\.partNumber), [1, 2, 3])
        XCTAssertEqual(completed.map(\.etag), ["etag-for-part1", "etag-for-part2", "etag-for-part3"])

        let bodies = StubURLProtocol.box.capturedRequests.map(\.bodyData)
        XCTAssertEqual(bodies[0], Data("0123".utf8))
        XCTAssertEqual(bodies[1], Data("4567".utf8))
        XCTAssertEqual(bodies[2], Data("89".utf8))
    }

    func testUploadPartsThrowsOnPartCountMismatchWithoutMakingAnyRequest() async throws {
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("upload-\(UUID().uuidString).bin")
        try Data("0123456789".utf8).write(to: fileURL)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let plan = try ResumableChunkPlan(relativePath: "clips/session.mov", fileSizeBytes: 10, chunkSizeBytes: 4)

        StubURLProtocol.box.handler = { _ in
            XCTFail("must not PUT any part when the part-URL count does not match the chunk plan")
            return StubbedResponse(statusCode: 500)
        }
        let client = PresignedUploadClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession())

        do {
            _ = try await client.uploadParts(
                plan: plan,
                fileURL: fileURL,
                partURLs: [PresignedUploadPartURL(partNumber: 1, url: "https://s3.example.test/part1")]
            )
            XCTFail("expected partCountMismatch")
        } catch PresignedUploadClientError.partCountMismatch(let expected, let got) {
            XCTAssertEqual(expected, 3)
            XCTAssertEqual(got, 1)
        }
    }

    func testUploadPartsThrowsWhenSuccessfulResponseHasNoETagHeader() async throws {
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("upload-\(UUID().uuidString).bin")
        try Data("0123".utf8).write(to: fileURL)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let plan = try ResumableChunkPlan(relativePath: "clips/session.mov", fileSizeBytes: 4, chunkSizeBytes: 4)

        StubURLProtocol.box.handler = { _ in
            StubbedResponse(statusCode: 200) // no ETag header
        }
        let client = PresignedUploadClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession())

        do {
            _ = try await client.uploadParts(
                plan: plan,
                fileURL: fileURL,
                partURLs: [PresignedUploadPartURL(partNumber: 1, url: "https://s3.example.test/part1")]
            )
            XCTFail("expected missingETag")
        } catch PresignedUploadClientError.missingETag(let partNumber) {
            XCTAssertEqual(partNumber, 1)
        }
    }

    func testCompleteClipAssemblesPartsSortedByPartNumberFromStubbedETagsRegardlessOfInputOrder() async throws {
        StubURLProtocol.box.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/clips/clip_1/complete")
            return StubbedResponse(statusCode: 200, json: [
                "id": "clip_1",
                "status": "uploaded",
                "key": "raw/user_1/clip_1/session.mov",
            ])
        }
        let client = PresignedUploadClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession())
        let unsortedParts = [
            CompletedUploadPart(partNumber: 3, etag: "etag-3"),
            CompletedUploadPart(partNumber: 1, etag: "etag-1"),
            CompletedUploadPart(partNumber: 2, etag: "etag-2"),
        ]

        let result = try await client.completeClip(clipID: "clip_1", uploadID: "upload_1", parts: unsortedParts)

        XCTAssertEqual(result.id, "clip_1")
        XCTAssertEqual(result.status, "uploaded")

        let body = try XCTUnwrap(StubURLProtocol.box.capturedRequests.first?.bodyJSON)
        XCTAssertEqual(body["upload_id"] as? String, "upload_1")
        let parts = try XCTUnwrap(body["parts"] as? [[String: Any]])
        XCTAssertEqual(parts.map { $0["part_number"] as? Int }, [1, 2, 3])
        XCTAssertEqual(parts.map { $0["etag"] as? String }, ["etag-1", "etag-2", "etag-3"])
    }

    func testCompleteClipSurfacesServerFailureDetail() async throws {
        StubURLProtocol.box.handler = { _ in
            StubbedResponse(statusCode: 400, json: ["detail": "multipart completion failed: NoSuchUpload"])
        }
        let client = PresignedUploadClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession())

        do {
            _ = try await client.completeClip(clipID: "clip_1", uploadID: "stale-upload", parts: [CompletedUploadPart(partNumber: 1, etag: "e1")])
            XCTFail("expected completeClip to throw")
        } catch PresignedUploadClientError.httpStatus(let status, let message) {
            XCTAssertEqual(status, 400)
            XCTAssertTrue(message.contains("NoSuchUpload"))
        }
    }
}
