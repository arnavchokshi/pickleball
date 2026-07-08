import Foundation

/// One presigned S3 multipart-upload part URL, as returned by
/// `POST /api/clips` (`server/routes/clips.py::presign_multipart_put`).
public struct PresignedUploadPartURL: Codable, Equatable, Sendable {
    public var partNumber: Int
    public var url: String

    public init(partNumber: Int, url: String) {
        self.partNumber = partNumber
        self.url = url
    }
}

/// `POST /api/clips` response shape.
public struct PresignedClipUploadTarget: Codable, Equatable, Sendable {
    public var id: String
    public var filename: String
    public var key: String
    public var uploadId: String
    public var partCount: Int
    public var partUrls: [PresignedUploadPartURL]
    public var sidecarUploadUrl: String

    public init(
        id: String,
        filename: String,
        key: String,
        uploadId: String,
        partCount: Int,
        partUrls: [PresignedUploadPartURL],
        sidecarUploadUrl: String
    ) {
        self.id = id
        self.filename = filename
        self.key = key
        self.uploadId = uploadId
        self.partCount = partCount
        self.partUrls = partUrls
        self.sidecarUploadUrl = sidecarUploadUrl
    }
}

/// One completed part, ready for `POST /api/clips/{id}/complete`
/// (`CompletedPart` server-side: `part_number` + `etag`).
public struct CompletedUploadPart: Equatable, Sendable {
    public var partNumber: Int
    public var etag: String

    public init(partNumber: Int, etag: String) {
        self.partNumber = partNumber
        self.etag = etag
    }
}

/// `POST /api/clips/{id}/complete` response shape.
public struct PresignedUploadCompleteResult: Codable, Equatable, Sendable {
    public var id: String
    public var status: String
    public var key: String
}

public enum PresignedUploadClientError: Error, Equatable, Sendable {
    case invalidResponse
    case httpStatus(Int, String)
    case missingPartURL(Int)
    case missingETag(Int)
    case partCountMismatch(expected: Int, got: Int)
}

/// Presigned-multipart upload client (INFRA-4): `POST /api/clips` mints the
/// part URLs, each chunk (from `ResumableChunkPlan`) is PUT directly to S3,
/// and `POST /api/clips/{id}/complete` assembles the parts+ETags. Injectable
/// session for tests, mirroring `RenderGatewayClient`/`AuthApiClient`.
public final class PresignedUploadClient: @unchecked Sendable {
    public static let defaultBaseURL = RenderGatewayClient.defaultBaseURL

    private let baseURL: URL
    private let session: URLSession
    private let accessTokenProvider: @Sendable () -> String?

    public init(
        baseURL: URL = PresignedUploadClient.defaultBaseURL,
        session: URLSession = .shared,
        accessTokenProvider: @escaping @Sendable () -> String? = { nil }
    ) {
        self.baseURL = baseURL
        self.session = session
        self.accessTokenProvider = accessTokenProvider
    }

    /// `POST /api/clips`: creates the clip doc and mints presigned part URLs.
    public func createClip(filename: String, sizeBytes: Int64, partSizeBytes: Int64) async throws -> PresignedClipUploadTarget {
        var request = URLRequest(url: RenderGatewayClient.apiURL(path: "/api/clips", baseURL: baseURL))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        applyAuthHeader(&request)
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "filename": filename,
            "size_bytes": sizeBytes,
            "part_size_bytes": partSizeBytes,
        ])
        let (data, response) = try await session.data(for: request)
        try Self.assertSuccess(response: response, data: data)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(PresignedClipUploadTarget.self, from: data)
    }

    /// Reads `plan`'s chunks out of `fileURL` and PUTs each to its presigned
    /// part URL, in ascending part-number order, collecting the `ETag`
    /// response header per part. `partURLs` need not already be sorted.
    public func uploadParts(
        plan: ResumableChunkPlan,
        fileURL: URL,
        partURLs: [PresignedUploadPartURL],
        contentType: String = "application/octet-stream"
    ) async throws -> [CompletedUploadPart] {
        guard partURLs.count == plan.chunks.count else {
            throw PresignedUploadClientError.partCountMismatch(expected: plan.chunks.count, got: partURLs.count)
        }
        let partURLsByNumber = Dictionary(uniqueKeysWithValues: partURLs.map { ($0.partNumber, $0.url) })
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        var completed: [CompletedUploadPart] = []
        completed.reserveCapacity(plan.chunks.count)
        for chunk in plan.chunks.sorted(by: { $0.index < $1.index }) {
            let partNumber = chunk.index + 1
            guard let urlString = partURLsByNumber[partNumber], let url = URL(string: urlString) else {
                throw PresignedUploadClientError.missingPartURL(partNumber)
            }
            try handle.seek(toOffset: UInt64(chunk.offsetBytes))
            let chunkData = try handle.read(upToCount: Int(chunk.lengthBytes)) ?? Data()

            var request = URLRequest(url: url)
            request.httpMethod = "PUT"
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
            let (data, response) = try await session.upload(for: request, from: chunkData)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw PresignedUploadClientError.invalidResponse
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                let message = String(data: data, encoding: .utf8) ?? "part upload failed"
                throw PresignedUploadClientError.httpStatus(httpResponse.statusCode, message)
            }
            guard let etag = httpResponse.value(forHTTPHeaderField: "ETag") else {
                throw PresignedUploadClientError.missingETag(partNumber)
            }
            completed.append(CompletedUploadPart(partNumber: partNumber, etag: etag))
        }
        return completed
    }

    /// `POST /api/clips/{id}/complete`: sends parts sorted ascending by
    /// part number (S3's `CompleteMultipartUpload` requires ascending
    /// order; sorting here rather than trusting caller order keeps that
    /// invariant in one place).
    public func completeClip(
        clipID: String,
        uploadID: String,
        parts: [CompletedUploadPart]
    ) async throws -> PresignedUploadCompleteResult {
        var request = URLRequest(url: RenderGatewayClient.apiURL(path: "/api/clips/\(clipID)/complete", baseURL: baseURL))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        applyAuthHeader(&request)
        let sortedParts = parts.sorted { $0.partNumber < $1.partNumber }
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "upload_id": uploadID,
            "parts": sortedParts.map { ["part_number": $0.partNumber, "etag": $0.etag] },
        ])
        let (data, response) = try await session.data(for: request)
        try Self.assertSuccess(response: response, data: data)
        let decoder = JSONDecoder()
        return try decoder.decode(PresignedUploadCompleteResult.self, from: data)
    }

    /// Uploads the capture sidecar to the (non-multipart) presigned PUT URL
    /// `POST /api/clips` also mints.
    public func uploadSidecar(data: Data, to sidecarUploadURL: String, contentType: String = "application/json") async throws {
        guard let url = URL(string: sidecarUploadURL) else {
            throw PresignedUploadClientError.invalidResponse
        }
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        let (responseData, response) = try await session.upload(for: request, from: data)
        try Self.assertSuccess(response: response, data: responseData)
    }

    private func applyAuthHeader(_ request: inout URLRequest) {
        guard let token = accessTokenProvider(), !token.isEmpty else {
            return
        }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    private static func assertSuccess(response: URLResponse, data: Data) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw PresignedUploadClientError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "request failed"
            throw PresignedUploadClientError.httpStatus(httpResponse.statusCode, message)
        }
    }
}
