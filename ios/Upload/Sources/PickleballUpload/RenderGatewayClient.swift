import Foundation

public struct RenderGatewayUploadRequest: Equatable, Sendable {
    public var videoURL: URL
    public var captureSidecarURL: URL?
    public var courtCornersURL: URL?
    public var courtAssistSeedURL: URL?
    public var clip: String?
    public var maxFrames: Int?

    public init(
        videoURL: URL,
        captureSidecarURL: URL? = nil,
        courtCornersURL: URL? = nil,
        courtAssistSeedURL: URL? = nil,
        clip: String? = nil,
        maxFrames: Int? = nil
    ) {
        self.videoURL = videoURL
        self.captureSidecarURL = captureSidecarURL
        self.courtCornersURL = courtCornersURL
        self.courtAssistSeedURL = courtAssistSeedURL
        self.clip = clip
        self.maxFrames = maxFrames
    }
}

public struct RenderGatewayMultipartBody: Equatable, Sendable {
    public var fileURL: URL
    public var contentType: String

    public init(fileURL: URL, contentType: String) {
        self.fileURL = fileURL
        self.contentType = contentType
    }
}

public enum RenderGatewayMultipartBodyWriter {
    public static func write(
        _ upload: RenderGatewayUploadRequest,
        boundary: String = "pickleball-\(UUID().uuidString)",
        directory: URL = FileManager.default.temporaryDirectory
    ) throws -> RenderGatewayMultipartBody {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let bodyURL = directory.appendingPathComponent("render-upload-\(UUID().uuidString).multipart")
        try Data().write(to: bodyURL)
        let handle = try FileHandle(forWritingTo: bodyURL)
        defer { try? handle.close() }

        func append(_ value: String) {
            handle.write(Data(value.utf8))
        }

        func appendField(name: String, value: String) {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            append("\(value)\r\n")
        }

        func appendFile(name: String, url: URL, mimeType: String) throws {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(safeFilename(url.lastPathComponent))\"\r\n")
            append("Content-Type: \(mimeType)\r\n\r\n")
            handle.write(try Data(contentsOf: url))
            append("\r\n")
        }

        try appendFile(name: "video", url: upload.videoURL, mimeType: videoMimeType(for: upload.videoURL))
        if let captureSidecarURL = upload.captureSidecarURL {
            try appendFile(name: "capture_sidecar", url: captureSidecarURL, mimeType: "application/json")
        }
        if let courtCornersURL = upload.courtCornersURL {
            try appendFile(name: "court_corners", url: courtCornersURL, mimeType: "application/json")
        }
        if let courtAssistSeedURL = upload.courtAssistSeedURL {
            try appendFile(name: "court_assist_seed", url: courtAssistSeedURL, mimeType: "application/json")
        }
        if let clip = upload.clip?.trimmingCharacters(in: .whitespacesAndNewlines), !clip.isEmpty {
            appendField(name: "clip", value: clip)
        }
        if let maxFrames = upload.maxFrames {
            appendField(name: "max_frames", value: "\(maxFrames)")
        }
        append("--\(boundary)--\r\n")

        return RenderGatewayMultipartBody(fileURL: bodyURL, contentType: "multipart/form-data; boundary=\(boundary)")
    }

    private static func safeFilename(_ value: String) -> String {
        value.replacingOccurrences(of: "\"", with: "_")
    }

    private static func videoMimeType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "mov":
            return "video/quicktime"
        case "m4v":
            return "video/x-m4v"
        default:
            return "video/mp4"
        }
    }
}

public enum RenderGatewayJobStatus: String, Codable, Equatable, Sendable {
    case queued
    case running
    case complete
    case submitted
    case failed
}

public struct RenderGatewayJobStep: Codable, Equatable, Sendable {
    public var id: String
    public var label: String
    public var status: String
}

public struct RenderGatewayJobProgress: Codable, Equatable, Sendable {
    public var percent: Int
    public var stage: String
    public var message: String?
    public var etaSeconds: Int?
    public var steps: [RenderGatewayJobStep]?

    public var fractionComplete: Double {
        Double(max(0, min(100, percent))) / 100.0
    }

    public var etaText: String {
        guard let etaSeconds else {
            return "ETA calculating"
        }
        if etaSeconds < 60 {
            return "less than 1 min"
        }
        return "about \(max(1, Int((Double(etaSeconds) / 60.0).rounded()))) min"
    }
}

public struct RenderGatewayJobResult: Codable, Equatable, Sendable {
    public var manifestUrl: String?
    public var notes: [String]?
    public var remoteRunDir: String?
}

public struct RenderGatewayJobLinks: Codable, Equatable, Sendable {
    public var status: String
    public var manifest: String?
}

public struct RenderGatewayJob: Codable, Equatable, Sendable {
    public var id: String
    public var clip: String?
    public var status: RenderGatewayJobStatus
    public var progress: RenderGatewayJobProgress?
    public var error: String?
    public var result: RenderGatewayJobResult?
    public var links: RenderGatewayJobLinks

    public var isActive: Bool {
        status == .queued || status == .running || status == .submitted
    }

    public static func decode(_ data: Data) throws -> RenderGatewayJob {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(RenderGatewayJob.self, from: data)
    }
}

public enum RenderGatewayClientError: Error, Equatable, Sendable {
    case invalidResponse
    case httpStatus(Int, String)
}

public final class RenderGatewayClient: @unchecked Sendable {
    public static let defaultBaseURL = URL(string: "https://pickleball-gpu-gateway.onrender.com")!

    private let baseURL: URL
    private let session: URLSession
    private let accessTokenProvider: (@Sendable () -> String?)?

    public init(
        baseURL: URL = RenderGatewayClient.defaultBaseURL,
        session: URLSession = .shared,
        accessTokenProvider: (@Sendable () -> String?)? = nil
    ) {
        self.baseURL = baseURL
        self.session = session
        self.accessTokenProvider = accessTokenProvider
    }

    public func submitJob(upload: RenderGatewayUploadRequest) async throws -> RenderGatewayJob {
        let body = try RenderGatewayMultipartBodyWriter.write(upload)
        defer { try? FileManager.default.removeItem(at: body.fileURL) }

        var request = URLRequest(url: Self.apiURL(path: "/api/jobs", baseURL: baseURL))
        request.httpMethod = "POST"
        request.setValue(body.contentType, forHTTPHeaderField: "Content-Type")
        applyAuthHeader(&request)
        let (data, response) = try await session.upload(for: request, fromFile: body.fileURL)
        return try Self.decodeJobResponse(data: data, response: response)
    }

    public func fetchJobStatus(_ statusPath: String) async throws -> RenderGatewayJob {
        var request = URLRequest(url: Self.apiURL(path: statusPath, baseURL: baseURL))
        applyAuthHeader(&request)
        let (data, response) = try await session.data(for: request)
        return try Self.decodeJobResponse(data: data, response: response)
    }

    /// Sets `Authorization: Bearer <token>` when `accessTokenProvider` is
    /// configured and returns a non-empty token (INFRA-4). Existing callers
    /// that don't pass a provider are unaffected -- no header is added.
    private func applyAuthHeader(_ request: inout URLRequest) {
        guard let token = accessTokenProvider?(), !token.isEmpty else {
            return
        }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    public func replayURL(for job: RenderGatewayJob) -> URL? {
        guard let manifestPath = job.result?.manifestUrl else {
            return nil
        }
        let manifestURL = Self.apiURL(path: manifestPath, baseURL: baseURL)
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        components?.path = "/"
        components?.queryItems = [URLQueryItem(name: "manifest", value: manifestURL.absoluteString)]
        return components?.url
    }

    public static func apiURL(path: String, baseURL: URL) -> URL {
        if let absolute = URL(string: path), absolute.scheme != nil {
            return absolute
        }
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        components?.path = path.hasPrefix("/") ? path : "/\(path)"
        components?.query = nil
        return components?.url ?? baseURL.appendingPathComponent(path)
    }

    private static func decodeJobResponse(data: Data, response: URLResponse) throws -> RenderGatewayJob {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw RenderGatewayClientError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "request failed"
            throw RenderGatewayClientError.httpStatus(httpResponse.statusCode, message)
        }
        return try RenderGatewayJob.decode(data)
    }
}
