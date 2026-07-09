import Foundation

public enum CaptureUploadStateKind: String, Codable, Equatable, Sendable {
    case queued
    case uploading
    case uploaded
    case failed
}

/// Persisted beside `clip.mov` and `capture_sidecar.json` as
/// `upload_state.json`. The broad state and byte/part counters are the stable
/// contract; the remaining fields preserve server identity and honest status.
public struct CaptureUploadState: Codable, Equatable, Sendable {
    public var state: CaptureUploadStateKind
    public var clipId: String?
    public var uploadedParts: [Int]
    public var bytesUploaded: Int64
    public var totalBytes: Int64
    public var updatedAt: Date
    public var lastError: String?
    public var serverStatus: String?
    public var jobId: String?
    public var videoCompleted: Bool
    public var sidecarUploaded: Bool

    public init(
        state: CaptureUploadStateKind,
        clipId: String? = nil,
        uploadedParts: [Int] = [],
        bytesUploaded: Int64 = 0,
        totalBytes: Int64,
        updatedAt: Date = Date(),
        lastError: String? = nil,
        serverStatus: String? = nil,
        jobId: String? = nil,
        videoCompleted: Bool = false,
        sidecarUploaded: Bool = false
    ) {
        self.state = state
        self.clipId = clipId
        self.uploadedParts = uploadedParts
        self.bytesUploaded = bytesUploaded
        self.totalBytes = totalBytes
        self.updatedAt = updatedAt
        self.lastError = lastError
        self.serverStatus = serverStatus
        self.jobId = jobId
        self.videoCompleted = videoCompleted
        self.sidecarUploaded = sidecarUploaded
    }

    public var fractionCompleted: Double {
        guard totalBytes > 0 else { return 0 }
        return min(1, max(0, Double(bytesUploaded) / Double(totalBytes)))
    }

    private enum CodingKeys: String, CodingKey {
        case state
        case clipId = "clip_id"
        case uploadedParts = "uploaded_parts"
        case bytesUploaded = "bytes_uploaded"
        case totalBytes = "total_bytes"
        case updatedAt = "updated_at"
        case lastError = "last_error"
        case serverStatus = "server_status"
        case jobId = "job_id"
        case videoCompleted = "video_completed"
        case sidecarUploaded = "sidecar_uploaded"
    }
}

public struct CaptureUploadPackage: Equatable, Sendable, Identifiable {
    public var id: String { packageID }
    public var packageID: String
    public var packageDirectoryURL: URL
    public var videoURL: URL
    public var sidecarURL: URL

    public init(packageID: String, packageDirectoryURL: URL, videoURL: URL, sidecarURL: URL) {
        self.packageID = packageID
        self.packageDirectoryURL = packageDirectoryURL
        self.videoURL = videoURL
        self.sidecarURL = sidecarURL
    }

    public var stateURL: URL {
        packageDirectoryURL.appendingPathComponent(UploadQueue.stateFilename)
    }
}

public struct ServerClipRecord: Codable, Equatable, Sendable {
    public var id: String
    public var filename: String
    public var status: String
    public var sizeBytes: Int64
    public var key: String
    public var jobId: String?
    public var createdAt: String?

    private enum CodingKeys: String, CodingKey {
        case id, filename, status, key
        case sizeBytes = "size_bytes"
        case jobId = "job_id"
        case createdAt = "created_at"
    }
}

public protocol UploadQueueClient: Sendable {
    func createClip(filename: String, sizeBytes: Int64, partSizeBytes: Int64) async throws -> PresignedClipUploadTarget
    func uploadParts(
        plan: ResumableChunkPlan,
        fileURL: URL,
        partURLs: [PresignedUploadPartURL],
        contentType: String,
        onPartUploaded: @escaping @Sendable (CompletedUploadPart, Int64) async -> Void
    ) async throws -> [CompletedUploadPart]
    func completeClip(clipID: String, uploadID: String, parts: [CompletedUploadPart]) async throws -> PresignedUploadCompleteResult
    func uploadSidecar(data: Data, to sidecarUploadURL: String, contentType: String) async throws
    func listClips() async throws -> [ServerClipRecord]
}

public enum UploadQueueError: Error, Equatable, Sendable {
    case missingVideo(String)
    case missingSidecar(String)
    case invalidVideoSize(Int64)
    case noRetryableState(String)
}

/// A restart-safe, strictly serial upload actor. One actor drains one package
/// at a time; every transition is atomically persisted inside that package.
public actor UploadQueue {
    public static let stateFilename = "upload_state.json"
    public static let defaultPartSizeBytes: Int64 = 8 * 1024 * 1024

    private struct ActiveAttempt: Sendable {
        var target: PresignedClipUploadTarget
        var completedParts: [CompletedUploadPart]
    }

    private let client: any UploadQueueClient
    private let partSizeBytes: Int64
    private var pending: [CaptureUploadPackage] = []
    private var attempts: [String: ActiveAttempt] = [:]
    private var isDraining = false
    private var activePackageID: String?

    public init(client: any UploadQueueClient, partSizeBytes: Int64 = UploadQueue.defaultPartSizeBytes) {
        self.client = client
        self.partSizeBytes = partSizeBytes
    }

    @discardableResult
    public func enqueue(_ package: CaptureUploadPackage) throws -> CaptureUploadState {
        let totalBytes = try Self.validatedVideoSize(for: package)
        guard FileManager.default.fileExists(atPath: package.sidecarURL.path) else {
            throw UploadQueueError.missingSidecar(package.sidecarURL.path)
        }
        var state = (try? Self.readState(for: package))
            ?? CaptureUploadState(state: .queued, totalBytes: totalBytes)
        guard state.state != .uploaded else { return state }
        state.state = .queued
        state.totalBytes = totalBytes
        state.updatedAt = Date()
        state.lastError = nil
        try Self.writeState(state, for: package)
        appendPendingIfNeeded(package)
        return state
    }

    /// Re-queues only work that was queued/uploading when the process died.
    /// Failed uploads remain user-controlled retries; uploaded work is inert.
    @discardableResult
    public func resume(_ packages: [CaptureUploadPackage]) throws -> [String] {
        var resumed: [String] = []
        for package in packages {
            guard var state = try? Self.readState(for: package),
                  state.state == .queued || state.state == .uploading else {
                continue
            }
            state.state = .queued
            state.updatedAt = Date()
            state.lastError = nil
            try Self.writeState(state, for: package)
            appendPendingIfNeeded(package)
            resumed.append(package.packageID)
        }
        return resumed
    }

    @discardableResult
    public func retry(_ package: CaptureUploadPackage) throws -> CaptureUploadState {
        guard var state = try? Self.readState(for: package), state.state == .failed else {
            throw UploadQueueError.noRetryableState(package.packageID)
        }
        state.state = .queued
        state.updatedAt = Date()
        state.lastError = nil
        try Self.writeState(state, for: package)
        appendPendingIfNeeded(package)
        return state
    }

    public func processPending() async {
        guard !isDraining else { return }
        isDraining = true
        defer { isDraining = false }

        while !pending.isEmpty {
            let package = pending.removeFirst()
            activePackageID = package.packageID
            await process(package)
            activePackageID = nil
        }
    }

    @discardableResult
    public func refreshServerStatus(for package: CaptureUploadPackage) async -> CaptureUploadState? {
        guard var state = try? Self.readState(for: package),
              state.state == .uploaded,
              let clipID = state.clipId else {
            return try? Self.readState(for: package)
        }
        do {
            let clip = try await client.listClips().first { $0.id == clipID }
            if let clip {
                state.serverStatus = clip.status
                state.jobId = clip.jobId
                state.lastError = nil
            } else {
                state.lastError = "Uploaded clip \(clipID) was not returned by GET /api/clips"
            }
        } catch {
            state.lastError = "Status refresh failed: \(String(describing: error))"
        }
        state.updatedAt = Date()
        try? Self.writeState(state, for: package)
        return state
    }

    public nonisolated static func readState(for package: CaptureUploadPackage) throws -> CaptureUploadState {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(CaptureUploadState.self, from: Data(contentsOf: package.stateURL))
    }

    private func process(_ package: CaptureUploadPackage) async {
        do {
            var state = try Self.readState(for: package)
            state.state = .uploading
            state.updatedAt = Date()
            state.lastError = nil
            try Self.writeState(state, for: package)

            var attempt: ActiveAttempt
            if let existing = attempts[package.packageID] {
                attempt = existing
            } else {
                let target = try await client.createClip(
                    filename: package.videoURL.lastPathComponent,
                    sizeBytes: state.totalBytes,
                    partSizeBytes: partSizeBytes
                )
                attempt = ActiveAttempt(target: target, completedParts: [])
                attempts[package.packageID] = attempt
                state.clipId = target.id
                state.uploadedParts = []
                state.bytesUploaded = 0
                state.videoCompleted = false
                state.sidecarUploaded = false
                state.serverStatus = "uploading"
                state.updatedAt = Date()
                try Self.writeState(state, for: package)
            }

            if !state.videoCompleted {
                let plan = try ResumableChunkPlan(
                    relativePath: package.videoURL.lastPathComponent,
                    fileSizeBytes: state.totalBytes,
                    chunkSizeBytes: partSizeBytes
                )
                let completed = try await client.uploadParts(
                    plan: plan,
                    fileURL: package.videoURL,
                    partURLs: attempt.target.partUrls,
                    contentType: Self.videoContentType(for: package.videoURL),
                    onPartUploaded: { [weak self] part, byteCount in
                        await self?.recordUploadedPart(part, byteCount: byteCount, for: package)
                    }
                )
                attempt.completedParts = completed
                attempts[package.packageID] = attempt
                let completedResult = try await client.completeClip(
                    clipID: attempt.target.id,
                    uploadID: attempt.target.uploadId,
                    parts: completed
                )
                state = try Self.readState(for: package)
                state.videoCompleted = true
                state.serverStatus = completedResult.status
                state.updatedAt = Date()
                try Self.writeState(state, for: package)
            }

            if !state.sidecarUploaded {
                // Intentionally read the exact persisted bytes. No decode,
                // encode, regeneration, or mutation occurs in this path.
                let exactSidecarBytes = try Data(contentsOf: package.sidecarURL)
                try await client.uploadSidecar(
                    data: exactSidecarBytes,
                    to: attempt.target.sidecarUploadUrl,
                    contentType: "application/json"
                )
            }

            state = try Self.readState(for: package)
            state.state = .uploaded
            state.clipId = attempt.target.id
            state.bytesUploaded = state.totalBytes
            state.sidecarUploaded = true
            state.lastError = nil
            state.updatedAt = Date()
            try Self.writeState(state, for: package)
            attempts.removeValue(forKey: package.packageID)
            _ = await refreshServerStatus(for: package)
        } catch {
            var state = (try? Self.readState(for: package))
                ?? CaptureUploadState(state: .failed, totalBytes: 0)
            state.state = .failed
            state.updatedAt = Date()
            state.lastError = String(describing: error)
            try? Self.writeState(state, for: package)
        }
    }

    private func recordUploadedPart(
        _ part: CompletedUploadPart,
        byteCount: Int64,
        for package: CaptureUploadPackage
    ) {
        guard var state = try? Self.readState(for: package) else { return }
        if !state.uploadedParts.contains(part.partNumber) {
            state.uploadedParts.append(part.partNumber)
            state.uploadedParts.sort()
            state.bytesUploaded = min(state.totalBytes, state.bytesUploaded + byteCount)
        }
        state.updatedAt = Date()
        try? Self.writeState(state, for: package)
    }

    private func appendPendingIfNeeded(_ package: CaptureUploadPackage) {
        guard activePackageID != package.packageID else { return }
        guard !pending.contains(where: { $0.packageID == package.packageID }) else { return }
        pending.append(package)
    }

    private nonisolated static func validatedVideoSize(for package: CaptureUploadPackage) throws -> Int64 {
        guard FileManager.default.fileExists(atPath: package.videoURL.path) else {
            throw UploadQueueError.missingVideo(package.videoURL.path)
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: package.videoURL.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard size > 0 else { throw UploadQueueError.invalidVideoSize(size) }
        return size
    }

    private nonisolated static func writeState(_ state: CaptureUploadState, for package: CaptureUploadPackage) throws {
        try FileManager.default.createDirectory(at: package.packageDirectoryURL, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(state).write(to: package.stateURL, options: .atomic)
    }

    private nonisolated static func videoContentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "mov": return "video/quicktime"
        case "m4v": return "video/x-m4v"
        default: return "video/mp4"
        }
    }
}

public enum UploadAuthenticationDecision: Equatable, Sendable {
    case enqueue(String)
    case promptSignIn
}

/// Pure state machine used by the app's contextual sign-in-at-upload flow.
public struct UploadAuthenticationGate: Equatable, Sendable {
    public private(set) var pendingPackageID: String?

    public init(pendingPackageID: String? = nil) {
        self.pendingPackageID = pendingPackageID
    }

    public mutating func requestUpload(packageID: String, hasAccessToken: Bool) -> UploadAuthenticationDecision {
        if hasAccessToken {
            pendingPackageID = nil
            return .enqueue(packageID)
        }
        pendingPackageID = packageID
        return .promptSignIn
    }

    public mutating func completeSignIn() -> String? {
        defer { pendingPackageID = nil }
        return pendingPackageID
    }
}

public enum APIBaseURLResolver {
    public static let environmentKey = "DINKVISION_API_BASE_URL"
    public static let plistKey = "DINKVISION_API_BASE_URL"
    public static let launchArgumentKeys = ["-dinkvision.apiBaseURL", "--dinkvision-api-base-url"]

    public static func resolve(
        arguments: [String],
        environment: [String: String],
        infoDictionary: [String: Any],
        defaultURL: URL = RenderGatewayClient.defaultBaseURL
    ) -> URL {
        for key in launchArgumentKeys {
            if let value = value(after: key, in: arguments), let url = validatedURL(value) {
                return url
            }
        }
        if let value = environment[environmentKey], let url = validatedURL(value) {
            return url
        }
        if let value = infoDictionary[plistKey] as? String, let url = validatedURL(value) {
            return url
        }
        return defaultURL
    }

    private static func value(after key: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: key) else { return nil }
        let next = arguments.index(after: index)
        guard next < arguments.endIndex else { return nil }
        return arguments[next]
    }

    private static func validatedURL(_ value: String) -> URL? {
        guard let url = URL(string: value.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = url.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              url.host != nil else {
            return nil
        }
        return url
    }
}
