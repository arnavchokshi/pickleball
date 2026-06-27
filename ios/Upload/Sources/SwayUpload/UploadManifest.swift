import Foundation
import SwayCore

public struct UploadManifest: Codable, Equatable, Sendable {
    public var clipRelativePath: String
    public var sidecar: CaptureSidecar
    public var onDevicePoseTrack: String?
    public var lidarDepthRefs: [String]

    public init(
        clipRelativePath: String,
        sidecar: CaptureSidecar,
        onDevicePoseTrack: String? = nil,
        lidarDepthRefs: [String] = []
    ) {
        self.clipRelativePath = clipRelativePath
        self.sidecar = sidecar
        self.onDevicePoseTrack = onDevicePoseTrack
        self.lidarDepthRefs = lidarDepthRefs
    }
}

public enum UploadManifestValidationError: Equatable, Sendable {
    case unsupportedSchemaVersion(Int)
    case emptyRelativePath(field: String)
    case unsafeRelativePath(field: String, value: String)
    case unsupportedClipExtension(String)
    case invalidFPS(Int)
    case invalidResolution([Int])
    case invalidGravity([Double])
}

public struct UploadManifestValidationReport: Equatable, Sendable {
    public var errors: [UploadManifestValidationError]

    public var isValid: Bool {
        errors.isEmpty
    }

    public init(errors: [UploadManifestValidationError]) {
        self.errors = errors
    }
}

public enum UploadManifestValidator {
    public static func validate(_ manifest: UploadManifest) -> UploadManifestValidationReport {
        var errors: [UploadManifestValidationError] = []

        if manifest.sidecar.schemaVersion != 1 {
            errors.append(.unsupportedSchemaVersion(manifest.sidecar.schemaVersion))
        }

        appendPathIssue(
            field: "clipRelativePath",
            path: manifest.clipRelativePath,
            to: &errors
        )

        let clipExtension = (manifest.clipRelativePath as NSString).pathExtension.lowercased()
        if !manifest.clipRelativePath.isEmpty && !["mov", "mp4", "m4v"].contains(clipExtension) {
            errors.append(.unsupportedClipExtension(clipExtension))
        }

        if let poseTrack = manifest.onDevicePoseTrack {
            appendPathIssue(field: "onDevicePoseTrack", path: poseTrack, to: &errors)
        }

        for (index, depthRef) in manifest.lidarDepthRefs.enumerated() {
            appendPathIssue(field: "lidarDepthRefs[\(index)]", path: depthRef, to: &errors)
        }

        if manifest.sidecar.fps <= 0 {
            errors.append(.invalidFPS(manifest.sidecar.fps))
        }

        if manifest.sidecar.resolution.count != 2 || manifest.sidecar.resolution.contains(where: { $0 <= 0 }) {
            errors.append(.invalidResolution(manifest.sidecar.resolution))
        }

        if manifest.sidecar.gravity.count != 3 {
            errors.append(.invalidGravity(manifest.sidecar.gravity))
        }

        return UploadManifestValidationReport(errors: errors)
    }

    static func isSafeRelativePath(_ path: String) -> Bool {
        guard !path.isEmpty else {
            return false
        }

        if path.hasPrefix("/") || path.hasPrefix("~") || path.contains("://") {
            return false
        }

        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        return components.allSatisfy { component in
            component != "." && component != ".." && !component.isEmpty
        }
    }

    private static func appendPathIssue(
        field: String,
        path: String,
        to errors: inout [UploadManifestValidationError]
    ) {
        if path.isEmpty {
            errors.append(.emptyRelativePath(field: field))
        } else if !isSafeRelativePath(path) {
            errors.append(.unsafeRelativePath(field: field, value: path))
        }
    }
}

public struct UploadPart: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case captureSidecar
        case posePrior
        case lidarDepth
        case clip
    }

    public var kind: Kind
    public var relativePath: String
    public var priority: Int

    public init(kind: Kind, relativePath: String, priority: Int) {
        self.kind = kind
        self.relativePath = relativePath
        self.priority = priority
    }
}

public enum UploadPlanningError: Error, Equatable, Sendable {
    case invalidManifest([UploadManifestValidationError])
    case invalidSidecarPath(String)
    case invalidFileSize(Int64)
    case invalidChunkSize(Int64)
}

public enum UploadPlan {
    public static func sidecarFirstParts(
        for manifest: UploadManifest,
        sidecarRelativePath: String
    ) throws -> [UploadPart] {
        let report = UploadManifestValidator.validate(manifest)
        guard report.isValid else {
            throw UploadPlanningError.invalidManifest(report.errors)
        }
        guard UploadManifestValidator.isSafeRelativePath(sidecarRelativePath) else {
            throw UploadPlanningError.invalidSidecarPath(sidecarRelativePath)
        }

        var parts = [
            UploadPart(kind: .captureSidecar, relativePath: sidecarRelativePath, priority: 0)
        ]

        if let poseTrack = manifest.onDevicePoseTrack ?? manifest.sidecar.ondevicePoseTrack {
            parts.append(UploadPart(kind: .posePrior, relativePath: poseTrack, priority: 10))
        }

        for depthRef in manifest.lidarDepthRefs.sorted() {
            parts.append(UploadPart(kind: .lidarDepth, relativePath: depthRef, priority: 20))
        }

        parts.append(UploadPart(kind: .clip, relativePath: manifest.clipRelativePath, priority: 100))
        return parts
    }
}

public struct ResumableChunkPlan: Equatable, Sendable {
    public var relativePath: String
    public var fileSizeBytes: Int64
    public var chunkSizeBytes: Int64
    public var chunks: [UploadChunk]

    public init(
        relativePath: String,
        fileSizeBytes: Int64,
        chunkSizeBytes: Int64
    ) throws {
        guard UploadManifestValidator.isSafeRelativePath(relativePath) else {
            throw UploadPlanningError.invalidSidecarPath(relativePath)
        }
        guard fileSizeBytes > 0 else {
            throw UploadPlanningError.invalidFileSize(fileSizeBytes)
        }
        guard chunkSizeBytes > 0 else {
            throw UploadPlanningError.invalidChunkSize(chunkSizeBytes)
        }

        self.relativePath = relativePath
        self.fileSizeBytes = fileSizeBytes
        self.chunkSizeBytes = chunkSizeBytes

        var plannedChunks: [UploadChunk] = []
        var offset: Int64 = 0
        var index = 0
        while offset < fileSizeBytes {
            let length = min(chunkSizeBytes, fileSizeBytes - offset)
            plannedChunks.append(
                UploadChunk(
                    index: index,
                    offsetBytes: offset,
                    lengthBytes: length,
                    identifier: "\(relativePath):\(index):\(offset):\(length)"
                )
            )
            offset += length
            index += 1
        }
        self.chunks = plannedChunks
    }
}

public struct UploadChunk: Equatable, Sendable {
    public var index: Int
    public var offsetBytes: Int64
    public var lengthBytes: Int64
    public var identifier: String

    public init(index: Int, offsetBytes: Int64, lengthBytes: Int64, identifier: String) {
        self.index = index
        self.offsetBytes = offsetBytes
        self.lengthBytes = lengthBytes
        self.identifier = identifier
    }
}
