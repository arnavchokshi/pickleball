import Foundation
import PickleballCore

public struct CaptureLibraryItem: Equatable, Sendable, Identifiable {
    public var id: String { sessionID }
    public var sessionID: String
    public var clipRelativePath: String
    public var sidecarRelativePath: String
    public var provenance: CaptureProvenance
    public var durationSeconds: Double?
    public var fps: Int
    public var resolution: [Int]
    public var captureQualityGrade: CaptureQuality.Grade
    public var recordedAt: Date?

    public var isImported: Bool {
        provenance == .cameraRollImport
    }

    public var badgeText: String? {
        isImported ? "imported" : nil
    }

    public init(
        sessionID: String,
        clipRelativePath: String,
        sidecarRelativePath: String,
        provenance: CaptureProvenance,
        durationSeconds: Double?,
        fps: Int,
        resolution: [Int],
        captureQualityGrade: CaptureQuality.Grade,
        recordedAt: Date?
    ) {
        self.sessionID = sessionID
        self.clipRelativePath = clipRelativePath
        self.sidecarRelativePath = sidecarRelativePath
        self.provenance = provenance
        self.durationSeconds = durationSeconds
        self.fps = fps
        self.resolution = resolution
        self.captureQualityGrade = captureQualityGrade
        self.recordedAt = recordedAt
    }
}

public enum CaptureLibrary {
    public static func listPackages(
        packageRootURL: URL,
        fileManager: FileManager = .default
    ) throws -> [CaptureLibraryItem] {
        let capturesURL = packageRootURL.appendingPathComponent("captures", isDirectory: true)
        guard fileManager.fileExists(atPath: capturesURL.path) else {
            return []
        }

        let packageURLs = try fileManager.contentsOfDirectory(
            at: capturesURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        let items = packageURLs.compactMap { packageURL -> CaptureLibraryItem? in
            guard (try? packageURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else {
                return nil
            }
            return readPackage(packageRootURL: packageRootURL, packageURL: packageURL, fileManager: fileManager)
        }

        return items.sorted { lhs, rhs in
            switch (lhs.recordedAt, rhs.recordedAt) {
            case let (left?, right?):
                return left > right
            case (_?, nil):
                return true
            case (nil, _?):
                return false
            case (nil, nil):
                return lhs.sessionID > rhs.sessionID
            }
        }
    }

    private static func readPackage(
        packageRootURL: URL,
        packageURL: URL,
        fileManager: FileManager
    ) -> CaptureLibraryItem? {
        let sessionID = packageURL.lastPathComponent
        let clipURL = packageURL.appendingPathComponent("clip.mov")
        let sidecarURL = packageURL.appendingPathComponent("capture_sidecar.json")
        guard fileManager.fileExists(atPath: clipURL.path),
              fileManager.fileExists(atPath: sidecarURL.path),
              let data = try? Data(contentsOf: sidecarURL),
              let sidecar = try? JSONDecoder().decode(CaptureSidecar.self, from: data) else {
            return nil
        }

        return CaptureLibraryItem(
            sessionID: sessionID,
            clipRelativePath: relativePath(from: packageRootURL, to: clipURL),
            sidecarRelativePath: relativePath(from: packageRootURL, to: sidecarURL),
            provenance: sidecar.provenance,
            durationSeconds: sidecar.recordingDurationS,
            fps: sidecar.fps,
            resolution: sidecar.resolution,
            captureQualityGrade: sidecar.captureQuality.grade,
            recordedAt: sidecar.recordingStartedAt.flatMap(Self.iso8601Date)
        )
    }

    private static func relativePath(from rootURL: URL, to url: URL) -> String {
        let rootPath = rootURL.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath) else {
            return url.lastPathComponent
        }
        return String(path.dropFirst(rootPath.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    private static func iso8601Date(from value: String) -> Date? {
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: value) {
            return date
        }
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        return plain.date(from: value)
    }
}
