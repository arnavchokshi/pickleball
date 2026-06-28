import Foundation

public struct ReplayAsset: Codable, Equatable, Sendable {
    public var usdzURL: URL?
    public var glbURL: URL?
    public var durationSeconds: Double

    public init(usdzURL: URL? = nil, glbURL: URL? = nil, durationSeconds: Double) {
        self.usdzURL = usdzURL
        self.glbURL = glbURL
        self.durationSeconds = durationSeconds
    }
}

public enum ReplayAssetValidationError: Equatable, Sendable {
    case missingRenderableAsset
    case invalidDuration(Double)
    case unsupportedScheme(String)
    case unexpectedExtension(format: ReplayAssetFormat, actual: String)
}

public struct ReplayAssetValidationReport: Equatable, Sendable {
    public var errors: [ReplayAssetValidationError]

    public var isValid: Bool {
        errors.isEmpty
    }

    public init(errors: [ReplayAssetValidationError]) {
        self.errors = errors
    }
}

public enum ReplayAssetValidator {
    public static func validate(_ asset: ReplayAsset) -> ReplayAssetValidationReport {
        var errors: [ReplayAssetValidationError] = []

        if asset.usdzURL == nil && asset.glbURL == nil {
            errors.append(.missingRenderableAsset)
        }

        if asset.durationSeconds <= 0 || !asset.durationSeconds.isFinite {
            errors.append(.invalidDuration(asset.durationSeconds))
        }

        validate(asset.usdzURL, expectedFormat: .usdz, errors: &errors)
        validate(asset.glbURL, expectedFormat: .glb, errors: &errors)

        return ReplayAssetValidationReport(errors: errors)
    }

    private static func validate(
        _ url: URL?,
        expectedFormat: ReplayAssetFormat,
        errors: inout [ReplayAssetValidationError]
    ) {
        guard let url else {
            return
        }

        if let scheme = url.scheme?.lowercased(), !["http", "https"].contains(scheme) {
            errors.append(.unsupportedScheme(scheme))
        }

        let pathExtension = url.pathExtension.lowercased()
        if pathExtension != expectedFormat.rawValue {
            errors.append(.unexpectedExtension(format: expectedFormat, actual: pathExtension))
        }
    }
}

public enum ReplayAssetFormat: String, Codable, Equatable, Sendable {
    case usdz
    case glb
}

public enum ReplayAssetRole: String, Codable, Equatable, Sendable {
    case nativeRealityKit
    case webShare
}

public struct ReplayAssetReference: Codable, Equatable, Sendable {
    public var format: ReplayAssetFormat
    public var role: ReplayAssetRole
    public var url: URL
    public var pathExtension: String

    public init(format: ReplayAssetFormat, role: ReplayAssetRole, url: URL) {
        self.format = format
        self.role = role
        self.url = url
        self.pathExtension = url.pathExtension.lowercased()
    }

    public static func references(for asset: ReplayAsset) -> [ReplayAssetReference] {
        var references: [ReplayAssetReference] = []

        if let usdzURL = asset.usdzURL {
            references.append(ReplayAssetReference(format: .usdz, role: .nativeRealityKit, url: usdzURL))
        }

        if let glbURL = asset.glbURL {
            references.append(ReplayAssetReference(format: .glb, role: .webShare, url: glbURL))
        }

        return references
    }
}

public struct ReplayTimelinePoint: Codable, Equatable, Sendable {
    public var id: Int
    public var startSeconds: Double
    public var endSeconds: Double
    public var glbURL: URL
    public var sizeMB: Double

    public init(id: Int, startSeconds: Double, endSeconds: Double, glbURL: URL, sizeMB: Double) {
        self.id = id
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.glbURL = glbURL
        self.sizeMB = sizeMB
    }
}

public struct ReplayTimelineDescriptor: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var worldFrame: String
    public var fps: Double
    public var durationSeconds: Double
    public var points: [ReplayTimelinePoint]

    public init(
        schemaVersion: Int,
        worldFrame: String,
        fps: Double,
        durationSeconds: Double,
        points: [ReplayTimelinePoint]
    ) {
        self.schemaVersion = schemaVersion
        self.worldFrame = worldFrame
        self.fps = fps
        self.durationSeconds = durationSeconds
        self.points = points
    }
}

public enum ReplayValidationScope: String, Codable, Equatable, Sendable {
    case metadataOnly
}

public struct ReplayCapabilityDescriptor: Codable, Equatable, Sendable {
    public var supportsNativeUSDZ: Bool
    public var supportsWebGLB: Bool
    public var supportsTimelineScrubbing: Bool
    public var supportsFreeViewpointMetadata: Bool
    public var hasRealityKitRuntimeValidation: Bool
    public var validationScope: ReplayValidationScope

    public init(
        supportsNativeUSDZ: Bool,
        supportsWebGLB: Bool,
        supportsTimelineScrubbing: Bool,
        supportsFreeViewpointMetadata: Bool,
        hasRealityKitRuntimeValidation: Bool,
        validationScope: ReplayValidationScope
    ) {
        self.supportsNativeUSDZ = supportsNativeUSDZ
        self.supportsWebGLB = supportsWebGLB
        self.supportsTimelineScrubbing = supportsTimelineScrubbing
        self.supportsFreeViewpointMetadata = supportsFreeViewpointMetadata
        self.hasRealityKitRuntimeValidation = hasRealityKitRuntimeValidation
        self.validationScope = validationScope
    }

    public static func describe(
        asset: ReplayAsset,
        timeline: ReplayTimelineDescriptor
    ) -> ReplayCapabilityDescriptor {
        ReplayCapabilityDescriptor(
            supportsNativeUSDZ: asset.usdzURL != nil,
            supportsWebGLB: asset.glbURL != nil,
            supportsTimelineScrubbing: timeline.fps > 0 && timeline.durationSeconds > 0 && !timeline.points.isEmpty,
            supportsFreeViewpointMetadata: timeline.worldFrame == "court_Z0",
            hasRealityKitRuntimeValidation: false,
            validationScope: .metadataOnly
        )
    }
}
