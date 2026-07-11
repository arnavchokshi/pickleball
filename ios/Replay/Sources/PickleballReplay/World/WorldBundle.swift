import Foundation

/// One loaded "world" -- everything the GLUE-4 iOS viewer needs to render a
/// clip: the manifest, the joints/track-tier `virtual_world.json`, the
/// optional mesh-tier `body_mesh.json`, and the optional
/// `contact_windows.json` used both for timeline markers and MESH-tier
/// gating. This is the native-app equivalent of the web viewer's
/// `App.tsx` load effect (`parseViewerManifest` -> `parseVirtualWorld` ->
/// `parseBodyMesh` / `parseContactWindows`).
public struct WorldBundle: Equatable, Sendable {
    public var manifest: WorldViewerManifest
    public var world: VirtualWorld
    public var bodyMesh: BodyMesh?
    public var contactWindows: ContactWindows?
    public var assetIssues: [WorldBundleAssetIssue]

    public init(
        manifest: WorldViewerManifest,
        world: VirtualWorld,
        bodyMesh: BodyMesh?,
        contactWindows: ContactWindows?,
        assetIssues: [WorldBundleAssetIssue] = []
    ) {
        self.manifest = manifest
        self.world = world
        self.bodyMesh = bodyMesh
        self.contactWindows = contactWindows
        self.assetIssues = assetIssues
    }
}

public struct WorldBundleAssetIssue: Equatable, Sendable {
    public var manifestField: String
    public var url: URL
    public var message: String

    public init(manifestField: String, url: URL, message: String) {
        self.manifestField = manifestField
        self.url = url
        self.message = message
    }
}

public typealias WorldBundleDataLoader = @Sendable (URL) async throws -> Data

public enum WorldBundleError: Error, Equatable {
    case missingManifestResource
    case missingArtifactResource(String)
}

extension WorldBundle {
    /// Loads a world bundle from a manifest file at `manifestURL`. Sibling
    /// artifact URLs in the manifest are resolved relative to the
    /// manifest's own directory (so a bundle can be copied anywhere --
    /// on-device Documents, a review-server mount, or the app bundle --
    /// without hardcoded absolute paths).
    public static func load(manifestURL: URL, decoder: JSONDecoder = JSONDecoder()) throws -> WorldBundle {
        let manifest = try WorldViewerManifest.load(from: manifestURL, decoder: decoder)
        let worldURL = resolve(manifest.virtualWorldURL, relativeTo: manifestURL)
        let world = try VirtualWorld.load(from: worldURL, decoder: decoder)
        var issues: [WorldBundleAssetIssue] = []
        let bodyMesh: BodyMesh? = loadOptional(
            manifest.bodyMeshURL,
            field: "body_mesh_url",
            relativeTo: manifestURL,
            issues: &issues
        ) { try BodyMesh.load(from: $0, decoder: decoder) }
        let contactWindows: ContactWindows? = loadOptional(
            manifest.contactWindowsURL,
            field: "contact_windows_url",
            relativeTo: manifestURL,
            issues: &issues
        ) { try ContactWindows.load(from: $0, decoder: decoder) }
        return WorldBundle(
            manifest: manifest,
            world: world,
            bodyMesh: bodyMesh,
            contactWindows: contactWindows,
            assetIssues: issues
        )
    }

    /// Loads one authenticated/local manifest and every native-consumed asset
    /// relative to that exact manifest URL. Optional asset failures are
    /// exposed as issues; they never trigger fixture or cross-capture data.
    public static func load(
        manifestURL: URL,
        dataLoader: @escaping WorldBundleDataLoader
    ) async throws -> WorldBundle {
        let manifestData = try await dataLoader(manifestURL)
        let manifest = try WorldViewerManifest.decode(manifestData)
        let worldURL = resolve(manifest.virtualWorldURL, relativeTo: manifestURL)
        let worldData = try await dataLoader(worldURL)
        let world = try JSONDecoder().decode(VirtualWorld.self, from: worldData)

        var issues: [WorldBundleAssetIssue] = []
        let bodyMesh: BodyMesh? = await loadOptional(
            manifest.bodyMeshURL,
            field: "body_mesh_url",
            relativeTo: manifestURL,
            dataLoader: dataLoader,
            issues: &issues
        ) { try JSONDecoder().decode(BodyMesh.self, from: $0) }
        let contactWindows: ContactWindows? = await loadOptional(
            manifest.contactWindowsURL,
            field: "contact_windows_url",
            relativeTo: manifestURL,
            dataLoader: dataLoader,
            issues: &issues
        ) { try JSONDecoder().decode(ContactWindows.self, from: $0) }
        return WorldBundle(
            manifest: manifest,
            world: world,
            bodyMesh: bodyMesh,
            contactWindows: contactWindows,
            assetIssues: issues
        )
    }

    /// Loads the fixture bundled into the app resources
    /// (`Resources/WorldFixture/`) -- a compact excerpt of the verified
    /// Wolverine `process_video` glue run
    /// (`runs/process_video_glue_20260702T_live_wolverine2/...`), with the
    /// huge BODY mesh artifact intentionally omitted from app resources.
    public static func loadBundledSample(bundle: Bundle? = nil, decoder: JSONDecoder = JSONDecoder()) throws -> WorldBundle {
        let resourceBundle = bundle ?? Bundle.module
        guard let manifestURL = resourceBundle.url(forResource: "replay_viewer_manifest", withExtension: "json", subdirectory: "WorldFixture") else {
            throw WorldBundleError.missingManifestResource
        }
        return try load(manifestURL: manifestURL, decoder: decoder)
    }

    public static func resolve(_ reference: String, relativeTo documentURL: URL) -> URL {
        if let absolute = URL(string: reference), absolute.scheme != nil {
            return absolute
        }
        let baseURL = documentURL.deletingLastPathComponent()
        if documentURL.isFileURL {
            return URL(fileURLWithPath: reference, relativeTo: baseURL).standardizedFileURL
        }
        return URL(string: reference, relativeTo: baseURL)?.absoluteURL
            ?? baseURL.appendingPathComponent(reference)
    }

    private static func loadOptional<Value>(
        _ reference: String?,
        field: String,
        relativeTo manifestURL: URL,
        issues: inout [WorldBundleAssetIssue],
        load: (URL) throws -> Value
    ) -> Value? {
        guard let reference else { return nil }
        let url = resolve(reference, relativeTo: manifestURL)
        do {
            return try load(url)
        } catch {
            issues.append(WorldBundleAssetIssue(
                manifestField: field,
                url: url,
                message: String(describing: error)
            ))
            return nil
        }
    }

    private static func loadOptional<Value>(
        _ reference: String?,
        field: String,
        relativeTo manifestURL: URL,
        dataLoader: @escaping WorldBundleDataLoader,
        issues: inout [WorldBundleAssetIssue],
        decode: (Data) throws -> Value
    ) async -> Value? {
        guard let reference else { return nil }
        let url = resolve(reference, relativeTo: manifestURL)
        do {
            return try decode(try await dataLoader(url))
        } catch {
            issues.append(WorldBundleAssetIssue(
                manifestField: field,
                url: url,
                message: String(describing: error)
            ))
            return nil
        }
    }
}
