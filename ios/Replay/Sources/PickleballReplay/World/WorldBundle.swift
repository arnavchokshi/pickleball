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

    public init(manifest: WorldViewerManifest, world: VirtualWorld, bodyMesh: BodyMesh?, contactWindows: ContactWindows?) {
        self.manifest = manifest
        self.world = world
        self.bodyMesh = bodyMesh
        self.contactWindows = contactWindows
    }
}

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
        let baseURL = manifestURL.deletingLastPathComponent()
        let world = try VirtualWorld.load(from: resolve(manifest.virtualWorldURL, against: baseURL), decoder: decoder)
        let bodyMesh = try manifest.bodyMeshURL.map { try BodyMesh.load(from: resolve($0, against: baseURL), decoder: decoder) }
        let contactWindows = try manifest.contactWindowsURL.map { try ContactWindows.load(from: resolve($0, against: baseURL), decoder: decoder) }
        return WorldBundle(manifest: manifest, world: world, bodyMesh: bodyMesh, contactWindows: contactWindows)
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

    private static func resolve(_ relativePath: String, against baseURL: URL) -> URL {
        URL(fileURLWithPath: relativePath, relativeTo: baseURL).standardizedFileURL
    }
}
