import Foundation

public struct RealityReplayAsset: Equatable, Sendable {
    public var clipID: String
    public var assetURL: URL
    public var sourceAssetPath: String
    public var sourceAssetByteCount: Int
    public var bundledAssetByteCount: Int
    public var timelineDurationSeconds: Double
    public var preferredFrameRate: Double
    public var usdStartTimeCode: Double
    public var usdEndTimeCode: Double
    public var usdTimeCodesPerSecond: Double
    public var badgeTitle: String
    public var badgeDetail: String

    public var usdTimelineStartSeconds: Double {
        usdStartTimeCode / usdTimeCodesPerSecond
    }

    public var usdTimelineEndSeconds: Double {
        usdEndTimeCode / usdTimeCodesPerSecond
    }

    public var animationDurationSeconds: Double {
        max(0, usdTimelineEndSeconds - usdTimelineStartSeconds)
    }
}

public enum RealityReplayAssetError: Error, Equatable {
    case missingManifestResource
    case missingAssetResource(String)
    case invalidAssetExtension(String)
}

private struct RealityReplayFixtureManifest: Decodable {
    var clipID: String
    var assetFilename: String
    var sourceAssetPath: String
    var sourceAssetByteCount: Int
    var timelineDurationSeconds: Double
    var preferredFrameRate: Double
    var usdStartTimeCode: Double
    var usdEndTimeCode: Double
    var usdTimeCodesPerSecond: Double
    var badgeTitle: String
    var badgeDetail: String

    enum CodingKeys: String, CodingKey {
        case clipID = "clip_id"
        case assetFilename = "asset_filename"
        case sourceAssetPath = "source_asset_path"
        case sourceAssetByteCount = "source_asset_byte_count"
        case timelineDurationSeconds = "timeline_duration_seconds"
        case preferredFrameRate = "preferred_frame_rate"
        case usdStartTimeCode = "usd_start_time_code"
        case usdEndTimeCode = "usd_end_time_code"
        case usdTimeCodesPerSecond = "usd_time_codes_per_second"
        case badgeTitle = "badge_title"
        case badgeDetail = "badge_detail"
    }
}

extension RealityReplayAsset {
    public static func loadBundledFixture(bundle: Bundle? = nil, decoder: JSONDecoder = JSONDecoder()) throws -> RealityReplayAsset {
        let resourceBundle = bundle ?? Bundle.module
        guard let manifestURL = resourceBundle.url(
            forResource: "reality_replay_manifest",
            withExtension: "json",
            subdirectory: "RealityReplayFixture"
        ) else {
            throw RealityReplayAssetError.missingManifestResource
        }

        let manifest = try decoder.decode(RealityReplayFixtureManifest.self, from: Data(contentsOf: manifestURL))
        let filenameURL = URL(fileURLWithPath: manifest.assetFilename)
        let assetName = filenameURL.deletingPathExtension().lastPathComponent
        let assetExtension = filenameURL.pathExtension
        guard assetExtension == "usdz" else {
            throw RealityReplayAssetError.invalidAssetExtension(assetExtension)
        }
        guard let assetURL = resourceBundle.url(
            forResource: assetName,
            withExtension: assetExtension,
            subdirectory: "RealityReplayFixture"
        ) else {
            throw RealityReplayAssetError.missingAssetResource(manifest.assetFilename)
        }

        let values = try assetURL.resourceValues(forKeys: [.fileSizeKey])
        return RealityReplayAsset(
            clipID: manifest.clipID,
            assetURL: assetURL,
            sourceAssetPath: manifest.sourceAssetPath,
            sourceAssetByteCount: manifest.sourceAssetByteCount,
            bundledAssetByteCount: values.fileSize ?? 0,
            timelineDurationSeconds: manifest.timelineDurationSeconds,
            preferredFrameRate: manifest.preferredFrameRate,
            usdStartTimeCode: manifest.usdStartTimeCode,
            usdEndTimeCode: manifest.usdEndTimeCode,
            usdTimeCodesPerSecond: manifest.usdTimeCodesPerSecond,
            badgeTitle: manifest.badgeTitle,
            badgeDetail: manifest.badgeDetail
        )
    }
}
