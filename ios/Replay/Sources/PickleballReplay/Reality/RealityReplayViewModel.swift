import Combine
import Foundation

@MainActor
public final class RealityReplayViewModel: ObservableObject {
    public let asset: RealityReplayAsset
    public let timeline: ReplayTimelineModel

    private var timelineCancellable: AnyCancellable?

    public init(asset: RealityReplayAsset, timeline: ReplayTimelineModel? = nil) throws {
        self.asset = asset
        self.timeline = timeline ?? ReplayTimelineModel(
            durationSeconds: asset.timelineDurationSeconds,
            preferredFrameRate: asset.preferredFrameRate
        )
        self.timelineCancellable = self.timeline.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
    }

    public var currentTime: Double {
        timeline.currentTime
    }

    public var durationSeconds: Double {
        timeline.durationSeconds
    }

    public var isPlaying: Bool {
        timeline.isPlaying
    }

    public var animationTimeSeconds: Double {
        animationTimeSeconds(for: currentTime)
    }

    public func animationTimeSeconds(for worldTimeSeconds: Double) -> Double {
        min(max(0, worldTimeSeconds - asset.usdTimelineStartSeconds), asset.animationDurationSeconds)
    }

    public func seek(to timeSeconds: Double) {
        timeline.seek(to: timeSeconds)
    }

    public func togglePlayback() {
        timeline.togglePlayback()
    }
}
