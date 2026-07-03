import Combine
import Foundation

@MainActor
public final class ReplayTimelineModel: ObservableObject {
    @Published public private(set) var currentTime: Double
    @Published public private(set) var isPlaying: Bool

    public let durationSeconds: Double
    public let preferredFrameRate: Double

    private var playbackTask: Task<Void, Never>?

    public init(currentTime: Double = 0, durationSeconds: Double, preferredFrameRate: Double = 15) {
        self.durationSeconds = max(0, durationSeconds)
        self.preferredFrameRate = max(1, preferredFrameRate)
        self.currentTime = min(max(0, currentTime), self.durationSeconds)
        self.isPlaying = false
    }

    public func seek(to timeSeconds: Double) {
        currentTime = min(max(0, timeSeconds), durationSeconds)
    }

    public func togglePlayback() {
        isPlaying ? pause() : play()
    }

    public func play() {
        guard durationSeconds > 0 else { return }
        isPlaying = true
        startPlaybackLoop()
    }

    public func pause() {
        isPlaying = false
        playbackTask?.cancel()
        playbackTask = nil
    }

    private func startPlaybackLoop() {
        playbackTask?.cancel()
        playbackTask = Task { @MainActor [weak self] in
            while let self, !Task.isCancelled, self.isPlaying {
                let next = self.currentTime + (1.0 / self.preferredFrameRate)
                if next >= self.durationSeconds {
                    self.seek(to: 0)
                    self.pause()
                    return
                }
                self.seek(to: next)
                try? await Task.sleep(nanoseconds: UInt64((1.0 / self.preferredFrameRate) * 1_000_000_000))
            }
        }
    }
}
