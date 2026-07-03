import Foundation
import Combine
import SceneKit

/// Drives the GLUE-4 world-viewer screen: owns the loaded bundle, the
/// scrub position, camera preset, and the low-confidence dim toggle, and
/// keeps the SceneKit scene in sync. All gate/tier logic is delegated to
/// `WorldFrameSnapshotBuilder`/`WorldSceneBuilder` -- this class is just
/// state plus wiring.
@MainActor
public final class WorldViewerViewModel: ObservableObject {
    @Published public private(set) var bundle: WorldBundle
    @Published public var cameraPreset: WorldCameraPreset = .broadcast {
        didSet { applyCameraPreset() }
    }
    @Published public var dimLowConfidence: Bool = true {
        didSet { applySnapshot() }
    }
    @Published public private(set) var snapshot: WorldFrameSnapshot

    public let timeline: ReplayTimelineModel
    public let sceneBuilder: WorldSceneBuilder

    private var timelineTimeCancellable: AnyCancellable?
    private var timelineStateCancellable: AnyCancellable?

    public var currentTime: Double {
        timeline.currentTime
    }

    public var durationSeconds: Double {
        timeline.durationSeconds
    }

    public var isPlaying: Bool {
        timeline.isPlaying
    }

    public init(bundle: WorldBundle, timeline: ReplayTimelineModel? = nil) {
        let startTime = bundle.world.players.flatMap { $0.frames.map(\.t) }.min() ?? 0
        let allTimes = bundle.world.players.flatMap { $0.frames.map(\.t) } + bundle.world.ball.frames.map(\.t)

        self.bundle = bundle
        self.timeline = timeline ?? ReplayTimelineModel(
            currentTime: startTime,
            durationSeconds: allTimes.max() ?? 0,
            preferredFrameRate: 15
        )
        self.sceneBuilder = WorldSceneBuilder(court: bundle.world.court, meshFaces: bundle.bodyMesh?.meshFaces ?? [])
        self.snapshot = WorldFrameSnapshotBuilder.build(bundle: bundle, at: self.timeline.currentTime)

        applyCameraPreset()
        applySnapshot()
        timelineTimeCancellable = self.timeline.$currentTime.sink { [weak self] timeSeconds in
            self?.applySnapshot(at: timeSeconds)
        }
        timelineStateCancellable = self.timeline.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
    }

    public func seek(to timeSeconds: Double) {
        timeline.seek(to: timeSeconds)
    }

    public func selectCameraPreset(_ preset: WorldCameraPreset) {
        cameraPreset = preset
    }

    public func togglePlayback() {
        timeline.togglePlayback()
    }

    private func applySnapshot() {
        applySnapshot(at: currentTime)
    }

    private func applySnapshot(at timeSeconds: Double) {
        snapshot = WorldFrameSnapshotBuilder.build(bundle: bundle, at: timeSeconds)
        sceneBuilder.apply(snapshot, dimLowConfidence: dimLowConfidence)
    }

    private func applyCameraPreset() {
        let pose = WorldCameraPlanner.pose(for: cameraPreset, court: bundle.world.court)
        sceneBuilder.cameraNode.simdPosition = SIMD3(Float(pose.position.x), Float(pose.position.y), Float(pose.position.z))
        sceneBuilder.cameraNode.look(
            at: WorldSceneGeometry.scnVector(pose.target),
            up: SCNVector3(0, 0, 1),
            localFront: SCNVector3(0, 0, -1)
        )
    }
}
