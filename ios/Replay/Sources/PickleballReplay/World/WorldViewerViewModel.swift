import Foundation
import Combine
import SceneKit

public enum WorldViewerPlaybackSpeed: String, CaseIterable, Equatable, Sendable {
    case normal = "1x"
    case double = "2x"
    case half = "0.5x"

    public var rate: Double {
        switch self {
        case .half: return 0.5
        case .normal: return 1.0
        case .double: return 2.0
        }
    }

    public var next: WorldViewerPlaybackSpeed {
        switch self {
        case .normal: return .double
        case .double: return .half
        case .half: return .normal
        }
    }
}

public protocol WorldViewerCoachMarkStoring: AnyObject {
    var didShowViewerCoachMark: Bool { get set }
}

public final class UserDefaultsWorldViewerCoachMarkStore: WorldViewerCoachMarkStoring {
    private let defaults: UserDefaults
    private let key = "dinkvision.worldViewer.didShowCoachMark.v1"

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    public var didShowViewerCoachMark: Bool {
        get { defaults.bool(forKey: key) }
        set { defaults.set(newValue, forKey: key) }
    }
}

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
    @Published public private(set) var playbackSpeed: WorldViewerPlaybackSpeed = .normal
    @Published public private(set) var followedPlayerID: Int?
    @Published public private(set) var isCoachMarkVisible: Bool
    @Published public var dimLowConfidence: Bool = true {
        didSet { applySnapshot() }
    }
    @Published public private(set) var snapshot: WorldFrameSnapshot

    public let timeline: ReplayTimelineModel
    public let sceneBuilder: WorldSceneBuilder

    private var timelineTimeCancellable: AnyCancellable?
    private var timelineStateCancellable: AnyCancellable?
    private let coachMarkStore: WorldViewerCoachMarkStoring

    public var currentTime: Double {
        timeline.currentTime
    }

    public var durationSeconds: Double {
        timeline.durationSeconds
    }

    public var isPlaying: Bool {
        timeline.isPlaying
    }

    public init(
        bundle: WorldBundle,
        timeline: ReplayTimelineModel? = nil,
        coachMarkStore: WorldViewerCoachMarkStoring = UserDefaultsWorldViewerCoachMarkStore()
    ) {
        let startTime = bundle.world.players.flatMap { $0.frames.map(\.t) }.min() ?? 0
        let allTimes = bundle.world.players.flatMap { $0.frames.map(\.t) } + bundle.world.ball.frames.map(\.t)

        self.bundle = bundle
        self.coachMarkStore = coachMarkStore
        self.isCoachMarkVisible = !coachMarkStore.didShowViewerCoachMark
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

    public func autoplayOnOpen() {
        timeline.play()
    }

    public func cyclePlaybackSpeed() {
        playbackSpeed = playbackSpeed.next
        timeline.setPlaybackRate(playbackSpeed.rate)
    }

    @discardableResult
    public func selectFollowedPlayer(id: Int) -> Bool {
        guard snapshot.players.contains(where: { $0.id == id }) else {
            return false
        }
        followedPlayerID = id
        applyFollowCameraIfNeeded()
        return true
    }

    public func clearFollowedPlayer() {
        followedPlayerID = nil
        applyCameraPreset()
    }

    public func dismissCoachMark() {
        isCoachMarkVisible = false
        coachMarkStore.didShowViewerCoachMark = true
    }

    private func applySnapshot() {
        applySnapshot(at: currentTime)
    }

    private func applySnapshot(at timeSeconds: Double) {
        snapshot = WorldFrameSnapshotBuilder.build(bundle: bundle, at: timeSeconds)
        sceneBuilder.apply(snapshot, dimLowConfidence: dimLowConfidence)
        applyFollowCameraIfNeeded()
    }

    private func applyCameraPreset() {
        if cameraPreset == .ballFollow, let ballPosition = snapshot.ball.frame?.worldXYZ {
            setCamera(position: WorldVec3(ballPosition.x, ballPosition.y - 4.0, max(1.6, ballPosition.z + 1.2)), target: ballPosition)
            return
        }
        let pose = WorldCameraPlanner.pose(for: cameraPreset, court: bundle.world.court)
        setCamera(position: pose.position, target: pose.target)
    }

    private func applyFollowCameraIfNeeded() {
        guard let followedPlayerID,
              let player = snapshot.players.first(where: { $0.id == followedPlayerID }),
              let floor = player.floorPosition else {
            return
        }
        setCamera(
            position: WorldVec3(floor.x, floor.y - 3.4, max(1.7, floor.z + 1.8)),
            target: WorldVec3(floor.x, floor.y, floor.z + 0.8)
        )
    }

    private func setCamera(position: WorldVec3, target: WorldVec3) {
        sceneBuilder.cameraNode.simdPosition = SIMD3(Float(position.x), Float(position.y), Float(position.z))
        sceneBuilder.cameraNode.look(
            at: WorldSceneGeometry.scnVector(target),
            up: SCNVector3(0, 0, 1),
            localFront: SCNVector3(0, 0, -1)
        )
    }
}
