import SwiftUI

public struct RealityReplayView: View {
    @StateObject private var viewModel: RealityReplayViewModel
    public let onClose: (() -> Void)?

    public init(asset: RealityReplayAsset, onClose: (() -> Void)? = nil) throws {
        let model = try RealityReplayViewModel(asset: asset)
        _viewModel = StateObject(wrappedValue: model)
        self.onClose = onClose
    }

    public init(viewModel: @autoclosure @escaping () throws -> RealityReplayViewModel, onClose: (() -> Void)? = nil) throws {
        let model = try viewModel()
        _viewModel = StateObject(wrappedValue: model)
        self.onClose = onClose
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            ZStack(alignment: .topLeading) {
                RealityReplaySceneHost(
                    assetURL: viewModel.asset.assetURL,
                    animationTimeSeconds: viewModel.animationTimeSeconds,
                    isPlaying: viewModel.isPlaying
                )
                .accessibilityIdentifier("RealityReplaySceneHost")
                badge
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .layoutPriority(1)
            timelineControls
        }
        .background(Color.black.opacity(0.94))
        .foregroundStyle(.white)
    }

    private var header: some View {
        HStack {
            if let onClose {
                Button(action: onClose) {
                    Image(systemName: "chevron.left")
                        .font(.headline.weight(.heavy))
                        .frame(width: 34, height: 34)
                }
                .buttonStyle(.plain)
                .background(.ultraThinMaterial, in: Circle())
                .accessibilityLabel("Back")
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.asset.clipID)
                    .font(.headline.weight(.heavy))
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
                Text("RealityKit Replay")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.62))
            }
            Spacer()
            Text(String(format: "%.1f MB source", Double(viewModel.asset.sourceAssetByteCount) / 1_000_000.0))
                .font(.caption.weight(.bold))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
                .accessibilityIdentifier("RealityReplaySourceSizeBadge")
        }
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 6)
    }

    private var badge: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(viewModel.asset.badgeTitle)
                .font(.caption.weight(.heavy))
            Text(viewModel.asset.badgeDetail)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.white.opacity(0.76))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .padding(10)
        .accessibilityIdentifier("RealityReplayHonestyBadge")
    }

    private var timelineControls: some View {
        HStack {
            Button(action: viewModel.togglePlayback) {
                Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .background(.ultraThinMaterial, in: Circle())
            .accessibilityIdentifier("RealityReplayPlayPauseButton")

            Slider(
                value: Binding(
                    get: { viewModel.currentTime },
                    set: { viewModel.seek(to: $0) }
                ),
                in: 0...max(viewModel.durationSeconds, 0.001)
            )
            .accessibilityIdentifier("RealityReplayTimelineSlider")

            Text(String(format: "%.2fs", viewModel.currentTime))
                .font(.caption.monospacedDigit())
                .frame(width: 56, alignment: .trailing)
        }
        .padding(.horizontal, 14)
        .padding(.bottom, ReplayScrubberLayout.bottomPadding)
        .padding(.top, 8)
    }
}

#if os(iOS) && canImport(RealityKit) && canImport(UIKit)
import RealityKit
import UIKit

public struct RealityReplaySceneHost: UIViewRepresentable {
    public var assetURL: URL
    public var animationTimeSeconds: Double
    public var isPlaying: Bool

    public init(assetURL: URL, animationTimeSeconds: Double, isPlaying: Bool) {
        self.assetURL = assetURL
        self.animationTimeSeconds = animationTimeSeconds
        self.isPlaying = isPlaying
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    @MainActor
    public func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
        arView.environment.background = .color(.black)
        context.coordinator.loadIfNeeded(assetURL: assetURL, in: arView)
        context.coordinator.apply(animationTimeSeconds: animationTimeSeconds, isPlaying: isPlaying)
        return arView
    }

    @MainActor
    public func updateUIView(_ uiView: ARView, context: Context) {
        context.coordinator.loadIfNeeded(assetURL: assetURL, in: uiView)
        context.coordinator.apply(animationTimeSeconds: animationTimeSeconds, isPlaying: isPlaying)
    }

    @MainActor
    public final class Coordinator {
        private var loadedAssetURL: URL?
        private var animationController: AnimationPlaybackController?

        func loadIfNeeded(assetURL: URL, in arView: ARView) {
            guard loadedAssetURL != assetURL else { return }
            arView.scene.anchors.removeAll()
            animationController = nil

            let anchor = AnchorEntity(world: .zero)
            do {
                let entity = try Entity.load(contentsOf: assetURL)
                entity.scale = SIMD3<Float>(repeating: 0.35)
                anchor.addChild(entity)
                if let animation = entity.availableAnimations.first {
                    animationController = entity.playAnimation(animation, transitionDuration: 0, startsPaused: true)
                }
            } catch {
                let box = ModelEntity(
                    mesh: .generateBox(size: 0.25),
                    materials: [SimpleMaterial(color: .systemRed, isMetallic: false)]
                )
                anchor.addChild(box)
            }
            arView.scene.addAnchor(anchor)
            loadedAssetURL = assetURL
        }

        func apply(animationTimeSeconds: Double, isPlaying: Bool) {
            guard let animationController else { return }
            animationController.time = animationTimeSeconds
            if isPlaying {
                animationController.resume()
            } else {
                animationController.pause()
            }
        }
    }
}
#else
public struct RealityReplaySceneHost: View {
    public var assetURL: URL
    public var animationTimeSeconds: Double
    public var isPlaying: Bool

    public init(assetURL: URL, animationTimeSeconds: Double, isPlaying: Bool) {
        self.assetURL = assetURL
        self.animationTimeSeconds = animationTimeSeconds
        self.isPlaying = isPlaying
    }

    public var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "cube.transparent")
                .font(.largeTitle)
            Text("RealityKit playback is available in the iOS app build.")
                .font(.headline.weight(.semibold))
            Text(assetURL.lastPathComponent)
                .font(.caption.monospaced())
                .foregroundStyle(.white.opacity(0.64))
            Text(String(format: "Animation %.2fs", animationTimeSeconds))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white.opacity(0.64))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }
}
#endif
