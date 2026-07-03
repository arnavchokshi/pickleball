import SceneKit
import SwiftUI

/// The GLUE-4 in-app 3D world viewer screen: the same court + tiered
/// player representation + trust badges as the web scrubber
/// (`web/replay/src/App.tsx`), driven by `WorldViewerViewModel` off a
/// `WorldBundle` (bundled sample by default). Free orbit/pan/zoom comes
/// from SceneKit's built-in camera controller (`allowsCameraControl`);
/// camera presets, the timeline scrubber, and the trust/dim toggle are
/// custom controls layered on top.
public struct WorldViewerView: View {
    @StateObject private var viewModel: WorldViewerViewModel
    public let onClose: (() -> Void)?

    public init(bundle: WorldBundle, onClose: (() -> Void)? = nil) {
        _viewModel = StateObject(wrappedValue: WorldViewerViewModel(bundle: bundle))
        self.onClose = onClose
    }

    public init(viewModel: @autoclosure @escaping () -> WorldViewerViewModel, onClose: (() -> Void)? = nil) {
        _viewModel = StateObject(wrappedValue: viewModel())
        self.onClose = onClose
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            cameraPresetBar
            ZStack(alignment: .bottomLeading) {
                SceneView(
                    scene: viewModel.sceneBuilder.scene,
                    pointOfView: viewModel.sceneBuilder.cameraNode,
                    options: [.allowsCameraControl, .autoenablesDefaultLighting],
                    delegate: nil
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .accessibilityIdentifier("WorldSceneView")
                legend
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .layoutPriority(1)
            trustBandRow
            timelineControls
        }
        .background(Color.black.opacity(0.92))
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
                Text(viewModel.bundle.manifest.clip)
                    .font(.headline.weight(.heavy))
                Text("3D World Viewer")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.6))
            }
            Spacer()
            Text("Players visible: \(viewModel.snapshot.visiblePlayerCount)")
                .font(.caption.weight(.bold))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
                .accessibilityIdentifier("WorldPlayersVisibleLabel")
        }
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 6)
    }

    private var cameraPresetBar: some View {
        HStack(spacing: 8) {
            ForEach(WorldCameraPreset.allCases, id: \.self) { preset in
                Button(preset.displayName) {
                    viewModel.selectCameraPreset(preset)
                }
                .font(.caption.weight(.bold))
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(preset == viewModel.cameraPreset ? Color.accentColor : Color.white.opacity(0.12), in: Capsule())
                .foregroundStyle(preset == viewModel.cameraPreset ? Color.black : Color.white)
                .accessibilityIdentifier("WorldCameraPreset-\(preset.rawValue)")
            }
            Spacer()
            Toggle(isOn: $viewModel.dimLowConfidence) {
                Text("Dim low-confidence")
                    .font(.caption.weight(.semibold))
            }
            .toggleStyle(.switch)
            .fixedSize()
            .accessibilityIdentifier("WorldDimLowConfidenceToggle")
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 6)
    }

    private var legend: some View {
        HStack(spacing: 10) {
            legendSwatch(color: WorldTrustColors.swiftUIColor(for: .verified), label: "verified")
            legendSwatch(color: WorldTrustColors.swiftUIColor(for: .preview), label: "preview")
            legendSwatch(color: WorldTrustColors.swiftUIColor(for: .lowConfidence), label: "low confidence")
            Text(viewModel.snapshot.ball.mode.readoutText)
                .font(.caption2.weight(.semibold))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: Capsule())
        .padding(10)
    }

    private func legendSwatch(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label).font(.caption2.weight(.semibold))
        }
    }

    private var trustBandRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                trustChip(label: "Court (CAL)", trustBand: viewModel.bundle.world.court.trustBand)
                trustChip(label: "Ball (BALL)", trustBand: viewModel.bundle.world.ball.trustBand)
                ForEach(viewModel.bundle.world.players, id: \.id) { player in
                    trustChip(label: "Player \(player.id) (\(player.representation.rawValue))", trustBand: player.trustBand)
                }
            }
            .padding(.horizontal, 14)
        }
        .padding(.vertical, 6)
    }

    private func trustChip(label: String, trustBand: TrustBand?) -> some View {
        let badge = TrustBandPresentation.badge(for: trustBand)
        return VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2.weight(.semibold)).foregroundStyle(.white.opacity(0.7))
            Text(TrustBandPresentation.chipText(trustBand))
                .font(.caption.weight(.bold))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(WorldTrustColors.swiftUIColor(for: badge).opacity(0.85), in: Capsule())
                .foregroundStyle(Color.black)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("WorldTrustChip-\(label)")
    }

    private var timelineControls: some View {
        VStack(spacing: 4) {
            HStack {
                Button(action: viewModel.togglePlayback) {
                    Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .background(.ultraThinMaterial, in: Circle())
                .accessibilityIdentifier("WorldPlayPauseButton")

                Slider(
                    value: Binding(
                        get: { viewModel.currentTime },
                        set: { viewModel.seek(to: $0) }
                    ),
                    in: 0...max(viewModel.durationSeconds, 0.001)
                )
                .accessibilityIdentifier("WorldTimelineSlider")

                Text(String(format: "%.2fs", viewModel.currentTime))
                    .font(.caption.monospacedDigit())
                    .frame(width: 56, alignment: .trailing)
            }
        }
        .padding(.horizontal, 14)
        .padding(.bottom, ReplayScrubberLayout.bottomPadding)
        .padding(.top, 4)
    }
}
