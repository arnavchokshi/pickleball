import SceneKit
import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

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
        ZStack {
            LinearGradient(
                colors: [WorldViewerChromeColor.deepGreen, WorldViewerChromeColor.ink],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                ZStack(alignment: .bottomLeading) {
                    WorldSceneKitView(
                        scene: viewModel.sceneBuilder.scene,
                        cameraNode: viewModel.sceneBuilder.cameraNode,
                        onPlayerTap: { playerID in _ = viewModel.selectFollowedPlayer(id: playerID) },
                        onEmptyTap: { viewModel.clearFollowedPlayer() },
                        onDoubleTap: { viewModel.selectCameraPreset(.broadcast) }
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .accessibilityIdentifier("WorldSceneView")

                    legend

                    if let followedPlayerID = viewModel.followedPlayerID {
                        followedChip(playerID: followedPlayerID)
                            .padding(.leading, 14)
                            .padding(.bottom, 54)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .layoutPriority(1)

                bottomControlCard
                    .padding(.horizontal, 14)
                    .padding(.bottom, ReplayScrubberLayout.bottomPadding)
                    .padding(.top, 8)
            }

            VStack(spacing: 10) {
                header
                cameraPresetBar
                Spacer()
            }

            if viewModel.isCoachMarkVisible {
                coachMarkOverlay
                    .transition(.opacity)
            }
        }
        .foregroundStyle(WorldViewerChromeColor.ink)
        .task {
            viewModel.autoplayOnOpen()
        }
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
                    .font(.system(size: 17, weight: .heavy, design: .rounded))
                Text("3D World Viewer")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(WorldViewerChromeColor.ink.opacity(0.62))
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 6) {
                Text("Players visible: \(viewModel.snapshot.visiblePlayerCount)")
                    .font(.caption.weight(.bold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(WorldViewerChromeColor.cream.opacity(0.92), in: Capsule())
                    .accessibilityIdentifier("WorldPlayersVisibleLabel")
                HStack(spacing: 6) {
                    chromeChip("Trusted 3D \(TrustBandPresentation.chipText(viewModel.bundle.world.court.trustBand))")
                    chromeChip(viewModel.bundle.manifest.notes.isEmpty ? "sample" : "sample")
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 6)
        .background(
            LinearGradient(colors: [WorldViewerChromeColor.cream.opacity(0.96), WorldViewerChromeColor.cream.opacity(0)], startPoint: .top, endPoint: .bottom)
                .frame(height: 130),
            alignment: .top
        )
    }

    private var cameraPresetBar: some View {
        HStack(spacing: 8) {
            ForEach(WorldCameraPreset.allCases, id: \.self) { preset in
                Button(preset.displayName) {
                    viewModel.selectCameraPreset(preset)
                }
                .font(.caption.weight(.bold))
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .frame(minHeight: 44)
                .background(preset == viewModel.cameraPreset ? WorldViewerChromeColor.ballYellow : WorldViewerChromeColor.cream.opacity(0.92), in: Capsule())
                .foregroundStyle(WorldViewerChromeColor.ink)
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
        .background(WorldViewerChromeColor.cream.opacity(0.92), in: Capsule())
        .padding(10)
    }

    private func legendSwatch(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label).font(.caption2.weight(.semibold))
        }
    }

    private func followedChip(playerID: Int) -> some View {
        chromeChip("Following P\(playerID)")
            .accessibilityIdentifier("WorldFollowedPlayerChip")
    }

    private var bottomControlCard: some View {
        VStack(spacing: 4) {
            HStack {
                Button(action: viewModel.togglePlayback) {
                    Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title3.weight(.black))
                        .frame(width: 52, height: 52)
                }
                .buttonStyle(.plain)
                .background(WorldViewerChromeColor.ballYellow, in: Circle())
                .accessibilityIdentifier("WorldPlayPauseButton")

                ZStack(alignment: .topLeading) {
                    rallyTicks
                        .padding(.horizontal, 3)
                        .offset(y: -4)
                    Slider(
                        value: Binding(
                            get: { viewModel.currentTime },
                            set: { viewModel.seek(to: $0) }
                        ),
                        in: 0...max(viewModel.durationSeconds, 0.001)
                    )
                    .accessibilityIdentifier("WorldTimelineSlider")
                }

                Text(String(format: "%.2fs", viewModel.currentTime))
                    .font(.caption.monospacedDigit())
                    .frame(width: 56, alignment: .trailing)

                Button(viewModel.playbackSpeed.rawValue) {
                    viewModel.cyclePlaybackSpeed()
                }
                .font(.caption.weight(.black))
                .frame(minWidth: 48, minHeight: 44)
                .background(WorldViewerChromeColor.ink, in: Capsule())
                .foregroundStyle(WorldViewerChromeColor.cream)
                .accessibilityIdentifier("WorldSpeedCycleButton")
            }
        }
        .padding(12)
        .background(WorldViewerChromeColor.cream, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .shadow(color: Color.black.opacity(0.18), radius: 18, y: 8)
        .accessibilityIdentifier("WorldBottomControlCard")
    }

    private var rallyTicks: some View {
        GeometryReader { proxy in
            ForEach(rallyTickFractions.indices, id: \.self) { index in
                Capsule()
                    .fill(WorldViewerChromeColor.trailRed.opacity(0.75))
                    .frame(width: 3, height: 10)
                    .position(x: proxy.size.width * rallyTickFractions[index], y: 5)
            }
        }
        .frame(height: 12)
    }

    private var rallyTickFractions: [CGFloat] {
        let duration = max(viewModel.durationSeconds, 0.001)
        return (viewModel.bundle.contactWindows?.events ?? [])
            .filter { $0.type == "contact" }
            .prefix(18)
            .map { CGFloat(min(1, max(0, $0.t / duration))) }
    }

    private var coachMarkOverlay: some View {
        ZStack {
            Color.black.opacity(0.38).ignoresSafeArea()
            VStack(spacing: 18) {
                Spacer()
                VStack(spacing: 12) {
                    Text("drag to orbit - tap a player to follow")
                        .font(.system(size: 18, weight: .heavy, design: .rounded))
                        .multilineTextAlignment(.center)
                    HStack(spacing: 24) {
                        CoachArrowShape()
                            .stroke(WorldViewerChromeColor.trailBlue, style: StrokeStyle(lineWidth: 5, lineCap: .round, lineJoin: .round))
                            .frame(width: 92, height: 62)
                        CoachArrowShape()
                            .stroke(WorldViewerChromeColor.trailRed, style: StrokeStyle(lineWidth: 5, lineCap: .round, lineJoin: .round))
                            .frame(width: 92, height: 62)
                            .scaleEffect(x: -1, y: 1)
                    }
                    Button("Got it") {
                        viewModel.dismissCoachMark()
                    }
                    .font(.system(size: 14, weight: .black, design: .rounded))
                    .frame(minWidth: 120, minHeight: 44)
                    .background(WorldViewerChromeColor.ballYellow, in: Capsule())
                }
                .padding(22)
                .background(WorldViewerChromeColor.cream, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
                .padding(.horizontal, 22)
                Spacer()
            }
        }
        .accessibilityIdentifier("WorldViewerCoachMark")
    }

    private func chromeChip(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.black))
            .lineLimit(1)
            .minimumScaleFactor(0.72)
            .padding(.horizontal, 9)
            .padding(.vertical, 6)
            .background(WorldViewerChromeColor.cream.opacity(0.92), in: Capsule())
            .foregroundStyle(WorldViewerChromeColor.ink)
    }
}

private enum WorldViewerChromeColor {
    static let cream = Color(red: 244.0 / 255.0, green: 238.0 / 255.0, blue: 227.0 / 255.0)
    static let ink = Color(red: 20.0 / 255.0, green: 20.0 / 255.0, blue: 20.0 / 255.0)
    static let deepGreen = Color(red: 35.0 / 255.0, green: 71.0 / 255.0, blue: 49.0 / 255.0)
    static let ballYellow = Color(red: 242.0 / 255.0, green: 198.0 / 255.0, blue: 63.0 / 255.0)
    static let trailBlue = Color(red: 62.0 / 255.0, green: 142.0 / 255.0, blue: 240.0 / 255.0)
    static let trailRed = Color(red: 232.0 / 255.0, green: 80.0 / 255.0, blue: 58.0 / 255.0)
}

private struct CoachArrowShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX + rect.width * 0.10, y: rect.maxY * 0.78))
        path.addQuadCurve(
            to: CGPoint(x: rect.maxX * 0.86, y: rect.minY + rect.height * 0.22),
            control: CGPoint(x: rect.midX * 0.8, y: rect.minY + rect.height * 0.10)
        )
        path.move(to: CGPoint(x: rect.maxX * 0.72, y: rect.minY + rect.height * 0.16))
        path.addLine(to: CGPoint(x: rect.maxX * 0.88, y: rect.minY + rect.height * 0.22))
        path.addLine(to: CGPoint(x: rect.maxX * 0.78, y: rect.minY + rect.height * 0.38))
        return path
    }
}

private struct WorldSceneKitView: View {
    let scene: SCNScene
    let cameraNode: SCNNode
    let onPlayerTap: (Int) -> Void
    let onEmptyTap: () -> Void
    let onDoubleTap: () -> Void

    var body: some View {
#if canImport(UIKit)
        WorldSceneUIKitView(
            scene: scene,
            cameraNode: cameraNode,
            onPlayerTap: onPlayerTap,
            onEmptyTap: onEmptyTap,
            onDoubleTap: onDoubleTap
        )
#else
        SceneView(
            scene: scene,
            pointOfView: cameraNode,
            options: [.allowsCameraControl, .autoenablesDefaultLighting],
            delegate: nil
        )
        .onTapGesture {
            onEmptyTap()
        }
#endif
    }
}

#if canImport(UIKit)
private struct WorldSceneUIKitView: UIViewRepresentable {
    let scene: SCNScene
    let cameraNode: SCNNode
    let onPlayerTap: (Int) -> Void
    let onEmptyTap: () -> Void
    let onDoubleTap: () -> Void

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = scene
        view.pointOfView = cameraNode
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .clear

        let singleTap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.singleTap(_:)))
        singleTap.numberOfTapsRequired = 1
        let doubleTap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.doubleTap(_:)))
        doubleTap.numberOfTapsRequired = 2
        singleTap.require(toFail: doubleTap)
        let pinch = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.pinch(_:)))
        view.addGestureRecognizer(singleTap)
        view.addGestureRecognizer(doubleTap)
        view.addGestureRecognizer(pinch)
        context.coordinator.view = view
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        view.scene = scene
        view.pointOfView = cameraNode
        context.coordinator.onPlayerTap = onPlayerTap
        context.coordinator.onEmptyTap = onEmptyTap
        context.coordinator.onDoubleTap = onDoubleTap
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(onPlayerTap: onPlayerTap, onEmptyTap: onEmptyTap, onDoubleTap: onDoubleTap)
    }

    @MainActor
    final class Coordinator: NSObject {
        weak var view: SCNView?
        var onPlayerTap: (Int) -> Void
        var onEmptyTap: () -> Void
        var onDoubleTap: () -> Void

        init(onPlayerTap: @escaping (Int) -> Void, onEmptyTap: @escaping () -> Void, onDoubleTap: @escaping () -> Void) {
            self.onPlayerTap = onPlayerTap
            self.onEmptyTap = onEmptyTap
            self.onDoubleTap = onDoubleTap
        }

        @objc func singleTap(_ recognizer: UITapGestureRecognizer) {
            guard let view else { return }
            let location = recognizer.location(in: view)
            for hit in view.hitTest(location, options: nil) {
                if let playerID = Self.playerID(from: hit.node) {
                    onPlayerTap(playerID)
                    return
                }
            }
            onEmptyTap()
        }

        @objc func doubleTap(_: UITapGestureRecognizer) {
            onDoubleTap()
        }

        @objc func pinch(_ recognizer: UIPinchGestureRecognizer) {
            guard let camera = view?.pointOfView?.camera else { return }
            let next = max(35, min(70, camera.fieldOfView / Double(recognizer.scale)))
            camera.fieldOfView = next
            recognizer.scale = 1
        }

        private static func playerID(from node: SCNNode?) -> Int? {
            var current = node
            while let node = current {
                if let name = node.name, name.hasPrefix("player-") {
                    return Int(name.dropFirst("player-".count))
                }
                current = node.parent
            }
            return nil
        }
    }
}
#endif
