import SwiftUI
import PickleballCapture

private enum PickleballPalette {
    static let ink = Color(red: 0.03, green: 0.045, blue: 0.04)
    static let felt = Color(red: 0.06, green: 0.22, blue: 0.16)
    static let mint = Color(red: 0.67, green: 1.0, blue: 0.58)
    static let lime = Color(red: 0.88, green: 1.0, blue: 0.24)
    static let coral = Color(red: 1.0, green: 0.43, blue: 0.30)
    static let cyan = Color(red: 0.40, green: 0.86, blue: 1.0)
    static let cream = Color(red: 0.94, green: 0.97, blue: 0.89)
}

struct AppRootView: View {
    @StateObject private var flow = PickleballAppFlow()

    var body: some View {
        ZStack {
            PickleballCourtBackdrop()
                .ignoresSafeArea()

            switch flow.screen {
            case .splash:
                PickleballSplashScreen {
                    flow.finishSplash()
                }
                .transition(.opacity.combined(with: .scale(scale: 0.985)))
            case .home:
                PickleballHomeScreen {
                    flow.openCamera()
                }
                .transition(.asymmetric(
                    insertion: .opacity.combined(with: .move(edge: .bottom)),
                    removal: .opacity
                ))
            case .camera:
                CameraCaptureScreen {
                    flow.returnHome()
                }
                .transition(.asymmetric(
                    insertion: .opacity.combined(with: .move(edge: .trailing)),
                    removal: .opacity.combined(with: .move(edge: .leading))
                ))
            }
        }
        .animation(.smooth(duration: 0.48), value: flow.screen)
        .preferredColorScheme(.dark)
    }
}

private struct PickleballSplashScreen: View {
    let onFinish: () -> Void
    @State private var markScale: CGFloat = 0.86
    @State private var wordOpacity = 0.0

    var body: some View {
        ZStack {
            PickleballCourtBackdrop()

            VStack(spacing: 20) {
                PickleballBrandMark(size: 112)
                    .scaleEffect(markScale)
                    .shadow(color: PickleballPalette.lime.opacity(0.28), radius: 34, y: 12)

                Text("Pickleball")
                    .font(.system(size: 34, weight: .heavy, design: .rounded))
                    .foregroundStyle(PickleballPalette.cream)
                    .opacity(wordOpacity)
            }
        }
        .ignoresSafeArea()
        .task {
            withAnimation(.smooth(duration: 0.55)) {
                markScale = 1.0
                wordOpacity = 1.0
            }
            try? await Task.sleep(nanoseconds: 1_080_000_000)
            onFinish()
        }
    }
}

private struct PickleballHomeScreen: View {
    let openCamera: () -> Void

    var body: some View {
        GeometryReader { proxy in
            let isCompact = proxy.size.width < 520
            ZStack {
                PickleballCourtBackdrop()

                ScrollView(.vertical, showsIndicators: false) {
                    VStack(spacing: isCompact ? 20 : 24) {
                        homeHeader
                        hero(isCompact: isCompact)
                        signalStrip
                        CapturePreviewPanel()
                        Spacer(minLength: 16)
                    }
                    .frame(minHeight: proxy.size.height)
                    .padding(.horizontal, isCompact ? 18 : 28)
                    .padding(.top, max(18, proxy.safeAreaInsets.top + 12))
                    .padding(.bottom, max(28, proxy.safeAreaInsets.bottom + 18))
                }
            }
            .ignoresSafeArea()
        }
    }

    private var homeHeader: some View {
        HStack(spacing: 12) {
            PickleballBrandMark(size: 42)
            VStack(alignment: .leading, spacing: 1) {
                Text("Pickleball")
                    .font(.system(size: 18, weight: .heavy, design: .rounded))
                Text("Court intelligence")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.64))
            }
            Spacer()
            Text("Local")
                .font(.caption.weight(.bold))
                .foregroundStyle(PickleballPalette.ink)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(PickleballPalette.lime, in: Capsule())
        }
        .foregroundStyle(PickleballPalette.cream)
    }

    private func hero(isCompact: Bool) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Game-speed capture,\nready on the first tap.")
                .font(.system(size: isCompact ? 39 : 48, weight: .black, design: .rounded))
                .minimumScaleFactor(0.68)
                .lineLimit(4)
                .foregroundStyle(PickleballPalette.cream)
                .shadow(color: .black.opacity(0.32), radius: 18, y: 8)

            Text("Record clean court video with locked camera settings, package-safe sidecars, and a review stack that stays honest about what is verified.")
                .font(.system(size: 17, weight: .medium))
                .lineSpacing(3)
                .foregroundStyle(.white.opacity(0.72))
                .fixedSize(horizontal: false, vertical: true)

            Button(action: openCamera) {
                HStack(spacing: 12) {
                    Image(systemName: "camera.viewfinder")
                        .font(.title3.weight(.bold))
                    Text("Open Camera")
                        .font(.headline.weight(.heavy))
                    Spacer(minLength: 0)
                    Image(systemName: "arrow.right")
                        .font(.headline.weight(.heavy))
                }
                .foregroundStyle(PickleballPalette.ink)
                .padding(.horizontal, 18)
                .frame(height: 58)
                .background(
                    LinearGradient(
                        colors: [PickleballPalette.lime, PickleballPalette.mint],
                        startPoint: .leading,
                        endPoint: .trailing
                    ),
                    in: Capsule()
                )
                .shadow(color: PickleballPalette.lime.opacity(0.24), radius: 26, y: 14)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Open Camera")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 18)
    }

    private var signalStrip: some View {
        // The "Review 3D" tile is a placeholder for playback of server-rendered
        // replay assets. It is not an ON-DEVICE LIVE deep reconstruction claim.
        HStack(spacing: 10) {
            HomeSignalTile(title: "Capture", value: "60 FPS", icon: "bolt.fill", tint: PickleballPalette.lime)
            HomeSignalTile(title: "Sidecar", value: "Ready", icon: "doc.badge.gearshape", tint: PickleballPalette.cyan)
            HomeSignalTile(title: "Review", value: "3D", icon: "scope", tint: PickleballPalette.coral)
        }
    }
}

private struct CameraCaptureScreen: View {
    let returnHome: () -> Void
    @StateObject private var model = CaptureViewModel()

    var body: some View {
        GeometryReader { proxy in
            let isLandscape = proxy.size.width > proxy.size.height
            ZStack {
                CameraPreviewView(
                    session: model.session,
                    videoRotationAngle: model.previewRotationAngle
                )
                .ignoresSafeArea()

                Color.black.opacity(model.status == .idle || model.status == .requestingAccess ? 0.22 : 0)
                    .ignoresSafeArea()

                if isLandscape {
                    landscapeOverlay
                } else {
                    portraitOverlay
                }
            }
            .task {
                await model.prepare()
                await model.updateOrientation(isLandscapeViewport: isLandscape)
            }
            .onChange(of: isLandscape) {
                Task {
                    await model.updateOrientation(isLandscapeViewport: isLandscape)
                }
            }
        }
    }

    private var portraitOverlay: some View {
        VStack(spacing: 12) {
            header
            Spacer(minLength: 20)
            recordButton
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var landscapeOverlay: some View {
        HStack(alignment: .top, spacing: 14) {
            header
            Spacer()
            recordButton
        }
        .padding(14)
    }

    private var header: some View {
        HStack(spacing: 10) {
            Button(action: returnHome) {
                Image(systemName: "chevron.left")
                    .font(.headline.weight(.heavy))
                    .frame(width: 38, height: 38)
            }
            .buttonStyle(.plain)
            .foregroundStyle(PickleballPalette.cream)
            .background(.ultraThinMaterial, in: Circle())
            .accessibilityLabel("Back")

            statusPill
            Spacer(minLength: 8)
            Label(model.captureOrientationTitle, systemImage: orientationIconName)
                .font(.caption.weight(.semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.regularMaterial, in: Capsule())
        }
    }

    private var recordButton: some View {
        Button {
            Task {
                await model.toggleRecording()
            }
        } label: {
            Image(systemName: model.isRecording ? "stop.fill" : "record.circle")
                .font(.system(size: 32, weight: .semibold))
                .frame(width: 72, height: 72)
        }
        .buttonStyle(.borderedProminent)
        .buttonBorderShape(.circle)
        .tint(model.isRecording ? PickleballPalette.coral : PickleballPalette.lime)
        .disabled(!canRecord)
        .shadow(color: (model.isRecording ? PickleballPalette.coral : PickleballPalette.lime).opacity(0.28), radius: 20, y: 10)
        .accessibilityLabel(model.isRecording ? "Stop recording" : "Start recording")
    }

    private var statusPill: some View {
        Label(statusText, systemImage: statusIconName)
            .font(.caption.weight(.semibold))
            .foregroundStyle(PickleballPalette.cream)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(.regularMaterial, in: Capsule())
    }

    private var canRecord: Bool {
        switch model.status {
        case .ready, .recording, .finished:
            return true
        case .idle, .requestingAccess, .blocked:
            return false
        }
    }

    private var statusText: String {
        switch model.status {
        case .idle:
            return "Idle"
        case .requestingAccess:
            return "Access"
        case .ready:
            return "Ready"
        case .recording:
            return "Recording"
        case .finished:
            return "Saved"
        case .blocked(let message):
            return message
        }
    }

    private var statusIconName: String {
        switch model.status {
        case .idle:
            return "pause"
        case .requestingAccess:
            return "lock.open"
        case .ready:
            return "checkmark"
        case .recording:
            return "record.circle"
        case .finished:
            return "checkmark.circle"
        case .blocked:
            return "exclamationmark.triangle"
        }
    }

    private var orientationIconName: String {
        model.captureOrientationTitle == "Portrait" ? "iphone" : "iphone.landscape"
    }
}

private struct PickleballCourtBackdrop: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    PickleballPalette.ink,
                    Color(red: 0.025, green: 0.10, blue: 0.08),
                    Color(red: 0.04, green: 0.055, blue: 0.055)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            CourtLineGrid()
                .stroke(.white.opacity(0.12), lineWidth: 1)
                .blendMode(.screen)
                .padding(.horizontal, -80)

            LinearGradient(
                colors: [
                    Color.clear,
                    PickleballPalette.felt.opacity(0.38),
                    Color.black.opacity(0.58)
                ],
                startPoint: .top,
                endPoint: .bottom
            )
        }
    }
}

private struct CourtLineGrid: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let step = max(70, rect.width / 5)
        var x = rect.minX - step
        while x <= rect.maxX + step {
            path.move(to: CGPoint(x: x, y: rect.minY))
            path.addLine(to: CGPoint(x: x + rect.height * 0.18, y: rect.maxY))
            x += step
        }
        var y = rect.minY + step * 0.6
        while y <= rect.maxY {
            path.move(to: CGPoint(x: rect.minX, y: y))
            path.addLine(to: CGPoint(x: rect.maxX, y: y + step * 0.16))
            y += step
        }
        path.addRoundedRect(in: rect.insetBy(dx: rect.width * 0.16, dy: rect.height * 0.18), cornerSize: CGSize(width: 18, height: 18))
        return path
    }
}

private struct PickleballBrandMark: View {
    var size: CGFloat

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.24, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            PickleballPalette.lime,
                            PickleballPalette.mint,
                            PickleballPalette.cyan
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
            CourtGlyph()
                .stroke(PickleballPalette.ink.opacity(0.86), style: StrokeStyle(lineWidth: max(2, size * 0.045), lineCap: .round, lineJoin: .round))
                .padding(size * 0.20)
        }
        .frame(width: size, height: size)
    }
}

private struct CourtGlyph: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.addRoundedRect(in: rect, cornerSize: CGSize(width: rect.width * 0.08, height: rect.width * 0.08))
        path.move(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
        path.move(to: CGPoint(x: rect.minX, y: rect.midY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.midY))
        path.move(to: CGPoint(x: rect.minX, y: rect.minY + rect.height * 0.28))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + rect.height * 0.28))
        path.move(to: CGPoint(x: rect.minX, y: rect.maxY - rect.height * 0.28))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - rect.height * 0.28))
        return path
    }
}

private struct HomeSignalTile: View {
    let title: String
    let value: String
    let icon: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: icon)
                .font(.caption.weight(.heavy))
                .foregroundStyle(tint)
            Text(value)
                .font(.system(size: 18, weight: .heavy, design: .rounded))
                .foregroundStyle(PickleballPalette.cream)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(title)
                .font(.caption2.weight(.bold))
                .foregroundStyle(.white.opacity(0.56))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.white.opacity(0.14), lineWidth: 1)
        )
    }
}

private struct CapturePreviewPanel: View {
    var body: some View {
        ZStack(alignment: .bottomLeading) {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.black.opacity(0.28))
                .overlay(CourtLineGrid().stroke(.white.opacity(0.16), lineWidth: 1))
                .overlay(alignment: .topTrailing) {
                    VStack(alignment: .trailing, spacing: 6) {
                        Text("LOCKED")
                            .font(.caption2.weight(.black))
                            .foregroundStyle(PickleballPalette.lime)
                        Text("HEVC / Motion / Sidecar")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.white.opacity(0.68))
                    }
                    .padding(16)
                }

            VStack(alignment: .leading, spacing: 8) {
                Text("Review-ready packages")
                    .font(.title3.weight(.heavy))
                    .foregroundStyle(PickleballPalette.cream)
                Text("Each capture keeps the video, orientation, timing, gravity, and quality flags together.")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white.opacity(0.66))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(18)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 190)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.white.opacity(0.14), lineWidth: 1)
        )
    }
}

#Preview {
    AppRootView()
}
