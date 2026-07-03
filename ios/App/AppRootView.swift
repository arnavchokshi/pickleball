import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import PickleballCapture
import PickleballCore
import PickleballFastTier
import PickleballGuidance
import PickleballUpload
import PickleballReplay

private enum PickleballPalette {
    static let ink = Color(red: 0.03, green: 0.045, blue: 0.04)
    static let felt = Color(red: 0.06, green: 0.22, blue: 0.16)
    static let mint = Color(red: 0.67, green: 1.0, blue: 0.58)
    static let lime = Color(red: 0.88, green: 1.0, blue: 0.24)
    static let coral = Color(red: 1.0, green: 0.43, blue: 0.30)
    static let cyan = Color(red: 0.40, green: 0.86, blue: 1.0)
    static let cream = Color(red: 0.94, green: 0.97, blue: 0.89)
}

private enum CameraRollImportStatus: Equatable {
    case idle
    case importing
    case imported(String)
    case failed(String)
}

private enum RenderUploadDisplayStatus: Equatable {
    case idle
    case submitting(sessionID: String)
    case running(sessionID: String, job: RenderGatewayJob, replayURL: URL?)
    case complete(sessionID: String, job: RenderGatewayJob, replayURL: URL?)
    case failed(sessionID: String, message: String)

    var sessionID: String? {
        switch self {
        case .idle:
            return nil
        case .submitting(let sessionID),
                .running(let sessionID, _, _),
                .complete(let sessionID, _, _),
                .failed(let sessionID, _):
            return sessionID
        }
    }

    var isBusy: Bool {
        switch self {
        case .submitting, .running:
            return true
        case .idle, .complete, .failed:
            return false
        }
    }

    var progressFraction: Double {
        switch self {
        case .idle:
            return 0
        case .submitting:
            return 0.05
        case .running(_, let job, _), .complete(_, let job, _):
            return job.progress?.fractionComplete ?? (job.status == .complete ? 1.0 : 0.35)
        case .failed(_, _):
            return 1.0
        }
    }

    var stageText: String {
        switch self {
        case .idle:
            return "Ready"
        case .submitting:
            return "Uploading to Render"
        case .running(_, let job, _), .complete(_, let job, _):
            return job.progress?.stage ?? job.status.rawValue
        case .failed(_, let message):
            return message
        }
    }

    var etaText: String? {
        switch self {
        case .running(_, let job, _):
            return job.progress?.etaText
        case .submitting:
            return "ETA calculating"
        case .idle, .complete, .failed:
            return nil
        }
    }

    var replayURL: URL? {
        switch self {
        case .running(_, _, let replayURL), .complete(_, _, let replayURL):
            return replayURL
        case .idle, .submitting, .failed:
            return nil
        }
    }
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
                PickleballHomeScreen(
                    openCamera: { flow.openCamera() },
                    openWorldViewer: { flow.openWorldViewer() },
                    openRealityReplay: { flow.openRealityReplay() }
                )
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
            case .worldViewer:
                WorldViewerScreen {
                    flow.returnHome()
                }
                .transition(.asymmetric(
                    insertion: .opacity.combined(with: .move(edge: .trailing)),
                    removal: .opacity.combined(with: .move(edge: .leading))
                ))
            case .realityReplay:
                RealityReplayScreen {
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
    let openWorldViewer: () -> Void
    let openRealityReplay: () -> Void
    @State private var isImportPickerPresented = false
    @State private var importStatus: CameraRollImportStatus = .idle
    @State private var captureItems: [CaptureLibraryItem] = []
    @State private var importSummary: PostStopPreviewSummary?
    @State private var renderUploadStatus: RenderUploadDisplayStatus = .idle

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
                        CapturePreviewPanel(
                            items: captureItems,
                            importStatus: importStatus,
                            renderUploadStatus: renderUploadStatus,
                            processPackage: { item in
                                Task {
                                    await uploadPackage(item)
                                }
                            }
                        )
                        if let importSummary {
                            PostStopSummaryCard(summary: importSummary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .transition(.opacity.combined(with: .scale(scale: 0.96)))
                        }
                        Spacer(minLength: 16)
                    }
                    .frame(minHeight: proxy.size.height)
                    .padding(.horizontal, isCompact ? 18 : 28)
                    .padding(.top, max(18, proxy.safeAreaInsets.top + 12))
                    .padding(.bottom, max(28, proxy.safeAreaInsets.bottom + 18))
                }
            }
            .ignoresSafeArea()
            .task {
                await refreshCaptureItems()
            }
            .sheet(isPresented: $isImportPickerPresented) {
                VideoImportPicker(
                    onPickedTemporaryURL: { url in
                        Task {
                            await importPickedVideo(from: url)
                        }
                    },
                    onFailure: { message in
                        importStatus = .failed(message)
                    }
                )
                .ignoresSafeArea()
            }
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

            Button {
                isImportPickerPresented = true
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "photo.on.rectangle")
                        .font(.title3.weight(.bold))
                    Text("Import Video")
                        .font(.headline.weight(.heavy))
                    Spacer(minLength: 0)
                    Image(systemName: "tray.and.arrow.down")
                        .font(.headline.weight(.heavy))
                }
                .foregroundStyle(PickleballPalette.cream)
                .padding(.horizontal, 18)
                .frame(height: 52)
                .background(Color.white.opacity(0.10), in: Capsule())
                .overlay(Capsule().stroke(PickleballPalette.cyan.opacity(0.38), lineWidth: 1))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Import Video")
            .disabled(importStatus == .importing)

            Button(action: openWorldViewer) {
                HStack(spacing: 12) {
                    Image(systemName: "scope")
                        .font(.title3.weight(.bold))
                    Text("View 3D World")
                        .font(.headline.weight(.heavy))
                    Spacer(minLength: 0)
                    Image(systemName: "arrow.right")
                        .font(.headline.weight(.heavy))
                }
                .foregroundStyle(PickleballPalette.cream)
                .padding(.horizontal, 18)
                .frame(height: 52)
                .background(Color.white.opacity(0.10), in: Capsule())
                .overlay(Capsule().stroke(.white.opacity(0.22), lineWidth: 1))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("View 3D World")

            Button(action: openRealityReplay) {
                HStack(spacing: 12) {
                    Image(systemName: "cube.transparent")
                        .font(.title3.weight(.bold))
                    Text("Play Baked Replay")
                        .font(.headline.weight(.heavy))
                    Spacer(minLength: 0)
                    Image(systemName: "arrow.right")
                        .font(.headline.weight(.heavy))
                }
                .foregroundStyle(PickleballPalette.cream)
                .padding(.horizontal, 18)
                .frame(height: 52)
                .background(Color.white.opacity(0.08), in: Capsule())
                .overlay(Capsule().stroke(PickleballPalette.mint.opacity(0.36), lineWidth: 1))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Play Baked Replay")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 18)
    }

    private func refreshCaptureItems() async {
        do {
            captureItems = try CaptureLibrary.listPackages(
                packageRootURL: CameraCaptureController.defaultPackageRootURL()
            )
        } catch {
            importStatus = .failed(Self.message(for: error))
        }
    }

    private func importPickedVideo(from temporaryURL: URL) async {
        let startedAt = CFAbsoluteTimeGetCurrent()
        importStatus = .importing
        importSummary = nil
        do {
            let rootURL = CameraCaptureController.defaultPackageRootURL()
            let result = try await Task.detached(priority: .userInitiated) {
                try await CameraRollVideoImporter().importVideo(
                    sourceURL: temporaryURL,
                    packageRootURL: rootURL
                )
            }.value
            try? FileManager.default.removeItem(at: temporaryURL)
            let elapsed = max(0, CFAbsoluteTimeGetCurrent() - startedAt)
            importSummary = PostStopPreviewBuilder.summarize(
                durationSeconds: result.sidecar.recordingDurationS ?? 0,
                requestedFPS: result.sidecar.fps,
                measuredFPS: Double(result.sidecar.fps),
                captureQuality: result.sidecar.captureQuality,
                sampledFrameDetectionCounts: [],
                elapsedBuildSeconds: elapsed,
                provenance: .cameraRollImport
            )
            importStatus = .imported(result.descriptor.sessionID)
            await refreshCaptureItems()
        } catch {
            try? FileManager.default.removeItem(at: temporaryURL)
            importStatus = .failed(Self.message(for: error))
        }
    }

    private func uploadPackage(_ item: CaptureLibraryItem) async {
        guard !renderUploadStatus.isBusy else {
            return
        }

        let rootURL = CameraCaptureController.defaultPackageRootURL()
        let clipURL = rootURL.appendingPathComponent(item.clipRelativePath)
        let sidecarURL = rootURL.appendingPathComponent(item.sidecarRelativePath)
        let client = RenderGatewayClient()
        renderUploadStatus = .submitting(sessionID: item.sessionID)

        do {
            var job = try await client.submitJob(
                upload: RenderGatewayUploadRequest(
                    videoURL: clipURL,
                    captureSidecarURL: sidecarURL,
                    clip: item.sessionID
                )
            )
            renderUploadStatus = .running(sessionID: item.sessionID, job: job, replayURL: client.replayURL(for: job))

            while job.isActive && !Task.isCancelled {
                try await Task.sleep(nanoseconds: 2_500_000_000)
                job = try await client.fetchJobStatus(job.links.status)
                let replayURL = client.replayURL(for: job)
                if job.status == .complete {
                    renderUploadStatus = .complete(sessionID: item.sessionID, job: job, replayURL: replayURL)
                } else if job.status == .failed {
                    renderUploadStatus = .failed(sessionID: item.sessionID, message: job.error ?? "Render job failed")
                } else {
                    renderUploadStatus = .running(sessionID: item.sessionID, job: job, replayURL: replayURL)
                }
            }
        } catch is CancellationError {
            renderUploadStatus = .idle
        } catch {
            renderUploadStatus = .failed(sessionID: item.sessionID, message: Self.message(for: error))
        }
    }

    private static func message(for error: Error) -> String {
        switch error {
        case CameraRollVideoImportError.missingVideoTrack:
            return "No video track"
        case CameraRollVideoImportError.invalidResolution:
            return "Resolution unavailable"
        case CameraRollVideoImportError.invalidFPS:
            return "FPS unavailable"
        case CameraRollVideoImportError.invalidDuration:
            return "Duration unavailable"
        default:
            return String(describing: error)
        }
    }

    private var signalStrip: some View {
        // The "Review 3D" tile mirrors the "View 3D World" CTA above --
        // tapping it opens the same in-app SceneKit world viewer (GLUE-4),
        // rendering the same locked mesh/joints tier + trust badges as the
        // web scrubber. It is not an ON-DEVICE LIVE deep reconstruction
        // claim.
        HStack(spacing: 10) {
            HomeSignalTile(title: "Capture", value: "60 FPS", icon: "bolt.fill", tint: PickleballPalette.lime)
            HomeSignalTile(title: "Sidecar", value: "Ready", icon: "doc.badge.gearshape", tint: PickleballPalette.cyan)
            Button(action: openWorldViewer) {
                HomeSignalTile(title: "Review", value: "3D", icon: "scope", tint: PickleballPalette.coral)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Review 3D")
        }
    }
}

private struct VideoImportPicker: UIViewControllerRepresentable {
    let onPickedTemporaryURL: (URL) -> Void
    let onFailure: (String) -> Void

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var configuration = PHPickerConfiguration(photoLibrary: .shared())
        configuration.filter = .videos
        configuration.selectionLimit = 1
        configuration.preferredAssetRepresentationMode = .current
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_: PHPickerViewController, context _: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onPickedTemporaryURL: onPickedTemporaryURL, onFailure: onFailure)
    }

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        private let onPickedTemporaryURL: (URL) -> Void
        private let onFailure: (String) -> Void

        init(onPickedTemporaryURL: @escaping (URL) -> Void, onFailure: @escaping (String) -> Void) {
            self.onPickedTemporaryURL = onPickedTemporaryURL
            self.onFailure = onFailure
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            guard let provider = results.first?.itemProvider else {
                return
            }
            guard let typeIdentifier = provider.registeredTypeIdentifiers.first(where: { identifier in
                guard let type = UTType(identifier) else {
                    return false
                }
                return type.conforms(to: .movie) || type.conforms(to: .video)
            }) else {
                DispatchQueue.main.async {
                    self.onFailure("Selected item is not a video")
                }
                return
            }

            provider.loadFileRepresentation(forTypeIdentifier: typeIdentifier) { sourceURL, error in
                if let error {
                    DispatchQueue.main.async {
                        self.onFailure(String(describing: error))
                    }
                    return
                }
                guard let sourceURL else {
                    DispatchQueue.main.async {
                        self.onFailure("Video file unavailable")
                    }
                    return
                }

                do {
                    let copiedURL = try Self.copyPickerFileToTemporaryURL(sourceURL)
                    DispatchQueue.main.async {
                        self.onPickedTemporaryURL(copiedURL)
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.onFailure(String(describing: error))
                    }
                }
            }
        }

        private static func copyPickerFileToTemporaryURL(_ sourceURL: URL) throws -> URL {
            let ext = sourceURL.pathExtension.isEmpty ? "mov" : sourceURL.pathExtension
            let destinationURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("camera-roll-import-\(UUID().uuidString).\(ext)")
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            return destinationURL
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

                LivePlayerFootRingOverlay(
                    rings: model.playerFootRings,
                    videoAspectRatio: model.liveOverlayVideoAspectRatio
                )
                .ignoresSafeArea()
                .allowsHitTesting(false)

                LiveBallTrajectoryOverlay(
                    trailPoints: model.ballTrailPoints,
                    contactMarkers: model.ballContactMarkers,
                    videoAspectRatio: model.liveOverlayVideoAspectRatio
                )
                .ignoresSafeArea()
                .allowsHitTesting(false)

                Color.black.opacity(model.status == .idle || model.status == .requestingAccess ? 0.22 : 0)
                    .ignoresSafeArea()

                if isLandscape {
                    landscapeOverlay
                } else {
                    portraitOverlay
                }

                if model.status == .ready || model.isRecording {
                    VStack {
                        Spacer()
                        HStack(alignment: .bottom, spacing: 10) {
                            LiveGuidancePanel(state: model.liveGuidanceState)
                            Spacer(minLength: 8)
                            VStack(alignment: .trailing, spacing: 8) {
                                BallIndicatorBadge(state: model.ballIndicatorState)
                                CourtDotMiniMap(
                                    points: model.courtDotMapPoints,
                                    statusText: model.courtOverlayStatusText,
                                    detailText: model.courtOverlayDetailText
                                )
                            }
                        }
                        .padding(.horizontal, 14)
                        .padding(.bottom, 10)
                    }
                }

                if let summary = model.postStopSummary {
                    PostStopSummaryCard(summary: summary)
                        .padding(20)
                        .transition(.opacity.combined(with: .scale(scale: 0.96)))
                }
            }
            .animation(.smooth(duration: 0.3), value: model.postStopSummary)
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

/// Hosts the GLUE-4 SceneKit world viewer (`PickleballReplay.WorldViewerView`)
/// loaded from the bundled Wolverine fixture
/// (`ios/Replay/Sources/PickleballReplay/Resources/WorldFixture/`, a
/// compact excerpt of `runs/process_video_glue_20260702T_live_wolverine2/...`).
/// Falls back to a plain error screen (never a placeholder/fake world) if
/// the bundled fixture somehow fails to load.
private struct WorldViewerScreen: View {
    let onClose: () -> Void
    @State private var loadedBundle: WorldBundle?
    @State private var loadError: String?

    var body: some View {
        ZStack {
            PickleballCourtBackdrop()
            if let loadedBundle {
                WorldViewerView(bundle: loadedBundle, onClose: onClose)
            } else if let loadError {
                worldLoadErrorView(message: loadError)
            } else {
                ProgressView("Loading world…")
                    .foregroundStyle(PickleballPalette.cream)
            }
        }
        .ignoresSafeArea(edges: .bottom)
        .task {
            do {
                loadedBundle = try WorldBundle.loadBundledSample()
            } catch {
                loadError = "Could not load the bundled world fixture: \(error)"
            }
        }
    }

    private func worldLoadErrorView(message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
            Text(message)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
            Button("Back", action: onClose)
                .buttonStyle(.borderedProminent)
        }
        .foregroundStyle(PickleballPalette.cream)
    }
}

/// Hosts the RealityKit/USDZ baked replay surface for W3-REPLAY-NATIVE
/// phase 2. This intentionally complements the GLUE-4 JSON/SceneKit world
/// viewer rather than replacing it: the USDZ is a baked contact-window mesh
/// flipbook with coarse animation control, while GLUE-4 remains the per-frame
/// tier/trust inspection surface.
private struct RealityReplayScreen: View {
    let onClose: () -> Void
    @State private var loadedAsset: RealityReplayAsset?
    @State private var loadError: String?

    var body: some View {
        ZStack {
            PickleballCourtBackdrop()
            if let loadedAsset {
                loadedReplayView(asset: loadedAsset)
            } else if let loadError {
                replayLoadErrorView(message: loadError)
            } else {
                ProgressView("Loading baked replay...")
                    .foregroundStyle(PickleballPalette.cream)
            }
        }
        .ignoresSafeArea(edges: .bottom)
        .task {
            do {
                loadedAsset = try RealityReplayAsset.loadBundledFixture()
            } catch {
                loadError = "Could not load the bundled Reality replay fixture: \(error)"
            }
        }
    }

    private func loadedReplayView(asset: RealityReplayAsset) -> some View {
        Group {
            if let view = try? RealityReplayView(asset: asset, onClose: onClose) {
                view
            } else {
                replayLoadErrorView(message: "Could not initialize the Reality replay timeline.")
            }
        }
    }

    private func replayLoadErrorView(message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
            Text(message)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
            Button("Back", action: onClose)
                .buttonStyle(.borderedProminent)
        }
        .foregroundStyle(PickleballPalette.cream)
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
    let items: [CaptureLibraryItem]
    let importStatus: CameraRollImportStatus
    let renderUploadStatus: RenderUploadDisplayStatus
    let processPackage: (CaptureLibraryItem) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("Review-ready packages")
                    .font(.title3.weight(.heavy))
                    .foregroundStyle(PickleballPalette.cream)
                Spacer()
                importStatusBadge
            }

            if items.isEmpty {
                Text("No local packages yet.")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white.opacity(0.66))
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                VStack(spacing: 8) {
                    ForEach(items.prefix(4)) { item in
                        CapturePackageRow(
                            item: item,
                            renderUploadStatus: renderUploadStatus,
                            processPackage: processPackage
                        )
                    }
                }
            }

            if renderUploadStatus != .idle {
                RenderUploadStatusCard(status: renderUploadStatus)
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
            }

            Text("Each package keeps `clip.mov` and `capture_sidecar.json` under Documents/captures for the same upload and process-video path.")
                .font(.caption.weight(.medium))
                .foregroundStyle(.white.opacity(0.52))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.black.opacity(0.28))
                .overlay(CourtLineGrid().stroke(.white.opacity(0.16), lineWidth: 1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.white.opacity(0.14), lineWidth: 1)
        )
    }

    @ViewBuilder
    private var importStatusBadge: some View {
        switch importStatus {
        case .idle:
            Text("video + sidecar")
                .foregroundStyle(.white.opacity(0.58))
        case .importing:
            Text("importing")
                .foregroundStyle(PickleballPalette.ink)
                .background(PickleballPalette.cyan, in: Capsule())
        case .imported:
            Text("imported")
                .foregroundStyle(PickleballPalette.ink)
                .background(PickleballPalette.lime, in: Capsule())
        case .failed:
            Text("import failed")
                .foregroundStyle(PickleballPalette.ink)
                .background(PickleballPalette.coral, in: Capsule())
        }
    }
}

private struct CapturePackageRow: View {
    let item: CaptureLibraryItem
    let renderUploadStatus: RenderUploadDisplayStatus
    let processPackage: (CaptureLibraryItem) -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: item.isImported ? "photo.on.rectangle" : "video.fill")
                .font(.caption.weight(.heavy))
                .foregroundStyle(item.isImported ? PickleballPalette.cyan : PickleballPalette.lime)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(item.sessionID)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(PickleballPalette.cream)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let badge = item.badgeText {
                        Text(badge)
                            .font(.caption2.weight(.black))
                            .foregroundStyle(PickleballPalette.ink)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(PickleballPalette.cyan, in: Capsule())
                    }
                }
                Text(detailText)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.white.opacity(0.56))
                    .lineLimit(1)
            }
            Spacer(minLength: 8)
            Button {
                processPackage(item)
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: uploadIconName)
                    Text(uploadButtonText)
                }
                .font(.caption2.weight(.black))
                .foregroundStyle(PickleballPalette.ink)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(uploadButtonColor, in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(renderUploadStatus.isBusy && renderUploadStatus.sessionID != item.sessionID)
            .accessibilityLabel("Process \(item.sessionID) on GPU")
            Text(gradeText)
                .font(.caption2.weight(.black))
                .foregroundStyle(PickleballPalette.ink)
                .padding(.horizontal, 7)
                .padding(.vertical, 4)
                .background(gradeColor, in: Capsule())
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var isCurrentUpload: Bool {
        renderUploadStatus.sessionID == item.sessionID
    }

    private var uploadButtonText: String {
        guard isCurrentUpload else {
            return "Process"
        }
        switch renderUploadStatus {
        case .submitting, .running:
            return "Running"
        case .complete:
            return "Ready"
        case .failed:
            return "Retry"
        case .idle:
            return "Process"
        }
    }

    private var uploadIconName: String {
        guard isCurrentUpload else {
            return "icloud.and.arrow.up"
        }
        switch renderUploadStatus {
        case .submitting, .running:
            return "hourglass"
        case .complete:
            return "checkmark"
        case .failed:
            return "arrow.clockwise"
        case .idle:
            return "icloud.and.arrow.up"
        }
    }

    private var uploadButtonColor: Color {
        guard isCurrentUpload else {
            return PickleballPalette.cyan
        }
        switch renderUploadStatus {
        case .complete:
            return PickleballPalette.lime
        case .failed:
            return PickleballPalette.coral
        case .idle, .submitting, .running:
            return PickleballPalette.cyan
        }
    }

    private var detailText: String {
        let resolution = item.resolution.count == 2 ? "\(item.resolution[0])x\(item.resolution[1])" : "unknown resolution"
        let duration = item.durationSeconds.map { String(format: "%.1fs", $0) } ?? "duration unknown"
        return "\(duration) · \(item.fps) fps · \(resolution)"
    }

    private var gradeText: String {
        switch item.captureQualityGrade {
        case .good:
            return "GOOD"
        case .warn:
            return "WARN"
        case .poor:
            return "POOR"
        }
    }

    private var gradeColor: Color {
        switch item.captureQualityGrade {
        case .good:
            return PickleballPalette.lime
        case .warn:
            return PickleballPalette.cyan
        case .poor:
            return PickleballPalette.coral
        }
    }
}

private struct RenderUploadStatusCard: View {
    let status: RenderUploadDisplayStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(status.stageText, systemImage: statusIconName)
                    .font(.caption.weight(.heavy))
                    .foregroundStyle(PickleballPalette.cream)
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
                Spacer(minLength: 8)
                if let etaText = status.etaText {
                    Text(etaText)
                        .font(.caption2.weight(.black))
                        .foregroundStyle(PickleballPalette.ink)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(PickleballPalette.cyan, in: Capsule())
                }
            }

            ProgressView(value: status.progressFraction)
                .tint(statusTint)

            if let replayURL = status.replayURL {
                Link(destination: replayURL) {
                    Label("Open replay", systemImage: "play.rectangle")
                        .font(.caption.weight(.heavy))
                        .foregroundStyle(PickleballPalette.lime)
                }
            }
        }
        .padding(10)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(statusTint.opacity(0.45), lineWidth: 1)
        )
    }

    private var statusIconName: String {
        switch status {
        case .idle:
            return "pause"
        case .submitting, .running:
            return "bolt.horizontal"
        case .complete:
            return "checkmark.circle"
        case .failed:
            return "exclamationmark.triangle"
        }
    }

    private var statusTint: Color {
        switch status {
        case .complete:
            return PickleballPalette.lime
        case .failed:
            return PickleballPalette.coral
        case .idle, .submitting, .running:
            return PickleballPalette.cyan
        }
    }
}

/// W3-LIVE-MLP surface 1: pre-record capture-quality guidance. Renders
/// `LiveGuidanceEvaluator` output as-is -- a check with no real signal shows
/// as a neutral "not measured yet" chip, never a fake green checkmark.
private struct LiveGuidancePanel: View {
    let state: LiveGuidanceState?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Setup check")
                .font(.caption2.weight(.black))
                .foregroundStyle(.white.opacity(0.56))

            if let state {
                FlowChips(checks: state.checks)
                Text(state.manualFramingTip)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.white.opacity(0.7))
                    .fixedSize(horizontal: false, vertical: true)
                if !state.setupTips.isEmpty {
                    ForEach(state.setupTips, id: \.self) { tip in
                        Text("· \(tip)")
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.5))
                    }
                }
            } else {
                Text("Reading camera signals…")
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.white.opacity(0.6))
            }
        }
        .padding(12)
        .frame(maxWidth: 260, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(.white.opacity(0.14), lineWidth: 1))
    }
}

private struct FlowChips: View {
    let checks: [LiveGuidanceCheck]

    var body: some View {
        // Simple wrapping-free horizontal scroll keeps this robust across
        // portrait/landscape without a custom flow layout for v0.
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(checks, id: \.id) { check in
                    Text(check.title)
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(chipForeground(check.status))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(chipBackground(check.status), in: Capsule())
                        .accessibilityLabel("\(check.title): \(check.detail)")
                }
            }
        }
    }

    private func chipBackground(_ status: LiveCheckStatus) -> Color {
        switch status {
        case .good:
            return PickleballPalette.lime.opacity(0.85)
        case .warn:
            return PickleballPalette.coral.opacity(0.85)
        case .unavailable:
            return Color.white.opacity(0.12)
        }
    }

    private func chipForeground(_ status: LiveCheckStatus) -> Color {
        switch status {
        case .good, .warn:
            return PickleballPalette.ink
        case .unavailable:
            return .white.opacity(0.6)
        }
    }
}

/// W3-LIVE-MLP surface 3: confidence-gated ball indicator. `ball_student` is
/// untrained in this build, so this ALWAYS renders the honest "coming soon"
/// badge -- see `LiveBallIndicatorPolicy` for the gate that guarantees this.
private struct BallIndicatorBadge: View {
    let state: LiveBallIndicatorState

    var body: some View {
        Label(state.badgeText, systemImage: state.availability == .tracking ? "circle.fill" : "clock.badge.questionmark")
            .font(.caption2.weight(.bold))
            .foregroundStyle(PickleballPalette.cream)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.regularMaterial, in: Capsule())
    }
}

/// W3-LIVE-MLP surface 2: live court-dot map. HONESTY: this is a
/// screen-space proxy (see `CourtDotMapBuilder`), not a true top-down court
/// projection -- there is no live calibration in v0. The label says so.
private struct CourtDotMiniMap: View {
    let points: [CourtDotMapPoint]
    let statusText: String
    let detailText: String

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(PickleballPalette.felt.opacity(0.55))
                GeometryReader { proxy in
                    ForEach(points, id: \.trackID) { point in
                        Circle()
                            .fill(dotColor(for: point))
                            .frame(width: 10, height: 10)
                            .position(
                                x: CGFloat(point.normalizedX) * proxy.size.width,
                                y: CGFloat(point.normalizedY) * proxy.size.height
                            )
                    }
                }
            }
            .frame(width: 120, height: 90)
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(.white.opacity(0.18), lineWidth: 1))
            .accessibilityLabel("Court dot map (screen-space proxy, not a calibrated top-down view): \(points.count) players")

            Text("screen-space proxy")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.45))
            Text(statusText)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.white.opacity(0.7))
            if !detailText.isEmpty {
                Text(detailText)
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.45))
                    .frame(maxWidth: 150, alignment: .trailing)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func dotColor(for point: CourtDotMapPoint) -> Color {
        let palette: [Color] = [PickleballPalette.lime, PickleballPalette.cyan, PickleballPalette.coral, PickleballPalette.mint]
        return palette[abs(point.trackID) % palette.count]
    }
}

private struct LivePlayerFootRingOverlay: View {
    let rings: [LivePlayerFootRing]
    let videoAspectRatio: Double

    var body: some View {
        GeometryReader { proxy in
            let renderedRings = LivePlayerFootRingLayout.layout(
                rings: rings,
                viewportWidth: Double(proxy.size.width),
                viewportHeight: Double(proxy.size.height),
                videoAspectRatio: videoAspectRatio
            )
            ZStack {
                ForEach(renderedRings, id: \.ring.trackID) { renderedRing in
                    ZStack {
                        Ellipse()
                            .fill(ringColor(for: renderedRing.ring).opacity(renderedRing.ring.fillOpacity))
                        Ellipse()
                            .stroke(
                                ringColor(for: renderedRing.ring).opacity(renderedRing.ring.strokeOpacity),
                                lineWidth: renderedRing.ring.isStale ? 2 : 3
                            )
                        Ellipse()
                            .stroke(.white.opacity(renderedRing.ring.isStale ? 0.14 : 0.22), lineWidth: 1)
                            .scaleEffect(0.70)
                    }
                    .frame(
                        width: max(26, CGFloat(renderedRing.width)),
                        height: max(10, CGFloat(renderedRing.height))
                    )
                    .position(
                        x: CGFloat(renderedRing.centerX),
                        y: CGFloat(renderedRing.centerY)
                    )
                    .shadow(
                        color: ringColor(for: renderedRing.ring).opacity(renderedRing.ring.strokeOpacity * 0.45),
                        radius: renderedRing.ring.isStale ? 4 : 8
                    )
                }
            }
        }
        .accessibilityHidden(true)
    }

    private func ringColor(for ring: LivePlayerFootRing) -> Color {
        let palette: [Color] = [
            PickleballPalette.lime,
            PickleballPalette.cyan,
            PickleballPalette.coral,
            Color(red: 0.98, green: 0.78, blue: 0.18),
        ]
        return palette[ring.colorIndex % palette.count]
    }
}

private struct LiveBallTrajectoryOverlay: View {
    let trailPoints: [LiveBallTrailPoint]
    let contactMarkers: [LiveBallContactMarker]
    let videoAspectRatio: Double

    var body: some View {
        GeometryReader { proxy in
            let rendered = LiveBallOverlayLayout.layout(
                overlay: LiveBallOverlayState(trailPoints: trailPoints, contactMarkers: contactMarkers),
                viewportWidth: Double(proxy.size.width),
                viewportHeight: Double(proxy.size.height),
                videoAspectRatio: videoAspectRatio
            )
            ZStack {
                ForEach(rendered.trailSegments.indices, id: \.self) { segmentIndex in
                    let segment = rendered.trailSegments[segmentIndex]
                    Path { path in
                        guard let first = segment.points.first else {
                            return
                        }
                        path.move(to: CGPoint(x: first.centerX, y: first.centerY))
                        for point in segment.points.dropFirst() {
                            path.addLine(to: CGPoint(x: point.centerX, y: point.centerY))
                        }
                    }
                    .stroke(
                        Color(red: 1.0, green: 0.84, blue: 0.18).opacity(trailOpacity(segment.points)),
                        style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round)
                    )
                }

                ForEach(rendered.trailPoints, id: \.point.frameIndex) { renderedPoint in
                    Circle()
                        .fill(Color(red: 1.0, green: 0.95, blue: 0.34).opacity(renderedPoint.point.opacity))
                        .frame(
                            width: max(5, CGFloat(renderedPoint.radius * 2)),
                            height: max(5, CGFloat(renderedPoint.radius * 2))
                        )
                        .position(x: CGFloat(renderedPoint.centerX), y: CGFloat(renderedPoint.centerY))
                        .shadow(color: Color(red: 1.0, green: 0.84, blue: 0.18).opacity(0.45), radius: 5)
                }

                ForEach(rendered.contactMarkers, id: \.marker.frameIndex) { renderedMarker in
                    ZStack {
                        Circle()
                            .stroke(
                                PickleballPalette.coral.opacity(renderedMarker.marker.opacity),
                                lineWidth: 3
                            )
                        Circle()
                            .stroke(.white.opacity(renderedMarker.marker.opacity * 0.45), lineWidth: 1)
                            .scaleEffect(0.58)
                    }
                    .frame(
                        width: max(24, CGFloat(renderedMarker.radius * 2)),
                        height: max(24, CGFloat(renderedMarker.radius * 2))
                    )
                    .position(x: CGFloat(renderedMarker.centerX), y: CGFloat(renderedMarker.centerY))
                    .shadow(color: PickleballPalette.coral.opacity(renderedMarker.marker.opacity * 0.42), radius: 8)
                }
            }
        }
        .accessibilityHidden(true)
    }

    private func trailOpacity(_ points: [RenderedLiveBallTrailPoint]) -> Double {
        points.map(\.point.opacity).max() ?? 0
    }
}

/// W3-LIVE-MLP surface 4: post-stop preview, must appear inside the <10s
/// gate. Every number here traces to a real readback (`PostStopPreviewSummary`)
/// -- the player-count estimate is explicitly labeled as an estimate from a
/// handful of sampled frames, never presented as an exact count.
private struct PostStopSummaryCard: View {
    let summary: PostStopPreviewSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(summary.provenance == .cameraRollImport ? "Clip imported" : "Clip saved")
                    .font(.headline.weight(.heavy))
                    .foregroundStyle(PickleballPalette.cream)
                Spacer()
                Text(summary.isWithinPreviewBudget ? "<10s preview" : "preview slow")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(PickleballPalette.ink)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(summary.isWithinPreviewBudget ? PickleballPalette.lime : PickleballPalette.coral, in: Capsule())
            }

            summaryRow(label: "Duration", value: String(format: "%.1fs", summary.durationSeconds))
            summaryRow(label: "Frame rate", value: frameRateText)
            summaryRow(label: "Capture quality", value: gradeText(summary.captureQualityGrade))
            summaryRow(
                label: "Players (estimate)",
                value: summary.estimatedPlayerCount.map { "~\($0) from \(summary.playerCountSampleFrameCount) sampled frames" }
                    ?? missingPlayerEstimateText
            )

            if !summary.captureQualityReasons.isEmpty {
                Text(summary.captureQualityReasons.map { LiveGuidanceEvaluator.humanReadableSetupTip(for: $0) }.joined(separator: " · "))
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.55))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .frame(maxWidth: 320, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(.white.opacity(0.16), lineWidth: 1))
        .shadow(color: .black.opacity(0.3), radius: 20, y: 10)
    }

    private func summaryRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white.opacity(0.6))
            Spacer()
            Text(value)
                .font(.caption.weight(.bold))
                .foregroundStyle(PickleballPalette.cream)
        }
    }

    private var frameRateText: String {
        summary.provenance == .cameraRollImport
            ? "\(summary.requestedFPS) fps probed"
            : "\(summary.requestedFPS) fps requested"
    }

    private var missingPlayerEstimateText: String {
        summary.provenance == .cameraRollImport
            ? "no sampled detector pass"
            : "not enough sampled frames yet"
    }

    private func gradeText(_ grade: CaptureQuality.Grade) -> String {
        switch grade {
        case .good:
            return "Good"
        case .warn:
            return "Warn"
        case .poor:
            return "Poor"
        }
    }
}

#Preview {
    AppRootView()
}
