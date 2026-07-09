import SwiftUI
import UIKit
import PickleballCapture
import PickleballCore
import PickleballFastTier
import PickleballReplay
import PickleballUpload

struct AppRootView: View {
    var body: some View {
        DinkVisionAppRootView()
    }
}

/// Auth is enabled for account-backed operations, but the shell itself is
/// intentionally never gated: local recording and Replays remain available
/// while signed out, and upload presents sign-in contextually.
let dinkVisionAuthGateEnabled = true

private struct DinkVisionAppRootView: View {
    private let configuration: DinkVisionRuntimeConfiguration
    @State private var isSplashVisible: Bool
    @State private var isSignedIn: Bool

    init(configuration: DinkVisionRuntimeConfiguration = .current()) {
        self.configuration = configuration
        _isSplashVisible = State(initialValue: !configuration.skipSplash)
        _isSignedIn = State(initialValue: !dinkVisionAuthGateEnabled || AuthTokenStore().hasAccessToken)
    }

    var body: some View {
        let access = DinkVisionLaunchAccessState(
            authGateEnabled: dinkVisionAuthGateEnabled,
            isSplashVisible: isSplashVisible,
            isSignedIn: isSignedIn
        )
        ZStack {
            DinkVisionTabShell(
                isActive: access.recordTabReachable,
                configuration: configuration,
                isSignedIn: $isSignedIn
            )
                .allowsHitTesting(!isSplashVisible)

            if isSplashVisible {
                DinkVisionSplashView {
                    withAnimation(.easeInOut(duration: 0.22)) {
                        isSplashVisible = false
                    }
                }
                .transition(.opacity)
                .zIndex(2)
            }
        }
        .background(DinkVisionColor.cream)
        .preferredColorScheme(.light)
    }
}

private enum DinkVisionChromeLayout {
    static let tabLayout = DinkVisionTabLayoutModel.brandV4

    static var tabOverlayHeight: CGFloat {
        tabLayout.totalOverlayHeight(tabBarHeight: DinkVisionMetric.tabBarHeight)
    }

    static var scrollBottomPadding: CGFloat {
        tabLayout.contentBottomPadding(tabBarHeight: DinkVisionMetric.tabBarHeight)
    }
}

private struct DinkVisionSplashView: View {
    let onFinish: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var machine = DinkVisionSplashStateMachine(reducedMotion: false)
    @State private var phase: DinkVisionSplashPhase = .settle
    @State private var settleScale: CGFloat = 1.06
    @State private var lidClosure: CGFloat = 0
    @State private var openUpScale: CGFloat = 1
    @State private var overlayOpacity: Double = 1

    var body: some View {
        GeometryReader { proxy in
            let markFrame = DinkVisionSplashLidGeometry.markFrame(in: proxy.size)
            ZStack {
                DinkVisionColor.cream
                    .opacity(overlayOpacity)
                    .ignoresSafeArea()

                DinkVisionSplashMarkComposition(lidClosure: lidClosure)
                    .frame(width: markFrame.width, height: markFrame.height)
                    .scaleEffect(settleScale * openUpScale, anchor: UnitPoint(
                        x: DinkVisionSplashLidGeometry.eyeCenterXRatio,
                        y: DinkVisionSplashLidGeometry.eyeCenterYRatio
                    ))
                    .opacity(overlayOpacity)
                    .position(x: markFrame.midX, y: markFrame.midY)
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .ignoresSafeArea()
        }
        .accessibilityHidden(true)
        .accessibilityIdentifier("DinkVisionSplash")
        .task {
            machine = DinkVisionSplashStateMachine(reducedMotion: reduceMotion)
            if reduceMotion {
                withAnimation(.easeInOut(duration: 0.18)) {
                    overlayOpacity = 0
                }
                try? await Task.sleep(nanoseconds: DinkVisionSplashTiming.reducedMotionCrossfadeNanoseconds)
                onFinish()
                return
            }

            phase = .settle
            settleScale = 1.06
            lidClosure = 0
            openUpScale = 1
            overlayOpacity = 1
            withAnimation(.interpolatingSpring(stiffness: 210, damping: 20)) {
                settleScale = 1
            }
            try? await Task.sleep(nanoseconds: DinkVisionSplashTiming.settleNanoseconds)

            phase = machine.advance()
            withAnimation(.easeInOut(duration: 0.26)) {
                lidClosure = 1
            }
            try? await Task.sleep(nanoseconds: DinkVisionSplashTiming.blinkCloseNanoseconds)
            try? await Task.sleep(nanoseconds: DinkVisionSplashTiming.blinkHoldNanoseconds)
            withAnimation(.easeInOut(duration: 0.34)) {
                lidClosure = 0
            }
            try? await Task.sleep(nanoseconds: DinkVisionSplashTiming.blinkOpenNanoseconds)

            phase = machine.advance()
            withAnimation(.easeInOut(duration: 0.38)) {
                openUpScale = 3.2
                overlayOpacity = 0
            }
            try? await Task.sleep(nanoseconds: DinkVisionSplashTiming.openUpNanoseconds)

            phase = machine.advance()
            onFinish()
        }
    }
}

private struct DinkVisionSplashMarkComposition: View {
    var lidClosure: CGFloat

    var body: some View {
        GeometryReader { proxy in
            let markFrame = CGRect(origin: .zero, size: proxy.size)
            ZStack {
                DinkVisionOwnerMark(height: proxy.size.height)
                    .frame(width: proxy.size.width, height: proxy.size.height)

                SplashLidCoverShape(isUpper: true, closure: lidClosure)
                    .fill(DinkVisionColor.cream)
                SplashLidCoverShape(isUpper: false, closure: lidClosure)
                    .fill(DinkVisionColor.cream)
                SplashLidStrokeShape(isUpper: true, closure: lidClosure)
                    .stroke(
                        DinkVisionColor.ink,
                        style: StrokeStyle(
                            lineWidth: DinkVisionSplashLidGeometry.strokeWidth(in: markFrame),
                            lineCap: .round,
                            lineJoin: .round
                        )
                    )
                SplashLidStrokeShape(isUpper: false, closure: lidClosure)
                    .stroke(
                        DinkVisionColor.ink,
                        style: StrokeStyle(
                            lineWidth: DinkVisionSplashLidGeometry.strokeWidth(in: markFrame),
                            lineCap: .round,
                            lineJoin: .round
                        )
                    )
            }
        }
    }
}

private struct SplashLidCoverShape: Shape {
    var isUpper: Bool
    var closure: CGFloat

    var animatableData: CGFloat {
        get { closure }
        set { closure = newValue }
    }

    func path(in rect: CGRect) -> Path {
        let cover = DinkVisionSplashLidGeometry.lidCover(isUpper: isUpper, closure: closure, markFrame: rect)
        guard cover.coverRect.height > 0.001 else {
            return Path()
        }
        var path = Path()
        path.move(to: cover.outerCurve.left)
        path.addQuadCurve(to: cover.outerCurve.right, control: cover.outerCurve.control)
        path.addLine(to: cover.innerCurve.right)
        path.addQuadCurve(to: cover.innerCurve.left, control: cover.innerCurve.control)
        path.closeSubpath()
        return path
    }
}

private struct SplashLidStrokeShape: Shape {
    var isUpper: Bool
    var closure: CGFloat

    var animatableData: CGFloat {
        get { closure }
        set { closure = newValue }
    }

    func path(in rect: CGRect) -> Path {
        guard closure > 0.001 else {
            return Path()
        }
        let curve = DinkVisionSplashLidGeometry.lidCover(isUpper: isUpper, closure: closure, markFrame: rect).innerCurve
        var path = Path()
        path.move(to: curve.left)
        path.addQuadCurve(to: curve.right, control: curve.control)
        return path
    }
}

private struct DinkVisionTabShell: View {
    var isActive: Bool = true
    private let configuration: DinkVisionRuntimeConfiguration
    @Binding private var isSignedIn: Bool
    @State private var selectedTab: DinkVisionTabKind = .record
    @StateObject private var recordModel: CaptureViewModel
    @StateObject private var uploadCoordinator: DinkVisionUploadCoordinator
    @State private var finishedRecordingPrompt: CameraRecordingResult?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    init(
        isActive: Bool = true,
        configuration: DinkVisionRuntimeConfiguration = .current(),
        isSignedIn: Binding<Bool> = .constant(false)
    ) {
        self.isActive = isActive
        self.configuration = configuration
        _isSignedIn = isSignedIn
        _recordModel = StateObject(wrappedValue: configuration.makeCaptureViewModel())
        _uploadCoordinator = StateObject(wrappedValue: configuration.makeUploadCoordinator())
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            Group {
                switch selectedTab {
                case .replays:
                    DinkVisionReplaysScreen(
                        dataSource: configuration.makeReplayDataSource(),
                        configuration: configuration,
                        uploadCoordinator: uploadCoordinator
                    )
                case .stats:
                    DinkVisionStatsScreen()
                case .record:
                    DinkVisionRecordScreen(isActive: isActive, model: recordModel)
                case .coach:
                    DinkVisionCoachScreen()
                case .profile:
                    DinkVisionProfileScreen(
                        isSignedIn: isSignedIn,
                        autoUploadAfterRecording: $uploadCoordinator.autoUploadAfterRecording,
                        onSignIn: { uploadCoordinator.isSignInPresented = true }
                    )
                }
            }
            .id(selectedTab)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityIdentifier("DinkVisionSelectedTab-\(selectedTab.rawValue)")
            .transition(reduceMotion ? .opacity : .asymmetric(
                insertion: .modifier(
                    active: DinkVisionStickerTransitionModifier(
                        offsetX: DinkVisionScreenMotionParameters.default.slidePoints,
                        rotationDegrees: DinkVisionScreenMotionParameters.default.rotationDegrees
                    ),
                    identity: DinkVisionStickerTransitionModifier(offsetX: 0, rotationDegrees: 0)
                ).combined(with: .opacity),
                removal: .modifier(
                    active: DinkVisionStickerTransitionModifier(
                        offsetX: -DinkVisionScreenMotionParameters.default.slidePoints,
                        rotationDegrees: -DinkVisionScreenMotionParameters.default.rotationDegrees
                    ),
                    identity: DinkVisionStickerTransitionModifier(offsetX: 0, rotationDegrees: 0)
                ).combined(with: .opacity)
            ))

            DinkVisionTabBar(
                selectedTab: $selectedTab,
                recordModel: recordModel,
                forceRecordPressed: configuration.forceRecordPressed
            )
        }
        .animation(.spring(response: 0.34, dampingFraction: 0.72), value: selectedTab)
        .ignoresSafeArea(edges: selectedTab == .record ? .all : .bottom)
        .onChange(of: recordModel.lastFinishedCapture) { _, recording in
            guard let recording else { return }
            uploadCoordinator.recordingFinished(recording)
            if !uploadCoordinator.autoUploadAfterRecording {
                finishedRecordingPrompt = recording
            }
        }
        .alert(
            "Recording saved",
            isPresented: Binding(
                get: { finishedRecordingPrompt != nil },
                set: { if !$0 { finishedRecordingPrompt = nil } }
            )
        ) {
            Button("Upload") {
                if let recording = finishedRecordingPrompt {
                    uploadCoordinator.uploadFinishedRecording(recording)
                }
                finishedRecordingPrompt = nil
            }
            Button("Keep Local", role: .cancel) { finishedRecordingPrompt = nil }
        } message: {
            Text("Your video and exact capture sidecar are stored locally. Upload now or later from Replays.")
        }
        .sheet(isPresented: $uploadCoordinator.isSignInPresented) {
            SignInView(authApiClient: configuration.makeAuthApiClient()) {
                isSignedIn = true
                uploadCoordinator.signedIn()
            }
        }
        .task {
            guard let items = try? CaptureLibrary.listPackages(
                packageRootURL: CameraCaptureController.defaultPackageRootURL()
            ) else { return }
            await uploadCoordinator.resumeInterruptedUploads(items: items)
        }
    }
}

private struct DinkVisionStickerTransitionModifier: ViewModifier {
    var offsetX: CGFloat
    var rotationDegrees: Double

    func body(content: Content) -> some View {
        content
            .offset(x: offsetX, y: 0)
            .rotationEffect(.degrees(rotationDegrees), anchor: .bottom)
    }
}

private struct DinkVisionTabBar: View {
    @Binding var selectedTab: DinkVisionTabKind
    @ObservedObject var recordModel: CaptureViewModel
    var forceRecordPressed: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let layout = DinkVisionChromeLayout.tabLayout

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .bottom) {
                HStack(alignment: .bottom, spacing: 0) {
                    ForEach(layout.tabs) { tab in
                        if tab == .record {
                            Color.clear
                                .frame(maxWidth: .infinity, minHeight: 54)
                                .accessibilityHidden(true)
                        } else {
                            Button {
                                selectedTab = tab
                            } label: {
                                DinkVisionTabItem(tab: tab, isSelected: selectedTab == tab)
                                    .frame(maxWidth: .infinity, minHeight: 54)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(tab.title)
                            .accessibilityIdentifier("DinkVisionTab-\(tab.rawValue)")
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.top, 15)
                .padding(.bottom, 18)
                .frame(height: DinkVisionMetric.tabBarHeight)
                .background(
                    TopRoundedRectangle(radius: DinkVisionMetric.tabBarRadius)
                        .fill(DinkVisionColor.ink)
                )
                .accessibilityIdentifier("DinkVisionTabBarRail")

                Button {
                    Task {
                        await handleRecordTap()
                    }
                } label: {
                    DinkVisionTexturedRecordButton(
                        isRecording: recordModel.isRecording,
                        isEnabled: canRecordFromTab,
                        reduceMotion: reduceMotion,
                        forcePressed: forceRecordPressed
                    )
                }
                .buttonStyle(.plain)
                .disabled(!canRecordFromTab)
                .frame(width: layout.recordButtonDiameter + 14, height: layout.recordButtonDiameter + 14)
                .position(
                    x: proxy.size.width / 2,
                    y: layout.recordButtonCenterY(tabBarHeight: DinkVisionMetric.tabBarHeight)
                )
                .accessibilityLabel(recordModel.isRecording ? "Stop recording" : "Start recording")
                .accessibilityIdentifier("DinkVisionRecordButton")
            }
        }
        .frame(height: layout.totalOverlayHeight(tabBarHeight: DinkVisionMetric.tabBarHeight))
        .accessibilityIdentifier("DinkVisionTabBarOverlay")
    }

    private var canRecordFromTab: Bool {
        recordModel.isRecordButtonEnabled
    }

    private func handleRecordTap() async {
        selectedTab = .record
        DinkVisionHaptics.impact(.medium)
        await recordModel.handleRecordTap()
    }
}

private struct DinkVisionTabItem: View {
    let tab: DinkVisionTabKind
    let isSelected: Bool

    var body: some View {
        VStack(spacing: 5) {
            Image(systemName: tab.symbolName)
                .font(.system(size: 20, weight: .heavy))
                .frame(width: 30, height: 26)
            Text(tab.title)
                .font(.system(size: 10, weight: .heavy, design: .rounded))
                .overlay(alignment: .bottom) {
                    if isSelected {
                        SketchyUnderline()
                            .stroke(DinkVisionColor.ballYellow, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                            .frame(height: 8)
                            .offset(y: 8)
                            .strokeDrawOn()
                    }
                }
        }
        .foregroundStyle(isSelected ? DinkVisionColor.cream : Color.white.opacity(0.48))
        .contentShape(Rectangle())
    }
}

private struct SketchyUnderline: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX + 2, y: rect.midY + 1))
        path.addQuadCurve(
            to: CGPoint(x: rect.maxX - 2, y: rect.midY),
            control: CGPoint(x: rect.midX, y: rect.minY)
        )
        return path
    }
}

private enum DinkVisionHaptics {
    @MainActor
    static func impact(_ style: UIImpactFeedbackGenerator.FeedbackStyle) {
        let generator = UIImpactFeedbackGenerator(style: style)
        generator.prepare()
        generator.impactOccurred()
    }
}

private struct DinkVisionTexturedRecordButton: View {
    var isRecording: Bool
    var isEnabled: Bool
    var reduceMotion: Bool
    var forcePressed: Bool = false
    @State private var isPressed = false
    @State private var breath = false
    @State private var wobbleDegrees: Double = 0

    private let layout = DinkVisionChromeLayout.tabLayout

    var body: some View {
        let visual = visualState
        ZStack {
            recordDisc(visual: visual)
                .frame(width: layout.recordButtonDiameter, height: layout.recordButtonDiameter)
                .scaleEffect(visual.scale * (breathScale(for: visual)))
                .rotationEffect(.degrees(wobbleDegrees))
                .shadow(color: .black.opacity(isPressed ? 0.16 : 0.28), radius: isPressed ? 5 : 12, y: isPressed ? 3 : 8)

            if isRecording {
                RoundedRectangle(cornerRadius: 11, style: .continuous)
                    .fill(DinkVisionColor.cream)
                    .frame(width: 30, height: 30)
                    .shadow(color: DinkVisionColor.cream.opacity(0.26), radius: 8)
                    .transition(.scale.combined(with: .opacity))
            }
        }
        .frame(width: 86, height: 86)
        .opacity(isEnabled ? 1 : 0.62)
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: DinkVisionRecordButtonVisual.idle.breathingDurationSeconds).repeatForever(autoreverses: true)) {
                breath = true
            }
        }
        .onChange(of: isRecording) { _, newValue in
            guard newValue, !reduceMotion else { return }
            wobbleDegrees = -2
            withAnimation(.easeInOut(duration: 0.09)) {
                wobbleDegrees = 2
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.09) {
                withAnimation(.easeInOut(duration: 0.09)) {
                    wobbleDegrees = 0
                }
            }
        }
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in isPressed = true }
                .onEnded { _ in isPressed = false }
        )
    }

    private var visualState: DinkVisionRecordButtonVisual {
        if isRecording {
            return .recording
        }
        return (isPressed || forcePressed) ? .pressed : .idle
    }

    private func breathScale(for visual: DinkVisionRecordButtonVisual) -> CGFloat {
        guard !reduceMotion, !isRecording, !isPressed, !forcePressed else {
            return 1
        }
        return breath ? visual.breathingScaleRange.upperBound : visual.breathingScaleRange.lowerBound
    }

    private func recordDisc(visual: DinkVisionRecordButtonVisual) -> some View {
        ZStack {
            if visual.centerShape == .roundedSquareStop {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(
                        RadialGradient(
                            colors: [Color(hex: 0xFF705D), DinkVisionColor.trailRed, Color(hex: 0xB82D22)],
                            center: .topLeading,
                            startRadius: 4,
                            endRadius: 68
                        )
                    )
                    .overlay {
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .stroke(DinkVisionColor.trailRed, lineWidth: 4)
                    }
                    .overlay {
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .stroke(Color.white.opacity(0.22), lineWidth: 1.5)
                            .blur(radius: 0.4)
                            .offset(x: -1, y: -1)
                            .padding(5)
                    }
            } else {
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [Color(hex: 0xFFD85A), DinkVisionColor.ballYellow, Color(hex: 0xE4B437)],
                            center: .topLeading,
                            startRadius: 4,
                            endRadius: 68
                        )
                    )
                    .overlay {
                        Circle()
                            .stroke(visual.ring == .trailRed ? DinkVisionColor.trailRed : DinkVisionColor.ink, lineWidth: 4)
                    }
                    .overlay {
                        Circle()
                            .stroke(Color.white.opacity(0.30), lineWidth: 1.5)
                            .blur(radius: 0.4)
                            .offset(x: -1, y: -1)
                            .mask(Circle().padding(5))
                    }

                ForEach(0..<visual.holeCount, id: \.self) { index in
                    embossedHole(index: index, visual: visual)
                }
            }
        }
    }

    private func embossedHole(index: Int, visual: DinkVisionRecordButtonVisual) -> some View {
        let diameter = layout.recordButtonDiameter * visual.holeDiameterRatio
        let point = Self.holePoint(index)
        return Circle()
            .fill(
                RadialGradient(
                    colors: [Color(hex: 0xB9851F), Color(hex: 0xD9A82F), Color(hex: 0xF5CC48).opacity(0.55)],
                    center: .bottomTrailing,
                    startRadius: 1,
                    endRadius: diameter
                )
            )
            .overlay(alignment: .topLeading) {
                Circle()
                    .stroke(Color.black.opacity(visual.innerShadowStrength), lineWidth: 2)
                    .blur(radius: 1.2)
                    .offset(x: -1.2, y: -1.2)
                    .mask(Circle())
            }
            .overlay(alignment: .bottomTrailing) {
                Circle()
                    .stroke(Color.white.opacity(0.28), lineWidth: 1)
                    .blur(radius: 0.5)
                    .offset(x: 1, y: 1)
                    .mask(Circle())
            }
            .frame(width: diameter, height: diameter)
            .position(
                x: layout.recordButtonDiameter * point.x,
                y: layout.recordButtonDiameter * point.y
            )
    }

    nonisolated private static func holePoint(_ index: Int) -> CGPoint {
        switch index % 8 {
        case 0: return CGPoint(x: 0.50, y: 0.25)
        case 1: return CGPoint(x: 0.71, y: 0.32)
        case 2: return CGPoint(x: 0.76, y: 0.55)
        case 3: return CGPoint(x: 0.62, y: 0.74)
        case 4: return CGPoint(x: 0.38, y: 0.74)
        case 5: return CGPoint(x: 0.24, y: 0.55)
        case 6: return CGPoint(x: 0.29, y: 0.32)
        default: return CGPoint(x: 0.50, y: 0.50)
        }
    }
}

private struct TopRoundedRectangle: Shape {
    var radius: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let r = min(radius, rect.width / 2, rect.height / 2)
        path.move(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY + r))
        path.addQuadCurve(to: CGPoint(x: rect.minX + r, y: rect.minY), control: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX - r, y: rect.minY))
        path.addQuadCurve(to: CGPoint(x: rect.maxX, y: rect.minY + r), control: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}

private struct DinkVisionRecordScreen: View {
    var isActive: Bool
    @ObservedObject private var model: CaptureViewModel
    @State private var isCourtOverlayEnabled = true
    @State private var selectedPolicyHint: String?

    init(
        isActive: Bool = true,
        model: CaptureViewModel = CaptureViewModel()
    ) {
        self.isActive = isActive
        self.model = model
    }

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                cameraLayer

                if isCourtOverlayEnabled {
                    LivePlayerFootRingOverlay(
                        rings: model.playerFootRings,
                        videoAspectRatio: model.liveOverlayVideoAspectRatio
                    )
                    .allowsHitTesting(false)

                    LiveBallTrajectoryOverlay(
                        trailPoints: model.ballTrailPoints,
                        contactMarkers: model.ballContactMarkers,
                        videoAspectRatio: model.liveOverlayVideoAspectRatio
                    )
                    .allowsHitTesting(false)
                }

                VStack(spacing: 0) {
                    recordHeader
                        .padding(.top, max(48, proxy.safeAreaInsets.top + 10))
                    policyChips
                    if let blockedReason = model.blockedReason {
                        blockedReasonBanner(blockedReason)
                            .padding(.top, 10)
                    }
                    Spacer(minLength: 20)
                    recordFooter
                        .padding(.bottom, DinkVisionChromeLayout.scrollBottomPadding)
                }
                .padding(.horizontal, 18)

                if case .saving = model.recordFlowPhase {
                    saveCard(title: "Saving\nsidecar", detail: "local")
                }
                if case .done = model.recordFlowPhase {
                    saveCard(title: "Capture\nsaved", detail: "sidecar ready")
                }

                if shouldShowPermissionPrimer {
                    permissionPrimer
                }

                if let selectedPolicyHint {
                    VStack {
                        Spacer()
                        Text(selectedPolicyHint)
                            .font(.system(size: 13, weight: .heavy, design: .rounded))
                            .foregroundStyle(DinkVisionColor.ink)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(DinkVisionColor.ballYellow, in: Capsule())
                            .padding(.bottom, DinkVisionChromeLayout.scrollBottomPadding + 72)
                            .onTapGesture {
                                self.selectedPolicyHint = nil
                            }
                    }
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }
            .animation(.spring(response: 0.32, dampingFraction: 0.86), value: model.recordFlowPhase)
            .accessibilityIdentifier("DinkVisionScreen-Record")
            .task(id: isActive) {
                guard isActive else {
                    return
                }
                if model.status == .idle {
                    await model.prepare()
                } else {
                    await model.refreshSetupPassIfNeeded()
                }
            }
            .onChange(of: proxy.size.width > proxy.size.height) {
                Task {
                    await model.updateOrientation(isLandscapeViewport: proxy.size.width > proxy.size.height)
                }
            }
        }
        .background(DinkVisionColor.courtGreen)
    }

    private var cameraLayer: some View {
        ZStack {
            DinkVisionCourtPreviewBackdrop()
            CameraPreviewView(session: model.session, videoRotationAngle: model.previewRotationAngle)
                .opacity(model.status == .ready || model.isRecording ? 1 : 0.16)
        }
        .ignoresSafeArea()
    }

    private var recordHeader: some View {
        HStack(alignment: .top) {
            DinkVisionOwnerMark(height: 44)
                .padding(.horizontal, 7)
                .padding(.vertical, 5)
                .background(DinkVisionColor.cream, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .shadow(color: .black.opacity(0.24), radius: 10, y: 4)
            Spacer()
            VStack(alignment: .trailing, spacing: 8) {
                if model.isRecording {
                    recordingBadge
                } else {
                    statusChip(title: preRecordStatusText, status: preRecordChipStatus)
                }
                Button {
                    isCourtOverlayEnabled.toggle()
                } label: {
                    Label(isCourtOverlayEnabled ? "Court overlay" : "Overlay off", systemImage: isCourtOverlayEnabled ? "viewfinder" : "viewfinder.circle")
                        .font(.system(size: 12, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 7)
                        .background(Color.black.opacity(0.45), in: Capsule())
                }
                .buttonStyle(.plain)
                .frame(minHeight: 44)
            }
        }
    }

    private var policyChips: some View {
        FlowLayout(spacing: 8) {
            ForEach(model.policyChips) { chip in
                Button {
                    selectedPolicyHint = chip.hint
                } label: {
                    statusChip(title: chip.title, status: chip.status)
                }
                .buttonStyle(.plain)
                .frame(minHeight: 44)
            }
        }
        .padding(.top, 22)
    }

    private var recordFooter: some View {
        VStack(spacing: 10) {
            Text(model.isRecording ? model.courtOverlayStatusText : "Frame the full court, then one tap")
                .font(.system(size: 15, weight: .heavy, design: .rounded))
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)
                .shadow(color: .black.opacity(0.38), radius: 8, y: 1)
        }
    }

    private func blockedReasonBanner(_ reason: String) -> some View {
        Label(reason, systemImage: "exclamationmark.triangle.fill")
            .font(.system(size: 14, weight: .heavy, design: .rounded))
            .foregroundStyle(DinkVisionColor.ink)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(DinkVisionColor.cream, in: Capsule())
            .shadow(color: .black.opacity(0.20), radius: 10, y: 4)
            .accessibilityIdentifier("DinkVisionRecordBlockedReason")
    }

    private var recordingBadge: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            let elapsed = model.recordingStartedAt.map { max(0, Int(context.date.timeIntervalSince($0))) } ?? 0
            statusChip(title: "REC \(elapsed / 60):\(String(format: "%02d", elapsed % 60))", status: .warning, fill: DinkVisionColor.trailRed)
        }
    }

    private func statusChip(title: String, status: DinkVisionPolicyChipStatus, fill: Color? = nil) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(status == .pass ? DinkVisionColor.success : DinkVisionColor.ballYellow)
                .frame(width: 8, height: 8)
            Text(title)
                .font(.system(size: 12, weight: .heavy, design: .rounded))
                .lineLimit(1)
                .minimumScaleFactor(0.72)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background((fill ?? Color.black.opacity(0.50)), in: Capsule())
    }

    private func saveCard(title: String, detail: String) -> some View {
        VStack {
            Spacer()
            BallTrailLoadingView(title: title, detail: detail)
                .padding(.horizontal, 18)
                .padding(.bottom, DinkVisionChromeLayout.scrollBottomPadding + 42)
        }
        .transition(.opacity.combined(with: .scale(scale: 0.98)))
    }

    private var permissionPrimer: some View {
        VStack(spacing: 12) {
            DinkVisionOwnerMark(height: 82)
            Text("Camera + mic")
                .font(.system(size: 22, weight: .heavy, design: .rounded))
                .foregroundStyle(DinkVisionColor.ink)
            Text("Accept access, then the record button is ready.")
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundStyle(DinkVisionColor.mutedText)
                .multilineTextAlignment(.center)
        }
        .padding(22)
        .frame(maxWidth: 270)
        .background(.white, in: RoundedRectangle(cornerRadius: DinkVisionMetric.cardRadius, style: .continuous))
        .overlay(alignment: .topTrailing) {
            SketchSlashes(color: DinkVisionColor.ink, lineWidth: 4)
                .frame(width: 38, height: 38)
                .offset(x: -18, y: -10)
        }
        .shadow(color: .black.opacity(0.20), radius: 24, y: 12)
        .transition(.opacity.combined(with: .scale(scale: 0.96)))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("DinkVisionPermissionPrimer")
    }

    private var shouldShowPermissionPrimer: Bool {
        if case .requestingAccess = model.status {
            return true
        }
        if case .permissionDenied = model.recordFlowPhase {
            return true
        }
        return false
    }

    private var canRecord: Bool {
        switch model.status {
        case .ready, .recording, .finished, .blocked:
            return true
        case .idle, .requestingAccess:
            return false
        }
    }

    private var statusText: String {
        switch model.status {
        case .idle:
            return "Starting"
        case .requestingAccess:
            return "Camera access"
        case .ready:
            return "Tripod steady"
        case .recording:
            return "Recording"
        case .finished:
            return "Saved"
        case .blocked(let message):
            return message
        }
    }

    private var preRecordStatusText: String {
        switch model.setupPassStatus {
        case .aligning, .aligned, .unavailable:
            return model.setupPassStatusText
        case .idle:
            return statusText
        }
    }

    private var preRecordChipStatus: DinkVisionPolicyChipStatus {
        switch model.setupPassStatus {
        case .aligning, .aligned, .unavailable:
            return model.setupPassChipStatus
        case .idle:
            return model.status == .ready ? .pass : .warning
        }
    }
}

private struct DinkVisionCourtPreviewBackdrop: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(hex: 0x37684A), DinkVisionColor.courtGreen, Color(hex: 0x26503A)],
                startPoint: .top,
                endPoint: .bottom
            )
            CourtPerspectiveLines()
                .stroke(DinkVisionColor.cream.opacity(0.85), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                .padding(.horizontal, 25)
                .padding(.top, 250)
                .padding(.bottom, 250)
        }
    }
}

private struct CourtPerspectiveLines: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.width * 0.24, y: rect.height * 0.08))
        path.addLine(to: CGPoint(x: rect.width * 0.76, y: rect.height * 0.08))
        path.addLine(to: CGPoint(x: rect.width * 0.94, y: rect.height * 0.88))
        path.addLine(to: CGPoint(x: rect.width * 0.06, y: rect.height * 0.88))
        path.closeSubpath()
        path.move(to: CGPoint(x: rect.width * 0.15, y: rect.height * 0.48))
        path.addLine(to: CGPoint(x: rect.width * 0.85, y: rect.height * 0.48))
        path.move(to: CGPoint(x: rect.width * 0.50, y: rect.height * 0.48))
        path.addLine(to: CGPoint(x: rect.width * 0.50, y: rect.height * 0.88))
        path.move(to: CGPoint(x: rect.width * 0.20, y: rect.height * 0.27))
        path.addLine(to: CGPoint(x: rect.width * 0.80, y: rect.height * 0.27))
        return path
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
            DinkVisionColor.trailYellow,
            DinkVisionColor.trailBlue,
            DinkVisionColor.trailRed,
            DinkVisionColor.success,
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
                        DinkVisionColor.ballYellow.opacity(trailOpacity(segment.points)),
                        style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round)
                    )
                }

                ForEach(rendered.trailPoints, id: \.point.frameIndex) { renderedPoint in
                    Circle()
                        .fill(DinkVisionColor.ballYellow.opacity(renderedPoint.point.opacity))
                        .frame(
                            width: max(5, CGFloat(renderedPoint.radius * 2)),
                            height: max(5, CGFloat(renderedPoint.radius * 2))
                        )
                        .position(x: CGFloat(renderedPoint.centerX), y: CGFloat(renderedPoint.centerY))
                        .shadow(color: DinkVisionColor.ballYellow.opacity(0.45), radius: 5)
                }

                ForEach(rendered.contactMarkers, id: \.marker.frameIndex) { renderedMarker in
                    Circle()
                        .stroke(DinkVisionColor.trailRed.opacity(renderedMarker.marker.opacity), lineWidth: 3)
                        .frame(
                            width: max(24, CGFloat(renderedMarker.radius * 2)),
                            height: max(24, CGFloat(renderedMarker.radius * 2))
                        )
                        .position(x: CGFloat(renderedMarker.centerX), y: CGFloat(renderedMarker.centerY))
                        .shadow(color: DinkVisionColor.trailRed.opacity(renderedMarker.marker.opacity * 0.42), radius: 8)
                }
            }
        }
        .accessibilityHidden(true)
    }

    private func trailOpacity(_ points: [RenderedLiveBallTrailPoint]) -> Double {
        points.map(\.point.opacity).max() ?? 0
    }
}

private struct DinkVisionCoachScreen: View {
    private let model = DinkVisionCoachPlaceholderModel.brandV4

    var body: some View {
        ZStack(alignment: .bottom) {
            DinkVisionColor.cream.ignoresSafeArea()
            VStack(spacing: 18) {
                Spacer(minLength: 44)
                ZStack(alignment: .topTrailing) {
                    DotGrid(rows: 4, columns: 5, dotSize: 9, color: DinkVisionColor.ink.opacity(0.18))
                        .frame(width: 150, height: 120)
                        .offset(x: 54, y: -34)
                    DinkVisionOwnerLockup(height: 148)
                        .padding(.top, 12)
                }
                Text(model.title)
                    .font(.system(size: 25, weight: .heavy, design: .rounded))
                    .foregroundStyle(DinkVisionColor.ink)
                    .multilineTextAlignment(.center)
                    .minimumScaleFactor(0.78)
                    .lineLimit(2)
                Text("Coming soon · roadmap \(model.roadmapID)")
                    .font(.system(size: 12, weight: .black, design: .rounded))
                    .foregroundStyle(DinkVisionColor.mutedText)
                    .textCase(.uppercase)
                Spacer(minLength: DinkVisionChromeLayout.scrollBottomPadding)
            }
            .padding(.horizontal, 24)

            HandArrow(color: DinkVisionColor.trailRed, lineWidth: 6)
                .frame(width: 118, height: 82)
                .rotationEffect(.degrees(112))
                .offset(x: -8, y: -DinkVisionChromeLayout.tabOverlayHeight - 8)
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("DinkVisionScreen-Coach")
    }
}

private struct DinkVisionReplaysScreen: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var rows: [DinkVisionReplayRow] = []
    @State private var isLoading = true
    @State private var errorText: String?
    @State private var selectedRow: DinkVisionReplayRow?
    @State private var openingRow: DinkVisionReplayRow?
    @State private var isVideoPickerPresented = false
    @StateObject private var importCoordinator: CameraRollImportCoordinator
    private let dataSource: DinkVisionReplayListDataSource
    private let configuration: DinkVisionRuntimeConfiguration
    @ObservedObject private var uploadCoordinator: DinkVisionUploadCoordinator

    init(
        dataSource: DinkVisionReplayListDataSource = DinkVisionReplayListDataSource(),
        configuration: DinkVisionRuntimeConfiguration = .current(),
        uploadCoordinator: DinkVisionUploadCoordinator
    ) {
        self.dataSource = dataSource
        self.configuration = configuration
        self.uploadCoordinator = uploadCoordinator
        _importCoordinator = StateObject(wrappedValue: CameraRollImportCoordinator(packageRootURL: dataSource.packageRootURL))
    }

    var body: some View {
        NavigationStack {
            ZStack {
                DinkVisionColor.cream.ignoresSafeArea()
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        DinkVisionScreenHeader(title: "Replays", subtitle: "Your rallies, rebuilt in 3D")

                        Button {
                            isVideoPickerPresented = true
                        } label: {
                            Label(importCoordinator.isImporting ? "Importing…" : "Import video", systemImage: "photo.on.rectangle")
                                .font(.system(size: 13, weight: .heavy, design: .rounded))
                                .foregroundStyle(DinkVisionColor.ink)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 10)
                                .background(DinkVisionColor.ballYellow, in: Capsule())
                        }
                        .disabled(importCoordinator.isImporting)
                        .accessibilityIdentifier("DinkVisionImportVideoButton")

                        if let importError = importCoordinator.errorMessage {
                            Text(importError)
                                .font(.footnote.weight(.semibold))
                                .foregroundStyle(.red)
                        }

                        if isLoading {
                            BallTrailLoadingView(title: "Scanning\ncaptures", detail: "sidecars")
                        } else if rows.isEmpty {
                            DinkVisionEmptyReplaysView(message: errorText ?? "record your first rally")
                        } else {
                            ForEach(rows) { row in
                                DinkVisionReplayRowView(
                                    row: row,
                                    uploadState: uploadCoordinator.state(for: row.id),
                                    onOpen: { openReplay(row) },
                                    onUpload: { uploadCoordinator.requestUpload(item: row.item) },
                                    onRetry: { uploadCoordinator.requestRetry(item: row.item) }
                                )
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 58)
                    .padding(.bottom, DinkVisionChromeLayout.scrollBottomPadding)
                }

                if let openingRow {
                    DinkVisionReplaySwooshOverlay(
                        durationText: openingRow.durationText,
                        reduceMotion: reduceMotion
                    )
                    .transition(.opacity)
                }
            }
            .accessibilityIdentifier("DinkVisionScreen-Replays")
            .navigationBarHidden(true)
            .task {
                loadRows()
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(15))
                    await uploadCoordinator.refreshUploadedStatuses()
                }
            }
            .sheet(isPresented: $isVideoPickerPresented) {
                CameraRollVideoPicker(
                    onPicked: { url in
                        isVideoPickerPresented = false
                        Task {
                            _ = await importCoordinator.importVideo(at: url)
                            try? FileManager.default.removeItem(at: url)
                            loadRows()
                        }
                    },
                    onError: { message in
                        isVideoPickerPresented = false
                        errorText = "Import failed: \(message)"
                    }
                )
            }
            .animation(.spring(response: 0.28, dampingFraction: 0.86), value: openingRow?.id)
            .fullScreenCover(item: $selectedRow) { row in
                DinkVisionReplayPlaybackScreen(row: row, configuration: configuration) {
                    selectedRow = nil
                }
            }
        }
    }

    private func loadRows() {
        isLoading = true
        do {
            rows = try dataSource.loadRows()
            uploadCoordinator.register(rows.map(\.item))
            errorText = nil
        } catch {
            rows = []
            errorText = "Could not read local captures"
        }
        isLoading = false
    }

    private func openReplay(_ row: DinkVisionReplayRow) {
        let delay = DinkVisionReplayOpenTransition.durationNanoseconds(reducedMotion: reduceMotion)
        guard delay > 0 else {
            selectedRow = row
            return
        }

        openingRow = row
        Task {
            try? await Task.sleep(nanoseconds: delay)
            await MainActor.run {
                if openingRow?.id == row.id {
                    selectedRow = row
                    openingRow = nil
                }
            }
        }
    }
}

private struct DinkVisionReplaySwooshOverlay: View {
    var durationText: String
    var reduceMotion: Bool
    @State private var progress: CGFloat = 0

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                if reduceMotion {
                    DinkVisionColor.ink.opacity(Double(progress) * 0.82)
                } else {
                    diagonalWipe(size: proxy.size)
                    trailComposition(size: proxy.size)
                }
                VStack {
                    Spacer()
                    Text(durationText)
                        .font(.system(size: 12, weight: .black, design: .rounded))
                        .foregroundStyle(DinkVisionColor.cream.opacity(0.74))
                        .padding(.bottom, DinkVisionChromeLayout.scrollBottomPadding)
                }
            }
            .ignoresSafeArea()
            .onAppear {
                let seconds = Double(DinkVisionReplayOpenTransition.durationNanoseconds(reducedMotion: reduceMotion)) / 1_000_000_000
                withAnimation(reduceMotion ? .easeOut(duration: seconds) : .easeOut(duration: seconds)) {
                    progress = 1
                }
            }
        }
        .accessibilityLabel("Opening replay")
        .accessibilityIdentifier("DinkVisionReplaySwooshOverlay")
    }

    private func diagonalWipe(size: CGSize) -> some View {
        let width = max(size.width, size.height) * 1.7
        let offsetX = -size.width * 0.82 + progress * size.width * 1.70
        let offsetY = size.height * 0.42 - progress * size.height * 0.88
        return DinkVisionDiagonalWipe(width: width, offsetX: offsetX, offsetY: offsetY)
    }

    private func trailComposition(size: CGSize) -> some View {
        let x = -size.width * 0.18 + progress * size.width * 0.94
        let y = size.height * 0.88 - progress * size.height * 0.62
        return ZStack {
            swooshLine(color: .white, width: 184, y: -42)
            swooshLine(color: DinkVisionColor.trailYellow, width: 132, y: -18)
            swooshLine(color: DinkVisionColor.trailBlue, width: 156, y: 10)
            swooshLine(color: DinkVisionColor.trailRed, width: 96, y: 35)
            PerforatedBallView(fill: .white, hole: DinkVisionColor.ink)
                .frame(width: 66, height: 66)
                .rotationEffect(.degrees(progress * 210))
                .offset(x: 94, y: -5)
        }
        .rotationEffect(.degrees(-17))
        .position(x: x, y: y)
    }

    private func swooshLine(color: Color, width: CGFloat, y: CGFloat) -> some View {
        Capsule()
            .fill(color)
            .frame(width: width * (0.34 + progress * 0.66), height: 6)
            .offset(x: -40 - progress * 20, y: y)
    }
}

private struct DinkVisionDiagonalWipe: View {
    var width: CGFloat
    var offsetX: CGFloat
    var offsetY: CGFloat

    var body: some View {
        Rectangle()
            .fill(DinkVisionColor.ink)
            .frame(width: width, height: width * 0.72)
            .rotationEffect(.degrees(-34))
            .offset(x: offsetX, y: offsetY)
            .shadow(color: DinkVisionColor.ink.opacity(0.5), radius: 24)
    }
}

private struct DinkVisionReplayRowView: View {
    let row: DinkVisionReplayRow
    let uploadState: CaptureUploadState?
    let onOpen: () -> Void
    let onUpload: () -> Void
    let onRetry: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button(action: onOpen) {
                rowContent
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("DinkVisionReplayRow-\(row.id)")

            HStack(spacing: 8) {
                Text(uploadState?.dinkVisionStateTitle ?? "Local")
                    .font(.system(size: 10, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(stateChipColor, in: Capsule())

                if let clipID = uploadState?.clipId {
                    Text("clip_id: \(clipID)")
                        .font(.system(size: 9, weight: .semibold, design: .monospaced))
                        .foregroundStyle(DinkVisionColor.mutedText)
                        .lineLimit(1)
                        .textSelection(.enabled)
                }

                Spacer()

                if uploadState?.state == .failed {
                    Button("Retry", action: onRetry)
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("DinkVisionRetryUpload-\(row.id)")
                } else if uploadState == nil {
                    Button("Upload", action: onUpload)
                        .buttonStyle(.borderedProminent)
                        .tint(DinkVisionColor.courtGreen)
                        .accessibilityIdentifier("DinkVisionUpload-\(row.id)")
                }
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(.white)
        )
        .accessibilityElement(children: .contain)
    }

    private var rowContent: some View {
        HStack(spacing: 14) {
            ZStack(alignment: .bottomTrailing) {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [Color(hex: 0x37684A), DinkVisionColor.courtGreenDeep],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 96, height: 64)
                TrailArcThumbnail()
                    .stroke(DinkVisionColor.trailBlue, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .frame(width: 86, height: 54)
                TrailArcThumbnail()
                    .stroke(DinkVisionColor.trailRed, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .rotationEffect(.degrees(10))
                    .frame(width: 74, height: 46)
                Circle()
                    .fill(DinkVisionColor.ballYellow)
                    .frame(width: 8, height: 8)
                    .offset(x: -62, y: -40)
                Text(row.durationText)
                    .font(.system(size: 10, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(6)
            }

            VStack(alignment: .leading, spacing: 5) {
                Text(row.title)
                    .font(.system(size: 17, weight: .heavy, design: .rounded))
                    .foregroundStyle(DinkVisionColor.ink)
                Text(row.subtitle)
                    .font(.system(size: 11, weight: .heavy, design: .rounded))
                    .foregroundStyle(DinkVisionColor.mutedText)
                    .textCase(.uppercase)
                    .lineLimit(2)
                HStack(spacing: 6) {
                    replayTrustChip(row.trusted3DText, fill: DinkVisionColor.courtGreen)
                    replayTrustChip(row.ballTrustText, fill: DinkVisionColor.ink.opacity(0.72))
                }
            }

            Spacer()
            Image(systemName: "chevron.right")
                .font(.system(size: 18, weight: .heavy))
                .foregroundStyle(DinkVisionColor.ink)
        }
        .accessibilityLabel("\(row.title), \(row.subtitle)")
    }

    private var stateChipColor: Color {
        switch uploadState?.state {
        case .queued: return DinkVisionColor.trailBlue
        case .uploading: return DinkVisionColor.ink
        case .uploaded: return DinkVisionColor.courtGreen
        case .failed: return DinkVisionColor.trailRed
        case nil: return DinkVisionColor.mutedText
        }
    }

    private func replayTrustChip(_ title: String, fill: Color) -> some View {
        Text(title)
            .font(.system(size: 10, weight: .heavy, design: .rounded))
            .foregroundStyle(.white)
            .lineLimit(1)
            .minimumScaleFactor(0.72)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(fill, in: Capsule())
    }
}

private struct TrailArcThumbnail: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX + 8, y: rect.maxY - 14))
        path.addQuadCurve(to: CGPoint(x: rect.maxX - 12, y: rect.midY + 4), control: CGPoint(x: rect.midX - 8, y: rect.minY + 4))
        return path
    }
}

private struct DinkVisionEmptyReplaysView: View {
    let message: String

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.system(size: 22, weight: .heavy, design: .rounded))
                    .foregroundStyle(DinkVisionColor.ink)
                Text(detail)
                    .font(.system(size: 12, weight: .heavy, design: .rounded))
                    .foregroundStyle(DinkVisionColor.mutedText)
                    .textCase(.uppercase)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            DinkVisionOwnerMark(height: 86)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(22)
        .background(.white, in: RoundedRectangle(cornerRadius: DinkVisionMetric.cardRadius, style: .continuous))
        .overlay(alignment: .topTrailing) {
            SketchSlashes(color: DinkVisionColor.ink, lineWidth: 5)
                .frame(width: 42, height: 42)
                .offset(x: -18, y: -14)
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("DinkVisionReplaysEmptyState")
    }

    private var title: String {
        message.localizedCaseInsensitiveContains("could") ? "Could not read captures" : "No rallies yet"
    }

    private var detail: String {
        message.localizedCaseInsensitiveContains("could") ? "Check local capture storage." : "Hit record - your first 3D replay lands here"
    }
}

private struct DinkVisionReplayPlaybackScreen: View {
    let row: DinkVisionReplayRow
    let configuration: DinkVisionRuntimeConfiguration
    let onClose: () -> Void
    @State private var loadedBundle: WorldBundle?
    @State private var loadError: String?

    var body: some View {
        ZStack(alignment: .top) {
            DinkVisionColor.ink.ignoresSafeArea()
            if let loadedBundle {
                if configuration.forceWorldCoachMark {
                    WorldViewerView(
                        viewModel: WorldViewerViewModel(
                            bundle: loadedBundle,
                            coachMarkStore: DinkVisionForcedCoachMarkStore()
                        ),
                        onClose: onClose
                    )
                    .ignoresSafeArea()
                } else {
                    WorldViewerView(bundle: loadedBundle, onClose: onClose)
                        .ignoresSafeArea()
                }
            } else if let loadError {
                VStack(spacing: 16) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                    Text(loadError)
                        .multilineTextAlignment(.center)
                    Button("Close", action: onClose)
                        .buttonStyle(.borderedProminent)
                }
                .foregroundStyle(DinkVisionColor.cream)
                .padding(24)
            } else {
                BallTrailLoadingView(title: "Loading\nreplay", detail: row.durationText)
                    .padding(22)
            }

            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(row.title)
                        .font(.system(size: 17, weight: .heavy, design: .rounded))
                    Text("Existing replay module. Bundled sample until this capture has server output.")
                        .font(.caption.weight(.semibold))
                        .lineLimit(2)
                }
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.headline.weight(.heavy))
                        .frame(width: 44, height: 44)
                }
                .buttonStyle(.plain)
            }
            .foregroundStyle(DinkVisionColor.cream)
            .padding(.horizontal, 18)
            .padding(.top, 54)
            .background(
                LinearGradient(colors: [Color.black.opacity(0.76), .clear], startPoint: .top, endPoint: .bottom)
                    .frame(height: 130),
                alignment: .top
            )
        }
        .accessibilityIdentifier("DinkVisionScreen-ReplayPlayback")
        .task {
            do {
                loadedBundle = try WorldBundle.loadBundledSample()
            } catch {
                loadError = "Could not load the replay viewer fixture."
            }
        }
    }
}

private struct DinkVisionStatsScreen: View {
    var body: some View {
        ZStack {
            DinkVisionColor.cream.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    DinkVisionScreenHeader(title: "Match Overview", subtitle: "Sample data - unlocks after your first match")
                    sampleWatermark
                    DinkVisionCourtMapCard()
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                        StatTile(title: "Shots tracked", value: "324", accent: DinkVisionColor.trailBlue, style: .spark)
                        StatTile(title: "Dink %", value: "41%", accent: DinkVisionColor.courtGreen, style: .ring)
                        StatTile(title: "Winners", value: "18", accent: DinkVisionColor.trailRed, style: .spark)
                        StatTile(title: "Avg speed", value: "31mph", accent: DinkVisionColor.trailYellow, style: .spark)
                    }
                    DinkVisionPlaceholderAnalysisCards()
                }
                .padding(.horizontal, 16)
                .padding(.top, 58)
                .padding(.bottom, DinkVisionChromeLayout.scrollBottomPadding)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("DinkVisionScreen-Stats")
    }

    private var sampleWatermark: some View {
        HStack {
            Spacer()
            ZStack(alignment: .topTrailing) {
                DotGrid(rows: 3, columns: 4, dotSize: 8, color: DinkVisionColor.ink.opacity(0.72))
                    .frame(width: 72, height: 54)
                    .offset(x: -72, y: 18)
                HandArrow()
                    .frame(width: 86, height: 60)
                    .rotationEffect(.degrees(4))
                    .offset(x: -42, y: 6)
                Text("Sample data")
                    .font(.system(size: 12, weight: .heavy, design: .rounded))
                    .foregroundStyle(DinkVisionColor.ink)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(DinkVisionColor.ballYellow, in: Capsule())
            }
            .frame(width: 180, height: 72, alignment: .topTrailing)
            .accessibilityLabel("Sample data, not real match stats")
        }
    }
}

private struct DinkVisionCourtMapCard: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: DinkVisionMetric.cardRadius, style: .continuous)
                .fill(DinkVisionColor.courtGreen)
            CourtMapLines()
                .stroke(DinkVisionColor.cream.opacity(0.9), lineWidth: 3)
                .padding(14)
            TrailArcThumbnail()
                .stroke(DinkVisionColor.trailBlue, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .padding(30)
            TrailArcThumbnail()
                .stroke(DinkVisionColor.trailRed, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .rotationEffect(.degrees(8))
                .padding(42)
            Circle()
                .fill(DinkVisionColor.ballYellow)
                .frame(width: 11, height: 11)
                .offset(x: 86, y: -16)
        }
        .frame(height: 164)
    }
}

private struct CourtMapLines: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.addRoundedRect(in: rect, cornerSize: CGSize(width: 10, height: 10))
        path.move(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
        path.move(to: CGPoint(x: rect.minX + rect.width * 0.33, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.minX + rect.width * 0.33, y: rect.maxY))
        path.move(to: CGPoint(x: rect.minX + rect.width * 0.66, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.minX + rect.width * 0.66, y: rect.maxY))
        return path
    }
}

private enum StatTileStyle {
    case spark
    case ring
}

private struct StatTile: View {
    let title: String
    let value: String
    let accent: Color
    let style: StatTileStyle

    var body: some View {
        DinkVisionCard {
            VStack(alignment: .leading, spacing: 7) {
                Text(title)
                    .font(.system(size: 12, weight: .heavy, design: .rounded))
                    .foregroundStyle(DinkVisionColor.mutedText)
                    .textCase(.uppercase)
                    .lineLimit(2)
                Text(value)
                    .font(.system(size: value.count > 4 ? 34 : 44, weight: .black, design: .rounded))
                    .foregroundStyle(DinkVisionColor.ink)
                    .minimumScaleFactor(0.64)
                    .lineLimit(1)
                if style == .spark {
                    Sparkline(accent: accent)
                        .stroke(accent, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        .frame(height: 28)
                } else {
                    RingMeter(accent: accent)
                        .frame(width: 46, height: 46)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .overlay(alignment: .topTrailing) {
            Text("sample")
                .font(.system(size: 8, weight: .heavy, design: .rounded))
                .foregroundStyle(DinkVisionColor.mutedText.opacity(0.76))
                .padding(10)
        }
    }
}

private struct Sparkline: Shape {
    let accent: Color

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX + 4, y: rect.midY + 5))
        path.addQuadCurve(to: CGPoint(x: rect.midX, y: rect.midY - 4), control: CGPoint(x: rect.width * 0.25, y: rect.minY + 2))
        path.addQuadCurve(to: CGPoint(x: rect.maxX - 4, y: rect.midY - 8), control: CGPoint(x: rect.width * 0.75, y: rect.maxY))
        return path
    }
}

private struct RingMeter: View {
    let accent: Color

    var body: some View {
        ZStack {
            Circle()
                .stroke(DinkVisionColor.line, lineWidth: 7)
            Circle()
                .trim(from: 0, to: 0.41)
                .stroke(accent, style: StrokeStyle(lineWidth: 7, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
    }
}

private struct DinkVisionPlaceholderAnalysisCards: View {
    var body: some View {
        VStack(spacing: 12) {
            placeholder(title: "Heat map", detail: "Sample placeholder until server wiring lands")
            placeholder(title: "Shot placement", detail: "Sample placeholder until your replay is processed")
        }
    }

    private func placeholder(title: String, detail: String) -> some View {
        DinkVisionCard {
            HStack {
                VStack(alignment: .leading, spacing: 6) {
                    Text(title)
                        .font(.system(size: 17, weight: .heavy, design: .rounded))
                        .foregroundStyle(DinkVisionColor.ink)
                    Text(detail)
                        .font(.system(size: 12, weight: .bold, design: .rounded))
                        .foregroundStyle(DinkVisionColor.mutedText)
                }
                Spacer()
                PerforatedBallView(fill: DinkVisionColor.line, hole: .white)
                    .frame(width: 46, height: 46)
            }
        }
    }
}

private struct DinkVisionProfileScreen: View {
    var isSignedIn: Bool
    @Binding var autoUploadAfterRecording: Bool
    var onSignIn: () -> Void
    @State private var flow = ProfileCaptureFlowState.h0Checklist()
    @State private var playerHeightCM: Double = 180
    @State private var ballSKU = "outdoor_yellow"

    var body: some View {
        ZStack {
            DinkVisionColor.cream.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    DinkVisionScreenHeader(title: "Set up your court", subtitle: "5 quick steps - better tracking forever")
                    accountAndUploadSettings
                    profileChecklist
                    capturePolicyExplainer
                    appInfo
                }
                .padding(.horizontal, 16)
                .padding(.top, 58)
                .padding(.bottom, DinkVisionChromeLayout.scrollBottomPadding)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("DinkVisionScreen-Profile")
    }

    private var accountAndUploadSettings: some View {
        DinkVisionCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(isSignedIn ? "Signed in" : "Local mode")
                            .font(.system(size: 17, weight: .heavy, design: .rounded))
                            .foregroundStyle(DinkVisionColor.ink)
                        Text(isSignedIn ? "Uploads use your account." : "Record and local replays work without an account.")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(DinkVisionColor.mutedText)
                    }
                    Spacer()
                    if !isSignedIn {
                        Button("Sign in", action: onSignIn)
                            .buttonStyle(.borderedProminent)
                            .tint(DinkVisionColor.courtGreen)
                            .accessibilityIdentifier("DinkVisionProfileSignIn")
                    }
                }

                Toggle("Auto-upload after recording", isOn: $autoUploadAfterRecording)
                    .font(.system(size: 14, weight: .heavy, design: .rounded))
                    .tint(DinkVisionColor.courtGreen)
                    .accessibilityIdentifier("DinkVisionAutoUploadToggle")
                Text("Off by default. When enabled, a signed-out upload prompts for sign-in without blocking recording.")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(DinkVisionColor.mutedText)
            }
        }
    }

    private var profileChecklist: some View {
        VStack(spacing: 10) {
            ForEach(flow.steps.indices, id: \.self) { index in
                let step = flow.steps[index]
                Button {
                    DinkVisionHaptics.impact(.light)
                    complete(step.kind)
                } label: {
                    HStack(spacing: 14) {
                        Text(step.status == .complete ? "OK" : "\(index + 1)")
                            .font(.system(size: 15, weight: .heavy, design: .rounded))
                            .foregroundStyle(DinkVisionColor.ink)
                            .frame(width: 40, height: 40)
                            .background(stepColor(step: step), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        VStack(alignment: .leading, spacing: 3) {
                            Text(step.kind.dinkVisionTitle)
                                .font(.system(size: 17, weight: .heavy, design: .rounded))
                                .foregroundStyle(DinkVisionColor.ink)
                            Text(step.kind.dinkVisionDetail)
                                .font(.system(size: 11, weight: .heavy, design: .rounded))
                                .foregroundStyle(DinkVisionColor.mutedText)
                                .textCase(.uppercase)
                        }
                        Spacer()
                        if flow.currentStep?.kind == step.kind {
                            Text("Next")
                                .font(.system(size: 10, weight: .heavy, design: .rounded))
                                .foregroundStyle(DinkVisionColor.ink)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(DinkVisionColor.ballYellow, in: Capsule())
                        }
                    }
                    .padding(14)
                    .background {
                        RoundedRectangle(cornerRadius: DinkVisionMetric.cardRadius, style: .continuous)
                            .fill(.white)
                        if step.status == .complete {
                            HStack {
                                Spacer()
                                PerforationPanel(style: .yellowEmbossed, cornerRadius: 16)
                                    .frame(width: 88, height: 48)
                                    .opacity(0.22)
                                    .padding(.trailing, 10)
                            }
                        }
                    }
                    .overlay(
                        RoundedRectangle(cornerRadius: DinkVisionMetric.cardRadius, style: .continuous)
                            .stroke(borderColor(step: step), lineWidth: 3)
                    )
                    .opacity(step.status == .complete || flow.currentStep?.kind == step.kind ? 1 : 0.55)
                }
                .buttonStyle(.plain)
                .frame(minHeight: 68)
            }
        }
    }

    private var capturePolicyExplainer: some View {
        DinkVisionCard {
            VStack(alignment: .leading, spacing: 8) {
                Text("Capture policy")
                    .font(.system(size: 17, weight: .heavy, design: .rounded))
                    .foregroundStyle(DinkVisionColor.ink)
                Text("DinkVision asks the camera for EIS off, AE/AF/WB lock, landscape, ARKit pose, gravity, and court-plane sidecar samples. If a device readback misses, the sidecar records the violation.")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(DinkVisionColor.mutedText)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var appInfo: some View {
        DinkVisionCard {
            HStack {
                DinkVisionOwnerMark(height: 62)
                VStack(alignment: .leading, spacing: 2) {
                    Text(DinkVisionBrand.displayName)
                        .font(.system(size: 20, weight: .heavy, design: .rounded))
                        .foregroundStyle(DinkVisionColor.ink)
                    Text("App-side capture UI. Server stats are not wired in this lane.")
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(DinkVisionColor.mutedText)
                }
                Spacer()
            }
        }
    }

    private func complete(_ kind: ProfileCaptureStepKind) {
        switch kind {
        case .emptyCourtClip:
            flow.recordStep(kind, artifactRef: "captures/profile/empty_court_clip.mov", metadata: ["clip_type": "empty_court"])
        case .calibrationGridSweep:
            flow.recordStep(kind, artifactRef: "captures/profile/calibration_grid_sweep.json", metadata: ["pattern": "charuco_or_aprilgrid"])
        case .paddleOrbit:
            flow.recordStep(kind, artifactRef: "captures/profile/paddle_orbit.mov", metadata: ["orbit": "single_paddle"])
        case .playerHeightEntry:
            flow.recordStep(kind, metadata: ["height_cm": "\(Int(playerHeightCM.rounded()))"])
        case .ballPick:
            flow.recordStep(kind, metadata: ["sku": ballSKU])
        }
    }

    private func stepColor(step: ProfileCaptureStepRecord) -> Color {
        if step.status == .complete {
            return DinkVisionColor.ballYellow
        }
        return flow.currentStep?.kind == step.kind ? DinkVisionColor.ballYellow : DinkVisionColor.line
    }

    private func borderColor(step: ProfileCaptureStepRecord) -> Color {
        if step.status == .complete {
            return Color(hex: 0xD9A82F)
        }
        return flow.currentStep?.kind == step.kind ? DinkVisionColor.ballYellow : .clear
    }
}

private struct DinkVisionScreenHeader: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.system(size: 28, weight: .heavy, design: .rounded))
                .foregroundStyle(DinkVisionColor.ink)
                .minimumScaleFactor(0.78)
                .lineLimit(2)
            Text(subtitle)
                .font(.system(size: 12, weight: .heavy, design: .rounded))
                .foregroundStyle(DinkVisionColor.mutedText)
                .textCase(.uppercase)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct FlowLayout<Content: View>: View {
    var spacing: CGFloat
    let content: Content

    init(spacing: CGFloat, @ViewBuilder content: () -> Content) {
        self.spacing = spacing
        self.content = content()
    }

    var body: some View {
        HStack(spacing: spacing) {
            content
        }
        .frame(maxWidth: .infinity, alignment: .center)
    }
}

#Preview("DinkVision shell") {
    DinkVisionAppRootView()
}

#Preview("Splash") {
    DinkVisionSplashView {}
}

#Preview("Splash closed lid snapshot") {
    ZStack {
        DinkVisionColor.cream
        DinkVisionSplashMarkComposition(lidClosure: 1)
            .frame(width: 148, height: 266)
    }
}

#Preview("Record") {
    DinkVisionRecordScreen()
}

#Preview("Replays empty") {
    DinkVisionReplaysScreen(
        dataSource: DinkVisionReplayListDataSource(loadPackages: { _ in [] }),
        uploadCoordinator: DinkVisionRuntimeConfiguration.current().makeUploadCoordinator()
    )
}

#Preview("Stats") {
    DinkVisionStatsScreen()
}

#Preview("Profile") {
    DinkVisionProfileScreen(
        isSignedIn: false,
        autoUploadAfterRecording: .constant(false),
        onSignIn: {}
    )
}
