import SwiftUI
import PickleballCapture

struct AppRootView: View {
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

                if isLandscape {
                    landscapeOverlay
                } else {
                    portraitOverlay
                }
            }
            .task {
                await model.prepare()
                model.updateOrientation(isLandscapeViewport: isLandscape)
            }
            .onChange(of: isLandscape) {
                model.updateOrientation(isLandscapeViewport: isLandscape)
            }
        }
    }

    private var portraitOverlay: some View {
        VStack(spacing: 12) {
            header
            Spacer(minLength: 20)
            capturePanel
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var landscapeOverlay: some View {
        HStack(alignment: .top, spacing: 14) {
            Spacer()
            VStack(spacing: 12) {
                header
                capturePanel
            }
            .frame(width: 360)
        }
        .padding(14)
    }

    private var header: some View {
        HStack(spacing: 10) {
            statusPill
            Spacer(minLength: 8)
            Label(model.captureOrientationTitle, systemImage: orientationIconName)
                .font(.caption.weight(.semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.regularMaterial, in: Capsule())
        }
    }

    private var capturePanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            modePicker
            metricsGrid
            pathRow
            replayBenchmarkRow
            HStack {
                Spacer()
                recordButton
                Spacer()
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private var modePicker: some View {
        Picker("Mode", selection: $model.selectedMode) {
            ForEach(CaptureViewModel.modes, id: \.self) { mode in
                Text(mode.title).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .controlSize(.small)
        .onChange(of: model.selectedMode) {
            guard !model.isRecording else {
                return
            }
            model.configure()
        }
    }

    private var metricsGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 10) {
            GridRow {
                MetricTile(title: "Format", value: model.modeSummary, systemImage: "video")
                    .gridCellColumns(2)
            }
            GridRow {
                MetricTile(title: "Rotation", value: model.videoRotationTitle, systemImage: "rotate.right")
                MetricTile(title: "State", value: statusText, systemImage: statusIconName)
            }
        }
    }

    private var pathRow: some View {
        HStack(spacing: 8) {
            Image(systemName: "folder")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(model.descriptor?.directoryRelativePath ?? "captures/new")
                .font(.caption2)
                .lineLimit(1)
                .foregroundStyle(.secondary)
            Spacer(minLength: 0)
        }
        .accessibilityLabel("Capture package path")
    }

    @ViewBuilder
    private var replayBenchmarkRow: some View {
        HStack(spacing: 10) {
            MetricTile(title: "Benchmark", value: model.replayBenchmarkDetail, systemImage: "person.3.sequence")
            Button {
                Task {
                    await model.runReplayBenchmarkFromStagedManifest()
                }
            } label: {
                Label(model.replayBenchmarkTitle, systemImage: model.isReplayBenchmarkRunning ? "hourglass" : "play.rectangle")
                    .labelStyle(.iconOnly)
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.bordered)
            .disabled(model.isReplayBenchmarkRunning)
            .accessibilityLabel("Run replay benchmark")
        }
        if let outputPath = model.replayBenchmarkOutputPath {
            HStack(spacing: 8) {
                Image(systemName: "doc.text")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(outputPath)
                    .font(.caption2)
                    .lineLimit(1)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            .accessibilityLabel("Replay benchmark output path")
        }
    }

    private var recordButton: some View {
        Button {
            model.toggleRecording()
        } label: {
            Image(systemName: model.isRecording ? "stop.fill" : "record.circle")
                .font(.system(size: 32, weight: .semibold))
                .frame(width: 68, height: 68)
        }
        .buttonStyle(.borderedProminent)
        .buttonBorderShape(.circle)
        .tint(model.isRecording ? .red : .blue)
        .disabled(!canRecord)
        .accessibilityLabel(model.isRecording ? "Stop recording" : "Start recording")
    }

    private var statusPill: some View {
        Label(statusText, systemImage: statusIconName)
            .font(.caption.weight(.semibold))
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

#Preview {
    AppRootView()
}

private struct MetricTile: View {
    var title: String
    var value: String
    var systemImage: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(value)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
    }
}
