import Foundation
import PickleballCapture
import PickleballUpload

@MainActor
final class DinkVisionUploadCoordinator: ObservableObject {
    @Published private(set) var states: [String: CaptureUploadState] = [:]
    @Published var isSignInPresented = false
    @Published var autoUploadAfterRecording: Bool {
        didSet { preferences.set(autoUploadAfterRecording, forKey: Self.autoUploadPreferenceKey) }
    }

    static let autoUploadPreferenceKey = "dinkvision.autoUploadAfterRecording"

    private let queue: UploadQueue
    private let packageRootURL: URL
    private let hasAccessToken: () -> Bool
    private let preferences: UserDefaults
    private var packages: [String: CaptureUploadPackage] = [:]
    private var authenticationGate = UploadAuthenticationGate()
    private var monitorTasks: [String: Task<Void, Never>] = [:]

    init(
        queue: UploadQueue,
        packageRootURL: URL,
        hasAccessToken: @escaping () -> Bool,
        preferences: UserDefaults = .standard
    ) {
        self.queue = queue
        self.packageRootURL = packageRootURL
        self.hasAccessToken = hasAccessToken
        self.preferences = preferences
        self.autoUploadAfterRecording = preferences.bool(forKey: Self.autoUploadPreferenceKey)
    }

    deinit {
        for task in monitorTasks.values { task.cancel() }
    }

    func register(_ items: [CaptureLibraryItem]) {
        for item in items {
            let package = makePackage(for: item)
            packages[item.sessionID] = package
            if let state = try? UploadQueue.readState(for: package) {
                states[item.sessionID] = state
            }
        }
    }

    func resumeInterruptedUploads(items: [CaptureLibraryItem]) async {
        register(items)
        let registered = items.compactMap { packages[$0.sessionID] }
        guard (try? await queue.resume(registered)) != nil else { return }
        for package in registered {
            if let state = try? UploadQueue.readState(for: package), state.state == .queued {
                states[package.packageID] = state
                startMonitoring(package)
            }
        }
        await queue.processPending()
        refreshLocalStates()
    }

    func requestUpload(item: CaptureLibraryItem) {
        register([item])
        switch authenticationGate.requestUpload(packageID: item.sessionID, hasAccessToken: hasAccessToken()) {
        case .enqueue(let packageID):
            enqueue(packageID: packageID, retry: false)
        case .promptSignIn:
            isSignInPresented = true
        }
    }

    func requestRetry(item: CaptureLibraryItem) {
        register([item])
        switch authenticationGate.requestUpload(packageID: item.sessionID, hasAccessToken: hasAccessToken()) {
        case .enqueue(let packageID):
            enqueue(packageID: packageID, retry: true)
        case .promptSignIn:
            isSignInPresented = true
        }
    }

    func recordingFinished(_ recording: CameraRecordingResult) {
        let itemPackage = CaptureUploadPackage(
            packageID: recording.descriptor.sessionID,
            packageDirectoryURL: packageRootURL.appendingPathComponent(recording.descriptor.directoryRelativePath, isDirectory: true),
            videoURL: packageRootURL.appendingPathComponent(recording.descriptor.clipRelativePath),
            sidecarURL: packageRootURL.appendingPathComponent(recording.descriptor.sidecarRelativePath)
        )
        packages[itemPackage.packageID] = itemPackage
        if autoUploadAfterRecording {
            switch authenticationGate.requestUpload(packageID: itemPackage.packageID, hasAccessToken: hasAccessToken()) {
            case .enqueue(let packageID): enqueue(packageID: packageID, retry: false)
            case .promptSignIn: isSignInPresented = true
            }
        }
    }

    func uploadFinishedRecording(_ recording: CameraRecordingResult) {
        recordingFinished(recording)
        guard !autoUploadAfterRecording else { return }
        switch authenticationGate.requestUpload(packageID: recording.descriptor.sessionID, hasAccessToken: hasAccessToken()) {
        case .enqueue(let packageID): enqueue(packageID: packageID, retry: false)
        case .promptSignIn: isSignInPresented = true
        }
    }

    func signedIn() {
        isSignInPresented = false
        if let pendingPackageID = authenticationGate.completeSignIn() {
            let retry = states[pendingPackageID]?.state == .failed
            enqueue(packageID: pendingPackageID, retry: retry)
        }
    }

    func refreshUploadedStatuses() async {
        for (packageID, package) in packages where states[packageID]?.state == .uploaded {
            if let state = await queue.refreshServerStatus(for: package) {
                states[packageID] = state
            }
        }
    }

    func state(for packageID: String) -> CaptureUploadState? {
        states[packageID]
    }

    private func enqueue(packageID: String, retry: Bool) {
        guard let package = packages[packageID] else { return }
        Task {
            do {
                let state = try await (retry ? queue.retry(package) : queue.enqueue(package))
                states[packageID] = state
                startMonitoring(package)
                await queue.processPending()
            } catch {
                states[packageID] = CaptureUploadState(
                    state: .failed,
                    captureId: packageID,
                    totalBytes: states[packageID]?.totalBytes ?? 0,
                    lastError: String(describing: error)
                )
            }
            if let state = try? UploadQueue.readState(for: package) {
                states[packageID] = state
                if state.state == .uploaded || state.state == .failed {
                    monitorTasks[packageID]?.cancel()
                    monitorTasks.removeValue(forKey: packageID)
                }
            }
        }
    }

    private func startMonitoring(_ package: CaptureUploadPackage) {
        monitorTasks[package.packageID]?.cancel()
        monitorTasks[package.packageID] = Task { [weak self] in
            while !Task.isCancelled {
                if let state = try? UploadQueue.readState(for: package) {
                    self?.states[package.packageID] = state
                    if state.state == .uploaded || state.state == .failed { return }
                }
                try? await Task.sleep(for: .milliseconds(150))
            }
        }
    }

    private func refreshLocalStates() {
        for (packageID, package) in packages {
            if let state = try? UploadQueue.readState(for: package) {
                states[packageID] = state
            }
        }
    }

    private func makePackage(for item: CaptureLibraryItem) -> CaptureUploadPackage {
        let videoURL = packageRootURL.appendingPathComponent(item.clipRelativePath)
        return CaptureUploadPackage(
            packageID: item.sessionID,
            packageDirectoryURL: videoURL.deletingLastPathComponent(),
            videoURL: videoURL,
            sidecarURL: packageRootURL.appendingPathComponent(item.sidecarRelativePath)
        )
    }
}

extension CaptureUploadState {
    var dinkVisionStateTitle: String {
        switch state {
        case .queued:
            return "Queued"
        case .uploading:
            return "Uploading \(Int((fractionCompleted * 100).rounded()))%"
        case .uploaded:
            if serverStatus == RenderGatewayJobStatus.partial.rawValue {
                let count = missingCapabilities.count
                return count == 1 ? "Partial — 1 capability missing" : "Partial — \(count) capabilities missing"
            }
            if serverStatus == RenderGatewayJobStatus.complete.rawValue, manifestUrl != nil {
                return "Replay ready"
            }
            if serverStatus == "uploaded", jobId == nil {
                return "Uploaded — processing not started"
            }
            return serverStatus.map { "Server: \($0)" } ?? "Uploaded"
        case .failed:
            return "Failed — Retry"
        }
    }
}
