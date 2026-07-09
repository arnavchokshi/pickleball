import Foundation
import PhotosUI
import PickleballCapture
import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class CameraRollImportCoordinator: ObservableObject {
    typealias ImportAction = (URL, URL) async throws -> CaptureLibraryItem

    @Published private(set) var isImporting = false
    @Published private(set) var errorMessage: String?

    private let packageRootURL: URL
    private let importAction: ImportAction

    init(
        packageRootURL: URL,
        importAction: @escaping ImportAction = { sourceURL, rootURL in
            let result = try await CameraRollVideoImporter().importVideo(sourceURL: sourceURL, packageRootURL: rootURL)
            guard let item = try CaptureLibrary.listPackages(packageRootURL: rootURL)
                .first(where: { $0.sessionID == result.descriptor.sessionID }) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            return item
        }
    ) {
        self.packageRootURL = packageRootURL
        self.importAction = importAction
    }

    @discardableResult
    func importVideo(at sourceURL: URL) async -> CaptureLibraryItem? {
        isImporting = true
        errorMessage = nil
        defer { isImporting = false }
        do {
            return try await importAction(sourceURL, packageRootURL)
        } catch {
            errorMessage = "Import failed: \(String(describing: error))"
            return nil
        }
    }
}

struct CameraRollVideoPicker: UIViewControllerRepresentable {
    var onPicked: (URL) -> Void
    var onError: (String) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var configuration = PHPickerConfiguration(photoLibrary: .shared())
        configuration.filter = .videos
        configuration.selectionLimit = 1
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        private let parent: CameraRollVideoPicker

        init(parent: CameraRollVideoPicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            guard let provider = results.first?.itemProvider else { return }
            provider.loadFileRepresentation(forTypeIdentifier: UTType.movie.identifier) { [parent] url, error in
                if let error {
                    Task { @MainActor in parent.onError(String(describing: error)) }
                    return
                }
                guard let url else {
                    Task { @MainActor in parent.onError("The selected video had no readable file.") }
                    return
                }
                do {
                    let staged = FileManager.default.temporaryDirectory
                        .appendingPathComponent("dinkvision-import-\(UUID().uuidString).\(url.pathExtension)")
                    try FileManager.default.copyItem(at: url, to: staged)
                    Task { @MainActor in parent.onPicked(staged) }
                } catch {
                    Task { @MainActor in parent.onError(String(describing: error)) }
                }
            }
        }
    }
}
