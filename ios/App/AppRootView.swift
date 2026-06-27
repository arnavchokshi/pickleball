import SwiftUI
import SwayCapture
import SwayCore

struct AppRootView: View {
    private let capture = CaptureSessionScaffold(mode: .standard60)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Sway Body")
                .font(.title)
            Text("Capture scaffold ready")
                .font(.subheadline)
            Text("\(capture.requestedFPS) fps · \(capture.orientation.rawValue)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}

#Preview {
    AppRootView()
}
