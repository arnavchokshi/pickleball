import SwiftUI
@preconcurrency import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let videoRotationAngle: Int

    func makeUIView(context _: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.applyVideoRotationAngle(videoRotationAngle)
        return view
    }

    func updateUIView(_ uiView: PreviewView, context _: Context) {
        uiView.previewLayer.session = session
        uiView.applyVideoRotationAngle(videoRotationAngle)
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    func applyVideoRotationAngle(_ angle: Int) {
        guard let connection = previewLayer.connection else {
            return
        }

        let rotationAngle = CGFloat(angle)
        if connection.isVideoRotationAngleSupported(rotationAngle) {
            connection.videoRotationAngle = rotationAngle
        }
    }
}
