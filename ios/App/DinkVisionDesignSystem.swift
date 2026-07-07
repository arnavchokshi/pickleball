import SwiftUI

enum DinkVisionBrand {
    static let displayName = "DinkVision"
}

enum DinkVisionColor {
    static let cream = Color(hex: 0xF4EEE3)
    static let courtGreen = Color(hex: 0x2E5B3F)
    static let courtGreenDeep = Color(hex: 0x234731)
    static let ink = Color(hex: 0x141414)
    static let ballYellow = Color(hex: 0xF2C63F)
    static let trailBlue = Color(hex: 0x3E8EF0)
    static let trailRed = Color(hex: 0xE8503A)
    static let trailYellow = Color(hex: 0xF2C63F)
    static let cardWhite = Color.white
    static let line = Color(hex: 0xE7DFD1)
    static let mutedText = Color(hex: 0x8D8577)
    static let success = Color(hex: 0x43D17C)
}

enum DinkVisionMetric {
    static let cardRadius: CGFloat = 24
    static let tabBarHeight: CGFloat = 88
    static let tabBarRadius: CGFloat = 32
}

extension Color {
    init(hex: Int) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0
        )
    }
}

struct DinkVisionCard<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(18)
            .background(
                RoundedRectangle(cornerRadius: DinkVisionMetric.cardRadius, style: .continuous)
                    .fill(DinkVisionColor.cardWhite)
            )
    }
}

struct DinkVisionOwnerMark: View {
    var height: CGFloat

    var body: some View {
        Image("DinkVisionMark")
            .renderingMode(.original)
            .resizable()
            .interpolation(.high)
            .scaledToFit()
            .frame(width: height * 322 / 579, height: height)
            .accessibilityLabel("\(DinkVisionBrand.displayName) mark")
    }
}

struct DinkVisionOwnerLockup: View {
    var height: CGFloat

    var body: some View {
        Image("DinkVisionLockup")
            .renderingMode(.original)
            .resizable()
            .interpolation(.high)
            .scaledToFit()
            .frame(width: height * 662 / 764, height: height)
            .accessibilityLabel(DinkVisionBrand.displayName)
    }
}

struct PaddleEyeMark: View {
    var size: CGFloat
    var foreground: Color = DinkVisionColor.ink
    var background: Color = DinkVisionColor.cream

    var body: some View {
        GeometryReader { proxy in
            let bounds = CGRect(origin: .zero, size: proxy.size)
            let scale = PaddleEyeMarkGeometry.scale(in: bounds)

            ZStack {
                PaddleHeadShape()
                    .stroke(foreground, style: StrokeStyle(lineWidth: 22 * scale, lineCap: .round, lineJoin: .round))
                PaddleNeckShape()
                    .stroke(foreground, style: StrokeStyle(lineWidth: 14 * scale, lineCap: .round))
                PaddleGripShape()
                    .stroke(foreground, style: StrokeStyle(lineWidth: 15 * scale, lineCap: .round, lineJoin: .round))
                PaddleGripStripeShape()
                    .stroke(foreground, style: StrokeStyle(lineWidth: 8 * scale, lineCap: .round))

                PaddleButtCircleShape()
                    .fill(background)
                PaddleButtCircleShape()
                    .stroke(foreground, style: StrokeStyle(lineWidth: 13 * scale, lineCap: .round, lineJoin: .round))

                PaddleEyeIrisView(fill: foreground, hole: background)
                PaddleEyeLidShape()
                    .stroke(foreground, style: StrokeStyle(lineWidth: 20 * scale, lineCap: .round, lineJoin: .round))
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
        }
        .frame(width: size, height: size * PaddleEyeMarkGeometry.aspectRatio)
        .accessibilityLabel("\(DinkVisionBrand.displayName) mark")
    }
}

struct PerforatedBallView: View {
    var fill: Color = DinkVisionColor.ink
    var hole: Color = DinkVisionColor.cream

    var body: some View {
        GeometryReader { proxy in
            let side = min(proxy.size.width, proxy.size.height)
            ZStack {
                Circle().fill(fill)
                ballHole(x: 0.50, y: 0.50, side: side)
                ballHole(x: 0.50, y: 0.22, side: side)
                ballHole(x: 0.75, y: 0.36, side: side)
                ballHole(x: 0.75, y: 0.64, side: side)
                ballHole(x: 0.50, y: 0.78, side: side)
                ballHole(x: 0.25, y: 0.64, side: side)
                ballHole(x: 0.25, y: 0.36, side: side)
            }
            .frame(width: side, height: side)
            .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private func ballHole(x: CGFloat, y: CGFloat, side: CGFloat) -> some View {
        Circle()
            .fill(hole)
            .frame(width: side * 0.174, height: side * 0.174)
            .position(x: side * x, y: side * y)
    }
}

struct BallTrailLoadingView: View {
    var title: String
    var detail: String
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let colors = [
        DinkVisionColor.trailBlue,
        DinkVisionColor.trailYellow,
        DinkVisionColor.trailRed,
        Color.white,
    ]
    private let widths: [CGFloat] = [0.38, 0.22, 0.52, 0.30]

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { timeline in
            let phase = reduceMotion ? 0 : timeline.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.2) / 1.2
            HStack(spacing: 14) {
                PerforatedBallView(fill: .white, hole: DinkVisionColor.ink)
                    .rotationEffect(.degrees(phase * 360))
                    .frame(width: 46, height: 46)

                VStack(spacing: 7) {
                    ForEach(colors.indices, id: \.self) { index in
                        HStack(spacing: 8) {
                            Spacer(minLength: 0)
                            Capsule()
                                .fill(colors[index])
                                .frame(width: 160 * widths[index] + CGFloat(index) * 6 + CGFloat(phase) * 10, height: 5)
                            Capsule()
                                .fill(colors[index].opacity(0.52))
                                .frame(width: 18 + CGFloat(index) * 4, height: 5)
                        }
                    }
                }
                .frame(maxWidth: .infinity)

                VStack(alignment: .trailing, spacing: 2) {
                    Text(title)
                        .font(.system(size: 13, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.trailing)
                    Text(detail)
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.58))
                        .multilineTextAlignment(.trailing)
                }
                .frame(width: 74, alignment: .trailing)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 18)
            .background(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(DinkVisionColor.ink)
            )
        }
        .accessibilityLabel("\(title), \(detail)")
    }
}

private nonisolated enum PaddleEyeMarkGeometry {
    static let designWidth: CGFloat = 360
    static let designHeight: CGFloat = 420
    static let aspectRatio = designHeight / designWidth

    static func scale(in rect: CGRect) -> CGFloat {
        min(rect.width / designWidth, rect.height / designHeight)
    }

    static func frame(in rect: CGRect) -> CGRect {
        let scale = scale(in: rect)
        let size = CGSize(width: designWidth * scale, height: designHeight * scale)
        return CGRect(
            x: rect.midX - size.width / 2,
            y: rect.midY - size.height / 2,
            width: size.width,
            height: size.height
        )
    }

    static func point(_ x: CGFloat, _ y: CGFloat, in rect: CGRect) -> CGPoint {
        let frame = frame(in: rect)
        let scale = scale(in: rect)
        return CGPoint(x: frame.minX + x * scale, y: frame.minY + y * scale)
    }

    static func rect(x: CGFloat, y: CGFloat, width: CGFloat, height: CGFloat, in rect: CGRect) -> CGRect {
        let frame = frame(in: rect)
        let scale = scale(in: rect)
        return CGRect(x: frame.minX + x * scale, y: frame.minY + y * scale, width: width * scale, height: height * scale)
    }
}

private struct PaddleHeadShape: Shape {
    func path(in rect: CGRect) -> Path {
        let scale = PaddleEyeMarkGeometry.scale(in: rect)
        let head = PaddleEyeMarkGeometry.rect(x: 60, y: 16, width: 240, height: 270, in: rect)
        var path = Path()
        path.addRoundedRect(in: head, cornerSize: CGSize(width: 104 * scale, height: 104 * scale))
        return path
    }
}

private struct PaddleNeckShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: PaddleEyeMarkGeometry.point(171, 286, in: rect))
        path.addLine(to: PaddleEyeMarkGeometry.point(171, 310, in: rect))
        path.move(to: PaddleEyeMarkGeometry.point(189, 286, in: rect))
        path.addLine(to: PaddleEyeMarkGeometry.point(189, 310, in: rect))
        return path
    }
}

private struct PaddleGripShape: Shape {
    func path(in rect: CGRect) -> Path {
        let scale = PaddleEyeMarkGeometry.scale(in: rect)
        let grip = PaddleEyeMarkGeometry.rect(x: 141, y: 310, width: 78, height: 72, in: rect)
        var path = Path()
        path.addRoundedRect(in: grip, cornerSize: CGSize(width: 22 * scale, height: 22 * scale))
        return path
    }
}

private struct PaddleGripStripeShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: PaddleEyeMarkGeometry.point(152, 332, in: rect))
        path.addLine(to: PaddleEyeMarkGeometry.point(208, 323, in: rect))
        path.move(to: PaddleEyeMarkGeometry.point(152, 351, in: rect))
        path.addLine(to: PaddleEyeMarkGeometry.point(208, 342, in: rect))
        path.move(to: PaddleEyeMarkGeometry.point(152, 368, in: rect))
        path.addLine(to: PaddleEyeMarkGeometry.point(208, 360, in: rect))
        return path
    }
}

private struct PaddleButtCircleShape: Shape {
    func path(in rect: CGRect) -> Path {
        let center = PaddleEyeMarkGeometry.point(180, 403, in: rect)
        let radius = 18 * PaddleEyeMarkGeometry.scale(in: rect)
        var path = Path()
        path.addEllipse(in: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
        return path
    }
}

private struct PaddleEyeIrisView: View {
    var fill: Color
    var hole: Color

    var body: some View {
        GeometryReader { proxy in
            let bounds = CGRect(origin: .zero, size: proxy.size)
            let scale = PaddleEyeMarkGeometry.scale(in: bounds)
            let center = PaddleEyeMarkGeometry.point(180, 151, in: bounds)
            ZStack {
                PerforatedBallView(fill: fill, hole: hole)
                    .frame(width: 132 * scale, height: 132 * scale)
                    .position(x: center.x, y: center.y)
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .clipShape(PaddleEyeApertureShape())
        }
    }
}

private struct PaddleEyeApertureShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: PaddleEyeMarkGeometry.point(86, 151, in: rect))
        path.addQuadCurve(
            to: PaddleEyeMarkGeometry.point(274, 151, in: rect),
            control: PaddleEyeMarkGeometry.point(180, 76, in: rect)
        )
        path.addQuadCurve(
            to: PaddleEyeMarkGeometry.point(86, 151, in: rect),
            control: PaddleEyeMarkGeometry.point(180, 226, in: rect)
        )
        path.closeSubpath()
        return path
    }
}

private struct PaddleEyeLidShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: PaddleEyeMarkGeometry.point(86, 151, in: rect))
        path.addQuadCurve(
            to: PaddleEyeMarkGeometry.point(274, 151, in: rect),
            control: PaddleEyeMarkGeometry.point(180, 76, in: rect)
        )
        path.move(to: PaddleEyeMarkGeometry.point(86, 151, in: rect))
        path.addQuadCurve(
            to: PaddleEyeMarkGeometry.point(274, 151, in: rect),
            control: PaddleEyeMarkGeometry.point(180, 226, in: rect)
        )
        return path
    }
}

#Preview("Paddle eye mark") {
    PaddleEyeMark(size: 160)
        .padding(40)
        .background(DinkVisionColor.cream)
}

#Preview("Ball trail loader") {
    BallTrailLoadingView(title: "Building\n3D world...", detail: "2 min left")
        .padding(24)
        .background(DinkVisionColor.cream)
}
