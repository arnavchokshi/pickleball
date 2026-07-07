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

struct PaddleEyeMark: View {
    var size: CGFloat
    var isBlinking: Bool = false
    var irisScale: CGFloat = 1.0
    var foreground: Color = DinkVisionColor.ink
    var background: Color = DinkVisionColor.cream

    var body: some View {
        GeometryReader { proxy in
            let side = min(proxy.size.width, proxy.size.height)
            let line = max(3, side * 0.045)

            ZStack {
                PaddleBodyShape()
                    .stroke(foreground, style: StrokeStyle(lineWidth: line, lineCap: .round, lineJoin: .round))

                if isBlinking {
                    BlinkedEyeShape()
                        .stroke(foreground, style: StrokeStyle(lineWidth: line * 0.94, lineCap: .round, lineJoin: .round))
                    BlinkLashesShape()
                        .stroke(foreground, style: StrokeStyle(lineWidth: line * 0.62, lineCap: .round))
                } else {
                    EyeShape()
                        .fill(background)
                    EyeShape()
                        .stroke(foreground, style: StrokeStyle(lineWidth: line * 0.86, lineCap: .round, lineJoin: .round))
                    PerforatedBallView(fill: foreground, hole: background)
                        .scaleEffect(irisScale)
                        .frame(width: side * 0.28, height: side * 0.28)
                        .position(x: side * 0.50, y: side * 0.38)
                }
            }
            .frame(width: side, height: side)
            .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
        }
        .frame(width: size, height: size * 1.42)
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
                ballHole(x: 0.35, y: 0.33, side: side)
                ballHole(x: 0.65, y: 0.33, side: side)
                ballHole(x: 0.27, y: 0.56, side: side)
                ballHole(x: 0.50, y: 0.67, side: side)
                ballHole(x: 0.73, y: 0.56, side: side)
            }
            .frame(width: side, height: side)
            .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private func ballHole(x: CGFloat, y: CGFloat, side: CGFloat) -> some View {
        Circle()
            .fill(hole)
            .frame(width: side * 0.12, height: side * 0.12)
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

private struct PaddleBodyShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let body = CGRect(
            x: rect.width * 0.20,
            y: rect.height * 0.07,
            width: rect.width * 0.60,
            height: rect.height * 0.62
        )
        path.addRoundedRect(in: body, cornerSize: CGSize(width: body.width * 0.48, height: body.width * 0.48))
        path.move(to: CGPoint(x: rect.midX, y: rect.height * 0.69))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.height * 0.84))
        let handle = CGRect(x: rect.width * 0.41, y: rect.height * 0.84, width: rect.width * 0.18, height: rect.height * 0.12)
        path.addRoundedRect(in: handle, cornerSize: CGSize(width: handle.width * 0.28, height: handle.width * 0.28))
        return path
    }
}

private struct EyeShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.width * 0.26, y: rect.height * 0.38))
        path.addQuadCurve(to: CGPoint(x: rect.width * 0.74, y: rect.height * 0.38), control: CGPoint(x: rect.midX, y: rect.height * 0.22))
        path.addQuadCurve(to: CGPoint(x: rect.width * 0.26, y: rect.height * 0.38), control: CGPoint(x: rect.midX, y: rect.height * 0.54))
        path.closeSubpath()
        return path
    }
}

private struct BlinkedEyeShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.width * 0.28, y: rect.height * 0.39))
        path.addQuadCurve(to: CGPoint(x: rect.width * 0.72, y: rect.height * 0.39), control: CGPoint(x: rect.midX, y: rect.height * 0.47))
        return path
    }
}

private struct BlinkLashesShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let anchors: [(CGFloat, CGFloat, CGFloat)] = [
            (0.39, 0.44, -0.05),
            (0.50, 0.46, 0.00),
            (0.61, 0.44, 0.05),
        ]
        for anchor in anchors {
            let start = CGPoint(x: rect.width * anchor.0, y: rect.height * anchor.1)
            path.move(to: start)
            path.addLine(to: CGPoint(x: start.x + rect.width * anchor.2, y: start.y + rect.height * 0.08))
        }
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
