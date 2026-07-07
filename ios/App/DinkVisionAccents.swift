import SwiftUI

struct StrokeDrawOn: ViewModifier {
    var parameters: DinkVisionStrokeDrawOnParameters
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var trimEnd: CGFloat = 1

    init(parameters: DinkVisionStrokeDrawOnParameters = .default) {
        self.parameters = parameters
    }

    func body(content: Content) -> some View {
        let resolved = reduceMotion ? .resolved(reducedMotion: true) : parameters
        content
            .mask(alignment: .leading) {
                Rectangle()
                    .scaleEffect(x: trimEnd, y: 1, anchor: .leading)
            }
            .onAppear {
                trimEnd = resolved.initialTrimEnd
                guard resolved.durationSeconds > 0 else {
                    trimEnd = resolved.finalTrimEnd
                    return
                }
                withAnimation(.easeOut(duration: resolved.durationSeconds)) {
                    trimEnd = resolved.finalTrimEnd
                }
            }
    }
}

extension View {
    func strokeDrawOn(parameters: DinkVisionStrokeDrawOnParameters = .default) -> some View {
        modifier(StrokeDrawOn(parameters: parameters))
    }
}

struct SketchSlashes: View {
    var color: Color = DinkVisionColor.ink
    var lineWidth: CGFloat = 5

    var body: some View {
        SketchSlashesShape()
            .stroke(color, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round, lineJoin: .round))
            .aspectRatio(1, contentMode: .fit)
            .strokeDrawOn()
            .accessibilityHidden(true)
    }

}

struct DotGrid: View {
    var rows: Int = 3
    var columns: Int = 4
    var dotSize: CGFloat = 9
    var color: Color = DinkVisionColor.ink

    var body: some View {
        GeometryReader { proxy in
            let safeRows = max(1, rows)
            let safeColumns = max(1, columns)
            let xStep = proxy.size.width / CGFloat(safeColumns)
            let yStep = proxy.size.height / CGFloat(safeRows)
            let diameter = min(dotSize, xStep * 0.62, yStep * 0.62)

            ZStack {
                ForEach(0..<safeRows, id: \.self) { row in
                    ForEach(0..<safeColumns, id: \.self) { column in
                        dot(row: row, column: column, xStep: xStep, yStep: yStep, diameter: diameter, width: proxy.size.width)
                    }
                }
            }
        }
        .accessibilityHidden(true)
    }

    private func dot(row: Int, column: Int, xStep: CGFloat, yStep: CGFloat, diameter: CGFloat, width: CGFloat) -> some View {
        let stagger: CGFloat = row.isMultiple(of: 2) ? 0 : xStep * 0.22
        let rawX: CGFloat = xStep * (CGFloat(column) + 0.5) + stagger
        let clampedX: CGFloat = min(width - diameter / 2, max(diameter / 2, rawX))
        let y: CGFloat = yStep * (CGFloat(row) + 0.5)
        return Circle()
            .fill(color)
            .frame(width: diameter, height: diameter)
            .position(x: clampedX, y: y)
    }
}

struct HandArrow: View {
    var color: Color = DinkVisionColor.trailRed
    var lineWidth: CGFloat = 6

    var body: some View {
        HandArrowShape()
            .stroke(color, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round, lineJoin: .round))
            .aspectRatio(10.0 / 7.0, contentMode: .fit)
            .strokeDrawOn()
            .accessibilityHidden(true)
    }
}

enum PerforationPanelStyle {
    case inkWhiteDots
    case yellowEmbossed
}

struct PerforationPanel: View {
    var style: PerforationPanelStyle
    var cornerRadius: CGFloat = 20

    var body: some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            .fill(background)
            .overlay {
                DotGrid(rows: 3, columns: 4, dotSize: style == .yellowEmbossed ? 20 : 18, color: dotColor)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 16)
                    .shadow(color: shadowColor, radius: style == .yellowEmbossed ? 1 : 0, y: style == .yellowEmbossed ? 2 : 0)
            }
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(innerStrokeColor, lineWidth: 2)
                    .blendMode(.multiply)
                    .padding(1)
            }
            .accessibilityHidden(true)
    }

    private var background: Color {
        switch style {
        case .inkWhiteDots:
            return DinkVisionColor.ink
        case .yellowEmbossed:
            return DinkVisionColor.ballYellow
        }
    }

    private var dotColor: Color {
        switch style {
        case .inkWhiteDots:
            return DinkVisionColor.cream
        case .yellowEmbossed:
            return Color(hex: 0xD9A82F)
        }
    }

    private var innerStrokeColor: Color {
        switch style {
        case .inkWhiteDots:
            return Color.white.opacity(0.08)
        case .yellowEmbossed:
            return Color.black.opacity(0.10)
        }
    }

    private var shadowColor: Color {
        switch style {
        case .inkWhiteDots:
            return .clear
        case .yellowEmbossed:
            return Color.black.opacity(0.26)
        }
    }
}

private struct SketchSlashesShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: point(16, 8, in: rect))
        path.addQuadCurve(to: point(10, 44, in: rect), control: point(13, 26, in: rect))
        path.move(to: point(34, 10, in: rect))
        path.addQuadCurve(to: point(27, 46, in: rect), control: point(30, 27, in: rect))
        return path
    }

    private func point(_ x: CGFloat, _ y: CGFloat, in rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + rect.width * x / 54, y: rect.minY + rect.height * y / 54)
    }
}

private struct HandArrowShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: point(8, 60, in: rect))
        path.addQuadCurve(to: point(88, 18, in: rect), control: point(40, 8, in: rect))
        path.move(to: point(74, 10, in: rect))
        path.addLine(to: point(90, 17, in: rect))
        path.addLine(to: point(80, 32, in: rect))
        return path
    }

    private func point(_ x: CGFloat, _ y: CGFloat, in rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + rect.width * x / 100, y: rect.minY + rect.height * y / 70)
    }
}
