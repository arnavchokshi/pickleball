import CoreGraphics
import SwiftUI

/// Mirrors `TRUST_BADGE_COLORS` / `trustBadgeColor` in
/// `web/replay/src/viewerData.ts` -- same three hex colors, so the native
/// viewer reads consistently with the web scrubber.
public enum WorldTrustColors {
    public static func swiftUIColor(for badge: TrustBadge) -> Color {
        switch badge {
        case .verified: return Color(red: 0x6c / 255, green: 0xb2 / 255, blue: 0xff / 255)
        case .preview: return Color(red: 0xff / 255, green: 0xb4 / 255, blue: 0x54 / 255)
        case .lowConfidence: return Color(red: 0x8a / 255, green: 0x8f / 255, blue: 0x98 / 255)
        }
    }

    public static func cgColor(for badge: TrustBadge, opacity: Double = 1) -> CGColor {
        let color = rgb(for: badge)
        return CGColor(red: color.0, green: color.1, blue: color.2, alpha: opacity)
    }

    /// Opacity applied to a low-confidence entity when the viewer's
    /// "dim low-confidence" toggle is on -- mirrors the web viewer's
    /// `dotOpacity = badge === "low_confidence" ? 0.55 : 1`.
    public static func opacity(for badge: TrustBadge, dimLowConfidence: Bool) -> Double {
        guard dimLowConfidence, badge == .lowConfidence else { return 1 }
        return 0.35
    }

    private static func rgb(for badge: TrustBadge) -> (CGFloat, CGFloat, CGFloat) {
        switch badge {
        case .verified: return (0x6c / 255, 0xb2 / 255, 0xff / 255)
        case .preview: return (0xff / 255, 0xb4 / 255, 0x54 / 255)
        case .lowConfidence: return (0x8a / 255, 0x8f / 255, 0x98 / 255)
        }
    }
}
