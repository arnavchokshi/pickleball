import Foundation
import PickleballCore

public enum CapturePolicyEnforcer {
    public static func requestedState(for policy: CapturePolicy) -> CapturePolicyRequestedState {
        CapturePolicyRequestedState(
            fps: policy.fps,
            resolution: policy.resolution.dimensions(for: policy.orientation),
            format: policy.codec,
            orientation: policy.orientation,
            electronicStabilizationEnabled: false,
            exposureLocked: true,
            focusLocked: true,
            whiteBalanceLocked: true
        )
    }

    public static func evaluate(policy: CapturePolicy, achieved: CapturePolicyAchievedState?) -> CapturePolicyEnforcementReport {
        let requested = requestedState(for: policy)
        guard let achieved else {
            return CapturePolicyEnforcementReport(
                requested: requested,
                achieved: nil,
                violations: ["policy_not_configured"]
            )
        }

        var violations: [String] = []
        if achieved.fps != nil && achieved.fps != requested.fps {
            violations.append("fps_mismatch")
        }
        if achieved.resolution != nil && achieved.resolution != requested.resolution {
            violations.append("resolution_mismatch")
        }
        if achieved.format != nil && achieved.format != requested.format {
            violations.append("format_mismatch")
        }
        if achieved.orientation != .landscape {
            violations.append("orientation_not_landscape")
        }
        if achieved.electronicStabilizationEnabled != false {
            violations.append("electronic_stabilization_enabled")
        }
        if achieved.exposureLocked != true {
            violations.append("exposure_not_locked")
        }
        if achieved.focusLocked != true {
            violations.append("focus_not_locked")
        }
        if achieved.whiteBalanceLocked != true {
            violations.append("white_balance_not_locked")
        }

        return CapturePolicyEnforcementReport(
            requested: requested,
            achieved: achieved,
            violations: violations
        )
    }
}
