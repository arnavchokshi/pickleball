import Foundation
import PickleballCore

public struct ProfileCaptureFlowState: Equatable, Sendable {
    public var steps: [ProfileCaptureStepRecord]

    public var currentStep: ProfileCaptureStepRecord? {
        steps.first { $0.status != .complete }
    }

    public var isComplete: Bool {
        currentStep == nil
    }

    public var payload: ProfileCapturePayload {
        ProfileCapturePayload(steps: steps)
    }

    public init(steps: [ProfileCaptureStepRecord]) {
        self.steps = steps
    }

    public static func h0Checklist() -> ProfileCaptureFlowState {
        ProfileCaptureFlowState(
            steps: ProfileCaptureStepKind.h0Order.map { kind in
                ProfileCaptureStepRecord(kind: kind, status: .pending)
            }
        )
    }

    public mutating func recordCurrentStep(
        artifactRef: String? = nil,
        metadata: [String: String] = [:]
    ) {
        guard let currentStep else {
            return
        }
        recordStep(currentStep.kind, artifactRef: artifactRef, metadata: metadata)
    }

    public mutating func recordStep(
        _ kind: ProfileCaptureStepKind,
        artifactRef: String? = nil,
        metadata: [String: String] = [:]
    ) {
        guard let index = steps.firstIndex(where: { $0.kind == kind }) else {
            return
        }
        steps[index] = ProfileCaptureStepRecord(
            kind: kind,
            status: .complete,
            artifactRef: artifactRef,
            metadata: metadata
        )
    }
}
