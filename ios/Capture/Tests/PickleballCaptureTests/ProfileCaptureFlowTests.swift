import XCTest
@testable import PickleballCapture
import PickleballCore

final class ProfileCaptureFlowTests: XCTestCase {
    func testH0ProfileFlowAdvancesThroughSetupStepsAndRecordsArtifacts() {
        var flow = ProfileCaptureFlowState.h0Checklist()

        XCTAssertEqual(flow.currentStep?.kind, .emptyCourtClip)
        XCTAssertFalse(flow.isComplete)

        flow.recordCurrentStep(
            artifactRef: "captures/profile-empty-court/clip.mov",
            metadata: ["duration_s": "8.0"]
        )

        XCTAssertEqual(flow.steps[0].status, .complete)
        XCTAssertEqual(flow.steps[0].artifactRef, "captures/profile-empty-court/clip.mov")
        XCTAssertEqual(flow.currentStep?.kind, .calibrationGridSweep)

        flow.recordStep(.calibrationGridSweep, artifactRef: "profiles/court/grid.json", metadata: ["pattern": "charuco"])
        flow.recordStep(.paddleOrbit, artifactRef: "profiles/gear/paddle_orbit.mov", metadata: ["hand": "right"])
        flow.recordStep(.playerHeightEntry, artifactRef: nil, metadata: ["height_cm": "180"])
        flow.recordStep(.ballPick, artifactRef: nil, metadata: ["sku": "outdoor_yellow"])

        XCTAssertTrue(flow.isComplete)
        XCTAssertEqual(flow.payload.steps.map(\.kind), ProfileCaptureStepKind.h0Order)
        XCTAssertEqual(flow.payload.steps[3].metadata["height_cm"], "180")
    }
}
