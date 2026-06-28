import XCTest
import PickleballCore
@testable import PickleballCapture

final class CapturePolicyTests: XCTestCase {
    func testSwingModeUsesLocked1080p120HEVCPolicy() {
        let policy = CapturePolicy.recommended(
            for: .swing120,
            deviceTier: .standard,
            capabilities: .allCodecs,
            orientation: .landscape
        )

        XCTAssertEqual(policy.mode, .swing120)
        XCTAssertEqual(policy.fps, 120)
        XCTAssertEqual(policy.resolution, .hd1080p)
        XCTAssertEqual(policy.codec, .hevc)
        XCTAssertEqual(policy.orientation, .landscape)
        XCTAssertFalse(policy.hdrEnabled)
        XCTAssertFalse(policy.cinematicEnabled)
        XCTAssertTrue(policy.liveFramesEnabled)
    }

    func testBallPhysicsModeUsesBinned720p240HEVCPolicy() {
        let policy = CapturePolicy.recommended(for: .ballPhysics240, deviceTier: .standard, capabilities: .allCodecs)

        XCTAssertEqual(policy.fps, 240)
        XCTAssertEqual(policy.resolution, .hd720p)
        XCTAssertEqual(policy.codec, .hevc)
    }

    func testTierAQualityModeUses4K60ProResWhenSupported() {
        let policy = CapturePolicy.recommended(for: .quality4K60, deviceTier: .lidar, capabilities: .allCodecs)

        XCTAssertEqual(policy.fps, 60)
        XCTAssertEqual(policy.resolution, .uhd4K)
        XCTAssertEqual(policy.codec, .prores422lt)
    }

    func testQualityModeFallsBackToHEVC1080p60OffTierAOrWithoutProRes() {
        let standardPolicy = CapturePolicy.recommended(for: .quality4K60, deviceTier: .standard, capabilities: .allCodecs)
        let unsupportedPolicy = CapturePolicy.recommended(for: .quality4K60, deviceTier: .lidar, capabilities: .hevcOnly)

        XCTAssertEqual(standardPolicy.fps, 60)
        XCTAssertEqual(standardPolicy.resolution, .hd1080p)
        XCTAssertEqual(standardPolicy.codec, .hevc)
        XCTAssertEqual(unsupportedPolicy.fps, 60)
        XCTAssertEqual(unsupportedPolicy.resolution, .hd1080p)
        XCTAssertEqual(unsupportedPolicy.codec, .hevc)
    }

    func testLockedSettingsClampShutterISOAndFocusToCaptureBounds() {
        let fastRequest = LockedCapturePolicy.request(
            requestedShutterSeconds: 1.0 / 2_000.0,
            requestedISO: 20,
            requestedFocusLensPosition: 1.4,
            isoBounds: ISOClampBounds(min: 34, max: 800)
        )

        XCTAssertEqual(fastRequest.exposureS, 1.0 / 1_000.0, accuracy: 0.000_000_1)
        XCTAssertEqual(fastRequest.iso, 34, accuracy: 0.000_000_1)
        XCTAssertEqual(fastRequest.focus, 1.0, accuracy: 0.000_000_1)
        XCTAssertTrue(fastRequest.wbLocked)

        let slowRequest = LockedCapturePolicy.request(
            requestedShutterSeconds: 1.0 / 250.0,
            requestedISO: 900,
            requestedFocusLensPosition: -0.2,
            isoBounds: ISOClampBounds(min: 34, max: 800)
        )

        XCTAssertEqual(slowRequest.exposureS, 1.0 / 500.0, accuracy: 0.000_000_1)
        XCTAssertEqual(slowRequest.iso, 800, accuracy: 0.000_000_1)
        XCTAssertEqual(slowRequest.focus, 0.0, accuracy: 0.000_000_1)
    }

    func testOrientationPolicySupportsPortraitAndLandscapeVideoTransforms() {
        XCTAssertEqual(CaptureOrientationPolicy.captureOrientation(for: .portrait), .portrait)
        XCTAssertEqual(CaptureOrientationPolicy.captureOrientation(for: .portraitUpsideDown), .portrait)
        XCTAssertEqual(CaptureOrientationPolicy.captureOrientation(for: .landscapeRight), .landscape)
        XCTAssertEqual(CaptureOrientationPolicy.captureOrientation(for: .landscapeLeft), .landscape)
        XCTAssertEqual(CaptureOrientationPolicy.rotationAngleDegrees(for: .portrait), 0)
        XCTAssertEqual(CaptureOrientationPolicy.rotationAngleDegrees(for: .portraitUpsideDown), 180)
        XCTAssertEqual(CaptureOrientationPolicy.rotationAngleDegrees(for: .landscapeRight), 90)
        XCTAssertEqual(CaptureOrientationPolicy.rotationAngleDegrees(for: .landscapeLeft), 270)
    }

    func testPortraitPolicyUsesDisplayResolutionForPhoneUprightCaptures() {
        let policy = CapturePolicy.recommended(
            for: .standard60,
            deviceTier: .standard,
            capabilities: .hevcOnly,
            orientation: .portrait
        )

        XCTAssertEqual(policy.orientation, .portrait)
        XCTAssertEqual(policy.resolution.dimensions(for: .portrait), [1080, 1920])
        XCTAssertEqual(policy.resolution.dimensions(for: .landscape), [1920, 1080])
    }
}
