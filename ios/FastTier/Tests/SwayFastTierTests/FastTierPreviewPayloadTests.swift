import XCTest
@testable import SwayFastTier

final class FastTierPreviewPayloadTests: XCTestCase {
    func testPreviewPayloadEncodesPoseBallAndRacketTrackMetadata() throws {
        let payload = FastTierPreviewPayload(
            clipID: "clip-001",
            tracks: [
                PreviewTrack(
                    id: "pose-main",
                    kind: .pose2D,
                    source: .appleVisionBody2D,
                    relativePath: "preview/pose2d.json",
                    fps: 60,
                    subjectLimit: 4
                ),
                PreviewTrack(
                    id: "ball",
                    kind: .ball,
                    source: .yoloCoreMLBall,
                    relativePath: "preview/ball.json",
                    fps: 60,
                    subjectLimit: 1
                ),
                PreviewTrack(
                    id: "racket",
                    kind: .racket,
                    source: .yoloCoreMLRacket,
                    relativePath: "preview/racket.json",
                    fps: 30,
                    subjectLimit: 4
                ),
            ],
            metricLabels: [
                PreviewMetricLabel(metricID: "court_spacing", title: "Court spacing", valueText: "balanced")
            ]
        )

        XCTAssertEqual(payload.status, .previewAvailable)
        XCTAssertEqual(payload.tracks.map(\.previewOnly), [true, true, true])
        XCTAssertEqual(payload.metricLabels.first?.previewOnly, true)
        XCTAssertEqual(payload.metricLabels.first?.displayTitle, "Preview only: Court spacing")
        XCTAssertEqual(payload.failureReasons, [])

        let data = try JSONEncoder().encode(payload)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(object["schema_version"] as? Int, 1)
        XCTAssertEqual(object["clip_id"] as? String, "clip-001")
        XCTAssertEqual(object["status"] as? String, "preview_available")

        let tracks = try XCTUnwrap(object["tracks"] as? [[String: Any]])
        XCTAssertEqual(tracks[0]["preview_only"] as? Bool, true)
        XCTAssertEqual(tracks[0]["subject_limit"] as? Int, 4)
    }

    func testPreviewPayloadFailsClosedWithoutPoseTrack() {
        let payload = FastTierPreviewPayload(
            clipID: "clip-002",
            tracks: [
                PreviewTrack(
                    id: "ball",
                    kind: .ball,
                    source: .yoloCoreMLBall,
                    relativePath: "preview/ball.json",
                    fps: 60,
                    subjectLimit: 1
                )
            ],
            metricLabels: []
        )

        XCTAssertEqual(payload.status, .failClosed)
        XCTAssertEqual(payload.failureReasons, ["missing_preview_pose_track"])
    }
}
