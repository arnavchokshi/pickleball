from __future__ import annotations

import cv2

from threed.racketsport.court_detector_v2_net import detect_court_net_evidence


def test_img1605_net_evidence_is_anchor_not_floor_homography() -> None:
    frame = cv2.imread(
        "runs/owner_data/owner_IMG_1605_8a193402780b/prelabels/review_frames/"
        "owner_IMG_1605_8a193402780b/frame_000151.jpg"
    )
    assert frame is not None

    evidence = detect_court_net_evidence(frame)

    assert evidence["anchor_role"] == "roi_orientation_scale_only"
    assert evidence["uses_top_net_as_floor_point"] is False
    assert evidence["roi"]["x_min"] >= 0
    assert evidence["roi"]["x_max"] <= frame.shape[1]
    assert evidence["roi"]["y_min"] >= 0
    assert evidence["roi"]["y_max"] <= frame.shape[0]
    assert 0.0 <= evidence["confidence"] <= 1.0
