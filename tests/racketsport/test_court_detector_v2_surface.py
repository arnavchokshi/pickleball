from __future__ import annotations

import numpy as np

from threed.racketsport.court_detector_v2_surface import build_surface_paint_evidence


def test_surface_paint_evidence_learns_local_line_color() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = [40, 120, 70]
    image[120:124, 40:280] = [210, 210, 60]

    evidence = build_surface_paint_evidence(image)

    assert evidence["mask_support_ratio"] > 0.001
    assert evidence["surface_color_bgr"][1] >= 80
    assert evidence["line_color_mode"] in {"white", "local_contrast"}
    assert len(evidence["line_color_bgr"]) == 3


def test_surface_paint_evidence_exposes_line_candidates() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = [40, 120, 70]
    image[80:84, 40:280] = [235, 235, 235]
    image[150:154, 40:280] = [235, 235, 235]
    image[50:210, 60:64] = [235, 235, 235]
    image[50:210, 260:264] = [235, 235, 235]

    evidence = build_surface_paint_evidence(image)

    assert len(evidence["semantic_line_candidates"]) >= 4
    assert {"p1", "p2", "angle_deg", "length_px"} <= set(evidence["semantic_line_candidates"][0])
