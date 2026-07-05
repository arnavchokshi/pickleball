from __future__ import annotations

from threed.racketsport.court_proposal_optimizer import accept_refinement, refine_homography_with_lines


def test_accept_refinement_rejects_better_line_residual_when_pixel_support_worsens() -> None:
    before = {"line_rmse_px": 20.0, "pixel_support": 0.70, "p95_px": 200.0, "median_px": 100.0}
    after = {"line_rmse_px": 10.0, "pixel_support": 0.40, "p95_px": 220.0, "median_px": 90.0}

    accepted, reasons = accept_refinement(before, after)

    assert accepted is False
    assert "pixel_support_worsened" in reasons


def test_accept_refinement_accepts_when_line_pixel_and_tail_improve() -> None:
    before = {"line_rmse_px": 20.0, "pixel_support": 0.70, "p95_px": 200.0, "median_px": 100.0}
    after = {"line_rmse_px": 12.0, "pixel_support": 0.78, "p95_px": 170.0, "median_px": 90.0}

    accepted, reasons = accept_refinement(before, after)

    assert accepted is True
    assert reasons == []


def test_refine_homography_shell_rejects_unwired_optimizer() -> None:
    initial_h = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    result = refine_homography_with_lines(initial_h, semantic_lines={}, line_distance_map=None)

    assert result["accepted"] is False
    assert result["homography_image_from_court"] == initial_h
    assert "optimizer_not_wired" in result["reject_reasons"]
