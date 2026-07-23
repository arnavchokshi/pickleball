from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from threed.racketsport.court_keypoint_geometric_loss import NET_KEYPOINT_NAMES  # noqa: E402
from threed.racketsport.court_keypoint_net import make_court_keypoint_heatmap_model  # noqa: E402
from threed.racketsport.court_structured_model import (  # noqa: E402
    STRUCTURED_FLOOR_KEYPOINT_COUNT,
    STRUCTURED_FLOOR_KEYPOINT_NAMES,
    covariance_matrices_from_params,
    initialize_structured_v3_from_v2,
    make_court_structured_v3_model,
)


def test_structured_v3_is_floor_only_and_emits_confidence_evidence() -> None:
    model = make_court_structured_v3_model()
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((2, 3, 96, 160), dtype=torch.float32))

    assert STRUCTURED_FLOOR_KEYPOINT_COUNT == 30
    assert not NET_KEYPOINT_NAMES.intersection(STRUCTURED_FLOOR_KEYPOINT_NAMES)
    assert output["keypoint_heatmaps"].shape == (2, 30, 24, 40)
    assert output["keypoint_vis_logits"].shape == (2, 30)
    assert output["keypoint_covariance"].shape == (2, 30, 2, 2)
    assert output["line_family_logits"].shape == (2, 5, 24, 40)
    assert output["line_distance_maps"].shape == (2, 5, 24, 40)
    assert output["supported_view_logit"].shape == (2,)
    assert torch.linalg.eigvalsh(output["keypoint_covariance"]).min().item() > 0.0


def test_covariance_parameterization_is_positive_and_bounded_correlation() -> None:
    params = torch.tensor([[[0.0, 0.0, 20.0], [-20.0, 20.0, -20.0]]])
    covariance = covariance_matrices_from_params(params)
    assert covariance.shape == (1, 2, 2, 2)
    assert torch.linalg.eigvalsh(covariance).min().item() > 0.0


def test_v2_warm_start_copies_shared_weights_and_twelve_floor_queries() -> None:
    v2 = make_court_keypoint_heatmap_model(15, architecture="court_unet_v2")
    v3 = make_court_structured_v3_model()
    report = initialize_structured_v3_from_v2(v3, v2.state_dict())

    assert len(report["initialized_canonical_queries"]) == 12
    assert "net_center" not in report["initialized_canonical_queries"]
    assert "stem.0.weight" in report["loaded_shared_keys"]
    assert "keypoint_head.weight" in report["skipped_v2_keys"]
