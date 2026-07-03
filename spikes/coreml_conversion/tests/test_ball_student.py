import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ball_student import WASBLiteBallStudent, count_parameters


def test_wasb_lite_student_shape_range_and_size():
    model = WASBLiteBallStudent().eval()
    x = torch.zeros(2, 9, 288, 512)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1, 72, 128)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0
    assert 1_000_000 <= count_parameters(model) <= 1_500_000
