import numpy as np
import torch

from src.gradcam import GradCAM, wafer_to_input
from src.model import WaferCNN
from src.synthetic import make_sample


def test_wafer_to_input_two_channels():
    m = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    x = wafer_to_input(m)
    assert tuple(x.shape) == (1, 2, 2, 3)
    assert torch.equal(x[0, 0], torch.tensor((m > 0).astype("float32")))
    assert torch.equal(x[0, 1], torch.tensor((m == 2).astype("float32")))


def test_gradcam_shape_and_range():
    torch.manual_seed(0)
    model = WaferCNN(num_classes=9, in_size=32)
    m = make_sample("Center", size=32, rng=np.random.default_rng(0))
    x = wafer_to_input(m)
    with GradCAM(model) as cam:
        heat, pred, probs = cam(x)
    assert heat.shape == (32, 32)
    assert heat.min() >= 0.0 and heat.max() <= 1.0 + 1e-6
    assert 0 <= pred < 9
    assert probs.shape == (9,)
    assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_gradcam_removes_hooks():
    model = WaferCNN(num_classes=9, in_size=32)
    before = sum(len(mod._forward_hooks) for mod in model.modules())
    x = wafer_to_input(make_sample("Donut", size=32, rng=np.random.default_rng(1)))
    with GradCAM(model) as cam:
        cam(x)
    after = sum(len(mod._forward_hooks) for mod in model.modules())
    assert after == before  # context manager cleaned up


def test_gradcam_class_specific():
    """The heatmap for a chosen class should differ from another class's."""
    torch.manual_seed(0)
    model = WaferCNN(num_classes=9, in_size=32)
    x = wafer_to_input(make_sample("Scratch", size=32, rng=np.random.default_rng(2)))
    with GradCAM(model) as cam:
        heat_a, _, _ = cam(x, class_idx=1)
    with GradCAM(model) as cam:
        heat_b, _, _ = cam(x, class_idx=5)
    assert not np.allclose(heat_a, heat_b)
