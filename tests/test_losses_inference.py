import numpy as np
import torch
import torch.nn.functional as F

from src.config import FAILURE_CLASSES
from src.inference import load_model, predict, predict_with_cam
from src.losses import FocalLoss
from src.synthetic import make_sample


def test_focal_reduces_to_ce_at_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(8, 9)
    target = torch.randint(0, 9, (8,))
    focal = FocalLoss(gamma=0.0)(logits, target)
    ce = F.cross_entropy(logits, target)
    assert torch.allclose(focal, ce, atol=1e-5)


def test_focal_is_scalar_and_backprops():
    logits = torch.randn(4, 9, requires_grad=True)
    target = torch.randint(0, 9, (4,))
    loss = FocalLoss(gamma=2.0)(logits, target)
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None


def test_load_model_without_checkpoint_returns_untrained():
    model = load_model(None, img_size=32)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_predict_returns_distribution():
    model = load_model(None, img_size=32)
    m = make_sample("Center", size=32, rng=np.random.default_rng(0))
    probs, idx = predict(model, m)
    assert probs.shape == (len(FAILURE_CLASSES),)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert 0 <= idx < len(FAILURE_CLASSES)


def test_predict_with_cam_shapes():
    model = load_model(None, img_size=32)
    m = make_sample("Edge-Ring", size=32, rng=np.random.default_rng(0))
    probs, idx, heat = predict_with_cam(model, m)
    assert heat.shape == (32, 32)
    assert 0 <= idx < len(FAILURE_CLASSES)
