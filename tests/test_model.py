import torch

from src.model import WaferCNN


def test_wafer_cnn_forward_shape():
    model = WaferCNN(num_classes=9, in_size=64)  # default in_channels=2
    x = torch.randn(4, 2, 64, 64)
    out = model(x)
    assert out.shape == (4, 9)


def test_wafer_cnn_forward_shape_smaller_input():
    model = WaferCNN(num_classes=9, in_size=32)
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 9)
