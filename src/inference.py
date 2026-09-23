"""
Small, UI-agnostic inference layer shared by the Streamlit app and the tests.

Keeping this separate from app.py means the prediction path can be unit-tested
without a running Streamlit server, and the app stays a thin presentation
layer over these functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import config
from .gradcam import GradCAM, wafer_to_input
from .model import WaferCNN


def load_model(checkpoint: str | Path | None = None, img_size: int = config.IMG_SIZE,
               device: str = "cpu") -> WaferCNN:
    """Load a WaferCNN. If `checkpoint` exists, its weights are restored;
    otherwise an untrained model is returned (callers should warn the user
    that predictions are then meaningless)."""
    model = WaferCNN(num_classes=len(config.FAILURE_CLASSES), in_size=img_size)
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model.to(device).eval()


def has_checkpoint(checkpoint: str | Path | None) -> bool:
    return checkpoint is not None and Path(checkpoint).exists()


def predict(model: WaferCNN, wafer_map: np.ndarray, device: str = "cpu"):
    """Return (probs over FAILURE_CLASSES, predicted_index) for one wafer map."""
    x = wafer_to_input(wafer_map).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs, int(probs.argmax())


def predict_with_cam(model: WaferCNN, wafer_map: np.ndarray, device: str = "cpu",
                     class_idx: int | None = None):
    """Return (probs, predicted_index, heatmap). The heatmap is HxW in [0,1]."""
    x = wafer_to_input(wafer_map).to(device)
    with GradCAM(model) as cam:
        heat, pred, probs = cam(x, class_idx=class_idx)
    return probs, pred, heat
