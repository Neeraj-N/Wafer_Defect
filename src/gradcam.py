"""
Grad-CAM explainability for the wafer-defect CNN.

Grad-CAM (Selvaraju et al., 2017) highlights the spatial regions a
convolutional network relied on for a given class. For wafer metrology this
answers the question an inspection engineer actually asks -- *where on the
wafer* is the model looking -- and shows that a "Scratch" or "Edge-Ring"
prediction is driven by the failed dies in that pattern rather than by
background noise.

The implementation has no third-party dependency: it registers a forward hook
to capture the last conv block's feature maps and a full-backward hook to
capture their gradients w.r.t. the target class score, then forms the usual
ReLU(sum_k alpha_k A_k) map and upsamples it to the input resolution.

CLI (needs a trained checkpoint and the real dataset):

    python -m src.gradcam --checkpoint checkpoints/best_cnn.pt \
        --classes Scratch Donut Edge-Loc --out docs/images

Each requested class yields <class>_raw.png and <class>_cam.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import config
from .model import WaferCNN


def _last_conv(model: nn.Module) -> nn.Conv2d:
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("model has no Conv2d layer to attach Grad-CAM to")
    return convs[-1]


def wafer_to_input(wafer_map: np.ndarray) -> torch.Tensor:
    """Turn a single {0,1,2} wafer map into the model's 2-channel input
    (die-exists mask, die-failed mask), matching src.dataset.WaferMapDataset.
    Returns a (1, 2, H, W) float tensor."""
    m = np.asarray(wafer_map)
    valid = (m > 0).astype(np.float32)
    fail = (m == 2).astype(np.float32)
    x = np.stack([valid, fail])[None]  # (1, 2, H, W)
    return torch.from_numpy(x)


class GradCAM:
    """Grad-CAM for a CNN whose last Conv2d produces the feature map of
    interest. Use as a context manager so the hooks are always removed:

        with GradCAM(model) as cam:
            heat, pred, probs = cam(x)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model.eval()
        self.target = target_layer or _last_conv(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fh = self.target.register_forward_hook(self._save_activation)
        self._bh = self.target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out):
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1, 2, H, W). Returns (heatmap HxW in [0,1], class_idx, probs)."""
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError(f"expected a single (1,C,H,W) input, got {tuple(x.shape)}")
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        act = self._activations[0]          # (K, h, w)
        grad = self._gradients[0]           # (K, h, w)
        weights = grad.mean(dim=(1, 2))     # (K,) global-average-pooled grads
        cam = torch.relu((weights[:, None, None] * act).sum(0))  # (h, w)

        cam = cam - cam.min()
        peak = cam.max()
        if peak > 0:
            cam = cam / peak
        cam = F.interpolate(
            cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        return cam.cpu().numpy(), class_idx, probs

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def save_panel(wafer_map, heatmap, title_left="Wafer map", title_right="Grad-CAM", path=None):
    """Render a raw-map | heatmap-overlay panel. Returns the matplotlib figure.
    Imported lazily so the core Grad-CAM math has no matplotlib dependency."""
    import matplotlib.pyplot as plt

    from .visualize import WAFER_CMAP

    m = np.asarray(wafer_map)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    axes[0].imshow(m, cmap=WAFER_CMAP, vmin=0, vmax=2)
    axes[0].set_title(title_left, fontsize=10)
    axes[1].imshow((m > 0), cmap="gray", vmin=0, vmax=1, alpha=0.35)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.65)
    axes[1].set_title(title_right, fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def save_raw(wafer_map, path):
    import matplotlib.pyplot as plt

    from .visualize import WAFER_CMAP

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.asarray(wafer_map), cmap=WAFER_CMAP, vmin=0, vmax=2)
    ax.axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_cam(wafer_map, heatmap, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow((np.asarray(wafer_map) > 0), cmap="gray", vmin=0, vmax=1, alpha=0.35)
    ax.imshow(heatmap, cmap="jet", alpha=0.65)
    ax.axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Generate Grad-CAM figures on real WM-811K wafers.")
    p.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "best_cnn.pt"))
    p.add_argument("--pkl-path", default=str(config.RAW_PKL))
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--out", default=str(config.ROOT / "docs" / "images"))
    p.add_argument(
        "--classes",
        nargs="+",
        default=["Scratch", "Donut", "Edge-Loc", "Center"],
        help="Which defect classes to visualise (must be in FAILURE_CLASSES).",
    )
    return p.parse_args()


def main():
    # Imported here so `import src.gradcam` (and the unit tests) never require
    # the dataset-loading stack.
    from .data_loader import load_labeled
    from .preprocessing import lot_group_split, resize_all

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WaferCNN(num_classes=len(config.FAILURE_CLASSES), in_size=args.img_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    df = load_labeled(args.pkl_path)
    _, _, test_df = lot_group_split(
        df, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE, seed=config.SEED
    )
    test_x = resize_all(test_df["waferMap"], size=args.img_size)
    test_y = test_df["label"].to_numpy()

    out = Path(args.out)
    rows = []
    with GradCAM(model) as cam:
        for cls in args.classes:
            cls_idx = config.FAILURE_CLASSES.index(cls)
            candidates = np.where(test_y == cls_idx)[0]
            if len(candidates) == 0:
                print(f"[skip] no test wafers for class {cls}")
                continue
            # pick the first wafer the model classifies correctly, for a clean example
            chosen = None
            for i in candidates:
                x = wafer_to_input(test_x[i]).to(device)
                heat, pred, probs = cam(x)
                if pred == cls_idx:
                    chosen = (test_x[i], heat, probs[cls_idx])
                    break
            if chosen is None:  # fall back to the first sample even if misclassified
                x = wafer_to_input(test_x[candidates[0]]).to(device)
                heat, _, probs = cam(x)
                chosen = (test_x[candidates[0]], heat, probs[cls_idx])
            wmap, heat, conf = chosen
            raw_path = out / f"{cls.lower().replace('-', '_')}_raw.png"
            cam_path = out / f"{cls.lower().replace('-', '_')}_cam.png"
            save_raw(wmap, raw_path)
            save_cam(wmap, heat, cam_path)
            rows.append((cls, conf, raw_path.name, cam_path.name))
            print(f"[ok] {cls}: p={conf:.2f} -> {raw_path.name}, {cam_path.name}")

    print("\nMarkdown table:\n")
    print("| Raw wafer map | Class | Grad-CAM overlay |")
    print("| :---: | :---: | :---: |")
    for cls, conf, raw, camf in rows:
        print(f"| ![raw](docs/images/{raw}) | {cls} (p={conf:.2f}) | ![cam](docs/images/{camf}) |")


if __name__ == "__main__":
    main()
