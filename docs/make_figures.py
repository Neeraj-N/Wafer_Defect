"""
Regenerate the figures embedded in the README.

Two figures, neither of which needs the 3.5 GB WM-811K download:

  --taxonomy       docs/images/defect_taxonomy.png
                   One illustrative synthetic map per class, so a reader sees
                   what each of the nine failure patterns looks like.

  --gradcam-demo   docs/images/gradcam_demo.png
                   Trains a small CNN on *synthetic* maps for a few epochs, then
                   runs the real src.gradcam on held-out synthetic wafers. This
                   exists to demonstrate the explainability pipeline end-to-end
                   without the dataset. It is a DEMONSTRATION on synthetic data,
                   not a report of model accuracy.

For Grad-CAM on real WM-811K wafers with your trained checkpoint, use the
`python -m src.gradcam` CLI instead (see the README).

Usage:
    python docs/make_figures.py --taxonomy --gradcam-demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FAILURE_CLASSES  # noqa: E402
from src.dataset import WaferMapDataset  # noqa: E402
from src.gradcam import GradCAM, wafer_to_input  # noqa: E402
from src.model import WaferCNN  # noqa: E402
from src.synthetic import make_batch, make_sample  # noqa: E402
from src.visualize import WAFER_CMAP  # noqa: E402

IMAGES = Path(__file__).resolve().parent / "images"
SIZE = 48


def make_taxonomy():
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))
    for ax, cls in zip(axes.ravel(), FAILURE_CLASSES):
        m = make_sample(cls, size=SIZE, rng=np.random.default_rng(hash(cls) % 2**32))
        ax.imshow(m, cmap=WAFER_CMAP, vmin=0, vmax=2)
        ax.set_title(cls, fontsize=11)
        ax.axis("off")
    axes.ravel()[-1].axis("off")
    fig.suptitle(
        "The nine WM-811K failure patterns (illustrative synthetic maps)",
        fontsize=12,
    )
    fig.tight_layout()
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = IMAGES / "defect_taxonomy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def _train_synthetic(epochs=6, per_class=300):
    torch.manual_seed(0)
    x_train, y_train = make_batch(n_per_class=per_class, size=SIZE, seed=1)
    ds = WaferMapDataset(x_train, y_train, augment=True)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model = WaferCNN(num_classes=len(FAILURE_CLASSES), in_size=SIZE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        tot, correct, n = 0.0, 0, 0
        for x, y in loader:
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        print(f"  [synthetic-train] epoch {ep + 1}/{epochs} loss {tot / n:.3f} acc {correct / n:.3f}")
    return model.eval()


def make_gradcam_demo(classes=("Center", "Donut", "Edge-Ring", "Scratch")):
    print("Training a small CNN on synthetic maps (demonstration only)...")
    model = _train_synthetic()
    rng = np.random.default_rng(999)
    n = len(classes)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.4 * n))
    with GradCAM(model) as cam:
        for i, cls in enumerate(classes):
            m = make_sample(cls, size=SIZE, rng=rng)
            heat, pred, probs = cam(wafer_to_input(m))
            axes[i, 0].imshow(m, cmap=WAFER_CMAP, vmin=0, vmax=2)
            axes[i, 0].set_ylabel(cls, fontsize=11, rotation=0, labelpad=32, va="center")
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            axes[i, 1].imshow((m > 0), cmap="gray", vmin=0, vmax=1, alpha=0.35)
            axes[i, 1].imshow(heat, cmap="jet", alpha=0.65)
            axes[i, 1].set_title(
                f"pred: {FAILURE_CLASSES[pred]} ({probs[pred] * 100:.0f}%)", fontsize=9
            )
            axes[i, 1].axis("off")
    axes[0, 0].set_title("Wafer map", fontsize=11)
    fig.suptitle(
        "Grad-CAM localisation (demonstration on synthetic data)", fontsize=12, y=1.005
    )
    fig.tight_layout()
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = IMAGES / "gradcam_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taxonomy", action="store_true")
    p.add_argument("--gradcam-demo", action="store_true")
    args = p.parse_args()
    if not (args.taxonomy or args.gradcam_demo):
        args.taxonomy = args.gradcam_demo = True
    if args.taxonomy:
        make_taxonomy()
    if args.gradcam_demo:
        make_gradcam_demo()


if __name__ == "__main__":
    main()
