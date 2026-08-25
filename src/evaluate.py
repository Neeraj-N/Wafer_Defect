"""
Evaluate a trained CNN checkpoint on the held-out test split.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best_cnn.pt
"""

import argparse

import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from . import config
from .data_loader import load_labeled
from .dataset import WaferMapDataset
from .model import WaferCNN
from .preprocessing import lot_group_split, resize_all
from .visualize import plot_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the CNN on the test split.")
    p.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "best_cnn.pt"))
    p.add_argument("--pkl-path", default=str(config.RAW_PKL))
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--out", default=None, help="Optional path to save the confusion matrix plot")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_labeled(args.pkl_path)
    _, _, test_df = lot_group_split(
        df, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE, seed=config.SEED
    )
    test_x = resize_all(test_df["waferMap"], size=args.img_size)
    test_y = test_df["label"].to_numpy()
    test_loader = DataLoader(
        WaferMapDataset(test_x, test_y, augment=False), batch_size=args.batch_size
    )

    model = WaferCNN(num_classes=len(config.FAILURE_CLASSES), in_size=args.img_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device))
            preds.extend(out.argmax(1).cpu().tolist())
            targets.extend(y.tolist())

    print(f"Macro-F1: {f1_score(targets, preds, average='macro'):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(targets, preds):.4f}")
    print(classification_report(targets, preds, target_names=config.FAILURE_CLASSES))

    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, config.FAILURE_CLASSES, save_path=args.out)


if __name__ == "__main__":
    main()
