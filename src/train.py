"""
Train the CNN baseline on WM-811K.

Usage:
    python -m src.train --epochs 40 --batch-size 128 --img-size 64

Selection and reporting use macro-F1, not accuracy: with ~85% of wafers
labeled 'none', accuracy rewards a model that leans on the majority class and
swings noisily epoch to epoch (a small shift in the none-vs-rest boundary moves
15+ points of accuracy) while telling you nothing about the minority defect
patterns you actually care about.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .data_loader import load_labeled
from .dataset import WaferMapDataset
from .losses import FocalLoss
from .model import WaferCNN
from .preprocessing import lot_group_split, resize_all


def parse_args():
    p = argparse.ArgumentParser(description="Train the wafer defect CNN.")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    p.add_argument(
        "--loss",
        choices=["ce", "focal"],
        default="ce",
        help="ce = class-weighted cross-entropy (default); focal = weighted focal loss "
        "(extra down-weighting of easy examples for the rare defect classes).",
    )
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Focusing parameter for --loss focal.")
    p.add_argument("--pkl-path", default=str(config.RAW_PKL))
    p.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    return p.parse_args()


def set_seed(seed: int):
    """Seed python/numpy/torch so a training run is reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    df = load_labeled(args.pkl_path)
    train_df, val_df, _ = lot_group_split(
        df, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE, seed=config.SEED
    )

    train_x = resize_all(train_df["waferMap"], size=args.img_size)
    val_x = resize_all(val_df["waferMap"], size=args.img_size)
    train_y = train_df["label"].to_numpy()
    val_y = val_df["label"].to_numpy()

    train_ds = WaferMapDataset(train_x, train_y, augment=True)
    val_ds = WaferMapDataset(val_x, val_y, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(config.FAILURE_CLASSES)), y=train_y
    )
    return train_loader, val_loader, torch.tensor(class_weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch; return (loss, accuracy, macro-F1). Checkpoint selection
    uses the macro-F1 -- see the module docstring for why not accuracy."""
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_targets = [], []
    torch.set_grad_enabled(train)
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        n += x.size(0)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(y.detach().cpu().tolist())
    torch.set_grad_enabled(True)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / n, correct / n, macro_f1


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_weights = build_loaders(args)

    model = WaferCNN(num_classes=len(config.FAILURE_CLASSES), in_size=args.img_size).to(device)
    weights = class_weights.to(device)
    if args.loss == "focal":
        criterion = FocalLoss(weight=weights, gamma=args.focal_gamma)
        print(f"Loss: weighted focal (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Loss: class-weighted cross-entropy")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Halve the LR when val macro-F1 plateaus; this tames the epoch-to-epoch
    # thrash you otherwise see with a fixed LR on this imbalanced data.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.LR_PATIENCE
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, False
        )
        scheduler.step(val_f1)
        print(
            f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_since_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_cnn.pt")
            print(f"  -> saved new best checkpoint (val macro-F1 {val_f1:.4f})")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"  -> early stopping: no val macro-F1 gain in {args.patience} epochs")
                break

    print(f"Best val macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
