"""
Train the classical-ML baseline (handcrafted features + Random Forest).

Useful as a fast sanity check before investing in the CNN, and as a
lightweight, interpretable point of comparison.

Usage:
    python -m src.train_baseline
"""

import argparse
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

from . import config
from .data_loader import load_labeled
from .features import extract_features_batch
from .preprocessing import lot_group_split


def parse_args():
    p = argparse.ArgumentParser(description="Train the Random Forest baseline.")
    p.add_argument("--pkl-path", default=str(config.RAW_PKL))
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    df = load_labeled(args.pkl_path)
    train_df, _val_df, test_df = lot_group_split(
        df, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE, seed=config.SEED
    )

    print(f"Extracting features for {len(train_df)} train / {len(test_df)} test wafers...")
    x_train = extract_features_batch(train_df["waferMap"])
    y_train = train_df["label"].to_numpy()
    x_test = extract_features_batch(test_df["waferMap"])
    y_test = test_df["label"].to_numpy()

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.SEED,
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(f"Macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=config.FAILURE_CLASSES))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "rf_baseline.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model to {ckpt_dir / 'rf_baseline.pkl'}")


if __name__ == "__main__":
    main()
