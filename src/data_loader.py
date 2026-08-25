"""
Load and clean the WM-811K wafer map dataset (LSWMD.pkl).

The raw pickle file is a pandas DataFrame with columns:
    waferMap        2D uint8 numpy array, per die: 0 = background/no die,
                     1 = pass, 2 = fail. Shape varies wafer to wafer.
    dieSize          float, number of dies on the wafer
    lotName          str, manufacturing lot identifier
    waferIndex       float, index of the wafer within its lot (usually 1-25)
    trainTestLabel   nested 1-elem array e.g. [['Training']], or [] if unlabeled
    failureType      nested 1-elem array e.g. [['Center']], or [] if unlabeled

The nested-array label fields are a leftover of the original MATLAB struct
this dataset was exported from; `clean()` unwraps them to plain strings
(or None for unlabeled wafers).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FAILURE_CLASSES


def _unwrap(cell):
    """Unwrap a possibly-nested 1-element array/list to a scalar, or None
    if empty (i.e. the wafer has no human label)."""
    arr = np.asarray(cell, dtype=object)
    if arr.size == 0:
        return None
    return arr.reshape(-1)[0]


def load_raw(pkl_path) -> pd.DataFrame:
    """Load the raw LSWMD.pkl file into a DataFrame."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"{pkl_path} not found. See scripts/download_data.sh for how "
            "to obtain LSWMD.pkl, then place it at data/raw/LSWMD.pkl."
        )
    return pd.read_pickle(pkl_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Unwrap nested label fields and add a few convenience columns."""
    df = df.copy()
    # The original .mat/pkl release has a typo in this column
    # ("trianTestLabel" instead of "trainTestLabel"). Normalize it so the
    # rest of the codebase can use the correctly spelled name regardless
    # of which version of the file you have.
    if "trianTestLabel" in df.columns and "trainTestLabel" not in df.columns:
        df = df.rename(columns={"trianTestLabel": "trainTestLabel"})

    df["failureType"] = df["failureType"].apply(_unwrap)
    df["trainTestLabel"] = df["trainTestLabel"].apply(_unwrap)
    df["waferMapDim"] = df["waferMap"].apply(lambda m: np.asarray(m).shape)
    return df


def labeled_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only wafers with a human-assigned failure type (~21% of the
    full dataset)."""
    return df[df["failureType"].notna()].reset_index(drop=True)


def encode_labels(df: pd.DataFrame, classes=FAILURE_CLASSES):
    """Map string failure types to integer class indices."""
    label_to_idx = {c: i for i, c in enumerate(classes)}
    df = df.copy()
    df["label"] = df["failureType"].map(label_to_idx)
    unmapped = df["label"].isna().sum()
    if unmapped:
        raise ValueError(
            f"{unmapped} rows have a failureType not in FAILURE_CLASSES: "
            f"{sorted(set(df.loc[df['label'].isna(), 'failureType']))}"
        )
    df["label"] = df["label"].astype(int)
    return df, label_to_idx


def load_labeled(pkl_path) -> pd.DataFrame:
    """Convenience one-shot: load, clean, filter to labeled, encode labels."""
    df = load_raw(pkl_path)
    df = clean(df)
    df = labeled_subset(df)
    df, _ = encode_labels(df)
    return df
