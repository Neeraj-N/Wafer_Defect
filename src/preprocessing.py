"""Resizing and lot-grouped train/val/test splitting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    import cv2

    def resize_map(wafer_map: np.ndarray, size: int = 64) -> np.ndarray:
        """Nearest-neighbor resize. Preserves the discrete {0,1,2} pixel
        semantics -- bilinear/bicubic would blend categorical values into
        meaningless intermediate numbers."""
        return cv2.resize(
            np.asarray(wafer_map).astype(np.uint8),
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )

except ImportError:  # pragma: no cover - fallback if opencv isn't installed
    from scipy.ndimage import zoom

    def resize_map(wafer_map: np.ndarray, size: int = 64) -> np.ndarray:
        m = np.asarray(wafer_map)
        h, w = m.shape
        return zoom(m, (size / h, size / w), order=0).astype(np.uint8)


def resize_all(wafer_maps, size: int = 64) -> np.ndarray:
    return np.stack([resize_map(m, size) for m in wafer_maps])


def lot_group_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
):
    """Split by `lotName` so wafers from the same manufacturing lot never
    appear in more than one split (avoids leakage from lot-correlated
    patterns). Rows with no lot name are treated as their own singleton
    group.
    """
    groups = df["lotName"].fillna(df.index.to_series().astype(str))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    trainval_df = df.iloc[trainval_idx]
    trainval_groups = groups.iloc[trainval_idx]

    relative_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_groups))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df
