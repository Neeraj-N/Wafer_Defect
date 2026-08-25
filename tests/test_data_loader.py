import numpy as np
import pandas as pd

from src.data_loader import clean, encode_labels, labeled_subset
from src.preprocessing import lot_group_split, resize_map


def make_toy_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    classes = ["none", "Center", "Donut"]
    for i in range(n):
        h, w = rng.integers(20, 30), rng.integers(20, 30)
        wafer_map = rng.integers(0, 3, size=(h, w))
        labeled = i % 3 != 0  # 2/3 labeled, 1/3 unlabeled
        rows.append(
            {
                "waferMap": wafer_map,
                "dieSize": float(h * w),
                "lotName": f"lot{i % 5}",
                "waferIndex": i % 5 + 1,
                "trainTestLabel": [["Training"]] if labeled else [],
                "failureType": [[classes[i % 3]]] if labeled else [],
            }
        )
    return pd.DataFrame(rows)


def test_clean_unwraps_nested_labels():
    df = clean(make_toy_df())
    val = df["failureType"].iloc[0]
    assert pd.isna(val) or val in {"none", "Center", "Donut"}
    # unlabeled rows (every 3rd, by construction of make_toy_df) should be
    # unwrapped to a missing value, not an empty array
    assert df.loc[df.index[0::3], "failureType"].isna().all()


def test_labeled_subset_drops_unlabeled():
    df = clean(make_toy_df(n=30))
    labeled = labeled_subset(df)
    assert labeled["failureType"].notna().all()
    assert len(labeled) < len(df)


def test_encode_labels_roundtrip():
    df = clean(make_toy_df(n=30))
    labeled = labeled_subset(df)
    encoded, mapping = encode_labels(labeled, classes=["none", "Center", "Donut"])
    assert set(encoded["label"].unique()) <= set(mapping.values())
    assert encoded["label"].dtype == int


def test_resize_map_preserves_discrete_values():
    m = np.array([[0, 1, 2], [2, 1, 0]])
    resized = resize_map(m, size=8)
    assert resized.shape == (8, 8)
    assert set(np.unique(resized)) <= {0, 1, 2}


def test_lot_group_split_no_leakage():
    df = clean(make_toy_df(n=60))
    df = labeled_subset(df)
    df, _ = encode_labels(df, classes=["none", "Center", "Donut"])
    train_df, val_df, test_df = lot_group_split(df, test_size=0.2, val_size=0.2, seed=1)

    train_lots = set(train_df["lotName"])
    val_lots = set(val_df["lotName"])
    test_lots = set(test_df["lotName"])

    assert not (train_lots & val_lots)
    assert not (train_lots & test_lots)
    assert not (val_lots & test_lots)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
