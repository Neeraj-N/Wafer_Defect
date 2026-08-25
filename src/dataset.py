"""PyTorch Dataset over resized wafer maps."""

import numpy as np
import torch
from torch.utils.data import Dataset


class WaferMapDataset(Dataset):
    """Expects wafer maps already resized to a fixed (H, W) shape (see
    src.preprocessing.resize_all) with values in {0, 1, 2}.

    Each map is turned into two binary channels -- a "die exists" mask and a
    "die failed" mask -- rather than a single ordinal channel, so the CNN never
    sees the false ordering background < pass < fail (they are categories).

    Wafers can be mounted in any orientation, so flips and 90-degree rotations
    are label-preserving augmentations. The *same* transform is applied to both
    channels so they stay aligned.
    """

    def __init__(self, wafer_maps: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.wafer_maps = wafer_maps
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.wafer_maps)

    def __getitem__(self, idx):
        m = np.asarray(self.wafer_maps[idx])
        valid = (m > 0).astype(np.float32)   # a die exists at this position
        fail = (m == 2).astype(np.float32)   # ... and it failed electrical test

        if self.augment:
            if np.random.rand() < 0.5:
                valid, fail = np.fliplr(valid), np.fliplr(fail)
            if np.random.rand() < 0.5:
                valid, fail = np.flipud(valid), np.flipud(fail)
            k = np.random.randint(0, 4)
            valid, fail = np.rot90(valid, k), np.rot90(fail, k)

        # np.stack returns a fresh C-contiguous array, so torch.from_numpy is
        # safe even though fliplr/rot90 produced negative-stride views.
        x = torch.from_numpy(np.stack([valid, fail]))  # (2, H, W)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y
