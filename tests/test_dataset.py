import numpy as np

from src.dataset import WaferMapDataset


def test_dataset_produces_two_binary_channels():
    m = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    ds = WaferMapDataset(np.stack([m]), np.array([3]), augment=False)
    x, y = ds[0]
    assert tuple(x.shape) == (2, 2, 3)      # (channels, H, W)
    assert set(x.unique().tolist()) <= {0.0, 1.0}  # binary masks
    # channel 0 = "die exists" (m > 0), channel 1 = "die failed" (m == 2)
    assert np.array_equal(x[0].numpy(), (m > 0).astype(np.float32))
    assert np.array_equal(x[1].numpy(), (m == 2).astype(np.float32))
    assert int(y) == 3


def test_augmentation_keeps_channels_aligned_and_binary():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 3, size=(8, 8)).astype(np.uint8)
    ds = WaferMapDataset(np.stack([m]), np.array([0]), augment=True)
    for _ in range(10):
        x, _ = ds[0]
        assert tuple(x.shape) == (2, 8, 8)
        assert set(x.unique().tolist()) <= {0.0, 1.0}
        # a failed die is always also a valid die -> channel1 implies channel0
        assert bool(((x[1] == 1) <= (x[0] == 1)).all())
