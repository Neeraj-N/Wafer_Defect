import numpy as np

from src.config import FAILURE_CLASSES
from src.synthetic import make_batch, make_sample


def test_make_sample_shape_and_values():
    for cls in FAILURE_CLASSES:
        m = make_sample(cls, size=40, rng=np.random.default_rng(0))
        assert m.shape == (40, 40)
        assert set(np.unique(m)) <= {0, 1, 2}


def test_defect_classes_have_more_fails_than_none():
    rng = np.random.default_rng(1)

    def fail_frac(cls):
        m = make_sample(cls, size=48, rng=rng)
        valid = (m > 0).sum()
        return (m == 2).sum() / max(valid, 1)

    none_frac = np.mean([fail_frac("none") for _ in range(5)])
    for cls in ["Center", "Edge-Ring", "Scratch", "Near-full"]:
        assert np.mean([fail_frac(cls) for _ in range(5)]) > none_frac


def test_make_batch_labels_align():
    maps, labels = make_batch(n_per_class=2, size=32, seed=3)
    assert maps.shape == (2 * len(FAILURE_CLASSES), 32, 32)
    assert labels.shape == (2 * len(FAILURE_CLASSES),)
    assert set(labels.tolist()) == set(range(len(FAILURE_CLASSES)))
