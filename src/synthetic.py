"""
Synthetic wafer-map generator.

This module fabricates *illustrative* wafer maps whose defect patterns
resemble the nine WM-811K classes. It exists so that parts of this repo can
run without the ~3.5 GB LSWMD.pkl download:

  * the defect-taxonomy figure in the README (what each class looks like),
  * a self-contained Grad-CAM demonstration (docs/make_figures.py),
  * the "no checkpoint / no data" demo mode of the Streamlit app,
  * fast unit tests that need realistic {0, 1, 2} maps.

These maps are NOT WM-811K data and must never be presented as real fab
results or used to report model performance. They only reproduce the coarse
geometry of each failure pattern (where the failed dies sit on the wafer).

Pixel convention matches the real dataset: 0 = background/no die, 1 = pass,
2 = fail.
"""

from __future__ import annotations

import numpy as np

from .config import FAILURE_CLASSES

BACKGROUND, PASS, FAIL = 0, 1, 2


def _wafer_base(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (map, inside_mask): a circular wafer of passing dies plus the
    boolean mask of die positions. A little edge jitter keeps the rim from
    looking artificially perfect."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = size / 2.0 - 0.5
    jitter = rng.normal(0.0, size * 0.01, size=(size, size))
    inside = r <= (radius + jitter)
    m = np.where(inside, PASS, BACKGROUND).astype(np.uint8)
    return m, inside


def _radial(size: int) -> tuple[np.ndarray, np.ndarray, float]:
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return r, np.arctan2(yy - cy, xx - cx), size / 2.0 - 0.5


def make_sample(
    cls: str,
    size: int = 48,
    rng: np.random.Generator | None = None,
    base_noise: float = 0.01,
) -> np.ndarray:
    """Generate one synthetic wafer map for class `cls`.

    `base_noise` is the fraction of passing dies that randomly fail, present
    in every class so that even `none` is not perfectly clean.
    """
    if rng is None:
        rng = np.random.default_rng()
    if cls not in FAILURE_CLASSES:
        raise ValueError(f"unknown class {cls!r}; expected one of {FAILURE_CLASSES}")

    m, inside = _wafer_base(size, rng)
    r, theta, radius = _radial(size)

    def fail_where(mask: np.ndarray, p: float = 1.0) -> None:
        sel = mask & inside
        if p < 1.0:
            sel = sel & (rng.random((size, size)) < p)
        m[sel] = FAIL

    # baseline speckle everywhere
    fail_where(rng.random((size, size)) < base_noise)

    if cls == "none":
        pass
    elif cls == "Center":
        fail_where(r < radius * rng.uniform(0.30, 0.45), p=0.9)
    elif cls == "Donut":
        lo, hi = radius * 0.45, radius * 0.70
        fail_where((r > lo) & (r < hi), p=0.85)
    elif cls == "Edge-Ring":
        fail_where(r > radius * rng.uniform(0.82, 0.90), p=0.9)
    elif cls == "Edge-Loc":
        a0 = rng.uniform(-np.pi, np.pi)
        span = rng.uniform(0.5, 1.1)
        arc = (np.abs(np.angle(np.exp(1j * (theta - a0)))) < span) & (r > radius * 0.72)
        fail_where(arc, p=0.85)
    elif cls == "Loc":
        a0 = rng.uniform(-np.pi, np.pi)
        rc = radius * rng.uniform(0.35, 0.6)
        cy0, cx0 = (size - 1) / 2 + rc * np.sin(a0), (size - 1) / 2 + rc * np.cos(a0)
        yy, xx = np.mgrid[0:size, 0:size]
        blob = np.sqrt((yy - cy0) ** 2 + (xx - cx0) ** 2) < radius * rng.uniform(0.18, 0.28)
        fail_where(blob, p=0.9)
    elif cls == "Scratch":
        a0 = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:size, 0:size]
        cy = cx = (size - 1) / 2.0
        # signed distance to a line through the centre at angle a0
        dist = np.abs((xx - cx) * np.sin(a0) - (yy - cy) * np.cos(a0))
        along = (xx - cx) * np.cos(a0) + (yy - cy) * np.sin(a0)
        line = (dist < rng.uniform(0.8, 1.6)) & (np.abs(along) < radius * rng.uniform(0.6, 0.95))
        fail_where(line, p=0.9)
    elif cls == "Random":
        fail_where(rng.random((size, size)) < rng.uniform(0.12, 0.22))
    elif cls == "Near-full":
        fail_where(np.ones((size, size), dtype=bool), p=rng.uniform(0.85, 0.95))

    return m


def make_batch(
    n_per_class: int = 1,
    size: int = 48,
    classes: list[str] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (maps, labels) with `n_per_class` samples for each class.

    labels are integer indices into FAILURE_CLASSES.
    """
    classes = classes or FAILURE_CLASSES
    rng = np.random.default_rng(seed)
    maps, labels = [], []
    for cls in classes:
        idx = FAILURE_CLASSES.index(cls)
        for _ in range(n_per_class):
            maps.append(make_sample(cls, size=size, rng=rng))
            labels.append(idx)
    return np.stack(maps), np.array(labels, dtype=np.int64)
