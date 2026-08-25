"""
Handcrafted features for the classical-ML baseline.

Two feature families, both standard in the wafer-map literature:
  - density_features: fraction of failed dies in each cell of a grid
    overlaid on the wafer -- cheap, and already separates e.g. Center
    (dense middle) from Edge-Ring (dense border).
  - radon_features: Radon-transform projections of the failure mask,
    which are good at picking up line/ring/scratch-like structure that a
    simple grid misses (used in Decision-Tree-Ensemble wafer-map papers).
"""

import numpy as np
from skimage.transform import radon


def density_features(wafer_map: np.ndarray, grid=(3, 3)) -> np.ndarray:
    """Fraction of failed dies per grid cell, normalized by the number of
    real dies in that cell (so empty/no-wafer regions don't skew it)."""
    m = np.asarray(wafer_map)
    fail = (m == 2).astype(float)
    valid = (m > 0).astype(float)
    h, w = m.shape
    gh, gw = grid
    feats = []
    for i in range(gh):
        for j in range(gw):
            r0, r1 = int(i * h / gh), int((i + 1) * h / gh)
            c0, c1 = int(j * w / gw), int((j + 1) * w / gw)
            v = valid[r0:r1, c0:c1].sum()
            f = fail[r0:r1, c0:c1].sum()
            feats.append(f / v if v > 0 else 0.0)
    return np.array(feats)


def radon_features(wafer_map: np.ndarray, n_angles: int = 20) -> np.ndarray:
    """Mean and std of the Radon transform of the failure mask across
    `n_angles` projection angles."""
    m = (np.asarray(wafer_map) == 2).astype(float)
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinogram = radon(m, theta=theta, circle=False)
    return np.concatenate([sinogram.mean(axis=0), sinogram.std(axis=0)])


def extract_features(wafer_map: np.ndarray, grid=(3, 3), n_angles: int = 20) -> np.ndarray:
    m = np.asarray(wafer_map)
    overall_defect_ratio = (m == 2).sum() / max((m > 0).sum(), 1)
    return np.concatenate(
        [
            density_features(m, grid),
            radon_features(m, n_angles),
            [overall_defect_ratio],
        ]
    )


def extract_features_batch(wafer_maps, grid=(3, 3), n_angles: int = 20) -> np.ndarray:
    return np.stack([extract_features(m, grid, n_angles) for m in wafer_maps])
