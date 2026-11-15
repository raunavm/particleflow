import math

import numpy as np
from scipy.spatial import cKDTree


def local_max(X, calo, r):
    """Calorimeter elements with the highest energy within dR < r in their calorimeter (ties: lower index)."""
    X = np.asarray(X)
    out = np.zeros(len(X), bool)
    e = X[:, 5]
    # ECAL if X col 10 > 0, else HCAL if X col 11 > 0, else other
    group = np.where(X[:, 10] > 0, 0, np.where(X[:, 11] > 0, 1, 2))
    for g in (0, 1, 2):
        idx = np.flatnonzero(calo & (group == g))
        if not len(idx):
            continue
        pts = np.stack([X[idx, 2] + 50.0, np.mod(np.arctan2(X[idx, 3], X[idx, 4]) + math.pi, 2 * math.pi)], 1)
        pts[:, 1] = np.minimum(pts[:, 1], np.nextafter(2 * math.pi, 0))
        # periodic in phi
        tree = cKDTree(pts, boxsize=[1e4, 2 * math.pi])
        pairs = tree.query_pairs(r, output_type="ndarray")
        keep = np.ones(len(idx), bool)
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            ei, ej = e[idx[i]], e[idx[j]]
            loser = np.where((ei < ej) | ((ei == ej) & (i > j)), i, j)
            keep[loser] = False
        out[idx[keep]] = True
    return out
