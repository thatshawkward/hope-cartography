"""streamed shape statistics for preactivation matrices too big for memory"""

from __future__ import annotations
import numpy as np
from scipy import stats as sps

_TINY = 1e-12


def ks_normal(z):
    n = z.shape[0]
    F = sps.norm.cdf(np.sort(z, axis=0))
    grid = np.arange(1, n + 1)[:, None] / n
    return np.maximum((grid - F).max(axis=0), (F - grid + 1.0 / n).max(axis=0))


class StreamingShapeStats:
    def __init__(self, n_total, units, ks_subsample=20000, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n_total
        self.idx = np.sort(rng.choice(n_total, size=min(ks_subsample, n_total),
                                      replace=False))
        self.sums = np.zeros((4, units))
        self.sub = np.empty((self.idx.size, units), dtype=np.float32)
        self.seen = 0

    def update(self, h):
        p = h.astype(np.float64)
        for k in range(4):
            self.sums[k] += p.sum(axis=0)
            if k < 3:
                p = p * h
        lo, hi = np.searchsorted(self.idx, (self.seen, self.seen + h.shape[0]))
        self.sub[lo:hi] = h[self.idx[lo:hi] - self.seen]
        self.seen += h.shape[0]

    def finalize(self):
        assert self.seen == self.n, (self.seen, self.n)
        e1, e2, e3, e4 = self.sums / self.n
        m = e1
        s = np.sqrt(np.maximum(e2 - m**2, 0.0)) + _TINY
        m3 = e3 - 3 * m * e2 + 2 * m**3
        m4 = e4 - 4 * m * e3 + 6 * m**2 * e2 - 3 * m**4
        z = (self.sub - m) / s
        return dict(mean=m, std=s, skew=m3 / s**3, exkurt=m4 / s**4 - 3.0,
                    ks=ks_normal(z), zsub=z.astype(np.float32))
