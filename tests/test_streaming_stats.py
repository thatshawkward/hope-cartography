"""StreamingShapeStats must reproduce gaussianity_stats without the matrix"""

import numpy as np
from scipy import stats as sps

from stage2.experiments import gaussianity_stats
from stage2.streaming import StreamingShapeStats, ks_normal

RNG = np.random.default_rng(5)


def _mixed_matrix(n=4096):
    cols = [RNG.normal(0.3, 1.7, n),
            RNG.exponential(2.0, n) - 2.0,
            RNG.standard_t(df=5, size=n) * 0.8 + 1.0,
            RNG.normal(-2.0, 0.05, n),
            RNG.uniform(-1.0, 1.0, n)]
    return np.stack(cols, axis=1).astype(np.float32)


def test_matches_gaussianity_stats():
    h = _mixed_matrix()
    ref = gaussianity_stats(h, ks_subsample=1500, seed=0)
    st = StreamingShapeStats(h.shape[0], h.shape[1], ks_subsample=1500, seed=0)
    for s in range(0, h.shape[0], 700):   
        st.update(h[s:s + 700])
    got = st.finalize()
    for key, tol in (("mean", 1e-5), ("std", 1e-5),
                     ("skew", 1e-3), ("exkurt", 1e-3)):
        np.testing.assert_allclose(got[key], ref[key], rtol=tol, atol=tol)
    #same seed => same subsample rows => same KS statistic, up to the reference's float32 moments
    np.testing.assert_allclose(got["ks"], ref["ks"], atol=1e-4)


def test_ks_normal_matches_scipy():
    z = RNG.normal(size=(400, 6)) * 1.3 + 0.2
    want = [sps.kstest(z[:, u], "norm").statistic for u in range(z.shape[1])]
    np.testing.assert_allclose(ks_normal(z), want, atol=1e-12)
