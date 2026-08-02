"""Demo progressive encoding of a synthetic BN + ReLU layer"""

import numpy as np

from minihope.greedy import format_ledger, greedy_encode
from minihope.layer import HopeLayer

rng = np.random.default_rng(42)

N_IN, N_OUT = 16, 8

#anisotropic data distribution.
Q, _ = np.linalg.qr(rng.normal(size=(N_IN, N_IN)))
S = Q @ np.diag(rng.uniform(0.3, 3.0, N_IN)) @ Q.T          
m = rng.normal(scale=0.5, size=N_IN)                    

blocks = []
core = rng.normal(size=(8, N_IN))
core_out = rng.normal(size=(8, N_OUT))
blocks.append((core, core_out, rng.uniform(0.8, 1.2, 8)))

dup_src = rng.choice(8, size=4, replace=False)
dups = core[dup_src] + 0.05 * rng.normal(size=(4, N_IN))
dups_out = core_out[dup_src] + 0.05 * rng.normal(size=(4, N_OUT))
blocks.append((dups, dups_out, rng.uniform(0.8, 1.2, 4)))

dead = rng.normal(size=(6, N_IN))
dead_out = rng.normal(size=(6, N_OUT))
blocks.append((dead, dead_out, rng.uniform(0.02, 0.08, 6)))

w_raw = np.vstack([b[0] for b in blocks])
w_out = np.vstack([b[1] for b in blocks])
gamma = np.concatenate([b[2] for b in blocks])
beta = rng.normal(scale=0.2, size=len(gamma))

#BN statistics under x ~ N(m, S)
mu = w_raw @ m
sigma = np.sqrt(np.einsum("ni,ij,nj->n", w_raw, S, w_raw))

layer = HopeLayer(w_raw, w_out, gamma, beta, mu, sigma)
print(f"initial: N={layer.n_neurons}, layer capacity E={layer.capacities().sum():.4f}")
print("planted duplicates of core neurons:", sorted(int(k) for k in dup_src),
      "-> rows 8-11; near-dead rows: 12-17\n")

final, ledger = greedy_encode(layer, target_density=0.5)
print(format_ledger(ledger))
print(f"\nfinal: N={final.n_neurons}, layer capacity E={final.capacities().sum():.4f}")
