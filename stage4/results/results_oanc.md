# Stage 4 results: trajectory fine-tuning (oanc)

Recovery blocks: 750 Adam steps at lr 0.0015 at each of 13 density checkpoints (0.95 down to 0.35); end-only gets the identical total budget of 9750 steps in one block. Recovery seeds are paired across arms.

## paper correlations

| arm | val @ 0.6 | val @ 0.5 | val @ 0.42 | val @ 0.35 |
|---|---|---|---|---|
| base | 2.2003 | 2.3455 | 2.4824 | 2.6355 |
| A: frozen plan | 1.8510 | 1.8893 | 1.9761 | 1.9634 |
| B: re-planned | 1.8258 | 1.8562 | 1.9290 | 1.9357 |
| end-only | 2.2003 | 2.3455 | 2.4824 | 1.8566 |
| capacity + recovery | 1.8510 | 1.8893 | 1.9761 | 1.9634 |
| L1 + recovery | 1.8606 | 1.9016 | 1.9887 | 1.9964 |
| random + recovery | 1.8462 | 1.8808 | 1.9506 | 1.9392 |
| L1 static | 2.4372 | 2.7158 | 2.8781 | 3.0639 |
| random static | 2.2685 | 2.4775 | 2.5990 | 2.7320 |

Eviction-set divergence, base vs re-planned (paper): Jaccard 0.661 over removed originals (199 common of 250/250), removal-order Spearman 0.747 on the common set.

## empirical correlations

| arm | val @ 0.6 | val @ 0.5 | val @ 0.42 | val @ 0.35 |
|---|---|---|---|---|
| base | 2.1823 | 2.3205 | 2.4901 | 2.5981 |
| A: frozen plan | 1.8363 | 1.8780 | 1.9661 | 1.9660 |
| B: re-planned | 1.8355 | 1.8693 | 1.9360 | 1.9424 |
| end-only | 2.1823 | 2.3205 | 2.4901 | 1.8497 |
| capacity + recovery | 1.8510 | 1.8893 | 1.9761 | 1.9634 |
| L1 + recovery | 1.8606 | 1.9016 | 1.9887 | 1.9964 |
| random + recovery | 1.8462 | 1.8808 | 1.9506 | 1.9392 |
| L1 static | 2.4372 | 2.7158 | 2.8781 | 3.0639 |
| random static | 2.2685 | 2.4775 | 2.5990 | 2.7320 |

Eviction-set divergence, base vs re-planned (empirical): Jaccard 0.598 over removed originals (189 common of 251/254), removal-order Spearman 0.726 on the common set.

## Capacity drift at recovery blocks

- paper: Spearman(pre, post) ranges 0.826 (at density 0.55) to 0.974.
- empirical: Spearman(pre, post) ranges 0.874 (at density 0.55) to 0.974.

Figures: fig8_oanc_recovery_trajectories.png, fig9_oanc_capacity_drift.png.
