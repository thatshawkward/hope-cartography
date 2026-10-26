# Stage 4 results: trajectory fine-tuning (shakespeare)

Recovery blocks: 750 Adam steps at lr 0.0015 at each of 13 density checkpoints (0.95 down to 0.35); end-only gets the identical total budget of 9750 steps in one block. Recovery seeds are paired across arms.

## paper correlations

| arm | val @ 0.6 | val @ 0.5 | val @ 0.42 | val @ 0.35 |
|---|---|---|---|---|
| base | 2.4837 | 2.6662 | 2.7617 | 2.8580 |
| A: frozen plan | 1.8724 | 1.8996 | 1.9935 | 1.9424 |
| B: re-planned | 1.8679 | 1.8909 | 1.9833 | 1.9426 |
| end-only | 2.4837 | 2.6662 | 2.7617 | 1.8931 |
| capacity + recovery | 1.8724 | 1.8996 | 1.9935 | 1.9424 |
| L1 + recovery | 1.8838 | 1.9147 | 2.0054 | 1.9396 |
| random + recovery | 1.8632 | 1.8848 | 1.9537 | 1.9229 |
| L1 static | 2.7013 | 2.9185 | 3.0902 | 3.3213 |
| random static | 2.4773 | 2.6135 | 2.7927 | 2.8965 |

Eviction-set divergence, base vs re-planned (paper): Jaccard 0.722 over removed originals (140 common of 167/167), removal-order Spearman 0.805 on the common set.

## empirical correlations

| arm | val @ 0.6 | val @ 0.5 | val @ 0.42 | val @ 0.35 |
|---|---|---|---|---|
| base | 2.4960 | 2.5898 | 2.7822 | 2.8558 |
| A: frozen plan | 1.8732 | 1.9048 | 2.0026 | 1.9371 |
| B: re-planned | 1.8902 | 1.9079 | 2.0330 | 1.9487 |
| end-only | 2.4960 | 2.5898 | 2.7822 | 1.8902 |
| capacity + recovery | 1.8724 | 1.8996 | 1.9935 | 1.9424 |
| L1 + recovery | 1.8838 | 1.9147 | 2.0054 | 1.9396 |
| random + recovery | 1.8632 | 1.8848 | 1.9537 | 1.9229 |
| L1 static | 2.7013 | 2.9185 | 3.0902 | 3.3213 |
| random static | 2.4773 | 2.6135 | 2.7927 | 2.8965 |

Eviction-set divergence, base vs re-planned (empirical): Jaccard 0.702 over removed originals (139 common of 167/170), removal-order Spearman 0.781 on the common set.

## Capacity drift at recovery blocks

- paper: Spearman(pre, post) ranges 0.947 (at density 0.40) to 0.990.
- empirical: Spearman(pre, post) ranges 0.952 (at density 0.45) to 0.990.

Figures: fig8_recovery_trajectories.png, fig9_capacity_drift.png.
