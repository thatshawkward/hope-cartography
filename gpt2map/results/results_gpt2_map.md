# GPT-2 map results

## block 2 (median exkurt +3.09)
- paper: 1997 prunes, 0 merges; NLL 3.5617 -> 4.2789 at density 0.35 (33 checkpoints)
- empirical: 1997 prunes, 0 merges; NLL 3.5617 -> 4.2789 at density 0.35 (33 checkpoints)
- baseline capacity: NLL 3.5617 -> 4.2789 at density 0.35
- baseline l1: NLL 3.5617 -> 4.3105 at density 0.35
- baseline random: NLL 3.5617 -> 3.5897 at density 0.35
- paper-vs-empirical removed-set Jaccard 1.000

## block 6 (median exkurt +0.29)
- paper: 1997 prunes, 0 merges; NLL 3.5617 -> 3.8425 at density 0.35 (33 checkpoints)
- empirical: 1997 prunes, 0 merges; NLL 3.5617 -> 3.8425 at density 0.35 (33 checkpoints)
- baseline capacity: NLL 3.5617 -> 3.8425 at density 0.35
- baseline l1: NLL 3.5617 -> 3.7698 at density 0.35
- baseline random: NLL 3.5617 -> 3.6638 at density 0.35
- paper-vs-empirical removed-set Jaccard 1.000

## block 12 (median exkurt +0.53)
- paper: 1997 prunes, 0 merges; NLL 3.5617 -> 3.6292 at density 0.35 (33 checkpoints)
- empirical: 1997 prunes, 0 merges; NLL 3.5617 -> 3.6292 at density 0.35 (33 checkpoints)
- baseline capacity: NLL 3.5617 -> 3.6292 at density 0.35
- baseline l1: NLL 3.5617 -> 3.6500 at density 0.35
- baseline random: NLL 3.5617 -> 3.6609 at density 0.35
- paper-vs-empirical removed-set Jaccard 1.000

- divergence has no variance across blocks (the two correlation models produced identical maps), so the kurtosis-tracking hypothesis is moot: rho never changed a decision. See the merge counts above.
- cross-register capacity Spearman (median) by block: b1 0.898, b2 0.912, b3 0.887, b4 0.808, b5 0.849, b6 0.839, b7 0.794, b8 0.700, b9 0.612, b10 0.552, b11 0.475, b12 0.517
