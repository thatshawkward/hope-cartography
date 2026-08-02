# Stage 3 results: the map

Model: the Stage 2 char-MLP (val 1.7833 nats); encoded to 35% density; probe 32,768 contexts.
Unit labels at |corr| >= 0.25: distributed 60, last=newline 16, last=punct 7, last=space 120, last=vowel 41, next=space 1, next=upper 9, next=vowel 2

## Trajectory (val loss at 50% / 35% density)
- HOPE full (paper rho): 2.6662 / 2.8630
- HOPE full (empirical rho): 2.5898 / 2.8510
- HOPE prune-only: 2.6662 / 2.8630
- L1-norm prune: 2.9185 / 3.3296
- random prune: 2.7636 / 3.1052

## The eviction order (paper run)
- actions: 167 total, 0 merges, 167 prunes
- median removal step by label: next=upper 65, distributed 76, last=vowel 83, last=punct 84, last=space 90, last=newline 116, next=vowel 150
- survival fraction by label: next=space 1.00, next=upper 0.56, last=newline 0.44, last=space 0.35, distributed 0.35, last=vowel 0.29, last=punct 0.14, next=vowel 0.00
- |excess kurtosis|: removed units median 0.198 vs surviving 0.216
- merges joining two original units: 0, of which same-label 0 (0%)

## Robustness to the correlation model
- removed-set Jaccard (paper vs empirical rho): 0.898
- removal-order Spearman on common units: 0.974
- merges under empirical rho: 17 (vs 0 under the MaxEnt warp)
- initial correlations: warped rho_hat max +0.13 (p99 +0.06) vs empirical max +0.67 (p99 +0.47); the warp leaves no pair redundant enough for a merge to ever outprice a prune

## Showcase units
- unit 2 [next=upper, pruned step 56]: top contexts "EASURE.\n"; "\nTYRREL:"
- unit 10 [last=newline, survives]: top contexts "\nWARWICK"; "\nWARWICK"
- unit 36 [next=space, survives]: top contexts "bject of"; "place of"

Total runtime 34s.
