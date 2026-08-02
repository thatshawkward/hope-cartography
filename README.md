# mini-HOPE

A NumPy implementation of the core machinery from **"Hilbert Operator
for Progressive Encoding (HOPE)"** (Mobahi & Bartlett, Google DeepMind, 2026;
arXiv:2607.21366)


## Status

**Stage 1** (`src/minihope/`, `demo.py`): closed-form ReLU
kernels, capacities, prune/merge costs, parent synthesis with BN recovery,
and the greedy encoder. Every closed form is Monte-Carlo verified; the
parent solve now uses the O(n) rank-2 reduction, pinned against the explicit
SVD by test.

**Stage 2** (`stage2/`): the Gaussian surrogate vs.
real text. On the char-MLP the median unit is nearly Gaussian but a heavy
tail is not; kernel values drift ~7-9% while the capacity ordering survives
(Spearman 0.982). On GPT-2 the fiction strains much harder (median excess
kurtosis +0.43, outlier units to +777, worst at the ends of the stack); see
`stage2/results/results_gpt2.md`.

**Stage 3** (`stage3/`): every action deployed into the live model,
units carrying identities, linguistic labels, and kurtosis; merges carrying
lineage; a second pass repricing merges with empirical correlations.
Findings, with ledgers and figures in `stage3/results/`:

- Under the paper's MaxEnt warp the encoder never merges: all 167
  actions are prunes, because the warp caps pairwise correlation at 0.13
  where the empirical maximum is 0.67. Stage 2's shrinkage becomes a
  behavioral fact; with empirical correlations, 17 merges fire and the
  trajectory improves slightly.
- The map is robust to that choice: removed-set Jaccard 0.898, removal-order
  Spearman 0.974.
- The capacity ordering earns its keep on text: 0.47 nats below L1-norm
  pruning and 0.24 below random at 35% density.
- Structure survives: newline/speaker-header units are disproportionately
  preserved, punctuation detectors are cut hardest, and heavy-tailed units
  skew toward surviving. No retraining between actions by design.

Next: trajectory fine-tuning, and GELU kernels so the GPT-2 map can be drawn where the
surrogate strains most.

## Broad corpus: OANC

`stage2/corpora.py` + `broad/run_broad.py` retrain and rerun everything on
the Open American National Corpus (contemporary American English, written
and spoken; the open analog of COCA). Where the full ~15M-word OANC isn't
downloaded, the pipeline runs on its MASC slice (~500K words, 20 genres,
auto-fetched). For the full corpus: anc.org's TLS certificate lapsed in
July 2025, so fetch it over plain HTTP (no certificate involved),
`curl -L -o OANC_GrAF.zip http://www.anc.org/OANC/OANC_GrAF.zip`, or
knowingly accept the expired certificate with `curl -kL`, or use the
Internet Archive snapshot
(`https://web.archive.org/web/2024/http://www.anc.org/OANC/OANC_GrAF.zip`).
Expect a ~0.6 GB zip unpacking to ~7.4 GB across 62,090 files (the site's
326 MB figure describes the separate original-XML packaging); extract with
`unzip -q OANC_GrAF.zip -d data/oanc`, and the `OANC-GrAF/` wrapper directory
inside is fine, since the loader anchors on the `data/` tree, reads the
plain-text primaries, and ignores the annotation standoff. After the first
run builds the cleaned-corpus cache, the raw tree is no longer read and can
be archived. Cleaned text is cached after the first parse, and ids
encode as uint8, so the 15M-word corpus stays light.

```
python -m broad.run_broad            # or: train | surrogate | map | finish
```

MASC-slice replication of the pilot (full numbers in
`broad/results/results_broad.md`): the surrogate findings hold on a bigger
model (k=12, hidden 384) and a 20-genre corpus. Median unit near-Gaussian
with a heavy tail, capacity ordering Spearman 0.963, the MaxEnt warp again
produces zero merges (41 fire under empirical correlations), map robustness
Jaccard 0.912, and HOPE's trajectory sits 0.55 nats below L1 at 35% density.
New here, the register-conditional map: calibrating the same weights on each
register separately shows a mostly shared core (median cross-register
capacity Spearman 0.966, top-half core overlap 0.83) with a legible
register-specific fringe; the most register-variable units are core in
movie scripts and slack in travel guides, led by an all-caps/header detector.
Which knowledge is core is, measurably, a property of the register the
surrogate is calibrated on: the DEFT premise, demonstrated.

At full OANC scale (88.5M chars, 8,823 docs, 8 genres; val 1.666 nats), the
core findings replicate a third time: near-Gaussian median with a heavy
tail, zero merges under the MaxEnt warp against 76 empirical, and a 0.43-nat
gap to L1 pruning at 35% density. Two honest scale effects: surrogate
fidelity decays with breadth (capacity Spearman 0.982 pilot, 0.963 MASC,
0.938 OANC; map robustness Jaccard 0.789), and random pruning nearly matches
capacity-ordered pruning at full scale (2.638 vs 2.636 nats at 35%), so at
saturation the ordering's value concentrates in what survives rather than
the raw curve. The map itself flips with the corpus, as it should: newline
units, best-preserved on Shakespeare, are all evicted on OANC, and the
register poles are conversation vs technical prose, with the most
register-variable units reading as transcript-formatting detectors on one
side and biomedical-text detectors on the other.

## GPT-2: the map with GELU

`src/minihope/gelu_kernels.py` lifts the PH-1 restriction: every kernel is
Gauss-Hermite quadrature under the same calibration-defined surrogate,
verified against Monte Carlo (GELU) and the closed forms (ReLU) in tests.
`gpt2map/run_gpt2_map.py` (torch + transformers; resumable parts
prep/calib/map/registers/finish, `--smoke` first) encodes chosen GPT-2 MLP
blocks with writeback and NLL checkpoints on held-out OANC, in paper-warp
and empirical-correlation modes, plus per-block baselines and a
register-by-depth capacity analysis. The pointed hypothesis, tying back to
`results_gpt2.md`: if the surrogate misleads where its shape assumption
fails, paper-vs-empirical map divergence should track the depth profile of
excess kurtosis (worst in blocks 1-3 and 12).

Ran on blocks 2, 6, and 12 (OANC calibration and eval), the hypothesis
resolved through a different channel than posed. Correlations never mattered:
zero merges under both models in every block (removed-set Jaccard 1.000
three times; the paper, empirical, and capacity-prune-only trajectories are
superimposed), because a 3072-unit block's reservoir of near-dead units
keeps the cheapest prune below any merge's distortion down to 35% density.
The fiction's failure surfaced in the capacity pricing instead: in block 2,
where median excess kurtosis peaks (+3.09), the data-free ordering is
anti-guidance, driving NLL from 3.56 to ~5.15 before a non-monotone partial
recovery, while random deletion of the same fractions is nearly free; block
6 slightly trails random; block 12's ordering beats every baseline. And the
register-conditional map gains a depth axis: cross-register capacity
Spearman declines near-monotonically from ~0.90 (blocks 1-3) to 0.48-0.55
(blocks 10-12), a profile decoupled from the kurtosis curve
(`fig_g2_register_depth.png`): register-universal foundations,
register-specific depths.
