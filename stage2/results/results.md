# Stage 2 results: the Gaussian surrogate vs. text

Model: char-MLP, context 8, embed 16, hidden 256 (ReLU); 25000 Adam steps on tiny Shakespeare (1,115,394 chars, vocab 65).
Final val loss 1.8311 nats/char (uniform 4.174, unigram 3.309).
Calibration batch 8,192 contexts; evaluation slice 120,000 contexts.

## Experiment A: shape of pre-activations
- excess kurtosis: median -0.008, 90th pct +0.447, max +3.082 (Gaussian control: median -0.002, max +0.052)
- |skewness|: median 0.145, max 1.287 (control median 0.005)
- KS distance to normal: median 0.0212 (control median 0.0057)

## Experiment B: kernel fidelity (moments matched by calibration)
- self-kernel K(i,i) relative error: median 0.0759, 90th pct 0.1945, max 0.4173
- self-kernel error with oracle (eval-slice) moments, i.e. pure shape error: median 0.0720, 90th pct 0.2001, max 0.4140
- cross-kernel scaled error, (a) full pipeline: median 0.0939, 90th pct 0.1848
- cross-kernel scaled error, (b) oracle correlation: median 0.1227, 90th pct 0.2152
- correlation model: Spearman(rho_hat, empirical rho) = 0.8257
- capacity ordering: Spearman(closed, empirical) = 0.982248

Total runtime 37s.
