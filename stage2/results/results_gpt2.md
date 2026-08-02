# Stage 2, Experiment A rerun: GPT-2 on tiny Shakespeare

Model: pretrained GPT-2 small (gpt2: 12 blocks, d_model 768, d_ff 3072, GELU). Pre-activations are each block's mlp.c_fc output -- the input to the GELU, the analog of the char-MLP's x @ W1 + b1.
Corpus: tiny Shakespeare, 1,115,394 chars -> 338,025 GPT-2 BPE tokens (vocab 50257).
Sanity: 4.1583 nats/token (ppl 64.0) on the eval slice.
Eval slice 120,832 positions (118 windows x 1024); calibration slice 8,192 positions, disjoint.
Control per block: iid Gaussian inputs with the calibration slice's MLP-input mean/covariance through the same c_fc weights (pre-activations exactly Gaussian).

## Experiment A, pooled over all 12 x 3072 units
- excess kurtosis: median +0.429, 90th pct +2.707, max +776.6 (Gaussian control: median +0.001, max +0.065)
- |skewness|: median 0.297, max 26.1 (control median 0.005)
- KS distance to normal: median 0.0319 (control median 0.0056)

## By block
| block | exkurt med | exkurt p90 | exkurt max | skew med (abs) | KS med | KS med (ctrl) |
|---|---|---|---|---|---|---|
| 1 | +1.147 | +6.598 | +148.9 | 0.848 | 0.0818 | 0.0053 |
| 2 | +2.445 | +9.312 | +533.0 | 0.999 | 0.0842 | 0.0055 |
| 3 | +0.989 | +4.548 | +776.6 | 0.535 | 0.0557 | 0.0055 |
| 4 | +0.346 | +1.695 | +414.9 | 0.325 | 0.0375 | 0.0060 |
| 5 | +0.234 | +1.222 | +250.9 | 0.261 | 0.0292 | 0.0059 |
| 6 | +0.237 | +1.277 | +37.4 | 0.267 | 0.0267 | 0.0057 |
| 7 | +0.220 | +1.357 | +115.5 | 0.237 | 0.0245 | 0.0056 |
| 8 | +0.231 | +1.175 | +83.1 | 0.211 | 0.0223 | 0.0056 |
| 9 | +0.243 | +1.043 | +12.0 | 0.188 | 0.0215 | 0.0056 |
| 10 | +0.310 | +1.071 | +15.1 | 0.188 | 0.0219 | 0.0060 |
| 11 | +0.426 | +1.225 | +98.3 | 0.204 | 0.0238 | 0.0055 |
| 12 | +0.727 | +1.749 | +28.6 | 0.270 | 0.0303 | 0.0056 |

Total runtime 160s (torch 2.8.0, transformers 4.57.6, device mps).
