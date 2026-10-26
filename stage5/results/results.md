# Stage 5 results: trajectory fine-tuning on GPT-2

Recovery: 300 AdamW steps at lr 2e-05 (dropout off) at 5 density checkpoints; re-projection is one closed-form c_proj refit per checkpoint, zero gradient steps; end-only gets the matched total SGD budget of 1500 steps.

All numbers here were measured on today's torch/transformers stack, including the re-measured static baselines: the stack has drifted since gpt2map's tracked curves were recorded (uncompressed NLL 3.73 today vs 3.56 tracked on identical windows), so cross-stack comparisons are avoided throughout.

Adapted ceiling (no eviction, same recovery budget and seeds): NLL 3.2293.

## Block 2

| run | nll @ 0.6 | nll @ 0.45 | nll @ 0.35 |
|---|---|---|---|
| HOPE static (no recovery) | 4.9680 | 4.4782 | 4.1455 |
| random static | 3.7532 | 3.7474 | 3.7557 |
| L1 static | 3.8994 | 4.0137 | 4.3565 |
| HOPE + re-projection | 3.7822 | 3.8959 | 3.9001 |
| HOPE + block SGD | 3.3475 | 3.3346 | 3.3296 |
| HOPE + full SGD | 3.2496 | 3.2507 | 3.2561 |
| random + full SGD | 3.2449 | 3.2459 | 3.2523 |
| re-planned + full SGD | 3.2511 | 3.2511 | 3.2541 |
| re-planned + re-projection | 3.7499 | 3.7513 | 3.7609 |
| re-planned + block SGD | 3.3345 | 3.3236 | 3.3222 |
| random + re-projection | 3.7374 | 3.7426 | 3.7522 |
| random + block SGD | 3.3136 | 3.3153 | 3.3189 |
| end-only full SGD | 4.9680 | 4.4782 | 3.2486 |
| end-only re-projection | 4.9680 | 4.4782 | 3.9001 |

Figure: fig_g3_b2_recovery.png

## Block 12

| run | nll @ 0.6 | nll @ 0.45 | nll @ 0.35 |
|---|---|---|---|
| HOPE static (no recovery) | 3.7338 | 3.7474 | 3.7496 |
| random static | 3.8410 | 3.7089 | 3.7014 |
| L1 static | 3.7982 | 3.8007 | 3.8039 |
| HOPE + re-projection | 3.7850 | 3.8030 | 3.8110 |
| HOPE + block SGD | 3.4761 | 3.4828 | 3.4850 |
| HOPE + full SGD | 3.2670 | 3.2753 | 3.2892 |
| random + full SGD | 3.2672 | 3.2819 | 3.3044 |
| re-planned + full SGD | 3.2646 | 3.2797 | 3.2979 |
| re-planned + re-projection | 3.7816 | 3.7989 | 3.8098 |
| re-planned + block SGD | 3.4962 | 3.4922 | 3.4915 |
| random + re-projection | 3.8096 | 3.8428 | 3.8599 |
| random + block SGD | 3.5544 | 3.4766 | 3.4983 |
| end-only full SGD | 3.7338 | 3.7474 | 3.2796 |
| end-only re-projection | 3.7338 | 3.7474 | 3.8110 |

Figure: fig_g4_b12_recovery.png

