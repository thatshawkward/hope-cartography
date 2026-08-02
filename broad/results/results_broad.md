# Broad-corpus run: OANC

Corpus: 88,544,779 chars, 8823 docs, 8 genres; model k=12, d=24, hidden=384; val loss 1.6658 nats/char.

## Surrogate replication (this corpus vs. Shakespeare pilot)
- excess kurtosis med/p90: +0.008/+0.496 (pilot -0.008/+0.447); max +7.3; control med -0.001
- self-kernel err med/p90: 0.167/0.453 (pilot 0.076)
- cross-kernel err, full pipeline / oracle rho: 0.245 / 0.319 (pilot 0.094 / 0.123)
- capacity ordering Spearman: 0.9380 (pilot 0.9822); correlation-model Spearman 0.820

## Trajectory (val loss at 50% / 35% density)
- HOPE full (paper rho): 2.3455 / 2.6355
- HOPE full (empirical rho): 2.3205 / 2.5981
- HOPE prune-only: 2.3455 / 2.6355
- L1-norm prune: 2.7158 / 3.0639
- random prune: 2.3850 / 2.6379

## The map
- merges: 0 under the MaxEnt warp, 76 under empirical rho (pilot: 0 / 17)
- robustness: removed-set Jaccard 0.789, order Spearman 0.899 (pilot 0.898 / 0.974)
- survival by label: last=vowel 0.53, last=space 0.35, distributed 0.30, last=punct 0.00, next=vowel 0.00, last=newline 0.00
- |excess kurtosis|: removed med 0.169 vs surviving 0.205

## The register-conditional map (7 registers)
- capacity Spearman across registers: median 0.9787, min 0.8551
- top-half core overlap (Jaccard): median 0.901, min 0.745
- most register-variable units (max/min capacity ratio):
- unit 243 [last=space] x2.3: core in conversation, slack in non-fiction; top context "o\n um\n \n \n y"
- unit 186 [distributed] x2.1: core in conversation, slack in non-fiction; top context "\n \n \n \n so h"
- unit 128 [distributed] x1.9: core in technical, slack in conversation; top context "ribe\ncompoun"
- unit 181 [distributed] x1.8: core in technical, slack in non-fiction; top context " \n \n \n \n \n \n"
- unit 11 [distributed] x1.8: core in technical, slack in conversation; top context "trol HPB-ALL"

Total runtime 68s.
