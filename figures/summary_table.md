# Training-statistics scan: summary

Scan root `/home/ubuntu/datascan`. x = training events of the signal hypothesis (N_full = 1,213,905). Toys: 48 weighted MC subsets per mu_true at 300 fb^-1. sigma_stat = mean 1-sigma half-width; the +/- is the std over seeds, or the toy standard error for single-seed points.

| mu_true | variant | fraction | training events | seeds | bias +/- seed std | mean 1-sigma width | bias / sigma_stat |
|---|---|---|---|---|---|---|---|
| 2.5 | frozen | 1% | 12,139 | 1 | -1.597 +/- 0.138 (toy SE) | 0.528 | -6.04 |
| 2.5 | frozen | 3% | 36,417 | 1 | -0.188 +/- 0.011 (toy SE) | 0.656 | -0.57 |
| 2.5 | frozen | 10% | 121,390 | 1 | -0.293 +/- 0.011 (toy SE) | 0.574 | -1.02 |
| 2.5 | frozen | 30% | 364,172 | 1 | -0.325 +/- 0.011 (toy SE) | 0.571 | -1.14 |
| 2.5 | frozen | 100% | 1,213,905 | 1 | -0.421 +/- 0.011 (toy SE) | 0.564 | -1.49 |
| 2.5 | poster | 100% | 1,213,905 | 1 | -0.166 +/- 0.011 (toy SE) | 0.573 | -0.58 |
| 2.5 | pretrained | 1% | 12,139 | 3 | -1.961 +/- 0.606 | 0.524 | -6.42 |
| 2.5 | pretrained | 3% | 36,417 | 3 | -1.686 +/- 1.157 | 0.546 | -5.21 |
| 2.5 | pretrained | 10% | 121,390 | 3 | -1.484 +/- 0.910 | 0.561 | -3.47 |
| 2.5 | pretrained | 30% | 364,172 | 1 | -0.224 +/- 0.018 (toy SE) | 0.479 | -0.94 |
| 2.5 | pretrained | 100% | 1,213,905 | 1 | -0.173 +/- 0.011 (toy SE) | 0.582 | -0.60 |
| 2.5 | scratch | 1% | 12,139 | 3 | -0.923 +/- 0.322 | 0.520 | -3.54 |
| 2.5 | scratch | 3% | 36,417 | 3 | -0.502 +/- 0.044 | 0.557 | -1.80 |
| 2.5 | scratch | 10% | 121,390 | 3 | -0.213 +/- 0.070 | 0.561 | -0.76 |
| 2.5 | scratch | 30% | 364,172 | 1 | -0.275 +/- 0.011 (toy SE) | 0.581 | -0.95 |
| 2.5 | scratch | 100% | 1,213,905 | 1 | -0.178 +/- 0.011 (toy SE) | 0.561 | -0.63 |
| 0.4 | frozen | 1% | 12,139 | 1 | +0.407 +/- 0.012 (toy SE) | 0.311 | +2.62 |
| 0.4 | frozen | 3% | 36,417 | 1 | +0.270 +/- 0.019 (toy SE) | 0.595 | +0.91 |
| 0.4 | frozen | 10% | 121,390 | 1 | +0.414 +/- 0.024 (toy SE) | 0.432 | +1.92 |
| 0.4 | frozen | 30% | 364,172 | 1 | +0.402 +/- 0.017 (toy SE) | 0.461 | +1.75 |
| 0.4 | frozen | 100% | 1,213,905 | 1 | +0.359 +/- 0.007 (toy SE) | 0.366 | +1.96 |
| 0.4 | poster | 100% | 1,213,905 | 1 | +0.508 +/- 0.017 (toy SE) | 0.405 | +2.51 |
| 0.4 | pretrained | 1% | 12,139 | 3 | +0.542 +/- 0.025 | 0.198 | +5.57 |
| 0.4 | pretrained | 3% | 36,417 | 3 | +0.479 +/- 0.197 | 0.237 | +5.70 |
| 0.4 | pretrained | 10% | 121,390 | 3 | +0.143 +/- 0.489 | 0.161 | -5.35 |
| 0.4 | pretrained | 30% | 364,172 | 1 | +0.552 +/- 0.021 (toy SE) | 0.333 | +3.31 |
| 0.4 | pretrained | 100% | 1,213,905 | 1 | +0.056 +/- 0.028 (toy SE) | 0.808 | +0.14 |
| 0.4 | scratch | 1% | 12,139 | 3 | +0.315 +/- 0.216 | 0.567 | +2.13 |
| 0.4 | scratch | 3% | 36,417 | 3 | +0.031 +/- 0.076 | 0.591 | +0.08 |
| 0.4 | scratch | 10% | 121,390 | 3 | -0.011 +/- 0.077 | 0.798 | -0.04 |
| 0.4 | scratch | 30% | 364,172 | 1 | -0.018 +/- 0.008 (toy SE) | 0.621 | -0.06 |
| 0.4 | scratch | 100% | 1,213,905 | 1 | -0.119 +/- 0.006 (toy SE) | 0.723 | -0.33 |

## Decision rule

Separation requires |bias_scratch - bias_pretrained| > sqrt(err_s^2 + err_p^2) and |bias_scratch| > 0.25 sigma_stat while |bias_pretrained| <= 0.25 sigma_stat (sigma_stat from the pretrained variant). Points marked 'open interval' have no sigma_stat and cannot separate.

- mu_true = 2.5: no separation seen down to 1% (12,139 training events).
- mu_true = 0.4: pretraining helps already at the largest compared fraction (100% = 1,213,905 training events).
