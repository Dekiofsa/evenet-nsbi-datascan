# Training-statistics scan: tier 1 results (single seed)

Ailun Shen, 12 September 2026

## Runs

Twenty trainings on an L4 GPU: two models, EveNet trained from scratch and EveNet fine-tuned from the pretrained checkpoint (all layers, the poster recipe), at five training-set sizes, 1, 3, 10, 30 and 100 % of the training split (12k to 1.21M signal events), each with both CARL classifiers (S/B and SBI/B). Every point uses the same optimizer-step budget (2,000 steps per "epoch", up to 50 epochs) with early stopping on a fixed validation subset; at 1 % one such epoch is about 30 passes over the training data. The validation split, the 48 pseudo-experiments per mu value (300 fb^-1) and the calibration reference are identical for all points. The signal strength is fitted on every toy; the bias is the mean of mu_hat - mu_true over the 48 toys.

Protocol notes: the scratch learning rate is 3e-4 (the authors' default of 1e-3 diverged at 3 %); the toys are weighted MC subsets without Poisson fluctuation, so the toy spread measures MC noise, not coverage; ratios are calibrated with C = 1/<r>_B on a fixed 1M-event background reference.

## Results

Bias of the fitted mu over 48 toys (toy standard error in brackets) and mean 1-sigma interval width. "open" means the interval hits the scan boundary; where only part of the toys are open the count is given and the width is the mean over the closed toys.

| mu_true | fraction | scratch bias | scratch width | pretrained bias | pretrained width |
|---|---|---|---|---|---|
| 2.5 | 1 % | -1.29 (0.13) | 0.54, 17 open | -2.08 (0.11) | 0.54, 37 open |
| 2.5 | 3 % | -0.55 (0.01) | 0.59 | -2.50 | open |
| 2.5 | 10 % | -0.20 (0.01) | 0.55 | -2.50 | open |
| 2.5 | 30 % | -0.27 (0.01) | 0.58 | -0.22 (0.02) | 0.48 |
| 2.5 | 100 % | -0.18 (0.01) | 0.56 | -0.17 (0.01) | 0.58 |
| 0.4 | 1 % | +0.20 (0.01) | 0.78 | +0.53 (0.02) | 0.16 |
| 0.4 | 3 % | -0.04 (0.00) | 0.54 | +0.59 (0.00) | 0.12 |
| 0.4 | 10 % | -0.10 (0.01) | 0.70 | -0.39 (0.00) | 0.03, 25 open |
| 0.4 | 30 % | -0.02 (0.01) | 0.62 | +0.55 (0.02) | 0.33 |
| 0.4 | 100 % | -0.12 (0.01) | 0.72 | +0.06 (0.03) | 0.81 |

Figure: `figures/bias_vs_training_events.png`. The toy standard error measures the MC noise of the toys only; model-to-model differences at one seed should be judged against the interval width (sigma_stat, about 0.3 to 0.4 half-width) until the seed repeats are in.

## Findings

1. At full statistics the two models agree. Bias -0.17 vs -0.18 at mu = 2.5, and +0.06 vs -0.12 at mu = 0.4; the latter two differ by 0.18, which is well inside the interval half-width of about 0.4 but far outside the toy standard error, so whether it is a real difference needs the seed repeats.

2. Below full statistics the fully fine-tuned pretrained model fails outright, while the scratch model degrades smoothly. At mu = 2.5 the pretrained fit runs to mu_hat = 0 with open intervals at 1 to 10 %; at mu = 0.4 it sticks near mu_hat = 1 with unphysically narrow intervals at 1, 3 and 30 %, and runs to 0 at 10 %. Only its 100 % point is healthy at both mu values. The scratch bias at mu = 2.5 goes -1.29, -0.55, -0.20, -0.27, -0.18 from 1 % to 100 %; at 1 % the fits are bimodal (17 of 48 toys at mu_hat = 0), from 3 % on all intervals are closed.

3. Which classifier is responsible, by mixing classifiers across training sizes. Pairing every S/B with every SBI/B of the same model:
   - The pretrained S/B classifier is the culprit at 1 to 30 %: every pairing that uses it collapses (at 30 % only at mu = 0.4). Its 100 % version is fine with every SBI/B partner.
   - The pretrained SBI/B classifier is fine at 1 and 3 %. Paired with the scratch S/B at 3 % it gives +0.02 at mu = 2.5 and -0.02 at mu = 0.4, both consistent with zero. Its 10 % version collapses every pairing, but for a different reason (item 5).
   - For scratch, the S/B classifier is good from 3 % on; at mu = 2.5 the bias is set by the SBI/B classifier (with the full-statistics S/B: -0.39, -0.63, -0.21, -0.35, -0.18 for SBI/B at 1, 3, 10, 30, 100 %), non-monotonically at one seed.

4. The pretrained S/B failure is a calibration failure, not a loss of discrimination. Its validation AUC (0.863 to 0.881) is close to scratch (0.850 to 0.885). Its density ratio has memorised tails: maximum calibrated ratios of 1,200 to 1,900 on toy events at 1 to 30 % against at most 82 for scratch, and events with r > 30 make up 0.1 to 0.2 % of the validation background weight yet carry 14 to 28 % of <r>_B. In the likelihood the S/B term cancels exactly at mu = 0 and mu = 1, which is where the collapsed fits land. Clipping the ratio at 30 and recalibrating turns the collapse into an ordinary bias (mu = 2.5: -0.47, -0.56, -0.28 at 1, 3, 10 %; mu = 0.4: +0.17, +0.31 at 1, 3 %), i.e. the underlying classifier is worse than scratch at low statistics but not catastrophic; the catastrophe is the tail. Closure chi2, calibration-factor gaps and the stopping epoch do not separate the failing from the healthy runs (scratch shows similar values at some points), so they are not diagnostic on their own. A cleaner test is robustness to the calibration sample: re-measuring every point with the calibration factor taken from the validation background instead of the 1M-event reference moves the scratch results by at most 0.03, but moves the pretrained ones by up to 0.35 (30 %, mu = 2.5: -0.22 becomes -0.58; 10 %, mu = 0.4: -0.39 becomes +0.53), because a calibration factor dominated by a handful of tail events changes with the sample. Only the pretrained 100 % point is stable under this change.

5. The fit is extremely sensitive to the calibration of the SBI/B ratio. Scaling the SBI/B ratio of the healthy 100 % models by a constant moves the bias at mu = 2.5 from -0.18 (scale 1) to +0.02 (1.005) and +0.22 (1.01), and to a full collapse to mu_hat = 0 at 0.995; at mu = 0.4 it moves by -0.3 to +0.7. Half a percent decides the result. This explains the pretrained SBI/B failure at 10 %: its reference calibration factor is 0.65 % below the value obtained on the validation background (the 1M-event reference contains an outlier; all other runs agree to 0.02 %), and recalibrating on validation makes that point healthy. It also puts the residual bias in perspective: the -0.17 to -0.27 at mu = 2.5 seen in every healthy configuration corresponds to an SBI/B ratio calibrated about 0.4 % low, and the dependence on the SBI/B training size in item 3 is a dependence of that sub-percent calibration on statistics. The corresponding S/B sensitivity is ten times weaker (5 % scale for the same effect).

6. Pipeline validation against the poster checkpoints. The fine-tuned checkpoints saved in May (S/B epoch 7, SBI/B epoch 20), pushed through the scan's export and measurement with the poster's calibration on the full background training set, reproduce the May notebook numbers on the poster's toy to 0.01: 2.278 [1.996, 2.567] at mu = 2.5 (notebook: 2.28 [2.00, 2.56]) and 0.962 [0.848, 1.195] at mu = 0.4 (notebook: 0.96 [0.85, 1.19]); the calibration factor is 1.028 as printed then. Two consequences: the measurement code is the same measurement as before, so the scan results are not a pipeline artefact; and the mu = 1 spike that breaks the pretrained model at low statistics was already present in the May fine-tuned model at full statistics (bias +0.49 over 48 toys at mu = 0.4, with an interval half the scratch width). The poster's 0.36 at mu = 0.4 came from a later checkpoint that was not kept. The scan's own 100 % pretrained run is milder (bias +0.06, one event with a ratio near 100 on the poster's toy) but shows the same mechanism.

## Conclusions

- The question "where does pretraining start to help" has, at one seed and with the poster fine-tuning recipe, the answer: nowhere below full statistics; full fine-tuning of all 20M parameters produces ratio tails that break the likelihood. Whether the pretrained features themselves are usable at low statistics is a separate question, answered by the frozen-backbone control and a gentler recipe (tier 2 and the next step).
- The interference classifier's calibration at the 0.1 % level, not either classifier's discrimination, limits the measurement at mu = 2.5. This is the mechanism ensembles average away and the natural home of the "bias term" in the ensemble study.
- Practical fixes to carry forward: calibrate on the full validation background (outlier-robust) rather than a subset, treat ratio tails explicitly (clipping or per-event calibration, reported next to the raw result), and quote model comparisons against seed spread rather than toy error.

## Caveats

- Every number is one seed. Seeds 1 and 2 at 1, 3 and 10 % and the frozen ladder are training now (tier 2, 34 runs).
- The automatic decision line of the analysis notebook flags "pretraining helps at 100 %, mu = 0.4"; it fires because the two biases differ by more than the toy error, but both are below a third of the statistical error and single-seed, so it should not be quoted.

## Next steps

1. Tier 2 (running, about a day): three seeds at 1, 3, 10 % for both models; frozen ladder as control.
2. A gentler pretrained recipe at 1 to 10 %: lower head learning rate, validation every 500 steps, an annealing step from the best checkpoint (tier-2 checkpoints are kept so this can resume from them).
3. Calibration study on the existing runs: validation-based and trimmed calibration factors, ratio clipping with recalibration, and the resulting bias, for all 20 runs. No GPU needed.

## Cost

The L4 sustained 8.5 optimizer steps per second for a run alone and 4.3 per run with two drivers sharing it (the Colab T4: 5.3). Runs took 0.5 to 2.9 h of wall-clock each, mostly two at a time; the GPU was busy for 17 h in total for tier 1. The measurement needs no GPU.
