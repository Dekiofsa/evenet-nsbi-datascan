# Results

Bias of the fitted signal strength, mean(μ̂) − μ_true over 48 toys at 300 fb⁻¹, for EveNet-Lite trained from scratch and the pretrained EveNet fine-tuned, against the fraction of the training split used for the two CARL classifiers. Setup and definitions are in the README; every table here comes from the CSV files listed at the end.

## 1. Early-stopped scan (11 to 13 September)

Both classifiers of every point were trained with 2,000-step epochs, at most 50, early stopping with patience 5 on a fixed validation subset, and the best-validation weights kept. Seeds 0 to 2 at 1, 3 and 10 %, seed 0 at 30 and 100 %; the frozen-backbone control at seed 0. 54 runs, 65 GPU-hours on an L4.

Bias per seed. "collapsed": all 48 fits at μ̂ = 0 with open intervals. "spike": most toys within 0.15 of μ̂ = 1, where the S/B term drops out of the likelihood. Toy standard error 0.01 to 0.02; statistical half-width of one fit about 0.28 at μ = 2.5 and 0.3 to 0.5 at μ = 0.4.

| μ_true | model | 1 % | 3 % | 10 % | 30 % | 100 % |
|---|---|---|---|---|---|---|
| 2.5 | scratch | −1.29 (17 open), −0.74, −0.74 | −0.55, −0.48, −0.47 | −0.20, −0.15, −0.29 | −0.27 | −0.18 |
| 2.5 | pretrained | −2.08 (37 open), collapsed, −1.31 (16 open) | collapsed, −2.20 (40 open), −0.36 | collapsed, −1.21 (20 open), −0.74 (3 open) | −0.22 | −0.17 |
| 2.5 | frozen | −1.60 (25 open) | −0.19 | −0.29 | −0.32 | −0.42 |
| 0.4 | scratch | +0.20, +0.56 (spike), +0.18 | −0.04, +0.11, +0.02 | −0.10, +0.03, +0.04 | −0.02 | −0.12 |
| 0.4 | pretrained | +0.53, +0.57, +0.52 (all spike) | +0.59 (spike), +0.59 (spike), +0.25 | −0.39 (25 open), +0.24, +0.58 (spike) | +0.55 (spike) | +0.06 |
| 0.4 | frozen | +0.41 | +0.27 | +0.41 | +0.40 | +0.36 |

Figures: `figures/bias_per_seed.png`, `figures/paired_bias_difference.png`, `figures/companions.png`, `figures/nll_profiles_three_models_one_toy.png`, `figures/nll_profiles_frozen_one_toy.png`, `figures/training_curves_tier1.png`.

- Scratch is reproducible from 3 % up: seed spread 0.04 to 0.07 at μ = 2.5, all six points at μ = 0.4 within 0.12 of zero. At 1 % it is at the edge, with one seed open at μ = 2.5 and one seed spiking at μ = 0.4.
- The fine-tuned model fails in 8 of 9 seed-fraction points at 1 to 10 % and still spikes at 30 % for μ = 0.4. Its S/B density ratio has heavy tails: events with r > 30 carry 3 to 28 % of the calibration average ⟨r⟩_B at 1 to 3 % (scratch 0 to 20 %), with maximum calibrated ratios on toy events up to 1,900 against at most 180 for scratch. The discrimination is the same (AUC 0.85 to 0.88 for both). For 0 < μ < 1 the S/B term enters with a negative sign, so those events make the mixture density negative; the clamp turns that into a wall and the fit retreats to μ = 0 or μ = 1.
- Pairing classifiers across models shows the S/B classifier is the culprit in every seed; the fine-tuned SBI/B is neither better nor worse than the scratch one.
- The μ = 2.5 bias is set by the SBI/B calibration: scaling the SBI/B ratio of a healthy model by 0.5 % moves μ̂ by 0.2, by 1 % it flips the sign. The residual −0.15 to −0.3 at μ = 2.5 in every healthy configuration is of that size.
- Recalibrating on the validation background instead of the fixed reference moves the scratch results by a few hundredths from 3 % up but the fine-tuned ones by up to 1.7, the signature of a calibration average dominated by a few tail events. Capping the S/B ratio at 30 and recalibrating turns most collapses into ordinary biases but does not make fine-tuning better than scratch (`figures/bias_clipped_vs_raw.png`); capping without recalibration makes things worse.
- The frozen backbone never collapses from 3 % up and never spikes, so the pretrained features transfer. Its +0.3 to +0.4 offset at μ = 0.4 comes from its S/B head, which early stopping left under-trained (best weights at 12,000 steps).
- Pretraining helps the classifier metric only at 1 % (S/B cross-entropy 0.460 against 0.543 for scratch); from 3 % up scratch is the better classifier.

## 2. Stopping rule (15 to 16 September)

The best-validation weights of the scan sit at epoch 1 to 10 of 2,000-step epochs with a cosine schedule sized for 50, so they are early and un-annealed. Two checks.

Poster checkpoints through the scan pipeline (`logs/may_checkpoints.log`, `figures/nll_profiles_may_best_vs_last.png`), bias at μ = 0.4 / 2.5 with the scan's calibration:

| checkpoints (full data, 20 full epochs) | μ = 0.4 | μ = 2.5 |
|---|---|---|
| fine-tuned, best-val S/B (epoch 7) + SBI/B epoch 20 | +0.51, 42 of 48 at μ ≈ 1 | −0.17 |
| fine-tuned, epoch 20 for both | +0.32, 17 at μ ≈ 1 | −0.13 |
| frozen, best-val S/B (epoch 8) + SBI/B epoch 20 | +0.12 | −0.08 |
| frozen, epoch 20 for both | +0.06 | −0.01 |

Annealing test (`tools/anneal_test.py`, `anneal/report_anneal.csv`, `figures/training_curves_anneal.png`, `figures/nll_profiles_anneal_best_vs_last.png`): 30 epochs of 2,000 steps, cosine to zero, early stopping off, best-val and last-epoch weights both measured; bias at μ = 0.4 / 2.5:

| | scratch | frozen | fine-tuned |
|---|---|---|---|
| 100 %, best-val | +0.05 / +0.04 (epoch 3) | −0.04 / +0.01 (epoch 29) | +0.58 (47 spikes) / −0.00 (epoch 29) |
| 100 %, last epoch | −0.11 / −0.75 | −0.06 / +0.06 | +0.58 (47 spikes) / −0.02 |
| 3 %, best-val (epoch 1) | +0.05 / −0.07 | +0.31 / −0.14 | +0.27 / −0.21 |
| 3 %, last epoch | +0.58 (48 spikes) / collapsed | +0.60 (48 spikes) / collapsed | +0.60 (48 spikes) / collapsed |

The frozen head needs the long schedule. Scratch and fine-tuned need early stopping at 100 %. The fine-tuned 100 % model spikes at μ = 0.4 after 47k, 58k and 136k steps alike, so this is an instability, not a length effect. At 3 % every model's optimum lies within the first few thousand steps and every fully trained checkpoint is destroyed. No single step budget fits all sizes.

## 3. Full-schedule fine-tuned ladder (18 September)

The fine-tuned model was retrained at all five fractions with the poster's recipe scaled to the data: 20 epochs of one full pass each, cosine to zero, no early stopping, every epoch saved (seed 0; 1,360 / 1,820 steps at 1 % up to 135,840 / 180,100 at 100 % for S/B / SBI/B). The SBI/B classifier is taken at epoch 20, where its validation loss is lowest. Every saved S/B checkpoint was measured with it; the tail share in brackets is the fraction of ⟨r⟩_B on the validation background carried by events with r > 30. All checkpoints: `figures/bias_fullsched_e05_e20.png`, `figures/nll_profiles_fullsched_toy0.png`.

| fraction | μ | best-val | epoch 5 | epoch 10 | epoch 15 | epoch 20 |
|---|---|---|---|---|---|---|
| 1 % | 2.5 | collapsed | collapsed | collapsed | collapsed | collapsed |
| 1 % | 0.4 | −0.24 | −0.24 [2.4 %] | −0.23 [2.1 %] | −0.25 [4.6 %] | −0.24 [3.9 %] |
| 3 % | 2.5 | collapsed (= epoch 20) | −0.51, 5 open [4.0 %] | collapsed [19 %] | collapsed [27 %] | collapsed [26 %] |
| 3 % | 0.4 | +0.58, 47 spikes | +0.39, 14 spikes | +0.58, 46 spikes | +0.58, 48 spikes | +0.58, 47 spikes |
| 10 % | 2.5 | −0.26 (epoch 9) [6.3 %] | −0.03 [0.14 %] | −1.05, 15 open [10 %] | collapsed [18 %] | −2.46, 47 open [14 %] |
| 10 % | 0.4 | +0.48, 40 spikes | +0.04 | +0.56, 47 spikes | +0.56, 44 spikes | +0.54, 45 spikes |
| 30 % | 2.5 | −0.14 (epoch 13) [2.7 %] | −0.11 [1.6 %] | −0.14 [1.1 %] | −0.13 [5.0 %] | −0.17 [4.9 %] |
| 30 % | 0.4 | +0.43, 29 spikes | −0.04 | +0.14, 7 spikes | +0.50, 37 spikes | +0.52, 38 spikes |
| 100 % | 2.5 | −0.06 (epoch 10) [12 %] | +0.02 [1.9 %] | −0.06 [12 %] | +0.05 [23 %] | +0.05 [32 %] |
| 100 % | 0.4 | +0.72, 34 spikes | +0.42, 26 spikes | +0.72, 34 spikes | +0.82, 6 spikes | +0.86, 5 spikes |

Epochs 1 to 4 were also measured at 3 % (−0.37, −0.21, −0.07, −0.13 at μ = 2.5; +0.08, −0.01, −0.00, +0.17 at μ = 0.4) and at 100 % (−0.08, +0.07, −0.12, −0.13; −0.03, +0.25, +0.47, +0.38). The tail grows with training at every size, the clean checkpoints are early (epoch 3 of 20 at 3 %, epoch 5 at 10 %, epoch 1 at 100 %), and the best-validation epoch is already past them. At 1 % every checkpoint collapses at μ = 2.5.

Checkpoint rule for the final curve: per fraction, the saved S/B epoch with the smallest validation tail share, no toys used: epochs 10, 3, 5, 10 and 1. At 30 % epoch 5 is shown instead of the rule's epoch 10, which has a partial spike (7 of 48 toys, +0.14 at μ = 0.4). Final curve, fine-tuned seed 0 against the early-stopped scratch scan (`fullsched/final_curve_pure_ep5at30.csv`, `figures/final_bias_pretrained_vs_scratch_pure_ep5at30.png`, profiles in `figures/nll_profiles_final_pretrained_vs_scratch_pure_ep5at30.png`); ± is the standard deviation over the three scratch seeds:

| fraction | fine-tuned, μ = 2.5 | scratch, μ = 2.5 | fine-tuned, μ = 0.4 | scratch, μ = 0.4 |
|---|---|---|---|---|
| 1 % | collapsed | −0.92 ± 0.32 | −0.23 | +0.32 ± 0.22 |
| 3 % | −0.07 | −0.50 ± 0.04 | −0.00 | +0.03 ± 0.08 |
| 10 % | −0.03 | −0.21 ± 0.07 | +0.04 | −0.01 ± 0.08 |
| 30 % | −0.11 | −0.27 | −0.04 | −0.02 |
| 100 % | −0.08 | −0.18 | −0.03 | −0.12 |

Caveats:

- One seed for the fine-tuned curve, and the early-stopped scan showed it to be the less reproducible model.
- Two protocols: early-stopped best-val for scratch, a 20-pass annealed schedule with a tail-based checkpoint choice for the fine-tuned model. A scratch ladder under the same protocol has not been run.
- Part of the μ = 2.5 gap is the SBI/B training length (4k to 14k early-stopped steps for scratch, the full schedule for fine-tuned). With the fine-tuned epoch-20 SBI/B for both, the scratch bias at μ = 2.5 becomes −0.26 / −0.20 / −0.09 / −0.16 at 3 / 10 / 30 / 100 % against −0.07 / −0.03 / −0.11 / −0.08 (`figures/final_bias_pretrained_vs_scratch_mintail.png`, `figures/final_muhat_pretrained_vs_scratch_mintail.png`, `fullsched/scratch_sb_x_long_sbi.csv`); a second scratch training with a longer SBI/B schedule gave −0.07 at 3 % and +0.04 at 100 %.
- The fine-tuned intervals are wider: half-widths 0.30 to 0.32 against 0.28 at μ = 2.5, 0.40 to 0.48 against 0.30 to 0.40 at μ = 0.4 (`figures/final_uncertainty_pretrained_vs_scratch.png`). The rule picks early, less sharp S/B classifiers (validation cross-entropy 0.436 against 0.427 for scratch at 100 %).

## Next

Seeds 1 and 2 for the fine-tuned ladder at 1, 3 and 10 %; a scratch ladder under the same 20-pass schedule and checkpoint rule; the tail treatment (cap and recalibrate) as part of the measurement.

## Files

- Early-stopped scan: `measurements.csv` (per toy, diagonal pairs), `aggregates_per_seed.csv`, `aggregates.csv`; `measurements_offdiag.csv`, `aggregates_offdiag.csv` (every S/B × SBI/B pairing); `measurements_cval.csv`, `aggregates_cval.csv` (validation-sample calibration); `diagnostics/` (per-seed table, tail statistics, attribution, cross-variant pairs, clipping, calibration-sample dependence).
- Annealing test: `anneal/report_anneal.csv`.
- Full schedule: `fullsched/report_cross.csv` (the table above), `fullsched/report_diagonal.csv`, `fullsched/final_curve_*.csv`, `fullsched/final_uncertainty.csv`, `fullsched/scratch_sb_x_long_sbi.csv`, `fullsched/measurements_fullsched*.csv` (per toy).
- Per-run training metadata under `runs/`, `anneal/runs/` and `fullsched/runs/`; box logs under `logs/`.
