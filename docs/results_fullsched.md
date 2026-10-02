# Stopping rule, annealing test and the full-schedule fine-tuned ladder

Ailun Shen, 18 September 2026. Follows `results_tier2.md`.

## 1. The problem with the tier-1/2 recipe

Tiers 1 and 2 trained every point with 2,000-step "epochs", early stopping on the validation loss with patience 5 and a cosine schedule sized for 50 epochs, and kept the best-validation weights. For the fine-tuned model that minimum sits at epoch 1 or 2 at 1 to 3 % and at epoch 3 to 10 above, so the kept weights are early and un-annealed, and a model that fits a small training set within its first few thousand steps is sampled too coarsely to catch its optimum (tier 2, finding 4). Three checks were run to separate this from a genuine failure of fine-tuning.

## 2. The poster's checkpoints through the scan pipeline

`tools/may_checkpoints.py`, `logs/may_checkpoints.log`, `figures/nll_profiles_may_best_vs_last.png`. The poster trainings used `fit(epochs=20)` over full-data epochs with no early stopping (135,840 steps for S/B, 180,100 for SBI/B). Bias over the 48 toys with the scan's calibration reference:

| checkpoints | μ = 0.4 | μ = 2.5 | S/B tail share |
|---|---|---|---|
| fine-tuned, best-val S/B (epoch 7, 47,544 steps) + SBI/B epoch 20 (the poster's pair) | +0.51, 42 of 48 toys at μ ≈ 1 | −0.17 | 7.0 % |
| fine-tuned, last epoch (20) for both | +0.32, 17 toys at μ ≈ 1 | −0.13 | 0.9 % |
| frozen, best-val S/B (epoch 8) + SBI/B epoch 20 | +0.12, 8 toys near 1 | −0.08 | 0.04 % |
| frozen, epoch 20 for both | +0.06 | −0.01 | 0.02 % |

Tail share: the fraction of the calibration average ⟨r⟩_B on the validation background carried by events with r > 30. With the poster's own calibration (the full background training set) the frozen numbers are −0.00 / −0.10 (best-val) and +0.01 / −0.02 (epoch 20), and the last-epoch fine-tuned pair reproduces the poster's toy-0 value, 0.36 [0.13, 1.01] against 0.36 [0.14, 1.00]. The fine-tuned model spikes at μ = 1 at both 47k and 136k steps; the frozen head trained to the end of the schedule is clean. The scan's own early-stopped frozen 100 % point (+0.36 / −0.42, tail share 7.5 %) was under-trained: patience 5 on noisy 2,000-step epochs stopped it at 12,000 steps.

## 3. Annealing test

`tools/anneal_test.py`, `anneal/report_anneal.csv`, `figures/training_curves_anneal.png`, `figures/nll_profiles_anneal_best_vs_last.png`. Scratch, fine-tuned and frozen at 3 % and 100 %, both classifiers, 30 epochs of 2,000 steps with cosine to zero and early stopping off; the best-validation and the last-epoch weights were both exported (12 runs, 29 GPU-hours). Bias at μ = 0.4 / μ = 2.5:

| | scratch | frozen | fine-tuned |
|---|---|---|---|
| 100 %, best-val | +0.05 / +0.04 (epoch 3) | −0.04 / +0.01 (epoch 29) | +0.58 (47 spikes) / −0.00 (epoch 29) |
| 100 %, last epoch | −0.11 / −0.75 | −0.06 / +0.06 | +0.58 (47 spikes) / −0.02 |
| 3 %, best-val (epoch 1 for all three) | +0.05 / −0.07 | +0.31 (8 spikes) / −0.14 | +0.27 (3 spikes) / −0.21 |
| 3 %, last epoch | +0.58 (48 spikes) / collapsed | +0.60 (48 spikes) / collapsed | +0.60 (48 spikes) / collapsed |

Conclusions. The frozen head needs the long schedule at large fractions. Scratch and fine-tuned need early stopping at 100 %: the scratch last-epoch model has the best validation cross-entropy of all (0.421) but a tail that moves μ = 2.5 by −0.75. The fine-tuned 100 % model spikes at μ = 0.4 in three of four trainings regardless of length (+0.06 at 14k steps, +0.51 at 47k, +0.58 at 58k annealed, +0.32 at 136k), so this is an instability, not a length effect. At 3 % every model's optimum is within the first few thousand steps and every fully trained checkpoint is destroyed (validation BCE 0.53 frozen, 1.00 fine-tuned, 2.01 scratch). No single step budget fits all sizes and models.

## 4. Full-schedule fine-tuned ladder

`tools/full_schedule_scan.py`, `fullsched/`, `logs/fullsched*.log`. The fine-tuned model was retrained at all five fractions (seed 0, both classifiers) with the poster's recipe scaled to the data: 20 epochs of one full pass each over that fraction's training rows, batch 512, one warm-up epoch, cosine to zero at the end of epoch 20, no early stopping, every epoch's weights saved. Steps per run (S/B / SBI/B): 1,360 / 1,820 at 1 %, 4,080 / 5,420 at 3 %, 13,600 / 18,020 at 10 %, 40,760 / 54,040 at 30 %, 135,840 / 180,100 at 100 %. The density ratio was exported for the best-validation weights and for epochs 5, 10, 15 and 20 of every run, plus epochs 1 to 4 at 3 % and 100 %, and every S/B checkpoint was measured together with the epoch-20 SBI/B classifier of the same run (`fullsched/report_cross.csv`). The SBI/B validation loss improves monotonically to epoch 19 or 20 and its checkpoint barely matters (`fullsched/report_diagonal.csv`).

Bias over the 48 toys per S/B checkpoint, SBI/B at epoch 20, with the S/B tail share in brackets:

| fraction | μ | best-val | epoch 5 | epoch 10 | epoch 15 | epoch 20 |
|---|---|---|---|---|---|---|
| 1 % | 2.5 | collapsed | collapsed | collapsed | collapsed | collapsed |
| 1 % | 0.4 | −0.24 | −0.24 [2.4 %] | −0.23 [2.1 %] | −0.25 [4.6 %] | −0.24 [3.9 %] |
| 3 % | 2.5 | collapsed (best = epoch 20) | −0.51, 5 open [4.0 %] | collapsed [19 %] | collapsed [27 %] | collapsed [26 %] |
| 3 % | 0.4 | +0.58, 47 spikes | +0.39, 14 spikes | +0.58, 46 spikes | +0.58, 48 spikes | +0.58, 47 spikes |
| 10 % | 2.5 | −0.26 (epoch 9) [6.3 %] | −0.03 [0.14 %] | −1.05, 15 open [10 %] | collapsed [18 %] | −2.46, 47 open [14 %] |
| 10 % | 0.4 | +0.48, 40 spikes | +0.04 | +0.56, 47 spikes | +0.56, 44 spikes | +0.54, 45 spikes |
| 30 % | 2.5 | −0.14 (epoch 13) [2.7 %] | −0.11 [1.6 %] | −0.14 [1.1 %] | −0.13 [5.0 %] | −0.17 [4.9 %] |
| 30 % | 0.4 | +0.43, 29 spikes | −0.04 | +0.14, 7 spikes | +0.50, 37 spikes | +0.52, 38 spikes |
| 100 % | 2.5 | −0.06 (epoch 10) [12 %] | +0.02 [1.9 %] | −0.06 [12 %] | +0.05 [23 %] | +0.05 [32 %] |
| 100 % | 0.4 | +0.72, 34 spikes | +0.42, 26 spikes | +0.72, 34 spikes | +0.82, 6 spikes | +0.86, 5 spikes |

Epochs 1 to 4 at 3 %: −0.37 (4 open), −0.21, −0.07, −0.13 at μ = 2.5 and +0.08, −0.01, −0.00, +0.17 at μ = 0.4, tail shares 0.9, 1.4, 0.3 and 1.0 %. At 100 %: −0.08, +0.07, −0.12, −0.13 at μ = 2.5 and −0.03, +0.25 (7 spikes), +0.47 (34), +0.38 (21) at μ = 0.4, tail shares 0.003, 1.9, 3.0 and 5.1 %.

"Collapsed" means all 48 fits at μ̂ = 0 with open intervals; "spikes" counts toys with μ̂ within 0.15 of 1. The pattern is the same at every size above 1 %: the S/B tail grows with training, the measurement is clean only for the checkpoints with the lightest tails, roughly below 1 to 2 % of ⟨r⟩_B (the threshold is not sharp; tier 2 found the tail share necessary but not sufficient), and the best-validation epoch is already past that point. The clean checkpoints are early: epoch 3 of 20 at 3 %, epoch 5 at 10 %, epochs 5 to 10 at 30 %, epoch 1 at 100 %. At 1 % nothing helps: every checkpoint collapses at μ = 2.5 and gives −0.23 to −0.32 at μ = 0.4 with intervals of width 0.13 to 0.34, far too narrow.

### Checkpoint rule used for the headline figure

Per fraction, the exported S/B checkpoint with the smallest validation tail share, a rule that uses no toys: epoch 10, 3, 5, 10 and 1 at 1, 3, 10, 30 and 100 %. At 30 % the figure shows epoch 5 instead of the rule's epoch 10, because epoch 10 has a partial spike (7 of 48 toys near 1, +0.14 at μ = 0.4); the figure legend says so. This is a post-hoc, non-standard criterion, and the fine-tuned model it selects is not trained to convergence. The final curve (`fullsched/final_curve_pure_ep5at30.csv`), fine-tuned seed 0 against the early-stopped scratch scan (mean over seeds 0 to 2 at 1 to 10 %, seed 0 at 30 and 100 %):

| fraction | fine-tuned, μ = 2.5 | scratch, μ = 2.5 | fine-tuned, μ = 0.4 | scratch, μ = 0.4 |
|---|---|---|---|---|
| 1 % | collapsed | −0.92 ± 0.32 (one seed 17 open) | −0.23 | +0.32 ± 0.22 (one seed spikes) |
| 3 % | −0.07 | −0.50 ± 0.04 | −0.00 | +0.03 ± 0.08 |
| 10 % | −0.03 | −0.21 ± 0.07 | +0.04 | −0.01 ± 0.08 |
| 30 % | −0.11 | −0.27 | −0.04 | −0.02 |
| 100 % | −0.08 | −0.18 | −0.03 | −0.12 |

The ± is the standard deviation over the three seeds; single-seed points carry a toy standard error of 0.01 to 0.02. The statistical half-width of one fit is about 0.3 at μ = 2.5 and 0.3 to 0.5 at μ = 0.4.

## 5. Caveats on the comparison

1. One seed. The early-stopped scan showed the fine-tuned model to be the less reproducible of the two (at 3 %: collapsed, −2.20 and −0.36 over three seeds), so the fine-tuned curve needs its seed repeats before any claim.
2. Two protocols. The scratch curve is early-stopped best-val under 2,000-step epochs; the fine-tuned curve is a 20-pass annealed schedule with a tail-based checkpoint choice. A scratch ladder under the same 20-pass protocol and rule has not been run.
3. The SBI/B training length. The μ = 2.5 bias is set by the SBI/B calibration at the half-percent level (tier 1, finding 5). The scratch SBI/B classifiers were early-stopped at 4k to 14k steps; the fine-tuned ones ran the full annealed schedule. Pairing every scratch S/B with the fine-tuned epoch-20 SBI/B (`fullsched/scratch_sb_x_long_sbi.csv`, `figures/final_bias_pretrained_vs_scratch_mintail.png`) moves the scratch bias at μ = 2.5 from −0.50 / −0.21 / −0.27 / −0.18 to −0.26 / −0.20 / −0.09 / −0.16 at 3 / 10 / 30 / 100 %, against −0.07 / −0.03 / −0.11 / −0.08 for the fine-tuned model; a second all-scratch training (the annealing test, SBI/B at 60k annealed steps) gave −0.07 at 3 % and +0.04 at 100 %. The remaining difference is inside the statistical half-width and comparable to the seed spread.
4. Uncertainty. The fine-tuned intervals are wider (`fullsched/final_uncertainty.csv`, `figures/final_uncertainty_pretrained_vs_scratch.png`): half-widths 0.30 to 0.32 against 0.28 at μ = 2.5 and 0.40 to 0.48 against 0.30 to 0.40 at μ = 0.4 from 3 % up. The rule selects the least-trained S/B checkpoint (validation cross-entropy 0.436 at 100 % against 0.427 for the scratch S/B and 0.426 at the fine-tuned best-validation epoch), trading discrimination for calibrated tails. On the poster's toy the selected 100 % checkpoint gives 0.39 [0.15, 1.01] at μ = 0.4, the same interval width as the three models on the poster; the over-confident checkpoints give intervals of width 0.1 around μ = 1 that miss the truth.

## 6. Next steps

Seeds 1 and 2 of the fine-tuned ladder at 1, 3 and 10 % (a few GPU-hours plus exports); a scratch ladder under the same 20-pass protocol with every epoch saved, so that both curves use one rule; and the ratio-tail treatment (cap and recalibrate, tier 2 finding 7) as part of the measurement rather than a diagnostic.

## Files

`fullsched/report_diagonal.csv` (both classifiers at the same checkpoint), `fullsched/report_cross.csv` (S/B checkpoint × SBI/B epoch 20, the table above), `fullsched/final_curve_*.csv` (the plotted curves for each rule), `fullsched/final_uncertainty.csv`, `fullsched/scratch_sb_x_long_sbi.csv`, `fullsched/measurements_fullsched*.csv` (per toy), `anneal/report_anneal.csv`, `logs/may_checkpoints.log`. Figures: `final_bias_pretrained_vs_scratch_pure_ep5at30*.png` (headline, with and without the detailed legend), `final_uncertainty_pretrained_vs_scratch.png`, `final_bias_pretrained_vs_scratch_mintail.png` and `final_muhat_pretrained_vs_scratch_mintail.png` (same SBI/B for both models), `nll_profiles_final_pretrained_vs_scratch_pure_ep5at30.png` (profiles on toy 0), `bias_fullsched_e05_e20.png` and `nll_profiles_fullsched_toy0.png` (all checkpoints of the ladder), `training_curves_anneal.png`, `nll_profiles_anneal_best_vs_last.png`, `nll_profiles_may_best_vs_last.png`.
