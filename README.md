# EveNet NSBI training-statistics scan

Signal-strength measurement of gg → (h* →) ZZ → 4ℓ with neural simulation-based inference (NSBI), using EveNet-Lite trained from scratch and the pretrained EveNet fine-tuned, as a function of how much training data the classifiers see: 1, 3, 10, 30 and 100 % of the training split.

## Setup

- MCFM samples for signal, background and the full process (SBI), 300 fb⁻¹.
- Two CARL classifiers per model, S/B and SBI/B. The calibrated density ratios go into the per-event mixture likelihood ratio, NLL(μ) is scanned over μ in [0, 4], and the best fit and 1σ interval are read off for 48 pseudo-experiments at μ_true = 0.4 and 48 at μ_true = 2.5.
- Figure of merit: bias = mean(μ̂) − μ_true over the 48 toys, against the number of training events.
- Models, all the same EveNet-Lite architecture (about 20M parameters):
  scratch (random init, lr 3e-4, early stopping on the validation loss);
  pretrained (the public EveNet checkpoint `Avencast/EveNet`, `checkpoints.20M.a4.last.ckpt`, all layers fine-tuned at lr 1e-4 / 5e-5 / 1e-5 for head / object encoder / body, 20 full passes, S/B checkpoint = the saved epoch with the lightest ratio tail on the validation background);
  frozen (same checkpoint, backbone frozen, head only) as a control.
- Scratch has 3 seeds at 1, 3 and 10 % and 1 seed at 30 and 100 %; the fine-tuned ladder has 1 seed.

## Result

![bias vs training events](figures/final_bias_pretrained_vs_scratch_pure_ep5at30_simple_nodiamonds.png)

- From 3 % of the data up, both models recover the true μ within or close to the statistical uncertainty, at μ = 0.4 and at μ = 2.5.
- At 1 % of the data neither model works.
- At μ = 2.5 pretrained is closer to the truth (scratch sits about 0.2 low from 10 % up, and 0.5 low at 3 %).
- The pretrained failure seen earlier came from training its classifier for too long, not from the pretraining itself. With best-validation or last-epoch checkpoints the fine-tuned S/B ratio develops heavy tails that pin the fits at μ = 1 or μ = 0; taking the saved epoch with the lightest tail fixes it. The frozen-backbone control is unbiased at 100 % once trained to convergence.

Caveats: the fine-tuned curve is one seed, its checkpoint rule was chosen after the fact (the 30 % point shows epoch 5 by hand where the rule picks epoch 10), and part of the gap at μ = 2.5 comes from the fine-tuned SBI/B classifier being trained longer than the early-stopped scratch one. Details and numbers are in the reports:

- `docs/results_tier1.md`: single seed, attribution to the two classifiers, calibration sensitivity (12 Sep 2026)
- `docs/results_tier2.md`: seed repeats and the frozen-backbone control (13 Sep)
- `docs/results_fullsched.md`: annealing test and the full-schedule fine-tuned ladder behind the figure above (18 Sep)
- `docs/poster_nsbi_evenet.pdf`: the poster the study started from (fine-tuned and frozen EveNet vs a CARL MLP on the full data, one toy)

## Layout

- `nsbi_scan.py`: all shared code (config, data cache and subsampling, training and ratio exports, calibration, likelihood scan, toy measurement, aggregation)
- `run_scan.py`, `remote_setup.sh`, `scan_config.json`: driver and setup for a Linux GPU box, scan configuration
- `00_*.ipynb` to `03_*.ipynb`: the notebooks (data cache, training, measurement, plots)
- `tools/`: the later studies (`anneal_test.py`, `full_schedule_scan.py`, `may_checkpoints.py`, `poster_export.py` on the GPU box; figure and diagnostics scripts on a laptop)
- `runs/`, `anneal/runs/`, `fullsched/runs/`: per-run `metrics.json`, `history.json`, `closure.json`, `train_meta.json`
- `measurements*.csv`, `aggregates*.csv`: per-toy fits and aggregates of the early-stopped scan
- `fullsched/`, `anneal/`: tables of the full-schedule ladder and the annealing test
- `diagnostics/`: tier-2 diagnostics (per-seed table, tail statistics, attribution, calibration, clipping)
- `figures/`, `logs/`, `docs/`

Not in the repository: the cached splits and toys (`data/*.npz`, rebuilt by notebook 00), the per-run ratio exports (`r_val.npz`, `r_toys_*.npy`, about 25 MB per run) and the weights. The figure scripts in `tools/figs_*.py` work from the CSV tables here; the profile scripts and `diag_tier2.py` need the ratio exports.

## Running

See `docs/running_the_scan.md`. Training needs torch and [EveNet-Lite](https://github.com/Dekiofsa/EveNet-Lite) (which includes the `nsbi` package); the measurement and figures need numpy, pandas and matplotlib (`requirements.txt`).
