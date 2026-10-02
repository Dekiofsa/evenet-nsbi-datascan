# Running the scan

Notes for the scan folder, called `datascan/` on Drive and on the GPU box. All logic is in `nsbi_scan.py`; the notebooks and `run_scan.py` only call it. Torch and EveNet-Lite are imported lazily, so the measurement and plotting half runs with numpy and pandas only.

## Choices

- The fraction f is a nested prefix of one seeded permutation of the seed-42 80 % training split, drawn per hypothesis. Validation split, toys, cross sections and calibration reference stay at full size.
- Every point gets the same optimizer-step budget: 2,000 steps per "epoch" (1,024,000 rows at batch 512), at most 50 epochs, early stopping with patience 5 on a fixed 262,144-row validation subset, best weights restored.
- scratch: EveNet-Lite with `pretrained=False`, one parameter group over Classification, ObjectEncoder, PET and GlobalEmbedding, lr 3e-4 (the authors' 1e-3 diverged at 3 %), weight decay 1e-2, one warm-up epoch.
- pretrained: `nsbi.train.build_classifier`, the Hugging Face checkpoint `Avencast/EveNet` / `checkpoints.20M.a4.last.ckpt`, all layers trained at lr 1e-4 / 5e-5 / 1e-5 (head / ObjectEncoder / body).
- frozen: `nsbi.train.build_classifier_ssl`, same checkpoint, backbone frozen, head only at lr 1e-3.
- Seed k sets the subsample permutation (1000 + k), the torch initialisation and the batch order.
- Per run: weighted BCE and AUC on the full validation split, calibration factors C_ref (fixed 1M-event bkg_train subset, seed 777), C_sub (own training denominator) and C_val, m4l reweighting-closure chi2/ndf, best epoch and step. Per point: bias, std and mean interval width over the 48 toys, then mean and std over seeds.
- The toys are weighted MC subsets whose weights sum to nu(mu). There is no Poisson fluctuation, so the toy spread is MC noise, not coverage.

## Layout

```
datascan/
  README.md, nsbi_scan.py, run_scan.py, scan_config.json
  00_setup_and_subsample.ipynb, 01_train_scan.ipynb, 02_measure_toys.ipynb, 03_aggregate_and_plot.ipynb
  data/
    {sig,bkg,sbi}_{train,val}.npz  kin (N,16) float32, msq (N,4) float64, wt (N,) float64
    summary.json                   rows / sum_wt / ESS / min_wt per split
    xs.json                        {sig, int, bkg, sbi} in fb from the full SBI sample
    toys_mu0p4.npz, toys_mu2p5.npz kin, n (event-unit weights), toy_id, mu_true
  runs/<variant>/<task>/f<NNNN>/seed<k>/
    history.json     per-epoch train/val loss and global step
    metrics.json     sizes, ESS, best epoch/step, val BCE/AUC, C_ref, C_sub, C_val, closure chi2/ndf, timings
    closure.json     m4l histograms of the reweighting closure
    r_val.npz        r_num, r_den (float32, uncalibrated) on the full validation split
    r_toys_mu0p4.npy, r_toys_mu2p5.npy   float32 r for every toy row, same order as data/toys_<tag>.npz
    done.json | failed.json
    final.pt         only with cfg.keep_checkpoint
  figures/
```

Fraction tags: `f0010` = 1 %, `f0030` = 3 %, `f0100` = 10 %, `f0300` = 30 %, `f1000` = 100 %. Tasks: `sig_over_bkg`, `sbi_over_bkg`. The r exports are uncalibrated; the measurement multiplies by the chosen C.

Inputs expected next to the folder on Drive: the MCFM CSVs in `spliced_21_columns/ggZZ_{sig,bkg,sbi}_spliced.csv` and the toys in `observed_mu0p4/observed_*.csv`, `observed_mu2p5/observed_*.csv`.

## How to run

1. Notebook 00, once: writes `scan_config.json`, caches the splits (a few minutes per CSV), the cross sections and the toys, and checks that each toy's yield is within 1 % of nu(mu). Rerun with `force=True` only if the raw CSVs change.
2. Notebook 01 on a Colab GPU, or `run_scan.py` on a Linux box: trains the run list of a tier, cheap fractions first. Runs with `done.json` are skipped, runs with `failed.json` are retried, interrupted runs restart. `stop_after_seconds` / `--stop-after-hours` stops starting new runs before the session ends.
3. Notebooks 02 and 03: measurement and figures, numpy only. Locally, sync the folder and set `NSBI_SCAN_ROOT` to the synced copy before starting Jupyter; only `scan_config.json`, `data/summary.json`, `data/xs.json`, `data/toys_*.npz` and `runs/**` are needed.

On an AWS box (Deep Learning AMI, torch in `/opt/pytorch`):

```bash
scp -r "Training Statistics Scan" ubuntu@<ip>:~/datascan
bash ~/datascan/remote_setup.sh
tmux new -s scan
source /opt/pytorch/bin/activate && cd ~/datascan
python run_scan.py --status
python run_scan.py --smoke
python -u run_scan.py --tier tier1 2>&1 | tee -a logs/tier1.log
```

Two drivers with disjoint `--variants` can share the GPU. Copy `runs/` back for notebooks 02 and 03.

Cost: a 2,000-step epoch takes about 6 min on a Colab T4 and 4 min on an L4. Tier 1 (20 runs) took 30 GPU-hours on the L4, tier 2 (34 runs) 36, the annealing test 29, the full-schedule ladder about a day including exports.

The later studies live in `tools/`: `anneal_test.py`, `full_schedule_scan.py`, `may_checkpoints.py` and `poster_export.py` run on the box under `~/datascan`; the figure and diagnostics scripts run on a laptop from the repository root.

## Dataset

| Process | Raw rows | Train (80 %, seed 42) | Val | ESS train | sum wt train [fb] |
|---|---|---|---|---|---|
| sig | 1,517,382 | 1,213,905 | 303,477 | 537k | 0.248 |
| bkg | 2,828,996 | 2,263,196 | 565,800 | 1.73M | 4.76 |
| sbi | 2,881,323 | 2,305,058 | 576,265 | 1.18M | 4.53 |

Training events per hypothesis at each fraction:

| f | sig | bkg | sbi |
|---|---|---|---|
| 1 % | 12,139 | 22,632 | 23,051 |
| 3 % | 36,417 | 67,896 | 69,152 |
| 10 % | 121,390 | 226,320 | 230,506 |
| 30 % | 364,172 | 678,959 | 691,517 |
| 100 % | 1,213,905 | 2,263,196 | 2,305,058 |

Cross sections from the full SBI sample: sig 0.3166, int −0.6242, bkg 5.9641, sbi 5.6565 fb. Expected yields at 300 fb⁻¹: nu(0.4) = 1708.8, nu(2.5) = 1730.6 events. Toys: 48 per mu, 58.6k to 63.1k weighted rows each, sum n within 0.1 % of nu(mu_true).

## Notes on EveNet-Lite

- No dots in run directory names: the trainer treats a path with a suffix as a file name, hence `f0010` rather than `f0.01`.
- Nothing seeds torch by default. `set_seeds(seed)` is called before building the model and before `fit`, and the weighted sampler is re-seeded per run. GPU nondeterminism remains.
- `compute_physics_metrics` is switched off; it runs a CPU AUC pass over all validation predictions every epoch. The AUC is computed once from the exported r.
- A failed Hugging Face download only logs a warning and training continues from random init. The smoke test checks the "Loaded N / M layers" line (504 / 535 expected).
- `build_classifier_ssl` loads the same checkpoint as `build_classifier`; the only differences are the frozen backbone and the head-only optimizer group. We call it `frozen`.
- Only modules named in `module_lists` get an optimizer, so the scratch recipe lists all four.
- The DataLoader is not the bottleneck (same steps/s with 0 and 2 workers); inference uses the batched `predict_logits_fast`, checked against `predict()` on 4096 events in every run.
- The best model is restored in memory at the end of `fit`; `final.pt` is written only with `keep_checkpoint`.
- Early stopping monitors a fixed 262,144-row subset; `val_bce_full` and `val_auc_full` use the full split.
- `fit_mu` walks outward from the grid minimum to the first crossing of t = 1 and t = 4, so a disconnected second dip never widens the interval. `lo` / `hi` are NaN when the crossing leaves [0, 4].
- `evenet-core` 0.3.0 pins numpy 1.26.4; install it and EveNet-Lite with `--no-deps`, then the runtime dependencies (see `remote_setup.sh`).
