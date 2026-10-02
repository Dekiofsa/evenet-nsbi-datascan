"""Pair every scratch S/B classifier with a fully trained SBI/B classifier (fine-tuned epoch 20 of the full
schedule, or the annealed scratch one) and measure the 48 toys. Writes fullsched/scratch_sb_x_long_sbi*.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1]   # the scan folder (repository root)
sys.path.insert(0, str(DIR))
import nsbi_scan as ns  # noqa: E402

cfg = ns.ScanConfig.load(str(DIR)).with_root(str(DIR))
xs = ns.load_cross_sections(cfg)
toys = {ns.MU_TAGS[m]: ns.load_toys(cfg, ns.MU_TAGS[m], with_kin=False) for m in cfg.mu_true_values}
main = ns.collect_runs(cfg)
full = ns.collect_runs(cfg.with_root(str(DIR / "fullsched")))
ann = ns.collect_runs(cfg.with_root(str(DIR / "anneal")))

sb_runs = main[(main.variant == "scratch") & (main.task == "sig_over_bkg")]
partners = {
    "ft_e20": full[(full.variant == "pretrained_e20") & (full.task == "sbi_over_bkg") & (full.seed == 0)],
    "scratch_long": ann[(ann.variant == "scratch") & (ann.task == "sbi_over_bkg") & (ann.seed == 0)],
}
out = []
for pname, P in partners.items():
    for _, a in sb_runs.iterrows():
        b = P[np.isclose(P.frac, a.frac)]
        if len(b) != 1:
            continue
        b = b.iloc[0]
        for mu_true in cfg.mu_true_values:
            tag = ns.MU_TAGS[mu_true]
            r_a = np.load(Path(a.run_dir) / f"r_toys_{tag}.npy")
            r_b = np.load(Path(b.run_dir) / f"r_toys_{tag}.npy")
            df = ns.measure_toys(r_a, r_b, toys[tag], xs, cfg, float(a.C_ref), float(b.C_ref))
            df.insert(0, "partner", pname)
            df.insert(1, "seed", int(a.seed))
            df.insert(2, "frac", float(a.frac))
            df["sbi_best_epoch"] = int(b.best_epoch) if "best_epoch" in b and pd.notna(b.best_epoch) else -1
            df["sbi_total_steps"] = int(b.total_steps) if "total_steps" in b and pd.notna(b.total_steps) else -1
            out.append(df)
            print(f"{pname:12s} scratch S/B seed {int(a.seed)} f={a.frac:<5} mu={mu_true}: "
                  f"bias {df.mu_hat.mean() - mu_true:+.3f}  open {int(df.width.isna().sum()):2d}  "
                  f"spike {int(((df.mu_hat > 0.85) & (df.mu_hat < 1.15)).sum()):2d}", flush=True)
toy = pd.concat(out, ignore_index=True)
toy.to_csv(DIR / "fullsched" / "scratch_sb_x_long_sbi_per_toy.csv", index=False)
g = toy.groupby(["partner", "seed", "frac", "mu_true"])
agg = g.agg(n_toys=("mu_hat", "size"), mean_muhat=("mu_hat", "mean"), median_muhat=("mu_hat", "median"),
            std_muhat=("mu_hat", "std"), mean_width=("width", "mean"), n_open=("width", lambda s: int(s.isna().sum())),
            n_spike=("mu_hat", lambda s: int(((s > 0.85) & (s < 1.15)).sum())),
            sbi_best_epoch=("sbi_best_epoch", "first"), sbi_total_steps=("sbi_total_steps", "first")).reset_index()
agg["bias"] = agg.mean_muhat - agg.mu_true
agg["bias_se_toys"] = agg.std_muhat / np.sqrt(agg.n_toys)
agg.to_csv(DIR / "fullsched" / "scratch_sb_x_long_sbi.csv", index=False)
pd.set_option("display.width", 200)
print(agg[["partner", "seed", "frac", "mu_true", "bias", "bias_se_toys", "median_muhat", "mean_width", "n_open", "n_spike",
           "sbi_best_epoch", "sbi_total_steps"]].round(3).to_string(index=False))
