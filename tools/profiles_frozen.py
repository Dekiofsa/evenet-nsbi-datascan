"""Likelihood profiles on toy 0 (the poster's toy) for the frozen-backbone model at every fraction, scratch overlaid."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = Path(__file__).resolve().parents[1]   # the scan folder (repository root)
sys.path.insert(0, str(DIR))
import nsbi_scan as ns

cfg = ns.ScanConfig.load(str(DIR)).with_root(str(DIR))
xs = ns.load_cross_sections(cfg)
runs = ns.collect_runs(cfg)
mu = cfg.mu_grid
TAGS = {2.5: "mu2p5", 0.4: "mu0p4"}
COL = {"scratch": "#1f77b4", "frozen": "#2ca02c"}
LAB = {"scratch": "scratch", "frozen": "frozen backbone"}
TOY = 0

def run(variant, task, frac):
    r = runs[(runs.variant == variant) & (runs.task == task) & np.isclose(runs.frac, frac) & (runs.seed == 0)]
    assert len(r) == 1, (variant, task, frac); return r.iloc[0]

def profile(variant, frac, tag, sel):
    a, b = run(variant, "sig_over_bkg", frac), run(variant, "sbi_over_bkg", frac)
    rs = np.load(Path(a.run_dir) / f"r_toys_{tag}.npy").astype(np.float64)[sel] * float(a.C_ref)
    rb = np.load(Path(b.run_dir) / f"r_toys_{tag}.npy").astype(np.float64)[sel] * float(b.C_ref)
    return rs, rb

fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=True)
for row, (mu_true, tag) in enumerate(TAGS.items()):
    toys = ns.load_toys(cfg, tag, with_kin=False)
    sel = toys["toy_id"] == TOY
    n = toys["n"][sel]
    for col, frac in enumerate(cfg.fractions):
        ax = axes[row, col]
        for variant in ("scratch", "frozen"):
            rs, rb = profile(variant, frac, tag, sel)
            t, _, _ = ns.nll_profile(rs, rb, n, xs, mu, cfg.lumi)
            f = ns.fit_mu(mu, t, mu_true)
            iv = "(open interval)" if not (np.isfinite(f["lo"]) and np.isfinite(f["hi"])) else f"[{f['lo']:.2f}, {f['hi']:.2f}]"
            ax.plot(mu, t, color=COL[variant], lw=2, label=f"{LAB[variant]}: mu_hat={f['mu_hat']:.2f} {iv}")
        ax.axhline(1, color="0.4", ls="--", lw=0.9); ax.axhline(4, color="0.4", ls=":", lw=0.9)
        ax.axvline(mu_true, color="0.3", ls=":", lw=1.2)
        ax.set_ylim(0, 12); ax.set_xlim(0, 4); ax.grid(alpha=0.3)
        ax.set_title(f"{int(round(frac*100))}% of training data, mu_true = {mu_true}, toy {TOY}", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper center")
        if col == 0: ax.set_ylabel("-2 log lambda")
        if row == 1: ax.set_xlabel("signal strength mu")
fig.suptitle("Likelihood profiles for one toy: frozen backbone vs scratch at every training-set size (dashed: 1 sigma, dotted: 2 sigma, vertical: truth)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = DIR / "figures" / "nll_profiles_frozen_one_toy.png"
fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
print("wrote", out)
