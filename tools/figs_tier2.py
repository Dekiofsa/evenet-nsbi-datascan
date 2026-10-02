"""Tier-2 figures: bias_per_seed.png and bias_clipped_vs_raw.png."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = Path(__file__).resolve().parents[1]   # the scan folder (repository root)
sys.path.insert(0, str(DIR))
import nsbi_scan as ns  # noqa: E402

cfg = ns.ScanConfig.load(str(DIR)).with_root(str(DIR))
FIG = DIR / "figures"
DIAG = DIR / "diagnostics"
COL = {"scratch": "#1f77b4", "pretrained": "#ff7f0e", "frozen": "#2ca02c"}
LAB = {"scratch": "scratch (random init)", "pretrained": "pretrained (all layers fine-tuned)", "frozen": "pretrained (frozen backbone)"}
MK = {0: "o", 1: "s", 2: "^"}
X = {f: ns.training_events_for(cfg, f) for f in cfg.fractions}
PIN = -2.5


def frac_axis(ax, top_labels=True):
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    if top_labels:
        top = ax.secondary_xaxis("top")
        top.xaxis.set_major_locator(FixedLocator([X[f] for f in cfg.fractions]))
        top.xaxis.set_major_formatter(FixedFormatter([f"{int(round(f * 100))}%" for f in cfg.fractions]))
        top.xaxis.set_minor_locator(NullLocator())
        top.set_xlabel("fraction of the training split")


ps = pd.read_csv(DIAG / "per_seed_diagonal.csv")
scratch_w = ps[(ps.variant == "scratch")].groupby(["mu_true", "frac"]).mean_width.mean()

# ------------------------------------------------------------------ figure 1: per seed
fig, axes = plt.subplots(2, 1, figsize=(9.5, 9), sharex=True)
for ax, mu in zip(axes, (2.5, 0.4)):
    frac_axis(ax, top_labels=(ax is axes[0]))
    hw = np.array([0.5 * scratch_w.loc[(mu, f)] for f in cfg.fractions])
    xs_ = np.array([X[f] for f in cfg.fractions])
    ax.fill_between(xs_, -hw, hw, color="0.85", label=r"$\pm\sigma_{\rm stat}$ (mean 1$\sigma$ half-width, scratch)")
    ax.axhline(0, color="k", lw=0.8)
    for variant in ("scratch", "pretrained", "frozen"):
        g = ps[(ps.variant == variant) & (ps.mu_true == mu)]
        means = g.groupby("frac").bias.mean()
        ax.plot([X[f] for f in means.index], means.values, "-", color=COL[variant], lw=1.2, alpha=0.6)
        for _, r in g.iterrows():
            open_ = r.n_open > 0
            y = r.bias
            ax.plot(X[r.frac] * (1 + 0.06 * (r.seed - 1)), y, MK[int(r.seed)], color=COL[variant], ms=8,
                    mfc="white" if open_ else COL[variant], mew=1.6, alpha=0.95)
        ax.plot([], [], MK[0], color=COL[variant], label=LAB[variant])
    ax.set_ylabel(r"bias  $\langle\hat\mu\rangle - \mu_{\rm true}$")
    ax.text(0.02, 0.05, rf"$\mu_{{\rm true}} = {mu}$", transform=ax.transAxes, fontsize=12)
axes[0].set_ylim(-2.75, 0.45)
axes[1].set_ylim(-0.55, 0.75)
for s, m in MK.items():
    axes[0].plot([], [], m, color="0.4", mfc="0.4", label=f"seed {s}")
axes[0].plot([], [], "o", color="0.4", mfc="white", mew=1.6, label="hollow: some 1$\\sigma$ intervals open at the scan edge")
axes[0].legend(fontsize=8.5, loc="lower right", ncol=2)
axes[1].set_xlabel("training events (signal hypothesis)")
fig.suptitle("NSBI signal-strength bias vs training statistics, every seed shown (48 weighted toys, 300 fb$^{-1}$)", fontsize=12)
fig.text(0.5, 0.005, "points at -2.5 are fits pinned at $\\hat\\mu$ = 0 with all 48 intervals open; seeds are offset horizontally for visibility",
         ha="center", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(FIG / "bias_per_seed.png", dpi=150)
fig.savefig(FIG / "bias_per_seed.pdf")
print("wrote", FIG / "bias_per_seed.png")

# ------------------------------------------------------------------ figure 2: clipped vs raw
if not (DIAG / "clipping_per_seed.csv").exists():
    print("clipping_per_seed.csv not there yet; skipping figure 2")
    sys.exit(0)
cl = pd.read_csv(DIAG / "clipping_per_seed.csv")
cl = cl[cl["mode"] == "clip_recal"]
fig, axes = plt.subplots(2, 1, figsize=(9.5, 9), sharex=True)
for ax, mu in zip(axes, (2.5, 0.4)):
    frac_axis(ax, top_labels=(ax is axes[0]))
    hw = np.array([0.5 * scratch_w.loc[(mu, f)] for f in cfg.fractions])
    xs_ = np.array([X[f] for f in cfg.fractions])
    ax.fill_between(xs_, -hw, hw, color="0.85", label=r"$\pm\sigma_{\rm stat}$ (scratch)")
    ax.axhline(0, color="k", lw=0.8)
    for variant in ("scratch", "pretrained", "frozen"):
        raw = ps[(ps.variant == variant) & (ps.mu_true == mu)].groupby("frac").agg(bias=("bias", "mean"), sd=("bias", "std"), n=("seed", "size"), open_=("n_open", "max"))
        cli = cl[(cl.variant == variant) & (cl.mu_true == mu)].groupby("frac").agg(bias=("bias", "mean"), sd=("bias", "std"), n=("seed", "size"), open_=("n_open", "max"))
        xr = [X[f] for f in raw.index]
        ax.plot(xr, raw.bias, "--", color=COL[variant], lw=1.0, alpha=0.5)
        ax.plot(xr, raw.bias, "o", color=COL[variant], mfc="white", mew=1.4, ms=7, alpha=0.7, label=f"{LAB[variant]}, raw")
        xc = [X[f] for f in cli.index]
        ax.errorbar(xc, cli.bias, yerr=cli.sd.fillna(0), fmt="o-", color=COL[variant], ms=7, lw=1.6, capsize=3,
                    label=f"{LAB[variant]}, S/B clipped at 30 + recalibrated")
    ax.set_ylabel(r"bias  $\langle\hat\mu\rangle - \mu_{\rm true}$")
    ax.text(0.02, 0.05, rf"$\mu_{{\rm true}} = {mu}$", transform=ax.transAxes, fontsize=12)
axes[0].set_ylim(-2.75, 0.45)
axes[1].set_ylim(-0.55, 0.75)
axes[0].legend(fontsize=8, loc="lower right", ncol=1)
axes[1].set_xlabel("training events (signal hypothesis)")
fig.suptitle("Effect of clipping the S/B ratio tail (r > 30) and recalibrating: mean over seeds, error bar = seed spread", fontsize=11.5)
fig.text(0.5, 0.005, "hollow: raw result with the fixed reference calibration; filled: calibrated S/B ratio capped at 30 (SBI/B at 3) and renormalised on the validation background",
         ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(FIG / "bias_clipped_vs_raw.png", dpi=150)
fig.savefig(FIG / "bias_clipped_vs_raw.pdf")
print("wrote", FIG / "bias_clipped_vs_raw.png")
