"""Bias vs training events for the full-schedule ladder with one checkpoint rule highlighted;
--rule e05+e20 pairs the S/B epoch-5 export with the SBI/B epoch-20 export.
"""
import argparse
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
FS = DIR / "fullsched"
X = {f: ns.training_events_for(cfg, f) for f in cfg.fractions}
PIN = -2.5
CK = ["best", "e05", "e10", "e15", "e20"]
CKCOL = {"best": "#7f7f7f", "e05": "#1b9e77", "e10": "#7570b3", "e15": "#e7298a", "e20": "#d95f02"}
CKLAB = {"best": "best-val epoch", "e05": "epoch 5", "e10": "epoch 10", "e15": "epoch 15", "e20": "epoch 20 (annealed)"}


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


def rule_frame(rule: str, diag: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    if "+" in rule:
        sb, sbi = rule.split("+", 1)
        if sbi != "e20":
            raise SystemExit("cross rules are only available against the SBI/B e20 export (report_cross.csv)")
        return cross[cross.checkpoint == sb].copy()
    return diag[diag.checkpoint == rule].copy()


def draw_curve(ax, g: pd.DataFrame, color, label, lw, ms, alpha, ls="-", marker="o", zorder=3):
    g = g.sort_values("frac_sig")
    xs_ = [X[f] for f in g.frac_sig]
    ax.plot(xs_, g.bias, ls, color=color, lw=lw, alpha=alpha, zorder=zorder)
    for x, b, n_open in zip(xs_, g.bias, g.n_open):
        ax.plot(x, b, marker, color=color, ms=ms, mfc="white" if n_open > 0 else color, mew=1.4, alpha=alpha, zorder=zorder + 1)
    ax.plot([], [], ls, marker=marker, color=color, lw=lw, ms=ms, alpha=alpha, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="e05+e20")
    ap.add_argument("--no-scratch", action="store_true")
    args = ap.parse_args()
    diag = pd.read_csv(FS / "report_diagonal.csv")
    cross = pd.read_csv(FS / "report_cross.csv")
    es = pd.read_csv(DIR / "diagnostics" / "per_seed_diagonal.csv")
    es = es[es.seed == 0].rename(columns={"frac": "frac_sig"})
    chosen = rule_frame(args.rule, diag, cross)
    rule_lab = (f"S/B {CKLAB[args.rule.split('+')[0]]} + SBI/B epoch 20" if "+" in args.rule else f"both classifiers at {CKLAB[args.rule]}")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 9.5), sharex=True)
    for ax, mu in zip(axes, (2.5, 0.4)):
        frac_axis(ax, top_labels=(ax is axes[0]))
        sw = es[(es.variant == "scratch") & (es.mu_true == mu)].groupby("frac_sig").mean_width.mean()
        fr = [f for f in cfg.fractions if f in sw.index]
        ax.fill_between([X[f] for f in fr], [-0.5 * sw[f] for f in fr], [0.5 * sw[f] for f in fr], color="0.9",
                        label=r"$\pm\sigma_{\rm stat}$ (mean 1$\sigma$ half-width, scratch, section 3)")
        ax.axhline(0, color="k", lw=0.8)
        for ck in CK:                                   # the other checkpoints, faint
            g = diag[(diag.checkpoint == ck) & (diag.mu_true == mu)]
            if ck == args.rule or g.empty:
                continue
            draw_curve(ax, g, CKCOL[ck], f"full schedule, {CKLAB[ck]} (both classifiers)", 1.0, 5, 0.35, marker="s", zorder=2)
        g = chosen[chosen.mu_true == mu]
        if not g.empty:
            draw_curve(ax, g, "#d62728", f"full schedule, chosen rule: {rule_lab}", 2.4, 9, 1.0, zorder=6)
        g = es[(es.variant == "pretrained") & (es.mu_true == mu)]
        draw_curve(ax, g, "0.35", "early-stopped scan (section 3), pretrained seed 0", 1.4, 7, 0.9, ls="--", marker="D", zorder=4)
        if not args.no_scratch:
            g = es[(es.variant == "scratch") & (es.mu_true == mu)]
            draw_curve(ax, g, "#1f77b4", "early-stopped scan (section 3), scratch seed 0 (reference)", 1.2, 6, 0.6, ls=":", marker="^", zorder=3)
        ax.set_ylabel(r"bias  $\langle\hat\mu\rangle - \mu_{\rm true}$")
        ax.text(0.02, 0.05, rf"$\mu_{{\rm true}} = {mu}$", transform=ax.transAxes, fontsize=12)
    axes[0].set_ylim(-2.75, 1.25)
    axes[1].set_ylim(-0.6, 0.8)
    axes[0].legend(fontsize=7.5, loc="upper center", ncol=2)
    axes[1].set_xlabel("training events (signal hypothesis)")
    fig.suptitle("Fine-tuned pretrained EveNet, full 20-pass schedule: bias vs training statistics (48 weighted toys, 300 fb$^{-1}$, seed 0)", fontsize=11.5)
    fig.text(0.5, 0.005, "hollow: some 1$\\sigma$ intervals open at the scan edge; points at -2.5 are fits pinned at $\\hat\\mu$ = 0 with all 48 intervals open",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    tag = args.rule.replace("+", "_")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"bias_fullsched_{tag}.png", dpi=150)
    fig.savefig(FIG / f"bias_fullsched_{tag}.pdf")
    print("wrote", FIG / f"bias_fullsched_{tag}.png")


if __name__ == "__main__":
    main()
