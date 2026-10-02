"""The final curve as the mean fitted mu over the 48 toys; reads fullsched/final_curve_<rule>.csv."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = Path(__file__).resolve().parents[1]   # the scan folder (repository root)
C = {"pretrained": "#d62728", "scratch S/B + same SBI/B": "#1f77b4"}
LAB = {"pretrained": "pretrained (fine-tuned), seed 0",
       "scratch S/B + same SBI/B": "scratch, mean over seeds (3 seeds at 1/3/10 %, 1 seed at 30/100 %)"}
HALF = {2.5: 1.0, 0.4: 0.4}                       # y range = mu_true +- HALF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="mintail")
    args = ap.parse_args()
    t = pd.read_csv(DIR / "fullsched" / f"final_curve_{args.rule}.csv")
    t["muhat"] = t.bias + t.mu_true
    fracs = sorted(t.frac.unique())
    xev = dict(zip(t.frac, t.train_events_sig))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8.6), sharex=True)
    for ax, mu in zip(axes, (2.5, 0.4)):
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        s = t[(t.model == "scratch S/B + same SBI/B") & (t.mu_true == mu)].sort_values("frac")
        hw = 0.5 * s.mean_width.fillna(s.mean_width.mean())
        ax.fill_between(s.train_events_sig, mu - hw, mu + hw, color="0.9", label=r"$\mu_{\rm true} \pm 1\sigma$ stat. of one fit")
        ax.axhline(mu, color="k", lw=1.2, ls="--", label=rf"truth, $\mu_{{\rm true}}$ = {mu}")
        lo, hi = mu - HALF[mu], mu + HALF[mu]
        for k, (model, color) in enumerate(C.items()):
            g = t[(t.model == model) & (t.mu_true == mu)].sort_values("frac")
            on = g[g.muhat >= lo]
            ax.errorbar(on.train_events_sig, on.muhat, yerr=on.err, fmt="o-", color=color, lw=2.3, ms=8.5, capsize=4,
                        label=LAB[model], zorder=5 + k)
            for _, r in g[g.muhat < lo].iterrows():            # all fits pinned at 0: off scale
                x = r.train_events_sig * (0.93 if k == 0 else 1.07)
                ax.annotate("", xy=(x, lo + 0.01 * (hi - lo)), xytext=(x, lo + 0.13 * (hi - lo)),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
            if (g.muhat < lo).any() and k == 0:
                x0 = g[g.muhat < lo].train_events_sig.iloc[0]
                ax.text(x0 * 1.18, lo + 0.06 * (hi - lo), r"both models: all fits pinned at $\hat\mu$ = 0", fontsize=8.5, va="center")
        ax.set_ylim(lo, hi)
        ax.set_ylabel(r"mean fitted $\hat\mu$ over 48 toys")
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    top = axes[0].secondary_xaxis("top")
    top.xaxis.set_major_locator(FixedLocator([xev[f] for f in fracs]))
    top.xaxis.set_major_formatter(FixedFormatter([f"{int(round(f * 100))}%" for f in fracs]))
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel("fraction of the training split")
    axes[0].legend(fontsize=8.5, loc="upper right")
    axes[1].set_xlabel("training events (signal hypothesis)")
    fig.suptitle(r"Fitted signal strength vs training statistics: pretrained vs scratch (48 toys, 300 fb$^{-1}$)", fontsize=12)
    fig.text(0.5, 0.005, "each model with its best recipe, both paired with the same fully trained SBI/B classifier; "
             "error bars: seed spread (3 seeds) or toy s.e. (1 seed)", ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = DIR / "figures" / f"final_muhat_pretrained_vs_scratch_{args.rule}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(t[t.model.isin(C)][["model", "frac", "mu_true", "muhat", "err", "n_seeds"]].round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
