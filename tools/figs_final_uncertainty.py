"""Companion of the final bias plot: mean 1-sigma uncertainty vs training events for the same checkpoints;
also prints the fraction of toys whose interval contains mu_true.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figs_final_pretrained_vs_scratch as F  # noqa: E402  (helpers only)

OVERRIDE = sys.argv[1] if len(sys.argv) > 1 else "0.3=e05"           # same checkpoints as the posted bias plot
ckmap = F.checkpoint_map("mintail")
for item in [x for x in OVERRIDE.split(",") if x.strip() and x != "none"]:
    k, v = item.split("=")
    ckmap[float(k)] = v.strip()
print("pretrained S/B checkpoint per fraction:", ckmap)


def per_run(g: pd.DataFrame, mu: float) -> dict:
    hw = 0.5 * g.width.dropna()
    cov = g.covers_1sig.fillna(False).astype(bool)
    return dict(n_toys=len(g), n_open=int(g.width.isna().sum()), sigma=hw.mean() if len(hw) else np.nan,
                sigma_se=hw.std() / np.sqrt(len(hw)) if len(hw) > 1 else np.nan,
                minus=(g.mu_hat - g.lo).mean(), plus=(g.hi - g.mu_hat).mean(), coverage=cov.mean(),
                bias=g.mu_hat.mean() - mu)


rows = []
m = pd.read_csv(F.FS / "measurements_fullsched_cross.csv")
m = m[(m.seed == 0) & np.isclose(m.frac_sig, m.frac_sbi)]
for f, ck in ckmap.items():
    v = "pretrained" if ck == "best" else f"pretrained_{ck}"
    for mu, g in m[(m.variant == v) & np.isclose(m.frac_sig, f)].groupby("mu_true"):
        rows.append(dict(model="pretrained", frac=f, mu_true=mu, seed=0, **per_run(g, mu)))
s = pd.read_csv(F.DIR / "measurements.csv")
s = s[(s.variant == "scratch") & np.isclose(s.frac_sig, s.frac_sbi)]
if "calibration" in s.columns:
    s = s[s.calibration == "C_ref"]
for (f, mu, seed), g in s.groupby(["frac_sig", "mu_true", "seed"]):
    rows.append(dict(model="scratch", frac=float(f), mu_true=mu, seed=int(seed), **per_run(g, mu)))
runs = pd.DataFrame(rows)

agg = []
for (model, f, mu), g in runs.groupby(["model", "frac", "mu_true"]):
    n = len(g)
    agg.append(dict(model=model, frac=f, mu_true=mu, n_seeds=n, sigma=g.sigma.mean(),
                    err=g.sigma.std() if n > 1 else g.sigma_se.iloc[0], err_kind="seed_std" if n > 1 else "toy_se",
                    minus=g.minus.mean(), plus=g.plus.mean(), coverage=g.coverage.mean(),
                    n_open="/".join(str(int(v)) for v in g.sort_values("seed").n_open), any_open=bool((g.n_open > 0).any()),
                    bias=g.bias.mean()))
agg = pd.DataFrame(agg).sort_values(["mu_true", "model", "frac"])
agg["train_events_sig"] = agg.frac.map(F.X)
agg.to_csv(F.FS / "final_uncertainty.csv", index=False)
pd.set_option("display.width", 220)
print(agg.round(3).to_string(index=False))

C = {"pretrained": F.C_P, "scratch": F.C_S}
LAB = {"pretrained": "pretrained (fine-tuned), 1 seed", "scratch": "scratch, mean of 3 seeds at 1/3/10 %, 1 seed at 30/100 %"}
fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.8), sharex=True)
for ax, mu in zip(axes, (2.5, 0.4)):
    F.frac_axis(ax, top_labels=(ax is axes[0]))
    for k, model in enumerate(("scratch", "pretrained")):
        a = agg[(agg.model == model) & (agg.mu_true == mu)].sort_values("frac")
        ok = a[a.sigma.notna()]
        ax.errorbar([F.X[f] for f in ok.frac], ok.sigma, yerr=ok.err, fmt="-", color=C[model], lw=2.3, capsize=4, zorder=5 + k)
        for _, r in ok.iterrows():
            ax.plot(F.X[r.frac], r.sigma, "o", color=C[model], ms=8.5, mfc="white" if r.any_open else C[model], mew=1.6, zorder=6 + k)
        for _, r in a[a.sigma.isna()].iterrows():
            ax.text(F.X[r.frac] * 0.97, 0.08, "pretrained at 1 %: all 48 fits\ncollapsed, no interval", fontsize=8.5,
                    color=C[model], va="center")
        ax.plot([], [], "o-", color=C[model], lw=2.3, ms=8.5, label=LAB[model])
        if model == "scratch":
            sr = runs[(runs.model == "scratch") & (runs.mu_true == mu)]
            for _, r in sr.iterrows():
                if (agg[(agg.model == "scratch") & (agg.mu_true == mu) & (agg.frac == r.frac)].n_seeds.iloc[0] > 1) and np.isfinite(r.sigma):
                    ax.plot(F.X[r.frac] * (1 + 0.05 * (r.seed - 1)), r.sigma, F.MK[int(r.seed)], color=C[model], ms=5,
                            alpha=0.45, mfc="white" if r.n_open > 0 else C[model], mew=1.0, zorder=4)
    ax.set_ylabel(r"mean 1$\sigma$ uncertainty on $\mu$")
    ax.set_ylim(0, 0.6)
    ax.text(0.98, 0.06, rf"$\mu_{{\rm true}} = {mu}$", transform=ax.transAxes, fontsize=12, ha="right")
axes[0].legend(fontsize=9.5, loc="upper right")
axes[1].set_xlabel("training events (signal hypothesis)")
fig.suptitle("NSBI fit uncertainty vs training statistics: fine-tuned pretrained EveNet vs scratch\n"
             "48 weighted toys, 300 fb$^{-1}$", fontsize=11.5)
fig.text(0.5, 0.006, "uncertainty = half-width of the fitted 68 % interval, averaged over the toys with a closed interval.\n"
         "Hollow marker: some intervals are open.  Error bars: spread over seeds (3 seeds) or error of the mean (1 seed).",
         ha="center", fontsize=8, color="0.35")
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
out = F.FIG / "final_uncertainty_pretrained_vs_scratch.png"
fig.savefig(out, dpi=150)
fig.savefig(out.with_suffix(".pdf"))
print("wrote", out)
