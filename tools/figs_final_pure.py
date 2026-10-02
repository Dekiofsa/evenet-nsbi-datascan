"""Bias vs training events with each model's own classifiers: fine-tuned (S/B at the smallest-tail checkpoint,
SBI/B at epoch 20) vs the early-stopped scratch scan.

    python tools/figs_final_pure.py [0.3=e05] [simple] [nodiamonds]
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figs_final_pretrained_vs_scratch as F  # noqa: E402  (helpers only; its main() is not called)

OVERRIDE = sys.argv[1] if len(sys.argv) > 1 else "none"
SIMPLE = "simple" in sys.argv[2:]
NO_AN = "nodiamonds" in sys.argv[2:]                                   # omit the second all-scratch training
ckmap = F.checkpoint_map("mintail")
for item in [x for x in OVERRIDE.split(",") if x.strip() and x != "none"]:
    k, v = item.split("=")
    ckmap[float(k)] = v.strip()
print("pretrained S/B checkpoint per fraction:", ckmap)
pre = F.pretrained_points(ckmap)
es_seeds = pd.read_csv(F.DIR / "diagnostics" / "per_seed_diagonal.csv")
es_seeds = es_seeds[es_seeds.variant == "scratch"]
es = F.seed_aggregate(es_seeds, "scratch")
an = pd.read_csv(F.DIR / "anneal" / "report_anneal.csv")
an = an[(an.model == "scratch") & (an.checkpoint == "best")]          # all-scratch, nothing mixed

if SIMPLE:
    L_PRE = "pretrained (fine-tuned), 1 seed"
    L_SCR = "scratch, mean of 3 seeds at 1/3/10 %, 1 seed at 30/100 %"
    L_AN = "scratch, a second independent training"
    L_BAND = r"$\pm 1\sigma$ statistical uncertainty of one fit"
else:
    L_PRE = ("pretrained, fine-tuned (its own S/B and SBI/B, 20-pass schedule), seed 0: S/B at the smallest-validation-tail checkpoint,"
             "\nSBI/B at epoch 20"
             + ("  (30 %: S/B epoch 5 shown; the rule's pick, epoch 10, gives +0.14 / -0.14)" if OVERRIDE != "none" else ""))
    L_SCR = ("scratch (its own S/B and SBI/B, early-stopped best-val): mean over seeds 0/1/2 (small markers) at 1/3/10 %, "
             "seed 0 at 30/100 %")
    L_AN = ("scratch, a second all-scratch training (annealing test, own S/B and SBI/B, best-val, seed 0):\n"
            "3 %: same S/B, its SBI/B trained 60k annealed steps instead of 8k;  100 %: a plain retraining")
    L_BAND = r"$\pm\sigma_{\rm stat}$: half of the mean 1$\sigma$ width of the scratch fits"

fig, axes = plt.subplots(2, 1, figsize=(9.5, 9.4 if SIMPLE else 10.6), sharex=True)
for ax, mu in zip(axes, (2.5, 0.4)):
    F.frac_axis(ax, top_labels=(ax is axes[0]))
    s = es[es.mu_true == mu].sort_values("frac")
    band = ax.fill_between([F.X[f] for f in s.frac], -0.5 * s.mean_width, 0.5 * s.mean_width, color="0.9", label=L_BAND)
    ax.axhline(0, color="k", lw=0.8)
    pos_s = {(r.frac, r.mu_true): r.bias for _, r in es.iterrows()}
    pos_p = {(r.frac, r.mu_true): r.bias for _, r in pre.iterrows()}
    h2 = F.draw(ax, es, es_seeds, mu, F.C_S, 2.3, 8.5, 1.0, "-", 5, L_SCR, False if SIMPLE else pos_p)
    h1 = F.draw(ax, pre, None, mu, F.C_P, 2.4, 9, 1.0, "-", 7, L_PRE, False if SIMPLE else pos_s)
    a = an[an.mu_true == mu]
    h3 = None
    if not NO_AN:
        h3, = ax.plot([F.X[f] * 1.09 for f in a.frac_sig], a.bias, "D", color=F.C_S, ms=8, mfc="white", mew=1.8, zorder=6,
                      ls="none", label=L_AN)
    ax.set_ylabel(r"bias  $\langle\hat\mu\rangle - \mu_{\rm true}$")
    ax.text(0.98, 0.05, rf"$\mu_{{\rm true}} = {mu}$", transform=ax.transAxes, fontsize=12, ha="right")
axes[0].set_ylim(-2.75, 0.6)
axes[1].set_ylim(-0.6, 0.8)
axes[1].set_xlabel("training events (signal hypothesis)")
fig.suptitle("NSBI signal-strength bias vs training statistics: fine-tuned pretrained EveNet vs scratch\n"
             "48 weighted toys, 300 fb$^{-1}$", fontsize=11.5)
HANDLES = [h for h in (h1, h2, h3, band) if h is not None]
if SIMPLE:
    axes[0].legend(handles=HANDLES, loc="center", bbox_to_anchor=(0.6, 0.33), fontsize=9.5, frameon=True)
    fig.text(0.5, 0.006, "error bars: spread over seeds (3 seeds) or statistical error of the 48-toy mean (1 seed).  "
             "Hollow marker: some fits failed; the point at -2.5: all 48 fits collapsed to $\\hat\\mu$ = 0.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
else:
    fig.legend(handles=HANDLES, loc="lower center", ncol=1, fontsize=7.4, frameon=True, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.006, "error bars: seed spread where 3 seeds exist, otherwise toy s.e. only (excludes run-to-run spread, about 0.07 at 10 %).\n"
             "Hollow: some 1$\\sigma$ intervals open; points at -2.5: all 48 fits pinned at $\\hat\\mu$ = 0.",
             ha="center", fontsize=7.8, color="0.35")
    fig.tight_layout(rect=(0, 0.135, 1, 0.97))
SUFFIX = "" if OVERRIDE == "none" else "_ep5at30"
out = F.FIG / f"final_bias_pretrained_vs_scratch_pure{SUFFIX}{'_simple' if SIMPLE else ''}{'_nodiamonds' if NO_AN else ''}.png"
fig.savefig(out, dpi=150)
fig.savefig(out.with_suffix(".pdf"))
cols = ["model", "frac", "mu_true", "sb_checkpoint", "bias", "err", "err_kind", "n_seeds", "mean_width", "n_open", "n_spike"]
table = pd.concat([pre[cols], es[cols]], ignore_index=True).sort_values(["mu_true", "model", "frac"])
table["train_events_sig"] = table.frac.map(F.X)
table.to_csv(F.FS / f"final_curve_pure{SUFFIX}.csv", index=False)
print("wrote", out)
