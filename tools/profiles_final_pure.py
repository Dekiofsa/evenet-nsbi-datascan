"""Likelihood profiles on one toy at every training-set size, scratch vs fine-tuned, each with its own classifiers."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = Path(__file__).resolve().parents[1]   # the scan folder (repository root)
sys.path.insert(0, str(DIR))
sys.path.insert(0, str(DIR / "tools"))
import nsbi_scan as ns  # noqa: E402
import figs_final_pretrained_vs_scratch as F  # noqa: E402

cfg = ns.ScanConfig.load(str(DIR)).with_root(str(DIR))
main_runs = ns.collect_runs(cfg)
full_runs = ns.collect_runs(cfg.with_root(str(DIR / "fullsched")))
xs = ns.load_cross_sections(cfg)
mu = cfg.mu_grid
TAGS = {2.5: "mu2p5", 0.4: "mu0p4"}


def run(runs, variant, task, frac):
    r = runs[(runs.variant == variant) & (runs.task == task) & np.isclose(runs.frac, frac) & (runs.seed == 0)]
    assert len(r) == 1, (variant, task, frac, len(r))
    return r.iloc[0]


def r_toys(a, tag, sel):
    return np.load(Path(a.run_dir) / f"r_toys_{tag}.npy").astype(np.float64)[sel] * float(a.C_ref)


ap = argparse.ArgumentParser()
ap.add_argument("--toy", type=int, default=0)
ap.add_argument("--override", default="", help="S/B checkpoint overrides of the mintail rule, e.g. 0.3=e05 (default: none, the rule's pick)")
args = ap.parse_args()
ckmap = F.checkpoint_map("mintail")
for item in [x for x in args.override.split(",") if x.strip()]:
    k, v = item.split("=")
    ckmap[float(k)] = v.strip()
print("pretrained S/B checkpoint per fraction:", ckmap)

fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=True)
for row, (mu_true, tag) in enumerate(TAGS.items()):
    toys = ns.load_toys(cfg, tag, with_kin=False)
    sel = toys["toy_id"] == args.toy
    n = toys["n"][sel]
    for col, frac in enumerate(cfg.fractions):
        ax = axes[row, col]
        ck = ckmap[float(frac)]
        curves = [
            ("scratch", run(main_runs, "scratch", "sig_over_bkg", frac), run(main_runs, "scratch", "sbi_over_bkg", frac), "#1f77b4"),
            (f"pretrained, S/B ep {int(ck[1:])}" if ck != "best" else "pretrained, S/B best-val",
             run(full_runs, "pretrained" if ck == "best" else f"pretrained_{ck}", "sig_over_bkg", frac),
             run(full_runs, "pretrained_e20", "sbi_over_bkg", frac), "#d62728"),
        ]
        for lab, a, b, color in curves:
            tt, _, _ = ns.nll_profile(r_toys(a, tag, sel), r_toys(b, tag, sel), n, xs, mu, cfg.lumi)
            f = ns.fit_mu(mu, tt, mu_true)
            iv = "(open)" if not (np.isfinite(f["lo"]) and np.isfinite(f["hi"])) else f"[{f['lo']:.2f}, {f['hi']:.2f}]"
            ax.plot(mu, tt, color=color, lw=2.2, label=f"{lab}: {f['mu_hat']:.2f} {iv}")
        ax.axhline(1, color="0.4", ls="--", lw=0.9)
        ax.axhline(4, color="0.4", ls=":", lw=0.9)
        ax.axvline(mu_true, color="0.3", ls=":", lw=1.2)
        ax.set_ylim(0, 12)
        ax.set_xlim(0, 4)
        ax.grid(alpha=0.3)
        ax.set_title(f"{int(round(frac * 100))}% of training data, mu_true = {mu_true}, toy {args.toy}", fontsize=11)
        ax.legend(fontsize=8, loc="upper center")
        if col == 0:
            ax.set_ylabel("-2 log lambda")
        if row == 1:
            ax.set_xlabel("signal strength mu")
fig.suptitle("Likelihood profiles for one toy (seed 0): scratch (early-stopped best-val) vs fine-tuned pretrained (S/B at the smallest-"
             "validation-tail epoch, SBI/B at epoch 20)\n"
             + ("30 %: S/B epoch 5 shown, the rule's pick (epoch 10) has a kink at mu 0.4;   " if args.override else "")
             + "dashed: 1 sigma, dotted: 2 sigma, vertical: truth", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = DIR / "figures" / ("nll_profiles_final_pretrained_vs_scratch_pure" + ("_ep5at30" if args.override else "") + ".png")
fig.savefig(out, dpi=130)
fig.savefig(out.with_suffix(".pdf"))
print("wrote", out)
