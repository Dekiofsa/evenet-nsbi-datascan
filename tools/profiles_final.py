"""Likelihood profiles on one toy at every training-set size, scratch vs fine-tuned, both with the same SBI/B
classifier (fine-tuned epoch 20); dashed: scratch as run.
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
main_runs = ns.collect_runs(cfg)
full_runs = ns.collect_runs(cfg.with_root(str(DIR / "fullsched")))
xs = ns.load_cross_sections(cfg)
mu = cfg.mu_grid
TAGS = {2.5: "mu2p5", 0.4: "mu0p4"}
C_S, C_P = "#1f77b4", "#ff7f0e"


def run(runs, variant, task, frac, seed=0):
    r = runs[(runs.variant == variant) & (runs.task == task) & np.isclose(runs.frac, frac) & (runs.seed == seed)]
    assert len(r) == 1, (variant, task, frac, len(r))
    return r.iloc[0]


def r_toys(a, tag, sel):
    return np.load(Path(a.run_dir) / f"r_toys_{tag}.npy").astype(np.float64)[sel] * float(a.C_ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="mintail")
    ap.add_argument("--toy", type=int, default=0)
    ap.add_argument("--override", default="", help="S/B checkpoint overrides, e.g. 0.3=e05,1.0=e01")
    ap.add_argument("--no-asrun", action="store_true", help="omit the dashed scratch-as-run reference curve")
    args = ap.parse_args()
    t = pd.read_csv(DIR / "fullsched" / f"final_curve_{args.rule}.csv")
    ckmap = {float(r.frac): r.sb_checkpoint for _, r in t[t.model == "pretrained"].iterrows()}
    for item in [x for x in args.override.split(",") if x.strip()]:
        k, v = item.split("=")
        ckmap[float(k)] = v.strip()
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=True)
    for row, (mu_true, tag) in enumerate(TAGS.items()):
        toys = ns.load_toys(cfg, tag, with_kin=False)
        sel = toys["toy_id"] == args.toy
        n = toys["n"][sel]
        for col, frac in enumerate(cfg.fractions):
            ax = axes[row, col]
            ck = ckmap[float(frac)]
            sbi_long = run(full_runs, "pretrained_e20", "sbi_over_bkg", frac)
            pre_sb = run(full_runs, "pretrained" if ck == "best" else f"pretrained_{ck}", "sig_over_bkg", frac)
            scr_sb = run(main_runs, "scratch", "sig_over_bkg", frac)
            scr_sbi = run(main_runs, "scratch", "sbi_over_bkg", frac)
            ep = "best-val" if ck == "best" else f"S/B epoch {int(ck[1:])}"
            curves = [
                ("scratch", scr_sb, sbi_long, C_S, "-", 2.2, 1.0),
                (f"pretrained (fine-tuned, {ep})", pre_sb, sbi_long, C_P, "-", 2.2, 1.0),
                ("scratch as run (early-stopped SBI/B)", scr_sb, scr_sbi, C_S, "--", 1.2, 0.55),
            ]
            if args.no_asrun:
                curves = curves[:2]
            for lab, a, b, color, ls, lw, alpha in curves:
                tt, _, _ = ns.nll_profile(r_toys(a, tag, sel), r_toys(b, tag, sel), n, xs, mu, cfg.lumi)
                f = ns.fit_mu(mu, tt, mu_true)
                iv = "(open interval)" if not (np.isfinite(f["lo"]) and np.isfinite(f["hi"])) else f"[{f['lo']:.2f}, {f['hi']:.2f}]"
                ax.plot(mu, tt, color=color, ls=ls, lw=lw, alpha=alpha, label=f"{lab}: mu_hat={f['mu_hat']:.2f} {iv}")
            ax.axhline(1, color="0.4", ls="--", lw=0.9)
            ax.axhline(4, color="0.4", ls=":", lw=0.9)
            ax.axvline(mu_true, color="0.3", ls=":", lw=1.2)
            ax.set_ylim(0, 12)
            ax.set_xlim(0, 4)
            ax.grid(alpha=0.3)
            ax.set_title(f"{int(round(frac * 100))}% of training data, mu_true = {mu_true}, toy {args.toy}", fontsize=11)
            ax.legend(fontsize=7.6, loc="upper center")
            if col == 0:
                ax.set_ylabel("-2 log lambda")
            if row == 1:
                ax.set_xlabel("signal strength mu")
    fig.suptitle("Likelihood profiles for one toy: scratch vs fine-tuned pretrained at every training-set size, best recipe of each, "
                 "both with the same fully trained SBI/B classifier (dashed: 1 sigma, dotted: 2 sigma, vertical: truth)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = DIR / "figures" / ("nll_profiles_final_pretrained_vs_scratch" + ("_alt" if args.override else "")
                             + ("_clean" if args.no_asrun else "") + ".png")
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out)


if __name__ == "__main__":
    main()
