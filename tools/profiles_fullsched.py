"""Likelihood profiles on toy 0 for the full-schedule ladder, one curve per checkpoint; the --rule pairing is drawn bold."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = Path(__file__).resolve().parents[1]   # the scan folder (repository root)
sys.path.insert(0, str(DIR))
import nsbi_scan as ns  # noqa: E402

cfg = ns.ScanConfig.load(str(DIR)).with_root(str(DIR))          # data/ (toys, xs.json) live here
runs = ns.collect_runs(cfg.with_root(str(DIR / "fullsched")))    # exports live under fullsched/runs
xs = ns.load_cross_sections(cfg)
mu = cfg.mu_grid
TAGS = {2.5: "mu2p5", 0.4: "mu0p4"}
CK = ["best", "e05", "e10", "e15", "e20"]
CKCOL = {"best": "#7f7f7f", "e05": "#1b9e77", "e10": "#7570b3", "e15": "#e7298a", "e20": "#d95f02"}
CKLAB = {"best": "best-val", "e05": "epoch 5", "e10": "epoch 10", "e15": "epoch 15", "e20": "epoch 20"}
TOY = 0


def variant_name(ck: str) -> str:
    return "pretrained" if ck == "best" else f"pretrained_{ck}"


def run(variant, task, frac):
    r = runs[(runs.variant == variant) & (runs.task == task) & np.isclose(runs.frac, frac) & (runs.seed == 0)]
    return r.iloc[0] if len(r) == 1 else None


def profile(sb_ck, sbi_ck, frac, tag, sel):
    a, b = run(variant_name(sb_ck), "sig_over_bkg", frac), run(variant_name(sbi_ck), "sbi_over_bkg", frac)
    if a is None or b is None:
        return None
    rs = np.load(Path(a.run_dir) / f"r_toys_{tag}.npy").astype(np.float64)[sel] * float(a.C_ref)
    rb = np.load(Path(b.run_dir) / f"r_toys_{tag}.npy").astype(np.float64)[sel] * float(b.C_ref)
    return rs, rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="e05+e20")
    args = ap.parse_args()
    cross = args.rule.split("+") if "+" in args.rule else None
    fracs = [f for f in cfg.fractions if not runs[np.isclose(runs.frac, f)].empty]
    fig, axes = plt.subplots(2, len(fracs), figsize=(4.4 * len(fracs) + 1, 9), sharey=True, squeeze=False)
    for row, (mu_true, tag) in enumerate(TAGS.items()):
        toys = ns.load_toys(cfg, tag, with_kin=False)
        sel = toys["toy_id"] == TOY
        n = toys["n"][sel]
        for col, frac in enumerate(fracs):
            ax = axes[row, col]
            curves = [(ck, ck, CKCOL[ck], 1.6, 0.85, CKLAB[ck]) for ck in CK]
            if cross:
                curves.append((cross[0], cross[1], "#d62728", 3.0, 1.0, f"S/B {CKLAB[cross[0]]} + SBI/B {CKLAB[cross[1]]}"))
            for sb_ck, sbi_ck, color, lw, alpha, lab in curves:
                p = profile(sb_ck, sbi_ck, frac, tag, sel)
                if p is None:
                    continue
                t, _, _ = ns.nll_profile(p[0], p[1], n, xs, mu, cfg.lumi)
                f = ns.fit_mu(mu, t, mu_true)
                iv = "(open)" if not (np.isfinite(f["lo"]) and np.isfinite(f["hi"])) else f"[{f['lo']:.2f}, {f['hi']:.2f}]"
                ax.plot(mu, t, color=color, lw=lw, alpha=alpha, label=f"{lab}: {f['mu_hat']:.2f} {iv}")
            ax.axhline(1, color="0.4", ls="--", lw=0.9)
            ax.axhline(4, color="0.4", ls=":", lw=0.9)
            ax.axvline(mu_true, color="0.3", ls=":", lw=1.2)
            ax.set_ylim(0, 12)
            ax.set_xlim(0, 4)
            ax.grid(alpha=0.3)
            ax.set_title(f"{int(round(frac * 100))}% of training data, mu_true = {mu_true}, toy {TOY}", fontsize=10.5)
            ax.legend(fontsize=7.5, loc="upper center")
            if col == 0:
                ax.set_ylabel("-2 log lambda")
            if row == 1:
                ax.set_xlabel("signal strength mu")
    fig.suptitle("Fine-tuned pretrained EveNet, full 20-pass schedule: likelihood profiles on toy 0 per checkpoint "
                 "(dashed: 1 sigma, dotted: 2 sigma, vertical: truth)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = DIR / "figures" / "nll_profiles_fullsched_toy0.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out)


if __name__ == "__main__":
    main()
