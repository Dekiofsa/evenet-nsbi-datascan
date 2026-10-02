"""Bias vs training events, fine-tuned vs scratch, both paired with the same SBI/B classifier (fine-tuned
epoch 20). --rule mintail picks the S/B checkpoint with the smallest validation tail share at each size.
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
C_S, C_P = "#1f77b4", "#d62728"
MK = {0: "o", 1: "s", 2: "^"}
SPIKE = lambda s: int(((s > 0.85) & (s < 1.15)).sum())


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


def checkpoint_map(rule: str) -> dict:
    """S/B checkpoint per fraction."""
    rep = pd.read_csv(FS / "report_cross.csv")
    rep = rep[rep.mu_true == rep.mu_true.min()]                     # tail_share is a property of the S/B run
    if rule == "mintail":
        return {float(f): g.sort_values("tail_share").checkpoint.iloc[0] for f, g in rep.groupby("frac_sig")}
    return {float(f): rule for f in rep.frac_sig.unique()}


def pretrained_points(ckmap: dict) -> pd.DataFrame:
    """Recomputed from the per-toy file: S/B checkpoint of ckmap paired with the SBI/B epoch-20 export."""
    m = pd.read_csv(FS / "measurements_fullsched_cross.csv")
    m = m[(m.seed == 0) & np.isclose(m.frac_sig, m.frac_sbi)]
    rows = []
    for f, ck in ckmap.items():
        v = "pretrained" if ck == "best" else f"pretrained_{ck}"
        for mu, g in m[(m.variant == v) & np.isclose(m.frac_sig, f)].groupby("mu_true"):
            closed = g[g.width.notna()]
            rows.append(dict(model="pretrained", frac=f, mu_true=mu, sb_checkpoint=ck, n_seeds=1, n_toys=len(g),
                             bias=g.mu_hat.mean() - mu, err=g.mu_hat.std() / np.sqrt(len(g)), err_kind="toy_se",
                             median_muhat=g.mu_hat.median(), mean_width=g.width.mean(),
                             n_open=str(int(g.width.isna().sum())), n_spike=str(SPIKE(g.mu_hat)),
                             any_open=bool(g.width.isna().any()), max_spike=SPIKE(g.mu_hat),
                             bias_closed_only=(closed.mu_hat.mean() - mu) if len(closed) else np.nan))
    return pd.DataFrame(rows)


def seed_aggregate(ps: pd.DataFrame, model: str) -> pd.DataFrame:
    """ps: per-seed rows with frac, mu_true, seed, bias, bias_se_toys, mean_width, n_open, n_spike."""
    rows = []
    for (f, mu), g in ps.sort_values("seed").groupby(["frac", "mu_true"]):
        n = len(g)
        rows.append(dict(model=model, frac=f, mu_true=mu, sb_checkpoint="early-stopped best-val", n_seeds=n,
                         n_toys=48 * n, bias=g.bias.mean(), err=g.bias.std() if n > 1 else g.bias_se_toys.mean(),
                         err_kind="seed_std" if n > 1 else "toy_se", median_muhat=np.nan, mean_width=g.mean_width.mean(),
                         n_open="/".join(str(int(v)) for v in g.n_open), n_spike="/".join(str(int(v)) for v in g.n_spike),
                         any_open=bool((g.n_open > 0).any()), max_spike=int(g.n_spike.max()), bias_closed_only=np.nan))
    return pd.DataFrame(rows)


def draw(ax, agg, seeds, mu, color, lw, ms, alpha, ls, zorder, label, annotate):
    a = agg[agg.mu_true == mu].sort_values("frac")
    if seeds is not None:
        for _, r in seeds[seeds.mu_true == mu].iterrows():
            if a[a.frac == r.frac].n_seeds.iloc[0] < 2:
                continue                                           # single-seed sizes: the big marker is the seed
            ax.plot(X[r.frac] * (1 + 0.05 * (r.seed - 1)), r.bias, MK[int(r.seed)], color=color, ms=5, alpha=0.45 * alpha,
                    mfc="white" if r.n_open > 0 else color, mew=1.0, zorder=zorder - 1)
    ax.errorbar([X[f] for f in a.frac], a.bias, yerr=a.err, fmt=ls, color=color, lw=lw, capsize=4, alpha=alpha, zorder=zorder)
    for _, r in a.iterrows():
        ax.plot(X[r.frac], r.bias, "o", color=color, ms=ms, mfc="white" if r.any_open else color, mew=1.6, alpha=alpha,
                zorder=zorder + 1)
        if not annotate:
            continue
        notes = []
        if r.model == "pretrained":
            notes.append(f"S/B {'best-val' if r.sb_checkpoint == 'best' else 'ep ' + str(int(r.sb_checkpoint[1:]))}")
        if r.model == "pretrained":
            if mu == 0.4 and r.max_spike >= 5:
                notes.append(f"{r.n_spike}/48 within 0.15 of 1")
            if r.any_open and r.bias > -2.49:
                notes.append(f"{r.n_open}/48 pinned at 0")
        elif mu == 0.4 and r.max_spike >= 10:
            notes.append(f"one seed: {r.max_spike}/48 near 1")
        if notes:
            other = annotate.get((r.frac, mu)) if isinstance(annotate, dict) else None
            up = (r.model == "pretrained") if other is None else (r.bias >= other if r.model == "pretrained" else r.bias > other)
            first, last = r.frac == agg.frac.min(), r.frac == agg.frac.max()   # keep the edge labels inside the axes
            ax.annotate("; ".join(notes), (X[r.frac], r.bias), textcoords="offset points",
                        xytext=(9 if first else -9 if last else 0, 10 if up else -16),
                        ha="left" if first else "right" if last else "center", fontsize=7, color=color)
    return ax.errorbar([], [], yerr=[], fmt=ls, marker="o", color=color, lw=lw, ms=ms, alpha=alpha, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="mintail")
    args = ap.parse_args()
    ckmap = checkpoint_map(args.rule)
    print("S/B checkpoint per fraction:", ckmap)
    pre = pretrained_points(ckmap)
    eq_seeds = pd.read_csv(FS / "scratch_sb_x_long_sbi.csv")
    eq_seeds = eq_seeds[eq_seeds.partner == "ft_e20"]
    eq = seed_aggregate(eq_seeds, "scratch S/B + same SBI/B")
    es_seeds = pd.read_csv(DIR / "diagnostics" / "per_seed_diagonal.csv")
    es_seeds = es_seeds[es_seeds.variant == "scratch"]
    es = seed_aggregate(es_seeds, "scratch, all early-stopped (as run)")
    rule_txt = ("checkpoint with the smallest validation tail share at each size (no toys used)" if args.rule == "mintail"
                else f"checkpoint {args.rule} at every size")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 11), sharex=True)
    handles = []
    for ax, mu in zip(axes, (2.5, 0.4)):
        frac_axis(ax, top_labels=(ax is axes[0]))
        s = eq[eq.mu_true == mu].sort_values("frac")
        band = ax.fill_between([X[f] for f in s.frac], -0.5 * s.mean_width.fillna(s.mean_width.mean()),
                               0.5 * s.mean_width.fillna(s.mean_width.mean()), color="0.9",
                               label=r"$\pm\sigma_{\rm stat}$: half of the mean 1$\sigma$ width of the scratch fits")
        ax.axhline(0, color="k", lw=0.8)
        h3 = draw(ax, es, None, mu, C_S, 1.3, 6, 0.45, "--", 3,
                  "reference: scratch exactly as run in the early-stopped scan (its own early-stopped SBI/B), mean over seeds", False)
        pos_s = {(r.frac, r.mu_true): r.bias for _, r in eq.iterrows()}      # label goes on the side away from the other curve
        pos_p = {(r.frac, r.mu_true): r.bias for _, r in pre.iterrows()}
        h2 = draw(ax, eq, eq_seeds, mu, C_S, 2.3, 8.5, 1.0, "-", 5,
                  "scratch: S/B early-stopped best-val + the same SBI/B; mean over seeds 0/1/2 (small markers) at 1/3/10 %, seed 0 at 30/100 %",
                  pos_p)
        h1 = draw(ax, pre, None, mu, C_P, 2.4, 9, 1.0, "-", 7,
                  f"pretrained, fine-tuned: S/B at the {rule_txt} + SBI/B at epoch 20; seed 0", pos_s)
        handles = [h1, h2, h3, band]
        ax.set_ylabel(r"bias  $\langle\hat\mu\rangle - \mu_{\rm true}$")
        ax.text(0.98, 0.05, rf"$\mu_{{\rm true}} = {mu}$", transform=ax.transAxes, fontsize=12, ha="right")
    axes[0].set_ylim(-2.75, 0.6)
    axes[1].set_ylim(-0.6, 0.8)
    axes[1].set_xlabel("training events (signal hypothesis)")
    fig.suptitle("NSBI signal-strength bias vs training statistics: fine-tuned pretrained EveNet vs scratch\n"
                 "best recipe of each model, both paired with the same fully trained SBI/B classifier; 48 weighted toys, 300 fb$^{-1}$",
                 fontsize=11)
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=7.8, frameon=True, bbox_to_anchor=(0.5, 0.048))
    fig.text(0.5, 0.006, "error bars: seed spread where 3 seeds exist, otherwise toy s.e. only (excludes run-to-run spread, about 0.07 at 10 %).\n"
             "Hollow: some 1$\\sigma$ intervals open; points at -2.5: all 48 fits pinned at $\\hat\\mu$ = 0.",
             ha="center", fontsize=7.8, color="0.35")
    fig.tight_layout(rect=(0, 0.15, 1, 0.97))
    out = FIG / f"final_bias_pretrained_vs_scratch_{args.rule}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    cols = ["model", "frac", "mu_true", "sb_checkpoint", "bias", "err", "err_kind", "n_seeds", "n_toys", "median_muhat",
            "mean_width", "n_open", "n_spike", "bias_closed_only"]
    table = pd.concat([pre[cols], eq[cols], es[cols]], ignore_index=True).sort_values(["mu_true", "model", "frac"])
    table["train_events_sig"] = table.frac.map(X)
    table.to_csv(FS / f"final_curve_{args.rule}.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.drop(columns=["n_toys", "train_events_sig"]).round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
