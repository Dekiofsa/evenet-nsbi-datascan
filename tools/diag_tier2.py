"""CPU-only diagnostics on the completed runs. Parts: offdiag (attribution grid), cval (validation-sample
calibration), cross (cross-variant pairs), clip (tail clipping with and without recalibration), tails
(tail statistics), seeds (per-seed tables).

    python tools/diag_tier2.py --part tails seeds
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1]   # the scan folder (repository root)
sys.path.insert(0, str(DIR))
import nsbi_scan as ns  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 1000)

cfg = ns.ScanConfig.load(str(DIR)).with_root(str(DIR))
xs = ns.load_cross_sections(cfg)
_nll_orig = ns.nll_profile
ns.nll_profile = lambda *a, **k: _nll_orig(*a, **{**k, "chunk": 2048})
TAGS = {2.5: "mu2p5", 0.4: "mu0p4"}
CLIP = {"sig_over_bkg": 30.0, "sbi_over_bkg": 3.0}
OUT = DIR / "diagnostics"
OUT.mkdir(exist_ok=True)
runs = ns.collect_runs(cfg)
toys = {tag: ns.load_toys(cfg, tag, with_kin=False) for tag in TAGS.values()}


def run_row(variant, task, frac, seed):
    r = runs[(runs.variant == variant) & (runs.task == task) & np.isclose(runs.frac, frac) & (runs.seed == seed)]
    return r.iloc[0] if len(r) == 1 else None


def toy_r(run, tag):
    return np.load(Path(run.run_dir) / f"r_toys_{tag}.npy").astype(np.float64)


def summarize(df):
    mh = df.mu_hat
    mu_true = float(df.mu_true.iloc[0])
    return dict(bias=float(mh.mean() - mu_true), se=float(mh.std(ddof=1) / np.sqrt(mh.notna().sum())),
                width=float(df.width.mean()), n_open=int(df.width.isna().sum()), n_boundary=int(df.at_boundary.sum()),
                mean_muhat=float(mh.mean()), std_muhat=float(mh.std(ddof=1)), coverage=float(df.covers_1sig.mean()),
                n_spike=int(((mh > 0.85) & (mh < 1.15)).sum()))


def measure(rs, rb, tag):
    return summarize(ns.measure_toys(rs, rb, toys[tag], xs, cfg))


def den_weights():
    bkg_val = ns.load_split(cfg, "bkg", "val")
    sbi_val = ns.load_split(cfg, "sbi", "val")
    return {"sig_over_bkg": ns._denominator_weights(bkg_val, None),
            "sbi_over_bkg": ns._denominator_weights(sbi_val, ("sbi", "bkg"))}


# ----------------------------------------------------------------------------------------------------------
def part_offdiag():
    t0 = time.time()
    meas = ns.measure_all(cfg, runs[runs.variant != "poster"], diagonal_only=False, calibration="C_ref",
                          cache_path=str(DIR / "measurements_offdiag.csv"))
    per_seed, over = ns.aggregate(meas)
    per_seed.to_csv(OUT / "attribution_per_seed.csv", index=False)
    over.to_csv(OUT / "attribution_over_seeds.csv", index=False)
    print(f"[offdiag] {len(meas) // 48} (pair, mu) measurements in {time.time() - t0:.0f} s", flush=True)
    for variant in ("scratch", "pretrained", "frozen"):
        for mu in (2.5, 0.4):
            for seed in sorted(per_seed.seed.unique()):
                g = per_seed[(per_seed.variant == variant) & (per_seed.mu_true == mu) & (per_seed.seed == seed)]
                if g.empty:
                    continue
                print(f"\n=== {variant} seed {seed} mu_true={mu}: bias (rows frac_sig, cols frac_sbi) ===")
                print(g.pivot(index="frac_sig", columns="frac_sbi", values="bias").round(2).to_string())
                print("   n_open:")
                print(g.pivot(index="frac_sig", columns="frac_sbi", values="n_open").to_string())
    print("[offdiag] DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------
def part_cval():
    t0 = time.time()
    meas = ns.measure_all(cfg, runs, diagonal_only=True, calibration="C_val",
                          cache_path=str(DIR / "measurements_cval.csv"))
    ps_val, _ = ns.aggregate(meas)
    ps_ref = diagonal_ref()
    key = ["variant", "seed", "frac_sig", "mu_true"]
    cols = key + ["bias", "n_open", "mean_width"]
    m = ps_ref[cols].merge(ps_val[cols], on=key, suffixes=("_ref", "_val"))
    m["shift"] = m.bias_val - m.bias_ref
    m = m.sort_values(["mu_true", "variant", "frac_sig", "seed"])
    m.to_csv(OUT / "cval_vs_cref_per_seed.csv", index=False)
    print(f"[cval] {len(meas) // 48} (pair, mu) measurements in {time.time() - t0:.0f} s")
    print(m.round(3).to_string(index=False))
    print("\n[cval] max |shift| per variant:")
    print(m.groupby("variant")["shift"].agg(lambda s: s.abs().max()).round(3).to_string())
    print("[cval] DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------
PAIRS = [("scratch", "pretrained"), ("pretrained", "scratch"), ("frozen", "scratch"), ("scratch", "frozen"),
         ("frozen", "pretrained"), ("pretrained", "frozen")]


def part_cross():
    t0 = time.time()
    rows = []
    for va, vb in PAIRS:
        for frac in cfg.fractions:
            for seed in (0, 1, 2):
                a = run_row(va, "sig_over_bkg", frac, seed)
                b = run_row(vb, "sbi_over_bkg", frac, seed)
                if a is None or b is None:
                    continue
                for mu, tag in TAGS.items():
                    res = measure(toy_r(a, tag) * float(a.C_ref), toy_r(b, tag) * float(b.C_ref), tag)
                    rows.append(dict(sig_variant=va, sbi_variant=vb, frac=frac, seed=seed, mu_true=mu, **res))
                    print(f"[cross] S/B {va:10s} + SBI/B {vb:10s} f={frac:<5} seed {seed} mu={mu}: "
                          f"bias={res['bias']:+.3f} width={res['width']:.3f} open={res['n_open']} spike={res['n_spike']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cross_variant_pairs.csv", index=False)
    print(f"\n[cross] {len(rows)} measurements in {time.time() - t0:.0f} s")
    for mu in (2.5, 0.4):
        g = df[df.mu_true == mu].groupby(["sig_variant", "sbi_variant", "frac"]).agg(bias=("bias", "mean"), n_seeds=("seed", "size"),
                                                                                    open_max=("n_open", "max")).reset_index()
        print(f"\n=== cross-variant pairs, mu_true={mu}, bias mean over available seeds ===")
        print(g.pivot(index=["sig_variant", "sbi_variant"], columns="frac", values="bias").round(2).to_string())
        print("   n_open max:")
        print(g.pivot(index=["sig_variant", "sbi_variant"], columns="frac", values="open_max").to_string())
    print("[cross] DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------
MODES = ("clip_recal", "clip_only")


def clipped(run, w_den, tag, hi, recal):
    """Calibrated toy ratios clipped FROM ABOVE at hi (the bulk below is untouched); if recal, re-normalised so that"""
    C = float(run.C_ref)
    C2 = 1.0
    if recal:
        r_den = np.load(Path(run.run_dir) / "r_val.npz")["r_den"].astype(np.float64) * C
        if len(r_den) != len(w_den):
            raise ValueError(f"{run.run_dir}: r_den {len(r_den)} vs weights {len(w_den)}")
        C2 = ns.calibration_factor_np(np.minimum(r_den, hi), w_den)
    return np.minimum(toy_r(run, tag) * C, hi) * C2, C2


def diagonal_ref():
    """The scan's own result: diagonal pairs with the reference calibration, aggregated per seed."""
    ref = pd.read_csv(DIR / "measurements.csv")
    ref = ref[np.isclose(ref.frac_sig, ref.frac_sbi) & (ref.calibration == "C_ref")]
    ps_ref, _ = ns.aggregate(ref)
    return ps_ref


def part_clip():
    t0 = time.time()
    W = den_weights()
    rows = []
    diag = runs[runs.task == "sig_over_bkg"][["variant", "frac", "seed"]].drop_duplicates()
    for _, d in diag.sort_values(["variant", "frac", "seed"]).iterrows():
        a = run_row(d.variant, "sig_over_bkg", d.frac, d.seed)
        b = run_row(d.variant, "sbi_over_bkg", d.frac, d.seed)
        if a is None or b is None:
            continue
        for mu, tag in TAGS.items():
            rs_r, C2s = clipped(a, W["sig_over_bkg"], tag, CLIP["sig_over_bkg"], recal=True)
            rb_r, C2b = clipped(b, W["sbi_over_bkg"], tag, CLIP["sbi_over_bkg"], recal=True)
            modes = {"clip_recal": (rs_r, rb_r)}
            if int(d.seed) == 0:
                rs_c, _ = clipped(a, W["sig_over_bkg"], tag, CLIP["sig_over_bkg"], recal=False)
                rb_c, _ = clipped(b, W["sbi_over_bkg"], tag, CLIP["sbi_over_bkg"], recal=False)
                modes["clip_only"] = (rs_c, rb_c)
            for mode, (rs, rb) in modes.items():
                res = measure(rs, rb, tag)
                rows.append(dict(variant=d.variant, frac=d.frac, seed=int(d.seed), mu_true=mu, mode=mode, C2_sig=C2s, C2_sbi=C2b, **res))
                print(f"[clip] {d.variant:10s} f={d.frac:<5} seed {int(d.seed)} mu={mu} {mode:20s}: bias={res['bias']:+.3f} "
                      f"width={res['width']:.3f} open={res['n_open']} spike={res['n_spike']} (C2 S/B {C2s:.4f}, SBI/B {C2b:.5f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "clipping_per_seed.csv", index=False)
    print(f"\n[clip] {len(rows)} measurements in {time.time() - t0:.0f} s")
    ps_ref = diagonal_ref().rename(columns={"frac_sig": "frac"})[["variant", "frac", "seed", "mu_true", "bias", "n_open", "mean_width"]]
    for mode in MODES:
        g = df[df["mode"] == mode].merge(ps_ref, on=["variant", "frac", "seed", "mu_true"], suffixes=("", "_raw"))
        print(f"\n=== {mode}: per seed, raw -> clipped (bias, n_open) ===")
        for mu in (2.5, 0.4):
            h = g[g.mu_true == mu].sort_values(["variant", "frac", "seed"])
            print(f"-- mu_true={mu}")
            print(h[["variant", "frac", "seed", "bias_raw", "n_open_raw", "bias", "n_open", "width", "n_spike"]].round(3).to_string(index=False))
        print(f"-- {mode}: mean over seeds (rows variant, cols frac)")
        for mu in (2.5, 0.4):
            print(f"   mu_true={mu}")
            print(g[g.mu_true == mu].groupby(["variant", "frac"]).bias.mean().unstack().round(2).to_string())
    print("[clip] DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------
def part_tails():
    W = den_weights()
    W_NUM = {"sig_over_bkg": ns.load_split(cfg, "sig", "val").weights.to_numpy(np.float64),
             "sbi_over_bkg": ns.load_split(cfg, "sbi", "val").weights.to_numpy(np.float64)}
    rows = []
    for _, run in runs.sort_values(["variant", "task", "frac", "seed"]).iterrows():
        hi = CLIP[run.task]
        w = W[run.task]
        rv = np.load(Path(run.run_dir) / "r_val.npz")
        r = rv["r_den"].astype(np.float64) * float(run.C_ref)
        if len(r) != len(w):
            raise ValueError(f"{run.run_dir}: r_den {len(r)} vs weights {len(w)}")
        p = w / w.sum()
        mean_r = float((r * p).sum())
        if abs(mean_r * float(run.C_val) / float(run.C_ref) - 1) > 1e-4:
            raise ValueError(f"{run.run_dir}: <r>_val * C_val / C_ref = {mean_r * run.C_val / run.C_ref:.6f}, weights misaligned?")
        big = r > hi
        r_num = rv["r_num"].astype(np.float64) * float(run.C_ref)
        w_num = W_NUM[run.task]
        if len(r_num) != len(w_num):
            raise ValueError(f"{run.run_dir}: r_num {len(r_num)} vs numerator weights {len(w_num)}")
        share_big_num = float(w_num[r_num > hi].sum() / w_num.sum())
        row = dict(variant=run.variant, task=run.task, frac=run.frac, seed=int(run.seed), auc=run.val_auc_full,
                   C_ref=run.C_ref, C_val=run.C_val, best_epoch=run.best_epoch, mean_r_val=mean_r, max_r_val=float(r.max()),
                   n_big_val=int(big.sum()), w_frac_big=float(p[big].sum()), share_big=float((r[big] * p[big]).sum() / mean_r),
                   share_big_num=share_big_num, tail_overestimate=float((r[big] * p[big]).sum() / mean_r / max(share_big_num, 1e-12)))
        for mu, tag in TAGS.items():
            rt = toy_r(run, tag) * float(run.C_ref)
            n = toys[tag]["n"]
            bigt = rt > hi
            row[f"max_r_toys_{tag}"] = float(rt.max())
            row[f"n_big_toys_{tag}"] = int(bigt.sum())
            row[f"wfrac_big_toys_{tag}"] = float(n[bigt].sum() / n.sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tail_stats_per_run.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("[tails] DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------
def part_seeds():
    m = pd.read_csv(DIR / "measurements.csv")
    m = m[np.isclose(m.frac_sig, m.frac_sbi) & (m.calibration == "C_ref")]
    ps, over = ns.aggregate(m)
    ps = ps.rename(columns={"frac_sig": "frac"})
    spike = m.groupby(["variant", "seed", "frac_sig", "mu_true"]).mu_hat.apply(lambda s: int(((s > 0.85) & (s < 1.15)).sum())).rename("n_spike").reset_index().rename(columns={"frac_sig": "frac"})
    ps = ps.merge(spike, on=["variant", "seed", "frac", "mu_true"])
    ps = ps.sort_values(["mu_true", "variant", "frac", "seed"])
    ps.to_csv(OUT / "per_seed_diagonal.csv", index=False)
    cols = ["mu_true", "variant", "frac", "seed", "bias", "bias_se_toys", "mean_width", "n_open", "n_spike", "frac_covering"]
    print(ps[cols].round(3).to_string(index=False))
    # paired scratch - pretrained per seed
    p = ps.pivot_table(index=["mu_true", "frac", "seed"], columns="variant", values="bias").reset_index()
    if {"scratch", "pretrained"} <= set(p.columns):
        p["abs_gain_pretrained"] = p.scratch.abs() - p.pretrained.abs()
        print("\n=== per-seed paired: |bias_scratch| - |bias_pretrained| (>0: pretraining helps) ===")
        print(p.round(3).to_string(index=False))
    # markdown table for the report
    lines = ["| mu_true | variant | fraction | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|---|---|"]

    def cell(r):
        if r is None:
            return "-"
        s = f"{r.bias:+.2f}"
        if r.n_open >= 47:
            return "collapsed (mu_hat = 0, 48 open)"
        if r.n_open > 0:
            s += f" ({int(r.n_open)} open)"
        if r.n_spike >= 24:
            s += f" (spike, {int(r.n_spike)} toys at mu = 1)"
        return s

    for mu in (2.5, 0.4):
        for variant in ("scratch", "pretrained", "frozen"):
            for frac in cfg.fractions:
                g = ps[(ps.mu_true == mu) & (ps.variant == variant) & np.isclose(ps.frac, frac)]
                if g.empty:
                    continue
                cells = []
                for seed in (0, 1, 2):
                    h = g[g.seed == seed]
                    cells.append(cell(h.iloc[0]) if len(h) else "-")
                lines.append(f"| {mu} | {variant} | {int(round(frac * 100))} % | " + " | ".join(cells) + " |")
    (OUT / "per_seed_table.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("[seeds] DONE", flush=True)


PARTS = {"offdiag": part_offdiag, "cval": part_cval, "cross": part_cross, "clip": part_clip, "tails": part_tails, "seeds": part_seeds}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", choices=list(PARTS), required=True)
    a = ap.parse_args()
    print(f"runs collected: {len(runs)} | xs {xs}", flush=True)
    for name in a.part:
        PARTS[name]()
