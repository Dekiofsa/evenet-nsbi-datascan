import sys, time
from pathlib import Path
import numpy as np, pandas as pd

DIR = str(Path(__file__).resolve().parents[1])   # the scan folder (repository root)
sys.path.insert(0, DIR)
import nsbi_scan as ns

cfg = ns.ScanConfig.load(DIR).with_root(DIR)
xs = ns.load_cross_sections(cfg)
print("xs:", xs, flush=True)
toys = {tag: ns.load_toys(cfg, tag, with_kin=False) for tag in ("mu2p5", "mu0p4")}
runs = ns.collect_runs(cfg)
print("runs collected:", len(runs), flush=True)


def get_run(variant, task, frac):
    r = runs[(runs.variant == variant) & (runs.task == task) & np.isclose(runs.frac, frac)]
    assert len(r) == 1, (variant, task, frac, len(r))
    return r.iloc[0]


def load_cal(variant, task, frac, tag):
    run = get_run(variant, task, frac)
    r = np.load(Path(run.run_dir) / f"r_toys_{tag}.npy").astype(np.float64)
    return r * float(run.C_ref)


def summarize(df, label):
    mh = df.mu_hat
    mu_true = float(df.mu_true.iloc[0])
    bias = mh.mean() - mu_true
    se = mh.std(ddof=1) / np.sqrt(mh.notna().sum())
    n_open = int(df.width.isna().sum())
    width = df.width.mean()
    print(f"{label:70s} bias={bias:+.4f} toySE={se:.4f} width={width:.3f} n_open={n_open} "
          f"n_boundary={int(df.at_boundary.sum())} mean_muhat={mh.mean():.3f}", flush=True)
    return bias, se, width, n_open


def measure(rs, rb, tag, label):
    t0 = time.time()
    df = ns.measure_toys(rs, rb, toys[tag], xs, cfg)
    out = summarize(df, label)
    print(f"    ({time.time()-t0:.0f} s)", flush=True)
    return out


# ---------------- E. max calibrated ratios on toy events ----------------
print("\n=== max calibrated ratio (r*C_ref) on toy events, and count beyond clip bounds ===")
for variant in ("scratch", "pretrained"):
    for task in ("sig_over_bkg", "sbi_over_bkg"):
        lo, hi = (1 / 30, 30) if task == "sig_over_bkg" else (1 / 3, 3)
        for frac in cfg.fractions:
            for tag in ("mu2p5", "mu0p4"):
                r = load_cal(variant, task, frac, tag)
                n_out = int(((r < lo) | (r > hi)).sum())
                print(f"{variant:10s} {task:12s} f={frac:<5} {tag}: max={r.max():>12.1f} min={r.min():.4f} "
                      f"n_events={len(r)} beyond[{lo:.3g},{hi}]={n_out} ({100*n_out/len(r):.3f} %)", flush=True)

# ---------------- A. sanity: reproduce two diagonal points ----------------
print("\n=== A. sanity reproduction of diagonal points ===")
measure(load_cal("scratch", "sig_over_bkg", 0.03, "mu2p5"), load_cal("scratch", "sbi_over_bkg", 0.03, "mu2p5"),
        "mu2p5", "scratch 3%/3% (aggregates.csv: -0.5525, w 0.593)")
measure(load_cal("pretrained", "sig_over_bkg", 0.3, "mu0p4"), load_cal("pretrained", "sbi_over_bkg", 0.3, "mu0p4"),
        "mu0p4", "pretrained 30%/30% (aggregates.csv: +0.5521, w 0.333)")

# ---------------- B. cross-variant pairing at 3 % ----------------
print("\n=== B. cross-variant pairing: scratch S/B 3% + pretrained SBI/B 3% ===")
for tag in ("mu2p5", "mu0p4"):
    measure(load_cal("scratch", "sig_over_bkg", 0.03, tag), load_cal("pretrained", "sbi_over_bkg", 0.03, tag),
            tag, f"scratch S/B 3% + pretrained SBI/B 3% [{tag}]")
print("--- reverse: pretrained S/B 3% + scratch SBI/B 3% ---")
for tag in ("mu2p5", "mu0p4"):
    measure(load_cal("pretrained", "sig_over_bkg", 0.03, tag), load_cal("scratch", "sbi_over_bkg", 0.03, tag),
            tag, f"pretrained S/B 3% + scratch SBI/B 3% [{tag}]")

# ---------------- D. clipping experiment on pretrained diagonal ----------------
print("\n=== D. clipping: pretrained diagonal, S/B clipped [1/30,30], SBI/B clipped [1/3,3] (post-calibration) ===")
res = []
for frac in cfg.fractions:
    for tag in ("mu2p5", "mu0p4"):
        rs = load_cal("pretrained", "sig_over_bkg", frac, tag)
        rb = load_cal("pretrained", "sbi_over_bkg", frac, tag)
        b_both = measure(np.clip(rs, 1 / 30, 30), np.clip(rb, 1 / 3, 3), tag,
                         f"pretrained f={frac} [{tag}] clip BOTH")
        b_sig = measure(np.clip(rs, 1 / 30, 30), rb, tag,
                        f"pretrained f={frac} [{tag}] clip S/B only")
        res.append((frac, tag, b_both, b_sig))

print("\n=== D summary (bias, n_open) ===")
for frac, tag, b_both, b_sig in res:
    print(f"f={frac:<5} {tag}: clip-both bias={b_both[0]:+.3f} (open {b_both[3]}) | clip-S/B-only bias={b_sig[0]:+.3f} (open {b_sig[3]})")
print("DONE", flush=True)
