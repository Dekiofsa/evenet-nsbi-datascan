#!/usr/bin/env python
"""Fine-tuned pretrained ladder with the poster's 20-epoch recipe scaled to the data size: one full pass per epoch,
cosine to 0, early stopping off, every epoch saved. Exports for the best-val weights and selected epochs go to
runs/pretrained/ and runs/pretrained_eXX/. Root ~/datascan/fullsched on the GPU box.

    python -u tools/full_schedule_scan.py --smoke
    python -u tools/full_schedule_scan.py --variants pretrained 2>&1 | tee -a logs/fullsched.log
    python -u tools/full_schedule_scan.py --export-only --export-epochs 3,7
    python -u tools/full_schedule_scan.py --measure
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
MAIN = Path.home() / "datascan"
sys.path.insert(0, str(MAIN))
_ev = os.environ.get("EVENET_LITE_DIR", str(Path.home() / "EveNet-Lite"))
if os.path.isdir(_ev) and _ev not in sys.path:
    sys.path.insert(0, _ev)

import nsbi_scan as ns  # noqa: E402

EPOCHS = 20
SEED = 0
TASKS = ("sig_over_bkg", "sbi_over_bkg")
FRACTIONS = (0.01, 0.03, 0.1, 0.3, 1.0)            # cheapest first
EXPORT_EPOCHS = (5, 10, 15, 20)
_STATE = {"epochs": EPOCHS}
_SUFFIX = re.compile(r"^(?P<model>.+?)_e(?P<epoch>\d{2})$")
_G = {}                                            # cfg for the measurement helpers


# ---------------------------------------------------------------------------------------------- roots and config
def make_root(name: str):
    root = MAIN / name
    root.mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    link = root / "data"
    if link.is_symlink() and not link.exists():
        link.unlink()
    if not link.exists():
        try:
            os.symlink(MAIN / "data", link, target_is_directory=True)
        except FileExistsError:
            pass
    cfg = ns.ScanConfig.load(str(MAIN)).with_root(str(root))
    cfg = dataclasses.replace(cfg, early_stop_patience=0, keep_checkpoint=True, wandb_project=None)
    return root, cfg


def n_train_rows(cfg: ns.ScanConfig, task: str, frac: float, seed: int) -> int:
    """Rows of the (num + den) classifier dataset as nsbi.data.make_classifier_data stacks them (= metrics.json"""
    P = ns.task_processes(cfg, task, frac, seed)
    return int(len(P["num_train"]) + len(P["den_train"]))


def run_cfg(cfg: ns.ScanConfig, task: str, frac: float, seed: int, epochs: int) -> ns.ScanConfig:
    """One full pass over the training rows per epoch, `epochs` epochs, cosine to 0 at the end."""
    n = n_train_rows(cfg, task, frac, seed)
    _STATE["epochs"] = epochs
    return dataclasses.replace(cfg, epoch_size=n, max_epochs=epochs, early_stop_patience=0,
                               keep_checkpoint=True, wandb_project=None)


def steps_per_epoch(cfg: ns.ScanConfig) -> int:
    return -(-cfg.epoch_size // cfg.batch_size)


# ---------------------------------------------------------------------------------------------- callback patch
_orig_make_history = ns.make_history_callback


def make_history_callback_full(path):
    """The scan's per-epoch history.json plus the lr of every optimizer group, and a checkpoint of EVERY epoch"""
    from evenet_lite.callbacks import Callback

    inner = _orig_make_history(path)
    out_dir = Path(path).parent

    class EveryEpochCallback(Callback):
        def __init__(self):
            self.inner = inner

        @property
        def rows(self):
            return self.inner.rows

        def on_epoch_end(self, trainer, epoch: int, metrics) -> None:
            self.inner.on_epoch_end(trainer, epoch, metrics)
            lr = [float(pg["lr"]) for opt in getattr(trainer, "optimizers", []) for pg in opt.param_groups]
            self.inner.rows[-1]["lr"] = lr
            with open(self.inner.out, "w") as fh:
                json.dump(self.inner.rows, fh, indent=1)
            f = out_dir / f"epoch{epoch + 1:02d}.pt"
            trainer.save_checkpoint(str(f), extra={"epoch": int(epoch), "kind": f"e{epoch + 1:02d}"})
            print(f"[full] epoch {epoch + 1}/{_STATE['epochs']} step {getattr(trainer, 'global_step', -1)} "
                  f"lr={['%.3g' % v for v in lr]} val_loss={metrics.get('val_loss', float('nan')):.4f} -> {f.name}",
                  flush=True)

    return EveryEpochCallback()


ns.make_history_callback = make_history_callback_full      # train_one looks the name up at call time


# ---------------------------------------------------------------------------------------------- exports
def epoch_variant(variant: str, epoch: int) -> str:
    return f"{variant}_e{epoch:02d}"


def export_epoch(cfg: ns.ScanConfig, variant: str, task: str, frac: float, seed: int, epoch: int,
                 device: str = "cuda") -> bool:
    """Exports r(x) of epochXX.pt into runs/<variant>_eXX/..."""
    import torch
    from evenet_lite import EvenetLiteClassifier
    from nsbi.data import FEATURE_NAMES

    src = ns.run_dir(cfg, variant, task, frac, seed)
    vname = epoch_variant(variant, epoch)
    dst = ns.run_dir(cfg, vname, task, frac, seed)
    ckpt = src / f"epoch{epoch:02d}.pt"
    if (dst / "done.json").exists():
        print(f"[skip] {dst} already done")
        return True
    if not ckpt.exists():
        print(f"[FAILED] {dst}: {ckpt} missing")
        return False
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "failed.json").unlink(missing_ok=True)
    t0 = time.time()
    try:
        ns.set_fast_math(False)
        clf = EvenetLiteClassifier(class_labels=["bkg", "sig"], device=device, sequential_input_dim=ns.SEQ_DIM,
                                   global_input_dim=ns.GLOBAL_DIM, pretrained=False)
        clf.load_checkpoint(str(ckpt), feature_names=FEATURE_NAMES)
        P = ns.task_processes(cfg, task, frac, seed)
        w_den_train = ns._denominator_weights(P["den_train"], P["reweight"])
        dev = ns.check_fast_predict(clf, P["den_val"])
        r_num_val = ns.compute_r_np(clf, P["num_val"])
        r_den_val = r_num_val if P["num_val"] is P["den_val"] else ns.compute_r_np(clf, P["den_val"])
        w_num_val = P["num_val"].weights.to_numpy(np.float64)
        w_den_val = ns._denominator_weights(P["den_val"], P["reweight"])
        np.savez(dst / "r_val.npz", r_num=r_num_val.astype(np.float32), r_den=r_den_val.astype(np.float32),
                 same_events=np.bool_(P["num_val"] is P["den_val"]))
        vm = ns.weighted_bce_auc(r_num_val, w_num_val, r_den_val, w_den_val)
        closure = ns.closure_chi2(P["num_val"].kinematics, w_num_val, P["den_val"].kinematics, w_den_val, r_den_val)
        ref = ns.calibration_reference(cfg)
        C_ref = ns.calibration_factor_np(ns.compute_r_np(clf, ref), ref.weights.to_numpy(np.float64))
        C_sub = ns.calibration_factor_np(ns.compute_r_np(clf, P["den_train"]), w_den_train)
        C_val = ns.calibration_factor_np(r_den_val, w_den_val)
        toy_info = {}
        for mu_true in cfg.mu_true_values:
            tag = ns.MU_TAGS[mu_true]
            toys = ns.load_toys(cfg, tag)
            r_toys = ns.compute_r_np(clf, ns.process_from_arrays(toys["kin"], None, toys["n"])).astype(np.float32)
            np.save(dst / f"r_toys_{tag}.npy", r_toys)
            toy_info[tag] = {"rows": int(len(r_toys)), "mean_r": float(r_toys.mean())}
        best_m = json.loads((src / "metrics.json").read_text())
        hist = json.loads((src / "history.json").read_text())
        row = hist[epoch - 1]
        keep = ("subsample_seed", "n_num_train", "n_den_train", "ess_num_train", "ess_den_train", "n_num_val",
                "n_den_val", "n_train_rows", "n_val_monitor_rows", "n_trainable_params", "n_nonfinite_batches",
                "steps_per_epoch", "tf32_training", "autocast_bf16_training", "fit_seconds", "device", "config")
        metrics = {"variant": vname, "task": task, "frac": frac, "frac_tag": ns.frac_tag(frac), "seed": seed,
                   "checkpoint_kind": f"e{epoch:02d}", "checkpoint": str(ckpt),
                   **{k: best_m[k] for k in keep if k in best_m},
                   "epochs_run": len(hist), "checkpoint_epoch": int(epoch),
                   "checkpoint_step": int(row.get("global_step") or 0),
                   "best_epoch": best_m.get("best_epoch"), "best_step": best_m.get("best_step"),   # of the paired best run
                   "best_val_loss_monitor": best_m.get("best_val_loss_monitor"),
                   "total_steps": int(hist[-1].get("global_step") or 0),
                   "checkpoint_val_loss_monitor": float(row.get("val_loss", float("nan"))),
                   "lr_at_checkpoint": row.get("lr"), "resumed_from_checkpoint": False,
                   **vm, "C_ref": C_ref, "C_sub": C_sub, "C_val": C_val, "fast_predict_max_rel_dev": dev,
                   "closure_chi2_ndf": closure["closure_chi2_ndf"], "closure_chi2": closure["closure_chi2"],
                   "closure_ndf": closure["closure_ndf"], "toys": toy_info, "total_seconds": time.time() - t0}
        (dst / "metrics.json").write_text(json.dumps(ns._jsonable(metrics), indent=1, default=str))
        (dst / "closure.json").write_text(json.dumps(ns._jsonable(closure)))
        json.loads((dst / "metrics.json").read_text())
        shutil.copyfile(src / "history.json", dst / "history.json")
        (dst / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                   "seconds": time.time() - t0}))
        print(f"[done] {vname}/{task}/{ns.frac_tag(frac)}/seed{seed}: val_auc={vm['val_auc_full']:.4f} "
              f"bce={vm['val_bce_full']:.4f} C_ref={C_ref:.4f} chi2/ndf={closure['closure_chi2_ndf']:.2f} "
              f"({(time.time() - t0) / 60:.1f} min)")
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as exc:
        (dst / "failed.json").write_text(json.dumps({"error": repr(exc), "traceback": traceback.format_exc(),
                                                     "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        print(f"[FAILED] {vname}/{task}/{ns.frac_tag(frac)}/seed{seed}: {exc!r}")
        return False


def post_best(cfg: ns.ScanConfig, variant: str, task: str, frac: float, seed: int) -> bool:
    """After train_one: mark the best pseudo-run, hard-link best.pt, check every epoch file exists, the epoch size"""
    out = ns.run_dir(cfg, variant, task, frac, seed)
    m = json.loads((out / "metrics.json").read_text())
    if m.get("checkpoint_kind") != "best":
        m["checkpoint_kind"] = "best"
        (out / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    if (out / "final.pt").exists() and not (out / "best.pt").exists():
        try:
            os.link(out / "final.pt", out / "best.pt")
        except OSError:
            shutil.copyfile(out / "final.pt", out / "best.pt")
    hist = json.loads((out / "history.json").read_text())
    lrs = [(r.get("lr") or [None])[0] for r in hist]
    fmt = lambda v: "%.3g" % v if isinstance(v, (int, float)) else "n/a"
    n_ep = cfg.max_epochs
    files = [(out / f"epoch{e:02d}.pt").exists() for e in range(1, n_ep + 1)]
    lr_ok = bool(lrs) and lrs[-1] is not None and bool(lrs[0]) and lrs[-1] <= 1e-9 * lrs[0]
    rows_ok = int(m.get("n_train_rows", -1)) == int(cfg.epoch_size)
    ok = lr_ok and rows_ok and len(hist) == n_ep and all(files)
    print(f"[full] {variant}/{task}/{ns.frac_tag(frac)}: {len(hist)}/{n_ep} epochs, {sum(files)}/{n_ep} epoch files, "
          f"steps/epoch {m.get('steps_per_epoch')} (n_train_rows {m.get('n_train_rows')} == epoch_size {cfg.epoch_size}: "
          f"{rows_ok}), total steps {m.get('total_steps')}, best epoch {m.get('best_epoch')}, "
          f"lr first->last {fmt(lrs[0] if lrs else None)} -> {fmt(lrs[-1] if lrs else None)}, schedule finished: {lr_ok}")
    if not ok:
        print(f"[full] WARNING: checks failed for {out}")
    return ok


# ---------------------------------------------------------------------------------------------- measurement
def split_variant(v: str):
    m = _SUFFIX.match(v)
    return (m.group("model"), "e" + m.group("epoch")) if m else (v, "best")


def tail_rows(runs):
    """Tail statistic of every S/B run: share of <r>_B on the validation background carried by events with r > 30."""
    import pandas as pd
    W = ns._denominator_weights(ns.load_split(_G["cfg"], "bkg", "val"), None)
    p = W / W.sum()
    rows = []
    for _, a in runs[runs.task == "sig_over_bkg"].iterrows():
        r = np.load(Path(a.run_dir) / "r_val.npz")["r_den"].astype(np.float64) * float(a.C_ref)
        big = r > 30
        rt = np.load(Path(a.run_dir) / "r_toys_mu0p4.npy").astype(np.float64) * float(a.C_ref)
        rows.append(dict(variant=a.variant, seed=int(a.seed), frac_sig=float(a.frac),
                         tail_share=float((r[big] * p[big]).sum() / (r * p).sum()),
                         max_r_toys=float(rt.max()), bce_sb=float(a.val_bce_full), auc_sb=float(a.val_auc_full)))
    return pd.DataFrame(rows)


def summarize(cfg, runs, meas, label):
    per_seed, _ = ns.aggregate(meas)
    keys = ["variant", "seed", "frac_sig", "frac_sbi", "mu_true"]
    spike = meas.groupby(keys).mu_hat.apply(lambda s: int(((s > 0.85) & (s < 1.15)).sum())).rename("n_spike").reset_index()
    t = per_seed.merge(spike, on=keys).merge(tail_rows(runs), on=["variant", "seed", "frac_sig"], how="left")
    sbi = runs[runs.task == "sbi_over_bkg"][["variant", "seed", "frac", "val_bce_full"]].rename(
        columns={"frac": "frac_sbi", "val_bce_full": "bce_sbi"})
    t = t.merge(sbi, on=["variant", "seed", "frac_sbi"], how="left")
    t["model"], t["checkpoint"] = zip(*t.variant.map(split_variant))
    sb = runs[runs.task == "sig_over_bkg"]
    best = sb[sb.variant.map(lambda v: split_variant(v)[1] == "best")][["variant", "seed", "frac", "best_epoch"]]
    best = best.rename(columns={"variant": "model", "frac": "frac_sig"})
    t = t.merge(best, on=["model", "seed", "frac_sig"], how="left")
    cols = ["model", "checkpoint", "seed", "frac_sig", "mu_true", "bias", "mean_width", "n_open", "n_spike", "tail_share",
            "max_r_toys", "bce_sb", "bce_sbi", "best_epoch"]
    if label == "cross":
        t["sbi_checkpoint"] = "e%02d" % EXPORT_EPOCHS[-1]
        cols.insert(2, "sbi_checkpoint")
    t = t.sort_values(["mu_true", "frac_sig", "model", "checkpoint", "seed"])
    print(f"\n=== {label}: 48 toys, calibration C_ref, seed {SEED} (n_spike: mu_hat within 0.15 of 1; "
          f"tail_share: share of <r>_B on validation background from r > 30) ===")
    print(t[cols].round(4).to_string(index=False))
    t[cols].to_csv(Path(cfg.root) / f"report_{label}.csv", index=False)
    return t


def report(cfg: ns.ScanConfig):
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    _G["cfg"] = cfg
    runs = ns.collect_runs(cfg)
    if runs.empty:
        print("no finished runs under", cfg.runs_dir)
        return
    runs["model"], runs["checkpoint"] = zip(*runs.variant.map(split_variant))
    have = runs.groupby(["model", "checkpoint", "frac"]).task.nunique()
    print("finished pairs (both classifiers):")
    print(have[have == 2].reset_index()[["model", "checkpoint", "frac"]].to_string(index=False))
    meas = ns.measure_all(cfg, runs, diagonal_only=True, calibration="C_ref",
                          cache_path=str(Path(cfg.root) / "measurements_fullsched.csv"))
    if meas.empty:
        print("no complete pairs yet")
        return
    summarize(cfg, runs, meas, "diagonal")
    last = "e%02d" % EXPORT_EPOCHS[-1]
    sbi_last = runs[(runs.task == "sbi_over_bkg") & (runs.checkpoint == last)]
    parts = []
    for _, a in runs[runs.task == "sig_over_bkg"].iterrows():
        b = sbi_last[(sbi_last.model == a.model) & np.isclose(sbi_last.frac, a.frac) & (sbi_last.seed == a.seed)]
        if len(b) == 1:
            parts.append(a.to_frame().T)
            parts.append(b.assign(variant=a.variant))
    if parts:
        cross = pd.concat(parts, ignore_index=True)
        for c in ("frac", "seed", "C_ref", "val_bce_full", "val_auc_full", "best_epoch"):
            cross[c] = pd.to_numeric(cross[c])
        meas_x = ns.measure_all(cfg, cross, diagonal_only=True, calibration="C_ref",
                                cache_path=str(Path(cfg.root) / "measurements_fullsched_cross.csv"))
        if not meas_x.empty:
            summarize(cfg, cross, meas_x, "cross")
    ref = MAIN / "aggregates_per_seed.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        r = r[(r.seed == 0) & np.isclose(r.frac_sig, r.frac_sbi) & (r.variant == "pretrained")]
        print("\n=== the scan's early-stopped seed-0 pretrained result (HANDOFF section 3) for comparison ===")
        print(r[["variant", "frac_sig", "mu_true", "bias", "mean_width", "n_open"]]
              .sort_values(["mu_true", "frac_sig"]).round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------- driver
def specs_for(variants, fractions, tasks=TASKS):
    return [dict(variant=v, task=t, frac=f, seed=SEED) for f in fractions for v in sorted(variants) for t in tasks]


def status(cfg, specs, export_epochs):
    for s in specs:
        out = ns.run_dir(cfg, **s)
        n_files = sum((out / f"epoch{e:02d}.pt").exists() for e in range(1, EPOCHS + 1))
        hist = json.loads((out / "history.json").read_text()) if (out / "history.json").exists() else []
        st = ("done" if (out / "done.json").exists() else "FAILED" if (out / "failed.json").exists()
              else "training" if hist else "pending")
        ex = "".join("+" if (ns.run_dir(cfg, epoch_variant(s["variant"], e), s["task"], s["frac"], s["seed"])
                             / "done.json").exists() else "-" for e in export_epochs)
        print(f"{s['variant']}/{s['task']}/{ns.frac_tag(s['frac'])}: {st:9s} epochs {len(hist):2d}/{EPOCHS} "
              f"files {n_files:2d} exports[{','.join('e%02d' % e for e in export_epochs)}] {ex}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="pretrained")
    ap.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    ap.add_argument("--export-epochs", default=",".join(str(e) for e in EXPORT_EPOCHS))
    ap.add_argument("--tasks", default=",".join(TASKS), help="classifiers to run/export (default both)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--export-only", action="store_true",
                    help="never train: only export --export-epochs of runs that already have done.json (safe while a driver runs)")
    args = ap.parse_args()
    variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    unknown = variants - set(ns.VARIANTS)
    if unknown:
        print("unknown variants", sorted(unknown))
        return 1
    fractions = [float(f) for f in args.fractions.split(",") if f.strip()]
    export_epochs = [int(e) for e in args.export_epochs.split(",") if e.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if set(tasks) - set(TASKS):
        print("unknown tasks", sorted(set(tasks) - set(TASKS)))
        return 1

    if args.smoke:
        root, cfg = make_root("fullsched_smoke")
        spec = dict(variant="pretrained", task="sig_over_bkg", frac=0.01, seed=SEED)
        n_ep = 3
        cfg = run_cfg(cfg, spec["task"], spec["frac"], spec["seed"], n_ep)
        print(f"[smoke] root {root}: pretrained S/B at 1 %, {n_ep} epochs x {steps_per_epoch(cfg)} steps "
              f"(epoch_size {cfg.epoch_size}), early stopping off")
        t0 = time.time()
        m = ns.train_one(cfg, device=args.device, force=True, **spec)
        if m is None:
            print("[smoke] FAIL: training failed, see failed.json")
            return 2
        ok = post_best(cfg, **spec)
        eps = list(range(1, n_ep + 1))
        for e in eps:
            if not export_epoch(cfg, epoch=e, device=args.device, **spec):
                print(f"[smoke] FAIL: export of epoch {e} failed")
                return 2
        runs = ns.collect_runs(cfg)
        rd = lambda v: np.load(ns.run_dir(cfg, v, "sig_over_bkg", 0.01, SEED) / "r_toys_mu0p4.npy")
        best = rd("pretrained")
        R = {e: rd(epoch_variant("pretrained", e)) for e in eps}
        best_epoch = int(m.get("best_epoch", -1))
        rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-6)))
        pairs = [(a, b) for a in eps for b in eps if a < b]
        devs = {f"e{a}-e{b}": round(rel(R[a], R[b]), 3) for a, b in pairs}
        differ = all(v > 1e-3 for v in devs.values())
        same_best = bool(np.allclose(best, R[best_epoch])) if best_epoch in R else False
        hist = json.loads((ns.run_dir(cfg, **spec) / "history.json").read_text())
        lrs = [r.get("lr") for r in hist]
        fit_s = float(m.get("fit_seconds", 0) or 0)
        print(f"[smoke] fit {fit_s:.0f} s for {m.get('total_steps')} steps "
              f"({int(m.get('total_steps', 0)) / max(fit_s, 1):.1f} steps/s), whole smoke {(time.time() - t0) / 60:.1f} min")
        print(f"[smoke] lr per epoch (head/ObjectEncoder/body): {[['%.3g' % v for v in l] for l in lrs]}")
        print(f"[smoke] best epoch {best_epoch}; best export == epoch{best_epoch:02d} export: {same_best}; "
              f"epoch exports pairwise different: {differ} (max rel devs {devs})")
        print(f"[smoke] collect_runs variants: {sorted(runs.variant)}")
        src = ns.run_dir(cfg, **spec)
        for f in ["best.pt", "final.pt", "history.json", "metrics.json", "done.json"] + [f"epoch{e:02d}.pt" for e in eps]:
            print(f"   best/{f}: {(src / f).exists()}")
        for e in eps:
            d = ns.run_dir(cfg, epoch_variant("pretrained", e), "sig_over_bkg", 0.01, SEED)
            print(f"   e{e:02d}/: " + " ".join(f"{f}={(d / f).exists()}" for f in
                                             ("r_val.npz", "r_toys_mu0p4.npy", "r_toys_mu2p5.npy", "metrics.json", "done.json")))
        expected = {"pretrained"} | {epoch_variant("pretrained", e) for e in eps}
        passed = ok and set(runs.variant) == expected and differ and same_best
        print("[smoke] PASS" if passed else "[smoke] CHECK THE LINES ABOVE")
        return 0 if passed else 3

    root, cfg0 = make_root("fullsched")
    specs = specs_for(variants, fractions, tasks)
    if args.measure:
        report(cfg0)
        return 0
    if args.status:
        status(cfg0, specs, export_epochs)
        return 0

    print(f"root {root} | {len(specs)} runs for variants {sorted(variants)} at fractions {fractions} | {EPOCHS} full-pass "
          f"epochs, early stopping off, every epoch saved, exports best + {['e%02d' % e for e in export_epochs]}")
    n_started, n_failed = 0, 0
    for spec in specs:
        cfg = run_cfg(cfg0, spec["task"], spec["frac"], spec["seed"], EPOCHS)
        best_done = (ns.run_dir(cfg, **spec) / "done.json").exists()
        ex_done = all((ns.run_dir(cfg, epoch_variant(spec["variant"], e), spec["task"], spec["frac"], spec["seed"])
                       / "done.json").exists() for e in export_epochs)
        if best_done and ex_done:
            print(f"[skip] {spec['variant']}/{spec['task']}/{ns.frac_tag(spec['frac'])} done (best + exports)")
            continue
        if args.export_only and not best_done:
            print(f"[export-only] {spec['variant']}/{spec['task']}/{ns.frac_tag(spec['frac'])} not trained yet, skipped")
            continue
        if args.max_runs is not None and n_started >= args.max_runs:
            print(f"[pending] {spec}")
            continue
        n_started += 1
        print(f"[run] {spec['variant']}/{spec['task']}/{ns.frac_tag(spec['frac'])}: epoch_size {cfg.epoch_size} = "
              f"{steps_per_epoch(cfg)} steps/epoch, {EPOCHS * steps_per_epoch(cfg)} steps total")
        m = ns.train_one(cfg, device=args.device, **spec)
        if m is None:
            n_failed += 1
            continue
        post_best(cfg, **spec)
        for e in export_epochs:
            if not export_epoch(cfg, epoch=e, device=args.device, **spec):
                n_failed += 1
    print(f"summary: started={n_started} failed={n_failed}")
    return 2 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
