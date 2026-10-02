#!/usr/bin/env python
"""Best-validation vs last-epoch checkpoints under a schedule that runs to completion: EPOCHS x 2,000 steps,
cosine to 0, early stopping off; both checkpoints are exported (runs/<variant>/ and runs/<variant>_last/).
Root ~/datascan/anneal on the GPU box.

    python -u tools/anneal_test.py --smoke
    python -u tools/anneal_test.py --variants frozen,pretrained 2>&1 | tee -a logs/anneal_a.log
    python -u tools/anneal_test.py --measure
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
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

EPOCHS = 30
SEED = 0
TASKS = ("sig_over_bkg", "sbi_over_bkg")
ORDER = [("frozen", 1.0), ("pretrained", 1.0), ("frozen", 0.03), ("pretrained", 0.03), ("scratch", 1.0), ("scratch", 0.03)]
_STATE = {"epochs": EPOCHS}          # the epoch index at which the callback saves last.pt


# ---------------------------------------------------------------------------------------------- roots and config
def make_root(name: str, max_epochs: int):
    root = MAIN / name
    root.mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    link = root / "data"
    if link.is_symlink() and not link.exists():
        link.unlink()
    if not link.exists():
        try:
            os.symlink(MAIN / "data", link, target_is_directory=True)
        except FileExistsError:          # two drivers starting at the same instant
            pass
    cfg = ns.ScanConfig.load(str(MAIN)).with_root(str(root))
    cfg = dataclasses.replace(cfg, max_epochs=max_epochs, early_stop_patience=0,   # patience 0 = early stopping off in EveNet-Lite
                              keep_checkpoint=True, wandb_project=None)
    _STATE["epochs"] = max_epochs
    return root, cfg


# ---------------------------------------------------------------------------------------------- callback patch
_orig_make_history = ns.make_history_callback


def make_history_callback_anneal(path):
    """Same per-epoch history.json as the scan, plus the learning rate of every optimizer group per epoch, and a"""
    from evenet_lite.callbacks import Callback

    inner = _orig_make_history(path)
    out_dir = Path(path).parent

    class AnnealCallback(Callback):
        def __init__(self):
            self.inner = inner
            self.lrs = []

        @property
        def rows(self):
            return self.inner.rows

        def on_epoch_end(self, trainer, epoch: int, metrics) -> None:
            self.inner.on_epoch_end(trainer, epoch, metrics)
            lr = [float(pg["lr"]) for opt in getattr(trainer, "optimizers", []) for pg in opt.param_groups]
            self.lrs.append(lr)
            self.inner.rows[-1]["lr"] = lr
            with open(self.inner.out, "w") as fh:
                json.dump(self.inner.rows, fh, indent=1)
            print(f"[anneal] epoch {epoch + 1}/{_STATE['epochs']} lr={['%.3g' % v for v in lr]} "
                  f"val_loss={metrics.get('val_loss', float('nan')):.4f}", flush=True)
            if epoch + 1 == _STATE["epochs"]:
                trainer.save_checkpoint(str(out_dir / "last.pt"), extra={"epoch": int(epoch), "kind": "last"})
                print(f"[anneal] last-epoch weights saved to {out_dir / 'last.pt'}", flush=True)

    return AnnealCallback()


ns.make_history_callback = make_history_callback_anneal      # train_one looks the name up at call time


# ---------------------------------------------------------------------------------------------- last-epoch export
def export_last(cfg: ns.ScanConfig, variant: str, task: str, frac: float, seed: int, device: str = "cuda"):
    """Exports r(x) of last.pt into runs/<variant>_last/..."""
    import torch
    from evenet_lite import EvenetLiteClassifier
    from nsbi.data import FEATURE_NAMES

    src = ns.run_dir(cfg, variant, task, frac, seed)
    dst = ns.run_dir(cfg, f"{variant}_last", task, frac, seed)
    if (dst / "done.json").exists():
        print(f"[skip] {dst} already done")
        return True
    if not (src / "last.pt").exists():
        print(f"[FAILED] {dst}: {src / 'last.pt'} missing")
        return False
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "failed.json").unlink(missing_ok=True)
    t0 = time.time()
    try:
        ns.set_fast_math(False)
        clf = EvenetLiteClassifier(class_labels=["bkg", "sig"], device=device, sequential_input_dim=ns.SEQ_DIM,
                                   global_input_dim=ns.GLOBAL_DIM, pretrained=False)
        clf.load_checkpoint(str(src / "last.pt"), feature_names=FEATURE_NAMES)
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
        keep = ("subsample_seed", "n_num_train", "n_den_train", "ess_num_train", "ess_den_train", "n_num_val", "n_den_val",
                "n_train_rows", "n_val_monitor_rows", "n_trainable_params", "n_nonfinite_batches", "steps_per_epoch",
                "tf32_training", "autocast_bf16_training", "fit_seconds", "device", "config")
        metrics = {"variant": f"{variant}_last", "task": task, "frac": frac, "frac_tag": ns.frac_tag(frac), "seed": seed,
                   "checkpoint_kind": "last", "checkpoint": str(src / "last.pt"),
                   **{k: best_m[k] for k in keep if k in best_m},
                   "epochs_run": len(hist), "checkpoint_epoch": len(hist),
                   "best_epoch": best_m.get("best_epoch"), "best_step": best_m.get("best_step"),    # of the paired best run
                   "best_val_loss_monitor": best_m.get("best_val_loss_monitor"),
                   "total_steps": int(hist[-1].get("global_step") or 0),
                   "last_val_loss_monitor": float(hist[-1].get("val_loss", float("nan"))),
                   "lr_last_epoch": hist[-1].get("lr"), "resumed_from_checkpoint": False,
                   **vm, "C_ref": C_ref, "C_sub": C_sub, "C_val": C_val, "fast_predict_max_rel_dev": dev,
                   "closure_chi2_ndf": closure["closure_chi2_ndf"], "closure_chi2": closure["closure_chi2"],
                   "closure_ndf": closure["closure_ndf"], "toys": toy_info, "total_seconds": time.time() - t0}
        (dst / "metrics.json").write_text(json.dumps(ns._jsonable(metrics), indent=1, default=str))
        (dst / "closure.json").write_text(json.dumps(ns._jsonable(closure)))
        json.loads((dst / "metrics.json").read_text())
        shutil.copyfile(src / "history.json", dst / "history.json")
        (dst / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                   "seconds": time.time() - t0}))
        print(f"[done] {variant}_last/{task}/{ns.frac_tag(frac)}/seed{seed}: epoch {len(hist)} val_auc={vm['val_auc_full']:.4f} "
              f"C_ref={C_ref:.4f} chi2/ndf={closure['closure_chi2_ndf']:.2f} ({(time.time() - t0) / 60:.1f} min)")
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as exc:
        (dst / "failed.json").write_text(json.dumps({"error": repr(exc), "traceback": traceback.format_exc(),
                                                     "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        print(f"[FAILED] {variant}_last/{task}/{ns.frac_tag(frac)}/seed{seed}: {exc!r}")
        return False


def post_best(cfg: ns.ScanConfig, variant: str, task: str, frac: float, seed: int) -> bool:
    """After train_one: mark the best pseudo-run, hard-link best.pt, and check that the schedule reached its end."""
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
    ok = (bool(lrs) and lrs[-1] is not None and lrs[0] and len(hist) == cfg.max_epochs
          and lrs[-1] <= 1e-9 * lrs[0])                      # min_lr is 0 for all three recipes: the last epoch must sit there
    print(f"[anneal] {variant}/{task}/{ns.frac_tag(frac)}: {len(hist)} epochs, best epoch {m.get('best_epoch')}, "
          f"lr first->last {fmt(lrs[0] if lrs else None)} -> {fmt(lrs[-1] if lrs else None)}, schedule finished: {ok}")
    if not ok:
        print(f"[anneal] WARNING: schedule did not finish or lr not at its minimum for {out}")
    return ok


# ---------------------------------------------------------------------------------------------- measurement
def report(cfg: ns.ScanConfig):
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    runs = ns.collect_runs(cfg)
    if runs.empty:
        print("no finished runs under", cfg.runs_dir)
        return
    runs["model"] = runs.variant.str.replace("_last", "", regex=False)
    runs["checkpoint"] = np.where(runs.variant.str.endswith("_last"), "last", "best")
    have = runs.groupby(["model", "checkpoint", "frac"]).task.nunique()
    print("finished pairs (both classifiers):")
    print(have[have == 2].reset_index()[["model", "checkpoint", "frac"]].to_string(index=False))
    meas = ns.measure_all(cfg, runs, diagonal_only=True, calibration="C_ref",
                          cache_path=str(Path(cfg.root) / "measurements_anneal.csv"))
    if meas.empty:
        print("no complete pairs yet")
        return
    per_seed, _ = ns.aggregate(meas)
    keys = ["variant", "seed", "frac_sig", "frac_sbi", "mu_true"]
    spike = meas.groupby(keys).mu_hat.apply(lambda s: int(((s > 0.85) & (s < 1.15)).sum())).rename("n_spike").reset_index()
    per_seed = per_seed.merge(spike, on=keys)
    W = ns._denominator_weights(ns.load_split(cfg, "bkg", "val"), None)
    p = W / W.sum()
    rows = []
    for _, a in runs[runs.task == "sig_over_bkg"].iterrows():
        r = np.load(Path(a.run_dir) / "r_val.npz")["r_den"].astype(np.float64) * float(a.C_ref)
        big = r > 30
        rt = np.load(Path(a.run_dir) / "r_toys_mu0p4.npy").astype(np.float64) * float(a.C_ref)
        rows.append(dict(variant=a.variant, frac_sig=a.frac, tail_share=float((r[big] * p[big]).sum() / (r * p).sum()),
                         max_r_toys=float(rt.max()), best_epoch=int(a.best_epoch), bce=float(a.val_bce_full), auc=float(a.val_auc_full)))
    tails = pd.DataFrame(rows)
    t = per_seed.merge(tails, on=["variant", "frac_sig"], how="left")
    t["model"] = t.variant.str.replace("_last", "", regex=False)
    t["checkpoint"] = np.where(t.variant.str.endswith("_last"), "last", "best")
    cols = ["model", "checkpoint", "frac_sig", "mu_true", "bias", "mean_width", "n_open", "n_spike", "tail_share", "max_r_toys", "best_epoch", "bce", "auc"]
    t = t.sort_values(["mu_true", "frac_sig", "model", "checkpoint"])
    print("\n=== annealing test: 48 toys, calibration C_ref (tail_share = share of <r>_B on the validation background from r > 30) ===")
    print(t[cols].round(4).to_string(index=False))
    t[cols].to_csv(Path(cfg.root) / "report_anneal.csv", index=False)
    if not (MAIN / "aggregates_per_seed.csv").exists():
        print("(no scan reference table ~/datascan/aggregates_per_seed.csv on this machine)")
        return
    ref = pd.read_csv(MAIN / "aggregates_per_seed.csv")
    ref = ref[(ref.seed == 0) & np.isclose(ref.frac_sig, ref.frac_sbi) & ref.frac_sig.isin([1.0, 0.03]) & ref.variant.isin(["scratch", "pretrained", "frozen"])]
    print("\n=== the scan's own seed-0 result (early-stopped, best-val checkpoint) for comparison ===")
    print(ref[["variant", "frac_sig", "mu_true", "bias", "mean_width", "n_open"]].sort_values(["mu_true", "frac_sig", "variant"]).round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------- driver
def specs_for(variants):
    return [dict(variant=v, task=t, frac=f, seed=SEED) for v, f in ORDER for t in TASKS if v in variants]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="frozen,pretrained,scratch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()
    variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    unknown = variants - set(ns.VARIANTS)
    if unknown:
        print("unknown variants", sorted(unknown))
        return 1

    if args.smoke:
        root, cfg = make_root("anneal_smoke", 2)
        spec = dict(variant="frozen", task="sig_over_bkg", frac=0.01, seed=SEED)
        print(f"[smoke] root {root}: frozen S/B at 1 %, 2 epochs, early stopping off")
        m = ns.train_one(cfg, device=args.device, force=True, **spec)
        if m is None:
            print("[smoke] FAIL: training failed, see failed.json")
            return 2
        ok = post_best(cfg, **spec)
        if not export_last(cfg, device=args.device, **spec):
            print("[smoke] FAIL: last export failed")
            return 2
        runs = ns.collect_runs(cfg)
        a = np.load(ns.run_dir(cfg, "frozen", "sig_over_bkg", 0.01, SEED) / "r_toys_mu0p4.npy")
        b = np.load(ns.run_dir(cfg, "frozen_last", "sig_over_bkg", 0.01, SEED) / "r_toys_mu0p4.npy")
        best_epoch = int(m.get("best_epoch", -1))
        same = bool(np.allclose(a, b))
        expect_same = best_epoch == cfg.max_epochs        # best-val epoch == last epoch -> the two exports must agree
        print(f"[smoke] collect_runs variants: {sorted(runs.variant)} | best epoch {best_epoch} of {cfg.max_epochs} | "
              f"r_toys best vs last identical: {same} (expected {expect_same}; max rel dev "
              f"{float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-6))):.3g}) | schedule finished: {ok}")
        for f in ("best.pt", "last.pt", "history.json", "r_val.npz", "r_toys_mu2p5.npy", "metrics.json", "done.json"):
            print(f"   best/{f}: {(ns.run_dir(cfg, 'frozen', 'sig_over_bkg', 0.01, SEED) / f).exists()}", end="")
            print(f"   last/{f}: {(ns.run_dir(cfg, 'frozen_last', 'sig_over_bkg', 0.01, SEED) / f).exists()}")
        passed = ok and set(runs.variant) == {"frozen", "frozen_last"} and same == expect_same
        print("[smoke] PASS" if passed else "[smoke] CHECK THE LINES ABOVE")
        return 0 if passed else 3

    root, cfg = make_root("anneal", EPOCHS)
    if args.measure:
        report(cfg)
        return 0

    specs = specs_for(variants)
    print(f"root {root} | {len(specs)} runs for variants {sorted(variants)} | {EPOCHS} epochs x {-(-cfg.epoch_size // cfg.batch_size)} steps, "
          f"early stopping off, both checkpoints exported")
    n_started, n_failed = 0, 0
    for spec in specs:
        best_done = (ns.run_dir(cfg, **spec) / "done.json").exists()
        last_done = (ns.run_dir(cfg, spec["variant"] + "_last", spec["task"], spec["frac"], spec["seed"]) / "done.json").exists()
        if best_done and last_done:
            print(f"[skip] {spec['variant']}/{spec['task']}/{ns.frac_tag(spec['frac'])} done (best + last)")
            continue
        if args.max_runs is not None and n_started >= args.max_runs:
            print(f"[pending] {spec}")
            continue
        n_started += 1
        m = ns.train_one(cfg, device=args.device, **spec)
        if m is None:
            n_failed += 1
            continue
        post_best(cfg, **spec)
        if not export_last(cfg, device=args.device, **spec):
            n_failed += 1
    print(f"summary: started={n_started} failed={n_failed}")
    return 2 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
