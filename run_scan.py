#!/usr/bin/env python
"""Terminal driver for the scan on a Linux GPU box; same tiers and ordering as notebook 01.

    python run_scan.py --status
    python run_scan.py --smoke
    python -u run_scan.py --tier tier1 [--variants scratch] 2>&1 | tee -a logs/tier1.log

Finished runs (done.json) are skipped on relaunch.
Exit codes: 0 all done, 1 setup problem, 2 a run failed, 3 runs still pending.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")          # no progress-bar spam in log files
try:
    sys.stdout.reconfigure(line_buffering=True)     # progress prints reach the tee'd log immediately
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_evenet = os.environ.get("EVENET_LITE_DIR", str(Path.home() / "EveNet-Lite"))
if os.path.isdir(_evenet) and _evenet not in sys.path:
    sys.path.insert(0, _evenet)                     # the `nsbi` package is not pip-installed, only evenet_lite is

import nsbi_scan as ns  # noqa: E402


def tier_specs(cfg: ns.ScanConfig, tier: str):
    """Same presets and cheap-first ordering as notebook 01."""
    tiers = {
        "tier1": ns.run_matrix(cfg, variants=["scratch", "pretrained"], seeds=[0]),
        "tier2": ns.run_matrix(cfg, variants=["scratch", "pretrained"], fractions=[0.01, 0.03, 0.10], seeds=[1, 2])
                 + ns.run_matrix(cfg, variants=["frozen"], seeds=[0]),
        "tier3": ns.run_matrix(cfg, variants=["scratch", "pretrained"], fractions=[0.30, 1.0], seeds=[1, 2]),
    }
    tiers["all"] = tiers["tier1"] + tiers["tier2"] + tiers["tier3"]

    def cheap_first(spec):
        return (spec["frac"], spec["seed"], ns.VARIANTS.index(spec["variant"]), ns.TASKS.index(spec["task"]))

    return sorted(tiers[tier], key=cheap_first)


def check_data_cache(cfg: ns.ScanConfig) -> list:
    needed = ([f"{p}_{s}.npz" for p in ns.PROCESSES for s in ("train", "val")] + ["summary.json", "xs.json"]
              + [f"toys_{ns.MU_TAGS[m]}.npz" for m in cfg.mu_true_values])
    return [f for f in needed if not (cfg.data_dir / f).exists()]


class _SoftLoadCatcher(logging.Handler):
    """Captures EveNet-Lite's 'Loaded N / M layers' line so a silent HF download failure cannot pass."""
    pat = re.compile(r"Loaded (\d+) / (\d+) layers")

    def __init__(self):
        super().__init__()
        self.loaded = None

    def emit(self, record):
        m = self.pat.search(record.getMessage())
        if m:
            self.loaded = (int(m.group(1)), int(m.group(2)))


def _fail(msg: str) -> None:
    raise SystemExit(f"[smoke] FAIL: {msg}")


def smoke(cfg: ns.ScanConfig, device: str) -> None:
    """Three 1 % points with a tiny budget (scratch S/B, pretrained SBI/B, pretrained S/B), then a measurement of"""
    import torch
    sroot = Path(cfg.root) / "smoke"
    sroot.mkdir(exist_ok=True)
    link = sroot / "data"
    if link.is_symlink():
        link.unlink()                                 # stale or dangling link from an earlier tree
    if not link.exists():
        os.symlink(Path(cfg.data_dir).resolve(), link, target_is_directory=True)
    cfg_s = dataclasses.replace(cfg, root=str(sroot), epoch_size=51_200, max_epochs=2, early_stop_patience=0,
                                val_monitor_rows=50_000, calib_ref_events=50_000, wandb_project=None)
    print(f"[smoke] root {sroot} | device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    t0 = time.time()

    def run(variant, task, catcher=None):
        if catcher is not None:
            logging.getLogger().addHandler(catcher)
        try:
            m = ns.train_one(cfg_s, variant, task, 0.01, 0, device=device, force=True)
        finally:
            if catcher is not None:
                logging.getLogger().removeHandler(catcher)
        if m is None:
            _fail(f"{variant}/{task} run failed, see {ns.run_dir(cfg_s, variant, task, 0.01, 0) / 'failed.json'}")
        return m

    m1 = run("scratch", "sig_over_bkg")
    catcher = _SoftLoadCatcher()
    m2 = run("pretrained", "sbi_over_bkg", catcher)
    if not catcher.loaded or catcher.loaded[0] < 400:
        _fail(f"pretrained weights did not load from Hugging Face: {catcher.loaded}")
    m3 = run("pretrained", "sig_over_bkg")
    runs = ns.collect_runs(cfg_s)
    meas = ns.measure_all(cfg_s, runs, diagonal_only=True)
    per_seed, over = ns.aggregate(meas)
    if meas.empty or over.empty or set(over.variant) != {"pretrained"} or len(meas) != 96:
        _fail(f"measurement produced {len(meas)} rows for variants {sorted(set(over.variant)) if len(over) else []}")
    print(over.to_string())
    print(f"[smoke] fast-vs-slow prediction max rel dev: {m1['fast_predict_max_rel_dev']:.2e}, "
          f"{m2['fast_predict_max_rel_dev']:.2e}, {m3['fast_predict_max_rel_dev']:.2e}")
    print(f"[smoke] PASS in {(time.time() - t0) / 60:.1f} min ({catcher.loaded[0]} / {catcher.loaded[1]} layers loaded)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ns.resolve_root(str(HERE)), help="scan tree (holds scan_config.json, data/, runs/)")
    p.add_argument("--tier", default="tier1", choices=["tier1", "tier2", "tier3", "all"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--stop-after-hours", type=float, default=None, help="do not START new runs after this wall time")
    p.add_argument("--max-runs", type=int, default=None, help="start at most this many new runs")
    p.add_argument("--no-retry-failed", action="store_true", help="skip runs that already have failed.json")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--variants", default=None,
                   help="comma-separated subset of variants, e.g. scratch or pretrained,frozen; lets two drivers "
                        "share one GPU without picking the same run")
    p.add_argument("--tf32", dest="tf32", action="store_true", default=None, help="override cfg.tf32 = True")
    p.add_argument("--no-tf32", dest="tf32", action="store_false", help="override cfg.tf32 = False")
    p.add_argument("--bf16", dest="autocast_bf16", action="store_true", default=None, help="override cfg.autocast_bf16 = True")
    p.add_argument("--no-bf16", dest="autocast_bf16", action="store_false", help="override cfg.autocast_bf16 = False")
    p.add_argument("--status", action="store_true", help="print the run table and exit")
    p.add_argument("--smoke", action="store_true", help="run the 1%% smoke test and exit")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = str(Path(args.root).resolve())
    cfg = ns.ScanConfig.load(root).with_root(root)
    overrides = {k: v for k, v in {"wandb_project": args.wandb_project, "wandb_entity": args.wandb_entity,
                                   "num_workers": args.num_workers, "tf32": args.tf32,
                                   "autocast_bf16": args.autocast_bf16}.items() if v is not None}
    cfg = dataclasses.replace(cfg, **overrides)
    steps = -(-cfg.epoch_size // cfg.batch_size)
    print(f"root {cfg.root}\nbudget: {steps} steps/epoch x max {cfg.max_epochs} epochs, patience {cfg.early_stop_patience}, "
          f"num_workers {cfg.num_workers}, wandb {cfg.wandb_project}, tf32 {cfg.tf32}, bf16 {cfg.autocast_bf16}")

    specs = tier_specs(cfg, args.tier)
    if args.variants:
        keep = {v.strip() for v in args.variants.split(',') if v.strip()}
        unknown = keep - set(ns.VARIANTS)
        if unknown:
            print(f'unknown variants {sorted(unknown)}; choose from {ns.VARIANTS}')
            return 1
        specs = [sp for sp in specs if sp['variant'] in keep]
    if args.status:
        status = ns.scan_status(cfg, specs)
        print(f"tier {args.tier}: {len(specs)} runs -> " + ", ".join(f"{k}={v}" for k, v in status.status.value_counts().items()))
        print(status.to_string(index=False))
        return 0

    missing = check_data_cache(cfg)
    if missing:
        print(f"data cache INCOMPLETE under {cfg.data_dir}: {missing}")
        return 1
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print("no CUDA device visible to torch: check nvidia-smi / the driver, or pass --device auto")
            return 1

    if args.smoke:
        smoke(cfg, args.device)
        return 0

    status = ns.scan_status(cfg, specs)
    print(f"tier {args.tier}: {len(specs)} runs -> " + ", ".join(f"{k}={v}" for k, v in status.status.value_counts().items()))
    if cfg.wandb_project:
        import wandb
        if not wandb.login(prompt=False):           # WANDB_API_KEY env var or ~/.netrc only; never prompts
            print("W&B: no API key found. Run `wandb login` once in this shell, or export WANDB_API_KEY, "
                  "or drop --wandb-project.")
            return 1
    stop = args.stop_after_hours * 3600 if args.stop_after_hours is not None else None
    result = ns.run_driver(cfg, specs, device=args.device, max_runs=args.max_runs,
                           stop_after_seconds=stop, retry_failed=not args.no_retry_failed)
    print(result.to_string(index=False))
    print("summary: " + ", ".join(f"{k}={v}" for k, v in result.status.value_counts().items()))
    if (result.status == "failed").any():
        return 2
    return 3 if (result.status == "pending").any() else 0


if __name__ == "__main__":
    sys.exit(main())
