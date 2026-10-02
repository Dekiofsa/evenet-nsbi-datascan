"""Shared code for the EveNet NSBI training-statistics scan: data cache and nested subsampling,
training with ratio exports, calibration, likelihood scan and toy measurement, aggregation.
torch and EveNet-Lite are imported lazily, so the measurement half runs with numpy and pandas only.
"""
from __future__ import annotations

import functools
import logging
import glob
import json
import math
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KIN_COLS = [f"l{i}_{v}" for i in range(1, 5) for v in ("pt", "eta", "phi", "energy")]
MSQ_COLS = ["msq_sig_sm", "msq_int_sm", "msq_sbi_sm", "msq_bkg_sm"]
PROCESSES = ("sig", "bkg", "sbi")
TASKS = ("sig_over_bkg", "sbi_over_bkg")
VARIANTS = ("scratch", "pretrained", "frozen")
MU_TAGS = {0.4: "mu0p4", 2.5: "mu2p5"}
SEQ_DIM = 4       # per-lepton features [pt, energy, eta, phi]  (nsbi/train.py)
GLOBAL_DIM = 3    # [m4l, sum_pt, sum_energy]

# Seeds that must NOT change between runs (they define the fixed reference sets)
SPLIT_SEED = 42            # nsbi.data.split_process default, same as the poster study
VAL_MONITOR_SEED = 12345   # fixed validation-monitoring subset
CALIB_REF_SEED = 777       # fixed background reference subset for the calibration factor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScanConfig:
    """All knobs of the scan."""

    # --- locations -------------------------------------------------------
    root: str = "/content/drive/MyDrive/Research/NSBI/datascan"
    raw_data_dir: str = "/content/drive/MyDrive/Research/NSBI/spliced_21_columns"
    toys_root: str = "/content/drive/MyDrive/Research/NSBI"   # holds observed_mu0p4/, observed_mu2p5/

    # --- data ------------------------------------------------------------
    train_frac: float = 0.8
    split_seed: int = SPLIT_SEED
    fractions: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.10, 0.30, 1.00])
    variants: List[str] = field(default_factory=lambda: ["scratch", "pretrained", "frozen"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    tasks: List[str] = field(default_factory=lambda: list(TASKS))

    # --- training budget (identical for every point) ---------------------
    batch_size: int = 512
    epoch_size: int = 1_024_000        # rows drawn per "epoch" = 2000 optimizer steps at batch 512
    max_epochs: int = 50               # cap = 100k optimizer steps
    early_stop_patience: int = 5       # epochs without val_loss improvement
    val_monitor_rows: int = 262_144    # fixed validation subset used for early stopping
    num_workers: int = 0

    # --- variant recipes -------------------------------------------------
    scratch_lr: float = 3e-4           # the authors' 1e-3 diverged at 3 %
    scratch_weight_decay: float = 1e-2

    tf32: bool = False
    autocast_bf16: bool = False

    # --- measurement -----------------------------------------------------
    lumi: float = 300.0
    mu_true_values: List[float] = field(default_factory=lambda: [0.4, 2.5])
    mu_min: float = 0.0
    mu_max: float = 4.0
    mu_steps: int = 401
    calib_ref_events: int = 1_000_000  # size of the fixed bkg_train subset used for C = 1/<r>_B

    # --- housekeeping ----------------------------------------------------
    keep_checkpoint: bool = False      # keep final.pt after r(x) export (80-240 MB per run)
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> str:
        path = path or str(Path(self.root) / "scan_config.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @classmethod
    def load(cls, root_or_path: str) -> "ScanConfig":
        p = Path(root_or_path)
        if p.is_dir():
            p = p / "scan_config.json"
        with open(p) as fh:
            d = json.load(fh)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        return cfg

    def with_root(self, root: str) -> "ScanConfig":
        return replace(self, root=root)

    # --- paths -----------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return Path(self.root) / "data"

    @property
    def runs_dir(self) -> Path:
        return Path(self.root) / "runs"

    @property
    def figures_dir(self) -> Path:
        return Path(self.root) / "figures"

    @property
    def mu_grid(self) -> np.ndarray:
        return np.linspace(self.mu_min, self.mu_max, self.mu_steps)


def resolve_root(default: str) -> str:
    """Root of the scan tree: env var NSBI_SCAN_ROOT wins (for local runs), else default."""
    return os.environ.get("NSBI_SCAN_ROOT", default)


# ---------------------------------------------------------------------------
# Process container (duck-type compatible with nsbi.data.Process)
# ---------------------------------------------------------------------------

@dataclass
class Process:
    kinematics: pd.DataFrame      # (N, 16) columns KIN_COLS
    components: pd.DataFrame      # (N, 4) columns MSQ_COLS (empty for toys)
    weights: pd.Series            # (N,)

    def __len__(self) -> int:
        return len(self.kinematics)

    def take(self, idx: np.ndarray) -> "Process":
        comp = self.components.iloc[idx].reset_index(drop=True) if len(self.components) else self.components
        return Process(
            kinematics=self.kinematics.iloc[idx].reset_index(drop=True),
            components=comp,
            weights=self.weights.iloc[idx].reset_index(drop=True),
        )


def process_from_arrays(kin: np.ndarray, msq: Optional[np.ndarray], wt: np.ndarray) -> Process:
    comp = pd.DataFrame(msq, columns=MSQ_COLS) if msq is not None else pd.DataFrame()
    return Process(pd.DataFrame(kin, columns=KIN_COLS), comp, pd.Series(wt))


def m4l(kin: pd.DataFrame) -> np.ndarray:
    """Four-lepton invariant mass (numpy port of nsbi.data._m4l)."""
    E = sum(kin[f"l{i}_energy"].values for i in range(1, 5))
    px = sum(kin[f"l{i}_pt"].values * np.cos(kin[f"l{i}_phi"].values) for i in range(1, 5))
    py = sum(kin[f"l{i}_pt"].values * np.sin(kin[f"l{i}_phi"].values) for i in range(1, 5))
    pz = sum(kin[f"l{i}_pt"].values * np.sinh(kin[f"l{i}_eta"].values) for i in range(1, 5))
    return np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0))


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    s2 = float((w ** 2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def _jsonable(o):
    """Recursively convert tensors / numpy / paths into plain JSON-serialisable Python objects."""
    if hasattr(o, "detach"):
        o = o.detach().cpu()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    return o


# ---------------------------------------------------------------------------
# Data caching (notebook 00)
# ---------------------------------------------------------------------------

def _resolve_csv(data_dir: str, proc: str) -> Path:
    base = Path(data_dir) / f"ggZZ_{proc}_spliced.csv"
    for cand in (base, Path(str(base) + ".gz")):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"{base}[.gz] not found")


def split_indices(n: int, train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exactly nsbi.data.split_process: permutation with default_rng(seed), first int(n*frac) = train."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    return idx[:split], idx[split:]


def cache_splits(cfg: ScanConfig, processes: Sequence[str] = PROCESSES, force: bool = False) -> Dict:
    """Read the MCFM CSVs once, apply the fixed train/val split, cache as npz."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.data_dir / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    for proc in processes:
        out_train = cfg.data_dir / f"{proc}_train.npz"
        out_val = cfg.data_dir / f"{proc}_val.npz"
        if out_train.exists() and out_val.exists() and not force:
            if f"{proc}_train" not in summary or f"{proc}_val" not in summary or f"{proc}_total_rows" not in summary:
                for name, out in (("train", out_train), ("val", out_val)):
                    w = np.load(out)["wt"]
                    summary[f"{proc}_{name}"] = {"rows": int(len(w)), "sum_wt": float(w.sum()),
                                                 "ess": effective_sample_size(w), "min_wt": float(w.min())}
                summary[f"{proc}_total_rows"] = summary[f"{proc}_train"]["rows"] + summary[f"{proc}_val"]["rows"]
            print(f"[cache_splits] {proc}: cached")
            continue
        csv = _resolve_csv(cfg.raw_data_dir, proc)
        t0 = time.time()
        df = pd.read_csv(csv, usecols=KIN_COLS + MSQ_COLS + ["wt"])
        n = len(df)
        tr, va = split_indices(n, cfg.train_frac, cfg.split_seed)
        for name, idx, out in (("train", tr, out_train), ("val", va, out_val)):
            sub = df.iloc[idx]
            np.savez(out,
                     kin=sub[KIN_COLS].to_numpy(np.float32),
                     msq=sub[MSQ_COLS].to_numpy(np.float64),
                     wt=sub["wt"].to_numpy(np.float64))
            summary[f"{proc}_{name}"] = {
                "rows": int(len(sub)),
                "sum_wt": float(sub["wt"].sum()),
                "ess": effective_sample_size(sub["wt"].to_numpy()),
                "min_wt": float(sub["wt"].min()),
            }
        summary[f"{proc}_total_rows"] = int(n)
        print(f"[cache_splits] {proc}: {n} rows -> train {len(tr)} / val {len(va)}  ({time.time()-t0:.0f}s)")
        del df
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


_SPLIT_CACHE: Dict[Tuple[str, str, str], Process] = {}


def load_split(cfg: ScanConfig, proc: str, part: str, use_cache: bool = True) -> Process:
    """Cached split loader: the driver trains many runs per session, and Drive reads are slow (~0.6 GB per run)."""
    key = (str(cfg.data_dir), proc, part)
    if use_cache and key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]
    d = np.load(cfg.data_dir / f"{proc}_{part}.npz")
    p = process_from_arrays(d["kin"], d["msq"], d["wt"])
    if use_cache:
        _SPLIT_CACHE[key] = p
    return p


def clear_split_cache() -> None:
    _SPLIT_CACHE.clear()


def cross_sections_from_sbi(components: pd.DataFrame, weights: pd.Series) -> Dict[str, float]:
    """Port of nsbi.measure._cross_sections: xs[comp] = sum(w * msq_comp / msq_sbi)  [fb]."""
    w = weights.to_numpy(np.float64)
    msq_sbi = components["msq_sbi_sm"].to_numpy(np.float64)
    return {c: float((w * components[f"msq_{c}_sm"].to_numpy(np.float64) / msq_sbi).sum())
            for c in ("sig", "int", "bkg", "sbi")}


def cache_cross_sections(cfg: ScanConfig) -> Dict[str, float]:
    """Cross sections from the FULL SBI sample (train + val), as nsbi.measure does with proc_sbi."""
    parts = [np.load(cfg.data_dir / f"sbi_{p}.npz") for p in ("train", "val")]
    comp = pd.DataFrame(np.concatenate([p["msq"] for p in parts]), columns=MSQ_COLS)
    wt = pd.Series(np.concatenate([p["wt"] for p in parts]))
    xs = cross_sections_from_sbi(comp, wt)
    with open(cfg.data_dir / "xs.json", "w") as fh:
        json.dump(xs, fh, indent=2)
    return xs


def load_cross_sections(cfg: ScanConfig) -> Dict[str, float]:
    return json.load(open(cfg.data_dir / "xs.json"))


def expected_yield(xs: Dict[str, float], mu: float, lumi: float) -> float:
    return lumi * (mu * xs["sig"] + math.sqrt(mu) * xs["int"] + xs["bkg"])


def _toy_files(cfg: ScanConfig, tag: str) -> List[str]:
    files = glob.glob(str(Path(cfg.toys_root) / f"observed_{tag}" / "observed_*.csv"))
    files.sort(key=lambda p: int(re.search(r"observed_(\d+)\.csv$", p).group(1)))
    if not files:
        raise FileNotFoundError(f"no toys under {Path(cfg.toys_root) / f'observed_{tag}'}")
    return files


def cache_toys(cfg: ScanConfig, force: bool = False) -> Dict[str, Dict]:
    """Concatenate the 48 toy CSVs per mu into <root>/data/toys_<tag>.npz (kin, n, toy_id)."""
    info = {}
    for mu_true, tag in MU_TAGS.items():
        if mu_true not in cfg.mu_true_values:
            continue
        out = cfg.data_dir / f"toys_{tag}.npz"
        if out.exists() and not force:
            d = np.load(out)
            info[tag] = {"n_toys": int(d["toy_id"].max()) + 1, "rows": int(len(d["n"])), "sum_n": float(d["n"].sum())}
            continue
        kin, n, tid = [], [], []
        for i, f in enumerate(_toy_files(cfg, tag)):
            df = pd.read_csv(f, usecols=KIN_COLS + ["n"])
            kin.append(df[KIN_COLS].to_numpy(np.float32))
            n.append(df["n"].to_numpy(np.float64))
            tid.append(np.full(len(df), i, dtype=np.int16))
        kin, n, tid = np.concatenate(kin), np.concatenate(n), np.concatenate(tid)
        np.savez(out, kin=kin, n=n, toy_id=tid, mu_true=np.float64(mu_true))
        info[tag] = {"n_toys": int(tid.max()) + 1, "rows": int(len(n)), "sum_n": float(n.sum())}
        print(f"[cache_toys] {tag}: {info[tag]}")
    return info


def load_toys(cfg: ScanConfig, tag: str, with_kin: bool = True) -> Dict[str, np.ndarray]:
    """with_kin=False skips the 185 MB kinematics array (the measurement only needs n and toy_id)."""
    d = np.load(cfg.data_dir / f"toys_{tag}.npz")
    out = {"n": d["n"], "toy_id": d["toy_id"], "mu_true": float(d["mu_true"])}
    if with_kin:
        out["kin"] = d["kin"]
    return out


def check_toys(cfg: ScanConfig, xs: Dict[str, float], tol: float = 0.01) -> pd.DataFrame:
    """Sum of n per toy must match the expected yield nu(mu_true) to within tol (relative)."""
    rows = []
    for mu_true, tag in MU_TAGS.items():
        if mu_true not in cfg.mu_true_values:
            continue
        toys = load_toys(cfg, tag)
        nu = expected_yield(xs, mu_true, cfg.lumi)
        for t in np.unique(toys["toy_id"]):
            m = toys["toy_id"] == t
            s = float(toys["n"][m].sum())
            rows.append({"tag": tag, "toy": int(t), "rows": int(m.sum()), "sum_n": s, "nu_expected": nu,
                         "rel_dev": s / nu - 1.0})
    df = pd.DataFrame(rows)
    bad = df[df.rel_dev.abs() > tol]
    if len(bad):
        raise AssertionError(f"{len(bad)} toys deviate from nu(mu) by more than {tol:.0%}:\n{bad}")
    return df


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_indices(n: int, frac: float, seed: int) -> np.ndarray:
    """Nested subsample: sorted prefix of one seeded permutation (same seed -> nested across fractions)."""
    k = n if frac >= 1.0 else max(1, int(round(frac * n)))
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[:k])


def subsample(process: Process, frac: float, seed: int) -> Process:
    if frac >= 1.0:
        return process
    return process.take(subsample_indices(len(process), frac, seed))


def subsample_seed(seed: int) -> int:
    """Permutation seed derived from the run seed (kept distinct from the torch seed)."""
    return 1000 + seed


def frac_tag(frac: float) -> str:
    """f0010 for 1%, f0100 for 10%, f1000 for 100%."""
    return f"f{int(round(frac * 1000)):04d}"


def frac_from_tag(tag: str) -> float:
    return int(tag[1:]) / 1000.0


def run_dir(cfg: ScanConfig, variant: str, task: str, frac: float, seed: int) -> Path:
    return cfg.runs_dir / variant / task / frac_tag(frac) / f"seed{seed}"


def run_matrix(cfg: ScanConfig, variants=None, fractions=None, seeds=None, tasks=None,
               seeds_by_fraction: Optional[Dict[float, List[int]]] = None) -> List[Dict]:
    """List of run specs."""
    specs = []
    for variant in (variants or cfg.variants):
        for frac in (fractions or cfg.fractions):
            fr_seeds = (seeds_by_fraction or {}).get(frac, seeds or cfg.seeds)
            for seed in fr_seeds:
                for task in (tasks or cfg.tasks):
                    specs.append({"variant": variant, "task": task, "frac": float(frac), "seed": int(seed)})
    return specs


def training_event_table(cfg: ScanConfig) -> pd.DataFrame:
    """Training events per hypothesis at each fraction (from data/summary.json)."""
    s = json.load(open(cfg.data_dir / "summary.json"))
    rows = []
    for f in cfg.fractions:
        row = {"fraction": f}
        for proc in PROCESSES:
            n = s[f"{proc}_train"]["rows"]
            row[f"{proc}_train_events"] = n if f >= 1.0 else int(round(f * n))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Training (Colab / Perlmutter only - needs torch + EveNet-Lite)
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_variant(name: str, cfg: ScanConfig, device: str = "auto"):
    """scratch: same EveNet-Lite architecture, random init, authors' scratch recipe."""
    import logging
    from evenet_lite import EvenetLiteClassifier
    from nsbi.train import build_classifier, build_classifier_ssl

    if name == "pretrained":
        clf = build_classifier(device=device)
    elif name == "frozen":
        clf = build_classifier_ssl(device=device)
    elif name == "scratch":
        clf = EvenetLiteClassifier(
            class_labels=["bkg", "sig"],
            device=device,
            lr=[cfg.scratch_lr],
            weight_decay=cfg.scratch_weight_decay,
            module_lists=[["Classification", "ObjectEncoder", "PET", "GlobalEmbedding"]],
            grad_clip=1.0,
            warmup_epochs=1,
            warmup_ratio=0.1,
            warmup_start_factor=0.1,
            sequential_input_dim=SEQ_DIM,
            global_input_dim=GLOBAL_DIM,
            pretrained=False,
            loss_gamma=0.0,
            log_level=logging.INFO,
        )
    else:
        raise ValueError(f"unknown variant {name!r}; expected one of {VARIANTS}")

    clf.config.compute_physics_metrics = False   # per-epoch CPU AUC/SIC pass is expensive; we compute AUC once
    clf.config.num_workers = cfg.num_workers
    install_nan_guard()
    return clf


def count_trainable(clf) -> int:
    return int(sum(p.numel() for p in clf.model.parameters() if p.requires_grad))


_NAN_GUARD = {"installed": False, "consecutive": 0, "total": 0, "limit": 20}


def install_nan_guard(limit: int = 20) -> None:
    """EveNet-Lite aborts a whole run on a single non-finite batch loss (trainer._run_epoch returns None and train()"""
    import torch
    import evenet_lite.trainer as tr
    _NAN_GUARD["limit"] = int(limit)
    if _NAN_GUARD["installed"]:
        return
    orig = tr.compute_loss

    def guarded(logits, targets, weights, *args, **kwargs):
        loss = orig(logits, targets, weights, *args, **kwargs)
        if bool(torch.isfinite(loss).all()):
            _NAN_GUARD["consecutive"] = 0
            return loss
        _NAN_GUARD["consecutive"] += 1
        _NAN_GUARD["total"] += 1
        logging.warning("non-finite loss in a batch (%d consecutive, %d in this run): update skipped",
                        _NAN_GUARD["consecutive"], _NAN_GUARD["total"])
        if _NAN_GUARD["consecutive"] >= _NAN_GUARD["limit"]:
            raise RuntimeError(f"training diverged: {_NAN_GUARD['limit']} consecutive non-finite losses")
        return torch.zeros((), device=loss.device, dtype=loss.dtype, requires_grad=True)

    tr.compute_loss = guarded
    _NAN_GUARD["installed"] = True


_FAST_MATH = {"tf32": False, "autocast_bf16": False, "active": False, "patched": False}


def set_fast_math(enabled: bool, tf32: bool = True, autocast_bf16: bool = False) -> None:
    """Enable TF32 matmuls (and optionally bf16 autocast in Trainer._forward) for TRAINING; call with enabled=False"""
    import torch
    on = bool(enabled)
    _FAST_MATH.update(tf32=bool(tf32), autocast_bf16=bool(autocast_bf16), active=on)
    use_tf32 = on and bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    try:
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    except Exception:
        pass
    if on and autocast_bf16:
        _install_autocast_patch()


def _install_autocast_patch() -> None:
    """Wrap EveNet-Lite's Trainer._forward in bf16 autocast while _FAST_MATH is active (training only)."""
    import torch
    import evenet_lite.trainer as tr
    if _FAST_MATH["patched"]:
        return
    orig = tr.Trainer._forward

    def fwd(self, model, features):
        if _FAST_MATH["active"] and _FAST_MATH["autocast_bf16"] and torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = orig(self, model, features)
            return out.float()
        return orig(self, model, features)

    tr.Trainer._forward = fwd
    _FAST_MATH["patched"] = True


def make_history_callback(path: Path):
    """Callback that appends the per-epoch merged metrics to history.json (survives disconnects)."""
    from evenet_lite.callbacks import Callback

    class HistoryCallback(Callback):
        def __init__(self, out: Path):
            self.out = Path(out)
            self.rows: List[Dict] = []
            self.t0 = time.time()

        def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> None:
            row = {"epoch": int(epoch) + 1, "global_step": int(getattr(trainer, "global_step", -1)),
                   "elapsed_s": time.time() - self.t0}
            row.update({k: float(v) for k, v in (metrics or {}).items() if isinstance(v, (int, float))})
            self.rows.append(row)
            with open(self.out, "w") as fh:
                json.dump(self.rows, fh, indent=1)

    return HistoryCallback(path)


def make_sampler_seed_callback(seed: int):
    """EveNet-Lite seeds its weighted sampler with the epoch index only (evenet_lite/data.py), so two runs on the"""
    from evenet_lite.callbacks import Callback

    class SamplerSeedCallback(Callback):
        def __init__(self, s: int):
            self.s = int(s)

        def on_epoch_start(self, trainer, epoch: int) -> None:
            sampler = getattr(trainer, "train_sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(int(epoch) + 100_000 * (self.s + 1))

    return SamplerSeedCallback(seed)


def task_processes(cfg: ScanConfig, task: str, frac: float, seed: int) -> Dict[str, Process]:
    """Training subsamples + full validation Processes for one task."""
    ss = subsample_seed(seed)
    if task == "sig_over_bkg":
        return {
            "num_train": subsample(load_split(cfg, "sig", "train"), frac, ss),
            "den_train": subsample(load_split(cfg, "bkg", "train"), frac, ss),
            "num_val": load_split(cfg, "sig", "val"),
            "den_val": load_split(cfg, "bkg", "val"),
            "reweight": None,
        }
    if task == "sbi_over_bkg":
        sbi_train = subsample(load_split(cfg, "sbi", "train"), frac, ss)
        sbi_val = load_split(cfg, "sbi", "val")
        return {"num_train": sbi_train, "den_train": sbi_train, "num_val": sbi_val, "den_val": sbi_val,
                "reweight": ("sbi", "bkg")}
    raise ValueError(task)


def _classifier_data(num: Process, den: Process, reweight):
    from nsbi.data import make_classifier_data
    return make_classifier_data(num, den, denominator_reweight=reweight)


def _subset_rows(features, labels, weights, n_rows: int, seed: int):
    import torch
    n = len(labels)
    if n_rows >= n:
        return features, labels, weights
    idx = torch.as_tensor(np.sort(np.random.default_rng(seed).permutation(n)[:n_rows]))
    return {k: v[idx] for k, v in features.items()}, labels[idx], weights[idx]


def _denominator_weights(proc: Process, reweight) -> np.ndarray:
    w = proc.weights.to_numpy(np.float64)
    if reweight is not None:
        a, b = reweight
        w = w * proc.components[f"msq_{b}_sm"].to_numpy(np.float64) / proc.components[f"msq_{a}_sm"].to_numpy(np.float64)
    return w


def predict_logits_fast(clf, features: Dict, batch_size: int = 8192):
    """GPU-batched inference without EveNet-Lite's per-sample DataLoader."""
    import torch
    trainer = getattr(clf, "trainer", None)
    if trainer is None or clf.normalizer is None:
        raise RuntimeError("classifier must be fitted or loaded from a checkpoint before predicting")
    model = trainer._unwrap_model()
    device = trainer.device
    model.to(device)
    feats = {k: torch.as_tensor(v) for k, v in features.items()}
    with torch.no_grad():
        feats = clf.normalizer.transform(feats)          # mirrors EvenetTensorDataset._prepare_features
    n = len(next(iter(feats.values())))
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = {}
            for k, v in feats.items():
                t = v[i:i + batch_size].to(device, non_blocking=True)
                if k in ("x", "globals"):                 # mirrors Trainer._prepare_features
                    t = t.float()
                batch[k] = t
            logits = trainer._forward(model, batch)
            if logits.dim() == 3:                         # ensemble [E, B, C] -> [B, C], as _collect_predictions
                logits = logits.mean(dim=0)
            out.append(logits.detach().float().cpu())
    if was_training:
        model.train()
    return torch.cat(out, dim=0)


def _r_from_logits(logits) -> np.ndarray:
    """Likelihood-ratio trick exactly as nsbi.measure.compute_r: r = s / (1 - s), s = softmax(logits)[:, 1]."""
    import torch
    probs = torch.softmax(logits.float(), dim=1)
    s = probs[:, 1]
    r = s / (1.0 - s.clamp(max=1.0 - 1e-7))
    return r.numpy().astype(np.float64)


def compute_r_np(clf, process: Process, batch_size: int = 8192, fast: bool = True) -> np.ndarray:
    """Per-event density ratio r(x) for a Process."""
    from nsbi.data import to_evenet_tensors
    feats = to_evenet_tensors(process)
    if fast:
        logits = predict_logits_fast(clf, feats, batch_size=batch_size)
    else:
        logits = clf.predict(feats, batch_size=min(batch_size, 1024))
    return _r_from_logits(logits)


def check_fast_predict(clf, process: Process, n: int = 4096, tol: float = 5e-2) -> float:
    """Compare the fast path with EveNet-Lite's predict() on a small subset; returns the max relative deviation."""
    sub = process.take(np.arange(min(n, len(process))))
    r_fast = compute_r_np(clf, sub, fast=True)
    r_slow = compute_r_np(clf, sub, fast=False)
    max_rel = float(np.max(np.abs(r_fast - r_slow) / np.maximum(np.abs(r_slow), 1e-6)))
    if not np.isfinite(max_rel) or max_rel > tol:
        raise RuntimeError(f"fast prediction path disagrees with EveNet-Lite predict(): max rel dev {max_rel:.3e}")
    return max_rel


def calibration_reference(cfg: ScanConfig) -> Process:
    """Fixed subset of bkg_train used for C = 1/<r>_B in EVERY run (same events for all runs)."""
    bkg = load_split(cfg, "bkg", "train")
    if cfg.calib_ref_events >= len(bkg):
        return bkg
    return bkg.take(subsample_indices(len(bkg), cfg.calib_ref_events / len(bkg), CALIB_REF_SEED))


def calibration_factor_np(r: np.ndarray, w: np.ndarray) -> float:
    p = w / w.sum()
    return float(1.0 / (r * p).sum())


def weighted_bce_auc(r_num: np.ndarray, w_num: np.ndarray, r_den: np.ndarray, w_den: np.ndarray) -> Dict[str, float]:
    """Balanced weighted cross-entropy and AUC from r on numerator (y=1) and denominator (y=0) events."""
    s = np.concatenate([r_num / (1 + r_num), r_den / (1 + r_den)])
    s = np.clip(s, 1e-7, 1 - 1e-7)
    y = np.concatenate([np.ones(len(r_num)), np.zeros(len(r_den))])
    w = np.concatenate([w_num / w_num.sum(), w_den / w_den.sum()])
    bce = float(-(w * (y * np.log(s) + (1 - y) * np.log(1 - s))).sum() / w.sum())
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, s, sample_weight=w))
    except Exception:
        auc = float("nan")
    return {"val_bce_full": bce, "val_auc_full": auc}


def closure_chi2(kin_num: pd.DataFrame, w_num: np.ndarray, kin_den: pd.DataFrame, w_den: np.ndarray,
                 r_den: np.ndarray, bins: np.ndarray = None) -> Dict[str, float]:
    """m4l reweighting closure: does r(x) * w_B reproduce the numerator m4l shape?  chi2/ndf with MC errors."""
    bins = np.linspace(180, 1000, 42) if bins is None else bins
    p_num = w_num / w_num.sum()
    w_rw = w_den * r_den
    p_rw = w_rw / w_rw.sum()
    m_num, m_den = m4l(kin_num), m4l(kin_den)
    h_num, _ = np.histogram(m_num, bins, weights=p_num)
    e_num, _ = np.histogram(m_num, bins, weights=p_num ** 2)
    h_rw, _ = np.histogram(m_den, bins, weights=p_rw)
    e_rw, _ = np.histogram(m_den, bins, weights=p_rw ** 2)
    var = e_num + e_rw
    ok = var > 0
    chi2 = float((((h_rw - h_num) ** 2)[ok] / var[ok]).sum())
    ndf = int(ok.sum() - 1)
    return {"closure_chi2": chi2, "closure_ndf": ndf, "closure_chi2_ndf": chi2 / max(ndf, 1),
            "closure_bins": bins.tolist(), "closure_h_num": h_num.tolist(), "closure_h_rw": h_rw.tolist()}


def train_one(cfg: ScanConfig, variant: str, task: str, frac: float, seed: int,
              device: str = "auto", force: bool = False, wandb_name: Optional[str] = None) -> Optional[Dict]:
    """Train one (variant, task, fraction, seed) point and export everything the measurement needs."""
    import shutil
    import torch
    from nsbi.data import FEATURE_NAMES, NORMALIZATION_RULES

    out = run_dir(cfg, variant, task, frac, seed)
    if (out / "done.json").exists() and not force:
        print(f"[skip] {out} already done")
        return json.load(open(out / "metrics.json"))
    if out.exists() and force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("done.json", "failed.json"):
        (out / stale).unlink(missing_ok=True)

    t_start = time.time()
    try:
        # ---------------- data ----------------
        ckpt = out / "final.pt"
        meta_path = out / "train_meta.json"
        resume = ckpt.exists() and meta_path.exists() and (out / "history.json").exists() and not force
        P = task_processes(cfg, task, frac, seed)
        w_den_train = _denominator_weights(P["den_train"], P["reweight"])
        sizes = {
            "n_num_train": len(P["num_train"]), "n_den_train": len(P["den_train"]),
            "ess_num_train": effective_sample_size(P["num_train"].weights.to_numpy()),
            "ess_den_train": effective_sample_size(w_den_train),
            "n_num_val": len(P["num_val"]), "n_den_val": len(P["den_val"]),
        }
        _NAN_GUARD["consecutive"] = 0
        _NAN_GUARD["total"] = 0
        set_seeds(seed)
        clf = build_variant(variant, cfg, device=device)
        if resume:
            from nsbi.data import FEATURE_NAMES as _FN
            print(f"[resume] {out}: trained weights found, skipping training")
            set_fast_math(False)
            clf.load_checkpoint(str(ckpt), feature_names=_FN)
            meta = json.loads(meta_path.read_text())
            hist = json.loads((out / "history.json").read_text())
            n_trainable = int(meta["n_trainable_params"])
            fit_seconds = float(meta["fit_seconds"])
            n_nonfinite = int(meta.get("n_nonfinite_batches", 0))
            sizes["n_train_rows"] = int(meta.get("n_train_rows", 0))
            sizes["n_val_monitor_rows"] = int(meta.get("n_val_monitor_rows", 0))
        else:
            train_data = _classifier_data(P["num_train"], P["den_train"], P["reweight"])
            val_full = _classifier_data(P["num_val"], P["den_val"], P["reweight"])
            val_monitor = _subset_rows(*val_full, cfg.val_monitor_rows, VAL_MONITOR_SEED)
            sizes["n_train_rows"] = int(len(train_data[1]))
            sizes["n_val_monitor_rows"] = int(len(val_monitor[1]))
            del val_full

            # ---------------- model ----------------
            if cfg.wandb_project:
                clf.config.use_wandb = True
                clf.config.wandb = {"project": cfg.wandb_project, "entity": cfg.wandb_entity,
                                    "name": wandb_name or f"{variant}_{task}_{frac_tag(frac)}_s{seed}",
                                    "group": f"{variant}_{task}", "tags": [variant, task, frac_tag(frac)]}
            n_trainable = count_trainable(clf)
            set_seeds(seed)
            history = make_history_callback(out / "history.json")
            sampler_seed = make_sampler_seed_callback(seed)
            set_fast_math(True, tf32=cfg.tf32, autocast_bf16=cfg.autocast_bf16)
            t_fit = time.time()
            clf.fit(
                train_data=train_data,
                val_data=val_monitor,
                feature_names=FEATURE_NAMES,
                normalization_rules=NORMALIZATION_RULES,
                callbacks=[history, sampler_seed],
                epochs=cfg.max_epochs,
                batch_size=cfg.batch_size,
                sampler="weighted",
                epoch_size=cfg.epoch_size,
                checkpoint_path=None,
                early_stop_metric="val_loss",
                early_stop_patience=cfg.early_stop_patience,
            )
            fit_seconds = time.time() - t_fit
            set_fast_math(False)                 # exports and the fast-vs-slow check run in exact fp32
            del train_data, val_monitor
            hist = history.rows
            n_nonfinite = int(_NAN_GUARD["total"])
            clf.save_checkpoint(str(ckpt))
            meta_path.write_text(json.dumps({"n_trainable_params": n_trainable, "fit_seconds": fit_seconds,
                                             "n_nonfinite_batches": n_nonfinite,
                                             "n_train_rows": sizes["n_train_rows"],
                                             "n_val_monitor_rows": sizes["n_val_monitor_rows"]}))

        # ---------------- best epoch / step ----------------
        val_losses = np.array([h.get("val_loss", float("nan")) for h in hist], dtype=float)
        steps_per_epoch = math.ceil(cfg.epoch_size / cfg.batch_size)
        if len(hist) and np.isfinite(val_losses).any():
            best_epoch = int(np.nanargmin(val_losses)) + 1
            best_step = int(hist[best_epoch - 1].get("global_step") or best_epoch * steps_per_epoch)
        else:
            best_epoch, best_step = 0, 0
        total_steps = int(hist[-1].get("global_step") or len(hist) * steps_per_epoch) if hist else 0

        # ---------------- r on validation ----------------
        fast_max_rel = check_fast_predict(clf, P["den_val"])     # guards the batched inference path every run
        r_num_val = compute_r_np(clf, P["num_val"])
        r_den_val = r_num_val if P["num_val"] is P["den_val"] else compute_r_np(clf, P["den_val"])
        w_num_val = P["num_val"].weights.to_numpy(np.float64)
        w_den_val = _denominator_weights(P["den_val"], P["reweight"])
        np.savez(out / "r_val.npz", r_num=r_num_val.astype(np.float32), r_den=r_den_val.astype(np.float32),
                 same_events=np.bool_(P["num_val"] is P["den_val"]))
        val_metrics = weighted_bce_auc(r_num_val, w_num_val, r_den_val, w_den_val)
        closure = closure_chi2(P["num_val"].kinematics, w_num_val, P["den_val"].kinematics, w_den_val, r_den_val)

        # ---------------- calibration ----------------
        ref = calibration_reference(cfg)
        C_ref = calibration_factor_np(compute_r_np(clf, ref), ref.weights.to_numpy(np.float64))
        r_den_train = compute_r_np(clf, P["den_train"])
        C_sub = calibration_factor_np(r_den_train, w_den_train)
        C_val = calibration_factor_np(r_den_val, w_den_val)

        # ---------------- toys ----------------
        toy_info = {}
        for mu_true in cfg.mu_true_values:
            tag = MU_TAGS[mu_true]
            toys = load_toys(cfg, tag)
            proc = process_from_arrays(toys["kin"], None, toys["n"])
            r_toys = compute_r_np(clf, proc).astype(np.float32)
            np.save(out / f"r_toys_{tag}.npy", r_toys)
            toy_info[tag] = {"rows": int(len(r_toys)), "mean_r": float(r_toys.mean())}

        # ---------------- normaliser + checkpoint ----------------
        try:
            norm_stats = _jsonable(clf.normalizer.state_dict())   # nested {'stats': {group: {'mean': Tensor, ...}}}
        except Exception as exc:
            norm_stats = {"error": repr(exc)}

        metrics = {
            "variant": variant, "task": task, "frac": frac, "frac_tag": frac_tag(frac), "seed": seed,
            "subsample_seed": subsample_seed(seed),
            **sizes,
            "n_trainable_params": n_trainable,
            "n_nonfinite_batches": n_nonfinite, "resumed_from_checkpoint": bool(resume),
            "tf32_training": bool(cfg.tf32), "autocast_bf16_training": bool(cfg.autocast_bf16),
            "epochs_run": len(hist), "best_epoch": best_epoch, "best_step": best_step, "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "best_val_loss_monitor": float(np.nanmin(val_losses)) if best_epoch else float("nan"),
            **val_metrics,
            "C_ref": C_ref, "C_sub": C_sub, "C_val": C_val,
            "fast_predict_max_rel_dev": fast_max_rel,
            "closure_chi2_ndf": closure["closure_chi2_ndf"], "closure_chi2": closure["closure_chi2"],
            "closure_ndf": closure["closure_ndf"],
            "toys": toy_info,
            "fit_seconds": fit_seconds, "total_seconds": time.time() - t_start,
            "device": str(getattr(getattr(clf, "trainer", None), "device", device)),
            "config": asdict(cfg),
            "normalizer": norm_stats,
        }
        metrics_text = json.dumps(_jsonable(metrics), indent=1, default=str)
        closure_text = json.dumps(_jsonable(closure))
        (out / "metrics.json").write_text(metrics_text)
        (out / "closure.json").write_text(closure_text)
        json.loads((out / "metrics.json").read_text())   # re-read: only then mark the run as done
        (out / "done.json").write_text(json.dumps(
            {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": time.time() - t_start}))
        if not cfg.keep_checkpoint:
            ckpt.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        print(f"[done] {variant}/{task}/{frac_tag(frac)}/seed{seed}: best_epoch={best_epoch} "
              f"val_auc={val_metrics['val_auc_full']:.4f} C_ref={C_ref:.4f} chi2/ndf={closure['closure_chi2_ndf']:.2f} "
              f"({(time.time()-t_start)/60:.1f} min)")
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics
    except Exception as exc:  # keep the driver alive; record the failure
        with open(out / "failed.json", "w") as fh:
            json.dump({"error": repr(exc), "traceback": traceback.format_exc(),
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
        print(f"[FAILED] {variant}/{task}/{frac_tag(frac)}/seed{seed}: {exc!r}")
        return None


def run_driver(cfg: ScanConfig, specs: List[Dict], device: str = "auto", max_runs: Optional[int] = None,
               stop_after_seconds: Optional[float] = None, retry_failed: bool = True) -> pd.DataFrame:
    """Loop over run specs, skipping completed ones."""
    t0 = time.time()
    rows, n_done = [], 0
    for spec in specs:
        out = run_dir(cfg, **spec)
        if (out / "done.json").exists():
            rows.append({**spec, "status": "done"})
            continue
        if not retry_failed and (out / "failed.json").exists():
            rows.append({**spec, "status": "failed"})
            continue
        if max_runs is not None and n_done >= max_runs:
            rows.append({**spec, "status": "pending"})
            continue
        if stop_after_seconds is not None and time.time() - t0 > stop_after_seconds:
            rows.append({**spec, "status": "pending"})
            continue
        m = train_one(cfg, device=device, **spec)
        rows.append({**spec, "status": "done" if m else "failed"})
        n_done += 1
    return pd.DataFrame(rows)


def scan_status(cfg: ScanConfig, specs: Optional[List[Dict]] = None) -> pd.DataFrame:
    if specs is None:
        specs = run_matrix(cfg)
    rows = []
    for spec in specs:
        out = run_dir(cfg, **spec)
        status = ("done" if (out / "done.json").exists() else "failed" if (out / "failed.json").exists()
                  else "trained" if (out / "final.pt").exists() else "partial" if (out / "history.json").exists()
                  else "pending")
        rows.append({**spec, "status": status})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Measurement (numpy only - runs locally)
# ---------------------------------------------------------------------------

def nll_profile(r_sig: np.ndarray, r_sbi: np.ndarray, n: np.ndarray, xs: Dict[str, float],
                mu_grid: np.ndarray, lumi: float, chunk: int = 32768) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """numpy port of nsbi.measure.neg_log_likelihood (rate + shape), chunked over events."""
    mu = np.asarray(mu_grid, dtype=np.float64)
    sq = np.sqrt(mu)
    den = xs["sig"] * mu + xs["int"] * sq + xs["bkg"]
    a_sig = xs["sig"] * (mu - sq) / den
    a_sbi = xs["sbi"] * sq / den
    a_bkg = xs["bkg"] * (1.0 - sq) / den
    r_sig = np.asarray(r_sig, dtype=np.float64)
    r_sbi = np.asarray(r_sbi, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    M = len(mu)
    rows = min(chunk, len(n)) if len(n) else 0
    buf = np.empty((rows, M))
    tmp = np.empty((rows, M))
    t_shape = np.zeros(M)
    for i in range(0, len(n), chunk):          # in-place ops on one buffer: memory-bandwidth bound otherwise
        rs = r_sig[i:i + chunk]
        rb = r_sbi[i:i + chunk]
        k = len(rs)
        b, t = buf[:k], tmp[:k]
        np.multiply(rs[:, None], a_sig[None, :], out=b)
        np.multiply(rb[:, None], a_sbi[None, :], out=t)
        b += t
        b += a_bkg[None, :]
        np.maximum(b, 1e-9, out=b)              # numerical safety, as nsbi.measure (clamp min 1e-9)
        np.log(b, out=b)
        t_shape += n[i:i + k] @ b               # weighted sum over events = matrix-vector product
    t_shape *= -2.0
    n_obs = float(n.sum())
    nu = lumi * den
    t_rate = -2.0 * (n_obs * np.log(nu) - nu)
    t_total = t_rate + t_shape
    return t_total - t_total.min(), t_rate - t_rate.min(), t_shape - t_shape.min()


def _crossing(mu: np.ndarray, t: np.ndarray, i0: int, level: float, direction: int) -> float:
    """Linear-interpolated mu where t crosses `level` walking from the minimum i0 (contiguous interval)."""
    i = i0
    while 0 <= i + direction < len(t) and t[i + direction] < level:
        i += direction
    j = i + direction
    if j < 0 or j >= len(t):
        return float("nan")            # interval hits the scan boundary
    # t[i] < level <= t[j]
    return float(mu[i] + (level - t[i]) * (mu[j] - mu[i]) / (t[j] - t[i]))


def fit_mu(mu_grid: np.ndarray, t: np.ndarray, mu_true: Optional[float] = None) -> Dict[str, float]:
    """Best fit (parabolic refinement of the grid minimum), contiguous 1-sigma interval, t at mu_true."""
    t = np.asarray(t, dtype=np.float64)
    n_nan = int(np.isnan(t).sum())
    nan = float("nan")
    if n_nan == len(t):
        res = {"mu_hat": nan, "lo": nan, "hi": nan, "width": nan, "lo_2sig": nan, "hi_2sig": nan,
               "at_boundary": True, "n_nan_t": n_nan}
        if mu_true is not None:
            res.update({"t_true": nan, "covers_1sig": nan})
        return res
    i0 = int(np.nanargmin(t))
    mu_hat = float(mu_grid[i0])
    if 0 < i0 < len(t) - 1:
        y0, y1, y2 = t[i0 - 1], t[i0], t[i0 + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom > 0:
            offset = 0.5 * (y0 - y2) / denom          # in grid steps; a true minimum keeps |offset| <= 0.5
            if abs(offset) <= 1.0:                    # guard against flat/noisy profiles
                mu_hat = float(mu_grid[i0] + offset * (mu_grid[1] - mu_grid[0]))
    lo = _crossing(mu_grid, t, i0, 1.0, -1)
    hi = _crossing(mu_grid, t, i0, 1.0, +1)
    lo2 = _crossing(mu_grid, t, i0, 4.0, -1)
    hi2 = _crossing(mu_grid, t, i0, 4.0, +1)
    res = {"mu_hat": mu_hat, "lo": lo, "hi": hi, "width": hi - lo, "lo_2sig": lo2, "hi_2sig": hi2,
           "at_boundary": bool(i0 == 0 or i0 == len(t) - 1), "n_nan_t": n_nan}
    if mu_true is not None:
        res["t_true"] = float(np.interp(mu_true, mu_grid, t))
        res["covers_1sig"] = float(lo <= mu_true <= hi) if np.isfinite(lo) and np.isfinite(hi) else nan
    return res


def measure_toys(r_sig_toys: np.ndarray, r_sbi_toys: np.ndarray, toys: Dict[str, np.ndarray],
                 xs: Dict[str, float], cfg: ScanConfig, C_sig: float = 1.0, C_sbi: float = 1.0,
                 n_jobs: Optional[int] = None) -> pd.DataFrame:
    """One row per toy: mu_hat, interval, width, t(mu_true)."""
    mu_grid = cfg.mu_grid
    rs = np.asarray(r_sig_toys, np.float64) * C_sig
    rb = np.asarray(r_sbi_toys, np.float64) * C_sbi
    if len(rs) != len(toys["n"]) or len(rb) != len(toys["n"]):
        raise ValueError(f"r arrays ({len(rs)}, {len(rb)}) do not match the toy table ({len(toys['n'])} rows)")
    finite = np.isfinite(rs) & np.isfinite(rb)
    toy_ids = np.unique(toys["toy_id"])
    nan = float("nan")

    def one(t):
        m = toys["toy_id"] == t
        bad = int((~finite[m]).sum())
        if bad:
            return {"toy": int(t), "n_nonfinite": bad, "mu_hat": nan, "lo": nan, "hi": nan, "width": nan,
                    "lo_2sig": nan, "hi_2sig": nan, "at_boundary": True, "n_nan_t": 0, "t_true": nan,
                    "covers_1sig": nan}
        tt, _, _ = nll_profile(rs[m], rb[m], toys["n"][m], xs, mu_grid, cfg.lumi)
        return {"toy": int(t), "n_nonfinite": 0, **fit_mu(mu_grid, tt, toys["mu_true"])}

    n_jobs = n_jobs or max(1, min(4, os.cpu_count() or 1))
    if n_jobs > 1 and len(toy_ids) > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(one, toy_ids))
    else:
        rows = [one(t) for t in toy_ids]
    df = pd.DataFrame(rows)
    df["mu_true"] = toys["mu_true"]
    return df


RUN_COLS = ["variant", "task", "frac", "seed"]


def collect_runs(cfg: ScanConfig) -> pd.DataFrame:
    """One row per completed run with the scalar entries of metrics.json (+ run_dir, finished timestamp)."""
    rows = []
    for done in sorted(glob.glob(str(cfg.runs_dir / "*" / "*" / "f*" / "seed*" / "done.json"))):
        d = Path(done).parent
        try:
            m = json.loads((d / "metrics.json").read_text())
            finished = str(json.loads(Path(done).read_text()).get("finished", ""))
        except (OSError, json.JSONDecodeError) as exc:      # e.g. a file Drive is still syncing
            print(f"[collect_runs] skipping {d}: {exc!r}")
            continue
        row = {k: v for k, v in m.items() if isinstance(v, (int, float, str, bool))}
        row["run_dir"] = str(d)
        row["finished"] = finished
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=RUN_COLS + ["run_dir", "finished", "C_ref", "C_sub", "C_val",
                                               "val_auc_full", "val_bce_full", "closure_chi2_ndf", "best_step"])
    return pd.DataFrame(rows).sort_values(RUN_COLS).reset_index(drop=True)


CALIBRATIONS = ("C_ref", "C_sub", "C_val", "none")
MEAS_KEY = ["variant", "seed", "frac_sig", "frac_sbi", "mu_true", "calibration", "finished_sig", "finished_sbi"]


def _write_measurement_cache(cache_path: str, cached: Dict, used: set, out: List[pd.DataFrame]) -> None:
    keep = [g for k, g in cached.items() if k not in used]          # rows of pairs not touched in this call
    frames = keep + out
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(cache_path, index=False)


def measure_all(cfg: ScanConfig, runs: pd.DataFrame, diagonal_only: bool = True,
                calibration: str = "C_ref", cache_path: Optional[str] = None,
                n_jobs: Optional[int] = None) -> pd.DataFrame:
    """Combine every available (S/B run, SBI/B run) pair of the same variant + seed and measure mu on the toys."""
    if calibration not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {CALIBRATIONS}, got {calibration!r}")
    if runs is None or not len(runs):
        return pd.DataFrame()
    xs = load_cross_sections(cfg)
    toys = {MU_TAGS[m]: load_toys(cfg, MU_TAGS[m], with_kin=False) for m in cfg.mu_true_values}
    sig_runs = runs[runs.task == "sig_over_bkg"]
    sbi_runs = runs[runs.task == "sbi_over_bkg"]
    cached: Dict[tuple, pd.DataFrame] = {}
    if cache_path and Path(cache_path).exists():
        old = pd.read_csv(cache_path)
        if all(c in old.columns for c in MEAS_KEY):                 # an older cache format is simply recomputed
            for c in ("finished_sig", "finished_sbi", "calibration"):
                old[c] = old[c].fillna("").astype(str)
            for k, g in old.groupby(MEAS_KEY, sort=False):
                cached[tuple(k)] = g
    out: List[pd.DataFrame] = []
    used: set = set()
    for _, a in sig_runs.iterrows():
        for _, b in sbi_runs.iterrows():
            if a.variant != b.variant or a.seed != b.seed:
                continue
            if diagonal_only and not np.isclose(a.frac, b.frac):
                continue
            C_a = 1.0 if calibration == "none" else float(a[calibration])
            C_b = 1.0 if calibration == "none" else float(b[calibration])
            fin_a, fin_b = str(a.get("finished", "") or ""), str(b.get("finished", "") or "")
            for mu_true in cfg.mu_true_values:
                tag = MU_TAGS[mu_true]
                key = (a.variant, int(a.seed), float(a.frac), float(b.frac), float(mu_true), calibration, fin_a, fin_b)
                used.add(key)
                if key in cached:
                    out.append(cached[key])
                    continue
                r_a = np.load(Path(a.run_dir) / f"r_toys_{tag}.npy")
                r_b = np.load(Path(b.run_dir) / f"r_toys_{tag}.npy")
                df = measure_toys(r_a, r_b, toys[tag], xs, cfg, C_a, C_b, n_jobs=n_jobs)
                df.insert(0, "variant", a.variant)
                df.insert(1, "seed", int(a.seed))
                df.insert(2, "frac_sig", float(a.frac))
                df.insert(3, "frac_sbi", float(b.frac))
                df["calibration"] = calibration
                df["finished_sig"] = fin_a
                df["finished_sbi"] = fin_b
                out.append(df)
                if cache_path:
                    _write_measurement_cache(cache_path, cached, used, out)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if cache_path and out:
        _write_measurement_cache(cache_path, cached, used, out)
    return res


def aggregate(measurements: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per (variant, seed, f_sig, f_sbi, mu_true) toy averages, then per (variant, f_sig, f_sbi, mu_true) over seeds."""
    if measurements is None or not len(measurements):
        return pd.DataFrame(), pd.DataFrame()
    m = measurements.reset_index(drop=True)
    keys = ["variant", "seed", "frac_sig", "frac_sbi", "mu_true"]
    per_seed = m.groupby(keys).agg(
        n_toys=("mu_hat", "size"),
        n_valid=("mu_hat", lambda s: int(s.notna().sum())),
        mean_muhat=("mu_hat", "mean"),
        std_muhat=("mu_hat", "std"),
        mean_width=("width", "mean"),
        n_open=("width", lambda s: int(s.isna().sum())),
        mean_t_true=("t_true", "mean"),
        frac_covering=("covers_1sig", "mean"),
        n_boundary=("at_boundary", "sum"),
    ).reset_index()
    per_seed["bias"] = per_seed["mean_muhat"] - per_seed["mu_true"]          # mu_true is a group key
    per_seed["mean_halfwidth"] = 0.5 * per_seed["mean_width"]
    per_seed["bias_se_toys"] = per_seed["std_muhat"] / np.sqrt(per_seed["n_valid"].clip(lower=1))
    per_seed["bias_over_sigma"] = per_seed["bias"] / per_seed["mean_halfwidth"]
    keys2 = ["variant", "frac_sig", "frac_sbi", "mu_true"]
    over_seeds = per_seed.groupby(keys2).agg(
        n_seeds=("seed", "size"),
        bias_mean=("bias", "mean"),
        bias_seed_std=("bias", "std"),
        abs_bias_mean=("bias", lambda s: s.abs().mean()),
        width_mean=("mean_width", "mean"),
        halfwidth_mean=("mean_halfwidth", "mean"),
        bias_over_sigma_mean=("bias_over_sigma", "mean"),
        std_muhat_mean=("std_muhat", "mean"),
        bias_se_toys_mean=("bias_se_toys", "mean"),
        n_open_max=("n_open", "max"),
    ).reset_index()
    multi = over_seeds["n_seeds"] > 1
    over_seeds["bias_err"] = np.where(multi, over_seeds["bias_seed_std"], over_seeds["bias_se_toys_mean"])
    over_seeds["err_kind"] = np.where(multi, "seed_std", "toy_se")
    return per_seed, over_seeds


@functools.lru_cache(maxsize=None)
def _summary_cached(data_dir: str) -> Dict:
    return json.loads((Path(data_dir) / "summary.json").read_text())


def training_events_for(cfg: ScanConfig, frac: float, task: str = "sig_over_bkg") -> int:
    """x-axis helper: training events of the smaller hypothesis at fraction f (summary.json read once)."""
    s = _summary_cached(str(cfg.data_dir))
    proc = "sig" if task == "sig_over_bkg" else "sbi"
    n = s[f"{proc}_train"]["rows"]
    return n if frac >= 1.0 else int(round(frac * n))


def toy_process_from_csv(path: str) -> Process:
    """The single-toy Process used by the old Graphing_Together notebooks (for regression checks)."""
    df = pd.read_csv(path)
    return Process(df[KIN_COLS], pd.DataFrame(), df["n"])


__all__ = [n for n in dir() if not n.startswith("_")]
