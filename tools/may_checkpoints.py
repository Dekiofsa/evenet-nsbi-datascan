"""Measure the poster-era checkpoints (fine-tuned and frozen, best-val vs last epoch) on the 48 toys through the
scan's export and measurement; assembles pseudo-runs under runs/may_*/.
"""
import json, os, shutil, sys, time
import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.expanduser("~/EveNet-Lite")); sys.path.insert(0, os.path.expanduser("~/datascan"))
import nsbi_scan as ns
from evenet_lite import EvenetLiteClassifier
from nsbi.data import FEATURE_NAMES

ROOT = os.path.expanduser("~/datascan")
CK = os.path.join(ROOT, "may_ckpt")
cfg = ns.ScanConfig.load(ROOT).with_root(ROOT)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FILES = {
    "pt_sig_final": "pretrain_sig/final.pt",
    "pt_sbi_final": "pretrain_sbi/final.pt",
    "fz_sig_ep8": "ssl_sig/sig_over_bkg-val_loss-epoch0008-0.4272.pt",
    "fz_sig_ep20": "ssl_sig/sig_over_bkg-val_loss-epoch0020-0.4283.pt",
    "fz_sig_final": "ssl_sig/final.pt",
    "fz_sbi_ep20": "ssl_sbi/sbi_over_bkg-val_loss-epoch0020-0.6926.pt",
    "fz_sbi_final": "ssl_sbi/final.pt",
}
VARIANTS = {
    "may_ft_last": {"sig_over_bkg": "pt_sig_final", "sbi_over_bkg": "pt_sbi_final"},
    "may_fz_best": {"sig_over_bkg": "fz_sig_ep8", "sbi_over_bkg": "fz_sbi_ep20"},
    "may_fz_ep20": {"sig_over_bkg": "fz_sig_ep20", "sbi_over_bkg": "fz_sbi_ep20"},
    "may_fz_last": {"sig_over_bkg": "fz_sig_final", "sbi_over_bkg": "fz_sbi_final"},
}
EXPORT_DIR = os.path.join(ROOT, "may_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


def state_dict(path):
    """EveNet-Lite checkpoints: {'model': OrderedDict of tensors, 'optimizer', 'normalizer', 'extra': {'epoch'"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else (ck["state_dict"] if "state_dict" in ck else ck)
    extra = ck.get("extra", {}) if isinstance(ck, dict) else {}
    ep = extra.get("epoch") if isinstance(extra, dict) else None
    step = None
    try:
        opt = ck.get("optimizer")
        st = (opt[0] if isinstance(opt, list) else opt)["state"]
        step = int(next(iter(st.values()))["step"])
    except Exception:
        pass
    return sd, (ep, step)


# ---------------------------------------------------------------- step 0: what is final.pt?
print("=== step 0: final.pt vs epoch checkpoints (tensor comparison) ===", flush=True)
alias = {}
for a, b in [("pt_sig_final", os.path.join(ROOT, "poster_ckpt", "EveNetPretrain_sig_over_bkg-val_loss-epoch0007-0.4119.pt")),
             ("pt_sbi_final", os.path.join(ROOT, "poster_ckpt", "EveNetPretrain_nsbi_over_bkg-val_loss-epoch0020-0.6926.pt")),
             ("fz_sig_final", os.path.join(CK, FILES["fz_sig_ep20"])), ("fz_sig_final", os.path.join(CK, FILES["fz_sig_ep8"])),
             ("fz_sbi_final", os.path.join(CK, FILES["fz_sbi_ep20"]))]:
    sa, ea = state_dict(os.path.join(CK, FILES[a])); sb, eb = state_dict(b)
    common = [k for k in sa if k in sb and sa[k].shape == sb[k].shape]
    nd = sum(1 for k in common if not torch.equal(sa[k], sb[k]))
    md = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in common)
    print(f"{a:13s} vs {os.path.basename(b)[:52]:52s}: {len(common)} tensors, {nd} differ, max|diff| {md:.2e}, epochs {ea} / {eb}", flush=True)
    if nd == 0:
        alias[a] = os.path.basename(b)
print("identical files:", alias, flush=True)
# fz_sig_final identical to ep20 -> export once
if "fz_sig_final" in alias and alias["fz_sig_final"].startswith("sig_over_bkg-val_loss-epoch0020"):
    FILES["fz_sig_final"] = FILES["fz_sig_ep20"]
if "fz_sbi_final" in alias:
    FILES["fz_sbi_final"] = FILES["fz_sbi_ep20"]

# ---------------------------------------------------------------- step 1: export once per unique checkpoint file
ref = ns.calibration_reference(cfg)
bkg_train = ns.load_split(cfg, "bkg", "train")
toys = {tag: ns.load_toys(cfg, tag) for tag in ns.MU_TAGS.values()}


def export(key, task):
    path = os.path.join(CK, FILES[key])
    out = os.path.join(EXPORT_DIR, os.path.basename(FILES[key]).replace(".pt", "") + f"__{task}")
    if os.path.exists(os.path.join(out, "done.json")):
        print(f"[export] {key}: cached at {out}", flush=True); return out
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    clf = EvenetLiteClassifier(class_labels=["bkg", "sig"], device="cuda", sequential_input_dim=ns.SEQ_DIM,
                               global_input_dim=ns.GLOBAL_DIM, pretrained=False)
    clf.load_checkpoint(path, feature_names=FEATURE_NAMES)
    P = ns.task_processes(cfg, task, 1.0, 0)
    dev = ns.check_fast_predict(clf, P["den_val"])
    w_den_val = ns._denominator_weights(P["den_val"], P["reweight"])
    r_num_val = ns.compute_r_np(clf, P["num_val"])
    r_den_val = r_num_val if P["num_val"] is P["den_val"] else ns.compute_r_np(clf, P["den_val"])
    np.savez(os.path.join(out, "r_val.npz"), r_num=r_num_val.astype(np.float32), r_den=r_den_val.astype(np.float32),
             same_events=np.bool_(P["num_val"] is P["den_val"]))
    C_ref = ns.calibration_factor_np(ns.compute_r_np(clf, ref), ref.weights.to_numpy(np.float64))
    C_sub = ns.calibration_factor_np(ns.compute_r_np(clf, bkg_train), bkg_train.weights.to_numpy(np.float64))
    C_val = ns.calibration_factor_np(r_den_val, w_den_val)
    vm = ns.weighted_bce_auc(r_num_val, P["num_val"].weights.to_numpy(np.float64), r_den_val, w_den_val)
    for tag, T in toys.items():
        np.save(os.path.join(out, f"r_toys_{tag}.npy"), ns.compute_r_np(clf, ns.process_from_arrays(T["kin"], None, T["n"])).astype(np.float32))
    _, (ep, step) = state_dict(path)
    metrics = {"task": task, "frac": 1.0, "frac_tag": "f1000", "seed": 0, "checkpoint": path, "checkpoint_epoch": ep,
               "checkpoint_step": step, "C_ref": C_ref, "C_sub": C_sub, "C_val": C_val, "fast_predict_max_rel_dev": dev, **vm,
               "best_epoch": ep if ep is not None else 20, "epochs_run": 20, "best_step": step if step is not None else 0,
               "closure_chi2_ndf": float("nan"), "total_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(ns._jsonable(metrics), fh, indent=1, default=str)
    with open(os.path.join(out, "done.json"), "w") as fh:
        json.dump({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}, fh)
    print(f"[export] {key} ({task}): C_ref={C_ref:.4f} C_sub={C_sub:.4f} C_val={C_val:.4f} auc={vm['val_auc_full']:.4f} "
          f"bce={vm['val_bce_full']:.4f} fast_dev={dev:.1e} ({(time.time()-t0)/60:.1f} min)", flush=True)
    del clf; torch.cuda.empty_cache()
    return out


for variant, spec in VARIANTS.items():
    for task, key in spec.items():
        src = export(key, task)
        dst = str(ns.run_dir(cfg, variant, task, 1.0, 0))
        os.makedirs(dst, exist_ok=True)
        for f in ("r_val.npz", "r_toys_mu0p4.npy", "r_toys_mu2p5.npy", "done.json"):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
        m = json.load(open(os.path.join(src, "metrics.json"))); m["variant"] = variant
        json.dump(m, open(os.path.join(dst, "metrics.json"), "w"), indent=1)

# ---------------------------------------------------------------- step 2: measure
runs = ns.collect_runs(cfg)
runs = runs[runs.variant.isin(["poster", "frozen", "pretrained", "scratch"] + list(VARIANTS))]
runs = runs[np.isclose(runs.frac, 1.0) & (runs.seed == 0)]
W = {"sig_over_bkg": ns._denominator_weights(ns.load_split(cfg, "bkg", "val"), None),
     "sbi_over_bkg": ns._denominator_weights(ns.load_split(cfg, "sbi", "val"), ("sbi", "bkg"))}
print("\n=== tail statistics of every S/B export at 100 % (validation background, calibrated with C_ref) ===")
for _, a in runs[runs.task == "sig_over_bkg"].sort_values("variant").iterrows():
    r = np.load(os.path.join(a.run_dir, "r_val.npz"))["r_den"].astype(np.float64) * float(a.C_ref)
    p = W["sig_over_bkg"] / W["sig_over_bkg"].sum(); big = r > 30
    rt = np.load(os.path.join(a.run_dir, "r_toys_mu0p4.npy")).astype(np.float64) * float(a.C_ref)
    print(f"{a.variant:12s} epoch {a.get('checkpoint_epoch', a.best_epoch)!s:>4}: bce {a.val_bce_full:.4f} auc {a.val_auc_full:.4f} "
          f"C_ref {a.C_ref:.4f} C_sub {a.C_sub:.4f} | r>30 share of <r>_B {100*(r[big]*p[big]).sum()/(r*p).sum():5.2f} % "
          f"(n={big.sum()}) | max r toys {rt.max():8.1f}, toy events r>30: {(rt>30).sum()}", flush=True)
for calib in ("C_sub", "C_ref"):
    meas = ns.measure_all(cfg, runs, diagonal_only=True, calibration=calib)
    per_seed, over = ns.aggregate(meas)
    per_seed["n_spike"] = meas.groupby(["variant", "seed", "frac_sig", "frac_sbi", "mu_true"]).mu_hat.apply(
        lambda s: int(((s > 0.85) & (s < 1.15)).sum())).values
    print(f"\n=== 48 toys, calibration {calib} ({'full bkg_train, the poster calibration' if calib == 'C_sub' else 'scan reference'}) ===")
    print(per_seed[["variant", "mu_true", "bias", "bias_se_toys", "mean_width", "n_open", "n_spike", "frac_covering"]]
          .sort_values(["mu_true", "variant"]).round(3).to_string(index=False))
    t0 = meas[meas.toy == 0][["variant", "mu_true", "mu_hat", "lo", "hi"]].sort_values(["mu_true", "variant"]).round(2)
    print("toy 0 (the poster's toy; poster plot: fine-tuned 0.36 [0.14, 1.00], frozen 0.42 [0.15, 1.07] at mu 0.4):")
    print(t0.to_string(index=False), flush=True)
print("MAY CHECKPOINTS DONE", flush=True)
