"""Export the poster's fine-tuned checkpoints as pseudo-runs (runs/poster/) so measure_all can pair them."""
import json, os, sys, time
import numpy as np
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.expanduser("~/EveNet-Lite")); sys.path.insert(0, os.path.expanduser("~/datascan"))
import torch
import nsbi_scan as ns
from evenet_lite import EvenetLiteClassifier
from nsbi.data import FEATURE_NAMES

ROOT = os.path.expanduser("~/datascan")
cfg = ns.ScanConfig.load(ROOT).with_root(ROOT)
CKPT = {
    "sig_over_bkg": os.path.join(ROOT, "poster_ckpt", "EveNetPretrain_sig_over_bkg-val_loss-epoch0007-0.4119.pt"),
    "sbi_over_bkg": os.path.join(ROOT, "poster_ckpt", "EveNetPretrain_nsbi_over_bkg-val_loss-epoch0020-0.6926.pt"),
}
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
for task, path in CKPT.items():
    t0 = time.time()
    clf = EvenetLiteClassifier(class_labels=["bkg", "sig"], device="cuda", sequential_input_dim=ns.SEQ_DIM,
                               global_input_dim=ns.GLOBAL_DIM, pretrained=False)
    clf.load_checkpoint(path, feature_names=FEATURE_NAMES)
    out = ns.run_dir(cfg, "poster", task, 1.0, 0); out.mkdir(parents=True, exist_ok=True)
    P = ns.task_processes(cfg, task, 1.0, 0)
    dev = ns.check_fast_predict(clf, P["den_val"])
    w_den_val = ns._denominator_weights(P["den_val"], P["reweight"])
    r_num_val = ns.compute_r_np(clf, P["num_val"])
    r_den_val = r_num_val if P["num_val"] is P["den_val"] else ns.compute_r_np(clf, P["den_val"])
    np.savez(out / "r_val.npz", r_num=r_num_val.astype(np.float32), r_den=r_den_val.astype(np.float32),
             same_events=np.bool_(P["num_val"] is P["den_val"]))
    ref = ns.calibration_reference(cfg)
    C_ref = ns.calibration_factor_np(ns.compute_r_np(clf, ref), ref.weights.to_numpy(np.float64))
    bkg_train = ns.load_split(cfg, "bkg", "train")
    C_poster = ns.calibration_factor_np(ns.compute_r_np(clf, bkg_train), bkg_train.weights.to_numpy(np.float64))
    C_val = ns.calibration_factor_np(r_den_val, w_den_val)
    val_metrics = ns.weighted_bce_auc(r_num_val, P["num_val"].weights.to_numpy(np.float64), r_den_val, w_den_val)
    for mu_true in cfg.mu_true_values:
        tag = ns.MU_TAGS[mu_true]; toys = ns.load_toys(cfg, tag)
        np.save(out / f"r_toys_{tag}.npy", ns.compute_r_np(clf, ns.process_from_arrays(toys["kin"], None, toys["n"])).astype(np.float32))
    metrics = {"variant": "poster", "task": task, "frac": 1.0, "frac_tag": "f1000", "seed": 0, "checkpoint": path,
               "C_ref": C_ref, "C_sub": C_poster, "C_val": C_val, "fast_predict_max_rel_dev": dev, **val_metrics,
               "best_epoch": int(path.split("epoch")[1][:4]), "epochs_run": 20, "best_step": 0, "closure_chi2_ndf": float("nan"),
               "total_seconds": time.time() - t0}
    (out / "metrics.json").write_text(json.dumps(ns._jsonable(metrics), indent=1, default=str))
    (out / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
    print(f"[poster] {task}: C_ref={C_ref:.4f} C_fullbkgtrain={C_poster:.4f} C_val={C_val:.4f} "
          f"val_auc={val_metrics['val_auc_full']:.4f} fast_dev={dev:.1e} ({(time.time()-t0)/60:.1f} min)")
    del clf; torch.cuda.empty_cache()

runs = ns.collect_runs(cfg); runs = runs[runs.variant == "poster"]
xs = ns.load_cross_sections(cfg)
for calib in ("C_sub", "C_ref"):
    meas = ns.measure_all(cfg, runs, diagonal_only=True, calibration=calib)
    per_seed, over = ns.aggregate(meas)
    print(f"\n=== poster checkpoints, calibration {calib} ({'full bkg_train, as the poster' if calib == 'C_sub' else 'scan reference'}) ===")
    print(over[["mu_true", "bias_mean", "bias_se_toys_mean", "width_mean", "n_open_max"]].round(3).to_string(index=False))
    t0rows = meas[meas.toy == 0][["mu_true", "mu_hat", "lo", "hi", "width"]].round(3)
    print("toy 0 (the poster's toy):"); print(t0rows.to_string(index=False))
print("POSTER EXPORT DONE")
