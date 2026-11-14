"""E7: Leave-One-Module-Out (LOMO) validation for A1 (random forest), A2 (classical fusion) and A4-ring (QEMF ring circuit).
Protocol (no test leakage):
  - Hyperparameters are NOT re-tuned per fold. The learning rate / epoch count / RF depth-leaf are copied from the 2014B
    validation results already produced by the main protocol (results/e6_tuning.csv, results/trees_tuning.csv).
  - For held-out module m: train on all presentations of the other six modules (all of 2013B,2013J,2014B,2014J minus module m),
    evaluate once on module m. Test presentation 2014J is therefore inside the training pool of folds != its module;
    this is deliberate: LOMO tests generalisation across subjects, not across time. Reported separately from the 2014J test.
  - Preprocessing statistics are fitted on the training folds only.
Outputs: results/lomo/lomo_<arm>_w<wk>_s<seed>.npy (per-row probabilities) and results/lomo/lomo_summary.csv
"""
import csv, json, sys, time, numpy as np, pandas as pd, jax
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, recall_score, precision_score
from paths import RESULTS, COMPACT
import data_lib, train_e6 as E, qsim_topo as T

OUT = RESULTS / "lomo"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]; WEEKS = (4, 8, 12)
ARMS = {"A1": "rf", "A2": "A2", "A4-ring": "A4-ring"}

def tuned(arm, wk):
    """Pick the setting that was selected on 2014B in the main protocol (no new tuning)."""
    if arm == "A1":
        rows = [r for r in csv.DictReader(open(RESULTS / "trees_tuning.csv")) if r["arm"] == "A1_RF" and int(r["week"]) == wk]
        best = max(rows, key=lambda r: float(r["val_auc"])); return dict(kind="rf", hp=eval(best["hp"]))
    rows = [r for r in csv.DictReader(open(RESULTS / "e6_tuning.csv")) if r["arm"] == arm and int(r["week"]) == wk]
    best = max(rows, key=lambda r: float(r["val_auc"])); return dict(kind="nn", lr=float(best["lr"]), epochs=int(best["best_epoch"]) + 1)

def fit_predict_fold(arm, d, tr, te, seed, cfg, sg):
    if cfg["kind"] == "rf":
        X = np.hstack([d["beh"], d["sem"], d["demo"]]) if arm == "A1" else None
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed, **cfg["hp"]).fit(X[tr], d["y"][tr])
        return m.predict_proba(X[te])[:, 1]
    # neural / quantum arms: presentation-free preprocessing fitted on the training folds only
    X = E.preprocess(d, tr); y = d["y"].astype("int32")
    p, _ = E.train(arm, X, y, np.where(tr)[0], cfg["lr"], seed, sg, epochs=cfg["epochs"])
    return E.predict(arm, p, X, np.where(te)[0], sg)["prob"]

if __name__ == "__main__":
    t0 = time.time(); sg = T.signs(E.N_Q); rows = []
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ARMS)
    for wk in WEEKS:
        d = data_lib.load_week(wk); mods = sorted(set(d["module"]))
        for arm in only:
            cfg = tuned(arm, wk)
            for seed in (SEEDS if cfg["kind"] == "nn" else [0]):
                for m in mods:
                    f = OUT / f"lomo_{arm}_w{wk}_s{seed}_{m}.npy"
                    te = d["module"] == m; tr = ~te
                    if f.exists(): continue
                    p = fit_predict_fold(arm, d, tr, te, seed, cfg, sg); y = d["y"][te].astype(int)
                    np.save(f, p.astype("float32"))
                    pred = (p > 0.5).astype(int)
                    rows.append(dict(arm=arm, week=wk, seed=seed, module=m, n_test=int(te.sum()), n_train=int(tr.sum()),
                                     auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), f1=f1_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                                     recall=recall_score(y, pred), brier=brier_score_loss(y, p), cfg=json.dumps({k: (str(v)) for k, v in cfg.items()})))
                    print(arm, wk, seed, m, "AUC %.4f" % rows[-1]["auc"], flush=True)
            # append (resumable)
            if rows:
                path = OUT / "lomo_fold_metrics.csv"; new = not path.exists()
                with open(path, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); (new and w.writeheader()); w.writerows(rows)
                rows = []
    print("done in %.0fs" % (time.time() - t0), flush=True)
